//! Minimal streaming chat client for a running attune proxy.
//!
//! The CLI keeps this logic here so `attune chat` (and later `attune run`) can
//! reuse it in-process. Requests are plain HTTP with `stream: true`; the SSE
//! body is parsed by [`parse_sse_line`] as it arrives, so tokens reach the
//! caller immediately.

use std::io::{BufRead, BufReader};
use std::time::Instant;

use serde::Serialize;
use ureq::Agent;

use crate::ServeError;

/// One turn in the conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Message {
    /// `system`, `user` or `assistant`.
    pub role: String,
    /// The message text.
    pub content: String,
}

/// Token count and decode speed for one reply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChatStats {
    /// Tokens generated in the reply.
    pub tokens: u64,
    /// Decode speed in tokens per second.
    pub tokens_per_s: f64,
}

/// What a single SSE line means.
#[derive(Debug, Clone, PartialEq)]
pub enum SseEvent {
    /// A content delta to print.
    Delta(String),
    /// A `reasoning_content` delta (the model's thinking, not the reply).
    Reasoning(String),
    /// `data: [DONE]`; the reply is complete.
    Done,
    /// llama-server's final `timings` block.
    Timings {
        predicted_n: u64,
        predicted_per_second: f64,
    },
    /// Anything else: empty lines, `:` comments, keep-alives, unknown chunks.
    Ignore,
}

/// Parse one line of an OpenAI-style SSE stream.
pub fn parse_sse_line(line: &str) -> SseEvent {
    let line = line.trim_end_matches(['\r', '\n']);
    let Some(rest) = line.strip_prefix("data:") else {
        return SseEvent::Ignore;
    };
    let data = rest.trim();
    if data.is_empty() {
        return SseEvent::Ignore;
    }
    if data == "[DONE]" {
        return SseEvent::Done;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
        return SseEvent::Ignore;
    };
    if let Some(timings) = value.get("timings") {
        let n = timings
            .get("predicted_n")
            .and_then(serde_json::Value::as_u64);
        let rate = timings
            .get("predicted_per_second")
            .and_then(serde_json::Value::as_f64);
        if let (Some(predicted_n), Some(predicted_per_second)) = (n, rate) {
            return SseEvent::Timings {
                predicted_n,
                predicted_per_second,
            };
        }
    }
    if let Some(reasoning) = value
        .pointer("/choices/0/delta/reasoning_content")
        .and_then(serde_json::Value::as_str)
    {
        if !reasoning.is_empty() {
            return SseEvent::Reasoning(reasoning.to_string());
        }
    }
    if let Some(delta) = value
        .pointer("/choices/0/delta/content")
        .and_then(serde_json::Value::as_str)
    {
        if !delta.is_empty() {
            return SseEvent::Delta(delta.to_string());
        }
    }
    SseEvent::Ignore
}

/// One streamed piece handed to [`ChatClient::send`]'s callback.
///
/// Reasoning text arrives before the visible reply for models that separate
/// their thinking from the answer. The client reports both; the caller decides
/// what to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatDelta<'a> {
    /// A reasoning/thinking delta.
    Thinking(&'a str),
    /// A visible reply delta to print.
    Text(&'a str),
}

/// A streaming chat conversation with an OpenAI-compatible server.
pub struct ChatClient {
    url: String,
    api_key: Option<String>,
    history: Vec<Message>,
}

impl ChatClient {
    /// Create a client. `system` (when set) is kept as the first message and
    /// survives [`ChatClient::clear`].
    pub fn new(url: impl Into<String>, api_key: Option<String>, system: Option<String>) -> Self {
        let mut history = Vec::new();
        if let Some(system) = system {
            history.push(Message {
                role: "system".to_string(),
                content: system,
            });
        }
        Self {
            url: url.into(),
            api_key,
            history,
        }
    }

    /// The server base URL this client talks to.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// The conversation so far, including the system message when there is one.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Forget every turn but the system message.
    pub fn clear(&mut self) {
        let system = self
            .history
            .first()
            .filter(|message| message.role == "system")
            .cloned();
        self.history.clear();
        if let Some(system) = system {
            self.history.push(system);
        }
    }

    /// Send one user message, streaming reply deltas through `on_delta`.
    ///
    /// A successful call appends the user and assistant messages to the
    /// history. Reasoning deltas are reported as [`ChatDelta::Thinking`] but
    /// are not stored in the history. Tokens and speed come from
    /// llama-server's `timings` when it sends them, otherwise from the text
    /// delta count over the elapsed time.
    pub fn send(
        &mut self,
        user: &str,
        on_delta: &mut dyn FnMut(ChatDelta<'_>),
    ) -> Result<ChatStats, ServeError> {
        let mut messages = self.history.clone();
        messages.push(Message {
            role: "user".to_string(),
            content: user.to_string(),
        });
        let body = serde_json::json!({
            "model": "attune",
            "stream": true,
            "messages": messages,
        })
        .to_string();

        let endpoint = format!("{}/v1/chat/completions", self.url.trim_end_matches('/'));
        let agent = chat_agent();
        let mut request = agent
            .post(endpoint.as_str())
            .header("Content-Type", "application/json");
        if let Some(key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {key}"));
        }

        let started = Instant::now();
        let response = request
            .send(body.as_str())
            .map_err(|error| map_send_error(error, &self.url))?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let mut response = response;
            let detail = match response.body_mut().read_to_string() {
                Ok(body) => body,
                Err(error) => format!("<could not read body: {error}>"),
            };
            let snippet: String = detail.chars().take(300).collect();
            let hint = if status == 401 {
                " (check --api-key)"
            } else {
                ""
            };
            return Err(ServeError::Chat(format!(
                "server returned HTTP {status}{hint}: {snippet}"
            )));
        }

        let reader = response.into_body().into_reader();
        let mut reader = BufReader::new(reader);
        let mut line = String::new();
        let mut assistant = String::new();
        let mut chunks: u64 = 0;
        let mut timings: Option<(u64, f64)> = None;

        loop {
            line.clear();
            let read = reader
                .read_line(&mut line)
                .map_err(|error| ServeError::Chat(format!("reading stream: {error}")))?;
            if read == 0 {
                break;
            }
            match parse_sse_line(&line) {
                SseEvent::Delta(delta) => {
                    assistant.push_str(&delta);
                    chunks += 1;
                    on_delta(ChatDelta::Text(&delta));
                }
                SseEvent::Reasoning(reasoning) => {
                    on_delta(ChatDelta::Thinking(&reasoning));
                }
                SseEvent::Done => break,
                SseEvent::Timings {
                    predicted_n,
                    predicted_per_second,
                } => timings = Some((predicted_n, predicted_per_second)),
                SseEvent::Ignore => {}
            }
        }

        let stats = match timings {
            Some((tokens, tokens_per_s)) => ChatStats {
                tokens,
                tokens_per_s,
            },
            None => {
                let seconds = started.elapsed().as_secs_f64();
                ChatStats {
                    tokens: chunks,
                    tokens_per_s: if seconds > 0.0 {
                        chunks as f64 / seconds
                    } else {
                        0.0
                    },
                }
            }
        };

        self.history.push(Message {
            role: "user".to_string(),
            content: user.to_string(),
        });
        self.history.push(Message {
            role: "assistant".to_string(),
            content: assistant,
        });
        Ok(stats)
    }
}

/// A local client: never route through `HTTP(S)_PROXY`.
fn chat_agent() -> Agent {
    Agent::config_builder()
        // Non-2xx must reach our status check so the body can be reported.
        .http_status_as_error(false)
        .proxy(None)
        .build()
        .into()
}

fn map_send_error(error: ureq::Error, url: &str) -> ServeError {
    if connection_refused(&error) {
        ServeError::NoServer {
            url: url.to_string(),
        }
    } else {
        ServeError::Chat(format!("request failed: {error}"))
    }
}

fn connection_refused(error: &ureq::Error) -> bool {
    fn io_refused(error: &std::io::Error) -> bool {
        if error.kind() == std::io::ErrorKind::ConnectionRefused {
            return true;
        }
        error
            .get_ref()
            .and_then(|inner| inner.downcast_ref::<std::io::Error>())
            .is_some_and(io_refused)
    }

    match error {
        ureq::Error::Io(error) => io_refused(error),
        ureq::Error::ConnectionFailed | ureq::Error::HostNotFound => true,
        other => {
            let text = other.to_string().to_ascii_lowercase();
            text.contains("connection refused") || text.contains("actively refused")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_deltas() {
        assert_eq!(
            parse_sse_line(r#"data: {"choices":[{"delta":{"content":"Hi"}}]}"#),
            SseEvent::Delta("Hi".to_string())
        );
        assert_eq!(
            parse_sse_line(r#"data: {"choices":[{"delta":{"content":""}}]}"#),
            SseEvent::Ignore
        );
    }

    #[test]
    fn parses_reasoning_deltas() {
        assert_eq!(
            parse_sse_line(r#"data: {"choices":[{"delta":{"reasoning_content":"let me think"}}]}"#),
            SseEvent::Reasoning("let me think".to_string())
        );
        assert_eq!(
            parse_sse_line(r#"data: {"choices":[{"delta":{"reasoning_content":""}}]}"#),
            SseEvent::Ignore
        );
    }

    #[test]
    fn recognises_done_and_ignorable_lines() {
        assert_eq!(parse_sse_line("data: [DONE]"), SseEvent::Done);
        assert_eq!(parse_sse_line(": keep-alive"), SseEvent::Ignore);
        assert_eq!(parse_sse_line(""), SseEvent::Ignore);
        assert_eq!(parse_sse_line("event: ping"), SseEvent::Ignore);
    }

    #[test]
    fn reads_timings_and_ignores_malformed_json() {
        assert_eq!(
            parse_sse_line(
                r#"data: {"choices":[],"timings":{"predicted_n":5,"predicted_per_second":12.5}}"#
            ),
            SseEvent::Timings {
                predicted_n: 5,
                predicted_per_second: 12.5,
            }
        );
        assert_eq!(parse_sse_line("data: {not json"), SseEvent::Ignore);
        assert_eq!(
            parse_sse_line(r#"data: {"timings":{"predicted_n":5}}"#),
            SseEvent::Ignore
        );
    }
}
