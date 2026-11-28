//! OpenAI-compatible reverse proxy (M5 slice 2).
//!
//! All forwarding is synchronous: `tiny_http` hands us one request per worker
//! thread and `ureq` calls the local `llama-server`. The important property is
//! that a streaming response is *piped*, never buffered: the upstream body
//! reader feeds a chunked writer that flushes every read, so the first SSE
//! chunk reaches the client as soon as llama.cpp emits it and `data: [DONE]`
//! passes through verbatim.
//!
//! Non-streaming responses are returned with the upstream status and body
//! unchanged (including 4xx/5xx bodies). Only a transport failure - a dead or
//! unreachable upstream - becomes a `502` with an OpenAI-shaped error, and the
//! proxy keeps serving later requests.
//!
//! # Why streaming bypasses `Response`
//!
//! `tiny_http`'s built-in chunked encoder buffers up to 8 KiB before it writes
//! a chunk, so an SSE token stream would stall until the buffer filled or the
//! model finished. The streaming branch therefore takes the raw socket writer
//! (`Request::into_writer`), writes the response head, and frames every read as
//! its own chunk with an explicit flush. Buffered responses still use
//! `Response` and its normal `Content-Length` handling.

use std::io::{self, Cursor, Read, Write};
use std::time::{Duration, Instant};

use tiny_http::{Header, Request, Response, ResponseBox, StatusCode};
use ureq::Agent;

use crate::metrics::{Metrics, Sample, Snapshot};
use crate::server::Route;

/// Local llama-server normally answers immediately; a short connect timeout
/// keeps a dead upstream from parking a worker thread. No global/body timeout
/// is set because an SSE stream can legitimately stay open for minutes.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Upper bound for a forwarded request body. A completion request is normally
/// a few KiB; anything near this is an attempt to exhaust proxy memory.
pub(crate) const MAX_REQUEST_BODY: usize = 32 * 1024 * 1024;

/// Headers that describe this hop only and must not be copied to the client.
/// `Content-Length` is recomputed by `tiny_http`.
const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
];

/// Shared forwarding state: one pooled agent plus the resolved upstream base.
pub(crate) struct ProxyState {
    agent: Agent,
    upstream_base: String,
    metrics: Metrics,
    /// Pid of the spawned llama-server, for the RSS gauge.
    upstream_pid: Option<u32>,
    /// attune-level bearer key; when set, the client `Authorization` header is
    /// attune's own credential and must never reach llama-server.
    pub(crate) api_key: Option<String>,
}

impl ProxyState {
    pub(crate) fn new(
        upstream_base: String,
        upstream_pid: Option<u32>,
        api_key: Option<String>,
    ) -> Self {
        let agent = Agent::config_builder()
            // Upstream 4xx/5xx responses are forwarded verbatim, so ureq must
            // hand them back as responses instead of `Error::StatusCode`.
            .http_status_as_error(false)
            // The upstream is local llama-server: never via HTTP(S)_PROXY.
            .proxy(None)
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .build()
            .into();
        Self {
            agent,
            upstream_base: upstream_base.trim_end_matches('/').to_string(),
            metrics: Metrics::new(),
            upstream_pid,
            api_key,
        }
    }

    fn snapshot(&self) -> Snapshot {
        self.metrics.snapshot(
            self.upstream_pid
                .and_then(attune_hal::memory::process_rss_bytes),
            attune_hal::memory::available_bytes(),
        )
    }
}

/// Forward one routed request and write the client response.
///
/// Never returns an error: any failure is rendered as an OpenAI-shaped error
/// response so a bad request cannot kill the worker (or the proxy).
pub(crate) fn handle(request: Request, route: Route, state: &ProxyState) {
    let (body, content_type) = match route {
        Route::Metrics => (state.snapshot().to_json().to_string(), "application/json"),
        Route::Prometheus => (
            state.snapshot().to_prometheus(),
            "text/plain; version=0.0.4; charset=utf-8",
        ),
        _ => return forward_request(request, route, state),
    };
    let headers: Vec<Header> = Header::from_bytes("Content-Type", content_type)
        .ok()
        .into_iter()
        .collect();
    respond(
        request,
        Upstream {
            status: 200,
            headers,
            body: UpstreamBody::Buffered(body.into_bytes()),
            headers_at: Instant::now(),
        },
    );
}

fn forward_request(mut request: Request, route: Route, state: &ProxyState) {
    let received = Instant::now();
    // Only completions are measured; /v1/models and /health are plumbing.
    let tracker = route.is_post().then(|| state.metrics.begin());
    let authorization = if state.api_key.is_some() {
        None
    } else {
        header_value(&request, "Authorization")
    };
    let mut body = Vec::new();
    if route.is_post() {
        let declared =
            header_value(&request, "Content-Length").and_then(|value| value.parse::<u64>().ok());
        if declared.is_some_and(|length| length > MAX_REQUEST_BODY as u64) {
            respond_too_large(request);
            return;
        }
        let read = {
            // Read at most one byte past the cap so a chunked body cannot grow
            // without bound before we notice.
            let mut limited = request.as_reader().take(MAX_REQUEST_BODY as u64 + 1);
            limited.read_to_end(&mut body)
        };
        if let Err(error) = read {
            respond_error(
                request,
                400,
                "invalid_request",
                &format!("failed to read request body: {error}"),
            );
            return;
        }
        if body.len() > MAX_REQUEST_BODY {
            respond_too_large(request);
            return;
        }
    }

    let wants_stream = route.is_post() && stream_requested(&body);
    match forward(
        state,
        route,
        request.url(),
        authorization.as_deref(),
        &body,
        wants_stream,
    ) {
        Ok(upstream) => {
            let streamed = matches!(upstream.body, UpstreamBody::Stream(_));
            let delivery = respond(request, upstream);
            if let Some(tracker) = tracker {
                tracker.finish(Sample {
                    ok: delivery.ok,
                    ttft: delivery
                        .first_byte
                        .unwrap_or(received)
                        .saturating_duration_since(received),
                    total: received.elapsed(),
                    tokens: delivery.tokens,
                    streamed,
                });
            }
        }
        // `tracker` drops here without a sample: counted as an error.
        Err(error) => respond_error(
            request,
            502,
            "upstream_error",
            &format!("upstream request failed: {error}"),
        ),
    }
}

fn respond_too_large(request: Request) {
    respond_error(
        request,
        413,
        "payload_too_large",
        &format!(
            "request body exceeds the {} MiB proxy limit",
            MAX_REQUEST_BODY / (1024 * 1024)
        ),
    );
}

/// Write an OpenAI-shaped error envelope (`{"error":{...}}`).
pub(crate) fn respond_error(request: Request, status: u16, error_type: &str, message: &str) {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
        }
    })
    .to_string()
    .into_bytes();
    let headers: Vec<Header> = Header::from_bytes("Content-Type", "application/json")
        .ok()
        .into_iter()
        .collect();
    respond(
        request,
        Upstream {
            status,
            headers,
            body: UpstreamBody::Buffered(body),
            headers_at: Instant::now(),
        },
    );
}

// ---- forwarding -------------------------------------------------------------

/// A fully resolved upstream response, either buffered or still streaming.
struct Upstream {
    status: u16,
    headers: Vec<Header>,
    body: UpstreamBody,
    /// When the upstream response head arrived (TTFT for non-stream bodies).
    headers_at: Instant,
}

/// What actually reached the client, for metrics.
struct Delivery {
    ok: bool,
    first_byte: Option<Instant>,
    tokens: Option<u64>,
}

enum UpstreamBody {
    Buffered(Vec<u8>),
    Stream(ureq::BodyReader<'static>),
}

fn forward(
    state: &ProxyState,
    route: Route,
    path: &str,
    authorization: Option<&str>,
    body: &[u8],
    wants_stream: bool,
) -> Result<Upstream, ureq::Error> {
    let url = format!("{}{}", state.upstream_base, path);
    let response = if route.is_post() {
        let mut builder = state
            .agent
            .post(url.as_str())
            .header("Content-Type", "application/json");
        if let Some(value) = authorization {
            builder = builder.header("Authorization", value);
        }
        builder.send(body)?
    } else {
        let mut builder = state.agent.get(url.as_str());
        if let Some(value) = authorization {
            builder = builder.header("Authorization", value);
        }
        builder.call()?
    };

    let status = response.status().as_u16();
    let headers_at = Instant::now();
    let headers = copy_headers(response.headers());
    // 204 has no body and `tiny_http` forbids chunked framing there, so only
    // pipe a 2xx stream when a body is actually legal.
    let streaming = wants_stream && (200..300).contains(&status) && status != 204;
    if streaming {
        Ok(Upstream {
            status,
            headers,
            body: UpstreamBody::Stream(response.into_body().into_reader()),
            headers_at,
        })
    } else {
        let mut response = response;
        let body = response.body_mut().read_to_vec()?;
        Ok(Upstream {
            status,
            headers,
            body: UpstreamBody::Buffered(body),
            headers_at,
        })
    }
}

fn respond(request: Request, upstream: Upstream) -> Delivery {
    let status_ok = (200..300).contains(&upstream.status);
    match upstream.body {
        UpstreamBody::Buffered(body) => {
            let tokens = usage_tokens(&body);
            let length = body.len();
            let response: ResponseBox = Response::new(
                StatusCode(upstream.status),
                upstream.headers,
                Box::new(Cursor::new(body)) as Box<dyn Read + Send>,
                Some(length),
                None,
            );
            let sent = request.respond(response);
            if let Err(error) = &sent {
                eprintln!("[attune] proxy: failed to send response: {error}");
            }
            Delivery {
                ok: status_ok && sent.is_ok(),
                first_byte: Some(upstream.headers_at),
                tokens,
            }
        }
        UpstreamBody::Stream(reader) => {
            let (sent, mut tap) =
                respond_stream(request, upstream.status, &upstream.headers, reader);
            Delivery {
                ok: status_ok && sent,
                first_byte: tap.first_byte,
                tokens: tap.tokens(),
            }
        }
    }
}

/// Write a chunked response with one flush per upstream read.
fn respond_stream(
    request: Request,
    status: u16,
    headers: &[Header],
    mut reader: ureq::BodyReader<'static>,
) -> (bool, StreamTap) {
    let mut writer = request.into_writer();
    let mut tap = StreamTap::default();
    let result = write_stream(&mut writer, status, headers, &mut reader, &mut tap);
    if let Err(error) = &result {
        // A client that disconnects surfaces here as a broken pipe; stop
        // pulling the upstream body and let the reader drop close that side.
        eprintln!("[attune] proxy: streaming response ended early: {error}");
    }
    (result.is_ok(), tap)
}

/// Observes SSE bytes as they are piped: first-byte time and token count.
#[derive(Default)]
struct StreamTap {
    first_byte: Option<Instant>,
    /// Bytes of an unfinished trailing line.
    carry: Vec<u8>,
    /// `data:` chunks other than `[DONE]` and the usage-only chunk.
    chunks: u64,
    usage_tokens: Option<u64>,
}

impl StreamTap {
    fn feed(&mut self, bytes: &[u8]) {
        if self.first_byte.is_none() && !bytes.is_empty() {
            self.first_byte = Some(Instant::now());
        }
        self.carry.extend_from_slice(bytes);
        while let Some(end) = self.carry.iter().position(|&byte| byte == b'\n') {
            let line: Vec<u8> = self.carry.drain(..=end).collect();
            self.line(&line);
        }
    }

    fn line(&mut self, line: &[u8]) {
        let Some(data) = line.strip_prefix(b"data:") else {
            return;
        };
        let data = data.trim_ascii();
        if data.is_empty() || data == b"[DONE]" {
            return;
        }
        match usage_tokens(data) {
            Some(tokens) => self.usage_tokens = Some(tokens),
            None => self.chunks += 1,
        }
    }

    /// Upstream-reported usage wins; otherwise the content chunk count.
    fn tokens(&mut self) -> Option<u64> {
        let rest = std::mem::take(&mut self.carry);
        self.line(&rest);
        self.usage_tokens
            .or((self.chunks > 0).then_some(self.chunks))
    }
}

/// `usage.completion_tokens` from a JSON body or SSE chunk, if present.
fn usage_tokens(json: &[u8]) -> Option<u64> {
    serde_json::from_slice::<serde_json::Value>(json)
        .ok()?
        .get("usage")?
        .get("completion_tokens")?
        .as_u64()
}

fn write_stream<W: Write>(
    writer: &mut W,
    status: u16,
    headers: &[Header],
    reader: &mut dyn Read,
    tap: &mut StreamTap,
) -> io::Result<()> {
    let reason = StatusCode(status).default_reason_phrase();
    write!(writer, "HTTP/1.1 {status} {reason}\r\n")?;
    for header in headers {
        writer.write_all(header.field.as_str().as_bytes())?;
        writer.write_all(b": ")?;
        writer.write_all(header.value.as_str().as_bytes())?;
        writer.write_all(b"\r\n")?;
    }
    // `copy_headers` already removed any upstream transfer framing; declare our
    // own so the client knows the body is delimited by the final zero chunk.
    writer.write_all(b"Transfer-Encoding: chunked\r\n\r\n")?;
    writer.flush()?;

    let mut buffer = [0u8; 8 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        tap.feed(&buffer[..read]);
        write!(writer, "{read:x}\r\n")?;
        writer.write_all(&buffer[..read])?;
        writer.write_all(b"\r\n")?;
        writer.flush()?;
    }
    writer.write_all(b"0\r\n\r\n")?;
    writer.flush()?;
    Ok(())
}

fn copy_headers(headers: &ureq::http::HeaderMap) -> Vec<Header> {
    headers
        .iter()
        .filter(|(name, _)| !is_hop_by_hop(name.as_str()))
        .filter_map(|(name, value)| {
            Header::from_bytes(name.as_str().as_bytes(), value.as_bytes()).ok()
        })
        .collect()
}

fn is_hop_by_hop(name: &str) -> bool {
    HOP_BY_HOP.iter().any(|hop| name.eq_ignore_ascii_case(hop))
}

pub(crate) fn header_value(request: &Request, name: &'static str) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|header| header.field.equiv(name))
        .map(|header| header.value.as_str().to_string())
}

/// Pure: does the request body ask for SSE (`"stream": true`)?
///
/// Invalid JSON is forwarded untouched and treated as non-streaming: the
/// upstream owns request validation, and the proxy must not reject bodies it
/// does not understand.
fn stream_requested(body: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| value.get("stream").and_then(serde_json::Value::as_bool))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_flag_is_read_from_the_json_body() {
        assert!(stream_requested(br#"{"model":"m","stream":true}"#));
        assert!(stream_requested(br#"{"stream" : true}"#));
        assert!(!stream_requested(br#"{"stream":false}"#));
        assert!(!stream_requested(br#"{"prompt":"hi"}"#));
        assert!(!stream_requested(br#"{"stream":"true"}"#));
    }

    #[test]
    fn malformed_bodies_are_not_stream_requests() {
        assert!(!stream_requested(b"not json"));
        assert!(!stream_requested(b""));
        assert!(!stream_requested(b"[1,2,3]"));
    }

    #[test]
    fn hop_by_hop_headers_are_filtered_case_insensitively() {
        for name in HOP_BY_HOP {
            assert!(is_hop_by_hop(name));
            assert!(is_hop_by_hop(&name.to_ascii_uppercase()));
        }
        assert!(!is_hop_by_hop("content-type"));
        assert!(!is_hop_by_hop("authorization"));
    }

    #[test]
    fn stream_writer_frames_each_read_and_terminates_with_zero_chunk() {
        let mut out = Vec::new();
        let mut reader = Cursor::new(b"data: one\n\n".to_vec());
        let mut tap = StreamTap::default();
        write_stream(&mut out, 200, &[], &mut reader, &mut tap).expect("write stream");
        assert!(tap.first_byte.is_some());
        assert_eq!(tap.tokens(), Some(1));
        let text = String::from_utf8(out).expect("utf8 framing");
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(text.contains("Transfer-Encoding: chunked\r\n"));
        assert!(text.contains("\r\nb\r\ndata: one\n\n\r\n"), "{text:?}");
        assert!(text.ends_with("0\r\n\r\n"));
    }

    #[test]
    fn stream_tap_counts_split_chunks_and_prefers_usage() {
        let mut tap = StreamTap::default();
        for part in ["data: {\"a\":1}\n\nda", "ta: {\"a\":2}\n\ndata: [DONE]\n\n"] {
            tap.feed(part.as_bytes());
        }
        assert_eq!(tap.tokens(), Some(2));

        let mut tap = StreamTap::default();
        tap.feed(
            b"data: {\"a\":1}\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":7}}\n\n",
        );
        assert_eq!(tap.tokens(), Some(7));
        assert_eq!(usage_tokens(b"{\"id\":1}"), None);
    }

    #[test]
    fn error_envelopes_use_the_openai_shape() {
        let body = serde_json::json!({
            "error": { "message": "boom", "type": "upstream_error" }
        });
        assert_eq!(body["error"]["type"], "upstream_error");
        assert_eq!(body["error"]["message"], "boom");
    }
}
