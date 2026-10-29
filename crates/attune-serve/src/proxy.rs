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
use std::time::Duration;

use tiny_http::{Header, Request, Response, ResponseBox, StatusCode};
use ureq::Agent;

use crate::server::Route;

/// Local llama-server normally answers immediately; a short connect timeout
/// keeps a dead upstream from parking a worker thread. No global/body timeout
/// is set because an SSE stream can legitimately stay open for minutes.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

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
}

impl ProxyState {
    pub(crate) fn new(upstream_base: String) -> Self {
        let agent = Agent::config_builder()
            // Upstream 4xx/5xx responses are forwarded verbatim, so ureq must
            // hand them back as responses instead of `Error::StatusCode`.
            .http_status_as_error(false)
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .build()
            .into();
        Self {
            agent,
            upstream_base: upstream_base.trim_end_matches('/').to_string(),
        }
    }
}

/// Forward one routed request and write the client response.
///
/// Never returns an error: any failure is rendered as an OpenAI-shaped error
/// response so a bad request cannot kill the worker (or the proxy).
pub(crate) fn handle(mut request: Request, route: Route, state: &ProxyState) {
    let authorization = header_value(&request, "Authorization");
    let mut body = Vec::new();
    if route.is_post() {
        if let Err(error) = request.as_reader().read_to_end(&mut body) {
            respond_error(
                request,
                400,
                "invalid_request",
                &format!("failed to read request body: {error}"),
            );
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
        Ok(upstream) => respond(request, upstream),
        Err(error) => respond_error(
            request,
            502,
            "upstream_error",
            &format!("upstream request failed: {error}"),
        ),
    }
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
        },
    );
}

// ---- forwarding -------------------------------------------------------------

/// A fully resolved upstream response, either buffered or still streaming.
struct Upstream {
    status: u16,
    headers: Vec<Header>,
    body: UpstreamBody,
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
    let headers = copy_headers(response.headers());
    // 204 has no body and `tiny_http` forbids chunked framing there, so only
    // pipe a 2xx stream when a body is actually legal.
    let streaming = wants_stream && (200..300).contains(&status) && status != 204;
    if streaming {
        Ok(Upstream {
            status,
            headers,
            body: UpstreamBody::Stream(response.into_body().into_reader()),
        })
    } else {
        let mut response = response;
        let body = response.body_mut().read_to_vec()?;
        Ok(Upstream {
            status,
            headers,
            body: UpstreamBody::Buffered(body),
        })
    }
}

fn respond(request: Request, upstream: Upstream) {
    match upstream.body {
        UpstreamBody::Buffered(body) => {
            let length = body.len();
            let response: ResponseBox = Response::new(
                StatusCode(upstream.status),
                upstream.headers,
                Box::new(Cursor::new(body)) as Box<dyn Read + Send>,
                Some(length),
                None,
            );
            if let Err(error) = request.respond(response) {
                eprintln!("[attune] proxy: failed to send response: {error}");
            }
        }
        UpstreamBody::Stream(reader) => {
            respond_stream(request, upstream.status, &upstream.headers, reader);
        }
    }
}

/// Write a chunked response with one flush per upstream read.
fn respond_stream(
    request: Request,
    status: u16,
    headers: &[Header],
    mut reader: ureq::BodyReader<'static>,
) {
    let mut writer = request.into_writer();
    if let Err(error) = write_stream(&mut writer, status, headers, &mut reader) {
        // A client that disconnects surfaces here as a broken pipe; stop
        // pulling the upstream body and let the reader drop close that side.
        eprintln!("[attune] proxy: streaming response ended early: {error}");
    }
}

fn write_stream<W: Write>(
    writer: &mut W,
    status: u16,
    headers: &[Header],
    reader: &mut dyn Read,
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

fn header_value(request: &Request, name: &'static str) -> Option<String> {
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
        write_stream(&mut out, 200, &[], &mut reader).expect("write stream");
        let text = String::from_utf8(out).expect("utf8 framing");
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(text.contains("Transfer-Encoding: chunked\r\n"));
        assert!(text.contains("\r\nb\r\ndata: one\n\n\r\n"), "{text:?}");
        assert!(text.ends_with("0\r\n\r\n"));
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
