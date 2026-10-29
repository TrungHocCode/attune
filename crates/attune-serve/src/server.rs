//! `tiny_http` listener and OpenAI-compatible routing (M5 slice 2).
//!
//! `tiny_http` already accepts connections on its own thread and reads request
//! headers off the caller thread; this module adds the openai surface
//! (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`) and
//! dispatches every request to its own worker thread so one long SSE stream
//! never serializes the next client behind it.
//!
//! Unknown paths answer `404` with an OpenAI-shaped envelope; a known path with
//! the wrong method answers `405`. Everything else is handed to
//! [`crate::proxy`].

use std::sync::Arc;
use std::time::Duration;

use tiny_http::{Method, Request, Server};

use crate::proxy::{self, ProxyState};
use crate::shutdown::ShutdownSignal;
use crate::ServeError;

/// How long the accept loop blocks before re-checking the shutdown flag.
///
/// The Ctrl+C handler wakes the loop immediately; this bound only covers the
/// rare missed wake-up and keeps shutdown latency below a fifth of a second.
const ACCEPT_POLL: Duration = Duration::from_millis(200);

/// Network settings for the OpenAI-compatible proxy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyConfig {
    /// Interface the proxy listens on (`--listen`).
    pub listen: String,
    /// Port the proxy listens on (`--listen-port`).
    pub listen_port: u16,
    /// Base URL of the running `llama-server`, e.g. `http://127.0.0.1:8081`.
    ///
    /// Shared by every worker through [`Proxy`]; plain HTTP is expected because
    /// the upstream is the local child process.
    pub upstream_base: String,
}

/// The OpenAI endpoints the proxy serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Route {
    ChatCompletions,
    Completions,
    Models,
    Health,
}

impl Route {
    /// `POST` routes carry a JSON request body; `GET` routes do not.
    pub(crate) fn is_post(self) -> bool {
        matches!(self, Self::ChatCompletions | Self::Completions)
    }

    /// The method this route accepts.
    fn allowed_method(self) -> &'static str {
        if self.is_post() {
            "POST"
        } else {
            "GET"
        }
    }

    /// Path shown in routing errors.
    fn path(self) -> &'static str {
        match self {
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Completions => "/v1/completions",
            Self::Models => "/v1/models",
            Self::Health => "/health",
        }
    }
}

/// What the router decided for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Routing {
    Serve(Route),
    MethodNotAllowed(Route),
    NotFound,
}

/// Pure routing table: method + path -> [`Routing`].
///
/// The query string is ignored for matching (it is still forwarded upstream
/// verbatim), and paths are matched exactly - no trailing-slash alias - so the
/// proxy surface stays predictable.
pub(crate) fn route(method: &Method, url: &str) -> Routing {
    let path = url.split('?').next().unwrap_or(url);
    let route = match path {
        "/v1/chat/completions" => Route::ChatCompletions,
        "/v1/completions" => Route::Completions,
        "/v1/models" => Route::Models,
        "/health" => Route::Health,
        _ => return Routing::NotFound,
    };
    let method_matches = if route.is_post() {
        *method == Method::Post
    } else {
        *method == Method::Get
    };
    if method_matches {
        Routing::Serve(route)
    } else {
        Routing::MethodNotAllowed(route)
    }
}

/// A bound proxy listener.
///
/// Cloning shares the listener and forwarding state; the production path keeps
/// a single value so dropping it after [`Proxy::run`] closes the socket.
#[derive(Clone)]
pub struct Proxy {
    server: Arc<Server>,
    state: Arc<ProxyState>,
    local_addr: String,
}

impl Proxy {
    /// Bind the proxy. Port `0` asks the OS for an ephemeral port (useful in
    /// tests); the actual address is available from [`Proxy::local_addr`].
    pub fn bind(config: &ProxyConfig) -> Result<Self, ServeError> {
        let host = config.listen.trim();
        if host.is_empty() {
            return Err(ServeError::InvalidConfig(
                "--listen must not be empty".to_string(),
            ));
        }
        if config.upstream_base.trim().is_empty() {
            return Err(ServeError::InvalidConfig(
                "upstream base URL must not be empty".to_string(),
            ));
        }

        let addr = format!("{host}:{}", config.listen_port);
        let server = Server::http(addr.as_str())
            .map_err(|error| ServeError::Proxy(format!("failed to bind {addr}: {error}")))?;
        let local_addr = server
            .server_addr()
            .to_ip()
            .map(|addr| addr.to_string())
            .unwrap_or_else(|| addr.clone());
        Ok(Self {
            server: Arc::new(server),
            state: Arc::new(ProxyState::new(config.upstream_base.clone())),
            local_addr,
        })
    }

    /// The address the listener actually bound (`host:port`).
    pub fn local_addr(&self) -> &str {
        &self.local_addr
    }

    /// Wake a blocking [`Proxy::run`] loop.
    pub fn unblock(&self) {
        self.server.unblock();
    }

    /// A signal-handler-safe wake-up hook that holds only a weak reference.
    ///
    /// The weak reference matters: a strong clone captured by the global ctrlc
    /// handler would keep the listener open after [`Proxy::run`] returns, so
    /// the socket would never close.
    pub fn unblocker(&self) -> impl Fn() + Send + 'static {
        let server = Arc::downgrade(&self.server);
        move || {
            if let Some(server) = server.upgrade() {
                server.unblock();
            }
        }
    }

    /// Serve until `shutdown` is requested, then close the listener.
    ///
    /// Returns `Ok(())` on a requested shutdown; the caller drops the listener
    /// (via this value) and then the [`attune_backend::ServerGuard`] so
    /// `llama-server` is killed and reaped.
    pub fn run(self, shutdown: &ShutdownSignal) -> Result<(), ServeError> {
        while !shutdown.is_requested() {
            match self.server.recv_timeout(ACCEPT_POLL) {
                Ok(Some(request)) => {
                    let state = Arc::clone(&self.state);
                    let spawned = std::thread::Builder::new()
                        .name("attune-proxy".to_string())
                        .spawn(move || handle_request(request, &state));
                    if let Err(error) = spawned {
                        // The request was dropped with the closure; tiny_http
                        // answers 500 and the next client is unaffected.
                        eprintln!("[attune] proxy: failed to spawn request worker: {error}");
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    if shutdown.is_requested() {
                        break;
                    }
                    eprintln!("[attune] proxy: accept failed: {error}");
                    std::thread::sleep(ACCEPT_POLL);
                }
            }
        }
        Ok(())
    }
}

fn handle_request(request: Request, state: &ProxyState) {
    match route(request.method(), request.url()) {
        Routing::Serve(route) => proxy::handle(request, route, state),
        Routing::MethodNotAllowed(route) => proxy::respond_error(
            request,
            405,
            "method_not_allowed",
            &format!("{} only accepts {}", route.path(), route.allowed_method()),
        ),
        Routing::NotFound => {
            let path = request
                .url()
                .split('?')
                .next()
                .unwrap_or(request.url())
                .to_string();
            proxy::respond_error(request, 404, "not_found", &format!("no route for `{path}`"))
        }
    }
}

// ---- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Mutex, MutexGuard};
    use std::thread::JoinHandle;
    use std::time::Instant;

    use tiny_http::{Header, Response};

    const UPSTREAM_CHAT: &str =
        "{\n  \"id\": \"chatcmpl-stub\",\n  \"choices\": [{\"index\": 0}]\n}\n";
    const UPSTREAM_ERROR: &str =
        "{\"error\":{\"message\":\"bad prompt\",\"type\":\"invalid_request\"}}";
    const UPSTREAM_MODELS: &str = "{\"object\":\"list\",\"data\":[{\"id\":\"stub-model\"}]}";
    const STREAM_CHUNKS: &[&str] = &["data: one\n\n", "data: two\n\ndata: [DONE]\n\n"];
    /// Delay between the first and second SSE chunk. Chosen well above the
    /// 1 s assertion so a proxy that buffered the whole upstream body cannot
    /// pass the streaming test, even on a loaded CI runner.
    const SLOW_CHUNK_DELAY: Duration = Duration::from_secs(2);

    // ---- routing (pure) ----------------------------------------------------

    #[test]
    fn routing_table_matches_the_openai_surface() {
        assert_eq!(
            route(&Method::Post, "/v1/chat/completions"),
            Routing::Serve(Route::ChatCompletions)
        );
        assert_eq!(
            route(&Method::Post, "/v1/completions"),
            Routing::Serve(Route::Completions)
        );
        assert_eq!(
            route(&Method::Get, "/v1/models"),
            Routing::Serve(Route::Models)
        );
        assert_eq!(
            route(&Method::Get, "/health"),
            Routing::Serve(Route::Health)
        );
    }

    #[test]
    fn routing_ignores_the_query_string_but_not_the_path() {
        assert_eq!(
            route(&Method::Get, "/v1/models?limit=1"),
            Routing::Serve(Route::Models)
        );
        assert_eq!(route(&Method::Get, "/v1/models/"), Routing::NotFound);
        assert_eq!(route(&Method::Get, "/v1/chat"), Routing::NotFound);
        assert_eq!(route(&Method::Post, "/"), Routing::NotFound);
    }

    #[test]
    fn routing_rejects_the_wrong_method_on_a_known_path() {
        assert_eq!(
            route(&Method::Get, "/v1/chat/completions"),
            Routing::MethodNotAllowed(Route::ChatCompletions)
        );
        assert_eq!(
            route(&Method::Post, "/v1/models"),
            Routing::MethodNotAllowed(Route::Models)
        );
        assert_eq!(
            route(&Method::Post, "/health"),
            Routing::MethodNotAllowed(Route::Health)
        );
    }

    #[test]
    fn post_routes_are_the_only_body_routes() {
        assert!(Route::ChatCompletions.is_post());
        assert!(Route::Completions.is_post());
        assert!(!Route::Models.is_post());
        assert!(!Route::Health.is_post());
    }

    // ---- end-to-end against a local stub ----------------------------------

    #[test]
    fn forwards_non_stream_responses_byte_for_byte() {
        let stub = StubUpstream::start();
        let proxy = TestProxy::start(&format!("http://{}", stub.addr));

        let body = r#"{"model":"stub","messages":[{"role":"user","content":"hi"}]}"#;
        let request = post_request(&proxy.addr, "/v1/chat/completions", body, None);
        let (response, _) = http_raw(&proxy.addr, &request);
        assert_eq!(response.status, 200);
        assert_eq!(response.body, UPSTREAM_CHAT.as_bytes());
        assert_eq!(
            test_header(&response.headers, "content-type"),
            Some("application/json")
        );

        // GET routes and 404/405 are served by the same listener.
        let models = http_raw(
            &proxy.addr,
            &format!(
                "GET /v1/models HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
                proxy.addr
            ),
        )
        .0;
        assert_eq!(models.status, 200);
        assert_eq!(models.body, UPSTREAM_MODELS.as_bytes());

        let health = http_raw(
            &proxy.addr,
            &format!(
                "GET /health HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
                proxy.addr
            ),
        )
        .0;
        assert_eq!(health.status, 200);
        assert_eq!(health.body, br#"{"status":"ok"}"#);
    }

    #[test]
    fn forwards_upstream_error_status_and_body_verbatim() {
        let stub = StubUpstream::start();
        let proxy = TestProxy::start(&format!("http://{}", stub.addr));

        let request = post_request(
            &proxy.addr,
            "/v1/completions",
            r#"{"prompt":"x","fail":true}"#,
            None,
        );
        let (response, _) = http_raw(&proxy.addr, &request);
        assert_eq!(response.status, 400);
        assert_eq!(response.body, UPSTREAM_ERROR.as_bytes());
    }

    #[test]
    fn streams_sse_chunks_in_order_without_buffering_the_body() {
        let stub = StubUpstream::start();
        let proxy = TestProxy::start(&format!("http://{}", stub.addr));

        let body = r#"{"model":"stub","messages":[],"stream":true}"#;
        let request = post_request(
            &proxy.addr,
            "/v1/chat/completions",
            body,
            Some("Bearer slice2-test"),
        );

        let started = Instant::now();
        let (response, first_chunk) = http_raw(&proxy.addr, &request);
        assert_eq!(response.status, 200);
        assert_eq!(
            test_header(&response.headers, "content-type"),
            Some("text/event-stream")
        );
        assert_eq!(
            test_header(&response.headers, "transfer-encoding"),
            Some("chunked")
        );

        // The first chunk must arrive while the stub is still sleeping before
        // the second one; a buffering proxy would take at least
        // SLOW_CHUNK_DELAY before sending anything.
        let first_chunk = first_chunk.expect("chunked response must contain a chunk");
        assert!(
            first_chunk.duration_since(started) < Duration::from_secs(1),
            "first SSE chunk took {:?}; the proxy buffered the body",
            first_chunk.duration_since(started)
        );

        assert_eq!(response.body, STREAM_CHUNKS.concat().as_bytes());
        assert!(
            stub.authorizations()
                .iter()
                .any(|value| value.as_deref() == Some("Bearer slice2-test")),
            "Authorization was not forwarded upstream: {:?}",
            stub.authorizations()
        );
    }

    #[test]
    fn unknown_paths_answer_404_in_the_openai_shape() {
        let stub = StubUpstream::start();
        let proxy = TestProxy::start(&format!("http://{}", stub.addr));

        let (response, _) = http_raw(
            &proxy.addr,
            &format!(
                "GET /v1/does-not-exist HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
                proxy.addr
            ),
        );
        assert_eq!(response.status, 404);
        let error: serde_json::Value =
            serde_json::from_slice(&response.body).expect("404 body must be JSON");
        assert_eq!(error["error"]["type"], "not_found");

        let (wrong_method, _) = http_raw(
            &proxy.addr,
            &format!(
                "GET /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
                proxy.addr
            ),
        );
        assert_eq!(wrong_method.status, 405);
        let error: serde_json::Value =
            serde_json::from_slice(&wrong_method.body).expect("405 body must be JSON");
        assert_eq!(error["error"]["type"], "method_not_allowed");
    }

    #[test]
    fn a_dead_upstream_becomes_502_and_the_proxy_keeps_serving() {
        let dead_port = {
            let listener = TcpListener::bind("127.0.0.1:0").expect("reserve a port");
            listener.local_addr().expect("local addr").port()
        };
        // The listener is dropped here, so connecting to `dead_port` is refused.
        let proxy = TestProxy::start(&format!("http://127.0.0.1:{dead_port}"));

        let request = post_request(
            &proxy.addr,
            "/v1/chat/completions",
            r#"{"model":"stub","messages":[]}"#,
            None,
        );
        let (response, _) = http_raw(&proxy.addr, &request);
        assert_eq!(response.status, 502);
        let error: serde_json::Value =
            serde_json::from_slice(&response.body).expect("502 body must be JSON");
        assert_eq!(error["error"]["type"], "upstream_error");

        // The worker failure must not take the listener down.
        let (again, _) = http_raw(&proxy.addr, &request);
        assert_eq!(again.status, 502);
        let (not_found, _) = http_raw(
            &proxy.addr,
            &format!(
                "GET /nope HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
                proxy.addr
            ),
        );
        assert_eq!(not_found.status, 404);
    }

    // ---- stub upstream -----------------------------------------------------

    struct StubUpstream {
        addr: String,
        stop: Arc<AtomicBool>,
        authorizations: Arc<Mutex<Vec<Option<String>>>>,
        handle: Option<JoinHandle<()>>,
    }

    impl StubUpstream {
        fn start() -> Self {
            let server = tiny_http::Server::http("127.0.0.1:0").expect("bind stub upstream");
            let addr = server
                .server_addr()
                .to_ip()
                .expect("stub must bind an IP address")
                .to_string();
            let stop = Arc::new(AtomicBool::new(false));
            let authorizations = Arc::new(Mutex::new(Vec::new()));
            let thread_stop = Arc::clone(&stop);
            let thread_authorizations = Arc::clone(&authorizations);
            let handle = std::thread::spawn(move || {
                while !thread_stop.load(Ordering::SeqCst) {
                    match server.recv_timeout(Duration::from_millis(20)) {
                        Ok(Some(request)) => {
                            let value = request
                                .headers()
                                .iter()
                                .find(|header| header.field.equiv("Authorization"))
                                .map(|header| header.value.as_str().to_string());
                            lock(&thread_authorizations).push(value);
                            respond_stub(request);
                        }
                        Ok(None) => {}
                        Err(_) => break,
                    }
                }
            });
            Self {
                addr,
                stop,
                authorizations,
                handle: Some(handle),
            }
        }

        fn authorizations(&self) -> MutexGuard<'_, Vec<Option<String>>> {
            lock(&self.authorizations)
        }
    }

    impl Drop for StubUpstream {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn respond_stub(mut request: Request) {
        let method = request.method().clone();
        let url = request.url().to_string();
        let path = url.split('?').next().unwrap_or(&url).to_string();
        let mut body = Vec::new();
        if let Err(error) = request.as_reader().read_to_end(&mut body) {
            eprintln!("stub: failed to read request body: {error}");
        }
        let text = String::from_utf8_lossy(&body);

        match (method, path.as_str()) {
            (Method::Post, "/v1/chat/completions") if text.contains("\"stream\":true") => {
                respond_sse(request);
            }
            (Method::Post, "/v1/chat/completions") => {
                let _ = request.respond(json_response(200, UPSTREAM_CHAT));
            }
            (Method::Post, "/v1/completions") if text.contains("\"fail\":true") => {
                let _ = request.respond(json_response(400, UPSTREAM_ERROR));
            }
            (Method::Post, "/v1/completions") => {
                let _ = request.respond(json_response(200, "{\"choices\":[]}"));
            }
            (Method::Get, "/v1/models") => {
                let _ = request.respond(json_response(200, UPSTREAM_MODELS));
            }
            (Method::Get, "/health") => {
                let _ = request.respond(json_response(200, "{\"status\":\"ok\"}"));
            }
            _ => {
                let _ = request.respond(json_response(404, "{\"error\":\"stub 404\"}"));
            }
        }
    }

    /// Write a chunked SSE response with an explicit flush per chunk.
    ///
    /// `tiny_http`'s own chunked encoder buffers up to 8 KiB, so a stub built
    /// on `Response::new` would itself hold the first chunk back until EOF and
    /// hide the proxy behaviour under test.
    fn respond_sse(request: Request) {
        let mut writer = request.into_writer();
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                    Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
        if writer.write_all(head.as_bytes()).is_err() || writer.flush().is_err() {
            return;
        }
        for (index, chunk) in STREAM_CHUNKS.iter().enumerate() {
            if index > 0 {
                std::thread::sleep(SLOW_CHUNK_DELAY);
            }
            let bytes = chunk.as_bytes();
            let framed = format!("{:x}\r\n", bytes.len());
            let write = writer
                .write_all(framed.as_bytes())
                .and_then(|()| writer.write_all(bytes))
                .and_then(|()| writer.write_all(b"\r\n"))
                .and_then(|()| writer.flush());
            if write.is_err() {
                return;
            }
        }
        let _ = writer.write_all(b"0\r\n\r\n");
        let _ = writer.flush();
    }

    fn json_response(status: u16, body: &str) -> Response<std::io::Cursor<Vec<u8>>> {
        Response::from_data(body.as_bytes().to_vec())
            .with_status_code(status)
            .with_header(
                Header::from_bytes("Content-Type", "application/json").expect("static header"),
            )
    }

    // ---- proxy harness -----------------------------------------------------

    struct TestProxy {
        addr: String,
        shutdown: ShutdownSignal,
        proxy: Proxy,
        handle: Option<JoinHandle<()>>,
    }

    impl TestProxy {
        fn start(upstream_base: &str) -> Self {
            let config = ProxyConfig {
                listen: "127.0.0.1".to_string(),
                listen_port: 0,
                upstream_base: upstream_base.to_string(),
            };
            let proxy = Proxy::bind(&config).expect("bind proxy");
            let addr = proxy.local_addr().to_string();
            let runner = proxy.clone();
            let shutdown = ShutdownSignal::new();
            let thread_shutdown = shutdown.clone();
            let handle = std::thread::spawn(move || {
                if let Err(error) = runner.run(&thread_shutdown) {
                    eprintln!("test proxy stopped with error: {error}");
                }
            });
            Self {
                addr,
                shutdown,
                proxy,
                handle: Some(handle),
            }
        }
    }

    impl Drop for TestProxy {
        fn drop(&mut self) {
            self.shutdown.request();
            self.proxy.unblock();
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    // ---- raw HTTP client ---------------------------------------------------

    struct RawResponse {
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    }

    fn post_request(addr: &str, path: &str, body: &str, authorization: Option<&str>) -> String {
        let auth_line = authorization
            .map(|value| format!("Authorization: {value}\r\n"))
            .unwrap_or_default();
        format!(
            "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
             {auth_line}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    /// Send a raw request and return the parsed response plus the instant the
    /// first non-empty chunk arrived (chunked responses) for streaming timing.
    fn http_raw(addr: &str, request: &str) -> (RawResponse, Option<Instant>) {
        let mut stream = TcpStream::connect(addr).expect("connect to proxy");
        stream
            .set_read_timeout(Some(Duration::from_secs(15)))
            .expect("set read timeout");
        stream.write_all(request.as_bytes()).expect("write request");
        stream.flush().expect("flush request");

        let mut reader = BufReader::new(stream);
        let mut status_line = String::new();
        reader
            .read_line(&mut status_line)
            .expect("read status line");
        let status = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|code| code.parse::<u16>().ok())
            .expect("parse status code");

        let mut headers = Vec::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).expect("read response header");
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
            }
        }

        let chunked = test_header(&headers, "transfer-encoding")
            .map(|value| value.to_ascii_lowercase().contains("chunked"))
            .unwrap_or(false);
        let (body, first_chunk) = if chunked {
            read_chunked(&mut reader)
        } else {
            let length = test_header(&headers, "content-length")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            let mut body = vec![0u8; length];
            reader.read_exact(&mut body).expect("read fixed body");
            (body, None)
        };

        (
            RawResponse {
                status,
                headers,
                body,
            },
            first_chunk,
        )
    }

    fn read_chunked(reader: &mut BufReader<TcpStream>) -> (Vec<u8>, Option<Instant>) {
        let mut body = Vec::new();
        let mut first_chunk = None;
        loop {
            let mut size_line = String::new();
            reader.read_line(&mut size_line).expect("read chunk size");
            let size_text = size_line.trim();
            let size = usize::from_str_radix(size_text.split(';').next().unwrap_or(size_text), 16)
                .expect("parse chunk size");
            if size == 0 {
                // Consume the final CRLF (or trailer terminator).
                let mut end = String::new();
                let _ = reader.read_line(&mut end);
                break;
            }
            let mut chunk = vec![0u8; size];
            reader.read_exact(&mut chunk).expect("read chunk data");
            if first_chunk.is_none() && !chunk.is_empty() {
                first_chunk = Some(Instant::now());
            }
            body.extend_from_slice(&chunk);
            let mut crlf = [0u8; 2];
            reader.read_exact(&mut crlf).expect("read chunk crlf");
        }
        (body, first_chunk)
    }

    fn test_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
        headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}
