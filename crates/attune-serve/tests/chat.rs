//! End-to-end streaming chat against a std `TcpListener` stub.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use attune_serve::{ChatClient, ChatDelta};

const SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"th\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"ink\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
    ": keep-alive\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{}}],\"timings\":{\"predicted_n\":2,\"predicted_per_second\":42.5}}\n\n",
    "data: [DONE]\n\n",
);

/// Serve one request and return the raw request bytes after answering it.
fn start_stub(body: &'static str) -> (String, JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub");
    let addr = listener.local_addr().expect("stub addr");
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept request");
        let request = read_request(&mut stream);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream
            .write_all(response.as_bytes())
            .expect("write response");
        stream.flush().expect("flush response");
        request
    });
    (format!("http://{addr}"), handle)
}

fn read_request(stream: &mut TcpStream) -> String {
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("set read timeout");
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 1024];
    let header_end = loop {
        let read = stream.read(&mut chunk).expect("read request");
        assert!(read > 0, "connection closed before the headers ended");
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(position) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
    };
    let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.eq_ignore_ascii_case("content-length") {
                value.trim().parse::<usize>().ok()
            } else {
                None
            }
        })
        .unwrap_or(0);
    while buffer.len() < header_end + length {
        let read = stream.read(&mut chunk).expect("read request body");
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    String::from_utf8_lossy(&buffer).to_string()
}

#[test]
fn streams_deltas_and_reports_timings() {
    let (url, handle) = start_stub(SSE_BODY);
    let mut client = ChatClient::new(url, Some("secret".to_string()), Some("be nice".to_string()));

    let mut deltas = Vec::new();
    let stats = client
        .send("hi", &mut |delta: ChatDelta<'_>| match delta {
            ChatDelta::Thinking(text) => deltas.push(("think", text.to_string())),
            ChatDelta::Text(text) => deltas.push(("text", text.to_string())),
        })
        .expect("send");

    assert_eq!(
        deltas,
        vec![
            ("think", "th".to_string()),
            ("think", "ink".to_string()),
            ("text", "Hel".to_string()),
            ("text", "lo".to_string()),
        ]
    );
    assert_eq!(stats.tokens, 2);
    assert!((stats.tokens_per_s - 42.5).abs() < 1e-9);

    let history = client.history();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].role, "system");
    assert_eq!(history[1].role, "user");
    assert_eq!(history[1].content, "hi");
    assert_eq!(history[2].role, "assistant");
    assert_eq!(history[2].content, "Hello");

    let request = handle.join().expect("stub finished");
    assert!(request.contains("\"stream\":true"), "{request}");
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer secret"),
        "{request}"
    );
}

#[test]
fn a_refused_connection_points_at_attune_serve() {
    let port = {
        let listener = TcpListener::bind("127.0.0.1:0").expect("reserve a port");
        listener.local_addr().expect("local addr").port()
    };
    let url = format!("http://127.0.0.1:{port}");
    let mut client = ChatClient::new(url, None, None);
    let error = client
        .send("hi", &mut |_delta| {})
        .expect_err("no server should be listening");
    assert!(error.to_string().contains("No attune server"), "{error}");
}
