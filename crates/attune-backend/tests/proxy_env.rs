//! A user's HTTP(S)_PROXY must not route the localhost health check through a
//! proxy (own test binary: it sets process-wide env vars).

use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

use attune_backend::health_wait;

#[test]
fn health_check_ignores_proxy_env_vars() {
    for var in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ] {
        std::env::set_var(var, "http://127.0.0.1:9");
    }
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}/health", listener.local_addr().expect("addr"));
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut buf = [0_u8; 1024];
            let _ = stream.read(&mut buf);
            let body = br#"{"status":"ok"}"#;
            let _ = stream.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .as_bytes(),
            );
            let _ = stream.write_all(body);
        }
    });
    health_wait(&url, Duration::from_secs(5)).expect("healthy despite proxy env");
}
