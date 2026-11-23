//! Proxy-measured serving metrics (M5 slice 3).
//!
//! Everything is measured at the proxy; nothing is instrumented in
//! `llama-server`. Latency/throughput samples live in bounded windows of the
//! most recent [`WINDOW`] requests so memory stays flat on a long-running
//! server. Counters (`total`, `ok`, `error`) are lifetime values.
//!
//! State is a single `Mutex`; each request touches it twice (start, finish), so
//! contention is irrelevant next to LLM inference time.

use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

/// How many of the most recent samples each statistic is computed over.
pub const WINDOW: usize = 1024;

/// One completed forwarded request, as seen by the proxy.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// Upstream answered 2xx and the body was delivered.
    pub ok: bool,
    /// Request received -> first upstream body byte.
    pub ttft: Duration,
    /// Request received -> last byte delivered.
    pub total: Duration,
    /// Completion tokens, when the upstream or the SSE chunk count gave one.
    pub tokens: Option<u64>,
    /// Streamed (SSE) response: decode time is `total - ttft`. Non-stream
    /// responses only arrive when generation is done, so decode time is `total`.
    pub streamed: bool,
}

/// `{count, mean, p50, p95}` over the current window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub count: usize,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
}

/// Mean and nearest-rank percentiles; `None` for an empty window.
pub fn stats(values: &VecDeque<f64>) -> Option<Stats> {
    if values.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = values.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    let rank = |p: f64| {
        let index = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1;
        sorted[index]
    };
    Some(Stats {
        count: sorted.len(),
        mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
        p50: rank(0.50),
        p95: rank(0.95),
    })
}

#[derive(Default)]
struct Inner {
    ok: u64,
    error: u64,
    in_flight: u64,
    ttft_ms: VecDeque<f64>,
    tpot_ms: VecDeque<f64>,
    tok_s: VecDeque<f64>,
}

fn push(window: &mut VecDeque<f64>, value: f64) {
    if window.len() == WINDOW {
        window.pop_front();
    }
    window.push_back(value);
}

/// Shared, thread-safe metrics registry.
pub struct Metrics {
    started: Instant,
    inner: Mutex<Inner>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            inner: Mutex::new(Inner::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Mark a forwarded request as started. Dropping the tracker without
    /// [`Tracker::finish`] counts it as an error (e.g. a panicking worker).
    pub fn begin(&self) -> Tracker<'_> {
        self.lock().in_flight += 1;
        Tracker {
            metrics: self,
            sample: None,
        }
    }

    /// Point-in-time copy used by both renderers.
    pub fn snapshot(&self, rss: Option<u64>, ram_available: u64) -> Snapshot {
        let inner = self.lock();
        Snapshot {
            uptime_s: self.started.elapsed().as_secs_f64(),
            ok: inner.ok,
            error: inner.error,
            in_flight: inner.in_flight,
            ttft_ms: stats(&inner.ttft_ms),
            tpot_ms: stats(&inner.tpot_ms),
            tok_s: stats(&inner.tok_s),
            rss,
            ram_available,
        }
    }
}

/// RAII handle for one in-flight request.
pub struct Tracker<'a> {
    metrics: &'a Metrics,
    sample: Option<Sample>,
}

impl Tracker<'_> {
    /// Record the outcome; failed requests (`ok == false`) only bump the
    /// error counter and add no latency samples.
    pub fn finish(mut self, sample: Sample) {
        self.sample = Some(sample);
    }
}

impl Drop for Tracker<'_> {
    fn drop(&mut self) {
        let mut inner = self.metrics.lock();
        inner.in_flight = inner.in_flight.saturating_sub(1);
        let Some(sample) = self.sample.filter(|sample| sample.ok) else {
            inner.error += 1;
            return;
        };
        inner.ok += 1;
        push(&mut inner.ttft_ms, sample.ttft.as_secs_f64() * 1e3);
        let decode = if sample.streamed {
            sample.total.saturating_sub(sample.ttft)
        } else {
            sample.total
        };
        // Skip rate stats when there is nothing to divide by.
        if let Some(tokens) = sample.tokens.filter(|&tokens| tokens > 0) {
            if !decode.is_zero() {
                push(&mut inner.tok_s, tokens as f64 / decode.as_secs_f64());
                push(
                    &mut inner.tpot_ms,
                    decode.as_secs_f64() * 1e3 / tokens as f64,
                );
            }
        }
    }
}

/// Immutable view of the registry at one instant.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub uptime_s: f64,
    pub ok: u64,
    pub error: u64,
    pub in_flight: u64,
    pub ttft_ms: Option<Stats>,
    pub tpot_ms: Option<Stats>,
    pub tok_s: Option<Stats>,
    pub rss: Option<u64>,
    pub ram_available: u64,
}

fn stats_json(stats: Option<Stats>) -> Value {
    match stats {
        Some(s) => json!({"count": s.count, "mean": s.mean, "p50": s.p50, "p95": s.p95}),
        None => json!({"count": 0, "mean": null, "p50": null, "p95": null}),
    }
}

impl Snapshot {
    /// Body of `GET /v1/metrics`.
    pub fn to_json(&self) -> Value {
        json!({
            "uptime_s": self.uptime_s,
            "requests": {
                "total": self.ok + self.error,
                "ok": self.ok,
                "error": self.error,
                "in_flight": self.in_flight,
            },
            "ttft_ms": stats_json(self.ttft_ms),
            "tpot_ms": stats_json(self.tpot_ms),
            "tok_s": stats_json(self.tok_s),
            "llama_server_rss_bytes": self.rss,
            "ram_available_bytes": self.ram_available,
        })
    }

    /// Body of `GET /metrics` (Prometheus text exposition format).
    pub fn to_prometheus(&self) -> String {
        let mut out = String::new();
        let mut line = |text: String| {
            out.push_str(&text);
            out.push('\n');
        };
        line("# TYPE attune_uptime_seconds gauge".into());
        line(format!("attune_uptime_seconds {}", self.uptime_s));
        line("# TYPE attune_requests_total counter".into());
        line(format!(
            "attune_requests_total{{status=\"ok\"}} {}",
            self.ok
        ));
        line(format!(
            "attune_requests_total{{status=\"error\"}} {}",
            self.error
        ));
        line("# TYPE attune_requests_in_flight gauge".into());
        line(format!("attune_requests_in_flight {}", self.in_flight));
        // Summaries over the recent window; latencies converted ms -> seconds.
        for (name, stats, scale) in [
            ("attune_ttft_seconds", self.ttft_ms, 1e-3),
            ("attune_tpot_seconds", self.tpot_ms, 1e-3),
            ("attune_tokens_per_second", self.tok_s, 1.0),
        ] {
            line(format!("# TYPE {name} summary"));
            if let Some(s) = stats {
                line(format!("{name}{{quantile=\"0.5\"}} {}", s.p50 * scale));
                line(format!("{name}{{quantile=\"0.95\"}} {}", s.p95 * scale));
                line(format!("{name}_sum {}", s.mean * s.count as f64 * scale));
            }
            line(format!("{name}_count {}", stats.map_or(0, |s| s.count)));
        }
        if let Some(rss) = self.rss {
            line("# TYPE attune_llama_server_rss_bytes gauge".into());
            line(format!("attune_llama_server_rss_bytes {rss}"));
        }
        line("# TYPE attune_ram_available_bytes gauge".into());
        line(format!("attune_ram_available_bytes {}", self.ram_available));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(values: &[f64]) -> VecDeque<f64> {
        values.iter().copied().collect()
    }

    fn sample(total_ms: u64, ttft_ms: u64, tokens: Option<u64>) -> Sample {
        Sample {
            ok: true,
            ttft: Duration::from_millis(ttft_ms),
            total: Duration::from_millis(total_ms),
            tokens,
            streamed: true,
        }
    }

    #[test]
    fn stats_of_empty_one_and_many() {
        assert_eq!(stats(&window(&[])), None);
        let one = stats(&window(&[7.0])).expect("one sample");
        assert_eq!((one.count, one.mean, one.p50, one.p95), (1, 7.0, 7.0, 7.0));
        let many: Vec<f64> = (1..=100).map(f64::from).collect();
        let s = stats(&window(&many)).expect("many samples");
        assert_eq!(s.count, 100);
        assert!((s.mean - 50.5).abs() < 1e-9);
        assert_eq!((s.p50, s.p95), (50.0, 95.0));
    }

    #[test]
    fn window_evicts_oldest_past_capacity() {
        let metrics = Metrics::new();
        for i in 0..(WINDOW as u64 + 10) {
            metrics.begin().finish(sample(100 + i, i + 1, Some(10)));
        }
        let snap = metrics.snapshot(None, 0);
        let ttft = snap.ttft_ms.expect("ttft stats");
        assert_eq!(ttft.count, WINDOW);
        // Samples 1..=10 ms were evicted, so the smallest left is 11 ms.
        assert!(ttft.mean > 11.0);
        assert_eq!(snap.ok, WINDOW as u64 + 10);
    }

    #[test]
    fn rates_are_skipped_without_tokens_or_decode_time_and_errors_add_no_samples() {
        let metrics = Metrics::new();
        metrics.begin().finish(sample(100, 100, Some(5))); // zero decode time
        metrics.begin().finish(sample(200, 100, Some(0))); // zero tokens
        metrics.begin().finish(sample(200, 100, None));
        metrics.begin().finish(Sample {
            ok: false,
            ..sample(1, 1, Some(1))
        });
        drop(metrics.begin()); // abandoned -> error
        let snap = metrics.snapshot(None, 0);
        assert_eq!((snap.ok, snap.error, snap.in_flight), (3, 2, 0));
        assert_eq!(snap.ttft_ms.map(|s| s.count), Some(3));
        assert!(snap.tok_s.is_none() && snap.tpot_ms.is_none());

        metrics.begin().finish(sample(300, 100, Some(10)));
        let snap = metrics.snapshot(None, 0);
        let (tok_s, tpot) = (snap.tok_s.expect("tok_s"), snap.tpot_ms.expect("tpot"));
        assert!((tok_s.mean - 50.0).abs() < 1e-9); // 10 tokens / 0.2 s
        assert!((tpot.mean - 20.0).abs() < 1e-9);
    }

    #[test]
    fn non_stream_decode_time_is_total_latency() {
        let metrics = Metrics::new();
        metrics.begin().finish(Sample {
            streamed: false,
            ..sample(500, 500, Some(10))
        });
        let tok_s = metrics.snapshot(None, 0).tok_s.expect("tok_s");
        assert!((tok_s.mean - 20.0).abs() < 1e-9);
    }

    #[test]
    fn json_has_null_stats_when_empty() {
        let value = Metrics::new().snapshot(None, 42).to_json();
        assert_eq!(value["requests"]["total"], 0);
        assert_eq!(value["ttft_ms"]["count"], 0);
        assert!(value["ttft_ms"]["p95"].is_null());
        assert!(value["llama_server_rss_bytes"].is_null());
        assert_eq!(value["ram_available_bytes"], 42);
    }

    #[test]
    fn prometheus_text_has_the_key_lines() {
        let metrics = Metrics::new();
        metrics.begin().finish(sample(300, 100, Some(10)));
        let text = metrics.snapshot(Some(1234), 99).to_prometheus();
        for expected in [
            "# TYPE attune_requests_total counter",
            "attune_requests_total{status=\"ok\"} 1",
            "attune_requests_total{status=\"error\"} 0",
            "attune_requests_in_flight 0",
            "attune_ttft_seconds{quantile=\"0.5\"} 0.1",
            "attune_ttft_seconds_count 1",
            "attune_tokens_per_second{quantile=\"0.95\"} 50",
            "attune_llama_server_rss_bytes 1234",
            "attune_ram_available_bytes 99",
        ] {
            assert!(
                text.lines().any(|l| l == expected),
                "missing `{expected}` in\n{text}"
            );
        }
    }
}
