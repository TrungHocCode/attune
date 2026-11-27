//! Download consent and progress for `serve` and `tune --verify` (M6).

use std::io::{BufRead, IsTerminal, Write};
use std::sync::atomic::{AtomicU64, Ordering};

pub use attune_serve::DownloadConsent;

/// `--yes` wins; otherwise ask only when a person can answer.
pub fn consent(yes: bool, interactive: bool) -> DownloadConsent {
    if yes {
        DownloadConsent::Yes
    } else if interactive {
        DownloadConsent::Ask(ask)
    } else {
        DownloadConsent::No
    }
}

/// Consent for this process: interactive when stdin is a terminal.
pub fn consent_for_stdin(yes: bool) -> DownloadConsent {
    consent(yes, std::io::stdin().is_terminal())
}

fn is_yes(answer: &str) -> bool {
    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

fn ask(_size: u64, what: &str) -> bool {
    eprint!("Download {what}? [y/N] ");
    let _ = std::io::stderr().flush();
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer).is_ok() && is_yes(&answer)
}

static LAST_STEP: AtomicU64 = AtomicU64::new(u64::MAX);

/// Human size: whole MB below 1 GB, two-decimal GB at or above it.
fn format_size(bytes: u64) -> String {
    if bytes < 1_000_000_000 {
        format!("{} MB", bytes / 1_000_000)
    } else {
        format!("{:.2} GB", bytes as f64 / 1e9)
    }
}

/// One progress line: `  412 MB / 2.50 GB (16%)`.
///
/// A zero total prints just the bytes done. Values below 1 GB never round to
/// `0.00 GB` because they stay in MB.
pub fn format_progress(done: u64, total: u64) -> String {
    if total == 0 {
        return format!("  {}", format_size(done));
    }
    let percent = (u128::from(done) * 100 / u128::from(total)).min(100);
    format!(
        "  {} / {} ({}%)",
        format_size(done),
        format_size(total),
        percent
    )
}

/// Percent rounded down to the nearest 10% step, for one-line-at-a-time output.
fn step_of(done: u64, total: u64) -> Option<u64> {
    if total == 0 {
        return None;
    }
    let percent = (u128::from(done) * 100 / u128::from(total)).min(100) as u64;
    Some(percent / 10 * 10)
}

/// The step to report when the terminal is not redrawable, or `None` when the
/// last reported step already covers this call.
fn crossed_step(last: u64, done: u64, total: u64) -> Option<u64> {
    step_of(done, total).filter(|step| *step != last)
}

/// Progress on stderr: one redrawn line on a terminal, one line per 10% step
/// otherwise (plus the final 100% line).
pub fn progress(done: u64, total: u64) {
    if std::io::stderr().is_terminal() {
        eprint!("\r{}", format_progress(done, total));
        if done == total {
            eprintln!();
        }
        let _ = std::io::stderr().flush();
        return;
    }

    if done == 0 {
        LAST_STEP.store(u64::MAX, Ordering::Relaxed);
    }
    let last = LAST_STEP.load(Ordering::Relaxed);
    if let Some(step) = crossed_step(last, done, total) {
        LAST_STEP.store(step, Ordering::Relaxed);
        eprintln!("{}", format_progress(done, total));
    }
}

/// No output (`--json`).
pub fn quiet(_done: u64, _total: u64) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_y_or_yes_confirms() {
        for yes in ["y", "Y", "yes", " YES\n"] {
            assert!(is_yes(yes), "{yes:?}");
        }
        for no in ["", "\n", "n", "no", "yep", "sure"] {
            assert!(!is_yes(no), "{no:?}");
        }
    }

    #[test]
    fn consent_follows_yes_then_terminal() {
        assert!(matches!(consent(true, false), DownloadConsent::Yes));
        assert!(matches!(consent(false, false), DownloadConsent::No));
        assert!(matches!(consent(false, true), DownloadConsent::Ask(_)));
    }

    #[test]
    fn progress_uses_mb_below_one_gb() {
        assert_eq!(
            format_progress(412_000_000, 2_500_000_000),
            "  412 MB / 2.50 GB (16%)"
        );
        assert_eq!(
            format_progress(412_000_000, 900_000_000),
            "  412 MB / 900 MB (45%)"
        );
        assert_eq!(
            format_progress(999_999_999, 1_000_000_000),
            "  999 MB / 1.00 GB (99%)"
        );
    }

    #[test]
    fn progress_uses_two_decimal_gb_above_one_gb() {
        assert_eq!(
            format_progress(1_500_000_000, 2_500_000_000),
            "  1.50 GB / 2.50 GB (60%)"
        );
        assert_eq!(
            format_progress(1_000_000_000, 1_000_000_000),
            "  1.00 GB / 1.00 GB (100%)"
        );
    }

    #[test]
    fn progress_with_zero_total_prints_only_done() {
        assert_eq!(format_progress(412_000_000, 0), "  412 MB");
    }

    #[test]
    fn step_is_reported_once_per_ten_percent() {
        assert_eq!(step_of(0, 100), Some(0));
        assert_eq!(step_of(9, 100), Some(0));
        assert_eq!(step_of(10, 100), Some(10));
        assert_eq!(step_of(99, 100), Some(90));
        assert_eq!(step_of(100, 100), Some(100));
        assert_eq!(step_of(1, 0), None);
        assert_eq!(crossed_step(u64::MAX, 5, 100), Some(0));
        assert_eq!(crossed_step(0, 5, 100), None);
        assert_eq!(crossed_step(0, 10, 100), Some(10));
        assert_eq!(crossed_step(90, 100, 100), Some(100));
        assert_eq!(crossed_step(100, 100, 100), None);
    }
}
