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

static LAST_PERCENT: AtomicU64 = AtomicU64::new(u64::MAX);

/// Progress on stderr, one line per 5%.
pub fn progress(done: u64, total: u64) {
    let percent = (done * 100).checked_div(total).unwrap_or(100);
    let step = percent / 5 * 5;
    if LAST_PERCENT.swap(step, Ordering::Relaxed) != step {
        eprintln!(
            "[attune] downloading… {step}% ({:.2} / {:.2} GB)",
            done as f64 / 1e9,
            total as f64 / 1e9
        );
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
}
