//! Keep measurement threads at full speed.
//!
//! Windows 11 power throttling (EcoQoS) moves a background console process to
//! efficiency scheduling after a short burst: the first bandwidth probe read
//! 34.7 GB/s and every later one 12-17 GB/s in the same process, while
//! llama.cpp (which opts out) sustained ~30 GB/s. Every probe calls
//! [`full_speed`] first so attune measures what llama.cpp will get.

use std::sync::OnceLock;

/// Opt this process out of execution-speed throttling (once). Returns whether
/// the process now runs unthrottled; always `true` off Windows.
pub fn full_speed() -> bool {
    static APPLIED: OnceLock<bool> = OnceLock::new();
    *APPLIED.get_or_init(opt_out)
}

#[cfg(windows)]
fn opt_out() -> bool {
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, ProcessPowerThrottling, SetProcessInformation,
        PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        PROCESS_POWER_THROTTLING_STATE,
    };
    let state = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        // Control execution speed, state 0 = never throttle it.
        ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        StateMask: 0,
    };
    // SAFETY: a pseudo-handle for this process and a correctly sized,
    // initialised struct that outlives the call.
    unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            (&state as *const PROCESS_POWER_THROTTLING_STATE).cast(),
            std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
        ) != 0
    }
}

#[cfg(not(windows))]
fn opt_out() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_speed_opt_out_succeeds_and_is_idempotent() {
        assert!(full_speed());
        assert!(full_speed());
    }
}
