use std::path::PathBuf;

use crate::FetchError;

/// Overrides the home dir (tests, portable installs).
pub const ENV_HOME: &str = "ATTUNE_HOME";

/// `$ATTUNE_HOME`, else `~/.attune` (`%USERPROFILE%\.attune` on Windows).
/// Created if missing.
pub fn home_dir() -> Result<PathBuf, FetchError> {
    let dir = match std::env::var_os(ENV_HOME).filter(|v| !v.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => {
            let user = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
            let base = std::env::var_os(user)
                .filter(|v| !v.is_empty())
                .ok_or_else(|| FetchError::Home(format!("${user} is not set")))?;
            PathBuf::from(base).join(".attune")
        }
    };
    std::fs::create_dir_all(&dir).map_err(|err| FetchError::io(&dir, err))?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn home_dir_honours_attune_home() {
        let dir = std::env::temp_dir().join(format!("attune-home-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var(ENV_HOME, &dir);
        let home = home_dir().expect("home");
        std::env::remove_var(ENV_HOME);
        assert_eq!(home, dir);
        assert!(dir.is_dir());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
