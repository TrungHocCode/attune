use crate::FetchError;

/// Which URLs attune may download from. Every redirect hop is checked too,
/// so a tampered catalog cannot point a download anywhere else.
#[derive(Debug, Clone, Copy)]
pub struct HostPolicy {
    allow_loopback_http: bool,
}

/// (host, required path prefix) pairs; an empty prefix allows any path.
const EXACT: &[(&str, &str)] = &[
    ("huggingface.co", ""),
    ("hf.co", ""),
    ("github.com", "/ggml-org/llama.cpp/"),
    ("objects.githubusercontent.com", ""),
    ("release-assets.githubusercontent.com", ""),
    ("raw.githubusercontent.com", "/TrungHocCode/attune/"),
];

/// Hugging Face LFS/Xet CDNs redirect to subdomains of these.
const SUFFIXES: &[&str] = &[".huggingface.co", ".hf.co"];

impl HostPolicy {
    /// The real allow-list: HTTPS to Hugging Face, llama.cpp releases and
    /// this repo's catalog only.
    pub fn production() -> Self {
        Self {
            allow_loopback_http: false,
        }
    }

    /// Production rules plus plain HTTP to `127.0.0.1`, for stub servers.
    #[doc(hidden)]
    pub fn allow_local_for_tests() -> Self {
        Self {
            allow_loopback_http: true,
        }
    }

    /// Ok when `url` may be fetched.
    pub fn check(&self, url: &str) -> Result<(), FetchError> {
        let denied = || FetchError::HostNotAllowed {
            url: url.to_string(),
        };
        let (scheme, rest) = url.split_once("://").ok_or_else(denied)?;
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        if authority.contains('@') {
            return Err(denied());
        }
        let host = authority
            .rsplit_once(':')
            .map_or(authority, |(host, _port)| host)
            .to_ascii_lowercase();
        let ok = match scheme {
            "https" => {
                EXACT
                    .iter()
                    .any(|(h, prefix)| host == *h && path.starts_with(prefix))
                    || SUFFIXES.iter().any(|s| host.ends_with(s))
            }
            "http" => self.allow_loopback_http && host == "127.0.0.1",
            _ => false,
        };
        if ok {
            Ok(())
        } else {
            Err(denied())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_allows_hf_and_llama_releases_only() {
        let policy = HostPolicy::production();
        for ok in [
            "https://huggingface.co/Qwen/x/resolve/abc/f.gguf",
            "https://cdn-lfs.huggingface.co/repos/ab/cd/file",
            "https://cas-bridge.xethub.hf.co/xet-bridge-us/abc",
            "https://github.com/ggml-org/llama.cpp/releases/download/b1/a.zip",
            "https://release-assets.githubusercontent.com/github-production-release-asset/1",
            "https://objects.githubusercontent.com/github-production-release-asset-2e65be/1",
            "https://raw.githubusercontent.com/TrungHocCode/attune/catalog/catalog.json",
        ] {
            assert!(policy.check(ok).is_ok(), "{ok}");
        }
        for bad in [
            "https://evil.example/f.gguf",
            "http://huggingface.co/Qwen/x/resolve/abc/f.gguf",
            "https://github.com/other/repo/releases/download/b1/a.zip",
            "https://raw.githubusercontent.com/someone/else/main/catalog.json",
            "https://huggingface.co.evil.example/x",
            "https://evilhuggingface.co/x",
            "http://127.0.0.1:8080/x",
        ] {
            assert!(
                matches!(policy.check(bad), Err(FetchError::HostNotAllowed { .. })),
                "{bad}"
            );
        }
    }

    #[test]
    fn test_policy_also_allows_loopback_http() {
        let policy = HostPolicy::allow_local_for_tests();
        assert!(policy.check("http://127.0.0.1:5000/f").is_ok());
        assert!(policy.check("http://10.0.0.1/x").is_err());
    }
}
