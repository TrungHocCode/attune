//! Home dir, host allow-list and verified, resumable downloads (M6).
//!
//! Leaf crate: no other attune crate is a dependency, so `attune-estimate`
//! (catalog fetch) and `attune-backend` (model and llama.cpp installs) can
//! share it without tangling the layering.

mod download;
mod error;
mod home;
mod install;
mod policy;

pub use download::{download, download_with, fetch_range, fetch_text, free_disk_bytes, Download};
pub use error::FetchError;
pub use home::{home_dir, ENV_HOME};
pub use install::{find_installed, install_archive, install_dir, installed_binary};
pub use policy::HostPolicy;
