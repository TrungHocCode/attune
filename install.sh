#!/bin/sh
# attune installer for Linux x86_64.
#
#   curl -fsSL https://raw.githubusercontent.com/TrungHocCode/attune/main/install.sh | sh
#
# Environment overrides:
#   ATTUNE_VERSION=v0.6.0   pin a release instead of using the latest one
#   ATTUNE_HOME=DIR         install root (default: ~/.attune)
#   ATTUNE_DOWNLOAD_BASE    asset base URL (or file:// directory, for tests)
#   ATTUNE_NO_MODIFY_PATH=1 skip ~/.profile and shell rc edits (used by CI)
set -eu

REPO=TrungHocCode/attune

os=$(uname -s)
arch=$(uname -m)

if [ "$os" != "Linux" ]; then
    if [ "$os" = "Darwin" ]; then
        echo "error: attune does not support macOS yet; only Linux x86_64 is available" >&2
    else
        echo "error: unsupported operating system: $os (only Linux x86_64 is supported)" >&2
    fi
    exit 1
fi

case "$arch" in
    x86_64 | amd64) ;;
    *)
        echo "error: unsupported architecture: $arch (only x86_64 is supported)" >&2
        exit 1
        ;;
esac

# Pick a downloader once. curl handles file:// URLs, which the CI test relies on.
if command -v curl >/dev/null 2>&1; then
    download() { curl -fsSL "$1" -o "$2"; }
    fetch_body() { curl -fsSL "$1" 2>/dev/null || true; }
elif command -v wget >/dev/null 2>&1; then
    download() { wget -q "$1" -O "$2"; }
    fetch_body() { wget -qO- "$1" 2>/dev/null || true; }
else
    echo "error: need curl or wget to download attune" >&2
    exit 1
fi

TAG=${ATTUNE_VERSION:-}
if [ -z "$TAG" ]; then
    body=$(fetch_body "https://api.github.com/repos/$REPO/releases/latest")
    TAG=$(printf '%s\n' "$body" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)
    if [ -z "$TAG" ]; then
        echo "error: could not find an attune release yet (set ATTUNE_VERSION to pin a version)" >&2
        exit 1
    fi
fi

BASE=${ATTUNE_DOWNLOAD_BASE:-https://github.com/$REPO/releases/download/$TAG}
BASE=${BASE%/}
archive="attune-$TAG-linux-x64.tar.gz"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM

download "$BASE/$archive" "$tmp/$archive"
download "$BASE/SHA256SUMS" "$tmp/SHA256SUMS"

line=$(awk -v f="$archive" '$2 == f || $2 == "*" f { print; exit }' "$tmp/SHA256SUMS")
if [ -z "$line" ]; then
    echo "error: $archive is not listed in SHA256SUMS" >&2
    exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
    if ! (cd "$tmp" && printf '%s\n' "$line" | sha256sum -c -); then
        echo "error: checksum verification failed for $archive" >&2
        exit 1
    fi
elif command -v shasum >/dev/null 2>&1; then
    if ! (cd "$tmp" && printf '%s\n' "$line" | shasum -a 256 -c -); then
        echo "error: checksum verification failed for $archive" >&2
        exit 1
    fi
else
    echo "error: need sha256sum or shasum to verify the download" >&2
    exit 1
fi

home=${ATTUNE_HOME:-$HOME/.attune}
bindir=$home/bin

mkdir -p "$tmp/unpack" "$bindir"
tar -xzf "$tmp/$archive" -C "$tmp/unpack"
if [ ! -f "$tmp/unpack/attune" ]; then
    echo "error: archive does not contain attune" >&2
    exit 1
fi
cp "$tmp/unpack/attune" "$bindir/attune"
chmod +x "$bindir/attune"

marker='# added by attune installer'

ensure_path() {
    file=$1
    if [ -f "$file" ] && grep -Fq "$marker" "$file" 2>/dev/null; then
        return 0
    fi
    {
        printf '\n%s\n' "$marker"
        printf 'export PATH="%s:$PATH"\n' "$bindir"
    } >>"$file"
}

case ":$PATH:" in
    *":$bindir:"*) on_path=1 ;;
    *) on_path=0 ;;
esac

if [ "$on_path" -eq 0 ] && [ "${ATTUNE_NO_MODIFY_PATH:-0}" != "1" ]; then
    ensure_path "$HOME/.profile"
    if [ -f "$HOME/.bashrc" ]; then
        ensure_path "$HOME/.bashrc"
    fi
    if [ -f "$HOME/.zshrc" ]; then
        ensure_path "$HOME/.zshrc"
    fi
fi

installed=$("$bindir/attune" --version 2>/dev/null || true)
echo "attune installed: ${installed:-$TAG}"
echo "Open a new terminal, then run: attune recommend"
