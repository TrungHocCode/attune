#!/bin/sh
# Pack a built attune binary into a Linux release archive.
#
# Usage: scripts/package.sh <tag> <binary> <outdir>
# Writes <outdir>/attune-<tag>-linux-x64.tar.gz with attune, LICENSE and
# README.md (when present) at the archive root, and prints the archive path.
set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <tag> <binary> <outdir>" >&2
    exit 2
fi

tag=$1
binary=$2
outdir=$3

if [ ! -f "$binary" ]; then
    echo "error: binary not found: $binary" >&2
    exit 1
fi

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$outdir"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT INT TERM

cp "$binary" "$work/attune"
chmod +x "$work/attune"
if [ -f "$root/LICENSE" ]; then
    cp "$root/LICENSE" "$work/LICENSE"
fi
if [ -f "$root/README.md" ]; then
    cp "$root/README.md" "$work/README.md"
fi

archive="$outdir/attune-$tag-linux-x64.tar.gz"
rm -f "$archive"
tar -czf "$archive" -C "$work" .
echo "$archive"
