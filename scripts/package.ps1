# Pack a built attune binary into a Windows release archive.
#
# Usage: scripts/package.ps1 -Tag <tag> -Binary <attune.exe> -OutDir <dir>
# Writes <OutDir>/attune-<tag>-windows-x64.zip with attune.exe, LICENSE and
# README.md (when present) at the archive root, and prints the archive path.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [Parameter(Mandatory = $true)][string]$Binary,
    [Parameter(Mandatory = $true)][string]$OutDir
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Binary -PathType Leaf)) {
    throw "binary not found: $Binary"
}

$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$work = Join-Path ([System.IO.Path]::GetTempPath()) ('attune-pkg-' + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $work | Out-Null

try {
    Copy-Item -LiteralPath $Binary -Destination (Join-Path $work 'attune.exe')
    foreach ($name in @('LICENSE', 'README.md')) {
        $src = Join-Path $root $name
        if (Test-Path -LiteralPath $src -PathType Leaf) {
            Copy-Item -LiteralPath $src -Destination (Join-Path $work $name)
        }
    }

    $archive = Join-Path $OutDir "attune-$Tag-windows-x64.zip"
    if (Test-Path -LiteralPath $archive) {
        Remove-Item -LiteralPath $archive -Force
    }
    Compress-Archive -Path (Join-Path $work '*') -DestinationPath $archive -CompressionLevel Optimal
    Write-Output $archive
}
finally {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}
