# attune installer for Windows x64.
#
#   irm https://raw.githubusercontent.com/TrungHocCode/attune/main/install.ps1 | iex
#
# Environment overrides:
#   ATTUNE_VERSION=v0.6.0   pin a release instead of using the latest one
#   ATTUNE_HOME=DIR         install root (default: %USERPROFILE%\.attune)
#   ATTUNE_DOWNLOAD_BASE    asset base URL, or a local directory for tests
#   ATTUNE_NO_MODIFY_PATH=1 skip the user PATH edit (used by CI)
# Works on Windows PowerShell 5.1 and PowerShell 7.
$ErrorActionPreference = 'Stop'

$Repo = 'TrungHocCode/attune'

if ($PSVersionTable.PSVersion.Major -lt 6) {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
}

if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    throw "attune supports Windows x64 only (detected: $env:PROCESSOR_ARCHITECTURE)."
}

$Tag = $env:ATTUNE_VERSION
if ([string]::IsNullOrWhiteSpace($Tag)) {
    $api = "https://api.github.com/repos/$Repo/releases/latest"
    try {
        $release = Invoke-RestMethod -Uri $api -Headers @{ 'User-Agent' = 'attune-installer' } -UseBasicParsing
    }
    catch {
        throw 'Could not find an attune release yet. Set ATTUNE_VERSION to pin a version.'
    }
    $Tag = $release.tag_name
}
if ([string]::IsNullOrWhiteSpace($Tag)) {
    throw 'Could not determine the attune version. Set ATTUNE_VERSION to pin a version.'
}

$Base = $env:ATTUNE_DOWNLOAD_BASE
if ([string]::IsNullOrWhiteSpace($Base)) {
    $Base = "https://github.com/$Repo/releases/download/$Tag"
}
$Base = $Base.TrimEnd([char[]]@('/', '\'))

$ArchiveName = "attune-$Tag-windows-x64.zip"

$HomeDir = $env:ATTUNE_HOME
if ([string]::IsNullOrWhiteSpace($HomeDir)) {
    $HomeDir = Join-Path $env:USERPROFILE '.attune'
}
$BinDir = Join-Path $HomeDir 'bin'
$Target = Join-Path $BinDir 'attune.exe'

$running = Get-Process -Name 'attune' -ErrorAction SilentlyContinue
if ($null -ne $running) {
    throw 'attune is running. Close it (and any chat or serve window) and run the installer again.'
}

function Get-AttuneFile {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    $source = "$Base/$Name"
    if ($Base -match '^(?i)https?://') {
        Invoke-WebRequest -Uri $source -OutFile $Destination -UseBasicParsing
    }
    else {
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "file not found: $source"
        }
        Copy-Item -LiteralPath $source -Destination $Destination -Force
    }
}

$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ('attune-install-' + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

try {
    $archive = Join-Path $Tmp $ArchiveName
    Get-AttuneFile -Name $ArchiveName -Destination $archive
    $sums = Join-Path $Tmp 'SHA256SUMS'
    Get-AttuneFile -Name 'SHA256SUMS' -Destination $sums

    $expected = $null
    foreach ($line in (Get-Content -LiteralPath $sums)) {
        $trimmed = $line.Trim()
        if ($trimmed.Length -eq 0) {
            continue
        }
        $match = [regex]::Match($trimmed, '^(?<hash>[0-9A-Fa-f]{64})\s+\*?(?<file>.+)$')
        if ($match.Success -and $match.Groups['file'].Value -eq $ArchiveName) {
            $expected = $match.Groups['hash'].Value
            break
        }
    }
    if ($null -eq $expected) {
        throw "$ArchiveName is not listed in SHA256SUMS"
    }

    $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
    if ($actual -ne $expected) {
        throw "checksum verification failed for ${ArchiveName} (expected $expected, got $actual)"
    }

    $unpack = Join-Path $Tmp 'unpack'
    Expand-Archive -LiteralPath $archive -DestinationPath $unpack -Force
    $exe = Join-Path $unpack 'attune.exe'
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        throw 'archive does not contain attune.exe'
    }

    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    Copy-Item -LiteralPath $exe -Destination $Target -Force

    if ($env:ATTUNE_NO_MODIFY_PATH -ne '1') {
        $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        if ($null -eq $userPath) {
            $userPath = ''
        }
        $inUserPath = $false
        foreach ($entry in ($userPath -split ';')) {
            if ($entry.Trim().TrimEnd('\') -ieq $BinDir.TrimEnd('\')) {
                $inUserPath = $true
                break
            }
        }
        if (-not $inUserPath) {
            if ([string]::IsNullOrWhiteSpace($userPath)) {
                $newPath = $BinDir
            }
            else {
                $newPath = $userPath.TrimEnd(';') + ';' + $BinDir
            }
            [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
        }

        $inSession = $false
        foreach ($entry in ($env:Path -split ';')) {
            if ($entry.Trim().TrimEnd('\') -ieq $BinDir.TrimEnd('\')) {
                $inSession = $true
                break
            }
        }
        if (-not $inSession) {
            $env:Path = $env:Path.TrimEnd(';') + ';' + $BinDir
        }
    }

    $installed = & $Target --version
    if ($LASTEXITCODE -ne 0) {
        throw "installed attune did not run (exit $LASTEXITCODE)"
    }
    Write-Host "attune installed: $installed"
    Write-Host 'Open a new terminal, then run: attune recommend'
    Write-Host 'Note: Windows SmartScreen may warn on the first run (the binary is unsigned).'
}
finally {
    Remove-Item -LiteralPath $Tmp -Recurse -Force -ErrorAction SilentlyContinue
}
