# Security Policy

## Supported versions

Attune is pre-1.0. Security fixes land on `main` and in the next release;
only the latest release is supported.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately through GitHub:
[Security → Report a vulnerability](https://github.com/TrungHocCode/attune/security/advisories/new).
Include what you found, how to reproduce it, and the impact you expect.

You should get a first reply within 7 days. Once a fix is ready we publish a
GitHub security advisory and credit you, unless you prefer otherwise.

## What is in scope

Attune downloads and runs third-party files, so these areas matter most:

- **Downloads** (`crates/attune-fetch`): the HTTPS host allow-list (checked on
  every redirect), SHA-256 verification, resumable `.part` files, and archive
  unpacking (entries or links that escape the install directory must be
  rejected).
- **Model catalog** (`catalog/`, the `catalog` branch, `attune-catalog/1`):
  a tampered catalog must not make attune download from an untrusted host or
  repository.
- **`attune serve`**: the OpenAI-compatible proxy binds `127.0.0.1` by default;
  any other address requires `--api-key`. Auth bypasses and request smuggling
  to llama-server are in scope.
- **Profiles**: a profile file must not be able to inject arbitrary
  llama-server arguments. This includes the GPU fields (`engine.variant`,
  `load_config.gpu_name`) and the `-dev` value attune derives from them.
- **GPU detection and builds**:
  - `detect` runs `nvidia-smi` and, on Windows, a read-only PowerShell CIM
    query. These must never run anything else.
  - The Vulkan and CUDA llama.cpp builds have the same pin, SHA-256 and host
    rules as the CPU build.

Out of scope: vulnerabilities in llama.cpp itself or in model weights (report
those upstream), and attacks that need an already-compromised machine.
