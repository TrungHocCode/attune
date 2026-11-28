# Model catalog

- **`snapshot.json`** is the built-in catalog (`attune-catalog/1`), compiled into attune. It's used when there's no network and no cache. Refresh it at each release by copying `catalog.json` from the `catalog` branch.
- **`publishers.json`** drives the daily GitHub Action (`.github/workflows/catalog.yml`), which regenerates `catalog.json` on the `catalog` branch. It holds:
  - `publishers` / `base_orgs`: the Hugging Face allow-list. It must match `TRUSTED_PUBLISHERS` in `crates/attune-estimate/src/model/registry.rs`, and a test enforces that.
  - `llama_cpp`: the pinned llama.cpp release and its assets per OS, arch and **variant**:
    - `cpu`: always installed.
    - `vulkan`: the default GPU build, installed by `setup --gpu auto` when an NVIDIA or AMD GPU is found.
    - `cuda12`: opt-in through `setup --gpu cuda`. It has two assets, the build plus cudart, and a `min_driver` field.

    Every asset is pinned by size and SHA-256 and must come from `github.com/ggml-org/llama.cpp/releases/download/`.
  - `max_params`, `quants` and `per_publisher_limit`: limits for the generator.
- **To bump llama.cpp**, edit the tag and every asset in `publishers.json`. Take the sizes and digests from `gh api repos/ggml-org/llama.cpp/releases/tags/<tag>`. Then copy the pin into `snapshot.json` and run `cargo test --workspace`.
