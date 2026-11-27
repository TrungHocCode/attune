# Model catalog

- `snapshot.json` is the built-in catalog (`attune-catalog/1`) compiled into attune, used when there is no network and no cache. Refresh it at each release by copying `catalog.json` from the `catalog` branch.
- `publishers.toml` is the allow-list the daily GitHub Action (`.github/workflows/catalog.yml`) uses to regenerate `catalog.json` on the `catalog` branch.
