# Notebooks

Minimal notebooks for the SymbioPan project. **All production logic lives in `symbiopan/` and `scripts/`** — notebooks are reserved for exploratory work and onboarding demos only.

| Notebook | Purpose |
|----------|---------|
| `01_quickstart.ipynb` | End-to-end pipeline walkthrough (config → preprocess → train → infer → visualize) |

## Conventions

- **Do not** copy production code into notebooks. Call the entry points in `scripts/` or use `symbiopan.*` APIs directly.
- **Do not** check in cell outputs (run `jupyter nbconvert --clear-output --inplace <nb>` before commit).
- **Do not** commit notebooks with Colab-specific code (Drive mounting, secret tokens, etc.) to `main`.

## Running locally

```bash
make install
jupyter lab notebooks/
```
