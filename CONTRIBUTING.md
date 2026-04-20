# Contributing

This repository is maintained as a research software companion for the SCILS hurricane landfall hazard model.

## Scope

- Keep the public repository code-first and publication-safe.
- Do not add restricted raw data, ERA5 files, IBTrACS files, or generated model outputs.
- Keep user-facing command-line workflows stable unless a change is documented.

## Development setup

```bash
conda env create -f environment.yml
conda activate scils_tc
python -m pip install -e . --no-deps
ruff check .
```

## Change guidelines

- Prefer focused changes over broad rewrites.
- Keep scientific behavior unchanged unless the change is explicitly intended to alter model logic.
- Add or update tests when behavior changes.
- Preserve reproducibility by documenting default changes, file naming updates, and workflow assumptions.

## Data policy

- Raw ERA5 and IBTrACS inputs live outside the public repository.
- Generated outputs under `preprocessed/`, `detrended/`, `simulated/`, and `resampled/` should not be committed.
- Only publish fixtures that are explicitly cleared for redistribution.