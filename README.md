# scils-hurricane-landfall-hazard

A simplified, climate-conditioned model for North Atlantic tropical cyclone landfall rates.

## Overview

This repository contains the SCILS tropical cyclone landfall hazard model and its workflow scripts for:

1. preprocessing climate and storm inputs,
2. detrending PI, CGI, and SST to target climate states,
3. simulating synthetic tropical cyclone events, and
4. deriving landfall-rate changes for YELT resampling.

The code is distributed as the Python package `scils-tc`, while the repository also keeps the top-level workflow scripts used in the paper pipeline.

## Installation

Conda-first installation is the supported setup.

```bash
conda env create -f environment.yml
conda activate scils_tc
python -m pip install -e . --no-deps
```

`python run_preprocessing.py` and `scils-preprocess` call the same underlying `main()` function. The top-level script form is convenient inside a cloned repository. The console entry-point form is convenient after installation.

## Data policy

This public repository versions code, small publishable support files, and approved test fixtures. It does not version the full raw climate/storm inputs or generated model outputs.

Included in the repository:

- `data/RONI.csv`
- `data/regionDefinitionsJewson.csv`
- `data/tl_2023_us_state/`
- `data/YELT_STORM_present_NA_10000_YEARS.csv`
- safe benchmark fixtures under `tests/fixtures/`

Excluded from the repository:

- ERA5 raw inputs
- IBTrACS raw input file
- generated outputs under `preprocessed/`, `detrended/`, `simulated/`, and `resampled/`

By default, raw ERA5 and IBTrACS inputs are expected under:

```text
external-data/
  ERA5/
  IBTrACS.NA.v04r01.nc
```

You can override that location with the `SCILS_TC_RAW_DATA_DIR` environment variable.

More detail is in `DATA_ACCESS.md`.

## Data sources

- ERA5: Climate Data Store. A sample API request will be added in a later release.
- IBTrACS: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/
- IBTrACS DOI: https://doi.org/10.25921/82ty-9e16

## Model workflow

### 1. Preprocessing

```bash
python run_preprocessing.py --region NorthAtlantic
scils-preprocess --region NorthAtlantic
```

### 2. Detrending

```bash
python run_detrending.py --target-gwl 2.00
scils-detrend --target-gwl 2.00
```

### 3. Simulation

```bash
python run_simulation.py --target-gwl 2.00 --n-iter 1000 --region NorthAtlantic
scils-simulate --target-gwl 2.00 --n-iter 1000 --region NorthAtlantic
```

### 4. Resampling

```bash
python run_resampling.py --base-historical --target-gwl 2.00 --region NorthAtlantic
python run_yelt_resampling.py --change-rates resampled/change_rates_base_historical_target_GWL2.00_NorthAtlantic.csv
```

## Batch workflows

- `run_batch_preprocessing.py`
- `run_batch_detrending.py`
- `run_batch_simulations.py`
- `run_batch_resampling.py`

Installed command equivalents:

- `scils-batch-preprocess`
- `scils-batch-detrend`
- `scils-batch-simulate`
- `scils-batch-resample`

## Naming conventions

Canonical filenames use normalized target labels.

- detrended files: `ERA5_PI_detrended_GWL2.00.nc`, `ERA5_CGI_detrended_to_2050.nc`
- simulation files: `simulated_events_GWL2.00_NorthAtlantic_n1000.csv`
- sensitivity simulations: `simulated_events_pi_only_GWL2.00_NorthAtlantic_n1000.csv`
- change-rate files: `change_rates_base_historical_target_GWL2.00_NorthAtlantic.csv`

Simulation outputs include placeholder rows and ensemble metadata so downstream rates and validation preserve zero-storm year-iteration combinations.

## Tests

Public smoke and fixture-based checks:

```bash
python -m unittest tests.test_public_repo -v
python -m unittest test_model.py -v
```

`test_model.py` runs the heavier regression checks when local preprocessing/detrending outputs are available, and skips those portions otherwise.

## Citation and licensing

- License: Apache 2.0. See `LICENSE`.
- Software citation metadata: `CITATION.cff`.

Version `0.1.0` is the initial public repository release prepared during manuscript typesetting. The Zenodo DOI and manuscript-linked software citation will be finalized for the paper-aligned `1.0.0` release.

## Release plan

- `0.1.0`: initial public repository release.
- `1.0.0`: paper-aligned release after publication.

Release notes live in `CHANGELOG.md`, and the release workflow is described in `RELEASING.md`.

## Repository layout

```text
scils-hurricane-landfall-hazard/
  scils_tc/
  config.py
  batch_utils.py
  run_preprocessing.py
  run_batch_preprocessing.py
  run_detrending.py
  run_batch_detrending.py
  run_simulation.py
  run_batch_simulations.py
  run_resampling.py
  run_batch_resampling.py
  run_yelt_resampling.py
  validate_simulation.py
  tests/
  data/
```

## Authors

- Benjamin Hohermuth, Schroders Capital, Zurich
- Juner Liu, ETH Zurich
