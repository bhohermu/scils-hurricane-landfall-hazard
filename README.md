# scils-hurricane-landfall-hazard

Companion code for:

> Liu, J., Steinmann, C. B., Bresch, D. N., Meiler, S., Lohmann, U., and Hohermuth, B.:
> *Recalibrating Risk: a simplified model for North Atlantic hurricanes in a warming climate.*
> <!-- TODO: add journal reference and DOI after publication -->

## Overview

**SCILS** (Schroders Capital Insurance-Linked Securities) is a physically grounded model that estimates how North Atlantic tropical cyclone (TC) landfall rates change between a historical baseline (1980–2025) and a target climate state defined by a Global Warming Level (GWL). The resulting category-specific landfall rate changes are used to resample a Year Event Loss Table (YELT), producing climate-adjusted loss distributions without rerunning full track simulations.

The model uses two physical climate proxies derived from ERA5 reanalysis:

- **Potential Intensity (PI)** — a thermodynamic upper bound on TC wind speed (Bister and Emanuel, 2002; Gilford, 2021), used as an intensity proxy.
- **Cyclone Genesis Index (CGI)** — a genesis potential metric (Bruyère et al., 2012), used as a frequency proxy.

Both proxies are detrended to a target GWL using pixel-wise Theil-Sen regression against observed GWL, preserving interannual variability while shifting the climate mean.

### Model workflow (paper Section 2.2, Figure 2)

The SCILS simulation generates stochastic TC events in four steps:

1. **TC count** — seasonal storm count drawn from a Poisson distribution with λ = scaled MDR CGI, then distributed to months using monthly CGI weights (`scils_tc/simulation/simulation.py`).
2. **LMI location** — Lifetime Maximum Intensity location sampled from ENSO-conditioned kernel density maps of historical LMI positions (`scils_tc/simulation/sampling.py`).
3. **Assign LMI** — PI at the sampled location is multiplied by an LMI/PI ratio drawn from the pooled historical distribution (paper Figure 1a) to obtain the storm's LMI (`scils_tc/simulation/sampling.py`).
4. **Calculate LFI** — Landfall Intensity is computed by multiplying LMI with a region-specific LFI/LMI ratio drawn from a zero-one-inflated Beta distribution (paper Figure 1b, Jewson 2023a regions) (`scils_tc/simulation/sampling.py`).

Event sets are generated for each target GWL and compared against a historical baseline to derive per-category landfall rate changes. These rate changes are applied to a YELT using incremental simulation (Jewson, 2023b) to obtain climate-adjusted loss estimates.

### Key assumptions and limitations

The following simplifications are intentional design choices documented in the paper:

- **No explicit track generation.** The model does not simulate TC trajectories. Landfall intensity is derived statistically from LMI via region-specific LFI/LMI distributions.
- **Constant LMI/PI ratio distribution.** The historical relationship between observed LMI and theoretical PI is assumed stationary across climate states. This follows the approach of Emanuel (2003, 2005) and Sparks and Toumi (2024).
- **Constant LFI/LMI ratio distributions.** The regional distributions relating landfall intensity to LMI are assumed stationary. Spatial shifts in storm tracks are not represented.
- **ENSO-conditioned LMI locations only.** LMI location maps are conditioned on ENSO phase but not on GWL. Spatial shifts in genesis or LMI location under warming are not captured.
- **Constant exposure and vulnerability.** YELT resampling modifies only event frequencies, not event severities. Loss per event is unchanged.
- **Resampling limitations.** The resampling method cannot create new events beyond those in the input YELT. This effectively caps extreme losses and underestimates tail risk at return periods above ~50 years (paper Section 3.5). Storm clustering and aggregate loss changes are not represented.
- **Thermodynamic proxies only.** Dynamic factors (e.g., wind shear trends, steering flow changes) enter only indirectly through the CGI. The model does not account for changes in large-scale circulation patterns.

### What is and is not represented

| Represented | Not represented |
|---|---|
| Climate-driven changes in TC frequency (via CGI) | Spatial shifts in TC tracks or genesis |
| Climate-driven changes in TC intensity (via PI) | Changes in TC translation speed or size |
| ENSO modulation of LMI location | Dynamic driver trends (independent of CGI) |
| Category-specific landfall rate changes | New event types beyond historical catalogue |
| YELT adjustment for near-term climate states | Tail risk beyond input YELT maximum |

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

- **ERA5 reanalysis** (Hersbach et al., 2023): monthly means from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/). Used to compute SST, wind shear, PI, and CGI.
- **IBTrACS** v04r01 (Gahtan et al., 2024): historical TC tracks for the North Atlantic. [Download](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/), DOI: [10.25921/82ty-9e16](https://doi.org/10.25921/82ty-9e16).
- **RONI** (Relative Oceanic Niño Index): ENSO classification from NOAA CPC (included in repo).
- **YELT**: generated with CLIMADA (Aznar-Siguan and Bresch, 2019) using STORM present-day tracks (Bloemendaal et al., 2020). See `DATA_ACCESS.md` for provenance.

See `DATA_ACCESS.md` for the full list of required input files and download instructions.

## Model workflow

The pipeline runs in four stages. Each stage corresponds to a section of the paper methodology.

### 1. Preprocessing (paper Section 2.1)

Processes ERA5 and IBTrACS inputs into the intermediate datasets used by the model: SST, wind shear, PI maps, CGI maps, ENSO classification, LMI/PI ratio histogram, LFI/LMI Beta distributions, and LMI location KDEs.

```bash
python run_preprocessing.py --region NorthAtlantic
scils-preprocess --region NorthAtlantic
```

### 2. Detrending (paper Section 2.1, "Climate variables and detrending")

Detrends PI and CGI maps to a target GWL using pixel-wise Theil-Sen regression against observed GWL. This shifts the climate mean while preserving interannual variability.

```bash
python run_detrending.py --target-gwl 2.00
scils-detrend --target-gwl 2.00
```

### 3. Simulation (paper Section 2.2, Figure 2)

Generates stochastic TC event sets for the target climate state using the 4-step model described above.

```bash
python run_simulation.py --target-gwl 2.00 --n-iter 1000 --region NorthAtlantic
scils-simulate --target-gwl 2.00 --n-iter 1000 --region NorthAtlantic
```

### 4. Resampling (paper Section 2.3)

Computes per-category landfall rate changes between base and target simulations, then adjusts a YELT using incremental simulation (Jewson, 2023b).

```bash
python run_resampling.py --base-historical --target-gwl 2.00 --region NorthAtlantic
python run_yelt_resampling.py --change-rates resampled/change_rates_base_historical_target_GWL2.00_NorthAtlantic.csv
```

## Batch workflows

Batch scripts run all standard scenarios (historical, GWL 1.24, GWL 2.00, and sensitivity experiments) used in the paper:

- `run_batch_preprocessing.py` / `scils-batch-preprocess`
- `run_batch_detrending.py` / `scils-batch-detrend`
- `run_batch_simulations.py` / `scils-batch-simulate`
- `run_batch_resampling.py` / `scils-batch-resample`

Sensitivity experiments (`pi_only`, `cgi_only`) correspond to paper Section 3.4.

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

## Validation (paper Section 3.1)

`validate_simulation.py` compares simulated historical landfall counts against IBTrACS observations and reports RMSE, CRPS, and likelihood metrics:

```bash
python validate_simulation.py --simulation-file simulated/simulated_events_historical_NorthAtlantic_n1000.csv
```

## Reproducibility

To reproduce the paper results from scratch:

1. Download ERA5 and IBTrACS inputs as described in `DATA_ACCESS.md`.
2. Run the full batch pipeline:
   ```bash
   python run_batch_preprocessing.py
   python run_batch_detrending.py
   python run_batch_simulations.py --n-iter 1000 --seed 42
   python run_batch_resampling.py --n-iter 1000 --seed 42
   ```
3. Validate the historical simulation:
   ```bash
   python validate_simulation.py --simulation-file simulated/simulated_events_historical_NorthAtlantic_n1000.csv
   ```

Random seed `42` is the default. Simulation outputs are written to `simulated/` and resampled outputs to `resampled/`. The default iteration count is 1000 per year (46,000 stochastic years total).

## Citation and licensing

- License: Apache 2.0. See `LICENSE`.
- Software citation metadata: `CITATION.cff`.

Version `0.1.0` is the initial public repository release prepared during manuscript typesetting. The Zenodo DOI and manuscript-linked software citation will be finalized for the paper-aligned `1.0.0` release.

## Release plan

- `0.1.0`: initial public repository release.
- `1.0.0`: paper-aligned release after publication.

Release notes live in `CHANGELOG.md`, and the release workflow is described in `RELEASING.md`.

## Paper-to-code mapping

| Paper section | Code |
|---|---|
| Section 2.1 — Input datasets and preprocessing | `run_preprocessing.py`, `scils_tc/preprocessing/` |
| Section 2.1 — Climate variable detrending (Theil-Sen vs GWL) | `run_detrending.py`, `scils_tc/detrending/` |
| Section 2.1 — GWL calculation (C3S methodology) | `scils_tc/preprocessing/gwl_calculation.py` |
| Section 2.1 — LMI/PI ratio (Figure 1a) | `scils_tc/preprocessing/ibtracs_statistics.py` |
| Section 2.1 — LFI/LMI distributions (Figure 1b) | `scils_tc/preprocessing/ibtracs_statistics.py` |
| Section 2.2 — Simulation flowchart (Figure 2) | `run_simulation.py`, `scils_tc/simulation/` |
| Section 2.3 — YELT resampling | `run_yelt_resampling.py`, `scils_tc/resampling/yelt_resampling.py` |
| Section 2.3 — Landfall rate changes | `run_resampling.py`, `scils_tc/resampling/resampling.py` |
| Section 3.1 — Historical validation | `validate_simulation.py` |
| Section 3.4 — Sensitivity (PI-only, CGI-only) | `batch_utils.py` (scenario definitions), `--detrend-pi-only` / `--detrend-cgi-only` flags |
| Configuration and constants | `config.py` |

## Repository layout

```text
scils-hurricane-landfall-hazard/
  scils_tc/                      # Python package
    preprocessing/               # ERA5/IBTrACS processing, KDE, distributions
    detrending/                  # Theil-Sen detrending of PI/CGI to target GWL
    simulation/                  # Stochastic event generation (4-step model)
    resampling/                  # Landfall rate changes and YELT adjustment
    utils/                       # Regions, Saffir-Simpson, simulation artifacts
  config.py                      # All paths, constants, and thresholds
  batch_utils.py                 # Scenario definitions for batch runs
  run_preprocessing.py           # Step 1 entry point
  run_detrending.py              # Step 2 entry point
  run_simulation.py              # Step 3 entry point
  run_resampling.py              # Step 4a: landfall rate changes
  run_yelt_resampling.py         # Step 4b: YELT adjustment
  run_batch_*.py                 # Batch runners for all scenarios
  validate_simulation.py         # Historical validation metrics
  test_model.py                  # Regression tests (requires local outputs)
  tests/                         # Public smoke tests and fixtures
  data/                          # Small included support files
```

## Authors

- Juner Liu, ETH Zurich
- Carmen B. Steinmann, ETH Zurich
- David N. Bresch, ETH Zurich
- Simona Meiler, Stanford University
- Ulrike Lohmann, ETH Zurich
- Benjamin Hohermuth, Schroders Capital, Zurich

Code contributions by Juner Liu and Benjamin Hohermuth.
