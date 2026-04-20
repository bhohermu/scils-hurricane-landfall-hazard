from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config
from scils_tc.simulation.simulation import run_simulation

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
PREPROCESSING_FIXTURE_DIR = FIXTURE_ROOT / "preprocessing"
DETRENDED_FIXTURE_DIR = FIXTURE_ROOT / "detrended"
SIMULATED_FIXTURE_DIR = FIXTURE_ROOT / "simulated"


def main() -> None:
    """Write publishable benchmark fixtures used by the regression suite."""
    PREPROCESSING_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    DETRENDED_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    SIMULATED_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    _write_preprocessing_fixtures()
    _write_detrended_fixtures()
    _write_simulation_fixtures()


def _write_preprocessing_fixtures() -> None:
    enso_df = pd.read_csv(config.PREPROCESSED_DIR / config.ENSO_STATE_FILE)
    enso_df.to_csv(PREPROCESSING_FIXTURE_DIR / "ENSO_state.csv", index=False)

    pd.read_csv(config.PREPROCESSED_DIR / "LFI_LMI_NorthAtlantic_beta_params.csv").to_csv(
        PREPROCESSING_FIXTURE_DIR / "LFI_LMI_NorthAtlantic_beta_params.csv",
        index=False,
    )
    pd.read_csv(config.PREPROCESSED_DIR / "LFI_LMI_CONUS_beta_params.csv").to_csv(
        PREPROCESSING_FIXTURE_DIR / "LFI_LMI_CONUS_beta_params.csv",
        index=False,
    )

    gwl_df = pd.read_csv(config.PREPROCESSED_DIR / "GWL_annual.csv")
    ibtracs_df = pd.read_csv(config.PREPROCESSED_DIR / config.IBTRACS_PROPERTIES_FILE)

    benchmarks = {
        "gwl_spot_checks": {
            str(year): float(gwl_df.loc[gwl_df["Year"] == year, "GWL"].iloc[0])
            for year in [1980, 2000, 2020]
        },
        "ibtracs_counts": {
            "rows": int(len(ibtracs_df)),
            "lmi_storms": int((ibtracs_df["LMI_ms"] >= config.TROPICAL_STORM_THRESHOLD_MS).sum()),
            "lmi_hurricanes": int((ibtracs_df["LMI_ms"] >= config.SSHS_THRESHOLDS_MS["Cat1"][0]).sum()),
            "north_atlantic_lfi_storms": int((ibtracs_df["NorthAtlantic_LFI_ms"] >= config.TROPICAL_STORM_THRESHOLD_MS).sum()),
            "north_atlantic_lfi_hurricanes": int((ibtracs_df["NorthAtlantic_LFI_ms"] >= config.SSHS_THRESHOLDS_MS["Cat1"][0]).sum()),
        },
    }

    with (PREPROCESSING_FIXTURE_DIR / "preprocessing_benchmarks.json").open("w", encoding="utf-8") as handle:
        json.dump(benchmarks, handle, indent=2, sort_keys=True)


def _write_detrended_fixtures() -> None:
    benchmarks = {
        "PI_regression_vs_GWL.nc": _build_regression_benchmark(config.DETRENDED_DIR / "PI_regression_vs_GWL.nc"),
        "CGI_regression_vs_GWL.nc": _build_regression_benchmark(config.DETRENDED_DIR / "CGI_regression_vs_GWL.nc"),
    }

    with (DETRENDED_FIXTURE_DIR / "regression_benchmarks.json").open("w", encoding="utf-8") as handle:
        json.dump(benchmarks, handle, indent=2, sort_keys=True)


def _build_regression_benchmark(path: Path) -> dict:
    dataset = xr.open_dataset(path)
    try:
        slope = dataset["slope"]
        intercept = dataset["intercept"]
        valid_indices = np.argwhere(np.isfinite(slope.values) & np.isfinite(intercept.values))
        sample_positions = np.linspace(0, len(valid_indices) - 1, num=5, dtype=int)

        samples = []
        for position in sample_positions:
            index_tuple = tuple(int(value) for value in valid_indices[position])
            coords = {}
            for dim, dim_index in zip(slope.dims, index_tuple):
                coords[dim] = float(slope.coords[dim].values[dim_index])
            samples.append(
                {
                    "coords": coords,
                    "slope": float(slope.values[index_tuple]),
                    "intercept": float(intercept.values[index_tuple]),
                }
            )

        return {
            "data_vars": sorted(dataset.data_vars),
            "sizes": {key: int(value) for key, value in dataset.sizes.items()},
            "non_null_cells": int(np.isfinite(slope.values).sum()),
            "slope_mean": float(np.nanmean(slope.values)),
            "intercept_mean": float(np.nanmean(intercept.values)),
            "samples": samples,
        }
    finally:
        dataset.close()


def _write_simulation_fixtures() -> None:
    historical_df = run_simulation(
        start_year=config.DEFAULT_START_YEAR,
        end_year=config.DEFAULT_END_YEAR,
        n_iter=3,
        seed=config.RANDOM_SEED,
        use_historical=True,
        region="NorthAtlantic",
        verbose=False,
    )
    historical_df.to_csv(SIMULATED_FIXTURE_DIR / "simulated_events_historical_NorthAtlantic_n3.csv", index=False)

    gwl_df = run_simulation(
        start_year=config.DEFAULT_START_YEAR,
        end_year=config.DEFAULT_END_YEAR,
        n_iter=3,
        seed=config.RANDOM_SEED,
        target_gwl=1.24,
        use_historical=False,
        region="NorthAtlantic",
        verbose=False,
    )
    gwl_df.to_csv(SIMULATED_FIXTURE_DIR / "simulated_events_GWL1.24_NorthAtlantic_n3.csv", index=False)


if __name__ == "__main__":
    main()