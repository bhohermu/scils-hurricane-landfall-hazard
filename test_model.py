import json
import unittest
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config
from scils_tc.resampling.resampling import calculate_change_rates, calculate_landfall_rates
from scils_tc.simulation.simulation import run_simulation
from scils_tc.utils import SimulationArtifact, TargetSpec

PROJECT_ROOT = Path(__file__).parent.resolve()
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures"
PREPROCESSING_FIXTURE_DIR = FIXTURE_ROOT / "preprocessing"
DETRENDED_FIXTURE_DIR = FIXTURE_ROOT / "detrended"
SIMULATED_FIXTURE_DIR = FIXTURE_ROOT / "simulated"

PREPROCESSING_OUTPUTS_PRESENT = all(
    path.exists()
    for path in [
        config.PREPROCESSED_DIR / config.ENSO_STATE_FILE,
        config.PREPROCESSED_DIR / "GWL_annual.csv",
        config.PREPROCESSED_DIR / "LFI_LMI_NorthAtlantic_beta_params.csv",
        config.PREPROCESSED_DIR / "LFI_LMI_CONUS_beta_params.csv",
        config.PREPROCESSED_DIR / config.IBTRACS_PROPERTIES_FILE,
    ]
)
DETRENDED_OUTPUTS_PRESENT = all(
    path.exists()
    for path in [
        config.DETRENDED_DIR / "PI_regression_vs_GWL.nc",
        config.DETRENDED_DIR / "CGI_regression_vs_GWL.nc",
        config.DETRENDED_DIR / "ERA5_PI_detrended_GWL1.24.nc",
        config.DETRENDED_DIR / "ERA5_CGI_detrended_GWL1.24.nc",
    ]
)
SIMULATION_INPUTS_PRESENT = all(
    path.exists()
    for path in [
        config.PREPROCESSED_DIR / config.ENSO_STATE_FILE,
        config.PREPROCESSED_DIR / config.CGI_MDR_FILE,
        config.PREPROCESSED_DIR / config.ASO_MDR_SST_FILE,
        config.PREPROCESSED_DIR / config.LMI_PI_RATIO_FILE.replace(".csv", "_single.csv"),
        config.PREPROCESSED_DIR / "LFI_LMI_NorthAtlantic_beta_params.csv",
    ]
) and DETRENDED_OUTPUTS_PRESENT


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _sort_simulation_frame(df: pd.DataFrame) -> pd.DataFrame:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    df = pd.read_csv(buffer)
    return df.sort_values(["year", "iteration", "event_id"], kind="stable").reset_index(drop=True)


@unittest.skipUnless(PREPROCESSING_OUTPUTS_PRESENT, "Requires local preprocessing outputs")
class TestPreprocessingBenchmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = _load_json(PREPROCESSING_FIXTURE_DIR / "preprocessing_benchmarks.json")
        cls.enso_fixture = _load_csv(PREPROCESSING_FIXTURE_DIR / "ENSO_state.csv")
        cls.na_beta_fixture = _load_csv(PREPROCESSING_FIXTURE_DIR / "LFI_LMI_NorthAtlantic_beta_params.csv")
        cls.conus_beta_fixture = _load_csv(PREPROCESSING_FIXTURE_DIR / "LFI_LMI_CONUS_beta_params.csv")

    def test_preprocessing_files_exist(self):
        required_paths = [
            config.PREPROCESSED_DIR / config.ENSO_STATE_FILE,
            config.PREPROCESSED_DIR / "GWL_annual.csv",
            config.PREPROCESSED_DIR / "LFI_LMI_NorthAtlantic_beta_params.csv",
            config.PREPROCESSED_DIR / "LFI_LMI_CONUS_beta_params.csv",
            config.PREPROCESSED_DIR / config.IBTRACS_PROPERTIES_FILE,
        ]
        for path in required_paths:
            self.assertTrue(path.exists(), f"Missing preprocessing file: {path}")

    def test_beta_params_match_current_benchmarks(self):
        pd.testing.assert_frame_equal(
            _load_csv(config.PREPROCESSED_DIR / "LFI_LMI_NorthAtlantic_beta_params.csv"),
            self.na_beta_fixture,
            check_dtype=False,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
        pd.testing.assert_frame_equal(
            _load_csv(config.PREPROCESSED_DIR / "LFI_LMI_CONUS_beta_params.csv"),
            self.conus_beta_fixture,
            check_dtype=False,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_gwl_spot_checks_match_benchmarks(self):
        gwl_df = _load_csv(config.PREPROCESSED_DIR / "GWL_annual.csv")
        for year_str, expected in self.benchmark["gwl_spot_checks"].items():
            year = int(year_str)
            actual = float(gwl_df.loc[gwl_df["Year"] == year, "GWL"].iloc[0])
            self.assertAlmostEqual(actual, expected, places=12, msg=f"GWL mismatch for {year}")

    def test_enso_states_match_benchmark(self):
        actual = _load_csv(config.PREPROCESSED_DIR / config.ENSO_STATE_FILE)
        pd.testing.assert_frame_equal(actual, self.enso_fixture, check_dtype=False)

    def test_ibtracs_counts_match_benchmark(self):
        ibtracs_df = _load_csv(config.PREPROCESSED_DIR / config.IBTRACS_PROPERTIES_FILE)
        actual_counts = {
            "rows": int(len(ibtracs_df)),
            "lmi_storms": int((ibtracs_df["LMI_ms"] >= config.TROPICAL_STORM_THRESHOLD_MS).sum()),
            "lmi_hurricanes": int((ibtracs_df["LMI_ms"] >= config.SSHS_THRESHOLDS_MS["Cat1"][0]).sum()),
            "north_atlantic_lfi_storms": int((ibtracs_df["NorthAtlantic_LFI_ms"] >= config.TROPICAL_STORM_THRESHOLD_MS).sum()),
            "north_atlantic_lfi_hurricanes": int((ibtracs_df["NorthAtlantic_LFI_ms"] >= config.SSHS_THRESHOLDS_MS["Cat1"][0]).sum()),
        }
        self.assertEqual(actual_counts, self.benchmark["ibtracs_counts"])


@unittest.skipUnless(DETRENDED_OUTPUTS_PRESENT, "Requires local detrended outputs")
class TestDetrendedBenchmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = _load_json(DETRENDED_FIXTURE_DIR / "regression_benchmarks.json")

    def test_regression_files_exist(self):
        required_paths = [
            config.DETRENDED_DIR / "PI_regression_vs_GWL.nc",
            config.DETRENDED_DIR / "CGI_regression_vs_GWL.nc",
        ]
        for path in required_paths:
            self.assertTrue(path.exists(), f"Missing detrended regression file: {path}")

    def test_regression_parameters_match_benchmarks(self):
        for filename in ["PI_regression_vs_GWL.nc", "CGI_regression_vs_GWL.nc"]:
            benchmark = self.benchmark[filename]
            dataset = xr.open_dataset(config.DETRENDED_DIR / filename)
            try:
                self.assertEqual(sorted(dataset.data_vars), benchmark["data_vars"])
                self.assertEqual(dict(dataset.sizes), benchmark["sizes"])

                slope = dataset["slope"]
                intercept = dataset["intercept"]
                self.assertEqual(int(np.isfinite(slope.values).sum()), benchmark["non_null_cells"])
                self.assertAlmostEqual(float(np.nanmean(slope.values)), benchmark["slope_mean"], places=12)
                self.assertAlmostEqual(float(np.nanmean(intercept.values)), benchmark["intercept_mean"], places=12)

                for sample in benchmark["samples"]:
                    indexer = {dim: sample["coords"][dim] for dim in slope.dims}
                    actual_slope = float(slope.sel(indexer, method="nearest").values)
                    actual_intercept = float(intercept.sel(indexer, method="nearest").values)
                    self.assertAlmostEqual(actual_slope, sample["slope"], places=12)
                    self.assertAlmostEqual(actual_intercept, sample["intercept"], places=12)
            finally:
                dataset.close()


@unittest.skipUnless(SIMULATION_INPUTS_PRESENT, "Requires local preprocessing and detrending inputs")
class TestSimulationBenchmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.historical_fixture = _sort_simulation_frame(
            _load_csv(SIMULATED_FIXTURE_DIR / "simulated_events_historical_NorthAtlantic_n3.csv")
        )
        cls.gwl_fixture = _sort_simulation_frame(
            _load_csv(SIMULATED_FIXTURE_DIR / "simulated_events_GWL1.24_NorthAtlantic_n3.csv")
        )
        cls.historical_run = _sort_simulation_frame(
            run_simulation(
                start_year=config.DEFAULT_START_YEAR,
                end_year=config.DEFAULT_END_YEAR,
                n_iter=3,
                seed=config.RANDOM_SEED,
                use_historical=True,
                region="NorthAtlantic",
                verbose=False,
            )
        )
        cls.gwl_run = _sort_simulation_frame(
            run_simulation(
                start_year=config.DEFAULT_START_YEAR,
                end_year=config.DEFAULT_END_YEAR,
                n_iter=3,
                seed=config.RANDOM_SEED,
                target_gwl=1.24,
                use_historical=False,
                region="NorthAtlantic",
                verbose=False,
            )
        )

    def test_fixture_simulation_files_exist(self):
        for path in [
            SIMULATED_FIXTURE_DIR / "simulated_events_historical_NorthAtlantic_n3.csv",
            SIMULATED_FIXTURE_DIR / "simulated_events_GWL1.24_NorthAtlantic_n3.csv",
        ]:
            self.assertTrue(path.exists(), f"Missing simulation fixture: {path}")

    def test_artifact_names_use_canonical_suffixes(self):
        historical = SimulationArtifact(region="NorthAtlantic", n_iter=3, data_mode="historical")
        gwl_target = TargetSpec.from_inputs(target_gwl=1.24)
        gwl_artifact = SimulationArtifact(region="NorthAtlantic", n_iter=3, target_spec=gwl_target)
        pi_only_year = SimulationArtifact.from_mode("pi_only_year_2050", "NorthAtlantic", 3)

        self.assertEqual(historical.simulation_filename(), "simulated_events_historical_NorthAtlantic_n3.csv")
        self.assertEqual(gwl_artifact.simulation_filename(), "simulated_events_GWL1.24_NorthAtlantic_n3.csv")
        self.assertEqual(pi_only_year.simulation_filename(), "simulated_events_pi_only_to_2050_NorthAtlantic_n3.csv")

    def test_historical_simulation_matches_benchmark(self):
        pd.testing.assert_frame_equal(
            self.historical_run,
            self.historical_fixture,
            check_dtype=False,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_gwl_simulation_matches_benchmark(self):
        pd.testing.assert_frame_equal(
            self.gwl_run,
            self.gwl_fixture,
            check_dtype=False,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_simulation_covers_full_year_iteration_grid(self):
        expected_combinations = (config.DEFAULT_END_YEAR - config.DEFAULT_START_YEAR + 1) * 3
        actual_combinations = len(self.historical_run[["year", "iteration"]].drop_duplicates())
        self.assertEqual(actual_combinations, expected_combinations)


class TestResamplingInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.historical = _load_csv(SIMULATED_FIXTURE_DIR / "simulated_events_historical_NorthAtlantic_n3.csv")
        cls.gwl = _load_csv(SIMULATED_FIXTURE_DIR / "simulated_events_GWL1.24_NorthAtlantic_n3.csv")

    def test_change_rates_are_well_formed(self):
        base_rates = calculate_landfall_rates(self.historical)
        target_rates = calculate_landfall_rates(self.gwl)
        change_rates = calculate_change_rates(base_rates, target_rates)

        self.assertEqual(list(base_rates.keys()), ["TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"])
        self.assertEqual(list(target_rates.keys()), ["TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"])
        self.assertEqual(list(change_rates.keys()), ["TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"])

        for category, rate in base_rates.items():
            self.assertGreaterEqual(rate, 0.0, f"Negative base rate for {category}")
        for category, rate in target_rates.items():
            self.assertGreaterEqual(rate, 0.0, f"Negative target rate for {category}")
        for category, rate in change_rates.items():
            self.assertFalse(pd.isna(rate), f"NaN change rate for {category}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
