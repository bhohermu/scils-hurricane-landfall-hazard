import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

from scils_tc import __version__
from scils_tc.resampling.resampling import calculate_change_rates, calculate_landfall_rates
from scils_tc.utils import SimulationArtifact, TargetSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIMULATED_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "simulated"


class TestPublicRepositorySurface(unittest.TestCase):
    def test_version_exposed(self):
        self.assertEqual(__version__, "0.1.0")

    def test_canonical_artifact_names(self):
        historical = SimulationArtifact(region="NorthAtlantic", n_iter=3, data_mode="historical")
        gwl_artifact = SimulationArtifact(
            region="NorthAtlantic",
            n_iter=3,
            target_spec=TargetSpec(effective_gwl=1.24, effective_year=2020.0, specified_by_year=False),
        )
        self.assertEqual(historical.simulation_filename(), "simulated_events_historical_NorthAtlantic_n3.csv")
        self.assertEqual(gwl_artifact.simulation_filename(), "simulated_events_GWL1.24_NorthAtlantic_n3.csv")

    def test_fixture_rate_calculation_runs(self):
        historical = pd.read_csv(SIMULATED_FIXTURE_DIR / "simulated_events_historical_NorthAtlantic_n3.csv")
        gwl = pd.read_csv(SIMULATED_FIXTURE_DIR / "simulated_events_GWL1.24_NorthAtlantic_n3.csv")

        base_rates = calculate_landfall_rates(historical)
        target_rates = calculate_landfall_rates(gwl)
        change_rates = calculate_change_rates(base_rates, target_rates)

        self.assertEqual(list(change_rates.keys()), ["TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"])
        self.assertTrue(all(rate >= 0 for rate in base_rates.values()))
        self.assertTrue(all(rate >= 0 for rate in target_rates.values()))

    def test_cli_help_runs(self):
        for script in [
            "run_preprocessing.py",
            "run_detrending.py",
            "run_simulation.py",
            "run_resampling.py",
            "run_yelt_resampling.py",
            "run_batch_preprocessing.py",
            "run_batch_detrending.py",
            "run_batch_simulations.py",
            "run_batch_resampling.py",
        ]:
            result = subprocess.run(
                [sys.executable, script, "--help"],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=f"Help failed for {script}: {result.stderr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
