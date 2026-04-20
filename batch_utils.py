#!/usr/bin/env python
"""Shared batch scenario definitions and command helpers for SCILS TC Model."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import config
from scils_tc.resampling import generate_output_filename

DEFAULT_REGIONS = [config.USE_REGION]
DEFAULT_SEED = 42


@dataclass(frozen=True)
class BatchScenario:
    """Define the arguments and labels for one batch scenario."""

    name: str
    target_label: str | None
    detrending_args: tuple[str, ...] = ()
    simulation_args: tuple[str, ...] = ()
    resampling_args: tuple[str, ...] = ()

    @property
    def is_historical(self) -> bool:
        """Return whether the scenario uses historical data without detrending."""
        return self.target_label is None

    @property
    def requires_detrending(self) -> bool:
        """Return whether the scenario requires detrended PI/CGI/SST inputs."""
        return bool(self.detrending_args)

    @property
    def detrending_key(self) -> tuple[str, ...]:
        """Return a hashable identifier for unique detrending targets."""
        return self.detrending_args

    def change_rates_filename(self, region: str, base_label: str = "historical") -> str:
        """Return the expected change-rate CSV name for the scenario and region."""
        if self.target_label is None:
            raise ValueError("Historical scenario does not produce change-rate files")
        return f"{generate_output_filename(base_label, self.target_label, region)}.csv"

    def change_rates_path(self, region: str, base_label: str = "historical") -> Path:
        """Return the expected change-rate CSV path for the scenario and region."""
        return config.RESAMPLED_DIR / self.change_rates_filename(region=region, base_label=base_label)

    def yelt_output_dir(self, region: str | None = None, base_label: str = "historical") -> Path:
        """Return the output directory used for YELT results for this scenario."""
        if self.target_label is None:
            raise ValueError("Historical scenario does not produce YELT results")
        region_suffix = f"_{region}" if region else ""
        return config.RESAMPLED_DIR / f"{base_label}_to_{self.target_label}{region_suffix}"


SIMULATION_SCENARIOS = [
    BatchScenario(
        name="historical",
        target_label=None,
        simulation_args=("--use-historical",),
    ),
    BatchScenario(
        name="2020",
        target_label="2020",
        detrending_args=("--target-year", "2020"),
        simulation_args=("--target-year", "2020"),
        resampling_args=("--target-year", "2020"),
    ),
    BatchScenario(
        name="2050",
        target_label="2050",
        detrending_args=("--target-year", "2050"),
        simulation_args=("--target-year", "2050"),
        resampling_args=("--target-year", "2050"),
    ),
    BatchScenario(
        name="GWL1.24",
        target_label="GWL1.24",
        detrending_args=("--target-gwl", "1.24"),
        simulation_args=("--target-gwl", "1.24"),
        resampling_args=("--target-gwl", "1.24"),
    ),
    BatchScenario(
        name="GWL2.00",
        target_label="GWL2.00",
        detrending_args=("--target-gwl", "2.00"),
        simulation_args=("--target-gwl", "2.00"),
        resampling_args=("--target-gwl", "2.00"),
    ),
    BatchScenario(
        name="GWL2.00_pi_only",
        target_label="GWL2.00_pi_only",
        detrending_args=("--target-gwl", "2.00"),
        simulation_args=("--target-gwl", "2.00", "--detrend-pi-only"),
        resampling_args=("--target-gwl", "2.00", "--target-pi-only"),
    ),
    BatchScenario(
        name="GWL2.00_cgi_only",
        target_label="GWL2.00_cgi_only",
        detrending_args=("--target-gwl", "2.00"),
        simulation_args=("--target-gwl", "2.00", "--detrend-cgi-only"),
        resampling_args=("--target-gwl", "2.00", "--target-cgi-only"),
    ),
]


def non_historical_scenarios() -> list[BatchScenario]:
    """Return scenarios that produce detrended simulations and change rates."""
    return [scenario for scenario in SIMULATION_SCENARIOS if not scenario.is_historical]


def resolve_scenarios(selected_names: list[str] | None = None) -> list[BatchScenario]:
    """Resolve a filtered scenario list while preserving the configured order."""
    if not selected_names:
        return list(SIMULATION_SCENARIOS)

    selected = set(selected_names)
    scenarios = [scenario for scenario in SIMULATION_SCENARIOS if scenario.name in selected]
    missing = selected.difference({scenario.name for scenario in scenarios})
    if missing:
        raise ValueError(f"Unknown scenarios: {', '.join(sorted(missing))}")
    return scenarios


def unique_detrending_scenarios(scenarios: list[BatchScenario]) -> list[BatchScenario]:
    """Return one scenario per unique detrending target in the provided order."""
    seen: set[tuple[str, ...]] = set()
    unique: list[BatchScenario] = []
    for scenario in scenarios:
        if not scenario.requires_detrending or scenario.detrending_key in seen:
            continue
        seen.add(scenario.detrending_key)
        unique.append(scenario)
    return unique


def run_command(cmd: list[str], description: str) -> bool:
    """Run one subprocess command and report a success flag."""
    print(f"\n{'=' * 72}")
    print(description)
    print(f"Command: {' '.join(cmd)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 72}\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed")
        return False

    print(f"\nCompleted: {description}")
    return True


def build_python_command(script_name: str, *args: str) -> list[str]:
    """Build a subprocess command that runs a repo script with the active Python."""
    return [sys.executable, script_name, *args]


def normalize_regions(regions: list[str] | None) -> list[str]:
    """Return a stable region list for batch scripts."""
    return regions if regions else list(DEFAULT_REGIONS)