#!/usr/bin/env python
"""Run batch simulations for the standard and sensitivity scenarios."""

from __future__ import annotations

import argparse
import sys

import config
from batch_utils import (
    DEFAULT_SEED,
    build_python_command,
    normalize_regions,
    resolve_scenarios,
    run_command,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch simulation runs."""
    parser = argparse.ArgumentParser(description="Run simulations for the configured target scenarios.")
    parser.add_argument("--start-year", type=int, default=config.START_YEAR, help=f"First year to simulate (default: {config.START_YEAR})")
    parser.add_argument("--end-year", type=int, default=config.END_YEAR, help=f"Last year to simulate (default: {config.END_YEAR})")
    parser.add_argument("--n-iter", type=int, default=1000, help="Iterations per year (default: 1000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--region",
        action="append",
        choices=["CONUS", "NorthAtlantic"],
        help="Region to simulate. Repeat the flag to run more than one region.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Scenario name to include. Repeat the flag to run a subset.",
    )
    parser.add_argument("--plot", action="store_true", help="Generate validation plots during simulation.")
    parser.add_argument("--force", action="store_true", help="Re-run even if a simulation output already exists.")
    return parser.parse_args()


def main() -> int:
    """Run simulations for all selected scenarios and regions."""
    args = parse_args()
    regions = normalize_regions(args.region)
    scenarios = resolve_scenarios(args.scenario)

    print("=" * 72)
    print("SCILS TC Model - Batch Simulations")
    print("=" * 72)
    print(f"Years: {args.start_year}-{args.end_year}")
    print(f"Iterations: {args.n_iter}")
    print(f"Regions: {regions}")
    print(f"Scenarios: {[scenario.name for scenario in scenarios]}")
    print("=" * 72)

    results: list[tuple[str, str, bool]] = []
    for scenario in scenarios:
        for region in regions:
            cmd = build_python_command(
                "run_simulation.py",
                "--start-year", str(args.start_year),
                "--end-year", str(args.end_year),
                "--n-iter", str(args.n_iter),
                "--region", region,
                "--seed", str(args.seed),
                *scenario.simulation_args,
            )
            if args.plot:
                cmd.append("--plot")
            if args.force:
                cmd.append("--force")

            success = run_command(cmd, f"Simulation: {scenario.name} / {region}")
            results.append((scenario.name, region, success))

    print("\n" + "=" * 72)
    print("BATCH SIMULATION SUMMARY")
    print("=" * 72)
    for scenario_name, region, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  {scenario_name:20} / {region:15} : {status}")

    return 0 if all(success for _, _, success in results) else 1


if __name__ == "__main__":
    sys.exit(main())
