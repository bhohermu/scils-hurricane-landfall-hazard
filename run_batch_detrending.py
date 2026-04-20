#!/usr/bin/env python
"""Run detrending for the configured batch scenarios."""

from __future__ import annotations

import argparse
import sys

import config
from batch_utils import (
    build_python_command,
    resolve_scenarios,
    run_command,
    unique_detrending_scenarios,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch detrending."""
    parser = argparse.ArgumentParser(description="Run detrending for the standard target scenarios.")
    parser.add_argument("--start-year", type=int, default=config.START_YEAR, help=f"First year to process (default: {config.START_YEAR})")
    parser.add_argument("--end-year", type=int, default=config.END_YEAR, help=f"Last year to process (default: {config.END_YEAR})")
    parser.add_argument(
        "--scenario",
        action="append",
        help="Scenario name to include. Repeat the flag to run a subset.",
    )
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots.")
    parser.add_argument(
        "--diagnostics",
        choices=["none", "first", "all"],
        default="first",
        help="Diagnostic mean-trend plots to generate (default: first).",
    )
    parser.add_argument("--force-recalculate", action="store_true", help="Force recalculation of regression fits.")
    return parser.parse_args()


def main() -> int:
    """Run detrending once per unique target scenario."""
    args = parse_args()
    scenarios = unique_detrending_scenarios(resolve_scenarios(args.scenario))

    print("=" * 72)
    print("SCILS TC Model - Batch Detrending")
    print("=" * 72)
    print(f"Years: {args.start_year}-{args.end_year}")
    print(f"Targets: {[scenario.name for scenario in scenarios]}")
    print(f"Diagnostics: {args.diagnostics}")
    print("=" * 72)

    results: list[tuple[str, bool]] = []
    for index, scenario in enumerate(scenarios):
        cmd = build_python_command(
            "run_detrending.py",
            "--start-year", str(args.start_year),
            "--end-year", str(args.end_year),
            *scenario.detrending_args,
        )
        if args.plot:
            cmd.append("--plot")

        diagnostics_mode = args.diagnostics
        if diagnostics_mode == "none" or (diagnostics_mode == "first" and index > 0):
            cmd.append("--skip-diagnostics")
        if args.force_recalculate:
            cmd.append("--force-recalculate")

        success = run_command(cmd, f"Detrending: {scenario.name}")
        results.append((scenario.name, success))

    print("\n" + "=" * 72)
    print("BATCH DETRENDING SUMMARY")
    print("=" * 72)
    for name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name:20} : {status}")

    return 0 if all(success for _, success in results) else 1


if __name__ == "__main__":
    sys.exit(main())