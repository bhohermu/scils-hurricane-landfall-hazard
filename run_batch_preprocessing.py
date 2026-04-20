#!/usr/bin/env python
"""Run preprocessing for one or more regions using the standard entrypoint."""

from __future__ import annotations

import argparse
import sys

import config
from batch_utils import build_python_command, normalize_regions, run_command


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch preprocessing."""
    parser = argparse.ArgumentParser(description="Run preprocessing for one or more regions.")
    parser.add_argument("--start-year", type=int, default=config.START_YEAR, help=f"First year to process (default: {config.START_YEAR})")
    parser.add_argument("--end-year", type=int, default=config.END_YEAR, help=f"Last year to process (default: {config.END_YEAR})")
    parser.add_argument(
        "--region",
        action="append",
        choices=["CONUS", "NorthAtlantic"],
        help="Region to preprocess. Repeat the flag to run more than one region.",
    )
    parser.add_argument("--plot", action="store_true", help="Generate preprocessing plots.")
    parser.add_argument("--force-recalculate", action="store_true", help="Force recalculation even if outputs already exist.")
    parser.add_argument("--plot-regions", action="store_true", help="Generate the Jewson region plot.")
    parser.add_argument("--skip-gwl", action="store_true", help="Skip GWL calculation.")
    return parser.parse_args()


def main() -> int:
    """Run preprocessing sequentially for the requested regions."""
    args = parse_args()
    regions = normalize_regions(args.region)

    print("=" * 72)
    print("SCILS TC Model - Batch Preprocessing")
    print("=" * 72)
    print(f"Years: {args.start_year}-{args.end_year}")
    print(f"Regions: {regions}")
    print("=" * 72)

    results: list[tuple[str, bool]] = []
    for region in regions:
        cmd = build_python_command(
            "run_preprocessing.py",
            "--start-year", str(args.start_year),
            "--end-year", str(args.end_year),
            "--region", region,
        )
        if args.plot:
            cmd.append("--plot")
        if args.force_recalculate:
            cmd.append("--force-recalculate")
        if args.plot_regions:
            cmd.append("--plot-regions")
        if args.skip_gwl:
            cmd.append("--skip-gwl")

        success = run_command(cmd, f"Preprocessing: {region}")
        results.append((region, success))

    print("\n" + "=" * 72)
    print("BATCH PREPROCESSING SUMMARY")
    print("=" * 72)
    for region, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  {region:15} : {status}")

    return 0 if all(success for _, success in results) else 1


if __name__ == "__main__":
    sys.exit(main())