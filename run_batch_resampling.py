#!/usr/bin/env python
"""Run change-rate and YELT resampling for all configured target scenarios."""

from __future__ import annotations

import argparse
import shutil
import sys

from batch_utils import (
    DEFAULT_SEED,
    build_python_command,
    normalize_regions,
    resolve_scenarios,
    run_command,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch resampling."""
    parser = argparse.ArgumentParser(description="Run change-rate and YELT resampling for all configured scenarios.")
    parser.add_argument("--n-iter", type=int, default=1000, help="Iterations per year for change-rate simulations (default: 1000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for YELT resampling (default: {DEFAULT_SEED})")
    parser.add_argument("--n-years", type=int, default=10000, help="Number of YELT years (default: 10000)")
    parser.add_argument(
        "--region",
        action="append",
        choices=["CONUS", "NorthAtlantic"],
        help="Region to resample. Repeat the flag to run more than one region.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Scenario name to include. Repeat the flag to run a subset.",
    )
    parser.add_argument(
        "--resampling-type",
        choices=["deterministic", "poisson"],
        default="deterministic",
        help="YELT resampling method (default: deterministic)",
    )
    parser.add_argument(
        "--resampling-iterations",
        type=int,
        default=100,
        help="Number of YELT Poisson iterations when --resampling-type poisson is used (default: 100)",
    )
    parser.add_argument("--skip-change-rates", action="store_true", help="Skip change-rate generation and only run YELT resampling.")
    parser.add_argument("--skip-yelt", action="store_true", help="Skip YELT resampling and only generate change-rate CSVs.")
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plots in the change-rate stage.")
    parser.add_argument("--force", action="store_true", help="Re-run change-rate generation even if outputs already exist.")
    return parser.parse_args()


def main() -> int:
    """Run change-rate generation and optional YELT resampling for each scenario."""
    args = parse_args()
    regions = normalize_regions(args.region)
    scenarios = [scenario for scenario in resolve_scenarios(args.scenario) if not scenario.is_historical]

    print("=" * 72)
    print("SCILS TC Model - Batch Resampling")
    print("=" * 72)
    print(f"Iterations: {args.n_iter}")
    print(f"Regions: {regions}")
    print(f"Scenarios: {[scenario.name for scenario in scenarios]}")
    print(f"Change rates: {'skipped' if args.skip_change_rates else 'enabled'}")
    print(f"YELT: {'skipped' if args.skip_yelt else args.resampling_type}")
    print("=" * 72)

    results: list[tuple[str, str, str, bool]] = []

    for scenario in scenarios:
        for region in regions:
            if not args.skip_change_rates:
                change_rates_cmd = build_python_command(
                    "run_resampling.py",
                    "--base-historical",
                    "--region", region,
                    "--n-iter", str(args.n_iter),
                    *scenario.resampling_args,
                )
                if args.plot:
                    change_rates_cmd.append("--plot")
                if args.force:
                    change_rates_cmd.append("--force")

                success = run_command(change_rates_cmd, f"Change rates: historical -> {scenario.name} / {region}")
                results.append((scenario.name, region, "change-rates", success))
                if not success:
                    continue

            if args.skip_yelt:
                continue

            change_rates_path = scenario.change_rates_path(region=region)
            if not change_rates_path.exists():
                print(f"\nERROR: Change-rate file not found: {change_rates_path}")
                results.append((scenario.name, region, "yelt", False))
                continue

            output_dir = scenario.yelt_output_dir(region=region)
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(change_rates_path, output_dir / change_rates_path.name)

            yelt_cmd = build_python_command(
                "run_yelt_resampling.py",
                "--change-rates", str(change_rates_path),
                "--output-dir", str(output_dir),
                "--resampling-type", args.resampling_type,
                "--seed", str(args.seed),
                "--n-years", str(args.n_years),
            )
            if args.resampling_type == "poisson":
                yelt_cmd.extend(["--resampling-iterations", str(args.resampling_iterations)])

            success = run_command(yelt_cmd, f"YELT: historical -> {scenario.name} / {region}")
            results.append((scenario.name, region, "yelt", success))

    print("\n" + "=" * 72)
    print("BATCH RESAMPLING SUMMARY")
    print("=" * 72)
    for scenario_name, region, stage, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  {stage:13} {scenario_name:20} / {region:15} : {status}")

    return 0 if all(success for _, _, _, success in results) else 1


if __name__ == "__main__":
    sys.exit(main())