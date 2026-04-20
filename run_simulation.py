#!/usr/bin/env python
"""
Run tropical cyclone simulation for SCILS TC Model (FINAL version).

This script generates synthetic tropical cyclone events based on:
- CGI-driven annual storm counts (Poisson distribution)
- Monthly distribution of storms (Multinomial)
- LMI location sampling from KDE maps conditioned on SST/ENSO
- PI-based intensity assignment using single histogram (1dist method)
- LFI/LMI ratio sampling using 0-1 inflated Beta distribution (beta method)

Usage:
    python run_simulation.py [options]

Examples:
    # Run with default settings (2023-2025, 100 iterations, target year 2025)
    python run_simulation.py

    # Run for specific years with more iterations
    python run_simulation.py --start-year 2020 --end-year 2024 --n-iter 500

    # Run with target GWL instead of year
    python run_simulation.py --target-gwl 1.5

    # Run with random (non-reproducible) seed
    python run_simulation.py --random-seed
"""

import argparse
import sys

import pandas as pd

import config
from scils_tc.simulation.plotting import plot_simulation_validation
from scils_tc.simulation.simulation import run_simulation, save_simulation_results
from scils_tc.utils import SimulationArtifact, TargetSpec, actual_event_rows, count_actual_events


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run tropical cyclone simulation for SCILS TC Model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--start-year', type=int, default=config.START_YEAR,
        help=f'First year to simulate (default: {config.START_YEAR})'
    )
    parser.add_argument(
        '--end-year', type=int, default=config.END_YEAR,
        help=f'Last year to simulate (default: {config.END_YEAR})'
    )
    parser.add_argument(
        '--n-iter', type=int, default=config.N_ITER,
        help=f'Number of iterations per year (default: {config.N_ITER})'
    )
    parser.add_argument(
        '--target-year', type=int, default=None,
        help=f'Target year for detrended data (default: {config.DEFAULT_TARGET_YEAR})'
    )
    parser.add_argument(
        '--target-gwl', type=float, default=None,
        help='Target GWL for detrended data (e.g., 1.5). Overridden by --target-year.'
    )
    parser.add_argument(
        '--use-historical', action='store_true',
        help='Use historical (non-detrended) PI, CGI, and SST instead of detrended data'
    )
    parser.add_argument(
        '--detrend-pi-only', action='store_true',
        help='Use detrended PI with historical CGI (mutually exclusive with --use-historical)'
    )
    parser.add_argument(
        '--detrend-cgi-only', action='store_true',
        help='Use detrended CGI with historical PI (mutually exclusive with --use-historical)'
    )
    parser.add_argument(
        '--seed', type=int, default=config.RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {config.RANDOM_SEED})'
    )
    parser.add_argument(
        '--random-seed', action='store_true',
        help='Use random (non-reproducible) seed instead of fixed seed'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not save results to file (print summary only)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate validation plots comparing simulated vs observed landfalls'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force re-run simulation even if results file already exists'
    )
    parser.add_argument(
        '--region', type=str, default=config.USE_REGION, choices=['CONUS', 'NorthAtlantic'],
        help=f'Region for landfall analysis (default: {config.USE_REGION})'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate mutually exclusive options
    options_set = sum([args.use_historical, args.detrend_pi_only, args.detrend_cgi_only])
    if options_set > 1:
        print("Error: --use-historical, --detrend-pi-only, and --detrend-cgi-only are mutually exclusive")
        sys.exit(1)

    # Handle seed
    seed = None if args.random_seed else args.seed

    # Handle target year/GWL - resolve using GWL lookup
    target_year = args.target_year
    target_gwl = args.target_gwl
    target_spec = None

    # Resolve target (use default GWL if neither specified)
    if target_year is None and target_gwl is None and not args.use_historical:
        target_gwl = config.DEFAULT_TARGET_GWL
    
    # Get effective values and filename suffix
    if not args.use_historical:
        target_spec = TargetSpec.from_inputs(
            target_year=target_year,
            target_gwl=target_gwl,
            default_gwl=config.DEFAULT_TARGET_GWL,
        )

    # Determine data mode and output filename for checking if results exist
    if args.use_historical:
        data_mode = "historical"
    elif args.detrend_pi_only:
        data_mode = "pi_only"
    elif args.detrend_cgi_only:
        data_mode = "cgi_only"
    else:
        data_mode = None  # Standard detrended

    artifact = SimulationArtifact(
        region=args.region,
        n_iter=args.n_iter,
        data_mode=data_mode,
        target_spec=target_spec,
    )
    output_path = config.SIMULATED_DIR / artifact.simulation_filename()

    # Check if we should skip simulation (default: skip if exists, unless --force)
    if output_path.exists() and not args.force:
        if not args.quiet:
            print("\n" + "=" * 60)
            print(f"Simulation results already exist: {output_path}")
            print("Skipping simulation (use --force to re-run)")
            print("=" * 60)

        # Load existing results
        df = pd.read_csv(output_path)

        if not args.quiet:
            print(f"\nLoaded {count_actual_events(df)} events from existing results")
    else:
        # Run simulation
        df = run_simulation(
            start_year=args.start_year,
            end_year=args.end_year,
            n_iter=args.n_iter,
            target_year=target_year,
            target_gwl=target_gwl,
            use_historical=args.use_historical,
            detrend_pi_only=args.detrend_pi_only,
            detrend_cgi_only=args.detrend_cgi_only,
            region=args.region,
            seed=seed,
            verbose=not args.quiet
        )

        # Save results
        if not args.no_save:
            output_path = save_simulation_results(df, target_year, target_gwl,
                                                   data_mode=data_mode, region=args.region,
                                                   n_iter=args.n_iter)
            if not args.quiet:
                print(f"\nResults saved to: {output_path}")

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Years: {args.start_year} - {args.end_year}")
        if not (output_path.exists() and not args.force):
            print(f"Iterations per year: {args.n_iter}")
        actual_df = actual_event_rows(df)
        print(f"Total events: {count_actual_events(df)}")

        if len(actual_df) > 0:
            print("\nEvents per year:")
            year_counts = actual_df.groupby('year').size()
            for year in range(args.start_year, args.end_year + 1):
                count = int(year_counts.get(year, 0))
                avg = count / args.n_iter
                print(f"  {year}: {count} total ({avg:.1f} avg/iter)")

            print("\nIntensity statistics (LMI, m/s):")
            print(f"  Mean: {actual_df['lmi'].mean():.1f}")
            print(f"  Median: {actual_df['lmi'].median():.1f}")
            print(f"  Max: {actual_df['lmi'].max():.1f}")
            print(f"  Min (valid): {actual_df['lmi'].dropna().min():.1f}")

            print("\nLandfall intensity statistics (LFI, m/s):")
            print(f"  Mean: {actual_df['lfi'].mean():.1f}")
            print(f"  Median: {actual_df['lfi'].median():.1f}")

            print("\nEvents by region:")
            region_counts = actual_df.groupby('region_name').size().sort_values(ascending=False)
            for region, count in region_counts.items():
                if region is not None:
                    print(f"  {region}: {count}")
        print("=" * 60)

    # Generate validation plots if requested
    if args.plot and not args.no_save:
        if not args.quiet:
            print("\nGenerating validation plots...")

        # Load observed data
        obs_file = config.get_output_path("IBTrACS_properties.csv")
        if obs_file.exists():
            obs_df = pd.read_csv(obs_file)

            # Generate plot filename with region and n_iter
            plot_path = config.SIMULATED_DIR / artifact.validation_plot_filename()

            plot_simulation_validation(
                sim_df=df,
                obs_df=obs_df,
                start_year=args.start_year,
                end_year=args.end_year,
                save_path=plot_path,
                region=args.region
            )
        else:
            print(f"Warning: Could not find observed data at {obs_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
