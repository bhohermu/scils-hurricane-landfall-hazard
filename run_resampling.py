#!/usr/bin/env python
"""
Resampling script for SCILS TC Model.

Calculates landfall rate change factors per SSHS category between a base period
and a target period, for use in resampling catastrophe model outputs.

Usage examples:
    # Base: historical (non-detrended), Target: GWL 2.0
    python run_resampling.py --base-historical --target-gwl 2.0 --region NorthAtlantic
    
    # Base: historical subsample 1998-2009, Target: GWL 2.0
    python run_resampling.py --base-start 1998 --base-end 2009 --base-type subsample --target-gwl 2.0 --region NorthAtlantic
    
    # Base: full simulation at GWL 1.2, Target: GWL 2.0
    python run_resampling.py --base-gwl 1.2 --base-type full --target-gwl 2.0 --region NorthAtlantic
    
    # Base: full simulation at midpoint of 1998-2009, Target: 2050
    python run_resampling.py --base-start 1998 --base-end 2009 --base-type full --target-year 2050 --region CONUS
    
    # With ENSO filtering
    python run_resampling.py --base-gwl 1.2 --target-gwl 2.0 --target-enso "La Nina" --region NorthAtlantic
    
    # With diagnostic plots for detrending/simulation
    python run_resampling.py --base-historical --target-gwl 2.0 --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ==============================================================================
# ARGUMENT PARSER - Edit this section to configure run parameters
# ==============================================================================

def create_parser():
    """Create and return the argument parser with all options."""
    parser = argparse.ArgumentParser(
        description='Calculate landfall rate change factors for resampling.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Base: historical (non-detrended), Target: GWL 2.0
  python run_resampling.py --base-historical --target-gwl 2.0
  
  # Base: historical subsample 1998-2009, Target: GWL 2.0
  python run_resampling.py --base-start 1998 --base-end 2009 --base-type subsample --target-gwl 2.0
  
  # Base: full simulation at GWL 1.2, Target: GWL 2.0  
  python run_resampling.py --base-gwl 1.2 --base-type full --target-gwl 2.0
  
  # Base: full simulation at midpoint of 1998-2009, Target: 2050
  python run_resampling.py --base-start 1998 --base-end 2009 --base-type full --target-year 2050
  
  # With diagnostic plots for detrending/simulation
  python run_resampling.py --base-historical --target-gwl 2.0 --plot
        """
    )
    
    # Base period specification
    base_group = parser.add_argument_group('Base period specification')
    base_group.add_argument('--base-start', type=int,
                           help='Start year of base period (use with --base-end)')
    base_group.add_argument('--base-end', type=int,
                           help='End year of base period (use with --base-start)')
    base_group.add_argument('--base-gwl', type=float,
                           help='GWL for base period (alternative to year range)')
    base_group.add_argument('--base-historical', action='store_true',
                           help='Use historical (non-detrended) data for base period')
    base_group.add_argument('--base-type', type=str, choices=['subsample', 'full'],
                           default='full',
                           help='Base period type: subsample (filter historical) or full (run simulation at midpoint/GWL)')
    
    # Target period specification
    target_group = parser.add_argument_group('Target period specification')
    target_exc = target_group.add_mutually_exclusive_group(required=True)
    target_exc.add_argument('--target-year', type=int,
                           help='Target year for detrending')
    target_exc.add_argument('--target-gwl', type=float,
                           help='Target GWL for detrending')
    target_exc.add_argument('--target-historical', action='store_true',
                           help='Use historical (non-detrended) data for target period')
    
    # Sensitivity modes
    target_group.add_argument('--target-pi-only', action='store_true',
                             help='Use PI-only simulation for target (detrended PI with historical CGI)')
    target_group.add_argument('--target-cgi-only', action='store_true',
                             help='Use CGI-only simulation for target (detrended CGI with historical PI)')
    
    # ENSO filtering
    enso_group = parser.add_argument_group('ENSO state filtering')
    enso_group.add_argument('--base-enso', type=str, 
                           choices=['El Nino', 'Neutral', 'La Nina', 'all'],
                           default='all',
                           help='ENSO state filter for base period (default: all)')
    enso_group.add_argument('--target-enso', type=str,
                           choices=['El Nino', 'Neutral', 'La Nina', 'all'],
                           default='all',
                           help='ENSO state filter for target period (default: all)')
    
    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation parameters')
    sim_group.add_argument('--region', type=str, choices=['CONUS', 'NorthAtlantic'],
                          default='NorthAtlantic',
                          help='Region for landfall analysis')
    sim_group.add_argument('--n-iter', type=int, default=1000,
                          help='Number of iterations per year for simulations')
    
    # Output options
    output_group = parser.add_argument_group('Output options')
    output_group.add_argument('--output', type=str,
                             help='Output CSV file path (default: auto-generated in resampled folder)')
    output_group.add_argument('--no-plot', action='store_true',
                             help='Skip generating comparison plot')
    output_group.add_argument('--plot', action='store_true',
                             help='Generate diagnostic plots for detrending and simulation steps')
    output_group.add_argument('--force', action='store_true',
                             help='Force re-run even if results file already exists')
    output_group.add_argument('--verbose', action='store_true', default=True,
                             help='Print progress information')
    
    return parser


# Create parser at module level for easy access
parser = create_parser()


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main entry point for resampling calculation."""
    import config
    from scils_tc.resampling import (
        SSHS_CATEGORIES,
        calculate_change_rates,
        calculate_landfall_rates,
        generate_output_filename,
        get_gwl_for_year,
        get_year_for_gwl,
        load_simulation,
        run_simulation_if_needed,
    )
    from scils_tc.resampling.plotting import plot_rate_comparison
    
    args = parser.parse_args()
    
    # Validate base period specification
    if args.base_historical:
        # Historical mode - no detrending
        if args.base_gwl is not None:
            parser.error("Cannot specify both --base-historical and --base-gwl")
        base_gwl = None
        base_years = list(range(args.base_start, args.base_end + 1)) if args.base_start else None
        base_label = "historical"
        base_is_historical = True
    elif args.base_gwl is not None:
        if args.base_start is not None or args.base_end is not None:
            parser.error("Cannot specify both --base-gwl and --base-start/--base-end")
        base_gwl = args.base_gwl
        base_years = None
        base_label = f"GWL{base_gwl:.2f}"
        base_is_historical = False
    else:
        if args.base_start is None or args.base_end is None:
            parser.error("Must specify either --base-gwl, --base-historical, or both --base-start and --base-end")
        base_years = list(range(args.base_start, args.base_end + 1))
        midpoint_year = (args.base_start + args.base_end) / 2
        base_gwl = get_gwl_for_year(midpoint_year)
        base_label = f"{args.base_start}-{args.base_end}"
        base_is_historical = False
    
    # Determine target specification
    if args.target_historical:
        target_mode = "historical"
        target_label = "historical"
    elif args.target_gwl is not None:
        if args.target_pi_only:
            target_mode = f"pi_only_GWL_{args.target_gwl:.2f}"
            target_label = f"GWL{args.target_gwl:.2f}_pi_only"
        elif args.target_cgi_only:
            target_mode = f"cgi_only_GWL_{args.target_gwl:.2f}"
            target_label = f"GWL{args.target_gwl:.2f}_cgi_only"
        else:
            target_mode = f"GWL_{args.target_gwl:.2f}"
            target_label = f"GWL{args.target_gwl:.2f}"
        get_year_for_gwl(args.target_gwl)
    else:
        if args.target_pi_only:
            target_mode = f"pi_only_year_{args.target_year}"
            target_label = f"{args.target_year}_pi_only"
        elif args.target_cgi_only:
            target_mode = f"cgi_only_year_{args.target_year}"
            target_label = f"{args.target_year}_cgi_only"
        else:
            target_mode = f"year_{args.target_year}"
            target_label = str(args.target_year)
    
    # ENSO filter conversion
    base_enso = None if args.base_enso == 'all' else args.base_enso
    target_enso = None if args.target_enso == 'all' else args.target_enso
    
    # Determine output path early to check if it exists
    if args.output:
        output_path = Path(args.output)
    else:
        output_basename = generate_output_filename(
            base_label, target_label, args.region,
            args.base_enso, args.target_enso
        )
        output_path = config.get_resampled_path(f"{output_basename}.csv")
    
    # Check if we should skip (default: skip if exists, unless --force)
    if output_path.exists() and not args.force:
        print("=" * 60)
        print("SCILS TC Model - Resampling Change Rate Calculation")
        print("=" * 60)
        print(f"Results already exist: {output_path}")
        print("Skipping calculation (use --force to re-run)")
        print("=" * 60)
        return 0
    
    print("=" * 60)
    print("SCILS TC Model - Resampling Change Rate Calculation")
    print("=" * 60)
    print(f"Region: {args.region}")
    print(f"Iterations: {args.n_iter}")
    print()
    print(f"Base period: {base_label} (type: {args.base_type})")
    if base_is_historical:
        print("  Using historical (non-detrended) data")
    elif base_years:
        print(f"  Years: {args.base_start}-{args.base_end} (midpoint GWL: {base_gwl:.2f}°C)")
    else:
        print(f"  GWL: {base_gwl:.2f}°C")
    if base_enso:
        print(f"  ENSO filter: {base_enso}")
    print()
    print(f"Target period: {target_label}")
    if args.target_historical:
        print("  Using historical (non-detrended) data")
    if target_enso:
        print(f"  ENSO filter: {target_enso}")
    if args.plot:
        print("  Diagnostic plots: enabled")
    print()
    
    # Step 1: Load or run base simulation
    print("Step 1: Preparing base period simulation...")
    
    if args.base_type == 'subsample':
        # Use historical simulation, will filter to specific years later
        base_mode = "historical"
        base_filepath = run_simulation_if_needed(base_mode, args.region, args.n_iter, args.verbose, 
                                                  generate_plots=args.plot)
        base_df = load_simulation(base_filepath)
        base_filter_years = base_years
    elif base_is_historical:
        # Use historical (non-detrended) simulation
        base_mode = "historical"
        base_filepath = run_simulation_if_needed(base_mode, args.region, args.n_iter, args.verbose,
                                                  generate_plots=args.plot)
        base_df = load_simulation(base_filepath)
        base_filter_years = base_years  # Optional year filter
    else:
        # Full: run simulation at the GWL
        if base_years:
            # Convert year range to equivalent simulation at midpoint
            midpoint_year = (args.base_start + args.base_end) / 2
            base_mode = f"year_{int(round(midpoint_year))}"
        else:
            base_mode = f"GWL_{base_gwl:.2f}"
        base_filepath = run_simulation_if_needed(base_mode, args.region, args.n_iter, args.verbose,
                                                  generate_plots=args.plot)
        base_df = load_simulation(base_filepath)
        base_filter_years = None  # Use all years from simulation
    
    # Step 2: Load or run target simulation
    print("\nStep 2: Preparing target period simulation...")
    target_filepath = run_simulation_if_needed(target_mode, args.region, args.n_iter, args.verbose,
                                                generate_plots=args.plot)
    target_df = load_simulation(target_filepath)
    
    # Step 3: Calculate landfall rates
    print("\nStep 3: Calculating landfall rates...")
    
    base_rates = calculate_landfall_rates(base_df, years=base_filter_years, enso_filter=base_enso)
    target_rates = calculate_landfall_rates(target_df, years=None, enso_filter=target_enso)
    
    print("\n  Base period rates (annual landfalls per category):")
    for cat in SSHS_CATEGORIES:
        print(f"    {cat}: {base_rates[cat]:.4f}")
    
    print("\n  Target period rates (annual landfalls per category):")
    for cat in SSHS_CATEGORIES:
        print(f"    {cat}: {target_rates[cat]:.4f}")
    
    # Step 4: Calculate change rates
    print("\nStep 4: Calculating change rates...")
    change_rates = calculate_change_rates(base_rates, target_rates)
    
    print("\n  Change rates (target / base):")
    for cat in SSHS_CATEGORIES:
        rate = change_rates[cat]
        if np.isinf(rate):
            print(f"    {cat}: inf (new category in target)")
        else:
            print(f"    {cat}: {rate:.4f}")
    
    # Step 5: Save results
    print("\nStep 5: Saving results...")
    
    # Ensure resampled directory exists
    config.ensure_resampled_dir()
    
    # Build output dataframe
    results = []
    for cat in SSHS_CATEGORIES:
        results.append({
            'sshs_category': cat,
            'base_rate': base_rates[cat],
            'target_rate': target_rates[cat],
            'change_rate': change_rates[cat],
        })
    
    results_df = pd.DataFrame(results)
    
    # Add metadata
    metadata = {
        'region': args.region,
        'base_label': base_label,
        'base_type': args.base_type,
        'base_enso': args.base_enso,
        'target_label': target_label,
        'target_enso': args.target_enso,
        'n_iter': args.n_iter,
    }
    
    # output_path was already determined at the start of the function
    
    # Save with metadata in header comments
    with open(output_path, 'w') as f:
        f.write("# SCILS TC Model - Landfall Rate Change Factors\n")
        f.write(f"# Region: {metadata['region']}\n")
        f.write(f"# Base: {metadata['base_label']} ({metadata['base_type']}, ENSO: {metadata['base_enso']})\n")
        f.write(f"# Target: {metadata['target_label']} (ENSO: {metadata['target_enso']})\n")
        f.write(f"# Iterations: {metadata['n_iter']}\n")
        f.write("#\n")
    
    results_df.to_csv(output_path, mode='a', index=False)
    print(f"  Results saved to: {output_path}")
    
    # Step 6: Generate plot
    if not args.no_plot:
        print("\nStep 6: Generating comparison plot...")
        
        plot_basename = generate_output_filename(
            base_label, target_label, args.region,
            args.base_enso, args.target_enso
        )
        plot_path = config.get_resampled_path(f"{plot_basename}.png")
        
        plot_rate_comparison(
            base_rates, target_rates, change_rates,
            base_label, target_label, args.region,
            base_enso=args.base_enso, target_enso=args.target_enso,
            output_path=plot_path
        )
    
    print("\n" + "=" * 60)
    print("Resampling change rate calculation complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
