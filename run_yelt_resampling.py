#!/usr/bin/env python
"""
YELT Resampling script for SCILS TC Model.

Adjusts a Year Event Loss Table (YELT) based on climate change landfall rate 
change factors per SSHS category.

Usage examples:
    # Deterministic resampling with change rates from run_resampling.py output
    python run_yelt_resampling.py --change-rates resampled/change_rates_base_historical_target_2050_NorthAtlantic.csv
    
    # Poisson resampling with multiple iterations
    python run_yelt_resampling.py --change-rates resampled/change_rates_base_historical_target_GWL2.0_NorthAtlantic.csv --resampling-type poisson --resampling-iterations 100
    
    # Specify custom YELT file
    python run_yelt_resampling.py --yelt data/my_yelt.csv --change-rates resampled/change_rates.csv
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
        description='Resample YELT based on climate change landfall rate change factors.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deterministic resampling
  python run_yelt_resampling.py --change-rates resampled/change_rates_base_historical_target_2050_NorthAtlantic.csv
  
  # Poisson resampling with 100 iterations
  python run_yelt_resampling.py --change-rates resampled/change_rates.csv --resampling-type poisson --resampling-iterations 100
  
  # Custom YELT and seed for reproducibility
  python run_yelt_resampling.py --yelt data/my_yelt.csv --change-rates resampled/change_rates.csv --seed 42
        """
    )
    
    # Input files
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('--yelt', type=str,
                            help='Path to YELT CSV file (default: data/YELT_STORM_present_NA_10000_YEARS.csv)')
    input_group.add_argument('--change-rates', type=str, required=True,
                            help='Path to change rates CSV file (output of run_resampling.py)')
    
    # Resampling options
    resample_group = parser.add_argument_group('Resampling options')
    resample_group.add_argument('--resampling-type', type=str, 
                               choices=['deterministic', 'poisson'],
                               default='deterministic',
                               help='Resampling method: deterministic (single output) or poisson (stochastic)')
    resample_group.add_argument('--resampling-iterations', type=int, default=1,
                               help='Number of resampling iterations for poisson method (default: 1)')
    resample_group.add_argument('--n-years', type=int, default=10000,
                               help='Number of simulation years in YELT (default: 10000)')
    resample_group.add_argument('--seed', type=int, default=None,
                               help='Random seed for reproducibility')
    
    # Output options
    output_group = parser.add_argument_group('Output options')
    output_group.add_argument('--output', type=str,
                             help='Output path for resampled YELT (default: auto-generated)')
    output_group.add_argument('--output-dir', type=str,
                             help='Output directory for all results (default: resampled/)')
    output_group.add_argument('--no-plot', action='store_true',
                             help='Skip generating comparison plots')
    output_group.add_argument('--verbose', action='store_true', default=True,
                             help='Print progress information')
    
    return parser


# Create parser at module level for easy access
parser = create_parser()


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main entry point for YELT resampling."""
    import config
    from scils_tc.resampling.plotting import plot_rate_comparison, plot_rate_comparison_poisson
    from scils_tc.resampling.yelt_plotting import plot_category_event_counts, plot_yelt_comparison
    from scils_tc.resampling.yelt_resampling import (
        SSHS_CATEGORIES,
        add_lfi_sshs_category,
        calculate_yelt_landfall_rates,
        clean_yelt_for_resampling,
        compare_yelts,
        load_change_rates,
        load_yelt,
        resample_yelt,
    )
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Load YELT
    print("=" * 60)
    print("SCILS TC Model - YELT Resampling")
    print("=" * 60)
    
    yelt_path = Path(args.yelt) if args.yelt else None
    print("\nStep 1: Loading YELT...")
    yelt_raw = load_yelt(yelt_path)
    print(f"  Loaded {len(yelt_raw):,} events")
    print(f"  Years: {yelt_raw['Iteration'].nunique()} ({yelt_raw['Iteration'].min()} to {yelt_raw['Iteration'].max()})")
    
    # Add LFI_SSHS column
    print("\nStep 2: Converting LFI_ms to SSHS categories...")
    yelt_raw = add_lfi_sshs_category(yelt_raw)
    
    # Show raw counts (including TD)
    all_landfalling = yelt_raw[yelt_raw['LFI_SSHS'].notna()]
    all_non_landfalling = yelt_raw[yelt_raw['LFI_SSHS'].isna()]
    td_count = (yelt_raw['LFI_SSHS'] == 'TD').sum()
    print(f"  All landfalling events (incl TD): {len(all_landfalling):,}")
    print(f"    TD (tropical depressions): {td_count:,} (excluded from resampling)")
    print(f"  Non-landfalling (bypassing) events: {len(all_non_landfalling):,}")
    
    # Clean YELT before resampling
    print("\nStep 2b: Cleaning YELT for resampling...")
    yelt, rows_removed = clean_yelt_for_resampling(yelt_raw)
    print(f"  Removed {rows_removed:,} zero-loss bypassing events")
    print(f"  Cleaned YELT has {len(yelt):,} events")
    
    # Count by category (TS-Cat5 only)
    landfalling = yelt[yelt['LFI_SSHS'].notna()]
    print(f"  Landfalling events (TS-Cat5): {len(landfalling):,}")
    
    # Show counts per category
    base_rates = calculate_yelt_landfall_rates(yelt, args.n_years)
    print("\n  Original landfall rates per category (TS-Cat5):")
    for cat in SSHS_CATEGORIES:
        count = (landfalling['LFI_SSHS'] == cat).sum()
        rate = base_rates[cat]
        print(f"    {cat}: {count:,} events ({rate:.4f} per year)")
    
    # Load change rates (and base/target rates for plotting)
    print(f"\nStep 3: Loading change rates from {args.change_rates}...")
    change_rates = load_change_rates(args.change_rates)
    
    # Also load base and target rates from the CSV for the change rates plot
    cr_csv_path = Path(args.change_rates)
    cr_lines = cr_csv_path.read_text().splitlines()
    cr_header_idx = next(i for i, line in enumerate(cr_lines) if not line.startswith('#'))
    cr_df = pd.read_csv(cr_csv_path, skiprows=cr_header_idx)
    cr_base_rates = dict(zip(cr_df['sshs_category'], cr_df['base_rate'])) if 'base_rate' in cr_df.columns else None
    cr_target_rates = dict(zip(cr_df['sshs_category'], cr_df['target_rate'])) if 'target_rate' in cr_df.columns else None
    
    print("  Change rates:")
    for cat in SSHS_CATEGORIES:
        rate = change_rates.get(cat, 1.0)
        if np.isinf(rate):
            print(f"    {cat}: inf (new category)")
        else:
            change_pct = (rate - 1) * 100
            direction = "increase" if rate > 1 else "decrease" if rate < 1 else "no change"
            print(f"    {cat}: {rate:.4f} ({change_pct:+.1f}% {direction})")
    
    # Parse labels from change rates filename
    cr_path = Path(args.change_rates)
    cr_name = cr_path.stem
    # Try to extract base and target labels from filename
    # e.g. "change_rates_base_historical_target_GWL2.00_pi_only_NorthAtlantic"
    if 'base_' in cr_name and '_target_' in cr_name:
        parts = cr_name.split('_target_')
        base_label = parts[0].replace('change_rates_base_', '')
        # Target label is everything after _target_ up to the region suffix
        # Remove known region suffixes from the end
        target_part = parts[1]
        for region_name in ['_NorthAtlantic', '_CONUS']:
            if target_part.endswith(region_name):
                target_part = target_part[:-len(region_name)]
                break
        target_label = target_part
    else:
        base_label = 'baseline'
        target_label = 'adjusted'
    
    # Perform resampling
    print(f"\nStep 4: Resampling YELT ({args.resampling_type} method)...")
    
    if args.resampling_type == 'deterministic':
        n_iterations = 1
    else:
        n_iterations = args.resampling_iterations
    
    print(f"  Iterations: {n_iterations}")
    
    adjusted_yelts = []
    for i in range(n_iterations):
        if args.verbose and n_iterations > 1 and (i + 1) % 10 == 0:
            print(f"    Processing iteration {i + 1}/{n_iterations}...")
        
        # Use different seed for each iteration
        iter_seed = args.seed + i if args.seed is not None else None
        
        adjusted_yelt = resample_yelt(
            yelt, change_rates, 
            method=args.resampling_type,
            n_years=args.n_years,
            seed=iter_seed
        )
        adjusted_yelts.append(adjusted_yelt)
    
    # Report statistics
    print("\n  Resampling complete.")
    if n_iterations == 1:
        adj = adjusted_yelts[0]
        adj_landfalling = adj[adj['LFI_SSHS'].notna()]
        print(f"  Original events: {len(yelt):,}")
        print(f"  Adjusted events: {len(adj):,}")
        print(f"  Original landfalling: {len(landfalling):,}")
        print(f"  Adjusted landfalling: {len(adj_landfalling):,}")
        
        print("\n  Adjusted landfall rates per category:")
        adj_rates = calculate_yelt_landfall_rates(adj, args.n_years)
        for cat in SSHS_CATEGORIES:
            adj_count = (adj_landfalling['LFI_SSHS'] == cat).sum()
            adj_rate = adj_rates[cat]
            orig_count = (landfalling['LFI_SSHS'] == cat).sum()
            ratio = adj_count / orig_count if orig_count > 0 else np.nan
            print(f"    {cat}: {adj_count:,} events ({adj_rate:.4f} per year, ratio: {ratio:.3f})")
    else:
        event_counts = [len(adj) for adj in adjusted_yelts]
        lf_counts = [len(adj[adj['LFI_SSHS'].notna()]) for adj in adjusted_yelts]
        print(f"  Original events: {len(yelt):,}")
        print(f"  Adjusted events: {np.mean(event_counts):,.0f} ± {np.std(event_counts):,.0f}")
        print(f"  Original landfalling: {len(landfalling):,}")
        print(f"  Adjusted landfalling: {np.mean(lf_counts):,.0f} ± {np.std(lf_counts):,.0f}")
    
    # Compare loss metrics
    print("\nStep 5: Calculating loss metrics...")
    return_periods = [2, 5, 10, 20, 50, 100, 200, 500]
    comparison = compare_yelts(yelt, adjusted_yelts, args.n_years, return_periods)
    
    orig_aal = comparison['original']['aal']
    print(f"\n  Original AAL: ${orig_aal:,.0f}")
    
    if n_iterations == 1:
        adj_aal = comparison['adjusted'][0]['aal']
        aal_ratio = comparison['aal_ratios'][0]
        print(f"  Adjusted AAL: ${adj_aal:,.0f} (ratio: {aal_ratio:.3f})")
        
        print("\n  Loss ratios at return periods:")
        for rp in return_periods:
            ratio = comparison['rp_ratios'][rp][0]
            orig_loss = comparison['original']['rp_losses'][rp]
            adj_loss = comparison['adjusted'][0]['rp_losses'][rp]
            print(f"    {rp:>3}yr: {ratio:.3f} (${orig_loss:,.0f} → ${adj_loss:,.0f})")
    else:
        adj_aals = [m['aal'] for m in comparison['adjusted']]
        aal_ratios = comparison['aal_ratios']
        print(f"  Adjusted AAL: ${np.mean(adj_aals):,.0f} ± ${np.std(adj_aals):,.0f}")
        print(f"  AAL ratio: {np.mean(aal_ratios):.3f} ± {np.std(aal_ratios):.3f}")
        
        print("\n  Loss ratios at return periods (mean ± std):")
        for rp in return_periods:
            ratios = comparison['rp_ratios'][rp]
            print(f"    {rp:>3}yr: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
    
    # Save results
    print("\nStep 6: Saving results...")
    config.ensure_resampled_dir()
    
    # Generate output filename
    if args.output:
        output_base = Path(args.output).stem
        output_dir = Path(args.output).parent
    elif args.output_dir:
        output_base = f"yelt_adjusted_{base_label}_to_{target_label}_{args.resampling_type}"
        output_dir = Path(args.output_dir)
    else:
        output_base = f"yelt_adjusted_{base_label}_to_{target_label}_{args.resampling_type}"
        output_dir = config.RESAMPLED_DIR
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save adjusted YELTs
    if n_iterations == 1:
        output_path = output_dir / f"{output_base}.csv"
        adjusted_yelts[0].to_csv(output_path, index=False)
        print(f"  Saved adjusted YELT to: {output_path}")
    else:
        for i, adj in enumerate(adjusted_yelts):
            output_path = output_dir / f"{output_base}_iter{i+1:03d}.csv"
            adj.to_csv(output_path, index=False)
        print(f"  Saved {n_iterations} adjusted YELTs to: {output_dir}")
        print(f"    Pattern: {output_base}_iter*.csv")
    
    # Generate plots
    if not args.no_plot:
        print("\nStep 7: Generating comparison plots...")
        
        # OEP and loss ratio plot
        plot_path = output_dir / f"{output_base}_comparison.png"
        plot_yelt_comparison(
            yelt, adjusted_yelts, comparison,
            method=args.resampling_type,
            base_label=base_label,
            target_label=target_label,
            output_path=plot_path
        )
        
        # Category event counts plot
        counts_path = output_dir / f"{output_base}_category_counts.png"
        plot_category_event_counts(
            yelt, adjusted_yelts,
            method=args.resampling_type,
            base_label=base_label,
            target_label=target_label,
            output_path=counts_path
        )
        
        # Change rates comparison plot
        if cr_base_rates is not None and cr_target_rates is not None:
            # Extract region from change rates filename
            cr_region = 'NorthAtlantic'  # default
            for rname in ['NorthAtlantic', 'CONUS']:
                if rname in cr_csv_path.stem:
                    cr_region = rname
                    break
            
            rates_plot_path = output_dir / f"{output_base}_change_rates.png"
            
            if args.resampling_type == 'poisson' and n_iterations > 1:
                # Compute realized change rates per iteration
                realized_change_rates_list = []
                for adj in adjusted_yelts:
                    adj_rates = calculate_yelt_landfall_rates(adj, args.n_years)
                    realized = {}
                    for cat in SSHS_CATEGORIES:
                        orig_rate = base_rates.get(cat, 0)
                        adj_rate = adj_rates.get(cat, 0)
                        realized[cat] = adj_rate / orig_rate if orig_rate > 0 else np.nan
                    realized_change_rates_list.append(realized)
                
                plot_rate_comparison_poisson(
                    cr_base_rates, cr_target_rates, change_rates,
                    realized_change_rates_list,
                    base_label=base_label,
                    target_label=target_label,
                    region=cr_region,
                    output_path=rates_plot_path
                )
            else:
                # Deterministic / single iteration: bar plot
                plot_rate_comparison(
                    cr_base_rates, cr_target_rates, change_rates,
                    base_label=base_label,
                    target_label=target_label,
                    region=cr_region,
                    output_path=rates_plot_path
                )
        else:
            print("  Skipping change rates plot (base/target rates not available in CSV)")
    
    print("\n" + "=" * 60)
    print("YELT resampling complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
