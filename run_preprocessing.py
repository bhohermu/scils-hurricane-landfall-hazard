"""
Main preprocessing script for SCILS TC Model.

This script orchestrates all preprocessing steps:
1. ERA5 processing (SST, wind shear, PI, CGI)
2. ENSO classification
3. IBTrACS properties extraction
4. IBTrACS statistics (KDE, LMI/PI, LFI/LMI)

Usage:
    python run_preprocessing.py [--start-year YEAR] [--end-year YEAR]
"""

import argparse

import pandas as pd

import config
from scils_tc.preprocessing.enso_classification import process_enso
from scils_tc.preprocessing.era5_processing import (
    calculate_aso_mdr_sst,
    calculate_cgi,
    calculate_potential_intensity,
    calculate_wind_shear,
)
from scils_tc.preprocessing.gwl_calculation import process_gwl
from scils_tc.preprocessing.ibtracs_properties import process_ibtracs_properties
from scils_tc.preprocessing.ibtracs_statistics import (
    calculate_lfi_lmi_distributions,
    calculate_lmi_kde,
    calculate_lmi_pi_ratio,
)
from scils_tc.preprocessing.plotting import create_all_plots
from scils_tc.utils.regions import load_jewson_regions, plot_jewson_regions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run SCILS TC Model preprocessing.'
    )
    parser.add_argument(
        '--start-year', type=int, default=config.START_YEAR,
        help=f'First year to process (default: {config.START_YEAR})'
    )
    parser.add_argument(
        '--end-year', type=int, default=config.END_YEAR,
        help=f'Last year to process (default: {config.END_YEAR})'
    )
    parser.add_argument(
        '--mdr-lat-min', type=float, default=config.MDR_LAT_MIN,
        help=f'MDR minimum latitude (default: {config.MDR_LAT_MIN})'
    )
    parser.add_argument(
        '--mdr-lat-max', type=float, default=config.MDR_LAT_MAX,
        help=f'MDR maximum latitude (default: {config.MDR_LAT_MAX})'
    )
    parser.add_argument(
        '--mdr-lon-min', type=float, default=config.MDR_LON_MIN,
        help=f'MDR minimum longitude (default: {config.MDR_LON_MIN})'
    )
    parser.add_argument(
        '--mdr-lon-max', type=float, default=config.MDR_LON_MAX,
        help=f'MDR maximum longitude (default: {config.MDR_LON_MAX})'
    )
    parser.add_argument(
        '--skip-era5', action='store_true',
        help='Skip ERA5 processing (use if already completed)'
    )
    parser.add_argument(
        '--skip-ibtracs', action='store_true',
        help='Skip IBTrACS processing (use if already completed)'
    )
    parser.add_argument(
        '--plot-regions', action='store_true',
        help='Plot Jewson region definitions'
    )
    parser.add_argument(
        '--force-recalculate', action='store_true',
        help='Force recalculation even if output files exist'
    )
    parser.add_argument(
        '--skip-gwl', action='store_true',
        help='Skip GWL calculation'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate visualization plots after preprocessing'
    )
    parser.add_argument(
        '--plot-only', action='store_true',
        help='Only generate plots (skip all preprocessing)'
    )
    parser.add_argument(
        '--region', type=str, default=config.USE_REGION, choices=['CONUS', 'NorthAtlantic'],
        help=f'Region for landfall analysis (default: {config.USE_REGION})'
    )
    
    return parser.parse_args()


def main():
    """Main preprocessing pipeline."""
    args = parse_args()
    
    print("=" * 60)
    print("SCILS TC Model - Preprocessing Pipeline")
    print("=" * 60)
    print(f"Processing years: {args.start_year} - {args.end_year}")
    print(f"MDR bounds: {args.mdr_lat_min}°N-{args.mdr_lat_max}°N, "
          f"{args.mdr_lon_min}°E-{args.mdr_lon_max}°E")
    print("Minimum storm strength: TS+ (tropical storm or above)")
    print("=" * 60)
    
    # Ensure output directory exists
    config.ensure_preprocessed_dir()
    
    # Plot-only mode
    if args.plot_only:
        print("\n[PLOT-ONLY MODE] Generating plots...")
        create_all_plots(
            start_year=args.start_year,
            end_year=args.end_year,
            mdr_lat_min=args.mdr_lat_min,
            mdr_lat_max=args.mdr_lat_max,
            mdr_lon_min=args.mdr_lon_min,
            mdr_lon_max=args.mdr_lon_max,
            region=args.region
        )
        print("\nPlot generation complete!")
        return
    
    # Plot regions if requested
    if args.plot_regions:
        print("\n[1/7] Plotting Jewson regions...")
        regions = load_jewson_regions(config.REGION_DEFINITIONS_FILE)
        plot_jewson_regions(
            regions, 
            output_path=config.get_output_path("jewson_regions.png")
        )
        print("  Jewson regions plot saved.")
    
    # Step 1: ERA5 Processing (must come first for PI)
    if not args.skip_era5:
        print("\n[2/7] Processing ERA5 data...")
        
        print("\n  [2a] Calculating ASO MDR SST...")
        calculate_aso_mdr_sst(
            start_year=args.start_year,
            end_year=args.end_year,
            mdr_lat_min=args.mdr_lat_min,
            mdr_lat_max=args.mdr_lat_max,
            mdr_lon_min=args.mdr_lon_min,
            mdr_lon_max=args.mdr_lon_max,
            force_recalculate=args.force_recalculate
        )
        
        print("\n  [2b] Calculating wind shear...")
        calculate_wind_shear(
            start_year=args.start_year,
            end_year=args.end_year,
            force_recalculate=args.force_recalculate
        )
        
        print("\n  [2c] Calculating Potential Intensity...")
        calculate_potential_intensity(
            start_year=args.start_year,
            end_year=args.end_year,
            force_recalculate=args.force_recalculate
        )
    else:
        print("\n[2/7] Skipping ERA5 processing...")
    
    # Step 2: ENSO Classification
    print("\n[3/7] Processing ENSO classification...")
    enso_df = process_enso(
        start_year=args.start_year,
        end_year=args.end_year
    )
    
    # Step 3: IBTrACS Properties
    if not args.skip_ibtracs:
        print("\n[4/7] Processing IBTrACS properties...")
        ibtracs_df = process_ibtracs_properties(
            start_year=args.start_year,
            end_year=args.end_year
        )
    else:
        print("\n[4/7] Skipping IBTrACS processing...")
        ibtracs_df = pd.read_csv(config.get_output_path(config.IBTRACS_PROPERTIES_FILE))
    
    # Step 4: CGI Calculation (needs IBTrACS for scaling)
    # Note: CGI table can be regenerated even with --skip-era5 if the map exists
    if not args.skip_era5:
        print("\n[5/7] Calculating CGI...")
        calculate_cgi(
            start_year=args.start_year,
            end_year=args.end_year,
            mdr_lat_min=args.mdr_lat_min,
            mdr_lat_max=args.mdr_lat_max,
            mdr_lon_min=args.mdr_lon_min,
            mdr_lon_max=args.mdr_lon_max,
            force_recalculate=args.force_recalculate
        )
    else:
        # Even with skip-era5, regenerate CGI table if map exists but table doesn't
        cgi_map_path = config.get_output_path(config.CGI_MAP_FILE)
        cgi_table_path = config.get_output_path(config.CGI_MDR_FILE)
        if cgi_map_path.exists() and (not cgi_table_path.exists() or args.force_recalculate):
            print("\n[5/7] Regenerating CGI MDR table (map exists)...")
            calculate_cgi(
                start_year=args.start_year,
                end_year=args.end_year,
                mdr_lat_min=args.mdr_lat_min,
                mdr_lat_max=args.mdr_lat_max,
                mdr_lon_min=args.mdr_lon_min,
                mdr_lon_max=args.mdr_lon_max,
                force_recalculate=False  # Don't force map recalc
            )
        else:
            print("\n[5/7] Skipping CGI calculation...")
    
    # Step 5: IBTrACS Statistics
    print("\n[6/8] Calculating IBTrACS statistics...")
    
    print("\n  [6a] LMI location KDEs...")
    calculate_lmi_kde(ibtracs_df, enso_df, force_recalculate=args.force_recalculate)
    
    print("\n  [6b] LMI/PI ratio distributions...")
    calculate_lmi_pi_ratio(ibtracs_df, force_recalculate=args.force_recalculate)
    
    print("\n  [6c] LFI/LMI distributions...")
    # Calculate for both CONUS and NorthAtlantic regions (all methods)
    calculate_lfi_lmi_distributions(ibtracs_df, region='CONUS', force_recalculate=args.force_recalculate)
    calculate_lfi_lmi_distributions(ibtracs_df, region='NorthAtlantic', force_recalculate=args.force_recalculate)
    
    # Step 6: GWL Calculation
    if not args.skip_gwl:
        print("\n[7/8] Calculating Global Warming Level (GWL)...")
        process_gwl(
            start_year=args.start_year,
            end_year=args.end_year,
            force_recalculate=args.force_recalculate
        )
    else:
        print("\n[7/8] Skipping GWL calculation...")
    
    print("\n[8/8] Preprocessing complete!")
    print("=" * 60)
    print("Output files saved to:", config.PREPROCESSED_DIR)
    print("=" * 60)
    
    # Generate plots if requested
    if args.plot:
        print("\n" + "=" * 60)
        print("Generating visualization plots...")
        print("=" * 60)
        create_all_plots(
            start_year=args.start_year,
            end_year=args.end_year,
            mdr_lat_min=args.mdr_lat_min,
            mdr_lat_max=args.mdr_lat_max,
            mdr_lon_min=args.mdr_lon_min,
            mdr_lon_max=args.mdr_lon_max,
            region=args.region
        )


if __name__ == "__main__":
    main()
