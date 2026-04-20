"""
Run detrending for SCILS TC Model.

This script performs Theil-Sen detrending on PI and CGI maps using GWL as the predictor.
SST detrending is optional and no longer used for TC classification (use --skip-sst to skip).

Usage:
    python run_detrending.py --target-gwl 1.41
    python run_detrending.py --target-year 2050
    python run_detrending.py --target-gwl 2.0 --plot

Options:
    --start-year         Start year for analysis (default: from config)
    --end-year           End year for analysis (default: from config)
    --target-year        Target year for detrending (converted to GWL via lookup)
    --target-gwl         Target GWL (°C) for direct GWL-based detrending (default: 1.41)
    --plot               Generate comparison plots
    --force-recalculate  Force recalculation of regression coefficients
    --skip-sst           Skip SST detrending (recommended)
    --skip-pi            Skip PI detrending
    --skip-cgi           Skip CGI detrending
"""

import argparse

import xarray as xr

import config
from scils_tc.detrending import (
    detrend_cgi,
    detrend_pi,
    detrend_sst,
)
from scils_tc.detrending.gwl_lookup import (
    format_target_label,
    get_filename_suffix,
    get_gwl_regression,
    load_gwl_data,
    resolve_target,
)
from scils_tc.detrending.plotting import (
    plot_detrending_comparison,
    plot_diagnostic_mean_trend,
    plot_gwl_regression,
)


def main():
    """Parse CLI arguments and run PI/CGI/SST detrending for one target state."""
    parser = argparse.ArgumentParser(
        description='Run detrending for SCILS TC Model'
    )
    parser.add_argument('--start-year', type=int, default=config.START_YEAR,
                        help=f'Start year (default: {config.START_YEAR})')
    parser.add_argument('--end-year', type=int, default=config.END_YEAR,
                        help=f'End year (default: {config.END_YEAR})')
    parser.add_argument('--target-year', type=float, default=None,
                        help='Target year for detrending (converted to GWL via lookup)')
    parser.add_argument('--target-gwl', type=float, default=None,
                        help=f'Target GWL (°C) for detrending (default: {config.DEFAULT_TARGET_GWL})')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')
    parser.add_argument('--force-recalculate', action='store_true',
                        help='Force recalculation of regression coefficients')
    parser.add_argument('--skip-sst', action='store_true',
                        help='Skip SST detrending')
    parser.add_argument('--skip-pi', action='store_true',
                        help='Skip PI detrending')
    parser.add_argument('--skip-cgi', action='store_true',
                        help='Skip CGI detrending')
    parser.add_argument('--skip-diagnostics', action='store_true',
                        help='Skip diagnostic mean/trend plots (faster)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SCILS TC Model - Detrending (GWL-based)")
    print("=" * 60)
    print(f"Year range: {args.start_year}-{args.end_year}")
    
    # Ensure detrended directory exists
    config.ensure_detrended_dir()
    
    # Resolve target (GWL or year)
    effective_gwl, effective_year, specified_by_year = resolve_target(
        target_year=args.target_year,
        target_gwl=args.target_gwl,
        default_gwl=config.DEFAULT_TARGET_GWL
    )
    
    target_label = format_target_label(effective_gwl, effective_year, specified_by_year)
    print(f"Detrending target: {target_label}")
    
    # Load GWL data for plotting
    print("\n" + "=" * 60)
    print("Step 1: Load GWL Data")
    print("=" * 60)
    
    gwl_df = load_gwl_data()
    gwl_regression = get_gwl_regression()
    
    print(f"Loaded GWL data: {len(gwl_df)} years ({gwl_df['Year'].min()}-{gwl_df['Year'].max()})")
    print(f"GWL regression (Year->GWL): slope={gwl_regression['slope']:.5f}°C/yr, "
          f"intercept={gwl_regression['intercept']:.3f}°C")
    print(f"GWL range: {gwl_df['GWL'].min():.2f}°C to {gwl_df['GWL'].max():.2f}°C")
    
    # Plot GWL regression if target specified by year
    if specified_by_year or args.plot:
        filename_suffix = get_filename_suffix(effective_gwl, effective_year, specified_by_year)
        plot_filename = f'GWL_regression_{filename_suffix}.png'
        gwl_plot_path = config.get_detrended_path(plot_filename)

        plot_gwl_regression(
            gwl_df, gwl_regression,
            gwl_df_fit=gwl_df,  # Use all data for fit
            target_year=effective_year,
            target_gwl=effective_gwl,
            save_path=gwl_plot_path
        )
    
    # Store original and detrended data for comparison plots
    results = {}
    regressions = {}
    
    # Detrend SST (optional)
    if not args.skip_sst:
        print("\n" + "=" * 60)
        print("Step 2: Detrend SST")
        print("=" * 60)
        
        detrended_sst, sst_regression, eff_gwl, eff_year, by_year = detrend_sst(
            start_year=args.start_year,
            end_year=args.end_year,
            target_year=args.target_year,
            target_gwl=args.target_gwl,
            force_recalculate=args.force_recalculate,
        )
        
        # Load original for comparison
        ds = xr.open_dataset(config.ERA5_SST_MSLP_FILE)
        original_sst = ds['sst'].sel(
            valid_time=ds['valid_time'].dt.month.isin(config.SEASON_MONTHS)
        )
        original_sst = original_sst.sel(
            valid_time=(original_sst['valid_time'].dt.year >= args.start_year) & 
                       (original_sst['valid_time'].dt.year <= args.end_year)
        ).rename({'valid_time': 'time'})
        
        results['original_sst'] = original_sst
        results['detrended_sst'] = detrended_sst
        regressions['sst'] = sst_regression
        ds.close()
    
    # Detrend PI
    if not args.skip_pi:
        print("\n" + "=" * 60)
        print("Step 3: Detrend PI")
        print("=" * 60)
        
        detrended_pi, pi_regression, eff_gwl, eff_year, by_year = detrend_pi(
            start_year=args.start_year,
            end_year=args.end_year,
            target_year=args.target_year,
            target_gwl=args.target_gwl,
            force_recalculate=args.force_recalculate,
        )
        
        # Load original for comparison
        ds = xr.open_dataset(config.get_output_path(config.PI_FILE))
        results['original_pi'] = ds['vmax']
        results['detrended_pi'] = detrended_pi
        regressions['pi'] = pi_regression
        ds.close()
    
    # Detrend CGI
    if not args.skip_cgi:
        print("\n" + "=" * 60)
        print("Step 4: Detrend CGI")
        print("=" * 60)
        
        detrended_cgi, cgi_regression, eff_gwl, eff_year, by_year = detrend_cgi(
            start_year=args.start_year,
            end_year=args.end_year,
            target_year=args.target_year,
            target_gwl=args.target_gwl,
            force_recalculate=args.force_recalculate,
        )
        
        # Load original for comparison
        ds = xr.open_dataset(config.get_output_path(config.CGI_MAP_FILE))
        results['original_cgi'] = ds['cgi']
        results['detrended_cgi'] = detrended_cgi
        regressions['cgi'] = cgi_regression
        ds.close()
    
    # Generate comparison plots
    if args.plot:
        print("\n" + "=" * 60)
        print("Step 5: Generate Comparison Plots")
        print("=" * 60)
        
        filename_suffix = get_filename_suffix(effective_gwl, effective_year, specified_by_year)
        plot_filename = f'detrending_comparison_{filename_suffix}.png'
        plot_path = config.get_detrended_path(plot_filename)
        
        plot_detrending_comparison(
            original_sst=results.get('original_sst'),
            detrended_sst=results.get('detrended_sst'),
            original_pi=results.get('original_pi'),
            detrended_pi=results.get('detrended_pi'),
            original_cgi=results.get('original_cgi'),
            detrended_cgi=results.get('detrended_cgi'),
            target_year=effective_year,
            target_gwl=effective_gwl,
            save_path=plot_path,
        )
        
        # Generate diagnostic plots (mean and trend maps)
        if not args.skip_diagnostics:
            print("\n" + "=" * 60)
            print("Step 6: Generate Diagnostic Plots")
            print("=" * 60)
            
            # SST diagnostic
            if 'original_sst' in results and 'sst' in regressions:
                sst_diag_path = config.get_detrended_diagnostics_path('SST_mean_trend.png')
                plot_diagnostic_mean_trend(
                    data=results['original_sst'],
                    regression=regressions['sst'],
                    variable_name='SST',
                    units='°C',
                    time_dim='time',
                    save_path=sst_diag_path,
                )
            
            # PI diagnostic
            if 'original_pi' in results and 'pi' in regressions:
                pi_diag_path = config.get_detrended_diagnostics_path('PI_mean_trend.png')
                plot_diagnostic_mean_trend(
                    data=results['original_pi'],
                    regression=regressions['pi'],
                    variable_name='PI',
                    units='m/s',
                    time_dim='time',
                    save_path=pi_diag_path,
                )
            
            # CGI diagnostic
            if 'original_cgi' in results and 'cgi' in regressions:
                cgi_diag_path = config.get_detrended_diagnostics_path('CGI_mean_trend.png')
                plot_diagnostic_mean_trend(
                    data=results['original_cgi'],
                    regression=regressions['cgi'],
                    variable_name='CGI',
                    units='',
                    time_dim='time',
                    save_path=cgi_diag_path,
                )
    
    print("\n" + "=" * 60)
    print("Detrending Complete!")
    print("=" * 60)
    print(f"Output directory: {config.DETRENDED_DIR}")
    
    # List output files
    if config.DETRENDED_DIR.exists():
        files = list(config.DETRENDED_DIR.glob('*'))
        diag_files = list(config.DETRENDED_DIAGNOSTICS_DIR.glob('*')) if config.DETRENDED_DIAGNOSTICS_DIR.exists() else []
        print(f"Generated {len([f for f in files if f.is_file()])} files:")
        for f in sorted(files):
            if f.is_file():
                print(f"  - {f.name}")
        if diag_files:
            print(f"Generated {len(diag_files)} diagnostic files:")
            for f in sorted(diag_files):
                print(f"  - diagnostics/{f.name}")


if __name__ == '__main__':
    main()
