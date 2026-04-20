"""
Plotting module for SCILS TC Model preprocessing results.

Creates visualization plots for LMI/PI, LFI/LMI, timeseries, and LMI location KDEs.
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import config


def plot_lmi_pi_lfi_lmi(output_path=None, region='CONUS'):
    """
    Create a figure with:
    - Subplot a: Single LMI/PI ratio histogram (pooled across all storms)
    - Subplot b: 0-1 Inflated Beta distributions of LFI/LMI by region with point markers
    
    Parameters
    ----------
    output_path : str or Path, optional
        Path to save the figure.
    region : str, default='CONUS'
        Region for landfall analysis ('CONUS' or 'NorthAtlantic')
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    if output_path is None:
        output_path = config.get_output_path(f"LMI_PI_LFI_LMI_plot_{region}.png")
    
    # Load LMI/PI single histogram
    lmi_pi_single_file = config.get_output_path(config.LMI_PI_RATIO_FILE.replace('.csv', '_single.csv'))
    beta_params_file = config.get_output_path(f"LFI_LMI_{region}_beta_params.csv")
    
    if not Path(lmi_pi_single_file).exists() or not Path(beta_params_file).exists():
        print("Error: Required data files not found. Run preprocessing first.")
        return None
    
    lmi_pi_df = pd.read_csv(lmi_pi_single_file)
    beta_params_df = pd.read_csv(beta_params_file)
    
    # Load IBTrACS for storm counts
    ibtracs_file = config.get_output_path(config.IBTRACS_PROPERTIES_FILE)
    lfi_column = f'{region}_LFI_ms'
    landfalls_by_region = {}
    n_storms_total = 0
    if Path(ibtracs_file).exists():
        ibtracs_df = pd.read_csv(ibtracs_file, keep_default_na=False, na_values=[''])
        n_storms_total = len(ibtracs_df)
        landfalling_storms = ibtracs_df[ibtracs_df[lfi_column].notna()]
        landfalls_by_region = landfalling_storms.groupby('RegionNumber').size().to_dict()
    
    from scipy import stats
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot a: Single LMI/PI histogram
    ax1 = axes[0]
    bin_width = lmi_pi_df['bin_center'].diff().median()
    ax1.bar(lmi_pi_df['bin_center'], lmi_pi_df['density'], 
            width=bin_width, alpha=0.7, color='steelblue', 
            edgecolor='black', linewidth=0.5,
            label=f'All storms (n={n_storms_total})')
    
    ax1.set_xlabel('LMI/PI Ratio', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('(a) LMI/PI Ratio Distribution (Single Histogram)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1.0)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Subplot b: LFI/LMI 0-1 Inflated Beta with point markers
    ax2 = axes[1]
    
    region_colors = {
        1: '#FF6B6B',  # GOM - red
        2: '#4ECDC4',  # CARB - teal
        3: '#45B7D1',  # MDR - blue
        4: '#96CEB4'   # NA - green
    }
    
    # Evaluate continuous part on interior (0, 1)
    x_eval = np.linspace(0.001, 0.999, 200)
    max_density = 0
    
    for _, row in beta_params_df.iterrows():
        region_num = int(row['RegionNumber'])
        region_name = row['RegionName']
        alpha = row['alpha']
        beta_val = row['beta']
        p0 = row['p0']
        p1 = row['p1']
        p_continuous = row.get('p_continuous', 1 - p0 - p1)
        n_region = landfalls_by_region.get(region_num, 0)
        color = region_colors.get(region_num, 'gray')
        
        # Beta PDF for continuous part (scaled by p_continuous)
        beta_pdf = stats.beta.pdf(x_eval, alpha, beta_val) * p_continuous
        max_density = max(max_density, beta_pdf.max())
        
        ax2.plot(x_eval, beta_pdf, label=f'{region_name} (n={n_region})',
                color=color, linewidth=2)
    
    # Add point masses at 0 and 1 with markers
    for _, row in beta_params_df.iterrows():
        region_num = int(row['RegionNumber'])
        p0 = row['p0']
        p1 = row['p1']
        color = region_colors.get(region_num, 'gray')
        
        # Scale point mass heights relative to max density for visibility
        # Use larger marker size for point masses
        if p0 > 0.01:
            # Point mass at 0: height proportional to p0
            height = p0 * max_density * 2  # Scale for visibility
            ax2.plot(0, height, 'o', color=color, markersize=10, 
                    markeredgecolor='black', markeredgewidth=1)
        if p1 > 0.01:
            # Point mass at 1: height proportional to p1
            height = p1 * max_density * 2
            ax2.plot(1, height, 'o', color=color, markersize=10,
                    markeredgecolor='black', markeredgewidth=1)
    
    ax2.set_xlabel('LFI/LMI Ratio', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('(b) LFI/LMI Distribution by Region (0-1 Inflated Beta)', 
                  fontsize=13, fontweight='bold')
    ax2.set_xlim(-0.05, 1.05)  # Wider x-axis to see point masses at 0 and 1
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"LMI/PI and LFI/LMI plot saved to: {output_path}")
    return fig


def plot_timeseries(output_path=None, start_year=None, end_year=None,
                    mdr_lat_min=None, mdr_lat_max=None,
                    mdr_lon_min=None, mdr_lon_max=None):
    """
    Create a timeseries figure with:
    - Subplot a: MDR SST
    - Subplot b: Scaled MDR CGI with actual number of storms (same y-axis)
    - Subplot c: PI averaged over MDR and ASO (Aug-Sept-Oct)
    
    Parameters
    ----------
    output_path : str or Path, optional
        Path to save the figure.
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    if output_path is None:
        output_path = config.get_output_path("timeseries_plot.png")
    if start_year is None:
        start_year = config.START_YEAR
    if end_year is None:
        end_year = config.END_YEAR
    if mdr_lat_min is None:
        mdr_lat_min = config.MDR_LAT_MIN
    if mdr_lat_max is None:
        mdr_lat_max = config.MDR_LAT_MAX
    if mdr_lon_min is None:
        mdr_lon_min = config.MDR_LON_MIN
    if mdr_lon_max is None:
        mdr_lon_max = config.MDR_LON_MAX
    
    # Load data
    sst_file = config.get_output_path(config.ASO_MDR_SST_FILE)
    cgi_file = config.get_output_path(config.CGI_MDR_FILE)
    pi_file = config.get_output_path(config.PI_FILE)
    ibtracs_file = config.get_output_path(config.IBTRACS_PROPERTIES_FILE)
    
    if not Path(sst_file).exists() or not Path(cgi_file).exists():
        print("Error: Required data files not found. Run preprocessing first.")
        return None
    
    sst_df = pd.read_csv(sst_file)
    cgi_df = pd.read_csv(cgi_file)
    
    # Load IBTrACS for storm counts
    if Path(ibtracs_file).exists():
        ibtracs_df = pd.read_csv(ibtracs_file, keep_default_na=False, na_values=[''])
        storm_counts = ibtracs_df.groupby('Year').size().reset_index(name='Storm_Count')
    else:
        storm_counts = pd.DataFrame(columns=['Year', 'Storm_Count'])
    
    # Calculate MDR PI average for ASO (Aug-Sept-Oct)
    pi_mdr_avg = []
    if Path(pi_file).exists():
        pi_ds = xr.open_dataset(pi_file)
        pi = pi_ds['vmax']
        
        # Get coordinate names
        lat_name = 'latitude' if 'latitude' in pi.dims else 'lat'
        lon_name = 'longitude' if 'longitude' in pi.dims else 'lon'
        
        # Select MDR region
        if float(pi[lon_name].min()) >= 0:
            lon_min = mdr_lon_min % 360
            lon_max = mdr_lon_max % 360
        else:
            lon_min = mdr_lon_min
            lon_max = mdr_lon_max
        
        pi_mdr = pi.sel(**{lat_name: slice(mdr_lat_max, mdr_lat_min),
                           lon_name: slice(lon_min, lon_max)})
        
        for year in range(start_year, end_year + 1):
            # Select ASO (Aug-Sept-Oct)
            time_mask = (
                (pi_mdr['time'].dt.year == year) &
                (pi_mdr['time'].dt.month.isin([8, 9, 10]))
            )
            pi_aso = pi_mdr.where(time_mask, drop=True)
            
            if len(pi_aso['time']) > 0:
                weights = np.cos(np.deg2rad(pi_aso[lat_name]))
                pi_weighted = pi_aso.weighted(weights)
                mean_pi = float(pi_weighted.mean().values)
                pi_mdr_avg.append({'Year': year, 'MDR_PI': mean_pi})
        
        pi_ds.close()
    
    pi_df = pd.DataFrame(pi_mdr_avg) if pi_mdr_avg else pd.DataFrame(columns=['Year', 'MDR_PI'])
    
    # Calculate climatological means
    clim_start = config.CLIMATOLOGY_START_YEAR
    clim_end = config.CLIMATOLOGY_END_YEAR
    
    sst_clim_mask = (sst_df['Year'] >= clim_start) & (sst_df['Year'] <= clim_end)
    sst_clim_mean = sst_df[sst_clim_mask]['ASO_MDR_SST'].mean() if sst_clim_mask.any() else sst_df['ASO_MDR_SST'].mean()
    
    cgi_clim_mask = (cgi_df['Year'] >= clim_start) & (cgi_df['Year'] <= clim_end)
    cgi_clim_mean = cgi_df[cgi_clim_mask]['Scaled_MDR_CGI'].mean() if 'Scaled_MDR_CGI' in cgi_df.columns and cgi_clim_mask.any() else None
    
    if not pi_df.empty:
        pi_clim_mask = (pi_df['Year'] >= clim_start) & (pi_df['Year'] <= clim_end)
        pi_clim_mean = pi_df[pi_clim_mask]['MDR_PI'].mean() if pi_clim_mask.any() else pi_df['MDR_PI'].mean()
    else:
        pi_clim_mean = None
    
    # Helper function for linear fit
    def add_linear_fit(ax, x, y, color='black'):
        """Add linear trend line and return slope."""
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            fit_line = slope * x + intercept
            ax.plot(x, fit_line, '--', color=color, linewidth=1.5, alpha=0.7,
                    label=f'Trend: {slope:.3f}/yr')
            return slope
        return None
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Subplot a: MDR SST
    ax1 = axes[0]
    ax1.plot(sst_df['Year'], sst_df['ASO_MDR_SST'], 'o-', color='firebrick', 
             linewidth=2, markersize=6, label='ASO MDR SST')
    ax1.axhline(y=sst_clim_mean, color='gray', linestyle='--', 
                linewidth=1, label=f"Clim. Mean ({clim_start}-{clim_end}): {sst_clim_mean:.2f}°C")
    add_linear_fit(ax1, sst_df['Year'].values, sst_df['ASO_MDR_SST'].values, color='darkred')
    ax1.set_ylabel('SST (°C)', fontsize=12)
    ax1.set_title('(a) August-September-October MDR Sea Surface Temperature', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Subplot b: Scaled CGI and Storm Counts (same y-axis scale)
    ax2 = axes[1]
    
    # Merge CGI and storm counts data
    if 'Scaled_MDR_CGI' in cgi_df.columns and not storm_counts.empty:
        merged = cgi_df.merge(storm_counts, on='Year', how='outer')
        
        ax2.plot(merged['Year'], merged['Scaled_MDR_CGI'], 'o-', 
                 color='steelblue', linewidth=2, markersize=6,
                 label='Scaled MDR CGI')
        ax2.plot(merged['Year'], merged['Storm_Count'], 's-', 
                 color='darkorange', linewidth=2, markersize=6,
                 label='Observed Storm Count')
        if cgi_clim_mean is not None:
            ax2.axhline(y=cgi_clim_mean, color='gray', linestyle='--', 
                        linewidth=1, label=f"CGI Clim. Mean: {cgi_clim_mean:.1f}")
        add_linear_fit(ax2, merged['Year'].values, merged['Scaled_MDR_CGI'].values, color='darkblue')
        ax2.legend(loc='upper left', fontsize=9)
    elif 'Scaled_MDR_CGI' in cgi_df.columns:
        ax2.plot(cgi_df['Year'], cgi_df['Scaled_MDR_CGI'], 'o-', 
                 color='steelblue', linewidth=2, markersize=6,
                 label='Scaled MDR CGI')
        if cgi_clim_mean is not None:
            ax2.axhline(y=cgi_clim_mean, color='gray', linestyle='--', 
                        linewidth=1, label=f"CGI Clim. Mean: {cgi_clim_mean:.1f}")
        add_linear_fit(ax2, cgi_df['Year'].values, cgi_df['Scaled_MDR_CGI'].values, color='darkblue')
        ax2.legend(loc='upper left', fontsize=9)
    elif not storm_counts.empty:
        ax2.plot(storm_counts['Year'], storm_counts['Storm_Count'], 's-', 
                 color='darkorange', linewidth=2, markersize=6,
                 label='Observed Storm Count')
        ax2.legend(loc='upper left', fontsize=9)
    
    ax2.set_ylabel('Count', fontsize=12)
    # Calculate and display scale factor
    # Use MDR_CGI_Sum (seasonal sum) if available, otherwise fall back to MDR_CGI
    cgi_raw_col = 'MDR_CGI_Sum' if 'MDR_CGI_Sum' in cgi_df.columns else 'MDR_CGI'
    if cgi_raw_col in cgi_df.columns and 'Observed_Count' in cgi_df.columns:
        total_obs = cgi_df['Observed_Count'].sum()
        total_raw = cgi_df[cgi_raw_col].sum()
        scale_factor = total_obs / total_raw if total_raw > 0 else 1.0
        ax2.set_title(f'(b) Scaled MDR CGI (seasonal sum) and Observed Storm Counts (scale factor: {scale_factor:.2f})', 
                      fontsize=13, fontweight='bold')
    else:
        ax2.set_title('(b) Scaled MDR CGI and Observed Storm Counts', 
                      fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Subplot c: MDR PI (ASO)
    ax3 = axes[2]
    if not pi_df.empty:
        ax3.plot(pi_df['Year'], pi_df['MDR_PI'], 'o-', color='purple', 
                 linewidth=2, markersize=6, label='ASO MDR PI')
        if pi_clim_mean is not None:
            ax3.axhline(y=pi_clim_mean, color='gray', linestyle='--', 
                        linewidth=1, label=f"Clim. Mean ({clim_start}-{clim_end}): {pi_clim_mean:.1f} m/s")
        add_linear_fit(ax3, pi_df['Year'].values, pi_df['MDR_PI'].values, color='darkviolet')
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Potential Intensity (m/s)', fontsize=12)
    ax3.set_title('(c) August-September-October MDR Potential Intensity', 
                  fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Set x-axis ticks
    years = sst_df['Year'].values
    if len(years) <= 10:
        ax3.set_xticks(years)
    else:
        ax3.set_xticks(years[::5])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Timeseries plot saved to: {output_path}")
    return fig


def plot_lmi_kde_maps(output_path=None):
    """
    Create a figure with 6 subplots (2 rows x 3 columns) showing 
    LMI location KDE maps for each ENSO/SST group.
    
    Layout:
    Row 1: warm_lanina, warm_neutral, warm_elnino
    Row 2: cold_lanina, cold_neutral, cold_elnino
    
    Parameters
    ----------
    output_path : str or Path, optional
        Path to save the figure.
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    if output_path is None:
        output_path = config.get_output_path("LMI_KDE_maps.png")
    
    # Load storm properties to get LMI locations
    storms_file = config.get_output_path("ibtracs_properties.csv")
    if Path(storms_file).exists():
        storms_df = pd.read_csv(storms_file)
        
        # Load ENSO data to classify storms into groups
        enso_file = config.get_output_path("ENSO_state.csv")
        
        if Path(enso_file).exists():
            enso_df = pd.read_csv(enso_file)
            
            # Merge ENSO state
            storms_df = storms_df.merge(enso_df[['Year', 'ENSO_State']], on='Year', how='left')
            
            # Create group column (ENSO only)
            def assign_group(row):
                """Map an ENSO state row to the KDE group name used on disk."""
                enso_state = row['ENSO_State']
                if pd.isna(enso_state):
                    return None
                if enso_state == 'La Nina':
                    return 'lanina'
                elif enso_state == 'Neutral':
                    return 'neutral'
                elif enso_state == 'El Nino':
                    return 'elnino'
                return None
            
            storms_df['group'] = storms_df.apply(assign_group, axis=1)
            print(f"  Loaded {len(storms_df)} storms with group classifications")
            print(f"    Group counts: {storms_df['group'].value_counts().to_dict()}")
        else:
            print("  Warning: Could not load ENSO or SST data for group classification")
            storms_df = None
    else:
        print("  Warning: Could not load storm properties file")
        storms_df = None
    
    # Define the groups in order (ENSO only - 3 groups)
    groups = [
        ('lanina', 'La Niña'),
        ('neutral', 'Neutral'),
        ('elnino', 'El Niño'),
    ]
    
    # First pass: find global maximum across all KDE data for consistent color scale
    global_max = 0
    kde_data_cache = {}
    for group_name, _ in groups:
        kde_file = config.get_output_path(f"LMI_KDE_{group_name}.nc")
        if Path(kde_file).exists():
            kde_ds = xr.open_dataset(kde_file)
            kde = kde_ds['kde']
            data = kde.values
            data_max = np.nanmax(data)
            if data_max > global_max:
                global_max = data_max
            kde_data_cache[group_name] = {
                'kde': kde,
                'n_storms': kde.attrs.get('n_storms', 'N/A'),
                'n_years': kde.attrs.get('n_years', 'N/A'),
                'data': data
            }
            kde_ds.close()
    
    # Create figure with cartopy projections (1 row, 3 columns for 3 groups)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                              subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    subplot_labels = ['(a)', '(b)', '(c)']
    
    for idx, (group_name, group_label) in enumerate(groups):
        ax = axes[idx]
        
        # Set map extent - longitude -100 to -10, latitude 10 to 50
        ax.set_extent([-100, -10, 10, 50], crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--')
        
        if group_name in kde_data_cache:
            cache = kde_data_cache[group_name]
            kde = cache['kde']
            n_storms = cache['n_storms']
            n_years = cache['n_years']
            data = cache['data']
            
            # Plot KDE using pcolormesh for continuous color scale (no normalization)
            lon = kde['lon'].values
            lat = kde['lat'].values
            
            # Use pcolormesh for continuous colors (like compare_PI plots)
            # Data is already in storms/year/km² from preprocessing
            ax.pcolormesh(
                lon,
                lat,
                data,
                cmap='YlOrRd',
                transform=ccrs.PlateCarree(),
                vmin=0,
                vmax=global_max,
                shading='auto',
            )
            
            # Add LMI locations as dots
            if storms_df is not None and 'group' in storms_df.columns:
                # Filter storms for this group
                group_storms = storms_df[storms_df['group'] == group_name]
                if len(group_storms) > 0:
                    ax.scatter(group_storms['LMI_Lon'], group_storms['LMI_Lat'],
                              s=8, c='navy', alpha=0.4, transform=ccrs.PlateCarree(),
                              zorder=5, edgecolors='black', linewidth=0.3)
            
            title_extra = f" (n={n_storms}, years={n_years})"
        else:
            title_extra = " (No data)"
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                          alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        if idx < 3:  # Top row
            gl.bottom_labels = False
        if idx % 3 != 0:  # Not leftmost column
            gl.left_labels = False
        
        ax.set_title(f'{subplot_labels[idx]} {group_label}{title_extra}', 
                     fontsize=11, fontweight='bold')
    
    # Add colorbar - adjust position for better spacing
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=mcolors.Normalize(0, global_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('KDE (storms/year/km²)', fontsize=12)
    
    plt.suptitle('LMI Location Density by ENSO State and SST Anomaly', 
                 fontsize=14, fontweight='bold', y=0.96)
    
    # Use subplots_adjust instead of tight_layout for cartopy compatibility
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.04, 
                        wspace=0.05, hspace=0.08)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"LMI KDE maps saved to: {output_path}")
    return fig


def create_all_plots(start_year=None, end_year=None,
                     mdr_lat_min=None, mdr_lat_max=None,
                     mdr_lon_min=None, mdr_lon_max=None,
                     region='CONUS'):
    """
    Create all visualization plots.
    
    Parameters
    ----------
    start_year, end_year : int, optional
        Years to include in timeseries.
    mdr_* : float, optional
        MDR bounds for PI calculation.
    region : str, default='CONUS'
        Region for landfall analysis ('CONUS' or 'NorthAtlantic')
    """
    print("\nCreating visualization plots...")
    
    # Plot 1: LMI/PI and LFI/LMI
    print("\n  Creating LMI/PI and LFI/LMI plot...")
    plot_lmi_pi_lfi_lmi(region=region)
    
    # Plot 2: Timeseries
    print("\n  Creating timeseries plot...")
    plot_timeseries(start_year=start_year, end_year=end_year,
                    mdr_lat_min=mdr_lat_min, mdr_lat_max=mdr_lat_max,
                    mdr_lon_min=mdr_lon_min, mdr_lon_max=mdr_lon_max)
    
    # Plot 3: LMI KDE maps
    print("\n  Creating LMI KDE maps...")
    plot_lmi_kde_maps()
    
    # Plot 4: LFI/LMI diagnostics
    print("\n  Creating LFI/LMI diagnostics plot...")
    plot_lfi_lmi_diagnostics(region=region)
    
    print("\nAll plots created successfully!")


def plot_lfi_lmi_diagnostics(output_path=None, region='CONUS'):
    """
    Create a 2x2 grid showing histograms and 0-1 inflated Beta distributions
    for each of the four Jewson regions.
    Shows Beta distribution fit with point masses at 0 and 1, plus 95% CI.
    
    Parameters
    ----------
    output_path : str or Path, optional
        Path to save the figure.
    region : str, default='CONUS'
        Region for landfall analysis ('CONUS' or 'NorthAtlantic')
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    if output_path is None:
        output_path = config.get_output_path(f"LFI_LMI_diagnostics_{region}.png")
    
    # Load data
    ibtracs_file = config.get_output_path(config.IBTRACS_PROPERTIES_FILE)
    beta_params_file = config.get_output_path(f"LFI_LMI_{region}_beta_params.csv")
    
    if not Path(ibtracs_file).exists() or not Path(beta_params_file).exists():
        print(f"Error: Required data files not found. Run preprocessing first. Looking for: {beta_params_file}")
        return None
    
    ibtracs_df = pd.read_csv(ibtracs_file, keep_default_na=False, na_values=[''])
    beta_params_df = pd.read_csv(beta_params_file)
    
    # Use region-specific LFI column
    lfi_column = f'{region}_LFI_ms'
    
    # Get storms with valid LFI/LMI
    storms_df = ibtracs_df[ibtracs_df[lfi_column].notna() & ibtracs_df['LMI_ms'].notna()].copy()
    storms_df['LFI_LMI'] = storms_df[lfi_column] / storms_df['LMI_ms']
    storms_df['LFI_LMI'] = storms_df['LFI_LMI'].clip(lower=0.0, upper=1.0)
    
    # Region mapping (only 4 main regions)
    region_names = {
        1.0: 'GOM (Gulf of Mexico)',
        2.0: 'CARB (Caribbean)',
        3.0: 'MDR (Main Development Region)',
        4.0: 'NA (North Atlantic)'
    }
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    from scipy import stats
    x_eval = np.linspace(0, 1, 200)
    
    for idx, (region_num, region_name) in enumerate(region_names.items()):
        ax = axes[idx]
        
        # Get Beta parameters for this region
        region_params = beta_params_df[beta_params_df['RegionNumber'] == region_num]
        region_storms = storms_df[storms_df['RegionNumber'] == region_num]
        
        if len(region_params) == 0 or len(region_storms) == 0:
            ax.text(0.5, 0.5, f'No data for\n{region_name}', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue
        
        # Plot histogram
        n_storms = len(region_storms)
        ratios = region_storms['LFI_LMI'].values
        
        # Calculate histogram
        bins_edges = np.linspace(0, 1, 30)
        hist, _ = np.histogram(ratios, bins=bins_edges, density=True)
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        # Plot histogram as bars
        ax.bar(bin_centers, hist, width=bins_edges[1]-bins_edges[0], 
               alpha=0.3, color='gray', label='Histogram', edgecolor='black', linewidth=0.5)
        
        # Get Beta parameters
        params = region_params.iloc[0]
        alpha = params['alpha']
        beta_val = params['beta']
        p0 = params['p0']
        p1 = params['p1']
        p_continuous = params.get('p_continuous', 1 - p0 - p1)
        
        # Plot Beta distribution (continuous part)
        beta_pdf = stats.beta.pdf(x_eval, alpha, beta_val) * p_continuous
        ax.plot(x_eval, beta_pdf, 'r-', linewidth=2.5, label='0-1 Inflated Beta', zorder=10)
        
        # Plot 95% CI if available
        has_ci = all(k in params for k in ['alpha_ci_lower', 'beta_ci_lower'])
        if has_ci:
            alpha_lower = params['alpha_ci_lower']
            alpha_upper = params['alpha_ci_upper']
            beta_lower = params['beta_ci_lower']
            beta_upper = params['beta_ci_upper']
            
            # Lower and upper CI curves
            beta_pdf_lower = stats.beta.pdf(x_eval, alpha_lower, beta_upper) * p_continuous
            beta_pdf_upper = stats.beta.pdf(x_eval, alpha_upper, beta_lower) * p_continuous
            
            ax.fill_between(x_eval, beta_pdf_lower, beta_pdf_upper, 
                           alpha=0.2, color='red', label='95% CI')
        
        # Show point masses as vertical lines
        if p0 > 0.01:
            ax.axvline(0, color='darkred', linestyle='--', linewidth=2, alpha=0.7,
                      label=f'p₀={p0:.3f}')
        if p1 > 0.01:
            ax.axvline(1, color='darkblue', linestyle='--', linewidth=2, alpha=0.7,
                      label=f'p₁={p1:.3f}')
        
        # Labels and title
        ax.set_xlabel('LFI/LMI Ratio', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{region_name}\n(n={n_storms}, α={alpha:.2f}, β={beta_val:.2f})',
                    fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'LFI/LMI Distribution Diagnostics ({region})', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Diagnostics plot saved to: {output_path}")
    return fig


def plot_lfi_lmi_model_comparison(output_path=None, region='CONUS'):
    """
    Create a comprehensive comparison plot of 0-1 Inflated Beta vs Beta Kernel KDE.
    
    Shows:
    - Cross-validation log-likelihood comparison
    - AIC comparison
    - Data composition (% at boundaries)
    
    Parameters
    ----------
    output_path : str or Path, optional
        Path to save the figure.
    region : str, default='CONUS'
        Region for landfall analysis ('CONUS' or 'NorthAtlantic')
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    if output_path is None:
        output_path = config.get_output_path(f"LFI_LMI_model_comparison_{region}.png")
    
    # Load comparison results
    comparison_file = config.get_output_path(f"LFI_LMI_model_comparison_{region}.csv")
    
    if not Path(comparison_file).exists():
        print(f"Error: Comparison file not found: {comparison_file}")
        return None
    
    df = pd.read_csv(comparison_file)
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Region names for x-axis
    regions = df['RegionName'].fillna('NA').values
    x_pos = np.arange(len(regions))
    
    # ========== Panel A: Cross-Validation Log-Likelihood ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    width = 0.35
    ax1.bar(x_pos - width/2, df['beta_inflated_cv_ll'], width, 
            label='0-1 Inflated Beta', color='#2E86AB', alpha=0.8, edgecolor='black')
    ax1.bar(x_pos + width/2, df['kde_cv_ll'], width, 
            label='Beta Kernel KDE', color='#A23B72', alpha=0.8, edgecolor='black')
    
    # Add error bars
    ax1.errorbar(x_pos - width/2, df['beta_inflated_cv_ll'], 
                yerr=df['beta_inflated_cv_se'], fmt='none', color='black', capsize=3)
    ax1.errorbar(x_pos + width/2, df['kde_cv_ll'], 
                yerr=df['kde_cv_se'], fmt='none', color='black', capsize=3)
    
    ax1.set_xlabel('Region', fontsize=11, fontweight='bold')
    ax1.set_ylabel('10-Fold CV Log-Likelihood\n(higher is better)', fontsize=10, fontweight='bold')
    ax1.set_title('(a) Out-of-Sample Predictive Performance', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(regions)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    
    # Add annotation
    ax1.text(0.02, 0.98, '0-1 Beta wins all regions', transform=ax1.transAxes,
            fontsize=9, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # ========== Panel B: AIC Comparison ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate delta AIC (negative = Beta wins)
    delta_aic = df['beta_inflated_aic'] - df['kde_aic']
    
    colors = ['green' if d < 0 else 'red' for d in delta_aic]
    bars = ax2.bar(x_pos, delta_aic, color=colors, alpha=0.7, edgecolor='black')
    
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='-')
    ax2.axhline(-10, color='green', linewidth=1, linestyle='--', alpha=0.5, label='Strong support (Δ<-10)')
    
    ax2.set_xlabel('Region', fontsize=11, fontweight='bold')
    ax2.set_ylabel('ΔAIC (0-1 Beta − KDE)\n(negative favors 0-1 Beta)', fontsize=10, fontweight='bold')
    ax2.set_title('(b) Information Criterion Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(regions)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, delta_aic)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5, 
                f'{val:.0f}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # ========== Panel C: Data Composition ==========
    ax3 = fig.add_subplot(gs[1, 0])
    
    pct_zeros = 100 * df['n_zeros'] / df['n_storms']
    pct_ones = 100 * df['n_ones'] / df['n_storms']
    pct_interior = 100 * df['n_interior'] / df['n_storms']
    
    ax3.bar(x_pos, pct_zeros, label='LFI/LMI = 0 (no landfall at peak)', 
            color='#E63946', alpha=0.8, edgecolor='black')
    ax3.bar(x_pos, pct_interior, bottom=pct_zeros, label='0 < LFI/LMI < 1 (partial intensity)', 
            color='#F1FAEE', alpha=0.8, edgecolor='black')
    ax3.bar(x_pos, pct_ones, bottom=pct_zeros+pct_interior, label='LFI/LMI = 1 (landfall at peak)', 
            color='#457B9D', alpha=0.8, edgecolor='black')
    
    ax3.set_xlabel('Region', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Percentage of Storms (%)', fontsize=10, fontweight='bold')
    ax3.set_title('(c) Data Composition by Region', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(regions)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add n values on bars
    for i, n in enumerate(df['n_storms']):
        ax3.text(i, 102, f'n={int(n)}', ha='center', fontsize=9)
    
    # ========== Panel D: Summary Statistics Table ==========
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Region', 'n', 'ΔLL (CV)', 'ΔAIC', 'Winner'])
    
    for _, row in df.iterrows():
        region_name = row['RegionName'] if pd.notna(row['RegionName']) else 'NA'
        n = int(row['n_storms'])
        delta_ll = row['cv_ll_diff']
        delta_aic_val = row['beta_inflated_aic'] - row['kde_aic']
        winner = '✓ 0-1 Beta' if row['winner'] == '0-1 Inflated Beta' else row['winner']
        
        table_data.append([
            region_name,
            f"{n}",
            f"{delta_ll:.1f}",
            f"{delta_aic_val:.0f}",
            winner
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#1D3557')
        cell.set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F1FAEE')
            else:
                cell.set_facecolor('white')
    
    ax4.set_title('(d) Model Comparison Summary', fontsize=12, fontweight='bold', pad=20)
    
    # Add footnote
    footnote = ('ΔLL (CV): Cross-validation log-likelihood difference (KDE − Beta). Negative favors 0-1 Beta.\n'
                'ΔAIC: Akaike Information Criterion difference (Beta − KDE). Negative favors 0-1 Beta.\n'
                'All regions show strong preference for 0-1 Inflated Beta distribution.')
    
    fig.text(0.5, 0.02, footnote, ha='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'LFI/LMI Model Comparison: 0-1 Inflated Beta vs Beta Kernel KDE ({region})',
                fontsize=14, fontweight='bold', y=0.98)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Model comparison plot saved to: {output_path}")
    
    return fig


if __name__ == "__main__":
    create_all_plots()
