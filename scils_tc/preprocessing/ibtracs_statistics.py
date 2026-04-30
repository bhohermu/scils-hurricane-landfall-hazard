"""
IBTrACS statistics module.

Calculates KDE for LMI locations, LMI/PI ratios, and LFI/LMI distributions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.optimize import minimize

import config


def get_sst_classification(year, sst_df, climatology_start=None, climatology_end=None):
    """
    Classify a year as 'warm' or 'cold' based on ASO MDR SST relative to climatology.
    
    Parameters
    ----------
    year : int
        Year to classify.
    sst_df : pandas.DataFrame
        DataFrame with Year and ASO_MDR_SST columns.
    climatology_start : int, optional
        Start year of climatology period (default from config).
    climatology_end : int, optional
        End year of climatology period (default from config).
    
    Returns
    -------
    str
        'warm' or 'cold'
    """
    if climatology_start is None:
        climatology_start = config.CLIMATOLOGY_START_YEAR
    if climatology_end is None:
        climatology_end = config.CLIMATOLOGY_END_YEAR
    
    # Calculate mean SST over climatology period
    clim_mask = (sst_df['Year'] >= climatology_start) & (sst_df['Year'] <= climatology_end)
    clim_sst = sst_df[clim_mask]['ASO_MDR_SST']
    
    if len(clim_sst) == 0:
        # Fall back to all available years if climatology period not available
        mean_sst = sst_df['ASO_MDR_SST'].mean()
    else:
        mean_sst = clim_sst.mean()
    
    year_sst = sst_df[sst_df['Year'] == year]['ASO_MDR_SST'].values
    
    if len(year_sst) == 0:
        return None
    
    return 'warm' if year_sst[0] >= mean_sst else 'cold'


def calculate_lmi_kde(ibtracs_df, enso_df, output_dir=None, force_recalculate=False):
    """
    Calculate LMI location kernel density estimates for 3 ENSO groups.
    
    Groups: lanina, neutral, elnino
    
    Parameters
    ----------
    ibtracs_df : pandas.DataFrame
        IBTrACS properties with LMI_Lat, LMI_Lon columns.
    enso_df : pandas.DataFrame
        ENSO state classification with Year, ENSO_State columns.
    output_dir : str or Path, optional
        Directory to save KDE outputs.
    force_recalculate : bool, optional
        If True, recalculate even if output files exist.
    
    Returns
    -------
    dict
        Dictionary of KDE results for each group.
    """
    if output_dir is None:
        output_dir = config.PREPROCESSED_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if outputs already exist
    expected_files = ['LMI_KDE_lanina.nc', 'LMI_KDE_neutral.nc', 'LMI_KDE_elnino.nc']
    all_exist = all((output_dir / f).exists() for f in expected_files)
    
    if all_exist and not force_recalculate:
        print("LMI KDE files already exist. Use --force-recalculate to regenerate.")
        return {}
    
    print("Calculating LMI location KDEs by ENSO group...")
    
    # Load land-sea mask from ERA5
    lsm_ds = xr.open_dataset(config.ERA5_SST_MSLP_FILE)
    if 'lsm' in lsm_ds:
        lsm = lsm_ds['lsm']
        # Get the first time step if time dimension exists
        if 'valid_time' in lsm.dims:
            lsm = lsm.isel(valid_time=0)
        elif 'time' in lsm.dims:
            lsm = lsm.isel(time=0)
        lsm_lat = lsm['latitude'].values if 'latitude' in lsm.dims else lsm['lat'].values
        lsm_lon = lsm['longitude'].values if 'longitude' in lsm.dims else lsm['lon'].values
        print(f"  Loaded land-sea mask: {lsm.shape}")
    else:
        lsm = None
        print("  Warning: No land-sea mask found in ERA5 file")
    lsm_ds.close()
    
    # Merge data
    df = ibtracs_df.merge(enso_df[['Year', 'ENSO_State']], on='Year', how='left')
    
    # Define groups (ENSO only)
    groups = {
        'lanina': 'La Nina',
        'neutral': 'Neutral',
        'elnino': 'El Nino',
    }
    
    # Create grid for KDE evaluation using configured resolution (~0.25° like ERA5)
    grid_res = config.KDE_GRID_RESOLUTION
    lon_grid = np.arange(-100, 10 + grid_res, grid_res)
    lat_grid = np.arange(0, 50 + grid_res, grid_res)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    positions = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()])
    
    print(f"  KDE grid: {len(lon_grid)}x{len(lat_grid)} at {grid_res}° resolution")
    
    results = {}
    
    for group_name, enso_state in groups.items():
        mask = (df['ENSO_State'] == enso_state)
        group_df = df[mask]
        
        # Calculate number of unique years for annual rate calculation
        n_years = group_df['Year'].nunique()
        
        # Skip if no data for this group
        if n_years == 0 or len(group_df) == 0:
            print(f"  {group_name}: No data (skipping)")
            continue
        
        print(f"  {group_name}: {len(group_df)} storms over {n_years} years (rate: {len(group_df)/n_years:.2f} storms/year)")
        
        if len(group_df) < 2:
            print("    Warning: Not enough data for KDE")
            continue
        
        # Get LMI locations
        lons = group_df['LMI_Lon'].dropna().values
        lats = group_df['LMI_Lat'].dropna().values
        
        if len(lons) < 2:
            continue
        
        # Calculate KDE
        try:
            values = np.vstack([lons, lats])
            kernel = stats.gaussian_kde(values, bw_method=config.KDE_BANDWIDTH)
            kde_values = kernel(positions).reshape(lon_mesh.shape)
            
            # Convert from total count density to annual rate density
            # by dividing by the number of years
            kde_values = kde_values / n_years
            
            # Convert to storms/year/km² (area-normalized like old method)
            earth_radius_km = 6371.0
            resolution_radians = np.radians(grid_res)
            cell_area_km2 = (earth_radius_km ** 2 * resolution_radians ** 2 * 
                           np.cos(np.radians(lat_grid)))
            kde_values = kde_values / cell_area_km2[:, np.newaxis]
            
            # Apply land-sea mask (set KDE to NaN over land)
            if lsm is not None:
                # Interpolate land-sea mask to KDE grid
                from scipy.interpolate import RegularGridInterpolator
                
                # Handle longitude convention (ERA5 may use 0-360)
                if lsm_lon.min() >= 0:
                    # Convert to -180 to 180
                    lsm_lon_converted = np.where(lsm_lon > 180, lsm_lon - 360, lsm_lon)
                    sort_idx = np.argsort(lsm_lon_converted)
                    lsm_lon_sorted = lsm_lon_converted[sort_idx]
                    lsm_values = lsm.values[:, sort_idx]
                else:
                    lsm_lon_sorted = lsm_lon
                    lsm_values = lsm.values
                
                # Flip latitude if needed (ERA5 is typically N-to-S)
                if lsm_lat[0] > lsm_lat[-1]:
                    lsm_lat_sorted = lsm_lat[::-1]
                    lsm_values = lsm_values[::-1, :]
                else:
                    lsm_lat_sorted = lsm_lat
                
                # Create interpolator
                interp = RegularGridInterpolator(
                    (lsm_lat_sorted, lsm_lon_sorted), 
                    lsm_values,
                    method='nearest',
                    bounds_error=False,
                    fill_value=1.0  # Treat out-of-bounds as land
                )
                
                # Interpolate to KDE grid
                points = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
                lsm_interp = interp(points).reshape(lat_mesh.shape)
                
                # Set KDE to NaN where land fraction > 0.5
                kde_values = np.where(lsm_interp > 0.5, np.nan, kde_values)
            
            # Apply Atlantic basin polygon mask (exclude Pacific, Mediterranean, etc.)
            if hasattr(config, 'ATLANTIC_BASIN_POLYGON') and config.ATLANTIC_BASIN_POLYGON:
                from matplotlib.path import Path as MplPath
                
                polygon = MplPath(config.ATLANTIC_BASIN_POLYGON)
                grid_points = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
                inside_atlantic = polygon.contains_points(grid_points).reshape(lon_mesh.shape)
                
                # Set KDE to NaN outside Atlantic basin
                kde_values = np.where(inside_atlantic, kde_values, np.nan)
                print("    Applied Atlantic basin mask")
            
            # Save as NetCDF
            kde_ds = xr.Dataset({
                'kde': (['lat', 'lon'], kde_values)
            }, coords={
                'lat': lat_grid,
                'lon': lon_grid
            })
            
            # Calculate number of unique years
            n_years = group_df['Year'].nunique()
            
            kde_ds['kde'].attrs = {
                'standard_name': 'LMI location annual rate density',
                'units': 'storms/year/km²',
                'long_name': 'Kernel density estimate of LMI location annual rate per km²',
                'group': group_name,
                'enso_state': enso_state,
                'n_storms': len(group_df),
                'n_years': n_years
            }
            
            output_file = output_dir / f"LMI_KDE_{group_name}.nc"
            kde_ds.to_netcdf(output_file)
            print(f"    Saved to: {output_file}")
            
            results[group_name] = {
                'kde': kde_values,
                'lon': lon_grid,
                'lat': lat_grid,
                'n_storms': len(group_df)
            }
        
        except Exception as e:
            print(f"    Error calculating KDE: {e}")
    
    return results


def calculate_lmi_pi_ratio(ibtracs_df, pi_file=None, output_path=None, force_recalculate=False):
    """
    Calculate LMI/PI ratio distribution using single histogram (1dist method).
    
    Uses the local PI at the LMI location and the month when LMI occurs.
    Constrains LMI/PI to <=1.
    
    Parameters
    ----------
    ibtracs_df : pandas.DataFrame
        IBTrACS properties with LMI_ms, LMI_Lat, LMI_Lon, LMI_month, Year columns.
    pi_file : str or Path, optional
        Path to PI NetCDF file.
    output_path : str or Path, optional
        Path to save histogram CSV.
    force_recalculate : bool, optional
        If True, recalculate even if output file exists.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with histogram data.
    """
    if pi_file is None:
        pi_file = config.get_output_path(config.PI_FILE)
    if output_path is None:
        output_path = config.get_output_path(config.LMI_PI_RATIO_FILE)
    
    # Check if output already exists
    if Path(output_path).exists() and not force_recalculate:
        print("LMI/PI ratio file already exists. Use --force-recalculate to regenerate.")
        return pd.read_csv(output_path)
    
    print("Calculating LMI/PI ratio distribution (single histogram)...")
    print("  Using month-specific PI at LMI location and time")
    
    # Load PI data
    pi_ds = xr.open_dataset(pi_file)
    pi = pi_ds['vmax']
    
    lat_name = 'latitude' if 'latitude' in pi.dims else 'lat'
    lon_name = 'longitude' if 'longitude' in pi.dims else 'lon'
    
    ratios = []
    pi_values = []
    n_valid = 0
    n_no_month = 0
    n_month_outside_season = 0
    n_pi_nan = 0
    
    for _, row in ibtracs_df.iterrows():
        lmi_ms = row['LMI_ms']
        lmi_lat = row['LMI_Lat']
        lmi_lon = row['LMI_Lon']
        year = row['Year']
        lmi_month = row.get('LMI_month', None)
        
        if pd.isna(lmi_ms) or pd.isna(lmi_lat) or pd.isna(lmi_lon):
            continue
        
        if pd.isna(lmi_month):
            n_no_month += 1
            continue
        
        lmi_month = int(lmi_month)
        
        if lmi_month not in config.SEASON_MONTHS:
            n_month_outside_season += 1
            continue
        
        try:
            if pi[lon_name].min() >= 0:
                lmi_lon_adj = lmi_lon % 360
            else:
                lmi_lon_adj = lmi_lon
            
            time_mask = (
                (pi['time'].dt.year == year) &
                (pi['time'].dt.month == lmi_month)
            )
            pi_month = pi.where(time_mask, drop=True)
            
            if len(pi_month.time) == 0:
                n_pi_nan += 1
                continue
            
            pi_local = float(pi_month.isel(time=0).sel(
                **{lat_name: lmi_lat, lon_name: lmi_lon_adj},
                method='nearest'
            ).values)
            
            if np.isnan(pi_local) or pi_local <= 0:
                n_pi_nan += 1
                continue
            
            ratio = lmi_ms / pi_local
            ratios.append(ratio)
            pi_values.append(pi_local)
            n_valid += 1
        
        except Exception:
            continue
    
    pi_ds.close()
    
    if len(ratios) == 0:
        print("  Warning: No valid LMI/PI ratios calculated")
        return None
    
    print(f"  Calculated {n_valid} LMI/PI ratios")
    print(f"  Excluded: {n_no_month} no LMI month, {n_month_outside_season} outside season, {n_pi_nan} PI=NaN")
    
    ratios_arr = np.array(ratios)
    
    # Constrain LMI/PI to <=1
    n_above_1 = np.sum(ratios_arr > 1.0)
    ratios_constrained = ratios_arr[ratios_arr <= 1.0]
    
    print(f"  {n_above_1} values ({100*n_above_1/len(ratios_arr):.1f}%) excluded (were >1)")
    print(f"  {len(ratios_constrained)} valid LMI/PI ratios (<=1)")
    
    # Compute single histogram
    n_bins = 50
    range_min = max(0.0, ratios_constrained.min() - 0.01)
    hist, edges = np.histogram(ratios_constrained, bins=n_bins, range=(range_min, 1.0), density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    # Save histogram
    hist_df = pd.DataFrame({
        'bin_center': bin_centers,
        'density': hist
    })
    hist_df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    # Print summary statistics
    print("\n  LMI/PI statistics (constrained <=1):")
    print(f"    Mean: {ratios_constrained.mean():.3f}")
    print(f"    Median: {np.median(ratios_constrained):.3f}")
    print(f"    Std: {ratios_constrained.std():.3f}")
    print(f"    Min: {ratios_constrained.min():.3f}, Max: {ratios_constrained.max():.3f}")
    
    return hist_df


def fit_zero_one_inflated_beta(data, x_points, return_ci=False, n_bootstrap=1000):
    """
    Fit a zero-one inflated Beta distribution to data in [0, 1].
    
    Model: f(x) = p0 * I(x=0) + p1 * I(x=1) + (1-p0-p1) * Beta(x; alpha, beta)
    
    Parameters
    ----------
    data : array-like
        Data values in [0, 1].
    x_points : array-like
        Points at which to evaluate the density.
    return_ci : bool, optional
        If True, compute bootstrap 95% confidence intervals for parameters.
    n_bootstrap : int, optional
        Number of bootstrap iterations for CI calculation.
    
    Returns
    -------
    density_values : ndarray
        Density values at x_points (continuous part only).
    params : dict
        Fitted parameters: alpha, beta, p0, p1, p_continuous.
    """
    data = np.asarray(data)
    n = len(data)
    
    # Identify exact zeros and ones (within numerical tolerance)
    tol = 1e-6
    n_zeros = np.sum(data < tol)
    n_ones = np.sum(data > 1 - tol)
    n_interior = n - n_zeros - n_ones
    
    # Estimate point mass probabilities
    p0 = n_zeros / n
    p1 = n_ones / n
    p_continuous = 1 - p0 - p1
    
    # Get interior data for Beta fitting
    interior_data = data[(data >= tol) & (data <= 1 - tol)]
    
    if len(interior_data) < 2:
        params = {'alpha': 1.0, 'beta': 1.0, 'p0': p0, 'p1': p1, 'p_continuous': p_continuous}
        density_values = np.ones_like(x_points) * p_continuous
        return density_values, params
    
    # Method of moments for initial guess
    mean_interior = np.mean(interior_data)
    var_interior = np.var(interior_data)
    
    if var_interior > 0 and var_interior < mean_interior * (1 - mean_interior):
        common = mean_interior * (1 - mean_interior) / var_interior - 1
        alpha_init = mean_interior * common
        beta_init = (1 - mean_interior) * common
    else:
        alpha_init = 2.0
        beta_init = 2.0
    
    # MLE for Beta parameters
    def neg_log_lik(params):
        """Return the negative log likelihood for Beta parameters on interior data."""
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return 1e10
        try:
            return -np.sum(stats.beta.logpdf(interior_data, alpha, beta))
        except:
            return 1e10
    
    result = minimize(neg_log_lik, [alpha_init, beta_init], 
                     method='L-BFGS-B', 
                     bounds=[(0.01, 100), (0.01, 100)])
    
    if result.success:
        alpha, beta = result.x
    else:
        alpha, beta = alpha_init, beta_init
    
    params = {
        'alpha': alpha, 
        'beta': beta, 
        'p0': p0, 
        'p1': p1, 
        'p_continuous': p_continuous,
        'n_zeros': n_zeros,
        'n_ones': n_ones,
        'n_interior': n_interior
    }
    
    # Bootstrap confidence intervals if requested
    if return_ci and n >= 10:
        alpha_boot, beta_boot, p0_boot, p1_boot = [], [], [], []
        np.random.seed(42)
        
        for _ in range(n_bootstrap):
            boot_data = np.random.choice(data, size=n, replace=True)
            n_zeros_b = np.sum(boot_data < tol)
            n_ones_b = np.sum(boot_data > 1 - tol)
            p0_boot.append(n_zeros_b / n)
            p1_boot.append(n_ones_b / n)
            
            interior_boot = boot_data[(boot_data >= tol) & (boot_data <= 1 - tol)]
            
            if len(interior_boot) >= 2:
                mean_b = np.mean(interior_boot)
                var_b = np.var(interior_boot)
                
                if var_b > 0 and var_b < mean_b * (1 - mean_b):
                    common_b = mean_b * (1 - mean_b) / var_b - 1
                    alpha_b = np.clip(mean_b * common_b, 0.1, 50)
                    beta_b = np.clip((1 - mean_b) * common_b, 0.1, 50)
                else:
                    alpha_b, beta_b = alpha, beta
                alpha_boot.append(alpha_b)
                beta_boot.append(beta_b)
            else:
                alpha_boot.append(alpha)
                beta_boot.append(beta)
        
        params['alpha_ci'] = (np.percentile(alpha_boot, 2.5), np.percentile(alpha_boot, 97.5))
        params['beta_ci'] = (np.percentile(beta_boot, 2.5), np.percentile(beta_boot, 97.5))
        params['p0_ci'] = (np.percentile(p0_boot, 2.5), np.percentile(p0_boot, 97.5))
        params['p1_ci'] = (np.percentile(p1_boot, 2.5), np.percentile(p1_boot, 97.5))
    
    density_values = p_continuous * stats.beta.pdf(x_points, alpha, beta)
    return density_values, params


def calculate_lfi_lmi_distributions(ibtracs_df, region='CONUS', output_path=None, force_recalculate=False):
    """
    Calculate LFI/LMI distributions using 0-1 inflated Beta distribution.
    
    Computes three different distribution estimates:
    - 'histogram': Direct histogram density estimation
    - 'kde': Gaussian KDE capped to [0,1] and rescaled
    - 'beta': Zero-one inflated Beta distribution (includes point masses at 0 and 1)
    
    The LFI/LMI ratio is clipped to [0, 1] since LFI cannot be negative
    or exceed LMI.
    
    Parameters
    ----------
    ibtracs_df : pandas.DataFrame
        IBTrACS properties with CONUS_LFI_ms, NorthAtlantic_LFI_ms, LMI_ms, RegionNumber, RegionName columns.
    region : str
        'CONUS' for US mainland only, 'NorthAtlantic' for all North Atlantic landfalls
    output_path : str or Path, optional
        Path to save output CSV.
    force_recalculate : bool, optional
        If True, recalculate even if output files exist.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with density values for all methods at each x point for each region.
        Columns: lfi_lmi, histogram, kde, beta, RegionName, RegionNumber
    """
    from scipy.stats import gaussian_kde
    
    if output_path is None:
        output_filename = f"LFI_LMI_{region}.csv"
        output_path = config.get_output_path(output_filename)
    
    # Also check for beta params file
    beta_params_path = config.get_output_path(f"LFI_LMI_{region}_beta_params.csv")
    
    # Check if outputs already exist
    if Path(output_path).exists() and Path(beta_params_path).exists() and not force_recalculate:
        print(f"LFI/LMI {region} files already exist. Use --force-recalculate to regenerate.")
        return pd.read_csv(output_path)
    
    print(f"Calculating LFI/LMI distributions (all methods) by region ({region})...")
    
    # Select appropriate LFI column based on region
    lfi_column = f'{region}_LFI_ms'
    
    # Filter for storms with valid LMI (includes storms with LFI=0 for non-landfalls)
    df = ibtracs_df[ibtracs_df[lfi_column].notna() & ibtracs_df['LMI_ms'].notna()].copy()
    
    # Calculate LFI/LMI ratio (including LFI=0 for non-landfalling storms)
    df['LFI_LMI'] = df[lfi_column] / df['LMI_ms']
    
    # Clip LFI/LMI to [0, 1] - LFI cannot be negative or exceed LMI
    df['LFI_LMI'] = df['LFI_LMI'].clip(lower=0.0, upper=1.0)
    
    # X-axis for evaluation (same for all methods)
    x_points = np.linspace(0, 1, config.LFI_LMI_DISCRETIZATION_POINTS)
    
    all_results = []
    beta_params_list = []  # Store beta parameters separately
    
    for region_num in sorted(df['RegionNumber'].dropna().unique()):
        region_df = df[df['RegionNumber'] == region_num]
        
        if len(region_df) < 2:
            continue
        
        region_name = region_df['RegionName'].iloc[0]
        ratios = region_df['LFI_LMI'].values
        
        print(f"  Region {region_num} ({region_name}): {len(region_df)} landfalls")
        
        # Method 1: Histogram (direct density estimation)
        n_bins = config.LFI_LMI_DISCRETIZATION_POINTS - 1
        hist, bin_edges = np.histogram(ratios, bins=n_bins, range=(0, 1), density=True)
        # Interpolate histogram to x_points
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist_density = np.interp(x_points, bin_centers, hist)
        # Ensure non-negative
        hist_density = np.maximum(hist_density, 0)
        print(f"    Histogram: {n_bins} bins")
        
        # Method 2: Gaussian KDE
        try:
            kde = gaussian_kde(ratios, bw_method='scott')
            kde_density = kde(x_points)
            # Clip to valid range and rescale
            kde_density = np.maximum(kde_density, 0)
            kde_density = kde_density / np.trapz(kde_density, x_points)  # Normalize
            print(f"    KDE: bandwidth={kde.factor:.4f}")
        except Exception as e:
            print(f"    KDE failed: {e}, using histogram")
            kde_density = hist_density.copy()
        
        # Method 3: Zero-one inflated Beta
        _, beta_params = fit_zero_one_inflated_beta(ratios, x_points, return_ci=True, n_bootstrap=2000)
        alpha = beta_params['alpha']
        beta_param = beta_params['beta']
        p0 = beta_params['p0']
        p1 = beta_params['p1']
        p_continuous = beta_params['p_continuous']
        
        # Calculate continuous Beta density (scaled by p_continuous)
        # The density for the Beta part is: p_continuous * Beta_pdf(x; alpha, beta)
        beta_density = p_continuous * stats.beta.pdf(x_points, alpha, beta_param)
        # Handle edge cases where x=0 or x=1 exactly
        beta_density = np.nan_to_num(beta_density, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"    Beta: alpha={alpha:.2f}, beta={beta_param:.2f}, p0={p0:.3f}, p1={p1:.3f}")
        
        # Store beta parameters for separate file
        beta_params['RegionName'] = region_name
        beta_params['RegionNumber'] = int(region_num)
        # Flatten CI tuples
        if 'alpha_ci' in beta_params:
            beta_params['alpha_ci_lower'] = beta_params['alpha_ci'][0]
            beta_params['alpha_ci_upper'] = beta_params['alpha_ci'][1]
            beta_params['beta_ci_lower'] = beta_params['beta_ci'][0]
            beta_params['beta_ci_upper'] = beta_params['beta_ci'][1]
            beta_params['p0_ci_lower'] = beta_params['p0_ci'][0]
            beta_params['p0_ci_upper'] = beta_params['p0_ci'][1]
            beta_params['p1_ci_lower'] = beta_params['p1_ci'][0]
            beta_params['p1_ci_upper'] = beta_params['p1_ci'][1]
            del beta_params['alpha_ci']
            del beta_params['beta_ci']
            del beta_params['p0_ci']
            del beta_params['p1_ci']
        beta_params_list.append(beta_params)
        
        # Build results for this region
        for i, x in enumerate(x_points):
            all_results.append({
                'lfi_lmi': x,
                'histogram': hist_density[i],
                'kde': kde_density[i],
                'beta': beta_density[i],
                'RegionName': region_name,
                'RegionNumber': int(region_num)
            })
    
    # Save main distributions file
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(output_path, index=False)
    print(f"  Distributions saved to: {output_path}")
    
    # Save beta parameters separately (includes p0, p1 for point masses)
    beta_params_path = str(output_path).replace('.csv', '_beta_params.csv')
    beta_params_df = pd.DataFrame(beta_params_list)
    beta_params_df.to_csv(beta_params_path, index=False)
    print(f"  Beta parameters (with p0, p1) saved to: {beta_params_path}")
    
    return result_df


if __name__ == "__main__":
    # Test the module (requires preprocessed files)
    ibtracs_file = config.get_output_path(config.IBTRACS_PROPERTIES_FILE)
    enso_file = config.get_output_path(config.ENSO_STATE_FILE)
    
    if Path(ibtracs_file).exists():
        ibtracs_df = pd.read_csv(ibtracs_file)
        
        if Path(enso_file).exists():
            enso_df = pd.read_csv(enso_file)
            calculate_lmi_kde(ibtracs_df, enso_df)
        
        calculate_lfi_lmi_distributions(ibtracs_df, region='NorthAtlantic')
