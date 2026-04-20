"""
Detrending functions for SCILS TC Model.

This module provides Theil-Sen regression-based detrending for PI and CGI maps,
using GWL (Global Warming Level) as the predictor variable.

The detrending process:
1. Merge annual PI/CGI data with GWL data (from preprocessing)
2. Fit Theil-Sen regression: variable ~ GWL for each grid cell
3. Save regression coefficients (slope, intercept) - these are constant
4. Apply detrending to extrapolate/interpolate to target GWL
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

import config

from .gwl_lookup import (
    format_target_label,
    get_filename_suffix,
    gwl_to_year,
    load_gwl_data,
    resolve_target,
    year_to_gwl,
)


def _theilsen_per_gridcell(data: np.ndarray, predictor: np.ndarray) -> tuple:
    """
    Calculate Theil-Sen regression for a single grid cell.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of values over time
    predictor : np.ndarray
        1D array of predictor values (GWL) corresponding to data
        
    Returns
    -------
    tuple
        (slope, intercept)
    """
    # Handle NaN values
    valid_mask = ~np.isnan(data) & ~np.isnan(predictor)
    if valid_mask.sum() < 2:
        return np.nan, np.nan
    
    try:
        result = stats.theilslopes(data[valid_mask], predictor[valid_mask])
        return result.slope, result.intercept
    except Exception:
        return np.nan, np.nan


def calculate_theilsen_regression_gwl(
    data: xr.DataArray,
    gwl_df: pd.DataFrame = None,
    time_dim: str = 'time'
) -> xr.Dataset:
    """
    Calculate Theil-Sen regression coefficients (vs GWL) for each grid cell.
    
    Parameters
    ----------
    data : xr.DataArray
        Data array with time dimension and spatial dimensions
    gwl_df : pd.DataFrame, optional
        DataFrame with Year and GWL columns. If None, loads from file.
    time_dim : str
        Name of the time dimension
        
    Returns
    -------
    xr.Dataset
        Dataset with 'slope' and 'intercept' variables (vs GWL)
    """
    # Load GWL data if not provided
    if gwl_df is None:
        gwl_df = load_gwl_data()
    
    # Annual mean for regression (average all months within each year)
    annual_data = data.groupby(f'{time_dim}.year').mean(dim=time_dim)
    annual_years = annual_data['year'].values.astype(int)
    
    # Get GWL for each year
    gwl_values = np.array([
        gwl_df.loc[gwl_df['Year'] == yr, 'GWL'].values[0] 
        if yr in gwl_df['Year'].values else np.nan
        for yr in annual_years
    ])
    
    # Check for missing GWL values
    n_missing = np.isnan(gwl_values).sum()
    if n_missing > 0:
        missing_years = annual_years[np.isnan(gwl_values)]
        print(f"  Warning: {n_missing} years missing GWL data: {missing_years}")
    
    # Stack spatial dimensions for vectorized processing
    stacked = annual_data.stack(gridcell=annual_data.dims[1:])
    
    # Calculate regression for each grid cell
    n_gridcells = stacked.shape[1]
    slopes = np.empty(n_gridcells)
    intercepts = np.empty(n_gridcells)
    
    print(f"  Calculating Theil-Sen regression (vs GWL) for {n_gridcells} grid cells...")
    
    for i in range(n_gridcells):
        if i % 10000 == 0 and i > 0:
            print(f"    Processed {i}/{n_gridcells} grid cells...")
        slopes[i], intercepts[i] = _theilsen_per_gridcell(
            stacked.values[:, i], 
            gwl_values
        )
    
    # Reshape back to original spatial dimensions
    slope_da = stacked.isel(year=0).copy()
    slope_da.values = slopes
    slope_da = slope_da.unstack('gridcell')
    
    intercept_da = stacked.isel(year=0).copy()
    intercept_da.values = intercepts
    intercept_da = intercept_da.unstack('gridcell')
    
    result = xr.Dataset({
        'slope': slope_da.drop_vars('year', errors='ignore'),
        'intercept': intercept_da.drop_vars('year', errors='ignore'),
    })
    
    # Add metadata
    result.attrs['regression_type'] = 'theil-sen'
    result.attrs['predictor'] = 'GWL'
    result.attrs['gwl_range'] = f"{gwl_values[~np.isnan(gwl_values)].min():.2f}-{gwl_values[~np.isnan(gwl_values)].max():.2f}"
    result.attrs['years_used'] = f"{int(annual_years.min())}-{int(annual_years.max())}"
    result.attrs['n_years'] = int((~np.isnan(gwl_values)).sum())
    
    return result


def apply_gwl_detrending(
    data: xr.DataArray,
    regression: xr.Dataset,
    target_gwl: float,
    gwl_df: pd.DataFrame = None,
    time_dim: str = 'time'
) -> xr.DataArray:
    """
    Apply GWL-based detrending to data.
    
    Y_{t, detrended to target_gwl} = Y_{t, observed} + Y_{target_gwl, estimated} - Y_{gwl_t, estimated}
    
    Where:
        Y_{gwl_t, estimated} = slope * gwl_t + intercept
        Y_{target_gwl, estimated} = slope * target_gwl + intercept
    
    Parameters
    ----------
    data : xr.DataArray
        Original data with time dimension
    regression : xr.Dataset
        Dataset with 'slope' and 'intercept' from calculate_theilsen_regression_gwl()
    target_gwl : float
        Target GWL to detrend to
    gwl_df : pd.DataFrame, optional
        DataFrame with Year and GWL columns. If None, loads from file.
    time_dim : str
        Name of the time dimension
        
    Returns
    -------
    xr.DataArray
        Detrended data
    """
    # Load GWL data if not provided
    if gwl_df is None:
        gwl_df = load_gwl_data()
    
    # Get years from time coordinate
    times = data[time_dim].values
    if np.issubdtype(times.dtype, np.datetime64):
        years = pd.to_datetime(times).year.values
    else:
        years = times.astype(int)
    
    # Get GWL for each time step (lookup by year)
    gwl_values = np.array([
        gwl_df.loc[gwl_df['Year'] == yr, 'GWL'].values[0] 
        if yr in gwl_df['Year'].values else year_to_gwl(yr, use_lookup=False)
        for yr in years
    ])
    
    # Create GWL coordinate for broadcasting
    gwl_da = xr.DataArray(gwl_values, dims=[time_dim], coords={time_dim: data[time_dim]})
    
    # Calculate estimated values at each time step's GWL
    y_estimated = regression['slope'] * gwl_da + regression['intercept']
    
    # Calculate estimated value at target GWL
    y_target = regression['slope'] * target_gwl + regression['intercept']
    
    # Apply detrending formula
    detrended = data + y_target - y_estimated
    
    # Copy attributes and add detrending info
    detrended.attrs = data.attrs.copy()
    detrended.attrs['detrended_to_gwl'] = target_gwl
    detrended.attrs['detrended_to_year_approx'] = gwl_to_year(target_gwl)
    detrended.attrs['detrending_method'] = 'gwl'
    
    return detrended


def detrend_pi(
    start_year: int = None,
    end_year: int = None,
    target_year: float = None,
    target_gwl: float = None,
    save_regression: bool = True,
    force_recalculate: bool = False,
) -> tuple:
    """
    Detrend PI (vmax) maps using Theil-Sen regression vs GWL.
    
    The regression coefficients are saved separately from the detrended data,
    so they only need to be calculated once.
    
    Parameters
    ----------
    start_year : int, optional
        Start year for data (default: config.START_YEAR)
    end_year : int, optional
        End year for data (default: config.END_YEAR)
    target_year : float, optional
        Target year for detrending (converted to GWL)
    target_gwl : float, optional
        Target GWL for detrending (default: config.DEFAULT_TARGET_GWL)
    save_regression : bool
        Whether to save regression coefficients
    force_recalculate : bool
        If True, recalculate regression even if it exists
        
    Returns
    -------
    tuple
        (detrended_data, regression_coefficients, effective_gwl, effective_year, specified_by_year)
    """
    print("\nDetrending PI (vmax)...")
    
    # Set defaults
    if start_year is None:
        start_year = config.START_YEAR
    if end_year is None:
        end_year = config.END_YEAR
    
    # Resolve target
    effective_gwl, effective_year, specified_by_year = resolve_target(
        target_year=target_year,
        target_gwl=target_gwl,
        default_gwl=config.DEFAULT_TARGET_GWL
    )
    
    print(f"  Target: {format_target_label(effective_gwl, effective_year, specified_by_year)}")
    
    # Load GWL data
    gwl_df = load_gwl_data()
    
    # Load PI data
    pi_path = config.get_output_path(config.PI_FILE)
    print(f"  Loading PI data from {pi_path}...")
    ds = xr.open_dataset(pi_path)
    vmax = ds['vmax']
    
    # Regression file (independent of target - coefficients are constant)
    reg_filename = 'PI_regression_vs_GWL.nc'
    reg_path = config.get_detrended_path(reg_filename)
    
    if reg_path.exists() and not force_recalculate:
        print(f"  Loading existing regression from {reg_path}")
        regression = xr.open_dataset(reg_path)
    else:
        # Calculate regression
        print("  Fitting Theil-Sen regression (PI vs GWL)...")
        regression = calculate_theilsen_regression_gwl(vmax, gwl_df, time_dim='time')
        
        # Save regression if requested
        if save_regression:
            regression.to_netcdf(reg_path)
            print(f"  Saved regression to {reg_path}")
    
    # Apply detrending to target GWL
    print(f"  Applying detrending to GWL {effective_gwl:.2f}°C...")
    detrended = apply_gwl_detrending(vmax, regression, effective_gwl, gwl_df, time_dim='time')
    
    # Clip negative PI values to NaN (physical constraint: PI must be >= 0)
    n_negative = (detrended < 0).sum().values
    if n_negative > 0:
        print(f"  Setting {n_negative} negative PI values to NaN (physical constraint)")
        detrended = detrended.where(detrended >= 0, np.nan)
    
    # Save detrended data with target in filename
    filename_suffix = get_filename_suffix(effective_gwl, effective_year, specified_by_year)
    filename = f'ERA5_PI_detrended_{filename_suffix}.nc'
    
    out_path = config.get_detrended_path(filename)
    detrended.to_netcdf(out_path)
    print(f"  Saved detrended PI to {out_path}")
    
    ds.close()
    
    return detrended, regression, effective_gwl, effective_year, specified_by_year


def detrend_cgi(
    start_year: int = None,
    end_year: int = None,
    target_year: float = None,
    target_gwl: float = None,
    save_regression: bool = True,
    force_recalculate: bool = False,
) -> tuple:
    """
    Detrend CGI maps using Theil-Sen regression vs GWL.
    
    The regression coefficients are saved separately from the detrended data,
    so they only need to be calculated once.
    
    Parameters
    ----------
    start_year : int, optional
        Start year for data (default: config.START_YEAR)
    end_year : int, optional
        End year for data (default: config.END_YEAR)
    target_year : float, optional
        Target year for detrending (converted to GWL)
    target_gwl : float, optional
        Target GWL for detrending (default: config.DEFAULT_TARGET_GWL)
    save_regression : bool
        Whether to save regression coefficients
    force_recalculate : bool
        If True, recalculate regression even if it exists
        
    Returns
    -------
    tuple
        (detrended_data, regression_coefficients, effective_gwl, effective_year, specified_by_year)
    """
    print("\nDetrending CGI...")
    
    # Set defaults
    if start_year is None:
        start_year = config.START_YEAR
    if end_year is None:
        end_year = config.END_YEAR
    
    # Resolve target
    effective_gwl, effective_year, specified_by_year = resolve_target(
        target_year=target_year,
        target_gwl=target_gwl,
        default_gwl=config.DEFAULT_TARGET_GWL
    )
    
    print(f"  Target: {format_target_label(effective_gwl, effective_year, specified_by_year)}")
    
    # Load GWL data
    gwl_df = load_gwl_data()
    
    # Load CGI data
    cgi_path = config.get_output_path(config.CGI_MAP_FILE)
    print(f"  Loading CGI data from {cgi_path}...")
    ds = xr.open_dataset(cgi_path)
    cgi = ds['cgi']
    
    # Regression file (independent of target - coefficients are constant)
    reg_filename = 'CGI_regression_vs_GWL.nc'
    reg_path = config.get_detrended_path(reg_filename)
    
    if reg_path.exists() and not force_recalculate:
        print(f"  Loading existing regression from {reg_path}")
        regression = xr.open_dataset(reg_path)
    else:
        # Calculate regression
        print("  Fitting Theil-Sen regression (CGI vs GWL)...")
        regression = calculate_theilsen_regression_gwl(cgi, gwl_df, time_dim='time')
        
        # Save regression if requested
        if save_regression:
            regression.to_netcdf(reg_path)
            print(f"  Saved regression to {reg_path}")
    
    # Apply detrending to target GWL
    print(f"  Applying detrending to GWL {effective_gwl:.2f}°C...")
    detrended = apply_gwl_detrending(cgi, regression, effective_gwl, gwl_df, time_dim='time')
    
    # Save detrended data with target in filename
    filename_suffix = get_filename_suffix(effective_gwl, effective_year, specified_by_year)
    filename = f'ERA5_CGI_detrended_{filename_suffix}.nc'
    
    out_path = config.get_detrended_path(filename)
    detrended.to_netcdf(out_path)
    print(f"  Saved detrended CGI to {out_path}")
    
    ds.close()
    
    return detrended, regression, effective_gwl, effective_year, specified_by_year


def detrend_sst(
    start_year: int = None,
    end_year: int = None,
    target_year: float = None,
    target_gwl: float = None,
    save_regression: bool = True,
    force_recalculate: bool = False,
) -> tuple:
    """
    Detrend SST maps using Theil-Sen regression vs GWL.
    
    Note: SST detrending is optional and not used in the main simulation.
    
    Parameters
    ----------
    start_year : int, optional
        Start year for data (default: config.START_YEAR)
    end_year : int, optional
        End year for data (default: config.END_YEAR)
    target_year : float, optional
        Target year for detrending (converted to GWL)
    target_gwl : float, optional
        Target GWL for detrending (default: config.DEFAULT_TARGET_GWL)
    save_regression : bool
        Whether to save regression coefficients
    force_recalculate : bool
        If True, recalculate regression even if it exists
        
    Returns
    -------
    tuple
        (detrended_data, regression_coefficients, effective_gwl, effective_year, specified_by_year)
    """
    print("\nDetrending SST...")
    
    # Set defaults
    if start_year is None:
        start_year = config.START_YEAR
    if end_year is None:
        end_year = config.END_YEAR
    
    # Resolve target
    effective_gwl, effective_year, specified_by_year = resolve_target(
        target_year=target_year,
        target_gwl=target_gwl,
        default_gwl=config.DEFAULT_TARGET_GWL
    )
    
    print(f"  Target: {format_target_label(effective_gwl, effective_year, specified_by_year)}")
    
    # Load GWL data
    gwl_df = load_gwl_data()
    
    # Load SST data
    print(f"  Loading SST data for {start_year}-{end_year}...")
    ds = xr.open_dataset(config.ERA5_SST_MSLP_FILE)
    
    # Filter to season months and year range
    sst = ds['sst'].sel(
        valid_time=ds['valid_time'].dt.month.isin(config.SEASON_MONTHS)
    )
    sst = sst.sel(
        valid_time=(sst['valid_time'].dt.year >= start_year) & 
                   (sst['valid_time'].dt.year <= end_year)
    )
    
    # Rename time dimension for consistency
    sst = sst.rename({'valid_time': 'time'})
    
    # Regression file (independent of target - coefficients are constant)
    reg_filename = 'SST_regression_vs_GWL.nc'
    reg_path = config.get_detrended_path(reg_filename)
    
    if reg_path.exists() and not force_recalculate:
        print(f"  Loading existing regression from {reg_path}")
        regression = xr.open_dataset(reg_path)
    else:
        # Calculate regression
        print("  Fitting Theil-Sen regression (SST vs GWL)...")
        regression = calculate_theilsen_regression_gwl(sst, gwl_df, time_dim='time')
        
        # Save regression if requested
        if save_regression:
            regression.to_netcdf(reg_path)
            print(f"  Saved regression to {reg_path}")
    
    # Apply detrending to target GWL
    print(f"  Applying detrending to GWL {effective_gwl:.2f}°C...")
    detrended = apply_gwl_detrending(sst, regression, effective_gwl, gwl_df, time_dim='time')
    
    # Save detrended data with target in filename
    filename_suffix = get_filename_suffix(effective_gwl, effective_year, specified_by_year)
    filename = f'ERA5_SST_detrended_{filename_suffix}.nc'
    
    out_path = config.get_detrended_path(filename)
    detrended.to_netcdf(out_path)
    print(f"  Saved detrended SST to {out_path}")
    
    ds.close()
    
    return detrended, regression, effective_gwl, effective_year, specified_by_year


def calculate_mdr_timeseries(
    data: xr.DataArray,
    time_dim: str = 'time',
    months: list = None,
    aggregation: str = 'mean'
) -> pd.DataFrame:
    """
    Calculate annual MDR timeseries from gridded data.
    
    Parameters
    ----------
    data : xr.DataArray
        Gridded data with time, latitude, longitude dimensions
    time_dim : str
        Name of the time dimension
    months : list, optional
        List of months to aggregate over. If None, uses ASO_MONTHS from config.
    aggregation : str
        How to aggregate across months: 'mean' or 'sum'.
        Use 'sum' for CGI (seasonal accumulated) and 'mean' for SST/PI.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'year' and 'value' columns
    """
    if months is None:
        months = config.ASO_MONTHS
    
    # Select MDR region
    mdr_data = data.sel(
        latitude=slice(config.MDR_LAT_MAX, config.MDR_LAT_MIN),
        longitude=slice(config.MDR_LON_MIN, config.MDR_LON_MAX)
    )
    
    # Filter to specified months
    month_data = mdr_data.sel(
        **{time_dim: mdr_data[time_dim].dt.month.isin(months)}
    )
    
    # Spatial mean (always average over space)
    spatial_mean = month_data.mean(dim=['latitude', 'longitude'])
    
    # Annual aggregation: mean or sum across months
    if aggregation == 'sum':
        annual = spatial_mean.groupby(f'{time_dim}.year').sum()
    else:
        annual = spatial_mean.groupby(f'{time_dim}.year').mean()
    
    return pd.DataFrame({
        'year': annual['year'].values,
        'value': annual.values
    })
