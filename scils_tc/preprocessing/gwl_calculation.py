"""
Global Warming Level (GWL) calculation using ERA5 2m temperature data.

Calculates GWL following the C3S methodology:
1. Compute monthly global average temperature anomalies against 1991-2020 climatology
2. Apply monthly offsets to convert to pre-industrial (1850-1900) reference
3. Calculate 30-year linear regression endpoint for each year

Data sources (pre-downloaded in data/GWL):
- ERA5 monthly 2m temperature (1951-2025)
- 1991-2020 climatology (monthly grib files)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

import config

# Monthly offsets to convert 1991-2020 anomalies to pre-industrial (1850-1900)
# Source: C3S Climate Bulletin documentation
MONTHLY_OFFSETS = [
    0.96, 0.96, 0.95, 0.91, 0.87, 0.83,
    0.80, 0.80, 0.81, 0.85, 0.89, 0.93
]

# Data paths
ERA5_FILE = Path("data/GWL/era5_2mTemperature_1951_2025.nc")
CLIMATOLOGY_DIR = Path("data/GWL/era5_2mTemperature_anomalies_1991_2020")

# Regression window
TREND_YEARS = 30


def load_era5_temperature():
    """Load ERA5 monthly 2m temperature data with automatic chunking for memory efficiency."""
    return xr.open_dataset(ERA5_FILE, chunks='auto')


def load_climatology():
    """Load 1991-2020 monthly climatology from GRIB files."""
    datasets = []
    for month in range(1, 13):
        grib_file = CLIMATOLOGY_DIR / f"climatology_0.25g_ea_2t_{month:02d}_1991-2020_v02.grib"
        # Use chunks='auto' for memory efficiency
        ds = xr.open_dataset(grib_file, engine='cfgrib', chunks='auto')
        # Extract only t2m variable with lat/lon coordinates to avoid time conflicts
        t2m_clean = ds['t2m'].drop_vars(
            ['time', 'valid_time', 'step', 'number', 'surface'], 
            errors='ignore'
        )
        # Create clean dataset with only essential data
        ds_clean = xr.Dataset({'t2m': t2m_clean})
        ds_clean = ds_clean.expand_dims('month').assign_coords(month=[month])
        datasets.append(ds_clean)
    return xr.concat(datasets, dim='month')


def calculate_latitude_weights(lat):
    """Calculate area weights based on latitude (cosine weighting)."""
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.sum()
    return xr.DataArray(weights, dims=['latitude'], coords={'latitude': lat})


def calculate_global_anomaly(era5, climatology, start_year=1971):
    """
    Calculate global temperature anomaly against pre-industrial baseline (vectorized).
    
    Parameters
    ----------
    era5 : xr.Dataset
        ERA5 monthly temperature data with 'valid_time' dimension
    climatology : xr.Dataset
        1991-2020 monthly climatology with 'month' dimension
    start_year : int
        First year to process (need 30 years before first trend year)
    
    Returns
    -------
    pd.DataFrame
        Monthly global anomalies with columns: year, month, date, anomaly
    """
    # Filter ERA5 to start year onwards
    era5_filtered = era5.sel(valid_time=era5.valid_time.dt.year >= start_year)
    
    # Rename time dimension for groupby compatibility
    era5_filtered = era5_filtered.rename({'valid_time': 'time'})
    
    # Calculate anomalies vs 1991-2020 climatology (vectorized with groupby)
    # Broadcast climatology to match ERA5 months
    clim_broadcast = climatology.t2m.sel(month=era5_filtered.time.dt.month)
    anomaly_1991_2020 = era5_filtered.t2m - clim_broadcast
    
    # Create monthly offset DataArray for vectorized addition
    offsets_da = xr.DataArray(
        MONTHLY_OFFSETS, 
        dims=['month'], 
        coords={'month': range(1, 13)}
    )
    
    # Apply pre-industrial offset (vectorized)
    offsets_broadcast = offsets_da.sel(month=era5_filtered.time.dt.month)
    anomaly_preindustrial = anomaly_1991_2020 + offsets_broadcast
    
    # Calculate latitude weights for area-weighted mean
    weights = np.cos(np.deg2rad(era5_filtered.latitude))
    weights_da = xr.DataArray(
        weights, 
        dims=['latitude'], 
        coords={'latitude': era5_filtered.latitude}
    )
    
    # Calculate global spatial average using xarray weighted mean with dask
    # This processes chunks efficiently without loading full array into memory
    global_anomaly = anomaly_preindustrial.weighted(weights_da).mean(
        dim=['latitude', 'longitude']
    )
    
    # Convert to DataFrame with proper date handling
    df = global_anomaly.to_dataframe(name='anomaly').reset_index()
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['date'] = pd.to_datetime(df['time'].dt.strftime('%Y-%m-15'))
    
    return df[['year', 'month', 'date', 'anomaly']]


def calculate_gwl_from_regression(anomaly_df, target_year, window=TREND_YEARS):
    """
    Calculate GWL as endpoint of 30-year linear regression ending at December of target_year.
    
    Parameters
    ----------
    anomaly_df : pd.DataFrame
        Monthly anomalies with 'date' and 'anomaly' columns
    target_year : int
        Year to calculate GWL for (uses data up to December of this year)
    window : int
        Number of years in regression window
    
    Returns
    -------
    tuple
        (gwl, slope, r_squared)
    """
    start_year = target_year - window + 1
    
    # Filter to 30-year window ending at December of target year
    mask = (anomaly_df['year'] >= start_year) & (anomaly_df['year'] <= target_year)
    subset = anomaly_df[mask].copy()
    
    if len(subset) < window * 12 - 11:  # Allow some missing months
        raise ValueError(f"Insufficient data for {target_year}")
    
    # Convert dates to numeric for regression
    subset['date_numeric'] = (subset['date'] - subset['date'].min()).dt.days
    
    # Linear regression
    slope, intercept, r_value, _, _ = stats.linregress(
        subset['date_numeric'], subset['anomaly']
    )
    
    # GWL is the regression value at December 15 of target year
    end_date = pd.Timestamp(f"{target_year}-12-15")
    end_numeric = (end_date - subset['date'].min()).days
    gwl = slope * end_numeric + intercept
    
    return gwl, slope, r_value ** 2


def calculate_gwl(start_year=1980, end_year=2025, force_recalculate=False):
    """
    Calculate GWL for all years using 30-year regression endpoint method.
    
    Parameters
    ----------
    start_year : int
        First year to calculate GWL for
    end_year : int
        Last year to calculate GWL for
    force_recalculate : bool
        Recalculate even if output exists
    
    Returns
    -------
    pd.DataFrame
        DataFrame with Year and GWL columns
    """
    output_file = config.get_output_path("GWL_annual.csv")
    anomaly_cache_file = config.get_output_path("GWL_monthly_anomalies.csv")
    
    # Load or calculate anomaly data (with caching)
    anomaly_start = start_year - TREND_YEARS + 1
    
    if anomaly_cache_file.exists() and not force_recalculate:
        print(f"    Loading cached anomaly data from {anomaly_cache_file}")
        anomaly_df = pd.read_csv(anomaly_cache_file, parse_dates=['date'])
        # Filter to required years
        anomaly_df = anomaly_df[anomaly_df['year'] >= anomaly_start]
    else:
        print("    Loading ERA5 temperature data...")
        era5 = load_era5_temperature()
        
        print("    Loading 1991-2020 climatology...")
        climatology = load_climatology()
        
        print("    Calculating global temperature anomalies (vectorized)...")
        anomaly_df = calculate_global_anomaly(era5, climatology, start_year=anomaly_start)
        
        # Cache anomaly data
        anomaly_df.to_csv(anomaly_cache_file, index=False)
        print(f"    Cached anomalies: {anomaly_cache_file}")
    
    # Check if annual GWL already exists and anomaly cache is fresh
    if output_file.exists() and not force_recalculate:
        print(f"    Loading existing GWL data from {output_file}")
        result_df = pd.read_csv(output_file)
        result_df._anomaly_df = anomaly_df
        return result_df
    
    print(f"    Calculating 30-year GWL trends for {start_year}-{end_year}...")
    results = []
    for year in range(start_year, end_year + 1):
        try:
            gwl, slope, r_sq = calculate_gwl_from_regression(anomaly_df, year)
            results.append({
                'Year': year,
                'GWL': round(gwl, 4),
                'Slope_per_day': slope,
                'R_squared': round(r_sq, 4)
            })
        except ValueError as e:
            print(f"      Warning: {e}")
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"    Saved: {output_file}")
    
    # Store anomaly data for plotting
    result_df._anomaly_df = anomaly_df
    
    return result_df


def plot_gwl(gwl_df, anomaly_df=None, start_year=1980, end_year=2025):
    """
    Create GWL plot with monthly anomalies as scatter and 30-year trend as line.
    
    Parameters
    ----------
    gwl_df : pd.DataFrame
        GWL data with Year and GWL columns
    anomaly_df : pd.DataFrame, optional
        Monthly anomaly data (if None, will recalculate)
    start_year : int
        First year to plot
    end_year : int
        Last year to plot
    """
    output_file = config.get_output_path("GWL_plot.png")
    
    # Load anomaly data if not provided
    if anomaly_df is None:
        anomaly_cache_file = config.get_output_path("GWL_monthly_anomalies.csv")
        if anomaly_cache_file.exists():
            print("    Loading cached anomaly data for plot...")
            anomaly_df = pd.read_csv(anomaly_cache_file, parse_dates=['date'])
        else:
            print("    Loading data for plot...")
            era5 = load_era5_temperature()
            climatology = load_climatology()
            anomaly_df = calculate_global_anomaly(era5, climatology, start_year=start_year - TREND_YEARS + 1)
    
    # Filter to plot range
    plot_mask = (anomaly_df['year'] >= start_year) & (anomaly_df['year'] <= end_year)
    plot_anomalies = anomaly_df[plot_mask]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Monthly anomalies as transparent scatter
    ax.scatter(
        plot_anomalies['date'], 
        plot_anomalies['anomaly'],
        alpha=0.3, 
        color='darkorange', 
        s=20,
        label='Monthly Global Surface Air Temperature (GSAT) Anomaly',
        zorder=1
    )
    
    # GWL trend as solid line (plotted at December of each year)
    gwl_dates = [pd.Timestamp(f"{year}-12-15") for year in gwl_df['Year']]
    ax.plot(
        gwl_dates, 
        gwl_df['GWL'],
        color='darkblue', 
        linewidth=2.5,
        label='Global Warming Level (GWL)',
        zorder=2
    )

    # --- Add linear regression line for GWL ---
    x = gwl_df['Year'].values
    y = gwl_df['GWL'].values

    # Fit regression
    slope, intercept = np.polyfit(x, y, 1)

    # Calculate regression values for actual years
    y_reg = slope * x + intercept

    ax.plot(
        gwl_dates,
        y_reg,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Linear Trend (GWL)',
        zorder=1.5
    )
    
    # Reference lines
    # ax.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='1.5°C threshold')
    # ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Temperature anomaly (1850-1900) (°C)', fontsize=12)
    ax.set_title('Global Warming Level derived from ERA5', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_file}")
    plt.close()


def process_gwl(start_year=1980, end_year=2025, force_recalculate=False):
    """
    Main entry point for GWL preprocessing.
    
    Parameters
    ----------
    start_year : int
        First year to calculate GWL
    end_year : int
        Last year to calculate GWL
    force_recalculate : bool
        Force recalculation even if outputs exist
    """
    print("  Calculating Global Warming Level (GWL)...")
    
    # Calculate GWL
    gwl_df = calculate_gwl(start_year, end_year, force_recalculate)
    
    # Get anomaly data for plotting (stored on dataframe if freshly calculated)
    anomaly_df = getattr(gwl_df, '_anomaly_df', None)
    
    # Generate plot
    print("  Generating GWL plot...")
    plot_gwl(gwl_df, anomaly_df, start_year, end_year)
    
    return gwl_df
