"""
Sampling functions for SCILS TC Model simulation.

This module provides functions to sample from distributions used in TC simulation:
- LMI location from KDE maps
- LMI/PI ratio from single histogram
- LFI/LMI ratio from 0-1 inflated Beta distribution
- PI lookup from gridded data
"""

import numpy as np
import pandas as pd
import xarray as xr


def load_lmi_kde(kde_file):
    """
    Load a KDE map for LMI location sampling.
    
    Parameters
    ----------
    kde_file : str or Path
        Path to the KDE NetCDF file.
    
    Returns
    -------
    xr.Dataset
        Dataset containing 'kde' variable on lat/lon grid.
    """
    return xr.open_dataset(kde_file)


def sample_lmi_location(kde_ds, rng=None, n_samples=1):
    """
    Sample LMI location(s) from a KDE map.
    
    Uses the KDE as a probability density and samples discrete grid cells.
    
    Parameters
    ----------
    kde_ds : xr.Dataset
        KDE dataset with 'kde' variable on lat/lon grid.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    n_samples : int
        Number of locations to sample.
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Arrays of (longitude, latitude) for sampled locations.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Get KDE values and coordinates
    kde = kde_ds['kde'].values
    lats = kde_ds['lat'].values
    lons = kde_ds['lon'].values
    
    # Handle NaN values - set to 0
    kde = np.nan_to_num(kde, nan=0.0)
    
    # Normalize to probability distribution
    kde_flat = kde.flatten()
    total = kde_flat.sum()
    
    if total <= 0:
        raise ValueError("KDE has no valid probability mass")
    
    probs = kde_flat / total
    
    # Sample indices
    indices = rng.choice(len(probs), size=n_samples, p=probs)
    
    # Convert flat indices to 2D
    lat_idx, lon_idx = np.unravel_index(indices, kde.shape)
    
    # Get coordinates directly from bins (no random offset)
    sampled_lats = lats[lat_idx]
    sampled_lons = lons[lon_idx]
    
    return sampled_lons, sampled_lats


def load_lmi_pi_ratio_distribution(ratio_file):
    """
    Load the LMI/PI ratio distribution (single histogram).
    
    Parameters
    ----------
    ratio_file : str or Path
        Path to the LMI_PI_ratio_histogram.csv file.
    
    Returns
    -------
    dict
        Dictionary with 'bins', 'probs' keys.
    """
    # Load single histogram
    ratio_file_single = str(ratio_file).replace('.csv', '_single.csv')
    df_single = pd.read_csv(ratio_file_single)
    bins = df_single['bin_center'].values
    probs = df_single['density'].values
    probs = probs / probs.sum()
    
    return {
        'bins': bins,
        'probs': probs
    }


def sample_lmi_pi_ratio(lmi_pi_dist, pi_value, rng=None, n_samples=1):
    """
    Sample LMI/PI ratio from the single histogram distribution.
    
    Parameters
    ----------
    lmi_pi_dist : dict
        Distribution parameters from load_lmi_pi_ratio_distribution().
    pi_value : float
        PI value in m/s (not used, kept for API consistency).
    rng : np.random.Generator, optional
        Random number generator.
    n_samples : int
        Number of samples to draw.
    
    Returns
    -------
    np.ndarray
        Sampled LMI/PI ratio values.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    bins = lmi_pi_dist['bins']
    probs = lmi_pi_dist['probs']
    
    indices = rng.choice(len(bins), size=n_samples, p=probs)
    return bins[indices]


def load_lfi_lmi_distributions(lfi_lmi_file):
    """
    Load LFI/LMI 0-1 inflated Beta distribution parameters.
    
    Parameters
    ----------
    lfi_lmi_file : str or Path
        Path to the main LFI_LMI_{region}.csv file.
    
    Returns
    -------
    dict
        Dictionary mapping region_number -> distribution dict with:
        - p0, p1: point mass probabilities at 0 and 1
        - alpha, beta_param: Beta distribution parameters
        - p_continuous: probability of continuous part
    """
    from pathlib import Path
    
    lfi_lmi_path = Path(lfi_lmi_file)
    beta_params_path = str(lfi_lmi_path).replace('.csv', '_beta_params.csv')
    beta_params_df = pd.read_csv(beta_params_path)
    
    distributions = {}
    for _, row in beta_params_df.iterrows():
        region_num = int(row['RegionNumber'])
        distributions[region_num] = {
            'p0': float(row['p0']),
            'p1': float(row['p1']),
            'alpha': float(row['alpha']),
            'beta_param': float(row['beta']),
            'p_continuous': float(row['p_continuous'])
        }
    
    return distributions


def sample_lfi_lmi_ratio(params, rng=None, n_samples=1):
    """
    Sample LFI/LMI ratio from 0-1 inflated Beta distribution.
    
    Samples from point masses (0 or 1) with probability p0/p1,
    otherwise samples from the continuous Beta distribution.
    
    Parameters
    ----------
    params : dict
        Distribution dict with 'p0', 'p1', 'alpha', 'beta_param'.
    rng : np.random.Generator, optional
        Random number generator.
    n_samples : int
        Number of samples to draw.
    
    Returns
    -------
    np.ndarray
        Sampled LFI/LMI ratio values in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    
    p0 = params['p0']
    p1 = params['p1']
    alpha = params['alpha']
    beta_param = params['beta_param']
    
    samples = np.zeros(n_samples)
    for i in range(n_samples):
        u = rng.random()
        if u < p0:
            samples[i] = 0.0
        elif u < p0 + p1:
            samples[i] = 1.0
        else:
            samples[i] = rng.beta(alpha, beta_param)
    
    return samples


def load_pi_data(pi_file):
    """
    Load PI (Potential Intensity) data.
    
    Parameters
    ----------
    pi_file : str or Path
        Path to the detrended PI NetCDF file or historical PI file.
    
    Returns
    -------
    xr.Dataset
        PI dataset with time, latitude, longitude dimensions.
    """
    ds = xr.open_dataset(pi_file)
    
    # Handle different variable names
    # Historical file: 'vmax'
    # Detrended file: 'pi' or '__xarray_dataarray_variable__'
    if 'vmax' in ds.data_vars:
        ds = ds.rename({'vmax': 'pi'})
    elif '__xarray_dataarray_variable__' in ds.data_vars:
        ds = ds.rename({'__xarray_dataarray_variable__': 'pi'})
    
    return ds


def get_pi_at_location(pi_ds, lon, lat, month, year=None):
    """
    Get PI value at a specific location and month.
    
    Parameters
    ----------
    pi_ds : xr.Dataset
        PI dataset with time, latitude, longitude dimensions.
    lon : float
        Longitude.
    lat : float
        Latitude.
    month : int
        Month (1-12).
    year : int, optional
        Year. If None, uses the first matching month.
    
    Returns
    -------
    float
        PI value at the location. Returns NaN if out of bounds.
    """
    # Handle coordinate name differences
    lat_dim = 'latitude' if 'latitude' in pi_ds.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in pi_ds.dims else 'lon'
    
    # Select the appropriate time
    times = pd.to_datetime(pi_ds['time'].values)
    
    if year is not None:
        # Find exact year-month match
        mask = (times.month == month) & (times.year == year)
        if mask.any():
            time_idx = np.where(mask)[0][0]
        else:
            # If year not found, use climatological mean for that month
            month_mask = times.month == month
            if not month_mask.any():
                return np.nan
            # Use mean over all years for that month
            pi_month = pi_ds['pi'].isel(time=np.where(month_mask)[0]).mean(dim='time')
            # Interpolate to location
            try:
                pi_val = pi_month.interp({lat_dim: lat, lon_dim: lon}, method='nearest').values
                return float(pi_val)
            except:
                return np.nan
    else:
        # Just match month
        mask = times.month == month
        if not mask.any():
            return np.nan
        time_idx = np.where(mask)[0][0]
    
    # Select time slice
    pi_slice = pi_ds['pi'].isel(time=time_idx)
    
    # Interpolate to exact location (using nearest neighbor)
    try:
        pi_val = pi_slice.interp({lat_dim: lat, lon_dim: lon}, method='nearest').values
        return float(pi_val)
    except:
        return np.nan


def load_cgi_data(cgi_file):
    """
    Load CGI (Cyclone Genesis Index) data.
    
    Parameters
    ----------
    cgi_file : str or Path
        Path to the detrended CGI NetCDF file or CGI_MDR_annual.csv.
    
    Returns
    -------
    xr.Dataset or pd.DataFrame
        CGI data.
    """
    cgi_file = str(cgi_file)
    if cgi_file.endswith('.csv'):
        return pd.read_csv(cgi_file)
    else:
        ds = xr.open_dataset(cgi_file)
        if '__xarray_dataarray_variable__' in ds.data_vars:
            ds = ds.rename({'__xarray_dataarray_variable__': 'cgi'})
        return ds


def load_sst_data(sst_file):
    """
    Load SST data for warm/cold classification.
    
    Parameters
    ----------
    sst_file : str or Path
        Path to the ASO_MDR_SST.csv or detrended SST NetCDF.
    
    Returns
    -------
    pd.DataFrame or xr.Dataset
        SST data.
    """
    sst_file = str(sst_file)
    if sst_file.endswith('.csv'):
        return pd.read_csv(sst_file)
    else:
        ds = xr.open_dataset(sst_file)
        if '__xarray_dataarray_variable__' in ds.data_vars:
            ds = ds.rename({'__xarray_dataarray_variable__': 'sst'})
        return ds


def get_mdr_sst_from_detrended(sst_ds, year):
    """
    Calculate MDR SST (ASO average) from detrended SST NetCDF.
    
    Parameters
    ----------
    sst_ds : xr.Dataset
        Detrended SST dataset.
    year : int
        Year to get SST for.
    
    Returns
    -------
    float
        ASO mean MDR SST.
    """
    import config
    
    # Handle coordinate names
    lat_dim = 'latitude' if 'latitude' in sst_ds.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in sst_ds.dims else 'lon'
    
    # Select MDR region
    sst_mdr = sst_ds['sst'].sel(
        **{lat_dim: slice(config.MDR_LAT_MAX, config.MDR_LAT_MIN),
           lon_dim: slice(config.MDR_LON_MIN, config.MDR_LON_MAX)}
    )
    
    # Select ASO months for the given year
    times = pd.to_datetime(sst_ds['time'].values)
    aso_mask = (times.year == year) & (times.month.isin(config.ASO_MONTHS))
    
    if not aso_mask.any():
        # If year not in data, return NaN
        return np.nan
    
    sst_aso = sst_mdr.isel(time=np.where(aso_mask)[0])
    
    # Return mean, converting from Kelvin to Celsius
    sst_kelvin = float(sst_aso.mean().values)
    sst_celsius = sst_kelvin - 273.15
    return sst_celsius


def get_mdr_cgi_from_maps(cgi_ds, year):
    """
    Calculate MDR CGI seasonal sum (June-November) from CGI NetCDF maps.
    
    Sums the monthly spatial-mean CGI values to get total seasonal genesis potential.
    
    Parameters
    ----------
    cgi_ds : xr.Dataset
        CGI dataset (historical or detrended).
    year : int
        Year to get CGI for.
    
    Returns
    -------
    float
        June-November sum of monthly MDR CGI (raw, not scaled).
    """
    import config
    
    # Handle coordinate names
    lat_dim = 'latitude' if 'latitude' in cgi_ds.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in cgi_ds.dims else 'lon'
    
    # Get CGI variable name
    if 'cgi' in cgi_ds.data_vars:
        cgi_var = 'cgi'
    elif '__xarray_dataarray_variable__' in cgi_ds.data_vars:
        cgi_var = '__xarray_dataarray_variable__'
    else:
        cgi_var = list(cgi_ds.data_vars)[0]
    
    # Select MDR region
    cgi_mdr = cgi_ds[cgi_var].sel(
        **{lat_dim: slice(config.MDR_LAT_MAX, config.MDR_LAT_MIN),
           lon_dim: slice(config.MDR_LON_MIN, config.MDR_LON_MAX)}
    )
    
    # Select hurricane season months (June-November) for the given year
    times = pd.to_datetime(cgi_ds['time'].values)
    season_mask = (times.year == year) & (times.month.isin(config.SEASON_MONTHS))
    
    if not season_mask.any():
        # If year not in data, return NaN
        return np.nan
    
    cgi_season = cgi_mdr.isel(time=np.where(season_mask)[0])
    
    # Apply latitude weighting for spatial average per month
    weights = np.cos(np.deg2rad(cgi_season[lat_dim]))
    cgi_weighted = cgi_season.weighted(weights)
    
    # Get spatial mean for each month, then SUM across months
    monthly_spatial_means = cgi_weighted.mean(dim=[lat_dim, lon_dim])
    seasonal_sum = float(monthly_spatial_means.sum(dim='time').values)
    
    return seasonal_sum


def classify_sst_warm_cold(mdr_sst, climatology_mean):
    """
    Classify SST as warm or cold relative to climatology.
    
    Parameters
    ----------
    mdr_sst : float
        MDR SST value.
    climatology_mean : float
        Climatological mean MDR SST.
    
    Returns
    -------
    str
        'warm' or 'cold'.
    """
    return 'warm' if mdr_sst >= climatology_mean else 'cold'


def get_monthly_mdr_cgi(cgi_ds, year):
    """
    Get monthly MDR CGI values for a given year from detrended CGI data.
    
    Parameters
    ----------
    cgi_ds : xr.Dataset
        Detrended CGI dataset with time, latitude, longitude dimensions.
    year : int
        Year to extract monthly values for.
    
    Returns
    -------
    dict
        Dictionary mapping month (6-11) -> MDR mean CGI value.
    """
    import config
    
    # Handle coordinate names
    lat_dim = 'latitude' if 'latitude' in cgi_ds.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in cgi_ds.dims else 'lon'
    
    # Handle variable name
    var_name = 'cgi' if 'cgi' in cgi_ds.data_vars else '__xarray_dataarray_variable__'
    
    # Select MDR region
    cgi_mdr = cgi_ds[var_name].sel(
        **{lat_dim: slice(config.MDR_LAT_MAX, config.MDR_LAT_MIN),
           lon_dim: slice(config.MDR_LON_MIN, config.MDR_LON_MAX)}
    )
    
    # Get times
    times = pd.to_datetime(cgi_ds['time'].values)
    
    # Extract monthly values for the given year
    monthly_cgi = {}
    for month in config.SEASON_MONTHS:  # June-November
        mask = (times.year == year) & (times.month == month)
        if mask.any():
            cgi_month = cgi_mdr.isel(time=np.where(mask)[0])
            monthly_cgi[month] = float(cgi_month.mean().values)
        else:
            # If year not in data, use climatological mean for that month
            month_mask = times.month == month
            if month_mask.any():
                cgi_month = cgi_mdr.isel(time=np.where(month_mask)[0]).mean(dim='time')
                monthly_cgi[month] = float(cgi_month.mean().values)
            else:
                monthly_cgi[month] = 0.0  # Fallback
    
    return monthly_cgi
