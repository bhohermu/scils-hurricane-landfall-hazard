"""
ERA5 data processing module.

Calculates SST, wind shear, potential intensity, and CGI from ERA5 data.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

import config


def load_era5_sst_mslp(filepath=None):
    """Load ERA5 SST and MSLP data."""
    if filepath is None:
        filepath = config.ERA5_SST_MSLP_FILE
    return xr.open_dataset(filepath)


def load_era5_wind(filepath=None):
    """Load ERA5 wind data."""
    if filepath is None:
        filepath = config.ERA5_WIND_FILE
    return xr.open_dataset(filepath)


def load_era5_temperature(filepath=None):
    """Load ERA5 temperature data."""
    if filepath is None:
        filepath = config.ERA5_TEMP_FILE
    return xr.open_dataset(filepath)


def load_era5_humidity(filepath=None):
    """Load ERA5 humidity data."""
    if filepath is None:
        filepath = config.ERA5_HUMIDITY_FILE
    return xr.open_dataset(filepath)


def get_time_coord_name(ds):
    """Get the time coordinate name from a dataset."""
    for name in ['time', 'valid_time']:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError("Could not find time coordinate in dataset")


def get_level_coord_name(ds):
    """Get the pressure level coordinate name from a dataset."""
    for name in ['level', 'pressure_level', 'isobaricInhPa']:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError("Could not find pressure level coordinate in dataset")


def check_output_exists(output_path, force_recalculate=False):
    """
    Check if output file already exists.
    
    Parameters
    ----------
    output_path : str or Path
        Path to the output file.
    force_recalculate : bool
        If True, always recalculate even if file exists.
    
    Returns
    -------
    bool
        True if file exists and should be skipped.
    """
    if force_recalculate:
        return False
    
    output_path = Path(output_path)
    if output_path.exists():
        print(f"  Output already exists: {output_path}")
        print("  Skipping calculation (use force_recalculate=True to override)")
        return True
    return False


def calculate_aso_mdr_sst(start_year=None, end_year=None, output_path=None,
                          mdr_lat_min=None, mdr_lat_max=None,
                          mdr_lon_min=None, mdr_lon_max=None,
                          force_recalculate=False):
    """
    Calculate average MDR SST over August-September-October for each year.
    
    Parameters
    ----------
    start_year : int, optional
        First year to process.
    end_year : int, optional
        Last year to process.
    output_path : str or Path, optional
        Path to save output CSV.
    mdr_lat_min, mdr_lat_max : float, optional
        MDR latitude bounds.
    mdr_lon_min, mdr_lon_max : float, optional
        MDR longitude bounds.
    force_recalculate : bool, optional
        If True, recalculate even if output exists.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with Year and ASO_MDR_SST columns.
    """
    if start_year is None:
        start_year = config.START_YEAR
    if end_year is None:
        end_year = config.END_YEAR
    if output_path is None:
        output_path = config.get_output_path(config.ASO_MDR_SST_FILE)
    if mdr_lat_min is None:
        mdr_lat_min = config.MDR_LAT_MIN
    if mdr_lat_max is None:
        mdr_lat_max = config.MDR_LAT_MAX
    if mdr_lon_min is None:
        mdr_lon_min = config.MDR_LON_MIN
    if mdr_lon_max is None:
        mdr_lon_max = config.MDR_LON_MAX
    
    # Check if output already exists
    if check_output_exists(output_path, force_recalculate):
        return pd.read_csv(output_path)
    
    print(f"Calculating ASO MDR SST for years {start_year}-{end_year}...")
    print(f"MDR bounds: {mdr_lat_min}°N-{mdr_lat_max}°N, {mdr_lon_min}°E-{mdr_lon_max}°E")
    
    # Load SST data
    ds = load_era5_sst_mslp()
    
    # Get coordinate names
    time_name = get_time_coord_name(ds)
    
    # Get SST variable name (could be 'sst' or 'sst_skin' or similar)
    sst_var = None
    for var in ['sst', 'sst_skin', 'sea_surface_temperature']:
        if var in ds:
            sst_var = var
            break
    
    if sst_var is None:
        # Check variable names
        print(f"Available variables: {list(ds.data_vars)}")
        raise ValueError("Could not find SST variable in dataset")
    
    sst = ds[sst_var]
    
    # Determine coordinate names
    lat_name = 'latitude' if 'latitude' in sst.dims else 'lat'
    lon_name = 'longitude' if 'longitude' in sst.dims else 'lon'
    
    # Select MDR region
    # Handle longitude convention (could be 0-360 or -180-180)
    if float(sst[lon_name].min()) >= 0:  # 0-360 convention
        lon_min = mdr_lon_min % 360
        lon_max = mdr_lon_max % 360
        if lon_min > lon_max:
            # Need to handle wrap-around
            sst_mdr = sst.where(
                ((sst[lon_name] >= lon_min) | (sst[lon_name] <= lon_max)) &
                (sst[lat_name] >= mdr_lat_min) & (sst[lat_name] <= mdr_lat_max),
                drop=True
            )
        else:
            sst_mdr = sst.sel(
                **{lat_name: slice(mdr_lat_max, mdr_lat_min),
                   lon_name: slice(lon_min, lon_max)}
            )
    else:  # -180-180 convention
        sst_mdr = sst.sel(
            **{lat_name: slice(mdr_lat_max, mdr_lat_min),
               lon_name: slice(mdr_lon_min, mdr_lon_max)}
        )
    
    results = []
    
    for year in range(start_year, end_year + 1):
        # Select ASO months for this year
        time_mask = (
            (sst_mdr[time_name].dt.year == year) &
            (sst_mdr[time_name].dt.month.isin(config.ASO_MONTHS))
        )
        sst_aso = sst_mdr.where(time_mask, drop=True)
        
        if len(sst_aso[time_name]) == 0:
            print(f"  Warning: No data for {year}")
            continue
        
        # Calculate spatial and temporal mean
        # Weight by cos(latitude) for proper area averaging
        weights = np.cos(np.deg2rad(sst_aso[lat_name]))
        sst_weighted = sst_aso.weighted(weights)
        mean_sst = float(sst_weighted.mean(dim=[lat_name, lon_name, time_name]).values)
        
        # Convert from Kelvin to Celsius if needed
        if mean_sst > 100:  # Likely in Kelvin
            mean_sst = mean_sst - 273.15
        
        results.append({
            'Year': year,
            'ASO_MDR_SST': float(mean_sst)
        })
        print(f"  {year}: {mean_sst:.2f}°C")
    
    ds.close()
    
    # Create DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    
    # Calculate mean for warm/cold classification
    mean_sst_all = result_df['ASO_MDR_SST'].mean()
    print(f"\nMean ASO MDR SST (for warm/cold classification): {mean_sst_all:.2f}°C")
    print(f"ASO MDR SST saved to: {output_path}")
    
    return result_df


def calculate_wind_shear(start_year=None, end_year=None, output_path=None,
                         force_recalculate=False):
    """
    Calculate wind shear between 850 and 200 hPa levels.
    
    Wind shear S = sqrt((U200 - U850)^2 + (V200 - V850)^2)
    
    Parameters
    ----------
    start_year : int, optional
        First year to process.
    end_year : int, optional
        Last year to process.
    output_path : str or Path, optional
        Path to save output NetCDF.
    force_recalculate : bool, optional
        If True, recalculate even if output exists.
    
    Returns
    -------
    xarray.Dataset
        Dataset with wind shear.
    """
    if start_year is None:
        start_year = config.START_YEAR
    if end_year is None:
        end_year = config.END_YEAR
    if output_path is None:
        output_path = config.get_output_path(config.WIND_SHEAR_FILE)
    
    # Check if output already exists
    if check_output_exists(output_path, force_recalculate):
        return xr.open_dataset(output_path)
    
    print(f"Calculating wind shear for years {start_year}-{end_year}...")
    
    # Load wind data
    ds = load_era5_wind()
    
    # Get coordinate names
    time_name = get_time_coord_name(ds)
    level_name = get_level_coord_name(ds)
    
    # Determine variable names
    u_var = 'u' if 'u' in ds else 'U'
    v_var = 'v' if 'v' in ds else 'V'
    
    # Filter by years
    time_mask = (
        (ds[time_name].dt.year >= start_year) &
        (ds[time_name].dt.year <= end_year)
    )
    ds_filtered = ds.where(time_mask, drop=True)
    
    # Get U and V at 850 and 200 hPa
    u = ds_filtered[u_var]
    v = ds_filtered[v_var]
    
    u_850 = u.sel(**{level_name: 850})
    u_200 = u.sel(**{level_name: 200})
    v_850 = v.sel(**{level_name: 850})
    v_200 = v.sel(**{level_name: 200})
    
    # Calculate wind shear magnitude
    shear = np.sqrt((u_200 - u_850)**2 + (v_200 - v_850)**2)
    
    # Create output dataset
    out_ds = xr.Dataset({
        'wind_shear': shear
    })
    out_ds['wind_shear'].attrs['standard_name'] = 'wind_shear_850_200hPa'
    out_ds['wind_shear'].attrs['units'] = 'm/s'
    out_ds['wind_shear'].attrs['long_name'] = 'Wind shear magnitude between 850 and 200 hPa'
    
    # Rename time coordinate to 'time' for consistency
    if time_name != 'time':
        out_ds = out_ds.rename({time_name: 'time'})
    
    # Save to NetCDF
    print(f"Saving wind shear to: {output_path}")
    with ProgressBar():
        out_ds.to_netcdf(output_path)
    
    ds.close()
    
    print("Wind shear calculation complete.")
    return out_ds


def calculate_potential_intensity(start_year=None, end_year=None, output_path=None,
                                   force_recalculate=False):
    """
    Calculate Potential Intensity (PI) using tcpyPI.
    
    Parameters
    ----------
    start_year : int, optional
        First year to process.
    end_year : int, optional
        Last year to process.
    output_path : str or Path, optional
        Path to save output NetCDF.
    force_recalculate : bool, optional
        If True, recalculate even if output exists.
    
    Returns
    -------
    xarray.Dataset
        Dataset with PI values.
    """
    from tcpyPI import pi
    
    if start_year is None:
        start_year = config.START_YEAR
    if end_year is None:
        end_year = config.END_YEAR
    if output_path is None:
        output_path = config.get_output_path(config.PI_FILE)
    
    # Check if output already exists
    if check_output_exists(output_path, force_recalculate):
        return xr.open_dataset(output_path)
    
    print(f"Calculating Potential Intensity for years {start_year}-{end_year}...")
    
    # Load data - load first, then rechunk to avoid chunk boundary warnings
    sst_ds = load_era5_sst_mslp()
    t_ds = xr.open_dataset(config.ERA5_TEMP_FILE).chunk({'valid_time': 3})
    q_ds = xr.open_dataset(config.ERA5_HUMIDITY_FILE).chunk({'valid_time': 3})
    
    # Get coordinate names
    time_name = get_time_coord_name(sst_ds)
    level_name = get_level_coord_name(t_ds)
    
    # Determine variable names
    sst_var = 'sst' if 'sst' in sst_ds else list(sst_ds.data_vars)[0]
    msl_var = 'msl' if 'msl' in sst_ds else 'mslp' if 'mslp' in sst_ds else 'sp'
    lsm_var = 'lsm' if 'lsm' in sst_ds else None
    
    # Filter by years
    time_mask = (
        (sst_ds[time_name].dt.year >= start_year) &
        (sst_ds[time_name].dt.year <= end_year)
    )
    
    sst = sst_ds[sst_var].where(time_mask, drop=True)
    mslp = sst_ds[msl_var].where(time_mask, drop=True)
    
    if lsm_var:
        lsm = sst_ds[lsm_var]
    else:
        lsm = None
    
    time_mask_t = (
        (t_ds[time_name].dt.year >= start_year) &
        (t_ds[time_name].dt.year <= end_year)
    )
    
    t = t_ds['t'].where(time_mask_t, drop=True)
    q = q_ds['q'].where(time_mask_t, drop=True)
    
    # Convert units
    # SST: Kelvin to Celsius
    sst = sst - 273.15
    # Temperature: Kelvin to Celsius
    t = t - 273.15
    # MSLP: Pa to hPa
    mslp = mslp / 100
    # Specific humidity: kg/kg to g/kg
    q = q * 1000
    
    # Apply land mask if available
    if lsm is not None:
        # Broadcast lsm to match sst time dimension
        lsm_broadcast = lsm.isel(**{time_name: 0})
        sst = sst.where((sst > 5) & (lsm_broadcast == 0))
        mslp = mslp.where((mslp > 0) & (lsm_broadcast == 0))
        t = t.where(lsm_broadcast == 0)
        q = q.where((q > 0) & (lsm_broadcast == 0))
    
    # Sort by pressure level (descending for tcpyPI)
    ds_combined = xr.Dataset({
        'sst': sst,
        'msl': mslp,
        't': t,
        'q': q,
    })
    ds_combined = ds_combined.sortby(level_name, ascending=False)
    
    # Rechunk to have single chunk along pressure level dimension (required for apply_ufunc)
    ds_combined = ds_combined.chunk({level_name: -1})
    
    # Calculate PI using xarray apply_ufunc
    print("  Running PI calculation (this may take a while)...")
    
    result = xr.apply_ufunc(
        pi,
        ds_combined['sst'], 
        ds_combined['msl'], 
        ds_combined[level_name], 
        ds_combined['t'], 
        ds_combined['q'],
        kwargs=dict(
            CKCD=config.PI_CKCD, 
            ascent_flag=config.PI_ASCENT_FLAG, 
            diss_flag=config.PI_DISS_FLAG, 
            V_reduc=config.PI_V_REDUC, 
            ptop=config.PI_PTOP, 
            miss_handle=1
        ),
        input_core_dims=[[], [], [level_name], [level_name], [level_name]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float, float, float, float],
    )
    
    gc.collect()
    
    # Store result
    vmax, pmin, ifl, t0, otl = result
    
    out_ds = xr.Dataset({
        'vmax': vmax,
        'pmin': pmin,
        'ifl': ifl,
        't0': t0,
        'otl': otl,
    })
    
    # Rename time coordinate to 'time' for consistency
    if time_name != 'time':
        out_ds = out_ds.rename({time_name: 'time'})
    
    # Add attributes
    out_ds['vmax'].attrs = {'standard_name': 'Maximum Potential Intensity', 'units': 'm/s'}
    out_ds['pmin'].attrs = {'standard_name': 'Minimum Central Pressure', 'units': 'hPa'}
    out_ds['ifl'].attrs = {'standard_name': 'pyPI Flag'}
    out_ds['t0'].attrs = {'standard_name': 'Outflow Temperature', 'units': 'K'}
    out_ds['otl'].attrs = {'standard_name': 'Outflow Temperature Level', 'units': 'hPa'}
    
    # Save to NetCDF
    print(f"  Saving PI to: {output_path}")
    with ProgressBar():
        out_ds.to_netcdf(output_path)
    
    sst_ds.close()
    t_ds.close()
    q_ds.close()
    
    print("  PI calculation complete.")
    return out_ds


def calculate_cgi(start_year=None, end_year=None, 
                  pi_file=None, shear_file=None,
                  output_map_path=None, output_table_path=None,
                  mdr_lat_min=None, mdr_lat_max=None,
                  mdr_lon_min=None, mdr_lon_max=None,
                  force_recalculate=False):
    """
    Calculate Cyclone Genesis Index (CGI).
    
    CGI = (PI/70)^3 * (1 + 0.1*S)^-2
    
    Scaled to match TS+ storm counts from IBTrACS.
    
    Parameters
    ----------
    start_year, end_year : int, optional
        Years to process.
    pi_file : str or Path, optional
        Path to PI NetCDF file.
    shear_file : str or Path, optional
        Path to wind shear NetCDF file.
    output_map_path : str or Path, optional
        Path to save CGI map NetCDF.
    output_table_path : str or Path, optional
        Path to save scaled MDR CGI table.
    mdr_* : float, optional
        MDR bounds.
    force_recalculate : bool, optional
        If True, recalculate even if output exists.
    
    Returns
    -------
    tuple (xarray.Dataset, pandas.DataFrame)
        CGI map and annual MDR CGI table.
    """
    if start_year is None:
        start_year = config.START_YEAR
    if end_year is None:
        end_year = config.END_YEAR
    if pi_file is None:
        pi_file = config.get_output_path(config.PI_FILE)
    if shear_file is None:
        shear_file = config.get_output_path(config.WIND_SHEAR_FILE)
    if output_map_path is None:
        output_map_path = config.get_output_path(config.CGI_MAP_FILE)
    if output_table_path is None:
        output_table_path = config.get_output_path(config.CGI_MDR_FILE)
    if mdr_lat_min is None:
        mdr_lat_min = config.MDR_LAT_MIN
    if mdr_lat_max is None:
        mdr_lat_max = config.MDR_LAT_MAX
    if mdr_lon_min is None:
        mdr_lon_min = config.MDR_LON_MIN
    if mdr_lon_max is None:
        mdr_lon_max = config.MDR_LON_MAX
    
    # Check if outputs already exist
    map_exists = check_output_exists(output_map_path, force_recalculate)
    table_exists = check_output_exists(output_table_path, force_recalculate)
    
    if map_exists and table_exists:
        return xr.open_dataset(output_map_path), pd.read_csv(output_table_path)
    
    # If map exists but table doesn't, load map and just regenerate table
    if map_exists and not table_exists:
        print("CGI map exists, regenerating MDR table only...")
        out_ds = xr.open_dataset(output_map_path)
        cgi = out_ds['cgi']
    else:
        # Full calculation needed
        print(f"Calculating CGI for years {start_year}-{end_year}...")
        
        # Load PI and wind shear
        pi_ds = xr.open_dataset(pi_file)
        shear_ds = xr.open_dataset(shear_file)
        
        pi = pi_ds['vmax']/config.PI_V_REDUC  # Adjusted PI
        shear = shear_ds['wind_shear']
        
        # Calculate CGI
        # CGI = (PI/70)^3 * (1 + 0.1*S)^-2
        cgi = (pi / config.CGI_PI_REFERENCE)**3 * (1 + 0.1 * shear)**(-2)
        
        # Create output dataset
        out_ds = xr.Dataset({
            'cgi': cgi
        })
        out_ds['cgi'].attrs = {
            'standard_name': 'Cyclone Genesis Index',
            'units': '1',
            'long_name': 'CGI = (PI/70)^3 * (1 + 0.1*S)^-2'
        }
        
        # Save CGI map
        print(f"  Saving CGI map to: {output_map_path}")
        out_ds.to_netcdf(output_map_path)
        
        pi_ds.close()
        shear_ds.close()
    
    # Calculate MDR sum for June-November each year (seasonal accumulated CGI)
    # We SUM monthly values rather than averaging them to represent total genesis potential
    lat_name = 'latitude' if 'latitude' in cgi.dims else 'lat'
    lon_name = 'longitude' if 'longitude' in cgi.dims else 'lon'
    
    # Select MDR region
    if cgi[lon_name].min() >= 0:  # 0-360 convention
        lon_min = mdr_lon_min % 360
        lon_max = mdr_lon_max % 360
    else:
        lon_min = mdr_lon_min
        lon_max = mdr_lon_max
    
    cgi_mdr = cgi.sel(
        **{lat_name: slice(mdr_lat_max, mdr_lat_min),
           lon_name: slice(lon_min, lon_max)}
    )
    
    # Calculate seasonal SUM of monthly MDR CGI for hurricane season
    results = []
    for year in range(start_year, end_year + 1):
        time_mask = (
            (cgi_mdr['time'].dt.year == year) &
            (cgi_mdr['time'].dt.month.isin(config.SEASON_MONTHS))
        )
        cgi_season = cgi_mdr.where(time_mask, drop=True)
        
        if len(cgi_season.time) == 0:
            continue
        
        # For each month: calculate area-weighted spatial mean
        # Then SUM across months to get seasonal accumulated CGI
        weights = np.cos(np.deg2rad(cgi_season[lat_name]))
        cgi_weighted = cgi_season.weighted(weights)
        
        # First get spatial mean for each month, then sum across months
        monthly_spatial_means = cgi_weighted.mean(dim=[lat_name, lon_name])
        seasonal_sum_cgi = float(monthly_spatial_means.sum(dim='time').values)
        
        results.append({
            'Year': year,
            'MDR_CGI_Sum': seasonal_sum_cgi
        })
    
    result_df = pd.DataFrame(results)
    
    # Get observed storm counts from IBTrACS properties if available
    # The counts are already filtered to TS+ in preprocessing
    ibtracs_file = config.get_output_path(config.IBTRACS_PROPERTIES_FILE)
    if Path(ibtracs_file).exists():
        ibtracs_df = pd.read_csv(ibtracs_file)
        storm_counts = ibtracs_df.groupby('Year').size().reset_index(name='Observed_Count')
        result_df = result_df.merge(storm_counts, on='Year', how='left')
        
        # Scale CGI sum to match observed counts
        total_observed = result_df['Observed_Count'].sum()
        total_cgi_sum = result_df['MDR_CGI_Sum'].sum()
        scale_factor = total_observed / total_cgi_sum if total_cgi_sum > 0 else 1
        
        result_df['Scaled_MDR_CGI'] = result_df['MDR_CGI_Sum'] * scale_factor
        print(f"  Scale factor (observed TS+/sum): {scale_factor:.4f}")
        print(f"  Total observed storms (TS+): {total_observed}")
    else:
        result_df['Scaled_MDR_CGI'] = result_df['MDR_CGI_Sum']
        print("  Warning: IBTrACS properties not found, CGI not scaled to observations")
    
    # Save table
    result_df.to_csv(output_table_path, index=False)
    print(f"  MDR CGI saved to: {output_table_path}")
    
    return out_ds, result_df


if __name__ == "__main__":
    # Test the module
    calculate_aso_mdr_sst()
    calculate_wind_shear()
    calculate_potential_intensity()
    calculate_cgi()
