"""
IBTrACS storm properties extraction module.

Extracts LMI, LFI, and other properties for tropical cyclones.
"""

import numpy as np
import pandas as pd
import xarray as xr

import config
from scils_tc.utils.regions import get_region_for_point, load_jewson_regions
from scils_tc.utils.saffir_simpson import get_sshs_category, is_tropical_storm_or_above, kts_to_ms
from scils_tc.utils.spatial import get_state_for_point, is_conus_state, load_us_states_shapefile

# Valid tropical status codes (exclude ET=Extratropical, EX=Extratropical, etc.)
VALID_TROPICAL_STATUS = {'TD', 'TS', 'HU', 'TY', 'ST', 'TC', 'SS', 'SD'}


def load_ibtracs(filepath=None):
    """
    Load IBTrACS NetCDF file.
    
    Parameters
    ----------
    filepath : str or Path, optional
        Path to IBTrACS NetCDF file.
    
    Returns
    -------
    xarray.Dataset
        IBTrACS dataset.
    """
    if filepath is None:
        filepath = config.IBTRACS_FILE
    
    ds = xr.open_dataset(filepath)
    return ds


def decode_string(data):
    """
    Decode byte strings from NetCDF to regular strings.
    
    Parameters
    ----------
    data : bytes or array of bytes
        Data to decode.
    
    Returns
    -------
    str
        Decoded string.
    """
    if isinstance(data, bytes):
        return data.decode('utf-8').strip()
    elif hasattr(data, 'values'):
        # xarray DataArray
        val = data.values
        if isinstance(val, bytes):
            return val.decode('utf-8').strip()
        elif isinstance(val, np.ndarray):
            return b''.join(val).decode('utf-8').strip()
    return str(data).strip()


def get_storm_year(ds, storm_idx):
    """
    Extract the year of a storm from its season or time data.
    
    Parameters
    ----------
    ds : xarray.Dataset
        IBTrACS dataset.
    storm_idx : int
        Storm index.
    
    Returns
    -------
    int
        Year of the storm.
    """
    # Try to get season first
    if 'season' in ds:
        season = ds['season'].isel(storm=storm_idx).values
        if not np.isnan(season):
            return int(season)
    
    # Fall back to time
    time = ds['time'].isel(storm=storm_idx, date_time=0).values
    if pd.notna(time):
        return pd.Timestamp(time).year
    
    return np.nan


def get_storm_name(ds, storm_idx):
    """
    Extract the name of a storm.
    
    Parameters
    ----------
    ds : xarray.Dataset
        IBTrACS dataset.
    storm_idx : int
        Storm index.
    
    Returns
    -------
    str
        Storm name.
    """
    name_data = ds['name'].isel(storm=storm_idx).values
    name = decode_string(name_data)
    # Remove any remaining byte string markers
    name = name.replace("b'", "").replace("'", "").strip()
    return name


def find_lifetime_maximum_intensity(ds, storm_idx):
    """
    Find the lifetime maximum intensity (LMI) for a storm.
    
    Only considers tropical storm (TS) or hurricane (HU) status observations.
    
    Parameters
    ----------
    ds : xarray.Dataset
        IBTrACS dataset.
    storm_idx : int
        Storm index.
    
    Returns
    -------
    dict
        Dictionary with LMI information:
        - lmi_ms: Maximum wind speed in m/s
        - lmi_sshs: SSHS category at LMI
        - lmi_lat: Latitude at LMI
        - lmi_lon: Longitude at LMI
        - lmi_time_idx: Time index of LMI
    """
    # Get wind data - use usa_wind as default, wmo_wind as fallback per timestep
    usa_wind = None
    wmo_wind = None
    
    if 'usa_wind' in ds:
        usa_wind = ds['usa_wind'].isel(storm=storm_idx).values
    
    if 'wmo_wind' in ds:
        wmo_wind = ds['wmo_wind'].isel(storm=storm_idx).values
    
    if usa_wind is None and wmo_wind is None:
        return None
    
    # Create combined wind array: prefer usa_wind, fallback to wmo_wind
    if usa_wind is not None and wmo_wind is not None:
        wind = np.where(np.isnan(usa_wind), wmo_wind, usa_wind)
    elif usa_wind is not None:
        wind = usa_wind
    else:
        wind = wmo_wind
    
    if np.all(np.isnan(wind)):
        return None
    
    # Get pressure data for tie-breaking
    usa_pres = None
    if 'usa_pres' in ds:
        usa_pres = ds['usa_pres'].isel(storm=storm_idx).values
    
    # Check USA_STATUS if available (filter for TS/HU status only)
    usa_status_valid = None
    if 'usa_status' in ds:
        usa_status = ds['usa_status'].isel(storm=storm_idx).values
        # Only TS or HU (exclude TD, ET, EX, etc.)
        valid_statuses = {b'TS', b'HU', 'TS', 'HU'}
        usa_status_valid = np.array([
            (s in valid_statuses if isinstance(s, bytes) else s in valid_statuses)
            for s in usa_status
        ])
    
    # Apply status mask to wind
    valid_wind = wind.copy()
    if usa_status_valid is not None:
        valid_wind[~usa_status_valid] = np.nan
    
    if np.all(np.isnan(valid_wind)):
        return None
    
    # Find maximum wind value
    max_wind = np.nanmax(valid_wind)
    
    # Find all indices where wind equals maximum (potential ties)
    max_indices = np.where(valid_wind == max_wind)[0]
    
    if len(max_indices) == 1:
        # No tie - use the single maximum
        lmi_idx = max_indices[0]
    else:
        # Tie in wind speed - use pressure as tie breaker (lowest pressure wins)
        if usa_pres is not None:
            # Get pressure values at max wind locations
            pres_at_max = usa_pres[max_indices]
            
            # Find indices with valid (non-NaN) pressure
            valid_pres_mask = ~np.isnan(pres_at_max)
            
            if np.any(valid_pres_mask):
                # Among valid pressures, find the minimum
                min_pres_idx = np.nanargmin(pres_at_max)
                lmi_idx = max_indices[min_pres_idx]
            else:
                # No valid pressure data - use first instance
                lmi_idx = max_indices[0]
        else:
            # No pressure data available - use first instance
            lmi_idx = max_indices[0]
    
    lmi_kts = wind[lmi_idx]
    
    if np.isnan(lmi_kts):
        return None
    
    # Convert to m/s
    lmi_ms = kts_to_ms(lmi_kts)
    
    # Get location
    lat = ds['lat'].isel(storm=storm_idx, date_time=lmi_idx).values
    lon = ds['lon'].isel(storm=storm_idx, date_time=lmi_idx).values
    
    # Get time at LMI
    lmi_time = ds['time'].isel(storm=storm_idx, date_time=lmi_idx).values
    lmi_month = None
    if pd.notna(lmi_time):
        lmi_month = int(pd.Timestamp(lmi_time).month)
    
    return {
        'lmi_ms': float(lmi_ms),
        'lmi_sshs': get_sshs_category(lmi_ms),
        'lmi_lat': float(lat),
        'lmi_lon': float(lon),
        'lmi_time_idx': int(lmi_idx),
        'lmi_month': lmi_month
    }


def find_landfalls(ds, storm_idx, states_gdf):
    """
    Find all landfalls for a storm.
    
    Landfall is identified by:
    - landfall variable = 0 (meaning minimum distance to land between 
      current and next observation is 0, i.e., storm crosses coastline)
    
    Only considers landfalls where storm has TS or HU status.
    
    Parameters
    ----------
    ds : xarray.Dataset
        IBTrACS dataset.
    storm_idx : int
        Storm index.
    states_gdf : geopandas.GeoDataFrame
        US states shapefile.
    
    Returns
    -------
    list of dict
        List of landfall events with wind speed, location, and type (CONUS or other).
    """
    landfalls = []
    
    # Get landfall variable (distance to land between current and next obs)
    if 'landfall' in ds:
        landfall_dist = ds['landfall'].isel(storm=storm_idx).values
    else:
        return landfalls
    
    # Get wind data - prefer wmo_wind, but use usa_wind as fallback
    wmo_wind = None
    usa_wind = None
    if 'wmo_wind' in ds:
        wmo_wind = ds['wmo_wind'].isel(storm=storm_idx).values
    if 'usa_wind' in ds:
        usa_wind = ds['usa_wind'].isel(storm=storm_idx).values
    
    if wmo_wind is None and usa_wind is None:
        return landfalls
    
    lat = ds['lat'].isel(storm=storm_idx).values
    lon = ds['lon'].isel(storm=storm_idx).values
    
    # Get USA_STATUS if available
    usa_status = None
    if 'usa_status' in ds:
        usa_status = ds['usa_status'].isel(storm=storm_idx).values
    
    n_times = len(landfall_dist)
    
    for t in range(n_times):
        # Landfall occurs when landfall distance = 0
        # (storm crosses coastline between this obs and next)
        if landfall_dist[t] == 0:
            # Get the location and wind at this observation
            lf_lat = float(lat[t])
            lf_lon = float(lon[t])
            
            # Use wmo_wind if available, otherwise usa_wind
            lf_wind_kts = np.nan
            if wmo_wind is not None and not np.isnan(wmo_wind[t]):
                lf_wind_kts = wmo_wind[t]
            elif usa_wind is not None and not np.isnan(usa_wind[t]):
                lf_wind_kts = usa_wind[t]
            
            if np.isnan(lf_wind_kts) or np.isnan(lf_lat) or np.isnan(lf_lon):
                continue
            
            # Get state at landfall location (will be None for non-US landfalls)
            state = get_state_for_point(lf_lon, lf_lat, states_gdf)
            
            # Mark whether this is a CONUS landfall based on state
            is_conus = is_conus_state(state)
            
            # Check USA_STATUS - filter for TS/HU status
            # This is a storm classification, not a location check
            status = None
            if usa_status is not None:
                status = usa_status[t]
                if isinstance(status, bytes):
                    status = status.decode('utf-8').strip()
            
            # Require valid tropical status for all landfalls
            # Only TS or HU (exclude TD, ET, EX, etc.)
            if status not in ['TS', 'HU']:
                continue
            
            # Convert wind to m/s
            lf_wind_ms = kts_to_ms(lf_wind_kts)
            
            # Include all landfalls (CONUS and non-CONUS)
            landfalls.append({
                'time_idx': t,
                'wind_ms': float(lf_wind_ms),
                'wind_sshs': get_sshs_category(lf_wind_ms),
                'lat': lf_lat,
                'lon': lf_lon,
                'state': state,
                'is_conus': is_conus
            })
    
    return landfalls


def find_maximum_landfall_intensity(landfalls, region='CONUS'):
    """
    Find the landfall with maximum intensity from a list of landfalls.
    
    Parameters
    ----------
    landfalls : list of dict
        List of landfall events.
    region : str
        'CONUS' for US mainland only, 'NorthAtlantic' for all North Atlantic landfalls
    
    Returns
    -------
    dict or None
        Landfall with maximum intensity, or None if no landfalls.
    """
    if not landfalls:
        return None
    
    # Filter by region
    if region == 'CONUS':
        filtered = [lf for lf in landfalls if lf.get('is_conus', False)]
    else:  # NorthAtlantic - all landfalls
        filtered = landfalls
    
    if not filtered:
        return None
    
    max_lf = max(filtered, key=lambda x: x['wind_ms'])
    return max_lf


def process_ibtracs_properties(start_year=None, end_year=None, output_path=None):
    """
    Process IBTrACS data and extract storm properties.
    
    Only includes storms reaching tropical storm strength (TS+) or above.
    
    Parameters
    ----------
    start_year : int, optional
        First year to process.
    end_year : int, optional
        Last year to process.
    output_path : str or Path, optional
        Path to save output CSV.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with storm properties.
    """
    if start_year is None:
        start_year = config.START_YEAR
    if end_year is None:
        end_year = config.END_YEAR
    if output_path is None:
        output_path = config.get_output_path(config.IBTRACS_PROPERTIES_FILE)
    
    strength_label = "TS+ (tropical storm or above)"
    print(f"Processing IBTrACS data for years {start_year}-{end_year}...")
    print(f"  Minimum strength: {strength_label}")
    
    # Load data
    ds = load_ibtracs()
    regions = load_jewson_regions(config.REGION_DEFINITIONS_FILE)
    states_gdf = load_us_states_shapefile(config.US_STATES_SHAPEFILE)
    
    # Get number of storms
    n_storms = ds.sizes['storm']
    
    results = []
    
    for storm_idx in range(n_storms):
        # Get year
        year = get_storm_year(ds, storm_idx)
        
        if np.isnan(year) or year < start_year or year > end_year:
            continue
        
        # Get storm name
        name = get_storm_name(ds, storm_idx)
        
        # Get LMI
        lmi_info = find_lifetime_maximum_intensity(ds, storm_idx)
        
        if lmi_info is None:
            continue
        
        # Skip if not tropical storm strength or above
        if not is_tropical_storm_or_above(lmi_info['lmi_ms']):
            continue
        
        # Get Jewson region for LMI location
        region_name, region_number = get_region_for_point(
            lmi_info['lmi_lon'], lmi_info['lmi_lat'], regions
        )
        
        # Find landfalls
        landfalls = find_landfalls(ds, storm_idx, states_gdf)
        
        # Get maximum landfall intensity for both regions
        max_landfall_conus = find_maximum_landfall_intensity(landfalls, region='CONUS')
        max_landfall_na = find_maximum_landfall_intensity(landfalls, region='NorthAtlantic')
        
        # Build result row
        result = {
            'Year': int(year),
            'Name': name,
            'LMI_ms': lmi_info['lmi_ms'],
            'LMI_SSHS': lmi_info['lmi_sshs'],
            'LMI_Lat': lmi_info['lmi_lat'],
            'LMI_Lon': lmi_info['lmi_lon'],
            'LMI_month': lmi_info['lmi_month'],
            'RegionName': region_name,
            'RegionNumber': region_number
        }
        
        # Add CONUS landfall info
        if max_landfall_conus:
            result['CONUS_LFI_ms'] = max_landfall_conus['wind_ms']
            result['CONUS_LFI_SSHS'] = max_landfall_conus['wind_sshs']
            result['CONUS_LFI_State'] = max_landfall_conus['state']
        else:
            result['CONUS_LFI_ms'] = 0.0
            result['CONUS_LFI_SSHS'] = None
            result['CONUS_LFI_State'] = None
        
        # Add NorthAtlantic landfall info
        if max_landfall_na:
            result['NorthAtlantic_LFI_ms'] = max_landfall_na['wind_ms']
            result['NorthAtlantic_LFI_SSHS'] = max_landfall_na['wind_sshs']
            result['NorthAtlantic_LFI_State'] = max_landfall_na.get('state', None)
        else:
            result['NorthAtlantic_LFI_ms'] = 0.0
            result['NorthAtlantic_LFI_SSHS'] = None
            result['NorthAtlantic_LFI_State'] = None
        
        results.append(result)
    
    ds.close()
    
    # Create DataFrame
    result_df = pd.DataFrame(results)
    
    # Save to CSV (use na_rep to avoid confusion with 'NA' region name)
    result_df.to_csv(output_path, index=False, na_rep='')
    
    print(f"IBTrACS properties saved to: {output_path}")
    print(f"Total storms processed: {len(result_df)}")
    print("\nSummary by year:")
    print(result_df.groupby('Year').size())
    
    return result_df


if __name__ == "__main__":
    df = process_ibtracs_properties()
    print(df.head(20))
