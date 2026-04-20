"""
GWL (Global Warming Level) lookup utilities.

This module provides functions to convert between years and GWL values
using the preprocessed GWL_annual.csv data.
"""

import numpy as np
import pandas as pd
from scipy import stats

import config

# Cache for GWL data to avoid repeated file reads
_gwl_cache = None
_gwl_regression_cache = None


def load_gwl_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Load GWL annual data from preprocessed file.
    
    Parameters
    ----------
    force_reload : bool
        Force reload from file even if cached
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Year, GWL, Slope_per_day, R_squared columns
    """
    global _gwl_cache
    
    if _gwl_cache is None or force_reload:
        gwl_file = config.GWL_FILE
        if not gwl_file.exists():
            raise FileNotFoundError(
                f"GWL file not found: {gwl_file}\n"
                "Run preprocessing first: python run_preprocessing.py"
            )
        _gwl_cache = pd.read_csv(gwl_file)
    
    return _gwl_cache.copy()


def get_gwl_regression(force_reload: bool = False) -> dict:
    """
    Get linear regression coefficients for GWL ~ Year.
    
    Uses Theil-Sen regression for robustness.
    
    Parameters
    ----------
    force_reload : bool
        Force recalculation even if cached
        
    Returns
    -------
    dict
        Dictionary with slope, intercept, lo_slope, up_slope
    """
    global _gwl_regression_cache
    
    if _gwl_regression_cache is None or force_reload:
        gwl_df = load_gwl_data(force_reload)
        result = stats.theilslopes(gwl_df['GWL'].values, gwl_df['Year'].values)
        _gwl_regression_cache = {
            'slope': result.slope,
            'intercept': result.intercept,
            'lo_slope': result.low_slope,
            'up_slope': result.high_slope,
        }
    
    return _gwl_regression_cache.copy()


def year_to_gwl(year: float, use_lookup: bool = True) -> float:
    """
    Convert a year to GWL.
    
    For historical years (in GWL_annual.csv), uses direct lookup.
    For future years, uses linear extrapolation from regression.
    
    Parameters
    ----------
    year : float
        Year to convert
    use_lookup : bool
        If True, use direct lookup for historical years.
        If False, always use regression.
        
    Returns
    -------
    float
        GWL in °C above pre-industrial
    """
    gwl_df = load_gwl_data()
    
    if use_lookup and year in gwl_df['Year'].values:
        # Direct lookup for historical years
        return float(gwl_df.loc[gwl_df['Year'] == year, 'GWL'].values[0])
    else:
        # Linear extrapolation for future years (or if lookup disabled)
        regression = get_gwl_regression()
        return regression['slope'] * year + regression['intercept']


def years_to_gwl(start_year: int, end_year: int) -> float:
    """
    Calculate mean GWL over a year range.
    
    Parameters
    ----------
    start_year : int
        First year of range
    end_year : int
        Last year of range (inclusive)
        
    Returns
    -------
    float
        Mean GWL over the year range
    """
    # Get GWL for each year in range
    gwl_values = []
    for year in range(start_year, end_year + 1):
        gwl_values.append(year_to_gwl(year))
    
    return np.mean(gwl_values)


def gwl_to_year(target_gwl: float) -> float:
    """
    Convert a GWL to the corresponding year using regression.
    
    GWL = slope * year + intercept
    => year = (GWL - intercept) / slope
    
    Parameters
    ----------
    target_gwl : float
        Target GWL in °C above pre-industrial
        
    Returns
    -------
    float
        Corresponding year (may be fractional or in future)
    """
    regression = get_gwl_regression()
    year = (target_gwl - regression['intercept']) / regression['slope']
    return year


def get_historical_gwl_range() -> tuple:
    """
    Get the range of GWL values in the historical record.
    
    Returns
    -------
    tuple
        (min_gwl, max_gwl, min_year, max_year)
    """
    gwl_df = load_gwl_data()
    return (
        gwl_df['GWL'].min(),
        gwl_df['GWL'].max(),
        int(gwl_df['Year'].min()),
        int(gwl_df['Year'].max()),
    )


def validate_target_gwl(target_gwl: float, allow_extrapolation: bool = True) -> bool:
    """
    Validate that a target GWL is reasonable.
    
    Parameters
    ----------
    target_gwl : float
        GWL to validate
    allow_extrapolation : bool
        If True, allow GWL values outside historical range
        
    Returns
    -------
    bool
        True if valid
        
    Raises
    ------
    ValueError
        If target_gwl is invalid
    """
    min_gwl, max_gwl, min_year, max_year = get_historical_gwl_range()
    
    if target_gwl < 0:
        raise ValueError(f"Target GWL must be >= 0, got {target_gwl}")
    
    if not allow_extrapolation:
        if target_gwl < min_gwl or target_gwl > max_gwl:
            raise ValueError(
                f"Target GWL {target_gwl} outside historical range "
                f"[{min_gwl:.2f}, {max_gwl:.2f}] ({min_year}-{max_year})"
            )
    
    return True


def resolve_target(
    target_year: float = None,
    target_gwl: float = None,
    default_gwl: float = None
) -> tuple:
    """
    Resolve target specification to both GWL and year.
    
    Returns the effective GWL and year, with a flag indicating
    whether the original specification was by year or GWL.
    
    Parameters
    ----------
    target_year : float, optional
        Target specified as year
    target_gwl : float, optional
        Target specified as GWL
    default_gwl : float, optional
        Default GWL if neither specified (uses config.DEFAULT_TARGET_GWL if None)
        
    Returns
    -------
    tuple
        (effective_gwl, effective_year, specified_by_year)
        specified_by_year is True if target_year was specified
    """
    if target_year is not None and target_gwl is not None:
        raise ValueError("Cannot specify both target_year and target_gwl")
    
    if target_year is not None:
        # Year specified -> lookup GWL
        effective_gwl = year_to_gwl(target_year)
        effective_year = target_year
        specified_by_year = True
    elif target_gwl is not None:
        # GWL specified -> calculate year
        effective_gwl = target_gwl
        effective_year = gwl_to_year(target_gwl)
        specified_by_year = False
    else:
        # Use default GWL
        if default_gwl is None:
            default_gwl = config.DEFAULT_TARGET_GWL
        effective_gwl = default_gwl
        effective_year = gwl_to_year(default_gwl)
        specified_by_year = False
    
    return effective_gwl, effective_year, specified_by_year


def format_target_label(
    target_gwl: float,
    target_year: float,
    specified_by_year: bool
) -> str:
    """
    Format a human-readable label for the target.
    
    Parameters
    ----------
    target_gwl : float
        Target GWL
    target_year : float
        Target year
    specified_by_year : bool
        Whether originally specified as year
        
    Returns
    -------
    str
        Human-readable label like "GWL 1.41°C (~2025)" or "Year 2050 (GWL ~2.0°C)"
    """
    if specified_by_year:
        return f"Year {int(target_year)} (GWL ~{target_gwl:.2f}°C)"
    else:
        return f"GWL {target_gwl:.2f}°C (~{int(round(target_year))})"


def get_filename_suffix(
    target_gwl: float,
    target_year: float,
    specified_by_year: bool
) -> str:
    """
    Get a filename suffix for the target.
    
    Parameters
    ----------
    target_gwl : float
        Target GWL
    target_year : float
        Target year
    specified_by_year : bool
        Whether originally specified as year
        
    Returns
    -------
    str
        Filename suffix like "GWL1.41" or "to_2050"
    """
    if specified_by_year:
        return f"to_{int(target_year)}"
    else:
        return f"GWL{target_gwl:.2f}"
