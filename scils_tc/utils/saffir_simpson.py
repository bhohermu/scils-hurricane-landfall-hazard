"""
Saffir-Simpson Hurricane Wind Scale utilities.

Provides conversion between wind speeds and SSHS categories.
"""

import numpy as np

# Conversion factor
KTS_TO_MS = 0.514444
MS_TO_KTS = 1.0 / KTS_TO_MS

# SSHS category thresholds in knots
# TD: < 34 kts, TS: 34-63 kts, Cat1: 64-82 kts, etc.
SSHS_THRESHOLDS_KTS = [
    ('TD', 0, 33),
    ('TS', 34, 63),
    ('Cat1', 64, 82),
    ('Cat2', 83, 95),
    ('Cat3', 96, 112),
    ('Cat4', 113, 136),
    ('Cat5', 137, np.inf)
]

# Convert thresholds to m/s
SSHS_THRESHOLDS_MS = [
    (cat, low * KTS_TO_MS, high * KTS_TO_MS)
    for cat, low, high in SSHS_THRESHOLDS_KTS
]


def kts_to_ms(speed_kts):
    """
    Convert wind speed from knots to meters per second.
    
    Parameters
    ----------
    speed_kts : float or array-like
        Wind speed in knots.
    
    Returns
    -------
    float or array-like
        Wind speed in m/s.
    """
    return speed_kts * KTS_TO_MS


def ms_to_kts(speed_ms):
    """
    Convert wind speed from meters per second to knots.
    
    Parameters
    ----------
    speed_ms : float or array-like
        Wind speed in m/s.
    
    Returns
    -------
    float or array-like
        Wind speed in knots.
    """
    return speed_ms * MS_TO_KTS


def get_sshs_category(wind_speed_ms):
    """
    Get the Saffir-Simpson Hurricane Scale category for a given wind speed.
    
    Parameters
    ----------
    wind_speed_ms : float
        Maximum sustained wind speed in m/s.
    
    Returns
    -------
    str
        SSHS category: 'TD', 'TS', 'Cat1', 'Cat2', 'Cat3', 'Cat4', or 'Cat5'.
        Returns 'Unknown' if wind speed is NaN or negative.
    """
    if np.isnan(wind_speed_ms) or wind_speed_ms < 0:
        return 'Unknown'
    
    for cat, low_ms, high_ms in SSHS_THRESHOLDS_MS:
        if low_ms <= wind_speed_ms <= high_ms:
            return cat
    
    # Should not reach here, but return Cat5 for very high winds
    return 'Cat5'


def get_sshs_category_from_kts(wind_speed_kts):
    """
    Get the Saffir-Simpson Hurricane Scale category for a given wind speed in knots.
    
    Parameters
    ----------
    wind_speed_kts : float
        Maximum sustained wind speed in knots.
    
    Returns
    -------
    str
        SSHS category: 'TD', 'TS', 'Cat1', 'Cat2', 'Cat3', 'Cat4', or 'Cat5'.
    """
    return get_sshs_category(kts_to_ms(wind_speed_kts))


# Tropical Storm threshold: 34 knots in m/s
TS_THRESHOLD_MS = 34 * KTS_TO_MS


def is_tropical_storm_or_above(wind_speed_ms):
    """
    Check if wind speed qualifies as tropical storm strength or above.
    
    Parameters
    ----------
    wind_speed_ms : float
        Wind speed in m/s.
    
    Returns
    -------
    bool
        True if wind speed >= 34 knots (17.49 m/s).
    """
    return wind_speed_ms >= TS_THRESHOLD_MS
