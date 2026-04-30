"""Configuration and filesystem layout for the SCILS TC Model."""

import os
from pathlib import Path

import numpy as np

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.resolve()
RAW_DATA_ENV_VAR = "SCILS_TC_RAW_DATA_DIR"

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = Path(os.environ.get(RAW_DATA_ENV_VAR, BASE_DIR / "external-data")).expanduser().resolve()
ERA5_DIR = RAW_DATA_DIR / "ERA5"
PREPROCESSED_DIR = BASE_DIR / "preprocessed"
SHAPEFILE_DIR = DATA_DIR / "tl_2023_us_state"

# Input files
IBTRACS_FILE = RAW_DATA_DIR / "IBTrACS.NA.v04r01.nc"
RONI_FILE = DATA_DIR / "RONI.csv"
REGION_DEFINITIONS_FILE = DATA_DIR / "regionDefinitionsJewson.csv"
US_STATES_SHAPEFILE = SHAPEFILE_DIR / "tl_2023_us_state.shp"
YELT_FILE = DATA_DIR / "YELT_STORM_present_NA_10000_YEARS.csv"

# ERA5 input files
ERA5_SST_MSLP_FILE = ERA5_DIR / "ERA5_monthly_1980_2025_SST_MSLP_lsm.nc"
ERA5_WIND_FILE = ERA5_DIR / "ERA5_monthly_1980_2025_U_V_850_200hPa.nc"
ERA5_TEMP_FILE = ERA5_DIR / "ERA5_monthly_1980_2025_t_1000to50hPa.nc"
ERA5_HUMIDITY_FILE = ERA5_DIR / "ERA5_monthly_1980_2025_q_1000to50hPa.nc"

# =============================================================================
# TIME CONFIGURATION
# =============================================================================

START_YEAR = 1980
END_YEAR = 2025

CLIMATOLOGY_START_YEAR = 1991
CLIMATOLOGY_END_YEAR = 2020

SEASON_MONTHS = [6, 7, 8, 9, 10, 11]
ENSO_MONTHS = [8, 9, 10]
ASO_MONTHS = [8, 9, 10]

# =============================================================================
# SPATIAL CONFIGURATION
# =============================================================================

MDR_LAT_MIN = 10.0
MDR_LAT_MAX = 20.0
MDR_LON_MIN = -60.0
MDR_LON_MAX = -15.0

NA_LAT_MIN = 0.0
NA_LAT_MAX = 65.0
NA_LON_MIN = -100.0
NA_LON_MAX = 10.0

ATLANTIC_BASIN_POLYGON = [
    (-99, 19), (-99, 32), (-83, 32), (-72, 50), (-10, 50),
    (-10, 0), (-75, 0), (-75, 7.5), (-77.6, 7.5), (-77.6, 8.5),
    (-79.1, 9.4), (-81, 8.5), (-82, 8.5), (-85, 11), (-85, 13),
    (-99, 19),
]

# =============================================================================
# THRESHOLD CONFIGURATION
# =============================================================================

ENSO_ELNINO_THRESHOLD = 0.5
ENSO_LANINA_THRESHOLD = -0.5

TROPICAL_STORM_THRESHOLD_KTS = 34.0
TROPICAL_STORM_THRESHOLD_MS = 17.4912

SSHS_THRESHOLDS_KTS = {
    'TD': (0, 33), 'TS': (34, 63), 'Cat1': (64, 82),
    'Cat2': (83, 95), 'Cat3': (96, 112), 'Cat4': (113, 136),
    'Cat5': (137, float('inf'))
}

KTS_TO_MS = 0.514444
SSHS_THRESHOLDS_MS = {
    cat: (low * KTS_TO_MS, high * KTS_TO_MS) 
    for cat, (low, high) in SSHS_THRESHOLDS_KTS.items()
}

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

KDE_BANDWIDTH = 'scott'
KDE_GRID_RESOLUTION = 0.25

LFI_LMI_DISCRETIZATION_POINTS = 200

CGI_PI_REFERENCE = 70.0

PI_CKCD = 0.9
PI_ASCENT_FLAG = 0
PI_DISS_FLAG = 1
PI_V_REDUC = 1.0
PI_PTOP = 50

# =============================================================================
# DETRENDING CONFIGURATION
# =============================================================================

DETRENDED_DIR = BASE_DIR / "detrended"
DETRENDED_DIAGNOSTICS_DIR = DETRENDED_DIR / "diagnostics"

# GWL data file (from preprocessing, using ERA5-based calculation)
GWL_FILE = PREPROCESSED_DIR / "GWL_annual.csv"

# GWL fit period (ERA5 period)
GWL_FIT_START_YEAR = 1980
GWL_FIT_END_YEAR = 2025

DEFAULT_START_YEAR = 1980
DEFAULT_END_YEAR = 2025

# Default GWL targets for detrending
# Base GWL 0.82 = mean GWL over 1980-2025 (historical)
# Target GWL 1.41 = GWL at 2025
DEFAULT_BASE_GWL = 0.82
DEFAULT_TARGET_GWL = 1.41

# Legacy: target year (deprecated, use GWL instead)
DEFAULT_TARGET_YEAR = 2025

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Output file names
ENSO_STATE_FILE = "ENSO_state.csv"
IBTRACS_PROPERTIES_FILE = "IBTrACS_properties.csv"
ASO_MDR_SST_FILE = "ASO_MDR_SST.csv"
WIND_SHEAR_FILE = "ERA5_wind_shear.nc"
PI_FILE = "ERA5_PI.nc"
CGI_MAP_FILE = "ERA5_CGI.nc"
CGI_MDR_FILE = "CGI_MDR_annual.csv"

# LMI/PI ratio histogram (single pooled distribution, paper Figure 1a)
LMI_PI_RATIO_FILE = "LMI_PI_ratio_histogram_single.csv"

# LFI/LMI Beta distribution parameters
LFI_LMI_BETA_PARAMS_FILE = "LFI_LMI_NorthAtlantic_beta_params.csv"
LFI_LMI_FILE = "LFI_LMI_NorthAtlantic.csv"

# LMI KDE files (one per ENSO group)
LMI_KDE_GROUPS = ["lanina", "neutral", "elnino"]

# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

SIMULATED_DIR = BASE_DIR / "simulated"
RESAMPLED_DIR = BASE_DIR / "resampled"

N_ITER = 100
RANDOM_SEED = 42

USE_REGION = 'NorthAtlantic'

SIMULATION_OUTPUT_FILE = "simulated_events_{target}.csv"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_preprocessed_dir():
    """Create the preprocessed directory if it doesn't exist."""
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PREPROCESSED_DIR

def ensure_raw_data_dir():
    """Create the external raw-data directory if it does not exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR

def ensure_detrended_dir():
    """Create the detrended directory if it doesn't exist."""
    DETRENDED_DIR.mkdir(parents=True, exist_ok=True)
    return DETRENDED_DIR

def ensure_detrended_diagnostics_dir():
    """Create the detrended diagnostics directory if it doesn't exist."""
    DETRENDED_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    return DETRENDED_DIAGNOSTICS_DIR

def get_output_path(filename):
    """Get the full path for an output file in the preprocessed directory."""
    ensure_preprocessed_dir()
    return PREPROCESSED_DIR / filename

def get_detrended_path(filename):
    """Get the full path for an output file in the detrended directory."""
    ensure_detrended_dir()
    return DETRENDED_DIR / filename

def get_detrended_diagnostics_path(filename):
    """Get the full path for a diagnostic file in the detrended/diagnostics directory."""
    ensure_detrended_diagnostics_dir()
    return DETRENDED_DIAGNOSTICS_DIR / filename

def ensure_simulated_dir():
    """Create the simulated directory if it doesn't exist."""
    SIMULATED_DIR.mkdir(parents=True, exist_ok=True)
    return SIMULATED_DIR

def get_simulated_path(filename):
    """Get the full path for an output file in the simulated directory."""
    ensure_simulated_dir()
    return SIMULATED_DIR / filename

def ensure_resampled_dir():
    """Create the resampled directory if it doesn't exist."""
    RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    return RESAMPLED_DIR

def get_resampled_path(filename):
    """Get the full path for an output file in the resampled directory."""
    ensure_resampled_dir()
    return RESAMPLED_DIR / filename

def get_sshs_category(wind_speed_ms):
    """Get Saffir-Simpson Hurricane Wind Scale category for wind speed in m/s."""
    if np.isnan(wind_speed_ms):
        return 'Unknown'
    
    for cat, (low, high) in SSHS_THRESHOLDS_MS.items():
        if low <= wind_speed_ms < high:
            return cat
    return 'Cat5'
