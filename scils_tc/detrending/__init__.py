"""
Detrending module for SCILS TC Model.

This module implements the climate-state adjustment described in Section 2.1
of the manuscript ("Climate variables and detrending"). For each grid cell
and month, PI and CGI are regressed against observed GWL using Theil-Sen
regression. The fitted trend is then used to shift the data to a target GWL,
preserving interannual variability while adjusting the climate mean.

The detrended PI and CGI maps are the primary climate inputs to the
simulation stage. SST detrending is retained for diagnostics but is not
used in the TC classification pipeline.
"""

from .detrending import (
    apply_gwl_detrending,
    calculate_mdr_timeseries,
    calculate_theilsen_regression_gwl,
    detrend_cgi,
    detrend_pi,
    detrend_sst,
)
from .gwl_lookup import (
    format_target_label,
    get_filename_suffix,
    get_gwl_regression,
    get_historical_gwl_range,
    gwl_to_year,
    load_gwl_data,
    resolve_target,
    year_to_gwl,
    years_to_gwl,
)
from .plotting import (
    plot_detrending_comparison,
    plot_diagnostic_mean_trend,
    plot_gwl_regression,
)

__all__ = [
    # Detrending functions
    'calculate_theilsen_regression_gwl',
    'apply_gwl_detrending',
    'detrend_sst',
    'detrend_pi',
    'detrend_cgi',
    'calculate_mdr_timeseries',
    # GWL lookup functions
    'load_gwl_data',
    'get_gwl_regression',
    'year_to_gwl',
    'gwl_to_year',
    'years_to_gwl',
    'resolve_target',
    'format_target_label',
    'get_filename_suffix',
    'get_historical_gwl_range',
    # Plotting functions
    'plot_gwl_regression',
    'plot_detrending_comparison',
    'plot_diagnostic_mean_trend',
]
