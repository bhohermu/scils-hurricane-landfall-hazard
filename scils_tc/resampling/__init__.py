"""
Resampling module for SCILS TC Model.

This module implements the YELT adjustment workflow described in Section 2.3
of the manuscript. It has two main components:

1. **Landfall rate changes** (``resampling.py``): compares per-category
   (Saffir-Simpson TS through Cat 5) landfall rates between a base
   simulation (typically historical) and a target simulation (at a
   specified GWL), producing rate-change factors.

2. **YELT resampling** (``yelt_resampling.py``): applies the rate-change
   factors to a Year Event Loss Table using incremental simulation
   (Jewson, 2023b). Events in categories with increased rates are
   duplicated; events in categories with decreased rates are thinned.
   Exposure, vulnerability, and per-event loss are held constant.

Important limitation: resampling modifies only event frequencies. It cannot
introduce new events beyond the input YELT catalogue, which caps extreme
losses and underestimates tail risk at high return periods (paper
Sections 3.2 and 3.5).
"""

from .plotting import (
    plot_multi_region_comparison,
    plot_rate_comparison,
)
from .resampling import (
    SSHS_CATEGORIES,
    calculate_change_rates,
    calculate_landfall_rates,
    generate_output_filename,
    get_gwl_for_year,
    get_simulation_filename,
    get_year_for_gwl,
    load_simulation,
    run_simulation_if_needed,
    simulation_exists,
)
from .yelt_plotting import (
    plot_category_event_counts,
    plot_yelt_comparison,
)
from .yelt_resampling import (
    add_lfi_sshs_category,
    calculate_aal,
    calculate_loss_metrics,
    calculate_oep_curve,
    calculate_yelt_landfall_rates,
    clean_yelt_for_resampling,
    compare_yelts,
    load_change_rates,
    load_yelt,
    resample_yelt,
    resample_yelt_deterministic,
    resample_yelt_poisson,
)

__all__ = [
    'calculate_landfall_rates',
    'calculate_change_rates',
    'get_simulation_filename',
    'simulation_exists',
    'run_simulation_if_needed',
    'load_simulation',
    'get_gwl_for_year',
    'get_year_for_gwl',
    'generate_output_filename',
    'plot_rate_comparison',
    'plot_multi_region_comparison',
    'SSHS_CATEGORIES',
    # YELT resampling
    'load_yelt',
    'add_lfi_sshs_category',
    'load_change_rates',
    'clean_yelt_for_resampling',
    'resample_yelt',
    'resample_yelt_deterministic',
    'resample_yelt_poisson',
    'calculate_yelt_landfall_rates',
    'calculate_oep_curve',
    'calculate_aal',
    'calculate_loss_metrics',
    'compare_yelts',
    'plot_yelt_comparison',
    'plot_category_event_counts',
]
