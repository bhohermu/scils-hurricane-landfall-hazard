"""
Preprocessing modules for SCILS TC Model.

This module implements the input processing described in Section 2.1 of the
manuscript. It converts raw ERA5 reanalysis and IBTrACS track data into the
intermediate datasets consumed by the detrending and simulation stages:

- ERA5 processing: SST, wind shear, Potential Intensity (PI; Bister and
  Emanuel 2002, computed via tcpyPI), and Cyclone Genesis Index (CGI;
  Bruyère et al. 2012).
- GWL calculation: Global Warming Level time series following C3S
  methodology, used as the predictor for detrending.
- ENSO classification: year-level El Niño / La Niña / Neutral labels
  based on ASO RONI anomalies, used to condition LMI location sampling.
- IBTrACS properties: per-storm LMI, LFI, and location extraction.
- IBTrACS statistics: pooled LMI/PI ratio histogram (paper Figure 1a),
  region-specific zero-one-inflated Beta LFI/LMI distributions (paper
  Figure 1b), and ENSO-conditioned LMI location KDE maps.
"""

from .enso_classification import classify_enso_state, process_enso
from .era5_processing import (
    calculate_aso_mdr_sst,
    calculate_cgi,
    calculate_potential_intensity,
    calculate_wind_shear,
)
from .ibtracs_properties import process_ibtracs_properties
from .ibtracs_statistics import (
    calculate_lfi_lmi_distributions,
    calculate_lmi_kde,
    calculate_lmi_pi_ratio,
)

__all__ = [
    'classify_enso_state',
    'process_enso',
    'process_ibtracs_properties',
    'calculate_lmi_kde',
    'calculate_lmi_pi_ratio',
    'calculate_lfi_lmi_distributions',
    'calculate_aso_mdr_sst',
    'calculate_wind_shear',
    'calculate_potential_intensity',
    'calculate_cgi',
]
