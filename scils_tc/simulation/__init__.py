"""
Simulation module for SCILS TC Model.

This module generates synthetic tropical cyclone events based on:
- CGI-driven annual storm counts (Poisson distribution)
- Monthly distribution of storms (Multinomial)
- LMI location sampling from KDE maps
- PI-based intensity assignment
- LFI/LMI ratio sampling by region
"""

from .sampling import (
    get_pi_at_location,
    sample_lfi_lmi_ratio,
    sample_lmi_location,
    sample_lmi_pi_ratio,
)
from .simulation import run_simulation, simulate_year

__all__ = [
    'run_simulation',
    'simulate_year',
    'sample_lmi_location',
    'sample_lmi_pi_ratio',
    'sample_lfi_lmi_ratio',
    'get_pi_at_location',
]
