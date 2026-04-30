"""
Simulation module for SCILS TC Model.

This module implements the four-step stochastic event generation described in
Section 2.2 and Figure 2 of the manuscript:

1. **TC count**: seasonal storm count drawn from Poisson(λ), where λ is the
   scaled MDR CGI averaged over June–November. Storms are distributed to
   months using monthly CGI weights (Multinomial).
2. **LMI location**: sampled from ENSO-conditioned KDE maps of historical
   LMI positions (Section 2.1, "Lifetime Maximum Intensity").
3. **Assign LMI**: PI at the sampled location × LMI/PI ratio drawn from the
   pooled historical distribution (paper Figure 1a). Rejection sampling
   ensures both PI and LMI exceed the tropical-storm threshold (≥34 kt).
4. **Calculate LFI**: LMI × LFI/LMI ratio drawn from a region-specific
   zero-one-inflated Beta distribution (paper Figure 1b, Jewson 2023a
   regions).

For each target GWL, 1 000 realisations are generated per historical year
(default), producing 46 000 stochastic years. A historical scenario using
non-detrended inputs serves as the baseline for rate-change calculations
and for model validation against IBTrACS observations.
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
