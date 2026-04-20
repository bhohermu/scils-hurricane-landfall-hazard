"""
Core simulation functions for SCILS TC Model.

This module implements the main simulation loop that generates
synthetic tropical cyclone events for each year and iteration.

Uses:
- LMI/PI: Single histogram distribution
- LFI/LMI: 0-1 inflated Beta distribution per Jewson region
- Rejection sampling with TS+ threshold (34 kts / 17.49 m/s)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config
from scils_tc.utils import SimulationArtifact, SimulationEnsemble, TargetSpec
from scils_tc.utils.regions import get_region_for_point, load_jewson_regions
from scils_tc.utils.saffir_simpson import TS_THRESHOLD_MS

from .sampling import (
    get_mdr_cgi_from_maps,
    get_mdr_sst_from_detrended,
    get_pi_at_location,
    load_cgi_data,
    load_lfi_lmi_distributions,
    load_lmi_kde,
    load_lmi_pi_ratio_distribution,
    load_pi_data,
    sample_lfi_lmi_ratio,
    sample_lmi_location,
    sample_lmi_pi_ratio,
)


def load_all_kdes(preprocessed_dir):
    """
    Load all 3 LMI KDE maps (by ENSO state).

    Parameters
    ----------
    preprocessed_dir : Path
        Path to preprocessed data directory.

    Returns
    -------
    dict
        Dictionary mapping kde_group_name -> xr.Dataset.
    """
    kdes = {}
    for group in config.LMI_KDE_GROUPS:
        kde_file = preprocessed_dir / f"LMI_KDE_{group}.nc"
        if kde_file.exists():
            kdes[group] = load_lmi_kde(kde_file)
        else:
            print(f"Warning: KDE file not found: {kde_file}")
    return kdes


def get_kde_group(enso_state):
    """
    Determine which KDE group to use based on ENSO state.

    Parameters
    ----------
    enso_state : str
        'El Nino', 'Neutral', or 'La Nina'.

    Returns
    -------
    str
        KDE group name (e.g., 'neutral').
    """
    enso_map = {
        'El Nino': 'elnino',
        'Neutral': 'neutral',
        'La Nina': 'lanina'
    }
    return enso_map.get(enso_state, 'neutral')


def simulate_year(year, iteration, cgi_value, enso_state,
                  kdes, pi_ds, cgi_ds, lmi_pi_dist,
                  lfi_lmi_distributions, regions, rng, ensemble,
                  mdr_sst=None, roni_anomaly=None, scaled_mdr_cgi=None,
                  rejection_stats=None):
    """
    Simulate all storm events for one year and iteration.

    Uses rejection sampling to ensure both PI and LMI meet the tropical storm
    threshold (TS+, >= 34 knots). This maintains consistency with the training
    data which only includes TS+ storms.

    Parameters
    ----------
    year : int
        Year being simulated.
    iteration : int
        Iteration number.
    cgi_value : float
        Scaled MDR CGI value (lambda for Poisson, expected storm count).
    enso_state : str
        'El Nino', 'Neutral', or 'La Nina'.
    kdes : dict
        Dictionary of KDE datasets.
    pi_ds : xr.Dataset
        PI dataset.
    lmi_pi_dist : dict
        LMI/PI ratio distribution parameters.
    lfi_lmi_distributions : dict
        LFI/LMI distributions by region.
    regions : dict
        Jewson regions dictionary.
    rng : np.random.Generator
        Random number generator.
    mdr_sst : float, optional
        Actual ASO MDR SST value in degC.
    roni_anomaly : float, optional
        ASO RONI anomaly value.
    scaled_mdr_cgi : float, optional
        Scaled MDR CGI value (June-November average, with scale factor applied).
    rejection_stats : dict, optional
        Dictionary to accumulate rejection statistics. If provided, will be updated with:
        'pi_rejections', 'lmi_rejections', 'total_attempts'.

    Returns
    -------
    list of dict
        List of event dictionaries.
    """
    events = []

    # Step 1: Draw number of storms from Poisson distribution
    n_storms = rng.poisson(cgi_value)

    if n_storms == 0:
        return [
            ensemble.placeholder_event(
                year=year,
                iteration=iteration,
                enso_state=enso_state,
                mdr_sst=mdr_sst,
                roni_anomaly=roni_anomaly,
                scaled_mdr_cgi=scaled_mdr_cgi,
            )
        ]

    # Step 2: Distribute storms across months using Multinomial with monthly MDR CGI as weights
    from .sampling import get_monthly_mdr_cgi
    monthly_cgi = get_monthly_mdr_cgi(cgi_ds, year)

    months = list(monthly_cgi.keys())
    cgi_values = np.array([monthly_cgi[m] for m in months])

    # Normalize to probabilities
    total_cgi = cgi_values.sum()
    if total_cgi > 0:
        probs = cgi_values / total_cgi
    else:
        # Fallback to uniform if no valid CGI
        probs = np.ones(len(months)) / len(months)

    month_counts = rng.multinomial(n_storms, probs)

    # Step 3: Select appropriate KDE
    kde_group = get_kde_group(enso_state)
    if kde_group not in kdes:
        print(f"Warning: KDE group {kde_group} not found, using neutral")
        kde_group = 'neutral'

    kde_ds = kdes[kde_group]

    # Step 4: Generate events for each month
    event_id = 0
    for month, count in zip(months, month_counts):
        if count == 0:
            continue

        # Sample LMI locations
        lons, lats = sample_lmi_location(kde_ds, rng, n_samples=count)

        for i in range(count):
            lon = lons[i]
            lat = lats[i]

            # Rejection sampling for PI and LMI (TS+ threshold)
            # We need both PI >= TS and LMI >= TS to match training data support
            max_resample_attempts = 100
            attempt = 0
            pi_rejections_this_storm = 0
            lmi_rejections_this_storm = 0
            
            while attempt < max_resample_attempts:
                # Get PI at location for this month and year
                pi_value = get_pi_at_location(pi_ds, lon, lat, month, year=year)
                
                # Check if PI is valid (not NaN, > 0)
                if np.isnan(pi_value) or pi_value <= 0:
                    # Resample location due to invalid PI (over land, etc.)
                    lon, lat = sample_lmi_location(kde_ds, rng, n_samples=1)
                    lon, lat = lon[0], lat[0]
                    attempt += 1
                    continue
                
                # Reject if PI < TS threshold
                if pi_value < TS_THRESHOLD_MS:
                    pi_rejections_this_storm += 1
                    lon, lat = sample_lmi_location(kde_ds, rng, n_samples=1)
                    lon, lat = lon[0], lat[0]
                    attempt += 1
                    continue
                
                # PI is valid and above threshold, now sample LMI/PI ratio
                lmi_pi_ratio = sample_lmi_pi_ratio(lmi_pi_dist, pi_value, rng, n_samples=1)[0]
                lmi = lmi_pi_ratio * pi_value
                
                # Reject if LMI < TS threshold
                if lmi < TS_THRESHOLD_MS:
                    lmi_rejections_this_storm += 1
                    # Resample location (not just ratio) to get fresh PI
                    lon, lat = sample_lmi_location(kde_ds, rng, n_samples=1)
                    lon, lat = lon[0], lat[0]
                    attempt += 1
                    continue
                
                # Both PI and LMI are valid and above TS threshold - success!
                break
            
            # Track rejection statistics
            if rejection_stats is not None:
                rejection_stats['pi_rejections'] += pi_rejections_this_storm
                rejection_stats['lmi_rejections'] += lmi_rejections_this_storm
                rejection_stats['total_attempts'] += attempt + 1
            
            # If still invalid after max attempts, skip this event
            if attempt >= max_resample_attempts:
                continue
            
            # At this point we have valid pi_value, lmi_pi_ratio, and lmi from the loop

            # Get Jewson region for LMI location
            region_name, region_num = get_region_for_point(lon, lat, regions)

            # Sample LFI/LMI ratio based on region using Beta distribution
            if region_num is not None and region_num in lfi_lmi_distributions:
                beta_params = lfi_lmi_distributions[region_num]
                lfi_lmi_ratio = sample_lfi_lmi_ratio(beta_params, rng, n_samples=1)[0]
            else:
                # Default: use region 1 (EGULF) if no region found
                default_region = 1
                if default_region in lfi_lmi_distributions:
                    beta_params = lfi_lmi_distributions[default_region]
                    lfi_lmi_ratio = sample_lfi_lmi_ratio(beta_params, rng, n_samples=1)[0]
                else:
                    lfi_lmi_ratio = np.nan

            # Calculate LFI
            if np.isnan(lmi) or np.isnan(lfi_lmi_ratio):
                lfi = np.nan
            else:
                lfi = max(0.0, lfi_lmi_ratio * lmi)  # Floor at 0

            # Get SSHS categories
            lmi_sshs = config.get_sshs_category(lmi)
            lfi_sshs = config.get_sshs_category(lfi)

            # Store event
            event = {
                'year': year,
                'iteration': iteration,
                'event_id': event_id,
                'month': month,
                'lmi_lon': lon,
                'lmi_lat': lat,
                'pi': pi_value,
                'lmi_pi_ratio': lmi_pi_ratio,
                'lmi': lmi,
                'lmi_sshs': lmi_sshs,
                'region_name': region_name,
                'region_number': region_num,
                'lfi_lmi_ratio': lfi_lmi_ratio,
                'lfi': lfi,
                'lfi_sshs': lfi_sshs,
                'mdr_sst_degc': mdr_sst,
                'scaled_mdr_cgi': scaled_mdr_cgi,
                'enso_state': enso_state,
                'roni_anomaly': roni_anomaly,
                'is_placeholder': False,
            }
            events.append(event)
            event_id += 1

    if not events:
        return [
            ensemble.placeholder_event(
                year=year,
                iteration=iteration,
                enso_state=enso_state,
                mdr_sst=mdr_sst,
                roni_anomaly=roni_anomaly,
                scaled_mdr_cgi=scaled_mdr_cgi,
            )
        ]

    return events


def run_simulation(start_year, end_year, n_iter, target_year=None, target_gwl=None,
                   use_historical=False, detrend_pi_only=False, detrend_cgi_only=False,
                   region='CONUS', seed=None, verbose=True):
    """
    Run the full simulation for a range of years.

    This FINAL version uses:
    - LMI/PI: Single histogram distribution
    - LFI/LMI: 0-1 inflated Beta distribution
    - Rejection sampling: Ensures PI and LMI >= TS threshold (34 kts)

    Parameters
    ----------
    start_year : int
        First year to simulate.
    end_year : int
        Last year to simulate (inclusive).
    n_iter : int
        Number of iterations per year.
    target_year : int, optional
        Target year for detrended data. Takes precedence over target_gwl.
    target_gwl : float, optional
        Target GWL for detrended data (e.g., 1.5).
    use_historical : bool, default False
        If True, use historical (non-detrended) PI, CGI, and SST.
    detrend_pi_only : bool, default False
        If True, use detrended PI with historical CGI.
    detrend_cgi_only : bool, default False
        If True, use detrended CGI with historical PI.
    region : str, default 'CONUS'
        Region for landfall analysis ('CONUS' or 'NorthAtlantic').
    seed : int or None
        Random seed. If None, uses random initialization.
    verbose : bool
        Print progress information.

    Returns
    -------
    pd.DataFrame
        DataFrame with all simulated events.
    """
    # Initialize random generator
    rng = np.random.default_rng(seed)
    ensemble = SimulationEnsemble(start_year=start_year, end_year=end_year, n_iter=n_iter)

    if verbose:
        print(f"Running simulation for years {start_year}-{end_year}")
        print(f"  Iterations per year: {n_iter}")
        print(f"  Random seed: {seed}")
        print(f"  Rejection sampling threshold: TS+ ({TS_THRESHOLD_MS:.2f} m/s)")

    # Determine data sources based on options
    if use_historical:
        data_mode = "historical"
        use_detrended_pi = False
        use_detrended_cgi = False
        if verbose:
            print("  Using historical (non-detrended) PI, CGI, and SST")
    elif detrend_pi_only:
        data_mode = "pi_only"
        use_detrended_pi = True
        use_detrended_cgi = False
        if verbose:
            print("  Using detrended PI with historical CGI")
    elif detrend_cgi_only:
        data_mode = "cgi_only"
        use_detrended_pi = False
        use_detrended_cgi = True
        if verbose:
            print("  Using detrended CGI with historical PI")
    else:
        data_mode = "detrended"
        use_detrended_pi = True
        use_detrended_cgi = True
        target_spec = TargetSpec.from_inputs(
            target_year=target_year,
            target_gwl=target_gwl,
            default_gwl=config.DEFAULT_TARGET_GWL,
        )
        if verbose:
            print(f"  Using detrended data: {target_spec.label}")

    # For partial detrending, still need target suffix
    if (use_detrended_pi or use_detrended_cgi) and 'target_spec' not in locals():
        target_spec = TargetSpec.from_inputs(
            target_year=target_year,
            target_gwl=target_gwl,
            default_gwl=config.DEFAULT_TARGET_GWL,
        )
    target_suffix = target_spec.suffix if (use_detrended_pi or use_detrended_cgi) else None

    # Load required data
    if verbose:
        print("Loading data...")

    # Load KDEs
    kdes = load_all_kdes(config.PREPROCESSED_DIR)
    if len(kdes) == 0:
        raise ValueError("No KDE files found in preprocessed directory")

    # Load PI data (detrended or historical based on options)
    if use_detrended_pi:
        pi_file = config.DETRENDED_DIR / f"ERA5_PI_detrended_{target_suffix}.nc"
        if not pi_file.exists():
            raise ValueError(f"Detrended PI file not found: {pi_file}")
    else:
        pi_file = config.get_output_path(config.PI_FILE)
        if not pi_file.exists():
            raise ValueError(f"Historical PI file not found: {pi_file}")
    pi_ds = load_pi_data(pi_file)

    # Load CGI data (detrended or historical based on options)
    if use_detrended_cgi:
        cgi_file = config.DETRENDED_DIR / f"ERA5_CGI_detrended_{target_suffix}.nc"
        if not cgi_file.exists():
            raise ValueError(f"Detrended CGI file not found: {cgi_file}")
    else:
        cgi_file = config.get_output_path(config.CGI_MAP_FILE)
        if not cgi_file.exists():
            raise ValueError(f"Historical CGI file not found: {cgi_file}")
    cgi_ds = load_cgi_data(cgi_file)

    # Load historical CGI CSV to get the scaling factor
    cgi_df_historical = pd.read_csv(config.PREPROCESSED_DIR / config.CGI_MDR_FILE)
    total_observed = cgi_df_historical['Observed_Count'].sum()
    cgi_col = 'MDR_CGI_Sum' if 'MDR_CGI_Sum' in cgi_df_historical.columns else 'MDR_CGI'
    total_raw_cgi = cgi_df_historical[cgi_col].sum()
    cgi_scale_factor = total_observed / total_raw_cgi if total_raw_cgi > 0 else 1.0

    if verbose:
        print(f"  CGI scaling factor: {cgi_scale_factor:.4f} (from historical {cgi_col})")

    # Load SST data for record-keeping
    sst_df_historical = pd.read_csv(config.PREPROCESSED_DIR / config.ASO_MDR_SST_FILE)

    # Calculate SST climatology mean from HISTORICAL data (1991-2020) for reference
    clim_mask = (sst_df_historical['Year'] >= config.CLIMATOLOGY_START_YEAR) & (sst_df_historical['Year'] <= config.CLIMATOLOGY_END_YEAR)
    sst_climatology = sst_df_historical.loc[clim_mask, 'ASO_MDR_SST'].mean()

    # Load SST data (just for record-keeping in output)
    if use_historical:
        use_detrended_sst = False
        sst_ds = None
        sst_df = sst_df_historical
    else:
        sst_file = config.DETRENDED_DIR / f"ERA5_SST_detrended_{target_suffix}.nc"
        if sst_file.exists():
            sst_ds = xr.open_dataset(sst_file)
            if '__xarray_dataarray_variable__' in sst_ds.data_vars:
                sst_ds = sst_ds.rename({'__xarray_dataarray_variable__': 'sst'})
            use_detrended_sst = True
            sst_df = None
        else:
            use_detrended_sst = False
            sst_ds = None
            sst_df = sst_df_historical

    # Load ENSO state
    enso_df = pd.read_csv(config.PREPROCESSED_DIR / config.ENSO_STATE_FILE)

    # Load LMI/PI ratio distribution (single histogram)
    lmi_pi_dist = load_lmi_pi_ratio_distribution(
        config.PREPROCESSED_DIR / config.LMI_PI_RATIO_FILE
    )

    # Load LFI/LMI distributions (0-1 inflated Beta)
    lfi_lmi_filename = f"LFI_LMI_{region}.csv"
    lfi_lmi_filepath = config.PREPROCESSED_DIR / lfi_lmi_filename

    if not lfi_lmi_filepath.exists():
        raise FileNotFoundError(
            f"LFI/LMI distribution file not found: {lfi_lmi_filepath}. "
            f"Run preprocessing first."
        )

    lfi_lmi_distributions = load_lfi_lmi_distributions(lfi_lmi_filepath)

    # Load Jewson regions
    regions = load_jewson_regions(config.REGION_DEFINITIONS_FILE)

    if verbose:
        print("Starting simulation...")

    # Run simulation
    all_events = []
    
    # Initialize rejection statistics
    rejection_stats = {
        'pi_rejections': 0,
        'lmi_rejections': 0,
        'total_attempts': 0
    }

    for year in range(start_year, end_year + 1):
        if verbose:
            print(f"  Year {year}...", end=" ")

        # Get CGI for this year from maps (June-November MDR average)
        mdr_cgi_raw = get_mdr_cgi_from_maps(cgi_ds, year)
        if np.isnan(mdr_cgi_raw):
            if verbose:
                print(f"Warning: No CGI data for year {year}, skipping")
            continue
        cgi_value = mdr_cgi_raw * cgi_scale_factor

        # Get MDR SST for record keeping
        if use_detrended_sst:
            mdr_sst = get_mdr_sst_from_detrended(sst_ds, year)
            if np.isnan(mdr_sst):
                mdr_sst = sst_climatology
        else:
            sst_row = sst_df[sst_df['Year'] == year]
            mdr_sst = sst_row['ASO_MDR_SST'].values[0] if len(sst_row) > 0 else sst_climatology

        # Get ENSO state and anomaly
        enso_row = enso_df[enso_df['Year'] == year]
        if len(enso_row) == 0:
            enso_state = 'Neutral'
            roni_anomaly = 0.0
        else:
            enso_state = enso_row['ENSO_State'].values[0]
            # Support both old (ASO_Nino34_Anomaly) and new (ASO_RONI) column names
            if 'ASO_RONI' in enso_df.columns:
                roni_anomaly = enso_row['ASO_RONI'].values[0]
            else:
                roni_anomaly = enso_row['ASO_Nino34_Anomaly'].values[0]

        # Run iterations
        year_events = []
        for iteration in range(n_iter):
            events = simulate_year(
                year=year,
                iteration=iteration,
                cgi_value=cgi_value,
                enso_state=enso_state,
                kdes=kdes,
                pi_ds=pi_ds,
                cgi_ds=cgi_ds,
                lmi_pi_dist=lmi_pi_dist,
                lfi_lmi_distributions=lfi_lmi_distributions,
                regions=regions,
                rng=rng,
                ensemble=ensemble,
                mdr_sst=mdr_sst,
                roni_anomaly=roni_anomaly,
                scaled_mdr_cgi=cgi_value,
                rejection_stats=rejection_stats,
            )
            year_events.extend(events)

        all_events.extend(year_events)

        if verbose:
            n_events = len(year_events)
            avg_per_iter = n_events / n_iter if n_iter > 0 else 0
            print(f"{n_events} events ({avg_per_iter:.1f} avg/iter), ENSO={enso_state}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_events)
    df = ensemble.attach_metadata(df)

    if verbose:
        print(f"\nSimulation complete. Total events: {len(df)}")
        
        # Print rejection statistics
        total_attempts = rejection_stats['total_attempts']
        if total_attempts > 0:
            pi_rate = 100 * rejection_stats['pi_rejections'] / total_attempts
            lmi_rate = 100 * rejection_stats['lmi_rejections'] / total_attempts
            total_rejections = rejection_stats['pi_rejections'] + rejection_stats['lmi_rejections']
            total_rate = 100 * total_rejections / total_attempts
            print("\nRejection sampling statistics (TS+ threshold):")
            print(f"  Total sampling attempts: {total_attempts}")
            print(f"  PI rejections: {rejection_stats['pi_rejections']} ({pi_rate:.1f}%)")
            print(f"  LMI rejections: {rejection_stats['lmi_rejections']} ({lmi_rate:.1f}%)")
            print(f"  Total rejection rate: {total_rate:.1f}%")

    return df


def save_simulation_results(df, target_year=None, target_gwl=None, data_mode=None,
                            region='CONUS', n_iter=None, output_path=None):
    """
    Save simulation results to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results.
    target_year : int, optional
        Target year used for detrending.
    target_gwl : float, optional
        Target GWL used for detrending.
    data_mode : str, optional
        Data mode: 'historical', 'pi_only', 'cgi_only', or None for standard detrended.
    region : str, default='CONUS'
        Region for landfall analysis ('CONUS' or 'NorthAtlantic')
    n_iter : int, optional
        Number of iterations per year (included in filename).
    output_path : Path or str, optional
        If provided, use this path directly instead of constructing filename.

    Returns
    -------
    Path
        Path to the saved file.
    """
    if output_path is not None:
        output_path = Path(output_path)
    else:
        target_spec = None
        if data_mode != "historical":
            target_spec = TargetSpec.from_inputs(
                target_year=target_year,
                target_gwl=target_gwl,
                default_gwl=config.DEFAULT_TARGET_GWL,
            )
        artifact = SimulationArtifact(
            region=region,
            n_iter=n_iter,
            data_mode=data_mode,
            target_spec=target_spec,
        )
        output_path = config.get_simulated_path(artifact.simulation_filename())

    df.to_csv(output_path, index=False)
    print(f"Saved simulation results to: {output_path}")

    return output_path
