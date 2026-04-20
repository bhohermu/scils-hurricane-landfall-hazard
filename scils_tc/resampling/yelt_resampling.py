"""
YELT (Year Event Loss Table) resampling functions.

This module provides functions for adjusting a YELT based on climate change
landfall rate change factors per SSHS category.
"""

import numpy as np
import pandas as pd

import config
from scils_tc.utils.saffir_simpson import get_sshs_category

# SSHS categories for resampling (excludes TD)
SSHS_CATEGORIES = ['TS', 'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5']

# Default YELT file path
DEFAULT_YELT_PATH = config.YELT_FILE


def load_yelt(filepath=None, encoding='latin-1'):
    """
    Load YELT from CSV file.
    
    Parameters
    ----------
    filepath : Path or str, optional
        Path to YELT CSV file. Defaults to DEFAULT_YELT_PATH.
    encoding : str
        File encoding.
        
    Returns
    -------
    pd.DataFrame
        YELT with columns: Loss, LMI_category, EventID, tc_id, Iteration, LFI_ms
    """
    if filepath is None:
        filepath = DEFAULT_YELT_PATH
    
    df = pd.read_csv(filepath, encoding=encoding)
    return df


def convert_lfi_to_sshs(lfi_ms):
    """
    Convert landfall intensity in m/s to SSHS category.
    
    Uses the same thresholds as the preprocessing module.
    
    Parameters
    ----------
    lfi_ms : float
        Landfall intensity in m/s.
        
    Returns
    -------
    str
        SSHS category: 'TD', 'TS', 'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5', or None
    """
    if pd.isna(lfi_ms):
        return None
    return get_sshs_category(lfi_ms)


def add_lfi_sshs_category(yelt):
    """
    Add LFI_SSHS column to YELT based on LFI_ms.
    
    Parameters
    ----------
    yelt : pd.DataFrame
        YELT DataFrame.
        
    Returns
    -------
    pd.DataFrame
        YELT with added LFI_SSHS column.
    """
    yelt = yelt.copy()
    yelt['LFI_SSHS'] = yelt['LFI_ms'].apply(convert_lfi_to_sshs)
    return yelt


def clean_yelt_for_resampling(yelt):
    """
    Clean YELT before resampling:
    - Remove zero-loss bypassing events (no landfall and no loss)
    - Keep all landfalling events (including zero-loss landfalls)
    - Keep bypassing events with positive losses
    - Exclude TD category from landfalling events for resampling (set LFI_SSHS to None)
    
    Parameters
    ----------
    yelt : pd.DataFrame
        YELT DataFrame with LFI_SSHS column.
        
    Returns
    -------
    pd.DataFrame
        Cleaned YELT.
    int
        Number of rows removed.
    """
    yelt = yelt.copy()
    original_len = len(yelt)
    
    # Identify bypassing events (no landfall intensity)
    is_bypassing = yelt['LFI_SSHS'].isna() | (yelt['LFI_SSHS'] == 'TD')
    has_zero_loss = yelt['Loss'] == 0
    
    # Remove zero-loss bypassing events
    remove_mask = is_bypassing & has_zero_loss
    yelt = yelt[~remove_mask].copy()
    
    # Set TD category to None (so they won't be resampled but will be kept)
    yelt.loc[yelt['LFI_SSHS'] == 'TD', 'LFI_SSHS'] = None
    
    rows_removed = original_len - len(yelt)
    return yelt, rows_removed


def load_change_rates(filepath):
    """
    Load change rates from CSV file (output of run_resampling.py).
    
    Parameters
    ----------
    filepath : Path or str
        Path to change rates CSV file.
        
    Returns
    -------
    dict
        Dictionary mapping SSHS category to change rate.
    """
    # Skip comment lines
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find first non-comment line
    header_idx = 0
    for i, line in enumerate(lines):
        if not line.startswith('#'):
            header_idx = i
            break
    
    df = pd.read_csv(filepath, skiprows=header_idx)
    return dict(zip(df['sshs_category'], df['change_rate']))


def calculate_yelt_landfall_rates(yelt, n_years=10000):
    """
    Calculate landfall rates per SSHS category from YELT.
    
    Parameters
    ----------
    yelt : pd.DataFrame
        YELT with LFI_SSHS column.
    n_years : int
        Number of simulation years.
        
    Returns
    -------
    dict
        Dictionary mapping SSHS category to annual landfall rate.
    """
    # Count events per category (only landfalling events)
    landfalling = yelt[yelt['LFI_SSHS'].notna()]
    category_counts = landfalling.groupby('LFI_SSHS').size()
    
    rates = {}
    for cat in SSHS_CATEGORIES:
        count = category_counts.get(cat, 0)
        rates[cat] = count / n_years
    
    return rates


def resample_yelt_deterministic(yelt, change_rates, n_years=10000, seed=None):
    """
    Deterministically resample YELT based on change rates.
    
    For d < 1: remove events with probability (1 - d)
    For d > 1: add exactly n_add = round(n_years * (d - 1) * r_c) events
    
    Parameters
    ----------
    yelt : pd.DataFrame
        YELT with LFI_SSHS column.
    change_rates : dict
        Dictionary mapping SSHS category to change rate.
    n_years : int
        Number of simulation years.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    pd.DataFrame
        Resampled YELT.
    """
    if seed is not None:
        np.random.seed(seed)
    
    yelt = yelt.copy()
    
    # Separate landfalling and non-landfalling events
    landfalling = yelt[yelt['LFI_SSHS'].notna()].copy()
    non_landfalling = yelt[yelt['LFI_SSHS'].isna()].copy()
    
    # Calculate base rates
    base_rates = calculate_yelt_landfall_rates(yelt, n_years)
    
    # Step 1: Handle removal of events (d < 1)
    keep_mask = np.ones(len(landfalling), dtype=bool)
    for cat in SSHS_CATEGORIES:
        d = change_rates.get(cat, 1.0)
        if d < 1:
            cat_mask = landfalling['LFI_SSHS'] == cat
            cat_indices = np.where(cat_mask)[0]
            # Remove events with probability (1 - d)
            remove_probs = np.random.rand(len(cat_indices))
            remove_mask = remove_probs >= d  # Keep if random < d
            keep_mask[cat_indices[remove_mask]] = False
    
    kept_events = landfalling[keep_mask].copy()
    
    # Step 2: Handle addition of events (d > 1)
    additional_events = []
    event_id_counter = {}
    
    for cat in SSHS_CATEGORIES:
        d = change_rates.get(cat, 1.0)
        if d > 1:
            r_c = base_rates.get(cat, 0)
            if r_c > 0:
                n_add = int(round(n_years * (d - 1) * r_c))
                if n_add > 0:
                    # Sample from existing events of this category
                    cat_events = landfalling[landfalling['LFI_SSHS'] == cat]
                    if len(cat_events) > 0:
                        sampled = cat_events.sample(n=n_add, replace=True).reset_index(drop=True).copy()
                        
                        # Assign random iterations
                        sampled['Iteration'] = np.random.randint(0, n_years, size=n_add)
                        
                        # Make EventIDs unique
                        new_event_ids = []
                        for i in range(len(sampled)):
                            orig_id = sampled.iloc[i]['EventID']
                            event_id_counter[orig_id] = event_id_counter.get(orig_id, 0) + 1
                            new_event_ids.append(f"{orig_id}_dup{event_id_counter[orig_id]}")
                        sampled['EventID'] = new_event_ids
                        
                        additional_events.append(sampled)
    
    # Combine all events
    result_parts = [non_landfalling, kept_events]
    if additional_events:
        result_parts.extend(additional_events)
    
    result = pd.concat(result_parts, ignore_index=True)
    result = result.sort_values(['Iteration', 'EventID']).reset_index(drop=True)
    
    return result


def resample_yelt_poisson(yelt, change_rates, n_years=10000, seed=None):
    """
    Stochastically resample YELT using Poisson sampling for additions.
    
    For d < 1: remove events with probability (1 - d)
    For d > 1: for each year, draw from Poisson((d-1) * r_c) for additions
    
    Parameters
    ----------
    yelt : pd.DataFrame
        YELT with LFI_SSHS column.
    change_rates : dict
        Dictionary mapping SSHS category to change rate.
    n_years : int
        Number of simulation years.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    pd.DataFrame
        Resampled YELT.
    """
    if seed is not None:
        np.random.seed(seed)
    
    yelt = yelt.copy()
    
    # Separate landfalling and non-landfalling events
    landfalling = yelt[yelt['LFI_SSHS'].notna()].copy()
    non_landfalling = yelt[yelt['LFI_SSHS'].isna()].copy()
    
    # Calculate base rates
    base_rates = calculate_yelt_landfall_rates(yelt, n_years)
    
    # Step 1: Handle removal of events (d < 1)
    keep_mask = np.ones(len(landfalling), dtype=bool)
    for cat in SSHS_CATEGORIES:
        d = change_rates.get(cat, 1.0)
        if d < 1:
            cat_mask = landfalling['LFI_SSHS'] == cat
            cat_indices = np.where(cat_mask)[0]
            # Remove events with probability (1 - d)
            remove_probs = np.random.rand(len(cat_indices))
            remove_mask = remove_probs >= d  # Keep if random < d
            keep_mask[cat_indices[remove_mask]] = False
    
    kept_events = landfalling[keep_mask].copy()
    
    # Step 2: Handle addition of events (d > 1) using Poisson per year
    additional_events = []
    event_id_counter = {}
    
    for cat in SSHS_CATEGORIES:
        d = change_rates.get(cat, 1.0)
        if d > 1:
            r_c = base_rates.get(cat, 0)
            if r_c > 0:
                delta_rate = (d - 1) * r_c  # Additional events per year
                cat_events = landfalling[landfalling['LFI_SSHS'] == cat]
                
                if len(cat_events) > 0:
                    # For each year, draw from Poisson
                    for year in range(n_years):
                        n_add = np.random.poisson(delta_rate)
                        if n_add > 0:
                            sampled = cat_events.sample(n=n_add, replace=True).reset_index(drop=True).copy()
                            sampled['Iteration'] = year
                            
                            # Make EventIDs unique
                            new_event_ids = []
                            for i in range(len(sampled)):
                                orig_id = sampled.iloc[i]['EventID']
                                event_id_counter[orig_id] = event_id_counter.get(orig_id, 0) + 1
                                new_event_ids.append(f"{orig_id}_dup{event_id_counter[orig_id]}")
                            sampled['EventID'] = new_event_ids
                            
                            additional_events.append(sampled)
    
    # Combine all events
    result_parts = [non_landfalling, kept_events]
    if additional_events:
        result_parts.extend(additional_events)
    
    result = pd.concat(result_parts, ignore_index=True)
    result = result.sort_values(['Iteration', 'EventID']).reset_index(drop=True)
    
    return result


def resample_yelt(yelt, change_rates, method='deterministic', n_years=10000, seed=None):
    """
    Resample YELT based on change rates.
    
    Parameters
    ----------
    yelt : pd.DataFrame
        YELT with LFI_SSHS column.
    change_rates : dict
        Dictionary mapping SSHS category to change rate.
    method : str
        'deterministic' or 'poisson'
    n_years : int
        Number of simulation years.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    pd.DataFrame
        Resampled YELT.
    """
    if method == 'deterministic':
        return resample_yelt_deterministic(yelt, change_rates, n_years, seed)
    elif method == 'poisson':
        return resample_yelt_poisson(yelt, change_rates, n_years, seed)
    else:
        raise ValueError(f"Unknown resampling method: {method}")


def calculate_oep_curve(yelt, n_years=10000, return_periods=None):
    """
    Calculate Occurrence Exceedance Probability curve.
    
    OEP(x) = probability that the largest loss in a year exceeds x.
    
    Parameters
    ----------
    yelt : pd.DataFrame
        YELT DataFrame with 'Loss' and 'Iteration' columns.
    n_years : int
        Number of simulation years.
    return_periods : array-like, optional
        Return periods to calculate losses for.
        
    Returns
    -------
    dict
        Dictionary with 'max_losses_per_year', 'exceedance_probs', 'losses',
        and optionally 'rp_losses' if return_periods is provided.
    """
    # Calculate maximum loss per year (iteration)
    max_losses = yelt.groupby('Iteration')['Loss'].max()
    
    # Fill in years with no events (loss = 0)
    max_losses = max_losses.reindex(range(n_years), fill_value=0.0)
    
    # Sort losses in descending order for OEP curve
    sorted_losses = np.sort(max_losses.values)[::-1]
    
    # Calculate exceedance probabilities (rank / n_years)
    n = len(sorted_losses)
    exceedance_probs = np.arange(1, n + 1) / n_years
    
    result = {
        'max_losses_per_year': max_losses.values,
        'sorted_losses': sorted_losses,
        'exceedance_probs': exceedance_probs,
    }
    
    # Calculate losses at specific return periods
    if return_periods is not None:
        rp_losses = {}
        for rp in return_periods:
            # Return period = 1 / exceedance_probability
            # So we need the loss at exceedance_probability = 1/rp
            target_prob = 1.0 / rp
            
            # Find index where exceedance_prob <= target_prob
            # sorted_losses is in descending order, exceedance_probs is ascending
            # We want the loss where prob >= target_prob (conservative)
            idx = np.searchsorted(exceedance_probs, target_prob, side='right') - 1
            if idx < 0:
                idx = 0
            if idx >= n:
                idx = n - 1
            rp_losses[rp] = sorted_losses[idx]
        result['rp_losses'] = rp_losses
    
    return result


def calculate_aal(yelt, n_years=10000):
    """
    Calculate Average Annual Loss.
    
    Parameters
    ----------
    yelt : pd.DataFrame
        YELT DataFrame with 'Loss' column.
    n_years : int
        Number of simulation years.
        
    Returns
    -------
    float
        Average annual loss.
    """
    total_loss = yelt['Loss'].sum()
    return total_loss / n_years


def calculate_loss_metrics(yelt, n_years=10000, return_periods=None):
    """
    Calculate loss metrics for a YELT.
    
    Parameters
    ----------
    yelt : pd.DataFrame
        YELT DataFrame.
    n_years : int
        Number of simulation years.
    return_periods : list, optional
        Return periods to calculate.
        
    Returns
    -------
    dict
        Dictionary with 'aal' and 'rp_losses'.
    """
    if return_periods is None:
        return_periods = [2, 5, 10, 20, 50, 100, 200, 500]
    
    aal = calculate_aal(yelt, n_years)
    oep = calculate_oep_curve(yelt, n_years, return_periods)
    
    return {
        'aal': aal,
        'rp_losses': oep['rp_losses'],
        'oep_curve': oep,
    }


def compare_yelts(orig_yelt, adjusted_yelts, n_years=10000, return_periods=None):
    """
    Compare original and adjusted YELTs.
    
    Parameters
    ----------
    orig_yelt : pd.DataFrame
        Original YELT.
    adjusted_yelts : list of pd.DataFrame
        List of adjusted YELTs (can be a single YELT in a list).
    n_years : int
        Number of simulation years.
    return_periods : list, optional
        Return periods to calculate.
        
    Returns
    -------
    dict
        Comparison results with original and adjusted metrics.
    """
    if return_periods is None:
        return_periods = [2, 5, 10, 20, 50, 100, 200, 500]
    
    orig_metrics = calculate_loss_metrics(orig_yelt, n_years, return_periods)
    
    adjusted_metrics_list = []
    for adj_yelt in adjusted_yelts:
        adjusted_metrics_list.append(calculate_loss_metrics(adj_yelt, n_years, return_periods))
    
    # Calculate ratios
    aal_ratios = [m['aal'] / orig_metrics['aal'] if orig_metrics['aal'] > 0 else np.nan 
                  for m in adjusted_metrics_list]
    
    rp_ratios = {rp: [] for rp in return_periods}
    for m in adjusted_metrics_list:
        for rp in return_periods:
            orig_loss = orig_metrics['rp_losses'].get(rp, 0)
            adj_loss = m['rp_losses'].get(rp, 0)
            ratio = adj_loss / orig_loss if orig_loss > 0 else np.nan
            rp_ratios[rp].append(ratio)
    
    return {
        'original': orig_metrics,
        'adjusted': adjusted_metrics_list,
        'aal_ratios': aal_ratios,
        'rp_ratios': rp_ratios,
    }
