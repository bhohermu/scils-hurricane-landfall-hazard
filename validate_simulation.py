#!/usr/bin/env python
"""
Validation script for SCILS TC Model simulations.

Calculates validation metrics comparing simulated results to historical observations:
1. RMSE (Root Mean Squared Error) - both climatology baseline and model RMSE
2. CRPS (Continuous Ranked Probability Score) for empirical distributions
3. Log-likelihood of observations under SCILS annual predictions
4. Log-likelihood of observations under static empirical distribution
5. Likelihood ratio (LR) comparing SCILS to static model

Metrics are calculated for three intensity thresholds:
- Cat0+: All landfalls (any intensity)
- Cat1+: Hurricane-strength landfalls (≥33 m/s)
- Cat3+: Major hurricane landfalls (≥50 m/s)

Usage:
    python validate_simulation.py --simulation-file PATH [--region REGION] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from scils_tc.utils import actual_event_rows, year_iteration_grid_from_dataframe

# Intensity thresholds (m/s)
# Using >= comparison, so threshold is inclusive lower bound
INTENSITY_THRESHOLDS = {
    'TS+': 17.5,     # Tropical Storm+ (≥34 kt = 17.5 m/s)
    'Cat1+': 33.0,   # Hurricane Cat1+ (≥64 kt = 33 m/s)
    'Cat3+': 50.0,   # Major Hurricane Cat3+ (≥96 kt = 50 m/s)
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate SCILS TC Model simulation results.'
    )
    parser.add_argument(
        '--simulation-file', type=str, required=True,
        help='Path to simulation CSV file (e.g., simulated_events_historical_NorthAtlantic_n1000.csv)'
    )
    parser.add_argument(
        '--region', type=str, default='NorthAtlantic', choices=['CONUS', 'NorthAtlantic'],
        help='Region for landfall analysis (default: NorthAtlantic)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results (default: same as simulation file)'
    )
    parser.add_argument(
        '--climatology-start', type=int, default=None,
        help='Start year for climatology (default: all years in data)'
    )
    parser.add_argument(
        '--climatology-end', type=int, default=None,
        help='End year for climatology (default: all years in data)'
    )
    return parser.parse_args()


def load_historical_landfalls(region='NorthAtlantic', start_year=None, end_year=None, 
                               intensity_threshold=0):
    """
    Load historical landfall counts by year from IBTrACS.
    
    Parameters
    ----------
    region : str
        'CONUS' or 'NorthAtlantic'
    start_year : int, optional
        First year to include
    end_year : int, optional
        Last year to include
    intensity_threshold : float
        Minimum LFI threshold in m/s (0 for all, 33 for Cat1+, 50 for Cat3+)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Year, landfall_count
    """
    ibtracs_file = config.get_output_path(config.IBTRACS_PROPERTIES_FILE)
    
    # Prevent "NA" from being interpreted as NaN
    df = pd.read_csv(ibtracs_file, keep_default_na=False, na_values=[''])
    
    # Count landfalls per year
    lfi_col = f'{region}_LFI_ms'
    
    # A landfall meeting threshold is when LFI >= threshold
    # (matching plotting.py convention)
    df['has_landfall'] = (df[lfi_col] >= intensity_threshold) & (df[lfi_col].notna())
    
    # Filter by year range if specified
    if start_year is not None:
        df = df[df['Year'] >= start_year]
    if end_year is not None:
        df = df[df['Year'] <= end_year]
    
    # Count landfalls per year (storms meeting threshold)
    landfall_counts = df[df['has_landfall']].groupby('Year').size().reset_index(name='landfall_count')
    
    # Include years with zero landfalls - important!
    all_years = pd.DataFrame({'Year': range(df['Year'].min(), df['Year'].max() + 1)})
    landfall_counts = all_years.merge(landfall_counts, on='Year', how='left')
    landfall_counts['landfall_count'] = landfall_counts['landfall_count'].fillna(0).astype(int)
    
    return landfall_counts


def load_historical_lmi(start_year=None, end_year=None, intensity_threshold=0):
    """
    Load historical LMI storm counts by year from IBTrACS.
    
    Parameters
    ----------
    start_year : int, optional
        First year to include
    end_year : int, optional
        Last year to include
    intensity_threshold : float
        Minimum LMI threshold in m/s (0 for all, 17.5 for TS+, 33 for Cat1+, 50 for Cat3+)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Year, lmi_count (number of storms with LMI >= threshold)
    """
    ibtracs_file = config.get_output_path(config.IBTRACS_PROPERTIES_FILE)
    
    # Prevent "NA" from being interpreted as NaN
    df = pd.read_csv(ibtracs_file, keep_default_na=False, na_values=[''])
    
    # A storm meeting threshold is when LMI >= threshold
    df['meets_threshold'] = (df['LMI_ms'] >= intensity_threshold) & (df['LMI_ms'].notna())
    
    # Filter by year range if specified
    if start_year is not None:
        df = df[df['Year'] >= start_year]
    if end_year is not None:
        df = df[df['Year'] <= end_year]
    
    # Count storms per year (storms meeting threshold)
    lmi_counts = df[df['meets_threshold']].groupby('Year').size().reset_index(name='lmi_count')
    
    # Include years with zero storms meeting threshold - important!
    all_years = pd.DataFrame({'Year': range(df['Year'].min(), df['Year'].max() + 1)})
    lmi_counts = all_years.merge(lmi_counts, on='Year', how='left')
    lmi_counts['lmi_count'] = lmi_counts['lmi_count'].fillna(0).astype(int)
    
    return lmi_counts


def load_simulation_lmi(simulation_file, intensity_threshold=0):
    """
    Load simulation LMI storm counts by year and iteration.
    
    Parameters
    ----------
    simulation_file : str or Path
        Path to simulation CSV file
    intensity_threshold : float
        Minimum LMI threshold in m/s (0 for all, 17.5 for TS+, 33 for Cat1+, 50 for Cat3+)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: year, iteration, lmi_count (number of storms with LMI >= threshold)
    """
    df = pd.read_csv(simulation_file, keep_default_na=False, na_values=[''])
    actual_df = actual_event_rows(df)
    all_combinations_df = year_iteration_grid_from_dataframe(df)
    
    # Count storms with LMI >= threshold per year and iteration
    actual_df['meets_threshold'] = (actual_df['lmi'] >= intensity_threshold) & (actual_df['lmi'].notna())
    
    lmi_counts = actual_df[actual_df['meets_threshold']].groupby(['year', 'iteration']).size().reset_index(name='lmi_count')
    
    # Merge with all combinations - years with no storms meeting threshold will have NaN, fill with 0
    lmi_counts = all_combinations_df.merge(lmi_counts, on=['year', 'iteration'], how='left')
    lmi_counts['lmi_count'] = lmi_counts['lmi_count'].fillna(0).astype(int)
    
    return lmi_counts


def load_simulation_results(simulation_file, region='NorthAtlantic', intensity_threshold=0):
    """
    Load simulation results and count landfalls per year and iteration.
    
    Note: Each year-iteration combination represents a complete simulation.
    If no storms in an iteration have lfi > threshold, that's a valid result
    of 0 landfalls for that iteration (not missing data).
    
    Parameters
    ----------
    simulation_file : str or Path
        Path to simulation CSV file
    region : str
        'CONUS' or 'NorthAtlantic'
    intensity_threshold : float
        Minimum LFI threshold in m/s (0 for all, 33 for Cat1+, 50 for Cat3+)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: year, iteration, landfall_count
    """
    df = pd.read_csv(simulation_file, keep_default_na=False, na_values=[''])
    actual_df = actual_event_rows(df)
    all_combinations_df = year_iteration_grid_from_dataframe(df)
    
    # Count landfalls (lfi >= threshold) per year and iteration
    # Note: lfi=0 means no landfall for that storm, which is valid model output
    actual_df['has_landfall'] = (actual_df['lfi'] >= intensity_threshold) & (actual_df['lfi'].notna())
    
    landfall_counts = actual_df[actual_df['has_landfall']].groupby(['year', 'iteration']).size().reset_index(name='landfall_count')
    
    # Merge with all combinations - years with no landfalls will have NaN, fill with 0
    # This is correct: an iteration with no storms meeting threshold = 0 landfalls
    landfall_counts = all_combinations_df.merge(landfall_counts, on=['year', 'iteration'], how='left')
    landfall_counts['landfall_count'] = landfall_counts['landfall_count'].fillna(0).astype(int)
    
    return landfall_counts


def calculate_rmse(observed, simulated_distributions):
    """
    Calculate RMSE metrics:
    1. Climatology RMSE = std of observations (baseline)
    2. Model RMSE = RMSE between observed and simulated means
    3. Bias = Mean model - mean obs (positive = overestimation)
    
    Parameters
    ----------
    observed : pd.DataFrame
        Historical observations with Year, landfall_count
    simulated_distributions : pd.DataFrame
        Simulated counts with year, iteration, landfall_count
    
    Returns
    -------
    dict
        Dictionary with climatology_rmse, model_rmse, skill_score, bias
    pd.DataFrame
        Merged data with observed and simulated means per year
    """
    # Climatology RMSE = standard deviation of observations
    obs_mean = observed['landfall_count'].mean()
    climatology_rmse = np.sqrt(np.mean((observed['landfall_count'] - obs_mean)**2))
    
    # Calculate mean count per year from simulations
    sim_means = simulated_distributions.groupby('year')['landfall_count'].mean().reset_index()
    sim_means.columns = ['Year', 'sim_mean']
    
    # Merge with observations
    merged = observed.merge(sim_means, on='Year', how='inner')
    
    # Model RMSE = RMSE between observations and simulated means
    model_rmse = np.sqrt(np.mean((merged['landfall_count'] - merged['sim_mean'])**2))
    
    # Bias = Mean model - mean obs (positive = overestimation, negative = underestimation)
    # IMPORTANT: Calculate from ALL simulated data, not just merged years
    sim_overall_mean = simulated_distributions['landfall_count'].mean()  # Mean over all year-iterations
    obs_overall_mean = observed['landfall_count'].mean()  # Mean over observed years
    bias = sim_overall_mean - obs_overall_mean
    
    # Skill score: how much better is model vs climatology (1 = perfect, 0 = no skill, <0 = worse)
    skill_score = 1 - (model_rmse / climatology_rmse) if climatology_rmse > 0 else 0
    
    return {
        'climatology_rmse': climatology_rmse,
        'model_rmse': model_rmse,
        'skill_score': skill_score,
        'obs_mean': obs_mean,
        'sim_mean': sim_overall_mean,
        'bias': bias
    }, merged


def calculate_crps(observed_value, simulated_samples):
    """
    Calculate Continuous Ranked Probability Score (CRPS) for a single observation.
    
    CRPS measures the difference between the empirical CDF of the forecast
    and the step function at the observation.
    
    Formula: CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent samples from forecast, y is observation.
    
    Parameters
    ----------
    observed_value : int or float
        The observed value
    simulated_samples : array-like
        Array of simulated values (ensemble members)
    
    Returns
    -------
    float
        CRPS value (lower is better)
    """
    simulated_samples = np.asarray(simulated_samples)
    n = len(simulated_samples)
    
    if n == 0:
        return np.nan
    if n == 1:
        return np.abs(simulated_samples[0] - observed_value)
    
    # Term 1: E|X - y| = mean absolute error between ensemble and observation
    term1 = np.mean(np.abs(simulated_samples - observed_value))
    
    # Term 2: E|X - X'| = mean of all pairwise absolute differences
    # For sorted samples x_1 <= x_2 <= ... <= x_n, this can be computed efficiently:
    # E|X - X'| = (2 / n^2) * sum_{i=1}^{n} x_i * (2i - n - 1)
    # where i is 1-indexed
    x_sorted = np.sort(simulated_samples)
    indices = np.arange(1, n + 1)  # 1-indexed: 1, 2, ..., n
    term2 = (2.0 / (n * n)) * np.sum(x_sorted * (2 * indices - n - 1))
    
    crps = term1 - 0.5 * term2
    
    return crps


def fit_negative_binomial(counts):
    """
    Fit a negative binomial distribution to count data using MLE.
    
    Parameters
    ----------
    counts : array-like
        Observed count data
    
    Returns
    -------
    tuple
        (n, p) parameters of the negative binomial distribution
        where n = number of successes, p = probability of success
    """
    counts = np.array(counts)
    mean = counts.mean()
    var = counts.var()
    
    # Handle edge case where variance <= mean (use Poisson approximation)
    if var <= mean:
        # Return params that approximate Poisson
        n = 1e6  # Large n makes NB approach Poisson
        p = n / (n + mean)
        return n, p
    
    # Method of moments estimation
    # For NB: mean = n(1-p)/p, var = n(1-p)/p^2
    # Solving: p = mean/var, n = mean^2 / (var - mean)
    p = mean / var
    n = mean * p / (1 - p)
    
    # Ensure valid parameters
    n = max(0.1, n)
    p = max(0.01, min(0.99, p))
    
    return n, p


def crps_negative_binomial(observed_value, n, p, max_k=100):
    """
    Calculate CRPS for a negative binomial distribution.
    
    Uses the formula: CRPS = sum over k of (F(k) - 1(k >= y))^2
    where F is the CDF and y is the observation.
    
    Parameters
    ----------
    observed_value : int
        The observed value
    n : float
        NB parameter (number of successes)
    p : float
        NB parameter (probability of success)
    max_k : int
        Maximum k to sum over
    
    Returns
    -------
    float
        CRPS value
    """
    from scipy.stats import nbinom
    
    # For scipy.stats.nbinom: pmf(k, n, p) where mean = n(1-p)/p
    crps = 0.0
    cdf = 0.0
    
    for k in range(max_k + 1):
        pmf_k = nbinom.pmf(k, n, p)
        cdf += pmf_k
        indicator = 1.0 if k >= observed_value else 0.0
        crps += (cdf - indicator) ** 2
    
    return crps


def calculate_lmi_statistics(observed_lmi, simulated_lmi):
    """
    Calculate RMSE, bias, and CRPS for LMI storm counts.
    
    Parameters
    ----------
    observed_lmi : pd.DataFrame
        Historical LMI storm counts with Year, lmi_count
    simulated_lmi : pd.DataFrame
        Simulated LMI storm counts with year, iteration, lmi_count
    
    Returns
    -------
    dict
        Dictionary with lmi_rmse, lmi_bias, lmi_crps, lmi_obs_mean, lmi_sim_mean
    """
    # Calculate mean count per year from simulations
    sim_means = simulated_lmi.groupby('year')['lmi_count'].mean().reset_index()
    sim_means.columns = ['Year', 'sim_mean']
    
    # Merge with observations
    merged = observed_lmi.merge(sim_means, on='Year', how='inner')
    
    # Calculate overall means
    obs_mean = observed_lmi['lmi_count'].mean()
    # IMPORTANT: Calculate from ALL simulated data, not just merged years
    sim_overall_mean = simulated_lmi['lmi_count'].mean()  # Mean over all year-iterations
    
    # Bias = Mean model - mean obs
    bias = sim_overall_mean - obs_mean
    
    # RMSE between observations and simulated means (only for years with observations)
    rmse = np.sqrt(np.mean((merged['lmi_count'] - merged['sim_mean'])**2))
    
    # Calculate CRPS: average over all years
    crps_values = []
    for year in observed_lmi['Year']:
        obs_count = observed_lmi[observed_lmi['Year'] == year]['lmi_count'].values[0]
        sim_counts = simulated_lmi[simulated_lmi['year'] == year]['lmi_count'].values
        
        if len(sim_counts) == 0:
            continue
        
        crps = calculate_crps(obs_count, sim_counts)
        crps_values.append(crps)
    
    mean_crps = np.mean(crps_values) if crps_values else np.nan
    
    return {
        'lmi_obs_mean': obs_mean,
        'lmi_sim_mean': sim_overall_mean,
        'lmi_bias': bias,
        'lmi_rmse': rmse,
        'lmi_crps': mean_crps
    }


def calculate_crps_for_years(observed, simulated_distributions):
    """
    Calculate CRPS for each year and return mean CRPS.
    
    Parameters
    ----------
    observed : pd.DataFrame
        Historical observations with Year, landfall_count
    simulated_distributions : pd.DataFrame
        Simulated counts with year, iteration, landfall_count
    
    Returns
    -------
    float
        Mean CRPS across all years
    pd.DataFrame
        CRPS for each year
    """
    results = []
    
    for year in observed['Year']:
        obs_count = observed[observed['Year'] == year]['landfall_count'].values[0]
        sim_counts = simulated_distributions[simulated_distributions['year'] == year]['landfall_count'].values
        
        if len(sim_counts) == 0:
            continue
        
        crps = calculate_crps(obs_count, sim_counts)
        results.append({'Year': year, 'observed': obs_count, 'crps': crps})
    
    results_df = pd.DataFrame(results)
    mean_crps = results_df['crps'].mean()
    
    return mean_crps, results_df


def calculate_crps_null_nb(observed):
    """
    Calculate CRPS for null model (negative binomial fitted to all observations).
    
    For each year, uses the same NB distribution fitted to all observations
    (including that year - this is not leave-one-out).
    
    Parameters
    ----------
    observed : pd.DataFrame
        Historical observations with Year, landfall_count
    
    Returns
    -------
    float
        Mean CRPS across all years
    pd.DataFrame
        CRPS for each year
    tuple
        (n, p) fitted NB parameters
    """
    counts = observed['landfall_count'].values
    n, p = fit_negative_binomial(counts)
    
    results = []
    
    for year in observed['Year']:
        obs_count = observed[observed['Year'] == year]['landfall_count'].values[0]
        crps = crps_negative_binomial(obs_count, n, p)
        results.append({'Year': year, 'observed': obs_count, 'crps_nb': crps})
    
    results_df = pd.DataFrame(results)
    mean_crps = results_df['crps_nb'].mean()
    
    return mean_crps, results_df, (n, p)


def calculate_log_likelihood_scils(observed, simulated_distributions, smoothing=True):
    """
    Calculate log-likelihood of observed counts under SCILS annual predictions.
    
    For each year, uses the empirical distribution from that year's simulations.
    
    Parameters
    ----------
    observed : pd.DataFrame
        Historical observations with Year, landfall_count
    simulated_distributions : pd.DataFrame
        Simulated counts with year, iteration, landfall_count
    smoothing : bool
        If True, use Laplace smoothing to avoid zero probabilities
    
    Returns
    -------
    float
        Total log-likelihood
    pd.DataFrame
        Log-likelihood for each year
    """
    results = []
    
    for year in observed['Year']:
        obs_count = observed[observed['Year'] == year]['landfall_count'].values[0]
        sim_counts = simulated_distributions[simulated_distributions['year'] == year]['landfall_count'].values
        
        if len(sim_counts) == 0:
            continue
        
        n_sim = len(sim_counts)
        
        if smoothing:
            # Laplace smoothing: add 1 to each possible count
            # This prevents log(0) while preserving the distribution shape
            max_count = max(obs_count, sim_counts.max()) + 1
            count_freq = np.zeros(max_count + 1)
            for c in sim_counts:
                count_freq[c] += 1
            # Add Laplace smoothing
            count_freq += 1
            prob = count_freq[obs_count] / count_freq.sum()
        else:
            # Simple empirical probability
            prob = np.sum(sim_counts == obs_count) / n_sim
            if prob == 0:
                prob = 1e-10
        
        log_lik = np.log(prob)
        results.append({
            'Year': year, 
            'observed': obs_count, 
            'probability': prob, 
            'log_likelihood': log_lik,
            'sim_mean': sim_counts.mean(),
            'sim_std': sim_counts.std()
        })
    
    results_df = pd.DataFrame(results)
    total_log_lik = results_df['log_likelihood'].sum()
    
    return total_log_lik, results_df


def calculate_95pi_coverage(observed, simulated_distributions, count_col='landfall_count'):
    """
    Calculate the 95% Prediction Interval coverage.
    
    Computes the percentage of years where the observed value falls within 
    the 2.5th to 97.5th percentile range of the simulated distribution.
    
    Parameters
    ----------
    observed : pd.DataFrame
        Historical observations with Year and count column
    simulated_distributions : pd.DataFrame
        Simulated counts with year, iteration, and count column
    count_col : str
        Name of the count column ('landfall_count' for LFI, 'lmi_count' for LMI)
    
    Returns
    -------
    dict
        Dictionary with coverage, n_covered, n_total, and per-year details
    """
    sim_col = count_col  # same column name in simulated
    obs_col = count_col
    
    # Determine year column names (observed uses 'Year', simulated uses 'year')
    results = []
    n_covered = 0
    n_total = 0
    
    for year in observed['Year']:
        obs_count = observed[observed['Year'] == year][obs_col].values[0]
        sim_counts = simulated_distributions[simulated_distributions['year'] == year][sim_col].values
        
        if len(sim_counts) == 0:
            continue
        
        # Calculate 95% PI (2.5th and 97.5th percentiles)
        q025 = np.percentile(sim_counts, 2.5)
        q975 = np.percentile(sim_counts, 97.5)
        
        # Check if observation is within the interval
        in_interval = (obs_count >= q025) and (obs_count <= q975)
        
        n_total += 1
        if in_interval:
            n_covered += 1
        
        results.append({
            'Year': year,
            'observed': obs_count,
            'q025': q025,
            'q975': q975,
            'in_95pi': in_interval
        })
    
    coverage = (n_covered / n_total * 100) if n_total > 0 else np.nan
    
    return {
        'coverage': coverage,
        'n_covered': n_covered,
        'n_total': n_total,
        'details': pd.DataFrame(results)
    }


def calculate_95pi_coverage_nb(observed, n, p, count_col='landfall_count'):
    """
    Calculate the 95% Prediction Interval coverage for a negative binomial model.
    
    For each year, checks if observation falls within the 2.5th to 97.5th 
    percentile of the fitted NB distribution.
    
    Parameters
    ----------
    observed : pd.DataFrame
        Historical observations with Year and count column
    n : float
        NB parameter (number of successes)
    p : float
        NB parameter (probability of success)
    count_col : str
        Name of the count column
    
    Returns
    -------
    dict
        Dictionary with coverage, n_covered, n_total, and per-year details
    """
    from scipy.stats import nbinom
    
    # Calculate 95% PI from NB distribution
    q025 = nbinom.ppf(0.025, n, p)
    q975 = nbinom.ppf(0.975, n, p)
    
    results = []
    n_covered = 0
    n_total = 0
    
    for year in observed['Year']:
        obs_count = observed[observed['Year'] == year][count_col].values[0]
        
        # Check if observation is within the interval
        in_interval = (obs_count >= q025) and (obs_count <= q975)
        
        n_total += 1
        if in_interval:
            n_covered += 1
        
        results.append({
            'Year': year,
            'observed': obs_count,
            'q025': q025,
            'q975': q975,
            'in_95pi': in_interval
        })
    
    coverage = (n_covered / n_total * 100) if n_total > 0 else np.nan
    
    return {
        'coverage': coverage,
        'n_covered': n_covered,
        'n_total': n_total,
        'q025': q025,
        'q975': q975,
        'details': pd.DataFrame(results)
    }


def calculate_log_likelihood_static(observed, climatology_start=None, climatology_end=None, 
                                    smoothing=True):
    """
    Calculate log-likelihood of observed counts under static empirical distribution.
    
    Uses the same empirical distribution (from climatology period) for all years.
    
    Parameters
    ----------
    observed : pd.DataFrame
        Historical observations with Year, landfall_count
    climatology_start : int, optional
        Start year for climatology
    climatology_end : int, optional
        End year for climatology
    smoothing : bool
        If True, use Laplace smoothing to avoid zero probabilities
    
    Returns
    -------
    float
        Total log-likelihood
    pd.DataFrame
        Log-likelihood for each year
    """
    # Define climatology period
    if climatology_start is None:
        climatology_start = observed['Year'].min()
    if climatology_end is None:
        climatology_end = observed['Year'].max()
    
    # Get climatology counts
    clim_counts = observed[
        (observed['Year'] >= climatology_start) & 
        (observed['Year'] <= climatology_end)
    ]['landfall_count'].values
    
    if smoothing:
        # Build empirical distribution with Laplace smoothing
        max_count = clim_counts.max() + 1
        count_freq = np.zeros(max_count + 1)
        for c in clim_counts:
            count_freq[c] += 1
        # Add Laplace smoothing
        count_freq += 1
        # Normalize
        prob_array = count_freq / count_freq.sum()
    else:
        # Build empirical distribution without smoothing
        unique_counts, count_freq = np.unique(clim_counts, return_counts=True)
        empirical_probs = count_freq / len(clim_counts)
        prob_dict = dict(zip(unique_counts, empirical_probs))
    
    results = []
    
    for year in observed['Year']:
        obs_count = observed[observed['Year'] == year]['landfall_count'].values[0]
        
        if smoothing:
            # Get probability from smoothed distribution
            if obs_count < len(prob_array):
                prob = prob_array[obs_count]
            else:
                # Observation is higher than anything in climatology
                prob = 1 / (len(clim_counts) + len(prob_array))  # Laplace-like
        else:
            prob = prob_dict.get(obs_count, 1e-10)
        
        log_lik = np.log(prob)
        results.append({
            'Year': year, 
            'observed': obs_count, 
            'probability': prob, 
            'log_likelihood': log_lik
        })
    
    results_df = pd.DataFrame(results)
    total_log_lik = results_df['log_likelihood'].sum()
    
    return total_log_lik, results_df


def calculate_log_likelihood_nb(observed, climatology_start=None, climatology_end=None):
    """
    Calculate log-likelihood of observed counts under negative binomial null model.
    
    Fits NB to climatology period and evaluates LL for all years.
    
    Parameters
    ----------
    observed : pd.DataFrame
        Historical observations with Year, landfall_count
    climatology_start : int, optional
        Start year for climatology
    climatology_end : int, optional
        End year for climatology
    
    Returns
    -------
    float
        Total log-likelihood
    pd.DataFrame
        Log-likelihood for each year
    tuple
        (n, p) fitted NB parameters
    """
    from scipy.stats import nbinom
    
    # Define climatology period
    if climatology_start is None:
        climatology_start = observed['Year'].min()
    if climatology_end is None:
        climatology_end = observed['Year'].max()
    
    # Get climatology counts
    clim_counts = observed[
        (observed['Year'] >= climatology_start) & 
        (observed['Year'] <= climatology_end)
    ]['landfall_count'].values
    
    # Fit NB to climatology
    n, p = fit_negative_binomial(clim_counts)
    
    results = []
    
    for year in observed['Year']:
        obs_count = observed[observed['Year'] == year]['landfall_count'].values[0]
        
        # Get probability from NB distribution
        prob = nbinom.pmf(obs_count, n, p)
        
        # Avoid log(0)
        if prob < 1e-10:
            prob = 1e-10
        
        log_lik = np.log(prob)
        results.append({
            'Year': year, 
            'observed': obs_count, 
            'probability': prob, 
            'log_likelihood': log_lik
        })
    
    results_df = pd.DataFrame(results)
    total_log_lik = results_df['log_likelihood'].sum()
    
    return total_log_lik, results_df, (n, p)


def plot_validation_results(observed, simulated_distributions, output_dir, threshold_name='Cat0+',
                            nb_params=None):
    """
    Create validation plots.
    
    Parameters
    ----------
    observed : pd.DataFrame
        Historical observations
    simulated_distributions : pd.DataFrame
        Simulated counts
    output_dir : Path
        Output directory
    threshold_name : str
        Name of intensity threshold for plot title
    nb_params : tuple, optional
        (n, p) parameters of fitted negative binomial distribution
    """
    from scipy.stats import nbinom
    
    # Calculate statistics per year
    sim_stats = simulated_distributions.groupby('year')['landfall_count'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('q05', lambda x: np.percentile(x, 5)),
        ('q25', lambda x: np.percentile(x, 25)),
        ('q75', lambda x: np.percentile(x, 75)),
        ('q95', lambda x: np.percentile(x, 95))
    ]).reset_index()
    
    # Merge with observations
    merged = observed.merge(sim_stats, left_on='Year', right_on='year', how='inner')
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Panel 1: Time series with uncertainty bands
    ax1 = axes[0]
    years = merged['Year']
    
    ax1.fill_between(years, merged['q05'], merged['q95'], 
                     alpha=0.2, color='blue', label='5-95% range')
    ax1.fill_between(years, merged['q25'], merged['q75'], 
                     alpha=0.3, color='blue', label='25-75% range')
    ax1.plot(years, merged['mean'], 'b-', linewidth=2, label='Simulated mean')
    ax1.plot(years, merged['landfall_count'], 'ko-', linewidth=1.5, 
             markersize=4, label='Observed')
    
    # Add climatology line
    clim_mean = observed['landfall_count'].mean()
    ax1.axhline(y=clim_mean, color='gray', linestyle='--', linewidth=1.5, 
                label=f'Climatology ({clim_mean:.1f})', alpha=0.7)
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Landfall Count', fontsize=12)
    ax1.set_title(f'Observed vs. Simulated Landfall Counts ({threshold_name})', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Histogram comparison
    ax2 = axes[1]
    
    obs_counts = observed['landfall_count'].values
    sim_counts = simulated_distributions['landfall_count'].values
    
    max_count = max(obs_counts.max(), sim_counts.max())
    bins = np.arange(0, max_count + 2) - 0.5
    
    ax2.hist(obs_counts, bins=bins, alpha=0.6, label='Observed', 
             density=True, color='black', edgecolor='black')
    ax2.hist(sim_counts, bins=bins, alpha=0.4, label='Simulated (SCILS)', 
             density=True, color='blue')
    
    # Add NB distribution if parameters provided
    if nb_params is not None:
        n, p = nb_params
        x_vals = np.arange(0, max_count + 2)
        nb_pmf = nbinom.pmf(x_vals, n, p)
        ax2.plot(x_vals, nb_pmf, 'r-', linewidth=2, marker='o', markersize=5,
                 label=f'Null NB (n={n:.2f}, p={p:.3f})')
    
    ax2.set_xlabel('Landfall Count', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title(f'Distribution of Landfall Counts ({threshold_name})', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    threshold_suffix = threshold_name.replace('+', 'plus')
    output_path = output_dir / f'validation_plots_{threshold_suffix}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"    Validation plots saved to: {output_path}")


def validate_for_threshold(threshold_name, threshold_value, simulation_file, region,
                           clim_start, clim_end, output_dir, start_year, end_year):
    """
    Run validation for a single intensity threshold.
    
    Returns
    -------
    dict
        Dictionary of metrics for this threshold
    """
    print(f"\n{'='*70}")
    print(f"INTENSITY THRESHOLD: {threshold_name} (≥{threshold_value} m/s)")
    print(f"{'='*70}")
    
    # Load data for this threshold
    observed = load_historical_landfalls(
        region=region, 
        start_year=start_year, 
        end_year=end_year,
        intensity_threshold=threshold_value
    )
    
    simulated = load_simulation_results(
        simulation_file, 
        region=region,
        intensity_threshold=threshold_value
    )
    
    print(f"\n  Observed: {len(observed)} years, total landfalls: {observed['landfall_count'].sum()}")
    print(f"  Mean observed landfalls/year: {observed['landfall_count'].mean():.2f}")
    print(f"  Simulated: {simulated['iteration'].nunique()} iterations per year")
    
    # Calculate metrics
    metrics = {'threshold': threshold_name}
    
    # 1. RMSE
    print("\n  [1] RMSE Metrics:")
    rmse_results, rmse_df = calculate_rmse(observed, simulated)
    metrics['climatology_rmse'] = rmse_results['climatology_rmse']
    metrics['model_rmse'] = rmse_results['model_rmse']
    metrics['skill_score'] = rmse_results['skill_score']
    metrics['obs_mean'] = rmse_results['obs_mean']
    metrics['sim_mean'] = rmse_results['sim_mean']
    metrics['bias'] = rmse_results['bias']
    print(f"      Observed mean: {rmse_results['obs_mean']:.4f} landfalls/year")
    print(f"      Simulated mean: {rmse_results['sim_mean']:.4f} landfalls/year")
    print(f"      Bias (model - obs): {rmse_results['bias']:.4f} ({'over' if rmse_results['bias'] > 0 else 'under'}estimation)")
    print(f"      Climatology RMSE (baseline): {rmse_results['climatology_rmse']:.4f}")
    print(f"      Model RMSE: {rmse_results['model_rmse']:.4f}")
    print(f"      Skill Score: {rmse_results['skill_score']:.4f} (1=perfect, 0=no skill)")
    
    # 2. CRPS (SCILS)
    print("\n  [2] CRPS:")
    mean_crps, crps_df = calculate_crps_for_years(observed, simulated)
    metrics['crps_scils'] = mean_crps
    print(f"      SCILS Mean CRPS: {mean_crps:.4f} (lower is better)")
    
    # 2b. CRPS (Null NB model)
    mean_crps_nb, crps_nb_df, nb_params = calculate_crps_null_nb(observed)
    metrics['crps_nb'] = mean_crps_nb
    metrics['nb_n'] = nb_params[0]
    metrics['nb_p'] = nb_params[1]
    print(f"      Null NB Mean CRPS: {mean_crps_nb:.4f}")
    print(f"      (NB params: n={nb_params[0]:.2f}, p={nb_params[1]:.4f})")
    
    crps_skill = 1 - (mean_crps / mean_crps_nb) if mean_crps_nb > 0 else 0
    metrics['crps_skill'] = crps_skill
    print(f"      CRPS Skill Score: {crps_skill:.4f}")
    
    # 2c. 95% PI Coverage for LFI
    print("\n  [2c] 95% PI Coverage (LFI):")
    pi95_lfi_scils = calculate_95pi_coverage(observed, simulated, count_col='landfall_count')
    metrics['lfi_95pi_coverage_scils'] = pi95_lfi_scils['coverage']
    print(f"      SCILS: {pi95_lfi_scils['coverage']:.1f}% ({pi95_lfi_scils['n_covered']}/{pi95_lfi_scils['n_total']} years)")
    
    pi95_lfi_nb = calculate_95pi_coverage_nb(observed, nb_params[0], nb_params[1], count_col='landfall_count')
    metrics['lfi_95pi_coverage_nb'] = pi95_lfi_nb['coverage']
    print(f"      Null NB: {pi95_lfi_nb['coverage']:.1f}% ({pi95_lfi_nb['n_covered']}/{pi95_lfi_nb['n_total']} years) [PI: {pi95_lfi_nb['q025']:.0f}-{pi95_lfi_nb['q975']:.0f}]")
    
    # 3. Log-likelihood (SCILS)
    print("\n  [3] Log-Likelihood (SCILS):")
    ll_scils, ll_scils_df = calculate_log_likelihood_scils(observed, simulated)
    metrics['ll_scils'] = ll_scils
    print(f"      Total: {ll_scils:.4f}")
    print(f"      Per year: {ll_scils / len(observed):.4f}")
    
    # 4. Log-likelihood (Static empirical)
    print("\n  [4] Log-Likelihood (Static Empirical):")
    ll_static, ll_static_df = calculate_log_likelihood_static(observed, clim_start, clim_end)
    metrics['ll_static'] = ll_static
    print(f"      Total: {ll_static:.4f}")
    print(f"      Per year: {ll_static / len(observed):.4f}")
    
    # 5. Log-likelihood (Null NB)
    print("\n  [5] Log-Likelihood (Null NB):")
    ll_nb, ll_nb_df, _ = calculate_log_likelihood_nb(observed, clim_start, clim_end)
    metrics['ll_nb'] = ll_nb
    print(f"      Total: {ll_nb:.4f}")
    print(f"      Per year: {ll_nb / len(observed):.4f}")
    
    # 6. Likelihood Ratios
    print("\n  [6] Likelihood Ratios:")
    
    lr_vs_static = ll_scils - ll_static
    metrics['lr_vs_static'] = lr_vs_static
    print(f"      LR(SCILS vs Static): {lr_vs_static:.4f}", end="")
    print(f" → {'SCILS better' if lr_vs_static > 0 else 'Static better'}")
    
    lr_vs_nb = ll_scils - ll_nb
    metrics['lr_vs_nb'] = lr_vs_nb
    print(f"      LR(SCILS vs NB):     {lr_vs_nb:.4f}", end="")
    print(f" → {'SCILS better' if lr_vs_nb > 0 else 'NB better'}")
    
    # 7. LMI Statistics
    print("\n  [7] LMI Statistics:")
    
    # Load LMI data for this threshold
    observed_lmi = load_historical_lmi(
        start_year=start_year,
        end_year=end_year,
        intensity_threshold=threshold_value
    )
    
    simulated_lmi = load_simulation_lmi(
        simulation_file,
        intensity_threshold=threshold_value
    )
    
    if len(observed_lmi) > 0 and len(simulated_lmi) > 0:
        lmi_stats = calculate_lmi_statistics(observed_lmi, simulated_lmi)
        metrics['lmi_obs_mean'] = lmi_stats['lmi_obs_mean']
        metrics['lmi_sim_mean'] = lmi_stats['lmi_sim_mean']
        metrics['lmi_bias'] = lmi_stats['lmi_bias']
        metrics['lmi_rmse'] = lmi_stats['lmi_rmse']
        metrics['lmi_crps'] = lmi_stats['lmi_crps']
        
        total_obs = observed_lmi['lmi_count'].sum()
        print(f"      Observed LMI storms/year: {lmi_stats['lmi_obs_mean']:.4f} (total: {total_obs})")
        print(f"      Simulated LMI storms/year: {lmi_stats['lmi_sim_mean']:.4f}")
        print(f"      Bias (model - obs): {lmi_stats['lmi_bias']:.4f} ({'over' if lmi_stats['lmi_bias'] > 0 else 'under'}estimation)")
        print(f"      RMSE: {lmi_stats['lmi_rmse']:.4f}")
        print(f"      CRPS: {lmi_stats['lmi_crps']:.4f} (lower is better)")
        
        # 95% PI Coverage for LMI
        print("\n  [7b] 95% PI Coverage (LMI):")
        pi95_lmi_scils = calculate_95pi_coverage(observed_lmi, simulated_lmi, count_col='lmi_count')
        metrics['lmi_95pi_coverage_scils'] = pi95_lmi_scils['coverage']
        print(f"      SCILS: {pi95_lmi_scils['coverage']:.1f}% ({pi95_lmi_scils['n_covered']}/{pi95_lmi_scils['n_total']} years)")
        
        # Fit NB to LMI observations for comparison
        lmi_nb_n, lmi_nb_p = fit_negative_binomial(observed_lmi['lmi_count'].values)
        pi95_lmi_nb = calculate_95pi_coverage_nb(observed_lmi, lmi_nb_n, lmi_nb_p, count_col='lmi_count')
        metrics['lmi_95pi_coverage_nb'] = pi95_lmi_nb['coverage']
        print(f"      Null NB: {pi95_lmi_nb['coverage']:.1f}% ({pi95_lmi_nb['n_covered']}/{pi95_lmi_nb['n_total']} years) [PI: {pi95_lmi_nb['q025']:.0f}-{pi95_lmi_nb['q975']:.0f}]")
    else:
        print(f"      No storms meeting threshold {threshold_value} m/s")
        metrics['lmi_obs_mean'] = np.nan
        metrics['lmi_sim_mean'] = np.nan
        metrics['lmi_bias'] = np.nan
        metrics['lmi_rmse'] = np.nan
        metrics['lmi_crps'] = np.nan
        metrics['lmi_95pi_coverage_scils'] = np.nan
        metrics['lmi_95pi_coverage_nb'] = np.nan
    
    # Generate plots (pass NB params for distribution overlay)
    plot_validation_results(observed, simulated, output_dir, threshold_name, nb_params=nb_params)
    
    return metrics


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 70)
    print("SCILS TC Model - Simulation Validation")
    print("=" * 70)
    print(f"Simulation file: {args.simulation_file}")
    print(f"Region: {args.region}")
    
    # Load simulation to get year range
    simulation_file = Path(args.simulation_file)
    
    if not simulation_file.exists():
        print(f"Error: Simulation file not found: {simulation_file}")
        sys.exit(1)
    
    # Quick load to get year range
    df_quick = pd.read_csv(simulation_file, usecols=['year', 'iteration'])
    start_year = df_quick['year'].min()
    end_year = df_quick['year'].max()
    n_iterations = df_quick['iteration'].nunique()
    
    print("\nSimulation info:")
    print(f"  Years: {start_year}-{end_year}")
    print(f"  Iterations: {n_iterations}")
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else simulation_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Climatology period
    clim_start = args.climatology_start if args.climatology_start else start_year
    clim_end = args.climatology_end if args.climatology_end else end_year
    print(f"  Climatology period: {clim_start}-{clim_end}")
    
    # Validate for each intensity threshold
    all_metrics = []
    
    for threshold_name, threshold_value in INTENSITY_THRESHOLDS.items():
        metrics = validate_for_threshold(
            threshold_name=threshold_name,
            threshold_value=threshold_value,
            simulation_file=simulation_file,
            region=args.region,
            clim_start=clim_start,
            clim_end=clim_end,
            output_dir=output_dir,
            start_year=start_year,
            end_year=end_year
        )
        all_metrics.append(metrics)
    
    # Save combined summary
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL THRESHOLDS")
    print("=" * 70)
    
    summary_df = pd.DataFrame(all_metrics)
    summary_df = summary_df[['threshold', 
                            'obs_mean', 'sim_mean', 'bias',
                            'climatology_rmse', 'model_rmse', 'skill_score',
                            'crps_scils', 'crps_nb', 'crps_skill',
                            'lfi_95pi_coverage_scils', 'lfi_95pi_coverage_nb',
                            'll_scils', 'll_static', 'll_nb', 
                            'lr_vs_static', 'lr_vs_nb',
                            'lmi_obs_mean', 'lmi_sim_mean', 'lmi_bias', 'lmi_rmse', 'lmi_crps',
                            'lmi_95pi_coverage_scils', 'lmi_95pi_coverage_nb']]
    
    print("\n" + summary_df.to_string(index=False))
    
    summary_path = output_dir / 'validation_summary_all_thresholds.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
