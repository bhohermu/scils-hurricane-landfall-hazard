"""
Plotting functions for simulation validation.

This module provides plots comparing simulated events with observations.
"""

import matplotlib.pyplot as plt
import numpy as np

from scils_tc.utils import actual_event_rows, year_iteration_grid_from_dataframe


def ms_to_category(wind_ms):
    """Convert wind speed in m/s to Saffir-Simpson category."""
    wind_kt = wind_ms * 1.94384  # m/s to knots
    if wind_kt < 34:
        return 0  # TD
    elif wind_kt < 64:
        return 0  # TS
    elif wind_kt < 83:
        return 1  # Cat 1
    elif wind_kt < 96:
        return 2  # Cat 2
    elif wind_kt < 113:
        return 3  # Cat 3
    elif wind_kt < 137:
        return 4  # Cat 4
    else:
        return 5  # Cat 5


def calculate_crps(observations, ensemble):
    """
    Calculate Continuous Ranked Probability Score (CRPS).
    
    Formula: CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent samples from forecast, y is observation.
    
    Parameters
    ----------
    observations : array-like
        Observed values (one per time point)
    ensemble : array-like (2D)
        Ensemble forecasts (rows = ensemble members, cols = time points)
        
    Returns
    -------
    float
        Mean CRPS across all time points
    """
    observations = np.array(observations)
    ensemble = np.array(ensemble)
    
    crps_values = []
    for i, obs in enumerate(observations):
        ens = ensemble[:, i]
        n = len(ens)
        
        if n == 0:
            continue
        if n == 1:
            crps_values.append(np.abs(ens[0] - obs))
            continue
        
        # Term 1: E|X - y| = mean absolute error between ensemble and observation
        term1 = np.mean(np.abs(ens - obs))
        
        # Term 2: E|X - X'| = mean of all pairwise absolute differences
        # For sorted samples x_1 <= x_2 <= ... <= x_n, this can be computed efficiently:
        # E|X - X'| = (2 / n^2) * sum_{i=1}^{n} x_i * (2i - n - 1)
        # where i is 1-indexed
        x_sorted = np.sort(ens)
        indices = np.arange(1, n + 1)  # 1-indexed: 1, 2, ..., n
        term2 = (2.0 / (n * n)) * np.sum(x_sorted * (2 * indices - n - 1))
        
        crps = term1 - 0.5 * term2
        crps_values.append(crps)
    
    return np.mean(crps_values) if crps_values else np.nan


def plot_simulation_validation(sim_df, obs_df, start_year, end_year, save_path=None, region='CONUS'):
    """
    Plot simulation validation comparing simulated landfalls with observations.
    
    Creates 3 subplots:
    - All storms with TS+ intensity at landfall
    - Cat 1+ storms
    - Cat 3+ (major hurricanes)
    
    Parameters
    ----------
    sim_df : pd.DataFrame
        Simulated events with columns: year, iteration, lfi
    obs_df : pd.DataFrame
        Observed IBTrACS properties with columns: Year, CONUS_LFI_ms or NorthAtlantic_LFI_ms
    start_year : int
        First year to plot
    end_year : int
        Last year to plot
    save_path : Path or str, optional
        Path to save the figure
    region : str, default='CONUS'
        Region for landfall analysis ('CONUS' or 'NorthAtlantic')
        
    Returns
    -------
    plt.Figure
    """
    years = np.arange(start_year, end_year + 1)
    
    # Select appropriate LFI column based on region
    lfi_column = f'{region}_LFI_ms'
    
    # Filter observations to landfalls only (LFI > 0)
    obs_landfalls = obs_df[obs_df[lfi_column] > 0].copy()
    
    # Define intensity categories
    categories = [
        ('TS+', 17.5),  # 34 knots = 17.5 m/s (TS threshold)
        ('Cat 1+', 33.0),  # 64 knots = 33.0 m/s (H1 threshold)
        ('Cat 3+', 50.0),  # 96 knots = 49.4 m/s (H3 threshold, use 50 for round number)
    ]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for idx, (cat_name, threshold) in enumerate(categories):
        ax = axes[idx]
        
        # Calculate observed counts per year
        obs_counts = []
        for year in years:
            year_obs = obs_landfalls[obs_landfalls['Year'] == year]
            count = np.sum(year_obs[lfi_column] >= threshold)
            obs_counts.append(count)
        obs_counts = np.array(obs_counts)
        
        # Calculate simulated statistics per year
        sim_means = []
        sim_q025 = []
        sim_q975 = []
        sim_q25 = []
        sim_q75 = []
        sim_ensemble = []  # For CRPS calculation
        
        for year in years:
            year_sim = sim_df[sim_df['year'] == year]
            actual_year_sim = actual_event_rows(year_sim)
            year_iter_grid = year_iteration_grid_from_dataframe(year_sim, years=[year])
            
            # Count landfalls per iteration
            iteration_counts = []
            for iter_num in year_iter_grid['iteration'].tolist():
                iter_data = actual_year_sim[actual_year_sim['iteration'] == iter_num]
                # Count storms with LFI >= threshold (LFI > 0 means landfall occurred)
                count = np.sum((iter_data['lfi'] >= threshold) & (iter_data['lfi'] > 0))
                iteration_counts.append(count)
            
            if len(iteration_counts) > 0:
                sim_means.append(np.mean(iteration_counts))
                sim_q025.append(np.percentile(iteration_counts, 2.5))
                sim_q975.append(np.percentile(iteration_counts, 97.5))
                sim_q25.append(np.percentile(iteration_counts, 25))
                sim_q75.append(np.percentile(iteration_counts, 75))
                sim_ensemble.append(iteration_counts)
            else:
                sim_means.append(0)
                sim_q025.append(0)
                sim_q975.append(0)
                sim_q25.append(0)
                sim_q75.append(0)
                sim_ensemble.append([0])
        
        sim_means = np.array(sim_means)
        sim_q025 = np.array(sim_q025)
        sim_q975 = np.array(sim_q975)
        sim_q25 = np.array(sim_q25)
        sim_q75 = np.array(sim_q75)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((obs_counts - sim_means)**2))
        
        # CRPS calculation
        max_iters = max(len(e) for e in sim_ensemble)
        # Pad ensemble to same length (repeat last value if needed)
        sim_ensemble_padded = np.array([
            np.pad(e, (0, max_iters - len(e)), mode='edge') 
            for e in sim_ensemble
        ]).T
        crps = calculate_crps(obs_counts, sim_ensemble_padded)
        
        # Percent of years in 95% range
        in_range = np.sum((obs_counts >= sim_q025) & (obs_counts <= sim_q975))
        pct_in_range = 100 * in_range / len(years)
        
        # Plot 95% confidence interval
        ax.fill_between(years, sim_q025, sim_q975, alpha=0.2, color='blue', 
                        label='95% range')
        
        # Plot IQR
        ax.fill_between(years, sim_q25, sim_q75, alpha=0.4, color='blue',
                        label='IQR')
        
        # Plot mean
        ax.plot(years, sim_means, 'b-', linewidth=2, label='Simulated mean')
        
        # Plot observations as dots
        ax.scatter(years, obs_counts, c='red', s=50, zorder=5, 
                  label='Observations', edgecolors='darkred', linewidth=1)
        
        # Add metrics text box
        metrics_text = (f'RMSE = {rmse:.2f}\n'
                       f'CRPS = {crps:.2f}\n'
                       f'{pct_in_range:.0f}% in 95% range')
        ax.text(0.98, 0.97, metrics_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, family='monospace')
        
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Number of Landfalls', fontsize=11)
        ax.set_title(f'{cat_name} {region} Landfalls', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(start_year - 1, end_year + 1)
    
    plt.suptitle('Simulation Validation: Simulated vs Observed Landfalls',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Validation plot saved to: {save_path}")
    
    return fig
