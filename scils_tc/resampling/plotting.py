"""
Plotting functions for resampling analysis.

This module provides visualization functions for landfall rate comparison
between base and target periods.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config

from .resampling import SSHS_CATEGORIES


def plot_rate_comparison(base_rates, target_rates, change_rates,
                         base_label, target_label, region,
                         base_enso='all', target_enso='all',
                         output_path=None):
    """
    Create a figure with two subplots showing rate comparison.
    
    Subplot a) Absolute landfall rates for base and target (side-by-side bars)
    Subplot b) Rate change factors per category
    
    Parameters
    ----------
    base_rates : dict
        Base period landfall rates per SSHS category
    target_rates : dict
        Target period landfall rates per SSHS category
    change_rates : dict
        Change rates (target/base) per SSHS category
    base_label : str
        Label for base period (e.g., '1998-2009')
    target_label : str
        Label for target period (e.g., '2050' or 'GWL2.0')
    region : str
        Region name
    base_enso : str
        ENSO filter for base period
    target_enso : str
        ENSO filter for target period
    output_path : Path, optional
        Path to save the figure. If None, shows interactively.
        
    Returns
    -------
    fig : plt.Figure
        The generated figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Setup common elements
    x = np.arange(len(SSHS_CATEGORIES))
    width = 0.35
    
    base_values = [base_rates.get(cat, 0) for cat in SSHS_CATEGORIES]
    target_values = [target_rates.get(cat, 0) for cat in SSHS_CATEGORIES]
    change_values = [change_rates.get(cat, 1.0) for cat in SSHS_CATEGORIES]
    
    # Colors
    base_color = '#1f77b4'  # Blue
    target_color = '#ff7f0e'  # Orange
    change_color = '#2ca02c'  # Green
    
    # ----- Subplot a) Absolute Rates -----
    bars1 = ax1.bar(x - width/2, base_values, width, label=f'Base: {base_label}',
                    color=base_color, alpha=0.8)
    bars2 = ax1.bar(x + width/2, target_values, width, label=f'Target: {target_label}',
                    color=target_color, alpha=0.8)
    
    ax1.set_xlabel('SSHS Category', fontsize=12)
    ax1.set_ylabel('Annual Landfall Rate', fontsize=12)
    ax1.set_title(f'a) Absolute Landfall Rates - {region}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(SSHS_CATEGORIES)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_bar_labels(ax, bars, fmt='{:.2f}'):
        """Annotate each non-zero bar with its numeric value."""
        for bar in bars:
            height = bar.get_height()
            if height > 0.001:  # Only label non-zero bars
                ax.annotate(fmt.format(height),
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    add_bar_labels(ax1, bars1)
    add_bar_labels(ax1, bars2)
    
    # ----- Subplot b) Change Rates -----
    # Handle infinite values for plotting
    plot_change_values = []
    for v in change_values:
        if np.isinf(v):
            plot_change_values.append(5.0)  # Cap for visualization
        else:
            plot_change_values.append(v)
    
    bars3 = ax2.bar(x, plot_change_values, width * 1.5, color=change_color, alpha=0.8)
    
    # Add horizontal line at 1.0 (no change)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='No Change')
    
    ax2.set_xlabel('SSHS Category', fontsize=12)
    ax2.set_ylabel('Rate Change Factor (Target / Base)', fontsize=12)
    ax2.set_title(f'b) Landfall Rate Change Factors - {region}', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(SSHS_CATEGORIES)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Color bars based on increase/decrease
    for bar, val in zip(bars3, change_values):
        if np.isinf(val):
            bar.set_color('#d62728')  # Red for infinite
            bar.set_hatch('//')
        elif val > 1.0:
            bar.set_color('#d62728')  # Red for increase
            bar.set_alpha(0.6 + 0.2 * min((val - 1) / 2, 1))  # Darker for bigger increase
        elif val < 1.0:
            bar.set_color('#2ca02c')  # Green for decrease
            bar.set_alpha(0.6 + 0.2 * min((1 - val) / 0.5, 1))  # Darker for bigger decrease
        else:
            bar.set_color('gray')  # Gray for no change
    
    # Add value labels
    for bar, val in zip(bars3, change_values):
        height = bar.get_height()
        label = '∞' if np.isinf(val) else f'{val:.2f}'
        ax2.annotate(label,
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Suptitle with ENSO info
    enso_info = ""
    if base_enso != 'all' or target_enso != 'all':
        if base_enso == target_enso:
            enso_info = f" (ENSO: {base_enso})"
        else:
            enso_info = f" (Base ENSO: {base_enso}, Target ENSO: {target_enso})"
    
    fig.suptitle(f'Landfall Rate Comparison: {base_label} → {target_label}{enso_info}',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save or show
    if output_path is not None:
        config.ensure_resampled_dir()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    return fig


def plot_rate_comparison_poisson(base_rates, target_rates, prescribed_change_rates,
                                 realized_change_rates_list,
                                 base_label, target_label, region,
                                 output_path=None):
    """
    Create a figure with two subplots for Poisson resampling results.
    
    Subplot a) Absolute landfall rates (base vs target, side-by-side bars)
    Subplot b) Violin plot of *realized* change rates across Poisson iterations,
               with the prescribed (input) change rate overlaid as markers.
    
    Parameters
    ----------
    base_rates : dict
        Base period landfall rates per SSHS category.
    target_rates : dict
        Target period landfall rates per SSHS category.
    prescribed_change_rates : dict
        Prescribed change rates (from CSV input) per SSHS category.
    realized_change_rates_list : list of dict
        One dict per iteration mapping SSHS category → realized change rate.
    base_label, target_label : str
        Labels for base / target periods.
    region : str
        Region name.
    output_path : Path, optional
        Save path. If None, shows interactively.
    
    Returns
    -------
    fig : plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(SSHS_CATEGORIES))
    width = 0.35
    
    base_values = [base_rates.get(cat, 0) for cat in SSHS_CATEGORIES]
    target_values = [target_rates.get(cat, 0) for cat in SSHS_CATEGORIES]
    
    # Colors
    base_color = '#1f77b4'
    target_color = '#ff7f0e'
    
    # ----- Subplot a) Absolute Rates (same as deterministic) -----
    bars1 = ax1.bar(x - width/2, base_values, width, label=f'Base: {base_label}',
                    color=base_color, alpha=0.8)
    bars2 = ax1.bar(x + width/2, target_values, width, label=f'Target: {target_label}',
                    color=target_color, alpha=0.8)
    
    ax1.set_xlabel('SSHS Category', fontsize=12)
    ax1.set_ylabel('Annual Landfall Rate', fontsize=12)
    ax1.set_title(f'a) Absolute Landfall Rates - {region}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(SSHS_CATEGORIES)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    def add_bar_labels(ax, bars, fmt='{:.2f}'):
        """Annotate each non-zero bar with its numeric value."""
        for bar in bars:
            height = bar.get_height()
            if height > 0.001:
                ax.annotate(fmt.format(height),
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    add_bar_labels(ax1, bars1)
    add_bar_labels(ax1, bars2)
    
    # ----- Subplot b) Violin plot of realized change rates -----
    # Build data matrix: rows = iterations, cols = categories
    n_iter = len(realized_change_rates_list)
    data_per_cat = []
    for cat in SSHS_CATEGORIES:
        vals = [d.get(cat, 1.0) for d in realized_change_rates_list]
        data_per_cat.append(vals)
    
    prescribed_values = [prescribed_change_rates.get(cat, 1.0) for cat in SSHS_CATEGORIES]
    
    # Violin plot
    parts = ax2.violinplot(data_per_cat, positions=x, showmeans=True,
                           showmedians=False, showextrema=True)
    
    # Color violins based on whether prescribed rate > or < 1
    for i, pc in enumerate(parts['bodies']):
        pv = prescribed_values[i]
        if np.isinf(pv) or pv > 1.0:
            pc.set_facecolor('#d62728')  # Red for increase
        elif pv < 1.0:
            pc.set_facecolor('#2ca02c')  # Green for decrease
        else:
            pc.set_facecolor('gray')
        pc.set_alpha(0.6)
    
    # Style the lines
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1)
    
    # Overlay prescribed change rates as diamond markers
    ax2.scatter(x, prescribed_values, marker='D', color='black', s=60, zorder=5,
                label='Prescribed change rate')
    
    # No-change reference line
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='No Change')
    
    ax2.set_xlabel('SSHS Category', fontsize=12)
    ax2.set_ylabel('Realized Change Rate (Target / Base)', fontsize=12)
    ax2.set_title(f'b) Realized Change Rates ({n_iter} Poisson iterations)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(SSHS_CATEGORIES)
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add mean ± std annotations above each violin
    for i, cat in enumerate(SSHS_CATEGORIES):
        vals = data_per_cat[i]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        ax2.annotate(f'{mean_v:.2f}±{std_v:.2f}',
                    xy=(x[i], max(vals)),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    fig.suptitle(f'Landfall Rate Comparison: {base_label} → {target_label} (Poisson)',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_path is not None:
        config.ensure_resampled_dir()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    return fig


def plot_multi_region_comparison(results_dict, base_label, target_label, output_path=None):
    """
    Create a comparison plot for multiple regions.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping region names to (base_rates, target_rates, change_rates) tuples
    base_label : str
        Label for base period
    target_label : str
        Label for target period
    output_path : Path, optional
        Path to save the figure
        
    Returns
    -------
    fig : plt.Figure
        The generated figure
    """
    n_regions = len(results_dict)
    fig, axes = plt.subplots(n_regions, 2, figsize=(14, 5 * n_regions))
    
    if n_regions == 1:
        axes = axes.reshape(1, -1)
    
    x = np.arange(len(SSHS_CATEGORIES))
    width = 0.35
    
    for i, (region, (base_rates, target_rates, change_rates)) in enumerate(results_dict.items()):
        ax1, ax2 = axes[i]
        
        base_values = [base_rates.get(cat, 0) for cat in SSHS_CATEGORIES]
        target_values = [target_rates.get(cat, 0) for cat in SSHS_CATEGORIES]
        change_values = [change_rates.get(cat, 1.0) for cat in SSHS_CATEGORIES]
        
        # Subplot a) Absolute Rates
        ax1.bar(x - width/2, base_values, width, label=f'Base: {base_label}',
                color='#1f77b4', alpha=0.8)
        ax1.bar(x + width/2, target_values, width, label=f'Target: {target_label}',
                color='#ff7f0e', alpha=0.8)
        ax1.set_xlabel('SSHS Category')
        ax1.set_ylabel('Annual Landfall Rate')
        ax1.set_title(f'{region} - Absolute Rates')
        ax1.set_xticks(x)
        ax1.set_xticklabels(SSHS_CATEGORIES)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Subplot b) Change Rates
        plot_change = [5.0 if np.isinf(v) else v for v in change_values]
        bars = ax2.bar(x, plot_change, width * 1.5, color='#2ca02c', alpha=0.8)
        ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('SSHS Category')
        ax2.set_ylabel('Change Factor')
        ax2.set_title(f'{region} - Change Factors')
        ax2.set_xticks(x)
        ax2.set_xticklabels(SSHS_CATEGORIES)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, change_values):
            if val > 1.0 or np.isinf(val):
                bar.set_color('#d62728')
            elif val < 1.0:
                bar.set_color('#2ca02c')
    
    fig.suptitle(f'Landfall Rate Comparison: {base_label} → {target_label}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path is not None:
        config.ensure_resampled_dir()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Multi-region plot saved to: {output_path}")
    
    return fig
