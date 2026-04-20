"""
Plotting functions for YELT resampling analysis.

This module provides visualization functions for comparing original and
resampled YELTs.
"""

import matplotlib.pyplot as plt
import numpy as np

import config


def plot_yelt_comparison(orig_yelt, adjusted_yelts, comparison_results,
                         method='deterministic', base_label='historical',
                         target_label='future', output_path=None):
    """
    Create a figure comparing original and adjusted YELTs.
    
    Subplot a) OEP curves for original and adjusted YELTs
    Subplot b) Loss ratios at AAL and different return periods
    
    Parameters
    ----------
    orig_yelt : pd.DataFrame
        Original YELT.
    adjusted_yelts : list of pd.DataFrame
        List of adjusted YELTs.
    comparison_results : dict
        Results from compare_yelts function.
    method : str
        'deterministic' or 'poisson'
    base_label : str
        Label for base period.
    target_label : str
        Label for target period.
    output_path : Path, optional
        Path to save the figure.
        
    Returns
    -------
    fig : plt.Figure
        The generated figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors
    orig_color = '#1f77b4'  # Blue
    adj_color = '#ff7f0e'   # Orange
    
    # Return periods for subplot b
    return_periods = [2, 5, 10, 20, 50, 100, 200, 500]
    x_labels = ['AAL'] + [f'{rp}yr' for rp in return_periods]
    
    # ----- Subplot a) OEP Curves -----
    # Original OEP curve - plot return period (1/prob) on x-axis
    orig_oep = comparison_results['original']['oep_curve']
    orig_return_periods = 1.0 / orig_oep['exceedance_probs']
    ax1.plot(orig_return_periods, orig_oep['sorted_losses'],
             color=orig_color, linewidth=2, label=f'Original ({base_label})')
    
    if method == 'poisson':
        # Multiple adjusted curves with transparency
        for i, adj_metrics in enumerate(comparison_results['adjusted']):
            adj_oep = adj_metrics['oep_curve']
            adj_return_periods = 1.0 / adj_oep['exceedance_probs']
            alpha = min(0.5, 10 / len(adjusted_yelts))  # Adjust transparency
            label = f'Adjusted ({target_label})' if i == 0 else None
            ax1.plot(adj_return_periods, adj_oep['sorted_losses'],
                     color=adj_color, alpha=alpha, linewidth=0.8, label=label)
    else:
        # Single adjusted curve
        adj_oep = comparison_results['adjusted'][0]['oep_curve']
        adj_return_periods = 1.0 / adj_oep['exceedance_probs']
        ax1.plot(adj_return_periods, adj_oep['sorted_losses'],
                 color=adj_color, linewidth=2, label=f'Adjusted ({target_label})')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Return Period (years)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('a) Occurrence Exceedance Probability (OEP) Curves', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 10000)  # Return period from 1 to 10000 years
    
    # ----- Subplot b) Loss Ratios -----
    x = np.arange(len(x_labels))
    
    # Prepare data
    aal_ratios = comparison_results['aal_ratios']
    rp_ratios = comparison_results['rp_ratios']
    
    if method == 'poisson':
        # Violin plots for Poisson method
        all_ratios = [aal_ratios] + [rp_ratios[rp] for rp in return_periods]
        
        # Filter out NaN values
        valid_data = []
        valid_positions = []
        for i, data in enumerate(all_ratios):
            clean_data = [v for v in data if not np.isnan(v)]
            if len(clean_data) > 0:
                valid_data.append(clean_data)
                valid_positions.append(i)
        
        if valid_data:
            parts = ax2.violinplot(valid_data, positions=valid_positions, showmeans=True, showmedians=True)
            
            # Color the violins
            for pc in parts['bodies']:
                pc.set_facecolor(adj_color)
                pc.set_alpha(0.6)
            
            # Add mean markers
            means = [np.mean(d) for d in valid_data]
            ax2.scatter(valid_positions, means, color='red', s=50, zorder=3, label='Mean')
    else:
        # Bar chart for deterministic method
        ratios = [aal_ratios[0]] + [rp_ratios[rp][0] for rp in return_periods]
        bars = ax2.bar(x, ratios, color=adj_color, alpha=0.8)
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax2.annotate(f'{ratio:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Add horizontal line at 1.0 (no change)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='No Change')
    
    ax2.set_xlabel('Return Period', fontsize=12)
    ax2.set_ylabel('Loss Ratio (Adjusted / Original)', fontsize=12)
    ax2.set_title('b) Loss Ratios at Different Return Periods', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Suptitle
    fig.suptitle(f'YELT Comparison: {base_label} → {target_label} ({method.capitalize()} Method)',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save or show
    if output_path is not None:
        config.ensure_resampled_dir()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"YELT comparison plot saved to: {output_path}")
    
    return fig


def plot_category_event_counts(orig_yelt, adjusted_yelts, method='deterministic',
                                base_label='historical', target_label='future',
                                output_path=None):
    """
    Plot event counts per SSHS category for original and adjusted YELTs.
    
    Parameters
    ----------
    orig_yelt : pd.DataFrame
        Original YELT with LFI_SSHS column.
    adjusted_yelts : list of pd.DataFrame
        List of adjusted YELTs.
    method : str
        'deterministic' or 'poisson'
    base_label : str
        Label for base period.
    target_label : str
        Label for target period.
    output_path : Path, optional
        Path to save the figure.
        
    Returns
    -------
    fig : plt.Figure
        The generated figure.
    """
    from .yelt_resampling import SSHS_CATEGORIES
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(SSHS_CATEGORIES))
    width = 0.35
    
    # Original counts
    orig_counts = []
    for cat in SSHS_CATEGORIES:
        count = (orig_yelt['LFI_SSHS'] == cat).sum()
        orig_counts.append(count)
    
    # Adjusted counts
    if method == 'poisson':
        # Box plot for multiple adjusted YELTs
        adj_counts_matrix = []
        for adj_yelt in adjusted_yelts:
            counts = [(adj_yelt['LFI_SSHS'] == cat).sum() for cat in SSHS_CATEGORIES]
            adj_counts_matrix.append(counts)
        adj_counts_matrix = np.array(adj_counts_matrix)
        
        # Plot original as bars
        ax.bar(x - width/2, orig_counts, width, label=f'Original ({base_label})',
               color='#1f77b4', alpha=0.8)
        
        # Plot adjusted as box plots
        bp = ax.boxplot([adj_counts_matrix[:, i] for i in range(len(SSHS_CATEGORIES))],
                        positions=x + width/2, widths=width*0.8, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#ff7f0e')
            patch.set_alpha(0.6)
    else:
        # Side-by-side bars for deterministic
        adj_yelt = adjusted_yelts[0]
        adj_counts = [(adj_yelt['LFI_SSHS'] == cat).sum() for cat in SSHS_CATEGORIES]
        
        ax.bar(x - width/2, orig_counts, width, label=f'Original ({base_label})',
               color='#1f77b4', alpha=0.8)
        ax.bar(x + width/2, adj_counts, width, label=f'Adjusted ({target_label})',
               color='#ff7f0e', alpha=0.8)
    
    ax.set_xlabel('SSHS Category', fontsize=12)
    ax.set_ylabel('Event Count', fontsize=12)
    ax.set_title(f'Landfalling Event Counts by Category ({method.capitalize()})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(SSHS_CATEGORIES)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is not None:
        config.ensure_resampled_dir()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Category event counts plot saved to: {output_path}")
    
    return fig
