"""
Plotting functions for detrending visualization.

This module provides plots for GWL regression and detrending comparisons.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import config

from .detrending import calculate_mdr_timeseries
from .gwl_lookup import (
    gwl_to_year,
    year_to_gwl,
)


def plot_gwl_regression(
    gwl_df: pd.DataFrame,
    gwl_regression: dict,
    gwl_df_fit: pd.DataFrame = None,
    target_year: float = None,
    target_gwl: float = None,
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot GWL timeseries with regression line.
    
    Parameters
    ----------
    gwl_df : pd.DataFrame
        DataFrame with 'Year' and 'GWL' columns (full data for plotting)
    gwl_regression : dict
        Regression coefficients from fit_gwl_regression()
    gwl_df_fit : pd.DataFrame, optional
        DataFrame with data used for fitting (to highlight fit period)
    target_year : float, optional
        Target year to highlight (computed from target_gwl if not provided)
    target_gwl : float, optional
        Target GWL to highlight
    save_path : Path, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot observed GWL (all data, lighter for out-of-fit period)
    if gwl_df_fit is not None:
        fit_start = gwl_df_fit['Year'].min()
        fit_end = gwl_df_fit['Year'].max()
        
        # Data outside fit period (lighter)
        outside_fit = gwl_df[(gwl_df['Year'] < fit_start) | (gwl_df['Year'] > fit_end)]
        ax.scatter(outside_fit['Year'], outside_fit['GWL'], 
                   color='steelblue', alpha=0.3, s=20, zorder=2)
        
        # Data inside fit period (darker)
        inside_fit = gwl_df[(gwl_df['Year'] >= fit_start) & (gwl_df['Year'] <= fit_end)]
        ax.scatter(inside_fit['Year'], inside_fit['GWL'], 
                   color='steelblue', alpha=0.8, s=30, 
                   label=f'Observed GWL (fit: {int(fit_start)}-{int(fit_end)})', zorder=2)
    else:
        ax.scatter(gwl_df['Year'], gwl_df['GWL'], 
                   color='steelblue', alpha=0.7, s=30, label='Observed GWL', zorder=2)
    
    # Plot regression line (extend from fit period to future)
    if gwl_df_fit is not None:
        fit_start = gwl_df_fit['Year'].min()
        years_range = np.linspace(fit_start, gwl_df['Year'].max() + 50, 100)
    else:
        years_range = np.linspace(gwl_df['Year'].min(), gwl_df['Year'].max() + 10, 100)
    
    gwl_fitted = gwl_regression['slope'] * years_range + gwl_regression['intercept']
    ax.plot(years_range, gwl_fitted, 'r-', linewidth=2, 
            label=f"Theil-Sen fit (slope={gwl_regression['slope']:.4f}°C/yr)", zorder=3)
    
    # Add 95% CI band - calculate proper CI using intercept at median x
    # The CI should be narrowest at the center of the data and widen towards the edges
    # For Theil-Sen, lo_slope/up_slope are the confidence bounds on slope
    # We need to calculate the intercept bounds too for proper CI visualization
    if gwl_df_fit is not None:
        x_median = gwl_df_fit['Year'].median()
    else:
        x_median = gwl_df['Year'].median()
    
    # At x_median, the regression line passes through (x_median, y_median_estimate)
    # CI expands as we move away from x_median
    y_at_median = gwl_regression['slope'] * x_median + gwl_regression['intercept']
    
    # Calculate lo/up lines that pivot around the median point
    gwl_lo = y_at_median + gwl_regression['lo_slope'] * (years_range - x_median)
    gwl_up = y_at_median + gwl_regression['up_slope'] * (years_range - x_median)
    
    ax.fill_between(years_range, gwl_lo, gwl_up, color='red', alpha=0.1, 
                    label='95% CI on slope', zorder=1)
    
    # Highlight target if specified
    if target_gwl is not None or target_year is not None:
        if target_year is None:
            target_year = gwl_to_year(target_gwl)
        if target_gwl is None:
            target_gwl = year_to_gwl(target_year)
        
        # Draw horizontal and vertical lines to target
        ax.axhline(target_gwl, color='green', linestyle='--', alpha=0.7, zorder=1)
        ax.axvline(target_year, color='green', linestyle='--', alpha=0.7, zorder=1)
        
        # Mark the target point
        ax.scatter([target_year], [target_gwl], color='green', s=100, marker='*', 
                   label=f'Target: {target_gwl:.2f}°C @ {target_year:.1f}', zorder=4)
        
        # Add text annotation
        ax.annotate(f'{target_gwl:.2f}°C\n{target_year:.1f}',
                    xy=(target_year, target_gwl),
                    xytext=(target_year + 5, target_gwl + 0.1),
                    fontsize=10, color='green',
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Global Warming Level (°C above pre-industrial)', fontsize=12)
    ax.set_title('Global Warming Level Timeseries with Theil-Sen Regression', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add data source
    ax.text(0.99, 0.01, 'Source: C3S Global Temperature Bulletin',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved GWL regression plot to {save_path}")
    
    return fig


def plot_detrending_comparison(
    original_sst: xr.DataArray = None,
    detrended_sst: xr.DataArray = None,
    original_pi: xr.DataArray = None,
    detrended_pi: xr.DataArray = None,
    original_cgi: xr.DataArray = None,
    detrended_cgi: xr.DataArray = None,
    target_year: float = None,
    target_gwl: float = None,
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot comparison of original and detrended MDR timeseries for SST, PI, and CGI.
    
    Parameters
    ----------
    original_sst, detrended_sst : xr.DataArray, optional
        Original and detrended SST data
    original_pi, detrended_pi : xr.DataArray, optional
        Original and detrended PI data
    original_cgi, detrended_cgi : xr.DataArray, optional
        Original and detrended CGI data
    target_year : float, optional
        Target year for labeling
    target_gwl : float, optional
        Target GWL for labeling
    save_path : Path, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    # Determine number of subplots based on what data is provided
    n_plots = sum([
        original_sst is not None,
        original_pi is not None,
        original_cgi is not None
    ])
    
    if n_plots == 0:
        print("  No data provided for detrending comparison plot")
        return None
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    ax_idx = 0
    
    # Determine label suffix
    if target_gwl is not None:
        label_suffix = f' (detrended to GWL={target_gwl:.1f}°C)'
    elif target_year is not None:
        label_suffix = f' (detrended to {int(target_year)})'
    else:
        label_suffix = ' (detrended)'
    
    # Plot SST
    if original_sst is not None:
        ax = axes[ax_idx]
        
        # Calculate MDR timeseries
        orig_ts = calculate_mdr_timeseries(original_sst, time_dim='time')
        
        # Plot original (semi-transparent)
        ax.plot(orig_ts['year'], orig_ts['value'] - 273.15, 
                'b-', linewidth=2, alpha=0.4, label='Original')
        ax.scatter(orig_ts['year'], orig_ts['value'] - 273.15, 
                   color='blue', alpha=0.4, s=30)
        
        if detrended_sst is not None:
            det_ts = calculate_mdr_timeseries(detrended_sst, time_dim='time')
            ax.plot(det_ts['year'], det_ts['value'] - 273.15, 
                    'b-', linewidth=2, label='Detrended' + label_suffix)
            ax.scatter(det_ts['year'], det_ts['value'] - 273.15, 
                       color='blue', s=30)
        
        ax.set_ylabel('MDR SST (°C)', fontsize=12)
        ax.set_title('MDR Sea Surface Temperature (ASO mean)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax_idx += 1
    
    # Plot PI
    if original_pi is not None:
        ax = axes[ax_idx]
        
        orig_ts = calculate_mdr_timeseries(original_pi, time_dim='time')
        
        ax.plot(orig_ts['year'], orig_ts['value'], 
                'r-', linewidth=2, alpha=0.4, label='Original')
        ax.scatter(orig_ts['year'], orig_ts['value'], 
                   color='red', alpha=0.4, s=30)
        
        if detrended_pi is not None:
            det_ts = calculate_mdr_timeseries(detrended_pi, time_dim='time')
            ax.plot(det_ts['year'], det_ts['value'], 
                    'r-', linewidth=2, label='Detrended' + label_suffix)
            ax.scatter(det_ts['year'], det_ts['value'], 
                       color='red', s=30)
        
        ax.set_ylabel('MDR PI (m/s)', fontsize=12)
        ax.set_title('MDR Potential Intensity (ASO mean)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax_idx += 1
    
    # Plot CGI (use full hurricane season June-November, scaled to storm counts)
    if original_cgi is not None:
        ax = axes[ax_idx]
        
        # Load CGI scaling factor from historical data
        # Use MDR_CGI_Sum (seasonal sum) if available, otherwise fall back to MDR_CGI
        cgi_df = pd.read_csv(config.PREPROCESSED_DIR / config.CGI_MDR_FILE)
        total_observed = cgi_df['Observed_Count'].sum()
        cgi_raw_col = 'MDR_CGI_Sum' if 'MDR_CGI_Sum' in cgi_df.columns else 'MDR_CGI'
        total_raw_cgi = cgi_df[cgi_raw_col].sum()
        cgi_scale_factor = total_observed / total_raw_cgi if total_raw_cgi > 0 else 1.0
        
        # Use aggregation='sum' for CGI (seasonal accumulated)
        orig_ts = calculate_mdr_timeseries(original_cgi, time_dim='time', months=config.SEASON_MONTHS, aggregation='sum')
        
        ax.plot(orig_ts['year'], orig_ts['value'] * cgi_scale_factor, 
                'g-', linewidth=2, alpha=0.4, label='Original')
        ax.scatter(orig_ts['year'], orig_ts['value'] * cgi_scale_factor, 
                   color='green', alpha=0.4, s=30)
        
        if detrended_cgi is not None:
            det_ts = calculate_mdr_timeseries(detrended_cgi, time_dim='time', months=config.SEASON_MONTHS, aggregation='sum')
            ax.plot(det_ts['year'], det_ts['value'] * cgi_scale_factor, 
                    'g-', linewidth=2, label='Detrended' + label_suffix)
            ax.scatter(det_ts['year'], det_ts['value'] * cgi_scale_factor, 
                       color='green', s=30)
        
        ax.set_ylabel('Scaled MDR CGI', fontsize=12)
        ax.set_title(f'MDR Cyclone Genesis Index (Jun-Nov sum, scale factor: {cgi_scale_factor:.2f})', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax_idx += 1
    
    # Set x-axis label on bottom plot
    axes[-1].set_xlabel('Year', fontsize=12)
    
    # Overall title
    if target_gwl is not None:
        fig.suptitle(f'Detrending Comparison (Target: GWL={target_gwl:.1f}°C, Year={target_year:.1f})', 
                     fontsize=14, y=1.02)
    elif target_year is not None:
        fig.suptitle(f'Detrending Comparison (Target Year: {int(target_year)})', 
                     fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved detrending comparison plot to {save_path}")
    
    return fig


def plot_diagnostic_mean_trend(
    data: xr.DataArray,
    regression: xr.Dataset,
    variable_name: str,
    units: str,
    time_dim: str = 'time',
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot diagnostic maps showing ASO mean and decadal trend.
    
    Creates a 2-row x 3-column plot:
    - Top row: Monthly means for Aug, Sep, Oct
    - Bottom row: Decadal trends (slope * 10) for Aug, Sep, Oct (hatched where significant)
    
    Parameters
    ----------
    data : xr.DataArray
        Original data with time dimension
    regression : xr.Dataset
        Dataset with 'slope' and 'intercept' from calculate_theilsen_regression()
    variable_name : str
        Name of the variable (e.g., 'SST', 'PI', 'CGI')
    units : str
        Units for the colorbar (e.g., '°C', 'm/s', '')
    time_dim : str
        Name of the time dimension
    save_path : Path, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from scipy import stats as scipy_stats
    
    # Month names for ASO
    months = {8: 'August', 9: 'September', 10: 'October'}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 9),
                              subplot_kw={'projection': ccrs.PlateCarree()})
    
    for col, month in enumerate([8, 9, 10]):
        # Filter data to this month
        month_data = data.sel(**{time_dim: data[time_dim].dt.month == month})
        
        # Calculate monthly mean across years
        mean_data = month_data.mean(dim=time_dim)
        
        # For trend, we need to calculate per-month regression
        # Get unique years
        times = month_data[time_dim].values
        if np.issubdtype(times.dtype, np.datetime64):
            years = pd.to_datetime(times).year.values.astype(float)
        else:
            years = times.astype(float)
        
        # Calculate slope and p-value for each grid cell for this month
        # Stack for efficiency
        stacked = month_data.stack(gridcell=month_data.dims[1:])
        
        slopes = np.empty(stacked.shape[1])
        pvalues = np.empty(stacked.shape[1])
        
        for i in range(stacked.shape[1]):
            cell_data = stacked.values[:, i]
            valid = ~np.isnan(cell_data)
            if valid.sum() >= 3:
                try:
                    result = scipy_stats.theilslopes(cell_data[valid], years[valid])
                    slopes[i] = result.slope
                    # Estimate p-value using Kendall's tau
                    tau, p = scipy_stats.kendalltau(years[valid], cell_data[valid])
                    pvalues[i] = p
                except:
                    slopes[i] = np.nan
                    pvalues[i] = np.nan
            else:
                slopes[i] = np.nan
                pvalues[i] = np.nan
        
        # Reshape back
        slope_da = stacked.isel(**{time_dim: 0}).copy()
        slope_da.values = slopes
        slope_da = slope_da.unstack('gridcell')
        
        pvalue_da = stacked.isel(**{time_dim: 0}).copy()
        pvalue_da.values = pvalues
        pvalue_da = pvalue_da.unstack('gridcell')
        
        # Decadal trend = slope * 10
        decadal_trend = slope_da * 10
        
        # Plot mean (top row)
        ax = axes[0, col]
        ax.set_extent([config.NA_LON_MIN, config.NA_LON_MAX, 
                       config.NA_LAT_MIN, config.NA_LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', zorder=2)
        
        # Determine colormap and limits based on variable
        if variable_name == 'SST':
            # Convert from K to C for display
            plot_data = mean_data - 273.15
            vmin, vmax = 15, 32
            cmap = 'RdYlBu_r'
        elif variable_name == 'PI':
            plot_data = mean_data
            vmin, vmax = 40, 90
            cmap = 'RdYlBu_r'
        else:  # CGI
            plot_data = mean_data
            vmin, vmax = 0, 3
            cmap = 'RdYlBu_r'
        
        im1 = ax.pcolormesh(plot_data.longitude, plot_data.latitude, plot_data,
                            transform=ccrs.PlateCarree(), cmap=cmap,
                            vmin=vmin, vmax=vmax, zorder=0)
        ax.set_title(f'{months[month]} Mean', fontsize=11)
        
        if col == 0:
            ax.set_ylabel('Mean', fontsize=12)
        
        # Plot trend (bottom row)
        ax = axes[1, col]
        ax.set_extent([config.NA_LON_MIN, config.NA_LON_MAX, 
                       config.NA_LAT_MIN, config.NA_LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', zorder=2)
        
        # Trend colormap (diverging)
        if variable_name == 'SST':
            trend_vmax = 0.5
        elif variable_name == 'PI':
            trend_vmax = 2
        else:
            trend_vmax = 0.5
        
        im2 = ax.pcolormesh(decadal_trend.longitude, decadal_trend.latitude, decadal_trend,
                            transform=ccrs.PlateCarree(), cmap='RdBu_r',
                            vmin=-trend_vmax, vmax=trend_vmax, zorder=0)
        
        # Add hatching where significant (p < 0.05)
        sig_mask = pvalue_da < 0.05
        if sig_mask.sum() > 0:
            # Create hatching for significant areas
            sig_data = np.where(sig_mask.values, 1, 0)
            with mpl.rc_context({'hatch.linewidth': 0.3}):
                cs_hatch = ax.contourf(pvalue_da.longitude, pvalue_da.latitude, 
                                    sig_data,
                                    levels=[0.5, 1.5], 
                                    colors='none',
                                    hatches=['/////'],
                                    extend='neither',
                                    zorder=10,
                                    transform=ccrs.PlateCarree())
            # Try to set hatch properties (GeoContourSet might wrap collections differently)
            if hasattr(cs_hatch, 'collections'):
                for collection in cs_hatch.collections:
                    collection.set_edgecolor('black')
                    collection.set_facecolor('none')
                    collection.set_linewidth(0.4)
        
        ax.set_title(f'{months[month]} Trend', fontsize=11)
        
        if col == 0:
            ax.set_ylabel('Decadal Trend', fontsize=12)
    
    # Add colorbars (after loop completes)
    cbar1 = fig.colorbar(im1, ax=axes[0, :].tolist(), shrink=0.7, 
                         orientation='vertical', pad=0.08, aspect=20)
    cbar1.set_label(f'{variable_name} ({units})', fontsize=10)
    
    cbar2 = fig.colorbar(im2, ax=axes[1, :].tolist(), shrink=0.7,
                         orientation='vertical', pad=0.08, aspect=20)
    cbar2.set_label(f'{variable_name} trend ({units}/decade)', fontsize=10)
    
    # Add gridlines
    for ax in axes.flat:
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
    
    # Overall title
    years_str = regression.attrs.get('years_used', 'N/A')
    fig.suptitle(f'{variable_name} Mean and Decadal Trend (ASO, {years_str})\n'
                 f'Stippling indicates p < 0.05', fontsize=14)
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved {variable_name} diagnostic plot to {save_path}")
    
    return fig
