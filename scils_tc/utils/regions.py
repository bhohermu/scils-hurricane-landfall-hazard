"""
Jewson region handling utilities.

Loads and processes the region definitions from regionDefinitionsJewson.csv.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def load_jewson_regions(filepath):
    """
    Load Jewson region definitions from CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to regionDefinitionsJewson.csv.
    
    Returns
    -------
    dict
        Dictionary with region information:
        {
            region_number: {
                'name': str,
                'polygon': shapely.geometry.Polygon,
                'vertices': list of (lon, lat) tuples
            }
        }
    """
    # Read CSV, keeping 'NA' as string (not as NaN)
    df = pd.read_csv(filepath, keep_default_na=False, na_values=[''])
    
    # Clean up column values (remove whitespace)
    df['Name'] = df['Name'].str.strip()
    df['Lat'] = pd.to_numeric(df['Lat'].astype(str).str.strip(), errors='coerce')
    df['Lon'] = pd.to_numeric(df['Lon'].astype(str).str.strip(), errors='coerce')
    
    regions = {}
    
    for region_num in df['Number'].unique():
        region_data = df[df['Number'] == region_num]
        region_name = region_data['Name'].iloc[0]
        
        # Get vertices as (lon, lat) tuples for Shapely
        vertices = list(zip(region_data['Lon'], region_data['Lat']))
        
        # Create polygon
        polygon = Polygon(vertices)
        
        regions[region_num] = {
            'name': region_name,
            'polygon': polygon,
            'vertices': vertices
        }
    
    return regions


def get_region_for_point(lon, lat, regions):
    """
    Determine which Jewson region a point belongs to.
    
    Parameters
    ----------
    lon : float
        Longitude of the point.
    lat : float
        Latitude of the point.
    regions : dict
        Dictionary of regions from load_jewson_regions().
    
    Returns
    -------
    tuple (str, int) or (None, None)
        (region_name, region_number) if point is in a region,
        (None, None) otherwise.
    """
    if np.isnan(lon) or np.isnan(lat):
        return None, None
    
    point = Point(lon, lat)
    
    for region_num, region_info in regions.items():
        if region_info['polygon'].contains(point):
            return region_info['name'], region_num
    
    return None, None


def plot_jewson_regions(regions, output_path=None, figsize=(14, 10)):
    """
    Plot the Jewson regions on a map.
    
    Parameters
    ----------
    regions : dict
        Dictionary of regions from load_jewson_regions().
    output_path : str or Path, optional
        Path to save the figure. If None, displays the plot.
    figsize : tuple
        Figure size (width, height) in inches.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    # Define colors for each region
    colors = {
        1: '#FF6B6B',  # GOM - red
        2: '#4ECDC4',  # CARB - teal
        3: '#45B7D1',  # MDR - blue
        4: '#96CEB4'   # NA - green
    }
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set map extent to cover North Atlantic
    ax.set_extent([-100, 10, -5, 55], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--')
    ax.add_feature(cfeature.STATES, linewidth=0.2, linestyle=':')
    
    # Plot each region
    for region_num, region_info in regions.items():
        vertices = region_info['vertices']
        lons = [v[0] for v in vertices]
        lats = [v[1] for v in vertices]
        
        # Plot polygon
        ax.fill(lons, lats, 
                alpha=0.4, 
                facecolor=colors.get(region_num, 'gray'),
                edgecolor='black',
                linewidth=2,
                transform=ccrs.PlateCarree(),
                label=f"{region_info['name']} (Region {region_num})")
        
        # Plot vertices as points
        ax.scatter(lons, lats, 
                   c='black', 
                   s=20, 
                   zorder=5,
                   transform=ccrs.PlateCarree())
        
        # Add region label at centroid
        centroid = region_info['polygon'].centroid
        ax.text(centroid.x, centroid.y, 
                region_info['name'],
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='center',
                transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Title and legend
    ax.set_title('Jewson Region Definitions for Atlantic Tropical Cyclones', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Region map saved to: {output_path}")
    
    return fig
