"""
Spatial utility functions.

Provides point-in-polygon tests and other spatial operations.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon


def create_polygon_from_vertices(vertices):
    """
    Create a Shapely Polygon from a list of vertices.
    
    Parameters
    ----------
    vertices : list of tuples
        List of (lon, lat) tuples defining the polygon vertices.
        The polygon should be closed (first and last vertex are the same).
    
    Returns
    -------
    shapely.geometry.Polygon
        Polygon object.
    """
    return Polygon(vertices)


def point_in_polygon(lon, lat, polygon):
    """
    Check if a point is inside a polygon.
    
    Parameters
    ----------
    lon : float
        Longitude of the point.
    lat : float
        Latitude of the point.
    polygon : shapely.geometry.Polygon
        Polygon to test against.
    
    Returns
    -------
    bool
        True if point is inside the polygon.
    """
    point = Point(lon, lat)
    return polygon.contains(point)


def load_us_states_shapefile(shapefile_path):
    """
    Load US states shapefile.
    
    Parameters
    ----------
    shapefile_path : str or Path
        Path to the shapefile.
    
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with state geometries.
    """
    gdf = gpd.read_file(shapefile_path)
    return gdf


def get_state_for_point(lon, lat, states_gdf):
    """
    Get the US state name for a given point.
    
    Parameters
    ----------
    lon : float
        Longitude of the point.
    lat : float
        Latitude of the point.
    states_gdf : geopandas.GeoDataFrame
        GeoDataFrame with state geometries (must have 'NAME' column).
    
    Returns
    -------
    str or None
        State name if point is within a state, None otherwise.
    """
    if np.isnan(lon) or np.isnan(lat):
        return None
    
    point = Point(lon, lat)
    
    for idx, row in states_gdf.iterrows():
        if row.geometry.contains(point):
            return row['NAME']
    
    return None


def is_conus_state(state_name):
    """
    Check if a state is part of the Continental United States (CONUS).
    
    Parameters
    ----------
    state_name : str or None
        Name of the state.
    
    Returns
    -------
    bool
        True if state is in CONUS.
    """
    if state_name is None:
        return False
    
    # Non-CONUS states/territories
    non_conus = [
        'Alaska', 'Hawaii', 'Puerto Rico', 'U.S. Virgin Islands',
        'Guam', 'American Samoa', 'Northern Mariana Islands',
        'United States Virgin Islands'
    ]
    
    return state_name not in non_conus
