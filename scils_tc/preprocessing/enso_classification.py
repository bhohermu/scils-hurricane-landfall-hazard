"""
ENSO state classification module.

Classifies years as El Niño, La Niña, or Neutral based on ASO RONI anomalies.
"""

import numpy as np
import pandas as pd

import config


def load_roni_data(filepath=None):
    """
    Load RONI (Relative Oceanic Niño Index) data from CSV file.
    
    Parameters
    ----------
    filepath : str or Path, optional
        Path to RONI.csv. If None, uses config default.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: SEAS, YR, ANOM
    """
    if filepath is None:
        filepath = config.RONI_FILE
    
    df = pd.read_csv(filepath)
    return df


def classify_enso_state(anomaly, 
                        elnino_threshold=None, 
                        lanina_threshold=None):
    """
    Classify ENSO state based on RONI anomaly value.
    
    Parameters
    ----------
    anomaly : float
        ASO RONI anomaly value.
    elnino_threshold : float, optional
        Threshold for El Niño classification. Default from config.
    lanina_threshold : float, optional
        Threshold for La Niña classification. Default from config.
    
    Returns
    -------
    str
        'El Nino', 'La Nina', or 'Neutral'
    """
    if elnino_threshold is None:
        elnino_threshold = config.ENSO_ELNINO_THRESHOLD
    if lanina_threshold is None:
        lanina_threshold = config.ENSO_LANINA_THRESHOLD
    
    if anomaly >= elnino_threshold:
        return 'El Nino'
    elif anomaly <= lanina_threshold:
        return 'La Nina'
    else:
        return 'Neutral'


def get_aso_roni(roni_df, year):
    """
    Get the ASO RONI anomaly for a given year.
    
    Parameters
    ----------
    roni_df : pandas.DataFrame
        RONI data with SEAS, YR, ANOM columns.
    year : int
        Year to look up.
    
    Returns
    -------
    float
        ASO RONI anomaly value.
    """
    mask = (roni_df['YR'] == year) & (roni_df['SEAS'] == 'ASO')
    subset = roni_df[mask]
    
    if len(subset) == 0:
        return np.nan
    
    return subset['ANOM'].values[0]


def process_enso(start_year=None, end_year=None, output_path=None):
    """
    Process ENSO classification for all years and save to CSV.
    
    Uses ASO RONI anomaly from RONI.csv for classification.
    
    Parameters
    ----------
    start_year : int, optional
        First year to process. Default from config.
    end_year : int, optional
        Last year to process. Default from config.
    output_path : str or Path, optional
        Path to save output CSV. Default from config.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: Year, ASO_RONI, ENSO_State
    """
    if start_year is None:
        start_year = config.START_YEAR
    if end_year is None:
        end_year = config.END_YEAR
    if output_path is None:
        output_path = config.get_output_path(config.ENSO_STATE_FILE)
    
    # Load RONI data
    roni_df = load_roni_data()
    
    # Process each year
    results = []
    for year in range(start_year, end_year + 1):
        aso_roni = get_aso_roni(roni_df, year)
        enso_state = classify_enso_state(aso_roni)
        
        results.append({
            'Year': year,
            'ASO_RONI': aso_roni,
            'ENSO_State': enso_state
        })
    
    # Create DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    
    print(f"ENSO state classification saved to: {output_path}")
    print(result_df.to_string(index=False))
    
    return result_df


if __name__ == "__main__":
    # Test the module
    df = process_enso()
