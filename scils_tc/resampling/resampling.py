"""
Core resampling functions for SCILS TC Model.

This module provides functions for calculating landfall rate change factors
between base and target periods.
"""

import subprocess
import sys

import numpy as np
import pandas as pd

import config
from scils_tc.utils import SimulationArtifact, actual_event_rows, year_iteration_grid_from_dataframe

# SSHS categories for landfall rate calculation (TS and up)
SSHS_CATEGORIES = ['TS', 'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5']


def get_simulation_filename(mode, region, n_iter):
    """
    Find simulation file matching the mode parameters.
    
    Searches for files containing the year or GWL value to be flexible with
    different naming conventions (e.g., year_2020, to_2020, GWL1.24, GWL2.0, GWL2.00).
    
    Parameters
    ----------
    mode : str
        'historical', 'year_YYYY', 'GWL_X.X', 'GWL_X.XX', 'pi_only_year_YYYY', 
        'pi_only_GWL_X.X', 'cgi_only_year_YYYY', or 'cgi_only_GWL_X.X'
    region : str
        'CONUS' or 'NorthAtlantic'
    n_iter : int
        Number of iterations
        
    Returns
    -------
    Path or None
        Path to simulation file if found, None otherwise
    """
    artifact = SimulationArtifact.from_mode(mode, region, n_iter)
    canonical_path = artifact.simulation_path()
    if canonical_path.exists():
        return canonical_path

    # Extract the search pattern - include prefix for pi_only/cgi_only
    search_patterns = []
    
    if mode == "historical":
        search_patterns = ["historical"]
    elif mode.startswith("pi_only_GWL_"):
        gwl_val = float(mode.replace("pi_only_GWL_", ""))
        search_patterns = [f"pi_only_*GWL{gwl_val:.2f}", f"pi_only_*GWL{gwl_val}"]
    elif mode.startswith("pi_only_year_"):
        year_val = int(mode.replace("pi_only_year_", ""))
        # Must include pi_only prefix to avoid matching regular year files
        search_patterns = [f"pi_only_to_{year_val}", f"pi_only_*year_{year_val}", f"pi_only_*{year_val}"]
    elif mode.startswith("cgi_only_GWL_"):
        gwl_val = float(mode.replace("cgi_only_GWL_", ""))
        search_patterns = [f"cgi_only_*GWL{gwl_val:.2f}", f"cgi_only_*GWL{gwl_val}"]
    elif mode.startswith("cgi_only_year_"):
        year_val = int(mode.replace("cgi_only_year_", ""))
        # Must include cgi_only prefix to avoid matching regular year files
        search_patterns = [f"cgi_only_to_{year_val}", f"cgi_only_*year_{year_val}", f"cgi_only_*{year_val}"]
    elif mode.startswith("GWL_"):
        gwl_val = float(mode.replace("GWL_", ""))
        # Exclude pi_only and cgi_only files by being more specific
        search_patterns = [f"GWL{gwl_val:.2f}", f"GWL{gwl_val}"]
    elif mode.startswith("year_"):
        year_val = int(mode.replace("year_", ""))
        search_patterns = [f"to_{year_val}", f"year_{year_val}", f"*{year_val}"]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Search for matching files in simulated directory
    if not config.SIMULATED_DIR.exists():
        return None
    
    # Try each search pattern with exact n_iter first
    for search_value in search_patterns:
        pattern = f"simulated_events_{search_value}*{region}_n{n_iter}.csv"
        matching_files = list(config.SIMULATED_DIR.glob(pattern))
        
        # For non-sensitivity modes, filter out pi_only and cgi_only files
        if not mode.startswith("pi_only_") and not mode.startswith("cgi_only_"):
            matching_files = [f for f in matching_files 
                           if "pi_only" not in f.name and "cgi_only" not in f.name]
        
        if matching_files:
            return matching_files[0]
    
    # Fallback: try with any n_iter value
    for search_value in search_patterns:
        pattern = f"simulated_events_{search_value}*{region}_n*.csv"
        matching_files = list(config.SIMULATED_DIR.glob(pattern))
        
        # For non-sensitivity modes, filter out pi_only and cgi_only files
        if not mode.startswith("pi_only_") and not mode.startswith("cgi_only_"):
            matching_files = [f for f in matching_files 
                           if "pi_only" not in f.name and "cgi_only" not in f.name]
        
        if matching_files:
            # Pick the one with the highest n_iter
            def extract_n_iter(f):
                """Extract the trailing `_nNNN` iteration count from a simulation filename."""
                stem = f.stem
                n_idx = stem.rfind('_n')
                if n_idx >= 0:
                    try:
                        return int(stem[n_idx + 2:])
                    except ValueError:
                        return 0
                return 0
            matching_files.sort(key=extract_n_iter, reverse=True)
            return matching_files[0]
    
    return None


def simulation_exists(mode, region, n_iter):
    """Check if a simulation file already exists."""
    filepath = get_simulation_filename(mode, region, n_iter)
    return filepath is not None and filepath.exists()


def run_simulation_if_needed(mode, region, n_iter, verbose=True, generate_plots=False):
    """
    Run a simulation if it doesn't already exist.
    
    Parameters
    ----------
    mode : str
        'historical', 'year_YYYY', or 'GWL_X.X'
    region : str
        'CONUS' or 'NorthAtlantic'
    n_iter : int
        Number of iterations
    verbose : bool
        Print progress
    generate_plots : bool
        If True, generate diagnostic plots for detrending and simulation
        
    Returns
    -------
    Path
        Path to simulation file
    """
    filepath = get_simulation_filename(mode, region, n_iter)
    
    if filepath is not None and filepath.exists():
        if verbose:
            print(f"  Reusing existing simulation: {filepath.name}")
        return filepath
    
    # Check if detrending files exist for non-historical modes
    if mode != "historical":
        # Parse mode to extract target and sensitivity type
        if mode.startswith("pi_only_") or mode.startswith("cgi_only_"):
            parts = mode.split("_")
            if "year" in parts:
                idx = parts.index("year")
                target_year = int(parts[idx + 1])
                suffix = f"to_{target_year}"
            elif "GWL" in parts:
                idx = parts.index("GWL")
                target_gwl = float(parts[idx + 1])
                suffix = f"GWL{target_gwl}"
            else:
                raise ValueError(f"Unknown sensitivity mode: {mode}")
        elif mode.startswith("year_"):
            target_year = int(mode.split("_")[1])
            suffix = f"to_{target_year}"
        elif mode.startswith("GWL_"):
            target_gwl = float(mode.split("_")[1])
            suffix = f"GWL{target_gwl:.2f}"
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Check for detrended files
        if mode.startswith("pi_only_") or mode.startswith("cgi_only_") or mode.startswith("year_") or mode.startswith("GWL_"):
            target_spec = SimulationArtifact.from_mode(mode, region, n_iter).require_target()
            suffix = target_spec.suffix

        pi_file = config.DETRENDED_DIR / f"ERA5_PI_detrended_{suffix}.nc"
        cgi_file = config.DETRENDED_DIR / f"ERA5_CGI_detrended_{suffix}.nc"
        sst_file = config.DETRENDED_DIR / f"ERA5_SST_detrended_{suffix}.nc"
        
        if not (pi_file.exists() and cgi_file.exists() and sst_file.exists()):
            if verbose:
                print(f"  Detrended files for {mode} not found. Running detrending first...")
            
            # Run detrending
            detrend_cmd = [
                sys.executable, "run_detrending.py",
                "--start-year", str(config.DEFAULT_START_YEAR),
                "--end-year", str(config.DEFAULT_END_YEAR),
            ]
            if not generate_plots:
                detrend_cmd.append("--skip-diagnostics")
            if "year" in mode:
                detrend_cmd.extend(["--target-year", str(target_year)])
            else:
                detrend_cmd.extend(["--target-gwl", str(target_gwl)])
            
            if verbose:
                print(f"    Detrending command: {' '.join(detrend_cmd)}")
            
            result = subprocess.run(detrend_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("ERROR: Detrending failed")
                print(result.stderr)
                raise RuntimeError(f"Detrending failed for mode {mode}")
            
            if verbose:
                print("    Detrending completed.")
    
    # Build simulation command
    cmd = [
        sys.executable, "run_simulation.py",
        "--start-year", str(config.DEFAULT_START_YEAR),
        "--end-year", str(config.DEFAULT_END_YEAR),
        "--n-iter", str(n_iter),
        "--region", region,
        "--seed", "42",
    ]
    
    if generate_plots:
        cmd.append("--plot")
    
    if mode == "historical":
        cmd.append("--use-historical")
    elif mode.startswith("pi_only_"):
        if "year" in mode:
            parts = mode.split("_")
            idx = parts.index("year")
            target_year = int(parts[idx + 1])
            cmd.extend(["--target-year", str(target_year), "--detrend-pi-only"])
        elif "GWL" in mode:
            parts = mode.split("_")
            idx = parts.index("GWL")
            target_gwl = float(parts[idx + 1])
            cmd.extend(["--target-gwl", str(target_gwl), "--detrend-pi-only"])
    elif mode.startswith("cgi_only_"):
        if "year" in mode:
            parts = mode.split("_")
            idx = parts.index("year")
            target_year = int(parts[idx + 1])
            cmd.extend(["--target-year", str(target_year), "--detrend-cgi-only"])
        elif "GWL" in mode:
            parts = mode.split("_")
            idx = parts.index("GWL")
            target_gwl = float(parts[idx + 1])
            cmd.extend(["--target-gwl", str(target_gwl), "--detrend-cgi-only"])
    elif mode.startswith("year_"):
        target_year = int(mode.split("_")[1])
        cmd.extend(["--target-year", str(target_year)])
    elif mode.startswith("GWL_"):
        target_gwl = float(mode.split("_")[1])
        cmd.extend(["--target-gwl", str(target_gwl)])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if verbose:
        print(f"  Running simulation: {mode}...")
        print(f"    Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("ERROR: Simulation failed")
        print(result.stderr)
        raise RuntimeError(f"Simulation failed for mode {mode}")
    
    # Re-search for the file after simulation completes
    filepath = get_simulation_filename(mode, region, n_iter)
    if filepath is None or not filepath.exists():
        raise RuntimeError(f"Simulation completed but file not found for mode {mode}")
    
    if verbose:
        print(f"    Completed: {filepath.name}")
    
    return filepath


def load_simulation(filepath):
    """Load simulation results from CSV."""
    return pd.read_csv(filepath)


def calculate_landfall_rates(df, years=None, enso_filter=None):
    """
    Calculate landfall rates per SSHS category.
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation results
    years : list, optional
        List of years to include (for subsample mode)
    enso_filter : str, optional
        ENSO state to filter by ('El Nino', 'Neutral', 'La Nina')
        
    Returns
    -------
    dict
        Dictionary mapping SSHS category to annual landfall rate
    """
    # Filter by years if specified
    if years is not None:
        df = df[df['year'].isin(years)]
    
    # Filter by ENSO state if specified
    if enso_filter is not None:
        df = df[df['enso_state'] == enso_filter]
    
    if len(df) == 0:
        print("WARNING: No events after filtering!")
        return {cat: 0.0 for cat in SSHS_CATEGORIES}
    
    year_iter_grid = year_iteration_grid_from_dataframe(df, years=years)
    n_year_iters = len(year_iter_grid)
    actual_df = actual_event_rows(df)
    
    # Count landfalls per category (using lfi_sshs)
    rates = {}
    for cat in SSHS_CATEGORIES:
        count = (actual_df['lfi_sshs'] == cat).sum()
        rates[cat] = count / n_year_iters if n_year_iters > 0 else 0.0
    
    return rates


def calculate_change_rates(base_rates, target_rates):
    """
    Calculate relative change rates between base and target.
    
    Parameters
    ----------
    base_rates : dict
        Base period landfall rates per category
    target_rates : dict
        Target period landfall rates per category
        
    Returns
    -------
    dict
        Dictionary mapping SSHS category to change rate (target/base)
    """
    change_rates = {}
    for cat in SSHS_CATEGORIES:
        base_rate = base_rates.get(cat, 0)
        target_rate = target_rates.get(cat, 0)
        
        if base_rate > 0:
            change_rates[cat] = target_rate / base_rate
        elif target_rate > 0:
            # Base is 0 but target is not - infinite increase
            change_rates[cat] = np.inf
        else:
            # Both are 0
            change_rates[cat] = 1.0
    
    return change_rates


def get_gwl_for_year(year):
    """
    Get GWL for a given year using the GWL lookup.
    
    Parameters
    ----------
    year : float
        Year to get GWL for
        
    Returns
    -------
    float
        GWL value
    """
    from scils_tc.detrending.gwl_lookup import year_to_gwl
    return year_to_gwl(year)


def get_year_for_gwl(gwl):
    """
    Get year for a given GWL using the GWL lookup.
    
    Parameters
    ----------
    gwl : float
        GWL value
        
    Returns
    -------
    float
        Year corresponding to GWL
    """
    from scils_tc.detrending.gwl_lookup import gwl_to_year
    return gwl_to_year(gwl)


def generate_output_filename(base_label, target_label, region, base_enso='all', target_enso='all'):
    """
    Generate output filename following naming convention.
    
    Parameters
    ----------
    base_label : str
        Base period label (e.g., '1998-2009', 'GWL1.2')
    target_label : str
        Target period label (e.g., '2050', 'GWL2.0')
    region : str
        Region name
    base_enso : str
        Base ENSO filter
    target_enso : str
        Target ENSO filter
        
    Returns
    -------
    str
        Filename for output
    """
    base_enso_str = f"_{base_enso.replace(' ', '')}" if base_enso != 'all' else ""
    target_enso_str = f"_{target_enso.replace(' ', '')}" if target_enso != 'all' else ""
    
    return f"change_rates_base_{base_label}{base_enso_str}_target_{target_label}{target_enso_str}_{region}"
