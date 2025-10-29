# IMPORTS
import os
import functions as fn
import pandas as pd
import numpy as np
import xarray as xr

# IMPORT DATA
# Import data and first look
data1 = "data/alexandre.nc"
dataset = fn.load_netcdf_data(data1)

# EXPLORE DATASET
# explore the variables
#fn.explore_variables(dataset)

# explore one variable in detail in a chosen period
#fn.explore_variable_detail(dataset, 'ACCRE_CYL')

# Accreation for winter and time period + plot
start_date = '1899-07-01T00:00:00'
end_date = '2021-06-30T23:30:00'
dates = pd.date_range(start_date, '2021-07-01T23:30:00', freq='YS-JUL')
#fn.accreation_per_winter(dataset, start_date, end_date)

# Load/calculate ice load data
print("Loading existing ice load data...")
ice_load_data = xr.open_dataarray("results/iceload_18990701_to_20210701.nc")
print(f"Loaded ice load data with shape: {ice_load_data.shape}")
#print("Calculating ice load...")
#ice_load_data = fn.calculate_ice_load(dataset, dates, 5)

# Spatial gradients analysis
#spatial_stats = fn.create_spatial_gradient_plots(ice_load_data)
#spatial_stats_2 = fn.create_spatial_gradient_time_evolution_plots(ice_load_data)

# Temporal gradients analysis
#temporal_stats = fn.create_temporal_gradient_plots(ice_load_data)

# Grid on Earth map
print("Creating grid overlay on Earth map...")
earth_map_info = fn.create_grid_on_earth_map(dataset)
if earth_map_info:
    print(f"Earth map created successfully!")
    print(f"  Domain size: {earth_map_info['domain_size_km'][0]:.1f} km × {earth_map_info['domain_size_km'][1]:.1f} km")
    print(f"  GDAL available: {earth_map_info['gdal_available']}")
    print(f"  Map data source: {earth_map_info['map_source']}")

# EMD DATA IMPORT FUNCTION
def import_emd_data(file_path):
    """
    Import EMD text data with metadata extraction and proper formatting
    
    Expected format:
    - Metadata lines (Orography, Latitude, Longitude)
    - Column headers line
    - Time-series data with tab-separated values
    """
    import pandas as pd
    
    metadata = {}
    data_start_line = 0
    
    # Read file and extract metadata
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse metadata section
    for i, line in enumerate(lines):
        line = line.strip()
        if ':' in line and not line.startswith('time'):
            # Extract metadata (Orography, Latitude, Longitude)
            key, value = line.split(':', 1)
            try:
                metadata[key.strip()] = float(value.strip())
            except ValueError:
                metadata[key.strip()] = value.strip()
        elif line.startswith('time'):
            # Found the header line
            data_start_line = i
            break
    
    # Read the data starting from header line
    try:
        # Try tab-separated first (most common for EMD data)
        df = pd.read_csv(file_path, delimiter='\t', skiprows=data_start_line, 
                        parse_dates=[0], index_col=0)
        
        # If tab doesn't work well, try other delimiters
        if len(df.columns) < 10:  # Expect many columns for EMD data
            for delimiter in [' ', ',', ';']:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter, skiprows=data_start_line,
                                   parse_dates=[0], index_col=0)
                    if len(df.columns) >= 10:
                        break
                except:
                    continue
        
        # Add metadata as attributes
        df.attrs['metadata'] = metadata
        
        print(f"Successfully imported EMD data:")
        print(f"- Data shape: {df.shape}")
        print(f"- Time range: {df.index[0]} to {df.index[-1]}")
        print(f"- Metadata: {metadata}")
        print(f"- Sample columns: {list(df.columns[:10])}")
        
        return df
        
    except Exception as e:
        print(f"Error importing EMD data: {e}")
        print("Trying basic import without metadata parsing...")
        
        # Fallback to simple import
        try:
            df = pd.read_csv(file_path, delimiter='\t', parse_dates=[0], index_col=0)
            return df
        except:
            # Last resort - try space-separated
            df = pd.read_csv(file_path, delimiter=' ', parse_dates=[0], index_col=0)
            return df

# Example usage (uncomment to use):
# emd_df = import_emd_data('data/EMD_data/your_emd_file.txt')
# print(f"Metadata: {emd_df.attrs.get('metadata', 'No metadata found')}")


