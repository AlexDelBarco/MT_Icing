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



