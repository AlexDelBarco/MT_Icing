# IMPORTS
import os
import functions as fn
import pandas as pd
import numpy as np
import xarray as xr

# PARAMTERES
height = 2  # Height level index to use (0-based): 0=50m; 1=100m; 2=150m
ice_load_method = 5  # Method for ice load calculation
calculate_new_ice_load = False  # Whether to calculate ice load or load existing data


# IMPORT DATA
# Import data and first look
data1 = "data/newa_wrf_for_jana_mstudent_extended.nc"
dataset = fn.load_netcdf_data(data1)


# EXPLORE DATASET
height_level = dataset.height.values[height]  # Height level in meters
print(f"Exploring dataset at height level: {height_level} m")
# explore the variables
#fn.explore_variables(dataset)

# explore one variable in detail in a chosen period
#fn.explore_variable_detail(dataset, 'LANDMASK')

# Plot grid location on map
#fn.plot_grid_points_cartopy_map(dataset, margin_degrees=1.5, zoom_level=8, title="Grid Points - Terrain Map")

# Offshore / Onshore classification
#landmask_results = fn.analyze_landmask(dataset, create_plot=True, save_results=True)


# CALCULATIONS AND PLOTS


# Period
start_date = '1989-01-01T00:00:00.000000000'
end_date = '2022-12-31T23:30:00.000000000'
dates = pd.date_range(start_date, end_date, freq='YS-JUL')

# Accreation for winter and time period + plot
#fn.accreation_per_winter(dataset, start_date, end_date, height_level=height)

# ice load data: load/calculate 
if calculate_new_ice_load:
    print("Calculating ice load...")
    #ice_load_data = fn.calculate_ice_load(dataset, dates, ice_load_method, height_level=height, create_figures=True)
else:
    print("Loading existing ice load data...")
    filename = f"results/iceload_19890701_to_20220701_h150m.nc"
    ice_load_data = xr.open_dataarray(filename)

    print(f"Loaded ice load data from: {filename}")
    print(f"Loaded ice load data with shape: {ice_load_data.shape}")
    
# Analyze ice load duration curves for all grid cells
#print("\n=== ICE LOAD DURATION CURVE ANALYSIS ===")
#duration_results = fn.plot_ice_load_duration_curves(ice_load_data, save_plots=True, ice_load_threshold=0.1)
#print("\n=== ICE LOAD PDF CURVE ANALYSIS ===")
#pdf_results = fn.plot_ice_load_pdf_curves(ice_load_data, save_plots=True, ice_load_threshold=0.1)
print("\n=== ICE LOAD CDF CURVE ANALYSIS ===")
cdf_results = fn.plot_ice_load_cdf_curves(ice_load_data, save_plots=True, ice_load_threshold=0.001)

# Analyze threshold exceedance spatial patterns
#print("\n=== ICE LOAD THRESHOLD EXCEEDANCE ANALYSIS ===")
#threshold_results = fn.plot_ice_load_threshold_exceedance_map(ice_load_data, ice_load_threshold=0.4, save_plots=True, units='hours')
