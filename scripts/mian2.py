# IMPORTS
import os
import functions as fn
import pandas as pd
import numpy as np
import xarray as xr

# Set working directory to project root if running from scripts folder
current_dir = os.getcwd()
if current_dir.endswith('scripts'):
    os.chdir('..')
    print(f"Changed working directory from {current_dir} to {os.getcwd()}")

# PARAMTERES
height = 2  # Height level index to use (0-based): 0=50m; 1=100m; 2=150m
ice_load_method = 5  # Method for ice load calculation
calculate_new_ice_load = False  # Whether to calculate ice load or load existing data


# IMPORT DATA
# Merge NetCDF files to combine all meteorological variables
print("=== MERGING NETCDF FILES ===")
main_file = "data/newa_wrf_for_jana_mstudent_extended.nc"
wd_file = "data/newa_wrf_for_jana_mstudent_extended_WD.nc"
merged_file = "data/newa_wrf_for_jana_mstudent_extended_merged.nc"

# Check if merged file already exists
if not os.path.exists(merged_file):
    print("Merged file not found. Creating merged dataset...")
    success = fn.merge_netcdf_files(main_file, wd_file, merged_file, verbose=True)
    if not success:
        print("Failed to merge files. Using main file only.")
        data1 = main_file
    else:
        data1 = merged_file
else:
    print("Merged file already exists. Using existing merged dataset...")
    data1 = merged_file

# Import merged NEWEA meteorological data
dataset = fn.load_netcdf_data(data1)

# Import EMD data
# data2 = "data/EMD_data/EmdWrf_N59.600_E019.960.txt"
# emd_data = fn.import_emd_data(data2)
# print(emd_data.head())


# EXPLORE DATASET
height_level = dataset.height.values[height]  # Height level in meters
print(f"Exploring dataset at height level: {height_level} m")
# explore the variables
# fn.explore_variables(dataset)

# explore one variable in detail in a chosen period
# fn.explore_variable_detail(dataset, 'WD')

# Plot grid location on map
#fn.plot_grid_points_cartopy_map(dataset, margin_degrees=1.5, zoom_level=8, title="Grid Points - Terrain Map")

# Offshore / Onshore classification
#landmask_results = fn.analyze_landmask(dataset, create_plot=True, save_results=True)


# CALCULATIONS AND PLOTS

# Period
start_date = '1989-01-01T00:00:00.000000000'
end_date = '2022-12-31T23:30:00.000000000'
dates = pd.date_range(start_date, end_date, freq='YS-JUL')


# ACCERATION

# Accreation for winter and time period + plot
#fn.accreation_per_winter(dataset, start_date, end_date, height_level=height)


# ICE LOAD

# ice load data: load/calculate 
# if calculate_new_ice_load:
#     print("Calculating ice load...")
#     #ice_load_data = fn.calculate_ice_load(dataset, dates, ice_load_method, height_level=height, create_figures=True)
# else:
#     print("Loading existing ice load data...")
#     filename = f"results/iceload_19890701_to_20220701_h150m.nc"
#     ice_load_data = xr.open_dataarray(filename)

#     print(f"Loaded ice load data from: {filename}")
#     print(f"Loaded ice load data with shape: {ice_load_data.shape}")


# SPATIAL GRADIENTS

# Analyze ice load mean gradient for all grid cells
# print("\n=== ICE LOAD MEAN GRADIENT ANALYSIS ===")
# mean_gradient_results = fn.create_spatial_gradient_mean_plots(ice_load_data)
   
# Analyze ice load load duration curves for all grid cells
# print("\n=== ICE LOAD DURATION CURVE ANALYSIS ===")
# duration_results = fn.plot_ice_load_duration_curves(ice_load_data, save_plots=True, ice_load_threshold=0.1)

# Analyze ice load PDF curves for all grid cells
# print("\n=== ICE LOAD PDF CURVE ANALYSIS ===")
# pdf_results = fn.plot_ice_load_pdf_curves(ice_load_data, save_plots=True, ice_load_threshold=0.1)

# Analyze ice load CDF curves for all grid cells
# print("\n=== ICE LOAD CDF CURVE ANALYSIS ===")
# cdf_results = fn.plot_ice_load_cdf_curves(ice_load_data, save_plots=True, ice_load_threshold=0.1)

# Analyze ice load 1-CDF curves for all grid cells
#print("\n=== ICE LOAD 1-CDF CURVE ANALYSIS ===")
#exceedance_cdf_results = fn.plot_ice_load_1_minus_cdf_curves(ice_load_data, save_plots=True, ice_load_threshold=0, months=[1,2,12])

# Analyze ice load CDF log curves for all grid cells
# print("\n=== ICE LOAD CDF LOG CURVE ANALYSIS ===")
# cdf_log_results = fn.plot_ice_load_cdf_curves_log_scale(ice_load_data, save_plots=True, ice_load_threshold=0, months=[1,2,12])

# Analyze ice load CDF curves for all grid cells after meteorological filtering
#print("\n=== METEOROLOGICAL FILTERING + AUTOMATIC CDF ANALYSIS ===")
# filtered_ds, results = fn.filter_dataset_by_thresholds(
#     dataset=dataset,
#     # Meteorological filters
#     PBLH_min=None,      # PBL Height (m)
#     PBLH_max=None,      # PBL Height (m)
#     PRECIP_min=None,    # Precipitation (mm/h)
#     PRECIP_max=None,    # Precipitation (mm/h)
#     QVAPOR_min=None,    # Water Vapor Mixing Ratio (kg/kg)
#     QVAPOR_max=None,    # Water Vapor Mixing Ratio (kg/kg)
#     RMOL_min=None,      # Monin-Obukhov Length (m)
#     RMOL_max=None,      # Monin-Obukhov Length (m)
#     T_min=None,         # Temperature (K)
#     T_max=None,         # Temperature (K)
#     WS_min=8.5,         # Wind Speed (m/s)
#     WS_max=None,        # Wind Speed (m/s)
    
#     height_level=height,
    
#     # Enable automatic ice load CDF analysis
#     calculate_ice_load_cdf=True,
#     dates=dates,
#     ice_load_method=ice_load_method,
#     ice_load_threshold=0.1,
#     months=None,  # Winter season if wanted
#     percentile=None,     # Remove extreme outliers if wanted
#     verbose=True   # Whether to print filtering information
# )
print("\n=== METEOROLOGICAL FILTERING + AUTOMATIC CDF ANALYSIS ; SYSTEMATIC ===")
results_sys_filter = fn.systematic_meteorological_filtering(
    dataset,
    WD_range=(0, 360, 10),           # Wind direction range: min=0, max=360, step=10 -> [0, 10, 20, ..., 360]
    WS_range=None,           # Wind speed range: min=5, max=15, step=5 -> [5, 10, 15]
    T_range=None,        # Temperature range  
    PBLH_range=None,    # Boundary layer height
    PRECIP_range=None,      # Precipitation
    QVAPOR_range=None,      # Water vapor mixing ratio
    RMOL_range=None,        # Monin-Obukhov length
    calculate_ice_load_cdf=True,
    dates=dates,
    ice_load_method=5,
    ice_load_threshold=0.1,
    months=None,       # Winter months
    save_results=True,
    height_level=height,
)


# Analyze threshold exceedance spatial patterns
#print("\n=== ICE LOAD THRESHOLD EXCEEDANCE ANALYSIS ===")
#threshold_results = fn.plot_ice_load_threshold_exceedance_map(ice_load_data, ice_load_threshold=0.1, save_plots=True, units='hours')


# TEMPORAL GRADIENTS

# Example: Enhanced filtering with automatic ice load CDF analysis

