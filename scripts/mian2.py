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

# PARAMETERS
height = 2  # Height level index to use (0-based): 0=50m; 1=100m; 2=150m
ice_load_method = 51  # Method for ice load calculation
calculate_new_ice_load = False  # Whether to calculate ice load or load existing data


# IMPORT DATA
# Merge NetCDF files to combine all meteorological variables
print("=== MERGING NETCDF FILES ===")
main_file = "data/newa_wrf_for_jana_mstudent_extended.nc"
wd_file = "data/newa_wrf_for_jana_mstudent_extended_WD.nc"
pdfc_file= "data/newa_wrf_for_jana_mstudent_extended_PSFC_SEAICE_SWDDNI.nc"
merged_file = "data/newa_wrf_for_jana_mstudent_extended_merged.nc"
final_file = "data/newa_wrf_final_merged.nc"

# Check if merged file already exists
if not os.path.exists(final_file):
    print("Merged file not found. Creating merged dataset...")
    success = fn.merge_netcdf_files(merged_file, pdfc_file, final_file, verbose=True)
    if not success:
        print("Failed to merge files. Using main file only.")
        data1 = main_file
    else:
        data1 = merged_file
else:
    print("Merged file already exists. Using existing merged dataset...")
    data1 = final_file

# Import merged NEWEA meteorological data
dataset = fn.load_netcdf_data(data1)


# EXPLORE DATASET
height_level = dataset.height.values[height]  # Height level in meters
print(f"Exploring dataset at height level: {height_level} m")
# explore the variables
# fn.explore_variables(dataset)

#explore one variable in detail in a chosen period
# fn.explore_variable_detail(dataset, 'QVAPOR')

# Plot grid location on map
#fn.plot_grid_points_cartopy_map(dataset, margin_degrees=1.5, zoom_level=8, title="Grid Points - Terrain Map")

# Offshore / Onshore classification
#landmask_results = fn.analyze_landmask(dataset, create_plot=True, save_results=True)


# CALCULATIONS AND PLOTS

# Period
start_date = '1989-01-01T00:00:00.000000000'
end_date = '2022-12-31T23:30:00.000000000'


# start_date = '2020-01-01T00:00:00.000000000'
# end_date = '2022-12-31T23:30:00.000000000'
dates = pd.date_range(start_date, end_date, freq='YS-JUL')


# COMPREHENSIVE CLIMATE ANALYSIS

print("\n=== COMPREHENSIVE CLIMATE ANALYSIS ===")
climate_results = fn.climate_analysis(
    dataset=dataset,
    height_level=height,
    save_plots=True,
    results_subdir="climate_analysis"
)


# ACCERATION

# Accreation for winter and time period + plot
#fn.accreation_per_winter(dataset, start_date, end_date, height_level=height)



# ICE LOAD

#ice load data: load/calculate 
if calculate_new_ice_load:
    print("Calculating ice load...")
    #ice_load_data = fn.calculate_ice_load(dataset, dates, ice_load_method, height_level=height, create_figures=True)

    #Add ice load directly to the dataset
    print("=== ADDING ICE LOAD TO DATASET ===")
    dataset_with_ice_load = fn.add_ice_load_to_dataset(
        ds=dataset,
        dates=dates,
        method=ice_load_method,
        height_level=height,
        variable_name='ICE_LOAD'
    )

    # # Now you can access ice load directly from the dataset
    # ice_load_data = dataset_with_ice_load['ICE_LOAD']
    # print(f"Ice load data shape: {ice_load_data.shape}")
    # print(f"Ice load available at height level {height}: {dataset.height.values[height]} m")

else:
    print("Loading existing complete dataset with ice load...")
    filename = f"results/dataset_iceload_19890701_20220701_h{height}.nc"
    dataset_with_ice_load = xr.open_dataset(filename)  # Load complete dataset

    print(f"Loaded dataset from: {filename}")
    print(f"Dataset dimensions: {dataset_with_ice_load.dims}")
    print(f"Available variables: {list(dataset_with_ice_load.data_vars.keys())}")
    print(f"Ice load variable 'ICE_LOAD' is ready for analysis at height level {height}: {dataset_with_ice_load.height.values[height]} m")

#Plot ice load values for each grid cell

# print("\n=== ICE LOAD GRID VALUES ANALYSIS ===")
# grid_results = fn.plot_grid_ice_load_values(
#      dataset_with_ice_load=dataset_with_ice_load,
#      ice_load_variable='ICE_LOAD',
#      height_level=height,
#      ice_load_threshold=0,
#      save_plots=True,
#      months=None,  # Can specify winter months like [12, 1, 2, 3] if desired
#      show_colorbar=True
#  )

# print("\n=== ICE LOAD GRID VALUES ANALYSIS HOURS ===")

# grid_results_hours = fn.plot_ice_load_threshold_exceedance_map(
#     dataset_with_ice_load=dataset_with_ice_load,
#     ice_load_variable='ICE_LOAD',
#     height_level=height,

#     ice_load_threshold=0.1,
#     save_plots=True
# )

# WIND ROSE

# results_wind_rose = fn.wind_rose(
#     dataset=dataset,
#     height_level=height,
#     title=f"Wind Rose at {height_level} m",
#     bins =32,
# )

# ICING TEMPERATURE AND HUMIDITY CRITERIA

# Calculate relative humidity
#The rh is calculated using surface P, T and mixing ratio at height 
# scale-height ~8km, from Ch.2 of 46100's book\notes, then consider d(ln p)/d(ln z)

# dataset_ice_load_rh = fn.add_rh(dataset_with_ice_load=dataset_with_ice_load, height_l=height,
#                                 phase= 'auto') #'liquid', 'solid', 'auto' – to make calculation valid in 'liquid' water (default) or 'solid' ice regimes. 'auto' will change regime based on determination of phase boundaries
                                

# # print("\n=== ICING TEMPERATURE AND HUMIDITY CRITERIA ANALYSIS HOURS ===")

# humidity_temperature_results = fn.temp_hum_criteria(dataset=dataset_ice_load_rh,
#                                                     humidity_threshold=0.95,  # Relative Humidity threshold (%)
#                                                     temperature_threshold=263.15,  # Temperature threshold (K)
#                                                     height_level=height,
#                                                     save_plots=True)

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
#     WS_min=None,         # Wind Speed (m/s) - Disable for WD testing
#     WS_max=None,        # Wind Speed (m/s)
#     WD_max=5,        # Wind Direction (degrees) - Less aggressive: North to South  
#     WD_min=0,        # Wind Direction (degrees) - This should leave more data
    
#     height_level=height,
    
#     # Enable automatic ice load CDF analysis
#     calculate_ice_load_cdf=True,    # Disable ice load calculation for faster testing
#     dates=dates,
#     ice_load_method=ice_load_method,
#     ice_load_threshold=0.1,
#     months=None,  # Winter season if wanted
#     percentile=None,     # Remove extreme outliers if wanted
#     verbose=True   # Whether to print filtering information
# )

# print("\n=== METEOROLOGICAL FILTERING + AUTOMATIC CDF ANALYSIS ; SYSTEMATIC ===")
# results_sys_filter = fn.systematic_meteorological_filtering(
#     dataset,
#     WD_range=(0, 360, 90),           # Wind direction range: min=0, max=360, step=10 -> [0, 10, 20, ..., 360]
#     WS_range=None,           # Wind speed range: min=5, max=15, step=5 -> [5, 10, 15]
#     T_range=None,        # Temperature range  
#     PBLH_range=None,    # Boundary layer height
#     PRECIP_range=None,      # Precipitation
#     QVAPOR_range=None,      # Water vapor mixing ratio
#     RMOL_range=None,        # Monin-Obukhov length
#     calculate_ice_load_cdf=True,
#     dates=dates,
#     ice_load_method=5,
#     ice_load_threshold=0.1,
#     months=None,       # Winter months
#     save_results=True,
#     height_level=height,
# )

# results_filters = fn.analyze_ice_load_with_filtering_and_cdf(
#     dataset_with_ice_load = dataset_with_ice_load,
#     ice_load_variable='ICE_LOAD',
#     height_level=height,
#     save_plots=True,
#     results_subdir="filtered_ice_load_cdf_analysis",
#     # Filtering parameters (min, max for each variable)
#     WD_range=(210, 230),        # (min, max) for Wind Direction
#     WS_range=None,        # (min, max) for Wind Speed
#     T_range=None,         # (min, max) for Temperature
#     PBLH_range=None,      # (min, max) for Boundary Layer Height
#     PRECIP_range=None,    # (min, max) for Precipitation
#     QVAPOR_range=None,    # (min, max) for Water Vapor
#     RMOL_range=None,      # (min, max) for Monin-Obukhov Length
#     # CDF analysis parameters
#     ice_load_threshold=0.1,
#     ice_load_bins=None,
#     months=None,
#     percentile=None
# )

# print("\n=== METEOROLOGICAL FILTERING + AUTOMATIC CDF ANALYSIS ; SYSTEMATIC + WEIGHTED NEIGHBOUR CELLS ===")

# results_w_weights = fn.analyze_ice_load_with_weighted_neighborhood_cdf(
#     dataset_with_ice_load=dataset_with_ice_load,  # Changed from 'dataset' to 'dataset_with_ice_load'
#     height_level=height,
#     neighborhood_type='24-neighbors', # '4-neighbors', '8-neighbors', '24-neighbors'
#     weight_scheme='distance',  # 'uniform', 'distance', 'custom'
#     # Filtering parameters (min, max for each variable)
#     WD_range=None,        # (min, max) for Wind Direction
#     WS_range=None,        # (min, max) for Wind Speed
#     T_range=None,         # (min, max) for Temperature
#     PBLH_range=None,      # (min, max) for Boundary Layer Height
#     PRECIP_range=None,    # (min, max) for Precipitation
#     QVAPOR_range=None,    # (min, max) for Water Vapor
#     RMOL_range=None,      # (min, max) for Monin-Obukhov Length
#     # CDF analysis parameters
#     ice_load_threshold=0.1,
#     ice_load_bins=None,
#     months=None,
#     percentile=None
# )

# If custom weights:
# custom_weights = {
#     'adjacent': 1.0,      # Full weight for direct neighbors
#     'diagonal': 0.7,      # Reduced weight for diagonal neighbors  
#     'second_line': 0.5,   # Half weight for 2-step neighbors
#     'second_diagonal': 0.3 # Lower weight for far diagonal neighbors
# }
# and remeber: custom_weights=custom_weights,


# TEMPORAL GRADIENTS

# Example: Enhanced filtering with automatic ice load CDF analysis

# ICE LOAD RESAMPLING ANALYSIS

# print("\n=== ICE LOAD RESAMPLING ANALYSIS ===")
# resampling_results = fn.ice_load_resampling_analysis(
#     dataset_with_ice_load=dataset_with_ice_load,
#     ice_load_variable='ICE_LOAD',
#     height_level=height,
#     resampling_years=1,  # Aggregate data into X-year periods
#     save_plots=True,
#     months=None,  # Use all months, or specify [12,1,2,3] for winter
#     ice_load_threshold=0  # Include all ice load values
# )

# print("\n=== ICE LOAD RESAMPLING ANALYSIS EXCEEDANCE HOURS ===")

# resampling_results_hours = fn.ice_load_resampling_analysis_hours(
#     dataset_with_ice_load=dataset_with_ice_load,
#     ice_load_variable='ICE_LOAD',
#     height_level=height,
#     resampling_years=1,  # Aggregate data into X-year periods
#     save_plots=True,
#     months=None,  # Use all months, or specify [12,1,2,3] for winter
#     ice_load_threshold=0.1  # Include all ice load values
# )

# CORRELATION WITH METEOROLOGICAL VARIABLES

# results_correlation = fn.correlation_with_met_variables(
#     dataset_with_ice_load=dataset_with_ice_load,
#     met_variable='RMOL',  # Meteorological variable to correlate with (e.g., 'WS', 'T', 'PRECIP', 'RMOL')
#     height_level=height,
#     ice_load_variable='ICE_LOAD',
#     n_bins=100,  # Number of meteorological variable bins
#     ice_load_threshold=0.05  # Threshold for ice load values (kg/m)
# )



