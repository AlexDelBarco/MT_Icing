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

site = "Offshore"  # Choose between "Onshore" or "Offshore", corresponding to the dayasets/sites

if site == "Onshore":
    onshore = True
    offshore = False
    OffOn = "Onshore"
if site == "Offshore":
    onshore = False
    offshore = True
    OffOn = "Offshore"

height = 0  # Height level index to use (0-based): 0=100m; 1=150m
ice_load_method = 51  # Method for ice load calculation
calculate_new_ice_load = False  # Whether to calculate ice load or load existing data

# IMPORT DATA
# Onshore

onshore_ice = "data/BigDomains/alexandre_onshore_domain/onshore_ice_step8.nc"
onshore_rh = "data/BigDomains/alexandre_onshore_domain/onshore_rh_step8.nc"
onshore_sfc = "data/BigDomains/alexandre_onshore_domain/onshore_sfc_step8.nc"
onshore_ws = "data/BigDomains/alexandre_onshore_domain/onshore_ws_step8.nc"
onshore_final = "data/BigDomains/alexandre_onshore_domain/onshore_final_step8.nc"

if onshore == True:

    if not os.path.exists(onshore_final):
        print("Merged file not found. Creating merged dataset...")
        data1 = fn.merge_netcdf_files2(onshore_ice, onshore_rh, onshore_sfc, onshore_ws, onshore_final, verbose=True)
    else:
        print("Merged file already exists. Using existing merged dataset...")
        data1 = xr.open_dataset(onshore_final)

# Offshore
offshore_ice = "data/BigDomains/alexandre_offshore_domain/offshore_ice_step8.nc"
offshore_rh = "data/BigDomains/alexandre_offshore_domain/offshore_rh_step8.nc"
offshore_sfc = "data/BigDomains/alexandre_offshore_domain/offshore_sfc_step8.nc"
offshore_ws = "data/BigDomains/alexandre_offshore_domain/offshore_ws_step8.nc"
offshore_final = "data/BigDomains/alexandre_offshore_domain/offshore_final_step8.nc"

# dataset_ice = fn.load_netcdf_data(offshore_ice)
# dataset_rh = fn.load_netcdf_data(offshore_rh)
# dataset_sfc = fn.load_netcdf_data(offshore_sfc)
# dataset_ws = fn.load_netcdf_data(offshore_ws)

if offshore == True:
    if not os.path.exists(offshore_final):
        print("Merged file not found. Creating merged dataset...")
        data1 = fn.merge_netcdf_files2(offshore_ice, offshore_rh, offshore_sfc, offshore_ws, offshore_final, verbose=True)
    else:
        print("Merged file already exists. Using existing merged dataset...")
        data1 = xr.open_dataset(offshore_final)


# EXPLORE DATASET

# Plot grid location on map
# fn.plot_grid_points_cartopy_map(data1, margin_degrees=2.3, zoom_level=8, title="Grid Points - Terrain Map")

# Get the dataset with only the nearest point
# target_latitude = 59.585484    # Your target latitude
# target_longitude = 19.970825  # Your target longitude
# nearest_point_data = fn.find_nearest_point(data1, target_latitude, target_longitude)


# CALCULATIONS AND PLOTS

# Period
start_date = '1989-01-01T00:00:00.000000000'
end_date = '2022-12-31T23:30:00.000000000'

dates = pd.date_range(start_date, end_date, freq='YS-JUL')

# ACCERATION

# # Accreation for winter and time period + plot
# fn.accreation_per_winter(dataset, start_date, end_date, height_level=height)

# ICE LOAD AND ICING TEMPERATURE AND HUMIDITY CRITERIA

#ice load data: load/calculate 
if calculate_new_ice_load:
    print("Calculating ice load...")
    #ice_load_data = fn.calculate_ice_load(dataset, dates, ice_load_method, height_level=height, create_figures=True)

    #Add ice load directly to the dataset
    print("=== ADDING ICE LOAD TO DATASET ===")

    dataset_with_ice_load = fn.add_ice_load_to_dataset(

        ds=data1,
        dates=dates,
        OffOn=OffOn,
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
    filename = f"results/dataset_iceload_{OffOn}_19890701_20220701_h{height}.nc"
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
#      OffOn=OffOn,
#      BigDomain=True,
#      months=None,  # Can specify winter months like [12, 1, 2, 3] if desired
#      show_colorbar=True
#  )

# Calculate relative humidity first (needed for color scale calculation)
#The rh is calculated using surface P, T and mixing ratio at height 
# scale-height ~8km, from Ch.2 of 46100's book\notes, then consider d(ln p)/d(ln z)

dataset_ice_load_rh = fn.add_rh(dataset_with_ice_load=dataset_with_ice_load, height_l=height,
                                phase= 'auto') #'liquid', 'solid', 'auto' 
                                                #– to make calculation valid in 'liquid' water 
                                                #(default) or 'solid' ice regimes. 'auto' will 
                                                # change regime based on determination of phase 
                                                # boundaries

# Plot ice load exceedance hours both th criteria and ice load criteria with same color scale

print("\n=== CALCULATING COMBINED COLOR SCALE RANGE ===")

# Calculate ice load exceedance matrix for color scale determination
print("Calculating ice load exceedance values...")
ice_load_data = dataset_with_ice_load['ICE_LOAD'].isel(height=height)
n_time = ice_load_data.sizes['time']
n_south_north = ice_load_data.sizes['south_north']
n_west_east = ice_load_data.sizes['west_east']

time_index = pd.to_datetime(ice_load_data.time.values)
n_years = len(time_index.year.unique())
time_step_hours = (time_index[1] - time_index[0]).total_seconds() / 3600 if len(time_index) > 1 else 0.5

# Calculate ice load exceedance matrix
ice_data_clean = ice_load_data.where(ice_load_data >= 0, 0)
ice_exceedance_matrix = np.zeros((n_south_north, n_west_east))

for i in range(n_south_north):
    for j in range(n_west_east):
        cell_data = ice_data_clean.isel(south_north=i, west_east=j)
        cell_values = cell_data.values
        valid_mask = ~np.isnan(cell_values)
        cell_values_clean = cell_values[valid_mask]
        
        if len(cell_values_clean) > 0:
            exceedances = np.sum(cell_values_clean >= 0.1)  # Use same threshold as in main call
            hours_per_year = (exceedances * time_step_hours) / n_years
            ice_exceedance_matrix[i, j] = hours_per_year
        else:
            ice_exceedance_matrix[i, j] = np.nan

# Calculate temperature-humidity criteria matrix for color scale determination
print("Calculating temperature-humidity criteria values...")
temp_data = dataset_ice_load_rh['T'].isel(height=height)
humidity_data = dataset_ice_load_rh['relative_humidity']
temp_hum_matrix = np.zeros((n_south_north, n_west_east))

for i in range(n_south_north):
    for j in range(n_west_east):
        cell_temp = temp_data.isel(south_north=i, west_east=j)
        cell_humidity = humidity_data.isel(south_north=i, west_east=j)
        
        temp_values = cell_temp.values
        humidity_values = cell_humidity.values
        
        valid_mask = ~(np.isnan(temp_values) | np.isnan(humidity_values))
        temp_clean = temp_values[valid_mask]
        humidity_clean = humidity_values[valid_mask]
        
        if len(temp_clean) > 0 and len(humidity_clean) > 0:
            temp_criteria = temp_clean <= 263.15  # Use same threshold as in main call
            humidity_criteria = humidity_clean >= 0.95  # Use same threshold as in main call
            both_criteria = temp_criteria & humidity_criteria
            criteria_count = np.sum(both_criteria)
            hours_per_year = (criteria_count * time_step_hours) / n_years
            temp_hum_matrix[i, j] = hours_per_year
        else:
            temp_hum_matrix[i, j] = np.nan

# Calculate combined min and max for consistent color scale
valid_ice_values = ice_exceedance_matrix[~np.isnan(ice_exceedance_matrix)]
valid_temp_hum_values = temp_hum_matrix[~np.isnan(temp_hum_matrix)]

if len(valid_ice_values) > 0 and len(valid_temp_hum_values) > 0:
    combined_min = min(np.min(valid_ice_values), np.min(valid_temp_hum_values))
    combined_max = max(np.max(valid_ice_values), np.max(valid_temp_hum_values))
    
    # Calculate combined 90th percentile for consistent clipping
    all_combined_values = np.concatenate([valid_ice_values, valid_temp_hum_values])
    combined_90p = np.percentile(all_combined_values, 90)
    
    # Check if we need 90th percentile clipping
    outlier_ratio = combined_max / combined_90p if combined_90p > 0 else 1
    if outlier_ratio > 2.0:
        combined_max_clipped = combined_90p
        clipping_applied = True
        print(f"Outliers detected (ratio: {outlier_ratio:.1f}), applying 90th percentile clipping")
    else:
        combined_max_clipped = combined_max
        clipping_applied = False
        print(f"No significant outliers detected (ratio: {outlier_ratio:.1f}), using full range")
else:
    combined_min, combined_max_clipped = 0, 1
    clipping_applied = False


print(f"Combined color scale range: {combined_min:.1f} to {combined_max_clipped:.1f} hours/year")
print(f"Ice load range: {np.min(valid_ice_values):.1f} to {np.max(valid_ice_values):.1f} hours/year")
print(f"Temp-humidity range: {np.min(valid_temp_hum_values):.1f} to {np.max(valid_temp_hum_values):.1f} hours/year")
if clipping_applied:
    print(f"90th percentile clipping applied: {combined_90p:.1f} hours/year (original max was {combined_max:.1f})")

# Now call both functions with the same color scale
grid_results_hours = fn.plot_ice_load_threshold_exceedance_map(
    dataset_with_ice_load=dataset_with_ice_load,
    ice_load_variable='ICE_LOAD',
    height_level=height,

    ice_load_threshold=0.05,
    save_plots=True,
    OffOn=OffOn,
    BigDomain=True,
    margin_degrees=0.5,  # Margin around grid in degrees for cartopy map
    zoom_level=6,  # Zoom level for terrain tiles
    custom_vmin=combined_min,
    custom_vmax=combined_max_clipped
)

# # print("\n=== ICING TEMPERATURE AND HUMIDITY CRITERIA ANALYSIS HOURS ===")

humidity_temperature_results = fn.temp_hum_criteria(dataset=dataset_ice_load_rh,
                                                    humidity_threshold=0.95,  # Relative Humidity threshold (%)
                                                    temperature_threshold=263.15,  # Temperature threshold (K)
                                                    height_level=height,
                                                    save_plots=True,
                                                    OffOn=OffOn,
                                                    BigDomain=True,
                                                    margin_degrees=0.5,  # Margin around grid in degrees for cartopy map
                                                    zoom_level=6,  # Zoom level for terrain tiles
                                                    custom_vmin=combined_min,
                                                    custom_vmax=combined_max_clipped)

# SPATIAL GRADIENTS

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
#     ice_load_threshold=0,
#     ice_load_bins=None,
#     months=None,
#     percentile=None,
#     OffOn=OffOn,
#     BigDomain=True,
#     margin_degrees=0.5,
#     zoom_level=6,
# )

# ICE LOAD RESAMPLING ANALYSIS

# print("\n=== ICE LOAD RESAMPLING ANALYSIS ===")
# resampling_results = fn.ice_load_resampling_analysis(
#     dataset_with_ice_load=dataset_with_ice_load,
#     ice_load_variable='ICE_LOAD',
#     height_level=height,
#     resampling_years=1,  # Aggregate data into X-year periods
#     save_plots=True,
#     months=None,  # Use all months, or specify [12,1,2,3] for winter
#     ice_load_threshold=0,  # Include all ice load values
#     OffOn=OffOn,
#     BigDomain=True
# )

# print("\n=== ICE LOAD RESAMPLING ANALYSIS EXCEEDANCE HOURS ===")

# resampling_results_hours = fn.ice_load_resampling_analysis_hours(
#     dataset_with_ice_load=dataset_with_ice_load,
#     ice_load_variable='ICE_LOAD',
#     height_level=height,
#     resampling_years=1,  # Aggregate data into X-year periods
#     save_plots=True,
#     months=None,  # Use all months, or specify [12,1,2,3] for winter
#     ice_load_threshold=0.1,  # Include all ice load values
#     OffOn=OffOn,
#     BigDomain=True
# )

