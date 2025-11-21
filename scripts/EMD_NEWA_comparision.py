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

# IMPORT NEWA DATA
height = 1  # Height level index to use (0-based): 0=50m; 1=100m; 2=150m
ice_load_method = 51  # Method for ice load calculation
calculate_new_ice_load = False  # Whether to calculate ice load or load existing data

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

start_date = '1989-01-01T00:00:00.000000000'
end_date = '2022-12-31T23:30:00.000000000'

dates = pd.date_range(start_date, end_date, freq='YS-JUL')

# Ice load

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

else:
    print("Loading existing complete dataset with ice load...")
    filename = f"results/dataset_iceload_19890701_20220701_h{height}.nc"
    dataset_with_ice_load = xr.open_dataset(filename)  # Load complete dataset

    print(f"Loaded dataset from: {filename}")
    print(f"Dataset dimensions: {dataset_with_ice_load.dims}")
    print(f"Available variables: {list(dataset_with_ice_load.data_vars.keys())}")
    print(f"Ice load variable 'ICE_LOAD' is ready for analysis at height level {height}: {dataset_with_ice_load.height.values[height]} m")

height_level = dataset_with_ice_load.height.values[height]

# IMPORT EMD DATA
data2 = "data/EMD_data/EmdWrf_N59.600_E019.960.txt"
emd_data = fn.import_emd_data(data2)


# EMD COMPARISON

# EMD point coordinates
emd_coords = (19.960, 59.600)  # EMD's data coordinates

# gridEMD_results = fn.plot_grid_with_extra_point(dataset=dataset_with_ice_load,
#                                                 extra_point_coords=emd_coords,
#                                                 extra_point_label='EMD',
#                                                 plot_title="NEWA Grid with EMD Location"
# )

EMD_NEW_results = fn.compare_ice_load_emd_newa(
    emd_data=emd_data,
    dataset_with_ice_load=dataset_with_ice_load,
    height=height_level,
    emd_coordinates=emd_coords
)