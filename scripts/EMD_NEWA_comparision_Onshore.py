# IMPORTS
import os
import functions as fn
import pandas as pd
import numpy as np
import xarray as xr
import dask

# Set working directory to project root if running from scripts folder
current_dir = os.getcwd()
if current_dir.endswith('scripts'):
    os.chdir('..')
    print(f"Changed working directory from {current_dir} to {os.getcwd()}")

# IMPORT NEWA DATA
# Parameters

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

# Iimport data
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

if offshore == True:
    if not os.path.exists(offshore_final):
        print("Merged file not found. Creating merged dataset...")
        data1 = fn.merge_netcdf_files2(offshore_ice, offshore_rh, offshore_sfc, offshore_ws, offshore_final, verbose=True)
    else:
        print("Merged file already exists. Using existing merged dataset...")
        data1 = xr.open_dataset(offshore_final)

# Import merged NEWEA meteorological data
dataset = fn.load_netcdf_data(data1)

start_date = '1989-01-01T00:00:00.000000000'
end_date = '2022-12-31T23:30:00.000000000'

dates = pd.date_range(start_date, end_date, freq='YS-JUL')

# Ice Load

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
    # Use automatic chunking to match how the file was written and avoid memory issues
    dataset_with_ice_load = xr.open_dataset(filename, chunks='auto')  # Load with automatic optimal chunks

    print(f"Loaded dataset from: {filename}")
    print(f"Dataset dimensions: {dataset_with_ice_load.dims}")
    print(f"Available variables: {list(dataset_with_ice_load.data_vars.keys())}")
    print(f"Ice load variable 'ICE_LOAD' is ready for analysis at height level {height}: {dataset_with_ice_load.height.values[height]} m")

height_level = dataset_with_ice_load.height.values[height]

# IMPORT EMD DATA
data2 = "data/EMD_data/EmdWrf_N62.630_E022.489.txt"
emd_data = fn.import_emd_data(data2)



# EMD COMPARISON

# EMD point coordinates

emd_coords = (22.489, 62.630)  # EMD's data coordinates Onshore

# ACCRETION

# EMD_NEWA_acc1 = fn.compare_accretion_emd_newa(
#     emd_data=emd_data,
#     dataset_with_ice_load=dataset_with_ice_load,
#     height=height_level,
#     emd_coordinates=emd_coords,
#     accretion_threshold=0,  # in g/h
#     non_zero_percentage=0
# )

# EMD_NEWA_acc2 = fn.emd_newa_accretion_typical(
#     emd_data=emd_data,
#     dataset_with_ice_load=dataset_with_ice_load,
#     height=height_level,
#     emd_coordinates=emd_coords,
#     ice_accretion_threshold=0.1,  # in g/h
#     non_zero_percentage=25
#  )

# EMD_NEWA_acc3 = fn.pdf_emd_newa_accretion(
#      emd_data=emd_data,
#      dataset_with_ice_load=dataset_with_ice_load,
#      height=height_level,
#      emd_coordinates=emd_coords,
#      ice_accretion_threshold=0.1,  # in g/h
#      non_zero_percentage=0 #Filtered by percentage of hours with ice accretion > 0 in a day, taking only valid days
#  )

# ICE LOAD

# EMD_NEWA_results = fn.compare_ice_load_emd_newa(
#     emd_data=emd_data,
#     dataset_with_ice_load=dataset_with_ice_load,
#     height=height_level,
#     emd_coordinates=emd_coords
# )

# EMD_NEWA_meanDWY = fn.emd_newa_typical(
#     emd_data=emd_data,
#     dataset_with_ice_load=dataset_with_ice_load,
#     height=height_level,
#     emd_coordinates=emd_coords,
#     ice_load_threshold=0, # in g/h, day/week/year with accretion > threshold
#     non_zero_percentage=0
# )

EMD_NEWA_pdf = fn.pdf_emd_newa(
    emd_data=emd_data,
    dataset_with_ice_load=dataset_with_ice_load,
    height=height_level,
    emd_coordinates=emd_coords,
    ice_load_threshold=0.001, # in kg/h
    non_zero_percentage=0 #Filtered by percentage of hours with ice accretion > 0 in a day, taking only valid days
)