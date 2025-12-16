# Test script for temperature comparison function
import os
import sys

# Set working directory to project root if running from scripts folder
current_dir = os.getcwd()
if current_dir.endswith('scripts'):
    os.chdir('..')
    print(f"Changed working directory from {current_dir} to {os.getcwd()}")

# Add scripts directory to Python path
scripts_dir = os.path.join(os.getcwd(), 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import functions as fn
import pandas as pd
import numpy as np
import xarray as xr

# Load datasets as in EMD_NEWA_comparison.py
height = 0  # Height level index to use (0-based): 0=50m; 1=100m; 2=150m

print("=== LOADING NEWA DATA ===")
merged_file = "data/newa_wrf_for_jana_mstudent_extended_merged.nc"

if os.path.exists(merged_file):
    dataset = fn.load_netcdf_data(merged_file)
else:
    print("Merged file not found. Please run EMD_NEWA_comparison.py first to create merged dataset.")
    exit()

print("\n=== LOADING EMD DATA ===")
data2 = "data/EMD_data/EmdWrf_N59.600_E019.960.txt"
emd_data = fn.import_emd_data(data2)

print("\n=== RUNNING TEMPERATURE COMPARISON ===")
# EMD point coordinates
emd_coords = (19.960, 59.600)  # EMD's data coordinates

# Run temperature comparison
temp_results = fn.compare_temperature_emd_newa(
    emd_data=emd_data,
    newa_data=dataset,
    height=height,
    emd_coordinates=emd_coords,
    save_plots=True
)

if temp_results:
    print("\n=== TEMPERATURE COMPARISON COMPLETED SUCCESSFULLY ===")
    print(f"Results saved in: {temp_results['output_directory']}")
    print(f"Height analyzed: {temp_results['height_meters']} m")
    print(f"Number of data points: {temp_results['statistics']['n_points']}")
    print(f"Mean bias (EMD - NEWA): {temp_results['statistics']['mean_bias']:.3f} K")
    print(f"RMSE: {temp_results['statistics']['rmse']:.3f} K")
    print(f"Correlation: {temp_results['statistics']['correlation']:.3f}")
else:
    print("ERROR: Temperature comparison failed!")