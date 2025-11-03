# Debug script to find the exact dimension issue
import os
import functions as fn
import pandas as pd
import numpy as np
import xarray as xr

# Set working directory
current_dir = os.getcwd()
if current_dir.endswith('scripts'):
    os.chdir('..')

# Load data
data1 = "data/newa_wrf_for_jana_mstudent_extended_merged.nc"
dataset = fn.load_netcdf_data(data1)
height = 2

print("=== DEBUGGING DIMENSION ISSUE ===")

# Filter with WD
print("\n1. Filtering with WD...")
filtered_wd, results_wd = fn.filter_dataset_by_thresholds(
    dataset=dataset,
    WD_min=0,
    WD_max=45,
    height_level=height,
    calculate_ice_load_cdf=False,
    verbose=False
)

print(f"Filtered dataset has {len(filtered_wd.time)} timesteps")

# Take a small sample
small_sample = filtered_wd.isel(time=slice(0, 10))

print("\n2. Examining small sample dataset structure...")
print(f"Dataset dims: {small_sample.dims}")
print(f"Dataset sizes: {small_sample.sizes}")

# Check ACCRE_CYL specifically
print("\n3. Examining ACCRE_CYL variable...")
accre_var = small_sample['ACCRE_CYL']
print(f"ACCRE_CYL shape: {accre_var.shape}")
print(f"ACCRE_CYL dims: {accre_var.dims}")
print(f"ACCRE_CYL sizes: {accre_var.sizes}")

# Check what happens when we select height level
print("\n4. Selecting height level...")
accre_height = accre_var.isel(height=height)
print(f"After height selection shape: {accre_height.shape}")
print(f"After height selection dims: {accre_height.dims}")
print(f"After height selection sizes: {accre_height.sizes}")

# Check if there's an issue with time selection
print("\n5. Testing time selection...")
try:
    dates = pd.date_range('1989-07-01', '1991-06-30', freq='YS-JUL')  # 2 years for testing
    date = dates[0]
    print(f"Dates available: {len(dates)} - {dates}")
    print(f"Selecting time from {date} to {dates[1] - pd.to_timedelta('30min')}")
    
    time_slice = slice(date, dates[1] - pd.to_timedelta('30min'))
    winter_accre = accre_height.sel(time=time_slice)
    print(f"After time selection: {winter_accre.shape}")
    print(f"After time selection dims: {winter_accre.dims}")
    print(f"After time selection sizes: {winter_accre.sizes}")
    
    # Try to access .time to see where the error occurs
    print(f"Accessing time coordinate...")
    time_coord = winter_accre.time
    print(f"Time coordinate shape: {time_coord.shape}")
    print(f"Time coordinate dims: {time_coord.dims}")
    
    # Try len() which is causing the error
    print(f"Trying len(winter_accre.time)...")
    time_len = len(winter_accre.time)
    print(f"✓ Length: {time_len}")
    
except Exception as e:
    print(f"✗ Error in time selection: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DEBUG COMPLETE ===")