# Debug script to isolate WD vs WS filtering differences
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
dates = pd.date_range('1989-01-01T00:00:00.000000000', '2022-12-31T23:30:00.000000000', freq='YS-JUL')

print("=== COMPARING WS vs WD FILTERING ===")

# Test 1: WS filtering (known to work)
print("\n1. Testing WS filtering...")
try:
    filtered_ws, results_ws = fn.filter_dataset_by_thresholds(
        dataset=dataset,
        WS_min=10,
        height_level=height,
        calculate_ice_load_cdf=False,  # First test without ice load
        verbose=True
    )
    print(f"✓ WS filtering successful: {results_ws['final_timesteps']} timesteps")
    print(f"  Filtered dataset shape: {filtered_ws.sizes}")
    print(f"  Variables in filtered dataset: {list(filtered_ws.data_vars.keys())}")
except Exception as e:
    print(f"✗ WS filtering failed: {e}")

# Test 2: WD filtering (problematic)
print("\n2. Testing WD filtering...")
try:
    filtered_wd, results_wd = fn.filter_dataset_by_thresholds(
        dataset=dataset,
        WD_min=0,
        WD_max=45,
        height_level=height,
        calculate_ice_load_cdf=False,  # First test without ice load
        verbose=True
    )
    print(f"✓ WD filtering successful: {results_wd['final_timesteps']} timesteps")
    print(f"  Filtered dataset shape: {filtered_wd.sizes}")
    print(f"  Variables in filtered dataset: {list(filtered_wd.data_vars.keys())}")
except Exception as e:
    print(f"✗ WD filtering failed: {e}")

# Test 3: Compare dataset structures
print("\n3. Comparing dataset structures...")
if 'filtered_ws' in locals() and 'filtered_wd' in locals():
    print("\nWS-filtered dataset:")
    print(f"  Coordinates: {list(filtered_ws.coords.keys())}")
    print(f"  WS variable shape: {filtered_ws.WS.shape}")
    print(f"  WS variable dims: {filtered_ws.WS.dims}")
    
    print("\nWD-filtered dataset:")
    print(f"  Coordinates: {list(filtered_wd.coords.keys())}")
    print(f"  WS variable shape: {filtered_wd.WS.shape}")
    print(f"  WS variable dims: {filtered_wd.WS.dims}")
    
    # Check if coordinates are the same
    ws_coords = set(filtered_ws.coords.keys())
    wd_coords = set(filtered_wd.coords.keys())
    if ws_coords != wd_coords:
        print(f"\n⚠️  Coordinate mismatch!")
        print(f"  WS-only coords: {ws_coords - wd_coords}")
        print(f"  WD-only coords: {wd_coords - ws_coords}")
    else:
        print("\n✓ Coordinates match")

# Test 4: Try ice load calculation on smaller datasets
print("\n4. Testing ice load calculation on filtered datasets...")

if 'filtered_ws' in locals():
    print("\n4a. Testing ice load on WS-filtered dataset (small sample)...")
    try:
        # Get the actual time range of the filtered data for a proper subset
        start_time = filtered_ws.time.min().values
        end_time = filtered_ws.time.max().values
        print(f"  WS filtered data spans: {pd.Timestamp(start_time)} to {pd.Timestamp(end_time)}")
        
        # Create dates array that matches the actual data range
        start_year = pd.Timestamp(start_time).year
        test_dates = [pd.Timestamp(f'{start_year}-07-01'), pd.Timestamp(f'{start_year+1}-07-01')]
        
        ice_load_ws = fn.calculate_ice_load(
            filtered_ws, 
            test_dates,  # Use appropriate date range
            5, 
            height_level=height, 
            create_figures=False
        )
        print(f"✓ Ice load calculation successful on WS-filtered data")
        print(f"  Ice load shape: {ice_load_ws.shape if ice_load_ws is not None else 'None'}")
    except Exception as e:
        print(f"✗ Ice load calculation failed on WS-filtered data: {e}")

if 'filtered_wd' in locals():
    print("\n4b. Testing ice load on WD-filtered dataset (small sample)...")
    try:
        # Get the actual time range of the filtered data for a proper subset
        start_time = filtered_wd.time.min().values
        end_time = filtered_wd.time.max().values
        print(f"  WD filtered data spans: {pd.Timestamp(start_time)} to {pd.Timestamp(end_time)}")
        
        # Create dates array that matches the actual data range
        start_year = pd.Timestamp(start_time).year
        test_dates = [pd.Timestamp(f'{start_year}-07-01'), pd.Timestamp(f'{start_year+1}-07-01')]
        
        ice_load_wd = fn.calculate_ice_load(
            filtered_wd, 
            test_dates,  # Use appropriate date range
            5, 
            height_level=height, 
            create_figures=False
        )
        print(f"✓ Ice load calculation successful on WD-filtered data")
        print(f"  Ice load shape: {ice_load_wd.shape if ice_load_wd is not None else 'None'}")
    except Exception as e:
        print(f"✗ Ice load calculation failed on WD-filtered data: {e}")
        import traceback
        traceback.print_exc()

print("\n=== DEBUG COMPLETE ===")