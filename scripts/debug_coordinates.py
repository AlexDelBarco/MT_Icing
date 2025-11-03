#!/usr/bin/env python3
"""
Debug script to compare coordinate structures between WS and WD filtered datasets
"""

import sys
import xarray as xr
import numpy as np
import pandas as pd

# Add the scripts directory to the Python path
sys.path.append('c:/Users/alexm/OneDrive/MIS DOCUMENTOS/DTU/MastersThesis/MT_Icing/scripts')
import functions as fn

def compare_coordinates():
    print("=== COMPARING WS vs WD COORDINATE STRUCTURES ===\n")
    
    # Load dataset
    ds = xr.open_dataset('data/newa_wrf_for_jana_mstudent_extended_merged.nc')
    height = 2
    
    # Apply WS filtering
    print("1. WS filtering...")
    result_ws = fn.filter_dataset_by_thresholds(ds, WS_min=10, height_level=height, verbose=False)
    filtered_ws = result_ws[0] if isinstance(result_ws, tuple) else result_ws
    small_ws = filtered_ws.isel(time=slice(0, 100))
    
    # Apply WD filtering
    print("2. WD filtering...")
    result_wd = fn.filter_dataset_by_thresholds(ds, WD_min=0, WD_max=45, height_level=height, verbose=False)
    filtered_wd = result_wd[0] if isinstance(result_wd, tuple) else result_wd
    small_wd = filtered_wd.isel(time=slice(0, 100))
    
    print("\n3. Detailed coordinate comparison...")
    
    print("\nWS Dataset:")
    print(f"  Dimensions: {dict(small_ws.dims)}")
    print(f"  Coordinates: {list(small_ws.coords.keys())}")
    print(f"  Time coordinate shape: {small_ws.time.shape}")
    print(f"  Time coordinate dims: {small_ws.time.dims}")
    
    # Check ACCRE_CYL variable specifically
    accre_ws = small_ws['ACCRE_CYL'].isel(height=2)
    print(f"  ACCRE_CYL at height 2 shape: {accre_ws.shape}")
    print(f"  ACCRE_CYL at height 2 dims: {accre_ws.dims}")
    print(f"  ACCRE_CYL at height 2 coords: {list(accre_ws.coords.keys())}")
    
    print("\nWD Dataset:")
    print(f"  Dimensions: {dict(small_wd.dims)}")
    print(f"  Coordinates: {list(small_wd.coords.keys())}")
    print(f"  Time coordinate shape: {small_wd.time.shape}")
    print(f"  Time coordinate dims: {small_wd.time.dims}")
    
    # Check ACCRE_CYL variable specifically
    accre_wd = small_wd['ACCRE_CYL'].isel(height=2)
    print(f"  ACCRE_CYL at height 2 shape: {accre_wd.shape}")
    print(f"  ACCRE_CYL at height 2 dims: {accre_wd.dims}")
    print(f"  ACCRE_CYL at height 2 coords: {list(accre_wd.coords.keys())}")
    
    print("\n4. Testing time slicing behavior...")
    
    start_date = pd.Timestamp('1989-07-01')
    end_date = pd.Timestamp('1989-12-31')
    
    print("\nWS time slicing:")
    try:
        ws_slice = accre_ws.sel(time=slice(start_date, end_date))
        print(f"  ✓ Success: shape {ws_slice.shape}, dims {ws_slice.dims}")
        print(f"  Time coord shape: {ws_slice.time.shape}")
        print(f"  Time coord dims: {ws_slice.time.dims}")
        print(f"  Time coord values (first 3): {ws_slice.time.values[:3]}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\nWD time slicing:")
    try:
        wd_slice = accre_wd.sel(time=slice(start_date, end_date))
        print(f"  ✓ Success: shape {wd_slice.shape}, dims {wd_slice.dims}")
        print(f"  Time coord shape: {wd_slice.time.shape}")
        print(f"  Time coord dims: {wd_slice.time.dims}")
        print(f"  Time coord values (first 3): {wd_slice.time.values[:3]}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Let's also check the raw coordinate details
    print("\n5. Raw coordinate inspection...")
    
    print(f"\nWS time coordinate details:")
    print(f"  Type: {type(small_ws.time)}")
    print(f"  Shape: {small_ws.time.shape}")
    print(f"  Size: {small_ws.time.size}")
    print(f"  Dims: {small_ws.time.dims}")
    print(f"  Coordinates: {list(small_ws.time.coords.keys())}")
    
    print(f"\nWD time coordinate details:")
    print(f"  Type: {type(small_wd.time)}")
    print(f"  Shape: {small_wd.time.shape}")
    print(f"  Size: {small_wd.time.size}")
    print(f"  Dims: {small_wd.time.dims}")
    print(f"  Coordinates: {list(small_wd.time.coords.keys())}")
    
    ds.close()

if __name__ == "__main__":
    compare_coordinates()