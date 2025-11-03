#!/usr/bin/env python3
"""
Debug script to isolate the time slicing issue
"""

import sys
import xarray as xr
import pandas as pd
import numpy as np

# Add the scripts directory to the Python path
sys.path.append('c:/Users/alexm/OneDrive/MIS DOCUMENTOS/DTU/MastersThesis/MT_Icing/scripts')
import functions as fn

def test_time_slicing():
    print("=== TIME SLICING DEBUG ===\n")
    
    # Load and filter datasets
    print("1. Loading and filtering datasets...")
    ds = xr.open_dataset('data/newa_wrf_for_jana_mstudent_extended_merged.nc')
    
    # Apply WS filtering
    ws_result = fn.filter_dataset_by_thresholds(ds, WS_min=10, height_level=2, verbose=False)
    ws_filtered = ws_result[0] if isinstance(ws_result, tuple) else ws_result
    
    print(f"WS filtered: {len(ws_filtered.time)} timesteps")
    print(f"WS time range: {ws_filtered.time.min().values} to {ws_filtered.time.max().values}")
    
    # Get ACCRE variable at height level 2
    print("\n2. Getting ACCRE variable...")
    accre_ws = ws_filtered['ACCRE_CYL'].isel(height=2)
    print(f"ACCRE WS shape: {accre_ws.shape}")
    print(f"ACCRE WS time coordinate type: {type(accre_ws.time.values[0])}")
    print(f"ACCRE WS time sample: {accre_ws.time.values[:3]}")
    
    # Test different time slice approaches
    print("\n3. Testing time slice approaches...")
    
    # Define test period
    start_date = pd.Timestamp('1989-01-01')
    end_date = pd.Timestamp('1989-02-01')
    
    print(f"Slicing from {start_date} to {end_date}")
    
    # Method 1: Direct slice
    print("\n3a. Method 1 - Direct slice...")
    try:
        slice1 = accre_ws.sel(time=slice(start_date, end_date))
        print(f"✓ Direct slice successful: {slice1.shape}")
    except Exception as e:
        print(f"✗ Direct slice failed: {e}")
    
    # Method 2: String-based slice
    print("\n3b. Method 2 - String slice...")
    try:
        slice2 = accre_ws.sel(time=slice('1989-01-01', '1989-02-01'))
        print(f"✓ String slice successful: {slice2.shape}")
    except Exception as e:
        print(f"✗ String slice failed: {e}")
    
    # Method 3: Index-based approach
    print("\n3c. Method 3 - Index-based...")
    try:
        # Find indices for the time range
        time_mask = (accre_ws.time >= start_date) & (accre_ws.time <= end_date)
        slice3 = accre_ws.isel(time=time_mask)
        print(f"✓ Index slice successful: {slice3.shape}")
    except Exception as e:
        print(f"✗ Index slice failed: {e}")
    
    # Method 4: Load first, then slice
    print("\n3d. Method 4 - Load then slice...")
    try:
        accre_loaded = accre_ws.load()
        slice4 = accre_loaded.sel(time=slice(start_date, end_date))
        print(f"✓ Load-then-slice successful: {slice4.shape}")
    except Exception as e:
        print(f"✗ Load-then-slice failed: {e}")
    
    # Check time coordinate integrity
    print("\n4. Checking time coordinate integrity...")
    print(f"Time coordinate is monotonic: {accre_ws.time.to_index().is_monotonic_increasing}")
    print(f"Time coordinate has duplicates: {accre_ws.time.to_index().has_duplicates}")
    print(f"Time coordinate dtype: {accre_ws.time.dtype}")
    
    # Close dataset
    ds.close()

if __name__ == "__main__":
    test_time_slicing()