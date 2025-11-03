#!/usr/bin/env python3
"""
Minimal test to identify where ice load calculation hangs
"""

import sys
import xarray as xr
import numpy as np
import pandas as pd

# Add the scripts directory to the Python path
sys.path.append('c:/Users/alexm/OneDrive/MIS DOCUMENTOS/DTU/MastersThesis/MT_Icing/scripts')
import functions as fn

def test_ice_load_minimal():
    print("=== Minimal Ice Load Test ===\n")
    
    # Load and filter dataset
    ds = xr.open_dataset('data/newa_wrf_for_jana_mstudent_extended_merged.nc')
    print("Original dataset loaded successfully")
    
    # Apply WD filtering to get a small dataset
    result = fn.filter_dataset_by_thresholds(ds, WD_min=0, WD_max=45, height_level=2, verbose=False)
    filtered_ds = result[0]  # Get the dataset from the tuple
    
    print(f"Filtered dataset: {filtered_ds.dims}")
    print(f"Time range: {filtered_ds.time.min().values} to {filtered_ds.time.max().values}")
    
    # Create a very small test with just a few months
    start_date = pd.Timestamp('1989-07-01')
    end_date = pd.Timestamp('1989-12-31')
    small_ds = filtered_ds.sel(time=slice(start_date, end_date))
    
    print(f"Small test dataset: {small_ds.dims}")
    print(f"Small time range: {small_ds.time.min().values} to {small_ds.time.max().values}")
    
    # Test each component of ice load calculation step by step
    print("\n1. Testing ACCRE_CYL access...")
    accre = small_ds['ACCRE_CYL'].isel(height=2)
    print(f"ACCRE shape: {accre.shape}")
    
    print("\n2. Testing ABLAT_CYL access...")
    ablat = small_ds['ABLAT_CYL'].isel(height=2)
    print(f"ABLAT shape: {ablat.shape}")
    
    print("\n3. Testing simple time slice...")
    winter_accre = accre.sel(time=slice(start_date, end_date))
    winter_ablat = ablat.sel(time=slice(start_date, end_date))
    print(f"Winter ACCRE shape: {winter_accre.shape}")
    print(f"Winter ABLAT shape: {winter_ablat.shape}")
    
    print("\n4. Testing coordinate access...")
    print(f"Winter ACCRE time length: {len(winter_accre.time)}")
    print(f"Winter ABLAT time length: {len(winter_ablat.time)}")
    
    print("\n5. Testing ice_load function directly...")
    try:
        # Call the ice_load function directly (not calculate_ice_load)
        print("Calling ice_load function...")
        load_result = fn.ice_load(winter_accre, winter_ablat, 5)
        print(f"✓ ice_load function successful!")
        print(f"Result shape: {load_result.shape if load_result is not None else 'None'}")
        print(f"Result type: {type(load_result)}")
    except Exception as e:
        print(f"✗ ice_load function failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n6. Testing calculate_ice_load with minimal dates...")
    try:
        # Use a very simple date range
        test_dates = [pd.Timestamp('1989-07-01'), pd.Timestamp('1989-12-31')]
        print(f"Test dates: {test_dates}")
        
        print("Calling calculate_ice_load...")
        result = fn.calculate_ice_load(
            small_ds, 
            test_dates,
            5,  # method
            height_level=2, 
            create_figures=False
        )
        print(f"✓ calculate_ice_load successful!")
        print(f"Result shape: {result.shape if result is not None else 'None'}")
    except Exception as e:
        print(f"✗ calculate_ice_load failed: {e}")
        import traceback
        traceback.print_exc()
    
    ds.close()

if __name__ == "__main__":
    test_ice_load_minimal()