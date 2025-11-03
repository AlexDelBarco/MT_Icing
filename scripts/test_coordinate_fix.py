#!/usr/bin/env python3
"""
Test script to verify coordinate fix works for ice load calculation
"""

import sys
import xarray as xr
import numpy as np
import pandas as pd

# Add the scripts directory to the Python path
sys.path.append('c:/Users/alexm/OneDrive/MIS DOCUMENTOS/DTU/MastersThesis/MT_Icing/scripts')
import functions as fn

def test_coordinate_structure():
    print("=== Testing Coordinate Structure After Filtering ===\n")
    
    # Load dataset
    ds = xr.open_dataset('data/newa_wrf_for_jana_mstudent_extended_merged.nc')
    print(f"Original dataset dimensions: {dict(ds.dims)}")
    print(f"Original time coordinate shape: {ds.time.shape}")
    print(f"Original time coordinate dims: {ds.time.dims}")
    
    # Apply WS filtering
    print("\n1. Testing WS filtering...")
    result = fn.filter_dataset_by_thresholds(
        ds, WS_min=10, height_level=2, verbose=False
    )
    
    # Check what the function returns
    print(f"Function returned type: {type(result)}")
    if isinstance(result, tuple):
        print(f"Tuple length: {len(result)}")
        ws_filtered = result[0]  # Assume first element is the dataset
    else:
        ws_filtered = result
    
    print(f"WS filtered dimensions: {dict(ws_filtered.dims)}")
    print(f"WS filtered time coordinate shape: {ws_filtered.time.shape}")
    print(f"WS filtered time coordinate dims: {ws_filtered.time.dims}")
    print(f"WS filtered coordinates: {list(ws_filtered.coords.keys())}")
    
    # Test accessing ACCRE_CYL variable
    print("\n2. Testing ACCRE_CYL variable access...")
    try:
        accre_var = ws_filtered['ACCRE_CYL']
        print(f"ACCRE_CYL shape: {accre_var.shape}")
        print(f"ACCRE_CYL dims: {accre_var.dims}")
        print(f"ACCRE_CYL coordinates: {list(accre_var.coords.keys())}")
        
        # Test height indexing
        print("\n3. Testing height indexing...")
        accre_at_height = accre_var.isel(height=2)
        print(f"ACCRE at height 2 shape: {accre_at_height.shape}")
        print(f"ACCRE at height 2 dims: {accre_at_height.dims}")
        print(f"ACCRE at height 2 coordinates: {list(accre_at_height.coords.keys())}")
        
        # Test time slicing
        print("\n4. Testing time slicing...")
        first_date = pd.Timestamp('1989-07-01')
        second_date = pd.Timestamp('1990-06-30')
        
        try:
            winter_slice = accre_at_height.sel(time=slice(first_date, second_date))
            print(f"Winter slice shape: {winter_slice.shape}")
            print(f"Winter slice dims: {winter_slice.dims}")
            print(f"Winter slice coordinates: {list(winter_slice.coords.keys())}")
            
            # The problematic line
            print(f"Checking len(winter_slice.time): {len(winter_slice.time)}")
            print("✓ Time coordinate access successful!")
            
        except Exception as e:
            print(f"✗ Time slicing failed: {e}")
            print(f"Error type: {type(e)}")
            
            # Try to understand the coordinate structure
            print("\nDebugging coordinate structure...")
            print(f"accre_at_height.sizes: {accre_at_height.sizes}")
            print(f"accre_at_height.dims: {accre_at_height.dims}")
            print(f"accre_at_height.shape: {accre_at_height.shape}")
            
    except Exception as e:
        print(f"✗ ACCRE_CYL access failed: {e}")
    
    # Close dataset
    ds.close()

if __name__ == "__main__":
    test_coordinate_structure()