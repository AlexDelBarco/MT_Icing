#!/usr/bin/env python3
"""
Direct test of the ice load calculation steps to find the exact issue
"""

import sys
import xarray as xr
import numpy as np
import pandas as pd

# Add the scripts directory to the Python path
sys.path.append('c:/Users/alexm/OneDrive/MIS DOCUMENTOS/DTU/MastersThesis/MT_Icing/scripts')
import functions as fn

def test_ice_load_steps():
    print("=== DIRECT ICE LOAD CALCULATION TEST ===\n")
    
    # Load dataset
    ds = xr.open_dataset('data/newa_wrf_for_jana_mstudent_extended_merged.nc')
    height = 2
    
    print("1. Creating test datasets...")
    
    # Get datasets 
    result_ws = fn.filter_dataset_by_thresholds(ds, WS_min=10, height_level=height, verbose=False)
    filtered_ws = result_ws[0] if isinstance(result_ws, tuple) else result_ws
    
    result_wd = fn.filter_dataset_by_thresholds(ds, WD_min=0, WD_max=45, height_level=height, verbose=False)
    filtered_wd = result_wd[0] if isinstance(result_wd, tuple) else result_wd
    
    # Find a time period that exists in both datasets
    ws_start = pd.Timestamp(filtered_ws.time.min().values)
    ws_end = pd.Timestamp(filtered_ws.time.max().values)
    wd_start = pd.Timestamp(filtered_wd.time.min().values)
    wd_end = pd.Timestamp(filtered_wd.time.max().values)
    
    # Get overlap period
    overlap_start = max(ws_start, wd_start)
    overlap_end = min(ws_end, wd_end)
    
    print(f"WS data range: {ws_start} to {ws_end}")
    print(f"WD data range: {wd_start} to {wd_end}")
    print(f"Overlap period: {overlap_start} to {overlap_end}")
    
    # Create small test datasets from overlap period
    test_start = overlap_start + pd.Timedelta(days=30)  # Start a bit later
    test_end = test_start + pd.Timedelta(days=90)       # 3 months of data
    
    print(f"\nUsing test period: {test_start} to {test_end}")
    
    ws_test = filtered_ws.sel(time=slice(test_start, test_end))
    wd_test = filtered_wd.sel(time=slice(test_start, test_end))
    
    print(f"WS test data: {len(ws_test.time)} timesteps")
    print(f"WD test data: {len(wd_test.time)} timesteps")
    
    if len(ws_test.time) == 0 or len(wd_test.time) == 0:
        print("No overlapping data found. Using alternative approach...")
        # Just take first 100 points from each
        ws_test = filtered_ws.isel(time=slice(0, min(100, len(filtered_ws.time))))
        wd_test = filtered_wd.isel(time=slice(0, min(100, len(filtered_wd.time))))
        test_start = pd.Timestamp(ws_test.time.min().values)
        test_end = pd.Timestamp(ws_test.time.max().values)
        print(f"Using WS period: {test_start} to {test_end}")
        print(f"WS test data: {len(ws_test.time)} timesteps")
        print(f"WD test data: {len(wd_test.time)} timesteps")
    
    print("\n2. Manually stepping through ice load calculation...")
    
    # Define test dates that cover our data
    dates = [test_start - pd.Timedelta(days=30), test_end + pd.Timedelta(days=30)]
    print(f"Test dates: {dates}")
    
    # Test the ice load calculation steps manually for WS data
    print("\n2a. Testing WS calculation steps...")
    try:
        print("  Getting ACCRE_CYL variable...")
        accre_ws = ws_test['ACCRE_CYL'].isel(height=height)
        print(f"  ACCRE shape: {accre_ws.shape}, dims: {accre_ws.dims}")
        
        print("  Getting ABLAT_CYL variable...")
        ablat_ws = ws_test['ABLAT_CYL'].isel(height=height)
        print(f"  ABLAT shape: {ablat_ws.shape}, dims: {ablat_ws.dims}")
        
        print("  Testing time slice...")
        winter_accre = accre_ws.sel(time=slice(dates[0], dates[1] - pd.Timedelta(minutes=30))).load()
        winter_ablat = ablat_ws.sel(time=slice(dates[0], dates[1] - pd.Timedelta(minutes=30))).load()
        
        print(f"  Winter ACCRE shape: {winter_accre.shape}")
        print(f"  Winter ABLAT shape: {winter_ablat.shape}")
        
        print("  Testing coordinate access...")
        print(f"  len(winter_accre.time): {len(winter_accre.time)}")
        print("  ✓ WS coordinate access successful!")
        
    except Exception as e:
        print(f"  ✗ WS failed at: {e}")
        import traceback
        traceback.print_exc()
    
    # Test the ice load calculation steps manually for WD data
    print("\n2b. Testing WD calculation steps...")
    try:
        print("  Getting ACCRE_CYL variable...")
        accre_wd = wd_test['ACCRE_CYL'].isel(height=height)
        print(f"  ACCRE shape: {accre_wd.shape}, dims: {accre_wd.dims}")
        
        print("  Getting ABLAT_CYL variable...")
        ablat_wd = wd_test['ABLAT_CYL'].isel(height=height)
        print(f"  ABLAT shape: {ablat_wd.shape}, dims: {ablat_wd.dims}")
        
        print("  Testing time slice...")
        winter_accre = accre_wd.sel(time=slice(dates[0], dates[1] - pd.Timedelta(minutes=30))).load()
        winter_ablat = ablat_wd.sel(time=slice(dates[0], dates[1] - pd.Timedelta(minutes=30))).load()
        
        print(f"  Winter ACCRE shape: {winter_accre.shape}")
        print(f"  Winter ABLAT shape: {winter_ablat.shape}")
        
        print("  Testing coordinate access...")
        print(f"  len(winter_accre.time): {len(winter_accre.time)}")
        print("  ✓ WD coordinate access successful!")
        
    except Exception as e:
        print(f"  ✗ WD failed at: {e}")
        import traceback
        traceback.print_exc()
    
    ds.close()

if __name__ == "__main__":
    test_ice_load_steps()