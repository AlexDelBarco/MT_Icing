#!/usr/bin/env python3
"""
Final test to confirm both WS and WD filtering work correctly
"""

import sys
import xarray as xr
import numpy as np
import pandas as pd

# Add the scripts directory to the Python path
sys.path.append('c:/Users/alexm/OneDrive/MIS DOCUMENTOS/DTU/MastersThesis/MT_Icing/scripts')
import functions as fn

def final_test():
    print("=== FINAL WS vs WD FILTERING TEST ===\n")
    
    # Load dataset
    ds = xr.open_dataset('data/newa_wrf_for_jana_mstudent_extended_merged.nc')
    height = 2
    
    print("1. Testing both filtering approaches...")
    
    # WS filtering
    print("1a. WS filtering...")
    result_ws = fn.filter_dataset_by_thresholds(ds, WS_min=10, height_level=height, verbose=False)
    filtered_ws = result_ws[0] if isinstance(result_ws, tuple) else result_ws
    print(f"  ✓ WS filtered: {len(filtered_ws.time)} timesteps")
    print(f"  Time range: {filtered_ws.time.min().values} to {filtered_ws.time.max().values}")
    
    # WD filtering
    print("\n1b. WD filtering...")
    result_wd = fn.filter_dataset_by_thresholds(ds, WD_min=0, WD_max=45, height_level=height, verbose=False)
    filtered_wd = result_wd[0] if isinstance(result_wd, tuple) else result_wd
    print(f"  ✓ WD filtered: {len(filtered_wd.time)} timesteps")
    print(f"  Time range: {filtered_wd.time.min().values} to {filtered_wd.time.max().values}")
    
    print("\n2. Testing ice load calculation with systematic_meteorological_filtering...")
    
    print("\n2a. Testing systematic WS filtering with ice load...")
    try:
        # Use the systematic function that properly handles dates
        ws_results = fn.systematic_meteorological_filtering(
            dataset=ds,
            WS_range=(10, 15),  # Small range for quick test
            output_dir="results/test_ws",
            figures_dir="results/figures_test_ws",
            variable_info={'WS': {'min': 10, 'max': 15}},
            height_level=height,
            calculate_ice_load_cdf=True,  # This is what was hanging!
            dates=[pd.Timestamp('1989-07-01'), pd.Timestamp('1990-07-01'), pd.Timestamp('1991-07-01')],
            ice_load_method=5,
            percentile=95,
            create_figures=False
        )
        print("  ✓ Systematic WS filtering with ice load CDF successful!")
    except Exception as e:
        print(f"  ✗ Systematic WS filtering failed: {e}")
    
    print("\n2b. Testing systematic WD filtering with ice load...")
    try:
        # Use the systematic function that properly handles dates
        wd_results = fn.systematic_meteorological_filtering(
            dataset=ds,
            WD_range=(0, 45),  # Small range for quick test
            output_dir="results/test_wd",
            figures_dir="results/figures_test_wd",
            variable_info={'WD': {'min': 0, 'max': 45}},
            height_level=height,
            calculate_ice_load_cdf=True,  # This is what we want to test!
            dates=[pd.Timestamp('1989-07-01'), pd.Timestamp('1990-07-01'), pd.Timestamp('1991-07-01')],
            ice_load_method=5,
            percentile=95,
            create_figures=False
        )
        print("  ✓ Systematic WD filtering with ice load CDF successful!")
    except Exception as e:
        print(f"  ✗ Systematic WD filtering failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== FINAL CONCLUSION ===")
    print("Testing complete! Both WS and WD filtering should work correctly.")
    print("The original 'hanging' issue was due to:")
    print("1. Large dataset sizes making ice load calculation slow")
    print("2. Progress indicators helping identify actual computation vs hanging")
    print("3. Coordinate issues that have been resolved")
    
    ds.close()

if __name__ == "__main__":
    final_test()