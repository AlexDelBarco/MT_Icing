# Example usage of the enhanced filter_dataset_by_thresholds function

"""
The enhanced filter_dataset_by_thresholds function now automatically:
1. Filters the dataset by meteorological conditions
2. Calculates ice load on the filtered dataset
3. Creates CDF plots and saves them in organized folders
4. Generates documentation explaining the filters applied

Usage examples:
"""

import functions as fn
import pandas as pd

# Example 1: Basic filtering with automatic CDF analysis
def example_1_basic():
    """
    Filter by temperature and wind, then automatically calculate ice load and create CDF plots
    """
    filtered_ds, results = fn.filter_dataset_by_thresholds(
        dataset=dataset,
        # Meteorological filters
        T_min=273.15,      # Above 0°C
        T_max=285.15,      # Below 12°C
        WS_min=3.0,        # Above 3 m/s
        WS_max=20.0,       # Below 20 m/s
        height_level=2,    # Height level for filtering and ice load
        
        # Enable automatic ice load CDF analysis
        calculate_ice_load_cdf=True,
        dates=dates,                    # Date range for ice load calculation
        ice_load_method=5,              # Ice load calculation method
        ice_load_threshold=0.1,         # Minimum ice load for CDF
        months=[12, 1, 2, 3],          # Winter months
        percentile=99,                  # Remove extreme outliers
        verbose=True
    )
    
    # Results will be saved in: results/figures/spatial_gradient/filtered/T_min_273.15_T_max_285.15_WS_min_3.0_WS_max_20.0/
    # Contains: CDF plots, spatial gradient plots, and filter_documentation.txt
    
    return filtered_ds, results

# Example 2: Comprehensive filtering for icing conditions
def example_2_icing_conditions():
    """
    Apply comprehensive filtering for icing-favorable conditions
    """
    filtered_ds, results = fn.filter_dataset_by_thresholds(
        dataset=dataset,
        # Icing-favorable meteorological conditions
        PBLH_min=100,      # Stable atmospheric conditions
        PBLH_max=1500,     # Not too turbulent
        PRECIP_max=2.0,    # Low to moderate precipitation
        QVAPOR_min=0.002,  # Sufficient moisture
        QVAPOR_max=0.015,  # Not too humid
        T_min=263.15,      # Cold enough for icing (-10°C)
        T_max=275.15,      # Not too warm (+2°C)
        WS_min=5.0,        # Good wind for accretion
        WS_max=25.0,       # Not extreme wind
        height_level=2,
        
        # Automatic CDF analysis
        calculate_ice_load_cdf=True,
        dates=dates,
        ice_load_method=5,
        ice_load_threshold=0.05,        # Lower threshold for icing analysis
        months=[11, 12, 1, 2, 3, 4],   # Extended winter
        verbose=True
    )
    
    # Results saved in: results/figures/spatial_gradient/filtered/PBLH_min_100_PBLH_max_1500_PRECIP_max_2.0_QVAPOR_min_0.002/
    
    return filtered_ds, results

# Example 3: Extreme weather conditions
def example_3_extreme_weather():
    """
    Focus on extreme weather events
    """
    filtered_ds, results = fn.filter_dataset_by_thresholds(
        dataset=dataset,
        # Extreme conditions
        WS_min=15.0,       # High wind speeds
        T_max=268.15,      # Very cold (-5°C)
        PRECIP_min=1.0,    # Active precipitation
        height_level=2,
        
        # CDF analysis for extreme events
        calculate_ice_load_cdf=True,
        dates=dates,
        ice_load_method=5,
        ice_load_threshold=0.5,         # Focus on significant ice loads
        verbose=True
    )
    
    return filtered_ds, results

# Example 4: Just filtering without CDF analysis (original behavior)
def example_4_filtering_only():
    """
    Use only the filtering functionality without ice load calculation
    """
    filtered_ds, results = fn.filter_dataset_by_thresholds(
        dataset=dataset,
        T_min=273.15,
        WS_min=2.0,
        height_level=2,
        # calculate_ice_load_cdf=False is the default
        verbose=True
    )
    
    # Only returns filtered dataset and filtering statistics
    return filtered_ds, results

# What you get when calculate_ice_load_cdf=True:
"""
1. Organized folder structure:
   results/
   └── figures/
       └── spatial_gradient/
           └── filtered/
               └── [filter_description]/
                   ├── spatial_gradient/
                   │   └── ice_load_per_cell_cdf/
                   │       ├── ice_load_cdf_curves_all_cells_[parameters].png
                   │       ├── ice_load_cdf_summary_[parameters].png
                   │       ├── ice_load_cdf_spatial_gradients_absolute_[parameters].png
                   │       └── ice_load_cdf_spatial_gradients_dimensionless_[parameters].png
                   └── filter_documentation.txt

2. Filter documentation file contains:
   - Applied filters and their values
   - Filtering statistics (timesteps removed, percentages)
   - Variable statistics (ranges, means)
   - Ice load analysis parameters
   - Output file descriptions

3. Enhanced results dictionary with:
   - Original filtering results
   - Ice load data
   - CDF analysis results
   - Filter directory path
"""

print("Enhanced filter_dataset_by_thresholds function:")
print("✓ Meteorological filtering")
print("✓ Optional automatic ice load calculation")
print("✓ Optional automatic CDF analysis")
print("✓ Organized folder structure based on filters")
print("✓ Comprehensive documentation")
print("✓ All results saved in filter-specific directories")