# IMPORTS
import os
import functions as fn
import pandas as pd
import numpy as np

# IMPORT DATA
# Import data and first look
data1 = "data/alexandre.nc"
dataset = fn.load_netcdf_data(data1)

# EXPLORE DATASET
# explore the variables
fn.explore_variables(dataset)

# explore one variable in detail in a chosen period
fn.explore_variable_detail(dataset, 'ACCRE_CYL')

# Accreation for winter and time period + plot

start_date = '1899-07-01T00:00:00'
end_date = '2021-06-30T23:30:00'
dates = pd.date_range(start_date, '2021-07-01T23:30:00', freq='YS-JUL')

fn.accreation_per_winter(dataset, start_date, end_date)

# ice load calculation + plots
print("Calculating ice load...")
ice_load_data = fn.calculate_ice_load(dataset, dates)

# Spatial gradients analysis
ice_load_clean = ice_load_data.where(~np.isnan(ice_load_data), drop=True)
max_ice_load = ice_load_clean.max(dim='time')
spatial_stats = fn.create_spatial_gradient_plots(ice_load_clean, max_ice_load)

# Temporal gradients analysis
temporal_stats = fn.create_temporal_gradient_plots(ice_load_clean)

# Print gradient statistics
print(f"\n=== Gradient Analysis Summary ===")
print(f"Max spatial gradient magnitude (from max ice load): {spatial_stats['max_gradient_magnitude']:.3f}")
print(f"Mean spatial gradient magnitude (from max ice load): {spatial_stats['mean_gradient_magnitude']:.3f}")
print(f"Max mean spatial gradient magnitude (over time): {spatial_stats['max_mean_gradient_magnitude']:.3f}")
print(f"Mean spatial gradient magnitude (over time): {spatial_stats['mean_mean_gradient_magnitude']:.3f}")
print(f"Max temporal gradient: {temporal_stats['max_temporal_gradient']:.3f} kg/m per 30min")
print(f"Min temporal gradient: {temporal_stats['min_temporal_gradient']:.3f} kg/m per 30min")
