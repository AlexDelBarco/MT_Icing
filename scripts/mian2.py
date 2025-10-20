# IMPORTS
import os
import functions as fn
import pandas as pd
import numpy as np
import xarray as xr

# IMPORT DATA
# Import data and first look
data1 = "data/newa_wrf_for_jana_mstudent_extended.nc"
dataset = fn.load_netcdf_data(data1)

# EXPLORE DATASET
# explore the variables
#fn.explore_variables(dataset)

# explore one variable in detail in a chosen period
fn.explore_variable_detail(dataset, 'ACCRE_CYL')

# Accreation for winter and time period + plot
start_date = '1989-01-01T00:00:00.000000000'
end_date = '2022-12-31T23:30:00.000000000'
dates = pd.date_range(start_date, end_date, freq='YS-JUL')
#fn.accreation_per_winter(dataset, start_date, end_date)

# Load/calculate ice load data
#print("Loading existing ice load data...")
#ice_load_data = xr.open_dataarray(f"results/iceload_{start_date[:10]}_to_{end_date[:10]}.nc")
#print(f"Loaded ice load data with shape: {ice_load_data.shape}")
#print("Calculating ice load...")
#ice_load_data = fn.calculate_ice_load(dataset, dates, 5)