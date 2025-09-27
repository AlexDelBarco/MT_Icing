# IMPORTS
import os
import functions as fn
import pandas as pd

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

start_date = '2019-07-01T00:00:00'
end_date = '2020-06-30T23:30:00'
dates = pd.date_range(start_date, '2021-07-01T23:30:00', freq='YS-JUL')

fn.accreation_per_winter(dataset, start_date, end_date)

# ice load
print("Calculating ice load...")
ice_load_result = fn.calculate_ice_load(dataset, dates)
 
# ice_load_result = fn.calculate_ice_load_for_dataset(dataset, start_date=start_date, end_date=end_date, accre_var='ACCRE_CYL', ablat_var='ABLAT_CYL', method=5, max_load=5.0)