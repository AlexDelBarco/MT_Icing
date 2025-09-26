# IMPORTS
import os
import functions as fn

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

start_date = '2016-11-01T00:00:00'
end_date = '2020-11-03T23:30:00'

fn.accreation_per_winter(dataset, start_date, end_date)

# ice load
 
# ice_load_result = fn.calculate_ice_load_for_dataset(dataset, start_date=start_date, end_date=end_date, accre_var='ACCRE_CYL', ablat_var='ABLAT_CYL', method=5, max_load=5.0)