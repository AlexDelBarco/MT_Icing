# IMPORTS
import os
import functions as fn

# IMPORT DATA
# Import data and first look
data1 = "data/alexandre.nc"
dataset = fn.load_netcdf_data(data1)

# EXPLORE DATASET
# Count spatial points
#fn.count_spatial_points(dataset)

# explore the variables
#fn.explore_variables(dataset)

# explore one variable in detail
#fn.explore_variable_detail(dataset, 'ACCRE_CYL')

# ice load

fn.calculate_ice_load_for_dataset(
    dataset,
    accre_var='ACCRE_CYL',
    ablat_var='ABLAT_CYL',
    method=3,
    max_load=5.0
)