# IMPORTS
import os
import functions as fn
import pandas as pd
import numpy as np
import xarray as xr

pdfc_file= "data/newa_wrf_for_jana_mstudent_extended_PSFC_SEAICE_SWDDNI.nc"

pp = fn.load_netcdf_data(pdfc_file)

pp1 = fn.explore_variables(pp)