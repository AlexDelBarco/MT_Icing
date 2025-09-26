import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# Figures directory - relative path from scripts/ to figures/
figures_dir = "figures"


def load_netcdf_data(file_path):
    """Load and explore NetCDF data using xarray"""
    try:
        # Open the NetCDF file
        ds = xr.open_dataset(file_path)
        
        print("=== NetCDF File Information ===")
        print(f"File: {file_path}")
        print(f"Dimensions: {dict(ds.dims)}")
        print(f"Data variables: {list(ds.data_vars)}")
        print(f"Coordinates: {list(ds.coords)}")
        
        return ds
    
    except Exception as e:
        print(f"Error loading NetCDF file: {e}")
        return None
    
def explore_variables(ds):
    """Explore variables in the dataset"""
    if ds is None:
        return
    
    print("\n=== Variable Details ===")
    for var_name in ds.data_vars:
        var = ds[var_name]
        print(f"\nVariable: {var_name}")
        print(f"  Shape: {var.shape}")
        print(f"  Dimensions: {var.dims}")
        print(f"  Data type: {var.dtype}")
        if hasattr(var, 'attrs') and var.attrs:
            print(f"  Attributes: {var.attrs}")

def explore_variable_detail(ds, var_name):
    """Explore a specific variable in detail"""
    if ds is None or var_name not in ds:
        print(f"Variable '{var_name}' not found in dataset.")
        return
    
    var = ds[var_name]
    print(f"\n=== Detailed Exploration of Variable: {var_name} ===")
    print(var)
        
    # Show time range
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Show spatial coordinates
    print(f"Latitude range: {ds.XLAT.values.min():.3f} to {ds.XLAT.values.max():.3f}")
    print(f"Longitude range: {ds.XLON.values.min():.3f} to {ds.XLON.values.max():.3f}")
    
    # Convert to pandas DataFrame for easier exploration if 1D or 2D
    if len(var.dims) <= 2:
        df = var.to_dataframe().reset_index()
        print("\nData as DataFrame:")
        print(df.head())
    else:
        print("Variable has more than 2 dimensions; skipping DataFrame conversion.")


def ice_load(accre,ablat,method):
    if method == 1:

        load = accre.cumsum().load()
        load.loc[{'time':load.where(load>5,drop=True).time}] = 5
        load.loc[{'time':load.where(accre==0,drop=True).time}] = load.where(accre==0,drop=True) + ablat.where(accre==0,drop=True)
        load.loc[{'time':load.where(load<0,drop=True).time}] = 0

    if method ==2:
        min_mass = 0.01
        run_mass = 0
        load = accre.copy() + np.nan
        load.rename('ice_load')
        for houroi in accre.time:
            new_mass = accre.sel(time=houroi) + ablat.sel(time=houroi)
            run_mass = xr.where(run_mass + new_mass>0,run_mass + new_mass,0)
            run_mass = xr.where(run_mass < min_mass, 0, run_mass)
            load.loc[{'time':houroi}] = run_mass

    if method == 3:
        load = accre.copy() * 0 
        for ihouroi,houroi in enumerate(accre.time):
            print (ihouroi,houroi.values)
            if houroi == accre.time.isel(time=0):
                acm = load.sel(time=houroi)
            else:
                acm = load.isel(time=ihouroi-1).copy()
            # Add ice when there is new accretion and remove when there is ablation
            acm += xr.where(accre.sel(time=houroi) > 0.0005, accre.sel(time=houroi), xr.where(ablat.sel(time=houroi) < 0., ablat.sel(time=houroi), 0.))
            # Ensure that ice never goes negative
            load.loc[{'time':houroi}] =  xr.where(acm < 0., 0., acm)
            # Limit to 5kg of ice / m
            load.loc[{'time':houroi}] = xr.where(acm > 5., 5., load.loc[{'time':houroi}])
    if method == 4:
        loadnp = np.zeros_like(accre.values)
        acm = np.zeros_like(accre.isel(time=0))
        
        for i in range(1, len(accre.time)):
            print(i,len(accre.time))
            # Calculate change in ice load
            #delta = xr.where(accre.isel(time=i) > 0.0005, accre.isel(time=i), 0) + xr.where(ablat.isel(time=i) < 0, ablat.isel(time=i), 0)
            delta = xr.where(accre.isel(time=i) > 0.0005, accre.isel(time=i), xr.where(ablat.isel(time=i) < 0, ablat.isel(time=i), 0))
            
            # Update ice load
            acm += delta
            acm = xr.where(acm < 0, 0, acm)
            loadnp[i,:,:] = acm

        load = xr.zeros_like(accre)
        load.data = loadnp
    if method == 5:
        loadnp = np.zeros_like(accre.values)
        acm = np.zeros_like(accre.isel(time=0))
        
        for i in range(1, len(accre.time)):
            print(i,len(accre.time))
            # Calculate change in ice load
            #delta = xr.where(accre.isel(time=i) > 0.0005, accre.isel(time=i), 0) + xr.where(ablat.isel(time=i) < 0, ablat.isel(time=i), 0)
            delta = xr.where(accre.isel(time=i) > 0.0005, accre.isel(time=i), xr.where(ablat.isel(time=i) < 0, ablat.isel(time=i), 0))
            
            # Update ice load
            acm += delta
            acm = xr.where(acm < 0, 0, acm)
            acm = xr.where(acm > 5, 5, acm)
            loadnp[i,:,:] = acm

        load = xr.zeros_like(accre)
        load.data = loadnp
    #
    # rename data array
    load = load.rename('ice_load')
    #
    return load


def accreation_per_winter(ds, start_date, end_date):
    # Check if dataset is None
    if ds is None:
        print("Dataset is None")
        return None

    # Subset dataset to the specified date range
    ds1 = ds.sel(time=slice(start_date,end_date))

    # Define Winters
    dates = pd.date_range(start_date, end_date,freq='YS-JUL')

    # Add winter number to dataset
    df = ds1['time'].to_pandas()
    winter_numbers = np.full(len(df), np.nan)  # Initialize with NaN
    
    for iwinter,winterstartdate in enumerate(dates[:-1]):
        winterenddate = dates[iwinter+1]-pd.to_timedelta('30min')
        print(iwinter,winterstartdate,winterenddate)
        
        # Find indices for this winter period
        mask = (df >= winterstartdate) & (df <= winterenddate)
        winter_numbers[mask] = iwinter
    
    ds1 = ds1.assign_coords(winterno=('time', winter_numbers))

    # Plot accretion sum - one plot per winter

    # Only group by valid (non-NaN) winter numbers
    valid_winters = ds1.where(~np.isnan(ds1.winterno), drop=True)
    
    if len(valid_winters.time) > 0:
        plot_data = valid_winters.ACCRE_CYL.isel(height=0).groupby('winterno').sum(dim='time')
                
        # Create one plot for each winter
        for i, winter_idx in enumerate(plot_data.winterno.values):
            plt.figure(figsize=(10, 6))
            plot_data.isel(winterno=i).plot()
            
            winter_start = dates[int(winter_idx)] if int(winter_idx) < len(dates)-1 else "N/A"
            winter_end = dates[int(winter_idx)+1] - pd.to_timedelta('30min') if int(winter_idx)+1 < len(dates) else "N/A"
            plt.title(f'Ice Accretion Sum for Winter starting on: {winter_start} and ending on {winter_end}')
            plt.xlabel('West-East')
            plt.ylabel('South-North')
            plt.tight_layout()
            
            # Save figure
            filename = f"{figures_dir}/ice_accretion_winter_{int(winter_idx)}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()  # Close figure to free memory
            
        print(f"All plots saved to {figures_dir}/ directory")
    else:
        print("No valid winter data available for plotting")


def calculate_ice_load_for_dataset(ds, start_date, end_date, accre_var='ACCRE_CYL', ablat_var='ABLAT_CYL', method=3, max_load=5.0):
    """
    Calculate ice load for the dataset using available variables
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing ice accretion and ablation data
    accre_var : str
        Name of the accretion variable in the dataset (default: 'ACCRE_CYL')
    ablat_var : str
        Name of the ablation variable in the dataset (default: 'ABLAT_CYL')
    method : int
        Calculation method (1-5, default: 3)
    max_load : float
        Maximum ice load limit in kg/m (default: 5.0)
    
    Returns:
    --------
    xarray.DataArray
        Ice load data with same dimensions as input
    """
    # Check if dataset is None
    if ds is None:
        print("Dataset is None")
        return None

    # Subset dataset to the specified date range
    ds1 = ds.sel(time=slice(start_date,end_date))

    # Define Winters
    dates = pd.date_range(start_date, end_date,freq='YS-JUL')

    # Add winter number to dataset
    df = ds1['time'].to_pandas()
    for iwinter,winterstartdate in enumerate(dates[:-1]):
        winterenddate = dates[iwinter+1]-pd.to_timedelta('30min')
        print(iwinter,winterstartdate,winterenddate)
        datesperwinter = pd.date_range(winterstartdate,winterenddate,freq='30min')
        df.loc[datesperwinter]=iwinter
    ds1 = ds1.assign_coords(winterno=('time',df.values))

    # Check if required variables exist
    if accre_var not in ds1.data_vars:
        print(f"Accretion variable '{accre_var}' not found in dataset.")
        print(f"Available variables: {list(ds1.data_vars)}")
        return None
    
    if ablat_var not in ds1.data_vars:
        print(f"Ablation variable '{ablat_var}' not found in dataset.")
        print(f"Available variables: {list(ds1.data_vars)}")
        return None
    
    print(f"\n=== Calculating Ice Load ===")
    print(f"Using accretion variable: {accre_var}")
    print(f"Using ablation variable: {ablat_var}")
    print(f"Method: {method}")
    print(f"Maximum load limit: {max_load} kg/m")
    
    # Extract the variables
    accre = ds1[accre_var]
    ablat = ds1[ablat_var]
    
    print(f"Accretion data shape: {accre.shape}")
    print(f"Ablation data shape: {ablat.shape}")
    print(f"Time steps: {len(accre.time)}")
    
    # Call the original ice_load function
    ice_load_result = ice_load(accre, ablat, method)
    
    # Add metadata
    ice_load_result.attrs['description'] = f'Ice load calculated using method {method}'
    ice_load_result.attrs['units'] = 'kg/m'
    ice_load_result.attrs['max_load_limit'] = max_load
    ice_load_result.attrs['accretion_source'] = accre_var
    ice_load_result.attrs['ablation_source'] = ablat_var
    
    print(f"Ice load calculation completed!")
    print(f"Result shape: {ice_load_result.shape}")
    print(f"Max ice load in dataset: {ice_load_result.max().values:.3f} kg/m")
    print(f"Min ice load in dataset: {ice_load_result.min().values:.3f} kg/m")

   
    return ice_load_result
