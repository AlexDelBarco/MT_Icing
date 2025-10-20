import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# Results directories
figures_dir = "results/figures"
results_dir = "results"

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
        print(f"First timestamp: {ds.time.values[0]}")
        print(f"Last timestamp: {ds.time.values[-1]}")
        print(f"Total time steps: {len(ds.time)}")
        print(f"Time frequency: {pd.to_datetime(ds.time.values[1]) - pd.to_datetime(ds.time.values[0])}")
        print(f"Height levels in dataset: {ds.height.values}")
        print(f"South north values in dataset: {ds.south_north.values}")
        print(f"West East values in dataset: {ds.west_east.values}")

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

    # Show dimensions and shape
    print(f"dimensions: {var.dims}")
    print(f"shape: {var.shape}")

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
    # Check if input data is empty
    if len(accre.time) == 0:
        print(f"Warning: No data available for this time period. Skipping ice load calculation.")
        return None
    
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
            #print(i,len(accre.time))
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
            #print(i,len(accre.time))
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


def accreation_per_winter(ds, start_date, end_date, height_level=0):
    """
    Analyze ice accretion per winter season at a specific height level
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing ice accretion data
    start_date : str
        Start date for analysis
    end_date : str
        End date for analysis
    height_level : int, optional
        Height level index to analyze (default: 0)
        Use 0, 1, or 2 for the three available height levels
    
    Returns:
    --------
    None
        Creates and saves plots for each winter season at specified height
    """
    # Check if dataset is None
    if ds is None:
        print("Dataset is None")
        return None

    # Check if height level is valid and get height information
    if 'height' in ds.dims:
        max_height_idx = ds.dims['height'] - 1
        if height_level > max_height_idx:
            print(f"Warning: height_level {height_level} is out of range. Maximum available: {max_height_idx}")
            print(f"Available height values: {ds.height.values}")
            height_level = 0
            print(f"Using height_level {height_level} instead")
        
        height_value = ds.height.values[height_level]
        height_units = ds.height.attrs.get('units', 'units')
        print(f"Analyzing ice accretion at height level {height_level} ({height_value} {height_units})")
        height_label = f"h{int(height_value)}{height_units.replace(' ', '')}"
    else:
        print("No height dimension found in dataset")
        height_level = 0
        height_value = "unknown"
        height_units = ""
        height_label = "h0"

    # Subset dataset to the specified date range
    ds1 = ds.sel(time=slice(start_date,end_date))

    # Define Winters - extend range to capture proper winter boundaries
    winter_start = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    winter_end = pd.to_datetime(end_date) + pd.DateOffset(years=1)
    dates = pd.date_range(winter_start, winter_end, freq='YS-JUL')
    
    # Create figures directory and ice_accretion subfolder if they don't exist
    ice_accretion_dir = os.path.join(figures_dir, "ice_accretion")
    os.makedirs(ice_accretion_dir, exist_ok=True)

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

    # Check if we have any valid winter numbers
    valid_winter_mask = ~np.isnan(winter_numbers)
    
    if np.any(valid_winter_mask):
        # Filter dataset to only include times with valid winter numbers
        valid_times = ds1.time.values[valid_winter_mask]
        ds1_filtered = ds1.sel(time=valid_times)
        
        # Use the specified height level instead of hardcoded height=0
        plot_data = ds1_filtered.ACCRE_CYL.isel(height=height_level).groupby('winterno').sum(dim='time')
                
        # Create one plot for each winter
        for i, winter_idx in enumerate(plot_data.winterno.values):
            plt.figure(figsize=(10, 6))
            plot_data.isel(winterno=i).plot()
            
            winter_start = dates[int(winter_idx)] if int(winter_idx) < len(dates)-1 else "N/A"
            winter_end = dates[int(winter_idx)+1] - pd.to_timedelta('30min') if int(winter_idx)+1 < len(dates) else "N/A"
            
            # Include height information in the title
            plt.title(f'Ice Accretion Sum for Winter starting on: {winter_start} and ending on {winter_end}\nHeight: {height_value} {height_units}')
            plt.xlabel('West-East')
            plt.ylabel('South-North')
            plt.tight_layout()
            
            # Include height information in the filename
            filename = os.path.join(ice_accretion_dir, f"ice_accretion_winter_{int(winter_idx)}_{height_label}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()  # Close figure to free memory
            
        print(f"All plots saved to {ice_accretion_dir}/ directory")
    else:
        print("No valid winter data available for plotting")


def create_spatial_gradient_plots(ice_load_data):
    """
    Create spatial gradient analysis plots for ice load data
    
    Parameters:
    -----------
    ice_load_data : xarray.DataArray
        Raw ice load data (may contain NaN values)
    
    Returns:
    --------
    dict
        Dictionary containing gradient statistics
    """
    print("Creating spatial gradient analysis...")
    
    # Create spatial_gradient subfolder if it doesn't exist
    spatial_gradient_dir = os.path.join(figures_dir, "spatial_gradient")
    os.makedirs(spatial_gradient_dir, exist_ok=True)
    
    # Clean the data by removing NaN values
    ice_load_clean = ice_load_data.where(~np.isnan(ice_load_data), drop=True)
    max_ice_load = ice_load_clean.max(dim='time')
    
    # Calculate spatial gradients using numpy gradient
    max_ice_2d = max_ice_load.values
    
    # Calculate gradients in both directions
    grad_y, grad_x = np.gradient(max_ice_2d)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original maximum ice load
    im1 = axes[0,0].imshow(max_ice_2d, cmap='Blues', aspect='auto')
    axes[0,0].set_title('Maximum Ice Load')
    axes[0,0].set_xlabel('West-East')
    axes[0,0].set_ylabel('South-North')
    plt.colorbar(im1, ax=axes[0,0], label='Ice Load (kg/m)')
    
    # X-direction gradient (West-East)
    im2 = axes[0,1].imshow(grad_x, cmap='RdBu_r', aspect='auto')
    axes[0,1].set_title('Spatial Gradient (West-East)')
    axes[0,1].set_xlabel('West-East')
    axes[0,1].set_ylabel('South-North')
    plt.colorbar(im2, ax=axes[0,1], label='Gradient (kg/m per grid)')
    
    # Y-direction gradient (South-North)
    im3 = axes[1,0].imshow(grad_y, cmap='RdBu_r', aspect='auto')
    axes[1,0].set_title('Spatial Gradient (South-North)')
    axes[1,0].set_xlabel('West-East')
    axes[1,0].set_ylabel('South-North')
    plt.colorbar(im3, ax=axes[1,0], label='Gradient (kg/m per grid)')
    
    # Gradient magnitude
    im4 = axes[1,1].imshow(gradient_magnitude, cmap='plasma', aspect='auto')
    axes[1,1].set_title('Gradient Magnitude')
    axes[1,1].set_xlabel('West-East')
    axes[1,1].set_ylabel('South-North')
    plt.colorbar(im4, ax=axes[1,1], label='Gradient Magnitude')
    
    plt.tight_layout()
    
    spatial_grad_filename = os.path.join(spatial_gradient_dir, 'ice_load_maximum_spatial_gradients.png')
    plt.savefig(spatial_grad_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {spatial_grad_filename}")
    plt.close()
    
    # Mean spatial gradient analysis over all time periods
    print("Creating mean temporal spatial gradient analysis...")
    
    # Calculate spatial gradients for each time step
    print("Computing gradients for all time steps...")
    
    # Initialize arrays to store gradients for all time steps
    all_grad_x = np.zeros_like(ice_load_clean.values)
    all_grad_y = np.zeros_like(ice_load_clean.values)
    
    # Calculate gradients for each time step
    for i, time_step in enumerate(ice_load_clean.time.values):
        ice_2d_t = ice_load_clean.sel(time=time_step).values
        
        # Skip if all NaN
        if not np.all(np.isnan(ice_2d_t)):
            grad_y_t, grad_x_t = np.gradient(ice_2d_t)
            all_grad_x[i, :, :] = grad_x_t
            all_grad_y[i, :, :] = grad_y_t
        else:
            all_grad_x[i, :, :] = np.nan
            all_grad_y[i, :, :] = np.nan
    
    # Compute mean gradients over time (ignoring NaN values)
    mean_grad_x = np.nanmean(all_grad_x, axis=0)
    mean_grad_y = np.nanmean(all_grad_y, axis=0)
    mean_gradient_magnitude = np.sqrt(mean_grad_x**2 + mean_grad_y**2)
    
    # Also compute the mean ice load for comparison
    mean_ice_load = ice_load_clean.mean(dim='time').values
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Mean ice load over time
    im1 = axes[0,0].imshow(mean_ice_load, cmap='Blues', aspect='auto')
    axes[0,0].set_title('Mean Ice Load Over Time')
    axes[0,0].set_xlabel('West-East')
    axes[0,0].set_ylabel('South-North')
    plt.colorbar(im1, ax=axes[0,0], label='Mean Ice Load (kg/m)')
    
    # Mean X-direction gradient (West-East)
    im2 = axes[0,1].imshow(mean_grad_x, cmap='RdBu_r', aspect='auto')
    axes[0,1].set_title('Mean Spatial Gradient (West-East)')
    axes[0,1].set_xlabel('West-East')
    axes[0,1].set_ylabel('South-North')
    plt.colorbar(im2, ax=axes[0,1], label='Mean Gradient (kg/m per grid)')
    
    # Mean Y-direction gradient (South-North)
    im3 = axes[1,0].imshow(mean_grad_y, cmap='RdBu_r', aspect='auto')
    axes[1,0].set_title('Mean Spatial Gradient (South-North)')
    axes[1,0].set_xlabel('West-East')
    axes[1,0].set_ylabel('South-North')
    plt.colorbar(im3, ax=axes[1,0], label='Mean Gradient (kg/m per grid)')
    
    # Mean gradient magnitude
    im4 = axes[1,1].imshow(mean_gradient_magnitude, cmap='plasma', aspect='auto')
    axes[1,1].set_title('Mean Gradient Magnitude')
    axes[1,1].set_xlabel('West-East')
    axes[1,1].set_ylabel('South-North')
    plt.colorbar(im4, ax=axes[1,1], label='Mean Gradient Magnitude')
    
    plt.tight_layout()
    
    mean_spatial_grad_filename = os.path.join(spatial_gradient_dir, 'ice_load_mean_spatial_gradients.png')
    plt.savefig(mean_spatial_grad_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {mean_spatial_grad_filename}")
    plt.close()
    
    # Return statistics for use in summary
    return {
        'max_gradient_magnitude': gradient_magnitude.max(),
        'mean_gradient_magnitude': gradient_magnitude.mean(),
        'max_mean_gradient_magnitude': np.nanmax(mean_gradient_magnitude),
        'mean_mean_gradient_magnitude': np.nanmean(mean_gradient_magnitude)
    }


def create_temporal_gradient_plots(ice_load_data):
    """
    Create temporal gradient analysis plots for ice load data
    
    Parameters:
    -----------
    ice_load_data : xarray.DataArray
        Raw ice load data (may contain NaN values)
    
    Returns:
    --------
    dict
        Dictionary containing temporal gradient statistics
    """
    print("Creating temporal gradient analysis...")
    
    # Create temporal_gradient subfolder if it doesn't exist
    temporal_gradient_dir = os.path.join(figures_dir, "temporal_gradient")
    os.makedirs(temporal_gradient_dir, exist_ok=True)
    
    # Clean the data by removing NaN values
    ice_load_clean = ice_load_data.where(~np.isnan(ice_load_data), drop=True)
    
    # Calculate temporal gradient (rate of change over time)
    temporal_gradient = ice_load_clean.diff(dim='time')
    
    # Get statistics of temporal changes
    temp_grad_mean = temporal_gradient.mean(dim=['south_north', 'west_east'])
    temp_grad_std = temporal_gradient.std(dim=['south_north', 'west_east'])
    temp_grad_max = temporal_gradient.max(dim=['south_north', 'west_east'])
    temp_grad_min = temporal_gradient.min(dim=['south_north', 'west_east'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Mean temporal gradient over time
    axes[0,0].plot(temp_grad_mean.time, temp_grad_mean.values)
    axes[0,0].set_title('Mean Temporal Gradient (Rate of Change)')
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Mean Rate of Change (kg/m per 30min)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Standard deviation of temporal gradient
    axes[0,1].plot(temp_grad_std.time, temp_grad_std.values, color='orange')
    axes[0,1].set_title('Temporal Gradient Variability')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Std Dev of Rate of Change (kg/m per 30min)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Maximum and minimum temporal gradients
    axes[1,0].plot(temp_grad_max.time, temp_grad_max.values, label='Maximum', color='red')
    axes[1,0].plot(temp_grad_min.time, temp_grad_min.values, label='Minimum', color='blue')
    axes[1,0].set_title('Extreme Temporal Gradients')
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Rate of Change (kg/m per 30min)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Histogram of all temporal gradients
    all_temp_gradients = temporal_gradient.values.flatten()
    all_temp_gradients = all_temp_gradients[~np.isnan(all_temp_gradients)]
    
    axes[1,1].hist(all_temp_gradients, bins=50, alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Distribution of Temporal Gradients')
    axes[1,1].set_xlabel('Rate of Change (kg/m per 30min)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    temporal_grad_filename = os.path.join(temporal_gradient_dir, 'ice_load_temporal_gradients.png')
    plt.savefig(temporal_grad_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {temporal_grad_filename}")
    plt.close()
    
    # Return statistics for use in summary
    return {
        'max_temporal_gradient': all_temp_gradients.max(),
        'min_temporal_gradient': all_temp_gradients.min()
    }


def create_spatial_gradient_time_evolution_plots(ice_load_data):
    """
    Create spatial gradient analysis showing how gradients evolve over time
    
    Parameters:
    -----------
    ice_load_data : xarray.DataArray
        Raw ice load data (may contain NaN values)
    
    Returns:
    --------
    dict
        Dictionary containing gradient evolution statistics
    """
    print("Creating spatial gradient time evolution analysis...")
    
    # Create spatial_gradient subfolder if it doesn't exist
    spatial_gradient_dir = os.path.join(figures_dir, "spatial_gradient")
    os.makedirs(spatial_gradient_dir, exist_ok=True)
    
    # Clean the data by removing NaN values
    ice_load_clean = ice_load_data.where(~np.isnan(ice_load_data), drop=True)
    
    if len(ice_load_clean.time) == 0:
        print("No valid ice load data found!")
        return None
    
    print("Computing spatial gradients for all time steps...")
    
    # Initialize arrays to store gradient statistics over time
    time_values = ice_load_clean.time.values
    gradient_magnitude_mean = np.zeros(len(time_values))
    gradient_magnitude_max = np.zeros(len(time_values))
    gradient_magnitude_std = np.zeros(len(time_values))
    gradient_x_mean = np.zeros(len(time_values))
    gradient_y_mean = np.zeros(len(time_values))
    
    # Calculate spatial gradients for each time step
    for i, time_step in enumerate(time_values):
        ice_2d_t = ice_load_clean.sel(time=time_step).values
        
        # Skip if all NaN
        if not np.all(np.isnan(ice_2d_t)):
            grad_y_t, grad_x_t = np.gradient(ice_2d_t)
            gradient_magnitude_t = np.sqrt(grad_x_t**2 + grad_y_t**2)
            
            # Store statistics for this time step
            gradient_magnitude_mean[i] = np.nanmean(gradient_magnitude_t)
            gradient_magnitude_max[i] = np.nanmax(gradient_magnitude_t)
            gradient_magnitude_std[i] = np.nanstd(gradient_magnitude_t)
            gradient_x_mean[i] = np.nanmean(grad_x_t)
            gradient_y_mean[i] = np.nanmean(grad_y_t)
        else:
            gradient_magnitude_mean[i] = np.nan
            gradient_magnitude_max[i] = np.nan
            gradient_magnitude_std[i] = np.nan
            gradient_x_mean[i] = np.nan
            gradient_y_mean[i] = np.nan
    
    # Create the time evolution plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert time to pandas datetime for better plotting
    time_pd = pd.to_datetime(time_values)
    
    # Plot 1: Gradient magnitude statistics over time
    axes[0,0].plot(time_pd, gradient_magnitude_mean, label='Mean', color='blue', alpha=0.8)
    axes[0,0].plot(time_pd, gradient_magnitude_max, label='Maximum', color='red', alpha=0.8)
    axes[0,0].fill_between(time_pd, 
                          gradient_magnitude_mean - gradient_magnitude_std, 
                          gradient_magnitude_mean + gradient_magnitude_std, 
                          alpha=0.3, color='blue', label='±1 Std Dev')
    axes[0,0].set_title('Spatial Gradient Magnitude Evolution')
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Gradient Magnitude (kg/m per grid)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Directional gradient evolution (West-East)
    axes[0,1].plot(time_pd, gradient_x_mean, color='orange', alpha=0.8)
    axes[0,1].set_title('Mean West-East Gradient Evolution')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('West-East Gradient (kg/m per grid)')
    axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Directional gradient evolution (South-North)
    axes[1,0].plot(time_pd, gradient_y_mean, color='green', alpha=0.8)
    axes[1,0].set_title('Mean South-North Gradient Evolution')
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('South-North Gradient (kg/m per grid)')
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Gradient variability over time
    axes[1,1].plot(time_pd, gradient_magnitude_std, color='purple', alpha=0.8)
    axes[1,1].set_title('Spatial Gradient Variability Over Time')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Gradient Std Dev (kg/m per grid)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    time_evolution_filename = os.path.join(spatial_gradient_dir, 'spatial_gradient_time_evolution.png')
    plt.savefig(time_evolution_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {time_evolution_filename}")
    plt.close()
    
    # Create a second figure showing seasonal patterns if multiple years of data
    print("Creating seasonal gradient analysis...")
    
    # Extract hour of day and month for seasonal analysis
    hours = pd.to_datetime(time_values).hour
    months = pd.to_datetime(time_values).month
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Gradient magnitude by hour of day
    hourly_grad_mean = []
    hourly_grad_std = []
    for hour in range(24):
        hour_mask = hours == hour
        if np.any(hour_mask):
            hourly_grad_mean.append(np.nanmean(gradient_magnitude_mean[hour_mask]))
            hourly_grad_std.append(np.nanstd(gradient_magnitude_mean[hour_mask]))
        else:
            hourly_grad_mean.append(np.nan)
            hourly_grad_std.append(np.nan)
    
    axes[0,0].plot(range(24), hourly_grad_mean, marker='o', color='blue')
    axes[0,0].fill_between(range(24), 
                          np.array(hourly_grad_mean) - np.array(hourly_grad_std),
                          np.array(hourly_grad_mean) + np.array(hourly_grad_std),
                          alpha=0.3, color='blue')
    axes[0,0].set_title('Average Gradient Magnitude by Hour of Day')
    axes[0,0].set_xlabel('Hour of Day')
    axes[0,0].set_ylabel('Mean Gradient Magnitude')
    axes[0,0].set_xticks(range(0, 24, 3))
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Gradient magnitude by month
    monthly_grad_mean = []
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month in range(1, 13):
        month_mask = months == month
        if np.any(month_mask):
            monthly_grad_mean.append(np.nanmean(gradient_magnitude_mean[month_mask]))
        else:
            monthly_grad_mean.append(np.nan)
    
    axes[0,1].bar(month_names, monthly_grad_mean, color='orange', alpha=0.7)
    axes[0,1].set_title('Average Gradient Magnitude by Month')
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('Mean Gradient Magnitude')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: West-East gradient by hour
    hourly_grad_x = []
    for hour in range(24):
        hour_mask = hours == hour
        if np.any(hour_mask):
            hourly_grad_x.append(np.nanmean(gradient_x_mean[hour_mask]))
        else:
            hourly_grad_x.append(np.nan)
    
    axes[1,0].plot(range(24), hourly_grad_x, marker='s', color='red')
    axes[1,0].set_title('Average West-East Gradient by Hour of Day')
    axes[1,0].set_xlabel('Hour of Day')
    axes[1,0].set_ylabel('Mean West-East Gradient')
    axes[1,0].set_xticks(range(0, 24, 3))
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: South-North gradient by hour
    hourly_grad_y = []
    for hour in range(24):
        hour_mask = hours == hour
        if np.any(hour_mask):
            hourly_grad_y.append(np.nanmean(gradient_y_mean[hour_mask]))
        else:
            hourly_grad_y.append(np.nan)
    
    axes[1,1].plot(range(24), hourly_grad_y, marker='^', color='green')
    axes[1,1].set_title('Average South-North Gradient by Hour of Day')
    axes[1,1].set_xlabel('Hour of Day')
    axes[1,1].set_ylabel('Mean South-North Gradient')
    axes[1,1].set_xticks(range(0, 24, 3))
    axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the seasonal figure
    seasonal_filename = os.path.join(spatial_gradient_dir, 'spatial_gradient_seasonal_patterns.png')
    plt.savefig(seasonal_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {seasonal_filename}")
    plt.close()
    
    # Return statistics
    return {
        'max_gradient_evolution': np.nanmax(gradient_magnitude_mean),
        'mean_gradient_evolution': np.nanmean(gradient_magnitude_mean),
        'gradient_temporal_variability': np.nanstd(gradient_magnitude_mean),
        'peak_hour_gradient': int(np.nanargmax(hourly_grad_mean)),
        'min_hour_gradient': int(np.nanargmin(hourly_grad_mean))
    }



    """
    Create geographical maps of ice load data using latitude and longitude coordinates
    
    Parameters:
    -----------
    ice_load_data : xarray.DataArray
        Ice load data to plot
    dataset : xarray.Dataset
        Original dataset containing XLAT and XLON coordinates
    
    Returns:
    --------
    dict
        Dictionary containing geographical plotting statistics
    """
    print("Creating geographical ice load maps...")
    
    # Create geographical_maps subfolder if it doesn't exist
    geo_maps_dir = os.path.join(figures_dir, "geographical_maps")
    os.makedirs(geo_maps_dir, exist_ok=True)
    
    # Clean the data by removing NaN values
    ice_load_clean = ice_load_data.where(~np.isnan(ice_load_data), drop=True)
    
    if len(ice_load_clean.time) == 0:
        print("No valid ice load data found!")
        return None
    
    # Get geographical coordinates
    if 'XLAT' in dataset.coords and 'XLON' in dataset.coords:
        lats = dataset.XLAT.values
        lons = dataset.XLON.values
    elif 'XLAT' in dataset.data_vars and 'XLON' in dataset.data_vars:
        lats = dataset.XLAT.values
        lons = dataset.XLON.values
    else:
        print("Error: Could not find XLAT and XLON coordinates in dataset")
        return None
    
    # Calculate statistics for mapping
    max_ice_load = ice_load_clean.max(dim='time').values
    mean_ice_load = ice_load_clean.mean(dim='time').values
    
    print(f"Coordinate ranges - Lat: {lats.min():.3f} to {lats.max():.3f}, Lon: {lons.min():.3f} to {lons.max():.3f}")
    
    # Try to import cartopy for proper geographical projections
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        cartopy_available = True
        print("Using cartopy for geographical projections")
    except ImportError:
        cartopy_available = False
        print("Cartopy not available, using basic matplotlib plotting")
    
    # Create geographical plots
    if cartopy_available:
        # Create maps with cartopy (proper geographical projection)
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Maximum ice load with geographical features
        ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
        im1 = ax1.pcolormesh(lons, lats, max_ice_load, 
                            cmap='Blues', transform=ccrs.PlateCarree())
        ax1.add_feature(cfeature.COASTLINE, alpha=0.7)
        ax1.add_feature(cfeature.BORDERS, alpha=0.5)
        ax1.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax1.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        ax1.gridlines(draw_labels=True, alpha=0.5)
        ax1.set_title('Maximum Ice Load (Geographical)')
        plt.colorbar(im1, ax=ax1, label='Ice Load (kg/m)', shrink=0.8)
        
        # Plot 2: Mean ice load with geographical features
        ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
        im2 = ax2.pcolormesh(lons, lats, mean_ice_load, 
                            cmap='Blues', transform=ccrs.PlateCarree())
        ax2.add_feature(cfeature.COASTLINE, alpha=0.7)
        ax2.add_feature(cfeature.BORDERS, alpha=0.5)
        ax2.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax2.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        ax2.gridlines(draw_labels=True, alpha=0.5)
        ax2.set_title('Mean Ice Load (Geographical)')
        plt.colorbar(im2, ax=ax2, label='Ice Load (kg/m)', shrink=0.8)
        
        # Plot 3: Grid overlay showing actual grid points
        ax3 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
        ax3.scatter(lons.flatten(), lats.flatten(), s=1, c='red', alpha=0.6, 
                   transform=ccrs.PlateCarree())
        ax3.add_feature(cfeature.COASTLINE, alpha=0.7)
        ax3.add_feature(cfeature.BORDERS, alpha=0.5)
        ax3.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax3.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        ax3.gridlines(draw_labels=True, alpha=0.5)
        ax3.set_title('Model Grid Points')
        
        # Plot 4: Ice load with contours
        ax4 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
        im4 = ax4.pcolormesh(lons, lats, max_ice_load, 
                            cmap='Blues', transform=ccrs.PlateCarree(), alpha=0.8)
        contours = ax4.contour(lons, lats, max_ice_load, levels=10, 
                              colors='black', alpha=0.6, transform=ccrs.PlateCarree())
        ax4.clabel(contours, inline=True, fontsize=8)
        ax4.add_feature(cfeature.COASTLINE, alpha=0.7)
        ax4.add_feature(cfeature.BORDERS, alpha=0.5)
        ax4.gridlines(draw_labels=True, alpha=0.5)
        ax4.set_title('Maximum Ice Load with Contours')
        plt.colorbar(im4, ax=ax4, label='Ice Load (kg/m)', shrink=0.8)
        
    else:
        # Create basic maps without cartopy
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Maximum ice load
        im1 = axes[0,0].pcolormesh(lons, lats, max_ice_load, cmap='Blues')
        axes[0,0].set_title('Maximum Ice Load')
        axes[0,0].set_xlabel('Longitude')
        axes[0,0].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[0,0], label='Ice Load (kg/m)')
        
        # Plot 2: Mean ice load
        im2 = axes[0,1].pcolormesh(lons, lats, mean_ice_load, cmap='Blues')
        axes[0,1].set_title('Mean Ice Load')
        axes[0,1].set_xlabel('Longitude')
        axes[0,1].set_ylabel('Latitude')
        plt.colorbar(im2, ax=axes[0,1], label='Ice Load (kg/m)')
        
        # Plot 3: Grid points
        axes[1,0].scatter(lons.flatten(), lats.flatten(), s=1, c='red', alpha=0.6)
        axes[1,0].set_title('Model Grid Points')
        axes[1,0].set_xlabel('Longitude')
        axes[1,0].set_ylabel('Latitude')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Ice load with contours
        im4 = axes[1,1].pcolormesh(lons, lats, max_ice_load, cmap='Blues', alpha=0.8)
        contours = axes[1,1].contour(lons, lats, max_ice_load, levels=10, colors='black', alpha=0.6)
        axes[1,1].clabel(contours, inline=True, fontsize=8)
        axes[1,1].set_title('Maximum Ice Load with Contours')
        axes[1,1].set_xlabel('Longitude')
        axes[1,1].set_ylabel('Latitude')
        plt.colorbar(im4, ax=axes[1,1], label='Ice Load (kg/m)')
    
    plt.tight_layout()
    
    # Save the geographical map
    geo_filename = os.path.join(geo_maps_dir, 'ice_load_geographical_maps.png')
    plt.savefig(geo_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {geo_filename}")
    plt.close()
    
    # Create a detailed grid information plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Latitude grid
    im1 = axes[0,0].imshow(lats, cmap='viridis', aspect='auto')
    axes[0,0].set_title('Latitude Grid')
    axes[0,0].set_xlabel('West-East Grid Index')
    axes[0,0].set_ylabel('South-North Grid Index')
    plt.colorbar(im1, ax=axes[0,0], label='Latitude (degrees)')
    
    # Plot 2: Longitude grid
    im2 = axes[0,1].imshow(lons, cmap='plasma', aspect='auto')
    axes[0,1].set_title('Longitude Grid')
    axes[0,1].set_xlabel('West-East Grid Index')
    axes[0,1].set_ylabel('South-North Grid Index')
    plt.colorbar(im2, ax=axes[0,1], label='Longitude (degrees)')
    
    # Plot 3: Grid spacing in latitude
    lat_spacing = np.diff(lats, axis=0)
    if lat_spacing.size > 0:
        im3 = axes[1,0].imshow(lat_spacing, cmap='RdYlBu', aspect='auto')
        axes[1,0].set_title('Latitude Grid Spacing')
        axes[1,0].set_xlabel('West-East Grid Index')
        axes[1,0].set_ylabel('South-North Grid Index')
        plt.colorbar(im3, ax=axes[1,0], label='Lat Spacing (degrees)')
    
    # Plot 4: Grid spacing in longitude
    lon_spacing = np.diff(lons, axis=1)
    if lon_spacing.size > 0:
        im4 = axes[1,1].imshow(lon_spacing, cmap='RdYlBu', aspect='auto')
        axes[1,1].set_title('Longitude Grid Spacing')
        axes[1,1].set_xlabel('West-East Grid Index')
        axes[1,1].set_ylabel('South-North Grid Index')
        plt.colorbar(im4, ax=axes[1,1], label='Lon Spacing (degrees)')
    
    plt.tight_layout()
    
    # Save the grid information plot
    grid_info_filename = os.path.join(geo_maps_dir, 'grid_coordinate_information.png')
    plt.savefig(grid_info_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {grid_info_filename}")
    plt.close()
    
    # Return statistics
    return {
        'lat_range': [float(lats.min()), float(lats.max())],
        'lon_range': [float(lons.min()), float(lons.max())],
        'grid_shape': lats.shape,
        'cartopy_available': cartopy_available
    }


def save_ice_load_data(dsiceload, start_date, end_date, height_label="h0"):
    """
    Save calculated ice load data to disk with height information
    
    Parameters:
    -----------
    dsiceload : xarray.DataArray
        Calculated ice load data
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    height_label : str, optional
        Height label for filename (default: "h0")
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename based on date range and height
    start_str = pd.to_datetime(start_date).strftime('%Y%m%d')
    end_str = pd.to_datetime(end_date).strftime('%Y%m%d')
    filename = f"iceload_{start_str}_to_{end_str}_{height_label}.nc"
    filepath = os.path.join(results_dir, filename)
    
    print(f"Saving ice load data to: {filepath}")
    try:
        dsiceload.to_netcdf(filepath)
        print(f"Successfully saved ice load data with shape: {dsiceload.shape}")
    except Exception as e:
        print(f"Error saving ice load data: {e}")


def calculate_ice_load(ds1, dates, method, height_level=0, create_figures=True):
    """
    Calculate ice load and create basic visualization figures at a specific height level
    
    Parameters:
    -----------
    ds1 : xarray.Dataset
        Dataset containing ice accretion and ablation data
    dates : pandas.DatetimeIndex
        Date range for winter seasons
    method : int
        Ice load calculation method (1-5)
    height_level : int, optional
        Height level index to analyze (default: 0)
        Use 0, 1, or 2 for the three available height levels
    create_figures : bool
        Whether to create basic visualization figures (default: True)
        Note: Gradient analysis plots are created separately using dedicated functions
    
    Returns:
    --------
    xarray.DataArray
        Calculated ice load data for all time periods at specified height level
    """
    
    # Check if height level is valid and get height information
    if 'height' in ds1.dims:
        max_height_idx = ds1.dims['height'] - 1
        if height_level > max_height_idx:
            print(f"Warning: height_level {height_level} is out of range. Maximum available: {max_height_idx}")
            print(f"Available height values: {ds1.height.values}")
            height_level = 0
            print(f"Using height_level {height_level} instead")
        
        height_value = ds1.height.values[height_level]
        height_units = ds1.height.attrs.get('units', 'units')
        print(f"Calculating ice load at height level {height_level} ({height_value} {height_units})")
        height_label = f"h{int(height_value)}{height_units.replace(' ', '')}"
    else:
        print("No height dimension found in dataset")
        height_level = 0
        height_value = "unknown"
        height_units = ""
        height_label = "h0"
    
    print(f"Calculating ice load at {height_value} {height_units}...")
    
    dsiceload = xr.zeros_like(ds1['ACCRE_CYL'].isel(height=height_level)) * np.nan
    for idate,date in enumerate(dates[:-1]):
        print(f"Processing winter {idate+1}/{len(dates)-1}: {date} to {dates[idate+1]-pd.to_timedelta('30min')}")
        
        # Get data for this winter period at specified height level
        winter_accre = ds1['ACCRE_CYL'].isel(height=height_level).sel(time=slice(date,dates[idate+1]-pd.to_timedelta('30min'))).load()
        winter_ablat = ds1['ABLAT_CYL'].isel(height=height_level).sel(time=slice(date,dates[idate+1]-pd.to_timedelta('30min'))).load()
        
        # Check if there's data for this winter
        if len(winter_accre.time) == 0:
            print(f"  No data available for winter {idate+1}. Skipping...")
            continue
            
        load = ice_load(winter_accre, winter_ablat, method)
        
        # Only assign if load calculation was successful
        if load is not None:
            dsiceload.loc[{'time':load.time}] = load
            print(f"Winter {idate+1} completed")
        else:
            print(f"Winter {idate+1} skipped due to insufficient data")
    
    print("\nCreating representative figures of ice load results...")
    
    # Create figures directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)
    
    # Remove NaN values for analysis
    ice_load_clean = dsiceload.where(~np.isnan(dsiceload), drop=True)
    
    if len(ice_load_clean.time) == 0:
        print("No valid ice load data found!")
        return None
    
    # 1. Maximum ice load over the entire period
    print("Creating maximum ice load map...")
    max_ice_load = ice_load_clean.max(dim='time')
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(max_ice_load.values, cmap='Blues', aspect='auto')
    plt.colorbar(im, label='Maximum Ice Load (kg/m)')
    plt.title(f'Maximum Ice Load Over All Winters\nHeight: {height_value} {height_units}')
    plt.xlabel('West-East Grid Points')
    plt.ylabel('South-North Grid Points')
    plt.tight_layout()
    
    max_ice_filename = os.path.join(figures_dir, f'max_ice_load_map_{height_label}.png')
    plt.savefig(max_ice_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {max_ice_filename}")
    plt.close()
    
    # 2. Time series of average ice load
    print("Creating time series plot...")
    avg_ice_load_time = ice_load_clean.mean(dim=['south_north', 'west_east'])
    
    plt.figure(figsize=(15, 6))
    avg_ice_load_time.plot(x='time')
    plt.title(f'Average Ice Load Over Time (All Grid Points)\nHeight: {height_value} {height_units}')
    plt.xlabel('Time')
    plt.ylabel('Average Ice Load (kg/m)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    timeseries_filename = os.path.join(figures_dir, f'ice_load_timeseries_{height_label}.png')
    plt.savefig(timeseries_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {timeseries_filename}")
    plt.close()
    
    # 3. Histogram of ice load values
    print("Creating ice load distribution histogram...")
    ice_values = ice_load_clean.values.flatten()
    ice_values_no_zero = ice_values[ice_values > 0]  # Remove zeros for better visualization
    
    plt.figure(figsize=(10, 6))
    plt.hist(ice_values_no_zero, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Ice Load (kg/m)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Ice Load Values (Non-zero)\nHeight: {height_value} {height_units}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    hist_filename = os.path.join(figures_dir, f'ice_load_distribution_{height_label}.png')
    plt.savefig(hist_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {hist_filename}")
    plt.close()
    
    # 4. Seasonal patterns (if multiple winters)
    if len(dates) > 2:  # More than one winter
        print("Creating seasonal comparison...")
        
        plt.figure(figsize=(15, 8))
        
        for idate in range(len(dates)-1):
            winter_start = dates[idate]
            winter_end = dates[idate+1] - pd.to_timedelta('30min')
            
            winter_data = ice_load_clean.sel(time=slice(winter_start, winter_end))
            if len(winter_data.time) > 0:
                winter_avg = winter_data.mean(dim=['south_north', 'west_east'])
                winter_avg.plot(x='time', label=f'Winter {idate+1}')
        
        plt.title(f'Ice Load Comparison Across Winters\nHeight: {height_value} {height_units}')
        plt.xlabel('Time')
        plt.ylabel('Average Ice Load (kg/m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        seasonal_filename = os.path.join(figures_dir, f'ice_load_seasonal_comparison_{height_label}.png')
        plt.savefig(seasonal_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {seasonal_filename}")
        plt.close()
    
    # Print summary statistics
    print(f"\n=== Ice Load Analysis Summary (Height: {height_value} {height_units}) ===")
    print(f"Height level: {height_level} ({height_value} {height_units})")
    print(f"Data shape: {dsiceload.shape}")
    print(f"Valid data points: {len(ice_values):,}")
    print(f"Non-zero data points: {len(ice_values_no_zero):,}")
    print(f"Maximum ice load: {float(ice_load_clean.max().values):.3f} kg/m")
    print(f"Average ice load: {float(ice_load_clean.mean().values):.3f} kg/m")
    
    # Save the calculated ice load data for future reference with height information
    start_date = dates[0].strftime('%Y-%m-%d')
    end_date = dates[-1].strftime('%Y-%m-%d')
    save_ice_load_data(dsiceload, start_date, end_date, height_label)
    
    return dsiceload


def plot_grid_points_cartopy_map(dataset, margin_degrees=0.2, zoom_level=6, title="Grid Points on Terrain Map"):
    """
    Plot grid points on a terrain map using Cartopy with Stamen terrain background
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Dataset containing XLAT and XLON coordinates
    margin_degrees : float, optional
        Margin to add around the grid in degrees (default: 0.2)
    zoom_level : int, optional
        Zoom level for the terrain tiles (default: 6)
        Higher values = more detail but slower loading
    title : str, optional
        Title for the plot (default: "Grid Points on Terrain Map")
    
    Returns:
    --------
    dict
        Dictionary containing grid extent and map information
    """
    print(f"Creating Cartopy terrain map with {margin_degrees}° margin and zoom level {zoom_level}...")
    
    # Create geographical_maps subfolder if it doesn't exist
    geo_maps_dir = os.path.join(figures_dir, "geographical_maps")
    os.makedirs(geo_maps_dir, exist_ok=True)
    
    # Get geographical coordinates
    if 'XLAT' in dataset.coords and 'XLON' in dataset.coords:
        lats = dataset.XLAT.values
        lons = dataset.XLON.values
    elif 'XLAT' in dataset.data_vars and 'XLON' in dataset.data_vars:
        lats = dataset.XLAT.values
        lons = dataset.XLON.values
    else:
        print("Error: Could not find XLAT and XLON coordinates in dataset")
        return None
    
    print(f"Grid: {lats.shape[0]}×{lats.shape[1]} = {lats.size} points")
    print(f"Center: {lats.mean():.4f}°N, {lons.mean():.4f}°E")
    print(f"Grid extent: {lats.min():.4f}° to {lats.max():.4f}°N, {lons.min():.4f}° to {lons.max():.4f}°E")
    print(f"Map extent with margin: {lats.min()-margin_degrees:.4f}° to {lats.max()+margin_degrees:.4f}°N, {lons.min()-margin_degrees:.4f}° to {lons.max()+margin_degrees:.4f}°E")
    
    try:
        # Import Cartopy components
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cartopy.io.img_tiles as cimgt
        
        print("Using Cartopy with Stamen terrain background")
        
        # Create figure with Cartopy projection
        fig = plt.figure(figsize=(14, 10))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Set map extent with margins
        west = lons.min() - margin_degrees
        east = lons.max() + margin_degrees
        south = lats.min() - margin_degrees
        north = lats.max() + margin_degrees
        
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
        
        # Add basic geographical features first (always visible)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.7, zorder=1)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.8, zorder=2)
        ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.8, zorder=3)
        
        # Try to add terrain background (optional enhancement)
        try:
            # Use OpenStreetMap (free, no API key required)
            terrain = cimgt.OSM()
            ax.add_image(terrain, zoom_level)
            print("Successfully loaded OpenStreetMap tiles")
        except Exception as e:
            try:
                # Fallback to GoogleTiles (satellite view)
                terrain = cimgt.GoogleTiles(style='satellite')
                ax.add_image(terrain, zoom_level)
                print("Successfully loaded Google satellite tiles")
            except Exception as e2:
                print(f"All tile services unavailable, using basic land/ocean features")
        
        # Add geographical features on top
        ax.add_feature(cfeature.BORDERS, linewidth=1.5, color='black', alpha=0.8, zorder=8)
        ax.add_feature(cfeature.COASTLINE, linewidth=2, color='black', alpha=0.9, zorder=9)
        
        # Plot grid points
        scatter = ax.scatter(lons.flatten(), lats.flatten(), 
                            s=120, c='red', marker='o', alpha=0.9,
                            edgecolors='white', linewidth=2, zorder=10,
                            transform=ccrs.PlateCarree(),
                            label=f'Grid Points ({lats.size})')
        
        # Add gridlines with labels
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98),
                 fancybox=True, shadow=True, fontsize=10)
        
        # Set title
        ax.set_title(title, fontsize=14, weight='bold', pad=20)
        
        # Add margin information
        margin_text = f"Margin: {margin_degrees}° | Zoom: {zoom_level} | Terrain: Stamen"
        ax.text(0.02, 0.02, margin_text, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        margin_str = str(margin_degrees).replace('.', 'p')
        filename = os.path.join(geo_maps_dir, f"grid_cartopy_terrain_margin_{margin_str}deg_zoom{zoom_level}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved Cartopy terrain map: {filename}")
        plt.close()
        
        # Return map information
        return {
            'grid_center': (lats.mean(), lons.mean()),
            'grid_extent': {
                'lat_min': lats.min(), 'lat_max': lats.max(),
                'lon_min': lons.min(), 'lon_max': lons.max()
            },
            'map_extent_with_margin': {
                'lat_min': lats.min() - margin_degrees, 'lat_max': lats.max() + margin_degrees,
                'lon_min': lons.min() - margin_degrees, 'lon_max': lons.max() + margin_degrees
            },
            'margin_degrees': margin_degrees,
            'margin_km_approx': margin_degrees * 111,
            'zoom_level': zoom_level,
            'total_grid_points': lats.size,
            'grid_shape': lats.shape,
            'map_type': 'cartopy_stamen_terrain'
        }
        
    except ImportError as e:
        print(f"Error: Cartopy required for this function")
        print(f"Please install with: conda install cartopy")
        print(f"Import error: {e}")
        return None
        
    except Exception as e:
        print(f"Error creating Cartopy map: {e}")
        print("Make sure you have internet connection for terrain tiles")
        return None


def plot_wind_rose(dataset, grid_point=None, height=50, time_filter='all', bins=16):
    """
    Plot a wind rose for a specific grid point showing wind direction and speed distribution.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        The WRF dataset containing wind data
    grid_point : tuple or None
        Tuple of (south_north_index, west_east_index) for the grid point.
        If None, uses the center point of the grid.
    height : int
        Height level in meters (default: 50)
    time_filter : str
        Time filter option: 'all', 'winter', 'summer', 'spring', 'autumn'
    bins : int
        Number of directional bins for the wind rose (default: 16)
    
    Returns:
    --------
    dict : Wind statistics and plot information
    """
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        
        if dataset is None:
            print("Error: No dataset provided")
            return None
            
        # Find center point if not specified
        if grid_point is None:
            center_sn = len(dataset.south_north) // 2
            center_we = len(dataset.west_east) // 2
            grid_point = (center_sn, center_we)
            print(f"Using center grid point: ({center_sn}, {center_we})")
        
        sn_idx, we_idx = grid_point
        
        # Check if indices are valid
        if sn_idx >= len(dataset.south_north) or we_idx >= len(dataset.west_east):
            print(f"Error: Grid point indices ({sn_idx}, {we_idx}) are out of bounds")
            return None
            
        # Get coordinates for the selected point
        if 'XLAT' in dataset.coords or 'XLAT' in dataset.data_vars:
            lat = float(dataset.XLAT.isel(south_north=sn_idx, west_east=we_idx).values)
            lon = float(dataset.XLON.isel(south_north=sn_idx, west_east=we_idx).values)
        elif 'xlat' in dataset.coords or 'xlat' in dataset.data_vars:
            lat = float(dataset.xlat.isel(south_north=sn_idx, west_east=we_idx).values)
            lon = float(dataset.xlon.isel(south_north=sn_idx, west_east=we_idx).values)
        else:
            print("Error: Could not find latitude/longitude coordinates (XLAT/XLON or xlat/xlon)")
            return None
        
        print(f"Selected grid point: ({sn_idx}, {we_idx})")
        print(f"Coordinates: Lat {lat:.3f}°N, Lon {lon:.3f}°E")
        
        # Apply seasonal filter if specified
        ds_filtered = dataset
        if time_filter != 'all':
            months_dict = {
                'winter': [12, 1, 2],
                'spring': [3, 4, 5], 
                'summer': [6, 7, 8],
                'autumn': [9, 10, 11]
            }
            if time_filter in months_dict:
                months = months_dict[time_filter]
                ds_filtered = ds_filtered.sel(time=ds_filtered.time.dt.month.isin(months))
        
        # Look for wind components in the dataset
        wind_vars = []
        possible_u_names = ['U', 'U10', 'u', 'u10', 'WSPD_U', 'wind_u']
        possible_v_names = ['V', 'V10', 'v', 'v10', 'WSPD_V', 'wind_v']
        
        u_var = None
        v_var = None
        
        # Find U component
        for name in possible_u_names:
            if name in ds_filtered.data_vars:
                u_var = name
                break
                
        # Find V component  
        for name in possible_v_names:
            if name in ds_filtered.data_vars:
                v_var = name
                break
        
        if u_var is None or v_var is None:
            print("Available variables in dataset:")
            for var in ds_filtered.data_vars:
                print(f"  - {var}")
            print("Error: Could not find wind components (U/V) in dataset")
            return None
            
        print(f"Using wind components: U='{u_var}', V='{v_var}'")
        
        # Select height level if available
        if 'height' in ds_filtered.dims:
            try:
                ds_filtered = ds_filtered.sel(height=height, method='nearest')
                actual_height = float(ds_filtered.height.values)
                print(f"Selected height: {actual_height}m")
            except:
                print(f"Warning: Could not select height {height}m, using all heights")
        
        # Extract wind components for the selected grid point
        try:
            u_wind = ds_filtered[u_var].isel(south_north=sn_idx, west_east=we_idx)
            v_wind = ds_filtered[v_var].isel(south_north=sn_idx, west_east=we_idx)
        except:
            print(f"Error: Could not extract wind data for grid point ({sn_idx}, {we_idx})")
            return None
        
        # Convert to numpy arrays and remove NaN values
        u_values = u_wind.values.flatten()
        v_values = v_wind.values.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(u_values) | np.isnan(v_values))
        u_clean = u_values[valid_mask]
        v_clean = v_values[valid_mask]
        
        if len(u_clean) == 0:
            print("Error: No valid wind data found")
            return None
            
        # Calculate wind speed and direction
        wind_speed = np.sqrt(u_clean**2 + v_clean**2)
        wind_direction = (270 - np.arctan2(v_clean, u_clean) * 180/np.pi) % 360
        
        # Create wind speed bins
        speed_bins = [0, 2, 4, 6, 8, 10, 15, np.inf]
        speed_labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-15', '15+']
        speed_colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6', '#34495e']
        
        # Create directional bins
        dir_bins = np.linspace(0, 360, bins + 1)
        dir_centers = (dir_bins[:-1] + dir_bins[1:]) / 2
        
        # Create the wind rose
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate frequency for each direction and speed bin
        theta = np.radians(dir_centers)
        bottom = np.zeros(len(dir_centers))
        
        for i, (speed_min, speed_max) in enumerate(zip(speed_bins[:-1], speed_bins[1:])):
            if speed_max == np.inf:
                speed_mask = wind_speed >= speed_min
            else:
                speed_mask = (wind_speed >= speed_min) & (wind_speed < speed_max)
            
            if np.any(speed_mask):
                speed_dir = wind_direction[speed_mask]
                
                # Calculate frequency in each direction bin
                freq = np.zeros(len(dir_centers))
                for j, (dir_min, dir_max) in enumerate(zip(dir_bins[:-1], dir_bins[1:])):
                    dir_mask = (speed_dir >= dir_min) & (speed_dir < dir_max)
                    freq[j] = np.sum(dir_mask) / len(wind_direction) * 100
                
                # Plot bars
                bars = ax.bar(theta, freq, width=2*np.pi/bins, bottom=bottom, 
                             color=speed_colors[i], alpha=0.8, edgecolor='white', linewidth=0.5)
                bottom += freq
        
        # Customize the plot
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, np.max(bottom) * 1.1)
        
        # Add labels
        ax.set_title(f'Wind Rose - Grid Point ({sn_idx}, {we_idx})\n' + 
                    f'Lat: {lat:.3f}°N, Lon: {lon:.3f}°E\n' +
                    f'Height: {height}m, Period: {time_filter}', 
                    pad=20, fontsize=14)
        
        # Add direction labels
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        for direction, angle in zip(directions, angles):
            ax.text(np.radians(angle), ax.get_ylim()[1] * 1.05, direction, 
                   ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=speed_colors[i], alpha=0.8, 
                          edgecolor='white') for i in range(len(speed_labels))]
        ax.legend(legend_elements, [f'{label} m/s' for label in speed_labels], 
                 loc='upper left', bbox_to_anchor=(1.1, 1))
        
        # Add radial labels (frequency %)
        ax.set_ylabel('Frequency (%)', labelpad=30)
        
        plt.tight_layout()
        
        # Create WindRose subdirectory if it doesn't exist
        wind_rose_dir = os.path.join(figures_dir, "WindRose")
        os.makedirs(wind_rose_dir, exist_ok=True)
        
        # Save plot with automatic filename
        filename = f"wind_rose_point_{sn_idx}_{we_idx}_h{height}m_{time_filter}.png"
        filepath = os.path.join(wind_rose_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Wind rose saved to: {filepath}")
        
        plt.show()
        
        # Calculate and return statistics
        stats = {
            'grid_point': grid_point,
            'coordinates': {'lat': lat, 'lon': lon},
            'height': height,
            'n_observations': len(wind_speed),
            'mean_speed': float(np.mean(wind_speed)),
            'max_speed': float(np.max(wind_speed)),
            'prevailing_direction': float(dir_centers[np.argmax(bottom)]),
            'calm_percentage': float(np.sum(wind_speed < 1) / len(wind_speed) * 100),
            'time_filter': time_filter,
            'wind_components_used': {'u': u_var, 'v': v_var}
        }
        
        print(f"\nWind Statistics:")
        print(f"  Mean wind speed: {stats['mean_speed']:.2f} m/s")
        print(f"  Maximum wind speed: {stats['max_speed']:.2f} m/s") 
        print(f"  Prevailing direction: {stats['prevailing_direction']:.1f}°")
        print(f"  Calm conditions (<1 m/s): {stats['calm_percentage']:.1f}%")
        print(f"  Total observations: {stats['n_observations']}")
        
        return stats
        
    except ImportError as e:
        print(f"Error: Missing required library: {e}")
        return None
    except Exception as e:
        print(f"Error creating wind rose: {e}")
        return None

