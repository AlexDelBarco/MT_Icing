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


def accreation_per_winter(ds, start_date, end_date):
    # Check if dataset is None
    if ds is None:
        print("Dataset is None")
        return None

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
        
        plot_data = ds1_filtered.ACCRE_CYL.isel(height=0).groupby('winterno').sum(dim='time')
                
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
            filename = os.path.join(ice_accretion_dir, f"ice_accretion_winter_{int(winter_idx)}.png")
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


def create_grid_on_earth_map(dataset):
    """
    Create a map showing the computational grid overlaid on an actual Earth map using GDAL
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Original dataset containing XLAT and XLON coordinates
    
    Returns:
    --------
    dict
        Dictionary containing map and grid information
    """
    print("Creating grid overlay on Earth map using GDAL...")
    
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
    
    print(f"Grid extent - Lat: {lats.min():.4f}° to {lats.max():.4f}°")
    print(f"Grid extent - Lon: {lons.min():.4f}° to {lons.max():.4f}°")
    
    # Try GDAL and related libraries
    try:
        from osgeo import gdal, ogr, osr
        import geopandas as gpd
        import contextily as ctx
        gdal_available = True
        print("Using GDAL for Earth map data access")
    except ImportError as e:
        gdal_available = False
        print(f"GDAL/GeoPandas not available ({e}), trying alternative approaches...")
    
    # Calculate map extent with padding
    padding = 0.1  # degrees
    extent = [lons.min() - padding, lons.max() + padding, 
             lats.min() - padding, lats.max() + padding]
    
    try:
        if gdal_available:
            print("Creating GDAL-based Earth map...")
            
            # Try to get Natural Earth data or OpenStreetMap data
            try:
                # Method 1: Try to download Natural Earth data using geopandas
                print("Attempting to download Natural Earth coastline data...")
                
                # Download world boundaries
                world = gpd.read_file("https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip")
                
                # Create figure with GDAL-based mapping
                fig, axes = plt.subplots(2, 2, figsize=(20, 16))
                
                # Plot 1: Grid on world map
                ax1 = axes[0, 0]
                
                # Clip world data to region of interest
                world_clipped = world.cx[extent[0]:extent[1], extent[2]:extent[3]]
                
                if not world_clipped.empty:
                    world_clipped.plot(ax=ax1, color='lightgray', alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Add grid points and lines
                ax1.scatter(lons.flatten(), lats.flatten(), s=50, c='red', 
                           marker='s', alpha=0.8, label='Grid Points', 
                           edgecolors='darkred', linewidth=1, zorder=5)
                
                # Add grid wireframe
                for i in range(lats.shape[0]):
                    ax1.plot(lons[i, :], lats[i, :], 'r-', alpha=0.6, linewidth=1, zorder=4)
                for j in range(lats.shape[1]):
                    ax1.plot(lons[:, j], lats[:, j], 'r-', alpha=0.6, linewidth=1, zorder=4)
                
                ax1.set_xlim(extent[0], extent[1])
                ax1.set_ylim(extent[2], extent[3])
                ax1.set_xlabel('Longitude (degrees)')
                ax1.set_ylabel('Latitude (degrees)')
                ax1.set_title('Computational Grid on Earth Map (GDAL)', fontsize=14, weight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Plot 2: Regional context
                ax2 = axes[0, 1]
                context_padding = 2.0
                context_extent = [lons.min() - context_padding, lons.max() + context_padding, 
                                lats.min() - context_padding, lats.max() + context_padding]
                
                world_context = world.cx[context_extent[0]:context_extent[1], context_extent[2]:context_extent[3]]
                if not world_context.empty:
                    world_context.plot(ax=ax2, color='lightgray', alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Add grid domain boundary
                boundary_lons = np.concatenate([lons[0, :], lons[:, -1], lons[-1, ::-1], lons[::-1, 0]])
                boundary_lats = np.concatenate([lats[0, :], lats[:, -1], lats[-1, ::-1], lats[::-1, 0]])
                ax2.plot(boundary_lons, boundary_lats, 'red', linewidth=3, label='Grid Domain')
                
                # Add grid center
                center_lon = lons.mean()
                center_lat = lats.mean()
                ax2.scatter(center_lon, center_lat, s=200, c='red', marker='*', 
                           label='Grid Center', edgecolors='darkred', linewidth=2)
                
                ax2.set_xlim(context_extent[0], context_extent[1])
                ax2.set_ylim(context_extent[2], context_extent[3])
                ax2.set_xlabel('Longitude (degrees)')
                ax2.set_ylabel('Latitude (degrees)')
                ax2.set_title('Regional Context (GDAL)', fontsize=14, weight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                print("Successfully created map using Natural Earth data")
                
            except Exception as e:
                print(f"Natural Earth download failed ({e}), trying alternative...")
                
                # Method 2: Create enhanced basic map with GDAL coordinate transformations
                fig, axes = plt.subplots(2, 2, figsize=(20, 16))
                
                # Use GDAL to create coordinate reference system
                source_srs = osr.SpatialReference()
                source_srs.ImportFromEPSG(4326)  # WGS84
                
                # Plot 1: Enhanced grid visualization
                ax1 = axes[0, 0]
                ax1.scatter(lons.flatten(), lats.flatten(), s=50, c='red', 
                           marker='s', alpha=0.8, label='Grid Points',
                           edgecolors='darkred', linewidth=1)
                
                # Add grid wireframe
                for i in range(lats.shape[0]):
                    ax1.plot(lons[i, :], lats[i, :], 'r-', alpha=0.6, linewidth=1)
                for j in range(lats.shape[1]):
                    ax1.plot(lons[:, j], lats[:, j], 'r-', alpha=0.6, linewidth=1)
                
                ax1.set_xlim(extent[0], extent[1])
                ax1.set_ylim(extent[2], extent[3])
                ax1.set_xlabel('Longitude (degrees)')
                ax1.set_ylabel('Latitude (degrees)')
                ax1.set_title('Computational Grid (GDAL Enhanced)', fontsize=14, weight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Add coordinate grid lines
                lat_ticks = np.linspace(extent[2], extent[3], 5)
                lon_ticks = np.linspace(extent[0], extent[1], 5)
                ax1.set_xticks(lon_ticks)
                ax1.set_yticks(lat_ticks)
                
                # Plot 2: Domain context
                ax2 = axes[0, 1]
                boundary_lons = np.concatenate([lons[0, :], lons[:, -1], lons[-1, ::-1], lons[::-1, 0]])
                boundary_lats = np.concatenate([lats[0, :], lats[:, -1], lats[-1, ::-1], lats[::-1, 0]])
                ax2.plot(boundary_lons, boundary_lats, 'red', linewidth=3, label='Grid Domain')
                ax2.scatter(lons.mean(), lats.mean(), s=200, c='red', marker='*', 
                           label='Grid Center', edgecolors='darkred', linewidth=2)
                ax2.set_xlabel('Longitude (degrees)')
                ax2.set_ylabel('Latitude (degrees)')
                ax2.set_title('Grid Domain (GDAL CRS)', fontsize=14, weight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Grid with coordinate system info
            ax3 = axes[1, 0]
            
            # Create a more detailed grid visualization
            if lats.size <= 50:
                for i in range(lats.shape[0]):
                    for j in range(lats.shape[1]):
                        ax3.scatter(lons[i, j], lats[i, j], s=60, c='red', marker='s')
                        ax3.text(lons[i, j], lats[i, j], f'{i},{j}', 
                                ha='center', va='center', fontsize=8, color='white', weight='bold')
            else:
                ax3.scatter(lons.flatten(), lats.flatten(), s=40, c='red', marker='s', alpha=0.8)
            
            ax3.set_xlim(extent[0], extent[1])
            ax3.set_ylim(extent[2], extent[3])
            ax3.set_xlabel('Longitude (degrees)')
            ax3.set_ylabel('Latitude (degrees)')
            ax3.set_title('Grid Points with GDAL CRS', fontsize=14, weight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: GDAL Information panel
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Calculate domain information using GDAL
            center_lat = lats.mean()
            center_lon = lons.mean()
            domain_width_km = (lons.max() - lons.min()) * 111.32 * np.cos(np.radians(center_lat))
            domain_height_km = (lats.max() - lats.min()) * 111.32
            
            # Get GDAL version and capabilities
            gdal_version = gdal.__version__ if hasattr(gdal, '__version__') else "Unknown"
            
            info_text = f"""GDAL-ENHANCED GRID ANALYSIS

GDAL Configuration:
• Version: {gdal_version}
• Coordinate Reference System: EPSG:4326 (WGS84)
• Spatial Reference: Geographic (Lat/Lon)

Grid Configuration:
• Shape: {lats.shape[0]} × {lats.shape[1]} = {lats.size} points
• Type: Structured rectangular grid

Geographic Location:
• Center: {center_lat:.4f}°N, {center_lon:.4f}°E
• Latitude range: {lats.min():.4f}° to {lats.max():.4f}°
• Longitude range: {lons.min():.4f}° to {lons.max():.4f}°

Domain Size (GDAL calculated):
• Width: {domain_width_km:.1f} km (E-W)
• Height: {domain_height_km:.1f} km (N-S)
• Total area: ~{domain_width_km * domain_height_km:.0f} km²

Grid Resolution:
• Latitude spacing: {np.abs(np.diff(lats, axis=0)).mean():.4f}°
• Longitude spacing: {np.abs(np.diff(lons, axis=1)).mean():.4f}°

Map Data Source: {'Natural Earth' if 'world' in locals() else 'GDAL Enhanced Coordinates'}"""
            
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        else:
            # Fallback without GDAL
            print("Creating basic grid map without GDAL features...")
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Basic grid visualization
            axes[0,0].scatter(lons.flatten(), lats.flatten(), s=50, c='red', marker='s', alpha=0.8)
            for i in range(lats.shape[0]):
                axes[0,0].plot(lons[i, :], lats[i, :], 'r-', alpha=0.6, linewidth=1)
            for j in range(lats.shape[1]):
                axes[0,0].plot(lons[:, j], lats[:, j], 'r-', alpha=0.6, linewidth=1)
            
            axes[0,0].set_title('Computational Grid (Basic View)')
            axes[0,0].set_xlabel('Longitude (degrees)')
            axes[0,0].set_ylabel('Latitude (degrees)')
            axes[0,0].grid(True, alpha=0.3)
            
            # Grid boundary
            boundary_lons = np.concatenate([lons[0, :], lons[:, -1], lons[-1, ::-1], lons[::-1, 0]])
            boundary_lats = np.concatenate([lats[0, :], lats[:, -1], lats[-1, ::-1], lats[::-1, 0]])
            
            axes[0,1].plot(boundary_lons, boundary_lats, 'red', linewidth=3, label='Grid Domain')
            axes[0,1].scatter(lons.mean(), lats.mean(), s=200, c='red', marker='*', label='Grid Center')
            axes[0,1].set_title('Grid Domain Boundary')
            axes[0,1].set_xlabel('Longitude (degrees)')
            axes[0,1].set_ylabel('Latitude (degrees)')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Grid with indices
            if lats.size <= 50:
                for i in range(lats.shape[0]):
                    for j in range(lats.shape[1]):
                        axes[1,0].scatter(lons[i, j], lats[i, j], s=60, c='red', marker='s')
                        axes[1,0].text(lons[i, j], lats[i, j], f'{i},{j}', 
                                     ha='center', va='center', fontsize=8)
            else:
                axes[1,0].scatter(lons.flatten(), lats.flatten(), s=40, c='red', marker='s')
            
            axes[1,0].set_title('Grid Points with Indices')
            axes[1,0].set_xlabel('Longitude (degrees)')
            axes[1,0].set_ylabel('Latitude (degrees)')
            axes[1,0].grid(True, alpha=0.3)
            
            # Information
            axes[1,1].axis('off')
            info_text = f"""Grid Information:
            
Shape: {lats.shape[0]} × {lats.shape[1]} points
Center: {lats.mean():.4f}°N, {lons.mean():.4f}°E
Lat range: {lats.min():.4f}° to {lats.max():.4f}°
Lon range: {lons.min():.4f}° to {lons.max():.4f}°

For GDAL Earth maps, install:
pip install geopandas contextily
conda install -c conda-forge gdal"""
            
            axes[1,1].text(0.05, 0.95, info_text, transform=axes[1,1].transAxes,
                          fontsize=12, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the map
        earth_map_filename = os.path.join(geo_maps_dir, 'grid_on_earth_map_gdal.png')
        plt.savefig(earth_map_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {earth_map_filename}")
        plt.close()
        
        return {
            'grid_shape': lats.shape,
            'lat_range': [float(lats.min()), float(lats.max())],
            'lon_range': [float(lons.min()), float(lons.max())],
            'center_lat': float(lats.mean()),
            'center_lon': float(lons.mean()),
            'gdal_available': gdal_available,
            'domain_size_km': [float((lons.max() - lons.min()) * 111.32 * np.cos(np.radians(lats.mean()))),
                              float((lats.max() - lats.min()) * 111.32)],
            'map_source': 'Natural Earth' if gdal_available else 'Basic coordinates'
        }
        
    except Exception as e:
        print(f"Error creating GDAL Earth map: {e}")
        return None


def create_grid_overlay_map(dataset):
    """
    Create a simple map showing only the computational grid without any data overlay
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Original dataset containing XLAT and XLON coordinates
    
    Returns:
    --------
    dict
        Dictionary containing grid information
    """
    print("Creating grid overlay map...")
    
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
    
    print(f"Grid coordinates - Lat: {lats.min():.4f}° to {lats.max():.4f}°")
    print(f"Grid coordinates - Lon: {lons.min():.4f}° to {lons.max():.4f}°")
    print(f"Grid shape: {lats.shape}")
    
    # Create the grid overlay plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Grid points as scatter plot
    axes[0,0].scatter(lons.flatten(), lats.flatten(), s=20, c='red', alpha=0.8, marker='s')
    axes[0,0].set_title('Computational Grid Points')
    axes[0,0].set_xlabel('Longitude (degrees)')
    axes[0,0].set_ylabel('Latitude (degrees)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_aspect('equal', adjustable='box')
    
    # Add grid point numbers if grid is small enough
    if lats.size <= 50:  # Only for small grids
        for i in range(lats.shape[0]):
            for j in range(lats.shape[1]):
                axes[0,0].annotate(f'({i},{j})', 
                                 (lons[i,j], lats[i,j]), 
                                 xytext=(3, 3), textcoords='offset points',
                                 fontsize=6, alpha=0.7)
    
    # Plot 2: Grid with connecting lines (wireframe)
    axes[0,1].plot(lons, lats, 'b-', alpha=0.6, linewidth=0.5)  # Horizontal lines
    axes[0,1].plot(lons.T, lats.T, 'b-', alpha=0.6, linewidth=0.5)  # Vertical lines
    axes[0,1].scatter(lons.flatten(), lats.flatten(), s=15, c='red', alpha=0.8, marker='o')
    axes[0,1].set_title('Grid Wireframe')
    axes[0,1].set_xlabel('Longitude (degrees)')
    axes[0,1].set_ylabel('Latitude (degrees)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_aspect('equal', adjustable='box')
    
    # Plot 3: Grid boundary and center
    # Calculate grid boundary
    boundary_lons = np.concatenate([lons[0, :], lons[:, -1], lons[-1, ::-1], lons[::-1, 0]])
    boundary_lats = np.concatenate([lats[0, :], lats[:, -1], lats[-1, ::-1], lats[::-1, 0]])
    
    # Calculate grid center
    center_lon = lons.mean()
    center_lat = lats.mean()
    
    axes[1,0].plot(boundary_lons, boundary_lats, 'g-', linewidth=2, label='Grid Boundary')
    axes[1,0].scatter(lons.flatten(), lats.flatten(), s=10, c='blue', alpha=0.6, marker='.')
    axes[1,0].scatter(center_lon, center_lat, s=100, c='red', marker='*', label='Grid Center')
    axes[1,0].set_title('Grid Boundary and Center')
    axes[1,0].set_xlabel('Longitude (degrees)')
    axes[1,0].set_ylabel('Latitude (degrees)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_aspect('equal', adjustable='box')
    
    # Plot 4: Grid spacing analysis
    if lats.shape[0] > 1 and lats.shape[1] > 1:
        # Calculate grid spacing
        lat_spacing = np.abs(np.diff(lats, axis=0)).mean()
        lon_spacing = np.abs(np.diff(lons, axis=1)).mean()
        
        # Approximate physical distances (rough estimate)
        lat_km = lat_spacing * 111.32  # km per degree latitude
        lon_km = lon_spacing * 111.32 * np.cos(np.radians(lats.mean()))  # km per degree longitude
        
        # Display grid information
        info_text = f"""Grid Information:
        
Grid Shape: {lats.shape[0]} × {lats.shape[1]} = {lats.size} points

Coordinate Ranges:
• Latitude: {lats.min():.4f}° to {lats.max():.4f}°
• Longitude: {lons.min():.4f}° to {lons.max():.4f}°

Grid Spacing:
• Latitude: {lat_spacing:.4f}° (~{lat_km:.2f} km)
• Longitude: {lon_spacing:.4f}° (~{lon_km:.2f} km)

Grid Center:
• Lat: {center_lat:.4f}°, Lon: {center_lon:.4f}°

Approximate Domain Size:
• North-South: {(lats.max() - lats.min()) * 111.32:.1f} km
• East-West: {(lons.max() - lons.min()) * 111.32 * np.cos(np.radians(lats.mean())):.1f} km"""
        
        axes[1,1].text(0.05, 0.95, info_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1,1].set_title('Grid Specifications')
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
    
    plt.tight_layout()
    
    # Save the grid overlay map
    grid_filename = os.path.join(geo_maps_dir, 'computational_grid_overlay.png')
    plt.savefig(grid_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {grid_filename}")
    plt.close()
    
    # Create a detailed grid coordinate plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Longitude mesh
    im1 = axes[0].imshow(lons, cmap='plasma', aspect='auto')
    axes[0].set_title('Longitude Coordinate Grid')
    axes[0].set_xlabel('West-East Grid Index')
    axes[0].set_ylabel('South-North Grid Index')
    
    # Add coordinate values as text overlay for small grids
    if lats.size <= 25:  # Only for very small grids
        for i in range(lons.shape[0]):
            for j in range(lons.shape[1]):
                axes[0].text(j, i, f'{lons[i,j]:.3f}', 
                           ha='center', va='center', fontsize=8, color='white')
    
    plt.colorbar(im1, ax=axes[0], label='Longitude (degrees)')
    
    # Plot 2: Latitude mesh
    im2 = axes[1].imshow(lats, cmap='viridis', aspect='auto')
    axes[1].set_title('Latitude Coordinate Grid')
    axes[1].set_xlabel('West-East Grid Index')
    axes[1].set_ylabel('South-North Grid Index')
    
    # Add coordinate values as text overlay for small grids
    if lats.size <= 25:  # Only for very small grids
        for i in range(lats.shape[0]):
            for j in range(lats.shape[1]):
                axes[1].text(j, i, f'{lats[i,j]:.3f}', 
                           ha='center', va='center', fontsize=8, color='white')
    
    plt.colorbar(im2, ax=axes[1], label='Latitude (degrees)')
    
    plt.tight_layout()
    
    # Save the coordinate detail plot
    coord_detail_filename = os.path.join(geo_maps_dir, 'grid_coordinate_details.png')
    plt.savefig(coord_detail_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {coord_detail_filename}")
    plt.close()
    
    # Return grid information
    return {
        'grid_shape': lats.shape,
        'lat_range': [float(lats.min()), float(lats.max())],
        'lon_range': [float(lons.min()), float(lons.max())],
        'center_lat': float(center_lat),
        'center_lon': float(center_lon),
        'total_points': int(lats.size),
        'lat_spacing_deg': float(lat_spacing) if lats.shape[0] > 1 else 0,
        'lon_spacing_deg': float(lon_spacing) if lats.shape[1] > 1 else 0,
        'approx_lat_spacing_km': float(lat_km) if lats.shape[0] > 1 else 0,
        'approx_lon_spacing_km': float(lon_km) if lats.shape[1] > 1 else 0
    }


def create_geographical_ice_load_maps_gdal(ice_load_data, dataset):
    """
    Create geographical maps of ice load data using GDAL for proper geospatial handling
    
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
    print("Creating geographical ice load maps using GDAL...")
    
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
    
    # Try to import GDAL libraries
    try:
        from osgeo import gdal, osr
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject, Resampling
        gdal_available = True
        print("Using GDAL for geospatial processing")
    except ImportError as e:
        gdal_available = False
        print(f"GDAL not available ({e}), using basic matplotlib plotting")
    
    if gdal_available:
        # Create GeoTIFF files for proper geospatial handling
        try:
            # Define the coordinate system (assuming WGS84)
            crs = 'EPSG:4326'  # WGS84 geographic coordinate system
            
            # Calculate bounds and transform
            bounds = (lons.min(), lats.min(), lons.max(), lats.max())
            transform = from_bounds(*bounds, lons.shape[1], lons.shape[0])
            
            # Save ice load data as GeoTIFF
            max_ice_geotiff = os.path.join(geo_maps_dir, 'max_ice_load.tif')
            mean_ice_geotiff = os.path.join(geo_maps_dir, 'mean_ice_load.tif')
            
            # Write maximum ice load GeoTIFF
            with rasterio.open(
                max_ice_geotiff, 'w',
                driver='GTiff',
                height=max_ice_load.shape[0],
                width=max_ice_load.shape[1],
                count=1,
                dtype=max_ice_load.dtype,
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(max_ice_load, 1)
                # Add metadata
                dst.update_tags(
                    title='Maximum Ice Load',
                    description='Maximum ice load over study period',
                    units='kg/m'
                )
            
            # Write mean ice load GeoTIFF
            with rasterio.open(
                mean_ice_geotiff, 'w',
                driver='GTiff',
                height=mean_ice_load.shape[0],
                width=mean_ice_load.shape[1],
                count=1,
                dtype=mean_ice_load.dtype,
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(mean_ice_load, 1)
                dst.update_tags(
                    title='Mean Ice Load',
                    description='Mean ice load over study period',
                    units='kg/m'
                )
            
            print(f"Saved GeoTIFF files: {max_ice_geotiff}, {mean_ice_geotiff}")
            
            # Create enhanced plots with GDAL capabilities
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            
            # Plot 1: Maximum ice load with geographical grid
            im1 = axes[0,0].pcolormesh(lons, lats, max_ice_load, cmap='Blues', shading='auto')
            axes[0,0].set_title('Maximum Ice Load (GDAL Processed)')
            axes[0,0].set_xlabel('Longitude (degrees)')
            axes[0,0].set_ylabel('Latitude (degrees)')
            axes[0,0].grid(True, alpha=0.3)
            # Add coordinate grid lines
            axes[0,0].set_xticks(np.linspace(lons.min(), lons.max(), 5))
            axes[0,0].set_yticks(np.linspace(lats.min(), lats.max(), 5))
            plt.colorbar(im1, ax=axes[0,0], label='Ice Load (kg/m)')
            
            # Plot 2: Mean ice load with geographical grid
            im2 = axes[0,1].pcolormesh(lons, lats, mean_ice_load, cmap='Blues', shading='auto')
            axes[0,1].set_title('Mean Ice Load (GDAL Processed)')
            axes[0,1].set_xlabel('Longitude (degrees)')
            axes[0,1].set_ylabel('Latitude (degrees)')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].set_xticks(np.linspace(lons.min(), lons.max(), 5))
            axes[0,1].set_yticks(np.linspace(lats.min(), lats.max(), 5))
            plt.colorbar(im2, ax=axes[0,1], label='Ice Load (kg/m)')
            
            # Plot 3: Grid coordinate system analysis
            # Calculate grid cell areas (approximate)
            lat_rad = np.radians(lats)
            lon_rad = np.radians(lons)
            
            # Approximate cell areas in km²
            dlat = np.abs(np.diff(lat_rad, axis=0).mean()) if lat_rad.shape[0] > 1 else 0.01
            dlon = np.abs(np.diff(lon_rad, axis=1).mean()) if lat_rad.shape[1] > 1 else 0.01
            
            # Earth radius in km
            R = 6371.0
            cell_areas = R**2 * dlon * dlat * np.cos(lat_rad)
            
            im3 = axes[1,0].imshow(cell_areas, cmap='viridis', aspect='auto')
            axes[1,0].set_title('Grid Cell Areas (km²)')
            axes[1,0].set_xlabel('West-East Grid Index')
            axes[1,0].set_ylabel('South-North Grid Index')
            plt.colorbar(im3, ax=axes[1,0], label='Area (km²)')
            
            # Plot 4: Projection information and grid quality
            # Calculate grid distortion
            if lats.shape[0] > 1 and lats.shape[1] > 1:
                lat_spacing = np.abs(np.diff(lats, axis=0))
                lon_spacing = np.abs(np.diff(lons, axis=1))
                
                # Plot grid spacing variability
                axes[1,1].hist(lat_spacing.flatten(), bins=30, alpha=0.7, label='Latitude spacing', density=True)
                axes[1,1].hist(lon_spacing.flatten(), bins=30, alpha=0.7, label='Longitude spacing', density=True)
                axes[1,1].set_title('Grid Spacing Distribution')
                axes[1,1].set_xlabel('Spacing (degrees)')
                axes[1,1].set_ylabel('Density')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error in GDAL processing: {e}")
            gdal_available = False
    
    if not gdal_available:
        # Fallback to basic matplotlib plotting
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Maximum ice load
        im1 = axes[0,0].pcolormesh(lons, lats, max_ice_load, cmap='Blues', shading='auto')
        axes[0,0].set_title('Maximum Ice Load')
        axes[0,0].set_xlabel('Longitude (degrees)')
        axes[0,0].set_ylabel('Latitude (degrees)')
        axes[0,0].grid(True, alpha=0.3)
        plt.colorbar(im1, ax=axes[0,0], label='Ice Load (kg/m)')
        
        # Plot 2: Mean ice load
        im2 = axes[0,1].pcolormesh(lons, lats, mean_ice_load, cmap='Blues', shading='auto')
        axes[0,1].set_title('Mean Ice Load')
        axes[0,1].set_xlabel('Longitude (degrees)')
        axes[0,1].set_ylabel('Latitude (degrees)')
        axes[0,1].grid(True, alpha=0.3)
        plt.colorbar(im2, ax=axes[0,1], label='Ice Load (kg/m)')
        
        # Plot 3: Grid points
        axes[1,0].scatter(lons.flatten(), lats.flatten(), s=1, c='red', alpha=0.6)
        axes[1,0].set_title('Model Grid Points')
        axes[1,0].set_xlabel('Longitude (degrees)')
        axes[1,0].set_ylabel('Latitude (degrees)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Ice load with contours
        im4 = axes[1,1].pcolormesh(lons, lats, max_ice_load, cmap='Blues', alpha=0.8, shading='auto')
        contours = axes[1,1].contour(lons, lats, max_ice_load, levels=10, colors='black', alpha=0.6)
        axes[1,1].clabel(contours, inline=True, fontsize=8)
        axes[1,1].set_title('Maximum Ice Load with Contours')
        axes[1,1].set_xlabel('Longitude (degrees)')
        axes[1,1].set_ylabel('Latitude (degrees)')
        axes[1,1].grid(True, alpha=0.3)
        plt.colorbar(im4, ax=axes[1,1], label='Ice Load (kg/m)')
    
    plt.tight_layout()
    
    # Save the geographical map
    geo_filename = os.path.join(geo_maps_dir, 'ice_load_geographical_maps_gdal.png')
    plt.savefig(geo_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {geo_filename}")
    plt.close()
    
    # Create coordinate system information plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Coordinate system overview
    axes[0,0].text(0.1, 0.9, f"Coordinate System Information", fontsize=14, weight='bold', transform=axes[0,0].transAxes)
    axes[0,0].text(0.1, 0.8, f"Latitude range: {lats.min():.4f}° to {lats.max():.4f}°", fontsize=12, transform=axes[0,0].transAxes)
    axes[0,0].text(0.1, 0.7, f"Longitude range: {lons.min():.4f}° to {lons.max():.4f}°", fontsize=12, transform=axes[0,0].transAxes)
    axes[0,0].text(0.1, 0.6, f"Grid shape: {lats.shape}", fontsize=12, transform=axes[0,0].transAxes)
    axes[0,0].text(0.1, 0.5, f"Total grid points: {lats.size:,}", fontsize=12, transform=axes[0,0].transAxes)
    axes[0,0].text(0.1, 0.4, f"GDAL available: {gdal_available}", fontsize=12, transform=axes[0,0].transAxes)
    if gdal_available:
        axes[0,0].text(0.1, 0.3, f"Coordinate Reference System: EPSG:4326 (WGS84)", fontsize=12, transform=axes[0,0].transAxes)
        axes[0,0].text(0.1, 0.2, f"GeoTIFF files created", fontsize=12, transform=axes[0,0].transAxes)
    axes[0,0].set_xlim(0, 1)
    axes[0,0].set_ylim(0, 1)
    axes[0,0].set_title('Geospatial Metadata')
    axes[0,0].axis('off')
    
    # Plot 2: Latitude grid
    im2 = axes[0,1].imshow(lats, cmap='viridis', aspect='auto')
    axes[0,1].set_title('Latitude Grid')
    axes[0,1].set_xlabel('West-East Grid Index')
    axes[0,1].set_ylabel('South-North Grid Index')
    plt.colorbar(im2, ax=axes[0,1], label='Latitude (degrees)')
    
    # Plot 3: Longitude grid
    im3 = axes[1,0].imshow(lons, cmap='plasma', aspect='auto')
    axes[1,0].set_title('Longitude Grid')
    axes[1,0].set_xlabel('West-East Grid Index')
    axes[1,0].set_ylabel('South-North Grid Index')
    plt.colorbar(im3, ax=axes[1,0], label='Longitude (degrees)')
    
    # Plot 4: Grid quality assessment
    if lats.shape[0] > 1 and lats.shape[1] > 1:
        # Calculate grid regularity
        lat_diff = np.diff(lats, axis=0)
        lon_diff = np.diff(lons, axis=1)
        
        lat_regularity = np.std(lat_diff) / np.mean(np.abs(lat_diff)) if np.mean(np.abs(lat_diff)) > 0 else 0
        lon_regularity = np.std(lon_diff) / np.mean(np.abs(lon_diff)) if np.mean(np.abs(lon_diff)) > 0 else 0
        
        axes[1,1].bar(['Latitude\nRegularity', 'Longitude\nRegularity'], 
                     [lat_regularity, lon_regularity], 
                     color=['blue', 'red'], alpha=0.7)
        axes[1,1].set_title('Grid Regularity Index\n(lower = more regular)')
        axes[1,1].set_ylabel('Regularity Index')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the coordinate information plot
    coord_info_filename = os.path.join(geo_maps_dir, 'coordinate_system_info_gdal.png')
    plt.savefig(coord_info_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {coord_info_filename}")
    plt.close()
    
    # Return statistics
    return {
        'lat_range': [float(lats.min()), float(lats.max())],
        'lon_range': [float(lons.min()), float(lons.max())],
        'grid_shape': lats.shape,
        'gdal_available': gdal_available,
        'coordinate_system': 'EPSG:4326 (WGS84)' if gdal_available else 'Basic lat/lon',
        'geotiff_created': gdal_available
    }


def create_geographical_ice_load_maps(ice_load_data, dataset):
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


def save_ice_load_data(dsiceload, start_date, end_date):
    """
    Save calculated ice load data to disk
    
    Parameters:
    -----------
    dsiceload : xarray.DataArray
        Calculated ice load data
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename based on date range
    start_str = pd.to_datetime(start_date).strftime('%Y%m%d')
    end_str = pd.to_datetime(end_date).strftime('%Y%m%d')
    filename = f"iceload_{start_str}_to_{end_str}.nc"
    filepath = os.path.join(results_dir, filename)
    
    print(f"Saving ice load data to: {filepath}")
    try:
        dsiceload.to_netcdf(filepath)
        print(f"Successfully saved ice load data with shape: {dsiceload.shape}")
    except Exception as e:
        print(f"Error saving ice load data: {e}")


def calculate_ice_load(ds1, dates, method, create_figures=True):
    """
    Calculate ice load and create basic visualization figures
    
    Parameters:
    -----------
    ds1 : xarray.Dataset
        Dataset containing ice accretion and ablation data
    dates : pandas.DatetimeIndex
        Date range for winter seasons
    create_figures : bool
        Whether to create basic visualization figures (default: True)
        Note: Gradient analysis plots are created separately using dedicated functions
    
    Returns:
    --------
    xarray.DataArray
        Calculated ice load data for all time periods
    """
    
    print("Calculating ice load...")
    
    dsiceload = xr.zeros_like(ds1['ACCRE_CYL'].isel(height=0)) * np.nan
    for idate,date in enumerate(dates[:-1]):
        print(f"Processing winter {idate+1}/{len(dates)-1}: {date} to {dates[idate+1]-pd.to_timedelta('30min')}")
        
        # Get data for this winter period
        winter_accre = ds1['ACCRE_CYL'].isel(height=0).sel(time=slice(date,dates[idate+1]-pd.to_timedelta('30min'))).load()
        winter_ablat = ds1['ABLAT_CYL'].isel(height=0).sel(time=slice(date,dates[idate+1]-pd.to_timedelta('30min'))).load()
        
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
    plt.title('Maximum Ice Load Over All Winters')
    plt.xlabel('West-East Grid Points')
    plt.ylabel('South-North Grid Points')
    plt.tight_layout()
    
    max_ice_filename = os.path.join(figures_dir, 'max_ice_load_map.png')
    plt.savefig(max_ice_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {max_ice_filename}")
    plt.close()
    
    # 2. Time series of average ice load
    print("Creating time series plot...")
    avg_ice_load_time = ice_load_clean.mean(dim=['south_north', 'west_east'])
    
    plt.figure(figsize=(15, 6))
    avg_ice_load_time.plot(x='time')
    plt.title('Average Ice Load Over Time (All Grid Points)')
    plt.xlabel('Time')
    plt.ylabel('Average Ice Load (kg/m)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    timeseries_filename = os.path.join(figures_dir, 'ice_load_timeseries.png')
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
    plt.title('Distribution of Ice Load Values (Non-zero)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    hist_filename = os.path.join(figures_dir, 'ice_load_distribution.png')
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
        
        plt.title('Ice Load Comparison Across Winters')
        plt.xlabel('Time')
        plt.ylabel('Average Ice Load (kg/m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        seasonal_filename = os.path.join(figures_dir, 'ice_load_seasonal_comparison.png')
        plt.savefig(seasonal_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {seasonal_filename}")
        plt.close()
    
    # Print summary statistics
    print(f"\n=== Ice Load Analysis Summary ===")
    print(f"Data shape: {dsiceload.shape}")
    print(f"Valid data points: {len(ice_values):,}")
    print(f"Non-zero data points: {len(ice_values_no_zero):,}")
    print(f"Maximum ice load: {float(ice_load_clean.max().values):.3f} kg/m")
    print(f"Average ice load: {float(ice_load_clean.mean().values):.3f} kg/m")
    
    # Save the calculated ice load data for future reference
    start_date = dates[0].strftime('%Y-%m-%d')
    end_date = dates[-1].strftime('%Y-%m-%d')
    save_ice_load_data(dsiceload, start_date, end_date)
    
    return dsiceload

