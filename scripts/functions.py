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
    
    # Create figures directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)

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
            filename = f"{figures_dir}/ice_accretion_winter_{int(winter_idx)}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()  # Close figure to free memory
            
        print(f"All plots saved to {figures_dir}/ directory")
    else:
        print("No valid winter data available for plotting")


def calculate_ice_load(ds1, dates, create_figures=True):
    """
    Calculate ice load and create representative figures
    
    Parameters:
    -----------
    ds1 : xarray.Dataset
        Dataset containing ice accretion and ablation data
    dates : pandas.DatetimeIndex
        Date range for winter seasons
    create_figures : bool
        Whether to create visualization figures (default: True)
    
    Returns:
    --------
    str
        Path to the figures directory containing all generated plots
    """
    
    dsiceload = xr.zeros_like(ds1['ACCRE_CYL'].isel(height=0)) * np.nan
    for idate,date in enumerate(dates[:-1]):
        print(f"Processing winter {idate+1}/{len(dates)-1}: {date} to {dates[idate+1]-pd.to_timedelta('30min')}")
        load = ice_load(ds1['ACCRE_CYL'].isel(height=0).sel(time=slice(date,dates[idate+1]-pd.to_timedelta('30min'))).load(),\
                        ds1['ABLAT_CYL'].isel(height=0).sel(time=slice(date,dates[idate+1]-pd.to_timedelta('30min'))).load(),5)
        dsiceload.loc[{'time':load.time}] = load
        print(f"Winter {idate+1} completed")
    
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
    
    # 6. Spatial gradient analysis
    print("Creating spatial gradient analysis...")
    
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
    
    spatial_grad_filename = os.path.join(figures_dir, 'ice_load_spatial_gradients.png')
    plt.savefig(spatial_grad_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {spatial_grad_filename}")
    plt.close()
    
    # 6.5. Mean spatial gradient analysis over all time periods
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
    
    mean_spatial_grad_filename = os.path.join(figures_dir, 'ice_load_mean_spatial_gradients.png')
    plt.savefig(mean_spatial_grad_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {mean_spatial_grad_filename}")
    plt.close()
    
    # 7. Temporal gradient analysis
    print("Creating temporal gradient analysis...")
    
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
    
    temporal_grad_filename = os.path.join(figures_dir, 'ice_load_temporal_gradients.png')
    plt.savefig(temporal_grad_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {temporal_grad_filename}")
    plt.close()
    
    # Print summary statistics
    print(f"\n=== Ice Load Analysis Summary ===")
    print(f"Data shape: {dsiceload.shape}")
    print(f"Valid data points: {len(ice_values):,}")
    print(f"Non-zero data points: {len(ice_values_no_zero):,}")
    print(f"Maximum ice load: {float(ice_load_clean.max().values):.3f} kg/m")
    print(f"Average ice load: {float(ice_load_clean.mean().values):.3f} kg/m")
    
    # Gradient statistics
    print(f"\n=== Gradient Analysis Summary ===")
    print(f"Max spatial gradient magnitude (from max ice load): {gradient_magnitude.max():.3f}")
    print(f"Mean spatial gradient magnitude (from max ice load): {gradient_magnitude.mean():.3f}")
    print(f"Max mean spatial gradient magnitude (over time): {np.nanmax(mean_gradient_magnitude):.3f}")
    print(f"Mean spatial gradient magnitude (over time): {np.nanmean(mean_gradient_magnitude):.3f}")
    print(f"Max temporal gradient: {all_temp_gradients.max():.3f} kg/m per 30min")
    print(f"Min temporal gradient: {all_temp_gradients.min():.3f} kg/m per 30min")
    
    print(f"\nFigures saved to: {figures_dir}/")
    
    return figures_dir

