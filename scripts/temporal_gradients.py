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
    axes[0,0].set_title('Spatial Gradient Magnitude Evolution', fontsize=28)
    axes[0,0].set_xlabel('Time', fontsize=24)
    axes[0,0].set_ylabel('Gradient Magnitude (kg/m per grid)', fontsize=24)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Directional gradient evolution (West-East)
    axes[0,1].plot(time_pd, gradient_x_mean, color='orange', alpha=0.8)
    axes[0,1].set_title('Mean West-East Gradient Evolution', fontsize=28)
    axes[0,1].set_xlabel('Time', fontsize=24)
    axes[0,1].set_ylabel('West-East Gradient (kg/m per grid)', fontsize=24)
    axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Directional gradient evolution (South-North)
    axes[1,0].plot(time_pd, gradient_y_mean, color='green', alpha=0.8)
    axes[1,0].set_title('Mean South-North Gradient Evolution', fontsize=28)
    axes[1,0].set_xlabel('Time', fontsize=24)
    axes[1,0].set_ylabel('South-North Gradient (kg/m per grid)', fontsize=24)
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Gradient variability over time
    axes[1,1].plot(time_pd, gradient_magnitude_std, color='purple', alpha=0.8)
    axes[1,1].set_title('Spatial Gradient Variability Over Time', fontsize=28)
    axes[1,1].set_xlabel('Time', fontsize=24)
    axes[1,1].set_ylabel('Gradient Std Dev (kg/m per grid)', fontsize=24)
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
    axes[0,0].set_title('Average Gradient Magnitude by Hour of Day', fontsize=28)
    axes[0,0].set_xlabel('Hour of Day', fontsize=24)
    axes[0,0].set_ylabel('Mean Gradient Magnitude', fontsize=24)
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
    axes[0,1].set_title('Average Gradient Magnitude by Month', fontsize=28)
    axes[0,1].set_xlabel('Month', fontsize=24)
    axes[0,1].set_ylabel('Mean Gradient Magnitude', fontsize=24)
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
    axes[1,0].set_title('Average West-East Gradient by Hour of Day', fontsize=28)
    axes[1,0].set_xlabel('Hour of Day', fontsize=24)
    axes[1,0].set_ylabel('Mean West-East Gradient', fontsize=24)
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
    axes[1,1].set_title('Average South-North Gradient by Hour of Day', fontsize=28)
    axes[1,1].set_xlabel('Hour of Day', fontsize=24)
    axes[1,1].set_ylabel('Mean South-North Gradient', fontsize=24)
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
    axes[0,0].set_title('Mean Temporal Gradient (Rate of Change)', fontsize=28)
    axes[0,0].set_xlabel('Time', fontsize=24)
    axes[0,0].set_ylabel('Mean Rate of Change (kg/m per 30min)', fontsize=24)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Standard deviation of temporal gradient
    axes[0,1].plot(temp_grad_std.time, temp_grad_std.values, color='orange')
    axes[0,1].set_title('Temporal Gradient Variability', fontsize=28)
    axes[0,1].set_xlabel('Time', fontsize=24)
    axes[0,1].set_ylabel('Std Dev of Rate of Change (kg/m per 30min)', fontsize=24)
    axes[0,1].grid(True, alpha=0.3)
    
    # Maximum and minimum temporal gradients
    axes[1,0].plot(temp_grad_max.time, temp_grad_max.values, label='Maximum', color='red')
    axes[1,0].plot(temp_grad_min.time, temp_grad_min.values, label='Minimum', color='blue')
    axes[1,0].set_title('Extreme Temporal Gradients', fontsize=28)
    axes[1,0].set_xlabel('Time', fontsize=24)
    axes[1,0].set_ylabel('Rate of Change (kg/m per 30min)', fontsize=24)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Histogram of all temporal gradients
    all_temp_gradients = temporal_gradient.values.flatten()
    all_temp_gradients = all_temp_gradients[~np.isnan(all_temp_gradients)]
    
    axes[1,1].hist(all_temp_gradients, bins=50, alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Distribution of Temporal Gradients', fontsize=28)
    axes[1,1].set_xlabel('Rate of Change (kg/m per 30min)', fontsize=24)
    axes[1,1].set_ylabel('Frequency', fontsize=24)
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

def ice_load_resampling_analysis(
    dataset_with_ice_load,
    ice_load_variable='ICE_LOAD',
    height_level=0,
    resampling_years=5,
    save_plots=True,
    results_subdir="ice_load_resampling_analysis",
    months=None,
    ice_load_threshold=0.0,
    OffOn=None,
    BigDomain=False
):
    """
    Perform resampling analysis of ice load data over specified year intervals.
    
    This function aggregates data into consistent time intervals, defines "typical" 
    long-term averages, and expresses each year as a deviation from the average.
    Creates comprehensive visualizations showing temporal patterns and variability.
    
    Parameters:
    -----------
    dataset_with_ice_load : xarray.Dataset
        Dataset that already contains ice load as a variable
    ice_load_variable : str, optional
        Name of the ice load variable in the dataset (default: 'ICE_LOAD')
    height_level : int, optional
        Height level index to use for analysis (default: 0)
    resampling_years : int, optional
        Number of years for resampling intervals (default: 5)
    save_plots : bool, optional
        Whether to save plots to files (default: True)
    results_subdir : str, optional
        Subdirectory name for saving results (default: "ice_load_resampling_analysis")
    months : list, optional
        List of months to include (e.g., [12,1,2,3] for winter)
    ice_load_threshold : float, optional
        Minimum ice load threshold for analysis (default: 0.0)
    OffOn : str, optional
        Specifies 'Onshore' or 'Offshore' for BigDomain directory structure
    BigDomain : bool, default False
        If True, saves results to MT_Icing/results/figures/BigDomain/{OffOn}/Temporal_Gradients...
        
    Returns:
    --------
    dict
        Comprehensive results including time series, statistics, and deviation analysis
    """
    
    print("=== ICE LOAD RESAMPLING ANALYSIS ===")
    print(f"Resampling interval: {resampling_years} years")
    
    # Check if ice load variable exists
    if ice_load_variable not in dataset_with_ice_load.data_vars:
        raise ValueError(f"Ice load variable '{ice_load_variable}' not found in dataset. "
                        f"Available variables: {list(dataset_with_ice_load.data_vars.keys())}")
    
    # Create results directory with characteristics in folder name
    if save_plots:
        # Create organized directory structure
        if BigDomain and OffOn:
            temporal_base_dir = os.path.join("results", "figures", "BigDomain", OffOn, "Temporal_Gradients")
        else:
            temporal_base_dir = os.path.join("results", "figures", "Temporal_Gradients")
        os.makedirs(temporal_base_dir, exist_ok=True)
        
        # Generate folder name based on analysis characteristics
        folder_name_parts = []
        
        # Add resampling years
        folder_name_parts.append(f"resample_{resampling_years}years")
        
        # Add height level info
        height_value = dataset_with_ice_load.height.values[height_level]
        folder_name_parts.append(f"h{height_level}_{height_value:.0f}m")
        
        # Add ice load threshold
        folder_name_parts.append(f"threshold_{ice_load_threshold:.1f}")
        
        # Add month filtering if specified
        if months is not None:
            months_str = "_".join(map(str, sorted(months)))
            folder_name_parts.append(f"months_{months_str}")
        
        # Add timestamp to ensure uniqueness
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        folder_name_parts.append(timestamp)
        
        # Create unique folder name
        folder_name = "_".join(folder_name_parts)
        
        base_results_dir = os.path.join(temporal_base_dir, folder_name)
        os.makedirs(base_results_dir, exist_ok=True)
        print(f"Results will be saved to: {base_results_dir}")
    
    # Get ice load data
    ice_load_data = dataset_with_ice_load[ice_load_variable].isel(height=height_level)
    print(f"Using height level {height_level}: {dataset_with_ice_load.height.values[height_level]} m")
    
    # Apply month filtering if specified
    if months is not None:
        print(f"Filtering data to months: {months}")
        time_df = pd.to_datetime(ice_load_data.time.values)
        month_mask = time_df.month.isin(months)
        if month_mask.any():
            ice_load_data = ice_load_data.isel(time=month_mask)
            print(f"Time steps after month filtering: {len(ice_load_data.time)}")
        else:
            print("Warning: No data found for specified months")
            return None
    
    # Apply threshold filtering
    ice_load_clean = ice_load_data.where(ice_load_data >= ice_load_threshold, np.nan)
    
    # Get time information
    time_index = pd.to_datetime(ice_load_clean.time.values)
    start_year = time_index.year.min()
    end_year = time_index.year.max()
    total_years = end_year - start_year + 1
    
    print(f"\nData Overview:")
    print(f"Time range: {start_year} to {end_year} ({total_years} years)")
    print(f"Grid size: {ice_load_clean.sizes['south_north']} × {ice_load_clean.sizes['west_east']}")
    print(f"Ice load threshold: {ice_load_threshold} kg/m")
    
    # 1. YEARLY STATISTICS
    print(f"\n1. CALCULATING YEARLY STATISTICS")
    print("=" * 35)
    
    yearly_stats = {}
    yearly_max = []
    yearly_mean = []
    yearly_percentiles = {'p90': [], 'p95': [], 'p99': []}
    years_list = []
    
    for year in range(start_year, end_year + 1):
        year_mask = time_index.year == year
        if year_mask.any():
            year_data = ice_load_clean.isel(time=year_mask)
            
            # Calculate spatial statistics for this year
            year_values = year_data.values.flatten()
            year_values_clean = year_values[~np.isnan(year_values)]
            
            if len(year_values_clean) > 0:
                stats = {
                    'max': float(np.max(year_values_clean)),
                    'mean': float(np.mean(year_values_clean)),
                    'std': float(np.std(year_values_clean)),
                    'p90': float(np.percentile(year_values_clean, 90)),
                    'p95': float(np.percentile(year_values_clean, 95)),
                    'p99': float(np.percentile(year_values_clean, 99)),
                    'n_points': len(year_values_clean)
                }
                
                yearly_stats[year] = stats
                years_list.append(year)
                yearly_max.append(stats['max'])
                yearly_mean.append(stats['mean'])
                yearly_percentiles['p90'].append(stats['p90'])
                yearly_percentiles['p95'].append(stats['p95'])
                yearly_percentiles['p99'].append(stats['p99'])
    
    # 2. RESAMPLING ANALYSIS
    print(f"\n2. PERFORMING {resampling_years}-YEAR RESAMPLING ANALYSIS")
    print("=" * 50)
    
    # Create resampling periods
    resampling_periods = []
    for start in range(start_year, end_year + 1, resampling_years):
        end = min(start + resampling_years - 1, end_year)
        if end - start + 1 >= resampling_years:  # Only include complete periods
            resampling_periods.append((start, end))
    
    print(f"Resampling periods ({len(resampling_periods)}):")
    for start, end in resampling_periods:
        print(f"  {start}-{end}")
    
    # Calculate statistics for each resampling period
    period_stats = {}
    period_means = []
    period_stds = []
    period_labels = []
    
    for start, end in resampling_periods:
        period_mask = (time_index.year >= start) & (time_index.year <= end)
        if period_mask.any():
            period_data = ice_load_clean.isel(time=period_mask)
            
            # Calculate statistics for this period
            period_values = period_data.values.flatten()
            period_values_clean = period_values[~np.isnan(period_values)]
            
            if len(period_values_clean) > 0:
                stats = {
                    'mean': float(np.mean(period_values_clean)),
                    'std': float(np.std(period_values_clean)),
                    'max': float(np.max(period_values_clean)),
                    'p95': float(np.percentile(period_values_clean, 95)),
                    'n_points': len(period_values_clean)
                }
                
                period_key = f"{start}-{end}"
                period_stats[period_key] = stats
                period_means.append(stats['mean'])
                period_stds.append(stats['std'])
                period_labels.append(period_key)
    
    # 3. LONG-TERM AVERAGE AND DEVIATIONS
    print(f"\n3. CALCULATING LONG-TERM AVERAGE AND DEVIATIONS")
    print("=" * 50)
    
    # Calculate overall long-term average
    all_values = ice_load_clean.values.flatten()
    all_values_clean = all_values[~np.isnan(all_values)]
    
    long_term_stats = {
        'mean': float(np.mean(all_values_clean)),
        'std': float(np.std(all_values_clean)),
        'max': float(np.max(all_values_clean)),
        'p95': float(np.percentile(all_values_clean, 95))
    }
    
    print(f"Long-term average: {long_term_stats['mean']:.4f} kg/m")
    print(f"Long-term std dev: {long_term_stats['std']:.4f} kg/m")
    
    # Calculate yearly deviations from long-term average
    yearly_deviations = []
    yearly_normalized_deviations = []
    
    for year in years_list:
        if year in yearly_stats:
            deviation = yearly_stats[year]['mean'] - long_term_stats['mean']
            normalized_deviation = deviation / long_term_stats['std']
            yearly_deviations.append(deviation)
            yearly_normalized_deviations.append(normalized_deviation)
        else:
            yearly_deviations.append(np.nan)
            yearly_normalized_deviations.append(np.nan)
    
    # 4. CREATE VISUALIZATIONS
    print(f"\n4. CREATING VISUALIZATIONS")
    print("=" * 30)
    
    # Plot 1a: Yearly mean ice load
    fig1a = plt.figure(figsize=(10, 6))
    plt.plot(years_list, yearly_mean, 'b-o', linewidth=2, markersize=4, label='Yearly Mean')
    plt.axhline(y=long_term_stats['mean'], color='red', linestyle='--', linewidth=2, 
                label=f'Long-term Average ({long_term_stats["mean"]:.3f})')
    plt.fill_between(years_list, 
                     [long_term_stats['mean'] - long_term_stats['std']] * len(years_list),
                     [long_term_stats['mean'] + long_term_stats['std']] * len(years_list),
                     alpha=0.2, color='red', label='±1 Std Dev')
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Mean Ice Load (kg/m)', fontsize=20)
    plt.title('Yearly Mean Ice Load vs Long-term Average', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()
    
    if save_plots:
        plot1a_path = os.path.join(base_results_dir, "yearly_mean_ice_load.png")
        plt.savefig(plot1a_path, dpi=300, bbox_inches='tight')
        print(f"Yearly mean ice load plot saved to: {plot1a_path}")
    plt.close(fig1a)
    
    # Plot 1b: Yearly maximum ice load
    fig1b = plt.figure(figsize=(10, 6))
    plt.plot(years_list, yearly_max, 'g-s', linewidth=2, markersize=4, label='Yearly Maximum')
    plt.axhline(y=long_term_stats['max'], color='orange', linestyle='--', linewidth=2,
                label=f'Long-term Max ({long_term_stats["max"]:.3f})')
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Maximum Ice Load (kg/m)', fontsize=20)
    plt.title('Yearly Maximum Ice Load', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()
    
    if save_plots:
        plot1b_path = os.path.join(base_results_dir, "yearly_maximum_ice_load.png")
        plt.savefig(plot1b_path, dpi=300, bbox_inches='tight')
        print(f"Yearly maximum ice load plot saved to: {plot1b_path}")
    plt.close(fig1b)
    
    # Plot 1c: Yearly deviations from long-term average
    fig1c = plt.figure(figsize=(10, 6))
    colors = ['red' if x > 0 else 'blue' for x in yearly_deviations]
    plt.bar(years_list, yearly_deviations, color=colors, alpha=0.7, width=0.8)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Deviation from Long-term Mean (kg/m)', fontsize=20)
    plt.title('Yearly Deviations from Long-term Average', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plots:
        plot1c_path = os.path.join(base_results_dir, "yearly_deviations.png")
        plt.savefig(plot1c_path, dpi=300, bbox_inches='tight')
        print(f"Yearly deviations plot saved to: {plot1c_path}")
    plt.close(fig1c)
    
    # Plot 1d: Normalized deviations
    fig1d = plt.figure(figsize=(10, 6))
    colors_norm = ['red' if x > 0 else 'blue' for x in yearly_normalized_deviations]
    plt.bar(years_list, yearly_normalized_deviations, color=colors_norm, alpha=0.7, width=0.8)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='+1 Std Dev')
    plt.axhline(y=-1, color='red', linestyle='--', alpha=0.7, label='-1 Std Dev')
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Normalized Deviation (σ)', fontsize=20)
    plt.title('Normalized Yearly Deviations', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()
    
    if save_plots:
        plot1d_path = os.path.join(base_results_dir, "yearly_normalized_deviations.png")
        plt.savefig(plot1d_path, dpi=300, bbox_inches='tight')
        print(f"Yearly normalized deviations plot saved to: {plot1d_path}")
    plt.close(fig1d)
    
    # Plot 2: Resampling period analysis
    if len(period_means) > 1:
        # Convert period labels from "2020-2025" to "20-25" format, or "2022-2022" to "22"
        period_labels_short = []
        for label in period_labels:
            years = label.split('-')
            short_years = [year[-2:] for year in years]
            if len(short_years) == 2 and short_years[0] == short_years[1]:
                period_labels_short.append(short_years[0])
            else:
                period_labels_short.append('-'.join(short_years))
        
        # Plot 2a: Period means
        fig2a = plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(period_labels))
        plt.bar(x_pos, period_means, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.axhline(y=long_term_stats['mean'], color='red', linestyle='--', linewidth=2,
                    label=f'Overall Mean ({long_term_stats["mean"]:.3f})')
        plt.xlabel(f'{resampling_years}-Year Periods', fontsize=20)
        plt.ylabel('Mean Ice Load (kg/m)', fontsize=20)
        plt.title(f'{resampling_years}-Year Period Means', fontsize=22)
        plt.xticks(x_pos, period_labels_short, rotation=45, fontsize=16)
        plt.yticks(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=18)
        plt.tight_layout()
        
        if save_plots:
            plot2a_path = os.path.join(base_results_dir, f"{resampling_years}year_period_means.png")
            plt.savefig(plot2a_path, dpi=300, bbox_inches='tight')
            print(f"Period means plot saved to: {plot2a_path}")
        plt.close(fig2a)
        
        # Plot 2b: Period standard deviations
        fig2b = plt.figure(figsize=(10, 6))
        plt.bar(x_pos, period_stds, alpha=0.7, color='lightcoral', edgecolor='darkred')
        plt.axhline(y=long_term_stats['std'], color='blue', linestyle='--', linewidth=2,
                    label=f'Overall Std ({long_term_stats["std"]:.3f})')
        plt.xlabel(f'{resampling_years}-Year Periods', fontsize=20)
        plt.ylabel('Standard Deviation (kg/m)', fontsize=20)
        plt.title(f'{resampling_years}-Year Period Variability', fontsize=22)
        plt.xticks(x_pos, period_labels_short, rotation=45, fontsize=16)
        plt.yticks(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=18)
        plt.tight_layout()
        
        if save_plots:
            plot2b_path = os.path.join(base_results_dir, f"{resampling_years}year_period_variability.png")
            plt.savefig(plot2b_path, dpi=300, bbox_inches='tight')
            print(f"Period variability plot saved to: {plot2b_path}")
        plt.close(fig2b)
        
        # Plot 2c: Period deviations from overall mean
        fig2c = plt.figure(figsize=(10, 6))
        period_deviations = [mean - long_term_stats['mean'] for mean in period_means]
        colors_period = ['red' if x > 0 else 'blue' for x in period_deviations]
        plt.bar(x_pos, period_deviations, color=colors_period, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.xlabel(f'{resampling_years}-Year Periods', fontsize=20)
        plt.ylabel('Deviation from Overall Mean (kg/m)', fontsize=20)
        plt.title(f'{resampling_years}-Year Period Deviations', fontsize=22)
        plt.xticks(x_pos, period_labels_short, rotation=45, fontsize=16)
        plt.yticks(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plot2c_path = os.path.join(base_results_dir, f"{resampling_years}year_period_deviations.png")
            plt.savefig(plot2c_path, dpi=300, bbox_inches='tight')
            print(f"Period deviations plot saved to: {plot2c_path}")
        plt.close(fig2c)
        
        # Plot 2d: Coefficient of variation
        fig2d = plt.figure(figsize=(10, 6))
        cv_values = [std/mean if mean > 0 else 0 for mean, std in zip(period_means, period_stds)]
        overall_cv = long_term_stats['std'] / long_term_stats['mean']
        
        plt.bar(x_pos, cv_values, alpha=0.7, color='gold', edgecolor='orange')
        plt.axhline(y=overall_cv, color='purple', linestyle='--', linewidth=2,
                    label=f'Overall CV ({overall_cv:.3f})')
        plt.xlabel(f'{resampling_years}-Year Periods', fontsize=20)
        plt.ylabel('Coefficient of Variation', fontsize=20)
        plt.title(f'{resampling_years}-Year Period Relative Variability', fontsize=22)
        plt.xticks(x_pos, period_labels_short, rotation=45, fontsize=16)
        plt.yticks(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=18)
        plt.tight_layout()
        
        if save_plots:
            plot2d_path = os.path.join(base_results_dir, f"{resampling_years}year_period_coefficient_variation.png")
            plt.savefig(plot2d_path, dpi=300, bbox_inches='tight')
            print(f"Period coefficient of variation plot saved to: {plot2d_path}")
        plt.close(fig2d)
    
    # Plot 3a: Percentiles evolution over time
    fig3a = plt.figure(figsize=(12, 6))
    plt.plot(years_list, yearly_percentiles['p90'], 'g-o', linewidth=2, markersize=3, label='90th Percentile')
    plt.plot(years_list, yearly_percentiles['p95'], 'orange', marker='s', linewidth=2, markersize=3, label='95th Percentile')
    plt.plot(years_list, yearly_percentiles['p99'], 'r-^', linewidth=2, markersize=3, label='99th Percentile')
    plt.plot(years_list, yearly_mean, 'b-', linewidth=2, alpha=0.7, label='Mean')
    
    # Add long-term averages
    plt.axhline(y=np.percentile(all_values_clean, 90), color='green', linestyle='--', alpha=0.7)
    plt.axhline(y=np.percentile(all_values_clean, 95), color='orange', linestyle='--', alpha=0.7)
    plt.axhline(y=np.percentile(all_values_clean, 99), color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=long_term_stats['mean'], color='blue', linestyle='--', alpha=0.7)
    
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Ice Load (kg/m)', fontsize=20)
    plt.title('Ice Load Percentiles Evolution Over Time', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()
    
    if save_plots:
        plot3a_path = os.path.join(base_results_dir, "percentiles_evolution.png")
        plt.savefig(plot3a_path, dpi=300, bbox_inches='tight')
        print(f"Percentiles evolution plot saved to: {plot3a_path}")
    plt.close(fig3a)
    
    # Plot 3b: Distribution of yearly statistics
    fig3b = plt.figure(figsize=(10, 6))
    bp = plt.boxplot([yearly_mean, yearly_percentiles['p90'], yearly_percentiles['p95'], yearly_percentiles['p99']], 
                      labels=['Mean', 'P90', 'P95', 'P99'],
                      patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
    plt.ylabel('Ice Load (kg/m)', fontsize=20)
    plt.title('Distribution of Yearly Statistics', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plots:
        plot3b_path = os.path.join(base_results_dir, "yearly_statistics_distribution.png")
        plt.savefig(plot3b_path, dpi=300, bbox_inches='tight')
        print(f"Yearly statistics distribution plot saved to: {plot3b_path}")
    plt.close(fig3b)
    
    # 5. SAVE SUMMARY REPORT
    if save_plots:
        summary_path = os.path.join(base_results_dir, "resampling_analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("ICE LOAD RESAMPLING ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis timestamp: {pd.Timestamp.now()}\n")
            f.write(f"Resampling interval: {resampling_years} years\n")
            f.write(f"Height level: {height_level} ({dataset_with_ice_load.height.values[height_level]} m)\n")
            f.write(f"Ice load threshold: {ice_load_threshold} kg/m\n")
            if months:
                f.write(f"Months included: {months}\n")
            f.write(f"Time range: {start_year} to {end_year} ({total_years} years)\n\n")
            
            f.write("LONG-TERM STATISTICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Mean: {long_term_stats['mean']:.4f} kg/m\n")
            f.write(f"Std Dev: {long_term_stats['std']:.4f} kg/m\n")
            f.write(f"Maximum: {long_term_stats['max']:.4f} kg/m\n")
            f.write(f"95th Percentile: {long_term_stats['p95']:.4f} kg/m\n\n")
            
            f.write("YEARLY STATISTICS SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean of yearly means: {np.mean(yearly_mean):.4f} kg/m\n")
            f.write(f"Std of yearly means: {np.std(yearly_mean):.4f} kg/m\n")
            f.write(f"Max yearly mean: {np.max(yearly_mean):.4f} kg/m ({years_list[np.argmax(yearly_mean)]})\n")
            f.write(f"Min yearly mean: {np.min(yearly_mean):.4f} kg/m ({years_list[np.argmin(yearly_mean)]})\n\n")
            
            if len(period_means) > 1:
                f.write(f"{resampling_years}-YEAR PERIOD STATISTICS:\n")
                f.write("-" * 35 + "\n")
                for i, (label, mean, std) in enumerate(zip(period_labels, period_means, period_stds)):
                    deviation = mean - long_term_stats['mean']
                    f.write(f"{label}: Mean={mean:.4f}, Std={std:.4f}, Dev={deviation:+.4f} kg/m\n")
            
            f.write(f"\nFILES GENERATED:\n")
            f.write("-" * 20 + "\n")
            f.write("- yearly_mean_ice_load.png\n")
            f.write("- yearly_maximum_ice_load.png\n")
            f.write("- yearly_deviations.png\n")
            f.write("- yearly_normalized_deviations.png\n")
            if len(period_means) > 1:
                f.write(f"- {resampling_years}year_period_means.png\n")
                f.write(f"- {resampling_years}year_period_variability.png\n")
                f.write(f"- {resampling_years}year_period_deviations.png\n")
                f.write(f"- {resampling_years}year_period_coefficient_variation.png\n")
            f.write("- percentiles_evolution.png\n")
            f.write("- yearly_statistics_distribution.png\n")
            f.write("- resampling_analysis_summary.txt (this file)\n")
        
        print(f"Summary report saved to: {summary_path}")
    
    # Compile results
    results = {
        'long_term_stats': long_term_stats,
        'yearly_stats': yearly_stats,
        'yearly_deviations': yearly_deviations,
        'yearly_normalized_deviations': yearly_normalized_deviations,
        'period_stats': period_stats,
        'resampling_years': resampling_years,
        'years_list': years_list,
        'time_range': (start_year, end_year),
        'resampling_periods': resampling_periods
    }
    
    print(f"\n=== RESAMPLING ANALYSIS COMPLETED ===")
    print(f"Analyzed {total_years} years of data")
    print(f"Long-term mean: {long_term_stats['mean']:.4f} kg/m")
    print(f"Yearly variability: {np.std(yearly_mean):.4f} kg/m")
    if len(period_means) > 1:
        print(f"Number of {resampling_years}-year periods: {len(period_means)}")
    
    return results

def ice_load_resampling_analysis_hours(
    dataset_with_ice_load,
    ice_load_variable='ICE_LOAD',
    height_level=0,
    resampling_years=5,
    save_plots=True,
    results_subdir="ice_load_resampling_analysis_hours",
    months=None,
    ice_load_threshold=0.1,
    OffOn=None,
    BigDomain=False
):
    """
    Perform resampling analysis of ice load exceedance hours over specified year intervals.
    
    This function counts hours where ice load exceeds a threshold, aggregates into time intervals,
    and analyzes temporal patterns of exceedance frequency. Creates comprehensive visualizations
    showing exceedance hours patterns and variability over time.
    
    Parameters:
    -----------
    dataset_with_ice_load : xarray.Dataset
        Dataset that already contains ice load as a variable
    ice_load_variable : str, optional
        Name of the ice load variable in the dataset (default: 'ICE_LOAD')
    height_level : int, optional
        Height level index to use for analysis (default: 0)
    resampling_years : int, optional
        Number of years for resampling intervals (default: 5)
    save_plots : bool, optional
        Whether to save plots to files (default: True)
    results_subdir : str, optional
        Subdirectory name for saving results (default: "ice_load_resampling_analysis_hours")
    months : list, optional
        List of months to include (e.g., [12,1,2,3] for winter)
    ice_load_threshold : float, optional
        Ice load threshold for exceedance analysis (default: 0.1 kg/m)
    OffOn : str, optional
        Specifies 'Onshore' or 'Offshore' for BigDomain directory structure
    BigDomain : bool, default False
        If True, saves results to MT_Icing/results/figures/BigDomain/{OffOn}/Temporal_Gradients/hours...
        
    Returns:
    --------
    dict
        Comprehensive results including time series, statistics, and exceedance analysis
    """
    
    print("=== ICE LOAD EXCEEDANCE HOURS RESAMPLING ANALYSIS ===")
    print(f"Resampling interval: {resampling_years} years")
    print(f"Ice load threshold: {ice_load_threshold} kg/m")
    
    # Check if ice load variable exists
    if ice_load_variable not in dataset_with_ice_load.data_vars:
        raise ValueError(f"Ice load variable '{ice_load_variable}' not found in dataset. "
                        f"Available variables: {list(dataset_with_ice_load.data_vars.keys())}")
    
    # Create results directory with characteristics in folder name
    if save_plots:
        # Create organized directory structure
        if BigDomain and OffOn:
            temporal_base_dir = os.path.join("results", "figures", "BigDomain", OffOn, "Temporal_Gradients", "hours")
        else:
            temporal_base_dir = os.path.join("results", "figures", "Temporal_Gradients", "hours")
        os.makedirs(temporal_base_dir, exist_ok=True)
        
        # Generate folder name based on analysis characteristics
        folder_name_parts = []
        
        # Add resampling years
        folder_name_parts.append(f"resample_{resampling_years}years")
        
        # Add height level info
        height_value = dataset_with_ice_load.height.values[height_level]
        folder_name_parts.append(f"h{height_level}_{height_value:.0f}m")
        
        # Add ice load threshold
        folder_name_parts.append(f"threshold_{ice_load_threshold:.1f}")
        
        # Add month filtering if specified
        if months is not None:
            months_str = "_".join(map(str, sorted(months)))
            folder_name_parts.append(f"months_{months_str}")
        
        # Add timestamp to ensure uniqueness
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        folder_name_parts.append(timestamp)
        
        # Create unique folder name
        folder_name = "_".join(folder_name_parts)
        
        base_results_dir = os.path.join(temporal_base_dir, folder_name)
        os.makedirs(base_results_dir, exist_ok=True)
        print(f"Results will be saved to: {base_results_dir}")
    
    # Get ice load data
    ice_load_data = dataset_with_ice_load[ice_load_variable].isel(height=height_level)
    print(f"Using height level {height_level}: {dataset_with_ice_load.height.values[height_level]} m")
    
    # Apply month filtering if specified
    if months is not None:
        print(f"Filtering data to months: {months}")
        time_df = pd.to_datetime(ice_load_data.time.values)
        month_mask = time_df.month.isin(months)
        if month_mask.any():
            ice_load_data = ice_load_data.isel(time=month_mask)
            print(f"Time steps after month filtering: {len(ice_load_data.time)}")
        else:
            print("Warning: No data found for specified months")
            return None
    
    # Calculate exceedance mask (True where ice load > threshold)
    exceedance_mask = ice_load_data >= ice_load_threshold
    
    # Get time information
    time_index = pd.to_datetime(ice_load_data.time.values)
    start_year = time_index.year.min()
    end_year = time_index.year.max()
    
    # For winter years: June Year1 to May Year2 is "Winter Year2"
    # Adjust start and end years for winter year calculation
    winter_start_year = start_year
    winter_end_year = end_year
    
    # Check if we have data starting in June or later for proper winter year definition
    first_month = time_index.month.min()
    last_month = time_index.month.max()
    
    # If data starts after June, the first winter year is the following year
    if time_index[time_index.year == start_year].month.min() > 6:
        winter_start_year = start_year + 1
    
    # If data ends before June, the last winter year is the previous year
    if time_index[time_index.year == end_year].month.max() < 6:
        winter_end_year = end_year - 1
    
    total_winter_years = winter_end_year - winter_start_year + 1
    
    time_step_hours = 0.5  # Default to 30 minutes
    
    print(f"\nData Overview:")
    print(f"Calendar time range: {start_year} to {end_year}")
    print(f"Winter years range: {winter_start_year} to {winter_end_year} ({total_winter_years} winter years)")
    print(f"Grid size: {ice_load_data.sizes['south_north']} × {ice_load_data.sizes['west_east']}")
    print(f"Time step duration: {time_step_hours:.1f} hours")
    print(f"Ice load threshold: {ice_load_threshold} kg/m")
    print(f"Note: Winter Year N = June Year(N-1) to May Year(N)")
    
    # 1. YEARLY EXCEEDANCE HOURS STATISTICS
    print(f"\n1. CALCULATING YEARLY EXCEEDANCE HOURS STATISTICS")
    print("=" * 50)
    
    yearly_stats = {}
    yearly_total_hours = []
    yearly_max_hours = []
    yearly_mean_hours = []
    yearly_percentiles = {'p75': [], 'p90': [], 'p95': []}
    years_list = []
    
    for winter_year in range(winter_start_year, winter_end_year + 1):
        # Define winter year mask: June (year-1) to May (year)
        winter_start = pd.Timestamp(f"{winter_year-1}-06-01")
        winter_end = pd.Timestamp(f"{winter_year}-05-31 23:59:59")
        
        # Create mask for this winter year
        winter_mask = (time_index >= winter_start) & (time_index <= winter_end)
        
        if winter_mask.any():
            # Get exceedance data for this winter year
            winter_exceedance = exceedance_mask.isel(time=winter_mask)
            
            # Calculate grid mean exceedance for each time step, then sum over time
            # This gives the total exceedance hours for the grid mean
            grid_mean_exceedance = winter_exceedance.mean(dim=['south_north', 'west_east'])
            total_grid_mean_exceedance_hours = float(grid_mean_exceedance.sum() * time_step_hours)
            
            # Also calculate exceedance hours per individual grid cell for spatial statistics
            exceedance_hours_per_cell = winter_exceedance.sum(dim='time') * time_step_hours
            exceedance_hours_values = exceedance_hours_per_cell.values.flatten()
            exceedance_hours_clean = exceedance_hours_values[~np.isnan(exceedance_hours_values)]
            
            if len(exceedance_hours_clean) > 0:
                # Statistics per grid cell
                stats = {
                    'grid_mean_total_hours': total_grid_mean_exceedance_hours,  # Total hours for grid mean
                    'max_hours_per_cell': float(np.max(exceedance_hours_clean)),
                    'mean_hours_per_cell': float(np.mean(exceedance_hours_clean)),
                    'std_hours_per_cell': float(np.std(exceedance_hours_clean)),
                    'p75_hours_per_cell': float(np.percentile(exceedance_hours_clean, 75)),
                    'p90_hours_per_cell': float(np.percentile(exceedance_hours_clean, 90)),
                    'p95_hours_per_cell': float(np.percentile(exceedance_hours_clean, 95)),
                    'n_grid_cells': len(exceedance_hours_clean),
                    'cells_with_exceedance': int(np.sum(exceedance_hours_clean > 0)),
                    'winter_start': winter_start,
                    'winter_end': winter_end
                }
                
                yearly_stats[winter_year] = stats
                years_list.append(winter_year)
                yearly_total_hours.append(stats['grid_mean_total_hours'])  # Use grid mean total
                yearly_max_hours.append(stats['max_hours_per_cell'])
                yearly_mean_hours.append(stats['mean_hours_per_cell'])
                yearly_percentiles['p75'].append(stats['p75_hours_per_cell'])
                yearly_percentiles['p90'].append(stats['p90_hours_per_cell'])
                yearly_percentiles['p95'].append(stats['p95_hours_per_cell'])
    
    # 2. RESAMPLING ANALYSIS
    print(f"\n2. PERFORMING {resampling_years}-WINTER-YEAR RESAMPLING ANALYSIS")
    print("=" * 50)
    
    # Create resampling periods using winter years
    resampling_periods = []
    for start in range(winter_start_year, winter_end_year + 1, resampling_years):
        end = min(start + resampling_years - 1, winter_end_year)
        if end - start + 1 >= resampling_years:  # Only include complete periods
            resampling_periods.append((start, end))
    
    print(f"Winter year resampling periods ({len(resampling_periods)}):")
    for start, end in resampling_periods:
        calendar_start = start - 1
        calendar_end = end
        print(f"  Winter {start}-{end} (Jun {calendar_start} to May {calendar_end})")
    
    # Calculate statistics for each resampling period
    period_stats = {}
    period_total_hours = []
    period_mean_hours = []
    period_labels = []
    
    for start, end in resampling_periods:
        # Create mask for winter year range
        period_mask = np.zeros(len(time_index), dtype=bool)
        for winter_year in range(start, end + 1):
            winter_start = pd.Timestamp(f"{winter_year-1}-06-01")
            winter_end = pd.Timestamp(f"{winter_year}-05-31 23:59:59")
            winter_mask = (time_index >= winter_start) & (time_index <= winter_end)
            period_mask = period_mask | winter_mask  # Combine masks (winter_mask is already numpy array)
        
        if period_mask.any():
            # Get exceedance data for this period
            period_exceedance = exceedance_mask.isel(time=period_mask)
            
            # Calculate grid mean exceedance for each time step, then sum over time
            grid_mean_exceedance = period_exceedance.mean(dim=['south_north', 'west_east'])
            total_grid_mean_exceedance_hours = float(grid_mean_exceedance.sum() * time_step_hours)
            
            # Also calculate exceedance hours per individual grid cell for spatial statistics
            exceedance_hours_per_cell = period_exceedance.sum(dim='time') * time_step_hours
            exceedance_hours_values = exceedance_hours_per_cell.values.flatten()
            exceedance_hours_clean = exceedance_hours_values[~np.isnan(exceedance_hours_values)]
            
            if len(exceedance_hours_clean) > 0:
                stats = {
                    'grid_mean_total_hours': total_grid_mean_exceedance_hours,  # Grid mean total
                    'mean_hours_per_cell': float(np.mean(exceedance_hours_clean)),
                    'max_hours_per_cell': float(np.max(exceedance_hours_clean)),
                    'std_hours_per_cell': float(np.std(exceedance_hours_clean)),
                    'p95_hours_per_cell': float(np.percentile(exceedance_hours_clean, 95)),
                    'n_grid_cells': len(exceedance_hours_clean),
                    'cells_with_exceedance': int(np.sum(exceedance_hours_clean > 0)),
                    'years_in_period': end - start + 1
                }
                
                period_key = f"{start-1}-{end}"  # For winter years: show calendar year span
                period_stats[period_key] = stats
                period_total_hours.append(stats['grid_mean_total_hours'])  # Use grid mean total
                period_mean_hours.append(stats['mean_hours_per_cell'])
                period_labels.append(period_key)
    
    # 3. LONG-TERM AVERAGE AND DEVIATIONS
    print(f"\n3. CALCULATING LONG-TERM AVERAGE AND DEVIATIONS")
    print("=" * 50)
    
    # Calculate overall long-term average
    # Calculate grid mean exceedance for entire period
    grid_mean_exceedance_all = exceedance_mask.mean(dim=['south_north', 'west_east'])
    total_grid_mean_exceedance_hours_all = float(grid_mean_exceedance_all.sum() * time_step_hours)
    
    # Also calculate individual cell statistics for spatial analysis
    all_exceedance = exceedance_mask.sum(dim='time') * time_step_hours
    all_exceedance_values = all_exceedance.values.flatten()
    all_exceedance_clean = all_exceedance_values[~np.isnan(all_exceedance_values)]
    
    long_term_stats = {
        'grid_mean_total_hours': total_grid_mean_exceedance_hours_all,  # Grid mean total
        'mean_hours_per_cell': float(np.mean(all_exceedance_clean)),
        'std_hours_per_cell': float(np.std(all_exceedance_clean)),
        'max_hours_per_cell': float(np.max(all_exceedance_clean)),
        'p95_hours_per_cell': float(np.percentile(all_exceedance_clean, 95)),
        'cells_with_exceedance': int(np.sum(all_exceedance_clean > 0)),
        'total_grid_cells': len(all_exceedance_clean)
    }
    
    print(f"Long-term grid mean total exceedance hours: {long_term_stats['grid_mean_total_hours']:.1f} hours")
    print(f"Long-term average hours per cell: {long_term_stats['mean_hours_per_cell']:.2f} hours")
    print(f"Cells with exceedance: {long_term_stats['cells_with_exceedance']} / {long_term_stats['total_grid_cells']}")
    
    # Calculate yearly deviations from long-term average
    yearly_deviations = []
    yearly_normalized_deviations = []
    
    for winter_year in years_list:
        if winter_year in yearly_stats:
            deviation = yearly_stats[winter_year]['mean_hours_per_cell'] - long_term_stats['mean_hours_per_cell']
            if long_term_stats['std_hours_per_cell'] > 0:
                normalized_deviation = deviation / long_term_stats['std_hours_per_cell']
            else:
                normalized_deviation = 0
            yearly_deviations.append(deviation)
            yearly_normalized_deviations.append(normalized_deviation)
        else:
            yearly_deviations.append(np.nan)
            yearly_normalized_deviations.append(np.nan)
    
    # 4. CREATE VISUALIZATIONS
    print(f"\n4. CREATING VISUALIZATIONS")
    print("=" * 30)
    
    # Plot 2: Resampling period analysis (separated into 4 individual plots)
    if len(period_total_hours) > 1:
        # Convert period labels from "2020-2025" to "20-25" format, or "2022-2022" to "22"
        period_labels_short = []
        for label in period_labels:
            years = label.split('-')
            short_years = [year[-2:] for year in years]
            if len(short_years) == 2 and short_years[0] == short_years[1]:
                period_labels_short.append(short_years[0])
            else:
                period_labels_short.append('-'.join(short_years))
        
        x_pos = np.arange(len(period_labels))
        
        # Plot 2a: Period grid mean total hours
        fig2a = plt.figure(figsize=(10, 6))
        plt.bar(x_pos, period_total_hours, alpha=0.7, color='skyblue', edgecolor='navy')
        period_avg = long_term_stats['grid_mean_total_hours'] / total_winter_years * resampling_years
        plt.axhline(y=period_avg, color='red', linestyle='--', linewidth=2,
                    label=f'Expected Total ({period_avg:.1f})')
        plt.xlabel(f'{resampling_years}-Year Periods', fontsize=20)
        plt.ylabel('Grid Mean Total Exceedance Hours', fontsize=20)
        plt.title(f'{resampling_years}-Year Period Grid Mean Total Hours', fontsize=22)
        plt.xticks(x_pos, period_labels_short, rotation=45, fontsize=16)
        plt.yticks(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=18)
        plt.tight_layout()
        
        if save_plots:
            plot2a_path = os.path.join(base_results_dir, f"{resampling_years}year_period_grid_mean_total_hours.png")
            plt.savefig(plot2a_path, dpi=300, bbox_inches='tight')
            print(f"Period grid mean total hours plot saved to: {plot2a_path}")
        plt.close(fig2a)
        
        # Plot 2b: Period mean hours per cell
        fig2b = plt.figure(figsize=(10, 6))
        plt.bar(x_pos, period_mean_hours, alpha=0.7, color='lightcoral', edgecolor='darkred')
        # Calculate mean of all period means (average of the bars)
        mean_of_period_means = np.mean(period_mean_hours)
        plt.axhline(y=mean_of_period_means, color='blue', linestyle='--', linewidth=2,
                    label=f'Mean of Period Means ({mean_of_period_means:.1f})')
        plt.xlabel(f'{resampling_years}-Year Periods', fontsize=20)
        plt.ylabel('Mean Hours per Cell', fontsize=20)
        plt.title(f'{resampling_years}-Year Period Mean Hours per Cell', fontsize=22)
        plt.xticks(x_pos, period_labels_short, rotation=45, fontsize=16)
        plt.yticks(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=18)
        plt.tight_layout()
        
        if save_plots:
            plot2b_path = os.path.join(base_results_dir, f"{resampling_years}year_period_mean_hours_per_cell.png")
            plt.savefig(plot2b_path, dpi=300, bbox_inches='tight')
            print(f"Period mean hours per cell plot saved to: {plot2b_path}")
        plt.close(fig2b)
        
        # Plot 2c: Period deviations from mean of period means
        fig2c = plt.figure(figsize=(10, 6))
        period_deviations = [mean - mean_of_period_means for mean in period_mean_hours]
        colors_period = ['red' if x > 0 else 'blue' for x in period_deviations]
        plt.bar(x_pos, period_deviations, color=colors_period, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.xlabel(f'{resampling_years}-Year Periods', fontsize=20)
        plt.ylabel('Deviation from Period Means Average\n(hours)', fontsize=20)
        plt.title(f'{resampling_years}-Year Period Deviations', fontsize=22)
        plt.xticks(x_pos, period_labels_short, rotation=45, fontsize=16)
        plt.yticks(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plot2c_path = os.path.join(base_results_dir, f"{resampling_years}year_period_deviations.png")
            plt.savefig(plot2c_path, dpi=300, bbox_inches='tight')
            print(f"Period deviations plot saved to: {plot2c_path}")
        plt.close(fig2c)
        
        # Plot 2d: Cells with exceedance percentage
        fig2d = plt.figure(figsize=(10, 6))
        cells_with_exceedance_pct = [(period_stats[label]['cells_with_exceedance'] / 
                                     period_stats[label]['n_grid_cells'] * 100) 
                                    for label in period_labels]
        overall_pct = (long_term_stats['cells_with_exceedance'] / 
                      long_term_stats['total_grid_cells'] * 100)
        
        plt.bar(x_pos, cells_with_exceedance_pct, alpha=0.7, color='gold', edgecolor='orange')
        plt.axhline(y=overall_pct, color='purple', linestyle='--', linewidth=2,
                    label=f'Overall ({overall_pct:.1f}%)')
        plt.xlabel(f'{resampling_years}-Year Periods', fontsize=20)
        plt.ylabel('Cells with Exceedance (%)', fontsize=20)
        plt.title(f'{resampling_years}-Year Period Spatial Coverage', fontsize=22)
        plt.xticks(x_pos, period_labels_short, rotation=45, fontsize=16)
        plt.yticks(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=18)
        plt.tight_layout()
        
        if save_plots:
            plot2d_path = os.path.join(base_results_dir, f"{resampling_years}year_period_spatial_coverage.png")
            plt.savefig(plot2d_path, dpi=300, bbox_inches='tight')
            print(f"Period spatial coverage plot saved to: {plot2d_path}")
        plt.close(fig2d)
    
    # Plot 3a: Percentiles evolution over time
    fig3a = plt.figure(figsize=(12, 6))
    plt.plot(years_list, yearly_percentiles['p75'], 'g-o', linewidth=2, markersize=3, label='75th Percentile')
    plt.plot(years_list, yearly_percentiles['p90'], 'orange', marker='s', linewidth=2, markersize=3, label='90th Percentile')
    plt.plot(years_list, yearly_percentiles['p95'], 'r-^', linewidth=2, markersize=3, label='95th Percentile')
    plt.plot(years_list, yearly_mean_hours, 'b-', linewidth=2, alpha=0.7, label='Mean')
    
    # Add long-term averages (average of yearly percentiles and yearly means)
    plt.axhline(y=np.mean(yearly_percentiles['p75']), color='green', linestyle='--', alpha=0.7)
    plt.axhline(y=np.mean(yearly_percentiles['p90']), color='orange', linestyle='--', alpha=0.7)
    plt.axhline(y=np.mean(yearly_percentiles['p95']), color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=np.mean(yearly_mean_hours), color='blue', linestyle='--', alpha=0.7)
    
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Exceedance Hours per Cell', fontsize=20)
    plt.title('Exceedance Hours Percentiles Evolution Over Time', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()
    
    if save_plots:
        plot3a_path = os.path.join(base_results_dir, "exceedance_hours_percentiles_evolution.png")
        plt.savefig(plot3a_path, dpi=300, bbox_inches='tight')
        print(f"Percentiles evolution plot saved to: {plot3a_path}")
    plt.close(fig3a)
    
    # Plot 3b: Distribution of yearly statistics
    fig3b = plt.figure(figsize=(10, 6))
    bp = plt.boxplot([yearly_mean_hours, yearly_percentiles['p75'], yearly_percentiles['p90'], yearly_percentiles['p95']], 
                      labels=['Mean', 'P75', 'P90', 'P95'],
                      patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
    plt.ylabel('Exceedance Hours per Cell', fontsize=20)
    plt.title('Distribution of Yearly Exceedance Hours Statistics', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plots:
        plot3b_path = os.path.join(base_results_dir, "yearly_exceedance_hours_distribution.png")
        plt.savefig(plot3b_path, dpi=300, bbox_inches='tight')
        print(f"Yearly exceedance hours distribution plot saved to: {plot3b_path}")
    plt.close(fig3b)
    
    # 5. SAVE SUMMARY REPORT
    if save_plots:
        summary_path = os.path.join(base_results_dir, "exceedance_hours_analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("ICE LOAD EXCEEDANCE HOURS RESAMPLING ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis timestamp: {pd.Timestamp.now()}\n")
            f.write(f"Resampling interval: {resampling_years} years\n")
            f.write(f"Height level: {height_level} ({dataset_with_ice_load.height.values[height_level]} m)\n")
            f.write(f"Ice load threshold: {ice_load_threshold} kg/m\n")
            f.write(f"Time step duration: {time_step_hours:.1f} hours\n")
            if months:
                f.write(f"Months included: {months}\n")
            f.write(f"Calendar time range: {start_year} to {end_year}\n")
            f.write(f"Winter years range: {winter_start_year} to {winter_end_year} ({total_winter_years} winter years)\n")
            f.write(f"Note: Winter Year N = June Year(N-1) to May Year(N)\n\n")
            
            f.write("LONG-TERM EXCEEDANCE STATISTICS:\n")
            f.write("-" * 35 + "\n")
            f.write(f"Grid mean total exceedance hours: {long_term_stats['grid_mean_total_hours']:.1f} hours\n")
            f.write(f"Mean hours per cell: {long_term_stats['mean_hours_per_cell']:.2f} hours\n")
            f.write(f"Std Dev per cell: {long_term_stats['std_hours_per_cell']:.2f} hours\n")
            f.write(f"Maximum hours per cell: {long_term_stats['max_hours_per_cell']:.2f} hours\n")
            f.write(f"95th Percentile per cell: {long_term_stats['p95_hours_per_cell']:.2f} hours\n")
            f.write(f"Cells with exceedance: {long_term_stats['cells_with_exceedance']} / {long_term_stats['total_grid_cells']}\n")
            f.write(f"Spatial coverage: {(long_term_stats['cells_with_exceedance']/long_term_stats['total_grid_cells']*100):.1f}%\n\n")
            
            f.write("YEARLY EXCEEDANCE HOURS SUMMARY:\n")
            f.write("-" * 35 + "\n")
            f.write(f"Mean of yearly totals: {np.mean(yearly_total_hours):.0f} hours\n")
            f.write(f"Std of yearly totals: {np.std(yearly_total_hours):.0f} hours\n")
            f.write(f"Max yearly total: {np.max(yearly_total_hours):.0f} hours ({years_list[np.argmax(yearly_total_hours)]})\n")
            f.write(f"Min yearly total: {np.min(yearly_total_hours):.0f} hours ({years_list[np.argmin(yearly_total_hours)]})\n")
            f.write(f"Mean of yearly means per cell: {np.mean(yearly_mean_hours):.2f} hours\n")
            f.write(f"Std of yearly means per cell: {np.std(yearly_mean_hours):.2f} hours\n\n")
            
            if len(period_total_hours) > 1:
                f.write(f"{resampling_years}-YEAR PERIOD EXCEEDANCE STATISTICS:\n")
                f.write("-" * 45 + "\n")
                for i, (label, total, mean) in enumerate(zip(period_labels, period_total_hours, period_mean_hours)):
                    deviation = mean - long_term_stats['mean_hours_per_cell']
                    pct_coverage = (period_stats[label]['cells_with_exceedance'] / 
                                   period_stats[label]['n_grid_cells'] * 100)
                    f.write(f"{label}: GridMeanTotal={total:.1f}h, Mean={mean:.2f}h/cell, Dev={deviation:+.2f}h, Coverage={pct_coverage:.1f}%\n")
            
            f.write(f"\nFILES GENERATED:\n")
            f.write("-" * 20 + "\n")
            if len(period_total_hours) > 1:
                f.write(f"- {resampling_years}year_period_grid_mean_total_hours.png\n")
                f.write(f"- {resampling_years}year_period_mean_hours_per_cell.png\n")
                f.write(f"- {resampling_years}year_period_deviations.png\n")
                f.write(f"- {resampling_years}year_period_spatial_coverage.png\n")
            f.write("- exceedance_hours_percentiles_evolution.png\n")
            f.write("- yearly_exceedance_hours_distribution.png\n")
            f.write("- exceedance_hours_analysis_summary.txt (this file)\n")
        
        print(f"Summary report saved to: {summary_path}")
    
    # Compile results
    results = {
        'long_term_stats': long_term_stats,
        'yearly_stats': yearly_stats,
        'yearly_deviations': yearly_deviations,
        'yearly_normalized_deviations': yearly_normalized_deviations,
        'period_stats': period_stats,
        'resampling_years': resampling_years,
        'years_list': years_list,  # These are winter years
        'calendar_time_range': (start_year, end_year),
        'winter_year_range': (winter_start_year, winter_end_year),
        'total_winter_years': total_winter_years,
        'resampling_periods': resampling_periods,
        'ice_load_threshold': ice_load_threshold,
        'time_step_hours': time_step_hours
    }
    
    print(f"\n=== EXCEEDANCE HOURS ANALYSIS COMPLETED ===")
    print(f"Analyzed {total_winter_years} winter years of data")
    print(f"Grid mean total exceedance hours: {long_term_stats['grid_mean_total_hours']:.1f} hours")
    print(f"Mean exceedance hours per cell: {long_term_stats['mean_hours_per_cell']:.2f} hours")
    print(f"Spatial coverage: {(long_term_stats['cells_with_exceedance']/long_term_stats['total_grid_cells']*100):.1f}%")
    if len(period_total_hours) > 1:
        print(f"Number of {resampling_years}-winter-year periods: {len(period_total_hours)}")
    
    return results

# BEGINNING OF MAIN ANALYSIS

# IMPORTS
import os
import functions as fn
import pandas as pd
import numpy as np
import xarray as xr

# Set working directory to project root if running from scripts folder
current_dir = os.getcwd()
if current_dir.endswith('scripts'):
    os.chdir('..')
    print(f"Changed working directory from {current_dir} to {os.getcwd()}")

# PARAMETERS
height = 1  # Height level index to use (0-based): 0=50m; 1=100m; 2=150m
ice_load_method = 51  # Method for ice load calculation
calculate_new_ice_load = False  # Whether to calculate ice load or load existing data


# IMPORT DATA
# Merge NetCDF files to combine all meteorological variables
print("=== MERGING NETCDF FILES ===")
main_file = "data/newa_wrf_for_jana_mstudent_extended.nc"
wd_file = "data/newa_wrf_for_jana_mstudent_extended_WD.nc"
pdfc_file= "data/newa_wrf_for_jana_mstudent_extended_PSFC_SEAICE_SWDDNI.nc"
merged_file = "data/newa_wrf_for_jana_mstudent_extended_merged.nc"
final_file = "data/newa_wrf_final_merged.nc"

# Check if merged file already exists
if not os.path.exists(final_file):
    print("Merged file not found. Creating merged dataset...")
    success = fn.merge_netcdf_files(merged_file, pdfc_file, final_file, verbose=True)
    if not success:
        print("Failed to merge files. Using main file only.")
        data1 = main_file
    else:
        data1 = merged_file
else:
    print("Merged file already exists. Using existing merged dataset...")
    data1 = final_file

# Import merged NEWEA meteorological data
dataset = fn.load_netcdf_data(data1)


# EXPLORE DATASET
height_level = dataset.height.values[height]  # Height level in meters
print(f"Exploring dataset at height level: {height_level} m")
# explore the variables
fn.explore_variables(dataset)

#explore one variable in detail in a chosen period
# fn.explore_variable_detail(dataset, 'QVAPOR')

# Plot grid location on map
#fn.plot_grid_points_cartopy_map(dataset, margin_degrees=1.5, zoom_level=8, title="Grid Points - Terrain Map")

# Offshore / Onshore classification
#landmask_results = fn.analyze_landmask(dataset, create_plot=True, save_results=True)


# CALCULATIONS AND PLOTS

# Period
start_date = '1989-01-01T00:00:00.000000000'
end_date = '2022-12-31T23:30:00.000000000'


# start_date = '2020-01-01T00:00:00.000000000'
# end_date = '2022-12-31T23:30:00.000000000'
dates = pd.date_range(start_date, end_date, freq='YS-JUL')


# COMPREHENSIVE CLIMATE ANALYSIS

# print("\n=== COMPREHENSIVE CLIMATE ANALYSIS ===")
# climate_results = fn.climate_analysis(
#     dataset=dataset,
#     height_level=height,
#     save_plots=True,
#     results_subdir="climate_analysis"
# )


# ACCERATION

# Fixed color scale range for ice accretion plots (adjust these values as needed)
accretion_vmin = 0
accretion_vmax = 1.5  # Adjust based on your expected range

# Accreation for winter and time period + plot
# fn.accreation_per_winter(dataset, start_date, end_date, height_level=height,
#                          custom_vmin=accretion_vmin, custom_vmax=accretion_vmax)



# ICE LOAD

#ice load data: load/calculate 
if calculate_new_ice_load:
    print("Calculating ice load...")
    #ice_load_data = fn.calculate_ice_load(dataset, dates, ice_load_method, height_level=height, create_figures=True)

    #Add ice load directly to the dataset
    print("=== ADDING ICE LOAD TO DATASET ===")
    dataset_with_ice_load = fn.add_ice_load_to_dataset(
        ds=dataset,
        dates=dates,
        method=ice_load_method,
        height_level=height,
        variable_name='ICE_LOAD'
    )

    # # Now you can access ice load directly from the dataset
    # ice_load_data = dataset_with_ice_load['ICE_LOAD']
    # print(f"Ice load data shape: {ice_load_data.shape}")
    # print(f"Ice load available at height level {height}: {dataset.height.values[height]} m")

else:
    print("Loading existing complete dataset with ice load...")
    filename = f"results/dataset_iceload_19890701_20220701_h{height}.nc"
    dataset_with_ice_load = xr.open_dataset(filename)  # Load complete dataset

    print(f"Loaded dataset from: {filename}")
    print(f"Dataset dimensions: {dataset_with_ice_load.dims}")
    print(f"Available variables: {list(dataset_with_ice_load.data_vars.keys())}")
    print(f"Ice load variable 'ICE_LOAD' is ready for analysis at height level {height}: {dataset_with_ice_load.height.values[height]} m")

# TEMPORAL GRADIENTS


# ICE LOAD RESAMPLING ANALYSIS

print("\n=== ICE LOAD RESAMPLING ANALYSIS ===")
resampling_results = ice_load_resampling_analysis(
    dataset_with_ice_load=dataset_with_ice_load,
    ice_load_variable='ICE_LOAD',
    height_level=height,
    resampling_years=1,  # Aggregate data into X-year periods
    save_plots=True,
    months=None,  # Use all months, or specify [12,1,2,3] for winter
    ice_load_threshold=0  # Include all ice load values
)

print("\n=== ICE LOAD RESAMPLING ANALYSIS EXCEEDANCE HOURS ===")

resampling_results_hours = ice_load_resampling_analysis_hours(
    dataset_with_ice_load=dataset_with_ice_load,
    ice_load_variable='ICE_LOAD',
    height_level=height,
    resampling_years=1,  # Aggregate data into X-year periods
    save_plots=True,
    months=None,  # Use all months, or specify [12,1,2,3] for winter
    ice_load_threshold=0.1  # Include all ice load values
)