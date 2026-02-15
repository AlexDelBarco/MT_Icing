def compare_ice_load_emd_newa(emd_data, dataset_with_ice_load, height, emd_coordinates, save_plots=True):
    """
    Compare ice load data between EMD observations and NEWA model dataset.
    
    Parameters:
    -----------
    emd_data : pandas.DataFrame
        EMD observational data containing ice load columns (MIce.50, MIce.100, MIce.150)
    dataset_with_ice_load : xarray.Dataset
        NEWA model dataset containing ICE_LOAD variable
    height : int
        Height level to compare (50, 100, or 150 meters)
    emd_coordinates : tuple
        EMD coordinates as (longitude, latitude) in degrees
    save_plots : bool, optional
        Whether to save plots to file (default: True)
        
    Returns:
    --------
    dict
        Dictionary containing comparison statistics and analysis results
    """
    
    print(f"=== ICE LOAD COMPARISON: EMD vs NEWA at {height}m ===")
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        from scipy import stats
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Validate height input
        if height not in [50, 100, 150]:
            raise ValueError(f"Height must be 50, 100, or 150 meters. Got: {height}")
        
        # Check if EMD data contains required column
        # Handle both integer and float height values from NEWA dataset
        emd_column = f"MIce.{int(height)}"  # Convert to int to avoid .0 suffix
        if emd_column not in emd_data.columns:
            available_ice_cols = [col for col in emd_data.columns if 'ice' in col.lower()]
            raise ValueError(f"Column '{emd_column}' not found in EMD data. Available ice columns: {available_ice_cols}")
        
        # Verify NEWA dataset height
        if 'ICE_LOAD' not in dataset_with_ice_load.data_vars:
            raise ValueError("'ICE_LOAD' variable not found in NEWA dataset")
        
        # Get height information from NEWA dataset
        height_levels = dataset_with_ice_load.height.values
        height_idx = None
        for i, h in enumerate(height_levels):
            if abs(h - height) < 1:  # Allow 1m tolerance
                height_idx = i
                break
        
        if height_idx is None:
            raise ValueError(f"Height {height}m not found in NEWA dataset. Available heights: {height_levels}")
        
        print(f"Using NEWA height level {height_idx} ({height_levels[height_idx]}m)")
        
        # Get EMD coordinates
        emd_lon, emd_lat = emd_coordinates
        print(f"EMD coordinates: {emd_lon:.4f}°E, {emd_lat:.4f}°N")
        
        # Find the closest grid cell to EMD coordinates
        if 'XLAT' not in dataset_with_ice_load and 'XLAT' not in dataset_with_ice_load.coords:
            raise ValueError("'XLAT' coordinate not found in NEWA dataset")
        if 'XLON' not in dataset_with_ice_load and 'XLON' not in dataset_with_ice_load.coords:
            raise ValueError("'XLON' coordinate not found in NEWA dataset")
        
        # Extract coordinates - handle both data variables and coordinates
        if 'XLAT' in dataset_with_ice_load.coords:
            lats = dataset_with_ice_load.coords['XLAT'].values
            lons = dataset_with_ice_load.coords['XLON'].values
        else:
            lats = dataset_with_ice_load['XLAT'].values
            lons = dataset_with_ice_load['XLON'].values
        
        print(f"NEWA grid covers:")
        print(f"  Longitude range: {lons.min():.4f}° to {lons.max():.4f}°E")
        print(f"  Latitude range: {lats.min():.4f}° to {lats.max():.4f}°N")
        print(f"  Grid shape: {lats.shape}")
        
        # Calculate distance from EMD to each grid point
        # Using Euclidean distance in degrees (appropriate for small domains)
        distance_squared = (lons - emd_lon)**2 + (lats - emd_lat)**2
        
        # Find the index of the closest grid cell
        closest_indices = np.unravel_index(np.argmin(distance_squared), distance_squared.shape)
        closest_sn, closest_we = closest_indices
        
        # Get the actual coordinates of the closest grid cell
        closest_lon = lons[closest_sn, closest_we]
        closest_lat = lats[closest_sn, closest_we]
        closest_distance_deg = np.sqrt(distance_squared[closest_sn, closest_we])
        
        # More accurate distance conversion accounting for latitude
        # At EMD latitude (59.6°N), longitude degrees are shorter
        lat_correction = np.cos(np.radians(emd_lat))
        closest_distance_km = closest_distance_deg * 111.32 * lat_correction
        
        print(f"Closest NEWA grid cell:")
        print(f"  Grid indices: south_north={closest_sn}, west_east={closest_we}")
        print(f"  Grid coordinates: {closest_lon:.4f}°E, {closest_lat:.4f}°N")
        print(f"  Distance from EMD: {closest_distance_km:.2f} km")
        
        # Extract NEWA ice load data at specified height and closest grid cell
        newa_ice_load = dataset_with_ice_load['ICE_LOAD'].isel(height=height_idx, south_north=closest_sn, west_east=closest_we)
        
        # Convert to pandas DataFrame for easier manipulation
        newa_df = newa_ice_load.to_dataframe(name='ICE_LOAD').reset_index()
        newa_df['time'] = pd.to_datetime(newa_df['time'])
        newa_df = newa_df.set_index('time')
        
        print(f"NEWA data extracted from grid cell ({closest_sn}, {closest_we})")
        print(f"Closest cell coordinates: {closest_lon:.4f}°E, {closest_lat:.4f}°N")
        print(f"Distance from EMD location: {closest_distance_km:.2f} km")
        
        # Prepare EMD data
        if not isinstance(emd_data.index, pd.DatetimeIndex):
            if 'time' in emd_data.columns:
                emd_df = emd_data.copy()
                emd_df['time'] = pd.to_datetime(emd_df['time'])
                emd_df = emd_df.set_index('time')
            else:
                raise ValueError("EMD data must have datetime index or 'time' column")
        else:
            emd_df = emd_data.copy()
        
        print(f"EMD data period: {emd_df.index.min()} to {emd_df.index.max()}")
        print(f"NEWA data period: {newa_df.index.min()} to {newa_df.index.max()}")
        
        # Find common time period
        common_start = max(emd_df.index.min(), newa_df.index.min())
        common_end = min(emd_df.index.max(), newa_df.index.max())
        
        print(f"Common period: {common_start} to {common_end}")
        
        # Filter to common period
        emd_common = emd_df.loc[common_start:common_end, emd_column].copy()
        newa_common = newa_df.loc[common_start:common_end, 'ICE_LOAD'].copy()
        
        # Resample NEWA data to hourly to match EMD (from 30min to 1h)
        print("Resampling NEWA data from 30min to 1h resolution...")
        newa_hourly = newa_common.resample('1H').mean()
        
        # Align time indices
        common_times = emd_common.index.intersection(newa_hourly.index)
        emd_aligned = emd_common.loc[common_times]
        newa_aligned = newa_hourly.loc[common_times]
        
        print(f"Aligned data points: {len(common_times)}")
        print(f"EMD ice load range: {emd_aligned.min():.3f} to {emd_aligned.max():.3f}")
        print(f"NEWA ice load range: {newa_aligned.min():.3f} to {newa_aligned.max():.3f}")
        
        # Remove NaN values and filter out non-icing months (June-October)
        valid_mask = ~(np.isnan(emd_aligned) | np.isnan(newa_aligned))
        emd_clean_all = emd_aligned[valid_mask]
        newa_clean_all = newa_aligned[valid_mask]
        
        # Filter out non-icing months (June=6, July=7, August=8, September=9, October=10)
        non_icing_months = [6, 7, 8, 9, 10]
        icing_mask = ~emd_clean_all.index.month.isin(non_icing_months)
        emd_clean = emd_clean_all[icing_mask]
        newa_clean = newa_clean_all[icing_mask]
        
        print(f"Valid data points after NaN removal: {len(emd_clean_all)}")
        print(f"Icing season data points (excluding Jun-Oct): {len(emd_clean)}")
        print(f"Excluded {len(emd_clean_all) - len(emd_clean)} non-icing season points")
        
        if len(emd_clean) < 10:
            print("Warning: Very few valid data points for comparison!")
            return None
        
        # Calculate comparison statistics
        print("\nCalculating comparison statistics...")
        
        # Basic statistics
        bias = np.mean(newa_clean - emd_clean)
        mae = mean_absolute_error(emd_clean, newa_clean)
        rmse = np.sqrt(mean_squared_error(emd_clean, newa_clean))
        
        # Correlation
        correlation, correlation_p = stats.pearsonr(emd_clean, newa_clean)
        spearman_corr, spearman_p = stats.spearmanr(emd_clean, newa_clean)
        
        # R-squared
        r2 = r2_score(emd_clean, newa_clean)
        
        # Relative metrics
        mean_emd = np.mean(emd_clean)
        relative_bias = (bias / mean_emd) * 100 if mean_emd != 0 else np.nan
        relative_mae = (mae / mean_emd) * 100 if mean_emd != 0 else np.nan
        relative_rmse = (rmse / mean_emd) * 100 if mean_emd != 0 else np.nan
        
        # Agreement statistics
        agreement_threshold = 0.1  # kg/m
        within_threshold = np.sum(np.abs(newa_clean - emd_clean) <= agreement_threshold)
        agreement_percentage = (within_threshold / len(emd_clean)) * 100
        
        # Print statistics
        print(f"\n=== COMPARISON STATISTICS ===")
        print(f"Data points: {len(emd_clean)}")
        print(f"EMD mean: {mean_emd:.3f} kg/m")
        print(f"NEWA mean: {np.mean(newa_clean):.3f} kg/m")
        print(f"Bias (NEWA - EMD): {bias:.3f} kg/m ({relative_bias:.1f}%)")
        print(f"MAE: {mae:.3f} kg/m ({relative_mae:.1f}%)")
        print(f"RMSE: {rmse:.3f} kg/m ({relative_rmse:.1f}%)")
        print(f"Correlation: {correlation:.3f} (p={correlation_p:.4f})")
        print(f"Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.4f})")
        print(f"R²: {r2:.3f}")
        print(f"Agreement within ±{agreement_threshold} kg/m: {agreement_percentage:.1f}%")
        
        # Prepare results dictionary
        results = {
            'height': height,
            'n_points': len(emd_clean),
            'common_period': {'start': common_start, 'end': common_end},
            'emd_coordinates': {'longitude': emd_lon, 'latitude': emd_lat},
            'newa_grid_cell': {
                'south_north_index': int(closest_sn),
                'west_east_index': int(closest_we),
                'longitude': float(closest_lon),
                'latitude': float(closest_lat),
                'distance_from_emd_km': float(closest_distance_km)
            },
            'statistics': {
                'emd_mean': float(mean_emd),
                'newa_mean': float(np.mean(newa_clean)),
                'bias': float(bias),
                'mae': float(mae),
                'rmse': float(rmse),
                'relative_bias_percent': float(relative_bias),
                'relative_mae_percent': float(relative_mae),
                'relative_rmse_percent': float(relative_rmse),
                'correlation': float(correlation),
                'correlation_p_value': float(correlation_p),
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
                'r_squared': float(r2),
                'agreement_percentage': float(agreement_percentage)
            },
            'data': {
                'emd': emd_clean,
                'newa': newa_clean,
                'time': common_times
            }
        }
        
        # Create plots if requested
        if save_plots:
            print(f"\nCreating requested comparison plots...")
            
            # Create directory structure
            base_dir = os.path.join("results", "figures", "EMD", "Ice_Load", f"NEWA_EMD_comparison_{height}")
            os.makedirs(base_dir, exist_ok=True)
            
            # Plot 1A: Time series comparison with lines only
            print("1A. Creating full time series comparison (lines only)...")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))
            
            # Subplot 1: Original hourly data (lines only)
            ax1.plot(emd_clean.index, emd_clean.values, 'b-', alpha=0.7, linewidth=0.5, 
                    label=f'EMD Hourly ({emd_column})')
            ax1.plot(newa_clean.index, newa_clean.values, 'r-', alpha=0.7, linewidth=0.5, 
                    label=f'NEWA Hourly (ICE_LOAD)')
            ax1.set_ylabel('Ice Load (kg/m)', fontsize=20)
            ax1.set_title(f'Hourly Ice Load Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Lines', fontsize=28)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Calculate daily averages - hourly mean per each day
            emd_daily_avg = emd_clean.resample('D').mean()
            newa_daily_avg = newa_clean.resample('D').mean()
            
            # Subplot 2: Daily averages (lines only)
            ax2.plot(emd_daily_avg.index, emd_daily_avg.values, 'b-', alpha=0.8, linewidth=1.0, 
                    label=f'EMD Daily Mean ({emd_column})')
            ax2.plot(newa_daily_avg.index, newa_daily_avg.values, 'r-', alpha=0.8, linewidth=1.0, 
                    label=f'NEWA Daily Mean (ICE_LOAD)')
            ax2.set_ylabel('Ice Load (kg/m)', fontsize=20)
            ax2.set_title(f'Daily Mean Ice Load Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Lines', fontsize=28)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Calculate weekly averages - hourly mean per each week
            emd_weekly_avg = emd_clean.resample('W').mean()
            newa_weekly_avg = newa_clean.resample('W').mean()
            
            # Subplot 3: Weekly averages (lines only)
            ax3.plot(emd_weekly_avg.index, emd_weekly_avg.values, 'b-', alpha=0.9, linewidth=1.5, 
                    label=f'EMD Weekly Mean ({emd_column})')
            ax3.plot(newa_weekly_avg.index, newa_weekly_avg.values, 'r-', alpha=0.9, linewidth=1.5, 
                    label=f'NEWA Weekly Mean (ICE_LOAD)')
            ax3.set_xlabel('Time', fontsize=20)
            ax3.set_ylabel('Ice Load (kg/m)', fontsize=20)
            ax3.set_title(f'Weekly Mean Ice Load Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Lines', fontsize=28)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle(f'Multi-Scale Ice Load Comparison: EMD vs NEWA at {height}m (Lines Only)',
                        fontsize=28, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            timeseries_lines_path = os.path.join(base_dir, f'multi_scale_timeseries_lines_{height:.0f}m.png')
            plt.savefig(timeseries_lines_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {timeseries_lines_path}")
            
            # Plot 1B: Time series comparison with scatter only
            print("1B. Creating full time series comparison (scatter only)...")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))
            
            # Subplot 1: Original hourly data (scatter only)
            ax1.scatter(emd_clean.index, emd_clean.values, c='blue', s=0.5, alpha=0.6, label=f'EMD Hourly ({emd_column})')
            ax1.scatter(newa_clean.index, newa_clean.values, c='red', s=0.5, alpha=0.6, label=f'NEWA Hourly (ICE_LOAD)')
            ax1.set_ylabel('Ice Load (kg/m)', fontsize=20)
            ax1.set_title(f'Hourly Ice Load Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Scatter', fontsize=28)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: Daily averages (scatter only)
            ax2.scatter(emd_daily_avg.index, emd_daily_avg.values, c='blue', s=3, alpha=0.7, label=f'EMD Daily Mean ({emd_column})')
            ax2.scatter(newa_daily_avg.index, newa_daily_avg.values, c='red', s=3, alpha=0.7, label=f'NEWA Daily Mean (ICE_LOAD)')
            ax2.set_ylabel('Ice Load (kg/m)', fontsize=20)
            ax2.set_title(f'Daily Mean Ice Load Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Scatter', fontsize=28)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Subplot 3: Weekly averages (scatter only)
            ax3.scatter(emd_weekly_avg.index, emd_weekly_avg.values, c='blue', s=10, alpha=0.8, label=f'EMD Weekly Mean ({emd_column})')
            ax3.scatter(newa_weekly_avg.index, newa_weekly_avg.values, c='red', s=10, alpha=0.8, label=f'NEWA Weekly Mean (ICE_LOAD)')
            ax3.set_xlabel('Time', fontsize=20)
            ax3.set_ylabel('Ice Load (kg/m)', fontsize=20)
            ax3.set_title(f'Weekly Mean Ice Load Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Scatter', fontsize=28)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle(f'Multi-Scale Ice Load Comparison: EMD vs NEWA at {height}m (Scatter Only)',
                        fontsize=28, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            timeseries_scatter_path = os.path.join(base_dir, f'multi_scale_timeseries_scatter_{height:.0f}m.png')
            plt.savefig(timeseries_scatter_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {timeseries_scatter_path}")
            
            # Plot 2A: Difference over time with lines only
            print("2A. Creating difference time series (lines only)...")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))
            
            # Hourly differences - difference for every hour (lines only)
            differences = newa_clean - emd_clean
            ax1.plot(differences.index, differences.values, 'g-', alpha=0.7, linewidth=0.5)
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax1.axhline(y=bias, color='red', linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Mean Bias: {bias:.3f} kg/m')
            ax1.set_ylabel('Difference (NEWA - EMD) [kg/m]', fontsize=20)
            ax1.set_title(f'Hourly Ice Load Differences: NEWA - EMD at {height}m (Icing Season Only) - Lines', fontsize=28)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Daily differences - mean of hourly differences for each day (lines only)
            daily_differences_ts = newa_daily_avg - emd_daily_avg
            daily_bias = daily_differences_ts.mean()
            ax2.plot(daily_differences_ts.index, daily_differences_ts.values, 'g-', alpha=0.8, linewidth=1.0)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax2.axhline(y=daily_bias, color='red', linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Daily Mean Bias: {daily_bias:.3f} kg/m')
            ax2.set_ylabel('Difference (NEWA - EMD) [kg/m]', fontsize=20)
            ax2.set_title(f'Daily Mean Ice Load Differences: NEWA - EMD at {height}m (Icing Season Only) - Lines', fontsize=28)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Weekly differences - mean of hourly differences for each week (lines only)
            weekly_differences_ts = newa_weekly_avg - emd_weekly_avg
            weekly_bias = weekly_differences_ts.mean()
            ax3.plot(weekly_differences_ts.index, weekly_differences_ts.values, 'g-', alpha=0.9, linewidth=1.5)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax3.axhline(y=weekly_bias, color='red', linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Weekly Mean Bias: {weekly_bias:.3f} kg/m')
            ax3.set_xlabel('Time', fontsize=20)
            ax3.set_ylabel('Difference (NEWA - EMD) [kg/m]', fontsize=20)
            ax3.set_title(f'Weekly Mean Ice Load Differences: NEWA - EMD at {height}m (Icing Season Only) - Lines', fontsize=28)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle(f'Multi-Scale Ice Load Differences: NEWA - EMD at {height}m (Lines Only)',
                        fontsize=28, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            differences_lines_path = os.path.join(base_dir, f'multi_scale_differences_lines_{height:.0f}m.png')
            plt.savefig(differences_lines_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {differences_lines_path}")
            
            # Plot 2B: Difference over time with scatter only
            print("2B. Creating difference time series (scatter only)...")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))
            
            # Hourly differences - difference for every hour (scatter only)
            ax1.scatter(differences.index, differences.values, c='green', s=0.5, alpha=0.6)
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax1.axhline(y=bias, color='red', linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Mean Bias: {bias:.3f} kg/m')
            ax1.set_ylabel('Difference (NEWA - EMD) [kg/m]', fontsize=20)
            ax1.set_title(f'Hourly Ice Load Differences: NEWA - EMD at {height}m (Icing Season Only) - Scatter', fontsize=28)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Daily differences - mean of hourly differences for each day (scatter only)
            ax2.scatter(daily_differences_ts.index, daily_differences_ts.values, c='green', s=3, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax2.axhline(y=daily_bias, color='red', linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Daily Mean Bias: {daily_bias:.3f} kg/m')
            ax2.set_ylabel('Difference (NEWA - EMD) [kg/m]', fontsize=20)
            ax2.set_title(f'Daily Mean Ice Load Differences: NEWA - EMD at {height}m (Icing Season Only) - Scatter', fontsize=28)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Weekly differences - mean of hourly differences for each week (scatter only)
            ax3.scatter(weekly_differences_ts.index, weekly_differences_ts.values, c='green', s=10, alpha=0.8)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax3.axhline(y=weekly_bias, color='red', linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Weekly Mean Bias: {weekly_bias:.3f} kg/m')
            ax3.set_xlabel('Time', fontsize=20)
            ax3.set_ylabel('Difference (NEWA - EMD) [kg/m]', fontsize=20)
            ax3.set_title(f'Weekly Mean Ice Load Differences: NEWA - EMD at {height}m (Icing Season Only) - Scatter', fontsize=28)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle(f'Multi-Scale Ice Load Differences: NEWA - EMD at {height}m (Scatter Only)',
                        fontsize=28, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            differences_scatter_path = os.path.join(base_dir, f'multi_scale_differences_scatter_{height:.0f}m.png')
            plt.savefig(differences_scatter_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {differences_scatter_path}")
            
            # Plot 3: EMD vs NEWA scatter plot with 45° line and linear regression
            print("3. Creating EMD vs NEWA scatter plot with regression analysis...")
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Create scatter plot (NEWA on x-axis, EMD on y-axis)
            scatter = ax.scatter(newa_clean.values, emd_clean.values, alpha=0.6, s=20, c='blue', edgecolors='none', label='Data points')
            
            # Calculate plot limits
            min_val = min(np.min(newa_clean), np.min(emd_clean))
            max_val = max(np.max(newa_clean), np.max(emd_clean))
            plot_range = max_val - min_val
            margin = plot_range * 0.05  # 5% margin
            xlim = [min_val - margin, max_val + margin]
            ylim = [min_val - margin, max_val + margin]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
            # Plot 45-degree reference line (perfect agreement)
            ax.plot(xlim, ylim, 'k--', linewidth=2, alpha=0.8, label='Perfect agreement (1:1 line)')
            
            # Calculate and plot linear regression
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(newa_clean.values, emd_clean.values)
            
            # Create regression line
            regression_x = np.array(xlim)
            regression_y = slope * regression_x + intercept
            ax.plot(regression_x, regression_y, 'r-', linewidth=2, alpha=0.8, 
                   label=f'Linear regression (y = {slope:.3f}x + {intercept:.3f})')
            
            # Add statistics text box
            stats_text = (f'N = {len(emd_clean)}\n'
                         f'R² = {r2:.3f}\n'
                         f'Correlation = {correlation:.3f}\n'
                         f'RMSE = {rmse:.3f} kg/m\n'
                         f'MAE = {mae:.3f} kg/m\n'
                         f'Bias = {bias:.3f} kg/m\n'
                         f'Slope = {slope:.3f}\n'
                         f'Intercept = {intercept:.3f}')
            
            # Position text box in upper left corner
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=16,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # Set labels and title
            ax.set_xlabel(f'NEWA Ice Load (kg/m) at {height}m', fontsize=20)
            ax.set_ylabel(f'EMD Ice Load (kg/m) at {height}m', fontsize=20)
            ax.set_title(f'EMD vs NEWA Ice Load Scatter Plot at {height}m (Icing Season Only)',
                        fontsize=28, pad=15)
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=15)
            
            # Make axes equal for better visualization of agreement
            ax.set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            
            scatter_regression_path = os.path.join(base_dir, f'emd_vs_newa_scatter_{height:.0f}m.png')
            plt.savefig(scatter_regression_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {scatter_regression_path}")
            
            print(f"Regression analysis:")
            print(f"  Slope: {slope:.3f} (perfect agreement = 1.0)")
            print(f"  Intercept: {intercept:.3f} kg/m (perfect agreement = 0.0)")
            print(f"  R-value: {r_value:.3f}")
            print(f"  P-value: {p_value:.4e}")
            print(f"  Standard error: {std_err:.3f}")
            
            # Plot 4: EMD vs NEWA scatter plot (non-zero values only)
            print("4. Creating EMD vs NEWA scatter plot (non-zero values only)...")
            
            # Filter for non-zero values only
            non_zero_mask = (emd_clean > 0) & (newa_clean > 0)
            emd_nonzero = emd_clean[non_zero_mask]
            newa_nonzero = newa_clean[non_zero_mask]
            
            print(f"Non-zero data: {len(emd_nonzero)} points out of {len(emd_clean)} total")
            
            if len(emd_nonzero) > 1:
                # Create scatter plot with sample size information
                fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                
                # Create scatter plot
                sc = ax.scatter(newa_nonzero.values, emd_nonzero.values, 
                              c='blue', alpha=0.6, s=20, edgecolors='none', label='Data points')
                
                # Calculate plot limits with same margin as normal scatter plot
                min_val = min(emd_nonzero.min(), newa_nonzero.min())
                max_val = max(emd_nonzero.max(), newa_nonzero.max())
                plot_range = max_val - min_val
                margin = plot_range * 0.05  # 5% margin
                xlim = [min_val - margin, max_val + margin]
                ylim = [min_val - margin, max_val + margin]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                
                # Plot 45-degree reference line (perfect agreement)
                ax.plot(xlim, ylim, 'k--', linewidth=2, alpha=0.8, label='Perfect agreement (1:1 line)')
                
                # Calculate and plot linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(newa_nonzero.values, emd_nonzero.values)
                regression_x = np.array(xlim)
                regression_y = slope * regression_x + intercept
                ax.plot(regression_x, regression_y, 'r-', linewidth=2, alpha=0.8,
                       label=f'Linear regression (y = {slope:.3f}x + {intercept:.3f})')
                
                ax.set_xlabel(f'NEWA Ice Load (kg/m) at {height}m', fontsize=20)
                ax.set_ylabel(f'EMD Ice Load (kg/m) at {height}m', fontsize=20)
                ax.set_title(f'EMD vs NEWA Ice Load Scatter Plot at {height}m (Non-Zero Values Only)',
                            fontsize=28, pad=15)
                
                # Add grid and legend
                ax.grid(True, alpha=0.3)
                ax.legend(loc='lower right', fontsize=15)
                
                # Make axes equal for better visualization of agreement
                ax.set_aspect('equal', adjustable='box')
                
                # Add statistics text box (same format as normal scatter plot)
                stats_text = (f'N = {len(emd_nonzero)}\n'
                             f'R² = {r_value**2:.3f}\n'
                             f'Correlation = {np.corrcoef(emd_nonzero, newa_nonzero)[0,1]:.3f}\n'
                             f'RMSE = {np.sqrt(np.mean((newa_nonzero - emd_nonzero)**2)):.3f} kg/m\n'
                             f'MAE = {np.mean(np.abs(newa_nonzero - emd_nonzero)):.3f} kg/m\n'
                             f'Bias = {np.mean(newa_nonzero - emd_nonzero):.3f} kg/m\n'
                             f'Slope = {slope:.3f}\n'
                             f'Intercept = {intercept:.3f}')
                
                # Position text box in upper left corner (same as normal scatter plot)
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=16,
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
                
                plt.tight_layout()
                
                nonzero_scatter_path = os.path.join(base_dir, f'emd_vs_newa_scatter_nonzero_{height:.0f}m.png')
                plt.savefig(nonzero_scatter_path, dpi=150, facecolor='white')
                plt.close()
                print(f"Saved: {nonzero_scatter_path}")
                
                print(f"\nNon-zero scatter plot statistics:")
                print(f"  Sample size: {len(emd_nonzero)}")
                print(f"  EMD range (non-zero): {emd_nonzero.min():.6f} to {emd_nonzero.max():.6f} kg/m")
                print(f"  NEWA range (non-zero): {newa_nonzero.min():.6f} to {newa_nonzero.max():.6f} kg/m")
                print(f"  Correlation: {np.corrcoef(emd_nonzero, newa_nonzero)[0,1]:.3f}")
                print(f"  Linear regression: y = {slope:.3f}x + {intercept:.3f}")
                print(f"  R-squared: {r_value**2:.3f}")
                print(f"  Standard error: {std_err:.3f}")
            else:
                print("Insufficient non-zero data for scatter plot")
            
            # Plot 5: Zero values analysis - Bar plot
            print("5. Creating zero values analysis bar plot...")
            
            # Calculate zero value statistics
            emd_zero_count = (emd_clean == 0).sum()
            newa_zero_count = (newa_clean == 0).sum()
            total_timestamps = len(emd_clean)
            
            emd_zero_percentage = (emd_zero_count / total_timestamps) * 100
            newa_zero_percentage = (newa_zero_count / total_timestamps) * 100
            
            # Create bar plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            datasets = ['EMD', 'NEWA']
            zero_percentages = [emd_zero_percentage, newa_zero_percentage]
            zero_counts = [emd_zero_count, newa_zero_count]
            colors = ['steelblue', 'orange']
            
            bars = ax.bar(datasets, zero_percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for i, (bar, count, percentage) in enumerate(zip(bars, zero_counts, zero_percentages)):
                height_b = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height_b + 0.5,
                       f'{percentage:.1f}%\n({count:,} hours)',
                       ha='center', va='bottom', fontweight='bold', fontsize=16)
            
            ax.set_ylabel('Percentage of Zero Values (%)', fontsize=20, fontweight='bold')
            ax.set_title(f'Zero Value Analysis at {height:.0f}m\n'
                        f'Total timestamps: {total_timestamps:,} hours',
                        fontsize=28, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(zero_percentages) * 1.15)
            
            # Add summary statistics text
            stats_text = f'Summary:\n'
            stats_text += f'EMD zeros: {emd_zero_count:,} ({emd_zero_percentage:.1f}%)\n'
            stats_text += f'NEWA zeros: {newa_zero_count:,} ({newa_zero_percentage:.1f}%)\n'
            stats_text += f'Both zero: {((emd_clean == 0) & (newa_clean == 0)).sum():,}\n'
            stats_text += f'Either zero: {((emd_clean == 0) | (newa_clean == 0)).sum():,}'
            
            ax.text(0.02, 0.05, stats_text, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   verticalalignment='bottom', horizontalalignment='left', fontsize=16)
            
            plt.tight_layout()
            
            zero_analysis_path = os.path.join(base_dir, f'zero_values_analysis_{height:.0f}m.png')
            plt.savefig(zero_analysis_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {zero_analysis_path}")
            
            print(f"\nZero values analysis:")
            print(f"  EMD zeros: {emd_zero_count:,} timestamps ({emd_zero_percentage:.1f}%)")
            print(f"  NEWA zeros: {newa_zero_count:,} timestamps ({newa_zero_percentage:.1f}%)")
            print(f"  Both datasets zero: {((emd_clean == 0) & (newa_clean == 0)).sum():,} timestamps")
            print(f"  Either dataset zero: {((emd_clean == 0) | (newa_clean == 0)).sum():,} timestamps")
            
            # Plot 6: Box plot for positive values only
            print("6. Creating box plot for positive values distribution...")
            
            # Filter for positive values only (> 0 for both datasets)
            positive_mask = (emd_clean > 0) & (newa_clean > 0)
            emd_positive = emd_clean[positive_mask]
            newa_positive = newa_clean[positive_mask]
            
            if len(emd_positive) > 0 and len(newa_positive) > 0:
                # Create box plot
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                # Prepare data for box plot
                box_data = [emd_positive.values, newa_positive.values]
                labels = ['EMD', 'NEWA']
                colors = ['steelblue', 'orange']
                
                # Create box plot
                box_plot = ax.boxplot(box_data, labels=labels, patch_artist=True,
                                    showmeans=True, meanline=True,
                                    boxprops=dict(alpha=0.7),
                                    medianprops=dict(color='red', linewidth=2),
                                    meanprops=dict(color='black', linewidth=2, linestyle='--'))
                
                # Color the boxes
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_ylabel('Ice Load [kg/m]', fontsize=40, fontweight='bold')
                ax.set_title(f'Distribution of Positive Ice Load Values at {height}m\n'
                            f'Positive values: EMD={len(emd_positive):,}, NEWA={len(newa_positive):,}',
                            fontsize=28, fontweight='bold')
                ax.tick_params(axis='both', labelsize=40)
                ax.grid(True, alpha=0.3)
                
                # Calculate and display statistics
                emd_stats = {
                    'count': len(emd_positive),
                    'mean': emd_positive.mean(),
                    'std': emd_positive.std(),
                    'min': emd_positive.min(),
                    'q25': emd_positive.quantile(0.25),
                    'median': emd_positive.median(),
                    'q75': emd_positive.quantile(0.75),
                    'max': emd_positive.max()
                }
                
                newa_stats = {
                    'count': len(newa_positive),
                    'mean': newa_positive.mean(),
                    'std': newa_positive.std(),
                    'min': newa_positive.min(),
                    'q25': newa_positive.quantile(0.25),
                    'median': newa_positive.median(),
                    'q75': newa_positive.quantile(0.75),
                    'max': newa_positive.max()
                }
                
                # Add statistics text box
                stats_text = 'Positive Values Statistics:\n\n'
                stats_text += f'EMD (n={emd_stats["count"]:,}):\n'
                stats_text += f'  Mean: {emd_stats["mean"]:.4f} kg/m\n'
                stats_text += f'  Std:  {emd_stats["std"]:.4f} kg/m\n'
                stats_text += f'  Min:  {emd_stats["min"]:.4f} kg/m\n'
                stats_text += f'  Q25:  {emd_stats["q25"]:.4f} kg/m\n'
                stats_text += f'  Med:  {emd_stats["median"]:.4f} kg/m\n'
                stats_text += f'  Q75:  {emd_stats["q75"]:.4f} kg/m\n'
                stats_text += f'  Max:  {emd_stats["max"]:.4f} kg/m\n\n'
                
                stats_text += f'NEWA (n={newa_stats["count"]:,}):\n'
                stats_text += f'  Mean: {newa_stats["mean"]:.4f} kg/m\n'
                stats_text += f'  Std:  {newa_stats["std"]:.4f} kg/m\n'
                stats_text += f'  Min:  {newa_stats["min"]:.4f} kg/m\n'
                stats_text += f'  Q25:  {newa_stats["q25"]:.4f} kg/m\n'
                stats_text += f'  Med:  {newa_stats["median"]:.4f} kg/m\n'
                stats_text += f'  Q75:  {newa_stats["q75"]:.4f} kg/m\n'
                stats_text += f'  Max:  {newa_stats["max"]:.4f} kg/m'
                
                ax.text(1.02, 1.0, stats_text, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                       verticalalignment='top', fontsize=16, family='monospace')
                
                # Add legend explaining box plot elements
                legend_text = 'Box Plot Elements:\n'
                legend_text += '━ Red line: Median\n'
                legend_text += '┅ Black line: Mean\n'
                legend_text += '□ Box: Q25-Q75 (IQR)\n'
                legend_text += '┬ Whiskers: 1.5×IQR\n'
                legend_text += '○ Outliers'
                
                ax.text(0.98, 0.98, legend_text, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                       verticalalignment='top', horizontalalignment='right', fontsize=16)
                
                plt.tight_layout()
                
                positive_boxplot_path = os.path.join(base_dir, f'positive_values_boxplot_{height:.0f}m.png')
                plt.savefig(positive_boxplot_path, dpi=150, facecolor='white', bbox_inches='tight')
                plt.close()
                print(f"Saved: {positive_boxplot_path}")
                
                print(f"\nPositive values distribution statistics:")
                print(f"  EMD positive values: {len(emd_positive):,} timestamps")
                print(f"    Mean ± Std: {emd_stats['mean']:.4f} ± {emd_stats['std']:.4f} kg/m")
                print(f"    Median [Q25, Q75]: {emd_stats['median']:.4f} [{emd_stats['q25']:.4f}, {emd_stats['q75']:.4f}] kg/m")
                print(f"    Range: {emd_stats['min']:.4f} to {emd_stats['max']:.4f} kg/m")
                print(f"  NEWA positive values: {len(newa_positive):,} timestamps")
                print(f"    Mean ± Std: {newa_stats['mean']:.4f} ± {newa_stats['std']:.4f} kg/m")
                print(f"    Median [Q25, Q75]: {newa_stats['median']:.4f} [{newa_stats['q25']:.4f}, {newa_stats['q75']:.4f}] kg/m")
                print(f"    Range: {newa_stats['min']:.4f} to {newa_stats['max']:.4f} kg/m")
            else:
                print("No positive values available for box plot analysis")
            
            # Plot 7: Hourly mean differences grid (all months included)
            print("7. Creating hourly mean differences grid (all months)...")
            
            # Calculate daily hourly means for each specific day (not averaged across years)
            # Group by date (year-month-day) and calculate mean for each day
            emd_daily_means = emd_clean_all.resample('D').mean()
            newa_daily_means = newa_clean_all.resample('D').mean()
            
            # Align the daily data
            common_daily_dates = emd_daily_means.index.intersection(newa_daily_means.index)
            emd_daily_aligned = emd_daily_means.loc[common_daily_dates]
            newa_daily_aligned = newa_daily_means.loc[common_daily_dates]
            
            # Calculate daily differences for each specific day
            daily_differences_all = newa_daily_aligned - emd_daily_aligned
            
            # Create DataFrame with year, day of year, and differences for grid plotting
            grid_df = pd.DataFrame({
                'date': daily_differences_all.index,
                'difference': daily_differences_all.values
            })
            grid_df['year'] = grid_df['date'].dt.year
            grid_df['day_of_year'] = grid_df['date'].dt.dayofyear
            
            # Create pivot table for grid (each cell is unique year-day combination)
            pivot_grid = grid_df.pivot(index='year', columns='day_of_year', values='difference')
            
            # Fill missing values with NaN and ensure we have 365 days
            if pivot_grid.shape[1] < 365:
                for day in range(1, 366):
                    if day not in pivot_grid.columns:
                        pivot_grid[day] = np.nan
            
            # Sort columns to ensure proper day order and limit to 365 days
            pivot_grid = pivot_grid.reindex(columns=sorted(pivot_grid.columns)[:365])
            
            # Convert to numpy array for plotting
            grid_array = pivot_grid.values
            
            # Create the grid plot with improved clarity (all months)
            plt.figure(figsize=(24, 14))  # Even larger for all months
            
            # Use a diverging colormap centered at 0
            vmax = np.nanmax(np.abs(grid_array))
            vmin = -vmax
            
            # Create the heatmap with clear cell boundaries and darker colors
            im = plt.imshow(grid_array, cmap='seismic', aspect='auto', 
                          interpolation='nearest', vmin=vmin, vmax=vmax)
            
            # Add grid lines to separate cells clearly
            plt.gca().set_xticks(np.arange(-0.5, grid_array.shape[1], 1), minor=True)
            plt.gca().set_yticks(np.arange(-0.5, grid_array.shape[0], 1), minor=True)
            plt.grid(which="minor", color="black", linestyle='-', linewidth=0.1, alpha=0.2)
            
            # Add colorbar with better formatting
            cbar = plt.colorbar(im, shrink=0.6, pad=0.02)
            cbar.set_label('Hourly Mean Ice Load Difference (NEWA - EMD) [kg/m]', fontsize=16)
            cbar.ax.tick_params(labelsize=15)
            
            # Set labels and ticks with better formatting for all months
            plt.xlabel('Day of Year', fontsize=20)
            plt.ylabel('Year', fontsize=20)
            plt.title(f'Daily Mean Ice Load Differences Grid: NEWA - EMD at {height}m (All Months)\n'
                     f'Each cell = daily mean difference for that specific year and day', fontsize=28, pad=20)
            
            # Set year labels
            year_indices = np.arange(0, len(pivot_grid.index))
            year_step = max(1, len(pivot_grid.index)//15)
            year_ticks = year_indices[::year_step]
            plt.yticks(year_ticks, [pivot_grid.index[i] for i in year_ticks], fontsize=20)
            
            # Set day of year labels (all months)
            month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.xticks(month_starts, month_labels, rotation=0, fontsize=20)
            
            # Add secondary x-axis with day numbers
            ax2 = plt.gca().secondary_xaxis('top')
            day_ticks = np.arange(0, 366, 30)
            ax2.set_xticks(day_ticks)
            ax2.set_xlabel('Day of Year', fontsize=20)
            ax2.tick_params(labelsize=20)
            
            # Improve layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            
            daily_grid_path = os.path.join(base_dir, f'hourly_mean_grid_all_months_{height:.0f}m.png')
            plt.savefig(daily_grid_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {daily_grid_path}")
            
            print(f"\nHourly mean grid statistics (all months):")
            print(f"  Years covered: {pivot_grid.index.min()} to {pivot_grid.index.max()}")
            print(f"  Days per year: 365")
            print(f"  All months included (Jan-Dec)")
            print(f"  Hourly mean difference range: {np.nanmin(grid_array):.3f} to {np.nanmax(grid_array):.3f} kg/m")
            print(f"  Overall mean difference: {np.nanmean(grid_array):.3f} kg/m")
            
            print(f"\n=== PLOT SUMMARY ===")
            print(f"Created 7 plots:")
            print(f"  1A. Multi-scale time series (lines): {timeseries_lines_path}")
            print(f"  1B. Multi-scale time series (scatter): {timeseries_scatter_path}")
            print(f"  2A. Multi-scale differences (lines): {differences_lines_path}")
            print(f"  2B. Multi-scale differences (scatter): {differences_scatter_path}")
            print(f"  3. EMD vs NEWA scatter with regression: {scatter_regression_path}")
            print(f"  4. EMD vs NEWA scatter (non-zero only): {nonzero_scatter_path}")
            print(f"  5. Zero values analysis (bar plot): {zero_analysis_path}")
            print(f"  6. Positive values distribution (box plot): {positive_boxplot_path}")
            print(f"  7. Hourly mean differences grid: {daily_grid_path}")
        
        print(f"\n✓ Ice load comparison completed successfully!")
        print(f"Results saved to: {base_dir}")
        
        return results
        
    except Exception as e:
        print(f"Error in ice load comparison: {e}")
        import traceback
        traceback.print_exc()
        return None

def emd_newa_typical(emd_data, dataset_with_ice_load, height, emd_coordinates, save_plots=True, ice_load_threshold=0.0, non_zero_percentage=0.0):
    """
    Compare typical day, week, and year ice load patterns between EMD observations and NEWA model dataset.
    
    Parameters:
    -----------
    emd_data : pandas.DataFrame
        EMD observational data containing ice load columns (MIce.50, MIce.100, MIce.150)
    dataset_with_ice_load : xarray.Dataset
        NEWA model dataset containing ICE_LOAD variable
    height : int
        Height level to compare (50, 100, or 150 meters)
    emd_coordinates : tuple
        EMD coordinates as (longitude, latitude) in degrees
    save_plots : bool, optional
        Whether to save plots to file (default: True)
    ice_load_threshold : float, optional
        Minimum mean hourly ice load threshold (kg/m). Only temporal periods where both datasets
        have mean hourly ice load >= threshold are included in analysis (default: 0.0)
    non_zero_percentage : float, optional
        Minimum percentage (0-100) of hours that must be > 0 in both datasets for a temporal period
        to be included in analysis. Applied to daily, weekly, and yearly aggregations (default: 0.0)
        
    Returns:
    --------
    dict
        Dictionary containing typical patterns analysis results
    """
    
    print(f"=== TYPICAL PATTERNS ANALYSIS: EMD vs NEWA at {height}m ===")
    print(f"Ice load threshold: {ice_load_threshold} kg/m (minimum mean hourly ice load for inclusion)")
    print(f"Non-zero percentage threshold: {non_zero_percentage}% (minimum percentage of hours > 0 required)")
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        from scipy import stats
        
        # Validate height input
        if height not in [50, 100, 150]:
            raise ValueError(f"Height must be 50, 100, or 150 meters. Got: {height}")
        
        # Check if EMD data contains required column
        emd_column = f"MIce.{int(height)}"
        if emd_column not in emd_data.columns:
            available_ice_cols = [col for col in emd_data.columns if 'ice' in col.lower()]
            raise ValueError(f"Column '{emd_column}' not found in EMD data. Available ice columns: {available_ice_cols}")
        
        # Verify NEWA dataset height
        if 'ICE_LOAD' not in dataset_with_ice_load.data_vars:
            raise ValueError("'ICE_LOAD' variable not found in NEWA dataset")
        
        # Get height information from NEWA dataset
        height_levels = dataset_with_ice_load.height.values
        height_idx = None
        for i, h in enumerate(height_levels):
            if abs(h - height) < 1:  # Allow 1m tolerance
                height_idx = i
                break
        
        if height_idx is None:
            raise ValueError(f"Height {height}m not found in NEWA dataset. Available heights: {height_levels}")
        
        print(f"Using NEWA height level {height_idx} ({height_levels[height_idx]}m)")
        
        # Get EMD coordinates and find closest NEWA grid cell
        emd_lon, emd_lat = emd_coordinates
        print(f"EMD coordinates: {emd_lon:.4f}°E, {emd_lat:.4f}°N")
        
        # Extract coordinates - handle both data variables and coordinates
        if 'XLAT' in dataset_with_ice_load.coords:
            lats = dataset_with_ice_load.coords['XLAT'].values
            lons = dataset_with_ice_load.coords['XLON'].values
        else:
            lats = dataset_with_ice_load['XLAT'].values
            lons = dataset_with_ice_load['XLON'].values
        
        # Find the closest grid cell to EMD coordinates
        distance_squared = (lons - emd_lon)**2 + (lats - emd_lat)**2
        closest_indices = np.unravel_index(np.argmin(distance_squared), distance_squared.shape)
        closest_sn, closest_we = closest_indices
        
        # Get the actual coordinates of the closest grid cell
        closest_lon = lons[closest_sn, closest_we]
        closest_lat = lats[closest_sn, closest_we]
        closest_distance_deg = np.sqrt(distance_squared[closest_sn, closest_we])
        lat_correction = np.cos(np.radians(emd_lat))
        closest_distance_km = closest_distance_deg * 111.32 * lat_correction
        
        print(f"Closest NEWA grid cell:")
        print(f"  Grid indices: south_north={closest_sn}, west_east={closest_we}")
        print(f"  Grid coordinates: {closest_lon:.4f}°E, {closest_lat:.4f}°N")
        print(f"  Distance from EMD: {closest_distance_km:.2f} km")
        
        # Extract NEWA ice load data at specified height and closest grid cell
        newa_ice_load = dataset_with_ice_load['ICE_LOAD'].isel(height=height_idx, south_north=closest_sn, west_east=closest_we)
        
        # Convert to pandas DataFrame for easier manipulation
        newa_df = newa_ice_load.to_dataframe(name='ICE_LOAD').reset_index()
        newa_df['time'] = pd.to_datetime(newa_df['time'])
        newa_df = newa_df.set_index('time')
        
        # Prepare EMD data
        if not isinstance(emd_data.index, pd.DatetimeIndex):
            if 'time' in emd_data.columns:
                emd_df = emd_data.copy()
                emd_df['time'] = pd.to_datetime(emd_df['time'])
                emd_df = emd_df.set_index('time')
            else:
                raise ValueError("EMD data must have datetime index or 'time' column")
        else:
            emd_df = emd_data.copy()
        
        # Find common time period
        common_start = max(emd_df.index.min(), newa_df.index.min())
        common_end = min(emd_df.index.max(), newa_df.index.max())
        
        print(f"Common period: {common_start} to {common_end}")
        
        # Filter to common period
        emd_common = emd_df.loc[common_start:common_end, emd_column].copy()
        newa_common = newa_df.loc[common_start:common_end, 'ICE_LOAD'].copy()
        
        # Resample NEWA data to hourly to match EMD (from 30min to 1h)
        print("Resampling NEWA data from 30min to 1h resolution...")
        newa_hourly = newa_common.resample('1H').mean()
        
        # Align time indices
        common_times = emd_common.index.intersection(newa_hourly.index)
        emd_aligned = emd_common.loc[common_times]
        newa_aligned = newa_hourly.loc[common_times]
        
        # Remove NaN values and filter out non-icing months (June-October)
        valid_mask = ~(np.isnan(emd_aligned) | np.isnan(newa_aligned))
        emd_clean_all = emd_aligned[valid_mask]
        newa_clean_all = newa_aligned[valid_mask]
        
        # Filter out non-icing months (June=6, July=7, August=8, September=9, October=10)
        non_icing_months = [6, 7, 8, 9, 10]
        icing_mask = ~emd_clean_all.index.month.isin(non_icing_months)
        emd_clean = emd_clean_all[icing_mask]
        newa_clean = newa_clean_all[icing_mask]
        
        print(f"Valid data points after NaN removal: {len(emd_clean_all)}")
        print(f"Icing season data points (excluding Jun-Oct): {len(emd_clean)}")
        print(f"Excluded {len(emd_clean_all) - len(emd_clean)} non-icing season points")
        
        if len(emd_clean) < 10:
            print("Warning: Very few valid data points for analysis!")
            return None

        # Apply hourly threshold filtering
        if ice_load_threshold > 0:
            print(f"Applying hourly threshold filter (>= {ice_load_threshold} kg/m)...")
            hourly_threshold_mask = (emd_clean >= ice_load_threshold) & (newa_clean >= ice_load_threshold)
            emd_threshold_filtered = emd_clean[hourly_threshold_mask]
            newa_threshold_filtered = newa_clean[hourly_threshold_mask]
            
            print(f"Hours after threshold filter (>= {ice_load_threshold} kg/m): {len(emd_threshold_filtered)}")
            print(f"Excluded {len(emd_clean) - len(emd_threshold_filtered)} hours below threshold")
        else:
            print("No hourly threshold filter applied (ice_load_threshold = 0)")
            emd_threshold_filtered = emd_clean.copy()
            newa_threshold_filtered = newa_clean.copy()

        # Use threshold-filtered data for the rest of the analysis
        emd_clean = emd_threshold_filtered.copy()
        newa_clean = newa_threshold_filtered.copy()
        
        # Create DataFrames with time components for typical pattern analysis
        emd_df_analysis = pd.DataFrame({
            'ice_load': emd_clean.values,
            'datetime': emd_clean.index,
            'hour': emd_clean.index.hour,
            'day_of_week': emd_clean.index.dayofweek,  # Monday=0, Sunday=6
            'week_of_year': emd_clean.index.isocalendar().week,
            'month': emd_clean.index.month,
            'year': emd_clean.index.year,
            'date': emd_clean.index.date
        })
        
        newa_df_analysis = pd.DataFrame({
            'ice_load': newa_clean.values,
            'datetime': newa_clean.index,
            'hour': newa_clean.index.hour,
            'day_of_week': newa_clean.index.dayofweek,
            'week_of_year': newa_clean.index.isocalendar().week,
            'month': newa_clean.index.month,
            'year': newa_clean.index.year,
            'date': newa_clean.index.date
        })
        
        print(f"\nCalculating typical patterns...")
        
        # 1. TYPICAL DAY ANALYSIS (hourly averages for each day, then statistics across all days)
        print("1. Calculating typical day patterns...")
        
        # Calculate daily mean ice load using threshold-filtered data
        emd_daily_means = emd_threshold_filtered.resample('D').mean()
        newa_daily_means = newa_threshold_filtered.resample('D').mean()
        
        # Remove NaN values from daily means
        emd_daily_clean = emd_daily_means.dropna()
        newa_daily_clean = newa_daily_means.dropna()
        
        # Align daily data
        common_daily_dates = emd_daily_clean.index.intersection(newa_daily_clean.index)
        emd_daily_filtered = emd_daily_clean.loc[common_daily_dates]
        newa_daily_filtered = newa_daily_clean.loc[common_daily_dates]
        
        # Apply non-zero percentage filter for daily data
        if non_zero_percentage > 0:
            print(f"  Applying {non_zero_percentage}% non-zero filter to daily data...")
            daily_non_zero_mask = []
            
            for date in emd_daily_filtered.index:
                # Get hourly data for this specific day from both datasets
                day_start = date
                day_end = date + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
                
                # Extract hourly data for this day
                emd_day_hours = emd_threshold_filtered[(emd_threshold_filtered.index >= day_start) & (emd_threshold_filtered.index <= day_end)]
                newa_day_hours = newa_threshold_filtered[(newa_threshold_filtered.index >= day_start) & (newa_threshold_filtered.index <= day_end)]
                
                if len(emd_day_hours) > 0 and len(newa_day_hours) > 0:
                    # Calculate percentage of non-zero hours for both datasets
                    emd_nonzero_pct = (emd_day_hours > 0).mean() * 100
                    newa_nonzero_pct = (newa_day_hours > 0).mean() * 100
                    
                    # Include day if both datasets meet the non-zero percentage requirement
                    daily_non_zero_mask.append(emd_nonzero_pct >= non_zero_percentage and newa_nonzero_pct >= non_zero_percentage)
                else:
                    daily_non_zero_mask.append(False)
            
            daily_non_zero_mask = pd.Series(daily_non_zero_mask, index=emd_daily_filtered.index)
            emd_daily_aligned = emd_daily_filtered[daily_non_zero_mask]
            newa_daily_aligned = newa_daily_filtered[daily_non_zero_mask]
            
            print(f"  Daily means after non-zero filter ({non_zero_percentage}%): {len(emd_daily_aligned)} days")
        else:
            emd_daily_aligned = emd_daily_filtered
            newa_daily_aligned = newa_daily_filtered
        
        print(f"  Daily means before threshold filter: {len(emd_daily_temp)} days")
        print(f"  Daily means after threshold filter (>= {ice_load_threshold} kg/m): {len(emd_daily_filtered)} days")
        print(f"  Daily means final count: {len(emd_daily_aligned)} days")
        
        # 2. TYPICAL WEEK ANALYSIS (weekly averages for each week, then statistics across all weeks)
        print("2. Calculating typical week patterns...")
        
        # Calculate weekly mean ice load using threshold-filtered data
        emd_weekly_means = emd_threshold_filtered.resample('W').mean()
        newa_weekly_means = newa_threshold_filtered.resample('W').mean()
        
        # Remove NaN values from weekly means
        emd_weekly_clean = emd_weekly_means.dropna()
        newa_weekly_clean = newa_weekly_means.dropna()
        
        # Align weekly data
        common_weekly_dates = emd_weekly_clean.index.intersection(newa_weekly_clean.index)
        emd_weekly_filtered = emd_weekly_clean.loc[common_weekly_dates]
        newa_weekly_filtered = newa_weekly_clean.loc[common_weekly_dates]
        
        # Apply non-zero percentage filter for weekly data
        if non_zero_percentage > 0:
            print(f"  Applying {non_zero_percentage}% non-zero filter to weekly data...")
            weekly_non_zero_mask = []
            
            for week_end in emd_weekly_filtered.index:
                # Get hourly data for this specific week from both datasets
                week_start = week_end - pd.Timedelta(days=6, hours=23)  # 7 days total
                
                # Extract hourly data for this week
                emd_week_hours = emd_threshold_filtered[(emd_threshold_filtered.index >= week_start) & (emd_threshold_filtered.index <= week_end)]
                newa_week_hours = newa_threshold_filtered[(newa_threshold_filtered.index >= week_start) & (newa_threshold_filtered.index <= week_end)]
                
                if len(emd_week_hours) > 0 and len(newa_week_hours) > 0:
                    # Calculate percentage of non-zero hours for both datasets
                    emd_nonzero_pct = (emd_week_hours > 0).mean() * 100
                    newa_nonzero_pct = (newa_week_hours > 0).mean() * 100
                    
                    # Include week if both datasets meet the non-zero percentage requirement
                    weekly_non_zero_mask.append(emd_nonzero_pct >= non_zero_percentage and newa_nonzero_pct >= non_zero_percentage)
                else:
                    weekly_non_zero_mask.append(False)
            
            weekly_non_zero_mask = pd.Series(weekly_non_zero_mask, index=emd_weekly_filtered.index)
            emd_weekly_aligned = emd_weekly_filtered[weekly_non_zero_mask]
            newa_weekly_aligned = newa_weekly_filtered[weekly_non_zero_mask]
            
            print(f"  Weekly means after non-zero filter ({non_zero_percentage}%): {len(emd_weekly_aligned)} weeks")
        else:
            emd_weekly_aligned = emd_weekly_filtered
            newa_weekly_aligned = newa_weekly_filtered
        
        print(f"  Weekly means after threshold filter: {len(emd_weekly_filtered)} weeks")
        print(f"  Weekly means final count: {len(emd_weekly_aligned)} weeks")
        
        # 3. TYPICAL YEAR ANALYSIS (yearly averages for each year, then statistics across all years)
        print("3. Calculating typical year patterns...")
        
        # Calculate yearly mean ice load using threshold-filtered data
        emd_yearly_means = emd_threshold_filtered.resample('Y').mean()
        newa_yearly_means = newa_threshold_filtered.resample('Y').mean()
        
        # Remove NaN values from yearly means
        emd_yearly_clean = emd_yearly_means.dropna()
        newa_yearly_clean = newa_yearly_means.dropna()
        
        # Align yearly data
        common_yearly_dates = emd_yearly_clean.index.intersection(newa_yearly_clean.index)
        emd_yearly_filtered = emd_yearly_clean.loc[common_yearly_dates]
        newa_yearly_filtered = newa_yearly_clean.loc[common_yearly_dates]
        
        # Apply non-zero percentage filter for yearly data
        if non_zero_percentage > 0:
            print(f"  Applying {non_zero_percentage}% non-zero filter to yearly data...")
            yearly_non_zero_mask = []
            
            for year_end in emd_yearly_filtered.index:
                # Get hourly data for this specific year from both datasets
                year_start = year_end.replace(month=1, day=1, hour=0)
                year_end_actual = year_end.replace(month=12, day=31, hour=23)
                
                # Extract hourly data for this year
                emd_year_hours = emd_threshold_filtered[(emd_threshold_filtered.index >= year_start) & (emd_threshold_filtered.index <= year_end_actual)]
                newa_year_hours = newa_threshold_filtered[(newa_threshold_filtered.index >= year_start) & (newa_threshold_filtered.index <= year_end_actual)]
                
                if len(emd_year_hours) > 0 and len(newa_year_hours) > 0:
                    # Calculate percentage of non-zero hours for both datasets
                    emd_nonzero_pct = (emd_year_hours > 0).mean() * 100
                    newa_nonzero_pct = (newa_year_hours > 0).mean() * 100
                    
                    # Include year if both datasets meet the non-zero percentage requirement
                    yearly_non_zero_mask.append(emd_nonzero_pct >= non_zero_percentage and newa_nonzero_pct >= non_zero_percentage)
                else:
                    yearly_non_zero_mask.append(False)
            
            yearly_non_zero_mask = pd.Series(yearly_non_zero_mask, index=emd_yearly_filtered.index)
            emd_yearly_aligned = emd_yearly_filtered[yearly_non_zero_mask]
            newa_yearly_aligned = newa_yearly_filtered[yearly_non_zero_mask]
            
            print(f"  Yearly means after non-zero filter ({non_zero_percentage}%): {len(emd_yearly_aligned)} years")
        else:
            emd_yearly_aligned = emd_yearly_filtered
            newa_yearly_aligned = newa_yearly_filtered
        
        print(f"  Yearly means after threshold filter: {len(emd_yearly_filtered)} years")
        print(f"  Yearly means final count: {len(emd_yearly_aligned)} years")
        
        # Check if we have sufficient data after threshold filtering
        if len(emd_daily_aligned) == 0:
            print(f"Warning: No daily data above threshold {ice_load_threshold} kg/m. Consider lowering the threshold.")
            return None
        if len(emd_weekly_aligned) == 0:
            print(f"Warning: No weekly data above threshold {ice_load_threshold} kg/m. Consider lowering the threshold.")
        if len(emd_yearly_aligned) == 0:
            print(f"Warning: No yearly data above threshold {ice_load_threshold} kg/m. Consider lowering the threshold.")
        
        # Calculate summary statistics for each temporal scale
        def calculate_stats(data):
            return {
                'count': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'q25': data.quantile(0.25),
                'median': data.median(),
                'q75': data.quantile(0.75),
                'max': data.max()
            }
        
        # Statistics for each temporal scale
        emd_daily_stats = calculate_stats(emd_daily_aligned)
        newa_daily_stats = calculate_stats(newa_daily_aligned)
        
        emd_weekly_stats = calculate_stats(emd_weekly_aligned)
        newa_weekly_stats = calculate_stats(newa_weekly_aligned)
        
        emd_yearly_stats = calculate_stats(emd_yearly_aligned)
        newa_yearly_stats = calculate_stats(newa_yearly_aligned)
        
        print(f"\n=== TYPICAL PATTERNS STATISTICS ===")
        print(f"Filters Applied:")
        print(f"  Ice Load Threshold: >={ice_load_threshold} kg/m")
        print(f"  Non-Zero Percentage: >={non_zero_percentage}% hours > 0")
        print(f"Daily means - EMD: {emd_daily_stats['mean']:.4f} +/- {emd_daily_stats['std']:.4f} kg/m (n={emd_daily_stats['count']})")
        print(f"Daily means - NEWA: {newa_daily_stats['mean']:.4f} +/- {newa_daily_stats['std']:.4f} kg/m (n={newa_daily_stats['count']})")
        print(f"Weekly means - EMD: {emd_weekly_stats['mean']:.4f} +/- {emd_weekly_stats['std']:.4f} kg/m (n={emd_weekly_stats['count']})")
        print(f"Weekly means - NEWA: {newa_weekly_stats['mean']:.4f} +/- {newa_weekly_stats['std']:.4f} kg/m (n={newa_weekly_stats['count']})")
        print(f"Yearly means - EMD: {emd_yearly_stats['mean']:.4f} +/- {emd_yearly_stats['std']:.4f} kg/m (n={emd_yearly_stats['count']})")
        print(f"Yearly means - NEWA: {newa_yearly_stats['mean']:.4f} +/- {newa_yearly_stats['std']:.4f} kg/m (n={newa_yearly_stats['count']})")
        
        # Create plots if requested
        if save_plots:
            print(f"\nCreating typical patterns box plots...")
            
            # Create directory structure with threshold and non-zero percentage information
            threshold_str = f"threshold_{ice_load_threshold:.3f}" if ice_load_threshold > 0 else "no_threshold"
            nonzero_str = f"nonzero_{non_zero_percentage:.0f}pct" if non_zero_percentage > 0 else "no_nonzero_filter"
            base_dir = os.path.join("results", "figures", "EMD", "Ice_Load", "MeanDWM", f"{height:.0f}m_{threshold_str}_{nonzero_str}")
            os.makedirs(base_dir, exist_ok=True)
            
            # Create comprehensive box plot comparison
            fig, axes = plt.subplots(1, 3, figsize=(18, 8))
            
            # Color scheme
            colors = ['steelblue', 'orange']
            labels = ['EMD', 'NEWA']
            
            # Plot 1: Daily means box plot
            ax1 = axes[0]
            daily_data = [emd_daily_aligned.values, newa_daily_aligned.values]
            
            box_plot_daily = ax1.boxplot(daily_data, labels=labels, patch_artist=True,
                                       showmeans=True, meanline=True,
                                       boxprops=dict(alpha=0.7),
                                       medianprops=dict(color='red', linewidth=2),
                                       meanprops=dict(color='black', linewidth=2, linestyle='--'))
            
            # Color the boxes
            for patch, color in zip(box_plot_daily['boxes'], colors):
                patch.set_facecolor(color)
            
            ax1.set_ylabel('Mean Daily Ice Load [kg/m]\n(Mean of Hourly Values)', fontsize=20, fontweight='bold')
            ax1.set_title(f'Distribution of Daily Means of Hourly Ice Load (n={len(emd_daily_aligned)})', fontsize=28, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Weekly means box plot
            ax2 = axes[1]
            weekly_data = [emd_weekly_aligned.values, newa_weekly_aligned.values]
            
            box_plot_weekly = ax2.boxplot(weekly_data, labels=labels, patch_artist=True,
                                        showmeans=True, meanline=True,
                                        boxprops=dict(alpha=0.7),
                                        medianprops=dict(color='red', linewidth=2),
                                        meanprops=dict(color='black', linewidth=2, linestyle='--'))
            
            # Color the boxes
            for patch, color in zip(box_plot_weekly['boxes'], colors):
                patch.set_facecolor(color)
            
            ax2.set_ylabel('Mean Weekly Ice Load [kg/m]\n(Mean of Hourly Values)', fontsize=20, fontweight='bold')
            ax2.set_title(f'Distribution of Weekly Means of Hourly Ice Load (n={len(emd_weekly_aligned)})', fontsize=28, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Yearly means box plot
            ax3 = axes[2]
            yearly_data = [emd_yearly_aligned.values, newa_yearly_aligned.values]
            
            box_plot_yearly = ax3.boxplot(yearly_data, labels=labels, patch_artist=True,
                                        showmeans=True, meanline=True,
                                        boxprops=dict(alpha=0.7),
                                        medianprops=dict(color='red', linewidth=2),
                                        meanprops=dict(color='black', linewidth=2, linestyle='--'))
            
            # Color the boxes
            for patch, color in zip(box_plot_yearly['boxes'], colors):
                patch.set_facecolor(color)
            
            ax3.set_ylabel('Mean Yearly Ice Load [kg/m]\n(Mean of Hourly Values)', fontsize=20, fontweight='bold')
            ax3.set_title(f'Distribution of Yearly Means of Hourly Ice Load (n={len(emd_yearly_aligned)})', fontsize=28, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add overall title with filtering information
            threshold_text = f"Ice Load Threshold: >={ice_load_threshold} kg/m" if ice_load_threshold > 0 else "No Ice Load Threshold"
            nonzero_text = f"Non-Zero Filter: >={non_zero_percentage}% hours > 0" if non_zero_percentage > 0 else "No Non-Zero Filter"
            
            fig.suptitle(f'Typical Patterns Analysis: EMD vs NEWA at {height}m (Icing Season Only)\n'
                        f'{threshold_text} | {nonzero_text}',
                        fontsize=28, fontweight='bold', y=0.96)
            
            # Add legend explaining box plot elements
            legend_text = 'Box Plot Elements:\n━ Red line: Median\n┅ Black line: Mean\n□ Box: Q25-Q75 (IQR)\n┬ Whiskers: 1.5×IQR\n○ Outliers'
            fig.text(0.02, 0.02, legend_text, fontsize=16, 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                    verticalalignment='bottom')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.80, bottom=0.15)  # Move plots down to prevent overlap with title
            
            typical_patterns_path = os.path.join(base_dir, f'typical_patterns_comparison_{height:.0f}m.png')
            plt.savefig(typical_patterns_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {typical_patterns_path}")
            
            # Create detailed statistics table plot
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            ax.axis('off')
            
            # Create statistics table
            stats_table_data = []
            
            # Daily statistics
            stats_table_data.append(['Temporal Scale', 'Dataset', 'Count', 'Mean', 'Std', 'Min', 'Q25', 'Median', 'Q75', 'Max'])
            stats_table_data.append(['Daily', 'EMD', f"{emd_daily_stats['count']}", f"{emd_daily_stats['mean']:.4f}", 
                                   f"{emd_daily_stats['std']:.4f}", f"{emd_daily_stats['min']:.4f}", 
                                   f"{emd_daily_stats['q25']:.4f}", f"{emd_daily_stats['median']:.4f}", 
                                   f"{emd_daily_stats['q75']:.4f}", f"{emd_daily_stats['max']:.4f}"])
            stats_table_data.append(['', 'NEWA', f"{newa_daily_stats['count']}", f"{newa_daily_stats['mean']:.4f}", 
                                   f"{newa_daily_stats['std']:.4f}", f"{newa_daily_stats['min']:.4f}", 
                                   f"{newa_daily_stats['q25']:.4f}", f"{newa_daily_stats['median']:.4f}", 
                                   f"{newa_daily_stats['q75']:.4f}", f"{newa_daily_stats['max']:.4f}"])
            
            # Weekly statistics
            stats_table_data.append(['Weekly', 'EMD', f"{emd_weekly_stats['count']}", f"{emd_weekly_stats['mean']:.4f}", 
                                   f"{emd_weekly_stats['std']:.4f}", f"{emd_weekly_stats['min']:.4f}", 
                                   f"{emd_weekly_stats['q25']:.4f}", f"{emd_weekly_stats['median']:.4f}", 
                                   f"{emd_weekly_stats['q75']:.4f}", f"{emd_weekly_stats['max']:.4f}"])
            stats_table_data.append(['', 'NEWA', f"{newa_weekly_stats['count']}", f"{newa_weekly_stats['mean']:.4f}", 
                                   f"{newa_weekly_stats['std']:.4f}", f"{newa_weekly_stats['min']:.4f}", 
                                   f"{newa_weekly_stats['q25']:.4f}", f"{newa_weekly_stats['median']:.4f}", 
                                   f"{newa_weekly_stats['q75']:.4f}", f"{newa_weekly_stats['max']:.4f}"])
            
            # Yearly statistics
            stats_table_data.append(['Yearly', 'EMD', f"{emd_yearly_stats['count']}", f"{emd_yearly_stats['mean']:.4f}", 
                                   f"{emd_yearly_stats['std']:.4f}", f"{emd_yearly_stats['min']:.4f}", 
                                   f"{emd_yearly_stats['q25']:.4f}", f"{emd_yearly_stats['median']:.4f}", 
                                   f"{emd_yearly_stats['q75']:.4f}", f"{emd_yearly_stats['max']:.4f}"])
            stats_table_data.append(['', 'NEWA', f"{newa_yearly_stats['count']}", f"{newa_yearly_stats['mean']:.4f}", 
                                   f"{newa_yearly_stats['std']:.4f}", f"{newa_yearly_stats['min']:.4f}", 
                                   f"{newa_yearly_stats['q25']:.4f}", f"{newa_yearly_stats['median']:.4f}", 
                                   f"{newa_yearly_stats['q75']:.4f}", f"{newa_yearly_stats['max']:.4f}"])
            
            # Create table
            table = ax.table(cellText=stats_table_data[1:], colLabels=stats_table_data[0], 
                           cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(16)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(stats_table_data)):
                for j in range(len(stats_table_data[0])):
                    cell = table[(i, j)]
                    if i == 0:  # Header row
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    elif j == 1 and i > 0 and stats_table_data[i][j] == 'EMD':  # EMD rows
                        cell.set_facecolor('#E3F2FD')
                    elif j == 1 and i > 0 and stats_table_data[i][j] == 'NEWA':  # NEWA rows
                        cell.set_facecolor('#FFF3E0')
            
            threshold_text = f"Ice Load Threshold: >={ice_load_threshold} kg/m" if ice_load_threshold > 0 else "No Ice Load Threshold"
            nonzero_text = f"Non-Zero Filter: >={non_zero_percentage}% hours > 0" if non_zero_percentage > 0 else "No Non-Zero Filter"
            
            ax.set_title(f'Typical Patterns Statistics Summary at {height}m\n'
                        f'{threshold_text} | {nonzero_text}',
                        fontsize=28, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            stats_table_path = os.path.join(base_dir, f'typical_patterns_statistics_{height:.0f}m.png')
            plt.savefig(stats_table_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {stats_table_path}")
            
            print(f"\n=== PLOT SUMMARY ===")
            print(f"Created 2 plots:")
            print(f"  1. Typical patterns box plot comparison: {typical_patterns_path}")
            print(f"  2. Detailed statistics table: {stats_table_path}")
        
        # Prepare results dictionary
        results = {
            'height': height,
            'ice_load_threshold': ice_load_threshold,
            'non_zero_percentage': non_zero_percentage,
            'emd_coordinates': {'longitude': emd_lon, 'latitude': emd_lat},
            'newa_grid_cell': {
                'south_north_index': int(closest_sn),
                'west_east_index': int(closest_we),
                'longitude': float(closest_lon),
                'latitude': float(closest_lat),
                'distance_from_emd_km': float(closest_distance_km)
            },
            'typical_patterns': {
                'daily': {
                    'emd_stats': emd_daily_stats,
                    'newa_stats': newa_daily_stats,
                    'emd_data': emd_daily_aligned,
                    'newa_data': newa_daily_aligned
                },
                'weekly': {
                    'emd_stats': emd_weekly_stats,
                    'newa_stats': newa_weekly_stats,
                    'emd_data': emd_weekly_aligned,
                    'newa_data': newa_weekly_aligned
                },
                'yearly': {
                    'emd_stats': emd_yearly_stats,
                    'newa_stats': newa_yearly_stats,
                    'emd_data': emd_yearly_aligned,
                    'newa_data': newa_yearly_aligned
                }
            }
        }
        
        print(f"\n✓ Typical patterns analysis completed successfully!")
        print(f"Results saved to: {base_dir}")
        
        return results
        
    except Exception as e:
        print(f"Error in typical patterns analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def pdf_emd_newa(emd_data, dataset_with_ice_load, height, emd_coordinates, save_plots=True, ice_load_threshold=0.0, non_zero_percentage=0.0):
    """
    Generate probability density function plots comparing EMD observations and NEWA model ice load distributions.
    
    Parameters:
    -----------
    emd_data : pandas.DataFrame
        EMD observational data containing ice load columns (MIce.50, MIce.100, MIce.150)
    dataset_with_ice_load : xarray.Dataset
        NEWA model dataset containing ICE_LOAD variable
    height : int
        Height level to compare (50, 100, or 150 meters)
    emd_coordinates : tuple
        EMD coordinates as (longitude, latitude) in degrees
    save_plots : bool, optional
        Whether to save plots to file (default: True)
    ice_load_threshold : float, optional
        Minimum mean hourly ice load threshold (kg/m). Only temporal periods where both datasets
        have mean hourly ice load >= threshold are included in analysis (default: 0.0)
    non_zero_percentage : float, optional
        Minimum percentage (0-100) of hours that must be > 0 in both datasets for a temporal period
        to be included in analysis. Applied to daily, weekly, and yearly aggregations (default: 0.0)
        
    Returns:
    --------
    dict
        Dictionary containing PDF analysis results and statistics
    """
    
    print(f"=== PROBABILITY DENSITY FUNCTION ANALYSIS: EMD vs NEWA at {height}m ===")
    print(f"Ice load threshold: {ice_load_threshold} kg/m (minimum mean hourly ice load for inclusion)")
    print(f"Non-zero percentage threshold: {non_zero_percentage}% (minimum percentage of hours > 0 required)")
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import os
        from scipy import stats
        from sklearn.neighbors import KernelDensity
        
        # Validate height input
        if height not in [50, 100, 150]:
            raise ValueError(f"Height must be 50, 100, or 150 meters. Got: {height}")
        
        # Check if EMD data contains required column
        emd_column = f"MIce.{int(height)}"
        if emd_column not in emd_data.columns:
            available_ice_cols = [col for col in emd_data.columns if 'ice' in col.lower()]
            raise ValueError(f"Column '{emd_column}' not found in EMD data. Available ice columns: {available_ice_cols}")
        
        # Verify NEWA dataset height
        if 'ICE_LOAD' not in dataset_with_ice_load.data_vars:
            raise ValueError("'ICE_LOAD' variable not found in NEWA dataset")
        
        # Get height information from NEWA dataset
        height_levels = dataset_with_ice_load.height.values
        height_idx = None
        for i, h in enumerate(height_levels):
            if abs(h - height) < 1:  # Allow 1m tolerance
                height_idx = i
                break
        
        if height_idx is None:
            raise ValueError(f"Height {height}m not found in NEWA dataset. Available heights: {height_levels}")
        
        print(f"Using NEWA height level {height_idx} ({height_levels[height_idx]}m)")
        
        # Get EMD coordinates and find closest NEWA grid cell
        emd_lon, emd_lat = emd_coordinates
        print(f"EMD coordinates: {emd_lon:.4f}°E, {emd_lat:.4f}°N")
        
        # Extract coordinates - handle both data variables and coordinates
        if 'XLAT' in dataset_with_ice_load.coords:
            lats = dataset_with_ice_load.coords['XLAT'].values
            lons = dataset_with_ice_load.coords['XLON'].values
        else:
            lats = dataset_with_ice_load['XLAT'].values
            lons = dataset_with_ice_load['XLON'].values
        
        # Find the closest grid cell to EMD coordinates
        distance_squared = (lons - emd_lon)**2 + (lats - emd_lat)**2
        closest_indices = np.unravel_index(np.argmin(distance_squared), distance_squared.shape)
        closest_sn, closest_we = closest_indices
        
        # Get the actual coordinates of the closest grid cell
        closest_lon = lons[closest_sn, closest_we]
        closest_lat = lats[closest_sn, closest_we]
        closest_distance_deg = np.sqrt(distance_squared[closest_sn, closest_we])
        lat_correction = np.cos(np.radians(emd_lat))
        closest_distance_km = closest_distance_deg * 111.32 * lat_correction
        
        print(f"Closest NEWA grid cell:")
        print(f"  Grid indices: south_north={closest_sn}, west_east={closest_we}")
        print(f"  Grid coordinates: {closest_lon:.4f}°E, {closest_lat:.4f}°N")
        print(f"  Distance from EMD: {closest_distance_km:.2f} km")
        
        # Extract NEWA ice load data at specified height and closest grid cell
        newa_ice_load = dataset_with_ice_load['ICE_LOAD'].isel(height=height_idx, south_north=closest_sn, west_east=closest_we)
        
        # Convert to pandas DataFrame for easier manipulation
        newa_df = newa_ice_load.to_dataframe(name='ICE_LOAD').reset_index()
        newa_df['time'] = pd.to_datetime(newa_df['time'])
        newa_df = newa_df.set_index('time')
        
        # Prepare EMD data
        if not isinstance(emd_data.index, pd.DatetimeIndex):
            if 'time' in emd_data.columns:
                emd_df = emd_data.copy()
                emd_df['time'] = pd.to_datetime(emd_df['time'])
                emd_df = emd_df.set_index('time')
            else:
                raise ValueError("EMD data must have datetime index or 'time' column")
        else:
            emd_df = emd_data.copy()
        
        # Find common time period
        common_start = max(emd_df.index.min(), newa_df.index.min())
        common_end = min(emd_df.index.max(), newa_df.index.max())
        
        print(f"Common period: {common_start} to {common_end}")
        
        # Filter to common period
        emd_common = emd_df.loc[common_start:common_end, emd_column].copy()
        newa_common = newa_df.loc[common_start:common_end, 'ICE_LOAD'].copy()
        
        # Resample NEWA data to hourly to match EMD (from 30min to 1h)
        print("Resampling NEWA data from 30min to 1h resolution...")
        newa_hourly = newa_common.resample('1H').mean()
        
        # Align time indices
        common_times = emd_common.index.intersection(newa_hourly.index)
        emd_aligned = emd_common.loc[common_times]
        newa_aligned = newa_hourly.loc[common_times]
        
        # Remove NaN values and filter out non-icing months (June-October)
        valid_mask = ~(np.isnan(emd_aligned) | np.isnan(newa_aligned))
        emd_clean_all = emd_aligned[valid_mask]
        newa_clean_all = newa_aligned[valid_mask]
        
        # Filter out non-icing months (June=6, July=7, August=8, September=9, October=10)
        non_icing_months = [6, 7, 8, 9, 10]
        icing_mask = ~emd_clean_all.index.month.isin(non_icing_months)
        emd_clean = emd_clean_all[icing_mask]
        newa_clean = newa_clean_all[icing_mask]
        
        print(f"Valid data points after NaN removal: {len(emd_clean_all)}")
        print(f"Icing season data points (excluding Jun-Oct): {len(emd_clean)}")
        print(f"Excluded {len(emd_clean_all) - len(emd_clean)} non-icing season points")
        
        if len(emd_clean) < 10:
            print("Warning: Very few valid data points for analysis!")
            return None
        
        # Apply hourly threshold filtering
        if ice_load_threshold > 0:
            print(f"Applying hourly threshold filter (>= {ice_load_threshold} kg/m)...")
            hourly_threshold_mask = (emd_clean >= ice_load_threshold) & (newa_clean >= ice_load_threshold)
            emd_threshold_filtered = emd_clean[hourly_threshold_mask]
            newa_threshold_filtered = newa_clean[hourly_threshold_mask]
            
            print(f"Hours after threshold filter (>= {ice_load_threshold} kg/m): {len(emd_threshold_filtered)}")
            print(f"Excluded {len(emd_clean) - len(emd_threshold_filtered)} hours below threshold")
        else:
            print("No hourly threshold filter applied (ice_load_threshold = 0)")
            emd_threshold_filtered = emd_clean.copy()
            newa_threshold_filtered = newa_clean.copy()

        # Use threshold-filtered data for the rest of the analysis
        emd_clean = emd_threshold_filtered.copy()
        newa_clean = newa_threshold_filtered.copy()
        
        print(f"Final hourly data points for analysis: {len(emd_clean)}")
        
        if len(emd_clean) < 10:
            print("Warning: Very few valid data points after filtering!")
            return None
        
        # Calculate basic statistics
        def calculate_stats(data, name):
            stats_dict = {
                'count': len(data),
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'q25': np.percentile(data, 25),
                'median': np.median(data),
                'q75': np.percentile(data, 75),
                'max': np.max(data),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
            print(f"{name} statistics: mean={stats_dict['mean']:.4f}, std={stats_dict['std']:.4f}, skew={stats_dict['skewness']:.3f}")
            return stats_dict
        
        emd_stats = calculate_stats(emd_clean.values, "EMD")
        newa_stats = calculate_stats(newa_clean.values, "NEWA")
        
        # Create plots if requested
        if save_plots:
            print(f"\nCreating PDF plots...")
            
            # Create directory structure
            threshold_str = f"threshold_{ice_load_threshold:.3f}" if ice_load_threshold > 0 else "no_threshold"
            nonzero_str = f"nonzero_{non_zero_percentage:.0f}pct" if non_zero_percentage > 0 else "no_nonzero_filter"
            base_dir = os.path.join("results", "figures", "EMD", "Ice_Load", "pdf_EMD_NEWA", f"{height:.0f}m_{threshold_str}_{nonzero_str}")
            os.makedirs(base_dir, exist_ok=True)
            
            # Define common range for plotting
            emd_final = emd_clean.values
            newa_final = newa_clean.values
            data_min = min(np.min(emd_final), np.min(newa_final))
            data_max = max(np.max(emd_final), np.max(newa_final))
            x_range = np.linspace(data_min, data_max, 1000)

            threshold_text = f"Ice Load Threshold: >={ice_load_threshold} kg/m" if ice_load_threshold > 0 else "No Ice Load Threshold"
            nonzero_text = f"Non-Zero Filter: >={non_zero_percentage}% hours > 0" if non_zero_percentage > 0 else "No Non-Zero Filter"
            common_subtitle = f'{threshold_text} | {nonzero_text}'
            
            # Plot 1: Histogram PDF comparison
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
            bins = np.linspace(data_min, data_max, 50)
            ax1.hist(emd_final, bins=bins, alpha=0.6, density=True, color='steelblue', edgecolor='darkblue', linewidth=1, label=f'EMD (n={len(emd_final)})')
            ax1.hist(newa_final, bins=bins, alpha=0.6, density=True, color='orange', edgecolor='darkorange', linewidth=1, label=f'NEWA (n={len(newa_final)})')
            ax1.set_xlabel('Ice Load [kg/m]', fontweight='bold', fontsize=20)
            ax1.set_ylabel('Probability Density', fontweight='bold', fontsize=20)
            ax1.set_title(f'Probability Density Functions\n{common_subtitle}', fontsize=26)
            ax1.legend(fontsize=20)
            ax1.tick_params(labelsize=20)
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            hist_path = os.path.join(base_dir, f'pdf_histogram_{height:.0f}m.png')
            plt.savefig(hist_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {hist_path}")

            # Plot 2: KDE comparison
            fig, ax2 = plt.subplots(1, 1, figsize=(12, 8))
            if len(emd_final) > 10:
                kde_emd = stats.gaussian_kde(emd_final)
                ax2.plot(x_range, kde_emd(x_range), 'steelblue', linewidth=3, label=f'EMD (μ={emd_stats["mean"]:.3f}, σ={emd_stats["std"]:.3f})')
            if len(newa_final) > 10:
                kde_newa = stats.gaussian_kde(newa_final)
                ax2.plot(x_range, kde_newa(x_range), 'orange', linewidth=3, label=f'NEWA (μ={newa_stats["mean"]:.3f}, σ={newa_stats["std"]:.3f})')
            ax2.axvline(emd_stats['mean'], color='steelblue', linestyle='--', alpha=0.8, label='EMD Mean')
            ax2.axvline(newa_stats['mean'], color='orange', linestyle='--', alpha=0.8, label='NEWA Mean')
            ax2.set_xlabel('Ice Load [kg/m]', fontweight='bold', fontsize=20)
            ax2.set_ylabel('Probability Density', fontweight='bold', fontsize=20)
            ax2.set_title(f'Kernel Density Estimation (KDE) Comparison\n{common_subtitle}', fontsize=26)
            ax2.legend(fontsize=20)
            ax2.tick_params(labelsize=20)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            kde_path = os.path.join(base_dir, f'pdf_kde_{height:.0f}m.png')
            plt.savefig(kde_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {kde_path}")

            # Plot 3: Log-log PDF comparison
            fig, ax3 = plt.subplots(1, 1, figsize=(12, 8))
            emd_nonzero = emd_final[emd_final > 0]
            newa_nonzero = newa_final[newa_final > 0]
            
            if len(emd_nonzero) > 10 and len(newa_nonzero) > 10:
                log_min = max(1e-6, min(np.min(emd_nonzero), np.min(newa_nonzero)))
                log_max = max(np.max(emd_nonzero), np.max(newa_nonzero))
                bins = np.logspace(np.log10(log_min), np.log10(log_max), 50)
                
                emd_counts, _ = np.histogram(emd_nonzero, bins=bins)
                newa_counts, _ = np.histogram(newa_nonzero, bins=bins)
                
                bin_widths = np.diff(bins)
                emd_pdf = emd_counts / (len(emd_nonzero) * bin_widths)
                newa_pdf = newa_counts / (len(newa_nonzero) * bin_widths)
                
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                emd_nonzero_pdf = emd_pdf > 0
                newa_nonzero_pdf = newa_pdf > 0
                
                ax3.loglog(bin_centers[emd_nonzero_pdf], emd_pdf[emd_nonzero_pdf], 'o-', 
                          color='steelblue', linewidth=2, markersize=4, alpha=0.8,
                          label=f'EMD PDF (n={len(emd_nonzero)})')
                ax3.loglog(bin_centers[newa_nonzero_pdf], newa_pdf[newa_nonzero_pdf], 's-', 
                          color='orange', linewidth=2, markersize=4, alpha=0.8,
                          label=f'NEWA PDF (n={len(newa_nonzero)})')
                ax3.axvline(np.mean(emd_nonzero), color='steelblue', linestyle='--', alpha=0.8, label='EMD Mean')
                ax3.axvline(np.mean(newa_nonzero), color='orange', linestyle='--', alpha=0.8, label='NEWA Mean')
            else:
                ax3.text(0.5, 0.5, 'Insufficient non-zero data\nfor log-log PDF', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=16)
            
            ax3.set_xlabel('Ice Load [kg/m]', fontweight='bold', fontsize=20)
            ax3.set_ylabel('Probability Density', fontweight='bold', fontsize=20)
            ax3.set_title(f'PDF Comparison (Log-Log Scale)\n{common_subtitle}', fontsize=26)
            ax3.legend(fontsize=20)
            ax3.tick_params(labelsize=20)
            ax3.grid(True, alpha=0.3, which="both")
            plt.tight_layout()
            loglog_path = os.path.join(base_dir, f'pdf_loglog_{height:.0f}m.png')
            plt.savefig(loglog_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {loglog_path}")

            # Plot 4: Q-Q plot
            fig, ax4 = plt.subplots(1, 1, figsize=(12, 8))
            n_quantiles = min(len(emd_final), len(newa_final), 1000)
            quantiles = np.linspace(0.01, 0.99, n_quantiles)
            emd_quantiles = np.quantile(emd_final, quantiles)
            newa_quantiles = np.quantile(newa_final, quantiles)
            ax4.scatter(emd_quantiles, newa_quantiles, alpha=0.6, s=20, color='purple')
            min_val = min(np.min(emd_quantiles), np.min(newa_quantiles))
            max_val = max(np.max(emd_quantiles), np.max(newa_quantiles))
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Agreement')
            qq_correlation = np.corrcoef(emd_quantiles, newa_quantiles)[0, 1]
            ax4.set_xlabel('EMD Quantiles [kg/m]', fontweight='bold', fontsize=20)
            ax4.set_ylabel('NEWA Quantiles [kg/m]', fontweight='bold', fontsize=20)
            ax4.set_title(f'Q-Q Plot (r = {qq_correlation:.3f})\n{common_subtitle}', fontsize=26)
            ax4.legend(fontsize=20)
            ax4.tick_params(labelsize=20)
            ax4.grid(True, alpha=0.3)
            plt.tight_layout()
            qq_path = os.path.join(base_dir, f'pdf_qqplot_{height:.0f}m.png')
            plt.savefig(qq_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {qq_path}")

            # Plot 5: Box plots
            fig, ax5 = plt.subplots(1, 1, figsize=(12, 8))
            box_data = [emd_final, newa_final]
            labels = ['EMD', 'NEWA']
            colors = ['steelblue', 'orange']
            box_plot = ax5.boxplot(box_data, labels=labels, patch_artist=True, showmeans=True, meanline=True, boxprops=dict(alpha=0.7), medianprops=dict(color='red', linewidth=2), meanprops=dict(color='black', linewidth=2, linestyle='--'))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            ax5.set_ylabel('Ice Load [kg/m]', fontweight='bold', fontsize=20)
            ax5.set_title(f'Distribution Comparison (Box Plots) at {height}m\n{common_subtitle}', fontweight='bold', fontsize=28)
            ax5.tick_params(labelsize=20)
            ax5.grid(True, alpha=0.3)
            plt.tight_layout()
            boxplot_path = os.path.join(base_dir, f'pdf_boxplot_{height:.0f}m.png')
            plt.savefig(boxplot_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {boxplot_path}")

            # Plot 6: Statistical summary table
            fig, ax6 = plt.subplots(1, 1, figsize=(12, 8))
            ax6.axis('off')
            
            stats_data = [
                ['Statistic', 'EMD', 'NEWA'],
                ['Count', f"{emd_stats['count']}", f"{newa_stats['count']}"],
                ['Mean', f"{emd_stats['mean']:.3f}", f"{newa_stats['mean']:.3f}"],
                ['Std Dev', f"{emd_stats['std']:.3f}", f"{newa_stats['std']:.3f}"],
                ['Minimum', f"{emd_stats['min']:.3f}", f"{newa_stats['min']:.3f}"],
                ['Q25', f"{emd_stats['q25']:.3f}", f"{newa_stats['q25']:.3f}"],
                ['Median', f"{emd_stats['median']:.3f}", f"{newa_stats['median']:.3f}"],
                ['Q75', f"{emd_stats['q75']:.3f}", f"{newa_stats['q75']:.3f}"],
                ['Maximum', f"{emd_stats['max']:.3f}", f"{newa_stats['max']:.3f}"],
                ['Skewness', f"{emd_stats['skewness']:.3f}", f"{newa_stats['skewness']:.3f}"],
                ['Kurtosis', f"{emd_stats['kurtosis']:.3f}", f"{newa_stats['kurtosis']:.3f}"],
                ['Q-Q Corr', f"{qq_correlation:.3f}", '-']
            ]
            
            table = ax6.table(cellText=stats_data[1:], colLabels=stats_data[0], 
                             cellLoc='center', loc='center', bbox=[0.05, 0.1, 0.9, 0.8])
            table.auto_set_font_size(False)
            table.set_fontsize(16)
            table.scale(1, 1.5)
            
            for i in range(len(stats_data)):
                for j in range(len(stats_data[0])):
                    cell = table[(i, j)]
                    if i == 0:  # Header row
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    elif j == 0:  # EMD column
                        cell.set_facecolor('#E3F2FD')
                    elif j == 1:  # NEWA column
                        cell.set_facecolor('#FFF3E0')

            ax6.set_title(f'Statistical Summary at {height}m\n{common_subtitle}', fontweight='bold', pad=20, fontsize=28)
            plt.tight_layout()
            stats_table_path = os.path.join(base_dir, f'pdf_stats_table_{height:.0f}m.png')
            plt.savefig(stats_table_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {stats_table_path}")
            
            print(f"\n=== PLOT SUMMARY ===")
            print(f"Created 6 individual plots:")
            print(f"  1. Histogram PDF: {hist_path}")
            print(f"  2. KDE comparison: {kde_path}")
            print(f"  3. Log-log PDF: {loglog_path}")
            print(f"  4. Q-Q plot: {qq_path}")
            print(f"  5. Box plots: {boxplot_path}")
            print(f"  6. Statistics table: {stats_table_path}")
        
        # Perform statistical tests
        print(f"\n=== STATISTICAL TESTS ===")
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_p_value = stats.ks_2samp(emd_final, newa_final)
        print(f"Kolmogorov-Smirnov test:")
        print(f"  Statistic: {ks_statistic:.4f}")
        print(f"  P-value: {ks_p_value:.4f}")
        print(f"  Interpretation: {'Distributions are significantly different' if ks_p_value < 0.05 else 'No significant difference in distributions'}")
        
        # Mann-Whitney U test (non-parametric)
        mw_statistic, mw_p_value = stats.mannwhitneyu(emd_final, newa_final, alternative='two-sided')
        print(f"\nMann-Whitney U test:")
        print(f"  Statistic: {mw_statistic:.0f}")
        print(f"  P-value: {mw_p_value:.4f}")
        print(f"  Interpretation: {'Medians are significantly different' if mw_p_value < 0.05 else 'No significant difference in medians'}")
        
        # Prepare results dictionary
        results = {
            'height': height,
            'ice_load_threshold': ice_load_threshold,
            'non_zero_percentage': non_zero_percentage,
            'emd_coordinates': {'longitude': emd_lon, 'latitude': emd_lat},
            'newa_grid_cell': {
                'south_north_index': int(closest_sn),
                'west_east_index': int(closest_we),
                'longitude': float(closest_lon),
                'latitude': float(closest_lat),
                'distance_from_emd_km': float(closest_distance_km)
            },
            'data_counts': {
                'emd_final': len(emd_final),
                'newa_final': len(newa_final),
                'hourly_data_points': len(emd_clean)
            },
            'statistics': {
                'emd_stats': emd_stats,
                'newa_stats': newa_stats,
                'qq_correlation': float(qq_correlation)
            },
            'statistical_tests': {
                'ks_test': {
                    'statistic': float(ks_statistic),
                    'p_value': float(ks_p_value),
                    'significant': ks_p_value < 0.05
                },
                'mann_whitney_test': {
                    'statistic': float(mw_statistic),
                    'p_value': float(mw_p_value),
                    'significant': mw_p_value < 0.05
                }
            },
            'filtered_data': {
                'emd_data': emd_final,
                'newa_data': newa_final
            }
        }
        
        print(f"\n✓ PDF analysis completed successfully!")
        print(f"Results saved to: {base_dir}")
        
        return results
        
    except Exception as e:
        print(f"Error in PDF analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_accretion_emd_newa(emd_data, dataset_with_ice_load, height, emd_coordinates, save_plots=True, accretion_threshold=0.0, non_zero_percentage=0.0):
    """
    Compare ice accretion data between EMD observations and NEWA model dataset.
    Parameters:
    -----------
    emd_data : pandas.DataFrame
        EMD observational data containing ice accretion columns (iceInten.50, iceInten.100, iceInten.150)
    dataset_with_ice_load : xarray.Dataset
        NEWA model dataset containing ACCRE_CYL variable
    height : int
        Height level to compare (50, 100, or 150 meters)
    emd_coordinates : tuple
        EMD coordinates as (longitude, latitude) in degrees
    save_plots : bool, optional
        Whether to save plots to file (default: True)
    accretion_threshold : float, optional
        Minimum accretion threshold (g/h) for filtering (default: 0.0)
    non_zero_percentage : float, optional
        Minimum percentage (0-100) of hours that must be > 0 in both datasets for a temporal period to be included (default: 0.0)
    Returns:
    --------
    dict
        Dictionary containing comparison statistics and analysis results
    """
    print(f"=== ICE ACCRETION COMPARISON: EMD vs NEWA at {height}m ===")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Validate height input
        if height not in [50, 100, 150]:
            raise ValueError(f"Height must be 50, 100, or 150 meters. Got: {height}")

        # Check if EMD data contains required column
        emd_column = f"iceInten.{int(height)}"
        if emd_column not in emd_data.columns:
            available_ice_cols = [col for col in emd_data.columns if 'iceInten' in col.lower() or 'accre' in col.lower()]
            raise ValueError(f"Column '{emd_column}' not found in EMD data. Available accretion columns: {available_ice_cols}")

        # Verify NEWA dataset height
        if 'ACCRE_CYL' not in dataset_with_ice_load.data_vars:
            raise ValueError("'ACCRE_CYL' variable not found in NEWA dataset")

        # Get height information from NEWA dataset
        height_levels = dataset_with_ice_load.height.values
        height_idx = None
        for i, h in enumerate(height_levels):
            if abs(h - height) < 1:
                height_idx = i
                break
        if height_idx is None:
            raise ValueError(f"Height {height}m not found in NEWA dataset. Available heights: {height_levels}")

        print(f"Using NEWA height level {height_idx} ({height_levels[height_idx]}m)")

        # Get EMD coordinates
        emd_lon, emd_lat = emd_coordinates
        print(f"EMD coordinates: {emd_lon:.4f}°E, {emd_lat:.4f}°N")

        # Extract coordinates - handle both data variables and coordinates
        if 'XLAT' in dataset_with_ice_load.coords:
            lats = dataset_with_ice_load.coords['XLAT'].values
            lons = dataset_with_ice_load.coords['XLON'].values
        else:
            lats = dataset_with_ice_load['XLAT'].values
            lons = dataset_with_ice_load['XLON'].values

        # Find the closest grid cell to EMD coordinates
        distance_squared = (lons - emd_lon)**2 + (lats - emd_lat)**2
        closest_indices = np.unravel_index(np.argmin(distance_squared), distance_squared.shape)
        closest_sn, closest_we = closest_indices

        # Get the actual coordinates of the closest grid cell
        closest_lon = lons[closest_sn, closest_we]
        closest_lat = lats[closest_sn, closest_we]
        closest_distance_deg = np.sqrt(distance_squared[closest_sn, closest_we])
        lat_correction = np.cos(np.radians(emd_lat))
        closest_distance_km = closest_distance_deg * 111.32 * lat_correction

        print(f"Closest NEWA grid cell:")
        print(f"  Grid indices: south_north={closest_sn}, west_east={closest_we}")
        print(f"  Grid coordinates: {closest_lon:.4f}°E, {closest_lat:.4f}°N")
        print(f"  Distance from EMD: {closest_distance_km:.2f} km")

        # Extract NEWA ice accretion data at specified height and closest grid cell
        newa_ice_accretion = dataset_with_ice_load['ACCRE_CYL'].isel(height=height_idx, south_north=closest_sn, west_east=closest_we)
        newa_df = newa_ice_accretion.to_dataframe(name='ACCRE_CYL').reset_index()
        newa_df['time'] = pd.to_datetime(newa_df['time'])
        newa_df = newa_df.set_index('time')
        # Convert NEWA accretion from kg/30min to g/h
        newa_df['ACCRE_CYL'] = newa_df['ACCRE_CYL'] * 2 * 1000

        print(f"NEWA data extracted from grid cell ({closest_sn}, {closest_we})")
        print(f"Closest cell coordinates: {closest_lon:.4f}°E, {closest_lat:.4f}°N")
        print(f"Distance from EMD location: {closest_distance_km:.2f} km")

        # Prepare EMD data
        if not isinstance(emd_data.index, pd.DatetimeIndex):
            if 'time' in emd_data.columns:
                emd_df = emd_data.copy()
                emd_df['time'] = pd.to_datetime(emd_df['time'])
                emd_df = emd_df.set_index('time')
            else:
                raise ValueError("EMD data must have datetime index or 'time' column")
        else:
            emd_df = emd_data.copy()

        print(f"EMD data period: {emd_df.index.min()} to {emd_df.index.max()}")
        print(f"NEWA data period: {newa_df.index.min()} to {newa_df.index.max()}")

        # Find common time period
        common_start = max(emd_df.index.min(), newa_df.index.min())
        common_end = min(emd_df.index.max(), newa_df.index.max())

        print(f"Common period: {common_start} to {common_end}")

        # Filter to common period
        emd_common = emd_df.loc[common_start:common_end, emd_column].copy()
        newa_common = newa_df.loc[common_start:common_end, 'ACCRE_CYL'].copy()

        # Resample NEWA data to hourly to match EMD (from 30min to 1h)
        print("Resampling NEWA data from 30min to 1h resolution and converting units...")
        newa_hourly = newa_common.resample('1H').mean()

        # Align time indices
        common_times = emd_common.index.intersection(newa_hourly.index)
        emd_aligned = emd_common.loc[common_times]
        newa_aligned = newa_hourly.loc[common_times]

        print(f"Aligned data points: {len(common_times)}")
        print(f"EMD ice accretion range: {emd_aligned.min():.3f} to {emd_aligned.max():.3f} g/h")
        print(f"NEWA ice accretion range: {newa_aligned.min():.3f} to {newa_aligned.max():.3f} g/h")

        # Remove NaN values and filter out non-icing months (June-October)
        valid_mask = ~(np.isnan(emd_aligned) | np.isnan(newa_aligned))
        emd_clean_all = emd_aligned[valid_mask]
        newa_clean_all = newa_aligned[valid_mask]

        # Filter out non-icing months (June=6, July=7, August=8, September=9, October=10)
        non_icing_months = [6, 7, 8, 9, 10]
        icing_mask = ~emd_clean_all.index.month.isin(non_icing_months)
        emd_clean = emd_clean_all[icing_mask]
        newa_clean = newa_clean_all[icing_mask]

        print(f"Valid data points after NaN removal: {len(emd_clean_all)}")
        print(f"Icing season data points (excluding Jun-Oct): {len(emd_clean)}")
        print(f"Excluded {len(emd_clean_all) - len(emd_clean)} non-icing season points")

        # Apply hourly threshold filtering
        if accretion_threshold > 0:
            print(f"Applying hourly threshold filter (>= {accretion_threshold} g/h)...")
            hourly_threshold_mask = (emd_clean >= accretion_threshold) & (newa_clean >= accretion_threshold)
            emd_threshold_filtered = emd_clean[hourly_threshold_mask]
            newa_threshold_filtered = newa_clean[hourly_threshold_mask]
            
            print(f"Hours after threshold filter (>= {accretion_threshold} g/h): {len(emd_threshold_filtered)}")
            print(f"Excluded {len(emd_clean) - len(emd_threshold_filtered)} hours below threshold")
        else:
            print("No hourly threshold filter applied (accretion_threshold = 0)")
            emd_threshold_filtered = emd_clean.copy()
            newa_threshold_filtered = newa_clean.copy()

        # Use threshold-filtered data for the rest of the analysis
        emd_clean = emd_threshold_filtered.copy()
        newa_clean = newa_threshold_filtered.copy()

        if len(emd_clean) < 10:
            print("Warning: Very few valid data points for comparison!")
            return None

        # Calculate comparison statistics
        print("\nCalculating comparison statistics...")

        # Basic statistics
        bias = np.mean(newa_clean - emd_clean)
        mae = mean_absolute_error(emd_clean, newa_clean)
        rmse = np.sqrt(mean_squared_error(emd_clean, newa_clean))

        # Correlation
        correlation, correlation_p = stats.pearsonr(emd_clean, newa_clean)
        spearman_corr, spearman_p = stats.spearmanr(emd_clean, newa_clean)

        # R-squared
        r2 = r2_score(emd_clean, newa_clean)

        # Relative metrics
        mean_emd = np.mean(emd_clean)
        relative_bias = (bias / mean_emd) * 100 if mean_emd != 0 else np.nan
        relative_mae = (mae / mean_emd) * 100 if mean_emd != 0 else np.nan
        relative_rmse = (rmse / mean_emd) * 100 if mean_emd != 0 else np.nan

        # Agreement statistics
        agreement_threshold = 0.1  # g/h
        within_threshold = np.sum(np.abs(newa_clean - emd_clean) <= agreement_threshold)
        agreement_percentage = (within_threshold / len(emd_clean)) * 100

        # Print statistics
        print(f"\n=== COMPARISON STATISTICS ===")
        print(f"Data points: {len(emd_clean)}")
        print(f"EMD mean: {mean_emd:.3f} g/h")
        print(f"NEWA mean: {np.mean(newa_clean):.3f} g/h")
        print(f"Bias (NEWA - EMD): {bias:.3f} g/h ({relative_bias:.1f}%)")
        print(f"MAE: {mae:.3f} g/h ({relative_mae:.1f}%)")
        print(f"RMSE: {rmse:.3f} g/h ({relative_rmse:.1f}%)")
        print(f"Correlation: {correlation:.3f} (p={correlation_p:.4f})")
        print(f"Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.4f})")
        print(f"R²: {r2:.3f}")
        print(f"Agreement within ±{agreement_threshold} g/h: {agreement_percentage:.1f}%")

        # Prepare results dictionary
        results = {
            'height': height,
            'n_points': len(emd_clean),
            'common_period': {'start': common_start, 'end': common_end},
            'emd_coordinates': {'longitude': emd_lon, 'latitude': emd_lat},
            'newa_grid_cell': {
                'south_north_index': int(closest_sn),
                'west_east_index': int(closest_we),
                'longitude': float(closest_lon),
                'latitude': float(closest_lat),
                'distance_from_emd_km': float(closest_distance_km)
            },
            'statistics': {
                'emd_mean': float(mean_emd),
                'newa_mean': float(np.mean(newa_clean)),
                'bias': float(bias),
                'mae': float(mae),
                'rmse': float(rmse),
                'relative_bias_percent': float(relative_bias),
                'relative_mae_percent': float(relative_mae),
                'relative_rmse_percent': float(relative_rmse),
                'correlation': float(correlation),
                'correlation_p_value': float(correlation_p),
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
                'r_squared': float(r2),
                'agreement_percentage': float(agreement_percentage)
            },
            'data': {
                'emd': emd_clean,
                'newa': newa_clean,
                'time': common_times
            }
        }


        # Create plots if requested
        if save_plots:
            print(f"\nCreating requested comparison plots...")
            threshold_str = f"threshold_{accretion_threshold:.3f}" if accretion_threshold > 0 else "no_threshold"
            nonzero_str = f"nonzero_{non_zero_percentage:.0f}pct" if non_zero_percentage > 0 else "no_nonzero_filter"
            base_dir = os.path.join("results", "figures", "EMD", "Ice_Accretion", "NEWA_EMD_comparison", f"{height:.0f}m_{threshold_str}_{nonzero_str}")
            os.makedirs(base_dir, exist_ok=True)

            # Plot 1A: Time series comparison with lines only
            print("1A. Creating full time series comparison (lines only)...")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))
            ax1.plot(emd_clean.index, emd_clean.values, 'b-', alpha=0.7, linewidth=0.5, label=f'EMD Hourly ({emd_column})')
            ax1.plot(newa_clean.index, newa_clean.values, 'r-', alpha=0.7, linewidth=0.5, label=f'NEWA Hourly (ACCRE_CYL)')
            ax1.set_ylabel('Ice Accretion (g/h)', fontsize=20)
            ax1.set_title(f'Hourly Ice Accretion Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Lines', fontsize=28)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            emd_daily_avg = emd_clean.resample('D').mean()
            newa_daily_avg = newa_clean.resample('D').mean()
            ax2.plot(emd_daily_avg.index, emd_daily_avg.values, 'b-', alpha=0.8, linewidth=1.0, label=f'EMD Daily Mean ({emd_column})')
            ax2.plot(newa_daily_avg.index, newa_daily_avg.values, 'r-', alpha=0.8, linewidth=1.0, label=f'NEWA Daily Mean (ACCRE_CYL)')
            ax2.set_ylabel('Ice Accretion (g/h)', fontsize=20)
            ax2.set_title(f'Daily Mean Ice Accretion Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Lines', fontsize=28)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            emd_weekly_avg = emd_clean.resample('W').mean()
            newa_weekly_avg = newa_clean.resample('W').mean()
            ax3.plot(emd_weekly_avg.index, emd_weekly_avg.values, 'b-', alpha=0.9, linewidth=1.5, label=f'EMD Weekly Mean ({emd_column})')
            ax3.plot(newa_weekly_avg.index, newa_weekly_avg.values, 'r-', alpha=0.9, linewidth=1.5, label=f'NEWA Weekly Mean (ACCRE_CYL)')
            ax3.set_xlabel('Time', fontsize=20)
            ax3.set_ylabel('Ice Accretion (g/h)', fontsize=20)
            ax3.set_title(f'Weekly Mean Ice Accretion Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Lines', fontsize=28)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.suptitle(f'Multi-Scale Ice Accretion Comparison: EMD vs NEWA at {height}m (Lines Only)', fontsize=28, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            timeseries_lines_path = os.path.join(base_dir, f'multi_scale_timeseries_lines_{height:.0f}m.png')
            plt.savefig(timeseries_lines_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {timeseries_lines_path}")

            # Plot 1B: Time series comparison with scatter only
            print("1B. Creating full time series comparison (scatter only)...")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))
            ax1.scatter(emd_clean.index, emd_clean.values, c='blue', s=0.5, alpha=0.6, label=f'EMD Hourly')
            ax1.scatter(newa_clean.index, newa_clean.values, c='red', s=0.5, alpha=0.6, label=f'NEWA Hourly')
            ax1.set_ylabel('Ice Accretion (g/h)', fontsize=23)
            ax1.set_title(f'Hourly Ice Accretion Time Series', fontsize=26)
            ax1.legend(fontsize=18)
            ax1.tick_params(axis='both', labelsize=21)
            ax1.grid(True, alpha=0.3)
            ax2.scatter(emd_daily_avg.index, emd_daily_avg.values, c='blue', s=3, alpha=0.7, label=f'EMD Daily Mean')
            ax2.scatter(newa_daily_avg.index, newa_daily_avg.values, c='red', s=3, alpha=0.7, label=f'NEWA Daily Mean')
            ax2.set_ylabel('Ice Accretion (g/h)', fontsize=23)
            ax2.set_title(f'Daily Mean Ice Accretion Time Series', fontsize=26)
            ax2.legend(fontsize=18)
            ax2.tick_params(axis='both', labelsize=21)
            ax2.grid(True, alpha=0.3)
            ax3.scatter(emd_weekly_avg.index, emd_weekly_avg.values, c='blue', s=10, alpha=0.8, label=f'EMD Weekly Mean')
            ax3.scatter(newa_weekly_avg.index, newa_weekly_avg.values, c='red', s=10, alpha=0.8, label=f'NEWA Weekly Mean')
            ax3.set_xlabel('Time', fontsize=23)
            ax3.set_ylabel('Ice Accretion (g/h)', fontsize=23)
            ax3.set_title(f'Weekly Mean Ice Accretion Time Series', fontsize=26)
            ax3.legend(fontsize=18)
            ax3.tick_params(axis='both', labelsize=21)
            ax3.grid(True, alpha=0.3)
            plt.suptitle(f'Multi-Scale Ice Accretion Comparison: EMD vs NEWA', fontsize=28, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            timeseries_scatter_path = os.path.join(base_dir, f'multi_scale_timeseries_scatter_{height:.0f}m.png')
            plt.savefig(timeseries_scatter_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {timeseries_scatter_path}")

            # Plot 2A: Difference over time with lines only
            print("2A. Creating difference time series (lines only)...")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))
            differences = newa_clean - emd_clean
            ax1.plot(differences.index, differences.values, 'g-', alpha=0.7, linewidth=0.5)
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax1.axhline(y=bias, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Mean Bias: {bias:.3f} g/h')
            ax1.set_ylabel('Difference (NEWA - EMD) [g/h]', fontsize=20)
            ax1.set_title(f'Hourly Ice Accretion Differences', fontsize=26)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            daily_differences_ts = newa_daily_avg - emd_daily_avg
            daily_bias = daily_differences_ts.mean()
            ax2.plot(daily_differences_ts.index, daily_differences_ts.values, 'g-', alpha=0.8, linewidth=1.0)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax2.axhline(y=daily_bias, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Daily Mean Bias: {daily_bias:.3f} g/h')
            ax2.set_ylabel('Difference (NEWA - EMD) [g/h]', fontsize=20)
            ax2.set_title(f'Daily Mean Ice Accretion Differences', fontsize=25)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            weekly_differences_ts = newa_weekly_avg - emd_weekly_avg
            weekly_bias = weekly_differences_ts.mean()
            ax3.plot(weekly_differences_ts.index, weekly_differences_ts.values, 'g-', alpha=0.9, linewidth=1.5)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax3.axhline(y=weekly_bias, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Weekly Mean Bias: {weekly_bias:.3f} g/h')
            ax3.set_xlabel('Time', fontsize=20)
            ax3.set_ylabel('Difference (NEWA - EMD) [g/h]', fontsize=20)
            ax3.set_title(f'Weekly Mean Ice Accretion Differences', fontsize=25)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.suptitle(f'Multi-Scale Ice Accretion Differences: NEWA - EMD at {height}m (Lines Only)', fontsize=28, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            differences_lines_path = os.path.join(base_dir, f'multi_scale_differences_lines_{height:.0f}m.png')
            plt.savefig(differences_lines_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {differences_lines_path}")

            # Plot 2B: Difference over time with scatter only
            print("2B. Creating difference time series (scatter only)...")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))
            ax1.scatter(differences.index, differences.values, c='green', s=0.5, alpha=0.6)
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax1.axhline(y=bias, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Mean Bias: {bias:.3f} g/h')
            ax1.set_ylabel('Difference (NEWA - EMD) [g/h]', fontsize=20)
            ax1.set_title(f'Hourly Ice Accretion Differences', fontsize=28)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax2.scatter(daily_differences_ts.index, daily_differences_ts.values, c='green', s=3, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax2.axhline(y=daily_bias, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Daily Mean Bias: {daily_bias:.3f} g/h')
            ax2.set_ylabel('Difference (NEWA - EMD) [g/h]', fontsize=20)
            ax2.set_title(f'Daily Mean Ice Accretion Differences', fontsize=28)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax3.scatter(weekly_differences_ts.index, weekly_differences_ts.values, c='green', s=10, alpha=0.8)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax3.axhline(y=weekly_bias, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Weekly Mean Bias: {weekly_bias:.3f} g/h')
            ax3.set_xlabel('Time', fontsize=20)
            ax3.set_ylabel('Difference (NEWA - EMD) [g/h]', fontsize=20)
            ax3.set_title(f'Weekly Mean Ice Accretion Differences', fontsize=28)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.suptitle(f'Multi-Scale Ice Accretion Differences', fontsize=28, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            differences_scatter_path = os.path.join(base_dir, f'multi_scale_differences_scatter_{height:.0f}m.png')
            plt.savefig(differences_scatter_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {differences_scatter_path}")

            # Plot 3: EMD vs NEWA scatter plot with 45° line and linear regression
            print("3. Creating EMD vs NEWA scatter plot with regression analysis...")
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            scatter = ax.scatter(newa_clean.values, emd_clean.values, alpha=0.6, s=20, c='blue', edgecolors='none', label='Data points')
            min_val = min(np.min(newa_clean), np.min(emd_clean))
            max_val = max(np.max(newa_clean), np.max(emd_clean))
            plot_range = max_val - min_val
            margin = plot_range * 0.05
            xlim = [min_val - margin, max_val + margin]
            ylim = [min_val - margin, max_val + margin]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.plot(xlim, ylim, 'k--', linewidth=2, alpha=0.8, label='Perfect agreement (1:1 line)')
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(newa_clean.values, emd_clean.values)
            regression_x = np.array(xlim)
            regression_y = slope * regression_x + intercept
            ax.plot(regression_x, regression_y, 'r-', linewidth=2, alpha=0.8, label=f'Linear regression (y = {slope:.3f}x + {intercept:.3f})')
            stats_text = (f'N = {len(emd_clean)}\nR² = {r2:.3f}\nCorrelation = {correlation:.3f}\nRMSE = {rmse:.3f} g/h\nMAE = {mae:.3f} g/h\nBias = {bias:.3f} g/h\nSlope = {slope:.3f}\nIntercept = {intercept:.3f}')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=16, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            ax.set_xlabel(f'NEWA Ice Accretion (g/h) at {height}m', fontsize=20)
            ax.set_ylabel(f'EMD Ice Accretion (g/h) at {height}m', fontsize=20)
            ax.set_title(f'EMD vs NEWA Ice Accretion Scatter Plot at {height}m (Icing Season Only)', fontsize=28, pad=15)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=15)
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            scatter_regression_path = os.path.join(base_dir, f'emd_vs_newa_scatter_{height:.0f}m.png')
            plt.savefig(scatter_regression_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {scatter_regression_path}")

            # Plot 4: EMD vs NEWA scatter plot (non-zero values only)
            print("4. Creating EMD vs NEWA scatter plot (non-zero values only)...")
            non_zero_mask = (emd_clean > 0) & (newa_clean > 0)
            emd_nonzero = emd_clean[non_zero_mask]
            newa_nonzero = newa_clean[non_zero_mask]
            print(f"Non-zero data: {len(emd_nonzero)} points out of {len(emd_clean)} total")
            if len(emd_nonzero) > 1:
                fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                sc = ax.scatter(newa_nonzero.values, emd_nonzero.values, c='blue', alpha=0.6, s=20, edgecolors='none', label='Data points')
                min_val = min(emd_nonzero.min(), newa_nonzero.min())
                max_val = max(emd_nonzero.max(), newa_nonzero.max())
                plot_range = max_val - min_val
                margin = plot_range * 0.05
                xlim = [min_val - margin, max_val + margin]
                ylim = [min_val - margin, max_val + margin]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.plot(xlim, ylim, 'k--', linewidth=2, alpha=0.8, label='Perfect agreement (1:1 line)')
                slope, intercept, r_value, p_value, std_err = stats.linregress(newa_nonzero.values, emd_nonzero.values)
                regression_x = np.array(xlim)
                regression_y = slope * regression_x + intercept
                ax.plot(regression_x, regression_y, 'r-', linewidth=2, alpha=0.8, label=f'Linear regression (y = {slope:.3f}x + {intercept:.3f})')
                ax.set_xlabel(f'NEWA Ice Accretion (g/h) at {height}m', fontsize=20)
                ax.set_ylabel(f'EMD Ice Accretion (g/h) at {height}m', fontsize=20)
                ax.set_title(f'EMD vs NEWA Ice Accretion Scatter Plot at {height}m (Non-Zero Values Only)', fontsize=28, pad=15)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='lower right', fontsize=15)
                ax.set_aspect('equal', adjustable='box')
                stats_text = (f'N = {len(emd_nonzero)}\nR² = {r_value**2:.3f}\nCorrelation = {np.corrcoef(emd_nonzero, newa_nonzero)[0,1]:.3f}\nRMSE = {np.sqrt(np.mean((newa_nonzero - emd_nonzero)**2)):.3f} g/h\nMAE = {np.mean(np.abs(newa_nonzero - emd_nonzero)):.3f} g/h\nBias = {np.mean(newa_nonzero - emd_nonzero):.3f} g/h\nSlope = {slope:.3f}\nIntercept = {intercept:.3f}')
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=16, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
                plt.tight_layout()
                nonzero_scatter_path = os.path.join(base_dir, f'emd_vs_newa_scatter_nonzero_{height:.0f}m.png')
                plt.savefig(nonzero_scatter_path, dpi=150, facecolor='white')
                plt.close()
                print(f"Saved: {nonzero_scatter_path}")
                print(f"\nNon-zero scatter plot statistics:")
                print(f"  Sample size: {len(emd_nonzero)}")
                print(f"  EMD range (non-zero): {emd_nonzero.min():.6f} to {emd_nonzero.max():.6f} g/h")
                print(f"  NEWA range (non-zero): {newa_nonzero.min():.6f} to {newa_nonzero.max():.6f} g/h")
                print(f"  Correlation: {np.corrcoef(emd_nonzero, newa_nonzero)[0,1]:.3f}")
                print(f"  Linear regression: y = {slope:.3f}x + {intercept:.3f}")
                print(f"  R-squared: {r_value**2:.3f}")
                print(f"  Standard error: {std_err:.3f}")
            else:
                print("Insufficient non-zero data for scatter plot")


            # Plot 5A: Zero values analysis - Bar plot
            print("5A. Creating zero values analysis bar plot...")
            emd_zero_count = (emd_clean == 0).sum()
            newa_zero_count = (newa_clean == 0).sum()
            total_timestamps = len(emd_clean)
            emd_zero_percentage = (emd_zero_count / total_timestamps) * 100
            newa_zero_percentage = (newa_zero_count / total_timestamps) * 100
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            datasets = ['EMD', 'NEWA']
            zero_percentages = [emd_zero_percentage, newa_zero_percentage]
            zero_counts = [emd_zero_count, newa_zero_count]
            colors = ['steelblue', 'orange']
            bars = ax.bar(datasets, zero_percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            for i, (bar, count, percentage) in enumerate(zip(bars, zero_counts, zero_percentages)):
                height_b = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height_b + 0.5, f'{percentage:.1f}%\n({count:,} hours)', ha='center', va='bottom', fontweight='bold', fontsize=16)
            ax.set_ylabel('Percentage of Zero Values (%)', fontsize=20, fontweight='bold')
            ax.set_title(f'Zero Value Analysis at {height:.0f}m\nTotal timestamps: {total_timestamps:,} hours', fontsize=28, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(zero_percentages) * 1.15)
            stats_text = f'Summary:\n'
            stats_text += f'EMD zeros: {emd_zero_count:,} ({emd_zero_percentage:.1f}%)\n'
            stats_text += f'NEWA zeros: {newa_zero_count:,} ({newa_zero_percentage:.1f}%)\n'
            stats_text += f'Both zero: {((emd_clean == 0) & (newa_clean == 0)).sum():,}\n'
            stats_text += f'Either zero: {((emd_clean == 0) | (newa_clean == 0)).sum():,}'
            ax.text(0.02, 0.05, stats_text, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), verticalalignment='bottom', horizontalalignment='left', fontsize=16)
            plt.tight_layout()

            # Ensure saving directory exists
            os.makedirs(base_dir, exist_ok=True)
            zero_analysis_path = os.path.join(base_dir, f'zero_values_analysis_{height:.0f}m.png')
            plt.savefig(zero_analysis_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {zero_analysis_path}")

            # Plot 5B: Negative values analysis - Bar plot
            print("5B. Creating negative values analysis bar plot...")
            emd_neg_count = (emd_clean < 0).sum()
            newa_neg_count = (newa_clean < 0).sum()
            emd_neg_percentage = (emd_neg_count / total_timestamps) * 100
            newa_neg_percentage = (newa_neg_count / total_timestamps) * 100
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            neg_percentages = [emd_neg_percentage, newa_neg_percentage]
            neg_counts = [emd_neg_count, newa_neg_count]
            bars = ax.bar(datasets, neg_percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            for i, (bar, count, percentage) in enumerate(zip(bars, neg_counts, neg_percentages)):
                height_b = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height_b + 0.5, f'{percentage:.1f}%\n({count:,} hours)', ha='center', va='bottom', fontweight='bold', fontsize=16)
            ax.set_ylabel('Percentage of Negative Values (%)', fontsize=20, fontweight='bold')
            ax.set_title(f'Negative Value Analysis at {height:.0f}m\nTotal timestamps: {total_timestamps:,} hours', fontsize=28, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(neg_percentages) * 1.15)
            stats_text = f'Summary:\n'
            stats_text += f'EMD negatives: {emd_neg_count:,} ({emd_neg_percentage:.1f}%)\n'
            stats_text += f'NEWA negatives: {newa_neg_count:,} ({newa_neg_percentage:.1f}%)\n'
            stats_text += f'Both negative: {((emd_clean < 0) & (newa_clean < 0)).sum():,}\n'
            stats_text += f'Either negative: {((emd_clean < 0) | (newa_clean < 0)).sum():,}'
            ax.text(0.02, 0.05, stats_text, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), verticalalignment='bottom', horizontalalignment='left', fontsize=16)
            plt.tight_layout()
            neg_analysis_path = os.path.join(base_dir, f'negative_values_analysis_{height:.0f}m.png')
            plt.savefig(neg_analysis_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {neg_analysis_path}")

            # Plot 5C: Positive values analysis - Bar plot
            print("5C. Creating positive values analysis bar plot...")
            emd_pos_count = (emd_clean > 0).sum()
            newa_pos_count = (newa_clean > 0).sum()
            emd_pos_percentage = (emd_pos_count / total_timestamps) * 100
            newa_pos_percentage = (newa_pos_count / total_timestamps) * 100
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            pos_percentages = [emd_pos_percentage, newa_pos_percentage]
            pos_counts = [emd_pos_count, newa_pos_count]
            bars = ax.bar(datasets, pos_percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            for i, (bar, count, percentage) in enumerate(zip(bars, pos_counts, pos_percentages)):
                height_b = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height_b + 0.5, f'{percentage:.1f}%\n({count:,} hours)', ha='center', va='bottom', fontweight='bold', fontsize=16)
            ax.set_ylabel('Percentage of Positive Values (%)', fontsize=20, fontweight='bold')
            ax.set_title(f'Positive Value Analysis at {height:.0f}m\nTotal timestamps: {total_timestamps:,} hours', fontsize=28, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(pos_percentages) * 1.15)
            stats_text = f'Summary:\n'
            stats_text += f'EMD positives: {emd_pos_count:,} ({emd_pos_percentage:.1f}%)\n'
            stats_text += f'NEWA positives: {newa_pos_count:,} ({newa_pos_percentage:.1f}%)\n'
            stats_text += f'Both positive: {((emd_clean > 0) & (newa_clean > 0)).sum():,}\n'
            stats_text += f'Either positive: {((emd_clean > 0) | (newa_clean > 0)).sum():,}'
            ax.text(0.02, 0.05, stats_text, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), verticalalignment='bottom', horizontalalignment='left', fontsize=16)
            plt.tight_layout()
            pos_analysis_path = os.path.join(base_dir, f'positive_values_analysis_{height:.0f}m.png')
            plt.savefig(pos_analysis_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {pos_analysis_path}")

            # Plot 6: Box plot for positive values only
            print("6. Creating box plot for positive values distribution...")
            positive_mask = (emd_clean > 0) & (newa_clean > 0)
            emd_positive = emd_clean[positive_mask]
            newa_positive = newa_clean[positive_mask]
            if len(emd_positive) > 0 and len(newa_positive) > 0:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                box_data = [emd_positive.values, newa_positive.values]
                labels = ['EMD', 'NEWA']
                colors = ['steelblue', 'orange']
                box_plot = ax.boxplot(box_data, labels=labels, patch_artist=True, showmeans=True, meanline=True, boxprops=dict(alpha=0.7), medianprops=dict(color='red', linewidth=2), meanprops=dict(color='black', linewidth=2, linestyle='--'))
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                ax.set_ylabel('Ice Accretion [g/h]', fontsize=40, fontweight='bold')
                ax.set_title(f'Distribution of Positive Ice Accretion Values at {height}m\nPositive values: EMD={len(emd_positive):,}, NEWA={len(newa_positive):,}', fontsize=28, fontweight='bold')
                ax.tick_params(axis='both', labelsize=40)
                ax.grid(True, alpha=0.3)
                emd_stats = {
                    'count': len(emd_positive),
                    'mean': emd_positive.mean(),
                    'std': emd_positive.std(),
                    'min': emd_positive.min(),
                    'q25': emd_positive.quantile(0.25),
                    'median': emd_positive.median(),
                    'q75': emd_positive.quantile(0.75),
                    'max': emd_positive.max()
                }
                newa_stats = {
                    'count': len(newa_positive),
                    'mean': newa_positive.mean(),
                    'std': newa_positive.std(),
                    'min': newa_positive.min(),
                    'q25': newa_positive.quantile(0.25),
                    'median': newa_positive.median(),
                    'q75': newa_positive.quantile(0.75),
                    'max': newa_positive.max()
                }
                stats_text = 'Positive Values Statistics:\n\n'
                stats_text += f'EMD (n={emd_stats["count"]:,}):\n'
                stats_text += f'  Mean: {emd_stats["mean"]:.4f} g/h\n'
                stats_text += f'  Std:  {emd_stats["std"]:.4f} g/h\n'
                stats_text += f'  Min:  {emd_stats["min"]:.4f} g/h\n'
                stats_text += f'  Q25:  {emd_stats["q25"]:.4f} g/h\n'
                stats_text += f'  Med:  {emd_stats["median"]:.4f} g/h\n'
                stats_text += f'  Q75:  {emd_stats["q75"]:.4f} g/h\n'
                stats_text += f'  Max:  {emd_stats["max"]:.4f} g/h\n\n'
                stats_text += f'NEWA (n={newa_stats["count"]:,}):\n'
                stats_text += f'  Mean: {newa_stats["mean"]:.4f} g/h\n'
                stats_text += f'  Std:  {newa_stats["std"]:.4f} g/h\n'
                stats_text += f'  Min:  {newa_stats["min"]:.4f} g/h\n'
                stats_text += f'  Q25:  {newa_stats["q25"]:.4f} g/h\n'
                stats_text += f'  Med:  {newa_stats["median"]:.4f} g/h\n'
                stats_text += f'  Q75:  {newa_stats["q75"]:.4f} g/h\n'
                stats_text += f'  Max:  {newa_stats["max"]:.4f} g/h'
                ax.text(1.02, 1.0, stats_text, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), verticalalignment='top', fontsize=16, family='monospace')
                legend_text = 'Box Plot Elements:\n'
                legend_text += '━ Red line: Median\n'
                legend_text += '┅ Black line: Mean\n'
                legend_text += '□ Box: Q25-Q75 (IQR)\n'
                legend_text += '┬ Whiskers: 1.5×IQR\n'
                legend_text += '○ Outliers'
                ax.text(0.98, 0.98, legend_text, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), verticalalignment='top', horizontalalignment='right', fontsize=16)
                plt.tight_layout()
                positive_boxplot_path = os.path.join(base_dir, f'positive_values_boxplot_{height:.0f}m.png')
                plt.savefig(positive_boxplot_path, dpi=150, facecolor='white', bbox_inches='tight')
                plt.close()
                print(f"Saved: {positive_boxplot_path}")
                print(f"\nPositive values distribution statistics:")
                print(f"  EMD positive values: {len(emd_positive):,} timestamps")
                print(f"    Mean ± Std: {emd_stats['mean']:.4f} ± {emd_stats['std']:.4f} g/h")
                print(f"    Median [Q25, Q75]: {emd_stats['median']:.4f} [{emd_stats['q25']:.4f}, {emd_stats['q75']:.4f}] g/h")
                print(f"    Range: {emd_stats['min']:.4f} to {emd_stats['max']:.4f} g/h")
                print(f"  NEWA positive values: {len(newa_positive):,} timestamps")
                print(f"    Mean ± Std: {newa_stats['mean']:.4f} ± {newa_stats['std']:.4f} g/h")
                print(f"    Median [Q25, Q75]: {newa_stats['median']:.4f} [{newa_stats['q25']:.4f}, {newa_stats['q75']:.4f}] g/h")
                print(f"    Range: {newa_stats['min']:.4f} to {newa_stats['max']:.4f} g/h")
            else:
                print("No positive values available for box plot analysis")

            # Plot 7: Hourly mean differences grid (all months included)
            print("7. Creating hourly mean differences grid (all months)...")
            emd_daily_means = emd_clean_all.resample('D').mean()
            newa_daily_means = newa_clean_all.resample('D').mean()
            common_daily_dates = emd_daily_means.index.intersection(newa_daily_means.index)
            emd_daily_aligned = emd_daily_means.loc[common_daily_dates]
            newa_daily_aligned = newa_daily_means.loc[common_daily_dates]
            daily_differences_all = newa_daily_aligned - emd_daily_aligned
            grid_df = pd.DataFrame({'date': daily_differences_all.index, 'difference': daily_differences_all.values})
            grid_df['year'] = grid_df['date'].dt.year
            grid_df['day_of_year'] = grid_df['date'].dt.dayofyear
            pivot_grid = grid_df.pivot(index='year', columns='day_of_year', values='difference')
            if pivot_grid.shape[1] < 365:
                for day in range(1, 366):
                    if day not in pivot_grid.columns:
                        pivot_grid[day] = np.nan
            pivot_grid = pivot_grid.reindex(columns=sorted(pivot_grid.columns)[:365])
            grid_array = pivot_grid.values
            plt.figure(figsize=(24, 14))
            vmax = np.nanmax(np.abs(grid_array))
            vmin = -vmax
            im = plt.imshow(grid_array, cmap='seismic', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
            plt.gca().set_xticks(np.arange(-0.5, grid_array.shape[1], 1), minor=True)
            plt.gca().set_yticks(np.arange(-0.5, grid_array.shape[0], 1), minor=True)
            plt.grid(which="minor", color="black", linestyle='-', linewidth=0.1, alpha=0.2)
            cbar = plt.colorbar(im, shrink=0.6, pad=0.02)
            cbar.set_label('Hourly Mean Ice Accretion Difference (NEWA - EMD) [g/h]', fontsize=23)
            cbar.ax.tick_params(labelsize=25)
            plt.xlabel('Day of Year', fontsize=23)
            plt.ylabel('Year', fontsize=23)
            plt.title(f'Daily Mean Ice Accretion Differences Grid: NEWA - EMD\nEach cell = daily mean difference for that specific year and day', fontsize=28, pad=20)
            year_indices = np.arange(0, len(pivot_grid.index))
            year_step = max(1, len(pivot_grid.index)//15)
            year_ticks = year_indices[::year_step]
            plt.yticks(year_ticks, [pivot_grid.index[i] for i in year_ticks], fontsize=23)
            month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.xticks(month_starts, month_labels, rotation=0, fontsize=23)
            ax2 = plt.gca().secondary_xaxis('top')
            day_ticks = np.arange(0, 366, 30)
            ax2.set_xticks(day_ticks)
            ax2.set_xlabel('Day of Year', fontsize=23)
            ax2.tick_params(labelsize=25)
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            daily_grid_path = os.path.join(base_dir, f'hourly_mean_grid_all_months_{height:.0f}m.png')
            plt.savefig(daily_grid_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {daily_grid_path}")

        print(f"\n✓ Ice accretion comparison completed successfully!")
        print(f"Results saved to: {base_dir}")
        return results

    except Exception as e:
        print(f"Error in ice accretion comparison: {e}")
        import traceback
        traceback.print_exc()
        return None

def emd_newa_accretion_typical(emd_data, dataset_with_ice_load, height, emd_coordinates, save_plots=True, ice_accretion_threshold=0.0, non_zero_percentage=0.0):
    """
    Compare typical day, week, and year ice accretion patterns between EMD observations and NEWA model dataset.
    
    Parameters:
    -----------
    emd_data : pandas.DataFrame
        EMD observational data containing ice accretion columns (iceInten.50, iceInten.100, iceInten.150)
    dataset_with_ice_load : xarray.Dataset
        NEWA model dataset containing ACCRE_CYL variable
    height : int
        Height level to compare (50, 100, or 150 meters)
    emd_coordinates : tuple
        EMD coordinates as (longitude, latitude) in degrees
    save_plots : bool, optional
        Whether to save plots to file (default: True)
    ice_accretion_threshold : float, optional
        Minimum mean hourly ice accretion threshold (mm/h). Only temporal periods where both datasets
        have mean hourly ice accretion >= threshold are included in analysis (default: 0.0)
    non_zero_percentage : float, optional
        Minimum percentage (0-100) of hours that must be > 0 in both datasets for a temporal period
        to be included in analysis. Applied to daily, weekly, and yearly aggregations (default: 0.0)
        
    Returns:
    --------
    dict
        Dictionary containing typical patterns analysis results
    """
    
    print(f"=== TYPICAL ACCRETION PATTERNS ANALYSIS: EMD vs NEWA at {height}m ===")
    print(f"Ice accretion threshold: {ice_accretion_threshold} mm/h (minimum mean hourly ice accretion for inclusion)")
    print(f"Non-zero percentage threshold: {non_zero_percentage}% (minimum percentage of hours > 0 required)")
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import os
        from scipy import stats
        
        # Validate height input
        if height not in [50, 100, 150]:
            raise ValueError(f"Height must be 50, 100, or 150 meters. Got: {height}")
        
        # Check if EMD data contains required column
        emd_column = f"iceInten.{int(height)}"
        if emd_column not in emd_data.columns:
            available_ice_cols = [col for col in emd_data.columns if 'iceInten' in col.lower() or 'accre' in col.lower()]
            raise ValueError(f"Column '{emd_column}' not found in EMD data. Available accretion columns: {available_ice_cols}")
        
        # Verify NEWA dataset height
        if 'ACCRE_CYL' not in dataset_with_ice_load.data_vars:
            raise ValueError("'ACCRE_CYL' variable not found in NEWA dataset")
        
        # Get height information from NEWA dataset
        height_levels = dataset_with_ice_load.height.values
        height_idx = None
        for i, h in enumerate(height_levels):
            if abs(h - height) < 1:  # Allow 1m tolerance
                height_idx = i
                break
        
        if height_idx is None:
            raise ValueError(f"Height {height}m not found in NEWA dataset. Available heights: {height_levels}")
        
        print(f"Using NEWA height level {height_idx} ({height_levels[height_idx]}m)")
        
        # Get EMD coordinates and find closest NEWA grid cell
        emd_lon, emd_lat = emd_coordinates
        print(f"EMD coordinates: {emd_lon:.4f}°E, {emd_lat:.4f}°N")
        
        # Extract coordinates - handle both data variables and coordinates
        if 'XLAT' in dataset_with_ice_load.coords:
            lats = dataset_with_ice_load.coords['XLAT'].values
            lons = dataset_with_ice_load.coords['XLON'].values
        else:
            lats = dataset_with_ice_load['XLAT'].values
            lons = dataset_with_ice_load['XLON'].values
        
        # Find the closest grid cell to EMD coordinates
        distance_squared = (lons - emd_lon)**2 + (lats - emd_lat)**2
        closest_indices = np.unravel_index(np.argmin(distance_squared), distance_squared.shape)
        closest_sn, closest_we = closest_indices
        
        # Get the actual coordinates of the closest grid cell
        closest_lon = lons[closest_sn, closest_we]
        closest_lat = lats[closest_sn, closest_we]
        closest_distance_deg = np.sqrt(distance_squared[closest_sn, closest_we])
        lat_correction = np.cos(np.radians(emd_lat))
        closest_distance_km = closest_distance_deg * 111.32 * lat_correction
        
        print(f"Closest NEWA grid cell:")
        print(f"  Grid indices: south_north={closest_sn}, west_east={closest_we}")
        print(f"  Grid coordinates: {closest_lon:.4f}°E, {closest_lat:.4f}°N")
        print(f"  Distance from EMD: {closest_distance_km:.2f} km")
        
        # Extract NEWA ice accretion data at specified height and closest grid cell
        newa_ice_accretion = dataset_with_ice_load['ACCRE_CYL'].isel(height=height_idx, south_north=closest_sn, west_east=closest_we)
        newa_df = newa_ice_accretion.to_dataframe(name='ACCRE_CYL').reset_index()
        newa_df['time'] = pd.to_datetime(newa_df['time'])
        newa_df = newa_df.set_index('time')
        # Convert NEWA accretion from kg/30min to g/h
        newa_df['ACCRE_CYL'] = newa_df['ACCRE_CYL'] * 2 * 1000

        # Prepare EMD data
        if not isinstance(emd_data.index, pd.DatetimeIndex):
            if 'time' in emd_data.columns:
                emd_df = emd_data.copy()
                emd_df['time'] = pd.to_datetime(emd_df['time'])
                emd_df = emd_df.set_index('time')
            else:
                raise ValueError("EMD data must have datetime index or 'time' column")
        else:
            emd_df = emd_data.copy()

        # Find common time period
        common_start = max(emd_df.index.min(), newa_df.index.min())
        common_end = min(emd_df.index.max(), newa_df.index.max())
        print(f"Common period: {common_start} to {common_end}")

        emd_common = emd_df.loc[common_start:common_end, emd_column].copy()
        newa_common = newa_df.loc[common_start:common_end, 'ACCRE_CYL'].copy()
        print("Resampling NEWA data from 30min to 1h resolution and converting units...")
        newa_hourly = newa_common.resample('1H').mean()

        common_times = emd_common.index.intersection(newa_hourly.index)
        emd_aligned = emd_common.loc[common_times]
        newa_aligned = newa_hourly.loc[common_times]

        valid_mask = ~(np.isnan(emd_aligned) | np.isnan(newa_aligned))
        emd_clean_all = emd_aligned[valid_mask]
        newa_clean_all = newa_aligned[valid_mask]

        non_icing_months = [6, 7, 8, 9, 10]
        icing_mask = ~emd_clean_all.index.month.isin(non_icing_months)
        emd_clean = emd_clean_all[icing_mask]
        newa_clean = newa_clean_all[icing_mask]

        print(f"Valid data points after NaN removal: {len(emd_clean_all)}")
        print(f"Icing season data points (excluding Jun-Oct): {len(emd_clean)}")

        if len(emd_clean) < 10:
            print("Warning: Very few valid data points for analysis!")
            return None

        # 1. TYPICAL DAY ANALYSIS
        print("1. Calculating typical day patterns...")
        
        # Apply hourly ice accretion threshold filter first
        print(f"  Applying hourly threshold filter (>= {ice_accretion_threshold} g/h)...")
        hourly_threshold_mask = (emd_clean >= ice_accretion_threshold) & (newa_clean >= ice_accretion_threshold)
        emd_threshold_filtered = emd_clean[hourly_threshold_mask]
        newa_threshold_filtered = newa_clean[hourly_threshold_mask]
        
        print(f"  Hours before threshold filter: {len(emd_clean)}")
        print(f"  Hours after threshold filter: {len(emd_threshold_filtered)}")
        
        # Calculate daily means from threshold-filtered hourly data
        emd_daily_means = emd_threshold_filtered.resample('D').mean()
        newa_daily_means = newa_threshold_filtered.resample('D').mean()
        emd_daily_clean = emd_daily_means.dropna()
        newa_daily_clean = newa_daily_means.dropna()
        common_daily_dates = emd_daily_clean.index.intersection(newa_daily_clean.index)
        emd_daily_temp = emd_daily_clean.loc[common_daily_dates]
        newa_daily_temp = newa_daily_clean.loc[common_daily_dates]
        
        # Use all days since hourly filtering already applied
        emd_daily_filtered = emd_daily_temp
        newa_daily_filtered = newa_daily_temp
        if non_zero_percentage > 0:
            print(f"  Applying {non_zero_percentage}% non-zero filter to daily data...")
            daily_non_zero_mask = []
            for date in emd_daily_filtered.index:
                day_start = date
                day_end = date + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
                emd_day_hours = emd_threshold_filtered[(emd_threshold_filtered.index >= day_start) & (emd_threshold_filtered.index <= day_end)]
                newa_day_hours = newa_threshold_filtered[(newa_threshold_filtered.index >= day_start) & (newa_threshold_filtered.index <= day_end)]
                if len(emd_day_hours) > 0 and len(newa_day_hours) > 0:
                    emd_nonzero_pct = (emd_day_hours > 0).mean() * 100
                    newa_nonzero_pct = (newa_day_hours > 0).mean() * 100
                    daily_non_zero_mask.append(emd_nonzero_pct >= non_zero_percentage and newa_nonzero_pct >= non_zero_percentage)
                else:
                    daily_non_zero_mask.append(False)
            daily_non_zero_mask = pd.Series(daily_non_zero_mask, index=emd_daily_filtered.index)
            emd_daily_aligned = emd_daily_filtered[daily_non_zero_mask]
            newa_daily_aligned = newa_daily_filtered[daily_non_zero_mask]
            print(f"  Daily means after non-zero filter ({non_zero_percentage}%): {len(emd_daily_aligned)} days")
        else:
            emd_daily_aligned = emd_daily_filtered
            newa_daily_aligned = newa_daily_filtered
        print(f"  Hours before threshold filter: {len(emd_clean)}")
        print(f"  Hours after threshold filter: {len(emd_threshold_filtered)}")
        print(f"  Daily means final count: {len(emd_daily_aligned)} days")

        # 2. TYPICAL WEEK ANALYSIS
        print("2. Calculating typical week patterns...")
        # Calculate weekly means from threshold-filtered hourly data
        emd_weekly_means = emd_threshold_filtered.resample('W').mean()
        newa_weekly_means = newa_threshold_filtered.resample('W').mean()
        emd_weekly_clean = emd_weekly_means.dropna()
        newa_weekly_clean = newa_weekly_means.dropna()
        common_weekly_dates = emd_weekly_clean.index.intersection(newa_weekly_clean.index)
        emd_weekly_temp = emd_weekly_clean.loc[common_weekly_dates]
        newa_weekly_temp = newa_weekly_clean.loc[common_weekly_dates]
        # Use all weeks since hourly filtering already applied
        emd_weekly_filtered = emd_weekly_temp
        newa_weekly_filtered = newa_weekly_temp
        if non_zero_percentage > 0:
            print(f"  Applying {non_zero_percentage}% non-zero filter to weekly data...")
            weekly_non_zero_mask = []
            for week_end in emd_weekly_filtered.index:
                week_start = week_end - pd.Timedelta(days=6, hours=23)
                emd_week_hours = emd_threshold_filtered[(emd_threshold_filtered.index >= week_start) & (emd_threshold_filtered.index <= week_end)]
                newa_week_hours = newa_threshold_filtered[(newa_threshold_filtered.index >= week_start) & (newa_threshold_filtered.index <= week_end)]
                if len(emd_week_hours) > 0 and len(newa_week_hours) > 0:
                    emd_nonzero_pct = (emd_week_hours > 0).mean() * 100
                    newa_nonzero_pct = (newa_week_hours > 0).mean() * 100
                    weekly_non_zero_mask.append(emd_nonzero_pct >= non_zero_percentage and newa_nonzero_pct >= non_zero_percentage)
                else:
                    weekly_non_zero_mask.append(False)
            weekly_non_zero_mask = pd.Series(weekly_non_zero_mask, index=emd_weekly_filtered.index)
            emd_weekly_aligned = emd_weekly_filtered[weekly_non_zero_mask]
            newa_weekly_aligned = newa_weekly_filtered[weekly_non_zero_mask]
            print(f"  Weekly means after non-zero filter ({non_zero_percentage}%): {len(emd_weekly_aligned)} weeks")
        else:
            emd_weekly_aligned = emd_weekly_filtered
            newa_weekly_aligned = newa_weekly_filtered
        print(f"  Weekly means before threshold filter: {len(emd_weekly_temp)} weeks")
        print(f"  Weekly means after threshold filter (>= {ice_accretion_threshold} g/h): {len(emd_weekly_filtered)} weeks")
        print(f"  Weekly means final count: {len(emd_weekly_aligned)} weeks")

        # 3. TYPICAL YEAR ANALYSIS
        print("3. Calculating typical year patterns...")
        # Calculate yearly means from threshold-filtered hourly data
        emd_yearly_means = emd_threshold_filtered.resample('Y').mean()
        newa_yearly_means = newa_threshold_filtered.resample('Y').mean()
        emd_yearly_clean = emd_yearly_means.dropna()
        newa_yearly_clean = newa_yearly_means.dropna()
        common_yearly_dates = emd_yearly_clean.index.intersection(newa_yearly_clean.index)
        emd_yearly_temp = emd_yearly_clean.loc[common_yearly_dates]
        newa_yearly_temp = newa_yearly_clean.loc[common_yearly_dates]
        # Use all years since hourly filtering already applied
        emd_yearly_filtered = emd_yearly_temp
        newa_yearly_filtered = newa_yearly_temp
        if non_zero_percentage > 0:
            print(f"  Applying {non_zero_percentage}% non-zero filter to yearly data...")
            yearly_non_zero_mask = []
            for year_end in emd_yearly_filtered.index:
                year_start = year_end.replace(month=1, day=1, hour=0)
                year_end_actual = year_end.replace(month=12, day=31, hour=23)
                emd_year_hours = emd_threshold_filtered[(emd_threshold_filtered.index >= year_start) & (emd_threshold_filtered.index <= year_end_actual)]
                newa_year_hours = newa_threshold_filtered[(newa_threshold_filtered.index >= year_start) & (newa_threshold_filtered.index <= year_end_actual)]
                if len(emd_year_hours) > 0 and len(newa_year_hours) > 0:
                    emd_nonzero_pct = (emd_year_hours > 0).mean() * 100
                    newa_nonzero_pct = (newa_year_hours > 0).mean() * 100
                    yearly_non_zero_mask.append(emd_nonzero_pct >= non_zero_percentage and newa_nonzero_pct >= non_zero_percentage)
                else:
                    yearly_non_zero_mask.append(False)
            yearly_non_zero_mask = pd.Series(yearly_non_zero_mask, index=emd_yearly_filtered.index)
            emd_yearly_aligned = emd_yearly_filtered[yearly_non_zero_mask]
            newa_yearly_aligned = newa_yearly_filtered[yearly_non_zero_mask]
            print(f"  Yearly means after non-zero filter ({non_zero_percentage}%): {len(emd_yearly_aligned)} years")
        else:
            emd_yearly_aligned = emd_yearly_filtered
            newa_yearly_aligned = newa_yearly_filtered
        print(f"  Yearly means before threshold filter: {len(emd_yearly_temp)} years")
        print(f"  Yearly means after threshold filter (>= {ice_accretion_threshold} g/h): {len(emd_yearly_filtered)} years")
        print(f"  Yearly means final count: {len(emd_yearly_aligned)} years")

        if len(emd_daily_aligned) == 0:
            print(f"Warning: No daily data above threshold {ice_accretion_threshold} g/h. Consider lowering the threshold.")
            return None
        if len(emd_weekly_aligned) == 0:
            print(f"Warning: No weekly data above threshold {ice_accretion_threshold} g/h. Consider lowering the threshold.")
        if len(emd_yearly_aligned) == 0:
            print(f"Warning: No yearly data above threshold {ice_accretion_threshold} g/h. Consider lowering the threshold.")

        def calculate_stats(data):
            return {
                'count': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'q25': data.quantile(0.25),
                'median': data.median(),
                'q75': data.quantile(0.75),
                'max': data.max()
            }

        emd_daily_stats = calculate_stats(emd_daily_aligned)
        newa_daily_stats = calculate_stats(newa_daily_aligned)
        emd_weekly_stats = calculate_stats(emd_weekly_aligned)
        newa_weekly_stats = calculate_stats(newa_weekly_aligned)
        emd_yearly_stats = calculate_stats(emd_yearly_aligned)
        newa_yearly_stats = calculate_stats(newa_yearly_aligned)

        print(f"\n=== TYPICAL ACCRETION PATTERNS STATISTICS ===")
        print(f"Filters Applied:")
        print(f"  Ice Accretion Threshold: >={ice_accretion_threshold} g/h")
        print(f"  Non-Zero Percentage: >={non_zero_percentage}% hours > 0")
        print(f"Daily means - EMD: {emd_daily_stats['mean']:.4f} +/- {emd_daily_stats['std']:.4f} g/h (n={emd_daily_stats['count']})")
        print(f"Daily means - NEWA: {newa_daily_stats['mean']:.4f} +/- {newa_daily_stats['std']:.4f} g/h (n={newa_daily_stats['count']})")
        print(f"Weekly means - EMD: {emd_weekly_stats['mean']:.4f} +/- {emd_weekly_stats['std']:.4f} g/h (n={emd_weekly_stats['count']})")
        print(f"Weekly means - NEWA: {newa_weekly_stats['mean']:.4f} +/- {newa_weekly_stats['std']:.4f} g/h (n={newa_weekly_stats['count']})")
        print(f"Yearly means - EMD: {emd_yearly_stats['mean']:.4f} +/- {emd_yearly_stats['std']:.4f} g/h (n={emd_yearly_stats['count']})")
        print(f"Yearly means - NEWA: {newa_yearly_stats['mean']:.4f} +/- {newa_yearly_stats['std']:.4f} g/h (n={newa_yearly_stats['count']})")

        if save_plots:
            print(f"\nCreating typical accretion patterns box plots...")
            threshold_str = f"threshold_{ice_accretion_threshold:.3f}" if ice_accretion_threshold > 0 else "no_threshold"
            nonzero_str = f"nonzero_{non_zero_percentage:.0f}pct" if non_zero_percentage > 0 else "no_nonzero_filter"
            base_dir = os.path.join("results", "figures", "EMD", "Ice_Accretion", "MeanDWM", f"{height:.0f}m_{threshold_str}_{nonzero_str}")
            os.makedirs(base_dir, exist_ok=True)
            fig, axes = plt.subplots(1, 3, figsize=(18, 8))
            colors = ['steelblue', 'orange']
            labels = ['EMD', 'NEWA']
            ax1 = axes[0]
            daily_data = [emd_daily_aligned.values, newa_daily_aligned.values]
            box_plot_daily = ax1.boxplot(daily_data, labels=labels, patch_artist=True, showmeans=True, meanline=True, boxprops=dict(alpha=0.7), medianprops=dict(color='red', linewidth=2), meanprops=dict(color='black', linewidth=2, linestyle='--'))
            for patch, color in zip(box_plot_daily['boxes'], colors):
                patch.set_facecolor(color)
            ax1.set_ylabel('Mean Daily Ice Accretion [g/h]\n(Mean of Hourly Values)', fontsize=15, fontweight='bold')
            ax1.set_title(f'Daily Means(n={len(emd_daily_aligned)})', fontsize=16, fontweight='bold')
            ax1.tick_params(axis='both', labelsize=20)
            ax1.grid(True, alpha=0.3)
            ax2 = axes[1]
            weekly_data = [emd_weekly_aligned.values, newa_weekly_aligned.values]
            box_plot_weekly = ax2.boxplot(weekly_data, labels=labels, patch_artist=True, showmeans=True, meanline=True, boxprops=dict(alpha=0.7), medianprops=dict(color='red', linewidth=2), meanprops=dict(color='black', linewidth=2, linestyle='--'))
            for patch, color in zip(box_plot_weekly['boxes'], colors):
                patch.set_facecolor(color)
            ax2.set_ylabel('Mean Weekly Ice Accretion [g/h]\n(Mean of Hourly Values)', fontsize=15, fontweight='bold')
            ax2.set_title(f'Weekly Means(n={len(emd_weekly_aligned)})', fontsize=16, fontweight='bold')
            ax2.tick_params(axis='both', labelsize=20)
            ax2.grid(True, alpha=0.3)
            ax3 = axes[2]
            yearly_data = [emd_yearly_aligned.values, newa_yearly_aligned.values]
            box_plot_yearly = ax3.boxplot(yearly_data, labels=labels, patch_artist=True, showmeans=True, meanline=True, boxprops=dict(alpha=0.7), medianprops=dict(color='red', linewidth=2), meanprops=dict(color='black', linewidth=2, linestyle='--'))
            for patch, color in zip(box_plot_yearly['boxes'], colors):
                patch.set_facecolor(color)
            ax3.set_ylabel('Mean Yearly Ice Accretion [g/h]\n(Mean of Hourly Values)', fontsize=15, fontweight='bold')
            ax3.set_title(f'Yearly Means(n={len(emd_yearly_aligned)})', fontsize=16, fontweight='bold')
            ax3.tick_params(axis='both', labelsize=20)
            ax3.grid(True, alpha=0.3)
            threshold_text = f"Ice Accretion Threshold: >={ice_accretion_threshold} g/h" if ice_accretion_threshold > 0 else "No Ice Accretion Threshold"
            nonzero_text = f"Non-Zero Filter: >={non_zero_percentage}% hours > 0" if non_zero_percentage > 0 else "No Non-Zero Filter"
            fig.suptitle(f'Patterns Analysis: EMD vs NEWA\nTemporal Means of Hourly Ice Accretion Values\n{threshold_text} | {nonzero_text}', fontsize=20, fontweight='bold', y=0.96)
            legend_text = '━ Red line: Median\n┅ Black line: Mean\n□ Box: Q25-Q75 (IQR)\n┬ Whiskers: 1.5×IQR\n○ Outliers'
            fig.text(0.98, 0.98, legend_text, fontsize=15, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), horizontalalignment='right', verticalalignment='top')
            plt.tight_layout()
            plt.subplots_adjust(top=0.80, bottom=0.15)
            typical_patterns_path = os.path.join(base_dir, f'typical_accretion_patterns_comparison_{height:.0f}m.png')
            plt.savefig(typical_patterns_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {typical_patterns_path}")
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            ax.axis('off')
            stats_table_data = []
            stats_table_data.append(['Temporal Scale', 'Dataset', 'Count', 'Mean', 'Std', 'Min', 'Q25', 'Median', 'Q75', 'Max'])
            stats_table_data.append(['Daily', 'EMD', f"{emd_daily_stats['count']}", f"{emd_daily_stats['mean']:.4f}", f"{emd_daily_stats['std']:.4f}", f"{emd_daily_stats['min']:.4f}", f"{emd_daily_stats['q25']:.4f}", f"{emd_daily_stats['median']:.4f}", f"{emd_daily_stats['q75']:.4f}", f"{emd_daily_stats['max']:.4f}"])
            stats_table_data.append(['', 'NEWA', f"{newa_daily_stats['count']}", f"{newa_daily_stats['mean']:.4f}", f"{newa_daily_stats['std']:.4f}", f"{newa_daily_stats['min']:.4f}", f"{newa_daily_stats['q25']:.4f}", f"{newa_daily_stats['median']:.4f}", f"{newa_daily_stats['q75']:.4f}", f"{newa_daily_stats['max']:.4f}"])
            stats_table_data.append(['Weekly', 'EMD', f"{emd_weekly_stats['count']}", f"{emd_weekly_stats['mean']:.4f}", f"{emd_weekly_stats['std']:.4f}", f"{emd_weekly_stats['min']:.4f}", f"{emd_weekly_stats['q25']:.4f}", f"{emd_weekly_stats['median']:.4f}", f"{emd_weekly_stats['q75']:.4f}", f"{emd_weekly_stats['max']:.4f}"])
            stats_table_data.append(['', 'NEWA', f"{newa_weekly_stats['count']}", f"{newa_weekly_stats['mean']:.4f}", f"{newa_weekly_stats['std']:.4f}", f"{newa_weekly_stats['min']:.4f}", f"{newa_weekly_stats['q25']:.4f}", f"{newa_weekly_stats['median']:.4f}", f"{newa_weekly_stats['q75']:.4f}", f"{newa_weekly_stats['max']:.4f}"])
            stats_table_data.append(['Yearly', 'EMD', f"{emd_yearly_stats['count']}", f"{emd_yearly_stats['mean']:.4f}", f"{emd_yearly_stats['std']:.4f}", f"{emd_yearly_stats['min']:.4f}", f"{emd_yearly_stats['q25']:.4f}", f"{emd_yearly_stats['median']:.4f}", f"{emd_yearly_stats['q75']:.4f}", f"{emd_yearly_stats['max']:.4f}"])
            stats_table_data.append(['', 'NEWA', f"{newa_yearly_stats['count']}", f"{newa_yearly_stats['mean']:.4f}", f"{newa_yearly_stats['std']:.4f}", f"{newa_yearly_stats['min']:.4f}", f"{newa_yearly_stats['q25']:.4f}", f"{newa_yearly_stats['median']:.4f}", f"{newa_yearly_stats['q75']:.4f}", f"{newa_yearly_stats['max']:.4f}"])
            table = ax.table(cellText=stats_table_data[1:], colLabels=stats_table_data[0], cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(16)
            table.scale(1, 2)
            for i in range(len(stats_table_data)):
                for j in range(len(stats_table_data[0])):
                    cell = table[(i, j)]
                    if i == 0:
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    elif j == 1 and i > 0 and stats_table_data[i][j] == 'EMD':
                        cell.set_facecolor('#E3F2FD')
                    elif j == 1 and i > 0 and stats_table_data[i][j] == 'NEWA':
                        cell.set_facecolor('#FFF3E0')
            threshold_text = f"Ice Accretion Threshold: >={ice_accretion_threshold} g/h" if ice_accretion_threshold > 0 else "No Ice Accretion Threshold"
            nonzero_text = f"Non-Zero Filter: >={non_zero_percentage}% hours > 0" if non_zero_percentage > 0 else "No Non-Zero Filter"
            ax.set_title(f'Patterns Statistics Summary\nTemporal Means of Hourly Ice Accretion Values in g/h\n{threshold_text} | {nonzero_text}', fontsize=28, fontweight='bold', pad=20)
            plt.tight_layout()
            stats_table_path = os.path.join(base_dir, f'typical_accretion_patterns_statistics_{height:.0f}m.png')
            plt.savefig(stats_table_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {stats_table_path}")
            print(f"\n=== PLOT SUMMARY ===")
            print(f"Created 2 plots:")
            print(f"  1. Typical accretion patterns box plot comparison: {typical_patterns_path}")
            print(f"  2. Detailed statistics table: {stats_table_path}")

        results = {
            'height': height,
            'ice_accretion_threshold': ice_accretion_threshold,
            'non_zero_percentage': non_zero_percentage,
            'emd_coordinates': {'longitude': emd_lon, 'latitude': emd_lat},
            'newa_grid_cell': {
                'south_north_index': int(closest_sn),
                'west_east_index': int(closest_we),
                'longitude': float(closest_lon),
                'latitude': float(closest_lat),
                'distance_from_emd_km': float(closest_distance_km)
            },
            'typical_patterns': {
                'daily': {
                    'emd_stats': emd_daily_stats,
                    'newa_stats': newa_daily_stats,
                    'emd_data': emd_daily_aligned,
                    'newa_data': newa_daily_aligned
                },
                'weekly': {
                    'emd_stats': emd_weekly_stats,
                    'newa_stats': newa_weekly_stats,
                    'emd_data': emd_weekly_aligned,
                    'newa_data': newa_weekly_aligned
                },
                'yearly': {
                    'emd_stats': emd_yearly_stats,
                    'newa_stats': newa_yearly_stats,
                    'emd_data': emd_yearly_aligned,
                    'newa_data': newa_yearly_aligned
                }
            }
        }

        print(f"\n✓ Typical accretion patterns analysis completed successfully!")
        print(f"Results saved to: {base_dir}")
        return results

    except Exception as e:
        print(f"Error in typical accretion patterns analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def pdf_emd_newa_accretion(emd_data, dataset_with_ice_load, height, emd_coordinates, save_plots=True, ice_accretion_threshold=0.0, non_zero_percentage=0.0):
    """
    Generate probability density function plots comparing EMD observations and NEWA model ice accretion distributions.
    
    Parameters:
    -----------
    emd_data : pandas.DataFrame
        EMD observational data containing ice accretion columns (iceInten.50, iceInten.100, iceInten.150)
    dataset_with_ice_load : xarray.Dataset
        NEWA model dataset containing ACCRE_CYL variable
    height : int
        Height level to compare (50, 100, or 150 meters)
    emd_coordinates : tuple
        EMD coordinates as (longitude, latitude) in degrees
    save_plots : bool, optional
        Whether to save plots to file (default: True)
    ice_accretion_threshold : float, optional
        Minimum mean hourly ice accretion threshold (g/h). Only temporal periods where both datasets
        have mean hourly ice accretion >= threshold are included in analysis (default: 0.0)
    non_zero_percentage : float, optional
        Minimum percentage (0-100) of hours that must be > 0 in both datasets for a temporal period
        to be included in analysis. Applied to daily, weekly, and yearly aggregations (default: 0.0)
        
    Returns:
    --------
    dict
        Dictionary containing PDF analysis results and statistics
    """
    

    print(f"=== PROBABILITY DENSITY FUNCTION ANALYSIS: EMD vs NEWA ICE ACCRETION at {height}m ===")
    print(f"Ice accretion threshold: {ice_accretion_threshold} mm/h (minimum mean hourly ice accretion for inclusion)")
    print(f"Non-zero percentage threshold: {non_zero_percentage}% (minimum percentage of hours > 0 required)")

    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import os
        from scipy import stats
        from sklearn.neighbors import KernelDensity

        # Validate height input
        if height not in [50, 100, 150]:
            raise ValueError(f"Height must be 50, 100, or 150 meters. Got: {height}")

        # Check if EMD data contains required column
        emd_column = f"iceInten.{int(height)}"
        if emd_column not in emd_data.columns:
            available_ice_cols = [col for col in emd_data.columns if 'iceInten' in col.lower() or 'accre' in col.lower()]
            raise ValueError(f"Column '{emd_column}' not found in EMD data. Available accretion columns: {available_ice_cols}")

        # Verify NEWA dataset height
        if 'ACCRE_CYL' not in dataset_with_ice_load.data_vars:
            raise ValueError("'ACCRE_CYL' variable not found in NEWA dataset")

        # Get height information from NEWA dataset
        height_levels = dataset_with_ice_load.height.values
        height_idx = None
        for i, h in enumerate(height_levels):
            if abs(h - height) < 1:  # Allow 1m tolerance
                height_idx = i
                break

        if height_idx is None:
            raise ValueError(f"Height {height}m not found in NEWA dataset. Available heights: {height_levels}")

        print(f"Using NEWA height level {height_idx} ({height_levels[height_idx]}m)")

        # Get EMD coordinates and find closest NEWA grid cell
        emd_lon, emd_lat = emd_coordinates
        print(f"EMD coordinates: {emd_lon:.4f}°E, {emd_lat:.4f}°N")

        # Extract coordinates - handle both data variables and coordinates
        if 'XLAT' in dataset_with_ice_load.coords:
            lats = dataset_with_ice_load.coords['XLAT'].values
            lons = dataset_with_ice_load.coords['XLON'].values
        else:
            lats = dataset_with_ice_load['XLAT'].values
            lons = dataset_with_ice_load['XLON'].values

        # Find the closest grid cell to EMD coordinates
        distance_squared = (lons - emd_lon)**2 + (lats - emd_lat)**2
        closest_indices = np.unravel_index(np.argmin(distance_squared), distance_squared.shape)
        closest_sn, closest_we = closest_indices

        # Get the actual coordinates of the closest grid cell
        closest_lon = lons[closest_sn, closest_we]
        closest_lat = lats[closest_sn, closest_we]
        closest_distance_deg = np.sqrt(distance_squared[closest_sn, closest_we])
        lat_correction = np.cos(np.radians(emd_lat))
        closest_distance_km = closest_distance_deg * 111.32 * lat_correction

        print(f"Closest NEWA grid cell:")
        print(f"  Grid indices: south_north={closest_sn}, west_east={closest_we}")
        print(f"  Grid coordinates: {closest_lon:.4f}°E, {closest_lat:.4f}°N")
        print(f"  Distance from EMD: {closest_distance_km:.2f} km")


        # Extract NEWA accretion data at specified height and closest grid cell
        newa_accretion = dataset_with_ice_load['ACCRE_CYL'].isel(height=height_idx, south_north=closest_sn, west_east=closest_we)

        # Convert to pandas DataFrame for easier manipulation
        newa_df = newa_accretion.to_dataframe(name='ACCRE_CYL').reset_index()
        newa_df['time'] = pd.to_datetime(newa_df['time'])
        newa_df = newa_df.set_index('time')

        # Convert NEWA accretion from kg/30min to g/h
        # Each value is kg per 30min, so multiply by 2 (to get kg/h) and then by 1000 (to get g/h)
        newa_df['ACCRE_CYL'] = newa_df['ACCRE_CYL'] * 2 * 1000

        # Now both EMD iceInten and NEWA ACCRE_CYL are in grams per hour

        # Prepare EMD data
        if not isinstance(emd_data.index, pd.DatetimeIndex):
            if 'time' in emd_data.columns:
                emd_df = emd_data.copy()
                emd_df['time'] = pd.to_datetime(emd_df['time'])
                emd_df = emd_df.set_index('time')
            else:
                raise ValueError("EMD data must have datetime index or 'time' column")
        else:
            emd_df = emd_data.copy()

        # Find common time period
        common_start = max(emd_df.index.min(), newa_df.index.min())
        common_end = min(emd_df.index.max(), newa_df.index.max())

        print(f"Common period: {common_start} to {common_end}")

        # Filter to common period
        emd_common = emd_df.loc[common_start:common_end, emd_column].copy()
        newa_common = newa_df.loc[common_start:common_end, 'ACCRE_CYL'].copy()


        # Resample NEWA data to hourly to match EMD (from 30min to 1h)
        print("Resampling NEWA data from 30min to 1h resolution and converting units...")
        # Already converted to g/h above, so just resample
        newa_hourly = newa_common.resample('1H').mean()

        # Align time indices
        common_times = emd_common.index.intersection(newa_hourly.index)
        emd_aligned = emd_common.loc[common_times]
        newa_aligned = newa_hourly.loc[common_times]

        # Remove NaN values and filter out non-icing months (June-October)
        valid_mask = ~(np.isnan(emd_aligned) | np.isnan(newa_aligned))
        emd_clean_all = emd_aligned[valid_mask]
        newa_clean_all = newa_aligned[valid_mask]

        # Filter out non-icing months (June=6, July=7, August=8, September=9, October=10)
        non_icing_months = [6, 7, 8, 9, 10]
        icing_mask = ~emd_clean_all.index.month.isin(non_icing_months)
        emd_clean = emd_clean_all[icing_mask]
        newa_clean = newa_clean_all[icing_mask]

        print(f"Valid data points after NaN removal: {len(emd_clean_all)}")
        print(f"Icing season data points (excluding Jun-Oct): {len(emd_clean)}")

        if len(emd_clean) < 10:
            print("Warning: Very few valid data points for analysis!")
            return None

        # Apply filtering based on ice_accretion_threshold and non_zero_percentage
        print(f"\nApplying filters to hourly data...")

        # Apply hourly ice accretion threshold filter - exclude timestamps where either EMD or NEWA is below threshold
        print(f"  Applying hourly threshold filter (>= {ice_accretion_threshold} g/h)...")
        hourly_threshold_mask = (emd_clean >= ice_accretion_threshold) & (newa_clean >= ice_accretion_threshold)
        emd_threshold_filtered = emd_clean[hourly_threshold_mask]
        newa_threshold_filtered = newa_clean[hourly_threshold_mask]
        
        print(f"  Hours before threshold filter: {len(emd_clean)}")
        print(f"  Hours after threshold filter: {len(emd_threshold_filtered)}")

        # Apply non-zero percentage filter if specified (now applied to remaining hourly data)
        if non_zero_percentage > 0:
            print(f"  Applying {non_zero_percentage}% non-zero filter to daily aggregations...")
            
            # Group remaining hourly data by day and check non-zero percentage
            emd_daily_groups = emd_threshold_filtered.groupby(emd_threshold_filtered.index.date)
            newa_daily_groups = newa_threshold_filtered.groupby(newa_threshold_filtered.index.date)
            
            valid_dates = []
            for date in emd_daily_groups.groups.keys():
                if date in newa_daily_groups.groups.keys():
                    emd_day_data = emd_daily_groups.get_group(date)
                    newa_day_data = newa_daily_groups.get_group(date)
                    
                    if len(emd_day_data) > 0 and len(newa_day_data) > 0:
                        emd_nonzero_pct = (emd_day_data > 0).mean() * 100
                        newa_nonzero_pct = (newa_day_data > 0).mean() * 100
                        if emd_nonzero_pct >= non_zero_percentage and newa_nonzero_pct >= non_zero_percentage:
                            valid_dates.append(date)
            
            # Filter to only include hours from valid dates
            valid_dates_set = set(valid_dates)
            final_mask = emd_threshold_filtered.index.to_series().dt.date.isin(valid_dates)
            emd_final = emd_threshold_filtered[final_mask].values
            newa_final = newa_threshold_filtered[final_mask].values
            
            print(f"  Days passing non-zero filter: {len(valid_dates)}")
            print(f"  Final hours after non-zero filter: {len(emd_final)}")
        else:
            # No non-zero percentage filter, use threshold-filtered data directly
            emd_final = emd_threshold_filtered.values
            newa_final = newa_threshold_filtered.values

        print(f"  Final hourly data points for PDF: {len(emd_final)} (EMD), {len(newa_final)} (NEWA)")

        if len(emd_final) < 50:
            print("Warning: Very few data points for PDF analysis!")
            return None

        # Calculate basic statistics
        def calculate_stats(data, name):
            stats_dict = {
                'count': len(data),
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'q25': np.percentile(data, 25),
                'median': np.median(data),
                'q75': np.percentile(data, 75),
                'max': np.max(data),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
            print(f"{name} statistics: mean={stats_dict['mean']:.4f}, std={stats_dict['std']:.4f}, skew={stats_dict['skewness']:.3f}")
            return stats_dict

        emd_stats = calculate_stats(emd_final, "EMD")
        newa_stats = calculate_stats(newa_final, "NEWA")

        # Create plots if requested
        if save_plots:
            print(f"\nCreating PDF plots...")
            threshold_str = f"threshold_{ice_accretion_threshold:.3f}" if ice_accretion_threshold > 0 else "no_threshold"
            nonzero_str = f"nonzero_{non_zero_percentage:.0f}pct" if non_zero_percentage > 0 else "no_nonzero_filter"
            base_dir = os.path.join("results", "figures", "EMD", "Ice_Accretion", "pdf_EMD_NEWA", f"{height:.0f}m_{threshold_str}_{nonzero_str}")
            os.makedirs(base_dir, exist_ok=True)

            data_min = min(np.min(emd_final), np.min(newa_final))
            data_max = max(np.max(emd_final), np.max(newa_final))
            x_range = np.linspace(data_min, data_max, 1000)

            threshold_text = f"Ice Accretion Threshold: >={ice_accretion_threshold} g/h" if ice_accretion_threshold > 0 else "No Ice Accretion Threshold"
            nonzero_text = f"Non-Zero Filter: >={non_zero_percentage}% hours > 0" if non_zero_percentage > 0 else "No Non-Zero Filter"
            common_subtitle = f'{threshold_text} | {nonzero_text}'

            # Plot 1: Histogram PDF comparison
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
            bins = np.linspace(data_min, data_max, 50)
            ax1.hist(emd_final, bins=bins, alpha=0.6, density=True, color='steelblue', edgecolor='darkblue', linewidth=1, label=f'EMD (n={len(emd_final)})')
            ax1.hist(newa_final, bins=bins, alpha=0.6, density=True, color='orange', edgecolor='darkorange', linewidth=1, label=f'NEWA (n={len(newa_final)})')
            ax1.set_xlabel('Ice Accretion [g/h]', fontweight='bold', fontsize=20)
            ax1.set_ylabel('Probability Density', fontweight='bold', fontsize=20)
            ax1.set_title(f'Probability Density Functions\n{common_subtitle}', fontsize=26)
            ax1.legend(fontsize=20)
            ax1.tick_params(labelsize=20)
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            hist_path = os.path.join(base_dir, f'pdf_histogram_{height:.0f}m.png')
            plt.savefig(hist_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {hist_path}")

            # Plot 2: KDE comparison
            fig, ax2 = plt.subplots(1, 1, figsize=(12, 8))
            if len(emd_final) > 10:
                kde_emd = stats.gaussian_kde(emd_final)
                ax2.plot(x_range, kde_emd(x_range), 'steelblue', linewidth=3, label=f'EMD (μ={emd_stats["mean"]:.3f}, σ={emd_stats["std"]:.3f})')
            if len(newa_final) > 10:
                kde_newa = stats.gaussian_kde(newa_final)
                ax2.plot(x_range, kde_newa(x_range), 'orange', linewidth=3, label=f'NEWA (μ={newa_stats["mean"]:.3f}, σ={newa_stats["std"]:.3f})')
            ax2.axvline(emd_stats['mean'], color='steelblue', linestyle='--', alpha=0.8, label='EMD Mean')
            ax2.axvline(newa_stats['mean'], color='orange', linestyle='--', alpha=0.8, label='NEWA Mean')
            ax2.set_xlabel('Ice Accretion [g/h]', fontweight='bold', fontsize=20)
            ax2.set_ylabel('Probability Density', fontweight='bold', fontsize=20)
            ax2.set_title(f'Kernel Density Estimation (KDE) Comparison\n{common_subtitle}', fontsize=26)
            ax2.legend(fontsize=20)
            ax2.tick_params(labelsize=20)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            kde_path = os.path.join(base_dir, f'pdf_kde_{height:.0f}m.png')
            plt.savefig(kde_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {kde_path}")

            # Plot 3: Log-log PDF comparison
            fig, ax3 = plt.subplots(1, 1, figsize=(12, 8))
            emd_nonzero = emd_final[emd_final > 0]
            newa_nonzero = newa_final[newa_final > 0]
            
            if len(emd_nonzero) > 10 and len(newa_nonzero) > 10:
                log_min = max(1e-6, min(np.min(emd_nonzero), np.min(newa_nonzero)))
                log_max = max(np.max(emd_nonzero), np.max(newa_nonzero))
                bins = np.logspace(np.log10(log_min), np.log10(log_max), 50)
                
                emd_counts, _ = np.histogram(emd_nonzero, bins=bins)
                newa_counts, _ = np.histogram(newa_nonzero, bins=bins)
                
                bin_widths = np.diff(bins)
                emd_pdf = emd_counts / (len(emd_nonzero) * bin_widths)
                newa_pdf = newa_counts / (len(newa_nonzero) * bin_widths)
                
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                emd_nonzero_pdf = emd_pdf > 0
                newa_nonzero_pdf = newa_pdf > 0
                
                ax3.loglog(bin_centers[emd_nonzero_pdf], emd_pdf[emd_nonzero_pdf], 'o-', 
                          color='steelblue', linewidth=2, markersize=4, alpha=0.8,
                          label=f'EMD PDF (n={len(emd_nonzero)})')
                ax3.loglog(bin_centers[newa_nonzero_pdf], newa_pdf[newa_nonzero_pdf], 's-', 
                          color='orange', linewidth=2, markersize=4, alpha=0.8,
                          label=f'NEWA PDF (n={len(newa_nonzero)})')
                ax3.axvline(np.mean(emd_nonzero), color='steelblue', linestyle='--', alpha=0.8, label='EMD Mean')
                ax3.axvline(np.mean(newa_nonzero), color='orange', linestyle='--', alpha=0.8, label='NEWA Mean')
            else:
                ax3.text(0.5, 0.5, 'Insufficient non-zero data\nfor log-log PDF', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=16)
            
            ax3.set_xlabel('Ice Accretion [g/h]', fontweight='bold', fontsize=20)
            ax3.set_ylabel('Probability Density', fontweight='bold', fontsize=20)
            ax3.set_title(f'PDF Comparison (Log-Log Scale)\n{common_subtitle}', fontsize=26)
            ax3.legend(fontsize=20)
            ax3.tick_params(labelsize=20)
            ax3.grid(True, alpha=0.3, which="both")
            plt.tight_layout()
            loglog_path = os.path.join(base_dir, f'pdf_loglog_{height:.0f}m.png')
            plt.savefig(loglog_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {loglog_path}")

            # Plot 4: Q-Q plot
            fig, ax4 = plt.subplots(1, 1, figsize=(12, 8))
            n_quantiles = min(len(emd_final), len(newa_final), 1000)
            quantiles = np.linspace(0.01, 0.99, n_quantiles)
            emd_quantiles = np.quantile(emd_final, quantiles)
            newa_quantiles = np.quantile(newa_final, quantiles)
            ax4.scatter(emd_quantiles, newa_quantiles, alpha=0.6, s=20, color='purple')
            min_val = min(np.min(emd_quantiles), np.min(newa_quantiles))
            max_val = max(np.max(emd_quantiles), np.max(newa_quantiles))
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Agreement')
            qq_correlation = np.corrcoef(emd_quantiles, newa_quantiles)[0, 1]
            ax4.set_xlabel('EMD Quantiles [g/h]', fontweight='bold', fontsize=20)
            ax4.set_ylabel('NEWA Quantiles [g/h]', fontweight='bold', fontsize=20)
            ax4.set_title(f'Q-Q Plot (r = {qq_correlation:.3f})\n{common_subtitle}', fontsize=26)
            ax4.legend(fontsize=20)
            ax4.tick_params(labelsize=20)
            ax4.grid(True, alpha=0.3)
            plt.tight_layout()
            qq_path = os.path.join(base_dir, f'pdf_qqplot_{height:.0f}m.png')
            plt.savefig(qq_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {qq_path}")

            # Plot 5: Box plots
            fig, ax5 = plt.subplots(1, 1, figsize=(12, 8))
            box_data = [emd_final, newa_final]
            labels = ['EMD', 'NEWA']
            colors = ['steelblue', 'orange']
            box_plot = ax5.boxplot(box_data, labels=labels, patch_artist=True, showmeans=True, meanline=True, boxprops=dict(alpha=0.7), medianprops=dict(color='red', linewidth=2), meanprops=dict(color='black', linewidth=2, linestyle='--'))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            ax5.set_ylabel('Ice Accretion [g/h]', fontweight='bold', fontsize=20)
            ax5.set_title(f'Distribution Comparison (Box Plots) at {height}m\n{common_subtitle}', fontweight='bold', fontsize=28)
            ax5.tick_params(labelsize=20)
            ax5.grid(True, alpha=0.3)
            plt.tight_layout()
            boxplot_path = os.path.join(base_dir, f'pdf_boxplot_{height:.0f}m.png')
            plt.savefig(boxplot_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {boxplot_path}")

            # Plot 6: Statistical summary table (inline in subplot grid)
            fig, ax6 = plt.subplots(1, 1, figsize=(12, 8))
            ax6.axis('off')
            
            stats_data = [
                ['Statistic', 'EMD', 'NEWA'],
                ['Count', f"{emd_stats['count']}", f"{newa_stats['count']}"],
                ['Mean', f"{emd_stats['mean']:.3f}", f"{newa_stats['mean']:.3f}"],
                ['Std Dev', f"{emd_stats['std']:.3f}", f"{newa_stats['std']:.3f}"],
                ['Min', f"{emd_stats['min']:.3f}", f"{newa_stats['min']:.3f}"],
                ['Q25', f"{emd_stats['q25']:.3f}", f"{newa_stats['q25']:.3f}"],
                ['Median', f"{emd_stats['median']:.3f}", f"{newa_stats['median']:.3f}"],
                ['Q75', f"{emd_stats['q75']:.3f}", f"{newa_stats['q75']:.3f}"],
                ['Max', f"{emd_stats['max']:.3f}", f"{newa_stats['max']:.3f}"],
                ['Skewness', f"{emd_stats['skewness']:.3f}", f"{newa_stats['skewness']:.3f}"],
                ['Kurtosis', f"{emd_stats['kurtosis']:.3f}", f"{newa_stats['kurtosis']:.3f}"],
                ['Q-Q r', '-', f"{qq_correlation:.3f}"]
            ]
            
            table = ax6.table(cellText=stats_data[1:], colLabels=stats_data[0], 
                             cellLoc='center', loc='center', bbox=[0.05, 0.1, 0.9, 0.8])
            table.auto_set_font_size(False)
            table.set_fontsize(16)
            table.scale(1, 1.5)
            
            for i in range(len(stats_data)):
                for j in range(len(stats_data[0])):
                    cell = table[(i, j)]
                    if i == 0:
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    elif j == 1:
                        cell.set_facecolor('#E3F2FD')
                    elif j == 2:
                        cell.set_facecolor('#FFF3E0')
                        
            ax6.set_title(f'Statistical Summary at {height}m\n{common_subtitle}', fontweight='bold', pad=10, fontsize=28)
            plt.tight_layout()
            stats_inline_path = os.path.join(base_dir, f'pdf_stats_table_inline_{height:.0f}m.png')
            plt.savefig(stats_inline_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {stats_inline_path}")

            # Plot 7: Detailed statistics table
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            ax.axis('off')
            stats_data = [
                ['Statistic', 'EMD', 'NEWA', 'Difference (NEWA - EMD)'],
                ['Count', f"{emd_stats['count']}", f"{newa_stats['count']}", f"{newa_stats['count'] - emd_stats['count']}"] ,
                ['Mean', f"{emd_stats['mean']:.4f}", f"{newa_stats['mean']:.4f}", f"{newa_stats['mean'] - emd_stats['mean']:.4f}"],
                ['Std Dev', f"{emd_stats['std']:.4f}", f"{newa_stats['std']:.4f}", f"{newa_stats['std'] - emd_stats['std']:.4f}"],
                ['Minimum', f"{emd_stats['min']:.4f}", f"{newa_stats['min']:.4f}", f"{newa_stats['min'] - emd_stats['min']:.4f}"],
                ['Q25', f"{emd_stats['q25']:.4f}", f"{newa_stats['q25']:.4f}", f"{newa_stats['q25'] - emd_stats['q25']:.4f}"],
                ['Median', f"{emd_stats['median']:.4f}", f"{newa_stats['median']:.4f}", f"{newa_stats['median'] - emd_stats['median']:.4f}"],
                ['Q75', f"{emd_stats['q75']:.4f}", f"{newa_stats['q75']:.4f}", f"{newa_stats['q75'] - emd_stats['q75']:.4f}"],
                ['Maximum', f"{emd_stats['max']:.4f}", f"{newa_stats['max']:.4f}", f"{newa_stats['max'] - emd_stats['max']:.4f}"],
                ['Skewness', f"{emd_stats['skewness']:.4f}", f"{newa_stats['skewness']:.4f}", f"{newa_stats['skewness'] - emd_stats['skewness']:.4f}"],
                ['Kurtosis', f"{emd_stats['kurtosis']:.4f}", f"{newa_stats['kurtosis']:.4f}", f"{newa_stats['kurtosis'] - emd_stats['kurtosis']:.4f}"],
                ['Q-Q Correlation', '-', '-', f"{qq_correlation:.4f}"]
            ]
            table = ax.table(cellText=stats_data[1:], colLabels=stats_data[0], cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(16)
            table.scale(1, 2)
            for i in range(len(stats_data)):
                for j in range(len(stats_data[0])):
                    cell = table[(i, j)]
                    if i == 0:
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    elif j == 1:
                        cell.set_facecolor('#E3F2FD')
                    elif j == 2:
                        cell.set_facecolor('#FFF3E0')
                    elif j == 3:
                        cell.set_facecolor('#F3E5F5')
            threshold_text = f"Ice Accretion Threshold: >={ice_accretion_threshold} mm/h" if ice_accretion_threshold > 0 else "No Ice Accretion Threshold"
            nonzero_text = f"Non-Zero Filter: >={non_zero_percentage}% hours > 0" if non_zero_percentage > 0 else "No Non-Zero Filter"
            ax.set_title(f'PDF Statistical Comparison Summary at {height}m\nIce Accretion Distribution Statistics [g/h] (Icing Season Only)\n{threshold_text} | {nonzero_text}', fontsize=28, fontweight='bold', pad=20)
            plt.tight_layout()
            stats_table_path = os.path.join(base_dir, f'pdf_statistics_{height:.0f}m.png')
            plt.savefig(stats_table_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {stats_table_path}")
            print(f"\n=== PLOT SUMMARY ===")
            print(f"Created 8 individual plots:")
            print(f"  1. Histogram PDF: {hist_path}")
            print(f"  2. KDE comparison: {kde_path}")
            print(f"  3. Log-log PDF: {loglog_path}")
            print(f"  4. Q-Q plot: {qq_path}")
            print(f"  5. Box plots: {boxplot_path}")
            print(f"  6. Statistics table (summary): {stats_inline_path}")
            print(f"  7. Statistics table (detailed): {stats_table_path}")

        print(f"\n=== STATISTICAL TESTS ===")
        ks_statistic, ks_p_value = stats.ks_2samp(emd_final, newa_final)
        print(f"Kolmogorov-Smirnov test:")
        print(f"  Statistic: {ks_statistic:.4f}")
        print(f"  P-value: {ks_p_value:.4f}")
        print(f"  Interpretation: {'Distributions are significantly different' if ks_p_value < 0.05 else 'No significant difference in distributions'}")
        mw_statistic, mw_p_value = stats.mannwhitneyu(emd_final, newa_final, alternative='two-sided')
        print(f"\nMann-Whitney U test:")
        print(f"  Statistic: {mw_statistic:.0f}")
        print(f"  P-value: {mw_p_value:.4f}")
        print(f"  Interpretation: {'Medians are significantly different' if mw_p_value < 0.05 else 'No significant difference in medians'}")

        results = {
            'height': height,
            'ice_accretion_threshold': ice_accretion_threshold,
            'non_zero_percentage': non_zero_percentage,
            'emd_coordinates': {'longitude': emd_lon, 'latitude': emd_lat},
            'newa_grid_cell': {
                'south_north_index': int(closest_sn),
                'west_east_index': int(closest_we),
                'longitude': float(closest_lon),
                'latitude': float(closest_lat),
                'distance_from_emd_km': float(closest_distance_km)
            },
            'data_counts': {
                'emd_final': len(emd_final),
                'newa_final': len(newa_final),
                'hourly_data_points': len(emd_clean)
            },
            'statistics': {
                'emd_stats': emd_stats,
                'newa_stats': newa_stats,
                'qq_correlation': float(qq_correlation)
            },
            'statistical_tests': {
                'ks_test': {
                    'statistic': float(ks_statistic),
                    'p_value': float(ks_p_value),
                    'significant': ks_p_value < 0.05
                },
                'mann_whitney_test': {
                    'statistic': float(mw_statistic),
                    'p_value': float(mw_p_value),
                    'significant': mw_p_value < 0.05
                }
            },
            'filtered_data': {
                'emd_data': emd_final,
                'newa_data': newa_final
            }
        }

        print(f"\n✓ PDF accretion analysis completed successfully!")
        print(f"Results saved to: {base_dir}")
        return results

    except Exception as e:
        print(f"Error in PDF accretion analysis: {e}")
        import traceback
        traceback.print_exc()
        return None