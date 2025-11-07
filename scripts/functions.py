import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# Results directories
figures_dir = "results/figures"
results_dir = "results"

# IMPORT DATA

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
    

def merge_netcdf_files(main_file_path, additional_file_path, output_file_path, verbose=True):
    """
    Merge two NetCDF files by combining their data variables.
    The main file provides the structure and most variables, while additional variables
    are added from the additional file.
    
    Parameters:
    -----------
    main_file_path : str
        Path to the main NetCDF file (contains most variables)
    additional_file_path : str
        Path to the additional NetCDF file (contains variables to be added)
    output_file_path : str
        Path where the merged NetCDF file will be saved
    verbose : bool, optional
        Whether to print detailed information during the merge process (default: True)
        
    Returns:
    --------
    bool
        True if merge was successful, False otherwise
        
    Example:
    --------
    >>> success = merge_netcdf_files(
    ...     "data/newa_wrf_for_jana_mstudent_extended.nc",
    ...     "data/newa_wrf_for_jana_mstudent_extended_WD.nc", 
    ...     "data/newa_wrf_for_jana_mstudent_extended_merged.nc"
    ... )
    """
    
    if verbose:
        print("=== NETCDF FILE MERGING ===")
        print(f"Main file: {main_file_path}")
        print(f"Additional file: {additional_file_path}")
        print(f"Output file: {output_file_path}")
    
    try:
        import xarray as xr
        import os
        
        # Load both datasets
        if verbose:
            print(f"\n1. Loading datasets...")
        
        main_ds = xr.open_dataset(main_file_path)
        additional_ds = xr.open_dataset(additional_file_path)
        
        if verbose:
            print(f"   Main dataset variables: {list(main_ds.data_vars.keys())}")
            print(f"   Additional dataset variables: {list(additional_ds.data_vars.keys())}")
            print(f"   Main dataset shape: {main_ds.dims}")
            print(f"   Additional dataset shape: {additional_ds.dims}")
        
        # Check if dimensions are compatible
        if verbose:
            print(f"\n2. Checking dimension compatibility...")
        
        # Get dimensions that matter for data variables (excluding potential differences in coords)
        main_dims = dict(main_ds.dims)
        additional_dims = dict(additional_ds.dims)
        
        # Check critical dimensions
        critical_dims = ['time', 'south_north', 'west_east']
        if 'height' in main_dims:
            critical_dims.append('height')
            
        dimension_compatible = True
        for dim in critical_dims:
            if dim in main_dims and dim in additional_dims:
                if main_dims[dim] != additional_dims[dim]:
                    print(f"   Warning: Dimension mismatch for '{dim}': {main_dims[dim]} vs {additional_dims[dim]}")
                    dimension_compatible = False
                else:
                    if verbose:
                        print(f"   ✓ {dim}: {main_dims[dim]} (compatible)")
            elif dim in main_dims:
                if verbose:
                    print(f"   ✓ {dim}: {main_dims[dim]} (only in main file)")
            elif dim in additional_dims:
                if verbose:
                    print(f"   ✓ {dim}: {additional_dims[dim]} (only in additional file)")
        
        if not dimension_compatible:
            print("   Error: Incompatible dimensions between files")
            return False
        
        # Check for variable conflicts
        if verbose:
            print(f"\n3. Checking for variable conflicts...")
        
        main_vars = set(main_ds.data_vars.keys())
        additional_vars = set(additional_ds.data_vars.keys())
        
        conflicts = main_vars.intersection(additional_vars)
        if conflicts:
            print(f"   Warning: Variable conflicts found: {conflicts}")
            print(f"   Variables from main file will be kept, additional file variables will be skipped")
        
        new_vars = additional_vars - conflicts
        if verbose:
            print(f"   Variables to be added: {list(new_vars)}")
        
        # Merge datasets
        if verbose:
            print(f"\n4. Merging datasets...")
        
        # Start with the main dataset
        merged_ds = main_ds.copy(deep=True)
        
        # Add new variables from additional dataset
        for var_name in new_vars:
            if verbose:
                print(f"   Adding variable: {var_name}")
            
            # Get the variable from additional dataset
            var_data = additional_ds[var_name]
            
            # Add to merged dataset
            merged_ds[var_name] = var_data
        
        # Verify the merge
        if verbose:
            print(f"\n5. Verifying merged dataset...")
            print(f"   Original main variables: {len(main_vars)}")
            print(f"   Added variables: {len(new_vars)}")
            print(f"   Total variables in merged dataset: {len(merged_ds.data_vars)}")
            print(f"   Final variables: {list(merged_ds.data_vars.keys())}")
        
        # Save the merged dataset
        if verbose:
            print(f"\n6. Saving merged dataset...")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save with compression to reduce file size
        encoding = {}
        for var in merged_ds.data_vars:
            if merged_ds[var].dtype in ['float32', 'float64']:
                encoding[var] = {'zlib': True, 'complevel': 4}
        
        merged_ds.to_netcdf(output_file_path, encoding=encoding)
        
        if verbose:
            file_size_mb = os.path.getsize(output_file_path) / (1024 * 1024)
            print(f"   Merged file saved: {output_file_path}")
            print(f"   File size: {file_size_mb:.1f} MB")
        
        # Clean up
        main_ds.close()
        additional_ds.close()
        merged_ds.close()
        
        if verbose:
            print(f"\n✓ NetCDF file merging completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error merging NetCDF files: {e}")
        import traceback
        traceback.print_exc()
        return False

# EXPLORE DATASET

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

def analyze_landmask(dataset, create_plot=True, save_results=True):
    """
    Analyze the LANDMASK variable to count land vs water grid cells
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        The dataset containing LANDMASK variable
    create_plot : bool, default True
        Whether to create a visualization of the landmask
    save_results : bool, default True
        Whether to save the analysis results and plot
        
    Returns:
    --------
    dict : Dictionary containing landmask analysis results
    """
    print("=== LANDMASK ANALYSIS ===")
    
    try:
        # Extract LANDMASK data
        landmask = dataset.LANDMASK
        
        # Basic information
        print(f"\n1. Basic Information:")
        print(f"   Grid shape: {landmask.shape}")
        print(f"   Data type: {landmask.dtype}")
        print(f"   Dimensions: {landmask.dims}")
        
        # Count land and water cells
        landmask_values = landmask.values
        water_cells = np.sum(landmask_values == 0)
        land_cells = np.sum(landmask_values == 1)
        total_cells = landmask.size
        
        # Calculate percentages
        water_percentage = (water_cells / total_cells) * 100
        land_percentage = (land_cells / total_cells) * 100
        
        # Print results
        print(f"\n2. Land/Water Distribution:")
        print(f"   Water cells (value = 0): {water_cells} ({water_percentage:.1f}%)")
        print(f"   Land cells (value = 1):  {land_cells} ({land_percentage:.1f}%)")
        print(f"   Total grid cells:        {total_cells}")
        
        # Check for any unexpected values
        unique_values = np.unique(landmask_values)
        print(f"\n3. Data Quality Check:")
        print(f"   Unique values found: {unique_values}")
        if len(unique_values) > 2 or not all(val in [0, 1] for val in unique_values):
            print(f"   WARNING: Unexpected values found in LANDMASK!")
        else:
            print(f"   ✓ LANDMASK contains only expected values (0, 1)")
        
        # Spatial context
        if 'XLAT' in dataset and 'XLON' in dataset:
            lat_range = [float(dataset.XLAT.min()), float(dataset.XLAT.max())]
            lon_range = [float(dataset.XLON.min()), float(dataset.XLON.max())]
            print(f"\n4. Spatial Context:")
            print(f"   Latitude range:  {lat_range[0]:.3f}° to {lat_range[1]:.3f}°")
            print(f"   Longitude range: {lon_range[0]:.3f}° to {lon_range[1]:.3f}°")
        
        # Create detailed grid cell analysis
        print(f"\n5. Grid Cell Details:")
        print("   Row-by-row analysis (from south to north):")
        for i in range(landmask.shape[0]):
            row = landmask_values[i, :]
            land_in_row = np.sum(row == 1)
            water_in_row = np.sum(row == 0)
            print(f"   Row {i:2d}: Land={land_in_row:2d}, Water={water_in_row:2d} | {row}")
        
        # Create visualization if requested
        if create_plot:
            print(f"\n6. Creating visualization...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: LANDMASK spatial distribution
            im1 = ax1.imshow(landmask_values, cmap='RdYlBu_r', origin='lower', 
                           interpolation='nearest')
            ax1.set_title('LANDMASK - Spatial Distribution')
            ax1.set_xlabel('West-East Grid Points')
            ax1.set_ylabel('South-North Grid Points')
            
            # Add grid cell values as text
            for i in range(landmask.shape[0]):
                for j in range(landmask.shape[1]):
                    value = landmask_values[i, j]
                    color = 'white' if value < 0.5 else 'black'
                    ax1.text(j, i, f'{int(value)}', ha='center', va='center', 
                            color=color, fontsize=8, weight='bold')
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('LANDMASK (0=Water, 1=Land)')
            cbar1.set_ticks([0, 1])
            cbar1.set_ticklabels(['Water', 'Land'])
            
            # Plot 2: Summary statistics
            categories = ['Water\n(0)', 'Land\n(1)']
            counts = [water_cells, land_cells]
            colors = ['lightblue', 'lightgreen']
            
            bars = ax2.bar(categories, counts, color=colors, edgecolor='black', linewidth=1)
            ax2.set_title('Land vs Water Distribution')
            ax2.set_ylabel('Number of Grid Cells')
            ax2.grid(True, alpha=0.3)
            
            # Add count labels on bars
            for bar, count, percentage in zip(bars, counts, [water_percentage, land_percentage]):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{count}\n({percentage:.1f}%)', 
                        ha='center', va='bottom', fontweight='bold')
            
            # Add total cells information
            ax2.text(0.5, max(counts) * 0.8, f'Total: {total_cells} cells', 
                    ha='center', transform=ax2.transData, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot if requested
            if save_results:
                os.makedirs(figures_dir, exist_ok=True)
                plot_path = f"{figures_dir}/landmask_analysis.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"   Plot saved to: {plot_path}")
            
            plt.show()
        
        # Prepare results dictionary
        results = {
            'grid_shape': landmask.shape,
            'total_cells': total_cells,
            'water_cells': water_cells,
            'land_cells': land_cells,
            'water_percentage': water_percentage,
            'land_percentage': land_percentage,
            'unique_values': unique_values.tolist(),
            'landmask_array': landmask_values,
            'is_valid': len(unique_values) <= 2 and all(val in [0, 1] for val in unique_values)
        }
        
        # Add spatial info if available
        if 'XLAT' in dataset and 'XLON' in dataset:
            results['lat_range'] = lat_range
            results['lon_range'] = lon_range
        
        # Save results to file if requested
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            results_path = f"{results_dir}/landmask_analysis.txt"
            with open(results_path, 'w') as f:
                f.write("LANDMASK ANALYSIS RESULTS\n")
                f.write("========================\n\n")
                f.write(f"Grid shape: {landmask.shape}\n")
                f.write(f"Total cells: {total_cells}\n")
                f.write(f"Water cells (0): {water_cells} ({water_percentage:.1f}%)\n")
                f.write(f"Land cells (1): {land_cells} ({land_percentage:.1f}%)\n")
                f.write(f"Unique values: {unique_values.tolist()}\n")
                f.write(f"Data valid: {results['is_valid']}\n\n")
                
                if 'lat_range' in results:
                    f.write(f"Latitude range: {lat_range[0]:.3f}° to {lat_range[1]:.3f}°\n")
                    f.write(f"Longitude range: {lon_range[0]:.3f}° to {lon_range[1]:.3f}°\n\n")
                
                f.write("Grid cell details (row by row):\n")
                for i in range(landmask.shape[0]):
                    row = landmask_values[i, :]
                    land_in_row = np.sum(row == 1)
                    water_in_row = np.sum(row == 0)
                    f.write(f"Row {i:2d}: Land={land_in_row:2d}, Water={water_in_row:2d} | {row}\n")
            
            print(f"   Results saved to: {results_path}")
        
        print(f"\n✓ LANDMASK analysis completed successfully!")
        return results
        
    except KeyError:
        print("Error: LANDMASK variable not found in dataset")
        return None
    except Exception as e:
        print(f"Error analyzing LANDMASK: {e}")
        return None

# ACCREATION

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

# ICE LOAD

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
            if i % 1000 == 0:  # Progress indicator every 1000 steps
                print(f"  Ice load progress: {i}/{len(accre.time)} ({i/len(accre.time)*100:.1f}%)")
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
    if method == 51:
        # Optimized version of method 5 using numpy operations for faster computation
        accre_values = accre.values
        ablat_values = ablat.values
        
        # Initialize arrays
        loadnp = np.zeros_like(accre_values)
        acm = np.zeros_like(accre_values[0])  # Shape: (south_north, west_east)
        
        for i in range(1, len(accre.time)):
            if i % 1000 == 0:  # Progress indicator every 1000 steps
                print(f"  Ice load progress (optimized): {i}/{len(accre.time)} ({i/len(accre.time)*100:.1f}%)")
            
            # Get current timestep data
            accre_i = accre_values[i]
            ablat_i = ablat_values[i]
            
            # Calculate change in ice load using numpy operations (much faster)
            delta = np.where(accre_i > 0.0005, accre_i, 
                           np.where(ablat_i < 0, ablat_i, 0))
            
            # Update ice load with numpy operations
            acm += delta
            np.clip(acm, 0, 5, out=acm)  # Efficient in-place clipping: 0 <= acm <= 5
            loadnp[i] = acm

        load = xr.zeros_like(accre)
        load.data = loadnp
    #
    # rename data array
    load = load.rename('ice_load')
    #
    return load

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
        # Avoid .load() on sliced data for better performance with filtered datasets
        winter_accre = ds1['ACCRE_CYL'].isel(height=height_level).sel(time=slice(date,dates[idate+1]-pd.to_timedelta('30min')))
        winter_ablat = ds1['ABLAT_CYL'].isel(height=height_level).sel(time=slice(date,dates[idate+1]-pd.to_timedelta('30min')))
        
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

def add_ice_load_to_dataset(ds, dates, method=5, height_level=0, variable_name='ICE_LOAD'):
    """
    Calculate ice load and add it as a new variable to the xarray Dataset
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing ice accretion and ablation data
    dates : pandas.DatetimeIndex
        Date range for winter seasons
    method : int, optional
        Ice load calculation method (1-5) (default: 5)
    height_level : int, optional
        Height level index to analyze (default: 0)
        Use 0, 1, or 2 for the three available height levels
    variable_name : str, optional
        Name for the new ice load variable in the dataset (default: 'ICE_LOAD')
    
    Returns:
    --------
    xarray.Dataset
        Original dataset with added ice load variable
    """
    
    print(f"Adding ice load calculation to dataset using method {method}...")
    
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
        print(f"Calculating ice load at height level {height_level} ({height_value} {height_units})")
    else:
        print("No height dimension found in dataset")
        height_level = 0
        height_value = "unknown"
        height_units = ""
    
    print(f"Calculating ice load at {height_value} {height_units}...")
    
    # Create ice load array with same structure as accretion data
    dsiceload = xr.zeros_like(ds['ACCRE_CYL'].isel(height=height_level)) * np.nan
    
    # Calculate ice load for each winter period
    for idate, date in enumerate(dates[:-1]):
        print(f"Processing winter {idate+1}/{len(dates)-1}: {date} to {dates[idate+1]-pd.to_timedelta('30min')}")
        
        # Get data for this winter period at specified height level
        winter_accre = ds['ACCRE_CYL'].isel(height=height_level).sel(time=slice(date, dates[idate+1]-pd.to_timedelta('30min')))
        winter_ablat = ds['ABLAT_CYL'].isel(height=height_level).sel(time=slice(date, dates[idate+1]-pd.to_timedelta('30min')))
        
        # Check if there's data for this winter
        if len(winter_accre.time) == 0:
            print(f"  No data available for winter {idate+1}. Skipping...")
            continue
            
        load = ice_load(winter_accre, winter_ablat, method)
        
        # Only assign if load calculation was successful
        if load is not None:
            dsiceload.loc[{'time': load.time}] = load
            print(f"Winter {idate+1} completed")
        else:
            print(f"Winter {idate+1} skipped due to insufficient data")
    
    # Create a copy of the original dataset
    ds_with_ice_load = ds.copy()
    
    # Add ice load as a new variable to the dataset
    # We need to expand the ice load data to include the height dimension
    # Create a new DataArray with the height dimension
    ice_load_expanded = xr.zeros_like(ds['ACCRE_CYL']) * np.nan
    # Use the actual height coordinate value, not the height_level index
    ice_load_expanded.loc[dict(height=height_value)] = dsiceload
    
    # Add the ice load variable to the dataset
    ds_with_ice_load[variable_name] = ice_load_expanded
    
    # Add attributes to the new variable
    ds_with_ice_load[variable_name].attrs = {
        'long_name': f'Ice Load calculated using method {method}',
        'units': 'kg/m',
        'description': f'Ice load calculated at height level {height_level} ({height_value} {height_units}) using ice accumulation method {method}',
        'calculation_method': method,
        'height_level_used': height_level,
        'height_value': height_value,
        'height_units': height_units,
        'valid_range': [0.0, 5.0],
        'missing_value': np.nan
    }
    
    print(f"\n=== Ice Load Integration Summary ===")
    print(f"Successfully added '{variable_name}' variable to dataset")
    print(f"Ice load shape: {dsiceload.shape}")
    print(f"Height level used: {height_level} ({height_value} {height_units})")
    print(f"Calculation method: {method}")
    print(f"Valid data points: {np.sum(~np.isnan(dsiceload.values)):,}")
    
    # Print dataset info after adding ice load
    print(f"\nDataset now contains {len(ds_with_ice_load.data_vars)} variables:")
    for var in ds_with_ice_load.data_vars:
        print(f"  - {var}: {ds_with_ice_load[var].shape}")
    
    # Save the complete dataset with ice load to results directory
    try:
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Format start and end dates for filename (safe for file names)
        start_date = pd.to_datetime(dates[0]).strftime('%Y%m%d')
        end_date = pd.to_datetime(dates[-1]).strftime('%Y%m%d')
        
        # Create filename for the dataset with ice load
        dataset_filename = f"dataset_iceload_{start_date}_{end_date}.nc"
        dataset_filepath = os.path.join(results_dir, dataset_filename)
        
        print(f"\nSaving complete dataset with ice load to: {dataset_filepath}")
        
        # Save the dataset
        ds_with_ice_load.to_netcdf(dataset_filepath)
        
        print(f"Successfully saved dataset with ice load!")
        print(f"  File size: {os.path.getsize(dataset_filepath) / (1024*1024):.1f} MB")
        print(f"  Variables: {list(ds_with_ice_load.data_vars.keys())}")
        print(f"  Time range: {start_date} to {end_date}")
        print(f"  Ice load method: {method}")
        print(f"  Height level: {height_level}")
        
    except Exception as e:
        print(f"Warning: Could not save dataset with ice load: {e}")
        print("Continuing without saving...")
    
    return ds_with_ice_load

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

# SPATIAL GRADIENTS 

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

def plot_ice_load_duration_curves(ice_load_data, save_plots=True, ice_load_bins=None, ice_load_threshold=0.0):
    """
    Create cumulative duration curves showing hours per year for each ice load level
    for every grid cell. The function creates a mean over all years and plots
    X-axis: Ice load (kg/m)
    Y-axis: Hours per year
    
    Parameters:
    -----------
    ice_load_data : xarray.DataArray
        Ice load data with dimensions (time, south_north, west_east)
    save_plots : bool, default True
        Whether to save the plots to files
    ice_load_bins : array-like, optional
        Custom ice load bins for analysis. If None, uses automatic binning
    ice_load_threshold : float, default 0.0
        Minimum ice load value to be plotted (kg/m). Values below this threshold will be excluded
        
    Returns:
    --------
    dict : Dictionary containing analysis results and statistics
    """
    print("=== ICE LOAD DURATION CURVE ANALYSIS ===")
    
    try:
        # Check data structure
        print(f"\n1. Data Information:")
        print(f"   Shape: {ice_load_data.shape}")
        print(f"   Dimensions: {ice_load_data.dims}")
        print(f"   Time range: {ice_load_data.time.min().values} to {ice_load_data.time.max().values}")
        
        # Get spatial dimensions
        n_south_north = ice_load_data.sizes['south_north']
        n_west_east = ice_load_data.sizes['west_east']
        n_time = ice_load_data.sizes['time']
        
        print(f"   Grid size: {n_south_north} × {n_west_east} = {n_south_north * n_west_east} cells")
        print(f"   Time steps: {n_time}")
        
        # Convert time to pandas for easier manipulation
        time_index = pd.to_datetime(ice_load_data.time.values)
        n_years = len(time_index.year.unique())
        print(f"   Years covered: {n_years}")
        print(f"   Years: {sorted(time_index.year.unique())}")
        
        # Calculate time step in hours (assuming regular intervals)
        if len(time_index) > 1:
            time_step_hours = (time_index[1] - time_index[0]).total_seconds() / 3600
        else:
            time_step_hours = 0.5  # Default to 30 minutes
        print(f"   Time step: {time_step_hours} hours")
        
        # Remove NaN values and get overall data statistics
        ice_data_clean = ice_load_data.where(ice_load_data >= 0, 0)  # Replace negative/NaN with 0
        max_ice_load = float(ice_data_clean.max())
        min_ice_load = 0.0
        
        print(f"\n2. Ice Load Statistics:")
        print(f"   Range: {min_ice_load:.3f} to {max_ice_load:.3f} kg/m")
        print(f"   Mean: {float(ice_data_clean.mean()):.3f} kg/m")
        
        # Define ice load bins if not provided
        if ice_load_bins is None:
            if max_ice_load > ice_load_threshold:
                # Create logarithmic-like bins for better distribution, starting from threshold
                min_bin_value = max(ice_load_threshold, 0.01)  # Ensure we have a reasonable minimum for log scale
                ice_load_bins = np.concatenate([
                    [ice_load_threshold] if ice_load_threshold > 0 else [0],  # Include threshold or zero
                    np.logspace(np.log10(min_bin_value), np.log10(max_ice_load), 30)
                ])
                # Remove duplicates and sort
                ice_load_bins = np.unique(ice_load_bins)
                # Filter bins to only include those >= threshold
                ice_load_bins = ice_load_bins[ice_load_bins >= ice_load_threshold]
            else:
                ice_load_bins = np.array([ice_load_threshold, ice_load_threshold + 0.01, ice_load_threshold + 0.1, ice_load_threshold + 1, ice_load_threshold + 10])
        else:
            # Filter provided bins to only include those >= threshold
            ice_load_bins = ice_load_bins[ice_load_bins >= ice_load_threshold]
        
        print(f"   Using {len(ice_load_bins)} ice load bins")
        print(f"   Ice load threshold: {ice_load_threshold:.3f} kg/m")
        print(f"   Bin range: {ice_load_bins[0]:.4f} to {ice_load_bins[-1]:.3f} kg/m")
        
        # Prepare results storage
        results = {
            'grid_shape': (n_south_north, n_west_east),
            'n_years': n_years,
            'time_step_hours': time_step_hours,
            'ice_load_bins': ice_load_bins,
            'duration_curves': {},
            'statistics': {}
        }
        
        # Create figure for all grid cells
        n_cols = min(n_west_east, 5)  # Maximum 5 columns
        n_rows = int(np.ceil((n_south_north * n_west_east) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        print(f"\n3. Processing grid cells...")
        
        cell_count = 0
        for i in range(n_south_north):
            for j in range(n_west_east):
                # Extract time series for this grid cell
                cell_data = ice_data_clean.isel(south_north=i, west_east=j)
                cell_values = cell_data.values
                
                # Remove NaN values
                valid_mask = ~np.isnan(cell_values)
                cell_values_clean = cell_values[valid_mask]
                
                if len(cell_values_clean) == 0:
                    print(f"   Warning: No valid data for cell ({i},{j})")
                    continue
                
                # Calculate duration curve
                duration_hours = []
                
                for ice_threshold in ice_load_bins:
                    # Count hours when ice load >= threshold
                    hours_above_threshold = np.sum(cell_values_clean >= ice_threshold) * time_step_hours
                    # Convert to hours per year
                    hours_per_year = hours_above_threshold / n_years
                    duration_hours.append(hours_per_year)
                
                duration_hours = np.array(duration_hours)
                
                # Store results for this cell
                cell_key = f'cell_{i}_{j}'
                results['duration_curves'][cell_key] = {
                    'ice_load_bins': ice_load_bins,
                    'hours_per_year': duration_hours,
                    'position': (i, j)
                }
                
                # Calculate statistics for this cell
                total_hours_per_year = 365.25 * 24  # Account for leap years
                max_hours = duration_hours[0] if len(duration_hours) > 0 else 0
                zero_ice_hours = total_hours_per_year - max_hours
                
                results['statistics'][cell_key] = {
                    'max_ice_load': float(np.max(cell_values_clean)),
                    'mean_ice_load': float(np.mean(cell_values_clean)),
                    'hours_with_ice_per_year': max_hours,
                    'hours_without_ice_per_year': zero_ice_hours,
                    'ice_occurrence_percentage': (max_hours / total_hours_per_year) * 100
                }
                
                # Plot duration curve for this cell
                if cell_count < len(axes):
                    ax = axes[cell_count]
                    ax.plot(ice_load_bins, duration_hours, 'b-', linewidth=2, marker='o', markersize=3)
                    ax.set_xlabel('Ice Load (kg/m)')
                    ax.set_ylabel('Hours/Year')
                    ax.set_title(f'Cell ({i},{j})')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(left=ice_load_threshold)
                    ax.set_ylim(bottom=0)
                    
                    # Use log scale for x-axis if there's a wide range
                    if max_ice_load > 10:
                        ax.set_xscale('log')
                        ax.set_xlim(left=max(ice_load_threshold, 0.01))  # Ensure minimum for log scale
                
                cell_count += 1
                
                if cell_count % 5 == 0:
                    print(f"   Processed {cell_count}/{n_south_north * n_west_east} cells...")
        
        # Hide unused subplots
        for idx in range(cell_count, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            # Create specific directory structure for ice load per cell duration curve plots
            ice_load_plots_dir = os.path.join(figures_dir, "spatial_gradient", "ice_load_per_cell_duration_curve")
            os.makedirs(ice_load_plots_dir, exist_ok=True)
            plot_path = f"{ice_load_plots_dir}/ice_load_duration_curves_all_cells.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\n   Duration curves plot saved to: {plot_path}")
        
        # Create summary plot with mean curve
        print(f"\n4. Creating summary statistics...")
        
        # Calculate mean duration curve across all cells
        all_duration_curves = []
        for cell_key, cell_data in results['duration_curves'].items():
            all_duration_curves.append(cell_data['hours_per_year'])
        
        if all_duration_curves:
            mean_duration = np.mean(all_duration_curves, axis=0)
            std_duration = np.std(all_duration_curves, axis=0)
            
            plt.figure(figsize=(10, 6))
            plt.plot(ice_load_bins, mean_duration, 'r-', linewidth=3, label='Mean across all cells')
            plt.fill_between(ice_load_bins, mean_duration - std_duration, 
                           mean_duration + std_duration, alpha=0.3, color='red', 
                           label='±1 Standard Deviation')
            
            plt.xlabel('Ice Load (kg/m)')
            plt.ylabel('Hours per Year')
            plt.title('Ice Load Duration Curve - Domain Average')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(left=ice_load_threshold)
            plt.ylim(bottom=0)
            
            if max_ice_load > 10:
                plt.xscale('log')
                plt.xlim(left=max(ice_load_threshold, 0.01))  # Ensure minimum for log scale
            
            plt.tight_layout()
            
            if save_plots:
                summary_path = f"{ice_load_plots_dir}/ice_load_duration_curve_summary.png"
                plt.savefig(summary_path, dpi=300, bbox_inches='tight')
                print(f"   Summary plot saved to: {summary_path}")
            
            # Store summary statistics
            results['domain_statistics'] = {
                'mean_duration_curve': mean_duration,
                'std_duration_curve': std_duration,
                'ice_load_bins': ice_load_bins
            }
        
        # Print summary statistics
        print(f"\n5. Summary Statistics:")
        print(f"   Processed {len(results['duration_curves'])} grid cells")
        
        if results['statistics']:
            all_stats = list(results['statistics'].values())
            mean_ice_occurrence = np.mean([s['ice_occurrence_percentage'] for s in all_stats])
            max_ice_occurrence = np.max([s['ice_occurrence_percentage'] for s in all_stats])
            min_ice_occurrence = np.min([s['ice_occurrence_percentage'] for s in all_stats])
            
            print(f"   Ice occurrence across domain:")
            print(f"     Mean: {mean_ice_occurrence:.1f}% of time")
            print(f"     Range: {min_ice_occurrence:.1f}% to {max_ice_occurrence:.1f}%")
        
        # Save detailed results
        if save_plots:
            results_path = f"{results_dir}/ice_load_duration_analysis.txt"
            with open(results_path, 'w') as f:
                f.write("ICE LOAD DURATION CURVE ANALYSIS RESULTS\n")
                f.write("=======================================\n\n")
                f.write(f"Data shape: {ice_load_data.shape}\n")
                f.write(f"Years analyzed: {n_years}\n")
                f.write(f"Time step: {time_step_hours} hours\n")
                f.write(f"Ice load range: {min_ice_load:.3f} to {max_ice_load:.3f} kg/m\n\n")
                
                f.write("Grid Cell Statistics:\n")
                f.write("-" * 50 + "\n")
                for cell_key, stats in results['statistics'].items():
                    pos = results['duration_curves'][cell_key]['position']
                    f.write(f"Cell {pos}: ")
                    f.write(f"Max ice: {stats['max_ice_load']:.3f} kg/m, ")
                    f.write(f"Mean ice: {stats['mean_ice_load']:.3f} kg/m, ")
                    f.write(f"Ice occurrence: {stats['ice_occurrence_percentage']:.1f}%\n")
            
            print(f"   Detailed results saved to: {results_path}")
        
        # Create spatial gradient analysis using Earth Mover's Distance
        print(f"\n6. Spatial Gradient Analysis (Earth Mover's Distance)...")
        
        try:
            from scipy.stats import wasserstein_distance
            
            # Initialize gradient matrices
            n_south_north, n_west_east = ice_load_data.shape[1], ice_load_data.shape[2]
            
            # East-West gradients (comparing adjacent cells horizontally)
            ew_gradients = np.full((n_south_north, n_west_east-1), np.nan)
            
            # South-North gradients (comparing adjacent cells vertically)
            sn_gradients = np.full((n_south_north-1, n_west_east), np.nan)
            
            # Combined gradients (for each cell, average of all neighbor comparisons)
            combined_gradients = np.full((n_south_north, n_west_east), np.nan)
            
            print(f"   Computing Wasserstein distances between neighboring cells...")
            
            # Calculate East-West gradients
            for i in range(n_south_north):
                for j in range(n_west_east-1):
                    cell1_key = f'cell_{i}_{j}'
                    cell2_key = f'cell_{i}_{j+1}'
                    
                    if cell1_key in results['duration_curves'] and cell2_key in results['duration_curves']:
                        curve1 = results['duration_curves'][cell1_key]['hours_per_year']
                        curve2 = results['duration_curves'][cell2_key]['hours_per_year']
                        bins1 = results['duration_curves'][cell1_key]['ice_load_bins']
                        bins2 = results['duration_curves'][cell2_key]['ice_load_bins']
                        
                        # Calculate Earth Mover's distance between the duration curves
                        # We need to treat these as distributions, so we create weighted samples
                        try:
                            distance = wasserstein_distance(bins1, bins2, curve1, curve2)
                            ew_gradients[i, j] = distance
                        except:
                            print(f"   Warning: Could not compute Wasserstein distance for cells ({i},{j}) and ({i},{j+1})")
            
            # Calculate South-North gradients
            for i in range(n_south_north-1):
                for j in range(n_west_east):
                    cell1_key = f'cell_{i}_{j}'
                    cell2_key = f'cell_{i+1}_{j}'
                    
                    if cell1_key in results['duration_curves'] and cell2_key in results['duration_curves']:
                        curve1 = results['duration_curves'][cell1_key]['hours_per_year']
                        curve2 = results['duration_curves'][cell2_key]['hours_per_year']
                        bins1 = results['duration_curves'][cell1_key]['ice_load_bins']
                        bins2 = results['duration_curves'][cell2_key]['ice_load_bins']
                        
                        try:
                            distance = wasserstein_distance(bins1, bins2, curve1, curve2)
                            sn_gradients[i, j] = distance
                        except:
                            print(f"   Warning: Could not compute Earth Mover's distance for cells ({i},{j}) and ({i+1},{j})")
            
            # Calculate combined gradients (average of all valid neighbor distances)
            for i in range(n_south_north):
                for j in range(n_west_east):
                    distances = []
                    
                    # Check East neighbor
                    if j < n_west_east-1 and not np.isnan(ew_gradients[i, j]):
                        distances.append(ew_gradients[i, j])
                    
                    # Check West neighbor
                    if j > 0 and not np.isnan(ew_gradients[i, j-1]):
                        distances.append(ew_gradients[i, j-1])
                    
                    # Check North neighbor
                    if i < n_south_north-1 and not np.isnan(sn_gradients[i, j]):
                        distances.append(sn_gradients[i, j])
                    
                    # Check South neighbor
                    if i > 0 and not np.isnan(sn_gradients[i-1, j]):
                        distances.append(sn_gradients[i-1, j])
                    
                    if distances:
                        combined_gradients[i, j] = np.mean(distances)
            
            # Create the spatial gradient plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: East-West gradients
            im1 = axes[0, 0].imshow(ew_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[0, 0].set_title('East-West Gradient\n(Earth Mover\'s Distance)')
            axes[0, 0].set_xlabel('West-East Grid Points')
            axes[0, 0].set_ylabel('South-North Grid Points')
            cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
            cbar1.set_label('Earth Mover\'s Distance')
            
            # Add grid lines
            axes[0, 0].set_xticks(range(n_west_east-1))
            axes[0, 0].set_yticks(range(n_south_north))
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: South-North gradients
            im2 = axes[0, 1].imshow(sn_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[0, 1].set_title('South-North Gradient\n(Earth Mover\'s Distance)')
            axes[0, 1].set_xlabel('West-East Grid Points')
            axes[0, 1].set_ylabel('South-North Grid Points')
            cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
            cbar2.set_label('Earth Mover\'s Distance')
            
            axes[0, 1].set_xticks(range(n_west_east))
            axes[0, 1].set_yticks(range(n_south_north-1))
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Combined gradients
            im3 = axes[1, 0].imshow(combined_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 0].set_title('Combined Spatial Gradient\n(Average Neighbor Distance)')
            axes[1, 0].set_xlabel('West-East Grid Points')
            axes[1, 0].set_ylabel('South-North Grid Points')
            cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
            cbar3.set_label('Average Wasserstein Distance')
            
            axes[1, 0].set_xticks(range(n_west_east))
            axes[1, 0].set_yticks(range(n_south_north))
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Gradient magnitude (combining EW and SN)
            # Create a full-size gradient magnitude matrix
            gradient_magnitude = np.full((n_south_north, n_west_east), np.nan)
            
            for i in range(n_south_north):
                for j in range(n_west_east):
                    magnitudes = []
                    
                    # East-West component
                    if j < n_west_east-1 and not np.isnan(ew_gradients[i, j]):
                        magnitudes.append(ew_gradients[i, j]**2)
                    if j > 0 and not np.isnan(ew_gradients[i, j-1]):
                        magnitudes.append(ew_gradients[i, j-1]**2)
                    
                    # South-North component  
                    if i < n_south_north-1 and not np.isnan(sn_gradients[i, j]):
                        magnitudes.append(sn_gradients[i, j]**2)
                    if i > 0 and not np.isnan(sn_gradients[i-1, j]):
                        magnitudes.append(sn_gradients[i-1, j]**2)
                    
                    if magnitudes:
                        gradient_magnitude[i, j] = np.sqrt(np.mean(magnitudes))
            
            im4 = axes[1, 1].imshow(gradient_magnitude, cmap='plasma', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 1].set_title('Gradient Magnitude\n(RMS of EW and SN)')
            axes[1, 1].set_xlabel('West-East Grid Points')
            axes[1, 1].set_ylabel('South-North Grid Points')
            cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
            cbar4.set_label('RMS Gradient Magnitude')
            
            axes[1, 1].set_xticks(range(n_west_east))
            axes[1, 1].set_yticks(range(n_south_north))
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save spatial gradient plots
            if save_plots:
                gradient_path = f"{ice_load_plots_dir}/ice_load_spatial_gradients.png"
                plt.savefig(gradient_path, dpi=300, bbox_inches='tight')
                print(f"   Spatial gradient plots saved to: {gradient_path}")
            
            # Store gradient results
            results['spatial_gradients'] = {
                'east_west_gradients': ew_gradients,
                'south_north_gradients': sn_gradients,
                'combined_gradients': combined_gradients,
                'gradient_magnitude': gradient_magnitude,
                'ew_mean': np.nanmean(ew_gradients),
                'ew_std': np.nanstd(ew_gradients),
                'sn_mean': np.nanmean(sn_gradients),
                'sn_std': np.nanstd(sn_gradients),
                'combined_mean': np.nanmean(combined_gradients),
                'combined_std': np.nanstd(combined_gradients)
            }
            
            # Print gradient statistics
            print(f"   Gradient Statistics:")
            print(f"     East-West: Mean = {results['spatial_gradients']['ew_mean']:.3f}, Std = {results['spatial_gradients']['ew_std']:.3f}")
            print(f"     South-North: Mean = {results['spatial_gradients']['sn_mean']:.3f}, Std = {results['spatial_gradients']['sn_std']:.3f}")
            print(f"     Combined: Mean = {results['spatial_gradients']['combined_mean']:.3f}, Std = {results['spatial_gradients']['combined_std']:.3f}")
            
        except ImportError:
            print("   Warning: scipy not available, skipping Earth Mover's distance calculation")
            print("   Install scipy with: conda install scipy")
        except Exception as e:
            print(f"   Error in spatial gradient analysis: {e}")
        
        print(f"\n✓ Ice load duration curve analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in ice load duration analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_ice_load_pdf_curves(ice_load_data, save_plots=True, ice_load_bins=None, ice_load_threshold=0.0):
    """
    Create probability density function (PDF) curves showing the probability density for each ice load level
    for every grid cell. The function creates a mean over all years and plots
    X-axis: Ice load (kg/m)
    Y-axis: Probability density
    
    Parameters:
    -----------
    ice_load_data : xarray.DataArray
        Ice load data with dimensions (time, south_north, west_east)
    save_plots : bool, default True
        Whether to save the plots to files
    ice_load_bins : array-like, optional
        Custom ice load bins for analysis. If None, uses automatic binning
    ice_load_threshold : float, default 0.0
        Minimum ice load value to be plotted (kg/m). Values below this threshold will be excluded
        
    Returns:
    --------
    dict : Dictionary containing analysis results and statistics
    """
    print("=== ICE LOAD PDF CURVE ANALYSIS ===")
    
    try:
        # Check data structure
        print(f"\n1. Data Information:")
        print(f"   Shape: {ice_load_data.shape}")
        print(f"   Dimensions: {ice_load_data.dims}")
        print(f"   Time range: {ice_load_data.time.min().values} to {ice_load_data.time.max().values}")
        
        # Get spatial dimensions
        n_south_north = ice_load_data.sizes['south_north']
        n_west_east = ice_load_data.sizes['west_east']
        n_time = ice_load_data.sizes['time']
        
        print(f"   Grid size: {n_south_north} × {n_west_east} = {n_south_north * n_west_east} cells")
        print(f"   Time steps: {n_time}")
        
        # Convert time to pandas for easier manipulation
        time_index = pd.to_datetime(ice_load_data.time.values)
        n_years = len(time_index.year.unique())
        print(f"   Years covered: {n_years}")
        print(f"   Years: {sorted(time_index.year.unique())}")
        
        # Calculate time step in hours (assuming regular intervals)
        if len(time_index) > 1:
            time_step_hours = (time_index[1] - time_index[0]).total_seconds() / 3600
        else:
            time_step_hours = 0.5  # Default to 30 minutes
        print(f"   Time step: {time_step_hours} hours")
        
        # Remove NaN values and get overall data statistics
        ice_data_clean = ice_load_data.where(ice_load_data >= 0, 0)  # Replace negative/NaN with 0
        max_ice_load = float(ice_data_clean.max())
        min_ice_load = 0.0
        
        print(f"\n2. Ice Load Statistics:")
        print(f"   Range: {min_ice_load:.3f} to {max_ice_load:.3f} kg/m")
        print(f"   Mean: {float(ice_data_clean.mean()):.3f} kg/m")
        
        # Define ice load bins if not provided
        if ice_load_bins is None:
            if max_ice_load > ice_load_threshold:
                # Create bins for PDF analysis - use more bins for better resolution
                ice_load_bins = np.linspace(ice_load_threshold, max_ice_load, 50)
            else:
                ice_load_bins = np.array([ice_load_threshold, ice_load_threshold + 0.01, ice_load_threshold + 0.1, ice_load_threshold + 1, ice_load_threshold + 10])
        else:
            # Filter provided bins to only include those >= threshold
            ice_load_bins = ice_load_bins[ice_load_bins >= ice_load_threshold]
        
        print(f"   Using {len(ice_load_bins)} ice load bins")
        print(f"   Ice load threshold: {ice_load_threshold:.3f} kg/m")
        print(f"   Bin range: {ice_load_bins[0]:.4f} to {ice_load_bins[-1]:.3f} kg/m")
        
        # Prepare results storage
        results = {
            'grid_shape': (n_south_north, n_west_east),
            'n_years': n_years,
            'time_step_hours': time_step_hours,
            'ice_load_bins': ice_load_bins,
            'pdf_curves': {},
            'statistics': {}
        }
        
        # Create figure for all grid cells
        n_cols = min(n_west_east, 5)  # Maximum 5 columns
        n_rows = int(np.ceil((n_south_north * n_west_east) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        print(f"\n3. Processing grid cells...")
        
        cell_count = 0
        for i in range(n_south_north):
            for j in range(n_west_east):
                # Extract time series for this grid cell
                cell_data = ice_data_clean.isel(south_north=i, west_east=j)
                cell_values = cell_data.values
                
                # Remove NaN values
                valid_mask = ~np.isnan(cell_values)
                cell_values_clean = cell_values[valid_mask]
                
                if len(cell_values_clean) == 0:
                    print(f"   Warning: No valid data for cell ({i},{j})")
                    continue
                
                # Filter values to be >= threshold
                cell_values_filtered = cell_values_clean[cell_values_clean >= ice_load_threshold]
                
                if len(cell_values_filtered) == 0:
                    print(f"   Warning: No data above threshold for cell ({i},{j})")
                    continue
                
                # Calculate PDF using histogram
                hist, bin_edges = np.histogram(cell_values_filtered, bins=ice_load_bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Store results for this cell
                cell_key = f'cell_{i}_{j}'
                results['pdf_curves'][cell_key] = {
                    'ice_load_bins': bin_centers,
                    'pdf_values': hist,
                    'position': (i, j)
                }
                
                # Calculate statistics for this cell
                results['statistics'][cell_key] = {
                    'max_ice_load': float(np.max(cell_values_clean)),
                    'mean_ice_load': float(np.mean(cell_values_clean)),
                    'std_ice_load': float(np.std(cell_values_clean)),
                    'median_ice_load': float(np.median(cell_values_clean)),
                    'ice_occurrence_percentage': (len(cell_values_filtered) / len(cell_values_clean)) * 100
                }
                
                # Plot PDF curve for this cell
                if cell_count < len(axes):
                    ax = axes[cell_count]
                    ax.plot(bin_centers, hist, 'b-', linewidth=2, marker='o', markersize=3)
                    ax.set_xlabel('Ice Load (kg/m)')
                    ax.set_ylabel('Probability Density')
                    ax.set_title(f'Cell ({i},{j})')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(left=ice_load_threshold)
                    ax.set_ylim(bottom=0)
                
                cell_count += 1
                
                if cell_count % 5 == 0:
                    print(f"   Processed {cell_count}/{n_south_north * n_west_east} cells...")
        
        # Hide unused subplots
        for idx in range(cell_count, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            # Create specific directory structure for ice load per cell PDF plots
            ice_load_plots_dir = os.path.join(figures_dir, "spatial_gradient", "ice_load_per_cell_pdf")
            os.makedirs(ice_load_plots_dir, exist_ok=True)
            plot_path = f"{ice_load_plots_dir}/ice_load_pdf_curves_all_cells.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\n   PDF curves plot saved to: {plot_path}")
        
        # Create summary plot with mean curve
        print(f"\n4. Creating summary statistics...")
        
        # Calculate mean PDF curve across all cells
        all_pdf_curves = []
        all_bin_centers = []
        for cell_key, cell_data in results['pdf_curves'].items():
            all_pdf_curves.append(cell_data['pdf_values'])
            all_bin_centers.append(cell_data['ice_load_bins'])
        
        if all_pdf_curves:
            # Interpolate all PDFs to common bins for averaging
            common_bins = ice_load_bins[:-1] + np.diff(ice_load_bins) / 2  # bin centers
            interpolated_pdfs = []
            
            for pdf_vals, bin_centers in zip(all_pdf_curves, all_bin_centers):
                interp_pdf = np.interp(common_bins, bin_centers, pdf_vals)
                interpolated_pdfs.append(interp_pdf)
            
            mean_pdf = np.mean(interpolated_pdfs, axis=0)
            std_pdf = np.std(interpolated_pdfs, axis=0)
            
            plt.figure(figsize=(10, 6))
            plt.plot(common_bins, mean_pdf, 'r-', linewidth=3, label='Mean across all cells')
            plt.fill_between(common_bins, mean_pdf - std_pdf, 
                           mean_pdf + std_pdf, alpha=0.3, color='red', 
                           label='±1 Standard Deviation')
            
            plt.xlabel('Ice Load (kg/m)')
            plt.ylabel('Probability Density')
            plt.title('Ice Load PDF - Domain Average')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(left=ice_load_threshold)
            plt.ylim(bottom=0)
            
            plt.tight_layout()
            
            if save_plots:
                summary_path = f"{ice_load_plots_dir}/ice_load_pdf_summary.png"
                plt.savefig(summary_path, dpi=300, bbox_inches='tight')
                print(f"   Summary plot saved to: {summary_path}")
            
            # Store summary statistics
            results['domain_statistics'] = {
                'mean_pdf_curve': mean_pdf,
                'std_pdf_curve': std_pdf,
                'ice_load_bins': common_bins
            }
        
        # Print summary statistics
        print(f"\n5. Summary Statistics:")
        print(f"   Processed {len(results['pdf_curves'])} grid cells")
        
        if results['statistics']:
            all_stats = list(results['statistics'].values())
            mean_ice_occurrence = np.mean([s['ice_occurrence_percentage'] for s in all_stats])
            max_ice_occurrence = np.max([s['ice_occurrence_percentage'] for s in all_stats])
            min_ice_occurrence = np.min([s['ice_occurrence_percentage'] for s in all_stats])
            
            print(f"   Ice occurrence across domain:")
            print(f"     Mean: {mean_ice_occurrence:.1f}% of time")
            print(f"     Range: {min_ice_occurrence:.1f}% to {max_ice_occurrence:.1f}%")
        
        # Create spatial gradient analysis using Earth Mover's Distance
        print(f"\n6. Spatial Gradient Analysis (Earth Mover's Distance)...")
        
        try:
            from scipy.stats import wasserstein_distance
            
            # Initialize gradient matrices
            n_south_north, n_west_east = ice_load_data.shape[1], ice_load_data.shape[2]
            
            # East-West gradients (comparing adjacent cells horizontally)
            ew_gradients = np.full((n_south_north, n_west_east-1), np.nan)
            
            # South-North gradients (comparing adjacent cells vertically)
            sn_gradients = np.full((n_south_north-1, n_west_east), np.nan)
            
            # Combined gradients (for each cell, average of all neighbor comparisons)
            combined_gradients = np.full((n_south_north, n_west_east), np.nan)
            
            print(f"   Computing Earth Mover's distances between neighboring cells...")
            
            # Calculate East-West gradients
            for i in range(n_south_north):
                for j in range(n_west_east-1):
                    cell1_key = f'cell_{i}_{j}'
                    cell2_key = f'cell_{i}_{j+1}'
                    
                    if cell1_key in results['pdf_curves'] and cell2_key in results['pdf_curves']:
                        pdf1 = results['pdf_curves'][cell1_key]['pdf_values']
                        pdf2 = results['pdf_curves'][cell2_key]['pdf_values']
                        bins1 = results['pdf_curves'][cell1_key]['ice_load_bins']
                        bins2 = results['pdf_curves'][cell2_key]['ice_load_bins']
                        
                        # Calculate Earth Mover's distance between the PDF curves
                        try:
                            distance = wasserstein_distance(bins1, bins2, pdf1, pdf2)
                            ew_gradients[i, j] = distance
                        except:
                            print(f"   Warning: Could not compute Wasserstein distance for cells ({i},{j}) and ({i},{j+1})")
            
            # Calculate South-North gradients
            for i in range(n_south_north-1):
                for j in range(n_west_east):
                    cell1_key = f'cell_{i}_{j}'
                    cell2_key = f'cell_{i+1}_{j}'
                    
                    if cell1_key in results['pdf_curves'] and cell2_key in results['pdf_curves']:
                        pdf1 = results['pdf_curves'][cell1_key]['pdf_values']
                        pdf2 = results['pdf_curves'][cell2_key]['pdf_values']
                        bins1 = results['pdf_curves'][cell1_key]['ice_load_bins']
                        bins2 = results['pdf_curves'][cell2_key]['ice_load_bins']
                        
                        try:
                            distance = wasserstein_distance(bins1, bins2, pdf1, pdf2)
                            sn_gradients[i, j] = distance
                        except:
                            print(f"   Warning: Could not compute Earth Mover's distance for cells ({i},{j}) and ({i+1},{j})")
            
            # Calculate combined gradients (average of all valid neighbor distances)
            for i in range(n_south_north):
                for j in range(n_west_east):
                    distances = []
                    
                    # Check East neighbor
                    if j < n_west_east-1 and not np.isnan(ew_gradients[i, j]):
                        distances.append(ew_gradients[i, j])
                    
                    # Check West neighbor
                    if j > 0 and not np.isnan(ew_gradients[i, j-1]):
                        distances.append(ew_gradients[i, j-1])
                    
                    # Check North neighbor
                    if i < n_south_north-1 and not np.isnan(sn_gradients[i, j]):
                        distances.append(sn_gradients[i, j])
                    
                    # Check South neighbor
                    if i > 0 and not np.isnan(sn_gradients[i-1, j]):
                        distances.append(sn_gradients[i-1, j])
                    
                    if distances:
                        combined_gradients[i, j] = np.mean(distances)
            
            # Create the spatial gradient plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: East-West gradients
            im1 = axes[0, 0].imshow(ew_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[0, 0].set_title('East-West Gradient\n(PDF Earth Mover\'s Distance)')
            axes[0, 0].set_xlabel('West-East Grid Points')
            axes[0, 0].set_ylabel('South-North Grid Points')
            cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
            cbar1.set_label('Earth Mover\'s Distance')
            
            # Add grid lines
            axes[0, 0].set_xticks(range(n_west_east-1))
            axes[0, 0].set_yticks(range(n_south_north))
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: South-North gradients
            im2 = axes[0, 1].imshow(sn_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[0, 1].set_title('South-North Gradient\n(PDF Earth Mover\'s Distance)')
            axes[0, 1].set_xlabel('West-East Grid Points')
            axes[0, 1].set_ylabel('South-North Grid Points')
            cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
            cbar2.set_label('Earth Mover\'s Distance')
            
            axes[0, 1].set_xticks(range(n_west_east))
            axes[0, 1].set_yticks(range(n_south_north-1))
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Combined gradients
            im3 = axes[1, 0].imshow(combined_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 0].set_title('Combined Spatial Gradient\n(Average Neighbor Distance)')
            axes[1, 0].set_xlabel('West-East Grid Points')
            axes[1, 0].set_ylabel('South-North Grid Points')
            cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
            cbar3.set_label('Average Earth Mover\'s Distance')
            
            axes[1, 0].set_xticks(range(n_west_east))
            axes[1, 0].set_yticks(range(n_south_north))
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Gradient magnitude (combining EW and SN)
            # Create a full-size gradient magnitude matrix
            gradient_magnitude = np.full((n_south_north, n_west_east), np.nan)
            
            for i in range(n_south_north):
                for j in range(n_west_east):
                    magnitudes = []
                    
                    # East-West component
                    if j < n_west_east-1 and not np.isnan(ew_gradients[i, j]):
                        magnitudes.append(ew_gradients[i, j]**2)
                    if j > 0 and not np.isnan(ew_gradients[i, j-1]):
                        magnitudes.append(ew_gradients[i, j-1]**2)
                    
                    # South-North component  
                    if i < n_south_north-1 and not np.isnan(sn_gradients[i, j]):
                        magnitudes.append(sn_gradients[i, j]**2)
                    if i > 0 and not np.isnan(sn_gradients[i-1, j]):
                        magnitudes.append(sn_gradients[i-1, j]**2)
                    
                    if magnitudes:
                        gradient_magnitude[i, j] = np.sqrt(np.mean(magnitudes))
            
            im4 = axes[1, 1].imshow(gradient_magnitude, cmap='plasma', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 1].set_title('Gradient Magnitude\n(RMS of EW and SN)')
            axes[1, 1].set_xlabel('West-East Grid Points')
            axes[1, 1].set_ylabel('South-North Grid Points')
            cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
            cbar4.set_label('RMS Gradient Magnitude')
            
            axes[1, 1].set_xticks(range(n_west_east))
            axes[1, 1].set_yticks(range(n_south_north))
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save spatial gradient plots
            if save_plots:
                gradient_path = f"{ice_load_plots_dir}/ice_load_pdf_spatial_gradients.png"
                plt.savefig(gradient_path, dpi=300, bbox_inches='tight')
                print(f"   Spatial gradient plots saved to: {gradient_path}")
            
            # Store gradient results
            results['spatial_gradients'] = {
                'east_west_gradients': ew_gradients,
                'south_north_gradients': sn_gradients,
                'combined_gradients': combined_gradients,
                'gradient_magnitude': gradient_magnitude,
                'ew_mean': np.nanmean(ew_gradients),
                'ew_std': np.nanstd(ew_gradients),
                'sn_mean': np.nanmean(sn_gradients),
                'sn_std': np.nanstd(sn_gradients),
                'combined_mean': np.nanmean(combined_gradients),
                'combined_std': np.nanstd(combined_gradients)
            }
            
            # Print gradient statistics
            print(f"   Gradient Statistics:")
            print(f"     East-West: Mean = {results['spatial_gradients']['ew_mean']:.3f}, Std = {results['spatial_gradients']['ew_std']:.3f}")
            print(f"     South-North: Mean = {results['spatial_gradients']['sn_mean']:.3f}, Std = {results['spatial_gradients']['sn_std']:.3f}")
            print(f"     Combined: Mean = {results['spatial_gradients']['combined_mean']:.3f}, Std = {results['spatial_gradients']['combined_std']:.3f}")
            
        except ImportError:
            print("   Warning: scipy not available, skipping Earth Mover's distance calculation")
            print("   Install scipy with: conda install scipy")
        except Exception as e:
            print(f"   Error in spatial gradient analysis: {e}")
        
        print(f"\n✓ Ice load PDF analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in ice load PDF analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_ice_load_cdf_curves(ice_load_data, save_plots=True, ice_load_bins=None, ice_load_threshold=0.0, months=None, percentile=None):
    """
    Create cumulative distribution function (CDF) curves showing the cumulative probability for each ice load level
    for every grid cell. The function creates a mean over all years and plots
    X-axis: Ice load (kg/m)
    Y-axis: Cumulative probability
    
    Parameters:
    -----------
    ice_load_data : xarray.DataArray
        Ice load data with dimensions (time, south_north, west_east)
    save_plots : bool, default True
        Whether to save the plots to files
    ice_load_bins : array-like, optional
        Custom ice load bins for analysis. If None, uses automatic binning
    ice_load_threshold : float, default 0.0
        Minimum ice load value to be included in CDF calculation (kg/m). Values below this threshold 
        will be completely excluded from the probability calculation, not just from plotting
    months : list of int, optional
        List of months to include in analysis (1-12). If None, uses all months.
        Example: [12, 1, 2, 3, 4] for winter months (Dec-Apr)
    percentile : float, optional
        Percentile value (0-100) to use for filtering extreme values before CDF calculation.
        If specified, only data below this percentile will be used for analysis.
        Example: percentile=95 will exclude the top 5% of values
        
    Returns:
    --------
    dict : Dictionary containing analysis results and statistics
    """
    print("=== ICE LOAD CDF CURVE ANALYSIS ===")
    
    try:
        # Check data structure
        print(f"\n1. Data Information:")
        print(f"   Shape: {ice_load_data.shape}")
        print(f"   Dimensions: {ice_load_data.dims}")
        print(f"   Time range: {ice_load_data.time.min().values} to {ice_load_data.time.max().values}")
        
        # Get spatial dimensions
        n_south_north = ice_load_data.sizes['south_north']
        n_west_east = ice_load_data.sizes['west_east']
        n_time = ice_load_data.sizes['time']
        
        print(f"   Grid size: {n_south_north} × {n_west_east} = {n_south_north * n_west_east} cells")
        print(f"   Time steps: {n_time}")
        
        # Convert time to pandas for easier manipulation
        time_index = pd.to_datetime(ice_load_data.time.values)
        n_years = len(time_index.year.unique())
        print(f"   Years covered: {n_years}")
        print(f"   Years: {sorted(time_index.year.unique())}")
        
        # Calculate time step in hours (assuming regular intervals)
        if len(time_index) > 1:
            time_step_hours = (time_index[1] - time_index[0]).total_seconds() / 3600
        else:
            time_step_hours = 0.5  # Default to 30 minutes
        print(f"   Time step: {time_step_hours} hours")
        
        # Remove NaN values and get overall data statistics
        ice_data_clean = ice_load_data.where(ice_load_data >= 0, 0)  # Replace negative/NaN with 0
        
        # Filter data by months if specified
        if months is not None:
            print(f"\n   Filtering data to specified months only: {months}...")
            time_index_full = pd.to_datetime(ice_data_clean.time.values)
            month_mask = time_index_full.month.isin(months)
            ice_data_filtered = ice_data_clean.isel(time=month_mask)
            
            # Update time information after filtering
            time_index_filtered = pd.to_datetime(ice_data_filtered.time.values)
            n_filtered_timesteps = len(time_index_filtered)
            
            print(f"   Original timesteps: {n_time}")
            print(f"   Filtered timesteps: {n_filtered_timesteps}")
            print(f"   Months included: {sorted(time_index_filtered.month.unique())}")
            print(f"   Reduction: {((n_time - n_filtered_timesteps) / n_time * 100):.1f}% timesteps removed")
            
            # Use filtered data for analysis
            ice_data_analysis = ice_data_filtered
        else:
            print(f"\n   Using all months for analysis...")
            ice_data_analysis = ice_data_clean
        
        max_ice_load = float(ice_data_analysis.max())
        min_ice_load = 0.0
        
        # Apply percentile filtering if specified
        if percentile is not None:
            if 0 < percentile < 100:
                percentile_value = float(np.nanpercentile(ice_data_analysis.values, percentile))
                if percentile_value < max_ice_load:
                    print(f"   Applying {percentile}th percentile threshold: {percentile_value:.3f} kg/m")
                    print(f"   Original max ice load: {max_ice_load:.3f} kg/m")
                    max_ice_load = percentile_value
                    # Filter the data to exclude values above percentile threshold
                    ice_data_analysis = ice_data_analysis.where(ice_data_analysis <= percentile_value, np.nan)
                    print(f"   Data filtered to exclude ice loads > {percentile_value:.3f} kg/m ({100-percentile:.1f}% of extreme values removed)")
                else:
                    print(f"   {percentile}th percentile ({percentile_value:.3f} kg/m) is above current data maximum, no additional filtering applied")
            else:
                print(f"   Warning: Invalid percentile value ({percentile}). Must be between 0 and 100. Skipping percentile filtering.")
        
        month_info = f" ({months})" if months is not None else " (all months)"
        percentile_info = f", {percentile}th percentile: {max_ice_load:.3f}" if percentile is not None else ""
        print(f"\n2. Ice Load Statistics{month_info}:")
        print(f"   Range: {min_ice_load:.3f} to {max_ice_load:.3f} kg/m")
        print(f"   Thresholds - min: {ice_load_threshold:.3f}{percentile_info} kg/m")
        print(f"   Mean: {float(ice_data_analysis.mean()):.3f} kg/m")
        
        # Define ice load bins if not provided
        if ice_load_bins is None:
            if max_ice_load > ice_load_threshold:
                # Create bins for CDF analysis - use more bins for better resolution
                # Always start from 0 for proper CDF calculation, but filter plotting later
                ice_load_bins = np.linspace(0.0, max_ice_load, 100)
            else:
                ice_load_bins = np.array([0.0, ice_load_threshold + 0.01, ice_load_threshold + 0.1, ice_load_threshold + 1, ice_load_threshold + 10])
        else:
            # Ensure bins start from 0 for proper CDF calculation
            if ice_load_bins[0] > 0:
                ice_load_bins = np.concatenate([[0.0], ice_load_bins])
            ice_load_bins = np.sort(ice_load_bins)
        
        print(f"   Using {len(ice_load_bins)} ice load bins")
        print(f"   Ice load threshold: {ice_load_threshold:.3f} kg/m")
        print(f"   Bin range: {ice_load_bins[0]:.4f} to {ice_load_bins[-1]:.3f} kg/m")
        
        # Include filtering information in results
        filtering_applied = {
            'ice_load_threshold': ice_load_threshold,
            'percentile': percentile,
            'months': months
        }
        
        # Prepare results storage
        results = {
            'grid_shape': (n_south_north, n_west_east),
            'n_years': n_years,
            'time_step_hours': time_step_hours,
            'ice_load_bins': ice_load_bins,
            'cdf_curves': {},
            'statistics': {}
        }
        
        # Create figure for all grid cells
        n_cols = min(n_west_east, 5)  # Maximum 5 columns
        n_rows = int(np.ceil((n_south_north * n_west_east) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        print(f"\n3. Processing grid cells...")
        
        cell_count = 0
        for i in range(n_south_north):
            for j in range(n_west_east):
                # Extract time series for this grid cell (using filtered data)
                cell_data = ice_data_analysis.isel(south_north=i, west_east=j)
                cell_values = cell_data.values
                
                # Remove NaN values
                valid_mask = ~np.isnan(cell_values)
                cell_values_clean = cell_values[valid_mask]
                
                if len(cell_values_clean) == 0:
                    print(f"   Warning: No valid data for cell ({i},{j})")
                    continue
                
                # Filter values to be >= threshold
                cell_values_filtered = cell_values_clean[cell_values_clean >= ice_load_threshold]
                
                if len(cell_values_filtered) == 0:
                    print(f"   Warning: No data above threshold for cell ({i},{j})")
                    continue
                
                # Calculate CDF using only filtered values above threshold
                cdf_values = []
                for ice_threshold in ice_load_bins:
                    # Calculate cumulative probability P(X <= ice_threshold) using only filtered data
                    # This excludes datapoints below ice_load_threshold from the CDF calculation
                    prob = np.sum(cell_values_filtered <= ice_threshold) / len(cell_values_filtered)
                    cdf_values.append(prob)
                
                cdf_values = np.array(cdf_values)
                
                # Store results for this cell
                cell_key = f'cell_{i}_{j}'
                results['cdf_curves'][cell_key] = {
                    'ice_load_bins': ice_load_bins,
                    'cdf_values': cdf_values,
                    'position': (i, j)
                }
                
                # Calculate statistics for this cell (using filtered data for consistency with CDF)
                results['statistics'][cell_key] = {
                    'max_ice_load': float(np.max(cell_values_filtered)),
                    'mean_ice_load': float(np.mean(cell_values_filtered)),
                    'std_ice_load': float(np.std(cell_values_filtered)),
                    'median_ice_load': float(np.median(cell_values_filtered)),
                    'ice_occurrence_percentage': (len(cell_values_filtered) / len(cell_values_clean)) * 100,
                    'percentile_95': float(np.percentile(cell_values_filtered, 95)),
                    'percentile_99': float(np.percentile(cell_values_filtered, 99))
                }
                
                # Plot CDF curve for this cell
                if cell_count < len(axes):
                    ax = axes[cell_count]
                    # Only plot bins within thresholds for visibility
                    plot_mask = ice_load_bins >= ice_load_threshold
                    
                    plot_bins = ice_load_bins[plot_mask]
                    plot_cdf = cdf_values[plot_mask]
                    ax.plot(plot_bins, plot_cdf, 'b-', linewidth=2, marker='o', markersize=3)
                    ax.set_xlabel('Ice Load (kg/m)')
                    ax.set_ylabel('Cumulative Probability')
                    ax.set_title(f'Cell ({i},{j})')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(left=ice_load_threshold)
                    ax.set_ylim([0, 1])
                
                cell_count += 1
                
                if cell_count % 5 == 0:
                    print(f"   Processed {cell_count}/{n_south_north * n_west_east} cells...")
        
        # Hide unused subplots
        for idx in range(cell_count, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            # Create specific directory structure for ice load per cell CDF plots
            ice_load_plots_dir = os.path.join(figures_dir, "spatial_gradient", "ice_load_per_cell_cdf")
            os.makedirs(ice_load_plots_dir, exist_ok=True)
            
            # Create filename with filtering information
            filename_parts = ["ice_load_cdf_curves_all_cells"]
            if months is not None:
                months_str = "_".join(map(str, sorted(months)))
                filename_parts.append(f"months_{months_str}")
            if ice_load_threshold > 0:
                filename_parts.append(f"min_{ice_load_threshold:.1f}")
            if percentile is not None:
                filename_parts.append(f"p{percentile}")
            
            plot_filename = "_".join(filename_parts) + ".png"
            plot_path = f"{ice_load_plots_dir}/{plot_filename}"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\n   CDF curves plot saved to: {plot_path}")
        
        # Create summary plot with mean curve
        print(f"\n4. Creating summary statistics...")
        
        # Calculate mean CDF curve across all cells
        all_cdf_curves = []
        for cell_key, cell_data in results['cdf_curves'].items():
            all_cdf_curves.append(cell_data['cdf_values'])
        
        if all_cdf_curves:
            mean_cdf = np.mean(all_cdf_curves, axis=0)
            std_cdf = np.std(all_cdf_curves, axis=0)
            
            plt.figure(figsize=(10, 6))
            # Only plot bins within thresholds for visibility
            plot_mask = ice_load_bins >= ice_load_threshold
            
            plot_bins = ice_load_bins[plot_mask]
            plot_mean_cdf = mean_cdf[plot_mask]
            plot_std_cdf = std_cdf[plot_mask]
            
            plt.plot(plot_bins, plot_mean_cdf, 'r-', linewidth=3, label='Mean across all cells')
            plt.fill_between(plot_bins, plot_mean_cdf - plot_std_cdf, 
                           plot_mean_cdf + plot_std_cdf, alpha=0.3, color='red', 
                           label='±1 Standard Deviation')
            
            plt.xlabel('Ice Load (kg/m)')
            plt.ylabel('Cumulative Probability')
            plt.title('Ice Load CDF - Domain Average')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(left=ice_load_threshold)
            plt.ylim([0, 1])
            
            plt.tight_layout()
            
            if save_plots:
                # Create summary filename with filtering information
                summary_filename_parts = ["ice_load_cdf_summary"]
                if months is not None:
                    months_str = "_".join(map(str, sorted(months)))
                    summary_filename_parts.append(f"months_{months_str}")
                if ice_load_threshold > 0:
                    summary_filename_parts.append(f"min_{ice_load_threshold:.1f}")
                if percentile is not None:
                    summary_filename_parts.append(f"p{percentile}")
                
                summary_filename = "_".join(summary_filename_parts) + ".png"
                summary_path = f"{ice_load_plots_dir}/{summary_filename}"
                plt.savefig(summary_path, dpi=300, bbox_inches='tight')
                print(f"   Summary plot saved to: {summary_path}")
            
            # Store summary statistics
            results['domain_statistics'] = {
                'mean_cdf_curve': mean_cdf,
                'std_cdf_curve': std_cdf,
                'ice_load_bins': ice_load_bins
            }
        
        # Print summary statistics
        print(f"\n5. Summary Statistics:")
        print(f"   Processed {len(results['cdf_curves'])} grid cells")
        
        if results['statistics']:
            all_stats = list(results['statistics'].values())
            mean_ice_occurrence = np.mean([s['ice_occurrence_percentage'] for s in all_stats])
            max_ice_occurrence = np.max([s['ice_occurrence_percentage'] for s in all_stats])
            min_ice_occurrence = np.min([s['ice_occurrence_percentage'] for s in all_stats])
            mean_p95 = np.mean([s['percentile_95'] for s in all_stats])
            mean_p99 = np.mean([s['percentile_99'] for s in all_stats])
            
            print(f"   Ice occurrence across domain:")
            print(f"     Mean: {mean_ice_occurrence:.1f}% of time")
            print(f"     Range: {min_ice_occurrence:.1f}% to {max_ice_occurrence:.1f}%")
            print(f"   Domain average percentiles:")
            print(f"     95th percentile: {mean_p95:.3f} kg/m")
            print(f"     99th percentile: {mean_p99:.3f} kg/m")
        
        # Create spatial gradient analysis using Earth Mover's Distance
        print(f"\n6. Spatial Gradient Analysis (Earth Mover's Distance)...")
        
        try:
            from scipy.stats import wasserstein_distance
            
            # Initialize gradient matrices
            n_south_north, n_west_east = ice_load_data.shape[1], ice_load_data.shape[2]
            
            # East-West gradients (comparing adjacent cells horizontally)
            ew_gradients = np.full((n_south_north, n_west_east-1), np.nan)
            
            # South-North gradients (comparing adjacent cells vertically)
            sn_gradients = np.full((n_south_north-1, n_west_east), np.nan)
            
            # Combined gradients (for each cell, average of all neighbor comparisons)
            combined_gradients = np.full((n_south_north, n_west_east), np.nan)
            
            print(f"   Computing Earth Mover's distances between neighboring cells...")
            
            # Calculate East-West gradients
            for i in range(n_south_north):
                for j in range(n_west_east-1):
                    cell1_key = f'cell_{i}_{j}'
                    cell2_key = f'cell_{i}_{j+1}'
                    
                    if cell1_key in results['cdf_curves'] and cell2_key in results['cdf_curves']:
                        cdf1 = results['cdf_curves'][cell1_key]['cdf_values']
                        cdf2 = results['cdf_curves'][cell2_key]['cdf_values']
                        bins1 = results['cdf_curves'][cell1_key]['ice_load_bins']
                        bins2 = results['cdf_curves'][cell2_key]['ice_load_bins']
                        
                        # For CDF, Earth Mover's distance can be calculated using the L1 distance
                        # between CDFs, which is mathematically equivalent
                        try:
                            distance = np.trapz(np.abs(cdf1 - cdf2), bins1)
                            ew_gradients[i, j] = distance
                        except:
                            # Fallback to simple mean absolute difference
                            distance = np.mean(np.abs(cdf1 - cdf2))
                            ew_gradients[i, j] = distance
            
            # Calculate South-North gradients
            for i in range(n_south_north-1):
                for j in range(n_west_east):
                    cell1_key = f'cell_{i}_{j}'
                    cell2_key = f'cell_{i+1}_{j}'
                    
                    if cell1_key in results['cdf_curves'] and cell2_key in results['cdf_curves']:
                        cdf1 = results['cdf_curves'][cell1_key]['cdf_values']
                        cdf2 = results['cdf_curves'][cell2_key]['cdf_values']
                        bins1 = results['cdf_curves'][cell1_key]['ice_load_bins']
                        bins2 = results['cdf_curves'][cell2_key]['ice_load_bins']
                        
                        try:
                            distance = np.trapz(np.abs(cdf1 - cdf2), bins1)
                            sn_gradients[i, j] = distance
                        except:
                            distance = np.mean(np.abs(cdf1 - cdf2))
                            sn_gradients[i, j] = distance
            
            # Calculate combined gradients (average of all valid neighbor distances)
            for i in range(n_south_north):
                for j in range(n_west_east):
                    distances = []
                    
                    # Check East neighbor
                    if j < n_west_east-1 and not np.isnan(ew_gradients[i, j]):
                        distances.append(ew_gradients[i, j])
                    
                    # Check West neighbor
                    if j > 0 and not np.isnan(ew_gradients[i, j-1]):
                        distances.append(ew_gradients[i, j-1])
                    
                    # Check North neighbor
                    if i < n_south_north-1 and not np.isnan(sn_gradients[i, j]):
                        distances.append(sn_gradients[i, j])
                    
                    # Check South neighbor
                    if i > 0 and not np.isnan(sn_gradients[i-1, j]):
                        distances.append(sn_gradients[i-1, j])
                    
                    if distances:
                        combined_gradients[i, j] = np.mean(distances)
            
            # Create the spatial gradient plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: East-West gradients
            im1 = axes[0, 0].imshow(ew_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[0, 0].set_title('East-West Gradient\n(CDF Earth Mover\'s Distance)')
            axes[0, 0].set_xlabel('West-East Grid Points')
            axes[0, 0].set_ylabel('South-North Grid Points')
            cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
            cbar1.set_label('Earth Mover\'s Distance (kg/m)')
            
            # Add grid lines
            axes[0, 0].set_xticks(range(n_west_east-1))
            axes[0, 0].set_yticks(range(n_south_north))
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: South-North gradients
            im2 = axes[0, 1].imshow(sn_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[0, 1].set_title('South-North Gradient\n(CDF Earth Mover\'s Distance)')
            axes[0, 1].set_xlabel('West-East Grid Points')
            axes[0, 1].set_ylabel('South-North Grid Points')
            cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
            cbar2.set_label('Earth Mover\'s Distance (kg/m)')
            
            axes[0, 1].set_xticks(range(n_west_east))
            axes[0, 1].set_yticks(range(n_south_north-1))
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Combined gradients
            im3 = axes[1, 0].imshow(combined_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 0].set_title('Combined Spatial Gradient\n(Average Neighbor Distance)')
            axes[1, 0].set_xlabel('West-East Grid Points')
            axes[1, 0].set_ylabel('South-North Grid Points')
            cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
            cbar3.set_label('Average Earth Mover\'s Distance (kg/m)')
            
            axes[1, 0].set_xticks(range(n_west_east))
            axes[1, 0].set_yticks(range(n_south_north))
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Gradient magnitude (combining EW and SN)
            # Create a full-size gradient magnitude matrix
            gradient_magnitude = np.full((n_south_north, n_west_east), np.nan)
            
            for i in range(n_south_north):
                for j in range(n_west_east):
                    magnitudes = []
                    
                    # East-West component
                    if j < n_west_east-1 and not np.isnan(ew_gradients[i, j]):
                        magnitudes.append(ew_gradients[i, j]**2)
                    if j > 0 and not np.isnan(ew_gradients[i, j-1]):
                        magnitudes.append(ew_gradients[i, j-1]**2)
                    
                    # South-North component  
                    if i < n_south_north-1 and not np.isnan(sn_gradients[i, j]):
                        magnitudes.append(sn_gradients[i, j]**2)
                    if i > 0 and not np.isnan(sn_gradients[i-1, j]):
                        magnitudes.append(sn_gradients[i-1, j]**2)
                    
                    if magnitudes:
                        gradient_magnitude[i, j] = np.sqrt(np.mean(magnitudes))
            
            im4 = axes[1, 1].imshow(gradient_magnitude, cmap='plasma', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 1].set_title('Gradient Magnitude\n(RMS of EW and SN)')
            axes[1, 1].set_xlabel('West-East Grid Points')
            axes[1, 1].set_ylabel('South-North Grid Points')
            cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
            cbar4.set_label('RMS Gradient Magnitude')
            
            axes[1, 1].set_xticks(range(n_west_east))
            axes[1, 1].set_yticks(range(n_south_north))
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save absolute spatial gradient plots
            if save_plots:
                # Create gradient filename with filtering information
                gradient_filename_parts = ["ice_load_cdf_spatial_gradients_absolute"]
                if months is not None:
                    months_str = "_".join(map(str, sorted(months)))
                    gradient_filename_parts.append(f"months_{months_str}")
                if ice_load_threshold > 0:
                    gradient_filename_parts.append(f"min_{ice_load_threshold:.1f}")
                if percentile is not None:
                    gradient_filename_parts.append(f"p{percentile}")
                
                gradient_filename = "_".join(gradient_filename_parts) + ".png"
                gradient_path = f"{ice_load_plots_dir}/{gradient_filename}"
                plt.savefig(gradient_path, dpi=300, bbox_inches='tight')
                print(f"   Absolute spatial gradient plots saved to: {gradient_path}")
            
            plt.close()  # Close the absolute gradient figure
            
            # Calculate dimensionless values (relative to domain mean)
            print(f"   Creating dimensionless gradient plots...")
            
            # Calculate domain means for normalization
            ew_mean = np.nanmean(ew_gradients)
            sn_mean = np.nanmean(sn_gradients)
            combined_mean = np.nanmean(combined_gradients)
            gradient_magnitude_mean = np.nanmean(gradient_magnitude)
            
            # Create dimensionless matrices
            ew_gradients_normalized = ew_gradients / ew_mean if ew_mean > 0 else ew_gradients
            sn_gradients_normalized = sn_gradients / sn_mean if sn_mean > 0 else sn_gradients
            combined_gradients_normalized = combined_gradients / combined_mean if combined_mean > 0 else combined_gradients
            gradient_magnitude_normalized = gradient_magnitude / gradient_magnitude_mean if gradient_magnitude_mean > 0 else gradient_magnitude
            
            print(f"     Normalization factors:")
            print(f"       East-West mean: {ew_mean:.3f} kg/m")
            print(f"       South-North mean: {sn_mean:.3f} kg/m")
            print(f"       Combined mean: {combined_mean:.3f} kg/m")
            print(f"       Gradient magnitude mean: {gradient_magnitude_mean:.3f} kg/m")
            
            # Create the dimensionless spatial gradient plots
            fig_norm, axes_norm = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: East-West gradients (normalized)
            im1_norm = axes_norm[0, 0].imshow(ew_gradients_normalized, cmap='RdBu_r', origin='lower', 
                                             interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
            axes_norm[0, 0].set_title('East-West Gradient\n(Dimensionless, Relative to Domain Mean)')
            axes_norm[0, 0].set_xlabel('West-East Grid Points')
            axes_norm[0, 0].set_ylabel('South-North Grid Points')
            cbar1_norm = plt.colorbar(im1_norm, ax=axes_norm[0, 0], shrink=0.8)
            cbar1_norm.set_label('Gradient / Domain Mean')
            
            # Add grid lines
            axes_norm[0, 0].set_xticks(range(n_west_east-1))
            axes_norm[0, 0].set_yticks(range(n_south_north))
            axes_norm[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: South-North gradients (normalized)
            im2_norm = axes_norm[0, 1].imshow(sn_gradients_normalized, cmap='RdBu_r', origin='lower', 
                                             interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
            axes_norm[0, 1].set_title('South-North Gradient\n(Dimensionless, Relative to Domain Mean)')
            axes_norm[0, 1].set_xlabel('West-East Grid Points')
            axes_norm[0, 1].set_ylabel('South-North Grid Points')
            cbar2_norm = plt.colorbar(im2_norm, ax=axes_norm[0, 1], shrink=0.8)
            cbar2_norm.set_label('Gradient / Domain Mean')
            
            axes_norm[0, 1].set_xticks(range(n_west_east))
            axes_norm[0, 1].set_yticks(range(n_south_north-1))
            axes_norm[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Combined gradients (normalized)
            im3_norm = axes_norm[1, 0].imshow(combined_gradients_normalized, cmap='RdBu_r', origin='lower', 
                                             interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
            axes_norm[1, 0].set_title('Combined Spatial Gradient\n(Dimensionless, Relative to Domain Mean)')
            axes_norm[1, 0].set_xlabel('West-East Grid Points')
            axes_norm[1, 0].set_ylabel('South-North Grid Points')
            cbar3_norm = plt.colorbar(im3_norm, ax=axes_norm[1, 0], shrink=0.8)
            cbar3_norm.set_label('Gradient / Domain Mean')
            
            axes_norm[1, 0].set_xticks(range(n_west_east))
            axes_norm[1, 0].set_yticks(range(n_south_north))
            axes_norm[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Gradient magnitude (normalized)
            im4_norm = axes_norm[1, 1].imshow(gradient_magnitude_normalized, cmap='RdBu_r', origin='lower', 
                                             interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
            axes_norm[1, 1].set_title('Gradient Magnitude\n(Dimensionless, Relative to Domain Mean)')
            axes_norm[1, 1].set_xlabel('West-East Grid Points')
            axes_norm[1, 1].set_ylabel('South-North Grid Points')
            cbar4_norm = plt.colorbar(im4_norm, ax=axes_norm[1, 1], shrink=0.8)
            cbar4_norm.set_label('Gradient / Domain Mean')
            
            axes_norm[1, 1].set_xticks(range(n_west_east))
            axes_norm[1, 1].set_yticks(range(n_south_north))
            axes_norm[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save dimensionless spatial gradient plots
            if save_plots:
                # Create dimensionless gradient filename with filtering information
                dimensionless_filename_parts = ["ice_load_cdf_spatial_gradients_dimensionless"]
                if months is not None:
                    months_str = "_".join(map(str, sorted(months)))
                    dimensionless_filename_parts.append(f"months_{months_str}")
                if ice_load_threshold > 0:
                    dimensionless_filename_parts.append(f"min_{ice_load_threshold:.1f}")
                if percentile is not None:
                    dimensionless_filename_parts.append(f"p{percentile}")
                
                dimensionless_filename = "_".join(dimensionless_filename_parts) + ".png"
                dimensionless_path = f"{ice_load_plots_dir}/{dimensionless_filename}"
                plt.savefig(dimensionless_path, dpi=300, bbox_inches='tight')
                print(f"   Dimensionless spatial gradient plots saved to: {dimensionless_path}")
            
            plt.close()  # Close the dimensionless gradient figure
            
            # Store gradient results (including both absolute and dimensionless)
            results['spatial_gradients'] = {
                'east_west_gradients': ew_gradients,
                'south_north_gradients': sn_gradients,
                'combined_gradients': combined_gradients,
                'gradient_magnitude': gradient_magnitude,
                'east_west_gradients_normalized': ew_gradients_normalized,
                'south_north_gradients_normalized': sn_gradients_normalized,
                'combined_gradients_normalized': combined_gradients_normalized,
                'gradient_magnitude_normalized': gradient_magnitude_normalized,
                'normalization_factors': {
                    'ew_mean': ew_mean,
                    'sn_mean': sn_mean,
                    'combined_mean': combined_mean,
                    'gradient_magnitude_mean': gradient_magnitude_mean
                },
                'ew_mean': np.nanmean(ew_gradients),
                'ew_std': np.nanstd(ew_gradients),
                'sn_mean': np.nanmean(sn_gradients),
                'sn_std': np.nanstd(sn_gradients),
                'combined_mean': np.nanmean(combined_gradients),
                'combined_std': np.nanstd(combined_gradients)
            }
            
            # Print gradient statistics
            print(f"   Gradient Statistics:")
            print(f"     Absolute values:")
            print(f"       East-West: Mean = {results['spatial_gradients']['ew_mean']:.3f}, Std = {results['spatial_gradients']['ew_std']:.3f}")
            print(f"       South-North: Mean = {results['spatial_gradients']['sn_mean']:.3f}, Std = {results['spatial_gradients']['sn_std']:.3f}")
            print(f"       Combined: Mean = {results['spatial_gradients']['combined_mean']:.3f}, Std = {results['spatial_gradients']['combined_std']:.3f}")
            print(f"     Dimensionless values (relative to domain mean):")
            print(f"       East-West: Mean = {np.nanmean(ew_gradients_normalized):.3f}, Std = {np.nanstd(ew_gradients_normalized):.3f}")
            print(f"       South-North: Mean = {np.nanmean(sn_gradients_normalized):.3f}, Std = {np.nanstd(sn_gradients_normalized):.3f}")
            print(f"       Combined: Mean = {np.nanmean(combined_gradients_normalized):.3f}, Std = {np.nanstd(combined_gradients_normalized):.3f}")
            print(f"       Gradient Magnitude: Mean = {np.nanmean(gradient_magnitude_normalized):.3f}, Std = {np.nanstd(gradient_magnitude_normalized):.3f}")
            
        except ImportError:
            print("   Warning: scipy not available, skipping Earth Mover's distance calculation")
            print("   Install scipy with: conda install scipy")
        except Exception as e:
            print(f"   Error in spatial gradient analysis: {e}")
        
        print(f"\n✓ Ice load CDF analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in ice load CDF analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_ice_load_cdf_curves_log_scale(ice_load_data, save_plots=True, ice_load_bins=None, ice_load_threshold=0.0, months=None):
    """
    Create cumulative distribution function (CDF) curves in logarithmic scale showing the cumulative probability for each ice load level
    for every grid cell. The logarithmic scale helps visualize differences when most probabilities are close to 1.
    X-axis: Ice load (kg/m)
    Y-axis: Cumulative probability (log scale)
    
    Parameters:
    -----------
    ice_load_data : xarray.DataArray
        Ice load data with dimensions (time, south_north, west_east)
    save_plots : bool, default True
        Whether to save the plots to files
    ice_load_bins : array-like, optional
        Custom ice load bins for analysis. If None, uses automatic binning
    ice_load_threshold : float, default 0.0
        Minimum ice load value to be plotted (kg/m). Values below this threshold will be excluded
    months : list of int, optional
        List of months to include in analysis (1-12). If None, uses all months.
        Example: [12, 1, 2, 3, 4] for winter months (Dec-Apr)
        
    Returns:
    --------
    dict : Dictionary containing analysis results and statistics
    """
    print("=== ICE LOAD CDF CURVE ANALYSIS (LOG SCALE) ===")
    
    try:
        # Check data structure
        print(f"\n1. Data Information:")
        print(f"   Shape: {ice_load_data.shape}")
        print(f"   Dimensions: {ice_load_data.dims}")
        print(f"   Time range: {ice_load_data.time.min().values} to {ice_load_data.time.max().values}")
        
        # Get spatial dimensions
        n_south_north = ice_load_data.sizes['south_north']
        n_west_east = ice_load_data.sizes['west_east']
        n_time = ice_load_data.sizes['time']
        
        print(f"   Grid size: {n_south_north} × {n_west_east} = {n_south_north * n_west_east} cells")
        print(f"   Time steps: {n_time}")
        
        # Convert time to pandas for easier manipulation
        time_index = pd.to_datetime(ice_load_data.time.values)
        n_years = len(time_index.year.unique())
        print(f"   Years covered: {n_years}")
        print(f"   Years: {sorted(time_index.year.unique())}")
        
        # Calculate time step in hours (assuming regular intervals)
        if len(time_index) > 1:
            time_step_hours = (time_index[1] - time_index[0]).total_seconds() / 3600
        else:
            time_step_hours = 0.5  # Default to 30 minutes
        print(f"   Time step: {time_step_hours} hours")
        
        # Remove NaN values and get overall data statistics
        ice_data_clean = ice_load_data.where(ice_load_data >= 0, 0)  # Replace negative/NaN with 0
        
        # Filter data by months if specified
        if months is not None:
            print(f"\n   Filtering data to specified months only: {months}...")
            time_index_full = pd.to_datetime(ice_data_clean.time.values)
            month_mask = time_index_full.month.isin(months)
            ice_data_filtered = ice_data_clean.isel(time=month_mask)
            
            # Update time information after filtering
            time_index_filtered = pd.to_datetime(ice_data_filtered.time.values)
            n_filtered_timesteps = len(time_index_filtered)
            
            print(f"   Original timesteps: {n_time}")
            print(f"   Filtered timesteps: {n_filtered_timesteps}")
            print(f"   Months included: {sorted(time_index_filtered.month.unique())}")
            print(f"   Reduction: {((n_time - n_filtered_timesteps) / n_time * 100):.1f}% timesteps removed")
            
            # Use filtered data for analysis
            ice_data_analysis = ice_data_filtered
        else:
            print(f"\n   Using all months for analysis...")
            ice_data_analysis = ice_data_clean
        
        max_ice_load = float(ice_data_analysis.max())
        min_ice_load = 0.0
        
        month_info = f" ({months})" if months is not None else " (all months)"
        print(f"\n2. Ice Load Statistics{month_info}:")
        print(f"   Range: {min_ice_load:.3f} to {max_ice_load:.3f} kg/m")
        print(f"   Mean: {float(ice_data_analysis.mean()):.3f} kg/m")
        
        # Define ice load bins if not provided
        if ice_load_bins is None:
            if max_ice_load > ice_load_threshold:
                # Create bins for CDF analysis - use more bins for better resolution
                # Always start from 0 for proper CDF calculation, but filter plotting later
                ice_load_bins = np.linspace(0.0, max_ice_load, 100)
            else:
                ice_load_bins = np.array([0.0, ice_load_threshold + 0.01, ice_load_threshold + 0.1, ice_load_threshold + 1, ice_load_threshold + 10])
        else:
            # Ensure bins start from 0 for proper CDF calculation
            if ice_load_bins[0] > 0:
                ice_load_bins = np.concatenate([[0.0], ice_load_bins])
            ice_load_bins = np.sort(ice_load_bins)
        
        print(f"   Using {len(ice_load_bins)} ice load bins")
        print(f"   Ice load threshold: {ice_load_threshold:.3f} kg/m")
        print(f"   Bin range: {ice_load_bins[0]:.4f} to {ice_load_bins[-1]:.3f} kg/m")
        
        # Prepare results storage
        results = {
            'grid_shape': (n_south_north, n_west_east),
            'n_years': n_years,
            'time_step_hours': time_step_hours,
            'ice_load_bins': ice_load_bins,
            'cdf_curves': {},
            'statistics': {}
        }
        
        # Create figure for all grid cells
        n_cols = min(n_west_east, 5)  # Maximum 5 columns
        n_rows = int(np.ceil((n_south_north * n_west_east) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        print(f"\n3. Processing grid cells...")
        
        cell_count = 0
        for i in range(n_south_north):
            for j in range(n_west_east):
                # Extract time series for this grid cell (using filtered data)
                cell_data = ice_data_analysis.isel(south_north=i, west_east=j)
                cell_values = cell_data.values
                
                # Remove NaN values
                valid_mask = ~np.isnan(cell_values)
                cell_values_clean = cell_values[valid_mask]
                
                if len(cell_values_clean) == 0:
                    print(f"   Warning: No valid data for cell ({i},{j})")
                    continue
                
                # Filter values to be >= threshold
                cell_values_filtered = cell_values_clean[cell_values_clean >= ice_load_threshold]
                
                if len(cell_values_filtered) == 0:
                    print(f"   Warning: No data above threshold for cell ({i},{j})")
                    continue
                
                # Calculate CDF
                cdf_values = []
                for ice_threshold in ice_load_bins:
                    # Calculate cumulative probability P(X <= ice_threshold)
                    # Use ALL values including zeros for proper CDF calculation
                    prob = np.sum(cell_values_clean <= ice_threshold) / len(cell_values_clean)
                    cdf_values.append(prob)
                
                cdf_values = np.array(cdf_values)
                
                # Store results for this cell
                cell_key = f'cell_{i}_{j}'
                results['cdf_curves'][cell_key] = {
                    'ice_load_bins': ice_load_bins,
                    'cdf_values': cdf_values,
                    'position': (i, j)
                }
                
                # Calculate statistics for this cell
                results['statistics'][cell_key] = {
                    'max_ice_load': float(np.max(cell_values_clean)),
                    'mean_ice_load': float(np.mean(cell_values_clean)),
                    'std_ice_load': float(np.std(cell_values_clean)),
                    'median_ice_load': float(np.median(cell_values_clean)),
                    'ice_occurrence_percentage': (len(cell_values_filtered) / len(cell_values_clean)) * 100,
                    'percentile_95': float(np.percentile(cell_values_clean, 95)),
                    'percentile_99': float(np.percentile(cell_values_clean, 99))
                }
                
                # Plot CDF curve for this cell in log-log scale
                if cell_count < len(axes):
                    ax = axes[cell_count]
                    # Only plot bins >= threshold for visibility, and avoid log(0) for both axes
                    plot_mask = ice_load_bins > max(ice_load_threshold, 1e-6)  # Ensure positive values for log
                    plot_bins = ice_load_bins[plot_mask]
                    plot_cdf = cdf_values[plot_mask]
                    
                    # Avoid log(0) by adding a small epsilon to probabilities that are exactly 0
                    plot_cdf_safe = np.where(plot_cdf == 0, 1e-10, plot_cdf)
                    
                    ax.loglog(plot_bins, plot_cdf_safe, 'b-', linewidth=2, marker='o', markersize=3)
                    ax.set_xlabel('Ice Load (kg/m, log scale)')
                    ax.set_ylabel('Cumulative Probability (log scale)')
                    ax.set_title(f'Cell ({i},{j})')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(left=max(ice_load_threshold, 1e-6))
                    ax.set_ylim([1e-10, 1])
                
                cell_count += 1
                
                if cell_count % 5 == 0:
                    print(f"   Processed {cell_count}/{n_south_north * n_west_east} cells...")
        
        # Hide unused subplots
        for idx in range(cell_count, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            # Create specific directory structure for ice load per cell CDF plots
            ice_load_plots_dir = os.path.join(figures_dir, "spatial_gradient", "ice_load_per_cell_cdf_log")
            os.makedirs(ice_load_plots_dir, exist_ok=True)
            plot_path = f"{ice_load_plots_dir}/ice_load_cdf_curves_all_cells_log.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\n   CDF curves (log scale) plot saved to: {plot_path}")
        
        # Create summary plot with mean curve in log scale
        print(f"\n4. Creating summary statistics...")
        
        # Calculate mean CDF curve across all cells
        all_cdf_curves = []
        for cell_key, cell_data in results['cdf_curves'].items():
            all_cdf_curves.append(cell_data['cdf_values'])
        
        if all_cdf_curves:
            mean_cdf = np.mean(all_cdf_curves, axis=0)
            std_cdf = np.std(all_cdf_curves, axis=0)
            
            plt.figure(figsize=(10, 6))
            # Only plot bins > threshold for visibility, and avoid log(0) for ice loads
            plot_mask = ice_load_bins > max(ice_load_threshold, 1e-6)  # Ensure positive values for log
            plot_bins = ice_load_bins[plot_mask]
            plot_mean_cdf = mean_cdf[plot_mask]
            plot_std_cdf = std_cdf[plot_mask]
            
            # Calculate confidence bands
            plot_std_cdf_upper = plot_mean_cdf + plot_std_cdf
            plot_std_cdf_lower = np.maximum(plot_mean_cdf - plot_std_cdf, 0)  # Ensure non-negative
            
            # Use safe log probabilities for plotting
            plot_mean_cdf_safe = np.where(plot_mean_cdf == 0, 1e-10, plot_mean_cdf)
            plot_std_cdf_upper_safe = np.where(plot_std_cdf_upper == 0, 1e-10, plot_std_cdf_upper)
            plot_std_cdf_lower_safe = np.where(plot_std_cdf_lower == 0, 1e-10, plot_std_cdf_lower)
            
            plt.loglog(plot_bins, plot_mean_cdf_safe, 'r-', linewidth=3, label='Mean across all cells')
            plt.fill_between(plot_bins, plot_std_cdf_lower_safe, plot_std_cdf_upper_safe, 
                           alpha=0.3, color='red', label='±1 Standard Deviation')
            
            plt.xlabel('Ice Load (kg/m, log scale)')
            plt.ylabel('Cumulative Probability (log scale)')
            plt.title('Ice Load CDF - Domain Average (Log-Log Scale)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(left=max(ice_load_threshold, 1e-6))
            plt.ylim([1e-10, 1])
            
            plt.tight_layout()
            
            if save_plots:
                summary_path = f"{ice_load_plots_dir}/ice_load_cdf_summary_log.png"
                plt.savefig(summary_path, dpi=300, bbox_inches='tight')
                print(f"   Summary plot (log scale) saved to: {summary_path}")
            
            # Store summary statistics
            results['domain_statistics'] = {
                'mean_cdf_curve': mean_cdf,
                'std_cdf_curve': std_cdf,
                'ice_load_bins': ice_load_bins
            }
        
        # Print summary statistics
        print(f"\n5. Summary Statistics:")
        print(f"   Processed {len(results['cdf_curves'])} grid cells")
        
        if results['statistics']:
            all_stats = list(results['statistics'].values())
            mean_ice_occurrence = np.mean([s['ice_occurrence_percentage'] for s in all_stats])
            max_ice_occurrence = np.max([s['ice_occurrence_percentage'] for s in all_stats])
            min_ice_occurrence = np.min([s['ice_occurrence_percentage'] for s in all_stats])
            mean_p95 = np.mean([s['percentile_95'] for s in all_stats])
            mean_p99 = np.mean([s['percentile_99'] for s in all_stats])
            
            print(f"   Ice occurrence across domain:")
            print(f"     Mean: {mean_ice_occurrence:.1f}% of time")
            print(f"     Range: {min_ice_occurrence:.1f}% to {max_ice_occurrence:.1f}%")
            print(f"   Domain average percentiles:")
            print(f"     95th percentile: {mean_p95:.3f} kg/m")
            print(f"     99th percentile: {mean_p99:.3f} kg/m")
        
        # Create spatial gradient analysis using Earth Mover's Distance with log-log scale binning
        print(f"\n6. Spatial Gradient Analysis (Earth Mover's Distance with Log-Log Scale Binning)...")
        
        try:
            from scipy.stats import wasserstein_distance
            
            # Initialize gradient matrices
            n_south_north, n_west_east = ice_load_data.shape[1], ice_load_data.shape[2]
            
            # East-West gradients (comparing adjacent cells horizontally)
            ew_gradients = np.full((n_south_north, n_west_east-1), np.nan)
            
            # South-North gradients (comparing adjacent cells vertically)
            sn_gradients = np.full((n_south_north-1, n_west_east), np.nan)
            
            # Combined gradients (for each cell, average of all neighbor comparisons)
            combined_gradients = np.full((n_south_north, n_west_east), np.nan)
            
            print(f"   Computing Earth Mover's distances between neighboring cells (log-log scale)...")
            
            # Calculate East-West gradients
            for i in range(n_south_north):
                for j in range(n_west_east-1):
                    cell1_key = f'cell_{i}_{j}'
                    cell2_key = f'cell_{i}_{j+1}'
                    
                    if cell1_key in results['cdf_curves'] and cell2_key in results['cdf_curves']:
                        cdf1 = results['cdf_curves'][cell1_key]['cdf_values']
                        cdf2 = results['cdf_curves'][cell2_key]['cdf_values']
                        bins1 = results['cdf_curves'][cell1_key]['ice_load_bins']
                        bins2 = results['cdf_curves'][cell2_key]['ice_load_bins']
                        
                        # Transform both bins and CDF values to log scale for distance calculation
                        # Only use bins > 0 to avoid log(0), and CDF values > 0 for log scale
                        mask = (bins1 > 1e-6) & (cdf1 > 1e-10) & (cdf2 > 1e-10)
                        if np.sum(mask) > 1:
                            log_bins = np.log10(bins1[mask])
                            log_cdf1 = np.log10(np.maximum(cdf1[mask], 1e-10))
                            log_cdf2 = np.log10(np.maximum(cdf2[mask], 1e-10))
                            
                            # Earth Mover's distance with log-log scale
                            try:
                                distance = np.trapz(np.abs(log_cdf1 - log_cdf2), log_bins)
                                ew_gradients[i, j] = distance
                            except:
                                # Fallback to simple mean absolute difference
                                distance = np.mean(np.abs(log_cdf1 - log_cdf2))
                                ew_gradients[i, j] = distance
            
            # Calculate South-North gradients
            for i in range(n_south_north-1):
                for j in range(n_west_east):
                    cell1_key = f'cell_{i}_{j}'
                    cell2_key = f'cell_{i+1}_{j}'
                    
                    if cell1_key in results['cdf_curves'] and cell2_key in results['cdf_curves']:
                        cdf1 = results['cdf_curves'][cell1_key]['cdf_values']
                        cdf2 = results['cdf_curves'][cell2_key]['cdf_values']
                        bins1 = results['cdf_curves'][cell1_key]['ice_load_bins']
                        bins2 = results['cdf_curves'][cell2_key]['ice_load_bins']
                        
                        # Transform both bins and CDF values to log scale for distance calculation
                        # Only use bins > 0 to avoid log(0), and CDF values > 0 for log scale
                        mask = (bins1 > 1e-6) & (cdf1 > 1e-10) & (cdf2 > 1e-10)
                        if np.sum(mask) > 1:
                            log_bins = np.log10(bins1[mask])
                            log_cdf1 = np.log10(np.maximum(cdf1[mask], 1e-10))
                            log_cdf2 = np.log10(np.maximum(cdf2[mask], 1e-10))
                            
                            try:
                                distance = np.trapz(np.abs(log_cdf1 - log_cdf2), log_bins)
                                sn_gradients[i, j] = distance
                            except:
                                distance = np.mean(np.abs(log_cdf1 - log_cdf2))
                                sn_gradients[i, j] = distance
            
            # Calculate combined gradients (average of all valid neighbor distances)
            for i in range(n_south_north):
                for j in range(n_west_east):
                    distances = []
                    
                    # Check East neighbor
                    if j < n_west_east-1 and not np.isnan(ew_gradients[i, j]):
                        distances.append(ew_gradients[i, j])
                    
                    # Check West neighbor
                    if j > 0 and not np.isnan(ew_gradients[i, j-1]):
                        distances.append(ew_gradients[i, j-1])
                    
                    # Check North neighbor
                    if i < n_south_north-1 and not np.isnan(sn_gradients[i, j]):
                        distances.append(sn_gradients[i, j])
                    
                    # Check South neighbor
                    if i > 0 and not np.isnan(sn_gradients[i-1, j]):
                        distances.append(sn_gradients[i-1, j])
                    
                    if distances:
                        combined_gradients[i, j] = np.mean(distances)
            
            # Create the spatial gradient plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: East-West gradients
            im1 = axes[0, 0].imshow(ew_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[0, 0].set_title('East-West Gradient\n(CDF L1 Distance, Log X-axis)')
            axes[0, 0].set_xlabel('West-East Grid Points')
            axes[0, 0].set_ylabel('South-North Grid Points')
            cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
            cbar1.set_label('L1 Distance (Log X-axis)')
            
            # Add grid lines
            axes[0, 0].set_xticks(range(n_west_east-1))
            axes[0, 0].set_yticks(range(n_south_north))
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: South-North gradients
            im2 = axes[0, 1].imshow(sn_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[0, 1].set_title('South-North Gradient\n(CDF L1 Distance, Log X-axis)')
            axes[0, 1].set_xlabel('West-East Grid Points')
            axes[0, 1].set_ylabel('South-North Grid Points')
            cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
            cbar2.set_label('L1 Distance (Log X-axis)')
            
            axes[0, 1].set_xticks(range(n_west_east))
            axes[0, 1].set_yticks(range(n_south_north-1))
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Combined gradients
            im3 = axes[1, 0].imshow(combined_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 0].set_title('Combined Spatial Gradient\n(Average Neighbor Distance)')
            axes[1, 0].set_xlabel('West-East Grid Points')
            axes[1, 0].set_ylabel('South-North Grid Points')
            cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
            cbar3.set_label('Average L1 Distance (Log X-axis)')
            
            axes[1, 0].set_xticks(range(n_west_east))
            axes[1, 0].set_yticks(range(n_south_north))
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Gradient magnitude (combining EW and SN)
            # Create a full-size gradient magnitude matrix
            gradient_magnitude = np.full((n_south_north, n_west_east), np.nan)
            
            for i in range(n_south_north):
                for j in range(n_west_east):
                    magnitudes = []
                    
                    # East-West component
                    if j < n_west_east-1 and not np.isnan(ew_gradients[i, j]):
                        magnitudes.append(ew_gradients[i, j]**2)
                    if j > 0 and not np.isnan(ew_gradients[i, j-1]):
                        magnitudes.append(ew_gradients[i, j-1]**2)
                    
                    # South-North component  
                    if i < n_south_north-1 and not np.isnan(sn_gradients[i, j]):
                        magnitudes.append(sn_gradients[i, j]**2)
                    if i > 0 and not np.isnan(sn_gradients[i-1, j]):
                        magnitudes.append(sn_gradients[i-1, j]**2)
                    
                    if magnitudes:
                        gradient_magnitude[i, j] = np.sqrt(np.mean(magnitudes))
            
            im4 = axes[1, 1].imshow(gradient_magnitude, cmap='plasma', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 1].set_title('Gradient Magnitude\n(RMS of EW and SN)')
            axes[1, 1].set_xlabel('West-East Grid Points')
            axes[1, 1].set_ylabel('South-North Grid Points')
            cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
            cbar4.set_label('RMS Gradient Magnitude')
            
            axes[1, 1].set_xticks(range(n_west_east))
            axes[1, 1].set_yticks(range(n_south_north))
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save spatial gradient plots
            if save_plots:
                gradient_path = f"{ice_load_plots_dir}/ice_load_cdf_spatial_gradients_log.png"
                plt.savefig(gradient_path, dpi=300, bbox_inches='tight')
                print(f"   Spatial gradient plots (log scale) saved to: {gradient_path}")
            
            # Store gradient results
            results['spatial_gradients'] = {
                'east_west_gradients': ew_gradients,
                'south_north_gradients': sn_gradients,
                'combined_gradients': combined_gradients,
                'gradient_magnitude': gradient_magnitude,
                'ew_mean': np.nanmean(ew_gradients),
                'ew_std': np.nanstd(ew_gradients),
                'sn_mean': np.nanmean(sn_gradients),
                'sn_std': np.nanstd(sn_gradients),
                'combined_mean': np.nanmean(combined_gradients),
                'combined_std': np.nanstd(combined_gradients)
            }
            
            # Print gradient statistics
            print(f"   Gradient Statistics (Log X-axis Scale):")
            print(f"     East-West: Mean = {results['spatial_gradients']['ew_mean']:.3f}, Std = {results['spatial_gradients']['ew_std']:.3f}")
            print(f"     South-North: Mean = {results['spatial_gradients']['sn_mean']:.3f}, Std = {results['spatial_gradients']['sn_std']:.3f}")
            print(f"     Combined: Mean = {results['spatial_gradients']['combined_mean']:.3f}, Std = {results['spatial_gradients']['combined_std']:.3f}")
            
        except ImportError:
            print("   Warning: scipy not available, skipping Earth Mover's distance calculation")
            print("   Install scipy with: conda install scipy")
        except Exception as e:
            print(f"   Error in spatial gradient analysis: {e}")
        
        print(f"\n✓ Ice load CDF analysis (log scale) completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in ice load CDF analysis (log scale): {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_ice_load_1_minus_cdf_curves(ice_load_data, save_plots=True, ice_load_bins=None, ice_load_threshold=0.0, months=None):
    """
    Create exceedance probability curves (1 - CDF) showing the probability of exceeding each ice load level
    for every grid cell. The function creates a mean over all years and plots
    X-axis: Ice load (kg/m)
    Y-axis: Exceedance probability (P(X > threshold))
    
    Parameters:
    -----------
    ice_load_data : xarray.DataArray
        Ice load data with dimensions (time, south_north, west_east)
    save_plots : bool, default True
        Whether to save the plots to files
    ice_load_bins : array-like, optional
        Custom ice load bins for analysis. If None, uses automatic binning
    ice_load_threshold : float, default 0.0
        Minimum ice load value to be plotted (kg/m). Values below this threshold will be excluded
    months : list of int, optional
        List of months to include in analysis (1-12). If None, uses all months.
        Example: [12, 1, 2, 3, 4] for winter months (Dec-Apr)
        
    Returns:
    --------
    dict : Dictionary containing analysis results and statistics
    """
    print("=== ICE LOAD EXCEEDANCE PROBABILITY CURVE ANALYSIS ===")
    
    try:
        # Check data structure
        print(f"\n1. Data Information:")
        print(f"   Shape: {ice_load_data.shape}")
        print(f"   Dimensions: {ice_load_data.dims}")
        print(f"   Time range: {ice_load_data.time.min().values} to {ice_load_data.time.max().values}")
        
        # Get spatial dimensions
        n_south_north = ice_load_data.sizes['south_north']
        n_west_east = ice_load_data.sizes['west_east']
        n_time = ice_load_data.sizes['time']
        
        print(f"   Grid size: {n_south_north} × {n_west_east} = {n_south_north * n_west_east} cells")
        print(f"   Time steps: {n_time}")
        
        # Convert time to pandas for easier manipulation
        time_index = pd.to_datetime(ice_load_data.time.values)
        n_years = len(time_index.year.unique())
        print(f"   Years covered: {n_years}")
        print(f"   Years: {sorted(time_index.year.unique())}")
        
        # Calculate time step in hours (assuming regular intervals)
        if len(time_index) > 1:
            time_step_hours = (time_index[1] - time_index[0]).total_seconds() / 3600
        else:
            time_step_hours = 0.5  # Default to 30 minutes
        print(f"   Time step: {time_step_hours} hours")
        
        # Remove NaN values and get overall data statistics
        ice_data_clean = ice_load_data.where(ice_load_data >= 0, 0)  # Replace negative/NaN with 0
        
        # Filter data by months if specified
        if months is not None:
            print(f"\n   Filtering data to specified months only: {months}...")
            time_index_full = pd.to_datetime(ice_data_clean.time.values)
            month_mask = time_index_full.month.isin(months)
            ice_data_filtered = ice_data_clean.isel(time=month_mask)
            
            # Update time information after filtering
            time_index_filtered = pd.to_datetime(ice_data_filtered.time.values)
            n_filtered_timesteps = len(time_index_filtered)
            
            print(f"   Original timesteps: {n_time}")
            print(f"   Filtered timesteps: {n_filtered_timesteps}")
            print(f"   Months included: {sorted(time_index_filtered.month.unique())}")
            print(f"   Reduction: {((n_time - n_filtered_timesteps) / n_time * 100):.1f}% timesteps removed")
            
            # Use filtered data for analysis
            ice_data_analysis = ice_data_filtered
        else:
            print(f"\n   Using all months for analysis...")
            ice_data_analysis = ice_data_clean
        
        max_ice_load = float(ice_data_analysis.max())
        min_ice_load = 0.0
        
        month_info = f" ({months})" if months is not None else " (all months)"
        print(f"\n2. Ice Load Statistics{month_info}:")
        print(f"   Range: {min_ice_load:.3f} to {max_ice_load:.3f} kg/m")
        print(f"   Mean: {float(ice_data_analysis.mean()):.3f} kg/m")
        
        # Define ice load bins if not provided
        if ice_load_bins is None:
            if max_ice_load > ice_load_threshold:
                # Create bins for exceedance probability analysis - use more bins for better resolution
                # Always start from 0 for proper calculation, but filter plotting later
                ice_load_bins = np.linspace(0.0, max_ice_load, 100)
            else:
                ice_load_bins = np.array([0.0, ice_load_threshold + 0.01, ice_load_threshold + 0.1, ice_load_threshold + 1, ice_load_threshold + 10])
        else:
            # Ensure bins start from 0 for proper calculation
            if ice_load_bins[0] > 0:
                ice_load_bins = np.concatenate([[0.0], ice_load_bins])
            ice_load_bins = np.sort(ice_load_bins)
        
        print(f"   Using {len(ice_load_bins)} ice load bins")
        print(f"   Ice load threshold: {ice_load_threshold:.3f} kg/m")
        print(f"   Bin range: {ice_load_bins[0]:.4f} to {ice_load_bins[-1]:.3f} kg/m")
        
        # Prepare results storage
        results = {
            'grid_shape': (n_south_north, n_west_east),
            'n_years': n_years,
            'time_step_hours': time_step_hours,
            'ice_load_bins': ice_load_bins,
            'exceedance_curves': {},
            'statistics': {}
        }
        
        # Create figure for all grid cells
        n_cols = min(n_west_east, 5)  # Maximum 5 columns
        n_rows = int(np.ceil((n_south_north * n_west_east) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        print(f"\n3. Processing grid cells...")
        
        cell_count = 0
        for i in range(n_south_north):
            for j in range(n_west_east):
                # Extract time series for this grid cell (using filtered data)
                cell_data = ice_data_analysis.isel(south_north=i, west_east=j)
                cell_values = cell_data.values
                
                # Remove NaN values
                valid_mask = ~np.isnan(cell_values)
                cell_values_clean = cell_values[valid_mask]
                
                if len(cell_values_clean) == 0:
                    print(f"   Warning: No valid data for cell ({i},{j})")
                    continue
                
                # Filter values to be >= threshold
                cell_values_filtered = cell_values_clean[cell_values_clean >= ice_load_threshold]
                
                if len(cell_values_filtered) == 0:
                    print(f"   Warning: No data above threshold for cell ({i},{j})")
                    continue
                
                # Calculate exceedance probability (1 - CDF)
                exceedance_values = []
                for ice_threshold in ice_load_bins:
                    # Calculate exceedance probability P(X > ice_threshold) = 1 - P(X <= ice_threshold)
                    # Use ALL values including zeros for proper calculation
                    cdf_prob = np.sum(cell_values_clean <= ice_threshold) / len(cell_values_clean)
                    exceedance_prob = 1.0 - cdf_prob
                    exceedance_values.append(exceedance_prob)
                
                exceedance_values = np.array(exceedance_values)
                
                # Store results for this cell
                cell_key = f'cell_{i}_{j}'
                results['exceedance_curves'][cell_key] = {
                    'ice_load_bins': ice_load_bins,
                    'exceedance_values': exceedance_values,
                    'position': (i, j)
                }
                
                # Calculate statistics for this cell
                results['statistics'][cell_key] = {
                    'max_ice_load': float(np.max(cell_values_clean)),
                    'mean_ice_load': float(np.mean(cell_values_clean)),
                    'std_ice_load': float(np.std(cell_values_clean)),
                    'median_ice_load': float(np.median(cell_values_clean)),
                    'ice_occurrence_percentage': (len(cell_values_filtered) / len(cell_values_clean)) * 100,
                    'percentile_95': float(np.percentile(cell_values_clean, 95)),
                    'percentile_99': float(np.percentile(cell_values_clean, 99))
                }
                
                # Plot exceedance curve for this cell
                if cell_count < len(axes):
                    ax = axes[cell_count]
                    # Only plot bins >= threshold for visibility
                    plot_mask = ice_load_bins >= ice_load_threshold
                    plot_bins = ice_load_bins[plot_mask]
                    plot_exceedance = exceedance_values[plot_mask]
                    ax.plot(plot_bins, plot_exceedance, 'b-', linewidth=2, marker='o', markersize=3)
                    ax.set_xlabel('Ice Load (kg/m)')
                    ax.set_ylabel('Exceedance Probability')
                    ax.set_title(f'Cell ({i},{j})')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(left=ice_load_threshold)
                    ax.set_ylim([0, 1])
                
                cell_count += 1
                
                if cell_count % 5 == 0:
                    print(f"   Processed {cell_count}/{n_south_north * n_west_east} cells...")
        
        # Hide unused subplots
        for idx in range(cell_count, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            # Create specific directory structure for ice load per cell exceedance plots
            ice_load_plots_dir = os.path.join(figures_dir, "spatial_gradient", "ice_load_per_cell_exceedance")
            os.makedirs(ice_load_plots_dir, exist_ok=True)
            plot_path = f"{ice_load_plots_dir}/ice_load_exceedance_curves_all_cells.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\n   Exceedance curves plot saved to: {plot_path}")
        
        # Create summary plot with mean curve
        print(f"\n4. Creating summary statistics...")
        
        # Calculate mean exceedance curve across all cells
        all_exceedance_curves = []
        for cell_key, cell_data in results['exceedance_curves'].items():
            all_exceedance_curves.append(cell_data['exceedance_values'])
        
        if all_exceedance_curves:
            mean_exceedance = np.mean(all_exceedance_curves, axis=0)
            std_exceedance = np.std(all_exceedance_curves, axis=0)
            
            plt.figure(figsize=(10, 6))
            # Only plot bins >= threshold for visibility
            plot_mask = ice_load_bins >= ice_load_threshold
            plot_bins = ice_load_bins[plot_mask]
            plot_mean_exceedance = mean_exceedance[plot_mask]
            plot_std_exceedance = std_exceedance[plot_mask]
            
            plt.plot(plot_bins, plot_mean_exceedance, 'r-', linewidth=3, label='Mean across all cells')
            plt.fill_between(plot_bins, plot_mean_exceedance - plot_std_exceedance, 
                           plot_mean_exceedance + plot_std_exceedance, alpha=0.3, color='red', 
                           label='±1 Standard Deviation')
            
            plt.xlabel('Ice Load (kg/m)')
            plt.ylabel('Exceedance Probability')
            plt.title('Ice Load Exceedance Probability - Domain Average')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(left=ice_load_threshold)
            plt.ylim([0, 1])
            
            plt.tight_layout()
            
            if save_plots:
                summary_path = f"{ice_load_plots_dir}/ice_load_exceedance_summary.png"
                plt.savefig(summary_path, dpi=300, bbox_inches='tight')
                print(f"   Summary plot saved to: {summary_path}")
            
            # Store summary statistics
            results['domain_statistics'] = {
                'mean_exceedance_curve': mean_exceedance,
                'std_exceedance_curve': std_exceedance,
                'ice_load_bins': ice_load_bins
            }
        
        # Print summary statistics
        print(f"\n5. Summary Statistics:")
        print(f"   Processed {len(results['exceedance_curves'])} grid cells")
        
        if results['statistics']:
            all_stats = list(results['statistics'].values())
            mean_ice_occurrence = np.mean([s['ice_occurrence_percentage'] for s in all_stats])
            max_ice_occurrence = np.max([s['ice_occurrence_percentage'] for s in all_stats])
            min_ice_occurrence = np.min([s['ice_occurrence_percentage'] for s in all_stats])
            mean_p95 = np.mean([s['percentile_95'] for s in all_stats])
            mean_p99 = np.mean([s['percentile_99'] for s in all_stats])
            
            print(f"   Ice occurrence across domain:")
            print(f"     Mean: {mean_ice_occurrence:.1f}% of time")
            print(f"     Range: {min_ice_occurrence:.1f}% to {max_ice_occurrence:.1f}%")
            print(f"   Domain average percentiles:")
            print(f"     95th percentile: {mean_p95:.3f} kg/m")
            print(f"     99th percentile: {mean_p99:.3f} kg/m")
        
        # Create spatial gradient analysis using Earth Mover's Distance
        print(f"\n6. Spatial Gradient Analysis (Earth Mover's Distance)...")
        
        try:
            from scipy.stats import wasserstein_distance
            
            # Initialize gradient matrices
            n_south_north, n_west_east = ice_load_data.shape[1], ice_load_data.shape[2]
            
            # East-West gradients (comparing adjacent cells horizontally)
            ew_gradients = np.full((n_south_north, n_west_east-1), np.nan)
            
            # South-North gradients (comparing adjacent cells vertically)
            sn_gradients = np.full((n_south_north-1, n_west_east), np.nan)
            
            # Combined gradients (for each cell, average of all neighbor comparisons)
            combined_gradients = np.full((n_south_north, n_west_east), np.nan)
            
            print(f"   Computing Earth Mover's distances between neighboring cells...")
            
            # Calculate East-West gradients
            for i in range(n_south_north):
                for j in range(n_west_east-1):
                    cell1_key = f'cell_{i}_{j}'
                    cell2_key = f'cell_{i}_{j+1}'
                    
                    if cell1_key in results['exceedance_curves'] and cell2_key in results['exceedance_curves']:
                        exc1 = results['exceedance_curves'][cell1_key]['exceedance_values']
                        exc2 = results['exceedance_curves'][cell2_key]['exceedance_values']
                        bins1 = results['exceedance_curves'][cell1_key]['ice_load_bins']
                        bins2 = results['exceedance_curves'][cell2_key]['ice_load_bins']
                        
                        # For exceedance curves, Earth Mover's distance can be calculated using the L1 distance
                        # between exceedance curves, which is mathematically equivalent
                        try:
                            distance = np.trapz(np.abs(exc1 - exc2), bins1)
                            ew_gradients[i, j] = distance
                        except:
                            # Fallback to simple mean absolute difference
                            distance = np.mean(np.abs(exc1 - exc2))
                            ew_gradients[i, j] = distance
            
            # Calculate South-North gradients
            for i in range(n_south_north-1):
                for j in range(n_west_east):
                    cell1_key = f'cell_{i}_{j}'
                    cell2_key = f'cell_{i+1}_{j}'
                    
                    if cell1_key in results['exceedance_curves'] and cell2_key in results['exceedance_curves']:
                        exc1 = results['exceedance_curves'][cell1_key]['exceedance_values']
                        exc2 = results['exceedance_curves'][cell2_key]['exceedance_values']
                        bins1 = results['exceedance_curves'][cell1_key]['ice_load_bins']
                        bins2 = results['exceedance_curves'][cell2_key]['ice_load_bins']
                        
                        try:
                            distance = np.trapz(np.abs(exc1 - exc2), bins1)
                            sn_gradients[i, j] = distance
                        except:
                            distance = np.mean(np.abs(exc1 - exc2))
                            sn_gradients[i, j] = distance
            
            # Calculate combined gradients (average of all valid neighbor distances)
            for i in range(n_south_north):
                for j in range(n_west_east):
                    distances = []
                    
                    # Check East neighbor
                    if j < n_west_east-1 and not np.isnan(ew_gradients[i, j]):
                        distances.append(ew_gradients[i, j])
                    
                    # Check West neighbor
                    if j > 0 and not np.isnan(ew_gradients[i, j-1]):
                        distances.append(ew_gradients[i, j-1])
                    
                    # Check North neighbor
                    if i < n_south_north-1 and not np.isnan(sn_gradients[i, j]):
                        distances.append(sn_gradients[i, j])
                    
                    # Check South neighbor
                    if i > 0 and not np.isnan(sn_gradients[i-1, j]):
                        distances.append(sn_gradients[i-1, j])
                    
                    if distances:
                        combined_gradients[i, j] = np.mean(distances)
            
            # Create the spatial gradient plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: East-West gradients
            im1 = axes[0, 0].imshow(ew_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[0, 0].set_title('East-West Gradient\n(Exceedance L1 Distance)')
            axes[0, 0].set_xlabel('West-East Grid Points')
            axes[0, 0].set_ylabel('South-North Grid Points')
            cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
            cbar1.set_label('L1 Distance')
            
            # Add grid lines
            axes[0, 0].set_xticks(range(n_west_east-1))
            axes[0, 0].set_yticks(range(n_south_north))
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: South-North gradients
            im2 = axes[0, 1].imshow(sn_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[0, 1].set_title('South-North Gradient\n(Exceedance L1 Distance)')
            axes[0, 1].set_xlabel('West-East Grid Points')
            axes[0, 1].set_ylabel('South-North Grid Points')
            cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
            cbar2.set_label('L1 Distance')
            
            axes[0, 1].set_xticks(range(n_west_east))
            axes[0, 1].set_yticks(range(n_south_north-1))
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Combined gradients
            im3 = axes[1, 0].imshow(combined_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 0].set_title('Combined Spatial Gradient\n(Average Neighbor Distance)')
            axes[1, 0].set_xlabel('West-East Grid Points')
            axes[1, 0].set_ylabel('South-North Grid Points')
            cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
            cbar3.set_label('Average L1 Distance')
            
            axes[1, 0].set_xticks(range(n_west_east))
            axes[1, 0].set_yticks(range(n_south_north))
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Gradient magnitude (combining EW and SN)
            # Create a full-size gradient magnitude matrix
            gradient_magnitude = np.full((n_south_north, n_west_east), np.nan)
            
            for i in range(n_south_north):
                for j in range(n_west_east):
                    magnitudes = []
                    
                    # East-West component
                    if j < n_west_east-1 and not np.isnan(ew_gradients[i, j]):
                        magnitudes.append(ew_gradients[i, j]**2)
                    if j > 0 and not np.isnan(ew_gradients[i, j-1]):
                        magnitudes.append(ew_gradients[i, j-1]**2)
                    
                    # South-North component  
                    if i < n_south_north-1 and not np.isnan(sn_gradients[i, j]):
                        magnitudes.append(sn_gradients[i, j]**2)
                    if i > 0 and not np.isnan(sn_gradients[i-1, j]):
                        magnitudes.append(sn_gradients[i-1, j]**2)
                    
                    if magnitudes:
                        gradient_magnitude[i, j] = np.sqrt(np.mean(magnitudes))
            
            im4 = axes[1, 1].imshow(gradient_magnitude, cmap='plasma', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 1].set_title('Gradient Magnitude\n(RMS of EW and SN)')
            axes[1, 1].set_xlabel('West-East Grid Points')
            axes[1, 1].set_ylabel('South-North Grid Points')
            cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
            cbar4.set_label('RMS Gradient Magnitude')
            
            axes[1, 1].set_xticks(range(n_west_east))
            axes[1, 1].set_yticks(range(n_south_north))
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save spatial gradient plots
            if save_plots:
                gradient_path = f"{ice_load_plots_dir}/ice_load_exceedance_spatial_gradients.png"
                plt.savefig(gradient_path, dpi=300, bbox_inches='tight')
                print(f"   Spatial gradient plots saved to: {gradient_path}")
            
            # Store gradient results
            results['spatial_gradients'] = {
                'east_west_gradients': ew_gradients,
                'south_north_gradients': sn_gradients,
                'combined_gradients': combined_gradients,
                'gradient_magnitude': gradient_magnitude,
                'ew_mean': np.nanmean(ew_gradients),
                'ew_std': np.nanstd(ew_gradients),
                'sn_mean': np.nanmean(sn_gradients),
                'sn_std': np.nanstd(sn_gradients),
                'combined_mean': np.nanmean(combined_gradients),
                'combined_std': np.nanstd(combined_gradients)
            }
            
            # Print gradient statistics
            print(f"   Gradient Statistics:")
            print(f"     East-West: Mean = {results['spatial_gradients']['ew_mean']:.3f}, Std = {results['spatial_gradients']['ew_std']:.3f}")
            print(f"     South-North: Mean = {results['spatial_gradients']['sn_mean']:.3f}, Std = {results['spatial_gradients']['sn_std']:.3f}")
            print(f"     Combined: Mean = {results['spatial_gradients']['combined_mean']:.3f}, Std = {results['spatial_gradients']['combined_std']:.3f}")
            
        except ImportError:
            print("   Warning: scipy not available, skipping Earth Mover's distance calculation")
            print("   Install scipy with: conda install scipy")
        except Exception as e:
            print(f"   Error in spatial gradient analysis: {e}")
        
        print(f"\n✓ Ice load exceedance probability analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in ice load exceedance probability analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_ice_load_threshold_exceedance_map(ice_load_data, ice_load_threshold, save_plots=True, 
                                         colormap='viridis', grid_labels=True, units='hours'):
    """
    Create a spatial map showing how often each grid cell exceeds a specified ice load threshold
    per year on average. Uses a colorbar to show spatial differences in threshold exceedance.
    
    Parameters:
    -----------
    ice_load_data : xarray.DataArray
        Ice load data with dimensions (time, south_north, west_east)
    ice_load_threshold : float
        Ice load threshold value (kg/m) to analyze exceedance for
    save_plots : bool, default True
        Whether to save the plot to file
    colormap : str, default 'viridis'
        Matplotlib colormap to use for the spatial plot
    grid_labels : bool, default True
        Whether to add grid cell coordinate labels to the plot
    units : str, default 'hours'
        Units for the exceedance frequency ('hours', 'days', or 'percentage')
        
    Returns:
    --------
    dict : Dictionary containing exceedance analysis results and statistics
    """
    print(f"=== ICE LOAD THRESHOLD EXCEEDANCE ANALYSIS ===")
    print(f"Threshold: {ice_load_threshold:.3f} kg/m")
    
    try:
        # Check data structure
        print(f"\n1. Data Information:")
        print(f"   Shape: {ice_load_data.shape}")
        print(f"   Dimensions: {ice_load_data.dims}")
        
        # Get spatial dimensions
        n_time = ice_load_data.sizes['time']
        n_south_north = ice_load_data.sizes['south_north']
        n_west_east = ice_load_data.sizes['west_east']
        
        print(f"   Grid size: {n_south_north} × {n_west_east} = {n_south_north * n_west_east} cells")
        print(f"   Time steps: {n_time}")
        
        # Calculate temporal information
        time_index = pd.to_datetime(ice_load_data.time.values)
        n_years = len(time_index.year.unique())
        years = sorted(time_index.year.unique())
        
        # Calculate time step in hours (assuming regular intervals)
        if len(time_index) > 1:
            time_step_hours = (time_index[1] - time_index[0]).total_seconds() / 3600
        else:
            time_step_hours = 0.5  # Default to 30 minutes
            
        print(f"   Years covered: {n_years} ({years[0]} to {years[-1]})")
        print(f"   Time step: {time_step_hours} hours")
        
        # Clean the data (remove NaN values, replace negative with 0)
        ice_data_clean = ice_load_data.where(ice_load_data >= 0, 0)
        
        print(f"\n2. Threshold Exceedance Analysis:")
        print(f"   Analyzing exceedance of {ice_load_threshold:.3f} kg/m threshold...")
        
        # Initialize exceedance matrix
        exceedance_matrix = np.zeros((n_south_north, n_west_east))
        
        # Calculate exceedance for each grid cell
        total_cells = n_south_north * n_west_east
        processed_cells = 0
        
        for i in range(n_south_north):
            for j in range(n_west_east):
                # Extract time series for this grid cell
                cell_data = ice_data_clean.isel(south_north=i, west_east=j)
                cell_values = cell_data.values
                
                # Remove NaN values
                valid_mask = ~np.isnan(cell_values)
                cell_values_clean = cell_values[valid_mask]
                
                if len(cell_values_clean) > 0:
                    # Count exceedances
                    exceedances = np.sum(cell_values_clean >= ice_load_threshold)
                    
                    # Convert to the requested units
                    if units == 'hours':
                        # Hours per year
                        exceedance_value = (exceedances * time_step_hours) / n_years
                    elif units == 'days':
                        # Days per year
                        exceedance_value = (exceedances * time_step_hours) / (24 * n_years)
                    elif units == 'percentage':
                        # Percentage of time
                        total_hours_per_year = 365.25 * 24  # Account for leap years
                        hours_per_year = (exceedances * time_step_hours) / n_years
                        exceedance_value = (hours_per_year / total_hours_per_year) * 100
                    else:
                        # Default to hours
                        exceedance_value = (exceedances * time_step_hours) / n_years
                    
                    exceedance_matrix[i, j] = exceedance_value
                else:
                    exceedance_matrix[i, j] = np.nan
                
                processed_cells += 1
                if processed_cells % 20 == 0:
                    print(f"   Processed {processed_cells}/{total_cells} cells...")
        
        # Calculate statistics
        valid_exceedances = exceedance_matrix[~np.isnan(exceedance_matrix)]
        
        print(f"\n3. Exceedance Statistics:")
        if len(valid_exceedances) > 0:
            print(f"   Mean exceedance: {np.mean(valid_exceedances):.2f} {units}/year")
            print(f"   Std exceedance: {np.std(valid_exceedances):.2f} {units}/year")
            print(f"   Min exceedance: {np.min(valid_exceedances):.2f} {units}/year")
            print(f"   Max exceedance: {np.max(valid_exceedances):.2f} {units}/year")
            print(f"   Cells with exceedances: {np.sum(valid_exceedances > 0)}/{len(valid_exceedances)}")
        else:
            print(f"   No valid exceedance data found")
        
        # Create the spatial plot
        print(f"\n4. Creating spatial exceedance map...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Create the main plot
        im = ax.imshow(exceedance_matrix, cmap=colormap, origin='lower', 
                      interpolation='nearest', aspect='auto')
        
        # Set title and labels
        unit_label = units.capitalize()
        if units == 'percentage':
            unit_label = '% of Time'
        
        ax.set_title(f'Ice Load Threshold Exceedance Map\n'
                    f'Threshold: {ice_load_threshold:.3f} kg/m\n'
                    f'Mean Annual Exceedance ({unit_label})')
        ax.set_xlabel('West-East Grid Points')
        ax.set_ylabel('South-North Grid Points')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label(f'Exceedance ({unit_label}/Year)')
        
        # Add grid lines
        ax.set_xticks(range(n_west_east))
        ax.set_yticks(range(n_south_north))
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Add cell values as text labels if requested
        if grid_labels:
            for i in range(n_south_north):
                for j in range(n_west_east):
                    value = exceedance_matrix[i, j]
                    if not np.isnan(value):
                        # Choose text color based on background
                        text_color = 'white' if value > np.nanmean(exceedance_matrix) else 'black'
                        ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                               color=text_color, fontsize=8, weight='bold')
        
        # Add coordinate references
        ax.set_xticks(range(n_west_east))
        ax.set_yticks(range(n_south_north))
        ax.set_xticklabels([f'{j}' for j in range(n_west_east)])
        ax.set_yticklabels([f'{i}' for i in range(n_south_north)])
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            # Create specific directory for threshold analysis
            threshold_plots_dir = os.path.join(figures_dir, "spatial_gradient", "ice_load_grid_threshold_exceedance")
            os.makedirs(threshold_plots_dir, exist_ok=True)
            
            # Create filename with threshold value
            threshold_str = f"{ice_load_threshold:.1f}".replace('.', 'p')
            plot_path = f"{threshold_plots_dir}/ice_load_threshold_exceedance_{threshold_str}kgm.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   Exceedance map saved to: {plot_path}")
        
        plt.close()  # Close the plot to prevent it from showing
        
        # Create additional summary statistics plot
        if len(valid_exceedances) > 0:
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram of exceedance values
            ax1.hist(valid_exceedances, bins=min(20, len(np.unique(valid_exceedances))), 
                    alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel(f'Exceedance ({unit_label}/Year)')
            ax1.set_ylabel('Number of Grid Cells')
            ax1.set_title(f'Distribution of Threshold Exceedances\nThreshold: {ice_load_threshold:.3f} kg/m')
            ax1.grid(True, alpha=0.3)
            
            # Box plot by row (south-north variation)
            row_data = []
            row_labels = []
            for i in range(n_south_north):
                row_exceedances = exceedance_matrix[i, :]
                valid_row = row_exceedances[~np.isnan(row_exceedances)]
                if len(valid_row) > 0:
                    row_data.append(valid_row)
                    row_labels.append(f'Row {i}')
            
            if row_data:
                ax2.boxplot(row_data, labels=row_labels)
                ax2.set_xlabel('Grid Row (South to North)')
                ax2.set_ylabel(f'Exceedance ({unit_label}/Year)')
                ax2.set_title('Exceedance by Grid Row')
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_plots:
                summary_path = f"{threshold_plots_dir}/ice_load_threshold_exceedance_summary_{threshold_str}kgm.png"
                plt.savefig(summary_path, dpi=300, bbox_inches='tight')
                print(f"   Summary statistics saved to: {summary_path}")
            
            plt.close()  # Close the plot to prevent it from showing
        
        # Prepare results dictionary
        results = {
            'threshold': ice_load_threshold,
            'units': units,
            'exceedance_matrix': exceedance_matrix,
            'grid_shape': (n_south_north, n_west_east),
            'n_years': n_years,
            'years_range': (years[0], years[-1]),
            'time_step_hours': time_step_hours,
            'statistics': {
                'mean': np.nanmean(exceedance_matrix),
                'std': np.nanstd(exceedance_matrix),
                'min': np.nanmin(exceedance_matrix),
                'max': np.nanmax(exceedance_matrix),
                'cells_with_exceedances': np.sum(valid_exceedances > 0) if len(valid_exceedances) > 0 else 0,
                'total_valid_cells': len(valid_exceedances) if len(valid_exceedances) > 0 else 0
            }
        }
        
        # Save detailed results to file
        if save_plots:
            results_path = f"{results_dir}/ice_load_threshold_exceedance_{threshold_str}kgm.txt"
            with open(results_path, 'w') as f:
                f.write("ICE LOAD THRESHOLD EXCEEDANCE ANALYSIS RESULTS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Threshold: {ice_load_threshold:.3f} kg/m\n")
                f.write(f"Units: {units}\n")
                f.write(f"Grid shape: {n_south_north} × {n_west_east}\n")
                f.write(f"Years analyzed: {n_years} ({years[0]} to {years[-1]})\n")
                f.write(f"Time step: {time_step_hours} hours\n\n")
                
                f.write("Exceedance Statistics:\n")
                f.write("-" * 25 + "\n")
                if len(valid_exceedances) > 0:
                    f.write(f"Mean: {results['statistics']['mean']:.3f} {units}/year\n")
                    f.write(f"Std: {results['statistics']['std']:.3f} {units}/year\n")
                    f.write(f"Min: {results['statistics']['min']:.3f} {units}/year\n")
                    f.write(f"Max: {results['statistics']['max']:.3f} {units}/year\n")
                    f.write(f"Cells with exceedances: {results['statistics']['cells_with_exceedances']}\n")
                    f.write(f"Total valid cells: {results['statistics']['total_valid_cells']}\n\n")
                
                f.write("Grid Cell Exceedance Values:\n")
                f.write("-" * 30 + "\n")
                for i in range(n_south_north):
                    row_str = f"Row {i:2d}: "
                    for j in range(n_west_east):
                        value = exceedance_matrix[i, j]
                        if np.isnan(value):
                            row_str += "   NaN   "
                        else:
                            row_str += f"{value:7.2f} "
                    f.write(row_str + "\n")
            
            print(f"   Detailed results saved to: {results_path}")
        
        print(f"\n✓ Ice load threshold exceedance analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in threshold exceedance analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def filter_dataset_by_thresholds(dataset, PBLH_min=None, PBLH_max=None, PRECIP_min=None, PRECIP_max=None, 
                                 QVAPOR_min=None, QVAPOR_max=None, RMOL_min=None, RMOL_max=None, 
                                 T_min=None, T_max=None, WS_min=None, WS_max=None, 
                                 WD_min=None, WD_max=None,
                                 height_level=0, verbose=True, 
                                 calculate_ice_load_cdf=False, dates=None, ice_load_method=None, 
                                 ice_load_threshold=0.0, months=None, percentile=None):
    """
    Filter dataset by removing timesteps where variables fall outside specified thresholds.
    Optionally calculate ice load and create CDF analysis on the filtered dataset.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Input dataset containing meteorological variables
    PBLH_min, PBLH_max : float, optional
        Minimum and maximum Planetary Boundary Layer Height (m)
    PRECIP_min, PRECIP_max : float, optional
        Minimum and maximum Precipitation (mm/h)
    QVAPOR_min, QVAPOR_max : float, optional
        Minimum and maximum Water vapor mixing ratio (kg/kg)
    RMOL_min, RMOL_max : float, optional
        Minimum and maximum Monin-Obukhov length (m)
    T_min, T_max : float, optional
        Minimum and maximum Temperature (K)
    WS_min, WS_max : float, optional
        Minimum and maximum Wind speed (m/s)
    WD_min, WD_max : float, optional
        Minimum and maximum Wind direction (degrees, 0-360)
    height_level : int, optional
        Height level index for variables with height dimension (default: 0)
    verbose : bool, optional
        Whether to print filtering information (default: True)
    calculate_ice_load_cdf : bool, optional
        Whether to calculate ice load and create CDF plots on filtered dataset (default: False)
    dates : list or pd.DatetimeIndex, optional
        Date range for ice load calculation (required if calculate_ice_load_cdf=True)
    ice_load_method : int, optional
        Method for ice load calculation (required if calculate_ice_load_cdf=True)
    ice_load_threshold : float, optional
        Minimum ice load value for CDF analysis (default: 0.0)
    months : list of int, optional
        List of months to include in CDF analysis (1-12)
    percentile : float, optional
        Percentile value for filtering extreme values in CDF
        
    Returns:
    --------
    xarray.Dataset
        Filtered dataset with timesteps removed based on thresholds
    dict
        Dictionary containing filtering statistics, information, and optionally CDF results
    """
    if verbose:
        print("=== DATASET FILTERING BY VARIABLE THRESHOLDS ===")
    
    try:
        # Check if dataset is valid
        if dataset is None:
            print("Error: Dataset is None")
            return None, None
        
        # Get initial dataset information
        initial_timesteps = len(dataset.time)
        initial_timespan = f"{dataset.time.values[0]} to {dataset.time.values[-1]}"
        
        if verbose:
            print(f"\n1. Initial Dataset Information:")
            print(f"   Total timesteps: {initial_timesteps:,}")
            print(f"   Time range: {initial_timespan}")
            print(f"   Grid shape: {dataset.sizes['south_north']} × {dataset.sizes['west_east']}")
            print(f"   Height level used: {height_level}")
        
        # Create variable thresholds dictionary from individual parameters
        variable_thresholds = {}
        available_vars = list(dataset.data_vars.keys())
        
        # Process each variable if min/max values are provided
        if PBLH_min is not None or PBLH_max is not None:
            if 'PBLH' in available_vars:
                variable_thresholds['PBLH'] = (PBLH_min if PBLH_min is not None else -np.inf, 
                                               PBLH_max if PBLH_max is not None else np.inf)
        
        if PRECIP_min is not None or PRECIP_max is not None:
            if 'PRECIP' in available_vars:
                variable_thresholds['PRECIP'] = (PRECIP_min if PRECIP_min is not None else -np.inf, 
                                                  PRECIP_max if PRECIP_max is not None else np.inf)
        
        if QVAPOR_min is not None or QVAPOR_max is not None:
            if 'QVAPOR' in available_vars:
                variable_thresholds['QVAPOR'] = (QVAPOR_min if QVAPOR_min is not None else -np.inf, 
                                                  QVAPOR_max if QVAPOR_max is not None else np.inf)
        
        if RMOL_min is not None or RMOL_max is not None:
            if 'RMOL' in available_vars:
                variable_thresholds['RMOL'] = (RMOL_min if RMOL_min is not None else -np.inf, 
                                               RMOL_max if RMOL_max is not None else np.inf)
        
        if T_min is not None or T_max is not None:
            if 'T' in available_vars:
                variable_thresholds['T'] = (T_min if T_min is not None else -np.inf, 
                                            T_max if T_max is not None else np.inf)
        
        if WS_min is not None or WS_max is not None:
            if 'WS' in available_vars:
                variable_thresholds['WS'] = (WS_min if WS_min is not None else -np.inf, 
                                             WS_max if WS_max is not None else np.inf)
        
        if WD_min is not None or WD_max is not None:
            if 'WD' in available_vars:
                variable_thresholds['WD'] = (WD_min if WD_min is not None else -np.inf, 
                                             WD_max if WD_max is not None else np.inf)
        
        # Check for variables not in dataset
        requested_vars = []
        if PBLH_min is not None or PBLH_max is not None:
            requested_vars.append('PBLH')
        if PRECIP_min is not None or PRECIP_max is not None:
            requested_vars.append('PRECIP')
        if QVAPOR_min is not None or QVAPOR_max is not None:
            requested_vars.append('QVAPOR')
        if RMOL_min is not None or RMOL_max is not None:
            requested_vars.append('RMOL')
        if T_min is not None or T_max is not None:
            requested_vars.append('T')
        if WS_min is not None or WS_max is not None:
            requested_vars.append('WS')
        if WD_min is not None or WD_max is not None:
            requested_vars.append('WD')
        
        missing_vars = [var for var in requested_vars if var not in available_vars]
        
        if missing_vars:
            print(f"Warning: Variables not found in dataset: {missing_vars}")
        
        if not variable_thresholds:
            print("Error: No valid variable thresholds provided")
            return dataset, {'error': 'No valid thresholds'}
        
        # Validate thresholds
        valid_thresholds = {}
        for var_name, (lower, upper) in variable_thresholds.items():
            if lower != -np.inf and upper != np.inf and lower >= upper:
                print(f"Warning: Invalid threshold for {var_name}: lower ({lower}) >= upper ({upper}). Skipping.")
                continue
            valid_thresholds[var_name] = (lower, upper)
        
        if not valid_thresholds:
            print("Error: No valid variable thresholds provided after validation")
            return dataset, {'error': 'No valid thresholds after validation'}
        
        # Create active filters dictionary for later use in folder naming and documentation
        active_filters = {}
        filtering_params = {
            'PBLH_min': PBLH_min, 'PBLH_max': PBLH_max,
            'PRECIP_min': PRECIP_min, 'PRECIP_max': PRECIP_max,
            'QVAPOR_min': QVAPOR_min, 'QVAPOR_max': QVAPOR_max,
            'RMOL_min': RMOL_min, 'RMOL_max': RMOL_max,
            'T_min': T_min, 'T_max': T_max,
            'WS_min': WS_min, 'WS_max': WS_max,
            'WD_min': WD_min, 'WD_max': WD_max
        }
        active_filters = {k: v for k, v in filtering_params.items() if v is not None}
        
        if verbose:
            print(f"\n2. Variable Thresholds:")
            for var_name, (lower, upper) in valid_thresholds.items():
                var_info = dataset[var_name]
                has_height = 'height' in var_info.dims
                height_info = f" (at height level {height_level})" if has_height else ""
                lower_str = f"{lower}" if lower != -np.inf else "No limit"
                upper_str = f"{upper}" if upper != np.inf else "No limit"
                print(f"   {var_name}{height_info}: {lower_str} <= value <= {upper_str}")
        
        # Create combined mask for all variables
        if verbose:
            print(f"\n3. Applying Filters:")
        
        combined_mask = None
        filter_stats = {}
        
        for var_name, (lower, upper) in valid_thresholds.items():
            var_data = dataset[var_name]
            
            # Handle variables with height dimension
            if 'height' in var_data.dims:
                if height_level >= var_data.sizes['height']:
                    print(f"Warning: Height level {height_level} not available for {var_name}. Using level 0.")
                    var_values = var_data.isel(height=0)
                else:
                    var_values = var_data.isel(height=height_level)
            else:
                var_values = var_data
            
            # Create mask for this variable (True where values are within bounds)
            var_mask = (var_values >= lower) & (var_values <= upper)
            
            # For variables with spatial dimensions, require ALL grid points to satisfy the condition
            if 'south_north' in var_mask.dims and 'west_east' in var_mask.dims:
                # All spatial points must be within bounds for timestep to be valid
                timestep_mask = var_mask.all(dim=['south_north', 'west_east'])
            else:
                timestep_mask = var_mask
            
            # Calculate statistics for this variable
            total_timesteps = len(timestep_mask)
            valid_timesteps = int(timestep_mask.sum())
            removed_timesteps = total_timesteps - valid_timesteps
            removal_percentage = (removed_timesteps / total_timesteps) * 100
            
            # Use appropriate values for statistics calculation
            if var_name == 'WD' and 'south_north' in var_values.dims and 'west_east' in var_values.dims:
                stats_values = var_values.mean(dim=['south_north', 'west_east'])
            else:
                stats_values = var_values
            
            filter_stats[var_name] = {
                'threshold_lower': lower,
                'threshold_upper': upper,
                'valid_timesteps': valid_timesteps,
                'removed_timesteps': removed_timesteps,
                'removal_percentage': removal_percentage,
                'original_range': (float(stats_values.min()), float(stats_values.max())),
                'original_mean': float(stats_values.mean())
            }
            
            if verbose:
                print(f"   {var_name}: {removed_timesteps:,} timesteps removed ({removal_percentage:.1f}%)")
                print(f"     Original range: [{filter_stats[var_name]['original_range'][0]:.3f}, {filter_stats[var_name]['original_range'][1]:.3f}]")
                print(f"     Original mean: {filter_stats[var_name]['original_mean']:.3f}")
            
            # Combine with overall mask
            if combined_mask is None:
                combined_mask = timestep_mask
            else:
                # Ensure both masks have compatible coordinates before combining
                if hasattr(timestep_mask, 'coords') and 'height' in timestep_mask.coords:
                    timestep_mask = timestep_mask.drop_vars('height', errors='ignore')
                if hasattr(combined_mask, 'coords') and 'height' in combined_mask.coords:
                    combined_mask = combined_mask.drop_vars('height', errors='ignore')
                combined_mask = combined_mask & timestep_mask
        
        # Apply the combined filter
        if combined_mask is not None:
            # Ensure the mask only has time dimension (remove any height coordinates)
            if hasattr(combined_mask, 'coords') and 'height' in combined_mask.coords:
                combined_mask = combined_mask.drop_vars('height', errors='ignore')
            
            # Convert to boolean array if needed
            if hasattr(combined_mask, 'values'):
                mask_values = combined_mask.values
            else:
                mask_values = combined_mask
            
            # Count final results
            final_valid_timesteps = int(mask_values.sum())
            final_removed_timesteps = initial_timesteps - final_valid_timesteps
            final_removal_percentage = (final_removed_timesteps / initial_timesteps) * 100
            
            # Filter the dataset using boolean indexing
            try:
                # Method 1: Use sel with boolean indexing
                filtered_dataset = dataset.isel(time=mask_values)
                
                # Fix coordinate inconsistencies after filtering
                # This prevents xarray dimension/coordinate mismatch errors
                filtered_dataset = filtered_dataset.copy()
                
                # Ensure time coordinate is properly aligned
                if 'time' in filtered_dataset.coords:
                    # Reset time coordinate to prevent dimension mismatches
                    time_coord = filtered_dataset.coords['time']
                    if hasattr(time_coord, 'values'):
                        filtered_dataset = filtered_dataset.assign_coords(time=time_coord.values)
                
            except Exception as e:
                if verbose:
                    print(f"   Warning: Standard filtering failed, trying alternative method...")
                    print(f"   Error: {e}")
                try:
                    # Method 2: Use where with drop=True
                    filtered_dataset = dataset.where(combined_mask, drop=True)
                except Exception as e2:
                    if verbose:
                        print(f"   Warning: Alternative filtering also failed: {e2}")
                    # Method 3: Create new dataset with selected timesteps
                    time_coords = dataset.time.values[mask_values]
                    filtered_dataset = dataset.sel(time=time_coords)
            
            if verbose:
                print(f"\n4. Combined Filtering Results:")
                print(f"   Original timesteps: {initial_timesteps:,}")
                print(f"   Remaining timesteps: {final_valid_timesteps:,}")
                print(f"   Removed timesteps: {final_removed_timesteps:,}")
                print(f"   Removal percentage: {final_removal_percentage:.1f}%")
                
                if final_valid_timesteps > 0:
                    filtered_timespan = f"{filtered_dataset.time.values[0]} to {filtered_dataset.time.values[-1]}"
                    print(f"   Filtered time range: {filtered_timespan}")
                else:
                    print("   Warning: No timesteps remain after filtering!")
            
            # Prepare results dictionary
            results = {
                'initial_timesteps': initial_timesteps,
                'final_timesteps': final_valid_timesteps,
                'removed_timesteps': final_removed_timesteps,
                'removal_percentage': final_removal_percentage,
                'initial_timespan': initial_timespan,
                'filtered_timespan': f"{filtered_dataset.time.values[0]} to {filtered_dataset.time.values[-1]}" if final_valid_timesteps > 0 else "No data remaining",
                'variable_stats': filter_stats,
                'height_level_used': height_level,
                'thresholds_applied': valid_thresholds
            }
            
            # Optional: Calculate ice load and create CDF plots on filtered dataset
            if calculate_ice_load_cdf:
                if dates is None or ice_load_method is None:
                    print("Warning: dates and ice_load_method are required for ice load CDF calculation. Skipping CDF analysis.")
                    cdf_results = None
                else:
                    if verbose:
                        print(f"\n5. Calculating Ice Load and CDF Analysis on Filtered Dataset...")
                    
                    try:
                        # Create filter-specific directory structure
                        filter_base_dir = os.path.join("results", "figures", "spatial_gradient", "filtered")
                        os.makedirs(filter_base_dir, exist_ok=True)
                        
                        # Create folder name based on applied filters
                        folder_name_parts = []
                        for param, value in active_filters.items():
                            var_name = param.split('_')[0]
                            limit_type = param.split('_')[1]
                            folder_name_parts.append(f"{var_name}_{limit_type}_{value}")
                        
                        if folder_name_parts:
                            filter_folder_name = "_".join(folder_name_parts[:4])  # Limit to avoid too long names
                        else:
                            filter_folder_name = "no_filters"
                        
                        filter_specific_dir = os.path.join(filter_base_dir, filter_folder_name)
                        os.makedirs(filter_specific_dir, exist_ok=True)
                        
                        if verbose:
                            print(f"   Results will be saved to: {filter_specific_dir}")
                        
                        # Calculate ice load on filtered dataset
                        if verbose:
                            print(f"   Calculating ice load (method {ice_load_method}) at height level {height_level}...")
                        
                        ice_load_data = calculate_ice_load(
                            filtered_dataset, 
                            dates, 
                            ice_load_method, 
                            height_level=height_level, 
                            create_figures=False
                        )
                        
                        if ice_load_data is None:
                            print("   Error: Ice load calculation failed")
                            cdf_results = None
                        else:
                            # Temporarily change the figures directory for CDF plots
                            original_figures_dir = globals().get('figures_dir', 'figures')
                            globals()['figures_dir'] = filter_specific_dir
                            
                            # Create CDF plots
                            if verbose:
                                print(f"   Creating CDF plots...")
                            
                            cdf_results = plot_ice_load_cdf_curves(
                                ice_load_data=ice_load_data,
                                save_plots=True,
                                ice_load_bins=None,
                                ice_load_threshold=ice_load_threshold,
                                months=months,
                                percentile=percentile
                            )
                            
                            # Restore original figures directory
                            globals()['figures_dir'] = original_figures_dir
                            
                            # Create filter documentation file
                            doc_file_path = os.path.join(filter_specific_dir, "filter_documentation.txt")
                            with open(doc_file_path, 'w', encoding='utf-8') as f:
                                f.write("METEOROLOGICAL FILTERING DOCUMENTATION\n")
                                f.write("=" * 50 + "\n\n")
                                f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                                
                                f.write("APPLIED FILTERS:\n")
                                f.write("-" * 20 + "\n")
                                if active_filters:
                                    for param, value in active_filters.items():
                                        var_name = param.split('_')[0]
                                        limit_type = param.split('_')[1]
                                        f.write(f"{var_name} {limit_type}: {value}\n")
                                else:
                                    f.write("No meteorological filters applied\n")
                                
                                f.write(f"\nHeight level used: {height_level}\n")
                                
                                f.write(f"\nFILTERING RESULTS:\n")
                                f.write("-" * 20 + "\n")
                                f.write(f"Original timesteps: {initial_timesteps:,}\n")
                                f.write(f"Filtered timesteps: {final_valid_timesteps:,}\n")
                                f.write(f"Removed timesteps: {final_removed_timesteps:,}\n")
                                f.write(f"Removal percentage: {final_removal_percentage:.2f}%\n")
                                f.write(f"Original timespan: {initial_timespan}\n")
                                if final_valid_timesteps > 0:
                                    filtered_timespan = f"{filtered_dataset.time.values[0]} to {filtered_dataset.time.values[-1]}"
                                    f.write(f"Filtered timespan: {filtered_timespan}\n")
                                
                                f.write(f"\nVARIABLE STATISTICS:\n")
                                f.write("-" * 20 + "\n")
                                for var_name, stats in filter_stats.items():
                                    f.write(f"\n{var_name}:\n")
                                    f.write(f"  Threshold: {stats['threshold_lower']} <= value <= {stats['threshold_upper']}\n")
                                    f.write(f"  Original range: [{stats['original_range'][0]:.3f}, {stats['original_range'][1]:.3f}]\n")
                                    f.write(f"  Original mean: {stats['original_mean']:.3f}\n")
                                    f.write(f"  Removed timesteps: {stats['removed_timesteps']:,} ({stats['removal_percentage']:.1f}%)\n")
                                
                                f.write(f"\nICE LOAD ANALYSIS:\n")
                                f.write("-" * 20 + "\n")
                                f.write(f"Ice load calculation method: {ice_load_method}\n")
                                f.write(f"Height level: {height_level}\n")
                                f.write(f"Ice load threshold: {ice_load_threshold} kg/m\n")
                                if months:
                                    f.write(f"Months analyzed: {months}\n")
                                if percentile:
                                    f.write(f"Percentile filter: {percentile}%\n")
                                
                                if cdf_results and 'statistics' in cdf_results:
                                    f.write(f"Grid cells analyzed: {len(cdf_results['statistics'])}\n")
                                    if cdf_results['statistics']:
                                        all_stats = list(cdf_results['statistics'].values())
                                        max_ice = max([s['max_ice_load'] for s in all_stats])
                                        mean_ice = np.mean([s['mean_ice_load'] for s in all_stats])
                                        f.write(f"Maximum ice load across domain: {max_ice:.3f} kg/m\n")
                                        f.write(f"Average ice load across domain: {mean_ice:.3f} kg/m\n")
                                
                                f.write(f"\nOUTPUT FILES:\n")
                                f.write("-" * 20 + "\n")
                                f.write("This directory contains:\n")
                                f.write("- CDF plots for individual grid cells\n")
                                f.write("- Summary CDF plot for domain average\n")
                                f.write("- Spatial gradient plots (if applicable)\n")
                                f.write("- This documentation file\n")
                            
                            if verbose:
                                print(f"   Filter documentation saved to: {doc_file_path}")
                                print(f"   CDF analysis completed with {len(cdf_results['cdf_curves']) if cdf_results else 0} grid cells")
                        
                        # Add CDF results to the main results dictionary
                        results['ice_load_data'] = ice_load_data
                        results['cdf_results'] = cdf_results
                        results['filter_directory'] = filter_specific_dir
                        
                    except Exception as e:
                        print(f"   Error in ice load CDF calculation: {e}")
                        import traceback
                        traceback.print_exc()
                        results['ice_load_data'] = None
                        results['cdf_results'] = None
            
            if verbose:
                print(f"\n✓ Dataset filtering completed successfully!")
            
            return filtered_dataset, results
            
        else:
            print("Error: No valid filtering mask created")
            return dataset, {'error': 'No valid mask'}
            
    except Exception as e:
        print(f"Error in dataset filtering: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def systematic_meteorological_filtering(dataset, 
                                       PBLH_range=None, PRECIP_range=None, QVAPOR_range=None,
                                       RMOL_range=None, T_range=None, WS_range=None, WD_range=None,
                                       height_level=0, verbose=True, max_combinations=None,
                                       calculate_ice_load_cdf=False, dates=None, ice_load_method=None,
                                       ice_load_threshold=0.0, months=None, percentile=None,
                                       save_results=True, results_summary_file="systematic_filtering_results.txt"):
    """
    Systematically filter dataset using all possible combinations of meteorological parameter ranges.
    Each parameter range is defined as (min_val, max_val, step) and creates a grid search across
    all specified parameter combinations.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Input dataset containing meteorological variables
    PBLH_range : tuple (min, max, step), optional
        Range for Planetary Boundary Layer Height (m)
        Example: (100, 1000, 200) creates values [100, 300, 500, 700, 900]
    PRECIP_range : tuple (min, max, step), optional
        Range for Precipitation (mm/h)
    QVAPOR_range : tuple (min, max, step), optional
        Range for Water vapor mixing ratio (kg/kg)
    RMOL_range : tuple (min, max, step), optional
        Range for Monin-Obukhov length (m)
    T_range : tuple (min, max, step), optional
        Range for Temperature (K)
    WS_range : tuple (min, max, step), optional
        Range for Wind speed (m/s)
    WD_range : tuple (min, max, step), optional
        Range for Wind direction (degrees, 0-360)
    height_level : int, optional
        Height level index for variables with height dimension (default: 0)
    verbose : bool, optional
        Whether to print detailed progress information (default: True)
    max_combinations : int, optional
        Maximum number of combinations to process (for testing/limiting computation)
    calculate_ice_load_cdf : bool, optional
        Whether to calculate ice load and create CDF plots for each combination (default: False)
    dates : list or pd.DatetimeIndex, optional
        Date range for ice load calculation (required if calculate_ice_load_cdf=True)
    ice_load_method : int, optional
        Method for ice load calculation (required if calculate_ice_load_cdf=True)
    ice_load_threshold : float, optional
        Minimum ice load value for CDF analysis (default: 0.0)
    months : list of int, optional
        List of months to include in CDF analysis (1-12)
    percentile : float, optional
        Percentile value for filtering extreme values in CDF
    save_results : bool, optional
        Whether to save results summary to file (default: True)
    results_summary_file : str, optional
        Filename for results summary (default: "systematic_filtering_results.txt")
        
    Returns:
    --------
    dict
        Dictionary containing all combination results, statistics, and summary information
        
    Example:
    --------
    >>> # Test with small ranges
    >>> results = systematic_meteorological_filtering(
    ...     dataset, 
    ...     WS_range=(5, 15, 5),  # Wind speed: [5, 10, 15] m/s
    ...     T_range=(270, 280, 5),  # Temperature: [270, 275, 280] K
    ...     max_combinations=10,
    ...     calculate_ice_load_cdf=True,
    ...     dates=date_range,
    ...     ice_load_method=5
    ... )
    """
    
    if verbose:
        print("=== SYSTEMATIC METEOROLOGICAL FILTERING ===")
        print("Grid search across multiple parameter combinations")
    
    try:
        import itertools
        
        # Validate input dataset
        if dataset is None:
            raise ValueError("Dataset is None")
        
        # Initialize parameter ranges dictionary
        param_ranges = {}
        
        # Process each parameter range
        if PBLH_range is not None:
            if len(PBLH_range) != 3:
                raise ValueError("PBLH_range must be tuple (min, max, step)")
            min_val, max_val, step = PBLH_range
            param_ranges['PBLH'] = list(np.arange(min_val, max_val + step, step))
        
        if PRECIP_range is not None:
            if len(PRECIP_range) != 3:
                raise ValueError("PRECIP_range must be tuple (min, max, step)")
            min_val, max_val, step = PRECIP_range
            param_ranges['PRECIP'] = list(np.arange(min_val, max_val + step, step))
        
        if QVAPOR_range is not None:
            if len(QVAPOR_range) != 3:
                raise ValueError("QVAPOR_range must be tuple (min, max, step)")
            min_val, max_val, step = QVAPOR_range
            param_ranges['QVAPOR'] = list(np.arange(min_val, max_val + step, step))
        
        if RMOL_range is not None:
            if len(RMOL_range) != 3:
                raise ValueError("RMOL_range must be tuple (min, max, step)")
            min_val, max_val, step = RMOL_range
            param_ranges['RMOL'] = list(np.arange(min_val, max_val + step, step))
        
        if T_range is not None:
            if len(T_range) != 3:
                raise ValueError("T_range must be tuple (min, max, step)")
            min_val, max_val, step = T_range
            param_ranges['T'] = list(np.arange(min_val, max_val + step, step))
        
        if WS_range is not None:
            if len(WS_range) != 3:
                raise ValueError("WS_range must be tuple (min, max, step)")
            min_val, max_val, step = WS_range
            param_ranges['WS'] = list(np.arange(min_val, max_val + step, step))
        
        if WD_range is not None:
            if len(WD_range) != 3:
                raise ValueError("WD_range must be tuple (min, max, step)")
            min_val, max_val, step = WD_range
            param_ranges['WD'] = list(np.arange(min_val, max_val + step, step))
        
        if not param_ranges:
            raise ValueError("At least one parameter range must be specified")
        
        if verbose:
            print(f"\n1. Parameter Ranges Defined:")
            for param, values in param_ranges.items():
                print(f"   {param}: {len(values)} values [{min(values):.3f} to {max(values):.3f}, step {values[1]-values[0]:.3f}]")
        
        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        all_combinations = list(itertools.product(*param_values))
        
        total_combinations = len(all_combinations)
        
        if verbose:
            print(f"\n2. Combination Generation:")
            print(f"   Total possible combinations: {total_combinations:,}")
        
        # Limit combinations if specified
        if max_combinations is not None and total_combinations > max_combinations:
            if verbose:
                print(f"   Limiting to first {max_combinations:,} combinations")
            all_combinations = all_combinations[:max_combinations]
            total_combinations = len(all_combinations)
        
        # Initialize results storage
        results = {
            'parameter_ranges': param_ranges,
            'total_combinations': total_combinations,
            'completed_combinations': 0,
            'failed_combinations': 0,
            'combination_results': {},
            'summary_statistics': {},
            'best_combinations': {},
            'execution_info': {
                'start_time': pd.Timestamp.now(),
                'height_level': height_level,
                'ice_load_cdf_enabled': calculate_ice_load_cdf,
                'ice_load_method': ice_load_method,
                'ice_load_threshold': ice_load_threshold,
                'months_filter': months,
                'percentile_filter': percentile
            }
        }
        
        # Process each combination
        if verbose:
            print(f"\n3. Processing {total_combinations:,} combinations...")
            print("   Progress: ", end="")
        
        successful_results = []
        
        for combo_idx, combination in enumerate(all_combinations):
            try:
                # Create parameter dictionary for this combination
                combo_params = dict(zip(param_names, combination))
                
                # Prepare parameters for filter_dataset_by_thresholds
                filter_params = {
                    'dataset': dataset,
                    'height_level': height_level,
                    'verbose': combo_idx < 3 if verbose else False,  # Show details for first few combinations
                    'calculate_ice_load_cdf': calculate_ice_load_cdf,
                    'dates': dates,
                    'ice_load_method': ice_load_method,
                    'ice_load_threshold': ice_load_threshold,
                    'months': months,
                    'percentile': percentile
                }
                
                # Set parameter thresholds (using values as both min and max for exact filtering)
                # Note: This creates filters where values must be >= threshold
                for param_name, threshold_value in combo_params.items():
                    if param_name == 'PBLH':
                        filter_params['PBLH_min'] = threshold_value
                    elif param_name == 'PRECIP':
                        filter_params['PRECIP_min'] = threshold_value
                    elif param_name == 'QVAPOR':
                        filter_params['QVAPOR_min'] = threshold_value
                    elif param_name == 'RMOL':
                        filter_params['RMOL_min'] = threshold_value
                    elif param_name == 'T':
                        filter_params['T_min'] = threshold_value
                    elif param_name == 'WS':
                        filter_params['WS_min'] = threshold_value
                    elif param_name == 'WD':
                        filter_params['WD_min'] = threshold_value
                
                # Apply filtering
                filtered_dataset, filter_results = filter_dataset_by_thresholds(**filter_params)
                
                if filtered_dataset is not None and filter_results is not None and 'error' not in filter_results:
                    # Store successful result
                    combo_key = f"combo_{combo_idx:04d}"
                    
                    # Create a compact result summary
                    combo_result = {
                        'combination_index': combo_idx,
                        'parameters': combo_params,
                        'initial_timesteps': filter_results['initial_timesteps'],
                        'final_timesteps': filter_results['final_timesteps'],
                        'removal_percentage': filter_results['removal_percentage'],
                        'data_retention_percentage': 100 - filter_results['removal_percentage'],
                        'filtered_timespan': filter_results['filtered_timespan']
                    }
                    
                    # Add ice load results if available
                    if calculate_ice_load_cdf and 'cdf_results' in filter_results and filter_results['cdf_results']:
                        cdf_results = filter_results['cdf_results']
                        if 'statistics' in cdf_results and cdf_results['statistics']:
                            # Calculate domain-wide ice load statistics
                            all_stats = list(cdf_results['statistics'].values())
                            combo_result['ice_load_stats'] = {
                                'max_ice_load_domain': max([s['max_ice_load'] for s in all_stats]),
                                'mean_ice_load_domain': np.mean([s['mean_ice_load'] for s in all_stats]),
                                'mean_ice_occurrence': np.mean([s['ice_occurrence_percentage'] for s in all_stats]),
                                'processed_cells': len(all_stats)
                            }
                        
                        # Store information about where plots were saved
                        if 'filter_directory' in filter_results:
                            combo_result['plot_directory'] = filter_results['filter_directory']
                            if verbose and combo_idx < 3:  # Show directory info for first few combinations
                                print(f"   Combo {combo_idx:04d}: CDF plots saved to {filter_results['filter_directory']}")
                    
                    results['combination_results'][combo_key] = combo_result
                    successful_results.append(combo_result)
                    results['completed_combinations'] += 1
                    
                else:
                    results['failed_combinations'] += 1
                
                # Progress indicator
                if verbose and (combo_idx + 1) % max(1, total_combinations // 20) == 0:
                    progress_pct = ((combo_idx + 1) / total_combinations) * 100
                    print(f"{progress_pct:.0f}%...", end="")
                
            except Exception as e:
                if verbose and combo_idx < 5:  # Only show first few errors
                    print(f"\n   Error in combination {combo_idx}: {e}")
                results['failed_combinations'] += 1
                continue
        
        if verbose:
            print(" Done!")
        
        # Calculate summary statistics
        if successful_results:
            if verbose:
                print(f"\n4. Calculating Summary Statistics...")
            
            # Data retention statistics
            retention_percentages = [r['data_retention_percentage'] for r in successful_results]
            final_timesteps = [r['final_timesteps'] for r in successful_results]
            
            results['summary_statistics'] = {
                'successful_combinations': len(successful_results),
                'failed_combinations': results['failed_combinations'],
                'success_rate': (len(successful_results) / total_combinations) * 100,
                'data_retention': {
                    'mean': np.mean(retention_percentages),
                    'std': np.std(retention_percentages),
                    'min': np.min(retention_percentages),
                    'max': np.max(retention_percentages),
                    'median': np.median(retention_percentages)
                },
                'final_timesteps': {
                    'mean': np.mean(final_timesteps),
                    'std': np.std(final_timesteps),
                    'min': np.min(final_timesteps),
                    'max': np.max(final_timesteps),
                    'median': np.median(final_timesteps)
                }
            }
            
            # Ice load statistics (if available)
            if calculate_ice_load_cdf:
                ice_results = [r for r in successful_results if 'ice_load_stats' in r]
                if ice_results:
                    max_ice_loads = [r['ice_load_stats']['max_ice_load_domain'] for r in ice_results]
                    mean_ice_loads = [r['ice_load_stats']['mean_ice_load_domain'] for r in ice_results]
                    ice_occurrences = [r['ice_load_stats']['mean_ice_occurrence'] for r in ice_results]
                    
                    results['summary_statistics']['ice_load'] = {
                        'combinations_with_ice_data': len(ice_results),
                        'max_ice_load_domain': {
                            'mean': np.mean(max_ice_loads),
                            'std': np.std(max_ice_loads),
                            'min': np.min(max_ice_loads),
                            'max': np.max(max_ice_loads)
                        },
                        'mean_ice_load_domain': {
                            'mean': np.mean(mean_ice_loads),
                            'std': np.std(mean_ice_loads),
                            'min': np.min(mean_ice_loads),
                            'max': np.max(mean_ice_loads)
                        },
                        'ice_occurrence': {
                            'mean': np.mean(ice_occurrences),
                            'std': np.std(ice_occurrences),
                            'min': np.min(ice_occurrences),
                            'max': np.max(ice_occurrences)
                        }
                    }
            
            # Find best combinations based on different criteria
            results['best_combinations'] = {
                'highest_data_retention': max(successful_results, key=lambda x: x['data_retention_percentage']),
                'lowest_data_retention': min(successful_results, key=lambda x: x['data_retention_percentage']),
                'most_final_timesteps': max(successful_results, key=lambda x: x['final_timesteps']),
                'least_final_timesteps': min(successful_results, key=lambda x: x['final_timesteps'])
            }
            
            if calculate_ice_load_cdf and ice_results:
                results['best_combinations']['highest_max_ice_load'] = max(ice_results, key=lambda x: x['ice_load_stats']['max_ice_load_domain'])
                results['best_combinations']['lowest_max_ice_load'] = min(ice_results, key=lambda x: x['ice_load_stats']['max_ice_load_domain'])
                results['best_combinations']['highest_ice_occurrence'] = max(ice_results, key=lambda x: x['ice_load_stats']['mean_ice_occurrence'])
                results['best_combinations']['lowest_ice_occurrence'] = min(ice_results, key=lambda x: x['ice_load_stats']['mean_ice_occurrence'])
        
        # Complete execution info
        results['execution_info']['end_time'] = pd.Timestamp.now()
        results['execution_info']['total_duration'] = results['execution_info']['end_time'] - results['execution_info']['start_time']
        
        # Print summary
        if verbose:
            print(f"\n5. Results Summary:")
            print(f"   Total combinations processed: {total_combinations:,}")
            print(f"   Successful combinations: {results['completed_combinations']:,}")
            print(f"   Failed combinations: {results['failed_combinations']:,}")
            print(f"   Success rate: {results['summary_statistics']['success_rate']:.1f}%" if successful_results else "   Success rate: 0%")
            print(f"   Total execution time: {results['execution_info']['total_duration']}")
            
            if successful_results:
                print(f"\n   Data Retention Statistics:")
                print(f"     Mean: {results['summary_statistics']['data_retention']['mean']:.1f}%")
                print(f"     Range: {results['summary_statistics']['data_retention']['min']:.1f}% to {results['summary_statistics']['data_retention']['max']:.1f}%")
                
                # Show where plots were saved if ice load CDF was calculated
                if calculate_ice_load_cdf:
                    plot_dirs = [r.get('plot_directory') for r in successful_results if 'plot_directory' in r]
                    unique_plot_dirs = list(set([d for d in plot_dirs if d is not None]))
                    if unique_plot_dirs:
                        print(f"\n   CDF Plots Created:")
                        print(f"     Number of combinations with plots: {len(plot_dirs)}")
                        print(f"     Plot directories created: {len(unique_plot_dirs)}")
                        print(f"     Base directory: results/figures/spatial_gradient/filtered/")
                        if len(unique_plot_dirs) <= 5:  # Show directories if not too many
                            for i, plot_dir in enumerate(unique_plot_dirs[:5]):
                                folder_name = plot_dir.split('/')[-1] if '/' in plot_dir else plot_dir.split('\\')[-1]
                                print(f"       {i+1}. {folder_name}")
                        else:
                            print(f"       (Too many directories to list - check base directory)")
                
                print(f"\n   Best Combination (Highest Data Retention):")
                best_retention = results['best_combinations']['highest_data_retention']
                print(f"     Parameters: {best_retention['parameters']}")
                print(f"     Data retention: {best_retention['data_retention_percentage']:.1f}%")
                print(f"     Final timesteps: {best_retention['final_timesteps']:,}")
                if 'plot_directory' in best_retention:
                    print(f"     Plots saved in: {best_retention['plot_directory']}")
        
        # Save results to file
        if save_results:
            if verbose:
                print(f"\n6. Saving Results...")
            
            results_path = os.path.join(results_dir, results_summary_file)
            
            with open(results_path, 'w', encoding='utf-8') as f:
                f.write("SYSTEMATIC METEOROLOGICAL FILTERING RESULTS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {results['execution_info']['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {results['execution_info']['total_duration']}\n\n")
                
                f.write("PARAMETER RANGES:\n")
                f.write("-" * 20 + "\n")
                for param, values in param_ranges.items():
                    f.write(f"{param}: {len(values)} values [{min(values):.3f} to {max(values):.3f}]\n")
                
                f.write(f"\nEXECUTION SUMMARY:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total combinations: {total_combinations:,}\n")
                f.write(f"Successful: {results['completed_combinations']:,}\n")
                f.write(f"Failed: {results['failed_combinations']:,}\n")
                f.write(f"Success rate: {results['summary_statistics']['success_rate']:.1f}%\n" if successful_results else "Success rate: 0%\n")
                f.write(f"Height level: {height_level}\n")
                f.write(f"Ice load CDF: {calculate_ice_load_cdf}\n")
                
                if successful_results:
                    f.write(f"\nDATA RETENTION STATISTICS:\n")
                    f.write("-" * 30 + "\n")
                    stats = results['summary_statistics']['data_retention']
                    f.write(f"Mean retention: {stats['mean']:.1f}%\n")
                    f.write(f"Std deviation: {stats['std']:.1f}%\n")
                    f.write(f"Min retention: {stats['min']:.1f}%\n")
                    f.write(f"Max retention: {stats['max']:.1f}%\n")
                    f.write(f"Median retention: {stats['median']:.1f}%\n")
                    
                    f.write(f"\nBEST COMBINATIONS:\n")
                    f.write("-" * 20 + "\n")
                    for criterion, combo in results['best_combinations'].items():
                        f.write(f"\n{criterion.replace('_', ' ').title()}:\n")
                        f.write(f"  Parameters: {combo['parameters']}\n")
                        f.write(f"  Data retention: {combo['data_retention_percentage']:.1f}%\n")
                        f.write(f"  Final timesteps: {combo['final_timesteps']:,}\n")
                        if 'ice_load_stats' in combo:
                            f.write(f"  Max ice load: {combo['ice_load_stats']['max_ice_load_domain']:.3f} kg/m\n")
                            f.write(f"  Mean ice occurrence: {combo['ice_load_stats']['mean_ice_occurrence']:.1f}%\n")
                    
                    f.write(f"\nALL SUCCESSFUL COMBINATIONS:\n")
                    f.write("-" * 35 + "\n")
                    f.write("Index | Parameters | Retention% | Timesteps | Timespan\n")
                    f.write("-" * 80 + "\n")
                    
                    for result in successful_results[:50]:  # Limit to first 50 for readability
                        params_str = ", ".join([f"{k}={v:.1f}" for k, v in result['parameters'].items()])
                        f.write(f"{result['combination_index']:5d} | {params_str:30s} | {result['data_retention_percentage']:8.1f} | {result['final_timesteps']:9,d} | {result['filtered_timespan']}\n")
                    
                    if len(successful_results) > 50:
                        f.write(f"... and {len(successful_results) - 50} more combinations\n")
            
            if verbose:
                print(f"   Results saved to: {results_path}")
        
        if verbose:
            print(f"\n✓ Systematic meteorological filtering completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"Error in systematic meteorological filtering: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_ice_load_with_filtering_and_cdf(
    dataset_with_ice_load,
    ice_load_variable='ICE_LOAD',
    height_level=0,
    save_plots=True,
    results_subdir="filtered_ice_load_cdf_analysis",
    # Filtering parameters (min, max for each variable)
    WD_range=None,        # (min, max) for Wind Direction
    WS_range=None,        # (min, max) for Wind Speed
    T_range=None,         # (min, max) for Temperature
    PBLH_range=None,      # (min, max) for Boundary Layer Height
    PRECIP_range=None,    # (min, max) for Precipitation
    QVAPOR_range=None,    # (min, max) for Water Vapor
    RMOL_range=None,      # (min, max) for Monin-Obukhov Length
    # CDF analysis parameters
    ice_load_threshold=0.0,
    ice_load_bins=None,
    months=None,
    percentile=None
):
    """
    Comprehensive function that filters a dataset with ice load by meteorological variables
    and performs CDF analysis with spatial gradient visualization.
    
    This function combines meteorological filtering with detailed CDF analysis and creates
    spatial gradient plots showing Earth Mover's distance between neighboring grid cells.
    
    Parameters:
    -----------
    dataset_with_ice_load : xarray.Dataset
        Dataset that already contains ice load as a variable (from add_ice_load_to_dataset)
    ice_load_variable : str, optional
        Name of the ice load variable in the dataset (default: 'ICE_LOAD')
    height_level : int, optional
        Height level index to use for analysis (default: 0)
    save_plots : bool, optional
        Whether to save plots to files (default: True)
    results_subdir : str, optional
        Subdirectory name for saving results (default: "filtered_ice_load_cdf_analysis")
        
    Filtering Parameters (all optional):
    -----------------------------------
    WD_range : tuple, optional
        Wind direction range: (min, max) in degrees
    WS_range : tuple, optional
        Wind speed range: (min, max) in m/s
    T_range : tuple, optional
        Temperature range: (min, max) in K
    PBLH_range : tuple, optional
        Boundary layer height range: (min, max) in m
    PRECIP_range : tuple, optional
        Precipitation range: (min, max) in mm
    QVAPOR_range : tuple, optional
        Water vapor mixing ratio range: (min, max) in kg/kg
    RMOL_range : tuple, optional
        Monin-Obukhov length range: (min, max) in m
        
    CDF Analysis Parameters:
    ------------------------
    ice_load_threshold : float, optional
        Minimum ice load threshold for CDF analysis (default: 0.0)
    ice_load_bins : array-like, optional
        Custom ice load bins for CDF analysis
    months : list, optional
        List of months to include (e.g., [1,2,12] for winter)
    percentile : float, optional
        Percentile threshold for extreme value filtering
        
    Returns:
    --------
    dict
        Comprehensive results including filtering info, CDF data, and spatial gradients
    """
    
    print("=== ICE LOAD ANALYSIS WITH FILTERING AND CDF ===")
    
    # Check if ice load variable exists
    if ice_load_variable not in dataset_with_ice_load.data_vars:
        raise ValueError(f"Ice load variable '{ice_load_variable}' not found in dataset. "
                        f"Available variables: {list(dataset_with_ice_load.data_vars.keys())}")
    
    print(f"Using ice load variable: {ice_load_variable}")
    print(f"Height level: {height_level} ({dataset_with_ice_load.height.values[height_level]} m)")
    
    # Create results directory based on filters applied
    if save_plots:
        # Create organized directory structure
        filters_base_dir = os.path.join("results", "figures", "filters")
        os.makedirs(filters_base_dir, exist_ok=True)
        
        # Generate folder name based on applied filters
        folder_name_parts = []
        
        # Add height level info
        folder_name_parts.append(f"h{height_level}")
        
        # Add filter information to folder name
        filter_params = {
            'WD': WD_range,
            'WS': WS_range, 
            'T': T_range,
            'PBLH': PBLH_range,
            'PRECIP': PRECIP_range,
            'QVAPOR': QVAPOR_range,
            'RMOL': RMOL_range
        }
        
        for param, value_range in filter_params.items():
            if value_range is not None:
                min_val, max_val = value_range
                folder_name_parts.append(f"{param}_{min_val}to{max_val}")
        
        # Add month filtering if specified
        if months is not None:
            months_str = "_".join(map(str, sorted(months)))
            folder_name_parts.append(f"months_{months_str}")
        
        # Add percentile filtering if specified
        if percentile is not None:
            folder_name_parts.append(f"p{percentile}")
        
        # Add ice load threshold if specified
        if ice_load_threshold > 0:
            folder_name_parts.append(f"min_{ice_load_threshold:.1f}")
        
        # Create unique folder name
        if folder_name_parts:
            folder_name = "_".join(folder_name_parts)
        else:
            folder_name = "no_filters"
        
        # Add timestamp to ensure uniqueness
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{folder_name}_{timestamp}"
        
        base_results_dir = os.path.join(filters_base_dir, folder_name)
        os.makedirs(base_results_dir, exist_ok=True)
        print(f"Results will be saved to: {base_results_dir}")
    
    # Apply meteorological filtering
    print(f"\n1. APPLYING METEOROLOGICAL FILTERS")
    print(f"=" * 40)
    
    filtered_ds = dataset_with_ice_load.copy()
    filter_info = {}
    
    # Apply time filtering first (months)
    if months is not None:
        print(f"   Filtering by months: {months}")
        time_df = pd.to_datetime(filtered_ds.time.values)
        month_mask = time_df.month.isin(months)
        if month_mask.any():
            filtered_ds = filtered_ds.sel(time=filtered_ds.time[month_mask])
            filter_info['months'] = months
            print(f"   → Time steps after month filtering: {len(filtered_ds.time)}")
        else:
            print(f"   Warning: No data found for specified months")
            return None
    
    # Apply meteorological variable filters
    filter_params = {
        'WD': WD_range,
        'WS': WS_range, 
        'T': T_range,
        'PBLH': PBLH_range,
        'PRECIP': PRECIP_range,
        'QVAPOR': QVAPOR_range,
        'RMOL': RMOL_range
    }
    
    mask = xr.ones_like(filtered_ds.time, dtype=bool)
    
    for param, value_range in filter_params.items():
        if value_range is not None and param in filtered_ds.data_vars:
            min_val, max_val = value_range
            param_data = filtered_ds[param]
            
            # For all parameters, simple range filtering at specified height
            param_mask = (param_data.isel(height=height_level) >= min_val) & (param_data.isel(height=height_level) <= max_val)
            
            # Reduce spatial dimensions - use any() to keep time steps where ANY grid point meets criteria
            param_mask_time = param_mask.any(dim=['south_north', 'west_east'])
            mask = mask & param_mask_time
            filter_info[param] = value_range
            
            # Count remaining data points
            remaining_points = mask.sum().values
            print(f"   {param} filter [{min_val}, {max_val}]: {remaining_points} time steps remaining")
    
    # Apply the combined mask using integer indexing
    if not mask.any():
        print(f"   ERROR: No data remaining after filtering!")
        return None
    
    # Convert boolean mask to integer indices
    time_indices = np.where(mask.values)[0]
    filtered_ds = filtered_ds.isel(time=time_indices)
    print(f"   Final filtered dataset: {len(filtered_ds.time)} time steps")
    
    # Get ice load data after filtering
    ice_load_data = filtered_ds[ice_load_variable].isel(height=height_level)
    
    # Apply percentile filtering if specified
    if percentile is not None:
        print(f"   Applying {percentile}th percentile threshold...")
        percentile_value = float(np.nanpercentile(ice_load_data.values, percentile))
        ice_load_data = ice_load_data.where(ice_load_data <= percentile_value, np.nan)
        filter_info['percentile'] = percentile
        print(f"   → Percentile threshold: {percentile_value:.3f} kg/m")
    
    # Data statistics after filtering
    print(f"\n2. FILTERED DATA STATISTICS")
    print(f"=" * 30)
    
    ice_data_clean = ice_load_data.where(ice_load_data >= ice_load_threshold, np.nan)
    max_ice_load = float(ice_data_clean.max())
    mean_ice_load = float(ice_data_clean.mean())
    
    n_south_north = ice_load_data.sizes['south_north']
    n_west_east = ice_load_data.sizes['west_east']
    n_time = ice_load_data.sizes['time']
    
    print(f"   Grid size: {n_south_north} × {n_west_east}")
    print(f"   Time steps: {n_time}")
    print(f"   Ice load range: {ice_load_threshold:.3f} to {max_ice_load:.3f} kg/m")
    print(f"   Ice load mean: {mean_ice_load:.3f} kg/m")
    
    # Define ice load bins for CDF analysis
    if ice_load_bins is None:
        if max_ice_load > ice_load_threshold:
            ice_load_bins = np.linspace(0.0, max_ice_load, 100)
        else:
            ice_load_bins = np.array([0.0, ice_load_threshold + 0.01])
    
    # Perform CDF analysis for each grid cell
    print(f"\n3. CDF ANALYSIS")
    print(f"=" * 15)
    print(f"   Processing {n_south_north * n_west_east} grid cells...")
    
    cdf_results = {}
    cell_statistics = {}
    
    for i in range(n_south_north):
        for j in range(n_west_east):
            # Extract time series for this grid cell
            cell_data = ice_data_clean.isel(south_north=i, west_east=j)
            cell_values = cell_data.values
            
            # Remove NaN values
            valid_mask = ~np.isnan(cell_values)
            cell_values_clean = cell_values[valid_mask]
            
            if len(cell_values_clean) == 0:
                continue
            
            # Filter values to be >= threshold
            cell_values_filtered = cell_values_clean[cell_values_clean >= ice_load_threshold]
            
            if len(cell_values_filtered) == 0:
                continue
            
            # Calculate CDF
            cdf_values = []
            for ice_threshold in ice_load_bins:
                prob = np.sum(cell_values_filtered <= ice_threshold) / len(cell_values_filtered)
                cdf_values.append(prob)
            
            cdf_values = np.array(cdf_values)
            
            # Store results
            cell_key = f'cell_{i}_{j}'
            cdf_results[cell_key] = {
                'ice_load_bins': ice_load_bins,
                'cdf_values': cdf_values,
                'position': (i, j)
            }
            
            # Calculate statistics
            cell_statistics[cell_key] = {
                'max_ice_load': float(np.max(cell_values_filtered)),
                'mean_ice_load': float(np.mean(cell_values_filtered)),
                'std_ice_load': float(np.std(cell_values_filtered)),
                'median_ice_load': float(np.median(cell_values_filtered)),
                'percentile_95': float(np.percentile(cell_values_filtered, 95)),
                'percentile_99': float(np.percentile(cell_values_filtered, 99)),
                'n_valid_points': len(cell_values_filtered)
            }
    
    print(f"   Processed {len(cdf_results)} valid grid cells")
    
    # Calculate spatial gradients using Earth Mover's Distance
    print(f"\n4. SPATIAL GRADIENT ANALYSIS")
    print(f"=" * 30)
    
    try:
        from scipy.stats import wasserstein_distance
        
        print(f"   Computing Earth Mover's distances between neighboring cells...")
        
        # Initialize gradient matrices
        ew_gradients = np.full((n_south_north, n_west_east-1), np.nan)
        sn_gradients = np.full((n_south_north-1, n_west_east), np.nan)
        combined_gradients = np.full((n_south_north, n_west_east), np.nan)
        
        # Calculate East-West gradients
        for i in range(n_south_north):
            for j in range(n_west_east-1):
                cell1_key = f'cell_{i}_{j}'
                cell2_key = f'cell_{i}_{j+1}'
                
                if cell1_key in cdf_results and cell2_key in cdf_results:
                    cdf1 = cdf_results[cell1_key]['cdf_values']
                    cdf2 = cdf_results[cell2_key]['cdf_values']
                    bins1 = cdf_results[cell1_key]['ice_load_bins']
                    
                    # Calculate Earth Mover's distance using L1 distance between CDFs
                    try:
                        distance = np.trapz(np.abs(cdf1 - cdf2), bins1)
                        ew_gradients[i, j] = distance
                    except:
                        distance = np.mean(np.abs(cdf1 - cdf2))
                        ew_gradients[i, j] = distance
        
        # Calculate South-North gradients
        for i in range(n_south_north-1):
            for j in range(n_west_east):
                cell1_key = f'cell_{i}_{j}'
                cell2_key = f'cell_{i+1}_{j}'
                
                if cell1_key in cdf_results and cell2_key in cdf_results:
                    cdf1 = cdf_results[cell1_key]['cdf_values']
                    cdf2 = cdf_results[cell2_key]['cdf_values']
                    bins1 = cdf_results[cell1_key]['ice_load_bins']
                    
                    try:
                        distance = np.trapz(np.abs(cdf1 - cdf2), bins1)
                        sn_gradients[i, j] = distance
                    except:
                        distance = np.mean(np.abs(cdf1 - cdf2))
                        sn_gradients[i, j] = distance
        
        # Calculate combined gradients (average of all valid neighbor distances)
        for i in range(n_south_north):
            for j in range(n_west_east):
                distances = []
                
                # Check all four neighbors
                if j < n_west_east-1 and not np.isnan(ew_gradients[i, j]):
                    distances.append(ew_gradients[i, j])
                if j > 0 and not np.isnan(ew_gradients[i, j-1]):
                    distances.append(ew_gradients[i, j-1])
                if i < n_south_north-1 and not np.isnan(sn_gradients[i, j]):
                    distances.append(sn_gradients[i, j])
                if i > 0 and not np.isnan(sn_gradients[i-1, j]):
                    distances.append(sn_gradients[i-1, j])
                
                if distances:
                    combined_gradients[i, j] = np.mean(distances)
        
        print(f"   Computed gradients for {np.sum(~np.isnan(ew_gradients))} EW and {np.sum(~np.isnan(sn_gradients))} SN pairs")
        
        # Create spatial gradient plots
        print(f"\n5. CREATING SPATIAL GRADIENT PLOTS")
        print(f"=" * 35)
        
        # ABSOLUTE GRADIENT PLOTS
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: East-West gradients
        im1 = axes[0, 0].imshow(ew_gradients, cmap='viridis', origin='lower', 
                               interpolation='nearest', aspect='auto')
        axes[0, 0].set_title('East-West Gradient\n(CDF Earth Mover\'s Distance)')
        axes[0, 0].set_xlabel('West-East Grid Points')
        axes[0, 0].set_ylabel('South-North Grid Points')
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        cbar1.set_label('Earth Mover\'s Distance (kg/m)')
        
        # Add grid lines
        axes[0, 0].set_xticks(range(n_west_east-1))
        axes[0, 0].set_yticks(range(n_south_north))
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: South-North gradients
        im2 = axes[0, 1].imshow(sn_gradients, cmap='viridis', origin='lower', 
                               interpolation='nearest', aspect='auto')
        axes[0, 1].set_title('South-North Gradient\n(CDF Earth Mover\'s Distance)')
        axes[0, 1].set_xlabel('West-East Grid Points')
        axes[0, 1].set_ylabel('South-North Grid Points')
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        cbar2.set_label('Earth Mover\'s Distance (kg/m)')
        
        axes[0, 1].set_xticks(range(n_west_east))
        axes[0, 1].set_yticks(range(n_south_north-1))
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Combined gradients
        im3 = axes[1, 0].imshow(combined_gradients, cmap='viridis', origin='lower', 
                               interpolation='nearest', aspect='auto')
        axes[1, 0].set_title('Combined Spatial Gradient\n(Average Neighbor Distance)')
        axes[1, 0].set_xlabel('West-East Grid Points')
        axes[1, 0].set_ylabel('South-North Grid Points')
        cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
        cbar3.set_label('Average Earth Mover\'s Distance (kg/m)')
        
        axes[1, 0].set_xticks(range(n_west_east))
        axes[1, 0].set_yticks(range(n_south_north))
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Gradient magnitude (combining EW and SN)
        gradient_magnitude = np.full((n_south_north, n_west_east), np.nan)
        
        for i in range(n_south_north):
            for j in range(n_west_east):
                magnitudes = []
                
                # East-West component
                if j < n_west_east-1 and not np.isnan(ew_gradients[i, j]):
                    magnitudes.append(ew_gradients[i, j]**2)
                if j > 0 and not np.isnan(ew_gradients[i, j-1]):
                    magnitudes.append(ew_gradients[i, j-1]**2)
                
                # South-North component  
                if i < n_south_north-1 and not np.isnan(sn_gradients[i, j]):
                    magnitudes.append(sn_gradients[i, j]**2)
                if i > 0 and not np.isnan(sn_gradients[i-1, j]):
                    magnitudes.append(sn_gradients[i-1, j]**2)
                
                if magnitudes:
                    gradient_magnitude[i, j] = np.sqrt(np.mean(magnitudes))
        
        im4 = axes[1, 1].imshow(gradient_magnitude, cmap='plasma', origin='lower', 
                               interpolation='nearest', aspect='auto')
        axes[1, 1].set_title('Gradient Magnitude\n(RMS of EW and SN)')
        axes[1, 1].set_xlabel('West-East Grid Points')
        axes[1, 1].set_ylabel('South-North Grid Points')
        cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
        cbar4.set_label('RMS Gradient Magnitude')
        
        axes[1, 1].set_xticks(range(n_west_east))
        axes[1, 1].set_yticks(range(n_south_north))
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save absolute spatial gradient plots
        if save_plots:
            # Create gradient filename with filtering information (simplified for file system)
            gradient_filename = "ice_load_cdf_spatial_gradients_absolute.png"
            gradient_path = os.path.join(base_results_dir, gradient_filename)
            plt.savefig(gradient_path, dpi=300, bbox_inches='tight')
            print(f"   Absolute spatial gradient plots saved to: {gradient_path}")
        
        plt.close()  # Close the absolute gradient figure
        
        # DIMENSIONLESS GRADIENT PLOTS
        print(f"   Creating dimensionless gradient plots...")
        
        # Calculate domain means for normalization
        ew_mean = np.nanmean(ew_gradients)
        sn_mean = np.nanmean(sn_gradients)
        combined_mean = np.nanmean(combined_gradients)
        gradient_magnitude_mean = np.nanmean(gradient_magnitude)
        
        # Create dimensionless matrices
        ew_gradients_normalized = ew_gradients / ew_mean if ew_mean > 0 else ew_gradients
        sn_gradients_normalized = sn_gradients / sn_mean if sn_mean > 0 else sn_gradients
        combined_gradients_normalized = combined_gradients / combined_mean if combined_mean > 0 else combined_gradients
        gradient_magnitude_normalized = gradient_magnitude / gradient_magnitude_mean if gradient_magnitude_mean > 0 else gradient_magnitude
        
        print(f"     Normalization factors:")
        print(f"       East-West mean: {ew_mean:.3f} kg/m")
        print(f"       South-North mean: {sn_mean:.3f} kg/m")
        print(f"       Combined mean: {combined_mean:.3f} kg/m")
        print(f"       Gradient magnitude mean: {gradient_magnitude_mean:.3f} kg/m")
        
        # Create the dimensionless spatial gradient plots
        fig_norm, axes_norm = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: East-West gradients (normalized)
        im1_norm = axes_norm[0, 0].imshow(ew_gradients_normalized, cmap='RdBu_r', origin='lower', 
                                         interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
        axes_norm[0, 0].set_title('East-West Gradient\n(Dimensionless, Relative to Domain Mean)')
        axes_norm[0, 0].set_xlabel('West-East Grid Points')
        axes_norm[0, 0].set_ylabel('South-North Grid Points')
        cbar1_norm = plt.colorbar(im1_norm, ax=axes_norm[0, 0], shrink=0.8)
        cbar1_norm.set_label('Gradient / Domain Mean')
        
        # Add grid lines
        axes_norm[0, 0].set_xticks(range(n_west_east-1))
        axes_norm[0, 0].set_yticks(range(n_south_north))
        axes_norm[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: South-North gradients (normalized)
        im2_norm = axes_norm[0, 1].imshow(sn_gradients_normalized, cmap='RdBu_r', origin='lower', 
                                         interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
        axes_norm[0, 1].set_title('South-North Gradient\n(Dimensionless, Relative to Domain Mean)')
        axes_norm[0, 1].set_xlabel('West-East Grid Points')
        axes_norm[0, 1].set_ylabel('South-North Grid Points')
        cbar2_norm = plt.colorbar(im2_norm, ax=axes_norm[0, 1], shrink=0.8)
        cbar2_norm.set_label('Gradient / Domain Mean')
        
        axes_norm[0, 1].set_xticks(range(n_west_east))
        axes_norm[0, 1].set_yticks(range(n_south_north-1))
        axes_norm[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Combined gradients (normalized)
        im3_norm = axes_norm[1, 0].imshow(combined_gradients_normalized, cmap='RdBu_r', origin='lower', 
                                         interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
        axes_norm[1, 0].set_title('Combined Spatial Gradient\n(Dimensionless, Relative to Domain Mean)')
        axes_norm[1, 0].set_xlabel('West-East Grid Points')
        axes_norm[1, 0].set_ylabel('South-North Grid Points')
        cbar3_norm = plt.colorbar(im3_norm, ax=axes_norm[1, 0], shrink=0.8)
        cbar3_norm.set_label('Gradient / Domain Mean')
        
        axes_norm[1, 0].set_xticks(range(n_west_east))
        axes_norm[1, 0].set_yticks(range(n_south_north))
        axes_norm[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Gradient magnitude (normalized)
        im4_norm = axes_norm[1, 1].imshow(gradient_magnitude_normalized, cmap='RdBu_r', origin='lower', 
                                         interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
        axes_norm[1, 1].set_title('Gradient Magnitude\n(Dimensionless, Relative to Domain Mean)')
        axes_norm[1, 1].set_xlabel('West-East Grid Points')
        axes_norm[1, 1].set_ylabel('South-North Grid Points')
        cbar4_norm = plt.colorbar(im4_norm, ax=axes_norm[1, 1], shrink=0.8)
        cbar4_norm.set_label('Gradient / Domain Mean')
        
        axes_norm[1, 1].set_xticks(range(n_west_east))
        axes_norm[1, 1].set_yticks(range(n_south_north))
        axes_norm[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save dimensionless spatial gradient plots
        if save_plots:
            # Create dimensionless gradient filename (simplified for file system)
            dimensionless_filename = "ice_load_cdf_spatial_gradients_dimensionless.png"
            dimensionless_path = os.path.join(base_results_dir, dimensionless_filename)
            plt.savefig(dimensionless_path, dpi=300, bbox_inches='tight')
            print(f"   Dimensionless spatial gradient plots saved to: {dimensionless_path}")
        
        plt.close()  # Close the dimensionless gradient figure
        
        # Store gradient results
        gradient_results = {
            'east_west_gradients': ew_gradients,
            'south_north_gradients': sn_gradients,
            'combined_gradients': combined_gradients,
            'gradient_magnitude': gradient_magnitude,
            'east_west_gradients_normalized': ew_gradients_normalized,
            'south_north_gradients_normalized': sn_gradients_normalized,
            'combined_gradients_normalized': combined_gradients_normalized,
            'gradient_magnitude_normalized': gradient_magnitude_normalized,
            'normalization_factors': {
                'ew_mean': ew_mean,
                'sn_mean': sn_mean,
                'combined_mean': combined_mean,
                'gradient_magnitude_mean': gradient_magnitude_mean
            }
        }
        
        print(f"   ✓ Spatial gradient analysis completed")
        
    except ImportError:
        print("   Warning: scipy not available, skipping Earth Mover's distance calculation")
        gradient_results = None
    except Exception as e:
        print(f"   Error in spatial gradient analysis: {e}")
        gradient_results = None
    
    # Create summary CDF plot
    print(f"\n6. CREATING SUMMARY CDF PLOT")
    print(f"=" * 28)
    
    if cdf_results:
        # Calculate mean CDF curve across all cells
        all_cdf_curves = []
        for cell_data in cdf_results.values():
            all_cdf_curves.append(cell_data['cdf_values'])
        
        mean_cdf = np.mean(all_cdf_curves, axis=0)
        std_cdf = np.std(all_cdf_curves, axis=0)
        
        plt.figure(figsize=(10, 6))
        plot_mask = ice_load_bins >= ice_load_threshold
        
        plot_bins = ice_load_bins[plot_mask]
        plot_mean_cdf = mean_cdf[plot_mask]
        plot_std_cdf = std_cdf[plot_mask]
        
        plt.plot(plot_bins, plot_mean_cdf, 'r-', linewidth=3, label='Mean across all cells')
        plt.fill_between(plot_bins, plot_mean_cdf - plot_std_cdf, 
                       plot_mean_cdf + plot_std_cdf, alpha=0.3, color='red', 
                       label='±1 Standard Deviation')
        
        plt.xlabel('Ice Load (kg/m)')
        plt.ylabel('Cumulative Probability')
        plt.title('Ice Load CDF - Domain Average (Filtered Data)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(left=ice_load_threshold)
        plt.ylim([0, 1])
        
        plt.tight_layout()
        
        if save_plots:
            # Create summary filename (simplified for file system)
            summary_filename = "ice_load_cdf_summary_filtered.png"
            summary_path = os.path.join(base_results_dir, summary_filename)
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            print(f"   Summary CDF plot saved to: {summary_path}")
        
        plt.close()
    
    # Save analysis summary file with all filter details
    if save_plots:
        summary_file_path = os.path.join(base_results_dir, "analysis_summary.txt")
        with open(summary_file_path, 'w') as f:
            f.write("ICE LOAD ANALYSIS WITH FILTERING AND CDF\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis timestamp: {pd.Timestamp.now()}\n")
            f.write(f"Ice load variable: {ice_load_variable}\n")
            f.write(f"Height level: {height_level} ({dataset_with_ice_load.height.values[height_level]} m)\n\n")
            
            f.write("FILTERS APPLIED:\n")
            f.write("-" * 20 + "\n")
            if not filter_info:
                f.write("No meteorological filters applied\n")
            else:
                for param, value_range in filter_info.items():
                    if param == 'months':
                        f.write(f"Months: {value_range}\n")
                    elif param == 'percentile':
                        f.write(f"Percentile threshold: {value_range}%\n")
                    else:
                        f.write(f"{param}: {value_range[0]} to {value_range[1]}\n")
            
            if ice_load_threshold > 0:
                f.write(f"Ice load threshold: {ice_load_threshold} kg/m\n")
            
            f.write(f"\nDATA STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Grid size: {n_south_north} × {n_west_east}\n")
            f.write(f"Time steps after filtering: {n_time}\n")
            f.write(f"Valid grid cells: {len(cdf_results)}\n")
            f.write(f"Ice load range: {ice_load_threshold:.3f} to {max_ice_load:.3f} kg/m\n")
            f.write(f"Ice load mean: {mean_ice_load:.3f} kg/m\n")
            
            f.write(f"\nFILES GENERATED:\n")
            f.write("-" * 20 + "\n")
            f.write("- ice_load_cdf_spatial_gradients_absolute.png\n")
            f.write("- ice_load_cdf_spatial_gradients_dimensionless.png\n")
            f.write("- ice_load_cdf_summary_filtered.png\n")
            f.write("- analysis_summary.txt (this file)\n")
            
            if gradient_results:
                f.write(f"\nSPATIAL GRADIENT STATISTICS:\n")
                f.write("-" * 30 + "\n")
                norms = gradient_results['normalization_factors']
                f.write(f"East-West gradient mean: {norms['ew_mean']:.3f} kg/m\n")
                f.write(f"South-North gradient mean: {norms['sn_mean']:.3f} kg/m\n")
                f.write(f"Combined gradient mean: {norms['combined_mean']:.3f} kg/m\n")
                f.write(f"Gradient magnitude mean: {norms['gradient_magnitude_mean']:.3f} kg/m\n")
        
        print(f"   Analysis summary saved to: {summary_file_path}")
    
    # Compile final results
    results = {
        'filter_info': filter_info,
        'data_statistics': {
            'n_time_steps': n_time,
            'n_grid_cells': n_south_north * n_west_east,
            'n_valid_cells': len(cdf_results),
            'max_ice_load': max_ice_load,
            'mean_ice_load': mean_ice_load,
            'ice_load_threshold': ice_load_threshold
        },
        'cdf_results': cdf_results,
        'cell_statistics': cell_statistics,
        'spatial_gradients': gradient_results,
        'ice_load_bins': ice_load_bins
    }
    
    print(f"\n=== ANALYSIS COMPLETED SUCCESSFULLY ===")
    print(f"   Filters applied: {len(filter_info)}")
    print(f"   Valid grid cells: {len(cdf_results)}")
    print(f"   Spatial gradients: {'✓' if gradient_results else '✗'}")
    
    return results

# TEMPORAL GRADIENTS

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


# EMD DATA IMPORT


def import_emd_data(file_path):
    """
    Import EMD text data with metadata extraction and proper formatting
    
    Parameters:
    -----------
    file_path : str
        Path to the EMD text file
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with time as index, meteorological variables as columns,
        and metadata stored in .attrs['metadata']
    
    Expected file format:
    --------------------
    - Metadata lines with format "key: value" (e.g., "Orography: -0.002")
    - Column headers line starting with "time"
    - Time-series data with tab-separated values
    
    Example:
    --------
    >>> emd_df = import_emd_data('data/EMD_data/station_data.txt')
    >>> print(emd_df.shape)
    >>> print(emd_df.attrs['metadata'])
    >>> wind_speed = emd_df['wSpeed.10']  # Wind speed at 10m
    """
    
    metadata = {}
    data_start_line = 0
    
    try:
        # Read file and extract metadata
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse metadata section
        for i, line in enumerate(lines):
            line = line.strip()
            if ':' in line and not line.startswith('time'):
                # Extract metadata (Orography, Latitude, Longitude, etc.)
                key, value = line.split(':', 1)
                try:
                    metadata[key.strip()] = float(value.strip())
                except ValueError:
                    metadata[key.strip()] = value.strip()
            elif line.startswith('time'):
                # Found the header line
                data_start_line = i
                break
        
        # Read the data starting from header line
        try:
            # Try tab-separated first (most common for EMD data)
            df = pd.read_csv(file_path, delimiter='\t', skiprows=data_start_line, 
                            parse_dates=[0], index_col=0)
            
            # If tab doesn't work well, try other delimiters
            if len(df.columns) < 10:  # Expect many columns for EMD data
                for delimiter in [' ', ',', ';']:
                    try:
                        df = pd.read_csv(file_path, delimiter=delimiter, skiprows=data_start_line,
                                       parse_dates=[0], index_col=0)
                        if len(df.columns) >= 10:
                            break
                    except:
                        continue
            
            # Add metadata as attributes
            df.attrs['metadata'] = metadata
            
            print(f"Successfully imported EMD data from: {file_path}")
            print(f"- Data shape: {df.shape}")
            print(f"- Time range: {df.index[0]} to {df.index[-1]}")
            print(f"- Number of variables: {len(df.columns)}")
            if metadata:
                print(f"- Metadata extracted: {list(metadata.keys())}")
            print(f"- Sample columns: {list(df.columns[:10])}")
            
            return df
            
        except Exception as e:
            print(f"Error parsing EMD data: {e}")
            print("Attempting fallback import methods...")
            
            # Fallback method 1: Basic tab-separated
            try:
                df = pd.read_csv(file_path, delimiter='\t', skiprows=data_start_line)
                # Manually set time column as index
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                df.attrs['metadata'] = metadata
                print("Fallback import successful (tab-separated)")
                return df
            except Exception as e2:
                print(f"Fallback method 1 failed: {e2}")
            
            # Fallback method 2: Space-separated
            try:
                df = pd.read_csv(file_path, delimiter=' ', skiprows=data_start_line)
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                df.attrs['metadata'] = metadata
                print("Fallback import successful (space-separated)")
                return df
            except Exception as e3:
                print(f"Fallback method 2 failed: {e3}")
                raise Exception(f"All import methods failed. Original error: {e}")
                
    except Exception as e:
        print(f"Error reading EMD file: {e}")
        raise

def explore_emd_data(emd_df):
    """
    Explore and summarize EMD DataFrame structure and content
    
    Parameters:
    -----------
    emd_df : pandas.DataFrame
        EMD DataFrame from import_emd_data()
    """
    
    print("\n=== EMD Data Summary ===")
    print(f"Data shape: {emd_df.shape}")
    print(f"Time range: {emd_df.index[0]} to {emd_df.index[-1]}")
    print(f"Time frequency: {emd_df.index[1] - emd_df.index[0]}")
    
    # Metadata
    if hasattr(emd_df, 'attrs') and 'metadata' in emd_df.attrs:
        print(f"\nMetadata:")
        for key, value in emd_df.attrs['metadata'].items():
            print(f"  {key}: {value}")
    
    # Variable categories
    wind_vars = [col for col in emd_df.columns if 'wSpeed' in col or 'wDir' in col]
    temp_vars = [col for col in emd_df.columns if 'temp' in col]
    pressure_vars = [col for col in emd_df.columns if 'press' in col]
    humidity_vars = [col for col in emd_df.columns if 'rh' in col]
    ice_vars = [col for col in emd_df.columns if 'ice' in col.lower() or 'Ice' in col]
    
    print(f"\nVariable categories:")
    print(f"  Wind variables: {len(wind_vars)} (e.g., {wind_vars[:3]})")
    print(f"  Temperature variables: {len(temp_vars)} (e.g., {temp_vars[:3]})")
    print(f"  Pressure variables: {len(pressure_vars)} (e.g., {pressure_vars[:3]})")
    print(f"  Humidity variables: {len(humidity_vars)} (e.g., {humidity_vars[:3]})")
    print(f"  Ice-related variables: {len(ice_vars)} (e.g., {ice_vars[:3]})")
    
    # Height levels
    height_levels = set()
    for col in emd_df.columns:
        if '.' in col:
            try:
                height = col.split('.')[-1]
                if height.replace('hpa', '').replace('mb', '').replace('cm', '').isdigit():
                    height_levels.add(height)
            except:
                pass
    
    if height_levels:
        print(f"\nHeight levels detected: {sorted(height_levels)}")
    
    # Basic statistics
    print(f"\nData quality:")
    print(f"  Missing values: {emd_df.isnull().sum().sum()} total")
    print(f"  Complete time series: {emd_df.isnull().sum() == 0}.sum() variables")







