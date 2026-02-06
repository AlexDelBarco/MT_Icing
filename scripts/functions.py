


import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import metpy.calc as mpcalc
from metpy.units import units

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
        # print(f"Height levels in dataset: {ds.height.values}")
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

def merge_netcdf_files2(main_file_path, additional_file_path, additional2_file_path, additional3_file_path, output_file_path, verbose=True):
    """
    Merge four NetCDF files by combining their data variables.
    The main file provides the structure and most variables, while additional variables
    are added from the additional, additional2, and additional3 files.
    
    Parameters:
    -----------
    main_file_path : str
        Path to the main NetCDF file (contains most variables)
    additional_file_path : str
        Path to the first additional NetCDF file (contains variables to be added)
    additional2_file_path : str
        Path to the second additional NetCDF file (contains variables to be added)
    additional3_file_path : str
        Path to the third additional NetCDF file (contains variables to be added)
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
    >>> success = merge_netcdf_files2(
    ...     "data/newa_wrf_for_jana_mstudent_extended.nc",
    ...     "data/newa_wrf_for_jana_mstudent_extended_WD.nc", 
    ...     "data/newa_wrf_for_jana_mstudent_extended_PSFC_SEAICE_SWDDNI.nc",
    ...     "data/additional_file3.nc",
    ...     "data/newa_wrf_for_jana_mstudent_extended_merged.nc"
    ... )
    """
    
    if verbose:
        print("=== NETCDF FILE MERGING (4 FILES) ===")
        print(f"Main file: {main_file_path}")
        print(f"Additional file: {additional_file_path}")
        print(f"Additional2 file: {additional2_file_path}")
        print(f"Additional3 file: {additional3_file_path}")
        print(f"Output file: {output_file_path}")
    
    try:
        import xarray as xr
        import os
        
        # Load all four datasets with chunks for memory efficiency
        if verbose:
            print(f"\n1. Loading datasets...")
        
        main_ds = xr.open_dataset(main_file_path, chunks='auto')
        additional_ds = xr.open_dataset(additional_file_path, chunks='auto')
        additional2_ds = xr.open_dataset(additional2_file_path, chunks='auto')
        additional3_ds = xr.open_dataset(additional3_file_path, chunks='auto')
        
        if verbose:
            print(f"   Main dataset variables: {list(main_ds.data_vars.keys())}")
            print(f"   Additional dataset variables: {list(additional_ds.data_vars.keys())}")
            print(f"   Additional2 dataset variables: {list(additional2_ds.data_vars.keys())}")
            print(f"   Additional3 dataset variables: {list(additional3_ds.data_vars.keys())}")
            print(f"   Main dataset shape: {main_ds.dims}")
            print(f"   Additional dataset shape: {additional_ds.dims}")
            print(f"   Additional2 dataset shape: {additional2_ds.dims}")
            print(f"   Additional3 dataset shape: {additional3_ds.dims}")
        
        # Check if dimensions are compatible
        if verbose:
            print(f"\n2. Checking dimension compatibility...")
        
        # Get dimensions that matter for data variables (excluding potential differences in coords)
        main_dims = dict(main_ds.dims)
        additional_dims = dict(additional_ds.dims)
        additional2_dims = dict(additional2_ds.dims)
        additional3_dims = dict(additional3_ds.dims)
        
        # Check critical dimensions
        critical_dims = ['time', 'south_north', 'west_east']
        if 'height' in main_dims:
            critical_dims.append('height')
            
        dimension_compatible = True
        for dim in critical_dims:
            dims_to_check = []
            if dim in main_dims:
                dims_to_check.append(('main', main_dims[dim]))
            if dim in additional_dims:
                dims_to_check.append(('additional', additional_dims[dim]))
            if dim in additional2_dims:
                dims_to_check.append(('additional2', additional2_dims[dim]))
            if dim in additional3_dims:
                dims_to_check.append(('additional3', additional3_dims[dim]))
            
            if len(dims_to_check) > 1:
                # Check if all dimensions are the same
                sizes = [size for _, size in dims_to_check]
                if not all(size == sizes[0] for size in sizes):
                    print(f"   Warning: Dimension mismatch for '{dim}': {dims_to_check}")
                    dimension_compatible = False
                else:
                    if verbose:
                        print(f"   ✓ {dim}: {sizes[0]} (compatible across all files)")
            elif dims_to_check:
                if verbose:
                    file_name, size = dims_to_check[0]
                    print(f"   ✓ {dim}: {size} (only in {file_name} file)")
        
        if not dimension_compatible:
            print("   Error: Incompatible dimensions between files")
            return False
        
        # Check for variable conflicts
        if verbose:
            print(f"\n3. Checking for variable conflicts...")
        
        main_vars = set(main_ds.data_vars.keys())
        additional_vars = set(additional_ds.data_vars.keys())
        additional2_vars = set(additional2_ds.data_vars.keys())
        additional3_vars = set(additional3_ds.data_vars.keys())
        
        conflicts_main_additional = main_vars.intersection(additional_vars)
        conflicts_main_additional2 = main_vars.intersection(additional2_vars)
        conflicts_main_additional3 = main_vars.intersection(additional3_vars)
        conflicts_additional_additional2 = additional_vars.intersection(additional2_vars)
        conflicts_additional_additional3 = additional_vars.intersection(additional3_vars)
        conflicts_additional2_additional3 = additional2_vars.intersection(additional3_vars)
        
        all_conflicts = (conflicts_main_additional.union(conflicts_main_additional2)
                        .union(conflicts_main_additional3)
                        .union(conflicts_additional_additional2)
                        .union(conflicts_additional_additional3)
                        .union(conflicts_additional2_additional3))
        
        if all_conflicts:
            print(f"   Warning: Variable conflicts found: {all_conflicts}")
            print(f"   Priority: main file > additional file > additional2 file > additional3 file")
            print(f"   Conflicting variables will be kept from the highest priority file")
        
        # Determine which variables to add from each file
        new_vars_additional = additional_vars - main_vars
        new_vars_additional2 = additional2_vars - main_vars - additional_vars
        new_vars_additional3 = additional3_vars - main_vars - additional_vars - additional2_vars
        
        if verbose:
            print(f"   Variables to be added from additional file: {list(new_vars_additional)}")
            print(f"   Variables to be added from additional2 file: {list(new_vars_additional2)}")
            print(f"   Variables to be added from additional3 file: {list(new_vars_additional3)}")
        
        # Merge datasets
        if verbose:
            print(f"\n4. Merging datasets...")
        
        # Start with the main dataset
        merged_ds = main_ds.copy(deep=True)
        
        # Add new variables from additional dataset
        for var_name in new_vars_additional:
            if verbose:
                print(f"   Adding variable from additional file: {var_name}")
            
            # Get the variable from additional dataset
            var_data = additional_ds[var_name]
            
            # Add to merged dataset
            merged_ds[var_name] = var_data
        
        # Add new variables from additional2 dataset
        for var_name in new_vars_additional2:
            if verbose:
                print(f"   Adding variable from additional2 file: {var_name}")
            
            # Get the variable from additional2 dataset
            var_data = additional2_ds[var_name]
            
            # Add to merged dataset
            merged_ds[var_name] = var_data
        
        # Add new variables from additional3 dataset
        for var_name in new_vars_additional3:
            if verbose:
                print(f"   Adding variable from additional3 file: {var_name}")
            
            # Get the variable from additional3 dataset
            var_data = additional3_ds[var_name]
            
            # Add to merged dataset
            merged_ds[var_name] = var_data
        
        # Verify the merge
        if verbose:
            print(f"\n5. Verifying merged dataset...")
            print(f"   Original main variables: {len(main_vars)}")
            print(f"   Added from additional file: {len(new_vars_additional)}")
            print(f"   Added from additional2 file: {len(new_vars_additional2)}")
            print(f"   Added from additional3 file: {len(new_vars_additional3)}")
            print(f"   Total variables in merged dataset: {len(merged_ds.data_vars)}")
            print(f"   Final variables: {list(merged_ds.data_vars.keys())}")
        
        # Save the merged dataset
        if verbose:
            print(f"\n6. Saving merged dataset...")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save with compression to reduce file size - memory efficient approach
        encoding = {}
        for var in merged_ds.data_vars:
            if merged_ds[var].dtype in ['float32', 'float64']:
                encoding[var] = {
                    'zlib': True, 
                    'complevel': 4,
                    'chunksizes': None  # Let xarray decide optimal chunking
                }
        
        # Use memory-efficient saving approach
        try:
            merged_ds.to_netcdf(output_file_path, encoding=encoding)
        except (MemoryError, Exception) as e:
            if "allocate" in str(e).lower() or "memory" in str(e).lower():
                if verbose:
                    print(f"   Memory error encountered, trying alternative save method...")
                
                # Alternative approach: save without compression first
                encoding_no_compression = {}
                merged_ds.to_netcdf(output_file_path, encoding=encoding_no_compression)
            else:
                raise e
        
        if verbose:
            file_size_mb = os.path.getsize(output_file_path) / (1024 * 1024)
            print(f"   Merged file saved: {output_file_path}")
            print(f"   File size: {file_size_mb:.1f} MB")
        
        # Clean up
        main_ds.close()
        additional_ds.close()
        additional2_ds.close()
        additional3_ds.close()
        merged_ds.close()
        
        if verbose:
            print(f"\n✓ NetCDF file merging (4 files) completed successfully!")
        
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
        gl.xlabel_style = {'size': 30, 'color': 'black'}
        gl.ylabel_style = {'size': 30, 'color': 'black'}
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98),
                 fancybox=True, shadow=True, fontsize=30)
        
        # Set title
        ax.set_title(title, fontsize=28, weight='bold', pad=20)
        
        # Add margin information
        margin_text = f"Margin: {margin_degrees}° | Zoom: {zoom_level} | Terrain: Stamen"
        ax.text(0.02, 0.02, margin_text, transform=ax.transAxes, fontsize=30,
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
            # Create grid coordinates for consistent orientation
            x_coords = np.arange(landmask_values.shape[1])  # west_east dimension
            y_coords = np.arange(landmask_values.shape[0])  # south_north dimension
            
            im1 = ax1.pcolormesh(x_coords, y_coords, landmask_values, cmap='RdYlBu_r', shading='auto')
            ax1.set_title('LANDMASK - Spatial Distribution', fontsize=28)
            
            # Set custom tick labels for grid points (1-14)
            ax1.set_xticks(range(len(x_coords)))
            ax1.set_xticklabels(range(1, len(x_coords)+1))
            ax1.set_yticks(range(len(y_coords)))
            ax1.set_yticklabels(range(1, len(y_coords)+1))
            ax1.set_xlabel('West-East Grid Points', fontsize=24)
            ax1.set_ylabel('South-North Grid Points', fontsize=24)
            
            # Add grid cell values as text
            for i in range(landmask.shape[0]):
                for j in range(landmask.shape[1]):
                    value = landmask_values[i, j]
                    color = 'white' if value < 0.5 else 'black'
                    ax1.text(j, i, f'{int(value)}', ha='center', va='center', 
                            color=color, fontsize=24, weight='bold')
            
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
            ax2.set_title('Land vs Water Distribution', fontsize=28)
            ax2.set_ylabel('Number of Grid Cells', fontsize=24)
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

def accreation_per_winter(ds, start_date, end_date, height_level=0, OffOn=None, BigDomain=False,
                          margin_degrees=0.5, zoom_level=6):
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
    OffOn : str, optional
        Specifies 'Onshore' or 'Offshore' for BigDomain directory structure
    BigDomain : bool, default False
        If True, saves results to results/figures/BigDomain/{OffOn}/ice_accretion/
    margin_degrees : float, default 0.5
        Margin around grid in degrees for cartopy terrain map
    zoom_level : int, default 6
        Zoom level for terrain tiles in cartopy terrain map
    
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
    # Use BigDomain structure if requested
    if BigDomain and OffOn:
        ice_accretion_dir = os.path.join(figures_dir, "BigDomain", OffOn, "ice_accretion")
    else:
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
        
        # Debug: Print data structure information
        print(f"Plot data dimensions: {plot_data.dims}")
        print(f"Plot data shape: {plot_data.shape}")
        if len(plot_data.shape) > 1:
            sample_winter = plot_data.isel(winterno=0)
            print(f"Sample winter data dimensions: {sample_winter.dims}")
            print(f"Sample winter data shape: {sample_winter.shape}")
            print(f"Sample winter coordinate names: {list(sample_winter.coords.keys())}")
                
        # Create one plot for each winter
        for i, winter_idx in enumerate(plot_data.winterno.values):
            plt.figure(figsize=(10, 6))
            
            winter_data = plot_data.isel(winterno=i)
            
            # Debug: Print individual winter data info
            print(f"Winter {int(winter_idx)} data dimensions: {winter_data.dims}")
            print(f"Winter {int(winter_idx)} data shape: {winter_data.shape}")
            
            # Create grid coordinates (0-based for plotting, but will label as 1-based)
            # Check dimension order and assign coordinates correctly
            if 'south_north' in winter_data.dims and 'west_east' in winter_data.dims:
                # Get the dimension order
                dims_order = winter_data.dims
                south_north_idx = dims_order.index('south_north')
                west_east_idx = dims_order.index('west_east')
                
                print(f"Dimension order: {dims_order}")
                print(f"south_north is at index {south_north_idx}, west_east is at index {west_east_idx}")
                
                # Assign coordinates based on actual dimension positions
                if west_east_idx == 1 and south_north_idx == 0:  # (south_north, west_east)
                    x_coords = np.arange(winter_data.shape[1])  # west_east dimension
                    y_coords = np.arange(winter_data.shape[0])  # south_north dimension
                elif west_east_idx == 0 and south_north_idx == 1:  # (west_east, south_north)
                    x_coords = np.arange(winter_data.shape[0])  # west_east dimension  
                    y_coords = np.arange(winter_data.shape[1])  # south_north dimension
                    # Need to transpose data for proper orientation
                    winter_data = winter_data.T
                else:
                    print("Warning: Unexpected dimension order!")
                    x_coords = np.arange(winter_data.shape[1])
                    y_coords = np.arange(winter_data.shape[0])
            else:
                print("Warning: Cannot find south_north and west_east dimensions!")
                x_coords = np.arange(winter_data.shape[1])
                y_coords = np.arange(winter_data.shape[0])
            
            # Use pcolormesh with simple grid coordinates
            im = plt.pcolormesh(x_coords, y_coords, winter_data.values, shading='auto')
            plt.colorbar(im, label='Ice Accretion Sum (kg/m)')
            
            # Set custom tick labels for grid points (1-14)
            plt.xticks(range(len(x_coords)), range(1, len(x_coords)+1))
            plt.yticks(range(len(y_coords)), range(1, len(y_coords)+1))
            plt.xlabel('West-East Grid Points', fontsize=24)
            plt.ylabel('South-North Grid Points', fontsize=24)
            
            winter_start = dates[int(winter_idx)] if int(winter_idx) < len(dates)-1 else "N/A"
            winter_end = dates[int(winter_idx)+1] - pd.to_timedelta('30min') if int(winter_idx)+1 < len(dates) else "N/A"
            
            # Include height information in the title
            plt.title(f'Ice Accretion Sum for Winter starting on: {winter_start} and ending on {winter_end}\nHeight: {height_value} {height_units}', fontsize=28)
            plt.tight_layout()
            
            # Include height information in the filename
            filename = os.path.join(ice_accretion_dir, f"ice_accretion_winter_{int(winter_idx)}_{height_label}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()  # Close figure to free memory
        
        # Create final plot showing mean across all winters for each grid cell
        print("Creating mean ice accretion plot across all winters...")
        plt.figure(figsize=(10, 6))
        mean_accretion = plot_data.mean(dim='winterno')
        
        print(f"Mean data dimensions: {mean_accretion.dims}")
        print(f"Mean data shape: {mean_accretion.shape}")
        
        # Create grid coordinates (0-based for plotting, but will label as 1-based)
        # Check dimension order and assign coordinates correctly
        if 'south_north' in mean_accretion.dims and 'west_east' in mean_accretion.dims:
            # Get the dimension order
            dims_order = mean_accretion.dims
            south_north_idx = dims_order.index('south_north')
            west_east_idx = dims_order.index('west_east')
            
            print(f"Mean plot dimension order: {dims_order}")
            print(f"south_north is at index {south_north_idx}, west_east is at index {west_east_idx}")
            
            # Assign coordinates based on actual dimension positions
            if west_east_idx == 1 and south_north_idx == 0:  # (south_north, west_east)
                x_coords = np.arange(mean_accretion.shape[1])  # west_east dimension
                y_coords = np.arange(mean_accretion.shape[0])  # south_north dimension
            elif west_east_idx == 0 and south_north_idx == 1:  # (west_east, south_north)
                x_coords = np.arange(mean_accretion.shape[0])  # west_east dimension  
                y_coords = np.arange(mean_accretion.shape[1])  # south_north dimension
                # Need to transpose data for proper orientation
                mean_accretion = mean_accretion.T
            else:
                print("Warning: Unexpected dimension order!")
                x_coords = np.arange(mean_accretion.shape[1])
                y_coords = np.arange(mean_accretion.shape[0])
        else:
            print("Warning: Cannot find south_north and west_east dimensions!")
            x_coords = np.arange(mean_accretion.shape[1])
            y_coords = np.arange(mean_accretion.shape[0])
        
        # Use pcolormesh with simple grid coordinates
        im = plt.pcolormesh(x_coords, y_coords, mean_accretion.values, shading='auto')
        plt.colorbar(im, label='Mean Ice Accretion Sum (kg/m)')
        
        # Set custom tick labels for grid points (1-14)
        plt.xticks(range(len(x_coords)), range(1, len(x_coords)+1))
        plt.yticks(range(len(y_coords)), range(1, len(y_coords)+1))
        plt.xlabel('West-East Grid Points', fontsize=24)
        plt.ylabel('South-North Grid Points', fontsize=24)
        
        # Calculate year range for the title
        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year
        num_winters = len(plot_data.winterno.values)
        
        plt.title(f'Mean Ice Accretion Sum Across All Winters ({start_year}-{end_year})\n'
                 f'Height: {height_value} {height_units}, Number of winters: {num_winters}', fontsize=28)
        plt.tight_layout()
        
        # Save mean plot
        mean_filename = os.path.join(ice_accretion_dir, f"ice_accretion_mean_all_winters_{height_label}.png")
        plt.savefig(mean_filename, dpi=300, bbox_inches='tight')
        print(f"Saved mean accretion plot: {mean_filename}")
        plt.close()  # Close figure to free memory
        
        # Create cartopy terrain map with mean ice accretion
        print(f"\nCreating cartopy terrain map with mean ice accretion...")
        
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            import cartopy.io.img_tiles as cimgt
            
            # Get geographical coordinates
            if 'XLAT' in ds.coords and 'XLON' in ds.coords:
                lats = ds.coords['XLAT'].values
                lons = ds.coords['XLON'].values
            elif 'XLAT' in ds.data_vars and 'XLON' in ds.data_vars:
                lats = ds['XLAT'].values
                lons = ds['XLON'].values
            else:
                raise ValueError("No latitude/longitude coordinates found in dataset")
            
            print(f"   Grid coordinates: Lat {lats.min():.3f} to {lats.max():.3f}, Lon {lons.min():.3f} to {lons.max():.3f}")
            
            # Calculate grid cell edges for pcolormesh
            lon_edges = np.zeros(lons.shape[1] + 1)
            lat_edges = np.zeros(lats.shape[0] + 1)
            
            # Longitude edges
            for j in range(lons.shape[1]):
                if j == 0:
                    lon_edges[j] = lons[0, j] - (lons[0, 1] - lons[0, 0]) / 2
                else:
                    lon_edges[j] = (lons[0, j-1] + lons[0, j]) / 2
            lon_edges[-1] = lons[0, -1] + (lons[0, -1] - lons[0, -2]) / 2
            
            # Latitude edges
            for i in range(lats.shape[0]):
                if i == 0:
                    lat_edges[i] = lats[i, 0] - (lats[1, 0] - lats[0, 0]) / 2
                else:
                    lat_edges[i] = (lats[i-1, 0] + lats[i, 0]) / 2
            lat_edges[-1] = lats[-1, 0] + (lats[-1, 0] - lats[-2, 0]) / 2
            
            # Create figure with cartopy projection
            fig = plt.figure(figsize=(16, 12))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            # Set extent with margin
            grid_center_lon = (lon_edges.min() + lon_edges.max()) / 2
            grid_center_lat = (lat_edges.min() + lat_edges.max()) / 2
            grid_span_lon = lon_edges.max() - lon_edges.min()
            grid_span_lat = lat_edges.max() - lat_edges.min()
            extent_span_lon = grid_span_lon + 2 * margin_degrees
            extent_span_lat = grid_span_lat + 2 * margin_degrees
            
            west = grid_center_lon - extent_span_lon / 2
            east = grid_center_lon + extent_span_lon / 2
            south = grid_center_lat - extent_span_lat / 2
            north = grid_center_lat + extent_span_lat / 2
            
            ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
            
            # Add geographical features
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.7, zorder=1)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.8, zorder=2)
            ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.8, zorder=3)
            
            # Try to add terrain background
            try:
                terrain = cimgt.OSM()
                ax.add_image(terrain, zoom_level)
                print(f"   Successfully loaded OpenStreetMap tiles")
            except Exception as e:
                try:
                    terrain = cimgt.GoogleTiles(style='satellite')
                    ax.add_image(terrain, zoom_level)
                    print(f"   Successfully loaded Google satellite tiles")
                except Exception as e2:
                    print(f"   Tile services unavailable, using basic land/ocean features")
            
            # Add geographical features on top
            ax.add_feature(cfeature.BORDERS, linewidth=1.5, color='black', alpha=0.8, zorder=8)
            ax.add_feature(cfeature.COASTLINE, linewidth=2, color='black', alpha=0.9, zorder=9)
            
            # Create meshgrid for pcolormesh
            lon_mesh, lat_mesh = np.meshgrid(lon_edges, lat_edges)
            
            # Calculate statistics for better color scaling
            mean_accretion_values = mean_accretion.values
            valid_accretion = mean_accretion_values[~np.isnan(mean_accretion_values)]
            
            if len(valid_accretion) > 0:
                data_min = np.min(valid_accretion)
                data_max = np.max(valid_accretion)
                data_mean = np.mean(valid_accretion)
                data_90p = np.percentile(valid_accretion, 90)
                
                print(f"   Mean ice accretion statistics:")
                print(f"     Min: {data_min:.2f}, Max: {data_max:.2f}, Mean: {data_mean:.2f} kg/m")
                print(f"     90th percentile: {data_90p:.2f} kg/m")
                
                # Apply automatic scaling with outlier detection
                outlier_ratio = data_max / data_90p if data_90p > 0 else 1
                if outlier_ratio > 2.0:
                    vmin = data_min
                    vmax = data_90p
                    outlier_clipped = True
                    print(f"   Using 90th percentile clipping for better visualization: {vmin:.2f} - {vmax:.2f} kg/m")
                else:
                    vmin = data_min
                    vmax = data_max
                    outlier_clipped = False
                    print(f"   Using full data range: {vmin:.2f} - {vmax:.2f} kg/m")
            else:
                vmin, vmax = 0, 1
                outlier_clipped = False
            
            # Plot mean ice accretion as semi-transparent overlay
            accretion_plot = ax.pcolormesh(
                lon_mesh, lat_mesh, mean_accretion_values,
                cmap='viridis', alpha=0.8,
                vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree(),
                zorder=7
            )
            
            # Add colorbar
            cbar = plt.colorbar(accretion_plot, ax=ax, shrink=0.8, pad=0.02)
            if outlier_clipped:
                cbar_label = f'Mean Ice Accretion Sum (kg/m)\n[Clipped at 90th percentile: {vmax:.2f}]'
            else:
                cbar_label = f'Mean Ice Accretion Sum (kg/m)'
            cbar.set_label(cbar_label, fontsize=24)
            cbar.ax.tick_params(labelsize=30)
            
            # Add gridlines with labels
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 30, 'color': 'black'}
            gl.ylabel_style = {'size': 30, 'color': 'black'}
            
            # Add title
            title_text = (f'Mean Ice Accretion Across All Winters on Terrain Map\n'
                         f'Period: {start_year}-{end_year}, '
                         f'Height: {height_value} {height_units}, Number of winters: {num_winters}')
            ax.set_title(title_text, fontsize=28, weight='bold', pad=20)
            
            # Add statistics information
            if len(valid_accretion) > 0:
                if outlier_clipped:
                    info_text = (f"Range: {data_min:.2f} - {data_max:.2f} kg/m | "
                                f"Mean: {data_mean:.2f} kg/m\n"
                                f"Color scale: {vmin:.2f} - {vmax:.2f} kg/m (90th percentile clipped)")
                else:
                    info_text = (f"Range: {data_min:.2f} - {data_max:.2f} kg/m | "
                                f"Mean: {data_mean:.2f} kg/m")
                ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=27,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            
            plt.tight_layout()
            
            # Save the cartopy terrain map
            cartopy_filename = os.path.join(ice_accretion_dir, f"ice_accretion_mean_cartopy_{height_label}.png")
            plt.savefig(cartopy_filename, dpi=300, bbox_inches='tight')
            print(f"   Cartopy terrain map saved to: {cartopy_filename}")
            plt.close()  # Close the plot to prevent it from showing
            
        except ImportError as e:
            print(f"   Warning: Cartopy not available, skipping terrain map: {e}")
        except Exception as e:
            print(f"   Error creating cartopy terrain map: {e}")
            
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
    
    # Create grid coordinates for consistent orientation
    x_coords = np.arange(max_ice_load.shape[1])  # west_east dimension
    y_coords = np.arange(max_ice_load.shape[0])  # south_north dimension
    
    im = plt.pcolormesh(x_coords, y_coords, max_ice_load.values, cmap='Blues', shading='auto')
    plt.colorbar(im, label='Maximum Ice Load (kg/m)')
    plt.title(f'Maximum Ice Load Over All Winters\nHeight: {height_value} {height_units}', fontsize=28)
    
    # Set custom tick labels for grid points (1-14)
    plt.xticks(range(len(x_coords)), range(1, len(x_coords)+1))
    plt.yticks(range(len(y_coords)), range(1, len(y_coords)+1))
    plt.xlabel('West-East Grid Points', fontsize=24)
    plt.ylabel('South-North Grid Points', fontsize=24)
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
    plt.title(f'Average Ice Load Over Time (All Grid Points)\nHeight: {height_value} {height_units}', fontsize=28)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Average Ice Load (kg/m)', fontsize=24)
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
    plt.xlabel('Ice Load (kg/m)', fontsize=24)
    plt.ylabel('Frequency', fontsize=24)
    plt.title(f'Distribution of Ice Load Values (Non-zero)\nHeight: {height_value} {height_units}', fontsize=28)
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
        
        plt.title(f'Ice Load Comparison Across Winters\nHeight: {height_value} {height_units}', fontsize=28)
        plt.xlabel('Time', fontsize=24)
        plt.ylabel('Average Ice Load (kg/m)', fontsize=24)
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

def add_ice_load_to_dataset(ds, dates, OffOn, method=5, height_level=0, variable_name='ICE_LOAD'):
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
    
    # Create ice load array with same structure as accretion data at single height level
    template = ds['ACCRE_CYL'].isel(height=height_level)
    dsiceload = xr.zeros_like(template) * np.nan
    
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
    
    print(f"\n=== Ice Load Integration Summary ===")
    print(f"Successfully calculated '{variable_name}' variable")
    print(f"Ice load shape: {dsiceload.shape}")
    print(f"Height level used: {height_level} ({height_value} {height_units})")
    print(f"Calculation method: {method}")
    print(f"Valid data points: {np.sum(~np.isnan(dsiceload.values)):,}")
    
    # Memory-efficient dataset creation with chunking
    print(f"\nCreating memory-efficient dataset with ice load...")
    
    # Use chunking for memory efficiency - chunk by time to optimize I/O
    chunk_size = min(8760, len(dsiceload.time) // 10)  # ~1 year or 1/10 of data, whichever is smaller
    
    # Convert to dask arrays for lazy evaluation
    dsiceload_chunked = dsiceload.chunk({'time': chunk_size})
    
    # Create expanded ice load array with chunking
    ice_load_expanded = xr.full_like(ds['ACCRE_CYL'], np.nan, dtype=np.float32).chunk({
        'time': chunk_size,
        'height': 1,  # Keep height chunks small
        'south_north': ds.dims['south_north'],
        'west_east': ds.dims['west_east']
    })
    
    # Assign ice load data only to the specific height level
    ice_load_expanded.loc[dict(height=height_value)] = dsiceload_chunked
    
    # Add the ice load variable to the original dataset
    ds[variable_name] = ice_load_expanded
    
    # Add attributes
    ds[variable_name].attrs = {
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
    
    print(f"Dataset now contains {len(ds.data_vars)} variables:")
    for var in ds.data_vars:
        print(f"  - {var}: {ds[var].shape}")
    
    # Save the complete dataset with aggressive optimization
    print(f"\nSaving complete dataset with ice load (memory-optimized)...")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Format start and end dates for filename
    start_date = pd.to_datetime(dates[0]).strftime('%Y%m%d')
    end_date = pd.to_datetime(dates[-1]).strftime('%Y%m%d')
    
    # Create filename for the dataset with ice load
    dataset_filename = f"dataset_iceload_{OffOn}_{start_date}_{end_date}_h{height_level}.nc"
    dataset_filepath = os.path.join(results_dir, dataset_filename)
    
    print(f"Saving to: {dataset_filepath}")
    
    # Aggressive encoding and chunking for memory efficiency
    encoding = {}
    optimal_chunk_time = min(2920, len(ds.time) // 20)  # About 2 months or 1/20 of data
    
    for var_name, var in ds.data_vars.items():
        # Use smaller data types where possible
        if var.dtype == np.float64:
            dtype = np.float32
        elif var.dtype == np.int64:
            dtype = np.int32
        else:
            dtype = var.dtype
            
        base_encoding = {
            'zlib': True,
            'complevel': 6,  # Higher compression
            'shuffle': True,
            'fletcher32': True,  # Add checksums
            'dtype': dtype
        }
        
        # Set chunking based on variable dimensions
        if 'time' in var.dims:
            if len(var.dims) == 4:  # 4D variables (time, height, south_north, west_east)
                base_encoding['chunksizes'] = (optimal_chunk_time, 1, var.sizes['south_north'], var.sizes['west_east'])
            elif len(var.dims) == 3:  # 3D variables (time, south_north, west_east)
                base_encoding['chunksizes'] = (optimal_chunk_time, var.sizes['south_north'], var.sizes['west_east'])
        
        encoding[var_name] = base_encoding
    
    # Also encode coordinate variables for efficiency
    for coord_name, coord in ds.coords.items():
        if coord_name not in ['time', 'height', 'south_north', 'west_east']:
            encoding[coord_name] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True,
                'dtype': np.float32 if coord.dtype == np.float64 else coord.dtype
            }
    
    # Ensure dataset is properly chunked before saving
    ds_chunked = ds.chunk({
        'time': optimal_chunk_time,
        'height': 1,
        'south_north': ds.dims['south_north'],
        'west_east': ds.dims['west_east']
    })
    
    # Save with memory-efficient approach
    print("Writing NetCDF file with optimized chunks and compression...")
    ds_chunked.to_netcdf(
        dataset_filepath,
        encoding=encoding,
        engine='netcdf4',
        unlimited_dims=['time']  # Allow unlimited time dimension for efficiency
    )
    
    print(f"Successfully saved complete dataset with ice load!")
    print(f"  File size: {os.path.getsize(dataset_filepath) / (1024*1024):.1f} MB")
    print(f"  Variables: {list(ds.data_vars.keys())}")
    print(f"  Time range: {start_date} to {end_date}")
    print(f"  Ice load method: {method}")
    print(f"  Height level: {height_level}")
    
    return ds

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

def plot_grid_ice_load_values(dataset_with_ice_load, ice_load_variable='ICE_LOAD', height_level=0, 
                             ice_load_threshold=0.0, save_plots=True, 
                             months=None, show_colorbar=True, OffOn=None, BigDomain=False,
                             margin_degrees=0.5, zoom_level=6):
    """
    Plot annual mean total ice load for each grid cell.
    Calculates the total ice load per year for each grid cell, then computes the mean across all years.
    
    Parameters:
    -----------
    dataset_with_ice_load : xarray.Dataset
        Dataset containing ice load data (from add_ice_load_to_dataset)
    ice_load_variable : str, default 'ICE_LOAD'
        Name of the ice load variable in the dataset
    height_level : int, default 0
        Height level index to analyze
    ice_load_threshold : float, default 0.0
        Minimum ice load value to include in calculation (kg/m)
    save_plots : bool, default True
        Whether to save the plots to files
    months : list of int, optional
        List of months to include (1-12). If None, uses all months
    show_colorbar : bool, default True
        Whether to show colorbar in the grid plot
    OffOn : str, optional
        Specifies 'Onshore' or 'Offshore' for BigDomain directory structure
    BigDomain : bool, default False
        If True, saves results to MT_Icing/results/figures/BigDomain/{OffOn}/spatial_gradient...
    margin_degrees : float, default 0.5
        Margin around grid in degrees for cartopy terrain map
    zoom_level : int, default 6
        Zoom level for terrain tiles in cartopy terrain map
        
    Returns:
    --------
    dict : Dictionary containing annual totals and mean values for each grid cell
    """
    print("=== ANNUAL MEAN ICE LOAD GRID ANALYSIS ===")
    
    try:
        # Extract ice load data
        if ice_load_variable not in dataset_with_ice_load.data_vars:
            raise ValueError(f"Variable '{ice_load_variable}' not found in dataset. Available variables: {list(dataset_with_ice_load.data_vars.keys())}")
        
        ice_load_data = dataset_with_ice_load[ice_load_variable].isel(height=height_level)
        
        print(f"\n1. Data Information:")
        print(f"   Ice load variable: {ice_load_variable}")
        print(f"   Height level: {height_level} ({dataset_with_ice_load.height.values[height_level]} m)")
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
        years = sorted(time_index.year.unique())
        print(f"   Years covered: {n_years}")
        print(f"   Years: {years}")
        
        # Clean data and apply threshold
        ice_data_clean = ice_load_data.where(ice_load_data >= ice_load_threshold, 0)  # Replace below threshold with 0
        ice_data_clean = ice_data_clean.where(ice_data_clean.notnull(), 0)  # Replace NaN with 0
        
        # Filter data by months if specified
        if months is not None:
            print(f"\n   Filtering data to specified months: {months}...")
            time_index_full = pd.to_datetime(ice_data_clean.time.values)
            month_mask = time_index_full.month.isin(months)
            ice_data_filtered = ice_data_clean.isel(time=month_mask)
            
            time_index_filtered = pd.to_datetime(ice_data_filtered.time.values)
            n_filtered_timesteps = len(time_index_filtered)
            
            print(f"   Original timesteps: {n_time}")
            print(f"   Filtered timesteps: {n_filtered_timesteps}")
            print(f"   Months included: {sorted(time_index_filtered.month.unique())}")
            print(f"   Reduction: {((n_time - n_filtered_timesteps) / n_time * 100):.1f}% timesteps removed")
            
            ice_data_analysis = ice_data_filtered
        else:
            print(f"\n   Using all months for analysis...")
            ice_data_analysis = ice_data_clean
        
        print(f"\n2. Ice Load Statistics (threshold: {ice_load_threshold:.3f} kg/m):")
        print(f"   Range: {float(ice_data_analysis.min()):.3f} to {float(ice_data_analysis.max()):.3f} kg/m")
        print(f"   Overall mean: {float(ice_data_analysis.mean()):.3f} kg/m")
        
        # Calculate annual totals for each grid cell
        print(f"\n3. Calculating annual totals for each grid cell...")
        
        # Create time coordinate for grouping by year
        ice_data_analysis = ice_data_analysis.assign_coords(year=ice_data_analysis.time.dt.year)
        
        # Calculate annual sums for each grid cell - using chunking to avoid memory issues
        print(f"   Using chunked computation to avoid memory issues...")
        annual_totals = ice_data_analysis.groupby('year').sum(dim='time').compute()
        
        print(f"   Annual totals calculated for {len(annual_totals.year)} years")
        print(f"   Years in annual totals: {list(annual_totals.year.values)}")
        
        # Calculate mean annual total for each grid cell
        mean_annual_totals = annual_totals.mean(dim='year').compute()
        
        print(f"   Mean annual totals shape: {mean_annual_totals.shape}")
        print(f"   Mean annual range: {float(mean_annual_totals.min()):.3f} to {float(mean_annual_totals.max()):.3f} kg/m")
        
        # Prepare results storage
        results = {
            'grid_shape': (n_south_north, n_west_east),
            'n_years': n_years,
            'years': years,
            'ice_load_threshold': ice_load_threshold,
            'annual_totals': annual_totals,
            'mean_annual_totals': mean_annual_totals,
            'grid_statistics': {}
        }
        
        # Calculate statistics for each grid cell
        print(f"\n4. Computing statistics for each grid cell...")
        
        for i in range(n_south_north):
            for j in range(n_west_east):
                cell_annual_totals = annual_totals.isel(south_north=i, west_east=j).values
                cell_mean_annual = float(mean_annual_totals.isel(south_north=i, west_east=j).values)
                
                cell_key = f'cell_{i}_{j}'
                results['grid_statistics'][cell_key] = {
                    'position': (i, j),
                    'annual_totals': cell_annual_totals.tolist(),
                    'mean_annual_total': cell_mean_annual,
                    'std_annual_total': float(np.std(cell_annual_totals)),
                    'min_annual_total': float(np.min(cell_annual_totals)),
                    'max_annual_total': float(np.max(cell_annual_totals)),
                    'cv': float(np.std(cell_annual_totals) / cell_mean_annual) if cell_mean_annual > 0 else 0
                }
        
        print(f"   Computed statistics for {len(results['grid_statistics'])} grid cells")
        
        # Skip spatial grid visualization to save memory - only generate cartopy map
        print(f"\n5. Skipping spatial grid visualization (only generating cartopy map)...")
        
        # Prepare plot data for cartopy
        plot_data = mean_annual_totals.values
        
        # Create directory structure for saving
        height_m = int(dataset_with_ice_load.height.values[height_level])
        
        # Format ice threshold with appropriate precision and replace decimal point with 'p'
        if ice_load_threshold == int(ice_load_threshold):
            # If it's a whole number, format as integer
            ice_threshold_str = f"{int(ice_load_threshold)}"
        else:
            # For decimal values, use appropriate precision to avoid rounding
            ice_threshold_str = f"{ice_load_threshold:.2f}".rstrip('0').rstrip('.')
        ice_threshold_str = ice_threshold_str.replace('.', 'p')
        
        if BigDomain and OffOn:
            base_dir = os.path.join("results", "figures", "BigDomain", OffOn, "spatial_gradient", "Ice_load_grid")
        else:
            base_dir = os.path.join("results", "figures", "spatial_gradient", "Ice_load_grid")
        specific_dir = f"ice_load_grid_{height_m}_{ice_threshold_str}"
        ice_load_plots_dir = os.path.join(base_dir, specific_dir)
        os.makedirs(ice_load_plots_dir, exist_ok=True)
        
        print(f"   Will save plots to: {ice_load_plots_dir}")
        
        # Create cartopy terrain map with mean annual ice load
        print(f"\n6. Creating cartopy terrain map with mean annual ice load...")
        
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            import cartopy.io.img_tiles as cimgt
            
            # Get geographical coordinates
            if 'XLAT' in dataset_with_ice_load.coords and 'XLON' in dataset_with_ice_load.coords:
                lats = dataset_with_ice_load.coords['XLAT'].values
                lons = dataset_with_ice_load.coords['XLON'].values
            elif 'XLAT' in dataset_with_ice_load.data_vars and 'XLON' in dataset_with_ice_load.data_vars:
                lats = dataset_with_ice_load['XLAT'].values
                lons = dataset_with_ice_load['XLON'].values
            else:
                raise ValueError("No latitude/longitude coordinates found in dataset")
            
            print(f"   Grid coordinates: Lat {lats.min():.3f} to {lats.max():.3f}, Lon {lons.min():.3f} to {lons.max():.3f}")
            
            # Calculate grid cell edges for pcolormesh
            lon_edges = np.zeros(lons.shape[1] + 1)
            lat_edges = np.zeros(lats.shape[0] + 1)
            
            # Longitude edges
            for j in range(lons.shape[1]):
                if j == 0:
                    lon_edges[j] = lons[0, j] - (lons[0, 1] - lons[0, 0]) / 2
                else:
                    lon_edges[j] = (lons[0, j-1] + lons[0, j]) / 2
            lon_edges[-1] = lons[0, -1] + (lons[0, -1] - lons[0, -2]) / 2
            
            # Latitude edges
            for i in range(lats.shape[0]):
                if i == 0:
                    lat_edges[i] = lats[i, 0] - (lats[1, 0] - lats[0, 0]) / 2
                else:
                    lat_edges[i] = (lats[i-1, 0] + lats[i, 0]) / 2
            lat_edges[-1] = lats[-1, 0] + (lats[-1, 0] - lats[-2, 0]) / 2
            
            # Create figure with cartopy projection
            fig = plt.figure(figsize=(16, 12))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            # Set extent with margin
            grid_center_lon = (lon_edges.min() + lon_edges.max()) / 2
            grid_center_lat = (lat_edges.min() + lat_edges.max()) / 2
            grid_span_lon = lon_edges.max() - lon_edges.min()
            grid_span_lat = lat_edges.max() - lat_edges.min()
            extent_span_lon = grid_span_lon + 2 * margin_degrees
            extent_span_lat = grid_span_lat + 2 * margin_degrees
            
            west = grid_center_lon - extent_span_lon / 2
            east = grid_center_lon + extent_span_lon / 2
            south = grid_center_lat - extent_span_lat / 2
            north = grid_center_lat + extent_span_lat / 2
            
            ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
            
            # Add geographical features
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.7, zorder=1)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.8, zorder=2)
            ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.8, zorder=3)
            
            # Try to add terrain background
            try:
                terrain = cimgt.OSM()
                ax.add_image(terrain, zoom_level)
                print(f"   Successfully loaded OpenStreetMap tiles")
            except Exception as e:
                try:
                    terrain = cimgt.GoogleTiles(style='satellite')
                    ax.add_image(terrain, zoom_level)
                    print(f"   Successfully loaded Google satellite tiles")
                except Exception as e2:
                    print(f"   Tile services unavailable, using basic land/ocean features")
            
            # Add geographical features on top
            ax.add_feature(cfeature.BORDERS, linewidth=1.5, color='black', alpha=0.8, zorder=8)
            ax.add_feature(cfeature.COASTLINE, linewidth=2, color='black', alpha=0.9, zorder=9)
            
            # Create meshgrid for pcolormesh
            lon_mesh, lat_mesh = np.meshgrid(lon_edges, lat_edges)
            
            # Calculate 90th percentile for color scale clipping
            plot_data_flat = plot_data.flatten()
            valid_data = plot_data_flat[~np.isnan(plot_data_flat)]
            
            if len(valid_data) > 0:
                data_min = np.min(valid_data)
                data_max = np.max(valid_data)
                data_90p = np.percentile(valid_data, 90)
                
                print(f"   Ice load statistics:")
                print(f"     Min: {data_min:.1f}, Max: {data_max:.1f} kg/m")
                print(f"     90th percentile: {data_90p:.1f} kg/m")
                
                # Apply 90th percentile clipping
                vmin = data_min
                vmax = data_90p
                print(f"   Using 90th percentile clipping for color scale: {vmin:.1f} - {vmax:.1f} kg/m")
            else:
                vmin, vmax = 0, 1
            
            # Plot mean annual ice load values as semi-transparent overlay
            ice_load_plot = ax.pcolormesh(
                lon_mesh, lat_mesh, plot_data,
                cmap='viridis', alpha=0.8,
                vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree(),
                zorder=7
            )
            
            # Add colorbar
            cbar = plt.colorbar(ice_load_plot, ax=ax, shrink=0.8, pad=0.02)
            cbar_label = f'Mean Annual Total Ice Load (kg/m)\n[Clipped at 90th percentile: {vmax:.1f}]'
            cbar.set_label(cbar_label, fontsize=16)
            cbar.ax.tick_params(labelsize=15)
            
            # Add gridlines with labels
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 20, 'color': 'black'}
            gl.ylabel_style = {'size': 20, 'color': 'black'}
            
            # Add title
            title_text = (f'Mean Annual Total Ice Load on Terrain Map\n'
                         f'Threshold: {ice_load_threshold:.1f} kg/m, '
                         f'Height: {dataset_with_ice_load.height.values[height_level]} m')
            ax.set_title(title_text, fontsize=28, weight='bold', pad=20)
            
            plt.tight_layout()
            
            if save_plots:
                # Create filename for cartopy map
                cartopy_filename_parts = ["mean_annual_total_ice_load_cartopy"]
                if months is not None:
                    months_str = "_".join(map(str, sorted(months)))
                    cartopy_filename_parts.append(f"months_{months_str}")
                
                cartopy_plot_filename = "_".join(cartopy_filename_parts) + ".png"
                cartopy_plot_path = os.path.join(ice_load_plots_dir, cartopy_plot_filename)
                plt.savefig(cartopy_plot_path, dpi=300, bbox_inches='tight')
                print(f"   Cartopy terrain map saved to: {cartopy_plot_path}")
            
            plt.close()
            
        except ImportError:
            print(f"   Error: Cartopy required for terrain map")
            print(f"   Please install with: conda install cartopy")
        except Exception as e:
            print(f"   Error creating cartopy terrain map: {e}")
        
        # Skip time series plots to save memory
        print(f"\n7. Skipping time series plots (only generating cartopy map as requested)...")
        
        # Print summary statistics
        print(f"\n8. Summary Statistics:")
        print(f"   Processed {n_south_north * n_west_east} grid cells")
        print(f"   Years analyzed: {n_years}")
        
        all_means = [stats['mean_annual_total'] for stats in results['grid_statistics'].values()]
        all_stds = [stats['std_annual_total'] for stats in results['grid_statistics'].values()]
        all_cvs = [stats['cv'] for stats in results['grid_statistics'].values() if stats['cv'] > 0]
        
        print(f"   Mean annual total ice load across domain:")
        print(f"     Mean: {np.mean(all_means):.3f} kg/m")
        print(f"     Min: {np.min(all_means):.3f} kg/m")
        print(f"     Max: {np.max(all_means):.3f} kg/m")
        print(f"     Std: {np.std(all_means):.3f} kg/m")
        
        if all_cvs:
            print(f"   Coefficient of variation across domain:")
            print(f"     Mean CV: {np.mean(all_cvs):.3f}")
            print(f"     Min CV: {np.min(all_cvs):.3f}")
            print(f"     Max CV: {np.max(all_cvs):.3f}")
        
        return results
    
    except Exception as e:
        print(f"\nError in annual mean ice load grid analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_ice_load_threshold_exceedance_map(dataset_with_ice_load, ice_load_variable='ICE_LOAD', height_level=0,
                                         ice_load_threshold=0.1, save_plots=True, 
                                         colormap='viridis', grid_labels=True, units='hours',
                                         OffOn=None, BigDomain=False,
                                         margin_degrees=0.2, zoom_level=6,
                                         custom_vmin=None, custom_vmax=None):
    """
    Create a spatial map showing how often each grid cell exceeds a specified ice load threshold
    per year on average. Uses a colorbar to show spatial differences in threshold exceedance.
    
    Parameters:
    -----------
    dataset_with_ice_load : xarray.Dataset
        Dataset containing ice load data (from add_ice_load_to_dataset)
    ice_load_variable : str, default 'ICE_LOAD'
        Name of the ice load variable in the dataset
    height_level : int, default 0
        Height level index to analyze
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
    OffOn : str, optional
        Specifies 'Onshore' or 'Offshore' for BigDomain directory structure
    BigDomain : bool, default False
        If True, saves results to MT_Icing/results/figures/BigDomain/{OffOn}/spatial_gradient...
    margin_degrees : float, default 0.2
        Margin around grid in degrees for cartopy terrain map
    zoom_level : int, default 6
        Zoom level for terrain tiles in cartopy terrain map
        
    Returns:
    --------
    dict : Dictionary containing exceedance analysis results and statistics
    """
    print(f"=== ICE LOAD THRESHOLD EXCEEDANCE ANALYSIS ===")
    print(f"Height level: {height_level} ({dataset_with_ice_load.height.values[height_level]} m)")
    print(f"Threshold: {ice_load_threshold:.3f} kg/m")
    
    try:
        # Extract ice load data
        if ice_load_variable not in dataset_with_ice_load.data_vars:
            raise ValueError(f"Variable '{ice_load_variable}' not found in dataset. Available variables: {list(dataset_with_ice_load.data_vars.keys())}")
        
        ice_load_data = dataset_with_ice_load[ice_load_variable].isel(height=height_level)
        
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
        
        # Create the main plot with synchronized color scale
        vmin = custom_vmin if custom_vmin is not None else 0
        vmax = custom_vmax if custom_vmax is not None else None
        im = ax.imshow(exceedance_matrix, cmap=colormap, origin='lower', 
                      interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set title and labels
        unit_label = units.capitalize()
        if units == 'percentage':
            unit_label = '% of Time'
        
        ax.set_title(f'Ice Load Threshold Exceedance Map\n'
                    f'Threshold: {ice_load_threshold:.3f} kg/m\n'
                    f'Mean Annual Exceedance ({unit_label})', fontsize=28)
        ax.set_xlabel('West-East Grid Points', fontsize=24)
        ax.set_ylabel('South-North Grid Points', fontsize=24)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label(f'Exceedance ({unit_label}/Year)', fontsize=16)
        cbar.ax.tick_params(labelsize=15)
        
        # Set axis tick label size
        ax.tick_params(axis='both', labelsize=20)
        
        # Add grid lines
        ax.set_xticks(range(n_west_east))
        ax.set_yticks(range(n_south_north))
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Add coordinate references
        ax.set_xticks(range(n_west_east))
        ax.set_yticks(range(n_south_north))
        ax.set_xticklabels([f'{j}' for j in range(n_west_east)])
        ax.set_yticklabels([f'{i}' for i in range(n_south_north)])
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            # Create directory structure: results/figures/spatial_gradient/ice_load_grid_hours_exceedance/ice_load_hours_{height}_{ice_threshold}
            height_m = int(dataset_with_ice_load.height.values[height_level])
            
            # Format ice threshold with appropriate precision and replace decimal point with 'p'
            if ice_load_threshold == int(ice_load_threshold):
                # If it's a whole number, format as integer
                ice_threshold_str = f"{int(ice_load_threshold)}"
            else:
                # For decimal values, use appropriate precision to avoid rounding
                ice_threshold_str = f"{ice_load_threshold:.2f}".rstrip('0').rstrip('.')
            ice_threshold_str = ice_threshold_str.replace('.', 'p')
            
            # Create directory structure based on BigDomain flag
            if BigDomain and OffOn:
                base_dir = os.path.join(figures_dir, "BigDomain", OffOn, "spatial_gradient", "ice_load_grid_hours_exceedance")
            else:
                base_dir = os.path.join(figures_dir, "spatial_gradient", "ice_load_grid_hours_exceedance")
            
            specific_dir = f"ice_load_hours_{height_m}_{ice_threshold_str}"
            threshold_plots_dir = os.path.join(base_dir, specific_dir)
            os.makedirs(threshold_plots_dir, exist_ok=True)
            
            print(f"   Saving plots to: {threshold_plots_dir}")
            
            # Create filename with threshold value
            plot_path = os.path.join(threshold_plots_dir, f"ice_load_threshold_exceedance_{ice_threshold_str}kgm.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   Exceedance map saved to: {plot_path}")
        
        plt.close()  # Close the plot to prevent it from showing
        
        # Create cartopy terrain map with threshold exceedance (similar to analyze_ice_load_with_weighted_neighborhood_cdf)
        print(f"\\n4. Creating cartopy terrain map with threshold exceedance...")
        
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            import cartopy.io.img_tiles as cimgt
            
            # Get geographical coordinates
            if 'XLAT' in dataset_with_ice_load.coords and 'XLON' in dataset_with_ice_load.coords:
                lats = dataset_with_ice_load.coords['XLAT'].values
                lons = dataset_with_ice_load.coords['XLON'].values
            elif 'XLAT' in dataset_with_ice_load.data_vars and 'XLON' in dataset_with_ice_load.data_vars:
                lats = dataset_with_ice_load['XLAT'].values
                lons = dataset_with_ice_load['XLON'].values
            else:
                raise ValueError("No latitude/longitude coordinates found in dataset")
            
            print(f"   Grid coordinates: Lat {lats.min():.3f} to {lats.max():.3f}, Lon {lons.min():.3f} to {lons.max():.3f}")
            
            # Calculate grid cell edges for pcolormesh
            lon_edges = np.zeros(lons.shape[1] + 1)
            lat_edges = np.zeros(lats.shape[0] + 1)
            
            # Longitude edges
            for j in range(lons.shape[1]):
                if j == 0:
                    lon_edges[j] = lons[0, j] - (lons[0, 1] - lons[0, 0]) / 2
                else:
                    lon_edges[j] = (lons[0, j-1] + lons[0, j]) / 2
            lon_edges[-1] = lons[0, -1] + (lons[0, -1] - lons[0, -2]) / 2
            
            # Latitude edges
            for i in range(lats.shape[0]):
                if i == 0:
                    lat_edges[i] = lats[i, 0] - (lats[1, 0] - lats[0, 0]) / 2
                else:
                    lat_edges[i] = (lats[i-1, 0] + lats[i, 0]) / 2
            lat_edges[-1] = lats[-1, 0] + (lats[-1, 0] - lats[-2, 0]) / 2
            
            # Create figure with cartopy projection
            fig = plt.figure(figsize=(16, 12))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            # Set extent with margin
            grid_center_lon = (lon_edges.min() + lon_edges.max()) / 2
            grid_center_lat = (lat_edges.min() + lat_edges.max()) / 2
            grid_span_lon = lon_edges.max() - lon_edges.min()
            grid_span_lat = lat_edges.max() - lat_edges.min()
            extent_span_lon = grid_span_lon + 2 * margin_degrees
            extent_span_lat = grid_span_lat + 2 * margin_degrees
            
            west = grid_center_lon - extent_span_lon / 2
            east = grid_center_lon + extent_span_lon / 2
            south = grid_center_lat - extent_span_lat / 2
            north = grid_center_lat + extent_span_lat / 2
            
            ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
            
            # Add geographical features
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.7, zorder=1)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.8, zorder=2)
            ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.8, zorder=3)
            
            # Try to add terrain background
            try:
                terrain = cimgt.OSM()
                ax.add_image(terrain, zoom_level)
                print(f"   Successfully loaded OpenStreetMap tiles")
            except Exception as e:
                try:
                    terrain = cimgt.GoogleTiles(style='satellite')
                    ax.add_image(terrain, zoom_level)
                    print(f"   Successfully loaded Google satellite tiles")
                except Exception as e2:
                    print(f"   Tile services unavailable, using basic land/ocean features")
            
            # Add geographical features on top
            ax.add_feature(cfeature.BORDERS, linewidth=1.5, color='black', alpha=0.8, zorder=8)
            ax.add_feature(cfeature.COASTLINE, linewidth=2, color='black', alpha=0.9, zorder=9)
            
            # Create meshgrid for pcolormesh
            lon_mesh, lat_mesh = np.meshgrid(lon_edges, lat_edges)
            
            # Calculate percentiles for better color scaling
            if len(valid_exceedances) > 0:
                data_min = np.min(valid_exceedances)
                data_max = np.max(valid_exceedances)
                data_mean = np.mean(valid_exceedances)
                data_90p = np.percentile(valid_exceedances, 90)
                
                print(f"   Threshold exceedance statistics:")
                print(f"     Min: {data_min:.1f}, Max: {data_max:.1f}, Mean: {data_mean:.1f} {unit_label.lower()}/year")
                print(f"     90th percentile: {data_90p:.1f} {unit_label.lower()}/year")
                
                # Use custom color scale if provided, otherwise use automatic scaling
                if custom_vmin is not None and custom_vmax is not None:
                    vmin = custom_vmin
                    vmax = custom_vmax
                    outlier_clipped = False  # Clipping already handled in main script
                    print(f"   Using provided color scale: {vmin:.1f} - {vmax:.1f} {unit_label.lower()}/year")
                else:
                    # Apply automatic scaling with outlier detection
                    outlier_ratio = data_max / data_90p if data_90p > 0 else 1
                    if outlier_ratio > 2.0:
                        vmin = data_min
                        vmax = data_90p
                        outlier_clipped = True
                        print(f"   Using 90th percentile clipping for better visualization: {vmin:.1f} - {vmax:.1f} {unit_label.lower()}/year")
                    else:
                        vmin = data_min
                        vmax = data_max
                        outlier_clipped = False
                        print(f"   Using full data range: {vmin:.1f} - {vmax:.1f} {unit_label.lower()}/year")
            else:
                vmin, vmax = (custom_vmin, custom_vmax) if (custom_vmin is not None and custom_vmax is not None) else (0, 1)
                outlier_clipped = False
            
            # Plot threshold exceedance values as semi-transparent overlay
            exceedance_plot = ax.pcolormesh(
                lon_mesh, lat_mesh, exceedance_matrix,
                cmap=colormap, alpha=0.8,
                vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree(),
                zorder=7
            )
            
            # Add colorbar
            cbar = plt.colorbar(exceedance_plot, ax=ax, shrink=0.8, pad=0.02)
            if outlier_clipped:
                cbar_label = f'Threshold Exceedance ({unit_label}/Year)\\n[Clipped at 90th percentile: {vmax:.1f}]'
            else:
                cbar_label = f'Threshold Exceedance ({unit_label}/Year)'
            cbar.set_label(cbar_label, fontsize=24)
            cbar.ax.tick_params(labelsize=30)
            
            # Add gridlines with labels
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 30, 'color': 'black'}
            gl.ylabel_style = {'size': 30, 'color': 'black'}
            
            # Add title
            title_text = (f'Ice Load Threshold Exceedance on Terrain Map\\\\n'
                         f'Threshold: {ice_load_threshold:.3f} kg/m, '
                         f'Height: {dataset_with_ice_load.height.values[height_level]} m')
            ax.set_title(title_text, fontsize=28, weight='bold', pad=20)
            
            # Add statistics information
            if len(valid_exceedances) > 0:
                if outlier_clipped:
                    info_text = (f"Range: {data_min:.1f} - {data_max:.1f} {unit_label.lower()}/year | "
                                f"Mean: {data_mean:.1f} {unit_label.lower()}/year\\n"
                                f"Color scale: {vmin:.1f} - {vmax:.1f} {unit_label.lower()}/year (90th percentile clipped)")
                else:
                    info_text = (f"Range: {data_min:.1f} - {data_max:.1f} {unit_label.lower()}/year | "
                                f"Mean: {data_mean:.1f} {unit_label.lower()}/year")
                ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=27,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            
            plt.tight_layout()
            
            # Save the cartopy terrain map
            if save_plots:
                cartopy_filename = f"ice_load_threshold_exceedance_cartopy_{ice_threshold_str}kgm.png"
                cartopy_path = os.path.join(threshold_plots_dir, cartopy_filename)
                plt.savefig(cartopy_path, dpi=300, bbox_inches='tight')
                print(f"   Cartopy terrain map saved to: {cartopy_path}")
            
            plt.close()  # Close the plot to prevent it from showing
            
        except ImportError as e:
            print(f"   Warning: Cartopy not available, skipping terrain map: {e}")
        except Exception as e:
            print(f"   Error creating cartopy terrain map: {e}")
        
        # Create additional summary statistics plot
        if len(valid_exceedances) > 0:
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram of exceedance values
            ax1.hist(valid_exceedances, bins=min(20, len(np.unique(valid_exceedances))), 
                    alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel(f'Exceedance ({unit_label}/Year)')
            ax1.set_ylabel('Number of Grid Cells', fontsize=24)
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
                ax2.set_xlabel('Grid Row (South to North)', fontsize=24)
                ax2.set_ylabel(f'Exceedance ({unit_label}/Year)')
                ax2.set_title('Exceedance by Grid Row', fontsize=28)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_plots:
                summary_path = os.path.join(threshold_plots_dir, f"ice_load_threshold_exceedance_summary_{ice_threshold_str}kgm.png")
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
            results_path = os.path.join(results_dir, f"ice_load_threshold_exceedance_{ice_threshold_str}kgm.txt")
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


# WIND ROSE

def wind_rose(dataset, height_level=0, wd_variable='WD', save_plot=True, bins=16, title=None, output_dir="results/figures/wind_rose"):
    """
    Plot wind rose for wind direction (WD) at a specified height from an xarray dataset.
    Parameters:
    -----------
    dataset : xarray.Dataset
        Dataset containing wind direction variable (WD)
    height_level : int, optional
        Height index to use (default: 0)
    wd_variable : str, optional
        Name of wind direction variable (default: 'WD')
    save_plot : bool, optional
        Whether to save the plot (default: True)
    bins : int, optional
        Number of wind direction bins (default: 16)
    title : str, optional
        Title for the plot
    output_dir : str, optional
        Directory to save the plot
    Returns:
    -----------
    None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Extract wind direction data at specified height
    if wd_variable not in dataset.data_vars:
        raise ValueError(f"Variable '{wd_variable}' not found in dataset.")
    wd_data = dataset[wd_variable].isel(height=height_level).values.flatten()

    # Remove NaNs
    wd_data = wd_data[~np.isnan(wd_data)]

    # Bin wind directions
    bin_edges = np.linspace(0, 360, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts, _ = np.histogram(wd_data, bins=bin_edges)
    total = counts.sum()
    percentages = counts / total * 100

    # Convert to polar coordinates
    theta = np.deg2rad(bin_centers)
    width = 2 * np.pi / bins

    # Plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    bars = ax.bar(theta, percentages, width=width, bottom=0.0, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    # Show both compass directions and degree values
    compass_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    compass_degrees = np.arange(0, 360, 45)
    ax.set_xticks(np.deg2rad(compass_degrees))
    ax.set_xticklabels([f'{label}\n{deg}°' for label, deg in zip(compass_labels, compass_degrees)])
    # Show radial grid labels (percentages)
    max_pct = percentages.max()
    yticks = np.linspace(0, max_pct, num=5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.1f}%' for y in yticks])
    ax.set_ylabel('Frequency (%)', labelpad=30, fontsize=24)
    if title is None:
        title = f"Wind Rose at Height Index {height_level}"
    ax.set_title(title, va='bottom', fontsize=28)

    # Save plot (do not show)
    output_dir = os.path.join("results", "figures", "geographical_maps")
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"wind_rose_height_{height_level}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Wind rose plot saved to: {plot_path}")
    plt.close()

# TEMPERATURE AND HUMIDITY CRITERIA

def add_rh(dataset_with_ice_load, height_l, phase, verbose=True):
    """
    Compute relative humidity from temperature, pressure, and mixing ratio,
    and add it as a new variable to the dataset.
    
    Parameters:
    -----------
    dataset_with_ice_load : xarray.Dataset
        Dataset containing meteorological variables including:
        - PSFC: surface pressure (Pa)
        - T: air temperature (K) 
        - QVAPOR: humidity mixing ratio (kg/kg)
    verbose : bool, optional
        Whether to print detailed information during calculation (default: True)
        
    Returns:
    --------
    xarray.Dataset
        Dataset with added 'relative_humidity' variable (dimensionless, 0-1)
        
    Notes:
    ------
    Uses MetPy's relative_humidity_from_mixing_ratio function which expects:
    - pressure in pressure units (Pa, hPa, etc.)
    - temperature in temperature units (K, °C, etc.)
    - mixing_ratio in dimensionless units (kg/kg)
    """
    
    if verbose:
        print("=== RELATIVE HUMIDITY CALCULATION ===")
        print("Using MetPy to calculate relative humidity from mixing ratio")
    
    try:
        # Check if required variables exist
        required_vars = ['PSFC', 'T', 'QVAPOR']
        missing_vars = [var for var in required_vars if var not in dataset_with_ice_load.data_vars]
        
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
            
        if verbose:
            print(f"\n1. Input Variables:")
            for var in required_vars:
                data_var = dataset_with_ice_load[var]
                print(f"   {var}: {data_var.shape} {data_var.dims}")
                if hasattr(data_var, 'attrs') and 'units' in data_var.attrs:
                    print(f"        Units: {data_var.attrs['units']}")
                print(f"        Range: {float(data_var.min().values):.3f} to {float(data_var.max().values):.3f}")
        
        # Extract the variables
        pressure = dataset_with_ice_load['PSFC']  # Surface pressure (Pa)
        temperature = dataset_with_ice_load['T']   # Air temperature (K)
        mixing_ratio = dataset_with_ice_load['QVAPOR']  # Mixing ratio (kg/kg)
        
        if verbose:
            print(f"\n2. Data Processing:")
            print(f"   Checking variable shapes...")
            print(f"   PSFC shape: {pressure.shape}, dims: {pressure.dims}")
            print(f"   T shape: {temperature.shape}, dims: {temperature.dims}")
            print(f"   QVAPOR shape: {mixing_ratio.shape}, dims: {mixing_ratio.dims}")
        
        # Handle dimensional differences - ensure all variables have compatible shapes
        # PSFC is typically 2D (surface), while T and QVAPOR might be 3D (with height levels)
        
        if 'height' in temperature.dims and 'height' not in pressure.dims:
            # Temperature has height dimension but pressure doesn't
            # Use lower height level for temperature and mixing ratio
            if verbose:
                print(f"   Temperature has height dimension, using level (height={height_l})")
            temperature = temperature.isel(height=height_l)
            mixing_ratio = mixing_ratio.isel(height=height_l)
        elif 'height' in pressure.dims and 'height' not in temperature.dims:
            # Pressure has height dimension but temperature doesn't 
            # Use lower level for pressure
            if verbose:
                print(f"   Pressure has height dimension, using level ")
            pressure = pressure.isel(height=height_l)
        elif 'height' in temperature.dims and 'height' in pressure.dims:
            # Both have height dimensions - use same level)
            if verbose:
                print(f"   Both variables have height dimension, using level (height={height_l})")
            temperature = temperature.isel(height=height_l)
            mixing_ratio = mixing_ratio.isel(height=height_l)
            pressure = pressure.isel(height=height_l)
        
        # Verify shapes after processing
        if verbose:
            print(f"   After processing:")
            print(f"   PSFC shape: {pressure.shape}, dims: {pressure.dims}")
            print(f"   T shape: {temperature.shape}, dims: {temperature.dims}")
            print(f"   QVAPOR shape: {mixing_ratio.shape}, dims: {mixing_ratio.dims}")
            
        # Check if shapes are now compatible
        if pressure.shape != temperature.shape or pressure.shape != mixing_ratio.shape:
            raise ValueError(f"Shape mismatch after processing: "
                           f"PSFC {pressure.shape}, T {temperature.shape}, QVAPOR {mixing_ratio.shape}")
        
        if verbose:
            print(f"   Converting data to MetPy units...")
        
        # Calculate relative humidity using MetPy with chunked processing to avoid memory issues
        if verbose:
            print(f"   Processing data in chunks to avoid memory allocation issues...")
        
        # Determine chunk size based on available memory (process ~10% of data at a time)
        total_timesteps = pressure.shape[0]
        chunk_size = max(1000, total_timesteps // 10)  # At least 1000 timesteps per chunk
        
        if verbose:
            print(f"   Total timesteps: {total_timesteps}")
            print(f"   Chunk size: {chunk_size}")
            print(f"   Number of chunks: {(total_timesteps + chunk_size - 1) // chunk_size}")
        
        # Initialize output array
        rh_values = np.full_like(pressure.values, np.nan, dtype=np.float32)
        
        # Process data in chunks
        for start_idx in range(0, total_timesteps, chunk_size):
            end_idx = min(start_idx + chunk_size, total_timesteps)
            chunk_slice = slice(start_idx, end_idx)
            
            if verbose and start_idx % (chunk_size * 5) == 0:  # Print progress every 5 chunks
                print(f"   Processing chunk {start_idx//chunk_size + 1}: timesteps {start_idx} to {end_idx-1}")
            
            # Extract chunk data
            pressure_chunk = pressure.isel(time=chunk_slice).values * units.Pa
            temperature_chunk = temperature.isel(time=chunk_slice).values * units.K  
            mixing_ratio_chunk = mixing_ratio.isel(time=chunk_slice).values * units('kg/kg')
            
            try:
                # Calculate relative humidity for this chunk
                rh_chunk = mpcalc.relative_humidity_from_mixing_ratio(
                    pressure_chunk,
                    temperature_chunk, 
                    mixing_ratio_chunk,
                    phase=phase
                )
                
                # Store results
                rh_values[start_idx:end_idx] = rh_chunk.magnitude.astype(np.float32)
                
            except Exception as chunk_error:
                if verbose:
                    print(f"   Warning: Error processing chunk {start_idx//chunk_size + 1}: {chunk_error}")
                # Leave NaN values for this chunk
                continue
        
        if verbose:
            print(f"   Chunked relative humidity calculation completed")
            print(f"   RH range: {np.nanmin(rh_values):.3f} to {np.nanmax(rh_values):.3f}")
            print(f"   RH mean: {np.nanmean(rh_values):.3f}")
            print(f"   Valid data points: {np.sum(~np.isnan(rh_values)):,} / {rh_values.size:,}")
        
        # Create a copy of the dataset
        ds_with_rh = dataset_with_ice_load.copy()
        
        # Create the relative humidity DataArray with same structure as processed temperature
        rh_dataarray = xr.DataArray(
            data=rh_values,
            dims=temperature.dims,
            coords=temperature.coords,
            attrs={
                'long_name': 'Relative humidity',
                'description': 'Relative humidity calculated from mixing ratio using MetPy (surface level)',
                'units': 'dimensionless',
                'valid_range': [0.0, 1.0],
                'calculation_method': 'MetPy relative_humidity_from_mixing_ratio',
                'input_variables': 'PSFC (Pa), T (K), QVAPOR (kg/kg) - all at surface level',
                'missing_value': np.nan,
                'level': 'surface'
            }
        )
        
        # Add to dataset
        ds_with_rh['relative_humidity'] = rh_dataarray
        
        if verbose:
            print(f"\n3. Results:")
            print(f"   Successfully added 'relative_humidity' variable to dataset")
            print(f"   Variable shape: {ds_with_rh['relative_humidity'].shape}")
            print(f"   Variable dimensions: {ds_with_rh['relative_humidity'].dims}")
            
            # Calculate some statistics
            rh_clean = ds_with_rh['relative_humidity'].where(
                ~np.isnan(ds_with_rh['relative_humidity']) & 
                (ds_with_rh['relative_humidity'] >= 0) & 
                (ds_with_rh['relative_humidity'] <= 1)
            )
            
            if rh_clean.size > 0:
                print(f"\n4. Statistics:")
                print(f"   Valid data points: {rh_clean.count().values:,}")
                print(f"   Mean RH: {float(rh_clean.mean().values):.3f} ({float(rh_clean.mean().values)*100:.1f}%)")
                print(f"   Median RH: {float(rh_clean.median().values):.3f} ({float(rh_clean.median().values)*100:.1f}%)")
                print(f"   Min RH: {float(rh_clean.min().values):.3f} ({float(rh_clean.min().values)*100:.1f}%)")
                print(f"   Max RH: {float(rh_clean.max().values):.3f} ({float(rh_clean.max().values)*100:.1f}%)")
                print(f"   Std RH: {float(rh_clean.std().values):.3f}")
                
                # Check for potentially problematic values
                n_over_100 = (rh_clean > 1.0).sum().values
                n_under_0 = (rh_clean < 0.0).sum().values
                
                if n_over_100 > 0:
                    print(f"   Warning: {n_over_100} values > 100% RH found")
                if n_under_0 > 0:
                    print(f"   Warning: {n_under_0} values < 0% RH found")
            
            print(f"\n5. Dataset Summary:")
            print(f"   Total variables: {len(ds_with_rh.data_vars)}")
            print(f"   Variables: {list(ds_with_rh.data_vars.keys())}")
        
        print(f"\n✓ Relative humidity calculation completed successfully!")
        return ds_with_rh
        
    except ImportError as e:
        print(f"Error: MetPy package required for relative humidity calculation")
        print(f"Please install with: conda install -c conda-forge metpy")
        print(f"Import error: {e}")
        return dataset_with_ice_load
        
    except Exception as e:
        print(f"Error calculating relative humidity: {e}")
        print(f"Returning original dataset without relative humidity")
        import traceback
        traceback.print_exc()
        return dataset_with_ice_load

def temp_hum_criteria(dataset, humidity_threshold, temperature_threshold, height_level=0,
                      save_plots=True, colormap='viridis', grid_labels=True,
                      OffOn=None, BigDomain=False,
                      margin_degrees=0.2, zoom_level=6,
                      custom_vmin=None, custom_vmax=None):
    """
    Create a spatial map showing how often each grid cell meets temperature and humidity criteria
    per year on average. Temperature must be equal or below the threshold, and relative humidity 
    must be equal or above the threshold.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Dataset containing meteorological data
    humidity_threshold : float
        Minimum relative humidity threshold value (0-1, dimensionless). Values must be >= this threshold
    temperature_threshold : float
        Maximum temperature threshold value (K). Values must be <= this threshold
    height_level : int, default 0
        Height level index to analyze
    save_plots : bool, default True
        Whether to save the plot to file
    colormap : str, default 'viridis'
        Matplotlib colormap to use for the spatial plot
    grid_labels : bool, default True
        Whether to add grid cell coordinate labels to the plot
    OffOn : str, optional
        Specifies 'Onshore' or 'Offshore' for BigDomain directory structure
    BigDomain : bool, default False
        If True, saves results to MT_Icing/results/figures/BigDomain/{OffOn}/spatial_gradient...
    margin_degrees : float, default 0.2
        Margin around grid in degrees for cartopy terrain map
    zoom_level : int, default 6
        Zoom level for terrain tiles in cartopy terrain map
        
    Returns:
    --------
    dict : Dictionary containing criteria analysis results and statistics
    """
    print(f"=== TEMPERATURE-HUMIDITY CRITERIA ANALYSIS ===")
    print(f"Height level: {height_level} ({dataset.height.values[height_level]} m)")
    print(f"Temperature threshold: <= {temperature_threshold:.2f} K")
    print(f"Relative humidity threshold: >= {humidity_threshold:.3f} (dimensionless)")
    
    try:
        # Check for required variables
        required_vars = ['T', 'relative_humidity']  # Temperature and relative humidity
        missing_vars = [var for var in required_vars if var not in dataset.data_vars]
        if missing_vars:
            raise ValueError(f"Required variables not found in dataset: {missing_vars}. Available variables: {list(dataset.data_vars.keys())}")
        
        # Extract temperature and humidity data at specified height
        temp_data = dataset['T'].isel(height=height_level)
        humidity_data = dataset['relative_humidity']
        
        # Check data structure
        print(f"\n1. Data Information:")
        print(f"   Temperature shape: {temp_data.shape}")
        print(f"   Humidity shape: {humidity_data.shape}")
        print(f"   Dimensions: {temp_data.dims}")
        
        # Get spatial dimensions
        n_time = temp_data.sizes['time']
        n_south_north = temp_data.sizes['south_north']
        n_west_east = temp_data.sizes['west_east']
        
        print(f"   Grid size: {n_south_north} × {n_west_east} = {n_south_north * n_west_east} cells")
        print(f"   Time steps: {n_time}")
        
        # Calculate temporal information
        time_index = pd.to_datetime(temp_data.time.values)
        n_years = len(time_index.year.unique())
        years = sorted(time_index.year.unique())
        
        # Calculate time step in hours (assuming regular intervals)
        if len(time_index) > 1:
            time_step_hours = (time_index[1] - time_index[0]).total_seconds() / 3600
        else:
            time_step_hours = 0.5  # Default to 30 minutes
            
        print(f"   Years covered: {n_years} ({years[0]} to {years[-1]})")
        print(f"   Time step: {time_step_hours} hours")
        
        # Clean the data (remove NaN values)
        temp_data_clean = temp_data.where(~np.isnan(temp_data))
        humidity_data_clean = humidity_data.where(~np.isnan(humidity_data))
        
        print(f"\n2. Temperature and Humidity Statistics:")
        print(f"   Temperature range: {float(temp_data_clean.min()):.2f} to {float(temp_data_clean.max()):.2f} K")
        print(f"   Relative humidity range: {float(humidity_data_clean.min()):.3f} to {float(humidity_data_clean.max()):.3f} (dimensionless)")
        
        print(f"\n3. Criteria Analysis:")
        print(f"   Analyzing when T <= {temperature_threshold:.2f} K AND relative_humidity >= {humidity_threshold:.3f}...")
        
        # Initialize criteria exceedance matrix
        criteria_matrix = np.zeros((n_south_north, n_west_east))
        
        # Calculate criteria exceedance for each grid cell
        total_cells = n_south_north * n_west_east
        processed_cells = 0
        
        for i in range(n_south_north):
            for j in range(n_west_east):
                # Extract time series for this grid cell
                cell_temp = temp_data_clean.isel(south_north=i, west_east=j)
                cell_humidity = humidity_data_clean.isel(south_north=i, west_east=j)
                
                temp_values = cell_temp.values
                humidity_values = cell_humidity.values
                
                # Remove NaN values from both arrays
                valid_mask = ~(np.isnan(temp_values) | np.isnan(humidity_values))
                temp_clean = temp_values[valid_mask]
                humidity_clean = humidity_values[valid_mask]
                
                if len(temp_clean) > 0 and len(humidity_clean) > 0:
                    # Count timesteps meeting both criteria
                    temp_criteria = temp_clean <= temperature_threshold
                    humidity_criteria = humidity_clean >= humidity_threshold
                    both_criteria = temp_criteria & humidity_criteria
                    
                    criteria_count = np.sum(both_criteria)
                    
                    # Convert to hours per year
                    hours_per_year = (criteria_count * time_step_hours) / n_years
                    
                    criteria_matrix[i, j] = hours_per_year
                else:
                    criteria_matrix[i, j] = np.nan
                
                processed_cells += 1
                if processed_cells % 20 == 0:
                    print(f"   Processed {processed_cells}/{total_cells} cells...")
        
        # Calculate statistics
        valid_criteria = criteria_matrix[~np.isnan(criteria_matrix)]
        
        print(f"\n4. Criteria Statistics:")
        if len(valid_criteria) > 0:
            print(f"   Mean criteria met: {np.mean(valid_criteria):.2f} hours/year")
            print(f"   Std criteria met: {np.std(valid_criteria):.2f} hours/year")
            print(f"   Min criteria met: {np.min(valid_criteria):.2f} hours/year")
            print(f"   Max criteria met: {np.max(valid_criteria):.2f} hours/year")
            print(f"   Cells meeting criteria: {np.sum(valid_criteria > 0)}/{len(valid_criteria)}")
        else:
            print(f"   No valid criteria data found")
        
        # Create the spatial plot
        print(f"\n5. Creating spatial criteria map...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Create the main plot with synchronized color scale
        vmin = custom_vmin if custom_vmin is not None else 0
        vmax = custom_vmax if custom_vmax is not None else None
        im = ax.imshow(criteria_matrix, cmap=colormap, origin='lower', 
                      interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set title and labels
        ax.set_title(f'Temperature-Humidity Criteria Exceedance Map\n'
                    f'T ≤ {temperature_threshold:.2f} K AND relative_humidity ≥ {humidity_threshold:.3f}\n'
                    f'Mean Annual Hours Meeting Criteria', fontsize=28)
        ax.set_xlabel('West-East Grid Points', fontsize=24)
        ax.set_ylabel('South-North Grid Points', fontsize=24)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Hours/Year Meeting Criteria', fontsize=16)
        cbar.ax.tick_params(labelsize=15)
        
        # Set axis tick label size
        ax.tick_params(axis='both', labelsize=20)
        
        # Add grid lines
        ax.set_xticks(range(n_west_east))
        ax.set_yticks(range(n_south_north))
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Add coordinate references
        ax.set_xticks(range(n_west_east))
        ax.set_yticks(range(n_south_north))
        ax.set_xticklabels([f'{j}' for j in range(n_west_east)])
        ax.set_yticklabels([f'{i}' for i in range(n_south_north)])
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            # Create directory structure: results/figures/spatial_gradient/temp_hum_criteria_exceedance/criteria_{height}_{temp}K_{hum}kgkg
            height_m = int(dataset.height.values[height_level])
            
            # Format thresholds for directory name
            temp_str = f"{temperature_threshold:.2f}".replace('.', 'p')
            hum_str = f"{humidity_threshold:.3f}".rstrip('0').rstrip('.').replace('.', 'p')
            
            # Create directory structure based on BigDomain flag
            if BigDomain and OffOn:
                base_dir = os.path.join(figures_dir, "BigDomain", OffOn, "spatial_gradient", "temp_hum_criteria_exceedance")
            else:
                base_dir = os.path.join(figures_dir, "spatial_gradient", "temp_hum_criteria_exceedance")
            
            specific_dir = f"criteria_{height_m}_{temp_str}K_{hum_str}rh"
            criteria_plots_dir = os.path.join(base_dir, specific_dir)
            os.makedirs(criteria_plots_dir, exist_ok=True)
            
            print(f"   Saving plots to: {criteria_plots_dir}")
            
            # Create filename
            plot_path = os.path.join(criteria_plots_dir, f"temp_hum_criteria_{temp_str}K_{hum_str}rh.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   Criteria map saved to: {plot_path}")
        
        plt.close()  # Close the plot to prevent it from showing
        
        # Create cartopy terrain map with criteria exceedance (similar to analyze_ice_load_with_weighted_neighborhood_cdf)
        print(f"\\n6. Creating cartopy terrain map with criteria exceedance...")
        
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            import cartopy.io.img_tiles as cimgt
            
            # Get geographical coordinates
            if 'XLAT' in dataset.coords and 'XLON' in dataset.coords:
                lats = dataset.coords['XLAT'].values
                lons = dataset.coords['XLON'].values
            elif 'XLAT' in dataset.data_vars and 'XLON' in dataset.data_vars:
                lats = dataset['XLAT'].values
                lons = dataset['XLON'].values
            else:
                raise ValueError("No latitude/longitude coordinates found in dataset")
            
            print(f"   Grid coordinates: Lat {lats.min():.3f} to {lats.max():.3f}, Lon {lons.min():.3f} to {lons.max():.3f}")
            
            # Calculate grid cell edges for pcolormesh
            lon_edges = np.zeros(lons.shape[1] + 1)
            lat_edges = np.zeros(lats.shape[0] + 1)
            
            # Longitude edges
            for j in range(lons.shape[1]):
                if j == 0:
                    lon_edges[j] = lons[0, j] - (lons[0, 1] - lons[0, 0]) / 2
                else:
                    lon_edges[j] = (lons[0, j-1] + lons[0, j]) / 2
            lon_edges[-1] = lons[0, -1] + (lons[0, -1] - lons[0, -2]) / 2
            
            # Latitude edges
            for i in range(lats.shape[0]):
                if i == 0:
                    lat_edges[i] = lats[i, 0] - (lats[1, 0] - lats[0, 0]) / 2
                else:
                    lat_edges[i] = (lats[i-1, 0] + lats[i, 0]) / 2
            lat_edges[-1] = lats[-1, 0] + (lats[-1, 0] - lats[-2, 0]) / 2
            
            # Create figure with cartopy projection
            fig = plt.figure(figsize=(16, 12))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            # Set extent with margin
            grid_center_lon = (lon_edges.min() + lon_edges.max()) / 2
            grid_center_lat = (lat_edges.min() + lat_edges.max()) / 2
            grid_span_lon = lon_edges.max() - lon_edges.min()
            grid_span_lat = lat_edges.max() - lat_edges.min()
            extent_span_lon = grid_span_lon + 2 * margin_degrees
            extent_span_lat = grid_span_lat + 2 * margin_degrees
            
            west = grid_center_lon - extent_span_lon / 2
            east = grid_center_lon + extent_span_lon / 2
            south = grid_center_lat - extent_span_lat / 2
            north = grid_center_lat + extent_span_lat / 2
            
            ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
            
            # Add geographical features
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.7, zorder=1)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.8, zorder=2)
            ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.8, zorder=3)
            
            # Try to add terrain background
            try:
                terrain = cimgt.OSM()
                ax.add_image(terrain, zoom_level)
                print(f"   Successfully loaded OpenStreetMap tiles")
            except Exception as e:
                try:
                    terrain = cimgt.GoogleTiles(style='satellite')
                    ax.add_image(terrain, zoom_level)
                    print(f"   Successfully loaded Google satellite tiles")
                except Exception as e2:
                    print(f"   Tile services unavailable, using basic land/ocean features")
            
            # Add geographical features on top
            ax.add_feature(cfeature.BORDERS, linewidth=1.5, color='black', alpha=0.8, zorder=8)
            ax.add_feature(cfeature.COASTLINE, linewidth=2, color='black', alpha=0.9, zorder=9)
            
            # Create meshgrid for pcolormesh
            lon_mesh, lat_mesh = np.meshgrid(lon_edges, lat_edges)
            
            # Calculate percentiles for better color scaling
            valid_criteria_values = criteria_matrix[~np.isnan(criteria_matrix)]
            if len(valid_criteria_values) > 0:
                data_min = np.min(valid_criteria_values)
                data_max = np.max(valid_criteria_values)
                data_mean = np.mean(valid_criteria_values)
                data_90p = np.percentile(valid_criteria_values, 90)
                
                print(f"   Criteria statistics:")
                print(f"     Min: {data_min:.1f}, Max: {data_max:.1f}, Mean: {data_mean:.1f} hours/year")
                print(f"     90th percentile: {data_90p:.1f} hours/year")
                
                # Use custom color scale if provided, otherwise use automatic scaling
                if custom_vmin is not None and custom_vmax is not None:
                    vmin = custom_vmin
                    vmax = custom_vmax
                    outlier_clipped = False  # Clipping already handled in main script
                    print(f"   Using provided color scale: {vmin:.1f} - {vmax:.1f} hours/year")
                else:
                    # Apply automatic scaling with outlier detection
                    outlier_ratio = data_max / data_90p if data_90p > 0 else 1
                    if outlier_ratio > 2.0:
                        vmin = data_min
                        vmax = data_90p
                        outlier_clipped = True
                        print(f"   Using 90th percentile clipping for better visualization: {vmin:.1f} - {vmax:.1f} hours/year")
                    else:
                        vmin = data_min
                        vmax = data_max
                        outlier_clipped = False
                        print(f"   Using full data range: {vmin:.1f} - {vmax:.1f} hours/year")
            else:
                vmin, vmax = (custom_vmin, custom_vmax) if (custom_vmin is not None and custom_vmax is not None) else (0, 1)
                outlier_clipped = False
            
            # Plot criteria values as semi-transparent overlay
            criteria_plot = ax.pcolormesh(
                lon_mesh, lat_mesh, criteria_matrix,
                cmap=colormap, alpha=0.8,
                vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree(),
                zorder=7
            )
            
            # Add colorbar
            cbar = plt.colorbar(criteria_plot, ax=ax, shrink=0.8, pad=0.02)
            if outlier_clipped:
                cbar_label = f'Hours/Year Meeting Criteria\\n[Clipped at 90th percentile: {vmax:.1f}]'
            else:
                cbar_label = 'Hours/Year Meeting Criteria'
            cbar.set_label(cbar_label, fontsize=24)
            cbar.ax.tick_params(labelsize=30)
            
            # Add gridlines with labels
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 30, 'color': 'black'}
            gl.ylabel_style = {'size': 30, 'color': 'black'}
            
            # Add title
            title_text = (f'Temperature-Humidity Criteria Exceedance on Terrain Map\\n'
                         f'T ≤ {temperature_threshold:.2f} K AND RH ≥ {humidity_threshold:.3f}, '
                         f'Height: {dataset.height.values[height_level]} m')
            ax.set_title(title_text, fontsize=28, weight='bold', pad=20)
            
            # Add statistics information
            if len(valid_criteria_values) > 0:
                if outlier_clipped:
                    info_text = (f"Range: {data_min:.1f} - {data_max:.1f} hours/year | "
                                f"Mean: {data_mean:.1f} hours/year\\n"
                                f"Color scale: {vmin:.1f} - {vmax:.1f} hours/year (90th percentile clipped)")
                else:
                    info_text = (f"Range: {data_min:.1f} - {data_max:.1f} hours/year | "
                                f"Mean: {data_mean:.1f} hours/year")
                ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=27,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            
            plt.tight_layout()
            
            # Save the cartopy terrain map
            if save_plots:
                cartopy_filename = f"temp_hum_criteria_cartopy_{temp_str}K_{hum_str}rh.png"
                cartopy_path = os.path.join(criteria_plots_dir, cartopy_filename)
                plt.savefig(cartopy_path, dpi=300, bbox_inches='tight')
                print(f"   Cartopy terrain map saved to: {cartopy_path}")
            
            plt.close()  # Close the plot to prevent it from showing
            
        except ImportError as e:
            print(f"   Warning: Cartopy not available, skipping terrain map: {e}")
        except Exception as e:
            print(f"   Error creating cartopy terrain map: {e}")
        
        # Create additional summary statistics plot
        if len(valid_criteria) > 0:
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram of criteria values
            ax1.hist(valid_criteria, bins=min(20, len(np.unique(valid_criteria))), 
                    alpha=0.7, color='lightgreen', edgecolor='black')
            ax1.set_xlabel('Hours/Year Meeting Criteria', fontsize=24)
            ax1.set_ylabel('Number of Grid Cells', fontsize=24)
            ax1.set_title(f'Distribution of Criteria Exceedances\n'
                         f'T ≤ {temperature_threshold:.2f} K AND relative_humidity ≥ {humidity_threshold:.3f}')
            ax1.grid(True, alpha=0.3)
            
            # Box plot by row (south-north variation)
            row_data = []
            row_labels = []
            for i in range(n_south_north):
                row_criteria = criteria_matrix[i, :]
                valid_row = row_criteria[~np.isnan(row_criteria)]
                if len(valid_row) > 0:
                    row_data.append(valid_row)
                    row_labels.append(f'Row {i}')
            
            if row_data:
                ax2.boxplot(row_data, labels=row_labels)
                ax2.set_xlabel('Grid Row (South to North)', fontsize=24)
                ax2.set_ylabel('Hours/Year Meeting Criteria', fontsize=24)
                ax2.set_title('Criteria Exceedance by Grid Row', fontsize=28)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_plots:
                summary_path = os.path.join(criteria_plots_dir, f"temp_hum_criteria_summary_{temp_str}K_{hum_str}rh.png")
                plt.savefig(summary_path, dpi=300, bbox_inches='tight')
                print(f"   Summary statistics saved to: {summary_path}")
            
            plt.close()  # Close the plot to prevent it from showing
        
        # Prepare results dictionary
        results = {
            'temperature_threshold': temperature_threshold,
            'humidity_threshold': humidity_threshold,
            'height_level': height_level,
            'criteria_matrix': criteria_matrix,
            'grid_shape': (n_south_north, n_west_east),
            'n_years': n_years,
            'years_range': (years[0], years[-1]),
            'time_step_hours': time_step_hours,
            'statistics': {
                'mean': np.nanmean(criteria_matrix),
                'std': np.nanstd(criteria_matrix),
                'min': np.nanmin(criteria_matrix),
                'max': np.nanmax(criteria_matrix),
                'cells_meeting_criteria': np.sum(valid_criteria > 0) if len(valid_criteria) > 0 else 0,
                'total_valid_cells': len(valid_criteria) if len(valid_criteria) > 0 else 0
            }
        }
        
        # Save detailed results to file
        if save_plots:
            results_path = os.path.join(criteria_plots_dir, f"temp_hum_criteria_results_{temp_str}K_{hum_str}rh.txt")
            with open(results_path, 'w') as f:
                f.write("TEMPERATURE-HUMIDITY CRITERIA ANALYSIS RESULTS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Temperature threshold: <= {temperature_threshold:.2f} K\n")
                f.write(f"Relative humidity threshold: >= {humidity_threshold:.3f} (dimensionless)\n")
                f.write(f"Height level: {height_level} ({dataset.height.values[height_level]} m)\n")
                f.write(f"Grid shape: {n_south_north} × {n_west_east}\n")
                f.write(f"Years analyzed: {n_years} ({years[0]} to {years[-1]})\n")
                f.write(f"Time step: {time_step_hours} hours\n\n")
                
                f.write("Criteria Statistics:\n")
                f.write("-" * 20 + "\n")
                if len(valid_criteria) > 0:
                    f.write(f"Mean: {results['statistics']['mean']:.3f} hours/year\n")
                    f.write(f"Std: {results['statistics']['std']:.3f} hours/year\n")
                    f.write(f"Min: {results['statistics']['min']:.3f} hours/year\n")
                    f.write(f"Max: {results['statistics']['max']:.3f} hours/year\n")
                    f.write(f"Cells meeting criteria: {results['statistics']['cells_meeting_criteria']}\n")
                    f.write(f"Total valid cells: {results['statistics']['total_valid_cells']}\n\n")
                
                f.write("Grid Cell Criteria Values (hours/year):\n")
                f.write("-" * 40 + "\n")
                for i in range(n_south_north):
                    row_str = f"Row {i:2d}: "
                    for j in range(n_west_east):
                        value = criteria_matrix[i, j]
                        if np.isnan(value):
                            row_str += "   NaN   "
                        else:
                            row_str += f"{value:7.2f} "
                    f.write(row_str + "\n")
            
            print(f"   Detailed results saved to: {results_path}")
        
        print(f"\n✓ Temperature-humidity criteria analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in temperature-humidity criteria analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    
    # Create grid coordinates for consistent orientation
    x_coords = np.arange(max_ice_2d.shape[1])  # west_east dimension
    y_coords = np.arange(max_ice_2d.shape[0])  # south_north dimension
    
    # Original maximum ice load
    im1 = axes[0,0].pcolormesh(x_coords, y_coords, max_ice_2d, cmap='Blues', shading='auto')
    axes[0,0].set_title('Maximum Ice Load', fontsize=28)
    axes[0,0].set_xticks(range(len(x_coords)))
    axes[0,0].set_xticklabels(range(1, len(x_coords)+1))
    axes[0,0].set_yticks(range(len(y_coords)))
    axes[0,0].set_yticklabels(range(1, len(y_coords)+1))
    axes[0,0].set_xlabel('West-East Grid Points', fontsize=24)
    axes[0,0].set_ylabel('South-North Grid Points', fontsize=24)
    plt.colorbar(im1, ax=axes[0,0], label='Ice Load (kg/m)')
    
    # X-direction gradient (West-East)
    im2 = axes[0,1].pcolormesh(x_coords, y_coords, grad_x, cmap='RdBu_r', shading='auto')
    axes[0,1].set_title('Spatial Gradient (West-East)', fontsize=28)
    axes[0,1].set_xticks(range(len(x_coords)))
    axes[0,1].set_xticklabels(range(1, len(x_coords)+1))
    axes[0,1].set_yticks(range(len(y_coords)))
    axes[0,1].set_yticklabels(range(1, len(y_coords)+1))
    axes[0,1].set_xlabel('West-East Grid Points', fontsize=24)
    axes[0,1].set_ylabel('South-North Grid Points', fontsize=24)
    plt.colorbar(im2, ax=axes[0,1], label='Gradient (kg/m per grid)')
    
    # Y-direction gradient (South-North)
    im3 = axes[1,0].pcolormesh(x_coords, y_coords, grad_y, cmap='RdBu_r', shading='auto')
    axes[1,0].set_title('Spatial Gradient (South-North)', fontsize=28)
    axes[1,0].set_xticks(range(len(x_coords)))
    axes[1,0].set_xticklabels(range(1, len(x_coords)+1))
    axes[1,0].set_yticks(range(len(y_coords)))
    axes[1,0].set_yticklabels(range(1, len(y_coords)+1))
    axes[1,0].set_xlabel('West-East Grid Points', fontsize=24)
    axes[1,0].set_ylabel('South-North Grid Points', fontsize=24)
    plt.colorbar(im3, ax=axes[1,0], label='Gradient (kg/m per grid)')
    
    # Gradient magnitude
    im4 = axes[1,1].pcolormesh(x_coords, y_coords, gradient_magnitude, cmap='plasma', shading='auto')
    axes[1,1].set_title('Gradient Magnitude', fontsize=28)
    axes[1,1].set_xticks(range(len(x_coords)))
    axes[1,1].set_xticklabels(range(1, len(x_coords)+1))
    axes[1,1].set_yticks(range(len(y_coords)))
    axes[1,1].set_yticklabels(range(1, len(y_coords)+1))
    axes[1,1].set_xlabel('West-East Grid Points', fontsize=24)
    axes[1,1].set_ylabel('South-North Grid Points', fontsize=24)
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
    axes[0,0].set_title('Mean Ice Load Over Time', fontsize=28)
    axes[0,0].set_xlabel('West-East', fontsize=24)
    axes[0,0].set_ylabel('South-North', fontsize=24)
    plt.colorbar(im1, ax=axes[0,0], label='Mean Ice Load (kg/m)')
    
    # Mean X-direction gradient (West-East)
    im2 = axes[0,1].imshow(mean_grad_x, cmap='RdBu_r', aspect='auto')
    axes[0,1].set_title('Mean Spatial Gradient (West-East)', fontsize=28)
    axes[0,1].set_xlabel('West-East', fontsize=24)
    axes[0,1].set_ylabel('South-North', fontsize=24)
    plt.colorbar(im2, ax=axes[0,1], label='Mean Gradient (kg/m per grid)')
    
    # Mean Y-direction gradient (South-North)
    im3 = axes[1,0].imshow(mean_grad_y, cmap='RdBu_r', aspect='auto')
    axes[1,0].set_title('Mean Spatial Gradient (South-North)', fontsize=28)
    axes[1,0].set_xlabel('West-East', fontsize=24)
    axes[1,0].set_ylabel('South-North', fontsize=24)
    plt.colorbar(im3, ax=axes[1,0], label='Mean Gradient (kg/m per grid)')
    
    # Mean gradient magnitude
    im4 = axes[1,1].imshow(mean_gradient_magnitude, cmap='plasma', aspect='auto')
    axes[1,1].set_title('Mean Gradient Magnitude', fontsize=28)
    axes[1,1].set_xlabel('West-East', fontsize=24)
    axes[1,1].set_ylabel('South-North', fontsize=24)
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
        ax1.set_title('Maximum Ice Load (Geographical)', fontsize=28)
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
        ax2.set_title('Mean Ice Load (Geographical)', fontsize=28)
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
        ax3.set_title('Model Grid Points', fontsize=28)
        
        # Plot 4: Ice load with contours
        ax4 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
        im4 = ax4.pcolormesh(lons, lats, max_ice_load, 
                            cmap='Blues', transform=ccrs.PlateCarree(), alpha=0.8)
        contours = ax4.contour(lons, lats, max_ice_load, levels=10, 
                              colors='black', alpha=0.6, transform=ccrs.PlateCarree())
        ax4.clabel(contours, inline=True, fontsize=24)
        ax4.add_feature(cfeature.COASTLINE, alpha=0.7)
        ax4.add_feature(cfeature.BORDERS, alpha=0.5)
        ax4.gridlines(draw_labels=True, alpha=0.5)
        ax4.set_title('Maximum Ice Load with Contours', fontsize=28)
        plt.colorbar(im4, ax=ax4, label='Ice Load (kg/m)', shrink=0.8)
        
    else:
        # Create basic maps without cartopy
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Maximum ice load
        im1 = axes[0,0].pcolormesh(lons, lats, max_ice_load, cmap='Blues')
        axes[0,0].set_title('Maximum Ice Load', fontsize=28)
        axes[0,0].set_xlabel('Longitude', fontsize=24)
        axes[0,0].set_ylabel('Latitude', fontsize=24)
        plt.colorbar(im1, ax=axes[0,0], label='Ice Load (kg/m)')
        
        # Plot 2: Mean ice load
        im2 = axes[0,1].pcolormesh(lons, lats, mean_ice_load, cmap='Blues')
        axes[0,1].set_title('Mean Ice Load', fontsize=28)
        axes[0,1].set_xlabel('Longitude', fontsize=24)
        axes[0,1].set_ylabel('Latitude', fontsize=24)
        plt.colorbar(im2, ax=axes[0,1], label='Ice Load (kg/m)')
        
        # Plot 3: Grid points
        axes[1,0].scatter(lons.flatten(), lats.flatten(), s=1, c='red', alpha=0.6)
        axes[1,0].set_title('Model Grid Points', fontsize=28)
        axes[1,0].set_xlabel('Longitude', fontsize=24)
        axes[1,0].set_ylabel('Latitude', fontsize=24)
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Ice load with contours
        im4 = axes[1,1].pcolormesh(lons, lats, max_ice_load, cmap='Blues', alpha=0.8)
        contours = axes[1,1].contour(lons, lats, max_ice_load, levels=10, colors='black', alpha=0.6)
        axes[1,1].clabel(contours, inline=True, fontsize=24)
        axes[1,1].set_title('Maximum Ice Load with Contours', fontsize=28)
        axes[1,1].set_xlabel('Longitude', fontsize=24)
        axes[1,1].set_ylabel('Latitude', fontsize=24)
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
    axes[0,0].set_title('Latitude Grid', fontsize=28)
    axes[0,0].set_xlabel('West-East Grid Index', fontsize=24)
    axes[0,0].set_ylabel('South-North Grid Index', fontsize=24)
    plt.colorbar(im1, ax=axes[0,0], label='Latitude (degrees)')
    
    # Plot 2: Longitude grid
    im2 = axes[0,1].imshow(lons, cmap='plasma', aspect='auto')
    axes[0,1].set_title('Longitude Grid', fontsize=28)
    axes[0,1].set_xlabel('West-East Grid Index', fontsize=24)
    axes[0,1].set_ylabel('South-North Grid Index', fontsize=24)
    plt.colorbar(im2, ax=axes[0,1], label='Longitude (degrees)')
    
    # Plot 3: Grid spacing in latitude
    lat_spacing = np.diff(lats, axis=0)
    if lat_spacing.size > 0:
        im3 = axes[1,0].imshow(lat_spacing, cmap='RdYlBu', aspect='auto')
        axes[1,0].set_title('Latitude Grid Spacing', fontsize=28)
        axes[1,0].set_xlabel('West-East Grid Index', fontsize=24)
        axes[1,0].set_ylabel('South-North Grid Index', fontsize=24)
        plt.colorbar(im3, ax=axes[1,0], label='Lat Spacing (degrees)')
    
    # Plot 4: Grid spacing in longitude
    lon_spacing = np.diff(lons, axis=1)
    if lon_spacing.size > 0:
        im4 = axes[1,1].imshow(lon_spacing, cmap='RdYlBu', aspect='auto')
        axes[1,1].set_title('Longitude Grid Spacing', fontsize=28)
        axes[1,1].set_xlabel('West-East Grid Index', fontsize=24)
        axes[1,1].set_ylabel('South-North Grid Index', fontsize=24)
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
                    ax.set_xlabel('Ice Load (kg/m)', fontsize=24)
                    ax.set_ylabel('Hours/Year', fontsize=24)
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
            
            plt.xlabel('Ice Load (kg/m)', fontsize=24)
            plt.ylabel('Hours per Year', fontsize=24)
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
            axes[0, 0].set_xlabel('West-East Grid Points', fontsize=24)
            axes[0, 0].set_ylabel('South-North Grid Points', fontsize=24)
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
            axes[0, 1].set_xlabel('West-East Grid Points', fontsize=24)
            axes[0, 1].set_ylabel('South-North Grid Points', fontsize=24)
            cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
            cbar2.set_label('Earth Mover\'s Distance')
            
            axes[0, 1].set_xticks(range(n_west_east))
            axes[0, 1].set_yticks(range(n_south_north-1))
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Combined gradients
            im3 = axes[1, 0].imshow(combined_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 0].set_title('Combined Spatial Gradient\n(Average Neighbor Distance)', fontsize=28)
            axes[1, 0].set_xlabel('West-East Grid Points', fontsize=24)
            axes[1, 0].set_ylabel('South-North Grid Points', fontsize=24)
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
            axes[1, 1].set_title('Gradient Magnitude\n(RMS of EW and SN)', fontsize=28)
            axes[1, 1].set_xlabel('West-East Grid Points', fontsize=24)
            axes[1, 1].set_ylabel('South-North Grid Points', fontsize=24)
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
                    ax.set_xlabel('Ice Load (kg/m)', fontsize=24)
                    ax.set_ylabel('Probability Density', fontsize=24)
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
            
            plt.xlabel('Ice Load (kg/m)', fontsize=24)
            plt.ylabel('Probability Density', fontsize=24)
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
            axes[0, 0].set_xlabel('West-East Grid Points', fontsize=24)
            axes[0, 0].set_ylabel('South-North Grid Points', fontsize=24)
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
            axes[0, 1].set_xlabel('West-East Grid Points', fontsize=24)
            axes[0, 1].set_ylabel('South-North Grid Points', fontsize=24)
            cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
            cbar2.set_label('Earth Mover\'s Distance')
            
            axes[0, 1].set_xticks(range(n_west_east))
            axes[0, 1].set_yticks(range(n_south_north-1))
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Combined gradients
            im3 = axes[1, 0].imshow(combined_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 0].set_title('Combined Spatial Gradient\n(Average Neighbor Distance)', fontsize=28)
            axes[1, 0].set_xlabel('West-East Grid Points', fontsize=24)
            axes[1, 0].set_ylabel('South-North Grid Points', fontsize=24)
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
            axes[1, 1].set_title('Gradient Magnitude\n(RMS of EW and SN)', fontsize=28)
            axes[1, 1].set_xlabel('West-East Grid Points', fontsize=24)
            axes[1, 1].set_ylabel('South-North Grid Points', fontsize=24)
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
    It creates a CDF per cell using all data available, then compare CDFs across the domain.
    
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
                    ax.set_xlabel('Ice Load (kg/m)', fontsize=24)
                    ax.set_ylabel('Cumulative Probability', fontsize=24)
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
            
            plt.xlabel('Ice Load (kg/m)', fontsize=24)
            plt.ylabel('Cumulative Probability', fontsize=24)
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
            axes[0, 0].set_xlabel('West-East Grid Points', fontsize=24)
            axes[0, 0].set_ylabel('South-North Grid Points', fontsize=24)
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
            axes[0, 1].set_xlabel('West-East Grid Points', fontsize=24)
            axes[0, 1].set_ylabel('South-North Grid Points', fontsize=24)
            cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
            cbar2.set_label('Earth Mover\'s Distance (kg/m)')
            
            axes[0, 1].set_xticks(range(n_west_east))
            axes[0, 1].set_yticks(range(n_south_north-1))
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Combined gradients
            im3 = axes[1, 0].imshow(combined_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 0].set_title('Combined Spatial Gradient\n(Average Neighbor Distance)', fontsize=28)
            axes[1, 0].set_xlabel('West-East Grid Points', fontsize=24)
            axes[1, 0].set_ylabel('South-North Grid Points', fontsize=24)
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
            axes[1, 1].set_title('Gradient Magnitude\n(RMS of EW and SN)', fontsize=28)
            axes[1, 1].set_xlabel('West-East Grid Points', fontsize=24)
            axes[1, 1].set_ylabel('South-North Grid Points', fontsize=24)
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
            axes_norm[0, 0].set_title('East-West Gradient\n(Dimensionless, Relative to Domain Mean)', fontsize=28)
            axes_norm[0, 0].set_xlabel('West-East Grid Points', fontsize=24)
            axes_norm[0, 0].set_ylabel('South-North Grid Points', fontsize=24)
            cbar1_norm = plt.colorbar(im1_norm, ax=axes_norm[0, 0], shrink=0.8)
            cbar1_norm.set_label('Gradient / Domain Mean')
            
            # Add grid lines
            axes_norm[0, 0].set_xticks(range(n_west_east-1))
            axes_norm[0, 0].set_yticks(range(n_south_north))
            axes_norm[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: South-North gradients (normalized)
            im2_norm = axes_norm[0, 1].imshow(sn_gradients_normalized, cmap='RdBu_r', origin='lower', 
                                             interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
            axes_norm[0, 1].set_title('South-North Gradient\n(Dimensionless, Relative to Domain Mean)', fontsize=28)
            axes_norm[0, 1].set_xlabel('West-East Grid Points', fontsize=24)
            axes_norm[0, 1].set_ylabel('South-North Grid Points', fontsize=24)
            cbar2_norm = plt.colorbar(im2_norm, ax=axes_norm[0, 1], shrink=0.8)
            cbar2_norm.set_label('Gradient / Domain Mean')
            
            axes_norm[0, 1].set_xticks(range(n_west_east))
            axes_norm[0, 1].set_yticks(range(n_south_north-1))
            axes_norm[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Combined gradients (normalized)
            im3_norm = axes_norm[1, 0].imshow(combined_gradients_normalized, cmap='RdBu_r', origin='lower', 
                                             interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
            axes_norm[1, 0].set_title('Combined Spatial Gradient\n(Dimensionless, Relative to Domain Mean)', fontsize=28)
            axes_norm[1, 0].set_xlabel('West-East Grid Points', fontsize=24)
            axes_norm[1, 0].set_ylabel('South-North Grid Points', fontsize=24)
            cbar3_norm = plt.colorbar(im3_norm, ax=axes_norm[1, 0], shrink=0.8)
            cbar3_norm.set_label('Gradient / Domain Mean')
            
            axes_norm[1, 0].set_xticks(range(n_west_east))
            axes_norm[1, 0].set_yticks(range(n_south_north))
            axes_norm[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Gradient magnitude (normalized)
            im4_norm = axes_norm[1, 1].imshow(gradient_magnitude_normalized, cmap='RdBu_r', origin='lower', 
                                             interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
            axes_norm[1, 1].set_title('Gradient Magnitude\n(Dimensionless, Relative to Domain Mean)', fontsize=28)
            axes_norm[1, 1].set_xlabel('West-East Grid Points', fontsize=24)
            axes_norm[1, 1].set_ylabel('South-North Grid Points', fontsize=24)
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
                    ax.set_xlabel('Ice Load (kg/m, log scale)', fontsize=24)
                    ax.set_ylabel('Cumulative Probability (log scale)', fontsize=24)
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
            
            plt.xlabel('Ice Load (kg/m, log scale)', fontsize=24)
            plt.ylabel('Cumulative Probability (log scale)', fontsize=24)
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
            axes[0, 0].set_title('East-West Gradient\n(CDF L1 Distance, Log X-axis)', fontsize=28)
            axes[0, 0].set_xlabel('West-East Grid Points', fontsize=24)
            axes[0, 0].set_ylabel('South-North Grid Points', fontsize=24)
            cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
            cbar1.set_label('L1 Distance (Log X-axis)')
            
            # Add grid lines
            axes[0, 0].set_xticks(range(n_west_east-1))
            axes[0, 0].set_yticks(range(n_south_north))
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: South-North gradients
            im2 = axes[0, 1].imshow(sn_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[0, 1].set_title('South-North Gradient\n(CDF L1 Distance, Log X-axis)', fontsize=28)
            axes[0, 1].set_xlabel('West-East Grid Points', fontsize=24)
            axes[0, 1].set_ylabel('South-North Grid Points', fontsize=24)
            cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
            cbar2.set_label('L1 Distance (Log X-axis)')
            
            axes[0, 1].set_xticks(range(n_west_east))
            axes[0, 1].set_yticks(range(n_south_north-1))
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Combined gradients
            im3 = axes[1, 0].imshow(combined_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 0].set_title('Combined Spatial Gradient\n(Average Neighbor Distance)', fontsize=28)
            axes[1, 0].set_xlabel('West-East Grid Points', fontsize=24)
            axes[1, 0].set_ylabel('South-North Grid Points', fontsize=24)
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
            axes[1, 1].set_title('Gradient Magnitude\n(RMS of EW and SN)', fontsize=28)
            axes[1, 1].set_xlabel('West-East Grid Points', fontsize=24)
            axes[1, 1].set_ylabel('South-North Grid Points', fontsize=24)
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
                    ax.set_xlabel('Ice Load (kg/m)', fontsize=24)
                    ax.set_ylabel('Exceedance Probability', fontsize=24)
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
            
            plt.xlabel('Ice Load (kg/m)', fontsize=24)
            plt.ylabel('Exceedance Probability', fontsize=24)
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
            axes[0, 0].set_title('East-West Gradient\n(Exceedance L1 Distance)', fontsize=28)
            axes[0, 0].set_xlabel('West-East Grid Points', fontsize=24)
            axes[0, 0].set_ylabel('South-North Grid Points', fontsize=24)
            cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
            cbar1.set_label('L1 Distance')
            
            # Add grid lines
            axes[0, 0].set_xticks(range(n_west_east-1))
            axes[0, 0].set_yticks(range(n_south_north))
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: South-North gradients
            im2 = axes[0, 1].imshow(sn_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[0, 1].set_title('South-North Gradient\n(Exceedance L1 Distance)', fontsize=28)
            axes[0, 1].set_xlabel('West-East Grid Points', fontsize=24)
            axes[0, 1].set_ylabel('South-North Grid Points', fontsize=24)
            cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
            cbar2.set_label('L1 Distance')
            
            axes[0, 1].set_xticks(range(n_west_east))
            axes[0, 1].set_yticks(range(n_south_north-1))
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Combined gradients
            im3 = axes[1, 0].imshow(combined_gradients, cmap='viridis', origin='lower', 
                                   interpolation='nearest', aspect='auto')
            axes[1, 0].set_title('Combined Spatial Gradient\n(Average Neighbor Distance)', fontsize=28)
            axes[1, 0].set_xlabel('West-East Grid Points', fontsize=24)
            axes[1, 0].set_ylabel('South-North Grid Points', fontsize=24)
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
            axes[1, 1].set_title('Gradient Magnitude\n(RMS of EW and SN)', fontsize=28)
            axes[1, 1].set_xlabel('West-East Grid Points', fontsize=24)
            axes[1, 1].set_ylabel('South-North Grid Points', fontsize=24)
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
    percentile=None,
    OffOn=None,
    BigDomain=False
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
    OffOn : str, optional
        Specifies 'Onshore' or 'Offshore' for BigDomain directory structure
    BigDomain : bool, default False
        If True, saves results to MT_Icing/results/figures/BigDomain/{OffOn}/filters...
        
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
        if BigDomain and OffOn:
            filters_base_dir = os.path.join("results", "figures", "BigDomain", OffOn, "filters")
        else:
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
        axes[0, 0].set_xlabel('West-East Grid Points', fontsize=24)
        axes[0, 0].set_ylabel('South-North Grid Points', fontsize=24)
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
        axes[0, 1].set_xlabel('West-East Grid Points', fontsize=24)
        axes[0, 1].set_ylabel('South-North Grid Points', fontsize=24)
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        cbar2.set_label('Earth Mover\'s Distance (kg/m)')
        
        axes[0, 1].set_xticks(range(n_west_east))
        axes[0, 1].set_yticks(range(n_south_north-1))
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Combined gradients
        im3 = axes[1, 0].imshow(combined_gradients, cmap='viridis', origin='lower', 
                               interpolation='nearest', aspect='auto')
        axes[1, 0].set_title('Combined Spatial Gradient\n(Average Neighbor Distance)', fontsize=28)
        axes[1, 0].set_xlabel('West-East Grid Points', fontsize=24)
        axes[1, 0].set_ylabel('South-North Grid Points', fontsize=24)
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
        axes[1, 1].set_title('Gradient Magnitude\n(RMS of EW and SN)', fontsize=28)
        axes[1, 1].set_xlabel('West-East Grid Points', fontsize=24)
        axes[1, 1].set_ylabel('South-North Grid Points', fontsize=24)
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
        axes_norm[0, 0].set_title('East-West Gradient\n(Dimensionless, Relative to Domain Mean)', fontsize=28)
        axes_norm[0, 0].set_xlabel('West-East Grid Points', fontsize=24)
        axes_norm[0, 0].set_ylabel('South-North Grid Points', fontsize=24)
        cbar1_norm = plt.colorbar(im1_norm, ax=axes_norm[0, 0], shrink=0.8)
        cbar1_norm.set_label('Gradient / Domain Mean')
        
        # Add grid lines
        axes_norm[0, 0].set_xticks(range(n_west_east-1))
        axes_norm[0, 0].set_yticks(range(n_south_north))
        axes_norm[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: South-North gradients (normalized)
        im2_norm = axes_norm[0, 1].imshow(sn_gradients_normalized, cmap='RdBu_r', origin='lower', 
                                         interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
        axes_norm[0, 1].set_title('South-North Gradient\n(Dimensionless, Relative to Domain Mean)', fontsize=28)
        axes_norm[0, 1].set_xlabel('West-East Grid Points', fontsize=24)
        axes_norm[0, 1].set_ylabel('South-North Grid Points', fontsize=24)
        cbar2_norm = plt.colorbar(im2_norm, ax=axes_norm[0, 1], shrink=0.8)
        cbar2_norm.set_label('Gradient / Domain Mean')
        
        axes_norm[0, 1].set_xticks(range(n_west_east))
        axes_norm[0, 1].set_yticks(range(n_south_north-1))
        axes_norm[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Combined gradients (normalized)
        im3_norm = axes_norm[1, 0].imshow(combined_gradients_normalized, cmap='RdBu_r', origin='lower', 
                                         interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
        axes_norm[1, 0].set_title('Combined Spatial Gradient\n(Dimensionless, Relative to Domain Mean)', fontsize=28)
        axes_norm[1, 0].set_xlabel('West-East Grid Points', fontsize=24)
        axes_norm[1, 0].set_ylabel('South-North Grid Points', fontsize=24)
        cbar3_norm = plt.colorbar(im3_norm, ax=axes_norm[1, 0], shrink=0.8)
        cbar3_norm.set_label('Gradient / Domain Mean')
        
        axes_norm[1, 0].set_xticks(range(n_west_east))
        axes_norm[1, 0].set_yticks(range(n_south_north))
        axes_norm[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Gradient magnitude (normalized)
        im4_norm = axes_norm[1, 1].imshow(gradient_magnitude_normalized, cmap='RdBu_r', origin='lower', 
                                         interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
        axes_norm[1, 1].set_title('Gradient magnitude (RMS)', fontsize=28)
        axes_norm[1, 1].set_xlabel('West-East Grid Points', fontsize=24)
        axes_norm[1, 1].set_ylabel('South-North Grid Points', fontsize=24)
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
        
        plt.xlabel('Ice Load (kg/m)', fontsize=24)
        plt.ylabel('Cumulative Probability', fontsize=24)
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

def analyze_ice_load_with_weighted_neighborhood_cdf(
    dataset_with_ice_load,
    ice_load_variable='ICE_LOAD',
    height_level=0,
    save_plots=True,
    results_subdir="weighted_neighborhood_ice_load_cdf_analysis",
    # Neighborhood parameters
    neighborhood_type='4-neighbors',  # '4-neighbors', '8-neighbors', '24-neighbors'
    weight_scheme='uniform',          # 'uniform', 'distance', 'custom'
    custom_weights=None,              # Custom weight dictionary for different neighbor types
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
    percentile=None,
    OffOn=None,
    BigDomain=False,
    # Cartopy map parameters for gradient visualization
    margin_degrees=0.2,   # Margin around grid in degrees for cartopy map
    zoom_level=6          # Zoom level for terrain tiles in cartopy map
):
    """
    Enhanced version of analyze_ice_load_with_filtering_and_cdf that allows customizable
    neighborhood analysis with different numbers of neighbors (4, 8, or 24) and 
    weighted distance calculations for spatial gradient analysis.
    
    This function combines meteorological filtering with detailed CDF analysis and creates
    spatial gradient plots using weighted Earth Mover's distance between neighboring grid cells.
    
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
        Subdirectory name for saving results (default: "weighted_neighborhood_ice_load_cdf_analysis")
        
    Neighborhood Parameters:
    ------------------------
    neighborhood_type : str, optional
        Type of neighborhood to analyze:
        - '4-neighbors': Only adjacent cells (N, S, E, W)
        - '8-neighbors': Adjacent + diagonal cells (N, S, E, W, NE, NW, SE, SW)
        - '24-neighbors': Extended neighborhood (5x5 grid around center, excluding center)
    weight_scheme : str, optional
        Weighting scheme for neighbor distances:
        - 'uniform': All neighbors have equal weight
        - 'distance': Weight inversely proportional to distance
        - 'custom': Use custom weights provided in custom_weights parameter
    custom_weights : dict, optional
        Custom weights for different neighbor types. Required if weight_scheme='custom'.
        Format: {
            'adjacent': float,      # Weight for N, S, E, W neighbors (distance 1)
            'diagonal': float,      # Weight for NE, NW, SE, SW neighbors (distance √2)
            'second_line': float,   # Weight for 2-step neighbors (distance 2)
            'second_diagonal': float # Weight for 2-step diagonal neighbors (distance 2√2)
        }
        
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
    OffOn : str, optional
        Specifies 'Onshore' or 'Offshore' for BigDomain directory structure
    BigDomain : bool, default False
        If True, saves results to MT_Icing/results/figures/BigDomain/{OffOn}/spatial_gradient...
    margin_degrees : float, optional
        Margin to add around the grid in degrees for cartopy map background (default: 0.2)
    zoom_level : int, optional
        Zoom level for terrain tiles in cartopy map (default: 6)
        Higher values = more detail but slower loading
        
    Returns:
    --------
    dict
        Comprehensive results including filtering info, CDF data, and weighted spatial gradients
    """
    
    print("=== ICE LOAD ANALYSIS WITH WEIGHTED NEIGHBORHOOD CDF ===")
    print(f"Neighborhood type: {neighborhood_type}")
    print(f"Weight scheme: {weight_scheme}")
    
    # Check if ice load variable exists
    if ice_load_variable not in dataset_with_ice_load.data_vars:
        raise ValueError(f"Ice load variable '{ice_load_variable}' not found in dataset. "
                        f"Available variables: {list(dataset_with_ice_load.data_vars.keys())}")
    
    print(f"Using ice load variable: {ice_load_variable}")
    print(f"Height level: {height_level} ({dataset_with_ice_load.height.values[height_level]} m)")
    
    # Validate neighborhood parameters
    valid_neighborhoods = ['4-neighbors', '8-neighbors', '24-neighbors']
    if neighborhood_type not in valid_neighborhoods:
        raise ValueError(f"Invalid neighborhood_type. Must be one of: {valid_neighborhoods}")
    
    valid_weights = ['uniform', 'distance', 'custom']
    if weight_scheme not in valid_weights:
        raise ValueError(f"Invalid weight_scheme. Must be one of: {valid_weights}")
    
    if weight_scheme == 'custom' and custom_weights is None:
        raise ValueError("custom_weights must be provided when weight_scheme='custom'")
    
    # Define neighbor offsets and weights for different neighborhood types
    def get_neighborhood_config(neighborhood_type, weight_scheme, custom_weights=None):
        """Get neighbor offsets and weights based on configuration"""
        
        if neighborhood_type == '4-neighbors':
            neighbor_offsets = [
                (-1, 0, 'adjacent'),  # North
                (1, 0, 'adjacent'),   # South
                (0, -1, 'adjacent'),  # West
                (0, 1, 'adjacent'),   # East
            ]
        elif neighborhood_type == '8-neighbors':
            neighbor_offsets = [
                (-1, 0, 'adjacent'),   # North
                (1, 0, 'adjacent'),    # South
                (0, -1, 'adjacent'),   # West
                (0, 1, 'adjacent'),    # East
                (-1, -1, 'diagonal'),  # Northwest
                (-1, 1, 'diagonal'),   # Northeast
                (1, -1, 'diagonal'),   # Southwest
                (1, 1, 'diagonal'),    # Southeast
            ]
        elif neighborhood_type == '24-neighbors':
            neighbor_offsets = []
            # 5x5 grid around center (excluding center itself)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    if di == 0 and dj == 0:
                        continue  # Skip center cell
                    
                    # Classify neighbor type by distance
                    distance = np.sqrt(di**2 + dj**2)
                    if distance == 1:
                        neighbor_type = 'adjacent'
                    elif distance == np.sqrt(2):
                        neighbor_type = 'diagonal'
                    elif distance == 2:
                        neighbor_type = 'second_line'
                    elif distance == 2 * np.sqrt(2):
                        neighbor_type = 'second_diagonal'
                    else:
                        neighbor_type = 'extended'
                    
                    neighbor_offsets.append((di, dj, neighbor_type))
        
        # Assign weights based on scheme
        neighbor_weights = {}
        for di, dj, ntype in neighbor_offsets:
            if weight_scheme == 'uniform':
                weight = 1.0
            elif weight_scheme == 'distance':
                distance = np.sqrt(di**2 + dj**2)
                weight = 1.0 / distance  # Inverse distance weighting
            elif weight_scheme == 'custom':
                if ntype in custom_weights:
                    weight = custom_weights[ntype]
                else:
                    print(f"Warning: No custom weight specified for neighbor type '{ntype}', using 1.0")
                    weight = 1.0
            
            neighbor_weights[(di, dj)] = weight
        
        return neighbor_offsets, neighbor_weights
    
    neighbor_offsets, neighbor_weights = get_neighborhood_config(neighborhood_type, weight_scheme, custom_weights)
    
    print(f"Using {len(neighbor_offsets)} neighbors with weights:")
    for (di, dj, ntype), weight in zip(neighbor_offsets, neighbor_weights.values()):
        print(f"  {ntype:>15} ({di:2d},{dj:2d}): weight = {weight:.3f}")
    
    # Create results directory based on filters and neighborhood config
    if save_plots:
        # Create organized directory structure
        if BigDomain and OffOn:
            filters_base_dir = os.path.join("results", "figures", "BigDomain", OffOn, "spatial_gradient", "weighted_neighborhood_filters")
        else:
            filters_base_dir = os.path.join("results", "figures", "spatial_gradient", "weighted_neighborhood_filters")
        os.makedirs(filters_base_dir, exist_ok=True)
        
        # Generate folder name based on applied filters and neighborhood config
        folder_name_parts = []
        
        # Add neighborhood configuration
        folder_name_parts.append(f"{neighborhood_type.replace('-', '')}")
        folder_name_parts.append(f"{weight_scheme}")
        
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
    
    # Apply meteorological filtering (same as original function)
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
    
    # Perform CDF analysis for each grid cell - OPTIMIZED WITH CHUNKING
    print(f"\n3. CDF ANALYSIS (Optimized with Chunking)")
    print(f"=" * 45)
    print(f"   Processing {n_south_north * n_west_east} grid cells...")
    
    cdf_results = {}
    cell_statistics = {}
    
    # Process in chunks to show progress and avoid memory issues
    total_cells = n_south_north * n_west_east
    cells_processed = 0
    valid_cells = 0
    
    # Process one row at a time to show progress
    for i in range(n_south_north):
        # Extract all cells in this row at once (more efficient)
        row_data = ice_data_clean.isel(south_north=i).values  # shape: (n_time, n_west_east)
        
        for j in range(n_west_east):
            # Extract time series for this grid cell
            cell_values = row_data[:, j]
            
            # Remove NaN values
            valid_mask = ~np.isnan(cell_values)
            cell_values_clean = cell_values[valid_mask]
            
            if len(cell_values_clean) == 0:
                cells_processed += 1
                continue
            
            # Filter values to be >= threshold
            cell_values_filtered = cell_values_clean[cell_values_clean >= ice_load_threshold]
            
            if len(cell_values_filtered) == 0:
                cells_processed += 1
                continue
            
            # Calculate CDF using vectorized operations (much faster)
            # Instead of looping through each bin, use searchsorted
            sorted_values = np.sort(cell_values_filtered)
            n_values = len(sorted_values)
            
            # Use numpy's searchsorted for efficient CDF calculation
            indices = np.searchsorted(sorted_values, ice_load_bins, side='right')
            cdf_values = indices / n_values
            
            # Store results
            cell_key = f'cell_{i}_{j}'
            cdf_results[cell_key] = {
                'ice_load_bins': ice_load_bins,
                'cdf_values': cdf_values,
                'position': (i, j)
            }
            
            # Calculate statistics using vectorized operations
            cell_statistics[cell_key] = {
                'max_ice_load': float(sorted_values[-1]),  # Already sorted
                'mean_ice_load': float(np.mean(cell_values_filtered)),
                'std_ice_load': float(np.std(cell_values_filtered)),
                'median_ice_load': float(sorted_values[n_values // 2]),  # Median from sorted
                'percentile_95': float(sorted_values[int(0.95 * n_values)]),
                'percentile_99': float(sorted_values[int(0.99 * n_values)]),
                'n_valid_points': n_values
            }
            
            cells_processed += 1
            valid_cells += 1
        
        # Print progress every row
        progress_pct = (i + 1) / n_south_north * 100
        print(f"   Progress: Row {i+1}/{n_south_north} ({progress_pct:.1f}%) - {valid_cells} valid cells found", end='\r')
    
    print(f"\n   Processed {cells_processed} total cells, {valid_cells} valid cells with data")
    
    # Calculate weighted spatial gradients using configurable neighborhood
    print(f"\n4. WEIGHTED SPATIAL GRADIENT ANALYSIS")
    print(f"=" * 40)
    
    try:
        from scipy.stats import wasserstein_distance
        
        print(f"   Computing weighted Earth Mover's distances using {neighborhood_type}...")
        
        # Initialize gradient matrix for neighborhood-based analysis
        weighted_gradients = np.full((n_south_north, n_west_east), np.nan)
        
        # Calculate weighted gradients for each grid cell
        for i in range(n_south_north):
            for j in range(n_west_east):
                cell_key = f'cell_{i}_{j}'
                
                if cell_key not in cdf_results:
                    continue
                
                cell_cdf = cdf_results[cell_key]['cdf_values']
                cell_bins = cdf_results[cell_key]['ice_load_bins']
                
                weighted_distances = []
                total_weights = 0
                
                # Check all neighbors in the defined neighborhood
                for di, dj in neighbor_weights.keys():
                    ni, nj = i + di, j + dj
                    
                    # Check if neighbor is within bounds
                    if 0 <= ni < n_south_north and 0 <= nj < n_west_east:
                        neighbor_key = f'cell_{ni}_{nj}'
                        
                        if neighbor_key in cdf_results:
                            neighbor_cdf = cdf_results[neighbor_key]['cdf_values']
                            neighbor_bins = cdf_results[neighbor_key]['ice_load_bins']
                            
                            # Calculate Earth Mover's distance using L1 distance between CDFs
                            try:
                                distance = np.trapz(np.abs(cell_cdf - neighbor_cdf), cell_bins)
                            except:
                                distance = np.mean(np.abs(cell_cdf - neighbor_cdf))
                            
                            # Weight the distance
                            weight = neighbor_weights[(di, dj)]
                            weighted_distances.append(distance * weight)
                            total_weights += weight
                
                # Calculate weighted average distance
                if weighted_distances and total_weights > 0:
                    weighted_gradients[i, j] = np.sum(weighted_distances) / total_weights
        
        print(f"   Computed weighted gradients for {np.sum(~np.isnan(weighted_gradients))} cells")
        
        # Create weighted spatial gradient plots
        print(f"\n5. CREATING WEIGHTED SPATIAL GRADIENT PLOTS")
        print(f"=" * 45)
        
        # Calculate statistics for normalization
        valid_gradients = weighted_gradients[~np.isnan(weighted_gradients)]
        if len(valid_gradients) > 0:
            gradient_mean = np.mean(valid_gradients)
            gradient_std = np.std(valid_gradients)
            gradient_min = np.min(valid_gradients)
            gradient_max = np.max(valid_gradients)
        else:
            gradient_mean = gradient_std = gradient_min = gradient_max = 0
        
        # ABSOLUTE GRADIENT PLOT (separate figure)
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111)
        
        # Plot 1: Absolute weighted gradients
        im1 = ax1.imshow(weighted_gradients, cmap='viridis', origin='lower', 
                           interpolation='nearest', aspect='auto')
        ax1.set_title(f'Weighted Spatial Gradient\n({neighborhood_type}, {weight_scheme} weighting)')
        ax1.set_xlabel('West-East Grid Points', fontsize=24)
        ax1.set_ylabel('South-North Grid Points', fontsize=24)
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Weighted Earth Mover\'s Distance (kg/m)')
        
        # Add grid lines and statistics text
        ax1.set_xticks(range(n_west_east))
        ax1.set_yticks(range(n_south_north))
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f'Mean: {gradient_mean:.3f}\nStd: {gradient_std:.3f}\nMin: {gradient_min:.3f}\nMax: {gradient_max:.3f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save absolute weighted spatial gradient plot
        if save_plots:
            gradient_filename_abs = f"weighted_spatial_gradients_absolute_{neighborhood_type}_{weight_scheme}.png"
            gradient_path_abs = os.path.join(base_results_dir, gradient_filename_abs)
            plt.savefig(gradient_path_abs, dpi=300, bbox_inches='tight')
            print(f"   Absolute weighted spatial gradient plot saved to: {gradient_path_abs}")
        
        plt.close()
        
        # NORMALIZED GRADIENT PLOT (separate figure)
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111)
        
        # Plot 2: Normalized weighted gradients
        if gradient_mean > 0:
            weighted_gradients_normalized = weighted_gradients / gradient_mean
        else:
            weighted_gradients_normalized = weighted_gradients
            
        im2 = ax2.imshow(weighted_gradients_normalized, cmap='RdBu_r', origin='lower', 
                           interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.5)
        ax2.set_title(f'Normalized Weighted Spatial Gradient\n(Relative to Domain Mean)', fontsize=28)
        ax2.set_xlabel('West-East Grid Points', fontsize=24)
        ax2.set_ylabel('South-North Grid Points', fontsize=24)
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Gradient / Domain Mean', fontsize=16)
        cbar2.ax.tick_params(labelsize=15)
        
        ax2.set_xticks(range(n_west_east))
        ax2.set_yticks(range(n_south_north))
        ax2.tick_params(axis='both', labelsize=20)
        ax2.grid(True, alpha=0.3)
        
        # Add normalization factor text
        norm_text = f'Normalization factor:\n{gradient_mean:.3f} kg/m'
        ax2.text(0.02, 0.98, norm_text, transform=ax2.transAxes, fontsize=15,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save normalized weighted spatial gradient plot
        if save_plots:
            gradient_filename_norm = f"weighted_spatial_gradients_normalized_{neighborhood_type}_{weight_scheme}.png"
            gradient_path_norm = os.path.join(base_results_dir, gradient_filename_norm)
            plt.savefig(gradient_path_norm, dpi=300, bbox_inches='tight')
            print(f"   Normalized weighted spatial gradient plot saved to: {gradient_path_norm}")
        
        plt.close()
        
        # CREATE CARTOPY-BASED GRADIENT PLOT
        print(f"\n   Creating cartopy-based normalized gradient map...")
        cartopy_success = False
        try:
            # Import Cartopy components
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            import cartopy.io.img_tiles as cimgt
            
            print(f"   Using Cartopy with terrain background (margin={margin_degrees}°, zoom={zoom_level})")
            
            # Get geographical coordinates
            if 'XLAT' in dataset_with_ice_load.coords and 'XLON' in dataset_with_ice_load.coords:
                lats = dataset_with_ice_load.XLAT.values
                lons = dataset_with_ice_load.XLON.values
            elif 'XLAT' in dataset_with_ice_load.data_vars and 'XLON' in dataset_with_ice_load.data_vars:
                lats = dataset_with_ice_load.XLAT.values
                lons = dataset_with_ice_load.XLON.values
            else:
                print("   Warning: Could not find XLAT and XLON coordinates, skipping cartopy plot")
                raise ValueError("Missing coordinates")
            
            # Create grid cell boundaries for pcolormesh first
            # Calculate cell boundaries (midpoints between cell centers)
            lon_edges = np.zeros(lons.shape[1] + 1)
            lat_edges = np.zeros(lats.shape[0] + 1)
            
            # Longitude edges
            for i in range(lons.shape[1]):
                if i == 0:
                    lon_edges[i] = lons[0, i] - (lons[0, 1] - lons[0, 0]) / 2
                else:
                    lon_edges[i] = (lons[0, i-1] + lons[0, i]) / 2
            lon_edges[-1] = lons[0, -1] + (lons[0, -1] - lons[0, -2]) / 2
            
            # Latitude edges
            for i in range(lats.shape[0]):
                if i == 0:
                    lat_edges[i] = lats[i, 0] - (lats[1, 0] - lats[0, 0]) / 2
                else:
                    lat_edges[i] = (lats[i-1, 0] + lats[i, 0]) / 2
            lat_edges[-1] = lats[-1, 0] + (lats[-1, 0] - lats[-2, 0]) / 2
            
            # Create figure with Cartopy projection
            fig = plt.figure(figsize=(16, 12))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            # Set map extent using actual grid boundaries with margins for better centering
            west = lon_edges.min() - margin_degrees
            east = lon_edges.max() + margin_degrees
            south = lat_edges.min() - margin_degrees
            north = lat_edges.max() + margin_degrees
            
            # Calculate grid center for better centering
            grid_center_lon = (lon_edges.min() + lon_edges.max()) / 2
            grid_center_lat = (lat_edges.min() + lat_edges.max()) / 2
            
            # Calculate grid span
            grid_span_lon = lon_edges.max() - lon_edges.min()
            grid_span_lat = lat_edges.max() - lat_edges.min()
            
            # Create balanced extent around grid center
            extent_span_lon = grid_span_lon + 2 * margin_degrees
            extent_span_lat = grid_span_lat + 2 * margin_degrees
            
            west = grid_center_lon - extent_span_lon / 2
            east = grid_center_lon + extent_span_lon / 2
            south = grid_center_lat - extent_span_lat / 2
            north = grid_center_lat + extent_span_lat / 2
            
            ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
            
            print(f"   Grid boundaries: {lon_edges.min():.4f}° to {lon_edges.max():.4f}°E, {lat_edges.min():.4f}° to {lat_edges.max():.4f}°N")
            print(f"   Map extent: {west:.4f}° to {east:.4f}°E, {south:.4f}° to {north:.4f}°N")
            print(f"   Grid center: {grid_center_lat:.4f}°N, {grid_center_lon:.4f}°E")
            
            # Add basic geographical features first (always visible)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.7, zorder=1)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.8, zorder=2)
            ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.8, zorder=3)
            
            # Try to add terrain background (optional enhancement)
            try:
                # Use OpenStreetMap (free, no API key required)
                terrain = cimgt.OSM()
                ax.add_image(terrain, zoom_level)
                print("   Successfully loaded OpenStreetMap tiles")
            except Exception as e:
                try:
                    # Fallback to GoogleTiles (satellite view)
                    terrain = cimgt.GoogleTiles(style='satellite')
                    ax.add_image(terrain, zoom_level)
                    print("   Successfully loaded Google satellite tiles")
                except Exception as e2:
                    print(f"   All tile services unavailable, using basic land/ocean features")
            
            # Add geographical features on top
            ax.add_feature(cfeature.BORDERS, linewidth=1.5, color='black', alpha=0.8, zorder=8)
            ax.add_feature(cfeature.COASTLINE, linewidth=2, color='black', alpha=0.9, zorder=9)
            
            # Create meshgrid for pcolormesh
            lon_mesh, lat_mesh = np.meshgrid(lon_edges, lat_edges)
            
            # Plot normalized gradient as semi-transparent overlay
            gradient_plot = ax.pcolormesh(
                lon_mesh, lat_mesh, weighted_gradients_normalized,
                cmap='RdBu_r', alpha=0.7, 
                vmin=0.5, vmax=1.5,
                transform=ccrs.PlateCarree(),
                zorder=7
            )
            
            # Add colorbar
            cbar = plt.colorbar(gradient_plot, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Normalized Weighted Spatial Gradient\n(Relative to Domain Mean)', fontsize=48)
            cbar.ax.tick_params(labelsize=30)
            
            # Add gridlines with labels
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 30, 'color': 'black'}
            gl.ylabel_style = {'size': 30, 'color': 'black'}
            
            # Add title with comprehensive information
            title_text = (f'Normalized Weighted Spatial Gradient on Terrain Map\n'
                         f'{neighborhood_type}, {weight_scheme} weighting, '
                         f'Height: {dataset_with_ice_load.height.values[height_level]} m')
            ax.set_title(title_text, fontsize=56, weight='bold', pad=20)
            
            # Add margin and method information
            info_text = (f"Margin: {margin_degrees}° | Zoom: {zoom_level} | "
                        f"Normalization: {gradient_mean:.3f} kg/m")
            ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=60,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            
            # Save the cartopy plot
            if save_plots:
                cartopy_filename = f"weighted_spatial_gradients_cartopy_{neighborhood_type}_{weight_scheme}.png"
                cartopy_path = os.path.join(base_results_dir, cartopy_filename)
                plt.savefig(cartopy_path, dpi=300, bbox_inches='tight')
                print(f"   Cartopy gradient plot saved to: {cartopy_path}")
                cartopy_success = True
            
            plt.close()
            
        except ImportError as e:
            print(f"   Warning: Cartopy not available, skipping cartopy gradient plot")
            print(f"   Install with: conda install cartopy")
        except Exception as e:
            print(f"   Error creating cartopy gradient plot: {e}")
            print(f"   Make sure you have internet connection for terrain tiles")
        
        # Store gradient results
        gradient_results = {
            'weighted_gradients': weighted_gradients,
            'weighted_gradients_normalized': weighted_gradients_normalized,
            'neighborhood_type': neighborhood_type,
            'weight_scheme': weight_scheme,
            'neighbor_weights': neighbor_weights,
            'gradient_statistics': {
                'mean': gradient_mean,
                'std': gradient_std,
                'min': gradient_min,
                'max': gradient_max
            },
            'cartopy_plot_created': cartopy_success,
            'cartopy_parameters': {
                'margin_degrees': margin_degrees,
                'zoom_level': zoom_level
            }
        }
        
        print(f"   ✓ Weighted spatial gradient analysis completed")
        print(f"   Statistics: Mean={gradient_mean:.3f}, Std={gradient_std:.3f}")
        
        # CREATE CARTOPY-BASED ICE LOAD VALUES PLOT
        print(f"\n   Creating cartopy-based ice load values map...")
        ice_load_cartopy_success = False
        try:
            # Import Cartopy components (already imported above, but keeping for clarity)
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            import cartopy.io.img_tiles as cimgt
            
            print(f"   Calculating mean annual ice load totals for each grid cell...")
            
            # Extract ice load data at the specific height level
            ice_load_data = dataset_with_ice_load[ice_load_variable].isel(height=height_level)
            
            # Clean data - replace values below threshold and NaN with 0
            ice_data_clean = ice_load_data.where(ice_load_data >= ice_load_threshold, 0)
            ice_data_clean = ice_data_clean.where(ice_data_clean.notnull(), 0)
            
            # Filter by months if specified (same logic as in the main filtering)
            if months is not None:
                time_index_full = pd.to_datetime(ice_data_clean.time.values)
                month_mask = time_index_full.month.isin(months)
                ice_data_clean = ice_data_clean.isel(time=month_mask)
                print(f"   Filtered to months: {months}")
            
            # Calculate annual totals for each grid cell
            ice_data_clean = ice_data_clean.assign_coords(year=ice_data_clean.time.dt.year)
            annual_totals = ice_data_clean.groupby('year').sum(dim='time')
            mean_annual_totals = annual_totals.mean(dim='year').values
            
            print(f"   Mean annual ice load range: {np.min(mean_annual_totals):.3f} to {np.max(mean_annual_totals):.3f} kg/m")
            
            # Get geographical coordinates (same as gradient plot)
            if 'XLAT' in dataset_with_ice_load.coords and 'XLON' in dataset_with_ice_load.coords:
                lats = dataset_with_ice_load.XLAT.values
                lons = dataset_with_ice_load.XLON.values
            elif 'XLAT' in dataset_with_ice_load.data_vars and 'XLON' in dataset_with_ice_load.data_vars:
                lats = dataset_with_ice_load.XLAT.values
                lons = dataset_with_ice_load.XLON.values
            else:
                print(f"   Warning: Could not find XLAT and XLON coordinates, skipping ice load cartopy plot")
                raise ValueError("Missing coordinates")
            
            # Use same grid boundaries calculation as gradient plot
            lon_edges = np.zeros(lons.shape[1] + 1)
            lat_edges = np.zeros(lats.shape[0] + 1)
            
            # Longitude edges
            for i in range(lons.shape[1]):
                if i == 0:
                    lon_edges[i] = lons[0, i] - (lons[0, 1] - lons[0, 0]) / 2
                else:
                    lon_edges[i] = (lons[0, i-1] + lons[0, i]) / 2
            lon_edges[-1] = lons[0, -1] + (lons[0, -1] - lons[0, -2]) / 2
            
            # Latitude edges
            for i in range(lats.shape[0]):
                if i == 0:
                    lat_edges[i] = lats[i, 0] - (lats[1, 0] - lats[0, 0]) / 2
                else:
                    lat_edges[i] = (lats[i-1, 0] + lats[i, 0]) / 2
            lat_edges[-1] = lats[-1, 0] + (lats[-1, 0] - lats[-2, 0]) / 2
            
            # Create figure with same setup as gradient plot
            fig = plt.figure(figsize=(16, 12))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            # Use same extent calculation as gradient plot
            grid_center_lon = (lon_edges.min() + lon_edges.max()) / 2
            grid_center_lat = (lat_edges.min() + lat_edges.max()) / 2
            grid_span_lon = lon_edges.max() - lon_edges.min()
            grid_span_lat = lat_edges.max() - lat_edges.min()
            extent_span_lon = grid_span_lon + 2 * margin_degrees
            extent_span_lat = grid_span_lat + 2 * margin_degrees
            
            west = grid_center_lon - extent_span_lon / 2
            east = grid_center_lon + extent_span_lon / 2
            south = grid_center_lat - extent_span_lat / 2
            north = grid_center_lat + extent_span_lat / 2
            
            ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
            
            # Add same geographical features as gradient plot
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.7, zorder=1)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.8, zorder=2)
            ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.8, zorder=3)
            
            # Try to add terrain background (same as gradient plot)
            try:
                terrain = cimgt.OSM()
                ax.add_image(terrain, zoom_level)
                print(f"   Successfully loaded OpenStreetMap tiles for ice load plot")
            except Exception as e:
                try:
                    terrain = cimgt.GoogleTiles(style='satellite')
                    ax.add_image(terrain, zoom_level)
                    print(f"   Successfully loaded Google satellite tiles for ice load plot")
                except Exception as e2:
                    print(f"   All tile services unavailable for ice load plot, using basic land/ocean features")
            
            # Add geographical features on top
            ax.add_feature(cfeature.BORDERS, linewidth=1.5, color='black', alpha=0.8, zorder=8)
            ax.add_feature(cfeature.COASTLINE, linewidth=2, color='black', alpha=0.9, zorder=9)
            
            # Create meshgrid for pcolormesh
            lon_mesh, lat_mesh = np.meshgrid(lon_edges, lat_edges)
            
            # Calculate percentiles for better color scaling (handle outliers)
            valid_values = mean_annual_totals[~np.isnan(mean_annual_totals)]
            if len(valid_values) > 0:
                data_min = np.min(valid_values)
                data_max = np.max(valid_values)
                data_mean = np.mean(valid_values)
                data_std = np.std(valid_values)
                data_90p = np.percentile(valid_values, 90)
                data_95p = np.percentile(valid_values, 95)
                
                print(f"   Ice load statistics:")
                print(f"     Min: {data_min:.1f}, Max: {data_max:.1f}, Mean: {data_mean:.1f} kg/m")
                print(f"     90th percentile: {data_90p:.1f}, 95th percentile: {data_95p:.1f} kg/m")
                
                # Check if there are significant outliers
                outlier_ratio = data_max / data_90p if data_90p > 0 else 1
                print(f"     Outlier ratio (max/90th percentile): {outlier_ratio:.1f}")
                
                # Use percentile-based scaling if there are significant outliers
                if outlier_ratio > 2.0:  # If max is more than 2x the 90th percentile
                    vmin = data_min
                    vmax = data_90p
                    outlier_clipped = True
                    print(f"   Using 90th percentile clipping for better color differentiation")
                    print(f"   Color scale: {vmin:.1f} to {vmax:.1f} kg/m (values above {vmax:.1f} will show as maximum color)")
                else:
                    vmin = data_min
                    vmax = data_max
                    outlier_clipped = False
                    print(f"   Using full range color scale: {vmin:.1f} to {vmax:.1f} kg/m")
            else:
                vmin, vmax = 0, 1
                outlier_clipped = False
            
            # Plot ice load values as semi-transparent overlay with improved scaling
            ice_load_plot = ax.pcolormesh(
                lon_mesh, lat_mesh, mean_annual_totals,
                cmap='viridis', alpha=0.8,  # Using viridis like in plot_grid_ice_load_values
                vmin=vmin, vmax=vmax,  # Use calculated range for better color differentiation
                transform=ccrs.PlateCarree(),
                zorder=7
            )
            
            # Add colorbar with enhanced information
            cbar = plt.colorbar(ice_load_plot, ax=ax, shrink=0.8, pad=0.02)
            if outlier_clipped:
                cbar_label = f'Mean Annual Total Ice Load (kg/m)\n[Clipped at 90th percentile: {vmax:.1f}]'
            else:
                cbar_label = 'Mean Annual Total Ice Load (kg/m)'
            cbar.set_label(cbar_label, fontsize=24)
            cbar.ax.tick_params(labelsize=30)
            
            # Add gridlines with labels
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 30, 'color': 'black'}
            gl.ylabel_style = {'size': 30, 'color': 'black'}
            
            # Add title with comprehensive information
            months_str = f", months {months}" if months else ""
            title_text = (f'Mean Annual Total Ice Load per Grid Cell on Terrain Map\\n'
                         f'Height: {dataset_with_ice_load.height.values[height_level]} m, '
                         f'Threshold: {ice_load_threshold:.1f} kg/m{months_str}')
            ax.set_title(title_text, fontsize=28, weight='bold', pad=20)
            
            # Add statistics information with outlier information
            if outlier_clipped:
                info_text = (f"Range: {np.min(mean_annual_totals):.1f} - {np.max(mean_annual_totals):.1f} kg/m | "
                            f"Mean: {np.mean(mean_annual_totals):.1f} kg/m\n"
                            f"Color scale: {vmin:.1f} - {vmax:.1f} kg/m (90th percentile clipped)")
            else:
                info_text = (f"Range: {np.min(mean_annual_totals):.1f} - {np.max(mean_annual_totals):.1f} kg/m | "
                            f"Mean: {np.mean(mean_annual_totals):.1f} kg/m")
            ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=27,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            
            plt.tight_layout()
            
            # Save the ice load cartopy plot
            if save_plots:
                ice_load_cartopy_filename = f"ice_load_values_cartopy_{neighborhood_type}_{weight_scheme}.png"
                ice_load_cartopy_path = os.path.join(base_results_dir, ice_load_cartopy_filename)
                plt.savefig(ice_load_cartopy_path, dpi=300, bbox_inches='tight')
                print(f"   Ice load cartopy plot saved to: {ice_load_cartopy_path}")
                ice_load_cartopy_success = True
            
            plt.close()
            
        except ImportError as e:
            print(f"   Warning: Cartopy not available, skipping ice load cartopy plot")
        except Exception as e:
            print(f"   Error creating ice load cartopy plot: {e}")
        
        # Update gradient_results to include ice load cartopy info
        if gradient_results:
            gradient_results['ice_load_cartopy_plot_created'] = ice_load_cartopy_success
        
    except ImportError:
        print("   Warning: scipy not available, skipping Earth Mover's distance calculation")
        gradient_results = None
    except Exception as e:
        print(f"   Error in spatial gradient analysis: {e}")
        gradient_results = None
    
    # # Create summary CDF plot (same as original)
    # print(f"\n6. CREATING SUMMARY CDF PLOT")
    # print(f"=" * 28)
    # 
    # if cdf_results:
    #     # Calculate mean CDF curve across all cells
    #     all_cdf_curves = []
    #     for cell_data in cdf_results.values():
    #         all_cdf_curves.append(cell_data['cdf_values'])
    #     
    #     mean_cdf = np.mean(all_cdf_curves, axis=0)
    #     std_cdf = np.std(all_cdf_curves, axis=0)
    #     
    #     plt.figure(figsize=(10, 6))
    #     plot_mask = ice_load_bins >= ice_load_threshold
    #     
    #     plot_bins = ice_load_bins[plot_mask]
    #     plot_mean_cdf = mean_cdf[plot_mask]
    #     plot_std_cdf = std_cdf[plot_mask]
    #     
    #     plt.plot(plot_bins, plot_mean_cdf, 'r-', linewidth=3, label='Mean across all cells')
    #     plt.fill_between(plot_bins, plot_mean_cdf - plot_std_cdf, 
    #                    plot_mean_cdf + plot_std_cdf, alpha=0.3, color='red', 
    #                    label='±1 Standard Deviation')
    #     
    #     plt.xlabel('Ice Load (kg/m)', fontsize=24)
    #     plt.ylabel('Cumulative Probability', fontsize=24)
    #     plt.title(f'Ice Load CDF - Domain Average\n({neighborhood_type}, {weight_scheme} weighting)')
    #     plt.grid(True, alpha=0.3)
    #     plt.legend()
    #     plt.xlim(left=ice_load_threshold)
    #     plt.ylim([0, 1])
    #     
    #     plt.tight_layout()
    #     
    #     if save_plots:
    #         summary_filename = f"ice_load_cdf_summary_{neighborhood_type}_{weight_scheme}.png"
    #         summary_path = os.path.join(base_results_dir, summary_filename)
    #         plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    #         print(f"   Summary CDF plot saved to: {summary_path}")
    #     
    #     plt.close()
    
    # Save enhanced analysis summary file
    if save_plots:
        summary_file_path = os.path.join(base_results_dir, "weighted_neighborhood_analysis_summary.txt")
        with open(summary_file_path, 'w') as f:
            f.write("WEIGHTED NEIGHBORHOOD ICE LOAD ANALYSIS WITH CDF\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis timestamp: {pd.Timestamp.now()}\n")
            f.write(f"Ice load variable: {ice_load_variable}\n")
            f.write(f"Height level: {height_level} ({dataset_with_ice_load.height.values[height_level]} m)\n\n")
            
            f.write("NEIGHBORHOOD CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Neighborhood type: {neighborhood_type}\n")
            f.write(f"Weight scheme: {weight_scheme}\n")
            f.write(f"Number of neighbors: {len(neighbor_weights)}\n")
            f.write("Neighbor weights:\n")
            for (di, dj), weight in neighbor_weights.items():
                f.write(f"  ({di:2d},{dj:2d}): {weight:.3f}\n")
            f.write("\n")
            
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
            
            if gradient_results:
                f.write(f"\nWEIGHTED SPATIAL GRADIENT STATISTICS:\n")
                f.write("-" * 40 + "\n")
                stats = gradient_results['gradient_statistics']
                f.write(f"Mean weighted gradient: {stats['mean']:.3f} kg/m\n")
                f.write(f"Standard deviation: {stats['std']:.3f} kg/m\n")
                f.write(f"Minimum gradient: {stats['min']:.3f} kg/m\n")
                f.write(f"Maximum gradient: {stats['max']:.3f} kg/m\n")
            
            f.write(f"\nFILES GENERATED:\n")
            f.write("-" * 20 + "\n")
            f.write(f"- weighted_spatial_gradients_absolute_{neighborhood_type}_{weight_scheme}.png\n")
            f.write(f"- weighted_spatial_gradients_normalized_{neighborhood_type}_{weight_scheme}.png\n")
            f.write("- weighted_neighborhood_analysis_summary.txt (this file)\n")
        
        print(f"   Enhanced analysis summary saved to: {summary_file_path}")
    
    # Compile final results
    results = {
        'neighborhood_config': {
            'neighborhood_type': neighborhood_type,
            'weight_scheme': weight_scheme,
            'neighbor_weights': neighbor_weights,
            'n_neighbors': len(neighbor_weights)
        },
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
        'weighted_spatial_gradients': gradient_results,
        'ice_load_bins': ice_load_bins
    }
    
    print(f"\n=== WEIGHTED NEIGHBORHOOD ANALYSIS COMPLETED SUCCESSFULLY ===")
    print(f"   Neighborhood: {neighborhood_type} ({len(neighbor_weights)} neighbors)")
    print(f"   Weight scheme: {weight_scheme}")
    print(f"   Filters applied: {len(filter_info)}")
    print(f"   Valid grid cells: {len(cdf_results)}")
    print(f"   Weighted gradients: {'✓' if gradient_results else '✗'}")
    
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
    
    # Plot 1: Yearly time series with long-term average
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Yearly mean ice load
    axes[0, 0].plot(years_list, yearly_mean, 'b-o', linewidth=2, markersize=4, label='Yearly Mean')
    axes[0, 0].axhline(y=long_term_stats['mean'], color='red', linestyle='--', linewidth=2, 
                       label=f'Long-term Average ({long_term_stats["mean"]:.3f})')
    axes[0, 0].fill_between(years_list, 
                            [long_term_stats['mean'] - long_term_stats['std']] * len(years_list),
                            [long_term_stats['mean'] + long_term_stats['std']] * len(years_list),
                            alpha=0.2, color='red', label='±1 Std Dev')
    axes[0, 0].set_xlabel('Year', fontsize=24)
    axes[0, 0].set_ylabel('Mean Ice Load (kg/m)', fontsize=24)
    axes[0, 0].set_title('Yearly Mean Ice Load vs Long-term Average', fontsize=28)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Subplot 2: Yearly maximum ice load
    axes[0, 1].plot(years_list, yearly_max, 'g-s', linewidth=2, markersize=4, label='Yearly Maximum')
    axes[0, 1].axhline(y=long_term_stats['max'], color='orange', linestyle='--', linewidth=2,
                       label=f'Long-term Max ({long_term_stats["max"]:.3f})')
    axes[0, 1].set_xlabel('Year', fontsize=24)
    axes[0, 1].set_ylabel('Maximum Ice Load (kg/m)', fontsize=24)
    axes[0, 1].set_title('Yearly Maximum Ice Load', fontsize=28)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Subplot 3: Yearly deviations from long-term average
    colors = ['red' if x > 0 else 'blue' for x in yearly_deviations]
    axes[1, 0].bar(years_list, yearly_deviations, color=colors, alpha=0.7, width=0.8)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_xlabel('Year', fontsize=24)
    axes[1, 0].set_ylabel('Deviation from Long-term Mean (kg/m)', fontsize=24)
    axes[1, 0].set_title('Yearly Deviations from Long-term Average', fontsize=28)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: Normalized deviations
    colors_norm = ['red' if x > 0 else 'blue' for x in yearly_normalized_deviations]
    axes[1, 1].bar(years_list, yearly_normalized_deviations, color=colors_norm, alpha=0.7, width=0.8)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='+1 Std Dev')
    axes[1, 1].axhline(y=-1, color='red', linestyle='--', alpha=0.7, label='-1 Std Dev')
    axes[1, 1].set_xlabel('Year', fontsize=24)
    axes[1, 1].set_ylabel('Normalized Deviation (σ)', fontsize=24)
    axes[1, 1].set_title('Normalized Yearly Deviations', fontsize=28)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_plots:
        plot1_path = os.path.join(base_results_dir, "ice_load_yearly_analysis.png")
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        print(f"Yearly analysis plot saved to: {plot1_path}")
    
    plt.close()
    
    # Plot 2: Resampling period analysis
    if len(period_means) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: Period means
        x_pos = np.arange(len(period_labels))
        axes[0, 0].bar(x_pos, period_means, alpha=0.7, color='skyblue', edgecolor='navy')
        axes[0, 0].axhline(y=long_term_stats['mean'], color='red', linestyle='--', linewidth=2,
                           label=f'Overall Mean ({long_term_stats["mean"]:.3f})')
        axes[0, 0].set_xlabel(f'{resampling_years}-Year Periods')
        axes[0, 0].set_ylabel('Mean Ice Load (kg/m)', fontsize=24)
        axes[0, 0].set_title(f'{resampling_years}-Year Period Means')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(period_labels, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Subplot 2: Period standard deviations
        axes[0, 1].bar(x_pos, period_stds, alpha=0.7, color='lightcoral', edgecolor='darkred')
        axes[0, 1].axhline(y=long_term_stats['std'], color='blue', linestyle='--', linewidth=2,
                           label=f'Overall Std ({long_term_stats["std"]:.3f})')
        axes[0, 1].set_xlabel(f'{resampling_years}-Year Periods')
        axes[0, 1].set_ylabel('Standard Deviation (kg/m)', fontsize=24)
        axes[0, 1].set_title(f'{resampling_years}-Year Period Variability')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(period_labels, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Subplot 3: Period deviations from overall mean
        period_deviations = [mean - long_term_stats['mean'] for mean in period_means]
        colors_period = ['red' if x > 0 else 'blue' for x in period_deviations]
        axes[1, 0].bar(x_pos, period_deviations, color=colors_period, alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 0].set_xlabel(f'{resampling_years}-Year Periods')
        axes[1, 0].set_ylabel('Deviation from Overall Mean (kg/m)', fontsize=24)
        axes[1, 0].set_title(f'{resampling_years}-Year Period Deviations')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(period_labels, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Subplot 4: Coefficient of variation
        cv_values = [std/mean if mean > 0 else 0 for mean, std in zip(period_means, period_stds)]
        overall_cv = long_term_stats['std'] / long_term_stats['mean']
        
        axes[1, 1].bar(x_pos, cv_values, alpha=0.7, color='gold', edgecolor='orange')
        axes[1, 1].axhline(y=overall_cv, color='purple', linestyle='--', linewidth=2,
                           label=f'Overall CV ({overall_cv:.3f})')
        axes[1, 1].set_xlabel(f'{resampling_years}-Year Periods')
        axes[1, 1].set_ylabel('Coefficient of Variation', fontsize=24)
        axes[1, 1].set_title(f'{resampling_years}-Year Period Relative Variability')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(period_labels, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plot2_path = os.path.join(base_results_dir, f"ice_load_{resampling_years}year_resampling_analysis.png")
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            print(f"Resampling analysis plot saved to: {plot2_path}")
        
        plt.close()
    
    # Plot 3: Percentiles evolution
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Subplot 1: Multiple percentiles over time
    axes[0].plot(years_list, yearly_percentiles['p90'], 'g-o', linewidth=2, markersize=3, label='90th Percentile')
    axes[0].plot(years_list, yearly_percentiles['p95'], 'orange', marker='s', linewidth=2, markersize=3, label='95th Percentile')
    axes[0].plot(years_list, yearly_percentiles['p99'], 'r-^', linewidth=2, markersize=3, label='99th Percentile')
    axes[0].plot(years_list, yearly_mean, 'b-', linewidth=2, alpha=0.7, label='Mean')
    
    # Add long-term averages
    axes[0].axhline(y=np.percentile(all_values_clean, 90), color='green', linestyle='--', alpha=0.7)
    axes[0].axhline(y=np.percentile(all_values_clean, 95), color='orange', linestyle='--', alpha=0.7)
    axes[0].axhline(y=np.percentile(all_values_clean, 99), color='red', linestyle='--', alpha=0.7)
    axes[0].axhline(y=long_term_stats['mean'], color='blue', linestyle='--', alpha=0.7)
    
    axes[0].set_xlabel('Year', fontsize=24)
    axes[0].set_ylabel('Ice Load (kg/m)', fontsize=24)
    axes[0].set_title('Ice Load Percentiles Evolution Over Time', fontsize=28)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Subplot 2: Yearly statistics distribution
    axes[1].boxplot([yearly_mean, yearly_percentiles['p90'], yearly_percentiles['p95'], yearly_percentiles['p99']], 
                    labels=['Mean', 'P90', 'P95', 'P99'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel('Ice Load (kg/m)', fontsize=24)
    axes[1].set_title('Distribution of Yearly Statistics', fontsize=28)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plot3_path = os.path.join(base_results_dir, "ice_load_percentiles_evolution.png")
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        print(f"Percentiles evolution plot saved to: {plot3_path}")
    
    plt.close()
    
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
            f.write("- ice_load_yearly_analysis.png\n")
            if len(period_means) > 1:
                f.write(f"- ice_load_{resampling_years}year_resampling_analysis.png\n")
            f.write("- ice_load_percentiles_evolution.png\n")
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
    
    # Plot 2: Resampling period analysis
    if len(period_total_hours) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: Period grid mean total hours
        x_pos = np.arange(len(period_labels))
        axes[0, 0].bar(x_pos, period_total_hours, alpha=0.7, color='skyblue', edgecolor='navy')
        period_avg = long_term_stats['grid_mean_total_hours'] / total_winter_years * resampling_years
        axes[0, 0].axhline(y=period_avg, color='red', linestyle='--', linewidth=2,
                           label=f'Expected Total ({period_avg:.1f})')
        axes[0, 0].set_xlabel(f'{resampling_years}-Year Periods')
        axes[0, 0].set_ylabel('Grid Mean Total Exceedance Hours', fontsize=24)
        axes[0, 0].set_title(f'{resampling_years}-Year Period Grid Mean Total Hours')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(period_labels, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Subplot 2: Period mean hours per cell
        axes[0, 1].bar(x_pos, period_mean_hours, alpha=0.7, color='lightcoral', edgecolor='darkred')
        # Calculate mean of all period means (average of the bars)
        mean_of_period_means = np.mean(period_mean_hours)
        axes[0, 1].axhline(y=mean_of_period_means, color='blue', linestyle='--', linewidth=2,
                           label=f'Mean of Period Means ({mean_of_period_means:.1f})')
        axes[0, 1].set_xlabel(f'{resampling_years}-Year Periods')
        axes[0, 1].set_ylabel('Mean Hours per Cell', fontsize=24)
        axes[0, 1].set_title(f'{resampling_years}-Year Period Mean Hours per Cell')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(period_labels, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Subplot 3: Period deviations from mean of period means
        period_deviations = [mean - mean_of_period_means for mean in period_mean_hours]
        colors_period = ['red' if x > 0 else 'blue' for x in period_deviations]
        axes[1, 0].bar(x_pos, period_deviations, color=colors_period, alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 0].set_xlabel(f'{resampling_years}-Year Periods')
        axes[1, 0].set_ylabel('Deviation from Period Means Average (hours)', fontsize=24)
        axes[1, 0].set_title(f'{resampling_years}-Year Period Deviations')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(period_labels, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Subplot 4: Cells with exceedance percentage
        cells_with_exceedance_pct = [(period_stats[label]['cells_with_exceedance'] / 
                                     period_stats[label]['n_grid_cells'] * 100) 
                                    for label in period_labels]
        overall_pct = (long_term_stats['cells_with_exceedance'] / 
                      long_term_stats['total_grid_cells'] * 100)
        
        axes[1, 1].bar(x_pos, cells_with_exceedance_pct, alpha=0.7, color='gold', edgecolor='orange')
        axes[1, 1].axhline(y=overall_pct, color='purple', linestyle='--', linewidth=2,
                           label=f'Overall ({overall_pct:.1f}%)')
        axes[1, 1].set_xlabel(f'{resampling_years}-Year Periods')
        axes[1, 1].set_ylabel('Cells with Exceedance (%)', fontsize=24)
        axes[1, 1].set_title(f'{resampling_years}-Year Period Spatial Coverage')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(period_labels, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plot2_path = os.path.join(base_results_dir, f"exceedance_hours_{resampling_years}year_resampling_analysis.png")
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            print(f"Resampling analysis plot saved to: {plot2_path}")
        
        plt.close()
    
    # Plot 3: Percentiles and distribution evolution
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Subplot 1: Multiple percentiles over time
    axes[0].plot(years_list, yearly_percentiles['p75'], 'g-o', linewidth=2, markersize=3, label='75th Percentile')
    axes[0].plot(years_list, yearly_percentiles['p90'], 'orange', marker='s', linewidth=2, markersize=3, label='90th Percentile')
    axes[0].plot(years_list, yearly_percentiles['p95'], 'r-^', linewidth=2, markersize=3, label='95th Percentile')
    axes[0].plot(years_list, yearly_mean_hours, 'b-', linewidth=2, alpha=0.7, label='Mean')
    
    # Add long-term averages (average of yearly percentiles and yearly means)
    axes[0].axhline(y=np.mean(yearly_percentiles['p75']), color='green', linestyle='--', alpha=0.7)
    axes[0].axhline(y=np.mean(yearly_percentiles['p90']), color='orange', linestyle='--', alpha=0.7)
    axes[0].axhline(y=np.mean(yearly_percentiles['p95']), color='red', linestyle='--', alpha=0.7)
    axes[0].axhline(y=np.mean(yearly_mean_hours), color='blue', linestyle='--', alpha=0.7)
    
    axes[0].set_xlabel('Year', fontsize=24)
    axes[0].set_ylabel('Exceedance Hours per Cell', fontsize=24)
    axes[0].set_title('Exceedance Hours Percentiles Evolution Over Time', fontsize=28)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Subplot 2: Yearly statistics distribution
    axes[1].boxplot([yearly_mean_hours, yearly_percentiles['p75'], yearly_percentiles['p90'], yearly_percentiles['p95']], 
                    labels=['Mean', 'P75', 'P90', 'P95'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel('Exceedance Hours per Cell', fontsize=24)
    axes[1].set_title('Distribution of Yearly Exceedance Hours Statistics', fontsize=28)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plot3_path = os.path.join(base_results_dir, "exceedance_hours_percentiles_evolution.png")
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        print(f"Percentiles evolution plot saved to: {plot3_path}")
    
    plt.close()
    
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
                f.write(f"- exceedance_hours_{resampling_years}year_resampling_analysis.png\n")
            f.write("- exceedance_hours_percentiles_evolution.png\n")
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


# CORRELATION WITH METEOROLOGICAL VARIABLES

def correlation_with_met_variables(dataset_with_ice_load, met_variable, ice_load_variable='ICE_LOAD', 
                                 height_level=0, n_bins=50, ice_load_threshold=0.0):
    """
    Binned correlation visualization between ice load and meteorological variables.
    
    Parameters:
    -----------
    dataset_with_ice_load : xarray.Dataset
        Dataset containing ice load and meteorological variables
    met_variable : str
        Name of the meteorological variable to correlate with ice load
    ice_load_variable : str, optional
        Name of the ice load variable (default: 'ICE_LOAD')
    height_level : int, optional
        Height level index to analyze (default: 0)
    n_bins : int, optional
        Number of meteorological variable bins (default: 50)
    ice_load_threshold : float, optional
        Minimum ice load threshold - only analyze data where ice load >= threshold (default: 0.0 kg/m)
        
    Returns:
    --------
    dict
        Dictionary containing binned correlation statistics and bin data
    """
    
    print(f"=== BINNED CORRELATION PLOT: {met_variable} vs {ice_load_variable} ===")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
        import os
        
        # Check if variables exist in dataset
        if ice_load_variable not in dataset_with_ice_load.data_vars:
            raise ValueError(f"Ice load variable '{ice_load_variable}' not found in dataset")
        
        if met_variable not in dataset_with_ice_load.data_vars:
            raise ValueError(f"Meteorological variable '{met_variable}' not found in dataset")
        
        print(f"Analyzing: {met_variable} vs {ice_load_variable}")
        print(f"Height level: {height_level} ({dataset_with_ice_load.height.values[height_level]} m)")
        print(f"Ice load threshold: {ice_load_threshold} kg/m")
        print(f"Number of bins: {n_bins}")
        
        # Extract ice load data at specified height level
        ice_load_data = dataset_with_ice_load[ice_load_variable].isel(height=height_level)
        
        # Extract meteorological data (handle both 2D and 3D variables)
        if 'height' in dataset_with_ice_load[met_variable].dims:
            met_data = dataset_with_ice_load[met_variable].isel(height=height_level)
            print(f"Using {met_variable} at height level {height_level}")
        else:
            met_data = dataset_with_ice_load[met_variable]
            print(f"Using {met_variable} (surface data)")
        
        # Flatten data
        ice_load_flat = ice_load_data.values.flatten()
        met_flat = met_data.values.flatten()
        
        # Remove NaN values and apply ice load threshold
        valid_mask = ~(np.isnan(ice_load_flat) | np.isnan(met_flat))
        threshold_mask = ice_load_flat >= ice_load_threshold
        combined_mask = valid_mask & threshold_mask
        
        ice_load_clean = ice_load_flat[combined_mask]
        met_clean = met_flat[combined_mask]
        
        print(f"Total data points: {len(ice_load_flat):,}")
        print(f"Valid data points (no NaN): {np.sum(valid_mask):,}")
        print(f"Points above threshold ({ice_load_threshold} kg/m): {np.sum(threshold_mask):,}")
        print(f"Final analysis points: {len(ice_load_clean):,}")
        print(f"Data reduction: {(1 - len(ice_load_clean)/len(ice_load_flat))*100:.1f}%")
        
        if len(ice_load_clean) == 0:
            print("No valid data points found!")
            return None
        
        # Use all valid data points (no sampling needed since we're binning)
        ice_load_sample = ice_load_clean
        met_sample = met_clean
        print(f"Using all {len(ice_load_sample):,} points for binning")
        
        # Create meteorological variable bins (X-axis)
        met_min = np.min(met_sample)
        met_max = np.max(met_sample)
        
        # Create bin edges for meteorological variable
        bin_edges = np.linspace(met_min, met_max, n_bins + 1)
        bin_width = (met_max - met_min) / n_bins
        
        print(f"{met_variable} range: {met_min:.3f} to {met_max:.3f}")
        print(f"Bin width: {bin_width:.3f}")
        
        # Calculate bin centers and statistics
        bin_centers = []
        bin_means = []
        bin_stds = []
        bin_counts = []
        
        for i in range(len(bin_edges) - 1):
            # Find data points in this meteorological bin
            mask = (met_sample >= bin_edges[i]) & (met_sample < bin_edges[i + 1])
            ice_load_in_bin = ice_load_sample[mask]
            
            if len(ice_load_in_bin) > 0:  # Only process bins with data
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                bin_mean = np.mean(ice_load_in_bin)
                bin_std = np.std(ice_load_in_bin)
                bin_count = len(ice_load_in_bin)
                
                bin_centers.append(bin_center)
                bin_means.append(bin_mean)
                bin_stds.append(bin_std)
                bin_counts.append(bin_count)
        
        # Convert to arrays
        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        bin_counts = np.array(bin_counts)
        
        print(f"Bins with data: {len(bin_centers)}")
        
        if len(bin_centers) == 0:
            print("No bins contain data!")
            return None
        
        # Calculate correlations on binned data
        pearson_corr, pearson_p = stats.pearsonr(bin_centers, bin_means)
        spearman_corr, spearman_p = stats.spearmanr(bin_centers, bin_means)
        
        print(f"Binned Pearson correlation: r = {pearson_corr:.4f}, p = {pearson_p:.4f}")
        print(f"Binned Spearman correlation: rho = {spearman_corr:.4f}, p = {spearman_p:.4f}")
        
        # Create binned plot
        plt.figure(figsize=(12, 8))
        
        # Main plot: meteorological variable vs mean ice load
        plt.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=3, 
                     markersize=4, linewidth=1, alpha=0.8, color='blue')
        
        plt.xlabel(f'{met_variable}')
        plt.ylabel(f'Mean {ice_load_variable} (kg/m)')
        plt.title(f'Binned Correlation: {met_variable} vs Ice Load\n'
                 f'Height: {dataset_with_ice_load.height.values[height_level]} m, '
                 f'Threshold: {ice_load_threshold} kg/m, Bins: {n_bins}\n'
                 f'Pearson r = {pearson_corr:.4f}, Spearman rho = {spearman_corr:.4f}')
        plt.grid(True, alpha=0.3)
        
        # Add secondary plot showing bin counts
        ax2 = plt.gca().twinx()
        ax2.bar(bin_centers, bin_counts, alpha=0.3, width=bin_width*0.8, 
                color='gray', label='Data count per bin')
        ax2.set_ylabel('Number of data points per bin', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # Add legend
        plt.legend([f'Mean {ice_load_variable} ± Std Dev'], loc='upper left')
        ax2.legend(loc='upper right')
        
        # Save plot
        results_dir = os.path.join("results", "figures", "correlation_ice_load", met_variable)
        os.makedirs(results_dir, exist_ok=True)
        
        height_m = int(dataset_with_ice_load.height.values[height_level])
        plot_path = os.path.join(results_dir, f"binned_correlation_{met_variable}_h{height_m}_thresh{ice_load_threshold:.3f}_bins{n_bins}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Plot saved: {plot_path}")
        
        # Return results
        results = {
            'bin_centers': bin_centers.tolist(),
            'bin_means': bin_means.tolist(),
            'bin_stds': bin_stds.tolist(),
            'bin_counts': bin_counts.tolist(),
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'n_bins': len(bin_centers),
            'total_data_points': len(ice_load_sample),
            'ice_load_threshold': ice_load_threshold,
            'n_bins_requested': n_bins,
            'met_variable_range': (met_min, met_max),
            'plot_path': plot_path
        }
        
        print("✓ Binned correlation analysis completed!")
        return results
        
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


# EMD

def plot_grid_with_extra_point(dataset, extra_point_coords, extra_point_label='EMD', 
                              save_plot=True, plot_title="Grid Points with Extra Point"):
    """
    Plot dataset grid coordinates (XLON vs XLAT) and add an extra point.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Dataset containing XLAT and XLON coordinates
    extra_point_coords : tuple
        Coordinates of the extra point as (longitude, latitude)
    extra_point_label : str, optional
        Label for the extra point (default: 'EMD')
    save_plot : bool, optional
        Whether to save the plot to file (default: True)
    plot_title : str, optional
        Title for the plot (default: "Grid Points with Extra Point")
        
    Returns:
    --------
    dict
        Dictionary containing plot information and grid statistics
    """
    
    print(f"=== GRID VISUALIZATION WITH EXTRA POINT ===")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Check if coordinates exist in dataset
        if 'XLAT' not in dataset and 'XLAT' not in dataset.coords:
            raise ValueError("'XLAT' coordinate not found in dataset")
        if 'XLON' not in dataset and 'XLON' not in dataset.coords:
            raise ValueError("'XLON' coordinate not found in dataset")
        
        # Extract coordinates - handle both data variables and coordinates
        if 'XLAT' in dataset.coords:
            lats = dataset.coords['XLAT'].values
            lons = dataset.coords['XLON'].values
        else:
            lats = dataset['XLAT'].values
            lons = dataset['XLON'].values
        
        # Get extra point coordinates
        extra_lon, extra_lat = extra_point_coords
        
        print(f"Grid information:")
        print(f"  Longitude range: {lons.min():.4f} to {lons.max():.4f}°")
        print(f"  Latitude range: {lats.min():.4f} to {lats.max():.4f}°")
        print(f"  Grid size: {lats.shape[0]} × {lats.shape[1]}")
        print(f"  Extra point ({extra_point_label}): ({extra_lon:.4f}°, {extra_lat:.4f}°)")
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Plot all grid points in blue
        plt.scatter(lons.flatten(), lats.flatten(), 
                   c='blue', s=100, alpha=0.7, edgecolors='darkblue', linewidth=1,
                   label='Grid Points')
        
        # Plot the extra point in red
        plt.scatter(extra_lon, extra_lat, 
                   c='red', s=200, alpha=0.9, edgecolors='darkred', linewidth=2,
                   marker='*', label=extra_point_label)
        
        # Customize the plot
        plt.xlabel('Longitude (°)', fontsize=24)
        plt.ylabel('Latitude (°)', fontsize=24)
        plt.title(plot_title)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=24)
        
        # Add coordinate labels for grid points (if not too many)
        if lats.size <= 25:  # Only for small grids
            for i in range(lats.shape[0]):
                for j in range(lats.shape[1]):
                    plt.annotate(f'({j},{i})', (lons[i,j], lats[i,j]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=24, alpha=0.7)
        
        # Add coordinate annotation for extra point
        plt.annotate(f'{extra_point_label}\n({extra_lon:.4f}°, {extra_lat:.4f}°)', 
                    (extra_lon, extra_lat), 
                    xytext=(10, 10), textcoords='offset points', 
                    fontsize=30, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Set equal aspect ratio to maintain grid proportions
        plt.axis('equal')
        
        # Adjust limits to show all points with some margin
        lon_margin = (lons.max() - lons.min()) * 0.1
        lat_margin = (lats.max() - lats.min()) * 0.1
        
        plt.xlim(min(lons.min(), extra_lon) - lon_margin, 
                max(lons.max(), extra_lon) + lon_margin)
        plt.ylim(min(lats.min(), extra_lat) - lat_margin, 
                max(lats.max(), extra_lat) + lat_margin)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            results_dir = os.path.join("results", "figures", "EMD")
            os.makedirs(results_dir, exist_ok=True)
            
            plot_filename = "grid_EMD_NEWA_points.png"
            plot_path = os.path.join(results_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved: {plot_path}")
        
        # Do not show the plot
        plt.close()
        
        # Prepare results
        results = {
            'grid_info': {
                'longitude_range': (float(lons.min()), float(lons.max())),
                'latitude_range': (float(lats.min()), float(lats.max())),
                'grid_size': lats.shape,
                'n_grid_points': lats.size
            },
            'extra_point': {
                'label': extra_point_label,
                'coordinates': extra_point_coords,
                'longitude': extra_lon,
                'latitude': extra_lat
            },
            'plot_info': {
                'title': plot_title,
                'saved': save_plot,
                'filename': plot_filename if save_plot else None
            }
        }
        
        print(f"✓ Grid visualization completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in grid visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

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
            ax1.set_ylabel('Ice Load (kg/m)', fontsize=24)
            ax1.set_title(f'Hourly Ice Load Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Lines')
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
            ax2.set_ylabel('Ice Load (kg/m)', fontsize=24)
            ax2.set_title(f'Daily Mean Ice Load Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Lines')
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
            ax3.set_xlabel('Time', fontsize=24)
            ax3.set_ylabel('Ice Load (kg/m)', fontsize=24)
            ax3.set_title(f'Weekly Mean Ice Load Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Lines')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle(f'Multi-Scale Ice Load Comparison: EMD vs NEWA at {height}m (Lines Only)\n'
                        f'NEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km',
                        fontsize=32, y=0.98)
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
            ax1.set_ylabel('Ice Load (kg/m)', fontsize=24)
            ax1.set_title(f'Hourly Ice Load Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Scatter')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: Daily averages (scatter only)
            ax2.scatter(emd_daily_avg.index, emd_daily_avg.values, c='blue', s=3, alpha=0.7, label=f'EMD Daily Mean ({emd_column})')
            ax2.scatter(newa_daily_avg.index, newa_daily_avg.values, c='red', s=3, alpha=0.7, label=f'NEWA Daily Mean (ICE_LOAD)')
            ax2.set_ylabel('Ice Load (kg/m)', fontsize=24)
            ax2.set_title(f'Daily Mean Ice Load Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Scatter')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Subplot 3: Weekly averages (scatter only)
            ax3.scatter(emd_weekly_avg.index, emd_weekly_avg.values, c='blue', s=10, alpha=0.8, label=f'EMD Weekly Mean ({emd_column})')
            ax3.scatter(newa_weekly_avg.index, newa_weekly_avg.values, c='red', s=10, alpha=0.8, label=f'NEWA Weekly Mean (ICE_LOAD)')
            ax3.set_xlabel('Time', fontsize=24)
            ax3.set_ylabel('Ice Load (kg/m)', fontsize=24)
            ax3.set_title(f'Weekly Mean Ice Load Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Scatter')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle(f'Multi-Scale Ice Load Comparison: EMD vs NEWA at {height}m (Scatter Only)\n'
                        f'NEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km',
                        fontsize=32, y=0.98)
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
            ax1.set_ylabel('Difference (NEWA - EMD) [kg/m]', fontsize=24)
            ax1.set_title(f'Hourly Ice Load Differences: NEWA - EMD at {height}m (Icing Season Only) - Lines')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Daily differences - mean of hourly differences for each day (lines only)
            daily_differences_ts = newa_daily_avg - emd_daily_avg
            daily_bias = daily_differences_ts.mean()
            ax2.plot(daily_differences_ts.index, daily_differences_ts.values, 'g-', alpha=0.8, linewidth=1.0)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax2.axhline(y=daily_bias, color='red', linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Daily Mean Bias: {daily_bias:.3f} kg/m')
            ax2.set_ylabel('Difference (NEWA - EMD) [kg/m]', fontsize=24)
            ax2.set_title(f'Daily Mean Ice Load Differences: NEWA - EMD at {height}m (Icing Season Only) - Lines')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Weekly differences - mean of hourly differences for each week (lines only)
            weekly_differences_ts = newa_weekly_avg - emd_weekly_avg
            weekly_bias = weekly_differences_ts.mean()
            ax3.plot(weekly_differences_ts.index, weekly_differences_ts.values, 'g-', alpha=0.9, linewidth=1.5)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax3.axhline(y=weekly_bias, color='red', linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Weekly Mean Bias: {weekly_bias:.3f} kg/m')
            ax3.set_xlabel('Time', fontsize=24)
            ax3.set_ylabel('Difference (NEWA - EMD) [kg/m]', fontsize=24)
            ax3.set_title(f'Weekly Mean Ice Load Differences: NEWA - EMD at {height}m (Icing Season Only) - Lines')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle(f'Multi-Scale Ice Load Differences: NEWA - EMD at {height}m (Lines Only)\n'
                        f'NEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km',
                        fontsize=32, y=0.98)
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
            ax1.set_ylabel('Difference (NEWA - EMD) [kg/m]', fontsize=24)
            ax1.set_title(f'Hourly Ice Load Differences: NEWA - EMD at {height}m (Icing Season Only) - Scatter')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Daily differences - mean of hourly differences for each day (scatter only)
            ax2.scatter(daily_differences_ts.index, daily_differences_ts.values, c='green', s=3, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax2.axhline(y=daily_bias, color='red', linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Daily Mean Bias: {daily_bias:.3f} kg/m')
            ax2.set_ylabel('Difference (NEWA - EMD) [kg/m]', fontsize=24)
            ax2.set_title(f'Daily Mean Ice Load Differences: NEWA - EMD at {height}m (Icing Season Only) - Scatter')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Weekly differences - mean of hourly differences for each week (scatter only)
            ax3.scatter(weekly_differences_ts.index, weekly_differences_ts.values, c='green', s=10, alpha=0.8)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax3.axhline(y=weekly_bias, color='red', linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Weekly Mean Bias: {weekly_bias:.3f} kg/m')
            ax3.set_xlabel('Time', fontsize=24)
            ax3.set_ylabel('Difference (NEWA - EMD) [kg/m]', fontsize=24)
            ax3.set_title(f'Weekly Mean Ice Load Differences: NEWA - EMD at {height}m (Icing Season Only) - Scatter')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle(f'Multi-Scale Ice Load Differences: NEWA - EMD at {height}m (Scatter Only)\n'
                        f'NEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km',
                        fontsize=32, y=0.98)
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
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=22,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # Set labels and title
            ax.set_xlabel(f'NEWA Ice Load (kg/m) at {height}m', fontsize=24)
            ax.set_ylabel(f'EMD Ice Load (kg/m) at {height}m', fontsize=24)
            ax.set_title(f'EMD vs NEWA Ice Load Scatter Plot at {height}m (Icing Season Only)\n'
                        f'NEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km',
                        fontsize=28, pad=15)
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=30)
            
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
                
                ax.set_xlabel(f'NEWA Ice Load (kg/m) at {height}m', fontsize=24)
                ax.set_ylabel(f'EMD Ice Load (kg/m) at {height}m', fontsize=24)
                ax.set_title(f'EMD vs NEWA Ice Load Scatter Plot at {height}m (Non-Zero Values Only)\n'
                            f'NEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km',
                            fontsize=28, pad=15)
                
                # Add grid and legend
                ax.grid(True, alpha=0.3)
                ax.legend(loc='lower right', fontsize=30)
                
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
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=22,
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
                       ha='center', va='bottom', fontweight='bold', fontsize=22)
            
            ax.set_ylabel('Percentage of Zero Values (%)', fontsize=24, fontweight='bold')
            ax.set_title(f'Zero Value Analysis at {height:.0f}m\n'
                        f'Total timestamps: {total_timestamps:,} hours\n'
                        f'NEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km',
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
                   verticalalignment='bottom', horizontalalignment='left', fontsize=30)
            
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
                
                ax.set_ylabel('Ice Load [kg/m]', fontsize=24, fontweight='bold')
                ax.set_title(f'Distribution of Positive Ice Load Values at {height}m\n'
                            f'Positive values: EMD={len(emd_positive):,}, NEWA={len(newa_positive):,}\n'
                            f'NEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km',
                            fontsize=28, fontweight='bold')
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
                       verticalalignment='top', fontsize=27, family='monospace')
                
                # Add legend explaining box plot elements
                legend_text = 'Box Plot Elements:\n'
                legend_text += '━ Red line: Median\n'
                legend_text += '┅ Black line: Mean\n'
                legend_text += '□ Box: Q25-Q75 (IQR)\n'
                legend_text += '┬ Whiskers: 1.5×IQR\n'
                legend_text += '○ Outliers'
                
                ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                       verticalalignment='top', fontsize=27)
                
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
            cbar.set_label('Hourly Mean Ice Load Difference (NEWA - EMD) [kg/m]', fontsize=28)
            cbar.ax.tick_params(labelsize=30)
            
            # Set labels and ticks with better formatting for all months
            plt.xlabel('Day of Year', fontsize=28)
            plt.ylabel('Year', fontsize=28)
            plt.title(f'Daily Mean Ice Load Differences Grid: NEWA - EMD at {height}m (All Months)\n'
                     f'NEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km\n'
                     f'Each cell = daily mean difference for that specific year and day', fontsize=32, pad=20)
            
            # Set year labels
            year_indices = np.arange(0, len(pivot_grid.index))
            year_step = max(1, len(pivot_grid.index)//15)
            year_ticks = year_indices[::year_step]
            plt.yticks(year_ticks, [pivot_grid.index[i] for i in year_ticks], fontsize=24)
            
            # Set day of year labels (all months)
            month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.xticks(month_starts, month_labels, rotation=0, fontsize=24)
            
            # Add secondary x-axis with day numbers
            ax2 = plt.gca().secondary_xaxis('top')
            day_ticks = np.arange(0, 366, 30)
            ax2.set_xticks(day_ticks)
            ax2.set_xlabel('Day of Year', fontsize=24)
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
            
            ax1.set_ylabel('Mean Daily Ice Load [kg/m]\n(Mean of Hourly Values)', fontsize=24, fontweight='bold')
            ax1.set_title(f'Typical Day Comparison at {height}m\n'
                         f'Distribution of Daily Means of Hourly Ice Load (n={len(emd_daily_aligned)})', fontsize=22, fontweight='bold')
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
            
            ax2.set_ylabel('Mean Weekly Ice Load [kg/m]\n(Mean of Hourly Values)', fontsize=24, fontweight='bold')
            ax2.set_title(f'Typical Week Comparison at {height}m\n'
                         f'Distribution of Weekly Means of Hourly Ice Load (n={len(emd_weekly_aligned)})', fontsize=22, fontweight='bold')
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
            
            ax3.set_ylabel('Mean Yearly Ice Load [kg/m]\n(Mean of Hourly Values)', fontsize=24, fontweight='bold')
            ax3.set_title(f'Typical Year Comparison at {height}m\n'
                         f'Distribution of Yearly Means of Hourly Ice Load (n={len(emd_yearly_aligned)})', fontsize=22, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add overall title with filtering information
            threshold_text = f"Ice Load Threshold: >={ice_load_threshold} kg/m" if ice_load_threshold > 0 else "No Ice Load Threshold"
            nonzero_text = f"Non-Zero Filter: >={non_zero_percentage}% hours > 0" if non_zero_percentage > 0 else "No Non-Zero Filter"
            
            fig.suptitle(f'Typical Patterns Analysis: EMD vs NEWA at {height}m (Icing Season Only)\n'
                        f'Temporal Means of Hourly Ice Load Values\n'
                        f'{threshold_text} | {nonzero_text}\n'
                        f'NEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km',
                        fontsize=28, fontweight='bold', y=0.96)
            
            # Add legend explaining box plot elements
            legend_text = 'Box Plot Elements:\n━ Red line: Median\n┅ Black line: Mean\n□ Box: Q25-Q75 (IQR)\n┬ Whiskers: 1.5×IQR\n○ Outliers'
            fig.text(0.02, 0.02, legend_text, fontsize=27, 
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
            table.set_fontsize(30)
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
                        f'Temporal Means of Hourly Ice Load Values in kg/m (Icing Season Only)\n'
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

            fig, axes = plt.subplots(2, 3, figsize=(24, 12))
            ax1 = axes[0, 0]
            bins = np.linspace(data_min, data_max, 50)
            ax1.hist(emd_final, bins=bins, alpha=0.6, density=True, color='steelblue', edgecolor='darkblue', linewidth=1, label=f'EMD (n={len(emd_final)})')
            ax1.hist(newa_final, bins=bins, alpha=0.6, density=True, color='orange', edgecolor='darkorange', linewidth=1, label=f'NEWA (n={len(newa_final)})')
            ax1.set_xlabel('Ice Load [kg/m]', fontweight='bold')
            ax1.set_ylabel('Probability Density', fontweight='bold')
            ax1.set_title('Probability Density Functions - Histograms', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = axes[0, 1]
            if len(emd_final) > 10:
                kde_emd = stats.gaussian_kde(emd_final)
                ax2.plot(x_range, kde_emd(x_range), 'steelblue', linewidth=3, label=f'EMD (μ={emd_stats["mean"]:.3f}, σ={emd_stats["std"]:.3f})')
            if len(newa_final) > 10:
                kde_newa = stats.gaussian_kde(newa_final)
                ax2.plot(x_range, kde_newa(x_range), 'orange', linewidth=3, label=f'NEWA (μ={newa_stats["mean"]:.3f}, σ={newa_stats["std"]:.3f})')
            ax2.axvline(emd_stats['mean'], color='steelblue', linestyle='--', alpha=0.8, label='EMD Mean')
            ax2.axvline(newa_stats['mean'], color='orange', linestyle='--', alpha=0.8, label='NEWA Mean')
            ax2.set_xlabel('Ice Load [kg/m]', fontweight='bold')
            ax2.set_ylabel('Probability Density', fontweight='bold')
            ax2.set_title('Kernel Density Estimation (KDE) Comparison', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # New log-log PDF comparison plot
            ax3 = axes[0, 2]
            # Filter out zero values for log-log plot
            emd_nonzero = emd_final[emd_final > 0]
            newa_nonzero = newa_final[newa_final > 0]
            
            if len(emd_nonzero) > 10 and len(newa_nonzero) > 10:
                # Create log-spaced bins for PDF calculation
                log_min = max(1e-6, min(np.min(emd_nonzero), np.min(newa_nonzero)))
                log_max = max(np.max(emd_nonzero), np.max(newa_nonzero))
                bins = np.logspace(np.log10(log_min), np.log10(log_max), 50)
                
                # Calculate PDF (normalized histogram)
                emd_counts, _ = np.histogram(emd_nonzero, bins=bins)
                newa_counts, _ = np.histogram(newa_nonzero, bins=bins)
                
                # Normalize to get PDF
                bin_widths = np.diff(bins)
                emd_pdf = emd_counts / (len(emd_nonzero) * bin_widths)
                newa_pdf = newa_counts / (len(newa_nonzero) * bin_widths)
                
                # Plot centers
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Remove zeros for log-log plot
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
                        ha='center', va='center', transform=ax3.transAxes, fontsize=24)
            
            ax3.set_xlabel('Ice Load [kg/m]', fontweight='bold')
            ax3.set_ylabel('Probability Density', fontweight='bold')
            ax3.set_title('PDF Comparison (Log-Log Scale)', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3, which="both")

            ax4 = axes[1, 0]
            n_quantiles = min(len(emd_final), len(newa_final), 1000)
            quantiles = np.linspace(0.01, 0.99, n_quantiles)
            emd_quantiles = np.quantile(emd_final, quantiles)
            newa_quantiles = np.quantile(newa_final, quantiles)
            ax4.scatter(emd_quantiles, newa_quantiles, alpha=0.6, s=20, color='purple')
            min_val = min(np.min(emd_quantiles), np.min(newa_quantiles))
            max_val = max(np.max(emd_quantiles), np.max(newa_quantiles))
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Agreement')
            qq_correlation = np.corrcoef(emd_quantiles, newa_quantiles)[0, 1]
            ax4.set_xlabel('EMD Quantiles [kg/m]', fontweight='bold')
            ax4.set_ylabel('NEWA Quantiles [kg/m]', fontweight='bold')
            ax4.set_title(f'Q-Q Plot (r = {qq_correlation:.3f})', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            ax5 = axes[1, 1]
            box_data = [emd_final, newa_final]
            labels = ['EMD', 'NEWA']
            colors = ['steelblue', 'orange']
            box_plot = ax5.boxplot(box_data, labels=labels, patch_artist=True, showmeans=True, meanline=True, boxprops=dict(alpha=0.7), medianprops=dict(color='red', linewidth=2), meanprops=dict(color='black', linewidth=2, linestyle='--'))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            ax5.set_ylabel('Ice Load [kg/m]', fontweight='bold')
            ax5.set_title('Distribution Comparison (Box Plots)', fontweight='bold')
            ax5.grid(True, alpha=0.3)

            # Add statistical summary table to the last subplot (axes[1, 2])
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            # Create statistical summary table
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
            table.set_fontsize(27)
            table.scale(1, 1.5)
            
            # Style the table
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

            ax6.set_title('Statistical Summary', fontweight='bold', pad=20)

            threshold_text = f"Ice Load Threshold: >={ice_load_threshold} kg/m" if ice_load_threshold > 0 else "No Ice Load Threshold"
            nonzero_text = f"Non-Zero Filter: >={non_zero_percentage}% hours > 0" if non_zero_percentage > 0 else "No Non-Zero Filter"
            
            fig.suptitle(f'PDF Analysis: EMD vs NEWA Ice Load at {height}m - 6 Comprehensive Plots\n'
                        f'Distribution Comparison (Icing Season Only) | {threshold_text} | {nonzero_text}\n'
                        f'NEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km',
                        fontsize=32, fontweight='bold', y=0.96)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            
            pdf_plot_path = os.path.join(base_dir, f'pdf_comparison_{height:.0f}m.png')
            plt.savefig(pdf_plot_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {pdf_plot_path}")
            
            print(f"\n=== PLOT SUMMARY ===")
            print(f"Created 1 comprehensive plot with 6 subplots:")
            print(f"  1. PDF comparison plot: {pdf_plot_path}")
        
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
            ax1.set_ylabel('Ice Accretion (g/h)', fontsize=24)
            ax1.set_title(f'Hourly Ice Accretion Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Lines')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            emd_daily_avg = emd_clean.resample('D').mean()
            newa_daily_avg = newa_clean.resample('D').mean()
            ax2.plot(emd_daily_avg.index, emd_daily_avg.values, 'b-', alpha=0.8, linewidth=1.0, label=f'EMD Daily Mean ({emd_column})')
            ax2.plot(newa_daily_avg.index, newa_daily_avg.values, 'r-', alpha=0.8, linewidth=1.0, label=f'NEWA Daily Mean (ACCRE_CYL)')
            ax2.set_ylabel('Ice Accretion (g/h)', fontsize=24)
            ax2.set_title(f'Daily Mean Ice Accretion Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Lines')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            emd_weekly_avg = emd_clean.resample('W').mean()
            newa_weekly_avg = newa_clean.resample('W').mean()
            ax3.plot(emd_weekly_avg.index, emd_weekly_avg.values, 'b-', alpha=0.9, linewidth=1.5, label=f'EMD Weekly Mean ({emd_column})')
            ax3.plot(newa_weekly_avg.index, newa_weekly_avg.values, 'r-', alpha=0.9, linewidth=1.5, label=f'NEWA Weekly Mean (ACCRE_CYL)')
            ax3.set_xlabel('Time', fontsize=24)
            ax3.set_ylabel('Ice Accretion (g/h)', fontsize=24)
            ax3.set_title(f'Weekly Mean Ice Accretion Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Lines')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.suptitle(f'Multi-Scale Ice Accretion Comparison: EMD vs NEWA at {height}m (Lines Only)\nNEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km', fontsize=32, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            timeseries_lines_path = os.path.join(base_dir, f'multi_scale_timeseries_lines_{height:.0f}m.png')
            plt.savefig(timeseries_lines_path, dpi=150, facecolor='white')
            plt.close()
            print(f"Saved: {timeseries_lines_path}")

            # Plot 1B: Time series comparison with scatter only
            print("1B. Creating full time series comparison (scatter only)...")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))
            ax1.scatter(emd_clean.index, emd_clean.values, c='blue', s=0.5, alpha=0.6, label=f'EMD Hourly ({emd_column})')
            ax1.scatter(newa_clean.index, newa_clean.values, c='red', s=0.5, alpha=0.6, label=f'NEWA Hourly (ACCRE_CYL)')
            ax1.set_ylabel('Ice Accretion (g/h)', fontsize=24)
            ax1.set_title(f'Hourly Ice Accretion Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Scatter')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax2.scatter(emd_daily_avg.index, emd_daily_avg.values, c='blue', s=3, alpha=0.7, label=f'EMD Daily Mean ({emd_column})')
            ax2.scatter(newa_daily_avg.index, newa_daily_avg.values, c='red', s=3, alpha=0.7, label=f'NEWA Daily Mean (ACCRE_CYL)')
            ax2.set_ylabel('Ice Accretion (g/h)', fontsize=24)
            ax2.set_title(f'Daily Mean Ice Accretion Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Scatter')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax3.scatter(emd_weekly_avg.index, emd_weekly_avg.values, c='blue', s=10, alpha=0.8, label=f'EMD Weekly Mean ({emd_column})')
            ax3.scatter(newa_weekly_avg.index, newa_weekly_avg.values, c='red', s=10, alpha=0.8, label=f'NEWA Weekly Mean (ACCRE_CYL)')
            ax3.set_xlabel('Time', fontsize=24)
            ax3.set_ylabel('Ice Accretion (g/h)', fontsize=24)
            ax3.set_title(f'Weekly Mean Ice Accretion Time Series: EMD vs NEWA at {height}m (Icing Season Only) - Scatter')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.suptitle(f'Multi-Scale Ice Accretion Comparison: EMD vs NEWA at {height}m (Scatter Only)\nNEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km', fontsize=32, y=0.98)
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
            ax1.set_ylabel('Difference (NEWA - EMD) [g/h]', fontsize=24)
            ax1.set_title(f'Hourly Ice Accretion Differences: NEWA - EMD at {height}m (Icing Season Only) - Lines')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            daily_differences_ts = newa_daily_avg - emd_daily_avg
            daily_bias = daily_differences_ts.mean()
            ax2.plot(daily_differences_ts.index, daily_differences_ts.values, 'g-', alpha=0.8, linewidth=1.0)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax2.axhline(y=daily_bias, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Daily Mean Bias: {daily_bias:.3f} g/h')
            ax2.set_ylabel('Difference (NEWA - EMD) [g/h]', fontsize=24)
            ax2.set_title(f'Daily Mean Ice Accretion Differences: NEWA - EMD at {height}m (Icing Season Only) - Lines')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            weekly_differences_ts = newa_weekly_avg - emd_weekly_avg
            weekly_bias = weekly_differences_ts.mean()
            ax3.plot(weekly_differences_ts.index, weekly_differences_ts.values, 'g-', alpha=0.9, linewidth=1.5)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax3.axhline(y=weekly_bias, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Weekly Mean Bias: {weekly_bias:.3f} g/h')
            ax3.set_xlabel('Time', fontsize=24)
            ax3.set_ylabel('Difference (NEWA - EMD) [g/h]', fontsize=24)
            ax3.set_title(f'Weekly Mean Ice Accretion Differences: NEWA - EMD at {height}m (Icing Season Only) - Lines')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.suptitle(f'Multi-Scale Ice Accretion Differences: NEWA - EMD at {height}m (Lines Only)\nNEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km', fontsize=32, y=0.98)
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
            ax1.set_ylabel('Difference (NEWA - EMD) [g/h]', fontsize=24)
            ax1.set_title(f'Hourly Ice Accretion Differences: NEWA - EMD at {height}m (Icing Season Only) - Scatter')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax2.scatter(daily_differences_ts.index, daily_differences_ts.values, c='green', s=3, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax2.axhline(y=daily_bias, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Daily Mean Bias: {daily_bias:.3f} g/h')
            ax2.set_ylabel('Difference (NEWA - EMD) [g/h]', fontsize=24)
            ax2.set_title(f'Daily Mean Ice Accretion Differences: NEWA - EMD at {height}m (Icing Season Only) - Scatter')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax3.scatter(weekly_differences_ts.index, weekly_differences_ts.values, c='green', s=10, alpha=0.8)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax3.axhline(y=weekly_bias, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Weekly Mean Bias: {weekly_bias:.3f} g/h')
            ax3.set_xlabel('Time', fontsize=24)
            ax3.set_ylabel('Difference (NEWA - EMD) [g/h]', fontsize=24)
            ax3.set_title(f'Weekly Mean Ice Accretion Differences: NEWA - EMD at {height}m (Icing Season Only) - Scatter')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.suptitle(f'Multi-Scale Ice Accretion Differences: NEWA - EMD at {height}m (Scatter Only)\nNEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km', fontsize=32, y=0.98)
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
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            ax.set_xlabel(f'NEWA Ice Accretion (g/h) at {height}m', fontsize=24)
            ax.set_ylabel(f'EMD Ice Accretion (g/h) at {height}m', fontsize=24)
            ax.set_title(f'EMD vs NEWA Ice Accretion Scatter Plot at {height}m (Icing Season Only)\nNEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km', fontsize=28, pad=15)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=30)
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
                ax.set_xlabel(f'NEWA Ice Accretion (g/h) at {height}m', fontsize=24)
                ax.set_ylabel(f'EMD Ice Accretion (g/h) at {height}m', fontsize=24)
                ax.set_title(f'EMD vs NEWA Ice Accretion Scatter Plot at {height}m (Non-Zero Values Only)\nNEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km', fontsize=28, pad=15)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='lower right', fontsize=30)
                ax.set_aspect('equal', adjustable='box')
                stats_text = (f'N = {len(emd_nonzero)}\nR² = {r_value**2:.3f}\nCorrelation = {np.corrcoef(emd_nonzero, newa_nonzero)[0,1]:.3f}\nRMSE = {np.sqrt(np.mean((newa_nonzero - emd_nonzero)**2)):.3f} g/h\nMAE = {np.mean(np.abs(newa_nonzero - emd_nonzero)):.3f} g/h\nBias = {np.mean(newa_nonzero - emd_nonzero):.3f} g/h\nSlope = {slope:.3f}\nIntercept = {intercept:.3f}')
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
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
                ax.text(bar.get_x() + bar.get_width()/2., height_b + 0.5, f'{percentage:.1f}%\n({count:,} hours)', ha='center', va='bottom', fontweight='bold', fontsize=22)
            ax.set_ylabel('Percentage of Zero Values (%)', fontsize=24, fontweight='bold')
            ax.set_title(f'Zero Value Analysis at {height:.0f}m\nTotal timestamps: {total_timestamps:,} hours\nNEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km', fontsize=28, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(zero_percentages) * 1.15)
            stats_text = f'Summary:\n'
            stats_text += f'EMD zeros: {emd_zero_count:,} ({emd_zero_percentage:.1f}%)\n'
            stats_text += f'NEWA zeros: {newa_zero_count:,} ({newa_zero_percentage:.1f}%)\n'
            stats_text += f'Both zero: {((emd_clean == 0) & (newa_clean == 0)).sum():,}\n'
            stats_text += f'Either zero: {((emd_clean == 0) | (newa_clean == 0)).sum():,}'
            ax.text(0.02, 0.05, stats_text, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), verticalalignment='bottom', horizontalalignment='left', fontsize=30)
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
                ax.text(bar.get_x() + bar.get_width()/2., height_b + 0.5, f'{percentage:.1f}%\n({count:,} hours)', ha='center', va='bottom', fontweight='bold', fontsize=22)
            ax.set_ylabel('Percentage of Negative Values (%)', fontsize=24, fontweight='bold')
            ax.set_title(f'Negative Value Analysis at {height:.0f}m\nTotal timestamps: {total_timestamps:,} hours\nNEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km', fontsize=28, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(neg_percentages) * 1.15)
            stats_text = f'Summary:\n'
            stats_text += f'EMD negatives: {emd_neg_count:,} ({emd_neg_percentage:.1f}%)\n'
            stats_text += f'NEWA negatives: {newa_neg_count:,} ({newa_neg_percentage:.1f}%)\n'
            stats_text += f'Both negative: {((emd_clean < 0) & (newa_clean < 0)).sum():,}\n'
            stats_text += f'Either negative: {((emd_clean < 0) | (newa_clean < 0)).sum():,}'
            ax.text(0.02, 0.05, stats_text, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), verticalalignment='bottom', horizontalalignment='left', fontsize=30)
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
                ax.text(bar.get_x() + bar.get_width()/2., height_b + 0.5, f'{percentage:.1f}%\n({count:,} hours)', ha='center', va='bottom', fontweight='bold', fontsize=22)
            ax.set_ylabel('Percentage of Positive Values (%)', fontsize=24, fontweight='bold')
            ax.set_title(f'Positive Value Analysis at {height:.0f}m\nTotal timestamps: {total_timestamps:,} hours\nNEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km', fontsize=28, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(pos_percentages) * 1.15)
            stats_text = f'Summary:\n'
            stats_text += f'EMD positives: {emd_pos_count:,} ({emd_pos_percentage:.1f}%)\n'
            stats_text += f'NEWA positives: {newa_pos_count:,} ({newa_pos_percentage:.1f}%)\n'
            stats_text += f'Both positive: {((emd_clean > 0) & (newa_clean > 0)).sum():,}\n'
            stats_text += f'Either positive: {((emd_clean > 0) | (newa_clean > 0)).sum():,}'
            ax.text(0.02, 0.05, stats_text, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), verticalalignment='bottom', horizontalalignment='left', fontsize=30)
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
                ax.set_ylabel('Ice Accretion [g/h]', fontsize=24, fontweight='bold')
                ax.set_title(f'Distribution of Positive Ice Accretion Values at {height}m\nPositive values: EMD={len(emd_positive):,}, NEWA={len(newa_positive):,}\nNEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km', fontsize=28, fontweight='bold')
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
                ax.text(1.02, 1.0, stats_text, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), verticalalignment='top', fontsize=27, family='monospace')
                legend_text = 'Box Plot Elements:\n'
                legend_text += '━ Red line: Median\n'
                legend_text += '┅ Black line: Mean\n'
                legend_text += '□ Box: Q25-Q75 (IQR)\n'
                legend_text += '┬ Whiskers: 1.5×IQR\n'
                legend_text += '○ Outliers'
                ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), verticalalignment='top', fontsize=27)
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
            cbar.set_label('Hourly Mean Ice Accretion Difference (NEWA - EMD) [g/h]', fontsize=28)
            cbar.ax.tick_params(labelsize=30)
            plt.xlabel('Day of Year', fontsize=28)
            plt.ylabel('Year', fontsize=28)
            plt.title(f'Daily Mean Ice Accretion Differences Grid: NEWA - EMD at {height}m (All Months)\nNEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km\nEach cell = daily mean difference for that specific year and day', fontsize=32, pad=20)
            year_indices = np.arange(0, len(pivot_grid.index))
            year_step = max(1, len(pivot_grid.index)//15)
            year_ticks = year_indices[::year_step]
            plt.yticks(year_ticks, [pivot_grid.index[i] for i in year_ticks], fontsize=24)
            month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.xticks(month_starts, month_labels, rotation=0, fontsize=24)
            ax2 = plt.gca().secondary_xaxis('top')
            day_ticks = np.arange(0, 366, 30)
            ax2.set_xticks(day_ticks)
            ax2.set_xlabel('Day of Year', fontsize=24)
            ax2.tick_params(labelsize=20)
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
            ax1.set_ylabel('Mean Daily Ice Accretion [g/h]\n(Mean of Hourly Values)', fontsize=24, fontweight='bold')
            ax1.set_title(f'Typical Day Comparison at {height}m\nDistribution of Daily Means of Hourly Ice Accretion (n={len(emd_daily_aligned)})', fontsize=22, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax2 = axes[1]
            weekly_data = [emd_weekly_aligned.values, newa_weekly_aligned.values]
            box_plot_weekly = ax2.boxplot(weekly_data, labels=labels, patch_artist=True, showmeans=True, meanline=True, boxprops=dict(alpha=0.7), medianprops=dict(color='red', linewidth=2), meanprops=dict(color='black', linewidth=2, linestyle='--'))
            for patch, color in zip(box_plot_weekly['boxes'], colors):
                patch.set_facecolor(color)
            ax2.set_ylabel('Mean Weekly Ice Accretion [g/h]\n(Mean of Hourly Values)', fontsize=24, fontweight='bold')
            ax2.set_title(f'Typical Week Comparison at {height}m\nDistribution of Weekly Means of Hourly Ice Accretion (n={len(emd_weekly_aligned)})', fontsize=22, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax3 = axes[2]
            yearly_data = [emd_yearly_aligned.values, newa_yearly_aligned.values]
            box_plot_yearly = ax3.boxplot(yearly_data, labels=labels, patch_artist=True, showmeans=True, meanline=True, boxprops=dict(alpha=0.7), medianprops=dict(color='red', linewidth=2), meanprops=dict(color='black', linewidth=2, linestyle='--'))
            for patch, color in zip(box_plot_yearly['boxes'], colors):
                patch.set_facecolor(color)
            ax3.set_ylabel('Mean Yearly Ice Accretion [g/h]\n(Mean of Hourly Values)', fontsize=24, fontweight='bold')
            ax3.set_title(f'Typical Year Comparison at {height}m\nDistribution of Yearly Means of Hourly Ice Accretion (n={len(emd_yearly_aligned)})', fontsize=22, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            threshold_text = f"Ice Accretion Threshold: >={ice_accretion_threshold} g/h" if ice_accretion_threshold > 0 else "No Ice Accretion Threshold"
            nonzero_text = f"Non-Zero Filter: >={non_zero_percentage}% hours > 0" if non_zero_percentage > 0 else "No Non-Zero Filter"
            fig.suptitle(f'Typical Accretion Patterns Analysis: EMD vs NEWA at {height}m (Icing Season Only)\nTemporal Means of Hourly Ice Accretion Values\n{threshold_text} | {nonzero_text}\nNEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km', fontsize=28, fontweight='bold', y=0.96)
            legend_text = 'Box Plot Elements:\n━ Red line: Median\n┅ Black line: Mean\n□ Box: Q25-Q75 (IQR)\n┬ Whiskers: 1.5×IQR\n○ Outliers'
            fig.text(0.02, 0.02, legend_text, fontsize=27, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), verticalalignment='bottom')
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
            table.set_fontsize(30)
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
            ax.set_title(f'Typical Accretion Patterns Statistics Summary at {height}m\nTemporal Means of Hourly Ice Accretion Values in g/h (Icing Season Only)\n{threshold_text} | {nonzero_text}', fontsize=28, fontweight='bold', pad=20)
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

            fig, axes = plt.subplots(2, 3, figsize=(24, 12))
            ax1 = axes[0, 0]
            bins = np.linspace(data_min, data_max, 50)
            ax1.hist(emd_final, bins=bins, alpha=0.6, density=True, color='steelblue', edgecolor='darkblue', linewidth=1, label=f'EMD (n={len(emd_final)})')
            ax1.hist(newa_final, bins=bins, alpha=0.6, density=True, color='orange', edgecolor='darkorange', linewidth=1, label=f'NEWA (n={len(newa_final)})')
            ax1.set_xlabel('Ice Accretion [g/h]', fontweight='bold')
            ax1.set_ylabel('Probability Density', fontweight='bold')
            ax1.set_title('Probability Density Functions - Histograms', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = axes[0, 1]
            if len(emd_final) > 10:
                kde_emd = stats.gaussian_kde(emd_final)
                ax2.plot(x_range, kde_emd(x_range), 'steelblue', linewidth=3, label=f'EMD (μ={emd_stats["mean"]:.3f}, σ={emd_stats["std"]:.3f})')
            if len(newa_final) > 10:
                kde_newa = stats.gaussian_kde(newa_final)
                ax2.plot(x_range, kde_newa(x_range), 'orange', linewidth=3, label=f'NEWA (μ={newa_stats["mean"]:.3f}, σ={newa_stats["std"]:.3f})')
            ax2.axvline(emd_stats['mean'], color='steelblue', linestyle='--', alpha=0.8, label='EMD Mean')
            ax2.axvline(newa_stats['mean'], color='orange', linestyle='--', alpha=0.8, label='NEWA Mean')
            ax2.set_xlabel('Ice Accretion [g/h]', fontweight='bold')
            ax2.set_ylabel('Probability Density', fontweight='bold')
            ax2.set_title('Kernel Density Estimation (KDE) Comparison', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # New log-log PDF comparison plot
            ax3 = axes[0, 2]
            # Filter out zero values for log-log plot
            emd_nonzero = emd_final[emd_final > 0]
            newa_nonzero = newa_final[newa_final > 0]
            
            if len(emd_nonzero) > 10 and len(newa_nonzero) > 10:
                # Create log-spaced bins for PDF calculation
                log_min = max(1e-6, min(np.min(emd_nonzero), np.min(newa_nonzero)))
                log_max = max(np.max(emd_nonzero), np.max(newa_nonzero))
                bins = np.logspace(np.log10(log_min), np.log10(log_max), 50)
                
                # Calculate PDF (normalized histogram)
                emd_counts, _ = np.histogram(emd_nonzero, bins=bins)
                newa_counts, _ = np.histogram(newa_nonzero, bins=bins)
                
                # Normalize to get PDF
                bin_widths = np.diff(bins)
                emd_pdf = emd_counts / (len(emd_nonzero) * bin_widths)
                newa_pdf = newa_counts / (len(newa_nonzero) * bin_widths)
                
                # Plot centers
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Remove zeros for log-log plot
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
                        ha='center', va='center', transform=ax3.transAxes, fontsize=24)
            
            ax3.set_xlabel('Ice Accretion [g/h]', fontweight='bold')
            ax3.set_ylabel('Probability Density', fontweight='bold')
            ax3.set_title('PDF Comparison (Log-Log Scale)', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3, which="both")

            ax4 = axes[1, 0]
            n_quantiles = min(len(emd_final), len(newa_final), 1000)
            quantiles = np.linspace(0.01, 0.99, n_quantiles)
            emd_quantiles = np.quantile(emd_final, quantiles)
            newa_quantiles = np.quantile(newa_final, quantiles)
            ax4.scatter(emd_quantiles, newa_quantiles, alpha=0.6, s=20, color='purple')
            min_val = min(np.min(emd_quantiles), np.min(newa_quantiles))
            max_val = max(np.max(emd_quantiles), np.max(newa_quantiles))
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Agreement')
            qq_correlation = np.corrcoef(emd_quantiles, newa_quantiles)[0, 1]
            ax4.set_xlabel('EMD Quantiles [g/h]', fontweight='bold')
            ax4.set_ylabel('NEWA Quantiles [g/h]', fontweight='bold')
            ax4.set_title(f'Q-Q Plot (r = {qq_correlation:.3f})', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            ax5 = axes[1, 1]
            box_data = [emd_final, newa_final]
            labels = ['EMD', 'NEWA']
            colors = ['steelblue', 'orange']
            box_plot = ax5.boxplot(box_data, labels=labels, patch_artist=True, showmeans=True, meanline=True, boxprops=dict(alpha=0.7), medianprops=dict(color='red', linewidth=2), meanprops=dict(color='black', linewidth=2, linestyle='--'))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            ax5.set_ylabel('Ice Accretion [g/h]', fontweight='bold')
            ax5.set_title('Distribution Comparison (Box Plots)', fontweight='bold')
            ax5.grid(True, alpha=0.3)

            # Add statistical summary table to the last subplot (axes[1, 2])
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            # Create statistical summary table
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
                             cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(30)
            table.scale(1, 1.5)
            
            # Style the table
            for i in range(len(stats_data)):
                for j in range(len(stats_data[0])):
                    cell = table[(i, j)]
                    if i == 0:  # Header row
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    elif j == 1:  # EMD column
                        cell.set_facecolor('#E3F2FD')
                    elif j == 2:  # NEWA column
                        cell.set_facecolor('#FFF3E0')
                        
            ax6.set_title('Statistical Summary', fontweight='bold', pad=10)

            threshold_text = f"Ice Accretion Threshold: >={ice_accretion_threshold} g/h" if ice_accretion_threshold > 0 else "No Ice Accretion Threshold"
            nonzero_text = f"Non-Zero Filter: >={non_zero_percentage}% hours > 0" if non_zero_percentage > 0 else "No Non-Zero Filter"
            fig.suptitle(f'Probability Density Function Analysis: EMD vs NEWA at {height}m\nIce Accretion Distribution Comparison (Icing Season Only) - 6 Analysis Views\n{threshold_text} | {nonzero_text}\nNEWA Grid Cell: ({closest_sn}, {closest_we}) - Distance: {closest_distance_km:.2f} km', fontsize=28, fontweight='bold', y=0.95)
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            pdf_plot_path = os.path.join(base_dir, f'pdf_comparison_{height:.0f}m.png')
            plt.savefig(pdf_plot_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            print(f"Saved: {pdf_plot_path}")

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
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
            table.set_fontsize(22)
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
            print(f"Created 2 plots:")
            print(f"  1. PDF comparison plot (5 subplots): {pdf_plot_path}")
            print(f"     - Histograms")
            print(f"     - Linear KDE comparison") 
            print(f"     - Log-log KDE comparison")
            print(f"     - Q-Q plot")
            print(f"     - Box plots")
            print(f"  2. Statistical summary table: {stats_table_path}")

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


def climate_analysis(dataset, height_level=0, save_plots=True, results_subdir="climate_analysis"):
    """
    Comprehensive climate analysis generating multiple plots to understand the local climate characteristics.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        NEWA dataset containing meteorological variables (T, PRECIP, WS, WD)
    height_level : int, optional
        Height level index to use (0=50m, 1=100m, 2=150m) (default: 0)
    save_plots : bool, optional
        Whether to save plots to file (default: True)
    results_subdir : str, optional
        Subdirectory name within results/ for saving plots (default: "climate_analysis")
        
    Returns:
    --------
    dict
        Dictionary containing analysis results and statistics
    """
    
    print(f"=== COMPREHENSIVE CLIMATE ANALYSIS ===")
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Circle
        import os
        from datetime import datetime
        
        # Get height information
        height_m = dataset.height.values[height_level]
        print(f"Analysis height: {height_m}m")
        
        # Create output directory
        if save_plots:
            base_dir = os.path.join("results", "figures", "geographical_maps")
            os.makedirs(base_dir, exist_ok=True)
            print(f"Saving plots to: {base_dir}")
        
        # Extract meteorological variables at specified height
        print("Extracting meteorological variables...")
        temperature = dataset['T'].isel(height=height_level)  # Kelvin
        
        # Check if PRECIP has height dimension, if not use it directly
        if 'height' in dataset['PRECIP'].dims:
            precipitation = dataset['PRECIP'].isel(height=height_level)  # mm/30min
        else:
            precipitation = dataset['PRECIP']  # mm/30min (no height dimension)
            print("Note: PRECIP variable has no height dimension, using surface values")
        
        wind_speed = dataset['WS'].isel(height=height_level)  # m/s
        wind_direction = dataset['WD'].isel(height=height_level)  # degrees
        
        # Convert temperature to Celsius
        temp_celsius = temperature - 273.15
        
        # Convert precipitation to mm/h (from mm/30min)
        precip_hourly = precipitation * 2
        
        # Create time-based analysis
        time_data = pd.to_datetime(dataset.time.values)
        
        # Calculate domain averages for time series analysis
        print("Calculating domain averages...")
        temp_mean = temp_celsius.mean(dim=['south_north', 'west_east'])
        precip_mean = precip_hourly.mean(dim=['south_north', 'west_east'])
        ws_mean = wind_speed.mean(dim=['south_north', 'west_east'])
        
        # Convert to pandas for easier manipulation
        df = pd.DataFrame({
            'time': time_data,
            'temperature': temp_mean.values,
            'precipitation': precip_mean.values,
            'wind_speed': ws_mean.values
        })
        df.set_index('time', inplace=True)
        
        # Add seasonal information
        df['month'] = df.index.month
        df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                       3: 'Spring', 4: 'Spring', 5: 'Spring',
                                       6: 'Summer', 7: 'Summer', 8: 'Summer',
                                       9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
        df['year'] = df.index.year
        
        # ========================================
        # PLOT 1: SEASONAL TEMPERATURE ANALYSIS
        # ========================================
        print("Creating seasonal temperature analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Monthly temperature statistics
        monthly_stats = df.groupby('month')['temperature'].agg(['mean', 'min', 'max', 'std'])
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        ax1 = axes[0, 0]
        x = range(1, 13)
        ax1.plot(x, monthly_stats['mean'], 'ro-', linewidth=2, markersize=6, label='Mean')
        ax1.fill_between(x, monthly_stats['min'], monthly_stats['max'], alpha=0.3, color='lightblue', label='Min-Max Range')
        ax1.set_xlabel('Month', fontsize=24)
        ax1.set_ylabel('Temperature (°C)', fontsize=24)
        ax1.set_title('Monthly Temperature Statistics', fontsize=28)
        ax1.set_xticks(x)
        ax1.set_xticklabels(months, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Seasonal boxplot
        ax2 = axes[0, 1]
        seasonal_order = ['Winter', 'Spring', 'Summer', 'Autumn']
        df_season = df[df['season'].isin(seasonal_order)]
        sns.boxplot(data=df_season, x='season', y='temperature', order=seasonal_order, ax=ax2)
        ax2.set_title('Seasonal Temperature Distribution', fontsize=28)
        ax2.set_ylabel('Temperature (°C)', fontsize=24)
        ax2.grid(True, alpha=0.3)
        
        # Temperature time series (annual means)
        ax3 = axes[1, 0]
        annual_temp = df.groupby('year')['temperature'].mean()
        ax3.plot(annual_temp.index, annual_temp.values, 'b-', linewidth=2)
        z = np.polyfit(annual_temp.index, annual_temp.values, 1)
        p = np.poly1d(z)
        ax3.plot(annual_temp.index, p(annual_temp.index), 'r--', alpha=0.8, 
                label=f'Trend: {z[0]:.3f}°C/year')
        ax3.set_xlabel('Year', fontsize=24)
        ax3.set_ylabel('Annual Mean Temperature (°C)', fontsize=24)
        ax3.set_title('Annual Temperature Trend', fontsize=28)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Temperature histogram
        ax4 = axes[1, 1]
        ax4.hist(df['temperature'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(df['temperature'].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {df["temperature"].mean():.1f}°C')
        ax4.axvline(0, color='blue', linestyle=':', linewidth=2, label='Freezing Point')
        ax4.set_xlabel('Temperature (°C)', fontsize=24)
        ax4.set_ylabel('Frequency', fontsize=24)
        ax4.set_title('Temperature Distribution', fontsize=28)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Temperature Analysis at {height_m}m Height', fontsize=32, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            temp_path = os.path.join(base_dir, f'temperature_analysis_{height_m:.0f}m.png')
            plt.savefig(temp_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {temp_path}")
        
        # ========================================
        # PLOT 2: PRECIPITATION ANALYSIS
        # ========================================
        print("Creating precipitation analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Monthly precipitation statistics
        monthly_precip = df.groupby('month')['precipitation'].agg(['mean', 'sum', 'std'])
        
        ax1 = axes[0, 0]
        bars = ax1.bar(x, monthly_precip['mean'], alpha=0.7, color='steelblue')
        ax1.set_xlabel('Month', fontsize=24)
        ax1.set_ylabel('Mean Precipitation Rate (mm/h)', fontsize=24)
        ax1.set_title('Monthly Mean Precipitation Rate', fontsize=28)
        ax1.set_xticks(x)
        ax1.set_xticklabels(months, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, monthly_precip['mean']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=24)
        
        # Seasonal precipitation
        ax2 = axes[0, 1]
        seasonal_precip = df.groupby('season')['precipitation'].sum()
        seasonal_precip = seasonal_precip.reindex(seasonal_order)
        colors = ['lightblue', 'lightgreen', 'orange', 'brown']
        bars = ax2.bar(seasonal_order, seasonal_precip.values, color=colors, alpha=0.7)
        ax2.set_title('Total Seasonal Precipitation', fontsize=28)
        ax2.set_ylabel('Total Precipitation (mm)', fontsize=24)
        ax2.grid(True, alpha=0.3)
        
        # Precipitation intensity distribution
        ax3 = axes[1, 0]
        precip_nonzero = df[df['precipitation'] > 0]['precipitation']
        ax3.hist(precip_nonzero, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xlabel('Precipitation Rate (mm/h)', fontsize=24)
        ax3.set_ylabel('Frequency', fontsize=24)
        ax3.set_title('Precipitation Intensity Distribution (Wet Hours Only)', fontsize=28)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Precipitation frequency
        ax4 = axes[1, 1]
        precip_freq = (df['precipitation'] > 0).groupby(df['month']).mean() * 100
        bars = ax4.bar(x, precip_freq.values, alpha=0.7, color='darkgreen')
        ax4.set_xlabel('Month', fontsize=24)
        ax4.set_ylabel('Precipitation Frequency (%)', fontsize=24)
        ax4.set_title('Monthly Precipitation Frequency', fontsize=28)
        ax4.set_xticks(x)
        ax4.set_xticklabels(months, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Precipitation Analysis at {height_m}m Height', fontsize=32, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            precip_path = os.path.join(base_dir, f'precipitation_analysis_{height_m:.0f}m.png')
            plt.savefig(precip_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {precip_path}")
        
        # ========================================
        # PLOT 3: WIND SPEED ANALYSIS
        # ========================================
        print("Creating wind speed analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Monthly wind speed statistics
        monthly_ws = df.groupby('month')['wind_speed'].agg(['mean', 'std', 'max'])
        
        ax1 = axes[0, 0]
        ax1.plot(x, monthly_ws['mean'], 'bo-', linewidth=2, markersize=6, label='Mean')
        ax1.fill_between(x, monthly_ws['mean'] - monthly_ws['std'], 
                        monthly_ws['mean'] + monthly_ws['std'], 
                        alpha=0.3, color='lightblue', label='±1 Std Dev')
        ax1.set_xlabel('Month', fontsize=24)
        ax1.set_ylabel('Wind Speed (m/s)', fontsize=24)
        ax1.set_title('Monthly Wind Speed Statistics', fontsize=28)
        ax1.set_xticks(x)
        ax1.set_xticklabels(months, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Seasonal wind speed
        ax2 = axes[0, 1]
        sns.boxplot(data=df_season, x='season', y='wind_speed', order=seasonal_order, ax=ax2)
        ax2.set_title('Seasonal Wind Speed Distribution', fontsize=28)
        ax2.set_ylabel('Wind Speed (m/s)', fontsize=24)
        ax2.grid(True, alpha=0.3)
        
        # Wind speed distribution
        ax3 = axes[1, 0]
        ax3.hist(df['wind_speed'], bins=50, alpha=0.7, color='orange', edgecolor='black', density=True)
        ax3.axvline(df['wind_speed'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {df["wind_speed"].mean():.1f} m/s')
        ax3.set_xlabel('Wind Speed (m/s)', fontsize=24)
        ax3.set_ylabel('Probability Density', fontsize=24)
        ax3.set_title('Wind Speed Distribution', fontsize=28)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Wind speed time series (annual means)
        ax4 = axes[1, 1]
        annual_ws = df.groupby('year')['wind_speed'].mean()
        ax4.plot(annual_ws.index, annual_ws.values, 'g-', linewidth=2)
        z_ws = np.polyfit(annual_ws.index, annual_ws.values, 1)
        p_ws = np.poly1d(z_ws)
        ax4.plot(annual_ws.index, p_ws(annual_ws.index), 'r--', alpha=0.8,
                label=f'Trend: {z_ws[0]:.3f} m/s/year')
        ax4.set_xlabel('Year', fontsize=24)
        ax4.set_ylabel('Annual Mean Wind Speed (m/s)', fontsize=24)
        ax4.set_title('Annual Wind Speed Trend', fontsize=28)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Wind Speed Analysis at {height_m}m Height', fontsize=32, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            ws_path = os.path.join(base_dir, f'wind_speed_analysis_{height_m:.0f}m.png')
            plt.savefig(ws_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {ws_path}")
        
        # ========================================
        # PLOT 4: WIND RESOURCE ANALYSIS
        # ========================================
        print("Creating wind resource analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Wind speed frequency distribution (for wind resource)
        ax1 = axes[0, 0]
        wind_bins = np.arange(0, df['wind_speed'].max() + 1, 1)
        hist, bins = np.histogram(df['wind_speed'], bins=wind_bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax1.bar(bin_centers, hist * 100, alpha=0.7, color='skyblue', edgecolor='black', width=0.8)
        ax1.set_xlabel('Wind Speed (m/s)', fontsize=24)
        ax1.set_ylabel('Frequency (%)', fontsize=24)
        ax1.set_title('Wind Speed Frequency Distribution', fontsize=28)
        ax1.grid(True, alpha=0.3)
        
        # Add wind power classes
        ax1.axvspan(0, 3, alpha=0.2, color='red', label='Low (<3 m/s)')
        ax1.axvspan(3, 6, alpha=0.2, color='yellow', label='Moderate (3-6 m/s)')
        ax1.axvspan(6, 10, alpha=0.2, color='green', label='Good (6-10 m/s)')
        ax1.axvspan(10, 25, alpha=0.2, color='blue', label='Excellent (>10 m/s)')
        ax1.legend(loc='upper right')
        
        # Wind power density (simplified)
        ax2 = axes[0, 1]
        # Assuming air density ≈ 1.225 kg/m³
        air_density = 1.225
        power_density = 0.5 * air_density * df['wind_speed']**3  # W/m²
        monthly_power = power_density.groupby(df['month']).mean()
        
        bars = ax2.bar(x, monthly_power.values, alpha=0.7, color='purple')
        ax2.set_xlabel('Month', fontsize=24)
        ax2.set_ylabel('Wind Power Density (W/m²)', fontsize=24)
        ax2.set_title('Monthly Mean Wind Power Density', fontsize=28)
        ax2.set_xticks(x)
        ax2.set_xticklabels(months, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Weibull distribution fit
        ax3 = axes[1, 0]
        from scipy import stats as scipy_stats
        
        # Fit Weibull distribution
        wind_data_clean = df['wind_speed'].dropna()
        weibull_params = scipy_stats.weibull_min.fit(wind_data_clean, floc=0)
        shape, loc, scale = weibull_params
        
        # Plot histogram and fitted distribution
        ax3.hist(wind_data_clean, bins=50, alpha=0.7, density=True, color='lightgreen', 
                edgecolor='black', label='Observed')
        
        x_weibull = np.linspace(0, wind_data_clean.max(), 1000)
        pdf_weibull = scipy_stats.weibull_min.pdf(x_weibull, shape, loc, scale)
        ax3.plot(x_weibull, pdf_weibull, 'r-', linewidth=3, 
                label=f'Weibull (k={shape:.2f}, λ={scale:.2f})')
        
        ax3.set_xlabel('Wind Speed (m/s)', fontsize=24)
        ax3.set_ylabel('Probability Density', fontsize=24)
        ax3.set_title('Wind Speed Distribution and Weibull Fit', fontsize=28)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Calm conditions analysis
        ax4 = axes[1, 1]
        calm_threshold = 3.0  # m/s
        calm_freq = (df['wind_speed'] < calm_threshold).groupby(df['month']).mean() * 100
        
        bars = ax4.bar(x, calm_freq.values, alpha=0.7, color='lightcoral')
        ax4.set_xlabel('Month', fontsize=24)
        ax4.set_ylabel(f'Calm Conditions Frequency (< {calm_threshold} m/s, %)')
        ax4.set_title('Monthly Calm Conditions Frequency', fontsize=28)
        ax4.set_xticks(x)
        ax4.set_xticklabels(months, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Wind Resource Analysis at {height_m}m Height', fontsize=32, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            resource_path = os.path.join(base_dir, f'wind_resource_analysis_{height_m:.0f}m.png')
            plt.savefig(resource_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {resource_path}")
        
        # ========================================
        # CALCULATE SUMMARY STATISTICS
        # ========================================
        print("Calculating summary statistics...")
        
        results = {
            'height_m': float(height_m),
            'analysis_period': {
                'start': str(df.index.min()),
                'end': str(df.index.max()),
                'total_years': df.index.year.nunique()
            },
            'temperature_stats': {
                'annual_mean_C': float(df['temperature'].mean()),
                'annual_std_C': float(df['temperature'].std()),
                'annual_min_C': float(df['temperature'].min()),
                'annual_max_C': float(df['temperature'].max()),
                'freezing_hours_percent': float((df['temperature'] < 0).mean() * 100),
                'seasonal_means': {
                    season: float(df[df['season'] == season]['temperature'].mean())
                    for season in seasonal_order
                },
                'trend_C_per_year': float(z[0])
            },
            'precipitation_stats': {
                'annual_mean_mm_h': float(df['precipitation'].mean()),
                'annual_total_mm': float(df['precipitation'].sum() * 0.5),  # Convert to mm (30min data)
                'wet_hours_percent': float((df['precipitation'] > 0).mean() * 100),
                'max_intensity_mm_h': float(df['precipitation'].max()),
                'seasonal_totals': {
                    season: float(df[df['season'] == season]['precipitation'].sum() * 0.5)
                    for season in seasonal_order
                }
            },
            'wind_speed_stats': {
                'annual_mean_ms': float(df['wind_speed'].mean()),
                'annual_std_ms': float(df['wind_speed'].std()),
                'annual_max_ms': float(df['wind_speed'].max()),
                'calm_hours_percent': float((df['wind_speed'] < 3).mean() * 100),
                'good_wind_percent': float((df['wind_speed'] >= 6).mean() * 100),
                'seasonal_means': {
                    season: float(df[df['season'] == season]['wind_speed'].mean())
                    for season in seasonal_order
                },
                'trend_ms_per_year': float(z_ws[0]),
                'weibull_parameters': {
                    'shape_k': float(shape),
                    'scale_lambda': float(scale)
                }
            },
            'wind_resource': {
                'annual_power_density_W_m2': float(power_density.mean()),
                'seasonal_power_density': {
                    season: float(power_density[df['season'] == season].mean())
                    for season in seasonal_order
                }
            }
        }
        
        print(f"\n=== CLIMATE ANALYSIS SUMMARY ===")
        print(f"Analysis period: {results['analysis_period']['total_years']} years")
        print(f"Temperature: {results['temperature_stats']['annual_mean_C']:.1f}°C (mean)")
        print(f"Precipitation: {results['precipitation_stats']['annual_total_mm']:.0f} mm/year")
        print(f"Wind speed: {results['wind_speed_stats']['annual_mean_ms']:.1f} m/s (mean)")
        print(f"Wind power density: {results['wind_resource']['annual_power_density_W_m2']:.0f} W/m²")
        
        if save_plots:
            print(f"\n=== PLOTS SAVED ===")
            print(f"1. Temperature analysis: temperature_analysis_{height_m:.0f}m.png")
            print(f"2. Precipitation analysis: precipitation_analysis_{height_m:.0f}m.png")
            print(f"3. Wind speed analysis: wind_speed_analysis_{height_m:.0f}m.png")
            print(f"4. Wind resource analysis: wind_resource_analysis_{height_m:.0f}m.png")
            print(f"5. Wind rose: wind_rose_{height_m:.0f}m.png")
        
        return results
        
    except Exception as e:
        print(f"Error in climate analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_temperature_emd_newa(emd_data, newa_data, height, emd_coordinates=None, save_plots=True):
    """
    Compare temperature values between EMD and NEWA datasets at specific height
    
    Parameters:
    -----------
    emd_data : pandas.DataFrame
        EMD dataset with temperature data in columns like 'temp.x' where x is height in meters
    newa_data : xarray.Dataset  
        NEWA dataset with temperature in 'T' variable at specified height
    height : int
        Height index (0-based) for NEWA data and corresponding height in meters for EMD
        (0=50m; 1=100m; 2=150m)
    emd_coordinates : tuple, optional
        (longitude, latitude) coordinates for extracting NEWA data at EMD location
        Default: (19.960, 59.600)
    save_plots : bool
        Whether to save resulting plots. Default: True
    
    Returns:
    --------
    dict
        Dictionary containing comparison results and statistics
    """
    try:
        # Set default coordinates if not provided
        if emd_coordinates is None:
            emd_coordinates = (19.960, 59.600)
        
        # Get actual height in meters for the height index
        height_meters = newa_data.height.values[height]
        
        print(f"=== TEMPERATURE COMPARISON: EMD vs NEWA ===")
        print(f"Height level: {height} (index) = {height_meters} m")
        print(f"EMD coordinates: {emd_coordinates}")
        
        # Create output directory
        output_dir = f"results/figures/EMD/Temperature/{height_meters:.0f}m"
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract EMD temperature data (convert from Celsius to Kelvin)
        emd_temp_col = f'temp.{height_meters:.0f}'
        if emd_temp_col not in emd_data.columns:
            print(f"Warning: Column {emd_temp_col} not found in EMD data")
            print(f"Available temperature columns: {[col for col in emd_data.columns if 'temp' in col]}")
            # Try to find closest height
            temp_cols = [col for col in emd_data.columns if 'temp' in col]
            if temp_cols:
                emd_temp_col = temp_cols[0]  # Use first available
                print(f"Using {emd_temp_col} instead")
            else:
                raise ValueError("No temperature columns found in EMD data")
        
        emd_temperature = emd_data[emd_temp_col].copy()
        emd_temperature_kelvin = emd_temperature + 273.15  # Convert Celsius to Kelvin
        
        # Extract NEWA temperature data at EMD location (already in Kelvin)
        lon, lat = emd_coordinates
        newa_temp_at_location = newa_data['T'].sel(
            west_east=lon, 
            south_north=lat, 
            height=height_meters, 
            method='nearest'
        )
        
        # Convert NEWA 30-min timestep to 1-hour by taking hourly means
        print("Converting NEWA timestep from 30min to 1hour...")
        newa_temp_hourly = newa_temp_at_location.resample(time='1H').mean()
        
        # Align time series for comparison
        print("Aligning time series...")
        # Convert to pandas for easier manipulation
        newa_temp_pd = newa_temp_hourly.to_pandas()
        
        # Find common time period
        common_start = max(emd_temperature_kelvin.index.min(), newa_temp_pd.index.min())
        common_end = min(emd_temperature_kelvin.index.max(), newa_temp_pd.index.max())
        
        print(f"Common time period: {common_start} to {common_end}")
        
        # Filter both series to common period
        emd_common = emd_temperature_kelvin[(emd_temperature_kelvin.index >= common_start) & 
                                           (emd_temperature_kelvin.index <= common_end)]
        newa_common = newa_temp_pd[(newa_temp_pd.index >= common_start) & 
                                  (newa_temp_pd.index <= common_end)]
        
        # Align exactly by reindexing
        common_index = emd_common.index.intersection(newa_common.index)
        emd_aligned = emd_common.reindex(common_index)
        newa_aligned = newa_common.reindex(common_index)
        
        # Remove NaN values
        valid_mask = ~(emd_aligned.isna() | newa_aligned.isna())
        emd_final = emd_aligned[valid_mask]
        newa_final = newa_aligned[valid_mask]
        
        print(f"Final comparison dataset size: {len(emd_final)} common time points")
        
        # Calculate differences and statistics
        differences = emd_final - newa_final
        
        # Statistics
        stats = {
            'mean_bias': differences.mean(),
            'rmse': np.sqrt((differences**2).mean()),
            'mae': np.abs(differences).mean(),
            'correlation': emd_final.corr(newa_final),
            'emd_mean': emd_final.mean(),
            'newa_mean': newa_final.mean(),
            'emd_std': emd_final.std(),
            'newa_std': newa_final.std(),
            'n_points': len(emd_final)
        }
        
        print(f"\n=== TEMPERATURE COMPARISON STATISTICS ===")
        print(f"Number of points: {stats['n_points']}")
        print(f"Mean bias (EMD - NEWA): {stats['mean_bias']:.3f} K")
        print(f"RMSE: {stats['rmse']:.3f} K")
        print(f"MAE: {stats['mae']:.3f} K")
        print(f"Correlation: {stats['correlation']:.3f}")
        print(f"EMD mean: {stats['emd_mean']:.1f} K")
        print(f"NEWA mean: {stats['newa_mean']:.1f} K")
        
        if save_plots:
            # Create comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Scatter plot of all values
            axes[0,0].scatter(newa_final, emd_final, alpha=0.5, s=1)
            axes[0,0].plot([min(newa_final.min(), emd_final.min()), 
                           max(newa_final.max(), emd_final.max())], 
                          [min(newa_final.min(), emd_final.min()), 
                           max(newa_final.max(), emd_final.max())], 'r--', alpha=0.8)
            axes[0,0].set_xlabel('NEWA Temperature (K)', fontsize=24)
            axes[0,0].set_ylabel('EMD Temperature (K)', fontsize=24)
            axes[0,0].set_title(f'Temperature Scatter Plot\nCorr = {stats["correlation"]:.3f}, N = {stats["n_points"]}')
            axes[0,0].grid(True, alpha=0.3)
            
            # 2. Difference scatter plot  
            axes[0,1].scatter(newa_final, differences, alpha=0.5, s=1)
            axes[0,1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
            axes[0,1].axhline(y=differences.mean(), color='g', linestyle='-', alpha=0.8, 
                             label=f'Mean bias = {differences.mean():.3f} K')
            axes[0,1].set_xlabel('NEWA Temperature (K)', fontsize=24)
            axes[0,1].set_ylabel('Difference (EMD - NEWA) (K)', fontsize=24)
            axes[0,1].set_title(f'Temperature Differences vs NEWA\nRMSE = {stats["rmse"]:.3f} K')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # 3. Time series of differences (sample)
            sample_size = min(1000, len(differences))
            sample_indices = np.random.choice(len(differences), sample_size, replace=False)
            sample_times = differences.iloc[sample_indices].sort_index()
            
            axes[1,0].plot(sample_times.index, sample_times.values, alpha=0.7)
            axes[1,0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
            axes[1,0].axhline(y=differences.mean(), color='g', linestyle='-', alpha=0.8)
            axes[1,0].set_xlabel('Time', fontsize=24)
            axes[1,0].set_ylabel('Difference (EMD - NEWA) (K)', fontsize=24)
            axes[1,0].set_title(f'Time Series of Differences (Random Sample: {sample_size} points)')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # 4. Histogram of differences
            axes[1,1].hist(differences, bins=50, alpha=0.7, density=True)
            axes[1,1].axvline(x=0, color='r', linestyle='--', alpha=0.8, label='Zero bias')
            axes[1,1].axvline(x=differences.mean(), color='g', linestyle='-', alpha=0.8, 
                             label=f'Mean bias = {differences.mean():.3f} K')
            axes[1,1].set_xlabel('Difference (EMD - NEWA) (K)', fontsize=24)
            axes[1,1].set_ylabel('Probability Density', fontsize=24)
            axes[1,1].set_title(f'Distribution of Differences\nStd = {differences.std():.3f} K')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{output_dir}/temperature_comparison_{height_meters:.0f}m.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"\nTemperature comparison plot saved: {plot_filename}")
            plt.close()
            
            # Create bias analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Monthly bias pattern
            monthly_bias = differences.groupby(differences.index.month).mean()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            axes[0,0].bar(range(1, 13), monthly_bias.values)
            axes[0,0].set_xticks(range(1, 13))
            axes[0,0].set_xticklabels(month_names)
            axes[0,0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
            axes[0,0].set_ylabel('Mean Bias (EMD - NEWA) (K)', fontsize=24)
            axes[0,0].set_title('Monthly Temperature Bias Pattern', fontsize=28)
            axes[0,0].grid(True, alpha=0.3)
            
            # 2. Hourly bias pattern (if enough data)
            if len(differences) > 24:
                hourly_bias = differences.groupby(differences.index.hour).mean()
                axes[0,1].plot(hourly_bias.index, hourly_bias.values, 'o-')
                axes[0,1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
                axes[0,1].set_xlabel('Hour of Day', fontsize=24)
                axes[0,1].set_ylabel('Mean Bias (EMD - NEWA) (K)', fontsize=24)
                axes[0,1].set_title('Hourly Temperature Bias Pattern', fontsize=28)
                axes[0,1].grid(True, alpha=0.3)
                axes[0,1].set_xticks(range(0, 24, 3))
            
            # 3. Bias vs temperature range
            temp_bins = np.percentile(newa_final, np.linspace(0, 100, 11))
            bin_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
            binned_bias = []
            
            for i in range(len(temp_bins)-1):
                mask = (newa_final >= temp_bins[i]) & (newa_final < temp_bins[i+1])
                if mask.sum() > 0:
                    binned_bias.append(differences[mask].mean())
                else:
                    binned_bias.append(np.nan)
            
            axes[1,0].plot(bin_centers, binned_bias, 'o-')
            axes[1,0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
            axes[1,0].set_xlabel('NEWA Temperature (K)', fontsize=24)
            axes[1,0].set_ylabel('Mean Bias (EMD - NEWA) (K)', fontsize=24)
            axes[1,0].set_title('Temperature Bias vs Temperature Range', fontsize=28)
            axes[1,0].grid(True, alpha=0.3)
            
            # 4. Annual bias trend (if multiple years available)
            if len(differences) > 365*24:
                annual_bias = differences.groupby(differences.index.year).mean()
                if len(annual_bias) > 1:
                    axes[1,1].plot(annual_bias.index, annual_bias.values, 'o-')
                    axes[1,1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
                    axes[1,1].set_xlabel('Year', fontsize=24)
                    axes[1,1].set_ylabel('Mean Bias (EMD - NEWA) (K)', fontsize=24)
                    axes[1,1].set_title('Annual Temperature Bias Trend', fontsize=28)
                    axes[1,1].grid(True, alpha=0.3)
                else:
                    axes[1,1].text(0.5, 0.5, 'Insufficient data\nfor annual trend', 
                                  ha='center', va='center', transform=axes[1,1].transAxes)
            else:
                axes[1,1].text(0.5, 0.5, 'Insufficient data\nfor annual analysis', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
            
            plt.tight_layout()
            
            # Save bias analysis plot
            bias_filename = f"{output_dir}/temperature_bias_analysis_{height_meters:.0f}m.png"
            plt.savefig(bias_filename, dpi=300, bbox_inches='tight')
            print(f"Temperature bias analysis plot saved: {bias_filename}")
            plt.close()
        
        # Prepare results
        results = {
            'statistics': stats,
            'emd_data': emd_final,
            'newa_data': newa_final,
            'differences': differences,
            'height_meters': height_meters,
            'output_directory': output_dir
        }
        
        return results
        
    except Exception as e:
        print(f"Error in temperature comparison: {e}")
        import traceback
        traceback.print_exc()
        return None


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r


def find_nearest_point(dataset, target_lat, target_lon, verbose=True):
    """
    Find the nearest point in the dataset to the given coordinates and return a dataset with only that point
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Dataset containing XLAT and XLON coordinates
    target_lat : float
        Target latitude in decimal degrees
    target_lon : float  
        Target longitude in decimal degrees
    verbose : bool, optional
        If True, print information about the search results
        
    Returns:
    --------
    xarray.Dataset
        Dataset with same structure as input but containing only the nearest point data
    """
    
    try:
        # Get geographical coordinates from dataset
        if 'XLAT' in dataset.coords and 'XLON' in dataset.coords:
            lats = dataset.XLAT.values
            lons = dataset.XLON.values
        elif 'XLAT' in dataset.data_vars and 'XLON' in dataset.data_vars:
            lats = dataset.XLAT.values
            lons = dataset.XLON.values
        else:
            raise ValueError("Could not find XLAT and XLON coordinates in dataset")
        
        if verbose:
            print(f"Searching for nearest point to coordinates: ({target_lat:.4f}, {target_lon:.4f})")
            print(f"Dataset grid size: {lats.shape}")
        
        # Calculate distances to all points
        distances = haversine_distance(target_lat, target_lon, lats, lons)
        
        # Find the indices of the minimum distance
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        south_north_idx, west_east_idx = min_idx
        
        # Get the coordinates and distance of the nearest point
        nearest_lat = lats[south_north_idx, west_east_idx]
        nearest_lon = lons[south_north_idx, west_east_idx]
        min_distance = distances[south_north_idx, west_east_idx]
        
        if verbose:
            print(f"Nearest point found at grid indices: ({south_north_idx}, {west_east_idx})")
            print(f"Nearest point coordinates: ({nearest_lat:.4f}, {nearest_lon:.4f})")
            print(f"Distance to nearest point: {min_distance:.2f} km")
        
        # Extract the nearest point data from the dataset
        # Select the specific grid point across all time steps and other dimensions
        nearest_dataset = dataset.isel(south_north=south_north_idx, west_east=west_east_idx)
        
        if verbose:
            print(f"Extracted dataset dimensions: {dict(nearest_dataset.dims)}")
            print(f"Available variables: {list(nearest_dataset.data_vars)}")
        
        return nearest_dataset
        
    except Exception as e:
        print(f"Error finding nearest point: {e}")
        import traceback
        traceback.print_exc()
        return None

