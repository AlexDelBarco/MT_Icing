# IMPORTS
import os
import functions as fn
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# IMPORT DATA
data1 = "data/alexandre.nc"
dataset = fn.load_netcdf_data(data1)

print("=== LANDMASK VARIABLE EXPLORATION ===")

# 1. Print basic information about LANDMASK
print("\n1. LANDMASK Variable Info:")
print(f"   Shape: {dataset.LANDMASK.shape}")
print(f"   Data type: {dataset.LANDMASK.dtype}")
print(f"   Dimensions: {dataset.LANDMASK.dims}")

# 2. Print the values
print("\n2. LANDMASK Values:")
print("   (Note: LANDMASK typically has values 0=water, 1=land)")
print(dataset.LANDMASK.values)

# 3. Print statistics
print("\n3. LANDMASK Statistics:")
print(f"   Minimum value: {dataset.LANDMASK.min().values}")
print(f"   Maximum value: {dataset.LANDMASK.max().values}")
print(f"   Unique values: {np.unique(dataset.LANDMASK.values)}")
print(f"   Mean value: {dataset.LANDMASK.mean().values:.3f}")

# 4. Count land vs water points
land_points = np.sum(dataset.LANDMASK.values == 1)
water_points = np.sum(dataset.LANDMASK.values == 0)
total_points = dataset.LANDMASK.size

print(f"\n4. Land/Water Distribution:")
print(f"   Land points: {land_points} ({land_points/total_points*100:.1f}%)")
print(f"   Water points: {water_points} ({water_points/total_points*100:.1f}%)")
print(f"   Total points: {total_points}")

# 5. Print coordinates for context
print(f"\n5. Spatial Context:")
print(f"   Latitude range: {dataset.XLAT.min().values:.3f} to {dataset.XLAT.max().values:.3f}")
print(f"   Longitude range: {dataset.XLON.min().values:.3f} to {dataset.XLON.max().values:.3f}")

# 6. Create a simple visualization
print("\n6. Creating LANDMASK visualization...")
plt.figure(figsize=(10, 8))
plt.imshow(dataset.LANDMASK.values, cmap='RdYlBu_r', origin='lower')
plt.colorbar(label='LANDMASK (0=Water, 1=Land)')
plt.title('LANDMASK - Åland Islands Domain')
plt.xlabel('West-East Grid Points')
plt.ylabel('South-North Grid Points')

# Add grid point numbers
for i in range(dataset.LANDMASK.shape[0]):
    for j in range(dataset.LANDMASK.shape[1]):
        value = dataset.LANDMASK.values[i, j]
        plt.text(j, i, f'{value:.0f}', ha='center', va='center', 
                color='white' if value < 0.5 else 'black', fontsize=8)

# Create results directory if it doesn't exist
os.makedirs('results/figures', exist_ok=True)
plt.tight_layout()
plt.savefig('results/figures/landmask_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   LANDMASK visualization saved to: results/figures/landmask_visualization.png")