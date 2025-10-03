Spatial gradients:
- Plot 1 (Top Left): Mean Ice Load Over Time
What it shows: The average ice load at each grid point across all time steps
Calculation: ice_load_clean.mean(dim='time')
Units: kg/m
Interpretation: Shows typical ice load patterns averaged over time
Use: Reference map for understanding average conditions

- Plot 2 (Top Right): Mean Spatial Gradient (West-East)
What it shows: Average west-east spatial gradient computed over all time steps
Calculation: Gradients calculated for each time step, then averaged
Units: kg/m per grid cell
Interpretation: Persistent east-west spatial patterns in ice load
Use: Identifies consistent east-west ice load transitions

- Plot 3 (Bottom Left): Mean Spatial Gradient (South-North)
What it shows: Average south-north spatial gradient computed over all time steps
Calculation: Gradients calculated for each time step, then averaged
Units: kg/m per grid cell
Interpretation: Persistent north-south spatial patterns in ice load
Use: Identifies consistent north-south ice load transitions

- Plot 4 (Bottom Right): Mean Gradient Magnitude
What it shows: Average overall gradient strength across all time steps
Calculation: √(mean_grad_x² + mean_grad_y²)
Units: Mean gradient magnitude
Interpretation: Regions with consistently strong spatial ice load contrasts
Use: Identifies persistently heterogeneous areas


Temporal gradient:
- Plot 1 (Top Left): Mean Temporal Gradient (Rate of Change)
What it shows: The average rate of ice load change across all grid points at each time step
Calculation: For each time step, calculates the mean of temporal gradients from all spatial locations
Units: kg/m per 30 minutes
Interpretation:
Positive values: Ice is accumulating on average across the domain
Negative values: Ice is melting/ablating on average across the domain
Zero line (red dashed): No net change in ice load
Use: Shows overall temporal trends of ice accumulation/ablation across your study area

- Plot 2 (Top Right): Temporal Gradient Variability
What it shows: The standard deviation of temporal gradients across all grid points at each time step
Calculation: For each time step, calculates how much the rate of change varies spatially
Units: kg/m per 30 minutes (standard deviation)
Interpretation:
High values: Large spatial variation - some areas gaining ice rapidly while others lose ice
Low values: Uniform behavior across the domain - all areas changing similarly
Use: Identifies periods when ice processes are spatially heterogeneous vs. homogeneous

- Plot 3 (Bottom Left): Extreme Temporal Gradients
What it shows: The maximum and minimum rates of change occurring anywhere in the domain at each time step
Calculation: For each time step, finds the highest and lowest temporal gradient values across all grid points
Units: kg/m per 30 minutes
Interpretation:
Red line (Maximum): Fastest ice accumulation rate occurring anywhere
Blue line (Minimum): Fastest ice ablation rate occurring anywhere
Gap between lines: Shows the range of temporal behavior across space
Use: Identifies extreme ice processes and their timing

- Plot 4 (Bottom Right): Distribution of Temporal Gradients
What it shows: Histogram of ALL temporal gradient values from all grid points and all time steps
Calculation: Flattens the entire temporal gradient array and plots frequency distribution
Units: Frequency vs. Rate of Change (kg/m per 30 minutes)
Interpretation:
Peak around zero: Most changes are small (ice load relatively stable)
Positive tail: Frequency of different accumulation rates
Negative tail: Frequency of different ablation rates
Red vertical line: Zero change reference
Use: Shows the overall statistical distribution of ice load change rates