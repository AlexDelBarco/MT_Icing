ds_c = xr.open_zarr(r'D:\PROJECTWIND\Alps_data\zarr_final_one_height_Alps.zarr').sel(time=slice(startdate,enddate))
ds = xr.open_zarr(r'D:\PROJECTWIND\Alps_data\zarr_iceload_final_one_height_Alps_max_5.zarr').sel(time=slice(startdate,enddate))

#adding data from the previous dataset to avoid errors and complete
ds['XLAT'] = ds_c.XLAT
ds['XLON'] = ds_c.XLON
ds['south_north'] = ds_c.south_north
ds['west_east'] = ds_c.west_east
ds['HGT']= ds_c.HGT
ds['T']= ds_c.T
ds['RH']= ds_c.RH
ds['height']=ds_c.height
ds['ABLAT_CYL']=ds_c.ABLAT_CYL

#convert to winter

dates = pd.date_range(startdate,'2019-06-30T00:00:00',freq='AS-JUL')

df = ds['time'].to_pandas()
for iwinter,winterstartdate in enumerate(dates[:-1]):
    winterenddate = dates[iwinter+1]-pd.to_timedelta('1h')
    print(iwinter,winterstartdate,winterenddate)
    datesperwinter = pd.date_range(winterstartdate,winterenddate,freq='1h')
    df.loc[datesperwinter]=iwinter
ds = ds.assign_coords(winterno=('time',df.values))

#specify the timeframe for calculations
start = pd.to_datetime('1989-07-01T00:00:00')
end = pd.to_datetime('2018-06-30T23:00:00')

# include a mask height

#new dataset with those filters
ds2=ds.sel(time=slice(start,end))

#mean cumulative hours iceload, ice load>1kg/h
dssum = xr.where(ds2['ice_load']>0.1,1,0).groupby('winterno').sum(dim='time')
dssum_mean=dssum.mean(dim='winterno')

stacked_elevation=ds2['HGT'].stack(desired=['south_north', 'west_east'])

#FLATTEN XRRAY Cumulative mean hours/year
stacked_iceblade=dssum_mean.stack(desired=['south_north', 'west_east', 'height']).load()

x_space= np.logspace(np.log10(1.0), np.log10(8760.0), 60)
fig, ax = plt.subplots(figsize=(10, 5))
hist=plt.hist2d(stacked_iceblade, stacked_elevation, bins= (x_space, 60), norm=LogNorm(), cmap='viridis')

cbar=plt.colorbar(hist[3], ax=ax, label='Counts')
cbar.ax.set_ylabel('Counts', fontsize=16)
cbar.ax.tick_params(labelsize=16)

plt.grid()
plt.ylabel('Terrain height [m]', fontsize=16)
plt.xscale('log')
ax.set_xlabel(' ice load > 0.1kg/m (iceblade) [hours/year]', fontsize=16)
ax.tick_params(axis='both', labelsize=16)