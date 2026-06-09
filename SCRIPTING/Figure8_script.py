#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 18:38:29 2025

@author: bradford
"""



import os
import numpy as np
import pandas as pd
import xarray as xr
import pygmt
from math import radians, sin, cos, sqrt, atan2
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import geopandas as gpd
import shapely
import rioxarray




active_dir = './'

region = 'TANGO_North'



# Geographic and depth limits
gmt_region = [-70.9, -63.44, -25, -20, -120, 12]
w, e, s, n, base, top = gmt_region



def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = sin(d_lat/2)**2 + cos(lat1) * cos(lat2) * sin(d_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = 6371 * c  # Earth radius in kilometers (approx.)
    return distance

def Return_Line_Elements(joints):


    points_per_km = 10

    # distance between joint-points
    d = np.array([0])
    num_points = np.array([])
    for i in range(len(joints)-1):
        d = np.append(d, haversine(joints[i,1], joints[i,0],joints[i+1,1], joints[i+1,0]))
        num_points = np.append(num_points, np.round(d[-1] * points_per_km))

    joints_dist = np.cumsum(d)


    # lat-lon coords of line
    line = np.array([0,0])
    for i in range(len(joints)-1):
        l = np.linspace(joints[i], joints[i+1], int(num_points[i]))
        line = np.vstack((line, l))

    line = line[1:len(line)]


    # cummulative distance along line
    d = np.array([0])
    for i in range(len(line)-1):
        d = np.append(d, haversine(line[i,1], line[i,0],line[i+1,1], line[i+1,0]))

    line_dist = np.cumsum(d)

    return joints_dist, line, line_dist

def rasterize(file):
    shape = rioxarray.open_rasterio(file)
    shape = shape.sel(band = 1).drop('band')
    shape.values[shape.values == 0] = np.nan

    
    shape_grid = shape.interp(x = grid.lon, y = grid.lat, method = 'nearest', kwargs={"fill_value": None})
    shape_grid.values = shape_grid.values * grid.values

    shape_grid.values = np.clip(shape_grid.values, np.nanmin(shape_grid.values), np.nanmax(shape_grid.values))


    return shape_grid



joints =  np.loadtxt('../DATA/MAPPING/{}_CCP_Line.txt'.format(region), delimiter = ',')
joints_dist, line, line_dist = Return_Line_Elements(joints)

x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')



dist_x = haversine(s,w,s,e)
figx = 20
ratio = (figx / dist_x)

dist_y = haversine(s,w,n,w)
figy = ratio * dist_y

aspect = 1
figz = ratio * aspect * np.abs(base)







#%% Plot full grid data

grid = pygmt.datasets.load_earth_relief(resolution="15s", region=[w,e,s,n])
grid.values = (grid.values / 1000) * 4
# grid.values[grid.values < 0] = np.nan
Lon, Lat = np.meshgrid(grid.lon.values, grid.lat.values)


for i in range(len(joints)-1):
    lons = np.unique(Lon[(Lon > joints[i,0]) & (Lon < joints[i+1,0])])
    lon_interp_func = interp1d([joints[i,0], joints[i+1,0]],[joints[i,1], joints[i+1,1]] , kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    lats = lon_interp_func(lons)
 
    for j in range(len(lons)):
        grid.values[(Lon == lons[j]) & (Lat < lats[j])] = np.nan
    
    
grid.values[(Lon < joints[:,0].min())] = np.nan
grid.values[(Lon > joints[:,0].max())] = np.nan






## Sea level plane

sea_level = grid.copy()
sea_level.values = grid.values * 0


fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):

    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["xg1", "yg1", "za20f10g20+lDepth (km)", "wesnZ"], perspective=[160, 5], zsize="{}c".format(figz),)
  
    
    fig.grdview(
        grid=sea_level,
        region = gmt_region,
        perspective = True,
        surftype="i",
        cmap = 'lightgray'
    )

    dem_color = pygmt.makecpt(cmap = '../DATA/MAPPING/natural_mod.cpt',   series = [0, np.nanmax(grid.values)])

    fig.grdview(
        grid=grid,
        shading = '+a135+nt0.6', 
        region = gmt_region,
        perspective = True,
        surftype="i",
        cmap= True,
    )
    
    
    

    shape_grid = rasterize('../GIS/APVC2_Raster.tif')
    pygmt.makecpt(cmap="pink", series=[np.nanmin(shape_grid.values)-1, np.nanmax(shape_grid.values)])
    fig.grdview(
        grid=shape_grid,
        shading = '+a135+nt0.6', 
        region = gmt_region,
        perspective = True,
        surftype="s",
        cmap= True,
        transparency = 50,
    )
    
    shape_grid = rasterize('../GIS/Calderas_Raster.tif')
    pygmt.makecpt(cmap="lightred", series=[np.nanmin(shape_grid.values)-1, np.nanmax(shape_grid.values)])
    fig.grdview(
        grid=shape_grid,
        shading = '+a135+nt0.6', 
        region = gmt_region,
        perspective = True,
        surftype="s",
        cmap= True,
        transparency = 50,
    )
    
    
    
    
    
    
    slab = xr.open_dataset('../DATA/MAPPING/sam_slab2_dep_02.23.18.grd')
    slab.coords['lon'] = slab.x - 360; slab.coords['lat'] = slab.y
    slab = slab.swap_dims({'x': 'lon', 'y': 'lat'})
    slab.z.values[slab.z.values < base] = np.nan
    Lon, Lat = np.meshgrid(slab.lon.values, slab.lat.values)
    for i in range(len(joints)-1):
        lons = np.unique(Lon[(Lon > joints[i,0]) & (Lon < joints[i+1,0])])
        lon_interp_func = interp1d([joints[i,0], joints[i+1,0]],[joints[i,1], joints[i+1,1]] , kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
        lats = lon_interp_func(lons)
     
        for j in range(len(lons)):
            slab.z.values[(Lon == lons[j]) & (Lat > lats[j])] = np.nan

    slab.z.values[(Lon < joints[:,0].min())] = np.nan
    slab.z.values[(Lon > joints[:,0].max())] = np.nan
    
    
    pygmt.makecpt(cmap="lightgray", series=[-100, 0, 10], continuous=False)
    fig.grdview(
        grid=slab.z,
        region = gmt_region,
        perspective = True,
        surftype = 's',
        contourpen = '.1p',
        cmap = True,
        transparency = 50

    )




    fig.plot3d(x = joints[:,0], y = joints[:,1], z = [10, 10],  style = 'u0.2c', pen = '0.2p,black', fill = 'green',
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=[160, 5], zsize="{}c".format(figz))
    
    

fig.show()



fig.savefig('../FIGURES/F8_CCP_FullGrid.png', dpi = 800)



#%% Plot Inidividual grids


fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):

    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["xf1", "yf1", "za20f10+lDepth (km)", "wesnZ"], perspective=[160, 5], zsize="{}c".format(figz),)
  
fig.show()
fig.savefig('../FIGURES/F8_CCP_EmptyGrid.png', dpi = 800, transparent = True)
    


fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):

    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["xf1g1", "yf1g1", "za20f10g10+lDepth (km)", "wesnZ"], perspective=[160, 5], zsize="{}c".format(figz),)
  
fig.show()
fig.savefig('../FIGURES/F8_CCP_DrawnGrid.png', dpi = 800, transparent = True)
    



fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):

    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["xf1g1", "yf1g1", "za20f10g20+lDepth (km)", "wesnZ"], perspective=[160, 5], zsize="{}c".format(figz),)
  
    fig.grdview(
        grid=sea_level,
        region = gmt_region,
        perspective = True,
        surftype="i",
        cmap = 'lightgray'
    )

    dem_color = pygmt.makecpt(cmap = '../DATA/MAPPING/natural_mod.cpt',   series = [0, np.nanmax(grid.values)])
    fig.grdview(
        grid=grid,
        shading = '+a135+nt0.6', 
        region = gmt_region,
        perspective = True,
        surftype="i",
        cmap= True,
    )
    shape_grid = rasterize('../GIS/APVC2_Raster.tif')
    pygmt.makecpt(cmap="pink", series=[np.nanmin(shape_grid.values)-1, np.nanmax(shape_grid.values)])
    fig.grdview(
        grid=shape_grid,
        shading = '+a135+nt0.6', 
        region = gmt_region,
        perspective = True,
        surftype="s",
        cmap= True,
        transparency = 50,
    ) 
    shape_grid = rasterize('../GIS/Calderas_Raster.tif')
    pygmt.makecpt(cmap="lightred", series=[np.nanmin(shape_grid.values)-1, np.nanmax(shape_grid.values)])
    fig.grdview(
        grid=shape_grid,
        shading = '+a135+nt0.6', 
        region = gmt_region,
        perspective = True,
        surftype="s",
        cmap= True,
        transparency = 50,
    )
    
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain", MAP_TICK_LENGTH = '2c'):
    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["xf1g1", "yf1g1", "wesn"], perspective=[160, 5], zsize="{}c".format(figz),)

    
    

fig.show()
fig.savefig('../FIGURES/F8_CCP_TopoGrid.png', dpi = 800, transparent = True)



fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):

    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["xf1", "yf1", "za20f10+lDepth (km)", "wesnZ"], perspective=[160, 5], zsize="{}c".format(figz),)
  
    slab = xr.open_dataset('../DATA/MAPPING/sam_slab2_dep_02.23.18.grd')
    slab.coords['lon'] = slab.x - 360; slab.coords['lat'] = slab.y
    slab = slab.swap_dims({'x': 'lon', 'y': 'lat'})
    slab.z.values[slab.z.values < base] = np.nan
    Lon, Lat = np.meshgrid(slab.lon.values, slab.lat.values)
    for i in range(len(joints)-1):
        lons = np.unique(Lon[(Lon > joints[i,0]) & (Lon < joints[i+1,0])])
        lon_interp_func = interp1d([joints[i,0], joints[i+1,0]],[joints[i,1], joints[i+1,1]] , kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
        lats = lon_interp_func(lons)
     
        for j in range(len(lons)):
            slab.z.values[(Lon == lons[j]) & (Lat > lats[j])] = np.nan

    slab.z.values[(Lon < joints[:,0].min())] = np.nan
    slab.z.values[(Lon > joints[:,0].max())] = np.nan
    
    
    pygmt.makecpt(cmap="lightgray", series=[-100, 0, 10], continuous=False)
    fig.grdview(
        grid=slab.z,
        region = gmt_region,
        perspective = True,
        surftype = 's',
        contourpen = '.1p',
        cmap = True,
        transparency = 50

    )

fig.show()
fig.savefig('../FIGURES/F8_CCP_SlabGrid.png', dpi = 800, transparent = True)


#%% Plot CCP Frames


Gs = [1.0, 2.5, 5.0, 7.5]

for G in Gs:
    fig = pygmt.Figure()
    with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):
        
        fig.basemap(region = [0, line_dist.max(), base, 0], projection = 'X{}c/{}c'.format(figx, figz), frame = ['wsen'])
    
        model = xr.open_dataset(f'../DATA/MAPPING/nc_models/CCP_TANGO_North_G-{G}_GDW_RE.nc')
        model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})
        model_slice.Amplitude.values = np.clip(model_slice.Amplitude.values, -0.5, 0.5)
    
    
        grid_color = pygmt.makecpt(cmap = '../DATA/MAPPING/no_green.cpt',   series = [-0.5, 0.5])
        fig.grdimage(grid = model_slice['Amplitude'], cmap = True, transparency = 25)
    
        # fig.basemap(region = [0, line_dist.max(), base, 0], projection = 'X{}c/{}c'.format(figx, figz), frame = ['WSen', 'xa100f10g50+lProfile Distance (km)', 'ya50f10g50+lDepth (km)'])
    
    
    fig.show()
    
    fig.savefig(f'../FIGURES/F8_CCP_G-{G}_Frame.png', dpi = 800, transparent = True)




fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):
    
    fig.basemap(region = [0, line_dist.max(), base, 0], projection = 'X{}c/{}c'.format(figx, figz), frame = ['wsen'])


    LAB = pygmt.xyz2grd(data = '../DATA/MAPPING/Tassara_2012_LAB.txt', spacing = (1, 1), region = gmt_region[0:4])
    
    lab = LAB.interp(y = y, x = x, method = 'linear')
    lab = lab.assign_coords({'Distance_Along_Trend': line_dist})
    
    fig.plot(x = lab.Distance_Along_Trend.values, y = lab.values, pen = '1p,black,--')

    
fig.show()
fig.savefig('../FIGURES/F8_CCP_LAB_Frame.png', dpi = 800, transparent = True)



