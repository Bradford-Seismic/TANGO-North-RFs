#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:31:08 2024

@author: bradford
"""


import numpy as np
import pandas as pd
import os
import sys
import subprocess
import shutil
import obspy
import obspy.taup.taup_geo as taup
from obspy.taup import TauPyModel
import matplotlib.pyplot as plt
from matplotlib import image
from obspy.core import UTCDateTime as utc
from obspy.core import Stream, read
from obspy.io.sac import SACTrace
import obspy.taup.taup_geo as taup
from glob import glob
import pygmt
import math
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely import geometry
from shapely.geometry import Point, Polygon
from math import radians, sin, cos, atan2, sqrt
from scipy.signal import savgol_filter

active_dir = './'

output_dir = './DATA/'
figures = './FIGURES/'
os.chdir(active_dir)


"""
    The goal of this script is to develop 3D velocity models using the constraints provided by other
    data, and combine separate models to create a single averaged model


"""


#%% Look at two datasets


"""
Only purpose is to make sure models are reproducible under different algorithms
or to simply cross compare

"""

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

joints =  np.loadtxt('../DATA/MAPPING/TANGO_North_CCP_Line.txt', delimiter = ',')

joints_dist, line, line_dist = Return_Line_Elements(joints)

fig_x = 20
depth = 150
aspect = 1

ratio = line_dist.max() / 20 # km per figure cm
fig_y = depth/ratio

x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')




m1 = 'FWT-SouthAmerica-2022.r0.0.nc'
m2 = 'BlendedModel_0.7APVC-ANT_0.3FWT_MantleSpec_Smoothed.nc'
m3 = 'BlendedModel_0.7APVC-ANT_0.3FWT_MantleSpec_Smoothed_RE.nc'
m4 = 'BlendedModel_0.7APVC-ANT_0.3FWT_BigLVZ_Smoothed_RE.nc'

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    with xr.open_dataset(os.path.join('../DATA/MAPPING/nc_models', m1)) as model:

        model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None}).vs
        model_slice.values = np.clip(model_slice.values, 2.0, np.nanmax(model_slice.values))
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})

        grid_color = pygmt.makecpt(cmap = 'jet', series = [2.0, 6.0, 0.2], reverse = True)
        fig.grdimage(grid = model_slice, cmap = True)
        fig.text(position = 'BR', text = m1)


    fig.shift_origin(yshift = '-{}c'.format(fig_y + 0.5))

    fig.basemap(region = [0, line_dist.max(), 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    with xr.open_dataset(os.path.join('../DATA/MAPPING/nc_models', m2)) as model:
        model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None}).vs
        model_slice.values = np.clip(model_slice.values, 2.0, np.nanmax(model_slice.values))
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})

        grid_color = pygmt.makecpt(cmap = 'jet', series = [2.0, 6.0, 0.2], reverse = True)
        fig.grdimage(grid = model_slice, cmap = True)
        fig.text(position = 'BR', text = m2)

    fig.shift_origin(yshift = '-{}c'.format(fig_y + 0.5))

    fig.basemap(region = [0, line_dist.max(), 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    with xr.open_dataset(os.path.join('../DATA/MAPPING/nc_models', m3)) as model:
        model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None}).vs
        model_slice.values = np.clip(model_slice.values, 2.0, np.nanmax(model_slice.values))
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})

        grid_color = pygmt.makecpt(cmap = 'jet', series = [2.0, 6.0, 0.2], reverse = True)
        fig.grdimage(grid = model_slice, cmap = True)
        fig.text(position = 'BR', text = m3)

    fig.shift_origin(yshift = '-{}c'.format(fig_y + 0.5))

    fig.basemap(region = [0, line_dist.max(), 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    with xr.open_dataset(os.path.join('../DATA/MAPPING/nc_models', m4)) as model:
        model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None}).vs
        model_slice.values = np.clip(model_slice.values, 2.0, np.nanmax(model_slice.values))
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})

        grid_color = pygmt.makecpt(cmap = 'jet', series = [2.0, 6.0, 0.2], reverse = True)
        fig.grdimage(grid = model_slice, cmap = True)
        fig.text(position = 'BR', text = m4)

fig.show()




m1 = 'ConstantVpVs_ContouredBy-FWT_Smoothed.nc'
m2 = 'ConstantVpVs_ContouredBy-FWT_RE_Smoothed.nc'
m3 = 'CentralAndes_Zoned_VpVs_Arc-1.85_Smoothed.nc'
m4 = 'CentralAndes_Zoned_VpVs_Arc-1.85_Puna-1.8_RE_Smoothed.nc'

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    with xr.open_dataset(os.path.join('../DATA/MAPPING/nc_models', m1)) as model:
        model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None}).VpVs
        model_slice.values = np.clip(model_slice.values, 1.72,  1.82)
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})

        grid_color = pygmt.makecpt(cmap = 'jet', series = [1.72, 1.82, 0.008], reverse = True)
        fig.grdimage(grid = model_slice, cmap = True)
        fig.text(position = 'BR', text = m1)

    fig.shift_origin(yshift = '-{}c'.format(fig_y + 0.5))

    fig.basemap(region = [0, line_dist.max(), 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    with xr.open_dataset(os.path.join('../DATA/MAPPING/nc_models', m2)) as model:
        model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None}).VpVs
        model_slice.values = np.clip(model_slice.values, 1.72,  1.82)
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})

        grid_color = pygmt.makecpt(cmap = 'jet', series = [1.72, 1.82, 0.008], reverse = True)
        fig.grdimage(grid = model_slice, cmap = True)
        fig.text(position = 'BR', text = m2)

    fig.shift_origin(yshift = '-{}c'.format(fig_y + 0.5))

    fig.basemap(region = [0, line_dist.max(), 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    with xr.open_dataset(os.path.join('../DATA/MAPPING/nc_models', m3)) as model:

        model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None}).VpVs
        model_slice.values = np.clip(model_slice.values, 1.72,  1.82)
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})

        grid_color = pygmt.makecpt(cmap = 'jet', series = [1.72, 1.82, 0.008], reverse = True)
        fig.grdimage(grid = model_slice, cmap = True)
        fig.text(position = 'BR', text = m3)

    fig.shift_origin(yshift = '-{}c'.format(fig_y + 0.5))

    fig.basemap(region = [0, line_dist.max(), 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    with xr.open_dataset(os.path.join('../DATA/MAPPING/nc_models', m4)) as model:

        model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None}).VpVs
        model_slice.values = np.clip(model_slice.values, 1.72,  1.82)
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})

        grid_color = pygmt.makecpt(cmap = 'jet', series = [1.72, 1.82, 0.008], reverse = True)
        fig.grdimage(grid = model_slice, cmap = True)
        fig.text(position = 'BR', text = m4)



fig.show()







#%% Create 3D model with Constant VpVs  within mantle and crust constrained at velocity contour
plt.close('all')


# the input data to serve as the constraint
data1 = xr.open_dataset('../DATA/MAPPING/nc_models/FWT-SouthAmerica-2022.r0.0.nc')

# plot slice of the data

# slice along latitude
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    vs_slice = data1.interp(latitude = -23, method = 'nearest').vs
    vs_slice.values = np.clip(vs_slice.values, 2.0, np.nanmax(vs_slice.values.max()))


    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    c = pygmt.makecpt(cmap = 'jet', series = [2, 7], continuous = True)
    fig.grdimage(grid = vs_slice, cmap =True)

    fig.text(position = 'BR', text = 'vs slice = -23 latitude', font ='24p')


    fig.colorbar()

fig.show()

# depth slice
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    vs_slice = data1.interp(depth = 15, method = 'nearest').vs
    vs_slice.values = np.clip(vs_slice.values, 2.0, np.nanmax(vs_slice.values.max()))


    fig.basemap(region = [-72, -62, -30, -18], projection = 'M20c', frame = ['WSne', 'xafg', 'yafg'])
    c = pygmt.makecpt(cmap = 'jet', series = [2, 7], continuous = True)
    fig.grdimage(grid = vs_slice, cmap =True)

    fig.coast(borders = ["1/1.2p,black,--"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

    fig.text(position = 'BR', text = 'vs slice = 15 km', font ='24p')

    fig.colorbar()

fig.show()





# construct a gridded mesh based on the size of the input model
def get_sample_interval(data):
    delta = np.array([])
    for i in range(len(data)-1):
        delta = np.append(delta, data[i+1] - data[i])

    delta = np.mean(delta)
    return delta


data1x, data1y = data1.longitude.values, data1.latitude.values
data1x_dim = np.round(get_sample_interval(data1x), 3)
data1y_dim = np.round(get_sample_interval(data1y), 3)



data1x, data1y = np.meshgrid(data1x, data1y)


z1 = data1.depth.values
z1_dim = np.round(get_sample_interval(z1), 1)


# Show grid points of input model
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):

    fig.basemap(region = [-72, -62, -30, -18], projection = 'M20c', frame = ['WSne', 'xafg', 'yafg'])

    fig.coast(borders = ["1/1.2p,black,--"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

    fig.plot(x = data1x.flatten(), y = data1y.flatten(), style = 'x0.05c', pen = '0.5p,black')
    fig.text(position = 'BR', text = 'x-dim = {}, y-dim = {}, z-dim = {}'.format(data1x_dim, data1y_dim, z1_dim),
             font ='24p')

fig.show()




west_limit =  -72 #np.ceil(data1x.min())
east_limit = -62 #np.ceil(data1x.max())

north_limit = -20 #np.ceil(data1y.max())
south_limit = -28 #np.ceil(data1y.min())

depth_lim = z1.max()

resampx = np.arange(west_limit, east_limit + data1x_dim, data1x_dim)
resampy = np.arange(south_limit, north_limit + data1y_dim, data1y_dim)
resampz = np.arange(0, 200 + z1_dim, z1_dim)

GRIDX, GRIDY = np.meshgrid(resampx, resampy)


print('resulting array will be of size {}'.format(len(resampy) * len(resampx) * len(resampz)))

coords = np.array(np.meshgrid(resampx, resampy, resampz)).T.reshape(-1, 3, order = 'F')

# Show resmpling grid points of input model
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):

    fig.basemap(region = [-72, -62, -30, -18], projection = 'M20c', frame = ['WSne', 'xafg', 'yafg'])

    fig.coast(borders = ["1/1.2p,black,--"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

    fig.plot(x = GRIDX.flatten(), y =  GRIDY.flatten(), style = 'x0.05c', pen = '0.5p,black')
    fig.text(position = 'BR', text = 'Resampling Area', font ='24p')

fig.show()






# Define constraint levels and values for how data will be assembled

mantle_vs_lim  = 4.1
mantle_depth_lim = 50

crust_vpvs = 1.74
mantle_vpvs = 1.79




# slice along latitude
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    vs_slice = data1.interp(latitude = -23, method = 'nearest').vs
    vs_slice.values = np.clip(vs_slice.values, 2.0, np.nanmax(vs_slice.values.max()))

    VSX, DEPTH = np.meshgrid(vs_slice.longitude.values, vs_slice.depth.values)

    frame_area = vs_slice.copy()
    frame_area.values[(vs_slice.values < mantle_vs_lim) & (DEPTH < mantle_depth_lim)] = crust_vpvs
    frame_area.values[frame_area != crust_vpvs] = mantle_vpvs

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    c = pygmt.makecpt(cmap = 'jet', series = [1.7, 1.9], continuous = True)
    fig.grdimage(grid = frame_area, cmap =True)


    fig.text(position = 'BR', text = 'Sample View', font ='24p')

    fig.colorbar()

fig.show()






# # Brute Force Cell by Cell solve
# v_resamp = np.empty(len(coords), dtype = 'float64')
# for i in range(len(coords)):
#     # check for data grid bounds with respect to Chile VpVs tomography model coverage

#     data1_val = data1.sel(longitude = coords[i][0], latitude = coords[i][1], depth = coords[i][2], method = 'nearest').vs.values

#     # check for mantle self-defined bounds
#     if (data1_val < mantle_vs_lim) and (coords[i][2] < mantle_depth_lim):
#         v_resamp[i] = crust_vpvs

#     # else (we are inside mantle)
#     else:
#         v_resamp[i] =  mantle_vpvs


#     print("\r", end="")
#     print("Averaging Datasets: {:.1%} ".format(i/(len(coords))), end="")



# v_model = np.hstack((coords, v_resamp.reshape((-1,1))))
# with open('../DATA/SPREADSHEETS/ConstantVpVs_ContouredBy-FWT_RE.xyz', 'w') as f:
#     for i in range(len(v_model)):
#         f.write('{:.3f},{:.3f},{:.3f},{:.2f}\n'.format(v_model[i][0], v_model[i][1], v_model[i][2], v_model[i][3]))




# xarray interpolation (smarter way)

grid_cut = data1.copy()
grid_cut = grid_cut.interp(longitude = resampx, latitude = resampy, depth = resampz, method = 'nearest')
VSY, DEPTH,  VSX = np.meshgrid(grid_cut.latitude, grid_cut.depth,  grid_cut.longitude)

grid2 = grid_cut.copy()
grid2 = grid2.rename_vars({'vs':'VpVs'})

grid2.VpVs.values[(grid_cut.vs.values < mantle_vs_lim) & (DEPTH < mantle_depth_lim)] = crust_vpvs
grid2.VpVs.values[(grid2.VpVs.values != crust_vpvs) & ~np.isnan(grid2.VpVs.values)] = mantle_vpvs


# slice along latitude
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):

    vpvs_slice = grid2.interp(latitude = -23, method = 'nearest').VpVs

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    c = pygmt.makecpt(cmap = 'jet', series = [1.7, 1.9], continuous = True)
    fig.grdimage(grid = vpvs_slice, cmap =True)


    fig.text(position = 'BR', text = 'Sample View', font ='24p')

    fig.colorbar()

fig.show()



# depth slice
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    vpvs_slice = grid2.interp(depth = 15, method = 'nearest').VpVs

    fig.basemap(region = [-72, -62, -30, -18], projection = 'M20c', frame = ['WSne', 'xafg', 'yafg'])
    c = pygmt.makecpt(cmap = 'jet', series = [1.7, 1.9], continuous = True)
    fig.grdimage(grid = vpvs_slice, cmap =True)

    fig.coast(borders = ["1/1.2p,black,--"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

    fig.text(position = 'BR', text = 'vs slice = 15 km', font ='24p')

    fig.colorbar()

fig.show()



model_encoding = {'depth': {'dtype': 'float32', '_FillValue': None },
            'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'VpVs': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }


grid2.to_netcdf('../DATA/MAPPING/nc_models/ConstantVpVs_ContouredBy-FWT_RE.nc', encoding = model_encoding)


model_smoothed = grid2.copy()


data_smooth = savgol_filter(grid2.VpVs.values, 5, 2, mode = 'nearest', axis = 0)
data_smooth = savgol_filter(data_smooth, 5, 2, mode = 'nearest', axis = 1)
data_smooth = savgol_filter(data_smooth, 5, 2, mode = 'nearest', axis = 2)

model_smoothed.VpVs.values = data_smooth

model_smoothed.to_netcdf('../DATA/MAPPING/nc_models/ConstantVpVs_ContouredBy-FWT_RE_Smoothed.nc', encoding = model_encoding)




#%% Create 3D Vp/Vs with defined tectonic zones  and other constraints



# pull data for the forearc that we want to preserve
data1 = xr.open_dataset('../DATA/MAPPING/nc_models/ChileCoast_VpVs_2024.nc')
# slice along latitude
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    model_slice = data1.interp(latitude = -23, method='nearest').vpvs
    model_slice.values = np.clip(model_slice.values, 1.72,  1.82)

    grid_color = pygmt.makecpt(cmap = 'jet', series = [1.72, 1.82, 0.008], reverse = True)
    fig.grdimage(grid = model_slice, cmap = True)
    fig.colorbar()

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

fig.show()



# pull data for the cordillera that we want to use as a constraint
data2 = xr.open_dataset('../DATA/MAPPING/nc_models/BlendedModel_0.7APVC-ANT_0.3FWT_MantleSpec_Smoothed.nc')
# slice along latitude
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    model_slice = data2.interp( latitude = -23, method='nearest').vs
    model_slice.values = np.clip(model_slice.values, 2.0, np.nanmax(model_slice.values))

    grid_color = pygmt.makecpt(cmap = 'jet', series = [2.0, 6.0, 0.2], reverse = True)
    fig.grdimage(grid = model_slice, cmap = True)
    fig.colorbar()

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

fig.show()




## Use tectonic zones within map projection to define the extent of areas
data_zone = gpd.read_file('../DATA/MAPPING/AndeanTectonicRegimes.shp')
Chile_vpvs_extent = gpd.read_file('../DATA/MAPPING/Chile_Sergio2024_VpVs_Extent.shp')


gmt_region = '-71.5/-60.5/-28.5/-19.5'
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    tecto_plates = gpd.read_file('../DATA/MAPPING/TectonicPlateBoundaries/TectonicPlateBoundaries.shp')
    APVC = gpd.read_file('../DATA/MAPPING/APVC_shp.shp')
    volcanoes = pd.read_csv('../DATA/SPREADSHEETS/SA_Holocene_Volcanoes.csv')
    prj='M15c'
    grid = pygmt.datasets.load_earth_relief(resolution="15s", region=gmt_region)

    shade = pygmt.grdgradient(grid=grid, azimuth="0/90", normalize="e1")

    fig.basemap(region=gmt_region, projection=prj, frame='ag4')

    fig.grdimage(grid=grid, shading = shade, cmap = '../DATA/MAPPING/natural_mod.cpt', projection = prj, transparency = 75)
    fig.coast(borders = ["1/1p,black,--"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

    fig.plot(data=APVC, region = gmt_region, pen = '0.7p,black', fill = 'pink', transparency = 50, label = 'APVC')
    fig.plot(data=Chile_vpvs_extent, region = gmt_region, pen = '0.7p,black', fill = 'lightblue', transparency = 50, label = 'APVC')
    fig.plot(data=data_zone, region = gmt_region, pen = '0.7p,black')

fig.show()




# construct a gridded mesh that reconciles sizes of input data
def get_sample_interval(data):
    delta = np.array([])
    for i in range(len(data)-1):
        delta = np.append(delta, data[i+1] - data[i])

    delta = np.mean(delta)
    return delta


data1x, data1y = data1.longitude.values, data1.latitude.values
data1x_dim = np.round(get_sample_interval(data1x), 3)
data1y_dim = np.round(get_sample_interval(data1y), 3)




data2x, data2y = data2.longitude.values, data2.latitude.values
data2x_dim = get_sample_interval(data2x)
data2y_dim = get_sample_interval(data2y)



data1x, data1y = np.meshgrid(data1x, data1y)
data2x, data2y = np.meshgrid(data2x, data2y)



z1 = data1.depth.values
z1_dim = get_sample_interval(z1)
z2 = data2.depth.values
z2_dim = get_sample_interval(z2)




west_limit =  -72 #np.ceil(data2.min())
east_limit = -62 #np.ceil(data2x.max())

north_limit = -20 #np.ceil(data2y.max())
south_limit = -28 #np.ceil(data2y.min())


depth_lim = z2.max()

resampx = np.arange(west_limit, east_limit + data1x_dim, data1x_dim)
resampy = np.arange(south_limit, north_limit + data1y_dim, data1y_dim)
resampz = np.arange(0, 200 + z2_dim, z2_dim)


GRIDX, GRIDY = np.meshgrid(resampx, resampy)

print('resulting array will be of size {}'.format(len(resampy) * len(resampx) * len(resampz)))

coords = np.array(np.meshgrid(resampx, resampy, resampz)).T.reshape(-1, 3, order = 'F')



# Show resmpling grid points of input model
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):

    fig.basemap(region = [-72, -62, -30, -18], projection = 'M20c', frame = ['WSne', 'xafg', 'yafg'])

    fig.coast(borders = ["1/1.2p,black,--"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

    fig.plot(x = GRIDX.flatten(), y =  GRIDY.flatten(), style = 'x0.05c', pen = '0.5p,black')
    fig.text(position = 'BR', text = 'Resampling Area', font ='24p')

fig.show()



mantle_vs_lim  = 4.1
mantle_depth_lim = 40

mantle_vpvs = 1.79


print('resulting array will be of size {}'.format(len(resampy) * len(resampx) * len(resampz)))

coords = np.array(np.meshgrid(resampx, resampy, resampz)).T.reshape(-1, 3, order = 'F')







# We are also identifying by zones, so we will separate the geometries and use them in assigning the grid space

WC = data_zone.geometry[data_zone.Region == 'WC'].reset_index(drop = True)[0]
altiplano = data_zone.geometry[data_zone.Region == 'Altiplano'].reset_index(drop = True)[0]
puna = data_zone.geometry[data_zone.Region == 'Puna'].reset_index(drop = True)[0]
EC = data_zone.geometry[data_zone.Region == 'EC'].reset_index(drop = True)[0]
SA = data_zone.geometry[data_zone.Region == 'SA'].reset_index(drop = True)[0]
SB = data_zone.geometry[data_zone.Region == 'SB'].reset_index(drop = True)[0]
SP = data_zone.geometry[data_zone.Region == 'SP'].reset_index(drop = True)[0]

# plus a zone identifying the data filled areas of the Chile VpVs map
vpvs_model_shape = Chile_vpvs_extent.geometry[0]


# order of priority:
#    Chile Vp/Vs model will not be averaged over
#   mantle Vp/Vs shall be 1.79 uniformly,
# Puna VpVs will be 1.71
# EC will be 1.77, Chaco will be 1.74
# SP will be 1.75



# stupid long brute force way
# v_resamp = np.empty(len(coords), dtype = 'float64')
# for i in range(len(coords)):
#     # check for data grid bounds with respect to Chile VpVs tomography model coverage
#     point = geometry.Point(coords[i][0], coords[i][1])


#     if vpvs_model_shape.contains(point):


#         if coords[i][2] > 100:
#             v_resamp[i] = mantle_vpvs

#         else:
#             data1_val = data1.sel(longitude = coords[i][0], latitude = coords[i][1], depth = coords[i][2], method = 'nearest').vpvs.values

#             v_resamp[i] = data1_val



#     # check if we're in data bounds for Arc coverage
#     elif WC.contains(point):
#         data2_val = data2.sel(longitude = coords[i][0], latitude = coords[i][1], depth = coords[i][2], method = 'nearest').vs.values

#         # check for mantle self-defined bounds
#         if (data2_val < mantle_vs_lim) and (coords[i][2] < mantle_depth_lim):
#             v_resamp[i] = 1.85

#         # else (we are inside mantle)
#         else:
#             v_resamp[i] =  mantle_vpvs


#     # check if we're in data bounds for altiplano coverage
#     elif altiplano.contains(point):
#         data2_val = data2.sel(longitude = coords[i][0], latitude = coords[i][1], depth = coords[i][2], method = 'nearest').vs.values

#         # check for mantle self-defined bounds
#         if (data2_val < mantle_vs_lim) and (coords[i][2] < mantle_depth_lim):
#             v_resamp[i] = 1.85

#         # else (we are inside mantle)
#         else:
#             v_resamp[i] =  mantle_vpvs


#     # check if we're in data bounds for puna coverage
#     elif puna.contains(point):
#         data2_val = data2.sel(longitude = coords[i][0], latitude = coords[i][1], depth = coords[i][2], method = 'nearest').vs.values

#         # check for mantle self-defined bounds
#         if (data2_val < mantle_vs_lim) and (coords[i][2] < mantle_depth_lim):
#             v_resamp[i] = 1.70

#         # else (we are inside mantle)
#         else:
#             v_resamp[i] =  mantle_vpvs


#     # check if we're in data bounds for EC coverage
#     elif EC.contains(point):
#         data2_val = data2.sel(longitude = coords[i][0], latitude = coords[i][1], depth = coords[i][2], method = 'nearest').vs.values

#         # check for mantle self-defined bounds
#         if (data2_val < mantle_vs_lim) and (coords[i][2] < mantle_depth_lim):
#             v_resamp[i] = 1.77

#         # else (we are inside mantle)
#         else:
#             v_resamp[i] =  mantle_vpvs



#     # we are outside any polygon coverage area, use flat value
#     else:
#         data2_val = data2.sel(longitude = coords[i][0], latitude = coords[i][1], depth = coords[i][2], method = 'nearest').vs.values

#         # check for mantle self-defined bounds
#         if (data2_val < mantle_vs_lim) and (coords[i][2] < mantle_depth_lim):
#             v_resamp[i] = 1.74

#         # else (we are inside mantle)
#         else:
#             v_resamp[i] =  mantle_vpvs


#     print("\r", end="")
#     print("Averaging Datasets: {:.1%} ".format(i/(len(coords))), end="")


# v_model = np.hstack((coords, v_resamp.reshape((-1,1))))
# with open('../DATA/SPREADSHEETS/CentralAndes_Zoned_VpVs_Arc-1.85_Plateau-1.85.xyz', 'w') as f:
#     for i in range(len(v_model)):
#         f.write('{:.3f},{:.3f},{:.3f},{:.2f}\n'.format(v_model[i][0], v_model[i][1], v_model[i][2], v_model[i][3]))




#### smarter array with xarray interpolation geopandas intersections

x_flat = GRIDX.ravel()
y_flat = GRIDY.ravel()
vpvs_flat = np.ones(x_flat.shape) *1.74

assign_zones = [WC, altiplano, puna, EC, vpvs_model_shape]
assign_vpvs = [1.85, 1.78, 1.82, 1.77, 0]
for i, zone in enumerate(assign_zones):
    points = gpd.GeoSeries([Point(x, y) for x, y in zip(x_flat, y_flat)], crs=data_zone.crs)
    mask = points.within(zone)

    vpvs_flat[mask] = assign_vpvs[i]

vpvs_grid = np.reshape(vpvs_flat, GRIDX.shape)
vpvs_da = xr.DataArray(data = vpvs_grid,  dims = ['latitude', 'longitude', ], coords = {'latitude': resampy, 'longitude': resampx})

# Show zoned model, note that we've left a whole where the vp/vs model will soon go
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):

    fig.basemap(region = [-72, -62, -30, -18], projection = 'M20c', frame = ['WSne', 'xafg', 'yafg'])

    fig.coast(borders = ["1/1.2p,black,--"], shorelines="1/0.5p", water = "skyblue", transparency = 10)
    grid_color = pygmt.makecpt(cmap = 'jet', series = [1.70, 1.88, 0.008], reverse = True)
    fig.grdimage(grid = vpvs_da,  cmap = True)
    fig.plot(data=data_zone, pen = '0.7p,black')

fig.show()




# expand vpvs into depth

vpvs_volume = np.repeat(vpvs_grid.T[np.newaxis, :, :], len(resampz), axis=0)

# constrain volume by data from vs model to segment off mantle, match the vpvs_volume orientation to match vs data grid

grid_cut = data2.copy()
grid_cut = grid_cut.interp(longitude = resampx, latitude = resampy, depth = resampz, method = 'nearest')
VSX, DEPTH, VSY= np.meshgrid(   grid_cut.longitude, grid_cut.depth,   grid_cut.latitude,)

crust_mask = np.zeros(vpvs_volume.shape)
crust_mask[(grid_cut.vs.values < mantle_vs_lim) & (DEPTH < mantle_depth_lim)] = 1
crust_mask[crust_mask != 1] = np.nan

vpvs_volume = vpvs_volume * crust_mask
vpvs_volume[np.isnan(vpvs_volume)] = 1.79


# mask area outside of vpvs region we want to punch in later
points = gpd.GeoSeries([Point(x, y) for x, y in zip(x_flat, y_flat)], crs=Chile_vpvs_extent.crs)
mask = points.within(vpvs_model_shape)

data1_flat = np.ones(x_flat.shape)
data1_flat[mask] = 0
data1_mask_grid = np.reshape(data1_flat, GRIDX.shape)

data1_mask_volume = np.repeat(data1_mask_grid.T[np.newaxis, :, :], len(resampz), axis=0)

vpvs_volume = vpvs_volume * data1_mask_volume


vpvs_da = xr.DataArray(data = vpvs_volume,  dims = ['depth',  'longitude', 'latitude',], coords = {'depth': resampz,  'longitude': resampx, 'latitude': resampy,})

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    model_slice = vpvs_da.interp( latitude = -23, method='nearest')
    model_slice.values = np.clip(model_slice.values, 1.72,  1.82)

    grid_color = pygmt.makecpt(cmap = 'jet', series = [1.72, 1.828, 0.008], reverse = True)
    fig.grdimage(grid = model_slice, cmap = True)
    fig.colorbar()

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])


fig.show()




# now punch in data1 vpvs model

data1_resamp = data1.interp(longitude = resampx, latitude = resampy, depth = resampz, method = 'nearest').vpvs
data1_resamp.values[data1_resamp.depth.values > 100] = 1.79     # bottom out model
data1_resamp.values[np.isnan(data1_resamp.values)] = 0

x_flat = GRIDX.ravel()
y_flat = GRIDY.ravel()
data1_flat = np.zeros(x_flat.shape)

# mask outside area
points = gpd.GeoSeries([Point(x, y) for x, y in zip(x_flat, y_flat)], crs=Chile_vpvs_extent.crs)
mask = points.within(vpvs_model_shape)

data1_flat[mask] = 1
data1_mask_grid = np.reshape(data1_flat, GRIDX.shape)

data1_mask_volume = np.repeat(data1_mask_grid[np.newaxis, :, :], len(resampz), axis=0)


# multply 3d mask and 3d vpvs to constrain section

data1_resamp.values = data1_resamp.values * data1_mask_volume

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    model_slice = data1_resamp.interp( latitude = -23, method='nearest')
    model_slice.values = np.clip(model_slice.values, 1.72,  1.82)

    grid_color = pygmt.makecpt(cmap = 'jet', series = [1.72, 1.828, 0.008], reverse = True)
    fig.grdimage(grid = model_slice, cmap = True)
    fig.colorbar()

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])


fig.show()


# now sum the two volumes, match the volume orientation to match vpvs_volume

data1_volume = np.transpose(data1_resamp.values, (0, 2, 1))
vpvs_volume = vpvs_volume + data1_volume

vpvs_da = xr.DataArray(data = vpvs_volume,  dims = ['depth',  'longitude', 'latitude',], coords = {'depth': resampz,  'longitude': resampx, 'latitude': resampy,})

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    model_slice = vpvs_da.interp( latitude = -23, method='nearest')
    model_slice.values = np.clip(model_slice.values, 1.72,  1.82)

    grid_color = pygmt.makecpt(cmap = 'jet', series = [1.72, 1.828, 0.008], reverse = True)
    fig.grdimage(grid = model_slice, cmap = True)
    fig.colorbar()

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])


fig.show()



# smooth the model
model_smoothed = vpvs_da.copy()


data_smooth = savgol_filter(model_smoothed.values, 20, 1, mode = 'nearest', axis = 0)
data_smooth = savgol_filter(data_smooth, 16, 1, mode = 'nearest', axis = 1)
data_smooth = savgol_filter(data_smooth, 16, 1, mode = 'nearest', axis = 2)

model_smoothed.values = data_smooth
model_smoothed = model_smoothed.to_dataset( name = 'VpVs')

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    model_slice = model_smoothed.interp( latitude = -23, method='nearest').VpVs
    model_slice.values = np.clip(model_slice.values, 1.72,  1.82)

    grid_color = pygmt.makecpt(cmap = 'jet', series = [1.72, 1.828, 0.008], reverse = True)
    fig.grdimage(grid = model_slice, cmap = True)
    fig.colorbar()

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])


fig.show()


model_encoding = {'depth': {'dtype': 'float32', '_FillValue': None },
            'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'VpVs': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }


model_smoothed.to_netcdf('../DATA/MAPPING/nc_models/CentralAndes_Zoned_VpVs_Arc-1.85_Puna-1.82_RE_Smoothed.nc', encoding = model_encoding)


#%% Create 3D vs model with blended averaging


data1 = xr.open_dataset(
    '../DATA/MAPPING/nc_models/FWT-SouthAmerica-2022.r0.0.nc', engine='netcdf4')
data2 = xr.open_dataset(
    '../DATA/MAPPING/nc_models/APVC+Puna.ANT+RF.Ward.2017_kmps.nc', engine='netcdf4')
data3 = xr.open_dataset(
    '../DATA/MAPPING/nc_models/Andes.ANT.Ward.2013_kmps.nc', engine='netcdf4')






# construct a gridded mesh that reconciles sizes of input data
def get_sample_interval(data):
    delta = np.array([])
    for i in range(len(data)-1):
        delta = np.append(delta, data[i+1] - data[i])

    delta = np.mean(delta)
    return delta


data1x, data1y = data1.longitude.values, data1.latitude.values
data1x_dim = get_sample_interval(data1x)
data1y_dim = get_sample_interval(data1y)

data2x, data2y = data2.longitude.values, data2.latitude.values
data2x_dim = get_sample_interval(data2x)
data2y_dim = get_sample_interval(data2y)

data3x, data3y = data3.longitude.values, data3.latitude.values
data3x_dim = get_sample_interval(data3x)
data3y_dim = get_sample_interval(data3y)

data1x, data1y = np.meshgrid(data1x, data1y)
data2x, data2y = np.meshgrid(data2x, data2y)
data3x, data3y = np.meshgrid(data3x, data3y)

z1 = data1.depth.values
z1_dim = get_sample_interval(z1)
z2 = data2.depth.values
z2_dim = get_sample_interval(z2)
z3 = data3.depth.values
z3_dim = get_sample_interval(z3)





west_limit =  -72 #np.ceil(data2.min())
east_limit = -63 #np.ceil(data2x.max())

north_limit = -20 #np.ceil(data2y.max())
south_limit = -28 #np.ceil(data2y.min())


depth_lim = z2.max()

resampx = np.arange(west_limit, east_limit + data2x_dim, data2x_dim)
resampy = np.arange(south_limit, north_limit + data2y_dim, data2y_dim)
resampz = np.arange(0, 200 + z2_dim, z2_dim)


GRIDX, GRIDY = np.meshgrid(resampx, resampy)

print('resulting array will be of size {}'.format(len(resampy) * len(resampx) * len(resampz)))

coords = np.array(np.meshgrid(resampx, resampy, resampz)).T.reshape(-1, 3, order = 'F')



# Show resmpling grid points of input model
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):

    fig.basemap(region = [-72, -62, -30, -18], projection = 'M20c', frame = ['WSne', 'xafg', 'yafg'])

    fig.coast(borders = ["1/1.2p,black,--"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

    fig.plot(x = GRIDX.flatten(), y =  GRIDY.flatten(), style = 'x0.05c', pen = '0.5p,black')
    fig.text(position = 'BR', text = 'Resampling Area', font ='24p')

fig.show()




# Interpolate each model over new grid space

data1_re = data1.interp(latitude = resampy, longitude = resampx, depth = resampz, method = 'nearest')
data2_re = data2.interp(latitude = resampy, longitude = resampx, depth = resampz, method = 'nearest')
data3_re = data3.interp(latitude = resampy, longitude = resampx, depth = resampz, method = 'nearest')

data3_re = data3_re.where(data3_re.longitude <= data2x.min())


# assign all nan areas to 0
data1_re.vs.values[np.isnan(data1_re.vs.values)] = 0
data2_re.vs.values[np.isnan(data2_re.vs.values)] = 0
data3_re.vs.values[np.isnan(data3_re.vs.values)] = 0

# pull the negative region by identifying all zeroed areas
data2_negative = data2_re.vs.copy()
data2_negative.values[data2_negative.values != 0] = np.nan
data2_negative.values[data2_negative.values == 0] = 1
data2_negative.values[np.isnan(data2_negative.values)] = 0


data3_negative = data3_re.vs.copy()
data3_negative.values[data3_negative.values != 0] = np.nan
data3_negative.values[data3_negative.values == 0] = 1
data3_negative.values[np.isnan(data3_negative.values)] = 0



# make a composite mask for the areas in data1 we want to preserve
data1_mask = data1_re.vs.copy()
data1_mask.values = (data2_negative.values.astype(bool) & data3_negative.values.astype(bool))*1

data1_mask_inv = data1_re.vs.copy()
data1_mask_inv.values = ~(data2_negative.values.astype(bool) & data3_negative.values.astype(bool))*1



fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):

    vs_slice = data1_re.interp(latitude = -23, method = 'nearest').vs
    vs_slice.values = np.clip(vs_slice.values, 2.0, np.nanmax(vs_slice.values))


    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    c = pygmt.makecpt(cmap = 'jet', series = [2, 7], continuous = True)
    fig.grdimage(grid = vs_slice, cmap =True)
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    fig.shift_origin(yshift = '-9c')


    vs_slice = data2_re.interp(latitude = -23, method = 'nearest').vs
    vs_slice.values = np.clip(vs_slice.values, 2.0, np.nanmax(vs_slice.values))

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    c = pygmt.makecpt(cmap = 'jet', series = [2, 7], continuous = True)
    fig.grdimage(grid = vs_slice, cmap =True)
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    fig.shift_origin(yshift = '-9c')


    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    vs_slice = data3_re.interp(latitude = -23, method = 'nearest', kwargs={"fill_value": None}).vs
    vs_slice.values = np.clip(vs_slice.values, 2.0, np.nanmax(vs_slice.values))

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    c = pygmt.makecpt(cmap = 'jet', series = [2, 7], continuous = True)
    fig.grdimage(grid = vs_slice, cmap =True)
    fig.colorbar()
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])



fig.show()




fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):

    vs_slice = data1_mask.interp(latitude = -23, method = 'nearest')

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    fig.grdimage(grid = vs_slice)
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    fig.shift_origin(yshift = '-9c')


    vs_slice = data2_negative.interp(latitude = -23, method = 'nearest')

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    fig.grdimage(grid = vs_slice)
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    fig.shift_origin(yshift = '-9c')


    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    vs_slice = data3_negative.interp(latitude = -23, method = 'nearest')

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    fig.grdimage(grid = vs_slice)
    fig.colorbar()
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])



fig.show()





# now each grid is of the same size in dimensions
# sum grids together using weighted scheme, note data2 and data3 don't overlap

vs_volume = data1_re.copy()

# blend overlapping zones
vs_volume.vs.values = 0.3*data1_re.vs.values + 0.7*data2_re.vs.values  + 0.7*data3_re.vs.values

# back-fill non-overlapping space
vs_volume.vs.values = (vs_volume.vs.values * data1_mask_inv.values) + (data1_re.vs.values * data1_mask.values)


fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):

    vs_slice = vs_volume.interp(latitude = -23, method = 'nearest').vs
    vs_slice.values = np.clip(vs_slice.values, 2.0, np.nanmax(vs_slice.values))


    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    c = pygmt.makecpt(cmap = 'jet', series = [2, 7], continuous = True)
    fig.grdimage(grid = vs_slice, cmap =True)
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])



fig.show()


## Reorganize and smooth data


vs_re = np.transpose(vs_volume.vs.values, (0, 2, 1))

vs_re = xr.DataArray(data = vs_re,
                     coords = {'depth': resampz, 'longitude': resampx, 'latitude': resampy},
                     dims = ['depth', 'longitude', 'latitude'])

vs_re = vs_re.to_dataset(name = 'vs')


# smooth the model
model_smoothed = vs_re.copy()


data_smooth = savgol_filter(model_smoothed.vs.values, 6, 1, mode = 'nearest', axis = 0)
data_smooth = savgol_filter(data_smooth, 4, 1, mode = 'nearest', axis = 1)
data_smooth = savgol_filter(data_smooth, 4, 1, mode = 'nearest', axis = 2)

model_smoothed.vs.values = data_smooth

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    model_slice = model_smoothed.interp( latitude = -23, method='nearest').vs
    model_slice.values = np.clip(model_slice.values, 2, 7)

    grid_color = pygmt.makecpt(cmap = 'jet', series = [2, 7, 0.2], reverse = True)
    fig.grdimage(grid = model_slice, cmap = True)
    fig.colorbar()

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])


fig.show()


model_encoding = {'depth': {'dtype': 'float32', '_FillValue': None },
            'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'vs': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }


model_smoothed.to_netcdf('../DATA/MAPPING/nc_models/BlendedModel_0.7APVC-ANT_0.3FWT_MantleSpec_Smoothed_RE.nc', encoding = model_encoding)



#%% Create 3D vs model with blended averaging and add a big LVZ


data1 = xr.open_dataset(
    '../DATA/MAPPING/nc_models/FWT-SouthAmerica-2022.r0.0.nc', engine='netcdf4')
data2 = xr.open_dataset(
    '../DATA/MAPPING/nc_models/APVC+Puna.ANT+RF.Ward.2017_kmps.nc', engine='netcdf4')
data3 = xr.open_dataset(
    '../DATA/MAPPING/nc_models/Andes.ANT.Ward.2013_kmps.nc', engine='netcdf4')






# construct a gridded mesh that reconciles sizes of input data
def get_sample_interval(data):
    delta = np.array([])
    for i in range(len(data)-1):
        delta = np.append(delta, data[i+1] - data[i])

    delta = np.mean(delta)
    return delta


data1x, data1y = data1.longitude.values, data1.latitude.values
data1x_dim = get_sample_interval(data1x)
data1y_dim = get_sample_interval(data1y)

data2x, data2y = data2.longitude.values, data2.latitude.values
data2x_dim = get_sample_interval(data2x)
data2y_dim = get_sample_interval(data2y)

data3x, data3y = data3.longitude.values, data3.latitude.values
data3x_dim = get_sample_interval(data3x)
data3y_dim = get_sample_interval(data3y)

data1x, data1y = np.meshgrid(data1x, data1y)
data2x, data2y = np.meshgrid(data2x, data2y)
data3x, data3y = np.meshgrid(data3x, data3y)

z1 = data1.depth.values
z1_dim = get_sample_interval(z1)
z2 = data2.depth.values
z2_dim = get_sample_interval(z2)
z3 = data3.depth.values
z3_dim = get_sample_interval(z3)





west_limit =  -72 #np.ceil(data2.min())
east_limit = -63 #np.ceil(data2x.max())

north_limit = -20 #np.ceil(data2y.max())
south_limit = -28 #np.ceil(data2y.min())


depth_lim = z2.max()

resampx = np.arange(west_limit, east_limit + data2x_dim, data2x_dim)
resampy = np.arange(south_limit, north_limit + data2y_dim, data2y_dim)
resampz = np.arange(0, 200 + z2_dim, z2_dim)


GRIDX, GRIDY = np.meshgrid(resampx, resampy)

print('resulting array will be of size {}'.format(len(resampy) * len(resampx) * len(resampz)))

coords = np.array(np.meshgrid(resampx, resampy, resampz)).T.reshape(-1, 3, order = 'F')



# Show resmpling grid points of input model
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):

    fig.basemap(region = [-72, -62, -30, -18], projection = 'M20c', frame = ['WSne', 'xafg', 'yafg'])

    fig.coast(borders = ["1/1.2p,black,--"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

    fig.plot(x = GRIDX.flatten(), y =  GRIDY.flatten(), style = 'x0.05c', pen = '0.5p,black')
    fig.text(position = 'BR', text = 'Resampling Area', font ='24p')

fig.show()



# Interpolate each model over new grid space

data1_re = data1.interp(latitude = resampy, longitude = resampx, depth = resampz, method = 'nearest')
data2_re = data2.interp(latitude = resampy, longitude = resampx, depth = resampz, method = 'nearest')
data3_re = data3.interp(latitude = resampy, longitude = resampx, depth = resampz, method = 'nearest')

data3_re = data3_re.where(data3_re.longitude <= data2x.min())



# make a big lvz pocket to add onto the blended model
# Define ranges
lvz = data1_re.vs.copy()
lvz.values= np.zeros(lvz.values.shape)

lon_min, lon_max = -68, -66
lat_min, lat_max = -24.3, -22.7
depth_min, depth_max = 6, 20

# Mask for the region
mask = (
    (lvz.longitude >= lon_min) & (lvz.longitude <= lon_max) &
    (lvz.latitude  >= lat_min) & (lvz.latitude  <= lat_max) &
    (lvz.depth     >= depth_min) & (lvz.depth    <= depth_max)
)

# Assign a fixed value (example: 5.0)
lvz = lvz.where(~mask, 2.7)

lvz_mask = data1_re.vs.copy()
lvz_mask.values= np.ones(lvz_mask.values.shape)
lvz_mask = lvz_mask.where(~mask, 0)



# assign all nan areas to 0
data1_re.vs.values[np.isnan(data1_re.vs.values)] = 0
data2_re.vs.values[np.isnan(data2_re.vs.values)] = 0
data3_re.vs.values[np.isnan(data3_re.vs.values)] = 0

# pull the negative region by identifying all zeroed areas
data2_negative = data2_re.vs.copy()
data2_negative.values[data2_negative.values != 0] = np.nan
data2_negative.values[data2_negative.values == 0] = 1
data2_negative.values[np.isnan(data2_negative.values)] = 0


data3_negative = data3_re.vs.copy()
data3_negative.values[data3_negative.values != 0] = np.nan
data3_negative.values[data3_negative.values == 0] = 1
data3_negative.values[np.isnan(data3_negative.values)] = 0



# make a composite mask for the areas in data1 we want to preserve
data1_mask = data1_re.vs.copy()
data1_mask.values = (data2_negative.values.astype(bool) & data3_negative.values.astype(bool))*1

data1_mask_inv = data1_re.vs.copy()
data1_mask_inv.values = ~(data2_negative.values.astype(bool) & data3_negative.values.astype(bool))*1



fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):

    vs_slice = data1_re.interp(latitude = -23, method = 'nearest').vs
    vs_slice.values = np.clip(vs_slice.values, 2.0, np.nanmax(vs_slice.values))


    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    c = pygmt.makecpt(cmap = 'jet', series = [2, 7], continuous = True)
    fig.grdimage(grid = vs_slice, cmap =True)
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    fig.shift_origin(yshift = '-9c')


    vs_slice = data2_re.interp(latitude = -23, method = 'nearest').vs
    vs_slice.values = np.clip(vs_slice.values, 2.0, np.nanmax(vs_slice.values))

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    c = pygmt.makecpt(cmap = 'jet', series = [2, 7], continuous = True)
    fig.grdimage(grid = vs_slice, cmap =True)
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    fig.shift_origin(yshift = '-9c')


    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    vs_slice = data3_re.interp(latitude = -23, method = 'nearest', kwargs={"fill_value": None}).vs
    vs_slice.values = np.clip(vs_slice.values, 2.0, np.nanmax(vs_slice.values))

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    c = pygmt.makecpt(cmap = 'jet', series = [2, 7], continuous = True)
    fig.grdimage(grid = vs_slice, cmap =True)
    fig.colorbar()
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])



fig.show()




fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):

    vs_slice = data1_mask.interp(latitude = -23, method = 'nearest')

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    fig.grdimage(grid = vs_slice)
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    fig.shift_origin(yshift = '-9c')


    vs_slice = data2_negative.interp(latitude = -23, method = 'nearest')

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    fig.grdimage(grid = vs_slice)
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    fig.shift_origin(yshift = '-9c')


    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    vs_slice = data3_negative.interp(latitude = -23, method = 'nearest')

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    fig.grdimage(grid = vs_slice)
    fig.colorbar()
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])



fig.show()





# now each grid is of the same size in dimensions
# sum grids together using weighted scheme, note data2 and data3 don't overlap

vs_volume = data1_re.copy()

# blend overlapping zones
vs_volume.vs.values = 0.3*data1_re.vs.values + 0.7*data2_re.vs.values  + 0.7*data3_re.vs.values

# back-fill non-overlapping space
vs_volume.vs.values = (vs_volume.vs.values * data1_mask_inv.values) + (data1_re.vs.values * data1_mask.values)


# punch in big LVZ

vs_volume.vs.values = (vs_volume.vs.values * lvz_mask.values) + lvz.values



fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):

    vs_slice = vs_volume.interp(latitude = -23, method = 'nearest').vs
    vs_slice.values = np.clip(vs_slice.values, 2.0, np.nanmax(vs_slice.values))


    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])
    c = pygmt.makecpt(cmap = 'jet', series = [2, 7], continuous = True)
    fig.grdimage(grid = vs_slice, cmap =True)
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])



fig.show()


## Reorganize and smooth data


vs_re = np.transpose(vs_volume.vs.values, (0, 2, 1))

vs_re = xr.DataArray(data = vs_re,
                     coords = {'depth': resampz, 'longitude': resampx, 'latitude': resampy},
                     dims = ['depth', 'longitude', 'latitude'])

vs_re = vs_re.to_dataset(name = 'vs')


# smooth the model
model_smoothed = vs_re.copy()


data_smooth = savgol_filter(model_smoothed.vs.values, 6, 1, mode = 'nearest', axis = 0)
data_smooth = savgol_filter(data_smooth, 4, 1, mode = 'nearest', axis = 1)
data_smooth = savgol_filter(data_smooth, 4, 1, mode = 'nearest', axis = 2)

model_smoothed.vs.values = data_smooth

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])

    model_slice = model_smoothed.interp( latitude = -23, method='nearest').vs
    model_slice.values = np.clip(model_slice.values, 2, 7)

    grid_color = pygmt.makecpt(cmap = 'jet', series = [2, 7, 0.2], reverse = True)
    fig.grdimage(grid = model_slice, cmap = True)
    fig.colorbar()

    fig.basemap(region = [-72, -62, 0, 200], projection = 'X20c/-8c', frame = ['WSne', 'xafg', 'yafg'])


fig.show()


model_encoding = {'depth': {'dtype': 'float32', '_FillValue': None },
            'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'vs': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }


model_smoothed.to_netcdf('../DATA/MAPPING/nc_models/BlendedModel_0.7APVC-ANT_0.3FWT_BigLVZ_Smoothed_RE.nc', encoding = model_encoding)



#%% Write and save v_model.xyz as exportable .nc file (DEPRICATED)




plt.close('all')

v_model = np.loadtxt('../DATA/SPREADSHEETS/CentralAndes_Zoned_VpVs_Arc-1.85.xyz', delimiter = ',', dtype = 'float32')



v_resamp = v_model[:,3].copy()

v_resamp = v_resamp.reshape((len(resampz),  len(resampx), len(resampy)), order = 'F')

new_model = xr.Dataset(data_vars = {'VpVs':(['depth',  'longitude', 'latitude'], v_resamp)},
                          coords = {'depth': resampz, 'longitude': resampx, 'latitude': resampy},
                          attrs = {'units': ""
                                   })


model_encoding = {'depth': {'dtype': 'float32', '_FillValue': None },
            'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'VpVs': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }





APVC = xr.open_dataset(
    '../DATA/MAPPING/sv_models/APVC+Puna.ANT+RF.Ward.2017_kmps.nc', engine='netcdf4')
FWT = xr.open_dataset(
    '../DATA/MAPPING/sv_models/FWT-SouthAmerica-2022.r0.0.nc', engine='netcdf4')
grid = pygmt.datasets.load_earth_relief(
    resolution="01m", region='-71.5/-60.5/-28.5/-19.5')




plt.figure()
FWT.sel(latitude=-23, method='nearest').vs.plot.contourf(yincrease=False, levels=10, cmap='jet',vmin=1.5, vmax=6, xlim = [-78, -62], ylim = [200, 0])

plt.figure()
APVC.sel(latitude=-23, method='nearest').vs.plot.contourf(yincrease=False, levels=10, cmap='jet', vmin=1.5, vmax=6, xlim = [-78, -62], ylim = [200, 0])


plt.figure()
new_model.sel(latitude = -36, method = 'nearest').VpVs.plot.contourf(yincrease = False, levels = 15, vmin = 1.69, vmax = 1.86, cmap = 'jet')





plt.figure()

plt.rcParams["figure.figsize"] = (15, 5)
grid2 = grid.sel(lat=-23, method='nearest')
grid2.values = grid2.values / -1000


new_model.sel(latitude = -23, method = 'nearest').VpVs.plot.contourf(yincrease = False, levels=np.arange(1.68, 1.88, 0.02), cmap = 'seismic', alpha = 0.5)
grid2.plot(xlim=[-78, -62], color='k')
plt.ylabel('z (km)')
plt.tight_layout()

# plt.plot([np.min(stlos), np.max(stlos)], [0, 0], color='cyan')
# plt.plot([np.min(stlos), np.max(stlos)], [0, 0], color='cyan', marker='|')

new_model.to_netcdf('../DATA/MAPPING/nc_models/CentralAndes_Zoned_VpVs_Arc-1.85.nc', encoding = model_encoding)




### Smooth the model


model = xr.open_dataset('../DATA/MAPPING/nc_models/CentralAndes_Zoned_VpVs_Arc-1.85.nc')
#
model_smoothed = model.copy()


data_smooth = savgol_filter(model.VpVs.values, 20, 2, mode = 'nearest', axis = 0)
data_smooth = savgol_filter(data_smooth, 20, 1, mode = 'nearest', axis = 1)
data_smooth = savgol_filter(data_smooth, 20, 1, mode = 'nearest', axis = 2)



# plt.figure()
# plt.pcolormesh(model.VpVs.values, cmap = 'jet');
# plt.colorbar()
# plt.gca().invert_yaxis()







model_encoding = {'depth': {'dtype': 'float32', '_FillValue': None },
            'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'VpVs': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }

model_smoothed.VpVs.values = data_smooth

data_smooth = model_smoothed.sel(latitude = -23, method='nearest')
plt.figure()
plt.pcolormesh(data_smooth.VpVs.values, cmap = 'jet');
plt.colorbar()
plt.gca().invert_yaxis()


model_smoothed.to_netcdf('../DATA/MAPPING/nc_models/CentralAndes_Zoned_VpVs_Arc-1.85_Smoothed.nc', encoding = model_encoding)

