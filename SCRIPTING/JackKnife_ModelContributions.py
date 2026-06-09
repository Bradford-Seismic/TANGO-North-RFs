#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:18:57 2026

@author: jimbradford
"""


import os
import sys
import io
import time
import gc
import warnings


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import pygmt
import math
from math import radians, sin, cos, sqrt, atan2
import xarray as xr
import random


import obspy
import obspy.taup.taup_geo as taup
from obspy.taup import TauPyModel
from obspy.core import UTCDateTime as utc
from obspy.core import Stream, read
from obspy.geodetics import degrees2kilometers
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
import multiprocessing
# from sklearn.neighbors import KDTree

from scipy.stats import binned_statistic_2d
from scipy.interpolate import griddata
import concurrent.futures


import geopandas as gpd

warnings.simplefilter('ignore', category = UserWarning)
warnings.simplefilter('ignore', category = FutureWarning)
warnings.simplefilter('ignore', category = RuntimeWarning)



active_dir = './'
os.chdir(active_dir)


region = 'TANGO_North'

"""

     The following attempts to determine the degree of contribution from 
     datapoints in an interpolated slice of the CCP data to return a result of
     grid points that impact the CCP image, and the piercing point rays and stations
     that contributed to that data point
     
     
     Since the CCP grid for TANGO-North is too large for an efficient JackKnife 
     procedure for the entire grid (lreave-one-out, interpolate, record difference, n>10E6, ~33 hrs computation time with multiprocessessing), 
     We will instead test only random points across the gridspace, then interpolate the 
     results to find the average sphere of influence.
     
     

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

    return joints_dist, np.unique(line, axis = 0), np.unique(line_dist)



    

joints =  np.loadtxt('../DATA/MAPPING/TANGO_North_CCP_Line.txt', delimiter = ',')
joints_dist, line, line_dist = Return_Line_Elements(joints)

x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')

model = '../DATA/MAPPING/nc_models/CCP_TANGO_North_G-2.5_GDW_RE.nc'



#%% Run JackKnife Resampling Algorithm



# read the CCP model
with xr.open_dataset(model) as base_model:
    model_slice = base_model.interp(longitude = x, latitude = y, method = 'linear', kwargs={"fill_value": None})
    model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})
    
    # take only amplitude data, and set all nan data to 0
    model_slice = model_slice['Amplitude']
    model_slice.values[np.where(np.isnan(model_slice.values))] = 0



# read the same model, prepared for jackknife resampling
with xr.open_dataset(model) as jackknife_model:
    
    Squared_Dif = np.empty(jackknife_model['Amplitude'].shape).reshape((-1,1))
    
    ## 1) define sizes of model
    i_size, j_size, k_size = jackknife_model['Amplitude'].shape
    total_size =  i_size * j_size * k_size



## Create list of random integers within total_size, note that since the domain
## is three dimensional, points will be evaluated across lat/lon/depth
## it will be necessary to collapse dimensions in the following kriging procedure

n_check = np.sort(np.random.choice(total_size, 100000, replace = False))




    
## 2) in loop, change i,j,k, iteration of datapoints to nan, 
## then, interpolate grid along joint coordinates
## calculate sum squared error and store in coordinate location
    
def JackKnife(n):
    
    
    with xr.open_dataset(model) as jackknife_model:
        
        squared_dif = np.nan


        print("\r", end="")
        print("Calculating Squared Differences Matrix: {:.4%} ".format(n/total_size), end="")
    
    
        jackknife_model_vals = np.reshape(jackknife_model['Amplitude'].values.copy(), (-1,1))
        
        # only continue if the contained amplitude data is not already nan
        if not np.isnan(jackknife_model_vals[n]):
            
            jackknife_model_vals[n] = np.nan
            jackknife_model['Amplitude'].values = np.reshape(jackknife_model_vals, jackknife_model['Amplitude'].shape)
            
        
            jackknife_slice = jackknife_model.interp(longitude = x, latitude = y, method = 'linear', kwargs={"fill_value": None})
            jackknife_slice = jackknife_slice.assign_coords({'Distance_Along_Trend': line_dist})
            
            jackknife_slice = jackknife_slice['Amplitude']
            jackknife_slice.values[np.where(np.isnan(jackknife_slice.values))] = 0
    
            
            # calculate sum of squared residuals against CCP model
            squared_dif = np.sum((model_slice.values - jackknife_slice.values)**2)
        
    return n, squared_dif

    



n_size = np.arange(0,total_size,1)
with concurrent.futures.ProcessPoolExecutor(max_workers = 28, mp_context=multiprocessing.get_context('fork')) as executor:
    summed_squares = executor.map(JackKnife, n_check)

    for trial in summed_squares:
        Squared_Dif[trial[0]] = trial[1]  
    




result = np.reshape( Squared_Dif, base_model['Amplitude'].shape)

data_description = "Sum of Squared Errors between linearly interpolated datasets with JackKnife comparisons along line ABCD"
export_result = xr.Dataset(data_vars = {'Squared_Error': (['depth',  'latitude', 'longitude'], result)},
                          coords = {'depth': base_model.depth.values, 'longitude': base_model.longitude.values, 'latitude': base_model.latitude.values},
                          attrs={'Comparison': data_description,
                                 'model': 'CCP_TANGO_South_G-5.0_FZB_CSEM2-Hicks2014-vp_Hick2014-Hk-vpvs.nc',
                                 })

model_encoding = {'depth': {'dtype': 'float32', '_FillValue': None },
            'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'Squared_Error': {'dtype': 'float32', '_FillValue': None , 'zlib': False},
            }




export_result.to_netcdf('../DATA/MAPPING/nc_models/JackKnife_Comparison_AB__CCP_TANGO_North_G-2.5_GDW_RE.nc', encoding = model_encoding)


sys.exit('Model JackKnife Completed')




#%% Examing grid point contributions and rays

# Since there is no spatial variance in sampling (binning) size, we treat the relative contribution of 
# grid points to the interpolated line as constant with depth (This is not necessarily true, but
# is a computational choice)
# We therefore only want to understand the average buffer zone of effect around the interpolation
# line. So, we will:
#   (1) read in the 3D JackKnife model
#   (2) average the values of Squared_Distance across depth
#   (3) Interpolate the gridpoints with values > 0 (assuming a mean of 0), into the entire CCP grid space
#   (4) evaluate ray contributions to the line



## Read JackKnife Model, and averange along depth dimension,
## note that this ignores nan values, which we will have to reinput later from
## the CCP model

with xr.open_dataset(model) as base_model:
    pass

with xr.open_dataset('../DATA/MAPPING/nc_models/JackKnife_Comparison_AB__CCP_TANGO_North_G-2.5_GDW_RE.nc') as jackknife:
    jackknife_flat = jackknife.mean(dim = 'depth', skipna = True)
 
    jackknife_choice = jackknife_flat.Squared_Error.where(jackknife_flat.Squared_Error > 0.001)
    
    # pull coordinate values where jackknife Squared_Error is greater than threshold
    da_nans=jackknife_flat.where(jackknife_flat.Squared_Error > 0.001)
    da_stacked=da_nans.Squared_Error.stack(z = ('latitude','longitude'))
    nona_coords = da_stacked[da_stacked.notnull()].z.values
    nona_coords = np.array(list(map(np.array, nona_coords)))
    nona_values = da_stacked[da_stacked.notnull()].values
    





LON, LAT = np.meshgrid(base_model.longitude.values, base_model.latitude.values)

interp_grid = griddata( np.flip(nona_coords, axis = 1), nona_values, (LON, LAT), method = 'linear', fill_value = 0)

interp_da = xr.DataArray(
        interp_grid,
        coords={"longitude": base_model.longitude.values, "latitude": base_model.latitude.values},
        dims=["latitude", "longitude"],
        name="Squared_Error"
    )


    





# plot the results

stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_FinalStations_RE.csv'.format(region))
stat_table['Code'] = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]


gmt_region = [-71.5, -63, -25, -21.4]
west, east, south, north = gmt_region

fig_x = 20

jackknife_limit = 0.0001


# for slice_depth in np.flip(np.arange(-120, 0, 10)):
fig = pygmt.Figure()    # initialize the main map
with pygmt.config(FONT = '14p', MAP_FRAME_TYPE = 'plain', FORMAT_GEO_MAP = 'ddd.xx'):

    prj = 'M{}c'.format(fig_x)            # set the map projection
    fig.basemap(region=gmt_region, projection=prj, frame=['xafg+lLongitude', 'yafg+lLatitude', 'wSnE'])
   
    # overlay national borders, coastlines, and make the water blue
    fig.coast(borders = ["1/1.5p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

    tecto_plates = gpd.read_file('../DATA/MAPPING/PlateBoundaries_Nazca.shp')
    fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = gmt_region, pen = '1.5p,black', style = "f1c/0.15c+r+t", fill = 'black')

    fig.plot(data='../DATA/MAPPING/APVC2.shp', region = gmt_region, pen = '0.75p,black', fill = 'pink', transparency = 50)
    fig.plot(data='../DATA/MAPPING/Calderas.shp', region = gmt_region, pen = '0.75p,black', fill = 'lightred', transparency = 50)


    grid_color = pygmt.makecpt(cmap = 'jet',   series = [0, 10], continuous = True)
    
    jackknife_slice = interp_da.copy()
    jackknife_slice.values = np.clip(jackknife_slice.values, 0.000001, 10)
    # fig.grdimage(grid=jackknife_slice.Squared_Error, cmap = True)
    fig.grdcontour(grid=jackknife_slice, annotation = [jackknife_limit])


    # now plot all station data
    fig.plot(x = stat_table.stlo, y = stat_table.stla, style='i0.3c', fill = 'gold',  pen='0.4p,black')
    fig.plot(x = stat_table[stat_table.net == 'XN'].stlo, y =  stat_table[stat_table.net == 'XN'].stla, style='i0.25c',fill = 'cyan', pen='black')
    fig.plot(x = stat_table[stat_table.net == 'XM'].stlo, y =  stat_table[stat_table.net == 'XM'].stla, style='i0.3c',fill = 'magenta', pen='0.6p,black')

    joints =  np.loadtxt('../DATA/MAPPING/TANGO_North_CCP_Line.txt', delimiter = ',')
    fig.plot(x = line[:,0], y = line[:,1], pen = '1.2p,black,--')
    Numerals = "AB"
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')

    fig.basemap(map_scale="jTR+o0.75c/0.25c+w100k+u")
    fig.text(x = -72.8, y = -36.85, text = 'Contributing Field', fill = 'white', pen = '1p,black')



fig.show()
    



    
# cycle through depths, pull grid point values that have squared error values
# greater than 'jackknife_limit'. These should be pulled from the CCP model so 
# that we pull the binning_radius data


with xr.open_dataset('../DATA/MAPPING/nc_models/CCP_TANGO_North_G-2.5_GDW_RE.nc') as model:
    df = (
        model.Bin_Radius.where(interp_da > jackknife_limit)
          .to_dataframe(name="bin_radius")
          .dropna()
          .reset_index()
    )
    

# read piercing point data
pierce_points = pd.read_csv('../DATA/MAPPING/PiercePoints_RE.csv')
pierce_points = pierce_points.dropna()
pierce_points = pierce_points.reset_index() # store original index information

depths = model.depth.values




contributing_stations = [] # stores values of station code strings
contributing_pierce= [] # stores indexes of contributing pierce points in 'pierce_points' dataframe
contributing_events = [] # stores origin-time IDs of events 
for k, z in enumerate(depths):
    
    print("\r", end="")
    print("Gathering Contributing Data: {:.2%} ".format(k / len(depths)), end="")

    # select data by depth
    df_depth = df[df.depth == z]
    if len(df_depth) == 0:
        continue
    
    pierce_depth = pierce_points[pierce_points.depth == -z].reset_index(drop = True)
    pierce_depth = pierce_depth.dropna().reset_index(drop = True)
    
    
    # make KDTree
    grid_tree = KDTree(np.c_[df_depth.longitude, df_depth.latitude])
    
    # query for nearby pierce points, boolean index cooresponds to whether pierce point is in range of grid space
    data_in_range = grid_tree.query_ball_point(x = np.c_[pierce_depth.longitude, pierce_depth.latitude], r = 24 / degrees2kilometers(1))
    data_in_range =[bool(data_in_range[j]) for j in range(len(data_in_range))]

    pierce_in_range = pierce_depth[data_in_range].reset_index(drop = True)
   
    for i in range(len(pierce_in_range)):
        if pierce_in_range.station[i] not in contributing_stations:
            contributing_stations.append(pierce_in_range.station[i])
        if pierce_in_range.event[i] not in contributing_events:
            contributing_events.append(pierce_in_range.event[i])
        if int(pierce_in_range['index'][i]) not in contributing_pierce:
            contributing_pierce.append(int(pierce_in_range['index'][i]))
    

pd.DataFrame({'Code': contributing_stations}).to_csv('../DATA/MAPPING/Contributing_Stations_RE.csv', index = False)
pd.DataFrame({'Event': contributing_events}).to_csv('../DATA/MAPPING/Contributing_events_RE.csv', index = False)
pd.DataFrame({'Index': contributing_pierce}).to_csv('../DATA/MAPPING/Contributing_Pierce_RE.csv', index = False)





#%% Plot Contributing Data

contributing_stations = pd.read_csv('../DATA/MAPPING/Contributing_Stations_RE.csv')
contributing_pierce = pd.read_csv('../DATA/MAPPING/Contributing_Pierce_RE.csv')

stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_FinalStations_RE.csv'.format(region))
stat_table['Code'] = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]

pierce_points = pd.read_csv('../DATA/MAPPING/PiercePoints_RE.csv')


stat_table = stat_table[stat_table.Code.isin(contributing_stations.Code)].reset_index(drop = True)
pierce_points = pierce_points.iloc[contributing_pierce.Index].reset_index(drop = True)


gmt_region = [-71.5, -63, -25, -21.4]
west, east, south, north = gmt_region

fig_x = 20


slice_depth = 200

fig = pygmt.Figure()    # initialize the main map
with pygmt.config(FONT = '14p', MAP_FRAME_TYPE = 'plain', FORMAT_GEO_MAP = 'ddd.xx'):

    prj = 'M{}c'.format(fig_x)            # set the map projection
    fig.basemap(region=gmt_region, projection=prj, frame=['xafg+lLongitude', 'yafg+lLatitude', 'wSnE'])
   
    # overlay national borders, coastlines, and make the water blue
    fig.coast(borders = ["1/1.5p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

    tecto_plates = gpd.read_file('../DATA/MAPPING/PlateBoundaries_Nazca.shp')
    fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = gmt_region, pen = '1.5p,black', style = "f1c/0.15c+r+t", fill = 'black')


    fig.plot(x = pierce_points.longitude[pierce_points.depth == slice_depth], y = pierce_points.latitude[pierce_points.depth  == slice_depth], style = 'x0.1c')
    

    # now plot all station data
    fig.plot(x = stat_table.stlo, y = stat_table.stla, style='i0.3c', fill = 'gold',  pen='0.4p,black')
    fig.plot(x = stat_table[stat_table.net == 'XN'].stlo, y =  stat_table[stat_table.net == 'XN'].stla, style='i0.25c',fill = 'cyan', pen='black')
    fig.plot(x = stat_table[stat_table.net == 'XM'].stlo, y =  stat_table[stat_table.net == 'XM'].stla, style='i0.3c',fill = 'magenta', pen='0.6p,black')

    joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')
    Numerals = "ABCDEF"
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')

    fig.basemap(map_scale="jTR+o0.75c/0.25c+w100k+u")



fig.show()




