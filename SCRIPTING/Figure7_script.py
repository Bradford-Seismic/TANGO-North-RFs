# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 15:38:49 2025

@author: 7418888
"""

import numpy as np
import pandas as pd
import os
import sys
import obspy
import matplotlib.pyplot as plt
from obspy.core import UTCDateTime as utc
from obspy.core import Stream, read
from obspy import read_inventory
from obspy.io.sac import SACTrace
from obspy.signal.trigger import recursive_sta_lta
import obspy.taup.taup_geo as taup
from obspy.taup import TauPyModel
from glob import glob
import warnings
import matplotlib.dates as dates
import pygmt
import subprocess
import math
import geopandas as gpd
import shapely
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from math import radians, sin, cos, sqrt, atan2
import io
import time
from shapely.ops import unary_union
from scipy.spatial import KDTree
from obspy.geodetics import degrees2kilometers


from pyproj import CRS, Transformer


warnings.simplefilter('ignore', category = UserWarning)
warnings.simplefilter('ignore', category = FutureWarning)


active_dir = './'



os.chdir(active_dir)
model = TauPyModel(model="iasp91")



stations = pd.read_csv('../DATA/SPREADSHEETS/SA_Stations.csv')
moveouts = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')
snr = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')
itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary_RE.csv', keep_default_na=False)

region = 'TANGO_North'

stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_FinalStations_RE.csv'.format(region))
stat_table['Code'] = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]

contributing_stations = pd.read_csv('../DATA/SPREADSHEETS/Contributing_Stations.csv')

stat_table = stat_table[stat_table.Code.isin(contributing_stations.Code)].reset_index(drop = True)



joints =  np.loadtxt('../DATA/MAPPING/{}_CCP_Line.txt'.format(region), delimiter = ',')

def closest_idx(lst, K):
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return idx


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

joints_dist, line, line_dist = Return_Line_Elements(joints)



gmt_region = [-71.5, -63, -25, -21.4]
west, east, south, north = gmt_region


grid = pygmt.datasets.load_earth_relief(resolution="15s", region=gmt_region)

fig_x = 20
depth = 70
dist_min = 500
dist_lim = 800
t_depth = 15
aspect = 3

ratio = dist_lim / fig_x # km per figure cm
fig_y = depth/ratio * aspect


x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')


fig = pygmt.Figure()
with pygmt.config(FONT = '14p', MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):
    fig.basemap(region = [dist_min, dist_lim, -depth, 0], projection = 'X{}c/{}c'.format(fig_x, fig_y),frame = ['WSne', 'xafg+lProfile Distance (km)', 'ya20f10g+lDepth (km)'])
   
    model = xr.open_dataset('../DATA/MAPPING/nc_models/CCP_TANGO_North_G-2.5_GDW_RE.nc')
    model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})    
    
    model_slice.Amplitude.values = np.clip(model_slice.Amplitude.values, -0.5, 0.5)

    grid_color = pygmt.makecpt(cmap = '../DATA/MAPPING/no_green.cpt',   series = [-0.5, 0.5])
    fig.grdimage(grid = model_slice['Amplitude'], cmap = True, transparency = 0)
    
    
    with pygmt.config(FONT_ANNOT_PRIMARY="20p", FONT_LABEL = '20p'):
        fig.colorbar(cmap = True, position = 'JMR+w{}+o0.25c/0c'.format(fig_y), frame = 'xa0.5f0.1+lAmplitude')


    spec = io.StringIO(
"""   
N 4
S 0.40c i 0.3c magenta 0.2p,black .75c TANGO Broadband
S 0.40c i 0.3c cyan 0.4p,black .75c TANGO Node



G 0.07c
G 0.07c

"""
    )
    fig.legend(spec = spec,  position='jBL+w{}c+o0c/-3c'.format(fig_x+2))



    fig.shift_origin(yshift = '+{}c'.format(fig_y))
    fig.basemap(region = [dist_min, dist_lim, 0,7], projection = 'X{}c/{}c'.format(fig_x, 1.5), frame = ['WsNe', 'xa100f10', 'ya5f1+e+lElev. (km)'])

    topo_x = np.append(line_dist[0], line_dist)
    topo_x = np.append(topo_x, line_dist[-1])


    # on-line topography

    dem = grid.interp(lat = y, lon = x, method = 'linear')
    dem = dem.assign_coords({'Distance_Along_Trend': line_dist})

    topo = np.append(0, dem.values)
    topo = np.append(topo, 0)


    fig.plot(x = topo_x, y = topo/1000, fill = 'lightgray', pen = '0.5p,black')



    # project volcanoes
    try:
        volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
        volc_data = pd.DataFrame({'longitude': volcanoes.Longitude.values, 'latitude': volcanoes.Latitude.values,
                                 'elevation': volcanoes['Elevation (m)'].values, 'volc_id': volcanoes['Volcano Number'].values})
        used_data = []
        for i in range(len(joints) - 1):

            # note the little buffer of on the 'length' parameter, for some reason, some lines if the vertex is
            # right on the point, it won't project
            track = pygmt.project(data=volc_data, center=joints[i], endpoint=joints[i+1], width=[-24, 24], length=[
                                  0, joints_dist[i+1] - joints_dist[i]], unit=True)
            track = track.rename(columns={
                                 0: 'longitude', 1: 'latitude', 2: 'elevation', 3: 'volc_id', 4: 'p', 5: 'q', 6: 'r', 7: 's'})

            if len(track) > 0:
                data_used_check = [track.volc_id[j]
                                   in used_data for j in range(len(track))]
                data_used_check = [not (bool(data_used_check[j]))
                                   for j in range(len(data_used_check))]
                track = track[data_used_check].reset_index(drop=True)
                used_data = np.append(
                    used_data, np.unique(track.volc_id.values))

                volc_elev_interp = grid.interp(lat=xr.DataArray(track.latitude, dims='Distance'), lon=xr.DataArray(
                    track.longitude, dims='Distance'), method='linear')

                fig.plot(x=track.p + joints_dist[i], y=volc_elev_interp.values/1000,
                         style='kvolcano/0.6c', fill='red', pen='0.3,black')
    except:
        pass

    # project APVC
    try:
        APVC = gpd.read_file('../DATA/MAPPING/APVC_shp.shp')
    
        line_gpd = gpd.GeoSeries(data = shapely.geometry.LineString([joints[0], joints[1]]), crs = APVC.crs)
    
        poly_int = APVC.intersection(line_gpd)
        poly_int = poly_int.get_coordinates().to_numpy()
            
        apvc_start = haversine(joints[0,1], joints[0,0], poly_int[0,1], poly_int[0,0]) 
        apvc_end = haversine(joints[0,1], joints[0,0], poly_int[1,1], poly_int[1,0]) 
            
        
        fig.plot(x = [apvc_start, apvc_start, apvc_end, apvc_end], y = [0, 3, 3, 0], fill = 'pink')
            
        ## Project Tectonic Zones
        
        tecto_zones = gpd.read_file('../DATA/MAPPING/AndeanTectonicRegimes2.shp')
    
        used_zones = []
        for j in range(len(joints)-1):
    
            line_gpd = gpd.GeoSeries(data = shapely.geometry.LineString([joints[j], joints[j+1]]), crs = tecto_zones.crs)
    
            int_dists = []
            areas = ['LV', 'CD', 'SdA', 'WC', 'Puna', 'EC', 'SB']
            for area in areas:
                try:
                    poly_int = tecto_zones[tecto_zones.Zone == area].union_all().intersection(line_gpd)
                    poly_int = poly_int.get_coordinates().to_numpy()[0,:]
    
                    int_dist = haversine(joints[j,1], joints[j,0], poly_int[1], poly_int[0]) + joints_dist[j]
                    int_elev = dem.values[closest_idx(dem['Distance_Along_Trend'], int_dist)]
                    int_dists.append(int_dist)
                    
                    fig.plot(x = [int_dist, int_dist], y = [0, int_elev/1000], pen = '1p,black,5_2')
    
    
                except:
                      pass
    except:
        pass


    # project stations
    codes = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i])
             for i in range(len(stat_table))]
    stat_table['code'] = codes

    stat_data = pd.DataFrame({'longitude': stat_table.stlo.values, 'latitude': stat_table.stla.values,
                             'elevation': stat_table.stel.values, 'code': stat_table.code.values})
    used_data = []
    for i in range(len(joints) - 1):
        track = pygmt.project(data=stat_data, center=joints[i], endpoint=joints[i+1], width=[
                              -24, 24], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
        track = track.rename(columns={
                             0: 'longitude', 1: 'latitude', 2: 'elevation', 3: 'p', 4: 'q', 5: 'r', 6: 's', 7: 'code'})

        if len(track) > 0:
            data_used_check = [track.code[j]
                               in used_data for j in range(len(track))]
            data_used_check = [not (bool(data_used_check[j]))
                               for j in range(len(data_used_check))]
            track = track[data_used_check].reset_index(drop=True)
            used_data = np.append(used_data, np.unique(track.code.values))

            stat_elev_interp = grid.interp(lat=xr.DataArray(track.latitude, dims='Distance'), lon=xr.DataArray(
                track.longitude, dims='Distance'), method='linear')

            fig.plot(x=track.p + joints_dist[i], y=stat_elev_interp.values /
                     1000, style='i0.3c', fill='gold', pen='0.3,black')


            try:
                fig.plot(x=track.p[track.code.str.contains('XN')] + joints_dist[i],
                         y=stat_elev_interp.values[track.code.str.contains('XN')]/1000, style='i0.3c', fill='cyan', pen='0.3,black')
            except:
                pass
            try:
                fig.plot(x=track.p[track.code.str.contains('XM')] + joints_dist[i], y=stat_elev_interp.values[track.code.str.contains(
                    'XM')]/1000, style='i0.3c', fill='magenta', pen='0.3,black')
            except:
                pass






fig.show()

fig.savefig('../FIGURES/Fig6_ForelandInterpretation.png', dpi = 300)