# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:52:23 2026

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

from pyproj import CRS, Transformer


warnings.simplefilter('ignore', category = UserWarning)
warnings.simplefilter('ignore', category = FutureWarning)


region = 'TANGO_North'




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

    return joints_dist, np.unique(line, axis = 0), np.unique(line_dist)

def plot_topo(fig):
    
    fig.basemap(region = [0, joints_dist[-1], 0,8], projection = 'X{}c/{}c'.format(fig_x, 1.5), frame = ['wsNE', 'xa100f10', 'ya5f1+e+lElev. (km)'])

    topo_x = np.append(line_dist[0], line_dist)
    topo_x = np.append(topo_x, line_dist[-1])


    # on-line topography

    dem = grid.interp(lat = y, lon = x, method = 'linear')
    dem = dem.assign_coords({'Distance_Along_Trend': line_dist})

    topo = np.append(0, dem.values)
    topo = np.append(topo, 0)


    fig.plot(x = topo_x, y = topo/1000, fill = 'lightgray', pen = '0.5p,black')

    # project tectonic zones
    
    tecto_zones = gpd.read_file('../DATA/MAPPING/AndeanTectonicRegimes2.shp')

    try:
        int_dists = []
        fig.plot(x = [30, 30], y = [4, 6], pen = '1p,black')
        fig.plot(x = [joints_dist[-1], joints_dist[-1]], y = [4, 6], pen = '1p,black')

        used_areas = []
        for j in range(len(joints)-1):
            
            line_gpd = gpd.GeoSeries(data = shapely.geometry.LineString([joints[j], joints[j+1]]), crs = tecto_zones.crs)

            int_dists = []
            areas = ['LV', 'CD', 'SdA', 'WC', 'Puna', 'EC', 'SB']

            for area in areas:
                try:
                    if area in used_areas:
                        continue

                    else:
                        poly_int = tecto_zones[tecto_zones.Zone == area].union_all().intersection(line_gpd)
                        poly_int = poly_int.get_coordinates().to_numpy()[0,:]


                        int_dist = haversine(joints[j,1], joints[j,0], poly_int[1], poly_int[0]) + joints_dist[j]
                        int_elev = dem.values[closest_idx(dem['Distance_Along_Trend'], int_dist)]
                        int_dists.append(int_dist)

                        fig.plot(x = [int_dist, int_dist], y = [4, 6], pen = '1p,black')
                        fig.plot(x = [int_dist, int_dist], y = [4, 6], pen = '1p,black')

                        used_areas.append(area)


                except:
                      pass
        fig.plot(x = [30, joints_dist[-1]], y = [6, 6], pen = '1p,black')

    except:
        pass


    # project APVC

    APVC = gpd.read_file('../DATA/MAPPING/APVC2.shp')
    line_gpd = gpd.GeoSeries(data = shapely.geometry.LineString([joints[0], joints[1]]), crs = APVC.crs)


    poly_int = APVC.union_all().intersection(line_gpd)
    poly_int = poly_int.get_coordinates().to_numpy()
    
    within_mask = shapely.within(shapely.points(line[:,0], line[:,1]), APVC.union_all())
    APVC_x = line_dist.copy()
    
    groups = np.split(APVC_x, np.flatnonzero(np.diff(within_mask)) + 1)
    mask_groups = np.split(within_mask, np.flatnonzero(np.diff(within_mask)) + 1)  
    true_groups = [g for g, m in zip(groups, mask_groups) if m[0]]    
    
    for sub_group_x in true_groups:
        sub_group_y = dem.interp(Distance_Along_Trend = sub_group_x, method = 'linear').copy().values/1000
        sub_group_x = np.append(sub_group_x[0], sub_group_x); sub_group_x = np.append(sub_group_x, sub_group_x[-1])
        sub_group_y = np.append(0, sub_group_y); sub_group_y = np.append(sub_group_y, 0)

    
        fig.plot(x = sub_group_x, y = sub_group_y,  fill = 'pink')




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
                fig.plot(x=track.p[track.code.str.contains('ZR')] + joints_dist[i],
                         y=stat_elev_interp.values[track.code.str.contains('ZR')]/1000, style='i0.3c', fill='lightred', pen='0.3,black')
            except:
                pass

            try:
                fig.plot(x=track.p[track.code.str.contains('XM')] + joints_dist[i], y=stat_elev_interp.values[track.code.str.contains(
                    'XM')]/1000, style='i0.3c', fill='magenta', pen='0.3,black')

            except:
                pass


    # Plot joints
    fig.text(x = joints_dist, y = np.ones(len(joints_dist)) * 7, text = list('AB'), font = '14p, Helvetica-Bold', fill = 'white', pen = '0.3p,black', no_clip = True)


        
        
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
                         style='kvolcano/0.5c', fill='red', pen='0.5,black')
    except:
        pass




    return fig




###### Read station objects and show only stations contributing to line

stations = pd.read_csv('../DATA/SPREADSHEETS/{}_FinalStations_RE.csv'.format(region))

stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_FinalStations_RE.csv'.format(region))
codes = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]
stat_table['Code'] = codes



contributing_stations = pd.read_csv('../DATA/MAPPING/Contributing_Stations_RE.csv')
stat_table = stat_table[stat_table['Code'].isin(contributing_stations.Code.to_list())]
stat_table.reset_index(drop = True, inplace = True)




gmt_region = [-71.5, -63, -25, -21.4]
west, east, south, north = gmt_region
joints =  np.loadtxt('../DATA/MAPPING/{}_CCP_Line.txt'.format(region), delimiter = ',')
joints_dist, line, line_dist = Return_Line_Elements(joints)

x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')

grid = pygmt.datasets.load_earth_relief(resolution="15s", region=[west, east, south, north])





fig_x = 15

fig = pygmt.Figure()
with pygmt.config(FONT = '12p', MAP_FRAME_TYPE = 'plain'):
    
    fig = plot_topo(fig)
    
    fig.shift_origin(yshift = '-7c')
    
    prj = 'M{}c'.format(fig_x)            # set the map projection
    fig.basemap(region=gmt_region, projection=prj, frame=['wESn', 'xa2f1g2+lLongitude', 'ya1f1g2+lLatitude'])

    # overlay the grid image DEM over the basemap
    shade = pygmt.grdgradient(grid=grid, azimuth="0/90", normalize="e1")
    elev_cmap = pygmt.makecpt(cmap = '../DATA/MAPPING/natural_mod.cpt', series=  [0, 6000, 1000])

    fig.grdimage(grid=grid, shading = shade, cmap = True, projection = prj, transparency = 65)

    # overlay national borders, coastlines, and make the water blue
    fig.coast(borders = ["1/1.5p,SlateGray"], shorelines="1/0.5p", water = "skyblue", transparency = 10)
    

    # Plot geographical markers
    tecto_plates = gpd.read_file('../DATA/MAPPING/PlateBoundaries_Nazca.shp')
    fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = gmt_region, pen = '1.5p,black', style = "f1c/0.3c+r+t", fill = 'black')
    fig.plot(data='../DATA/MAPPING/APVC2.shp', region = gmt_region, pen = '0.75p,black', fill = 'pink', transparency = 50)
    fig.plot(data='../DATA/MAPPING/Calderas.shp', region = gmt_region, pen = '0.75p,black', fill = 'lightred', transparency = 50)
    # fig.plot(data='../DATA/MAPPING/Lavayen_Valley.shp', region = gmt_region, pen = '1p,black,.-')


    Chile_Faults = gpd.read_file('../DATA/MAPPING/CHAF_Pangea_v1.shp')
    fig.plot(data = Chile_Faults[Chile_Faults.F_system == 'ATFS'], pen = '.5p,black')
    fig.plot(data = Chile_Faults[Chile_Faults.F_system == 'CdSF'], pen = '.5p,black')
    fig.plot(data = Chile_Faults[Chile_Faults.F_system == 'SdAF'], pen = '.5p,black')


    tecto_zones = gpd.read_file('../DATA/MAPPING/AndeanTectonicRegimes2.shp')
    areas = ['Altiplano', 'WC','EC', 'SB', 'SA','LV', 'SdA' ]
    tecto_zones = tecto_zones[tecto_zones.Zone.isin(areas)]
    tecto_zones = tecto_zones.dissolve(by = 'Zone')
    # fig.plot(data = tecto_zones, pen = '1p,white')
    fig.plot(data = tecto_zones, pen = '1p,black,6_6:6p')

    

    # Plot volcanoes
    
    # volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')
    # fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.4c', fill = 'orange', pen='0.75p,black')

    volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.4c', fill = 'red', pen='0.75p,black')



    # now plot station data
    fig.plot(x = stat_table.stlo[(stat_table.net != 'XM') | (stat_table.net != 'XN')], y = stat_table.stla[(stat_table.net != 'XM') | (stat_table.net != 'XN')], style='i0.23c', fill = 'gold',  pen='0.4p,black')
    
    try:
        fig.plot(x = stations[stations.net == 'XN'].stlo, y = stations[stations.net == 'XN'].stla, style='i0.25c',fill = 'cyan', pen='black', transparency = 75)
        fig.plot(x = stat_table[stat_table.net == 'XN'].stlo, y =  stat_table[stat_table.net == 'XN'].stla, style='i0.25c',fill = 'cyan', pen='black')
    except:
        pass
   
    try:
        fig.plot(x = stations[stations.net == 'XM'].stlo, y = stations[stations.net == 'XM'].stla, style='i0.3c',fill = 'magenta', pen='0.6p,black', transparency = 75)
        fig.plot(x = stat_table[stat_table.net == 'XM'].stlo, y =  stat_table[stat_table.net == 'XM'].stla, style='i0.3c',fill = 'magenta', pen='0.6p,black')
    except:
        pass
     



    # plot joints
    sub_joints =  np.array([[-65.5816, -24.1358], [-64.8716, -24.3922]]) # Sub-line for north array

    fig.plot(x = line[:,0], y = line[:,1], pen = '1.2p,black,')
    
    Numerals = "ABCDEFGHIJKLMNOP"
    # Numerals = ["A", "A'"]
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')
    

    
    fig.plot(x = sub_joints[:,0], y = sub_joints[:,1], pen = '1.2p,black,')
    fig.text(x = sub_joints[0,0], y = sub_joints[0,1], text = '1',  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')
    fig.text(x = sub_joints[1,0], y = sub_joints[1,1], text ="2",  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')
    
    
    
    # Plot cities
    cities = pd.read_csv('../DATA/MAPPING/Cities.csv')
    fig.plot(x = cities.Longitude, y = cities.Latitude, style = 'd0.4c', fill = 'darkorange', pen = '1p,black')




    # Plot inset map
    with fig.inset(position="jBL+w8c/12c+o-3.5c/-2c", box = False):
        
        # fig.basemap(frame = 'wsne')
        fig.coast(
           region=[-85, -30, -60, 15],
           projection="U19S/?",
           dcw= ["AR+gcoral+p0.2p",
                 "CL+gcoral+p0.2p",
                 "PE+gcoral+p0.2p",
                 "EC+gcoral+p0.2p",
                 "CO+gcoral+p0.2p",
                 "BO+gcoral+p0.2p"],
           area_thresh=10000,)
            
        
        # plot the general region we are focused on
        fig.plot(x = [west, west, east, east, west], y = [north, south, south, north, north], pen = '2p,blue')
        fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = [-85, -30, -60, 15], pen = '1p,black', style = "f0.75c/0.2c+r+t", fill = 'black')


spec = io.StringIO(
"""   
N 4
S 0.20c i 0.3c cyan 0.4p,black 0.65c TANGO Node
S 0.20c i 0.3c magenta 0.6p,black 0.65c TANGO Broadband
S 0.20c i 0.3c gold 0.6p,black 0.65c Other Stations
S 0.15c d 0.4c darkorange 1p,black 0.65c Major City

S 0.20c kvolcano 0.40c red 0.75p,black 0.65c Holocene Volcano
# S 0.20c kvolcano 0.40c orange 0.75p,black 0.65c Pleistocene Volcano
S 0.20c f+l+t 0.45c/-1/0.15c black 1.5,black 0.65c Chile Trench

S 0.05c - .5c - 1p,- 0.65c Tectonic Boundaries
S 0.20c - .5c - 1p,gray,- 0.65c Country Border
S 0.20c r 0.4/0.25 pink 0.8p,black 0.65c APVC
S 0.20c r 0.4/0.25 lightred 0.8p,black 0.65c Caldera


G 0.07c
G 0.07c

"""
)


with pygmt.config(FONT = '10p'):
    fig.legend(spec = spec,  position='jBL+w17c+o0c/-2.5c')

    
fig.show()



stat_fig_name = 'F1_Stations+Tectonics_Region-{}.png'.format(region)
fig.savefig(f'../FIGURES/{stat_fig_name}', dpi = 600)

