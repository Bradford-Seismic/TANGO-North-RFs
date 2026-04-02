# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 18:05:40 2025

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
from scipy.interpolate import interp1d



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


# depth, left dist, right dist
primaries = [[-6, 320, 390],
             [-3, 360, 385],
             [-14, 305, 350],
             [-11.5, 380, 410],
             [-5, 415, 435],
             [-9, 415, 435],
             [-17, 410, 540],
             [-7, 470, 510],
             [-12, 510, 540],
             [-10, 570, 600],
             [-15, 550, 570]]


time_primaries = [[0.6, 700, 735],
                  [0.8, 675, 700],
                  [1.15, 650, 675],
                  ]

p_ref = 0.064





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

def calc_multiple(H, vs_rms, vpvs_rms):
    qa = np.sqrt((1/vs_rms**2) - p_ref**2)
    qb = np.sqrt((1/(vs_rms*vpvs_rms)**2) - p_ref**2)
    
    tps = H * (qa - qb)     # arrival time of primary
    tppps = tps * ( (qa + qb) / (qa - qb) )     # arrival time of first mulitple

    return tppps


joints_dist, line, line_dist = Return_Line_Elements(joints)



gmt_region = [-71.5, -63, -25, -21.4]
west, east, south, north = gmt_region


grid = pygmt.datasets.load_earth_relief(resolution="15s", region=gmt_region)


x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')


shear_model = 'BlendedModel_0.7APVC-ANT_0.3FWT_MantleSpec_Smoothed.nc'
vpvs_model = 'CentralAndes_Zoned_VpVs_Arc-1.85_Smoothed.nc'

min_vs = 2.0
v_model = xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(shear_model))

min_vpvs = 1.72
max_vpvs = 1.82
vpvs_model = xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(vpvs_model))

fig = pygmt.Figure()
with pygmt.config(FONT = '14p', MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):
  
    # establish figure dimension
    fig_x = 20
    depth = 80
    t_depth = 15
    aspect = 1
 
    ratio = line_dist.max() / 20 # km per figure cm
    fig_y = depth/ratio*aspect
    
    total_y = fig_y
    
     
    fig.basemap(region = [300, 600, 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10g100', 'ya50f10g100+lDepth'])
    v_slice = v_model['vs'].interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})

    v_slice = v_slice.assign_coords({'Distance_Along_Trend': line_dist})
    v_slice.values = np.clip(v_slice.values, min_vs, v_slice.values.max())    
    pygmt.makecpt(cmap = 'jet', series = [min_vs, 6.0, 0.2], reverse = True)
    fig.grdimage(grid = v_slice, cmap = True)
    fig.grdcontour(grid = v_slice, levels =  np.arange(3.0, 5.2, 0.4), annotation = np.arange(3.0, 5.2, 0.4))

    with pygmt.config(FONT_ANNOT_PRIMARY="34p", FONT_LABEL = '34p'):
        fig.colorbar(cmap = True, position = 'JMR+w{}/0.4+o0.25c/0c'.format(fig_y), frame = 'xa2f1+lvs (km/s)')


    # shift origin up to plot topography
    
    fig.shift_origin(yshift = '+{}c'.format(fig_y))
    total_y = total_y + 1.5
    
    fig.basemap(region = [300, 600, 0, 7], projection = 'X{}c/{}c'.format(fig_x, 1.5), frame = ['WsNe', 'xa100f10', 'ya5f1+e+lElev'])
    


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
        APVC = gpd.read_file('../DATA/MAPPING/APVC2.shp')
    
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



    # shift origin back down to base of plot
    fig.shift_origin(yshift = '-{}c'.format(fig_y))


    # shift down to vp/vs plot
    
    fig.shift_origin(yshift = '-{}c'.format(fig_y+0.5))
    shifted_y = fig_y + 0.5
    total_y = total_y + fig_y + 0.5

    fig.basemap(region = [300, 600, 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10g100+lProfile Distance (km)', 'ya50f10g100+lDepth'])


    v_slice = vpvs_model['VpVs'].interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})

    v_slice = v_slice.assign_coords({'Distance_Along_Trend': line_dist})
    v_slice.values = np.clip(v_slice.values, min_vpvs, max_vpvs)    

    pygmt.makecpt(cmap = 'jet', series = [min_vpvs, max_vpvs, 0.008], reverse = True)
    fig.grdimage(grid = v_slice, cmap = True)
    fig.grdcontour(grid = v_slice, levels =  [1.74, 1.76, 1.78, 1.82], annotation = [1.74, 1.76, 1.78, 1.82])

    with pygmt.config(FONT_ANNOT_PRIMARY="34p", FONT_LABEL = '34p'):
        fig.colorbar(cmap = True, position = 'JMR+w{}/0.4+o0.25c/0c'.format(fig_y), frame = 'xa0.04f0.02+lvp/vs')

    

    # shift down to CCP plot 

    fig.shift_origin(yshift = '-{}c'.format(fig_y*3+0.5))
  
    shifted_y = shifted_y + fig_y*3 + 0.5
    total_y = total_y + fig_y*3 + 0.5


    fig_x = 20
    depth = 80
    aspect = 3
    
    ratio = line_dist.max() / 20 # km per figure cm
    fig_y = depth/ratio*aspect


    
    fig.basemap(region = [300, 600, -depth, 0], projection = 'X{}c/{}c'.format(fig_x, fig_y), frame = ['WSen', 'xa50f10g75+lProfile Distance (km)', 'ya20f10g75+lDepth (km)'])

    model = xr.open_dataset('../DATA/MAPPING/nc_models/CCP_TANGO_North_G-5.0_GDW_RE.nc')
    model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})
    model_slice.Amplitude.values = np.clip(model_slice.Amplitude.values, -0.25, 0.25)

    vs = xr.open_dataset('../DATA/MAPPING/nc_models/BlendedModel_0.7APVC-ANT_0.3FWT_MantleSpec_Smoothed.nc')
    vs_slice = vs.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    vs_slice = vs_slice.assign_coords({'Distance_Along_Trend': line_dist})
    vs_slice.vs.values = np.clip(vs_slice.vs.values, 2, 8)

    vpvs = xr.open_dataset('../DATA/MAPPING/nc_models/CentralAndes_Zoned_VpVs_Arc-1.85_Smoothed.nc')
    vpvs_slice = vpvs.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    vpvs_slice = vpvs_slice.assign_coords({'Distance_Along_Trend': line_dist})
    vpvs_slice.VpVs.values = np.clip(vpvs_slice.VpVs.values, 1.72, 1.80)

    dem = grid.interp(lat = y, lon = x, method = 'linear')
    dem = dem.assign_coords({'Distance_Along_Trend': line_dist})


    grid_color = pygmt.makecpt(cmap = '../DATA/MAPPING/no_green.cpt',   series = [-0.25, 0.25])
    fig.grdimage(grid = model_slice['Amplitude'], cmap = True)
    
    
    for primary in primaries:
        
        x_len = np.arange(primary[1], primary[2], 1)
        fig.plot(x = x_len, y = np.ones(x_len.shape) * primary[0], pen = '3p,white,--')
       
        vs_interp = vs_slice.interp(Distance_Along_Trend = x_len, depth = np.arange(0, primary[0] * -1, 0.5), method = 'nearest')
        vpvs_interp = vpvs_slice.interp(Distance_Along_Trend = x_len, depth = np.arange(0, primary[0] * -1, 0.5), method = 'nearest')
        

        topo = dem.interp(Distance_Along_Trend = np.arange(primary[1], primary[2], 1), method = 'nearest').values/1000
        vs_rms = np.sqrt(np.mean(vs_interp.vs.values**2, axis = 0)) # bulk velocity above chosen interface at depth
        vpvs_rms = np.sqrt(np.mean(vpvs_interp.VpVs.values**2, axis = 0))      # bulk vpvs above chosen interface
        H = topo + -primary[0] # thickness of crust to primary conversion selection
        
        tppps = calc_multiple(H, vs_rms, vpvs_rms) # estimated arrival time to tppps multiple
        tppps_up = calc_multiple(H, vs_rms + vs_rms*0.1, vpvs_rms) # estimated arrival time to tppps multiple
        tppps_down = calc_multiple(H, vs_rms - vs_rms*0.1, vpvs_rms) # estimated arrival time to tppps multiple

        
        
        # propagate the estimated multiple times as they would be in the CCP migration
        
        
        se = np.round(topo / 0.5) * 0.5
        z = np.arange(-se.min(),140,0.5)
        vd = np.ones((len(z), len(x_len)))
        kd = np.ones((len(z), len(x_len)))
        for i,x_l in enumerate(x_len):
            vsu = vs_slice.interp(Distance_Along_Trend = x_l,  method = 'linear').vs.values
            vsu[vsu <= 2.0] = vsu[vsu > 2.0].min()
            v_interp_func = interp1d(vs_slice.depth.values, vsu, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
            vsu = v_interp_func(z)
            vd[:,i] = vsu
            
            ksu = vpvs_slice.interp(Distance_Along_Trend = x_l,  method = 'linear').VpVs.values
            ksu[ksu <= 1.72] = ksu[ksu > 1.72].min()
            ksu[ksu >= 1.82] = ksu[ksu < 1.82].max()
    
            k_interp_func = interp1d(vpvs_slice.depth.values, ksu, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
            ksu = k_interp_func(z)
            kd[:,i] = ksu

        # vd is the velocity profile beneath points along primary lines
        
        
        
        ## colums are segments on line
        ## rows are depth
        
        qa = np.sqrt((1/(vd)**2) - p_ref**2)
        qb = np.sqrt((1/(vd*kd)**2) - p_ref**2)

        # solve for travel time within each depth interval
        dt = 0.5 * (qa - qb)

        tbot = np.cumsum(dt, axis = 0)    # total travel time at bottom of interval 
        ttop = tbot - dt        # total travel time at top of interval 
        tmid = np.mean([tbot, ttop], axis = 0) # estimated time at middle of interval
        
        

        # cycle through columns
        z_interp = interp1d(tmid[:,i], z)

        first_mult = z_interp(tppps) 
        first_mult_up = z_interp(tppps_up)  
        first_mult_down = z_interp(tppps_down)  


        

        fig.plot(x = np.append(x_len, np.flip(x_len)),
                 y = np.append(-first_mult_up, np.flip(-first_mult_down)),
                 pen = '2p,white', fill = 'blue', transparency = 50)
    
        fig.plot(x = x_len, y = -first_mult, pen = '3p,white,--')

        
    fig.basemap(region = [300, 600, -depth, 0], projection = 'X{}c/{}c'.format(fig_x, fig_y), frame = ['wsen', 'xa20f10g50+lProfile Distance (km)', 'ya20f10g50+lDepth (km)'])

    
    with pygmt.config(FONT_ANNOT_PRIMARY="20p", FONT_LABEL = '20p'):
        fig.colorbar(cmap = True, position = 'JMR+w{}/0.4+o0.25c/0c'.format(fig_y), frame = 'xa0.25f0.1+lAmplitude')
   
    
    


    spec = io.StringIO(
    """   
N 4
S 0.20c i 0.3c cyan 0.4p,black 0.65c TANGO Node
S 0.20c i 0.3c magenta 0.6p,black 0.65c TANGO Broadband
S 0.20c i 0.3c gold 0.6p,black 0.65c Other Network

S 0.20c kvolcano 0.40c red 0.75p,black 0.65c Holocene Volcano
# S 0.20c r 0.4/0.25 white 0.8p,black,-- 0.65c Tectonic Region
S 0.20c r 0.4/0.25 pink 0.8p,black 0.65c APVC

S 0.2c - 0.6c black 1p,black,dashed 0.65c PpPs Multiple
S 0.20c r 0.4/0.25 blue 0.8p,black 0.65c PpPs Mulitple \\261 10% vs




G 0.07c
G 0.07c

    """
    )


    fig.legend(spec = spec,  position='jBL+w20c+o0c/-3.5c')


    fig.shift_origin(xshift = '{}'.format(fig_x+3.5))
    
    # now not changing fig_y
    
    depth = 50
    aspect = 1.5
    ratio = depth /fig_y    # km per cm-y
    fig_x = (760 - 650)/ratio/aspect
    
    
    fig.basemap(region = [650, 760, -depth, 0], projection = 'X{}c/{}c'.format(fig_x, fig_y), frame = ['wSnE', 'xa50f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'] )


    model = xr.open_dataset('../DATA/MAPPING/nc_models/CCP_TANGO_North_G-5.0_GDW_RE.nc')
    model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})
    model_slice.Amplitude.values = np.clip(model_slice.Amplitude.values, -0.25, 0.25)

    grid_color = pygmt.makecpt(cmap = '../DATA/MAPPING/no_green.cpt',   series = [-0.25, 0.25])
    fig.grdimage(grid = model_slice['Amplitude'], cmap = True)

    
    for primary in time_primaries:
        
        x_len = np.arange(primary[1], primary[2], 1)

        tps = primary[0]
       
        
        vp_assume = 4.0
        vpvs_assume = 2.2

        qa = np.sqrt((1/(vp_assume/vpvs_assume)**2) - p_ref**2)
        qb = np.sqrt((1/vp_assume**2) - p_ref**2)
        
        tppps = tps * ( (qa + qb) / (qa - qb) )     # arrival time of first mulitple
        tppss = tps *( ( 2*qa )/ (qa-qb))           # arrival time of second multiple



        qa_up = np.sqrt((1/(vp_assume/(vpvs_assume-vpvs_assume*0.1))**2) - p_ref**2)
        qb = np.sqrt((1/vp_assume**2) - p_ref**2)
        
        tppps_up = tps * ( (qa_up + qb) / (qa_up - qb) )     # arrival time of first mulitple
        tppss_up = tps *( ( 2*qa_up )/ (qa_up-qb))           # arrival time of second multiple

        qa_down = np.sqrt((1/(vp_assume/(vpvs_assume+vpvs_assume*0.1))**2) - p_ref**2)
        qb = np.sqrt((1/vp_assume**2) - p_ref**2)
        
        tppps_down = tps * ( (qa_down + qb) / (qa_down - qb) )     # arrival time of first mulitple
        tppss_down = tps *( ( 2*qa_down )/ (qa_down-qb))           # arrival time of second multiple


        # propagate the estimated multiple times as the would be in the CCP migration
        
        z = np.arange(0,35,0.5)
        vsu = vs_slice.interp(Distance_Along_Trend = x_len, depth = z, method = 'linear').vs.values
        vsu[vsu <= 2.0] = vsu[vsu > 2.0].min()
        
        
        ksu = vpvs_slice.interp(Distance_Along_Trend = x_len, depth = z, method = 'linear').VpVs.values
        
        ## colums are segments on line
        ## rows are depth
        
        qa = np.sqrt((1/(vsu)**2) - p_ref**2)
        qb = np.sqrt((1/(vsu*ksu)**2) - p_ref**2)

        # solve for travel time within each depth interval
        dt = 0.5 * (qa - qb)

        tbot = np.cumsum(dt, axis = 0)    # total travel time at bottom of interval 
        ttop = tbot - dt        # total travel time at top of interval 
        tmid = np.mean([tbot, ttop], axis = 0) # estimated time at middle of interval
        
        # cycle through columns
        
        primary_in_depth = np.array([])
        first_mult = np.array([])
        first_mult_up = np.array([])
        first_mult_down = np.array([])
        second_mult = np.array([])
        second_mult_up = np.array([])
        second_mult_down = np.array([])

        for i in range(tmid.shape[1]):
            z_interp = interp1d(tmid[:,i], z)
            
            primary_in_depth = np.append(primary_in_depth, z_interp(tps))
            first_mult = np.append(first_mult, z_interp(tppps))
            second_mult = np.append(second_mult, z_interp(tppss))
        
            first_mult_up = np.append(first_mult_up, z_interp(tppps_up))
            second_mult_up = np.append(second_mult_up, z_interp(tppss_up))

            first_mult_down = np.append(first_mult_down, z_interp(tppps_down))
            second_mult_down = np.append(second_mult_down, z_interp(tppss_down))

        
        fig.plot(x = x_len, y = -primary_in_depth, pen = '3p,black,--')


        fig.plot(x = np.append(x_len, np.flip(x_len)),
                 y = np.append(-first_mult_up, np.flip(-first_mult_down)),
                 pen = '2p,white', fill = 'red', transparency = 50)

        fig.plot(x = np.append(x_len, np.flip(x_len)),
                 y = np.append(-second_mult_up, np.flip(-second_mult_down)),
                 pen = '2p,white', fill = 'blue', transparency = 50)
       
        fig.plot(x = x_len, y = -first_mult, pen = '3p,black,--')
        fig.plot(x = x_len, y = -second_mult, pen = '3p,white,--')


    
    # with pygmt.config(FONT_ANNOT_PRIMARY="18p", FONT_LABEL = '18p'):
    #     fig.colorbar(cmap = True, position = 'JMR+w{}+o0.75c/0c'.format(fig_y), frame = 'xa0.25f0.1+lAmplitude')

    spec = io.StringIO(
    """   
N 2

S 0.20c r 0.4/0.25 red 0.8p,black 0.65c PpPs Mulitple \\261 10% vp/vs
S 0.2c - 0.6c black 1p,black,dashed 0.65c Multiple
S 0.20c r 0.4/0.25 blue 0.8p,black 0.65c PpSs+PsPs Mulitple \\261 10% vp/vs



G 0.07c
G 0.07c

    """
    )


    fig.legend(spec = spec,  position='jBL+w{}c+o0c/-3.5c'.format(15))

   

    fig.shift_origin(yshift = '{}'.format(fig_y+0.5))


    fig.basemap(region = [650, 760, 0, 7], projection = 'X{}c/-{}c'.format(fig_x, total_y - fig_y-1.5-0.5), frame = ['wsnE', 'xa50f10', 'ya2f1+lTime (s)'] )
    
    
    ## Pull QC passing data
    itd_use = itd[itd.G == 5.0][(itd.Accept == 'Auto QC Pass') | (itd.Accept == 'Man QC Pass')][itd.Region == region]
    
    ## collect station location
    stlas = [stations.Latitude[stations.code == code.replace('_', '-')].values[0] for code in itd_use.Code]
    stlos = [stations.Longitude[stations.code == code.replace('_', '-')].values[0] for code in itd_use.Code]
    
    stat_coords = np.c_[stlos, stlas]
    
    itd_use['stlo'] = stlos
    itd_use['stla'] = stlas
    
    
    
    
    
    ## search for data in range of A-B line
    line_tree = KDTree(line)
    
    data_in_range = line_tree.query_ball_point(x = stat_coords, r = 30 / degrees2kilometers(1))
    data_in_range =[bool(data_in_range[j]) for j in range(len(data_in_range))]
    
    
    ## search for data in range of 1-2 line
    joints12 = np.array([[-65.5816, -24.1358], [-64.8716, -24.3922]]) # Sub-line for north array
    joints_dist12, line12, line_dist12 = Return_Line_Elements(joints12)

    
    line_tree12 = KDTree(line12)
    data_in_range12 = line_tree12.query_ball_point(x = stat_coords, r = 10 / degrees2kilometers(1))
    data_in_range12 =[bool(data_in_range12[j]) for j in range(len(data_in_range12))]
    
    
    ## Read file data and time-stack for average station RF
    files = np.unique(np.array(itd_use.File)[data_in_range])
    files12 = np.unique(np.array(itd_use.File)[data_in_range12])
    files = np.append(files, files12)
    
    
    nets, stlas, stlos, stels, tr_id = [],[],[],[],[]    # resetting stlas and stlos list to queried data
    st = Stream()
    for f in files:
        try:
            st += read('../DATA/{}/DATA_CCP/{}'.format(region, f))
        except:
            pass
    
    st.stack(group_by='{network}.{station}', npts_tol=2)    # T-stack by station for average station RF
    for i, tr in enumerate(st):
        nets.append(tr.stats.network)
        stlas.append(stations.Latitude[stations.code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stlos.append(stations.Longitude[stations.code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stels.append(grid.sel(lat = stlas[-1], lon = stlos[-1], method = 'nearest').values.item())
        tr_id.append(i)
    
    select_data = pd.DataFrame({'longitude': np.squeeze(stlos), 'latitude': np.squeeze(stlas), 'stel': np.squeeze(stels), 'tr_id': np.array(tr_id)})


    used_data = []
    for i in range(len(joints) - 1):                                                    ### width here is arbitrary
       track = pygmt.project(data = select_data,center=joints[i],endpoint=joints[i+1], width=[-100, 100], length = [0, joints_dist[i+1]], unit=True)
       track = track.rename(columns={0:'stlo',1:'stla',2:'stel',3:'tr_id',4:'p',5:'q',6:'r',7:'s'})

       ## check for re-used data from projection, remove those data
       data_used_check = [track.tr_id[j] in used_data for j in range(len(track))]
       data_used_check = [not(bool(data_used_check[j])) for j in range(len(data_used_check))]
       track = track[data_used_check].reset_index(drop = True)
       used_data = np.append(used_data, np.unique(track.tr_id.values))
       
       track = track.sort_values('p', ascending = False).reset_index(drop = True)

       for j in range(len(track)-2):

           station_dist = track.p[j] + joints_dist[i]

           # collect data within astack_ range of each center station
           center_tree = KDTree(np.c_[track.stlo[j], track.stla[j]])
           in_range = center_tree.query_ball_point(x = np.c_[select_data.longitude, select_data.latitude], r = 8 / degrees2kilometers(1))
           in_range =[bool(in_range[k]) for k in range(len(in_range))]
           in_range = select_data[in_range].tr_id.values

           st_stack = Stream()
           for trace in in_range:
               st_stack += st[trace].copy()

           st_stack.stack(npts_tol = 2)
           
           ## normalize to p-wave amplitude
           extrema_limit = st_stack[0].data.max() * 0.10
           # cycle through the data array to find all major extrema
           extrema=[]
           extrema_i = []
           k = 1
           for dat in st_stack[0].data[1:len(st_stack[0].data)-2]:
               left = st_stack[0].data[k - 1]
               right = st_stack[0].data[k + 1]
               if (dat > left) and (dat > right):
                       if dat > extrema_limit:
                           extrema.append(dat)
                           extrema_i.append(k)
               elif (dat < left) and (dat < right):
                   if dat < -extrema_limit:
                       extrema.append(dat)
                       extrema_i.append(k) 
               k += 1
               
           
           # get the first peak amplitude, and divide the data by this value
           # the data is now normalized
           p_amp = extrema[0]
           
           st_stack[0].data = st_stack[0].data / p_amp


           tr = st_stack[0]
           data = tr.data
           time = tr.times() - 10

           time_trim = time
           data_trim = data * 10 + station_dist

           pos = data_trim.copy()
           neg = data_trim.copy()

           pos[0] = station_dist
           pos[pos < station_dist] = station_dist
           pos[-1] = station_dist

           neg[0] = station_dist
           neg[neg > station_dist] = station_dist
           neg[-1] = station_dist

           fig.plot(x = pos, y = time_trim,pen = '0.1p,black', no_clip = False, fill = 'red', transparency = 20)
           fig.plot(x = neg, y = time_trim,pen = '0.1p,black', no_clip = False, fill = 'blue', transparency = 20)



    for primary in time_primaries:
        
        x_len = np.arange(primary[1], primary[2], 1)

        tps = primary[0]
        fig.plot(x = x_len, y = np.ones(x_len.shape) * tps, pen = '3p,black,--')
        
        vp_assume = 4.0
        vpvs_assume = 2.2

                
        qa = np.sqrt((1/(vp_assume/vpvs_assume)**2) - p_ref**2)
        qb = np.sqrt((1/vp_assume**2) - p_ref**2)
        
        tppps = tps * ( (qa + qb) / (qa - qb) )     # arrival time of first mulitple
        tppss = tps *( ( 2*qa )/ (qa-qb))
        
        qa = np.sqrt((1/(vp_assume/(vpvs_assume + 0.1*vpvs_assume))**2) - p_ref**2)
        qb = np.sqrt((1/vp_assume**2) - p_ref**2)
        tppps_up = tps * ( (qa + qb) / (qa - qb) )     # arrival time of first mulitple
        tppss_up = tps *( ( 2*qa )/ (qa-qb))


        qa = np.sqrt((1/(vp_assume/(vpvs_assume - 0.1*vpvs_assume))**2) - p_ref**2)
        qb = np.sqrt((1/vp_assume**2) - p_ref**2)
        tppps_down = tps * ( (qa + qb) / (qa - qb) )     # arrival time of first mulitple
        tppss_down = tps *( ( 2*qa )/ (qa-qb))

        

        fig.plot(x = np.append(x_len, np.flip(x_len)),
                 y = np.append(np.ones(x_len.shape) *tppps_up, np.ones(x_len.shape) *np.flip(tppps_down)),
                 pen = '2p,white', fill = 'red', transparency = 50)

        fig.plot(x = np.append(x_len, np.flip(x_len)),
                 y = np.append(np.ones(x_len.shape) *tppss_up, np.ones(x_len.shape) *np.flip(tppss_down)),
                 pen = '2p,white', fill = 'blue', transparency = 50)
        
        fig.plot(x = x_len, y = np.ones(x_len.shape) * tppps, pen = '3p,black,--')
        fig.plot(x = x_len, y = np.ones(x_len.shape) * tppss, pen = '3p,white,--')

    
    # shift origin up to plot topography
    
    fig.shift_origin(yshift = '+{}c'.format(total_y - fig_y-1.5-0.5))
    total_y = total_y + 1.5
    
    fig.basemap(region = [650, 760, 0, 7], projection = 'X{}c/{}c'.format(fig_x, 1.5), frame = ['wsNE', 'xa100f10', 'ya5f1+e+lElev. (km)'])
    


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
        APVC = gpd.read_file('../DATA/MAPPING/APVC2.shp')
    
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
    
    
fig.savefig('../FIGURES/S7_ARF_Multiples.png', dpi = 300)















