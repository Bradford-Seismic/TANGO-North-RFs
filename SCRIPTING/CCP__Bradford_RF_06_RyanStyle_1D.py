#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:36:54 2025

@author: bradford
"""


import os
import sys
import io
import time
import gc


import os
import sys
import io
import time
import gc


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
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
# from sklearn.neighbors import KDTree

from scipy.stats import binned_statistic_2d
import scipy
import concurrent.futures


import geopandas as gpd

import warnings
warnings.simplefilter('ignore', category = UserWarning)
warnings.simplefilter('ignore', category = FutureWarning)

active_dir = './'

os.chdir(active_dir)


region = 'TANGO_North'

CCP_dir = '../DATA/{}/DATA_CCP/'.format(region)
CCP_Files = os.listdir(os.path.join(CCP_dir))

v_model_1D = 'IASP91_mod_70km.csv'



#%% Grab Data

G = 5.0


itd_pass = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv',keep_default_na=False)

itd_pass = itd_pass[itd_pass.Accept == 'Man QC Pass'][itd_pass.G == G][itd_pass.Region == region][itd_pass.Phase == 'P']
itd_pass = itd_pass.reset_index()





unique_events = np.unique(itd_pass.Event.to_numpy())
unique_stations = np.unique(itd_pass.Code.to_numpy())


st  = Stream()
stlas = stlos = stels = codes = nets = stats = np.array([])
t = 0
for i, f in enumerate(itd_pass.File):
    t+=1
    print("\r", end="")
    print("Reading Stations: {:.1%} ".format(t/(len(itd_pass))), end="")


    if f in CCP_Files:
        st += read(os.path.join(CCP_dir, f))
        st[-1].fname = f

        # trace id refers to the row index of the file within the iterdecon summary
        st[-1].trace_id = itd_pass.index[i]

        code = '{}-{}'.format(st[-1].stats.network, st[-1].stats.station)

        if code not in codes:
            codes = np.append(codes, code)

            nets = np.append(nets, st[-1].stats.network)
            stats = np.append(stats, st[-1].stats.station)

            stlas = np.append(stlas, st[-1].stats.sac.stla)
            stlos = np.append(stlos, st[-1].stats.sac.stlo)
            stels = np.append(stels, st[-1].stats.sac.stel)

    else:
        continue

stat_table = pd.DataFrame({'net': nets, 'stat':stats, 'stla': stlas, 'stlo': stlos})




print('{} station within data space, {} RF traces available'.format(len(stat_table), len(st)))




del stlas, stlos, stels, codes, nets, stats, CCP_Files
gc.collect()





#%% Construct Area

select_region = 0  # manually select a region
use_region =  1      # use a saved region
save_region = 0    # save the region

dzi = .5   # depth increment in km
z_max = 150 # max depth in km

bin_sz = 5 # node spacing in km
bin_min_sm = 12 # base radius of bin in km around node, will increase to find more pierce points if allowed
bin_min_gauss = 4
hits_per_bin = 1


ref_angle = 20  # reference angle for calculating amplitude scaling factor
baz_range = [0, 360] # range of desired back-azimuths for CCP grid


# buffer zone to format to grid
edge_buff = 0.2



# Area defining the grid edge-boundaries , either select the region manually or defined boundary as station extent
plt.close('all')
if select_region:
    fig, ax = plt.subplots(figsize = (12, 12))

    ax.scatter(stat_table.stlo, stat_table.stla, marker = 'v', color = 'r', edgecolor = 'black')
    ax.set_aspect('equal')

    nw_se_corner = plt.ginput(n = 2, show_clicks = True)

    north = np.ceil(nw_se_corner[0][1]*1e2)/1e2+edge_buff
    south = np.ceil(nw_se_corner[1][1]*1e2)/1e2-edge_buff

    west = np.ceil(nw_se_corner[0][0]*1e2)/1e2-edge_buff
    east = np.ceil(nw_se_corner[1][0]*1e2)/1e2+edge_buff


else:

    if use_region:

        # north, south, east, west = np.loadtxt('../DATA/MAPPING/TANGO_North_CCP_GridArea.txt')
        north, south, east, west = np.loadtxt('../DATA/MAPPING/{}_CCP_GridArea-ABCD.txt'.format(region))

    else:
        print('assigning grid by provided station data')
        north = np.ceil(stat_table.stla.max()*1e2)/1e2+edge_buff
        south = np.floor(stat_table.stla.min()*1e2)/1e2-edge_buff

        east = np.ceil(stat_table.stlo.max() *1e2)/ 1e2+edge_buff
        west = np.floor(stat_table.stlo.min() *1e2)/ 1e2-edge_buff

plt.close('all')


if save_region:
    area = [north, south, east, west]
    np.savetxt('../DATA/MAPPING/{}_CCP_GridArea.txt'.format(region), area, delimiter = ',')







#%% Construct Grid spaces for models and vectors

# Filter the stream list and remove all traces not within baz_limit:
for tr in st:
    baz = tr.stats.sac.baz
    if (baz > baz_range[0]) and (baz < baz_range[1]):
        continue
    else:
        st.remove(tr)



def latitude_to_km(latitude):
    return latitude * (np.pi/180) * 6371


def longitude_to_km(longitude, latitude):
    # Convert latitude from degrees to radians
    latitude = radians(latitude)

    # Radius of the Earth at the given latitude
    earth_radius_at_latitude = 6371 * cos(latitude)  # Earth radius in kilometers

    return abs(longitude) * earth_radius_at_latitude


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

def haversine_rad(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = sin(d_lat/2)**2 + cos(lat1) * cos(lat2) * sin(d_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = c  # Angular distance in radians
    return distance


def arc_length_on_earth(lat1, lon1, lat2, lon2):
    distance = haversine_rad(lat1, lon1, lat2, lon2) * (180/np.pi)
    return distance

def deg_to_km(degrees):
    km = degrees*(np.pi/180) * 6371
    return km

def km_to_deg(km):
    degrees = km * (180/np.pi) / 6371
    return degrees

def km_to_latitude(km):
    deg_lat =  km/((np.pi/180) * 6371)
    return deg_lat



# converter for longitudinal distance, used for flattening
dd = deg_to_km(np.mean([arc_length_on_earth(south,west,south,east), arc_length_on_earth(north,west,north,east)])/(east - west))

# Parameterize grid as boxes

ydist = latitude_to_km(north - south)
xdist = dd*(east-west)


# construct grid as x-y in km
x = np.arange(0, xdist+bin_sz,bin_sz) # x values
y = np.arange(0,ydist+bin_sz,bin_sz) # y values






# grab elevation data in grid area
gmt_region = '{}/{}/{}/{}'.format(west,east,south,north)
with pygmt.datasets.load_earth_relief(resolution="15s", region=gmt_region) as g:
    glon = g.lon.values.astype(np.float32)
    glat = g.lat.values.astype(np.float32)
    elev = g.values.astype(np.float32)
    elev[elev < 0] = 0 # remove values below sea level

    gy = latitude_to_km(glat - south)
    gx = dd * (glon - west)

# dem  = xr.Dataset(data_vars = {'z':(['y', 'x'], elev)},
#                           coords = {'y': gy, 'x': gx})

# reorder elevation grid into a single vector

GX,GY = np.meshgrid(gx,gy)
elev_coords = np.c_[GX.reshape(GX.size, -1), GY.reshape(GY.size, -1)]
ELEV = np.c_[elev.reshape(np.size(elev), -1)]

# set up KDTree for distance-searches
elev_tree = KDTree(elev_coords)

del glon, glat, elev, gy, gx, elev_coords, GX, GY




### write stream coordinate information as x-y on coordinate grid
# station coordinate defined with reference to bottom-left corner of grid (west, south)


k = 0
st_len = len(st)
stzs = []
STX = []
STY = []
NET = []
for tr in st:
    k += 1
    print("\r", end="")
    print("locating stations on grid: {:.1%} ".format(k/st_len), end="")

    stla = tr.stats.sac.stla
    stlo = tr.stats.sac.stlo

    # reject trace if it falls out of grid range
    if (stla > north) or (stla < south) or (stlo < west) or (stlo > east):
        st.remove(tr)
        continue

    sty = np.round(latitude_to_km(stla - south), 4)
    stx = np.round(dd*(stlo - west), 4)

    # stz = float(dem.sel(y = sty, x = stx ,method = 'nearest').z.values) / 1000
    stz_idx = elev_tree.query([stx, sty], k = 1)[1]
    stz = ELEV[stz_idx][0] / 1000

    stzs.append(stz)
    STX.append(stx)
    STY.append(sty)
    NET.append(tr.stats.network)

    tr.x = stx
    tr.y = sty
    tr.z = stz




# set z to level at the maximum station elevation
z = np.arange(np.round((-np.max(stzs)) * 1/dzi) * dzi, z_max + dzi, dzi).astype('float32')

del stzs, st_len, k, ELEV



fig = pygmt.Figure()    # initialize the map

with fig.subplot(nrows = 1, ncols = 1, figsize=('15c', '15c')):
    with fig.set_panel(panel = 0):
        prj = 'M?'            # set the map projection
        with pygmt.config(FONT="14p", FORMAT_GEO_MAP="ddd.xx"):


            fig.basemap(region=gmt_region, projection=prj, frame='a2g4')


            fig.plot(x = stat_table.stlo, y = stat_table.stla, style='i0.23c', fill = 'gold', transparency = 50, pen='0.4p,black')


            try:
                fig.plot(x = stat_table[stat_table.net == 'XN'].stlo, y =  stat_table[stat_table.net == 'XN'].stla, region = gmt_region, style='i0.25c',fill = 'cyan', pen='black')
            except:
                pass
            try:
                fig.plot(x = stat_table[stat_table.net == '1X'].stlo, y =  stat_table[stat_table.net == '1X'].stla, region = gmt_region,style='i0.25c',fill = 'cyan', pen='black')
            except:
                pass

            try:
                fig.plot(x = stat_table[stat_table.net == 'XM'].stlo, y =  stat_table[stat_table.net == 'XM'].stla, region = gmt_region,style='i0.3c',fill = 'magenta', pen='0.6p,black')
            except:
                pass

fig.show()



gc.collect()


# plot in projected view
fig = pygmt.Figure()    # initialize the map

prj = 'X15c/15c'            # set the map projection
with pygmt.config(FONT="14p"):


    fig.basemap(region=[50,550, -100, 400], projection=prj, frame='a25g50')


    fig.plot(x = STX, y = STY, style='i0.23c', fill = 'gold', pen='0.4p,black')


    try:
        fig.plot(x = STX[NET=='XN'], y =  STY[NET=='XN'], region = gmt_region, style='i0.25c',fill = 'cyan', pen='black')
    except:
        pass
    try:
        fig.plot(x = STX[NET=='1X'], y =  STY[NET=='1X'], region = gmt_region,style='i0.25c',fill = 'cyan', pen='black')
    except:
        pass

    try:
        fig.plot(x = STX[NET=='XM'], y =  STY[NET=='XM'], region = gmt_region,style='i0.3c',fill = 'magenta', pen='0.6p,black')
    except:
        pass

fig.show()




# Plot v_model

v_data = pd.read_csv('../DATA/MAPPING/nc_models/' + v_model_1D,   skiprows = 1, header = None)

vsu = v_data[2].to_numpy()
vpu = v_data[1].to_numpy()
vd = v_data[0].to_numpy()



# Interpolate vs and k at mid-points between layers in z using nearest values
vs_interp_func = interp1d(vd, vsu, kind = 'nearest', bounds_error = False, fill_value = 'extrapolate')
vp_interp_func = interp1d(vd, vpu, kind = 'nearest', bounds_error = False, fill_value = 'extrapolate')

vs = vs_interp_func(z + 0.5*dzi).astype('float32')
vp = vp_interp_func(z + 0.5*dzi).astype('float32')


# vs[np.where(vs < minv)] = minv



fig = pygmt.Figure()    # initialize the map

with fig.subplot(nrows = 1, ncols = 1, figsize=('3c', '10c')):
    with fig.set_panel(panel = 0):
        with pygmt.config(FONT="14p"):
            fig.basemap(region=[vs.min() - 1, vp.max() + 1, z.min(), z.max()], projection='X3c/-10c', frame =[ 'WSne', 'xa2f1+lvel.', 'ya10f5g40+lz (km)'])

            fig.plot(x = vp, y = z, pen = '1p,blue')
            fig.plot(x = vs,  y = z, pen = '1p,red,--')

spec = io.StringIO(
    """
N 2
S 0.20c - 0.6c - 1p,blue 0.7c vp
S 0.20c - 0.6c - 1p,red 0.7c vs

    """
    )


with pygmt.config(FONT_ANNOT_PRIMARY="14p"):
    fig.legend(spec = spec,  position='jTL+w4c+o0c/4c')


fig.show()



#%% Ray Tracing


t_offset = 10   # time from tr time = 0 where p-wave peak should be centered

def Trace_Pierce_Point(trace_index):
    print("\r", end="")
    print("Tracing Rays: {:.1%} ".format(trace_index/len(st)), end="")


    t1 = time.perf_counter()





    tr = st[trace_index].copy()

    # 1) acquire station location and baz/ray parameter
    stx = tr.x
    sty = tr.y
    stz = tr.z

    # round station elevation to nearest z level position
    se = np.round(stz / dzi) * dzi

    baz = tr.stats.sac.baz
    p = tr.stats.sac.user4



   ## 2) normalize to p-wave amplitude
    extrema_limit = tr.data.max() * 0.10
    # cycle through the data array to find all major extrema
    extrema=[]
    extrema_i = []

    # Note, maintain k as iterable variable
    k = 1
    for dat in tr.data[1:len(tr.data)-2]:
        left = tr.data[k - 1]
        right = tr.data[k + 1]
        if (dat > left) and (dat > right):
                if dat > extrema_limit:
                    extrema.append(dat)
                    extrema_i.append(k)
        elif (dat < left) and (dat < right):
            if dat < -extrema_limit:
                extrema.append(dat)
                extrema_i.append(k)
        k += 1

    p_amp = extrema[0]
    dat = tr.data.copy() / p_amp
    times = tr.times().copy() - t_offset






    # Solve for incident angle scaling along vector
    i_ang = np.rad2deg(np.arcsin(p*vs))
    a_scale = ref_angle / i_ang

    # Solve for qa and qb coefficients
    qa = np.sqrt((1/(vs)**2) - p**2)
    qb = np.sqrt((1/(vp)**2) - p**2)

    # solve for travel time in each depth interval
    dt = dzi * (qa - qb)

    tbot = np.cumsum(dt)    # total travel time at bottom of interval
    ttop = tbot - dt        # total travel time at top of interval
    tmid = np.mean([tbot, ttop], axis = 0) # estimated time at middle of interval

    # control for indexes in tmid
    within_range_indexes = np.where(tbot <= times.max())[0]

    tbot = tbot[within_range_indexes]
    ttop = ttop[within_range_indexes]
    tmid = tmid[within_range_indexes]
    trace_z = z[within_range_indexes]
    qa = qa[within_range_indexes]
    qb = qb[within_range_indexes]

    ### time and depth vectors now reflect the unique range of coverage by ray
    ### Everything so far has been solved with respect to maximum surface elevation within z_vector

    # interpolate the amplitudes at tmid from trace data
    dat_interp_func = interp1d(times, dat, kind = 'linear')
    seis = dat_interp_func(tmid)

    # Identify indexes that encompass the z-space above the station elevation
    eci = int((-(z.min()) - se) / dzi) + 1

    # append Nan space by a length of eci above,
    # then remove that same amount of indices from the end of seis
    # this accounts for elevation
    empty_elev = np.ones(eci) * np.nan
    seis = np.append(empty_elev, seis)
    seis = seis[0:len(seis) - eci]

    seisout = seis * a_scale


    h_offset = np.ones(qa.shape) * np.nan
    h_cumm = np.ones(qa.shape) * np.nan

    # trace the ray across the grid field
    h_offset[eci-1 : len(h_offset)] = dzi * p / qa[eci-1 : len(qa)]

    # cummulative lateral travel
    h_cumm[eci-1 : len(h_cumm)] = np.append([0], np.nancumsum(h_offset[eci -1 :len(h_offset) - 1]))

    # project onto direction
    ew = h_cumm * np.sin(np.deg2rad(baz))
    ns = h_cumm * np.cos(np.deg2rad(baz))

    xpierce = ew + stx
    ypierce = ns + sty


    t2  = time.perf_counter()

    return trace_index, xpierce, ypierce, trace_z, seisout



## Multiprocessing Solve
trace_id = np.arange(0, len(st), 1)
with concurrent.futures.ProcessPoolExecutor(max_workers = 5) as executor:
    pierce_traces = executor.map(Trace_Pierce_Point, trace_id)

    for ray in pierce_traces:
        st[ray[0]].pierce = {'x': ray[1], 'y': ray[2], 'z': ray[3], 'seis': ray[4]}



# ## Brute Force Solve
# trace_id = np.arange(0, len(st), 1)
# for trace in trace_id:
#     ray = Trace_Pierce_Point(trace)

#     st[ray[0]].pierce = {'x': ray[1], 'y': ray[2], 'z': ray[3], 'seis': ray[4]}




#%% Plot the Piercing Points


pierce_depth = 50
print('Plotting Pierce Points at {} km depth'.format(pierce_depth))


save_pierce = 0


try:
    contributing = pd.read_csv('../DATA/SPREADSHEETS/Contributing_Stations.csv')
except:
    pass


x_depth = y_depth = stx = sty = np.array([])
n_x_depth = n_y_depth = s_x_depth = s_y_depth = np.array([])
for tr in st:

    try:
        code = '{}-{}'.format(tr.stats.network, tr.stats.station)
        if code not in contributing.Code.to_list():
            continue
    except:
        pass

    stx = np.append(stx, tr.x)
    sty = np.append(sty, tr.y)

    x_depth = np.append(x_depth, tr.pierce['x'][np.where(tr.pierce['z'] == pierce_depth)])
    y_depth = np.append(y_depth, tr.pierce['y'][np.where(tr.pierce['z'] == pierce_depth)])

    if (tr.stats.sac.baz > 290) and (tr.stats.sac.baz < 360):
        n_x_depth = np.append(n_x_depth, tr.pierce['x'][np.where(tr.pierce['z'] == pierce_depth)])
        n_y_depth = np.append(n_y_depth, tr.pierce['y'][np.where(tr.pierce['z'] == pierce_depth)])

    if (tr.stats.sac.baz > 90) and (tr.stats.sac.baz < 180):
        s_x_depth = np.append(s_x_depth, tr.pierce['x'][np.where(tr.pierce['z'] == pierce_depth)])
        s_y_depth = np.append(s_y_depth, tr.pierce['y'][np.where(tr.pierce['z'] == pierce_depth)])


pierce_region = '{}/{}/{}/{}'.format(x_depth.min() - 0.1, x_depth.max() + 0.1, y_depth.min() - 0.1, y_depth.max() + 0.1)


fig = pygmt.Figure()    # initialize the map

with fig.subplot(nrows = 1, ncols = 1, figsize=('15c', '15c')):
    with fig.set_panel(panel = 0):


        with pygmt.config(FONT="14p"):


            fig.basemap(region=[-40, 240, -50, 230], projection='X15c/15c', frame='a100g50')

            fig.plot(x = x_depth, y = y_depth, style = 'x0.2c', pen = '0.7p,black')

            fig.plot(x = 0, y = 0, style = 'c{}c'.format(bin_min_sm*2*(15/280)), fill = 'yellow', pen = '1p,black', transparency = 30)
            fig.plot(x = 0, y = 0, style = 'c{}c'.format((bin_min_sm/3)*2*(15/280)), fill = 'darkorange', transparency = 30)


            fig.plot(x = [x.min(), x.max(), x.max(), x.min(), x.min()], y = [y.min(), y.min(), y.max(), y.max(), y.min()], pen = '1p,red,--')
            try:
                fig.plot(x = stx, y =  sty,  style='i0.25c',fill = 'yellow', pen='black')
            except:
                pass


            # fig.plot(x =line_x, y = line_y, pen = '2p,black,--')



spec = io.StringIO(
"""
N 2
S 0.20c i 0.3c gold 0.6p,black 0.65c Station

S 0.2c x 0.2c black 0.7p,black 0.65c Pierce Point
S 0.2c c 0.2c white 0.3p,blue 0.65c NW - directed Pierce Point
S 0.2c c 0.2c white 0.3p,red 0.65c SE - directed Pierce Point

G 0.07c
G 0.07c

"""

    )
with pygmt.config(FONT_ANNOT_PRIMARY="12p"):
        fig.legend(spec = spec,  position='jBL+w12c+o0c/-2c')


fig.show()



if save_pierce:

    pierce_lat = south + km_to_latitude(y_depth)
    pierce_lon = west + x_depth/dd


    pierce_xy = np.c_[pierce_lon, pierce_lat]

    np.savetxt('../DATA/MAPPING/Pierce-Contributing_xy_z-{}.txt'.format(pierce_depth), pierce_xy, delimiter = ',')
    fig.savefig('../FIGURES/{}_PiercePoints-Contributing_z-{}.png'.format(region, pierce_depth), dpi = 300)


#%% Find Contributing Stations to line
joints = np.loadtxt('../DATA/MAPPING/LDM_Line2.txt', delimiter = ',')


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
    line = np.array([0,0])  # holding values
    for i in range(len(joints)-1):
        l = np.linspace(joints[i], joints[i+1], int(num_points[i]))

        if i == 0: # append all values for first iteration
            line = np.vstack((line, l))

        else: # append only values after the first so we don't duplicate values
            line = np.vstack((line, l[1:len(l)]))


    line = line[1:len(line)] # remove holding value


    # cummulative distance along line
    d = np.array([0])
    for i in range(len(line)-1):
        d = np.append(d, haversine(line[i,1], line[i,0],line[i+1,1], line[i+1,0]))

    line_dist = np.cumsum(d)

    return joints_dist, line, line_dist


joints_dist, line, line_dist  = Return_Line_Elements(joints)


contributing_stations = []
contributing_events = []

line_y = np.round(latitude_to_km(line[:,1] - south), 4)
line_x = np.round(dd*(line[:,0]  - west), 4)

line = np.c_[line_x, line_y]



line_tree = KDTree(line)
for tr in st:

    code = '{}-{}'.format(tr.stats.network, tr.stats.station)
    event = tr.fname.split('.')[3]

    xpierce = tr.pierce['x']
    ypierce = tr.pierce['y']

    pierce_coords = np.c_[xpierce, ypierce]
    rows_with_nan = np.isnan(pierce_coords).any(axis=1)
    pierce_coords = pierce_coords[~rows_with_nan]

    within = line_tree.query_ball_point(pierce_coords, bin_min_sm, return_length = True)

    if (within[within>0].any()) and (code not in contributing_stations):
        contributing_stations.append(code)
    if (within[within>0].any()) and (event not in contributing_events):
        contributing_events.append(event)



dat = pd.DataFrame({'Code': contributing_stations})
dat.to_csv('../DATA/SPREADSHEETS/Contributing_Stations.csv')

dat = pd.DataFrame({'Events': contributing_events})
dat.to_csv('../DATA/SPREADSHEETS/Contributing_Events.csv')



#%% Run CCP, with Parallel Pool Processing







def Bootstrap(sample, weights, num_bootstraps):
    sample_idx = np.arange(0,len(sample),1)

    bootstrap_averages = []
    for i in range(num_bootstraps):
        # bts_sample = np.random.choice(a = sample, size = len(sample), replace = True, p = weights)
        bts_sample = random.choices(sample, weights=weights, k = len(sample))
        bootstrap_averages.append(np.mean(bts_sample))

    bts_std = np.std(bootstrap_averages)
    bts_mean = np.mean(bootstrap_averages)
    return bts_mean, bts_std




def circular_variability(angles_deg, weights=None, axial=False):
    """
    Compute circular concentration/variability metrics for back-azimuths.

    Parameters
    ----------
    angles_deg : array-like
        Back-azimuths in degrees [0, 360).
    weights : array-like or None
        Optional nonnegative weights (e.g., RF quality, amplitude). Same length as angles.
    axial : bool
        If True, treat data as axial (mod 180°). Useful if 0° and 180° are equivalent.

    Returns
    -------
    dict with keys:
        Rbar : mean resultant length in [0, 1]  (higher = more concentrated)
        circ_std_deg : circular standard deviation in degrees
        mean_dir_deg : circular mean direction in degrees (0–360)
        kappa : approx. von Mises concentration parameter (NaN if undefined)
    """
    a = np.asarray(angles_deg, dtype=float)
    if axial:
        a = (2.0 * a) % 360.0

    theta = np.deg2rad(a)
    if weights is None:
        weights = np.ones_like(theta)
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)

    # Weighted unit vectors
    C = np.sum(w * np.cos(theta))
    S = np.sum(w * np.sin(theta))
    W = np.sum(w)

    if W <= 0:
        return dict(Rbar=np.nan, circ_std_deg=np.nan, mean_dir_deg=np.nan, kappa=np.nan)

    R = np.hypot(C, S)
    Rbar = R / W

    # Mean direction
    mean_dir = np.arctan2(S, C)
    mean_dir_deg = (np.degrees(mean_dir) % 360.0)
    if axial:
        mean_dir_deg = (mean_dir_deg / 2.0) % 360.0

    # Circular standard deviation (Fisher, 1993)
    if Rbar <= 0:
        circ_std_deg = np.nan
    else:
        circ_std_deg = np.degrees(np.sqrt(-2.0 * np.log(Rbar)))

    # Approximate von Mises kappa (Zar/Fisher approximations)
    # Useful if you want a concentration parameter rather than Rbar.
    if Rbar < 0.53:
        kappa = 2*Rbar + Rbar**3 + 5*Rbar**5/6
    elif Rbar < 0.85:
        kappa = -0.4 + 1.39*Rbar + 0.43/(1 - Rbar)
    else:
        kappa = 1/(Rbar**3 - 4*Rbar**2 + 3*Rbar)
    if not np.isfinite(kappa):
        kappa = np.nan

    return Rbar, circ_std_deg



def Gaussian_Distance_Weighted(depth):

    time.sleep(0.1*np.abs(depth))
    print("\r", end="")
    print("Solving CCP: {:.1%} ".format(depth/z_max), end="")

    t1 = time.perf_counter()

    # set coordinate grid locally within process core
    gridx, gridy = np.meshgrid(x, y)
    COORDS = np.c_[gridx.reshape(np.size(gridx), -1), gridy.reshape(np.size(gridy), -1)]


    # pull pierce point data of each ray
    xpierce = ypierce = seis = bazs = np.array([])
    trace_id = np.array([], dtype = int)
    for tr in st.copy():
        try:
            xpierce = np.append(xpierce, tr.pierce['x'][tr.pierce['z'] == depth])
            ypierce = np.append(ypierce, tr.pierce['y'][tr.pierce['z'] == depth])
            seis = np.append(seis, tr.pierce['seis'][tr.pierce['z'] == depth])
            trace_id = np.append(trace_id, tr.trace_id)
            bazs = np.append(bazs, tr.stats.sac.baz)
        except:
            pass


    # clean all nan positions
    xpierce = xpierce[np.where(~np.isnan(seis))]
    ypierce = ypierce[np.where(~np.isnan(seis))]
    seis = seis[np.where(~np.isnan(seis))]
    trace_id = trace_id[np.where(~np.isnan(seis))]
    bazs = bazs[np.where(~np.isnan(seis))]


    # set average, bootstrap standard deviation, hit_count,radius, and baz_dist vectors correlated to bin center position
    count = np.ones((len(COORDS),1))*0
    count_std = np.ones((len(COORDS),1))*0
    radius= np.ones((len(COORDS),1))*np.nan
    radius_std= np.ones((len(COORDS),1))*np.nan
    seis_avg = np.ones((len(COORDS),1))*np.nan
    seis_bstd = np.ones((len(COORDS),1))*np.nan
    baz_conc = np.ones((len(COORDS),1))*np.nan
    baz_std = np.ones((len(COORDS),1))*np.nan

    gauss_width = bin_min_gauss
    bin_sm = bin_min_sm

    # construct KDTree from pierce points within depth layer
    pierce_tree = KDTree(np.c_[xpierce, ypierce])

    # find indexes of pierce points that fall within binning range for each node coordinate
    # do the same for a metric of number of hits within one std
    hits = pierce_tree.query_ball_point(COORDS, bin_sm)
    hits_std = pierce_tree.query_ball_point(COORDS, gauss_width)

    # iterate through each grid coordinate position, and append a list of contributing trace_ids to each coordinate
    for i, coord in enumerate(COORDS):



        if len(hits[i]) < hits_per_bin:
            continue


        else:
            dists = np.array([(np.sqrt((xpierce[k] - coord[0])**2 + (ypierce[k] - coord[1])**2)) for k in hits[i]])
            seis_to_avg = seis[hits[i]]
            baz_to_var = bazs[hits[i]]

            # list for recording trace_ids that contribute to bin


            gauss = scipy.stats.norm(0, gauss_width)
            weights = gauss.pdf(dists)



            seis_avg_bin, seis_std_bin = Bootstrap(seis_to_avg, weights, 100)
            baz_conc_bin, baz_std_bin = circular_variability(baz_to_var, weights = None, axial = False)

            seis_avg[i] = seis_avg_bin
            seis_bstd[i] = seis_std_bin
            radius[i] = bin_sm
            radius_std[i] = gauss_width
            count[i] = len(hits[i])
            count_std[i] = len(hits_std[i])
            baz_conc[i] = baz_conc_bin
            baz_std[i] = baz_std_bin


    t2 = time.perf_counter()




    return depth, seis_avg.reshape(gridx.shape), seis_bstd.reshape(gridx.shape), count.reshape(gridx.shape), count_std.reshape(gridx.shape), radius.reshape(gridx.shape), radius_std.reshape(gridx.shape), baz_conc.reshape(gridx.shape), baz_std.reshape(gridx.shape)




def Fresnel_Zone_Binning(depth):

    time.sleep(0.1*np.abs(depth))
    print("\r", end="")
    print("Solving CCP: {:.1%} ".format(depth/z_max), end="")

    t1 = time.perf_counter()

    # set coordinate grid locally within process core
    gridx, gridy = np.meshgrid(x, y)
    COORDS = np.c_[gridx.reshape(np.size(gridx), -1), gridy.reshape(np.size(gridy), -1)]


    # pull pierce point data of each ray
    xpierce = ypierce = seis = bazs = np.array([])
    trace_id = np.array([], dtype = int)
    for tr in st.copy():
        try:
            xpierce = np.append(xpierce, tr.pierce['x'][tr.pierce['z'] == depth])
            ypierce = np.append(ypierce, tr.pierce['y'][tr.pierce['z'] == depth])
            seis = np.append(seis, tr.pierce['seis'][tr.pierce['z'] == depth])
            trace_id = np.append(trace_id, tr.trace_id)
            bazs = np.append(bazs, tr.stats.sac.baz)
        except:
            pass


    # clean all nan positions
    xpierce = xpierce[np.where(~np.isnan(seis))]
    ypierce = ypierce[np.where(~np.isnan(seis))]
    seis = seis[np.where(~np.isnan(seis))]
    trace_id = trace_id[np.where(~np.isnan(seis))]
    bazs = bazs[np.where(~np.isnan(seis))]


    # set average, bootstrap standard deviation, hit_count,radius, and baz_dist vectors correlated to bin center position
    count = np.ones((len(COORDS),1))*0
    count_std = np.ones((len(COORDS),1))*0
    radius = np.ones((len(COORDS),1))*np.nan
    radius_std = np.ones((len(COORDS),1))*np.nan
    seis_avg = np.ones((len(COORDS),1))*np.nan
    seis_bstd = np.ones((len(COORDS),1))*np.nan
    baz_conc = np.ones((len(COORDS),1))*np.nan
    baz_std = np.ones((len(COORDS),1))*np.nan


    # construct KDTree from pierce points within depth layer
    pierce_tree = KDTree(np.c_[xpierce, ypierce])



    vsu =  vs[z == depth]


    T = 1/ (G / 2)
    ts = depth / vsu
    D = vsu * np.sqrt((ts + (T/2))**2 - ts**2)

    gauss_width = np.round(D, 2)
    bin_sm = D * 3



    gauss_width[np.isnan(gauss_width)] = bin_min_gauss
    bin_sm[np.isnan(bin_sm)] = bin_min_sm

    gauss_width[gauss_width < bin_min_gauss] = bin_min_gauss
    bin_sm[bin_sm < bin_min_sm] = bin_min_sm

    # gauss_width = np.random.uniform(2, 40, size = len(COORDS))
    # bin_sm = np.random.uniform(2, 40, size = len(COORDS))




    # find indexes of pierce points that fall within binning range for each node coordinate
    # do the same for a metric of number of hits within one std
    hits = pierce_tree.query_ball_point(COORDS, bin_sm)
    hits_std = pierce_tree.query_ball_point(COORDS, gauss_width)

    # iterate through each grid coordinate position, and append a list of contributing trace_ids to each coordinate
    for i, coord in enumerate(COORDS):

        if len(hits[i]) < hits_per_bin:
            continue


        else:
            dists = np.array([(np.sqrt((xpierce[k] - coord[0])**2 + (ypierce[k] - coord[1])**2)) for k in hits[i]])
            seis_to_avg = seis[hits[i]]
            baz_to_var = bazs[hits[i]]

            # list for recording trace_ids that contribute to bin


            gauss = scipy.stats.norm(0, gauss_width)
            weights = gauss.pdf(dists)



            seis_avg_bin, seis_std_bin = Bootstrap(seis_to_avg, weights, 100)
            baz_conc_bin, baz_std_bin = circular_variability(baz_to_var, weights = None, axial = False)

            seis_avg[i] = seis_avg_bin
            seis_bstd[i] = seis_std_bin
            radius[i] = bin_sm
            radius_std[i] = gauss_width
            count[i] = len(hits[i])
            count_std[i] = len(hits_std[i])
            baz_conc[i] = baz_conc_bin
            baz_std[i] = baz_std_bin


    t2 = time.perf_counter()




    return depth, seis_avg.reshape(gridx.shape), seis_bstd.reshape(gridx.shape), count.reshape(gridx.shape), count_std.reshape(gridx.shape), radius.reshape(gridx.shape), radius_std.reshape(gridx.shape), baz_conc.reshape(gridx.shape), baz_std.reshape(gridx.shape)






process_start = time.perf_counter()


CCP = np.empty((len(y), len(x), len(z)), dtype = 'float32'); CCP[:] = np.nan    # records bootstrapped RF Amplitude
HITS = np.empty((len(y), len(x), len(z)),dtype = 'int32'); HITS[:] = 0          # records number of bin_hits in bin_sm range
HITS_STD = np.empty((len(y), len(x), len(z)),dtype = 'int32'); HITS_STD[:] = 0  # records number of hits in bin_gauss range
BSTD = np.empty((len(y), len(x), len(z)),dtype = 'float32'); BSTD[:] = np.nan   # records standard deviation of RF amp. in bin_sm range
RADS = np.empty((len(y), len(x), len(z)), dtype = 'float32'); RADS[:] = np.nan  # records bin_sm (size of search radius)
RADS_STD = np.empty((len(y), len(x), len(z)), dtype = 'float32'); RADS_STD[:] = np.nan  # records bin_sm (size of search radius)
BAZ_CONC = np.empty((len(y), len(x), len(z)), dtype = 'float32'); BAZ_CONC[:] = np.nan # records concentration of baz-azimuths of rays in each bin
BAZ_STD = np.empty((len(y), len(x), len(z)), dtype = 'float32'); BAZ_STD[:] = np.nan # records standard deviation of baz-azimuths of rays in each bin

with concurrent.futures.ProcessPoolExecutor(max_workers = 24) as executor:
    # layers = executor.map(Gaussian_Distance_Weighted, z)
    layers = executor.map(Fresnel_Zone_Binning, z)

    print('Writing CCP...')
    for layer in layers:
        CCP[:,:,np.where(z==layer[0])[0][0]] = layer[1]
        BSTD[:,:,np.where(z==layer[0])[0][0]] = layer[2]
        HITS[:,:,np.where(z==layer[0])[0][0]] = layer[3]
        HITS_STD[:,:,np.where(z==layer[0])[0][0]] = layer[4]
        RADS[:,:,np.where(z==layer[0])[0][0]] = layer[5]
        RADS_STD[:,:,np.where(z==layer[0])[0][0]] = layer[6]
        BAZ_CONC[:,:,np.where(z==layer[0])[0][0]] = layer[7]
        BAZ_STD[:,:,np.where(z==layer[0])[0][0]] = layer[8]





end_time = time.perf_counter()

print('Total Processing time: {} seconds'.format(end_time - process_start))


lat = south + km_to_latitude(y)
lon = west + x/dd





model = xr.Dataset(data_vars = {'Amplitude':(['depth',  'latitude', 'longitude'], np.swapaxes(CCP.T, 2,1)),
                                'Bin_Hits':(['depth',  'latitude', 'longitude'], np.swapaxes(HITS.T, 2,1)),
                                'Bin_Hits_std':(['depth',  'latitude', 'longitude'], np.swapaxes(HITS_STD.T, 2,1)),
                                'Bootstrap_std':(['depth',  'latitude', 'longitude'], np.swapaxes(BSTD.T, 2,1)),
                                'Bin_Radius':(['depth',  'latitude', 'longitude'], np.swapaxes(RADS.T, 2,1)),
                                'Bin_std_Radius':(['depth',  'latitude', 'longitude'], np.swapaxes(RADS_STD.T, 2,1)),
                                'BAZ_Concentration':(['depth',  'latitude', 'longitude'], np.swapaxes(BAZ_CONC.T, 2,1)),
                                'BAZ_STD':(['depth',  'latitude', 'longitude'], np.swapaxes(BAZ_STD.T, 2,1)),

                                },
                          coords = {'depth': -z, 'longitude': lon, 'latitude': lat},
                          attrs={'Velocity Model': v_model_1D,
                                 'description': 'CCP Stack of {} with Fresnel Zone Binning, with using algorithm of Ryan Porter (NAU)'.format(region),
                                 'Node Spacing': bin_sz, 'Bin Min Smoothing': bin_min_sm,'Bin Gauss': bin_min_gauss,
                                 'BAZ Range': '{}-{}'.format(baz_range[0], baz_range[1]),
                                 'Reference_File': 'Iterdecon_Summary.csv',
                                 'x_center': lon,
                                 'y_center': lat})



model_encoding = {'depth': {'dtype': 'float32', '_FillValue': None },
            'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'Amplitude': {'dtype': 'float32', '_FillValue': None , 'zlib': False},
            'Bin_Hits': {'dtype': 'int', '_FillValue': 0 , 'zlib': False},
            'Bin_Hits_std': {'dtype': 'int', '_FillValue': 0 , 'zlib': False},
            'Bootstrap_std': {'dtype': 'float32', '_FillValue': None , 'zlib': False},
            'Bin_Radius': {'dtype': 'float32', '_FillValue': None , 'zlib': False},
            }



model.to_netcdf('../DATA/MAPPING/nc_models/CCP_{}_G-{}_FZB_IASP91-40km.nc'.format(region, G), encoding = model_encoding)









