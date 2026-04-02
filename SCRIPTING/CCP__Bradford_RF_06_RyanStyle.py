#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 16:55:29 2025

@author: bradford
"""


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
from scipy.stats import norm
import concurrent.futures


import geopandas as gpd



active_dir = './'

os.chdir(active_dir)



shear_model = 'BlendedModel_0.7APVC-ANT_0.3FWT_MantleSpec_RE_Smoothed.nc'
vpvs_model = 'CentralAndes_Zoned_VpVs_Arc-1.85_RE_Smoothed.nc'



shear = xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(shear_model))
vpvs = xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(vpvs_model))

region = 'TANGO_North'

CCP_dir = '../DATA/{}/DATA_CCP/'.format(region)
CCP_Files = os.listdir(os.path.join(CCP_dir))


"""

Use of this version:

    Take a shear wave velocity model, and apply a constant bulk vp/vs in the crust and mantle
    at a set contour



"""



#%% Grab Data

G = 2.5


itd_pass = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary_RE.csv',keep_default_na=False)

itd_pass = itd_pass[(itd_pass.Accept == 'Auto QC Pass') | (itd_pass.Accept == 'Man QC Pass')][itd_pass.G == G][itd_pass.Region == region][itd_pass.Phase == 'P']#[itd_pass.Net != '1X']
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



select_region = False   # manually select a region
use_region =  True      # use a saved region
save_region = False     # save the region

dzi = 0.5   # depth increment in km
z_max = 200 # max depth in km

bin_sz = 5 # node spacing in km
bin_min_sm = 24 # base radius of bin in km around node, will increase to find more pierce points if allowed
bin_min_gauss = 8
hits_per_bin = 1    # Required number of hits for valid bin



minv = 2.0  # lowest allowable shear wave velocity in model, any point lower will be raised to this
max_vpvs = 1.82 # maximum allowable Vp/Vs in model
min_vpvs = 1.72 # lowest allowable Vp/Vs in model
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

        north, south, east, west = np.loadtxt('../DATA/MAPPING/TANGO_North_CCP_GridArea.txt')
        # north, south, east, west = np.loadtxt('../DATA/MAPPING/{}_CCP_GridArea.txt'.format(region))
        north, south, east, west = [-21.7, -24.5, -63.5, -70.8] # temp
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
i = 0
len_st = len(st)
for tr in st:
    i+=1
    baz = tr.stats.sac.baz
    print("\r", end="")
    print("Filtering BAZ: {:.1%} ".format(i/len_st), end="")

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
gmt_region = '{}/{}/{}/{}'.format(west,east,south, north)
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

    tr.x = stx
    tr.y = sty
    tr.z = stz




# set z to level at the maximum station elevation
z = np.arange(np.round((-np.max(stzs)) * 1/dzi) * dzi, z_max + dzi, dzi).astype(np.float16)

del stzs, st_len, k, ELEV



fig = pygmt.Figure()    # initialize the map

with fig.subplot(nrows = 1, ncols = 1, figsize=('15c', '15c')):
    with fig.set_panel(panel = 0):
        prj = 'X15c/15c'            # set the map projection
        with pygmt.config(FONT="14p", FORMAT_GEO_MAP="ddd.xx"):


            fig.basemap(region=[x.min(), x.max(), y.min(), y.max()], projection=prj, frame='afg')


            fig.plot(x = STX, y = STY, style='i0.23c', fill = 'gold', transparency = 50, pen='0.4p,black')

fig.show()



gc.collect()


#%% Ray Tracing


t_offset = 10   # time from tr time = 0 where p-wave peak should be centered


def Trace_Pierce_Point(trace_index):
    print("\r", end="")
    print("Tracing Rays: {:.1%} ".format(trace_index/len(st)), end="")


    t1 = time.perf_counter()




    ### Read in and convert velocity data into localized grid
    ### Perform this within function so we do not force several gigabytes of
    ### memory transfer to CPU cores
    ### Flatten the velocity model(s) to share the km-grid space
    with xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(shear_model)) as shear:
        s_lat = shear.latitude.values
        s_lon = shear.longitude.values
        sz = shear.depth.values
        s_val = shear.vs.values

        sy = latitude_to_km(s_lat - south)
        sx = dd * (s_lon - west)



    SX, SY = np.meshgrid(sx, sy)
    SCOORDS = np.c_[SX.reshape(SX.size, -1, order = 'F'), SY.reshape(SY.size, -1, order = 'F')]
    VS = s_val.T.reshape(len(SCOORDS), len(s_val), order = 'F')

    # set up KDTree for distance-searches
    s_tree = KDTree(SCOORDS)



    with xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(vpvs_model)) as k:

        k_lat = k.latitude.values
        k_lon = k.longitude.values
        kz = k.depth.values
        k_val = k.VpVs.values

        ky = latitude_to_km(k_lat - south)
        kx = dd * (k_lon - west)


    KX, KY = np.meshgrid(kx, ky)
    KCOORDS = np.c_[KX.reshape(KX.size, -1, order = 'F'), KY.reshape(KY.size, -1, order = 'F')]
    K = k_val.T.reshape(len(KCOORDS), len(k_val), order = 'F')

    # set up KDTree for distance-searches
    k_tree = KDTree(KCOORDS)









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


    # Pull velocity data at station location on grid using
    # vsu = s_model.sel(x = stx, y = sty, method = 'nearest').vs.values.copy()
    # vd = s_model.z.values.copy()

    vsu_idx = s_tree.query([stx, sty], k = 1)[1]
    vsu = VS[vsu_idx].copy()


    # ksu = k_model.sel(x = stx, y = sty, method = 'nearest').vpvs.values.copy()
    # kd = k_model.z.values.copy()

    ksu_idx = k_tree.query([stx, sty], k = 1)[1]
    ksu = K[ksu_idx].copy()



    # enforce global minimums by setting to next minimum in vector
    vsu[vsu <= minv] = vsu[vsu > minv].min()

    ksu[ksu <= min_vpvs] = ksu[ksu > min_vpvs].min()
    ksu[ksu >= max_vpvs] = ksu[ksu < max_vpvs].max()

    # Interpolate vs and k at mid-points between layers in z
    v_interp_func = interp1d(sz, vsu, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    k_interp_func = interp1d(kz, ksu, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')

    vs = v_interp_func(z + 0.5*dzi)
    ku = k_interp_func(z+ 0.5*dzi)

    # multiple vs*k to obtain vp
    vp = vs * ku


    # Solve for incident angle scaling along vector
    i_ang = np.rad2deg(np.arcsin(p*vs))
    a_scale = ref_angle / i_ang

    # Solve for qa and qb coefficients
    qa = np.sqrt((1/(vs)**2) - p**2)
    qb = np.sqrt((1/(vp)**2) - p**2)

    # solve for travel time within each depth interval
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

    gc.collect()

    return trace_index, xpierce, ypierce, trace_z, seisout





# note that the trace_id variable here only corresponds to trace index within st, and is not the trace_id dictionary within each trace
trace_id = np.arange(0, len(st), 1)
with concurrent.futures.ProcessPoolExecutor(max_workers = 24) as executor:
    pierce_traces = executor.map(Trace_Pierce_Point, trace_id)

    for ray in pierce_traces:
        st[ray[0]].pierce = {'x': ray[1], 'y': ray[2], 'z': ray[3], 'seis': ray[4]}






gc.collect()

#%% Plot the Piercing Points



pierce_depth = 140
print('Plotting Pierce Points at {} km depth'.format(pierce_depth))


save_pierce = 0


try:
    contributing = pd.read_csv('../DATA/SPREADSHEETS/Contributing_Stations.csv')
except:
    pass


x_depth = y_depth = stx = sty = np.array([])
n_x_depth = n_y_depth = s_x_depth = s_y_depth = np.array([])
for tr in st:

    # try:
    #     code = '{}-{}'.format(tr.stats.network, tr.stats.station)
    #     if code not in contributing.Code.to_list():
    #         continue
    # except:
    #     pass

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


            fig.basemap(region=[-40, 800, -180, 660], projection='X15c/15c', frame='a100g50')

            fig.plot(x = x_depth, y = y_depth, style = 'x0.2c', pen = '0.7p,black')
            fig.plot(x = n_x_depth, y = n_y_depth, style = 'c0.2c', pen = '0.3p,blue')
            # fig.plot(x = s_x_depth, y = s_y_depth, style = 'c0.2c', pen = '0.3p,red')

            fig.plot(x = 0, y = 0, style = 'c{}c'.format(bin_min_sm*2*(15/840)), fill = 'yellow', pen = '1p,black', transparency = 30)
            fig.plot(x = 0, y = 0, style = 'c{}c'.format((bin_min_gauss)*2*(15/840)), fill = 'darkorange', transparency = 30)


            fig.plot(x = [x.min(), x.max(), x.max(), x.min(), x.min()], y = [y.min(), y.min(), y.max(), y.max(), y.min()], pen = '1p,red,--')
            try:
                fig.plot(x = stx, y =  sty,  style='i0.25c',fill = 'yellow', pen='black')
            except:
                pass




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
joints = np.loadtxt('../DATA/MAPPING/{}_CCP_Line.txt'.format(region), delimiter = ',')

contributing_stations = []
contributing_events = []


line_y = np.round(latitude_to_km(joints[:,1] - south), 4)
line_x = np.round(dd*(joints[:,0]  - west), 4)

line = np.c_[line_x, line_y]
line = np.linspace(line[0], line[1], 800)


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




def Gaussian_Distance_Weighted(depth):

    time.sleep(0.1*np.abs(depth))
    print("\r", end="")
    print("Solving CCP: {:.1%} ".format(depth/z_max), end="")

    t1 = time.perf_counter()

    # set coordinate grid locally within process core
    gridx, gridy = np.meshgrid(x, y)
    COORDS = np.c_[gridx.reshape(np.size(gridx), -1), gridy.reshape(np.size(gridy), -1)]


    xpierce = ypierce = seis = np.array([])
    trace_id = np.array([], dtype = int)
    for tr in st.copy():
        xpierce = np.append(xpierce, tr.pierce['x'][tr.pierce['z'] == depth])
        ypierce = np.append(ypierce, tr.pierce['y'][tr.pierce['z'] == depth])
        seis = np.append(seis, tr.pierce['seis'][tr.pierce['z'] == depth])
        trace_id = np.append(trace_id, tr.trace_id)


    # clean all nan positions
    xpierce = xpierce[np.where(~np.isnan(seis))]
    ypierce = ypierce[np.where(~np.isnan(seis))]
    seis = seis[np.where(~np.isnan(seis))]
    trace_id = trace_id[np.where(~np.isnan(seis))]


    gauss_width = bin_min_gauss



    # set average, bootstrap standard deviation, hit_count,and radius vectors correlated to bin center position
    count = np.ones((len(COORDS),1))*0
    count_std = np.ones((len(COORDS),1))*0
    radius= np.ones((len(COORDS),1))*np.nan
    seis_avg = np.ones((len(COORDS),1))*np.nan
    seis_bstd = np.ones((len(COORDS),1))*np.nan

    contributions = np.empty(len(COORDS), dtype = 'object')


    # construct KDTree from pierce points within depth layer
    pierce_tree = KDTree(np.c_[xpierce, ypierce])

    # find indexes of pierce points that fall within binning range for each node coordinate
    # do the same for a metric of number of hits within one std
    hits = pierce_tree.query_ball_point(COORDS, bin_min_sm)
    hits_std = pierce_tree.query_ball_point(COORDS, gauss_width)

    # iterate through each grid coordinate position, and append a list of contributing trace_ids to each coordinate
    for i, coord in enumerate(COORDS):


        contributions[i] = ','.join(str(s) for s in trace_id[hits[i]])

        if len(hits[i]) < hits_per_bin:
            # append empty string to contributions array for coordinate position
            continue


        else:
            dists = np.array([(np.sqrt((xpierce[k] - coord[0])**2 + (ypierce[k] - coord[1])**2)) for k in hits[i]])
            seis_to_avg = seis[hits[i]]

            # list for recording trace_ids that contribute to bin


            gauss = norm(0, gauss_width)
            weights = gauss.pdf(dists)



            seis_avg_bin, seis_std_bin = Bootstrap(seis_to_avg, weights, 50)

            seis_avg[i] = seis_avg_bin
            seis_bstd[i] = seis_std_bin
            radius[i] = bin_min_sm
            count[i] = len(hits[i])
            count_std[i] = len(hits_std[i])


    t2 = time.perf_counter()




    return depth, seis_avg.reshape(gridx.shape), seis_bstd.reshape(gridx.shape), count.reshape(gridx.shape), count_std.reshape(gridx.shape), radius.reshape(gridx.shape), contributions.reshape(gridx.shape)






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
    baz_std = np.ones((len(COORDS),1))*np.nan


    # construct KDTree from pierce points within depth layer
    pierce_tree = KDTree(np.c_[xpierce, ypierce])

    ### Pull veloicty data through xarray interpolation of coordinates
    with xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(shear_model)) as mod:

        shear = mod.vs

        m_lat = mod.latitude.values
        m_lon = mod.longitude.values
        mz = mod.depth.values

        my = latitude_to_km(m_lat - south)
        mx = dd * (m_lon - west)

        shear = shear.assign_coords({'longitude': mx, 'latitude': my})
        vsu = shear.interp(longitude = x, latitude = y, depth = float(depth), method = 'nearest', kwargs={"fill_value": None}).values.astype("float32")


    # clip global minimums
    vsu = np.clip(vsu, minv, np.nanmax(vsu))


    T = 1/ (G)
    ts = depth / vsu
    D = vsu * np.sqrt((ts + (T/2))**2 - ts**2)
    D = D.reshape(np.size(gridx), -1).flatten()

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


            gauss = norm(0, gauss_width[i])
            weights = gauss.pdf(dists)



            seis_avg_bin, seis_std_bin = Bootstrap(seis_to_avg, weights, 100)

            seis_avg[i] = seis_avg_bin
            seis_bstd[i] = seis_std_bin
            radius[i] = bin_sm[i]
            radius_std[i] = gauss_width[i]
            count[i] = len(hits[i])
            count_std[i] = len(hits_std[i])


    t2 = time.perf_counter()




    return depth, seis_avg.reshape(gridx.shape), seis_bstd.reshape(gridx.shape), count.reshape(gridx.shape), count_std.reshape(gridx.shape), radius.reshape(gridx.shape), radius_std.reshape(gridx.shape)









process_start = time.perf_counter()


CCP = np.empty((len(y), len(x), len(z)), dtype = 'float32'); CCP[:] = np.nan
HITS = np.empty((len(y), len(x), len(z)),dtype = 'int32'); HITS[:] = 0
HITS_STD = np.empty((len(y), len(x), len(z)),dtype = 'int32'); HITS_STD[:] = 0
BSTD = np.empty((len(y), len(x), len(z)),dtype = 'float32'); BSTD[:] = np.nan
RADS = np.empty((len(y), len(x), len(z)), dtype = 'float32'); RADS[:] = np.nan
# CONTRIBUTIONS = np.empty((len(y), len(x), len(z)), dtype = 'object')

with concurrent.futures.ProcessPoolExecutor(max_workers = 24) as executor:
    layers = executor.map(Fresnel_Zone_Binning, z)

    print('Writing CCP...')
    for layer in layers:
        CCP[:,:,np.where(z==layer[0])[0][0]] = layer[1]
        BSTD[:,:,np.where(z==layer[0])[0][0]] = layer[2]
        HITS[:,:,np.where(z==layer[0])[0][0]] = layer[3]
        HITS_STD[:,:,np.where(z==layer[0])[0][0]] = layer[4]
        RADS[:,:,np.where(z==layer[0])[0][0]] = layer[5]
        # CONTRIBUTIONS[:,:,np.where(z==layer[0])[0][0]] = layer[6]






end_time = time.perf_counter()

print('Total Processing time: {} seconds'.format(end_time - process_start))


lat = south + km_to_latitude(y)
lon = west + x/dd





model = xr.Dataset(data_vars = {'Amplitude':(['depth',  'latitude', 'longitude'], np.swapaxes(CCP.T, 2,1)),
                                'Bin_Hits':(['depth',  'latitude', 'longitude'], np.swapaxes(HITS.T, 2,1)),
                                'Bin_Hits_std':(['depth',  'latitude', 'longitude'], np.swapaxes(HITS_STD.T, 2,1)),
                                'Bootstrap_std':(['depth',  'latitude', 'longitude'], np.swapaxes(BSTD.T, 2,1)),
                                'Bin_Radius':(['depth',  'latitude', 'longitude'], np.swapaxes(RADS.T, 2,1)),
                                },
                          coords = {'depth': -z, 'longitude': lon, 'latitude': lat},
                          attrs={'Velocity Model': shear_model, 'VpVs Model': vpvs_model,
                                 'description': 'CCP Stack of {} with Gaussian Weighted Distance, with using algorithm of Ryan Porter (NAU)'.format(region),
                                 'vmin': minv, 'VpVs Range': '{}-{}'.format(min_vpvs, max_vpvs),
                                 'Node Spacing': bin_sz, 'Bin Min Smoothing': bin_min_sm, 'Bin Min Gauss': bin_min_gauss,
                                 'Hit Requirement': hits_per_bin, 'BAZ Range': '{}-{}'.format(baz_range[0], baz_range[1]),
                                 'Reference_File': 'Iterdecon_Summary_RE.csv',
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



model.to_netcdf('../DATA/MAPPING/nc_models/CCP_{}_G-{}_FZB_RE.nc'.format(region, G), encoding = model_encoding)







#%% Fresnel Zone Binning, investigate bin size
with xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(shear_model)) as mod:

    shear = mod.vs
    vsu = shear.interp(longitude = -68, latitude = -23, method = 'nearest', kwargs={"fill_value": None}).values.astype("float32")
    depth = shear.depth.values

# clip global minimums
vsu = np.clip(vsu, minv, np.nanmax(vsu))


T = 1/ (G)
ts = depth / vsu
D = vsu * np.sqrt((ts + (T/2))**2 - ts**2)

gauss_width = np.round(D, 2)
bin_sm = D * 3

plt.plot(D)




#%% Sanity Check

### Does xarray breakdown into meshgrid of velocity models produce the same
### result as just using xarray.interpolate with nearest neighbor?


# converter for longitudinal distance, used for flattening
dd = deg_to_km(np.mean([arc_length_on_earth(south,west,south,east), arc_length_on_earth(north,west,north,east)])/(east - west))

# Parameterize grid as boxes

ydist = latitude_to_km(north - south)
xdist = dd*(east-west)


# construct grid as x-y in km
x = np.arange(0, xdist+bin_sz,bin_sz) # x values
y = np.arange(0,ydist+bin_sz,bin_sz) # y values

X,Y = np.meshgrid(x, y)

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    prj = 'X15c/15c'            # set the map projection

    fig.basemap(region=[x.min(), x.max(), y.min(), y.max()], projection=prj, frame='afg')


    fig.plot(x = STX, y = STY, style='i0.23c', fill = 'gold', transparency = 50, pen='0.4p,black')

fig.show()




stx = 400
sty = 200



############ 1) show output from meshgrid search

### Flatten the velocity model(s) to share the km-grid space
with xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(shear_model)) as shear:
    s_lat = shear.latitude.values
    s_lon = shear.longitude.values
    sz = shear.depth.values
    s_val = shear.vs.values

    sy = latitude_to_km(s_lat - south)
    sx = dd * (s_lon - west)



SX, SY = np.meshgrid(sx, sy)
SCOORDS = np.c_[SX.reshape(SX.size, -1, order = 'F'), SY.reshape(SY.size, -1, order = 'F')]
VS = s_val.T.reshape(len(SCOORDS), len(s_val), order = 'F')

# set up KDTree for distance-searches
s_tree = KDTree(SCOORDS)

with xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(vpvs_model)) as k:

    k_lat = k.latitude.values
    k_lon = k.longitude.values
    kz = k.depth.values
    k_val = k.VpVs.values

    ky = latitude_to_km(k_lat - south)
    kx = dd * (k_lon - west)


KX, KY = np.meshgrid(kx, ky)
KCOORDS = np.c_[KX.reshape(KX.size, -1, order = 'F'), KY.reshape(KY.size, -1, order = 'F')]
K = k_val.T.reshape(len(KCOORDS), len(k_val), order = 'F')

# set up KDTree for distance-searches
k_tree = KDTree(KCOORDS)



vsu_idx = s_tree.query([stx, sty], k = 1)[1]
vsu = VS[vsu_idx].copy()


# ksu = k_model.sel(x = stx, y = sty, method = 'nearest').vpvs.values.copy()
# kd = k_model.z.values.copy()

ksu_idx = k_tree.query([stx, sty], k = 1)[1]
ksu = K[ksu_idx].copy()



# enforce global minimums by setting to next minimum in vector
vsu[vsu <= minv] = vsu[vsu > minv].min()

ksu[ksu <= min_vpvs] = ksu[ksu > min_vpvs].min()
ksu[ksu >= max_vpvs] = ksu[ksu < max_vpvs].max()

# Interpolate vs and k at mid-points between layers in z
v_interp_func = interp1d(sz, vsu, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
k_interp_func = interp1d(kz, ksu, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')

vs = v_interp_func(z + 0.5*dzi)
ku = k_interp_func(z+ 0.5*dzi)


fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    prj = 'X6c/-15c'            # set the map projection
    fig.basemap(region=[2, 7, z.min(), z.max()], projection=prj, frame='afg')
    fig.plot(x = vs.astype("float64"), y = z.astype("float64"), pen = '1p,blue')

    fig.shift_origin(xshift = '7c')

    fig.basemap(region=[1.6, 2.0, z.min(), z.max()], projection=prj, frame='afg')
    fig.plot(x = ku.astype("float64"), y = z.astype("float64"), pen = '1p,blue')

fig.show()





################ 2) find vs and ku via xarray grid interpolation nearest neighbor



### Pull veloicty data through xarray interpolation of coordinates
with xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(shear_model)) as shear:
    s_lat = shear.latitude.values
    s_lon = shear.longitude.values
    sz = shear.depth.values

    sy = latitude_to_km(s_lat - south)
    sx = dd * (s_lon - west)

    shear = shear.assign_coords({'longitude': sx, 'latitude': sy})
    vsu = shear.interp(longitude = stx, latitude = sty, method = 'nearest', kwargs={"fill_value": None}).vs.values




### Pull veloicty data through xarray interpolation of coordinates
with xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(vpvs_model)) as k:
    k_lat = k.latitude.values
    k_lon = k.longitude.values
    kz = k.depth.values

    ky = latitude_to_km(k_lat - south)
    kx = dd * (k_lon - west)

    k = k.assign_coords({'longitude': kx, 'latitude': ky})
    ksu = k.interp(longitude = stx, latitude = sty, method = 'nearest', kwargs={"fill_value": None}).VpVs.values


# enforce global minimums by setting to next minimum in vector
vsu[vsu <= minv] = vsu[vsu > minv].min()

ksu[ksu <= min_vpvs] = ksu[ksu > min_vpvs].min()
ksu[ksu >= max_vpvs] = ksu[ksu < max_vpvs].max()


# Interpolate vs and k at mid-points between layers in z
v_interp_func = interp1d(sz, vsu, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
k_interp_func = interp1d(kz, ksu, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')

vs = v_interp_func(z + 0.5*dzi)
ku = k_interp_func(z+ 0.5*dzi)




fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    prj = 'X6c/-15c'            # set the map projection
    fig.basemap(region=[2, 7, z.min(), z.max()], projection=prj, frame='afg')
    fig.plot(x = vs.astype("float64"), y = z.astype("float64"), pen = '1p,blue')

    fig.shift_origin(xshift = '7c')

    fig.basemap(region=[1.6, 2.0, z.min(), z.max()], projection=prj, frame='afg')
    fig.plot(x = ku.astype("float64"), y = z.astype("float64"), pen = '1p,blue')

fig.show()



######## I am not going insane...





