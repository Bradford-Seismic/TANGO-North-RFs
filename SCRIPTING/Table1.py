# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 20:10:39 2025

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
from glob import glob


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
contributing_events = pd.read_csv('../DATA/SPREADSHEETS/Contributing_Events.csv')

stat_table = stat_table[stat_table.Code.isin(contributing_stations.Code)].reset_index(drop = True)



files = glob('../DATA/Tango_North/DATA_CCP/*02.5.itr')


nets = []
num_stations = []
num_RFs = []
for net in np.unique(stat_table.net):
    stats = stat_table[stat_table.net == net]
    
    # gather stations used in contributing
    num_station = len(stats)
    
    
    # gather events detected in network
    events_in_data = glob('../DATA/Tango_North/DATA_CCP/{}*.02.5.itr'.format(net))
    
    file_board = pd.DataFrame({'net': [net for i in range(len(events_in_data))],
                               'stat': [file.split('\\')[-1].split('.')[1] for file in events_in_data],
                               'event': [file.split('\\')[-1].split('.')[3] for file in events_in_data]})


    num_events = len(file_board)
    
    
    # gather passing RF data in contributing
    
    file_board = file_board[file_board.event.isin(contributing_events.Events)]

    num_RF = len(file_board)
    
    
    nets.append(net)
    num_stations.append(num_station)
    num_RFs.append(num_RF)


dat = pd.DataFrame({'net': nets, 'Num. Stations': num_stations, 'Num. RFs': num_RFs})

dat.to_csv('../DATA/SPREADSHEETS/Supplementary_Table1.csv')