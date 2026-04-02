#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:00:17 2024

@author: bradford
"""


import numpy as np
import pandas as pd
import os
import sys
from obspy.signal import rotate
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
import cv2
import math

warnings.simplefilter('ignore', category = UserWarning)

# # our home folder where everything is referred to
# active_dir = '/tango/bradford/Tango_2/SCRIPTING_LINUX/'
# # the folder where things will go
# output_dir = '/tango/bradford/Tango_2/DATA/'
# # the subfolder within the output_dir that contains are raw data
# raw_file_loc = '/tango/bradford/Tango_2/DATA/raw_mseed'
# # change our position to the active_dir
# os.chdir(active_dir)


active_dir = './'

output_dir = '../DATA/'
raw_file_loc = '../DATA/raw_mseed/'
os.chdir(active_dir)


model = TauPyModel(model="iasp91")




"""
                Part 2 of this journey
The user should now have three things:
    1) a Download_Summary.csv file that contains the event-station information
    and the donwloaded file data
    
    2) a big set of downloaded .mseed files
    
    3) a big set of downloaded instrument response files with the .xml extension

    
We will now move these files accordining to where we want to process them
and save them in the more user-friendly form of SAC


Prior to starting, in your DATA directory, create subdirectories of all the
locations of your project

Call them by whatever you feel is appropriate, but remember your list


"""


#%% Print download stats


## begin by reading our Download_Summary.csv file and print a quick summary
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')

P = len(downloads[downloads.PhaseType == 'P'])
PP = len(downloads[downloads.PhaseType == 'PP'])
PKP = len(downloads[downloads.PhaseType == 'PKP'])
print('Num Files downloaded: {}, P: {}, PP: {}, PKP: {}'.format(len(downloads), P, PP, PKP))



#%% Plot the downloaded stations



"""
    It's fun to look at our data and see how this changes as we process
    Let's start by making a map our study locations

"""

# first thing to handle, let's collect all of our station information
codes = stlas = stlos = np.array([])

for i in range(len(downloads)):
    if downloads.Network[i] + '.' + downloads.Station[i] not in codes:
        codes = np.append(codes, downloads.Network[i] + '.' + downloads.Station[i])
        stlas = np.append(stlas, downloads.STLA[i])
        stlos = np.append(stlos, downloads.STLO[i])
        



# Next, we want to have a PyGMT map, which gives us really good mapping result
# initialize our map this way

# region contains the bound extent of our data 
# organized here as 'west margin / east margin / south margin / north margin'
# region = '-75/-50/-40/-20'
edge_gap = 0.8
north, south, west, east = np.ceil(stlas.max())+edge_gap, np.floor(stlas.min())-edge_gap, np.floor(stlos.min())-edge_gap, np.ceil(stlos.max())+edge_gap
region = '{}/{}/{}/{}'.format(west, east, south, north)

fig = pygmt.Figure()    # initialize the map
prj = 'M15c'            # set the map projection


# download and use this grid dem file as our background
# for finer resolution, change the resolution to 15s
# but for large areas this can take awhile to process
grid = pygmt.datasets.load_earth_relief(resolution="15s", region=region)

# shade the map and make it look cool
shade = pygmt.grdgradient(grid=grid, azimuth="0/90", normalize="e1")


# make the basemap using our defined map projection and boundary
fig.basemap(region=region, projection=prj, frame='ag4')

# overlay the grid image DEM over the basemap
fig.grdimage(grid=grid, shading = shade, cmap = '../DATA/MAPPING/natural_mod.cpt', projection = prj, transparency = 60)

# overlay national borders, coastlines, and make the water blue
fig.coast(borders = ["1/0.7p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 10)



# now plot our data
fig.plot(x = stlos, y = stlas, style='i0.2c', fill = 'yellow', pen='black', label = 'Other Network')






"""
         Say we want to highlight specific networks, here, parse through the 
         download summary and location the specific networks we want to show off

"""
# fig.plot(x = downloads['STLO'][downloads['Network'].str.contains('XN')], y =downloads['STLA'][downloads['Network'].str.contains('XN')], style='i0.23c',fill = 'cyan', pen='black', label = 'TANGO Node')
fig.plot(x = downloads['STLO'][downloads['Network'].str.contains('1X')], y =downloads['STLA'][downloads['Network'].str.contains('1X')], style='i0.23c',fill = 'cyan', pen='black', label = 'TANGO Node')
fig.plot(x = downloads['STLO'][downloads['Network'].str.contains('XM')], y =downloads['STLA'][downloads['Network'].str.contains('XM')], style='i0.28c',fill = 'magenta', pen='0.6p,black', label = 'TANGO Broadband')
# fig.plot(x = downloads['STLO'][downloads['Network'].str.contains('ZA')][downloads['Station'].str.contains('RF')], y =downloads['STLA'][downloads['Network'].str.contains('ZA')][downloads['Station'].str.contains('RF')], style='i0.28c',fill = 'coral', pen='0.6p,black', label = 'REFUCA Broadband')



# Add additional informational text, such as country names
fig.text(x = -60, y=-22, text = 'Paraguay', font = '12p,Helvetica-Bold')
fig.text(x = -65.6, y=-20.5, text = 'Bolivia', font = '12p,Helvetica-Bold')
fig.text(x = -63, y=-30, text = 'Argentina', font = '12p,Helvetica-Bold')
fig.text(x = -70, y=-27, text = 'Chile', font = '12p,Helvetica-Bold')


# # visual aid for seeing lateral range in kilometers, you may otherwise ignore this
# fig.plot(x = [-63, -63 + 317/111.32],y =  [-30, -30])


# include a legend, and show the figure in a plot window
fig.legend()
fig.show()


fig.savefig(os.path.join('../FIGURES/Prelim_DownloadedStation_Map.png'), dpi = 300)





#%% Redistribute data sets into Project Location directories

# If you have only a single area you are considering in this project,
# you may otherwise just think nothing of this step

# Otherwise:
    
"""
        Receiver Functions generally only sample data within the vertical column
        directly beneath a station
        
        However,
        Multiples of converting boundaries can sample a lateral area and interfere 
        with the signal we care about
        
        To accomodate, we want to include all data available around our target analysis
        point or line or area, 
        
        Generally, a lateral range of 3 times the depth to our deepest converter
        of interest should be our lateral range of observation
        
        See: http://eqseis.geosc.psu.edu/cammon/HTML/RftnDocs/rftn01.html
        For more details
    
"""



# command to calculate the distance in kilometers between two points given 
# in latitude/longitude coordinates
def haversine(lat1,lon1,lat2,lon2):
    lat1_rad=math.radians(lat1)
    lat2_rad=math.radians(lat2)
    lon1_rad=math.radians(lon1)
    lon2_rad=math.radians(lon2)
    delta_lat=lat2_rad-lat1_rad
    delta_lon=lon2_rad-lon1_rad
    a=math.sqrt((math.sin(delta_lat/2))**2+math.cos(lat1_rad)*math.cos(lat2_rad)*(math.sin(delta_lon/2))**2)
    d=2*6371*math.asin(a)
    return d


# an mseed channel may contain more than one bandwidth 
# here, we assign preference to bandwidths if that is the case
# 'D' is for nodal networks
# 'H' for high-broadband
# 'B' for standard broadband
# 'S' for short period sensor
def rank_channels(st):
    bands = np.array([])
    # get list of all st contained bands for mseed file
    for tr in st:
        bands = np.append(bands, tr.stats.channel[0])
    
    if 'D' in bands:
        st=st.select(channel='DP?')
    elif 'H' in bands:
        st=st.select(channel='HH?')
    elif 'B' in bands:
        st=st.select(channel='BH?')
    elif 'S' in bands:
        st=st.select(channel='SH?')
    else:
        return 'fail'
        
    
    return st





# perform an auto quality control for mseed files to ensure they are full
# of the correct data
def QC_Mseed(st, fname):
    # stream file QC
    # stream contains all H*,B*,D* band channels
    
    # remove if there are less than three channels 
    if len(st) < 3:
        print('\n Channel Content Error with file: {}'.format(fname))
        with open('../DATA/SPREADSHEETS/FileData_Summary.csv', 'a') as f:
            f.write('{}, (4) Channel Content\n'.format(fname))

        return 1    
    
    
    
    # remove if there is not a Z,N,E, or 1,2,Z, component
    comps = []
    for i in st:
        comps.append(i.stats.component)
    if not(('E' in comps) or ('1' in comps) or ('N' in comps) or ('2' in comps) or ('Z' in comps)):
        print('\n Channel Listing Error with file: {}'.format(fname))
        with open('../DATA/SPREADSHEETS/FileData_Summary.csv', 'a') as f:
            f.write('{}, (5) Channel Listing\n'.format(fname))

        return 1
    
    
            
            
            
    
    # remove if the data is broken, or poorly sampled
    for i in st:
        if i.stats.npts < 1000:
            print('\n NPTS Error with file: {}'.format(fname))
            with open('../DATA/SPREADSHEETS/FileData_Summary.csv', 'a') as f:
                f.write('{}, (6) NPTS\n'.format(fname))
    
            return 1
        
        
        
        
        
    # remove the data if the sampling rate is inappropriate for our uses
    for i in st:
        if i.stats.delta > 0.1:
            print('\n Sampling Rate Error with file: {}'.format(fname))
            with open('../DATA/SPREADSHEETS/FileData_Summary.csv', 'a') as f:
                f.write('{}, (7) delta\n'.format(fname))
    
            return 1
        
        
        
        
    # reject the data if the file is flat, 0 all throughout
    for i in st:
        if np.min(i.data) == np.max(i.data):
            print('\n Flat Values Error with file: {}'.format(fname))
            with open('../DATA/SPREADSHEETS/FileData_Summary.csv', 'a') as f:
                f.write('{}, (8) Flat Value\n'.format(fname))

            return 1
        
        
        
        
        
    # empty data, padded with null/0
    for i in st:
        if np.absolute(np.mean(i.data)) > 10000000:
            print('\n Null Data Error with file: {}'.format(fname))
            with open('../DATA/SPREADSHEETS/FileData_Summary.csv', 'a') as f:
                f.write('{}, (9) Null\n'.format(fname))

            return 1


    return 0





# to a sac trace object, add this additional header information
def append_sac_data(trace, record):
    trace.stla = record.STLA.to_numpy()[0]
    trace.stlo = record.STLO.to_numpy()[0]
    trace.stel = record.STEL.to_numpy()[0]
    trace.evla = record.EVLA.to_numpy()[0]
    trace.evlo = record.EVLO.to_numpy()[0]
    trace.evdp = record.EVDP.to_numpy()[0]
    trace.mag = record.MAG.to_numpy()[0]
    trace.user4 = record.RayP.to_numpy()[0]
    trace.lcalda = True



    
def add_90_to_azimuth(azi):
    if azi >= 270:
        plus_90 = azi  + 90 - 360
        return plus_90
    elif azi < 270:
        plus_90 = azi + 90
        return plus_90



# take a obspy stream object, and transfer it into its location as a SAC file
def make_sac(st, record, destination):
    
    net = st[0].stats.network
    stat = st[0].stats.station
    event = record.Event.to_numpy()[0]
    phase = record.PhaseType.to_numpy()[0]
    location = st[0].stats.location
    

    
    
    # read the metadata .xml file from our response_xml folder
    inv = read_inventory(os.path.join('../DATA/raw_mseed/response_xml', '{}.{}.response.xml'.format(net, stat)))
    inv = inv.select(channel = '[D,H,B,S]??')
    
    
    
    # Correction for nodal polarity
    for tr in st:
        if (tr.stats.network == '1X') and (tr.stats.channel == 'DPZ') and (int(tr.stats.station) >= 91):
            tr.data = tr.data * -1
    
        elif (tr.stats.network == 'XN') and (tr.stats.channel == 'DPZ') and (int(tr.stats.station) >= 151):
            tr.data = tr.data * -1

    
    # convert the obspy Stream into a SAC trace object so we can insert SAC headers
    vertical = SACTrace.from_obspy_trace(st.select(component='Z')[0])
    radial = SACTrace.from_obspy_trace(st.select(component='[N,1]')[0])
    tangential = SACTrace.from_obspy_trace(st.select(component='[E,2]')[0])
    
    # append common SAC headers to each component
    append_sac_data(vertical, record)
    append_sac_data(radial, record)
    append_sac_data(tangential, record)

    
    # insert component azimuth from inventory metadata
    # manually control component inclination to be in line with SAC convention
    vertical.cmpaz = 0
    vertical.cmpinc = 0
    radial.cmpaz = inv.get_orientation('{}.{}.{}.{}'.format(net, stat, location, radial.kcmpnm))['azimuth']
    radial.cmpinc = 90
    tangential.cmpaz = inv.get_orientation('{}.{}.{}.{}'.format(net, stat, location, tangential.kcmpnm))['azimuth']
    tangential.cmpinc = 90

    

    ## Save the SAC files into their respective Data_By_Station and Data_By_Event folders
    # with the ZNE naming scheme
    vertical.write(os.path.join(output_dir,destination,'Data_By_Station','{}'.format(net+'_'+stat), '{}.{}.{}.{}.{}.sac'.format(net, stat, vertical.kcmpnm[0:2] + 'Z', event, phase)))
    radial.write(os.path.join(output_dir,destination,'Data_By_Station','{}'.format(net+'_'+stat), '{}.{}.{}.{}.{}.sac'.format(net, stat, radial.kcmpnm[0:2] + 'N', event, phase)))
    tangential.write(os.path.join(output_dir,destination,'Data_By_Station','{}'.format(net+'_'+stat), '{}.{}.{}.{}.{}.sac'.format(net, stat, tangential.kcmpnm[0:2] + 'E', event, phase)))
    
    
    vertical.write(os.path.join(output_dir,destination,'Data_By_Event','{}'.format(event), '{}.{}.{}.{}.{}.sac'.format(net, stat, vertical.kcmpnm[0:2] + 'Z', event, phase)))
    radial.write(os.path.join(output_dir,destination,'Data_By_Event','{}'.format(event), '{}.{}.{}.{}.{}.sac'.format(net, stat, radial.kcmpnm[0:2] + 'N', event, phase)))
    tangential.write(os.path.join(output_dir,destination,'Data_By_Event','{}'.format(event), '{}.{}.{}.{}.{}.sac'.format(net, stat, tangential.kcmpnm[0:2] + 'E', event, phase)))





    # Check the Orientations record for Rayleigh-Wave estimations
     
    if os.path.exists(os.path.join('../Sensor_Orientations/DATA/{}/{}/_orientation.txt'.format(net, stat))):
        # read the orientations file for that network
        orientations = pd.read_csv(os.path.join('../Sensor_Orientations/DATA/{}/{}/_orientation.txt'.format(net, stat)), delimiter = " ")
        
        
        # Do not use estimated Orientation if the error contained is extreme
        if orientations['4-SIG'][0] < 20:
        
            # add estimated orientations for each component, assuming component 2 is 90 degrees from component 1
            vertical.cmpaz = 0
            vertical.cmpinc = 0
            radial.cmpaz =  orientations[orientations['STA'] == net+'-'+stat]['MEAN'].to_numpy()[0]
            radial.cmpinc = 90
            tangential.cmpaz = add_90_to_azimuth(orientations[orientations['STA'] == net+'-'+stat]['MEAN'].to_numpy()[0])
            tangential.cmpinc = 90
            
            
            ## Save the SAC files into their respective Data_By_Station and Data_By_Event folders
            # with the Z12 naming scheme        
            vertical.write(os.path.join(output_dir,destination,'Data_By_Station','{}'.format(net+'_'+stat), '{}.{}.{}.{}.{}.sac'.format(net, stat, vertical.kcmpnm[0:2] + 'Z', event, phase)))
            radial.write(os.path.join(output_dir,destination,'Data_By_Station','{}'.format(net+'_'+stat), '{}.{}.{}.{}.{}.sac'.format(net, stat, radial.kcmpnm[0:2] + '1', event, phase)))
            tangential.write(os.path.join(output_dir,destination,'Data_By_Station','{}'.format(net+'_'+stat), '{}.{}.{}.{}.{}.sac'.format(net, stat, tangential.kcmpnm[0:2] + '2', event, phase)))
            
    
            vertical.write(os.path.join(output_dir,destination,'Data_By_Event','{}'.format(event), '{}.{}.{}.{}.{}.sac'.format(net, stat, vertical.kcmpnm[0:2] + 'Z', event, phase)))
            radial.write(os.path.join(output_dir,destination,'Data_By_Event','{}'.format(event), '{}.{}.{}.{}.{}.sac'.format(net, stat, radial.kcmpnm[0:2] + '1', event, phase)))
            tangential.write(os.path.join(output_dir,destination,'Data_By_Event','{}'.format(event), '{}.{}.{}.{}.{}.sac'.format(net, stat, tangential.kcmpnm[0:2] + '2', event, phase)))

        
    else:
        pass

        

    




# take a set of the station information, and where we want to put it
def convert_and_transfer_SAC(contributing_stations, destination):

    # if the user has not established the 'Data_By_Event'  or
    # 'Data_By_Station' directories, build those
    if not(os.path.exists(os.path.join(output_dir, destination, 'Data_By_Event'))):
        os.mkdir(os.path.join(output_dir, destination, 'Data_By_Event'))

    if not(os.path.exists(os.path.join(output_dir, destination, 'Data_By_Station'))):
        os.mkdir(os.path.join(output_dir, destination, 'Data_By_Station'))

    print('\n')
    
    
    
    # cycle through the list 'contributing_stations'
    k = 0
    for i in contributing_stations:
        k+=1
        print("\r", end="")
        print("converting mseed... {:.2f}% ".format(k/len(contributing_stations)*100), end="")

        # parse the network and station ID
        net = i.split('.')[0]
        stat = i.split('.')[1]
        
        
        
        ### Subprocess, continue if not an XM station
        # if net != 'XM':
        #     continue
        
        
        # if this specific station does not already exist in the 'Data_By_Station'
        # folder, build that now
        if not(os.path.exists(os.path.join(output_dir, destination, 'Data_By_Station', '{}_{}'.format(net, stat)))):
            os.mkdir(os.path.join(output_dir, destination, 'Data_By_Station', '{}_{}'.format(net, stat)))
    
        
        # locate all files in the 'raw_mseed' folder with this net-station code
        files = glob(os.path.join(raw_file_loc, '{}.{}.*.mseed'.format(net, stat)))
        for j in files:
            # parse for event ID, and the file name, since this needs to be separated
            # from the path name that is given in the glob command
            event = j.split('/')[-1].split('.')[2]
            fname = j.split('/')[-1]
            
            
            # find the network-station-event record in the download list so that
            # we can pull its associated location data
            record = downloads[downloads.Network == net][downloads.Station == stat][downloads.Event == event].copy()
            if len(record) == 0:
                continue




            # if this specific event ID does not already exist in the 'Data_By_Event'
            # folder, build that now
            if not(os.path.exists(os.path.join(output_dir, destination, 'Data_By_Event', event))):
                os.mkdir(os.path.join(output_dir, destination, 'Data_By_Event', event))
                
            # read the mseed file as an obspy stream object
            st = read(j)
            
            
            
            
            
            """
                    Here is where we need to watch our data quality from the 
                    downloaded set. Mseed files will come with several issues
                    that are simply out of our control since we are likely 
                    dealing with a large-N data set, so we'll automatic methods 
                    to get the best result that we can
                    
                    Feel free to check all the data yourself if you dare 
                    
                    
                    
                    If the file experiences any of these problems, we want
                    to record that error, and continue to the next file
                    so we don't have processing errors later
            """
            
            # first file length issue, mseed file must at least channels of data
            # at the moment, we don't care what bands these channels are associated with
            if len(st) < 3:
                print('\n Band Error with file: {}'.format(fname))
                with open('../DATA/SPREADSHEETS/FileData_Summary.csv', 'a') as f:
                    f.write('{}, (1) Iinitial Content Error\n'.format(fname))
                    
                continue
            
            
            
            # select data based on band coverage, H>B>S, unless band is D (node)
            st = rank_channels(st)
            if st == 'fail':
                print('\n Band Error with file: {}'.format(fname))
                with open('../DATA/SPREADSHEETS/FileData_Summary.csv', 'a') as f:
                    f.write('{}, (2) Band Content Error\n'.format(fname))
                    
                continue
            
            
            
            # We now have a file contain only bandwidth designation
            # sometimes, if the data is broken, there will still be more
            # than 3 traces within the stream, we can merge these together
            if len(st) > 3: 
                st.merge(method=0, fill_value = 'latest')
                
                
            # if that merge still leaves more than 3 channel traces within
            # the stream, there's another issue, and we'll move on
            if len(st)>3:
                print('\n Merge Error with file: {}'.format(fname))
                with open('../DATA/SPREADSHEETS/FileData_Summary.csv', 'a') as f:
                    f.write('{}, (3) Merge Error\n'.format(fname))


            # perform secondary QC step
            if QC_Mseed(st, fname):
                continue


            

            # If the mseed file has made it this far, we are ready to export it
            # to a SAC file in the DATA directory
            try:
                make_sac(st, record, destination)
                with open('../DATA/SPREADSHEETS/FileData_Summary.csv', 'a') as f:
                    f.write('{}, (0) {}\n'.format(fname, destination))


            except:     
                print('\n SAC Conversion Error with file: {}'.format(fname))
                with open('../DATA/SPREADSHEETS/FileData_Summary.csv', 'a') as f:
                    f.write('{}, (10) SAC Error\n'.format(fname))
            
        
    return













"""
    Those are all the subfunctions we'll use
    
    The rest of this program is presuming that the user has some form of 'target'
    or perhaps line of interest that they want to investigate
    
    As it is currently built, this program will sort the data into its location 
    in a SAC format if it meets the lateral coverage requirements with respect to 
    that target network. 
    
    This marks the station as 'contributing' to the investigation


"""



# Create an outfile about file location and fail status
with open('../DATA/SPREADSHEETS/FileData_Summary.csv', 'w') as f:
    f.write('File,Status\n')




# http://eqseis.geosc.psu.edu/cammon/HTML/RftnDocs/rftn01.html
p_ref = 0.065
vs_ref = 3.4
vp_ref = 6.0
coverage_depth = 250

# presumed lateral range from station for a reading to coverage_depth
# collect all stations within this range 
lateral_range = 3 * coverage_depth * np.tan(np.arcsin(p_ref * vp_ref))



# cycle through the download summary, pulling out only the first instances
# of stations location information (so that you don't do this long process more than you need to)
unique_kstnm = unique_stla = unique_stlo = np.array([])
for i in range(len(downloads)):
    if downloads.Network[i] + '.' + downloads.Station[i] not in unique_kstnm:
        unique_kstnm = np.append(unique_kstnm, downloads.Network[i] + '.' + downloads.Station[i])
        unique_stla = np.append(unique_stla, downloads.STLA[i])
        unique_stlo = np.append(unique_stlo, downloads.STLO[i])






# Lastly, a dictionary structure to contain our location information
# how this is organized, the key in the dictionary is the station location folder
# the first and second objects within the key is the network and station that defines 
# study area

# for example, on TANGO_North, the XN network defines my study area, and I want to collect
# all stations within the necessary lateral coverage around these stations
# theres not a specific station code I'm focusing on, so that part is empty

# 2nd example, on ReFUCA, the 'ZA' network, speficially the 'RF' coded stations within that
# network is what defines this region

locations = {'TANGO_North': ['XN', ''], 'TANGO_South': ['1X', ''], 'ReFUCA': ['ZA', 'RF']}
# locations = {'Laguna_Maule': ['ZR', '']}




# iterate through each location within 'locations'
for loc in locations:
    print('Collecting data within the {} location'.format(loc))
    
    
    defining_net = locations[loc][0]
    defining_stat = locations[loc][1]
    
    print('\nLocation defined by the {} network + {} station(s)'.format(defining_net, defining_stat))


    # Search for stations contributing to region array
    def_kstnm = def_stla = def_stlo = np.array([])
    for i in range(len(downloads)):
        if len(defining_stat) == 0:
            if downloads.Network[i] == defining_net:
                if downloads.Network[i] + '.' + downloads.Station[i] not in def_kstnm:
                    def_kstnm = np.append(def_kstnm, downloads.Network[i] + '.' + downloads.Station[i])
                    def_stla = np.append(def_stla, downloads.STLA[i])
                    def_stlo = np.append(def_stlo, downloads.STLO[i])
        elif len(defining_stat) > 0:
            if (downloads.Network[i] == defining_net) and (defining_stat in downloads.Station[i]):
                if downloads.Network[i] + '.' + downloads.Station[i] not in def_kstnm:
                    def_kstnm = np.append(def_kstnm, downloads.Network[i] + '.' + downloads.Station[i])
                    def_stla = np.append(def_stla, downloads.STLA[i])
                    def_stlo = np.append(def_stlo, downloads.STLO[i])
                    
                    
             
                    
    
    # now cycle though all downloaded stations, and find those that will contribute to 
    # this location, including the definition network (note the distance will be calculated as 0)
    contributing_stations = np.array([])
    k = 0
    for i in range(len(def_kstnm)):
        k+=1
        print("\r", end="")
        print("Searching... {:.2f}% ".format(k/len(def_kstnm)*100), end="")
    
        for j in range(len(unique_kstnm)):
            if unique_kstnm[j] not in contributing_stations:
                if haversine(def_stla[i], def_stlo[i], unique_stla[j], unique_stlo[j]) <= lateral_range:
                    
                    contributing_stations = np.append(contributing_stations, unique_kstnm[j])
                    
    
    
    # Finally, convert every mseed file found in contributing stations to a 
    # SAC file within that location directory
    
    # The output shall be 5 files if applicable, 3 files minimum
    # 1) Z component, labeled 'net'.'station'.??Z.'event'.'phase'.sac
    # 2) N component, labeled 'net'.'station'.??N.'event'.'phase'.sac
    # 3) E component, labeled 'net'.'station'.??E.'event'.'phase'.sac
    # 4) 1 component, labeled 'net'.'station'.??1.'event'.'phase'.sac
    # 5) 2 component, labeled 'net'.'station'.??2.'event'.'phase'.sac
 
    convert_and_transfer_SAC(contributing_stations, loc)

    """
             What is the difference between N/E and 1/2?
             N/E assumes orthogonal components aligned to 0 degrees and 90 degrees
             north respectively. This is typically how the stations are entered 
             within Earthscope metadata and is likely not the reality. For our 
             purposes, the N/E components contain the station metadata orientations 
             
             
             1/2 shall represent the files with estimated orientations based of Rayleigh Wave
             Polarization (See Jim Gaherty codes)
             
             This will have manually inserted cmpaz sac headers
     
        
             Note that this distinction only exists within the name of the file
             and not the SAC header. The SAC header will have kcmpnm set to its
             original setup so that we can still cross refer to the .xml response files
    """
    



















