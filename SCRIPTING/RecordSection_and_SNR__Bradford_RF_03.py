#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:44:35 2024

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
import obspy.taup.taup_geo as taup
from obspy.taup import TauPyModel
from glob import glob
import warnings
import matplotlib.dates as dates
import pygmt
import cv2
import math
import time



warnings.simplefilter('ignore', category = UserWarning)


active_dir = './'
output_dir = '../DATA/'
raw_file_loc = '../DATA/raw_mseed/'
os.chdir(active_dir)


model = TauPyModel(model="iasp91")




"""

                        Part 3 
        The user should now have three things:
            1) a Download_Summary.csv file that contains the event-station information
            and the donwloaded file data
            
            2) Sorted directories organized into Station and Events
            
            3) .sac files in these directories whose headers contain the
                desired earthquake-station information 
        
            
        
        Now we will start our first step of quality control. The moveout-check
        
        Ideally, we would like clear signal from the approaching p-waves and other
        phases to ensure that the receiver function calculation has minimal noise
        or other interferences. The best way to verify is to visually look at the
        earthquake moveout across stations.
        
        We will plot each event moveout, cut around the predicted arrival times,
        and request the program to count each event as good or bad.
        
        
        
        The second part of this pre-RF quality check, is to calculate a quantitative
        signal-to-noise ratio for each station. The cut-off for what we consider SNR
        to be significant is up to the user, but its nice to have this information
        


"""


#%% Plot the downloaded stations


## Plot only the stations that we have downloaded information for

station_data = pd.read_csv(os.path.join(output_dir, 'SPREADSHEETS', 'Download_Summary.csv'))

fig = pygmt.Figure()
region = '-75/-50/-40/-20'
# region = '-71.5/-70/-37/-35'
prj = 'M15c'
grid = pygmt.datasets.load_earth_relief(resolution="01m", region=region)

shade = pygmt.grdgradient(grid=grid, azimuth="0/90", normalize="e1")

fig.basemap(region=region, projection=prj, frame='ag4')
fig.grdimage(grid=grid, shading = shade, cmap = '../DATA/MAPPING/natural_mod.cpt', projection = prj, transparency = 60)
fig.coast(borders = ["1/0.7p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 10)







Areas = ['TANGO_North', 'TANGO_South', 'ReFUCA']
codes = stlas = stlos = np.array([])

for a in Areas:
    for i in glob(os.path.join(output_dir, a, 'Data_By_Station', '*')):
        if i.split('/')[-1] not in codes:
            
            stat = station_data[station_data.Network == i.split('/')[-1].split('_')[0]][station_data.Station == i.split('/')[-1].split('_')[1]]
            
                                
            codes = np.append(codes, i.split('/')[-1])
            stlas = np.append(stlas, stat.STLA.to_numpy()[0])
            stlos = np.append(stlos, stat.STLO.to_numpy()[0])





fig.plot(x = stlos, y = stlas, style='i0.2c', fill = 'yellow', pen='black', label = 'Other Network')



fig.legend()
fig.show()



#%% Plot event record sections

##TODO need to do south line and refuca moveouts

# Read the downloads summary for location information
downloads = pd.read_csv(os.path.join(output_dir, 'SPREADSHEETS', 'Download_Summary.csv'))



Areas = ['TANGO_South', 'ReFUCA']
re_use_RS_Summary = True
if not(re_use_RS_Summary):
    with open('../DATA/SPREADSHEETS/Moveout_Summary.csv', 'w') as f:
        f.write('Event,Region,num_Stations,Accept,Reason,Latitude,Longitude,Phase\n')

elif re_use_RS_Summary:
    continue_summary = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')




def read_sac(files, event):
    
    RS_freq_max = 2.0
    RS_freq_min = 0.1
    
    
    # files list contains full path name information, parse for single file name
    
    
    # event cut information, cut around minimum arrival time for event
    arrival = utc(downloads.ArriveTime[downloads.Event == event].min())
    B = arrival - 30
    E = arrival + 120
    
    
    
    
    st = Stream()
    for f in files:
        file = f.split('/')[-1]
        
        
        # collect Z component waveforms
        if 'Z' in file.split('.')[2]:

            st += read(f)
            
    # edit stream to center around event moveout
    st.trim(starttime = B, endtime = E)
    
    
    
    
    # edit stream to remove instrument responses

    
    pre_filt = [0.01, 0.1, 8.0, 10.0]

    for tr in st:
        net = tr.stats.network
        stat = tr.stats.station
        
        # remove the instrument response, if there is an error, we will check if the
        # station is a Broadband (FDSN H or B in first position of ???), if it is
        # we will ignore IR removal and continue on, otherwise, continue to next station
        try:
            response = read_inventory('../DATA/raw_mseed/response_xml/{}.{}.response.xml'.format(net, stat))
            tr.remove_response(inventory = response, pre_filt = pre_filt, output = 'VEL')

        except: # header error of some kind
            if ('H' in tr.stats.channel[0]) or ('B' in tr.stats.channel[0]): # pass for broadband stations
                pass        
            else:
                st.remove(tr)
  
        
    
    
    if len(st) > 0:
        st.filter('bandpass', freqmin=RS_freq_min, freqmax=RS_freq_max)
        st.normalize()
    
    
    return st


def plot_record_section(st, phase):

    Record_Section, ax = plt.subplots(figsize = (40,15))
    data_scale = 0.12


        
    for tr in st:
      
        tr_time = tr.times('matplotlib')
        arrival = utc(downloads.ArriveTime[downloads.Event == event][downloads.Network == tr.stats.network][downloads.Station == tr.stats.station].to_numpy()[0])

    
        
        # pull from SAC data first

        gcarc = tr.stats.sac.gcarc
        evla = tr.stats.sac.evla
        evlo = tr.stats.sac.evlo
        evdp = tr.stats.sac.evdp
        mag = tr.stats.sac.mag
        baz = tr.stats.sac.baz
        
        
        
        
        
        data = (tr.data*data_scale) + gcarc
             
        ax.plot(data, tr_time, color = 'k', linewidth=0.5)



        
        if ('XN' in tr.stats.network) or ('1X' in tr.stats.network):
            ax.scatter(gcarc, tr_time[0], marker = 'v', s = 250, color = 'cyan', edgecolors='k', linewidth = 2.5)
            ax.text(gcarc, tr_time[1400], tr.stats.station, horizontalalignment = 'right', verticalalignment = 'center_baseline', rotation = 90, fontsize = 25)


        elif ('XM' in tr.stats.network):
            ax.scatter(gcarc, tr_time[0], marker = 'v', s = 250, color = 'crimson', edgecolors='k', linewidth = 2.5)
            try:
                ax.text(gcarc, tr_time[600], tr.stats.station, horizontalalignment = 'right', verticalalignment = 'center_baseline', rotation = 90, fontsize = 25)
            except:
                pass
        else:
            ax.scatter(gcarc, tr_time[0], marker = 'v', s = 250, color = 'yellow', edgecolors='k', linewidth = 2.5)



        ax.scatter(gcarc, arrival.matplotlib_date, s = 250, color = 'b')
        
                        
                
                

    ax.yaxis_date()
    ax.set_xlabel('GCARC', fontsize = 36)
    ax.tick_params(axis = 'both', labelsize = 34)


    plt.suptitle('{}, Mw = {:.1f}, Lat/Lon={:.2f}/{:.2f}, depth = {:.1f} km, Phase: {}'.format(event, mag, evla, evlo, evdp, phase), fontsize = 40, y=1)


    
    return Record_Section




def input_status():
    while True:
        status = input('is good? [y/n]')
        
        if status == 'quit':
            break
        
        if not ((status == 'y') or (status == 'n')):
            print('Enter y/n')
        else:
            break

    return status
    
    
    




    


for a in Areas:
    events = glob(os.path.join(output_dir, a , 'Data_By_Event', '*'))
    print('Determing Moveouts in {}'.format(a))
    print('Num Events to judge: {}'.format(len(events)))
    
    t = 0
    for e in events:
        t += 1
        
        event = e.split('/')[-1]
        event_files = glob(os.path.join(output_dir, a , 'Data_By_Event', e, '*.sac'))



        print("\r", end="")
        print("Judging Events... {} ...  {:.2f}% ".format(event, t/len(events)*100), end="")        
    
        
        
        if re_use_RS_Summary:
            # check for event registered in the continue summary
            if len(continue_summary[continue_summary.Event == event][continue_summary.Region == a].to_numpy()) > 0:
                continue
        
        
        
        
        
        if len(event_files) == 0:
            os.rmdir(e)
            continue
            
        phase_type = event_files[0].split('.')[-2]
        

        
        st = read_sac(event_files, event)
        if len(st) == 0:
            with open('../DATA/SPREADSHEETS/Moveout_Summary.csv', 'a') as f:
                f.write('{},{},{},n,Data Issue,{},{},{}\n'.format(event, a, len(st), np.nan, np.nan, phase_type))
    
            continue


        
        
        evla = st[0].stats.sac.evla
        evlo = st[0].stats.sac.evlo
        
        # ignore events with only one station recording
        if len(st) <= 1:
            with open('../DATA/SPREADSHEETS/Moveout_Summary.csv', 'a') as f:
                f.write('{},{},{},n,inadequate station coverage,{},{},{}\n'.format(event, a, len(st), evla, evlo, phase_type))
    
            continue
        
        
        RS = plot_record_section(st, phase_type)
     
        
        plt.pause(0.2)
        
        
        
        # like the record section plot
        status = input_status()
        if status == 'quit':
            sys.exit('exited from movemout assessment')
    
        if status == 'y':
            
            RS.savefig(os.path.join(output_dir, a, 'Data_By_Event', event, 'RS_{}.png'.format(event)), dpi = 300)
            
            with open('../DATA/SPREADSHEETS/Moveout_Summary.csv', 'a') as f:
                f.write('{},{},{},{},none,{},{},{}\n'.format(event, a, len(st), status, evla, evlo, phase_type))
            
            
            
        
        elif status == 'n':
            with open('../DATA/SPREADSHEETS/Moveout_Summary.csv', 'a') as f:
                f.write('{},{},{},{},judged,{},{},{}\n'.format(event, a, len(st), status, evla, evlo, phase_type))
        
    
        



    
    
#%% SNR for good events



# read the moveout_summary, so we only do events that have accepted
moveout_summary = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')


# the seconds from the predicted phase arrival, we will calculate the rms
# 'snr_range' before and after the arrival and solve for the ratio betweeen the two
snr_range = 60

 


# set to True if you wish to continue from the end of the last listed summary
re_use_SNR_Summary = True
if not(re_use_SNR_Summary):
    with open('../DATA/SPREADSHEETS/SNR_Summary.csv', 'w') as f:
        f.write('Network,Station,Channel,Event,Phase,Region,Arrival,Filtered_SNR\n')

elif re_use_SNR_Summary:
    continue_summary = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')







# Error information from before, ignore this if evla and evlo parameters were
# provided in the moveout_summary
if ('Latitude' not in moveout_summary.columns) or ('Longitude' not in moveout_summary.columns):
    evlas = evlos = phases = np.array([])
    for i in range(len(moveout_summary.Event)):
        files = glob(os.path.join(output_dir, moveout_summary.Region[i], 'Data_By_Event', moveout_summary.Event[i], '*.sac'))
        st = read(files[0])
        evla = st[0].stats.sac.evla
        evlo = st[0].stats.sac.evlo
        phase = files[0].split('/')[-1].split('.')[-2]
        
        evlas = np.append(evlas, evla)
        evlos = np.append(evlos, evlo)
        phases = np.append(phases, phase)
        
    moveout_summary['Latitude'] = evlas
    moveout_summary['Longitude'] = evlos
    moveout_summary['Phase'] = phases
    
    
    




    
# Trim the moveout_summary to only those events that were accepted, and reset the index
accepted = moveout_summary[moveout_summary.Accept == 'y'].copy()
accepted = accepted.reset_index(drop = True)
print('{:.3f}% event pass rate'.format(len(accepted) / len(moveout_summary) * 100))








# produce a gmt map of the accepted events with respect to the phase type
fig = pygmt.Figure()
prj = "A-71.2/-20/155/15c"


fig.coast(projection=prj, region = 'g', frame = 'g60', land="gray", shorelines = '0.5p, black')

fig.plot(x = accepted.Longitude[accepted.Phase == 'P'], y = accepted.Latitude[accepted.Phase == 'P'], style='c0.23c',fill = 'cyan', pen='0.6p,black', label = 'P phase')
fig.plot(x = accepted.Longitude[accepted.Phase == 'PP'], y = accepted.Latitude[accepted.Phase == 'PP'], style='c0.23c',fill = 'red', pen='0.6p,black', label = 'PP phase')
fig.plot(x = accepted.Longitude[accepted.Phase == 'PKP'], y = accepted.Latitude[accepted.Phase == 'PKP'], style='c0.23c',fill = 'magenta', pen='0.6p,black', label = 'PKP phase')

# plot the general region we are focused on
fig.plot(x = [-72, -72, -56, -56, -72], y = [-19, -29, -29, -19, -19], pen = '0.4p,red,--')


fig.legend()
fig.show()



# cycle through each accepted event
t = 0
for i in range(len(accepted.Event)):
    t+=1
    
    print("\r", end="")
    print("Calculating SNR via RMS ratio... {:.2f}% ".format(t/len(accepted.Event)*100), end="")        

    
    # collect phase and event ID information from the Moveout_Summary
    phase = accepted.Phase[i]
    event = accepted.Event[i]
    area = accepted.Region[i]
    
    
    

    
    
    
    # list of files contained within that event directory
    files = glob(os.path.join(output_dir, area, 'Data_By_Event', accepted.Event[i], '*.sac'))
    if len(files) == 0:
        continue
    
    
    # initialize a stream object to read sac files into, also save the file name into trace
    st_event = Stream()
    codes = np.array([])
    for j in files:
        st_event += read(j)
        st_event[-1].stats.fname = j.split('/')[-1]     # identifier for original file name
        st_event[-1].stats.orien = j.split('/')[-1].split('.')[2][2]  # identifier for orientation type
        
        codes = np.append(codes, '{}.{}'.format(st_event[-1].stats.network, st_event[-1].stats.station))
        
        
        
    # cycle through each net-stat code
    for code in np.unique(codes):
        net, stat = code.split('.')[0],code.split('.')[1]
        
        

        
        
        # if this net-stat-event combination is within the continue summary, continue to next station
        if re_use_SNR_Summary:
            if len(continue_summary[continue_summary.Event == event][continue_summary.Region == area][continue_summary.Network == net][continue_summary.Station == stat]) > 0:
                continue      


        # collect the predicted arrival time from the downloads summary
        arrival = utc(downloads.ArriveTime[downloads.Event == event][downloads.Network == net][downloads.Station == stat].to_numpy()[0])

        # select this station from event Stream and trim to within the snr_range
        pre_filt = [0.01, 0.1, 8.0, 10.0]
        st_stat = st_event.select(network = net, station = stat).copy()
        
        st_stat.trim(starttime = arrival - snr_range, endtime = arrival + snr_range, pad=True, fill_value = 0)
       
        # check for null values within stream:
        null_data = False
        for tr in st_stat:
            if any(np.isnan(tr.data)):
                null_data = True
        if null_data:
            continue
        
        
        st_stat.interpolate(sampling_rate = st_stat[0].stats.sampling_rate, npts = st_stat[0].stats.npts)
       
        # check for null values again within stream:
        null_data = False
        for tr in st_stat:
            if any(np.isnan(tr.data)):
                null_data = True
        if null_data:
            continue

        
        # remove the instrument response, if there is an error, we will check if the
        # station is a Broadband (FDSN H or B in first position of ???), if it is
        # we will ignore IR removal and continue on, otherwise, continue to next station
        try:
            response = read_inventory('../DATA/raw_mseed/response_xml/{}.{}.response.xml'.format(net, stat))
            st_stat.remove_response(inventory = response, pre_filt = pre_filt, output = 'VEL')

        except: # header error of some kind
            if ('H' in st_stat[0].stats.channel[0]) or ('B' in st_stat[0].stats.channel[0]): # pass for broadband stations
                pass        
            else:
                continue
        
        
        # bandpass filter the Stream to within our range for RFs, and center each trace
        for tr in st_stat:
            tr.data = tr.data - np.mean(tr.data)
        
        st_stat.detrend()   # detrend the Stream
        st_stat.detrend('linear')   # detrend the Stream
        
        st_stat.filter('bandpass', freqmin = 0.1, freqmax = 8.0)



        
        # Rotate the data to the receiver-earthquake reference frame
        
        """
                To be aware of:
                    
                    Obspy station rotation options demand that the station orientations
                    be set within the Inventory object, or defaulted to assume that the
                    N or 1 is indeed at 0 degrees azimuth, and E or 2 component is orthogonal
                    and set to 90 degrees. It will ignore any SAC header telling it otherwise.
                    
                    If we rely on the Inventory (metadata) of the station, know that we cannot
                    change values from within obspy (which is wildly annoying, but ok) 
                    so, we need a workaround if we do not make orientation assumptions
        
        
        """


        # fname contains information on which data has default 0/90 degree orientations
        # and which have estimated orientations
        
        # if N/E in fname, -> default 0/90 orientations used
        # if 1/2 in fname, -> estimated orientations used

        # start by rotating the 1/2 component traces to the N/E orientation
        
        if len(st_stat) > 3:    # determine if there are actually 1/2 and N/E distinct traces
            rot_st = Stream()
            # cycle through traces, and pull the data arrays for 1/2 traces and Z trace
            for tr in st_stat:
                if (tr.stats.orien == '1'):
                    rot_st += tr.copy()
                    n1 = tr.data
                    n_azi = tr.stats.sac.cmpaz
                    st_stat.remove(tr)
                    
                elif (tr.stats.orien == '2'):
                    rot_st += tr.copy()
                    e2 = tr.data
                    e_azi = tr.stats.sac.cmpaz
                    st_stat.remove(tr)

                
                elif (tr.stats.orien == 'Z'):
                    rot_st += tr.copy()
                    z3 = tr.data


                
                # While here, we'll correct the component names of N/E convention files
                elif (tr.stats.orien == 'N'):
                    tr.stats.component = 'N'
            
                elif (tr.stats.orien == 'E'):
                    tr.stats.component = 'E'
                    

            # rotate using rotate2zne subroutine
            z,n,e = rotate.rotate2zne(z3, 0, -90, n1, n_azi, 0, e2, e_azi, 0)
            
            # update the correcting stream traces, and ensure that the compoent
            # is listed as 'N' or 'E' so obspy rotate command can recognize them
            
            # Note, the obspy trace 'channel' and 'component' parameters are 
            # mutually dependent, changing one will change the other
            for tr in rot_st:
                if (tr.stats.orien == '1'):
                    tr.data = n
                    tr.stats.component = 'N'
                    
                elif (tr.stats.orien == '2'):
                    tr.data = e
                    tr.stats.component = 'E'

                elif (tr.stats.orien == 'Z'):
                    tr.data = z
                    
 
 
            
            # Now, we have st_stat, containing unrotated data
            # and rot_st, containing NE rotation correction for 1 and 2 defined traces
            
            # rotate corrected components and find SNR
            try:
                rot_st.rotate('NE->RT', back_azimuth = rot_st[0].stats.sac.baz)
            except:
                print('Bad Rotation')
                continue
            
            for tr in rot_st:
                
                if tr.stats.orien == 'Z':
                    continue
                
                print_channel = tr.stats.channel[0:2] + tr.stats.orien
                
                snr_pre = tr.data[(tr.times('UTCDateTime')>=arrival - snr_range) & (tr.times('UTCDateTime') <= arrival)]
                snr_post = tr.data[(tr.times('UTCDateTime')>=arrival) & (tr.times('UTCDateTime') <=arrival + snr_range)]
                
                noise_rms = np.sqrt(np.sum(snr_pre**2)/len(snr_pre))
                signal_rms = np.sqrt(np.sum(snr_post**2)/len(snr_post))
            
                snr = signal_rms / noise_rms
                
                with open('../DATA/SPREADSHEETS/SNR_Summary.csv', 'a') as f:
                    f.write('{},{},{},{},{},{},{},{:.4f}\n'.format(net, stat, print_channel, event, phase, area, arrival, snr))


        




        # now, rotate uncorrected components and find SNR       
        try:
            st_stat.rotate('NE->RT', back_azimuth = st_stat[0].stats.sac.baz)
        except:
            print('Bad Rotation')
            continue
        
        for tr in st_stat:
            
            print_channel = tr.stats.channel[0:2] + tr.stats.orien


            snr_pre = tr.data[(tr.times('UTCDateTime')>=arrival - snr_range) & (tr.times('UTCDateTime') <= arrival)]
            snr_post = tr.data[(tr.times('UTCDateTime')>=arrival) & (tr.times('UTCDateTime') <=arrival + snr_range)]
            
            noise_rms = np.sqrt(np.sum(snr_pre**2)/len(snr_pre))
            signal_rms = np.sqrt(np.sum(snr_post**2)/len(snr_post))
        
            snr = signal_rms / noise_rms
            
            with open('../DATA/SPREADSHEETS/SNR_Summary.csv', 'a') as f:
                f.write('{},{},{},{},{},{},{},{:.4f}\n'.format(net, stat, print_channel, event, phase, area, arrival, snr))





snr_summary = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')

        
        
        
        
        



