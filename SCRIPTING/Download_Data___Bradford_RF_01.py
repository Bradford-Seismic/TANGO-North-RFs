# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:33:09 2023

@author: 7418888
"""


import numpy as np
import pandas as pd
import os
import sys
import shutil
from obspy.core import UTCDateTime as utc
import obspy.taup.taup_geo as taup
from obspy.taup import TauPyModel
import obspy.clients.fdsn as fdsn
from glob import glob
from obspy.core import Stream, read
import matplotlib.pyplot as plt
from obspy.geodetics import degrees2kilometers
import warnings

warnings.simplefilter('ignore', category = UserWarning)
warnings.simplefilter('ignore', category = FutureWarning)



"""
                    Usage Notes
    The code written here is by me, Jim Bradford,
    for the purposes of preforming the necessary procedural and analytical assessment
    for the article: TBD
    DOI: TBD


    This is first of a series of Scripts (labeled accordingly) that should be performed
    in sequential order for replicating the results and findings of the specified article

    I can not gaurantee that later works by myself or other workers will use precisely
    the same form or exact parameters, so please do your do diligence and check the work
    to make an informed decision if you wish to use this model

    Furthermore, I do not intend to update these scripts post-publication, as this is built specifically
    to address this particular work. If you are using this script, and you run into issues
    that are due to its compilation and execution, I cannot help with that.

    That said, if you find something that is procedurally wrong, PLEASE LET ME KNOW,
    as that is a scientific problem, and affects how we interpret the Earth, which is something
    I care much more about.



    I would love to see these scripts used as an education material and inspiration for other work.
    Don't let this be a black box object. I want users to be able to take bits and pieces of these
    scipts in order to supplement their own work. For instance, if you like how one of our
    figures are designed using PyGMT, please check it out here so you can understand that process.

    But please be sure to give credit where it is do.



    Enjoy,
    -Jim

"""





#### Initialize System

# Preferable Directory Structure:
# For first time usage, the user will need to manually make the first level of
# directories under the 'Project_Folder' location: 'DATA', 'SCRIPTING', 'FIGURES'
# Put this python script, into your 'SCRIPTING' directory
"""

                                    -Project_Folder
                                            |
                                            |
                |----------------------------------------------------------|
                    - DATA          -SCRIPTING (Language)       - FIGURES
                    |
                    |
                    |
                    |
               |---------------------------------------------------------|
                -raw_mseed      -MAPPING        -SPREADSHEETS       -PROJECT_AREA
                     |                                                  |
                     |                                                  |
                 |--------|                                             |
                   -response_xml                                        |
                                                                        |
                |-----------------------------------------------------------|
                    -Data_By_Station        -Data_By_Event          -DATA_CPP
                            |                       |                   |
                            |                       |                   |
                            |                       |                   |
                     |-------------|       |--------------|         |--------|
                        - Various_Events    - Various_Stations      .itr files



"""



# notice that these are absolute path names from the comupter name, to the user home and project folder
# example: 'Tango_2' is my project folder, 'SCRIPTING' is the folder where
# all these program files will be kept and refered to

# 'active_dir' will be our home folder, and will navigate to everywhere we need
# pull data based on its relative location to 'active_dir'
# active_dir = '/tango/bradford/User_Test_Folder/SCRIPTING'
active_dir = './'




# command: change the current location to the 'active_dir' to begin processing
os.chdir(active_dir)






user = ""
password = ""



# Initialize System
# Preferable Directory Structure:
# For first time usage, the user will need to manually make the first level of
# directories under the 'Project_Folder' location: 'DATA', 'SCRIPTING', 'FIGURES'
# Put this python script, into your 'SCRIPTING' directory
"""

                                    -Project_Folder
                                            |
                                            |
                |----------------------------------------------------------|
                    - DATA          -SCRIPTING (Language)       - FIGURES
                    |
                    |
                    |
                    |
               |---------------------------------------------------------|
                -raw_mseed      -MAPPING        -SPREADSHEETS       -PROJECT_AREA
                     |                                                  |
                     |                                                  |
                 |--------|                                             |
                   -response_xml                                        |
                                                                        |
                |-----------------------------------------------------------|
                    -Data_By_Station        -Data_By_Event
                            |                       |
                            |                       |
                            |                       |
                     |-------------|       |--------------|
                        - Various_Events        - Various_Stations



"""





#%% Build Directory Dependencies

# project 2 data structure

# DATA/
# # TANGO_North
# # SPREADSHEETS
# # MAPPING

# Additional subfolders based on locations within a project (user preference, add as many as you want)
Project_Areas = ['TANGO_North']
Sub_Dirs = ['SPREADSHEETS', 'MAPPING', 'raw_mseed', 'response_xml']
Area_Sub_Dirs  = ['Data_By_Station', 'Data_By_Event', 'Data_CCP']


for d in Sub_Dirs:
    os.mkdir('../DATA/{}'.format(d))

for a in Project_Areas:
    os.mkdir('../DATA/{}'.format(a))
    for d in Area_Sub_Dirs:
        os.mkdir('../DATA/{}/{}'.format(a, d))



#%%

"""

                        We require two things moving forward

1) Go to:   https://ds.iris.edu/gmap/

    and download the location information of the desired station within
    the study area

    -> Open the downloaded txt file in your preferred editor, and remove the
    unnecessary spaces contained in the header line, this will prevent the
    system from reader the data if you do not do this,



    -> you will need to know where each file is stored, its Data Center
    For SA, this is usually IRISDMC, Geofon, or IRISPH5
    Create a column in the station data file and add a column: DataCenter
    When downloaded from the IRIS station search engine, the rows will be separated
    corresponding to data center, remove the extra layers, and change all rows
    beneath that data center subheader in the DataCenter column to that data center



    -> Last editing step, you will add a 'code' column
    the station 'code' is the unique network_name-station_name combination.
    This makes file parsing easier, use you excel tools to concatenate the
    network and station columns together with a hyphen in the middle to create
    the 'code' column

    example: Network = 'XM', Station= 'CB09' --> code = 'XM-CB09'



    -> save the file as .csv under your prefered file name




2) Go to: https://earthquake.usgs.gov/earthquakes/map

    and download all global earthquakes between your desired timeframe

    do not worry about the EQ locations relative to the stations, we'll handle
    that later


    -> rename the file to your desired file name, there should no additional
    file editing needed for this step


"""






#%% Define Search Params

# read the downloaded station information
station_data = pd.read_csv('../DATA/SPREADSHEETS/SA_Stations.csv',
                           header = 0,
                           dtype={
                                  'Network': 'string',
                                  'Station': 'string'})



# remove any NA entires and reset the list index
station_data = station_data.dropna()
station_data = station_data.reset_index()


# find the minimum activation time and maximum end time within the downloaded
# station set, we'll trim the EQ events to within this time span
Record_Start = utc(np.min(station_data.StartTime))
Record_End = utc(np.max(station_data.EndTime))






# read the downloaded EQ information file, specifically grabbing the location and time information
EQ_data = pd.read_csv('../DATA/SPREADSHEETS/Global_EQs.csv',
                      usecols=['time', 'latitude', 'longitude', 'depth', 'mag'])


# restrict our Earthquakes to above a desired magnitude, this is fairly arbitrary
# in that you can have really good low magnitude events sometimes
# for RFs, typically 5-6 minimum magnitude is accepted,
EQ_data = EQ_data[EQ_data.mag > 5]


# restrict the event to those after the minimum station activation time
EQ_data = EQ_data[EQ_data.time > Record_Start]


# remove all NA entries, and reset the list index
EQ_data = EQ_data.dropna()
EQ_data = EQ_data.reset_index()


print("Preliminary availability prior to data download:\n{} Stations\n{} Events".format(len(np.unique(station_data.Code)), len(EQ_data)))
print("Maximum Potential of data: {} files".format(len(np.unique(station_data.Code)) *  len(EQ_data)))




#%% Match Station-EQ Pairs and download data from respective data center

# for each station, compare active time to EQ occurences
# then, compare station - EQ location and assign and phase type
# then, determine ray parameter, distance, phase, and arrival time
# then, download the event file as mseed type along with the instrument response xml file


model = TauPyModel(model="iasp91")
raw_file_loc = '../DATA/raw_mseed/'
response_file_loc = '../DATA/response_xml/'

# this definition will perform the data download, provide for it
# the: network name, station name, Origin time of the earthquake, begin time of file
# end time of the file, the phase type (P,PP,S, etc) and the data center

def download_data(network, station, event_ID, B, E, phase, center):
    b = str(B)[0:19]
    e = str(E)[0:19]
    # set the downloaded file name
    output = os.path.join(raw_file_loc, "{}.{}.{}.{}.mseed".format(network,
                                                    station,
                                                    event_ID, phase))

    if center == 'IRISDMC':
        os.system('curl --digest --user {}:{} --output {} "http://service.iris.edu/fdsnws/dataselect/1/queryauth?&format=mseed&net={}&sta={}&cha=?H?&loc=--&starttime={}&endtime={}"'.format(user,password, output, network, station, b, e))
        if not(os.path.exists(os.path.join(response_file_loc, '{}.{}.response.xml'.format(network, station)))):
            os.system('curl --digest --user {}:{} --output {} "http://service.iris.edu/fdsnws/station/1/query?&format=xml&net={}&sta={}&loc=--&level=response"'.format(user, password,os.path.join(response_file_loc,  '{}.{}.response.xml'.format(network, station)), network, station))


    elif center == 'IRISPH5':
        os.system('curl --digest --user {}:{} --output {} "http://service.iris.edu/ph5ws/dataselect/1/queryauth?reqtype=FDSN&format=mseed&net={}&sta={}&cha=DP?&loc=--&starttime={}&endtime={}"'.format(user,password, output, network, station, b, e))

        if not(os.path.exists(os.path.join(response_file_loc,  '{}.{}.response.xml'.format(network, station)))):
            os.system('curl --digest --user {}:{} --output {} "http://service.iris.edu/ph5ws/station/1/query?&format=xml&net={}&sta={}&loc=--&level=response"'.format(user, password,os.path.join(response_file_loc,  '{}.{}.response.xml'.format(network, station)), network, station))



    elif center == 'GEOFON':
        os.system('curl --digest --user {}:{} --output {} "http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query?&format=mseed&net={}&sta={}&cha=???&starttime={}&endtime={}"'.format(user,password, output, network, station, b, e))
        if not(os.path.exists(os.path.join(response_file_loc,  '{}.{}.response.xml'.format(network, station)))):
            os.system('curl --digest --user {}:{} --output {} "http://geofon.gfz-potsdam.de/fdsnws/station/1/query?&format=xml&net={}&sta={}&loc=--&level=response"'.format(user, password,os.path.join(response_file_loc, '{}.{}.response.xml'.format(network, station)), network, station))

    else:
        # if for some reason, this particular is not available anywhere, print this message
        # you should only be considered if you see this appear your feed consistently
        print('Data Location Error')


    return output, event_ID








#### Start the download process

# we will output the download information to this file stored in your SPREADSHEETS directory
output_file = '../DATA/SPREADSHEETS/Download_Summary.csv'





# if, for some reason prior download fails, we can use the current download summary
# to mark our progress

# set the 're_use_Download_Summary' variable to True if you want to continue from
# last download attempt,
# set to False if you want to restart and reset the Download_Summary.csv file

re_use_Download_Summary = False
if not(re_use_Download_Summary):
    with open(output_file, "w") as downloads:
         downloads.write("Network,Station,Event,STLA,STLO,STEL,EVLA,EVLO,EVDP,MAG,GCARC,BAZ,RayP,O,B,E,TravelTime,ArriveTime,PhaseType,FileName,FileFormat,Bytes\n")

elif re_use_Download_Summary:
    continue_summary = pd.read_csv(output_file)






# pull all unique station codes
# one station may have multiple data entries, due to it being activated and
# deactivated several times over the course of its deployment
# we only care about its absolute start and end times,
unique_stations = np.unique(station_data.Code)
print('{} unique stations in area'.format(len(unique_stations)))


# time window for file download before and after predicted arrival time
trim_B = 120
trim_E = 120


### define the parameters needed to accept event:
phase = 'P'


def event_consider(stla, stlo, evla, evlo, evdp, phase):


    geo = taup.calc_dist_azi(evla, evlo, stla, stlo, 6371, 0)
    dist = geo[0]   # distance in degrees
    baz = geo[2]    # receiver -> source back azimuth



    try:
        ray = model.get_ray_paths(evdp, dist, phase_list = [phase, 'S'])
        rayp = ray[0].ray_param  * (np.pi/180) * (1/degrees2kilometers(1)) # ray parameter converted from sec/rad to sec/km
        i_angle = ray[0].incident_angle

        P_arrive = ray[0].time
        S_arrive = ray[1].time



        ### list requirements as separate if/else statements based on above metrics

        check = 1
        ### Distance Requirement
        if (dist > 30) and (dist < 90):
        # if (dist < 10):
            check = 1
        else:
            check = 0

        # if i_angle < 30:
        #     check = 1
        # else:
        #     check = 0

        # if S_arrive - P_arrive > 10:
        #     check = 1
        # else:
        #     check = 0


    except:
        check = 0





    return check








# Counters for downloading data

num_phase = len(glob(os.path.join(raw_file_loc, '*.{}.mseed'.format(phase))))
num_reject = 0



total=0

k = 0


"""

                A note on naming and parsing format

for receiver function files, we code each file to be defined by its:
    - station code
    - event that its recording
    - type of ray phase being recorded

for mseed files, the file id will be:

      'net'.'station'.'event origin time'.'phase'.mseed


where 'event origin time' is the 'yyyy-mm-ddThh-mm-ss' format of event origin defined in UTC


"""


# iterate through each unique station code
for i in unique_stations:
    k+=1

    # pull the subset from station_data for this station specifically
    station_df = station_data[station_data.Code == i].copy().reset_index(drop = True)

    # get the minimum starttime and maximum endtime in the list
    start = utc(station_df.StartTime.min())
    end = utc(station_df.EndTime.max())

    # get the location information of the station
    stla = station_df.Latitude.unique()[0]
    stlo = station_df.Longitude.unique()[0]
    knetwk = i.split('-')[0]
    kstnm = i.split('-')[1]





    try:
        stel = station_df.Elevation.unique()[0] # station elevation may not be entered
    except:
        stel = 9999     # arbitrary filler value

    # the data center location for this network-station
    center = station_df.DataCenter.unique()[0]


    # select all earthquakes occuring within the station activity time
    EQ_trim = EQ_data[(EQ_data.time > start) & (EQ_data.time < end)].copy().reset_index(drop = True)



    # iterate through earthquake data
    for j in range(len(EQ_trim)):
        total += 1
        evla, evlo, evdp, mag = EQ_trim.latitude[j], EQ_trim.longitude[j], EQ_trim.depth[j], EQ_trim.mag[j]

        OTime = EQ_trim.time[j]


        # parse the OTime object and make it in terms of our event ID format
        event_ID = OTime[0:19].replace(':', '-')



        # now, refer to the continue_summary if we are continuing a download session
        # if the net-stat-event ID is registered in the continue summary, continue
        if re_use_Download_Summary:
            if len(continue_summary[continue_summary.Event == event_ID][continue_summary.Network == knetwk][continue_summary.Station == kstnm]) > 0:
                continue





        if event_consider(stla, stlo, evla, evlo, evdp, phase):

            # calculate the necessary geographical information
            geo = taup.calc_dist_azi(evla, evlo, stla, stlo, 6371, 0)
            dist = geo[0]   # distance in degrees
            baz = geo[2]    # receiver -> source back azimuth

            ray = model.get_ray_paths(evdp, dist, phase_list = [phase])



            travel = ray[0].time            # the predicted travel time of the ray
            arrive = utc(OTime) + travel    # the predicted arrival time
            B = arrive - trim_B             # subtract from arrivel time for trim window beginning
            E = arrive  + trim_E            # add to arrival time for trim window end
            rayp = ray[0].ray_param  * (np.pi/180) * (1/degrees2kilometers(1)) # ray parameter converted from sec/rad to sec/km





            # perform the data download
            file, event = download_data(knetwk, kstnm, event_ID, B, E, phase, center)



            # there's a chance of downloading an empty file with no data in it
            # this file will have considerably smaller size
            # remove temp file if output file is not filled
            if os.path.exists(file):
                if os.stat(file).st_size < 1000:
                    os.remove(file)
                    num_reject += 1
                else:
                    num_phase += 1
                    # if the file was successfully downloaded, record it in the Download_Summary.csv file
                    with open(output_file, "a") as downloads:
                        downloads.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                            knetwk,kstnm,event,
                            stla,stlo,stel,evla,evlo,evdp,mag,
                            np.round(dist, 4),np.round(baz, 4),np.round(rayp, 4),
                            str(utc(OTime)),str(utc(B)),str(utc(E)),
                            np.round(travel, 4),str(arrive),phase,
                            file.split('/')[-1], 'mseed', os.stat(file).st_size))







        print("\r", end="")
        print("Downloading... {:.4f}% , Phase: {}, Reject: {} ".format(k/len(unique_stations)*100, num_phase,num_reject), end="")




