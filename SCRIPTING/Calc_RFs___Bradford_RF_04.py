#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:55:26 2024

@author: bradford
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
import cv2
import subprocess

warnings.simplefilter('ignore', category = UserWarning)


active_dir = './'
output_dir = '../DATA/'
raw_file_loc = '../DATA/raw_mseed/'
os.chdir(active_dir)


model = TauPyModel(model="iasp91")




moveouts = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')
snr = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')







#%% Filter the snr and moveout summaries to all passing data

moveouts_trimmed = moveouts[moveouts.Accept == 'y'].reset_index(drop = True).copy()




# the SNR summary and parameters we care about will change from study to study, 
# take the time to edit the snr_trimmed variable as you feel


# accept SNR > 2.0 for all broadband networks
# accept SNR > 1.5 for all nodal networks

snr_BB = snr[snr.Filtered_SNR > 2.0][(snr.Network != 'XN') | (snr.Network != '1X')].reset_index(drop = True).copy()
snr_node = snr[snr.Filtered_SNR > 1.5][(snr.Network == 'XN') | (snr.Network == '1X')].reset_index(drop = True).copy()
snr_trimmed = pd.concat((snr_BB, snr_node)).reset_index(drop = True)



# SNR trimmed contains only the passing data, we'll save this as a new .csv file
with open('../DATA/SPREADSHEETS/SNR_Summary_Passing.csv', 'w') as f:
    f.write('Network,Station,Channels,Event,Phase,Region,Arrival,Vertical_SNR,Radial_SNR\n')


codes = np.array([])
for i in range(len(snr_trimmed)):
    print("\r", end="")
    print("Assessing Passed Files... {:.2f}% ".format(i/len(snr_trimmed)*100), end="")        

    net = snr_trimmed.Network[i]
    stat = snr_trimmed.Station[i]
    event = snr_trimmed.Event[i]
    phase = snr_trimmed.Phase[i]
    region = snr_trimmed.Region[i]
    arrival = snr_trimmed.Arrival[i]
    
    
    # cycle through codes, where there will be an additional event component
    code = "{}-{}-{}".format(net, stat, event)
    if code not in codes:
        codes = np.append(codes, code)
        
        sel = snr_trimmed[snr_trimmed.Network == net][snr_trimmed.Station == stat][snr_trimmed.Event == event].copy()
        # sel needs to contain both "??Z" and "??N" components for radial receiver function analysis
        channels = sel.Channel.to_numpy()
        snr = sel.Filtered_SNR.to_numpy()
        
        # fix channels to only contain the component identifier
        for j in range(len(channels)):
            channels[j] = channels[j][2]
            if channels[j] == 'Z':
                snr_Z = snr[j]
            elif channels[j] == 'N':
                snr_N = snr[j]
            elif channels[j] == '1':
                snr_1 = snr[j]
        
        # the identifier for Z-N or Z-1 is used for internal statistics, assessing the 
        # passing rate of orientation-corrected files and assumed ZNE files
        
        # assess channels content, checking for both a Z and N component or Z and 1 component
        if ('Z' in channels) and ('N' in channels):
            print_channel = 'Z-N'
            with open('../DATA/SPREADSHEETS/SNR_Summary_Passing.csv', 'a') as f:
                f.write('{},{},{},{},{},{},{},{},{}\n'.format(net,stat,print_channel, event,phase,region,arrival, snr_Z, snr_N))



        if ('Z' in channels) and ('1' in channels):
            print_channel = 'Z-1'
            with open('../DATA/SPREADSHEETS/SNR_Summary_Passing.csv', 'a') as f:
                f.write('{},{},{},{},{},{},{},{},{}\n'.format(net,stat,print_channel, event,phase,region,arrival, snr_Z, snr_1))



snr_pass = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary_Passing.csv')
        
print("\nCalculate {} IterDecon Files".format(len(snr_pass)))



            
            
#%% Calculate the RFs for passing data



os.chdir(active_dir)
snr_pass = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary_Passing.csv')
moveouts = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')


re_use_Iterdecon_Summary = True
if not(re_use_Iterdecon_Summary):
    with open('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', 'w') as outfile:
        outfile.write('Net,Stat,Code,Event,Phase,File,Region,G,Corr,Status,Accept\n')

elif re_use_Iterdecon_Summary:
    continue_summary = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv')




Use_Phases = ['P', 'PP', 'PKP']




def Calc_RF(path, Z_file, R_file, T_file, net, stat, R_comp, event, phase):
    
    # go to the given event to calculate RFs locally
    os.chdir(path)

    # remove old Iterdecon_Results Files
    prev_results = glob('./Iterdecon_Results_*.txt')
    if len(prev_results) != 0:
        for prev_file in prev_results:
            os.remove(prev_file)
    
    
    if R_comp == '1':
        T_comp = '2'
    elif R_comp == 'N':
        T_comp = 'E'
    
    
    files = []
    files = Z_file + R_file + T_file
    for file in files:
        f = file.split('/')[-1]
        # cut = 'echo "cut t0 -10 50; r {}; w append .cut; q" | sac\n'.format(f)
        interpolate = 'echo "r {}.ir.cut; interpolate delta 0.02; w append .dec; q" | sac\n'.format(f)
        preprocess = 'echo "r {}.ir.cut.dec; rmean; rtrend; taper; w append .tap; q" | sac\n'.format(f)
        bandpass = 'echo "r {}.ir.cut.dec.tap; bp n 4 p 2 c 0.1 8.0; taper; w append .bp; q" | sac\n'.format(f)
    
        # open file within event dir with overwrite, write the csh commands, and run the subprocess script
        # this is all happening within the event subdirectory
        with open("./preprocess.bash", "w") as ITD:
            ITD.write('#!/bin/bash\n')
            ITD.write('\n\n')
            # ITD.write(cut)
            ITD.write(interpolate)
            ITD.write(preprocess)
            ITD.writelines(bandpass)
    
        subprocess.run(['bash', './preprocess.bash'])
    
    
    
    
    
    
    if R_comp == '1':
        rot_to_gcp = 'echo "r {}.{}.??1.{}.{}.sac.ir.cut.dec.tap.bp {}.{}.??2.{}.{}.sac.ir.cut.dec.tap.bp; rot to gcp; w {}.{}.1.{}.{}.r {}.{}.2.{}.{}.t; q" | sac\n'.format(net, stat, event, phase,
                                                                                                                                                      net, stat, event, phase, net, stat, event, phase,
                                                                                                                                                      net, stat, event, phase)
        correct_comp = 'echo "r {}.{}.1.{}.{}.r; ch kcmpnm radial; wh; r {}.{}.2.{}.{}.t; ch kcmpnm tangential; wh; q" | sac\n'.format(net, stat, event, phase, net, stat, event, phase)

    
    elif R_comp == 'N':
        rot_to_gcp = 'echo "r {}.{}.??N.{}.{}.sac.ir.cut.dec.tap.bp {}.{}.??E.{}.{}.sac.ir.cut.dec.tap.bp; rot to gcp; w {}.{}.N.{}.{}.r {}.{}.E.{}.{}.t; q" | sac\n'.format(net, stat, event, phase,
                                                                                                                                                   net, stat, event, phase, net, stat, event, phase,
                                                                                                                                                   net, stat, event, phase)
        correct_comp = 'echo "r {}.{}.N.{}.{}.r; ch kcmpnm radial; wh; r {}.{}.E.{}.{}.t; ch kcmpnm tangential; wh; q" | sac\n'.format(net, stat, event, phase, net, stat, event, phase)

    
    
                                                                                                             
    correct_z_comp = 'echo "r {}.{}.??Z.{}.{}.sac.ir.cut.dec.tap.bp; ch kcmpnm vertical; w {}.{}.Z.{}.{}.z; q" | sac\n'.format(net, stat, event, phase,
                                                                                                                    net, stat, event, phase)

    with open("./rotate.bash", "w") as ITD:
        ITD.write('#!/bin/bash\n')
        ITD.write('\n\n')
        ITD.write(rot_to_gcp)
        ITD.write(correct_comp)
        ITD.write(correct_z_comp)

    subprocess.run(['bash', './rotate.bash'])

    

    params = """

    iterations=400          #Number of iterations
    phase=10                #Time before P-arrival
    trough=.001             #Trough filter / Sampling rate
    wave=1                  #Number of Waveforms
    end=0                   #Mystery value at end of .run file that can't change
    Gaussians=(01.0 02.5 05.0 07.5 10.0)    #Array of Gaussian values
                 """
    
    
    
    file_params = 'net={}\nstat={}\nevent={}\nphasetype={}\nrcomp={}\ntcomp={}'.format(net, stat, event, phase, R_comp, T_comp)


    ITD_command = """### This space is to build the ITERDECON function
    # Start looping through gaussian filter values
    for G in ${Gaussians[@]}; do
    echo "Solving Receiver Function"
    echo "iterdecon_batch << EOF" > current.run.bash
    
    echo ${net}.${stat}.${rcomp}.${event}.${phasetype}.r >> current.run.bash
    echo ${net}.${stat}.Z.${event}.${phasetype}.z >> current.run.bash
    echo ${net}.${stat}.${rcomp}.${event}.${phasetype}.${G}.itr >> current.run.bash
    echo ${iterations} >> current.run.bash
    echo ${phase}. >> current.run.bash
    echo ${trough} >> current.run.bash
    echo ${G} >> current.run.bash
    echo ${wave} >> current.run.bash
    echo ${end} >> current.run.bash
    
    echo ${net}.${stat}.${tcomp}.${event}.${phasetype}.t >> current.run.bash
    echo ${net}.${stat}.Z.${event}.${phasetype}.z >> current.run.bash
    echo ${net}.${stat}.${tcomp}.${event}.${phasetype}.${G}.itt >> current.run.bash
    echo ${iterations} >> current.run.bash
    echo ${phase}. >> current.run.bash
    echo ${trough} >> current.run.bash
    echo ${G} >> current.run.bash
    echo ${wave} >> current.run.bash
    echo ${end} >> current.run.bash
    echo "EOF" >> current.run.bash
    
    bash current.run.bash >> Iterdecon_Results_${G}.txt
    
    done            # end of G loop
    
    
    """

    with open("./make_ITD.bash", "w") as ITD:
        ITD.write('#!/bin/bash\n')
        ITD.write('\n\n')
        ITD.write(params)
        ITD.write('\n')
        ITD.write(file_params)
        ITD.write('\n\n')
        ITD.write(ITD_command)

    subprocess.run(['bash', './make_ITD.bash'])


    os.chdir(active_dir)


    return








# cycle through the passing SNR station list
for i in range(len(snr_pass)):
    print("\r", end="")
    print("Calculating RFs... {:.2f}% ".format(i/len(snr_pass)*100), end="")        

    net = snr_pass.Network[i]
    stat = snr_pass.Station[i]
    event = snr_pass.Event[i]
    location = snr_pass.Region[i]
    phase = snr_pass.Phase[i]
    component_combo = snr_pass.Channels[i]
    

    proposed_fname = '{}.{}.{}.{}.{}.02.5.itr'.format(net, stat, component_combo.split('-')[1], event, phase)
    
    
    Z_comp = 'Z'
    R_comp = component_combo.split('-')[1]
    if R_comp == '1':
        T_comp = '2'
    elif R_comp == 'N':
        T_comp = 'E'

    
    snr_Z = snr_pass.Vertical_SNR[i]
    snr_R = snr_pass.Radial_SNR[i]
    
    status_code = 'None'    
    do_IR = True     # we have some response files that do not work, we'll make note of these and pass through them
    
    
    
    if phase not in Use_Phases:
        continue
    
    
    
    if re_use_Iterdecon_Summary:
        if len(continue_summary[continue_summary.File == proposed_fname][continue_summary.Region == location]) > 0:
            continue
        
    



    
    
    # path to sac file suite
    f_path = os.path.join(output_dir, location, 'Data_By_Event', event)
    
    files = []
    Z_file = glob(os.path.join(f_path, '{}.{}.{}.{}.{}.sac'.format(net, stat, '??' + Z_comp, event, phase)))   
    R_file = glob(os.path.join(f_path, '{}.{}.{}.{}.{}.sac'.format(net, stat, '??' + R_comp, event, phase)))   
    T_file = glob(os.path.join(f_path, '{}.{}.{}.{}.{}.sac'.format(net, stat, '??' + T_comp, event, phase)))   

    files = Z_file + R_file + T_file
    
    # Arrival information
    arrive = downloads.ArriveTime[downloads.Network == net][downloads.Station == stat][downloads.Event == event].to_numpy()[0]
    O = downloads.O[downloads.Network == net][downloads.Station == stat][downloads.Event == event].to_numpy()[0]
    B = downloads.B[downloads.Network == net][downloads.Station == stat][downloads.Event == event].to_numpy()[0]

    arrive, O, B = utc(arrive), utc(O), utc(B)
     
    # t0 for trimming
    t0 = arrive - B
    
    
    
    
    # attempt to read the inventory metadata for this station
    pre_filt = [0.01, 0.1, 8.0, 10.0]
    try:
        response = read_inventory(os.path.join(raw_file_loc, 'response_xml', '{}.{}.response.xml'.format(net, stat)))
    except:
        status_code = 'IR Read-Error - IR Removal Failed'
        do_IR = False
        
        
    
    # cycle through file list of station channels
    for f in files:
        fname = f.split('/')[-1]
        st = read(f)
        tr = st[0]
        tr.stats.fname = fname
        
        
        
        # edit SAC header information to work with SAC processing format                
        if ('N' in tr.stats.channel) or ('E' in tr.stats.channel) or ('1' in tr.stats.channel) or ('2' in tr.stats.channel):
            tr.stats.sac.cmpinc = 90
        elif 'Z' in tr.stats.channel:
            tr.stats.sac.cmpinc = 0

        # tr.stats.sac.t0 = 10 

        
        # trim the trace to the arrival window:
        # why do this: typically we would only use t0, however, this is read as time in seconds
        # from the file starttime, which is not gaurunteed to be consistent within a web-download
        # trimming use absolute times relieves us of having to worry about inconsistent 
        # deconvolution time windows
         
        tr.trim(starttime = arrive  - 10, endtime = arrive + 50)
        
        
        

        
        if do_IR:
            try:
                tr.remove_response(inventory = response, pre_filt = pre_filt, output = 'VEL')
    
            except: # header error of some kind
                if ('H' in tr.stats.channel[0]) or ('B' in tr.stats.channel[0]): # pass for broadband stations
                    pass        
                else:
                    status_code = 'IR removal Failed'
                    

        tr.write(f + '.ir.cut', format = 'SAC')

    



    Calc_RF(f_path, Z_file, R_file, T_file, net, stat, R_comp, event, phase)
    
    
    
    
    
    
    
    # QC check for mal-produced RFs (i.e. the deconvolution exploded)
    # to pass the check loop, variable 'good' must be 5, representing that all five 
    # gaussian values have passed
    # OR, when the number of times we've tried this exceeds 10 times
    itr_files = glob(os.path.join(f_path, '{}.{}.{}.{}.{}.*.itr'.format(net, stat, R_comp, event, phase)))
    itt_files = glob(os.path.join(f_path, '{}.{}.{}.{}.{}.*.itt'.format(net, stat, R_comp, event, phase)))

    check_iter = 0
    while 1:
        good = 0
        check_iter += 1

        for file in itr_files:
            st = read(file)
            tr = st[0]
            
            # if found nan values in any trace, calculate new RFs and break the loop
            if np.isnan(tr.data.sum()):
                print("\r", end="")
                print("--- reprocess itr {}".format('-'*check_iter), end="")
                status_code = 'Reprocess Iterdecon'
                Calc_RF(f_path, Z_file, R_file, T_file, net, stat, R_comp, event, phase)

            else:
                good += 1
                
        if good == 5:
            break
        elif check_iter == 10:
            status_code = 'NaN Deconvolution'
            break
        
             
    
    # Copy .itr files to Data_By_Station directory
    for file in itr_files:
        subprocess.run(["cp", "{}".format(file), "../DATA/{}/Data_By_Station/{}_{}/".format(location, net, stat)])
    for file in itt_files:
        subprocess.run(["cp", "{}".format(file), "../DATA/{}/Data_By_Station/{}_{}/".format(location, net, stat)])
        
        
        
    
    # add deconvolution data to iterdecon summary
    for f in itr_files:
            st = read(f)
            tr = st[0]
            
            with open('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', 'a') as outfile:
              #  outfile.write('Net,Stat,Code,Event,Phase,File,Region,Corr,Accept')
                outfile.write('{},{},{},{},{},{},{},{},{:.4f},{},{}\n'.format(net, stat, net+'_'+stat, event, phase, f.split('/')[-1], location, tr.stats.sac.user0, tr.stats.sac.user9, status_code, 'Check'))



    # if 'RF01' in code:
    #     sys.exit('here')

        

#%% Post-RF Calc itr move


## Run this cell if for some reason the .itr file did not copy over to Data_By_Station area

itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', )


for i in range(len(itd)):
    net = itd.Net[i]
    stat = itd.Stat[i]
    event = itd.Event[i]
    location = itd.Region[i]
    code = itd.Code[i]
    
    file = itd.File[i]
    rcomp = file.split('.')[2]
    if rcomp == '1':
        tcomp = '2'
    elif rcomp == 'N':
        tcomp = 'E'
        
        
    itt_file = file.replace('.'+rcomp+'.', '.' + tcomp + '.')[0:-1] + 't'
    
    
    
    
    subprocess.run(["cp", "../DATA/{}/Data_By_Event/{}/{}".format(location, event, file),"../DATA/{}/Data_By_Station/{}".format(location, code)])
    subprocess.run(["cp", "../DATA/{}/Data_By_Event/{}/{}".format(location, event, itt_file),"../DATA/{}/Data_By_Station/{}".format(location, code)])
               





