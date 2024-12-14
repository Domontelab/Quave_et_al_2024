#!/usr/bin/env python

# coding: utf-8

'''
*Author: Milagros Marin Alejo (Milagros.MarinAlejo@uth.tmc.edu)*
May 2023

Analysis of the rat's neuronal activity and spatial representation in the chamber based on ephys and behavioral data.

Dimensions of the cage: 31cm x 62cm, each side being 31cm square

Description of this script: 
1. Import necessary libraries.
2. Define the functions to run the pipeline.
3. Set analysis variables based on task type (Conflict or Preference), including likelihood thresholds, number of bins, input files, etc.
4. Load inputs: ephys files, video files (`.mp4`), and body tracking data (`.csv`), along with metadata such as frames per second (fps), delay, and paired side.
5. Run analyses for each animal.
6. Generate outputs:
   - `.pkl` files with z-score matrices for excitatory and inhibitory responses by group (risk-takers, risk-avoiders, saline group).
   - A log file (`.txt`) detailing analysis steps and results.
   - Figures showing body tracking and likelihood from DeepLabCut, spatial mapping, and time and z-score distributions per bin.
'''

# In[1]:
# =============================================================================
# #### Libraries
# =============================================================================

import math
import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import moviepy.editor as vid
import numpy as np
import pandas as pd

# In[2]:
# =============================================================================
# #### Setup
# =============================================================================

num_x_bins = 12
num_y_bins = 6
num_bins =(num_x_bins * num_y_bins) # 72 bins
likelihood_threshold = 0.8 # 80% of likelihood
minimum_n_frames_filter = 5 #frames

new_square = True
temporal_bin_zscore = 6 #seconds of temporal bins to calculate av. and std. of firing rate for zscore formula; For the interval size of frames for z_score: 100 bins in 600s = 6s * fps = 87.72 frames = 88 frames 

# In[2]:
# =============================================================================
# #### Functions
# =============================================================================

def grid_calculation(frame1,num_x_bins,num_y_bins, video_name,num_frames,square_coords='', new_square=False):
    '''1. Extraction of the cage size and define the bins (grid)'''
    
    if new_square:
        
        # Show the image and wait for the user to select a square
        cv2.imshow('Select a square', frame1)
        points = cv2.selectROI('Select a square', frame1, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        # Extract the coordinates of the square
        x, y, w, h = points
        square_coords = {
            'top_left': (x, y),
            'top_right': (x + w, y),
            'bottom_left': (x, y + h),
            'bottom_right': (x + w, y + h)
        }
        
        #Save variable square_coords
        f = open('square_coords_{}_{}bins_{}frames.pkl'.format(video_name,str(num_bins),str(num_frames)), 'wb')
        pickle.dump(square_coords,f)
        f.close()
    else:
        try: square_coords
        except NameError: square_coords = {'top_left': (72, 69), 'top_right': (375, 69), 'bottom_left': (72, 219), 'bottom_right': (375, 219)}

    # Determine the range of values for x and y
    x_range = square_coords['bottom_right'][0] - square_coords['bottom_left'][0]
    y_range = square_coords['bottom_left'][1] - square_coords['top_left'][1]
    
    # Determine the size of the bin
    bin_size = np.sqrt((x_range * y_range) / (num_x_bins * num_y_bins))

    # Round the bin size to the nearest 0.5
    bin_size = np.floor(bin_size * 2) / 2

    print("Background dimensions: ",square_coords)
    print("Number of bins along x-axis:", num_x_bins)
    print("Number of bins along y-axis:", num_y_bins)
    
    xbins_vector = np.linspace(square_coords['bottom_left'][0], square_coords['bottom_right'][0], num_x_bins)  # 50 bins in the x direction
    ybins_vector = np.linspace(square_coords['top_left'][1], square_coords['bottom_left'][1], num_y_bins)  # 50 bins in the y direction

    return xbins_vector,ybins_vector,square_coords

def extract_plot_x_y_likelihood(likelihood_threshold,body_part,data,delay,reference_duration,plotting=False, save=False):
    '''
    2. Extracting DLC coordinates, cut them from the delay time and the reference_duration, and filter by likelihood using NaN.
    '''
    
    data.columns = data.iloc[0]
    line_number_len = len(data.index)-2
    line_number_list = np.arange(line_number_len)
    data2 = data.copy()
    columns_body_part = data2.pop(body_part)
    x = np.asarray(columns_body_part.iloc[2:line_number_list[-1]+3,0]).astype(float)
    y = np.asarray(columns_body_part.iloc[2:line_number_list[-1]+3,1]).astype(float)
    likelihood = np.asarray(columns_body_part.iloc[2:line_number_list[-1]+3,2]).astype(float)

    x_good = []
    y_good = []

    for i in range(0,len(likelihood)):
        if (float(likelihood[i])<likelihood_threshold) or (float(likelihood[i])>1.0):
            x_good.append(np.nan)
            y_good.append(np.nan)
        else:
            x_good.append(x[i])
            y_good.append(y[i])
    
    # Coordinates after filtering by likelihood
    x_good = np.array(x_good)
    y_good = np.array(y_good) 

    if plotting:
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True,figsize=(14, 10))
        plt.suptitle('{} - {} positions (blue) and NaN (red) (likelihood>{}% = {}%) \n From {} frames, {} frames are NaN'.format(video_name, body_part, likelihood_threshold*100,round((np.count_nonzero(np.isnan(x_good))/line_number_len)*100,2),len(x_good),len(x_good[np.isnan(x_good)])))

        # x coordinate
        ax0.plot(x_good,zorder=1)
        ax0.scatter(np.argwhere(np.isnan(x_good)),np.ones(len(np.argwhere(np.isnan(x_good)))), color='red',s=4,zorder=2)
        ax0.set_ylabel('X coordinate (pixels)')

        # y coordinate
        ax1.plot(y_good,zorder=1)
        ax1.scatter(np.argwhere(np.isnan(y_good)),np.ones(len(np.argwhere(np.isnan(y_good)))), color='red',s=4,zorder=2)
        ax1.set_ylabel('Y coordinate (pixels)')

        #likelihood
        ax2.plot(np.arange(len(likelihood)),likelihood,zorder=1)
        ax2.scatter(np.argwhere(np.isnan(x_good)),likelihood[np.argwhere(np.isnan(x_good))], color='red',s=4,zorder=2)

        #plt.ylim(0,1)
        ax2.set_xlabel('Time (h)')
        ax2.set_ylabel('Likelihood')
        ax0.set_ylim(0,width)
        ax0.set_xlim(0)
        ax1.set_ylim(0,height)
        ax1.set_xlim(0)
        ax2.set_xlim(0)
        if save:
            plt.savefig('x_y_likelihood_{}_{}.jpeg'.format(body_part,video_name), dpi=600)
        plt.show()

    return x_good, y_good 

def allign_behavior_and_neuronal_data(x1,y1,nd,fps,reference_duration,delay,neuron_name):
    '''
    Cutting at the beginning and end of th length of the total video (total frames). Not considering NaN positions from likelihood filtering.
    nd_ms_mod still contains values in the frame positions that are NaN from filtering by threshold.
    '''
    
    length_beh_data = round(len(x1)/fps,3) #sec. (counting NaNs from likelihood), length_beh_data = 600.
        
    # If video lasts more, or less, than 10 min:
    if reference_duration == length_beh_data:
        x1_mod = x1
        y1_mod = y1
    
    elif reference_duration < length_beh_data:
        x1_mod = x1[0:int(reference_duration*fps)]
        y1_mod = y1[0:int(reference_duration*fps)]
        
    elif reference_duration > length_beh_data:
        reference_duration = length_beh_data
        x1_mod = x1
        y1_mod = y1
        print('Length of behavioral video (with NaNs of likelihood filter) is less than 600sec, the video lasts ', reference_duration, ' s')
    
    print('Reference duration = ',reference_duration,' s')

    # If delay is positive or negative: 
    ## Note: x_mod and y_mod are going to last 10min, or maybe less if video is shorter.
    if delay >= 0.:
        nd_ms_mod = nd['{}'.format(neuron_name)][int(np.floor(delay*1000)):int((delay+reference_duration)*1000)] #delay is in seconds, to ms ; end is in frames to seconds, to ms
        nd_ms_mod = np.array(nd_ms_mod)

    else: #if delay is negative, then the first frame should be cut and aligned
        #cut frames based on negative delay
        frames_to_cut = math.ceil(abs(delay*fps)) #frames
        
        #Cut the behavioral data based on the frames to cut
        x1_mod = x1[frames_to_cut:int(reference_duration*fps)]
        y1_mod = y1[frames_to_cut:int(reference_duration*fps)]
        total_time_beh_data = len(x1_mod)/fps
        
        #align the neuronal recording to the cut in the video
        time_nd_to_cut = round((frames_to_cut+delay*fps)/fps*1000)
        nd_ms_mod = nd['{}'.format(neuron_name)][time_nd_to_cut:time_nd_to_cut + int(total_time_beh_data*1000)] 
        nd_ms_mod = np.array(nd_ms_mod)
        total_time_neuron_data =len(nd_ms_mod)/1000
        print('The behavioral video has been cut {} frames ({} ms), with a total duration of {} s'.format(frames_to_cut,round(frames_to_cut/fps*1000,2),total_time_beh_data))
        print('The neuronal recording has been cut from {} ms to {} ms (total of {} s)'.format(round((frames_to_cut+delay*fps)/fps*1000),round(total_time_beh_data+(frames_to_cut+delay*fps)/fps*1000),total_time_neuron_data))
        
    return x1_mod,y1_mod,nd_ms_mod,reference_duration

def nd_to_frames (nd_ms_mod,x1_mod,fps,path,title_file):
    '''
    Transform the time from the neuronal recordings (1-ms) to the frame time, averaging the spikes between frames, and starting from the Delay time.
    Substitute to NaN all the neuronal recordings that fall into the frames that are NaN. 

    Parameters
    ----------
    nd_ms_mod : numpy array
        Neuronal recordings of all the cells from one video and the bin left (time in 1ms). Vector starting at 0, after cutting the data.

    Returns
    -------
    nd_frames: numpy array
        List containing the accumulated spikes per interval of frames. 

    '''
    if type(nd_ms_mod) == pd.DataFrame:
        nd_ms_mod = nd_ms_mod.to_numpy()
    
    nd_frames = []
    interval_size= (1/fps) * 1000 #ms
    
    init_interval = 0
    end_interval = init_interval + interval_size #ms
    sum_spikes = 0
    count_interval = 0
    list_frames = [] #This starts AFTER cutting the first frames of the behavioral data to align with neuronal recordings.
    list_timestamps = [] #ms. This starts at time 0. Example, if list_timstamps is 39 ms, then we should add the cut ms if delay is negative: 39+26 = 65 ms.
    count_interval2 = 0
    spike_bool= 0

    for ms in np.arange(0,len(nd_ms_mod)):
        if ms >= init_interval and ms < end_interval:
            sum_spikes += nd_ms_mod[ms]/1000
            count_interval = 1  #This is for spikes/interval = spikes/frame
            if nd_ms_mod[ms]>0:
                list_timestamps.append(ms)
                spike_bool = 1
        else:
            if count_interval > 0:
                mean_spikes = sum_spikes / count_interval 
                nd_frames.append(mean_spikes)
                
                if spike_bool==1:
                    list_frames.append(count_interval2)
                count_interval2 += 1 #This is for spikes/ms    

                #Restart interval
                init_interval = end_interval
                end_interval += interval_size
                sum_spikes = 0
                count_interval = 0
                spike_bool = 0
                
    #Ensure that the last interval is added to the second vector
    if len(nd_frames) < len(x1_mod):
        if count_interval > 0:
            mean_spikes = sum_spikes / count_interval 

        nd_frames.append(mean_spikes)
        if spike_bool==1:
            list_frames.append(count_interval2)
        count_interval2 += 1 #This is for spikes/ms

    nd_frames= np.asarray(nd_frames)

    print('After grouping by frames, the neuronal recording (nd_frames) has a length of {} s\n'.format(len(nd_frames)*(1/fps)))
    
    #Substitute by NaN all the neuronal data that fall in the NaN frames from filtering by likelihood
    nd_frames[np.isnan(x1_mod)] = np.nan
    print('Now the NaN positions after threshold filtering are substitute also in the nd_frames ({} elements)'.format(len(nd_frames[np.isnan(nd_frames)])))
    #Save vector in a file
    np.save('{}/nd_frames_{}.npy'.format(path,title_file), nd_frames)
    return nd_frames
 
def plot_spatial_mapping(body_part, x1,y1,frame1,square_coords,xbins_vector,ybins_vector,cat_odor='',title='',save=False,path='',video_name='',number_bins='',number_frames=''):
    ''' 3. Plot spatial mapping'''
    fig = plt.figure(figsize=(14, 12))
    ax0 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2) 
    ax0.set_facecolor('gray')
    ax2.set_facecolor('gray')

    ### Subplot1 and subplot2: trajectories
    ax0.scatter(x1,y1,label='x versus y coordinates', alpha=0.7)
    ax0.imshow(frame1)    
        
    #Heatmap calculation: this heatmap represents the number of pixels that have covered each of the bins. It is not based on the time spent in each bin and velocity of the animal. 
    #square_coords
    heatmap, xedges, yedges = np.histogram2d(x1,y1,bins=(len(xbins_vector), len(ybins_vector)),range=[[square_coords['top_left'][0],square_coords['top_right'][0]],[square_coords['top_left'][1],square_coords['bottom_left'][1]]])
    
    #Substitute 0. to NaN = squares with no position of head are going to be NaN.
    nan_positions = np.full(heatmap.shape,False)
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i,j] == 0.:
                heatmap[i,j] = np.nan
                nan_positions[i,j] = True  

    print('Heatmap of pixels per spatial bin: substitution of bins with 0 positions with a NaN (and saved in nan_positions array): ',sum(sum(nan_positions)))   
    print('Total time in the heatmap = ',sum(sum(heatmap/fps)),' (',sum(sum(heatmap)),') frames')
    
    ### Subplot3: heatmap
    image = ax2.imshow(heatmap.T, cmap='viridis',extent=[xedges[0], xedges[-1], yedges[-1],yedges[0]])

    #Plot the cbar
    cbar = plt.colorbar(image)
    cbar.set_label('# Pixels', rotation=270)
    cbar.ax.get_yaxis().labelpad = 20
    
    # grid
    for xx in xedges:
        ax0.axvline(xx, color='lime', linewidth=0.7)
        #ax1.axvline(xx, color='lime', linewidth=0.7)
    for yy in yedges:
        ax0.axhline(yy, color='lime', linewidth=0.7)
        #ax1.axhline(yy, color='lime', linewidth=0.7) 
    
    for i in np.arange(heatmap.T.shape[0]): #yaxis
        for j in np.arange(heatmap.T.shape[1]): #xaxis
            text = ax2.text(xedges[j],yedges[i], heatmap.T[i,j],
                           ha="center", va="center", color="black")        

    #IMPORTANT: FLIP REPRESENTATIONS IF CAT ODOR ON THE LEFT:
    if cat_odor == 'S':
        ax0.invert_xaxis()
        ax2.invert_xaxis()
        print('Cat odor is on the strides side (left). Figure has been flipped, so paired side is now on the right ')
        fig.suptitle('{} trajectory during the video "{}" - {} bins \n Cat odor is on the strides side (left). Figure has been flipped (paired side is on the right)'.format(body_part, video_name,len(xbins_vector)*len(ybins_vector)))
    
    elif cat_odor == 'D':
        fig.suptitle('{} trajectory during the video "{}" - {} bins \n Cat odor is on the dots side (paired side is on the right)'.format(body_part, video_name,len(xbins_vector)*len(ybins_vector)))
    
    else:
        sys.exit("ERROR: Define the cat_odor position!")  
     
    plt.tight_layout()

    if save:
        plt.savefig('{}/spatial_mapping_{}_{}_frames_.jpeg'.format(path,video_name,len(xbins_vector)*len(ybins_vector),), dpi=600)
         
    plt.show()
    
    return nan_positions,xedges,yedges # data is not flipped, only figure
    
def calculate_time_spent_and_Hz_per_bin(x1,y1,nd_frames,nan_positions,xbins_vector,ybins_vector,minimum_n_frames_filter=0):
    """
    This is for one neuron
    
    bins_ms_times: time spent per bin
    bins_Hz_times: Hz per bin
    """
    bin_x = []
    bin_y = []
    #number of bins including the extra two in the beginning and end of the grid:
    grid_outliers_x = np.arange(0, len(xbins_vector)+1)
    grid_outliers_y = np.arange(0, len(ybins_vector)+1)
    for i in np.arange(0,len(x1)):
        bin_index_x = np.searchsorted(xbins_vector, x1[i], side='right')
        bin_index_y = np.searchsorted(ybins_vector, y1[i], side='right')
        if bin_index_x == grid_outliers_x[0] or bin_index_x == grid_outliers_x[-1] or bin_index_y == grid_outliers_y[0] or bin_index_y == grid_outliers_y[-1]:
            bin_x.append(np.nan)
            bin_y.append(np.nan)
        else:
            bin_x.append(bin_index_x)
            bin_y.append(bin_index_y)        
          
    frame_bins = []
    for i in range(len(bin_x)):
        frame_bins.append(tuple([bin_x[i]-1,bin_y[i]-1])) #[row, column]. Correcting the bins -1 .
    frame_bins = np.array(frame_bins) #This list contains frame bins for NaN elements too.
    
    #Fulfill the list of bins with (1) duration of animal in each bin and (2) total number of spikes in each bin
    bins_nd_times = np.zeros([len(xbins_vector)-1,len(ybins_vector)-1])
    bins_ms_times = np.zeros([len(xbins_vector)-1,len(ybins_vector)-1])
    bins_Hz_times = np.zeros([len(xbins_vector)-1,len(ybins_vector)-1])
    
    example_frames_bin_0_0 = [] #frames that falls into a specific bin
    for ind in np.arange(0,len(frame_bins)): 
        if ~np.isnan(frame_bins[ind][0]) or ~np.isnan(frame_bins[ind][1]):
            i = int(frame_bins[ind][0])
            j = int(frame_bins[ind][1])
            if i==0 and j==0:
                example_frames_bin_0_0.append(ind)
            bins_ms_times[i,j] += 1/fps*1000 #ms
            if ~np.isnan(nd_frames[ind]):
                bins_nd_times[i,j] += nd_frames[ind] #[14,7]
                
    for i in np.arange(bins_Hz_times.shape[0]):
        for j in np.arange(bins_Hz_times.shape[1]):
            if bins_ms_times[i,j]>=minimum_n_frames_filter/fps*1000:
                bins_Hz_times[i,j] = bins_nd_times[i,j]/(bins_ms_times[i,j]/1000) #from ms to s
            else:
                bins_Hz_times[i,j] = np.nan
                
    # Delete the values for squares that have no positions
    for i in range(nan_positions.shape[0]):
        for j in range(nan_positions.shape[1]):
            if nan_positions[i,j] == True:
                bins_nd_times[i,j] = np.nan
                bins_ms_times[i,j] = np.nan 
                bins_Hz_times[i,j] = np.nan
                
    print('bins_ms_times on which bins_Hz_times is based (before filtering by minimum number of frames) = {}'.format(sum(sum(bins_ms_times))))
    
    return bins_ms_times,bins_Hz_times #if bins_Hz_times has any NaN, means that the head was never into the bin.

def calculate_z_score_per_bin(bins_Hz_times,nd_frames,fps,interval_size_frames,log_file):
    """
    This is for one neuron
    """
    
    # 1. Create a vector of Hz per intervals of frames
    init_interval = 0
    end_interval = init_interval + interval_size_frames
    sum_spikes = 0
    count_interval = 0
    Hz_per_interval_list= []
    
    for frame in np.arange(len(nd_frames)):
        if frame >= init_interval and frame < end_interval and ~np.isnan(nd_frames[frame]):          
            sum_spikes += nd_frames[frame] #spike
            count_interval += 1 #number of frames. This is for dividing spikes/s = Hz
        elif np.isnan(nd_frames[frame]):
            continue
        else:
            if count_interval > 0:
                Hz_per_interval = sum_spikes / (count_interval/fps) 
                Hz_per_interval_list.append(Hz_per_interval)
            
                #Restart interval
                init_interval = end_interval
                end_interval += interval_size_frames
                sum_spikes = 0
                count_interval = 0
    
    #Ensure that the last interval is added to the second vector
    Hz_per_interval = sum_spikes / (count_interval/fps) if count_interval > 0 else 0
    Hz_per_interval_list.append(Hz_per_interval)
    Hz_per_interval_list= np.asarray(Hz_per_interval_list)
    
    #2. Calculate the average and std of the whole session for one neuron
    av_Hz_per_interval = np.mean(Hz_per_interval_list)
    std_Hz_per_interval = np.std(Hz_per_interval_list)
        
    print(['For a bin (interval) of ',interval_size_frames,'frames (= ',round(interval_size_frames/fps,4),'sec), the Hz/bin =',round(av_Hz_per_interval,4),' +- ',round(std_Hz_per_interval,4)])
    
    #3. Z_score
    z_score_matrix = (bins_Hz_times - av_Hz_per_interval)/std_Hz_per_interval
    
    return z_score_matrix, av_Hz_per_interval,std_Hz_per_interval

def plot_heatmaps_and_zscore(bins_ms_times,bins_Hz_times,minimum_n_frames_filter,number_bins, av_Hz_per_interval,std_Hz_per_interval,z_score_matrix,neuron_name='',cat_odor='',video_name='',plotting=True,save=False):
    """
    This is for one neuron
    """    
    fig = plt.figure(figsize=(14, 12))
    ax0 = plt.subplot(3,1,1)
    ax3 = plt.subplot(3,1,2)
    ax1 = plt.subplot(3,1,3) 
    ax1.set_facecolor('gray')
    ax0.set_facecolor('gray')
    ax3.set_facecolor('gray')

    #subplot1
    image = ax0.imshow(bins_ms_times.T/1000, cmap='viridis')
    cbar = plt.colorbar(image)
    cbar.set_label('Time spent (s)', rotation=270)
    cbar.ax.get_yaxis().labelpad = 20
    ax0.locator_params(axis='y',nbins=bins_ms_times.T.shape[0])        

    #subplot3
    image2 = ax1.imshow(z_score_matrix.T, cmap='bwr', vmin=-4, vmax=4)
    cbar2 = plt.colorbar(image2,extend='both')
    cbar2.set_label('z-score', rotation=270)
    cbar2.ax.get_yaxis().labelpad = 20
    ax1.locator_params(axis='y',nbins=bins_ms_times.T.shape[0])        
    
    #subplot2
    image3 = ax3.imshow(bins_Hz_times.T, cmap='viridis')
    cbar3 = plt.colorbar(image3,extend='both')
    cbar3.set_label('Firing frequency (Hz) over {} frames ({} s)'.format(minimum_n_frames_filter,round(minimum_n_frames_filter*(1/fps),3)), rotation=270)
    cbar3.ax.get_yaxis().labelpad = 20
    ax3.locator_params(axis='y',nbins=bins_Hz_times.T.shape[0])        
    
    fig_width, fig_height = plt.gcf().get_size_inches()
    
    # Loop over data dimensions and create text annotations on heatmaps
    for i in range(bins_ms_times.shape[0]):
        for j in range(bins_ms_times.shape[1]):
            text = ax0.text(i,j, round(bins_ms_times[i,j]/1000,2),
                           ha="center", va="center", color="white")        
    # Loop over data dimensions and create text annotations on heatmaps
    for i in range(z_score_matrix.shape[0]):
        for j in range(z_score_matrix.shape[1]):
            text = ax1.text(i,j, round(z_score_matrix[i, j],2),
                           ha="center", va="center", color="black")
    # Loop over data dimensions and create text annotations on heatmaps
    for i in range(bins_Hz_times.shape[0]):
        for j in range(bins_Hz_times.shape[1]):
            text = ax3.text(i,j, round(bins_Hz_times[i, j],2),
                           ha="center", va="center", color="white")  
              
    #IMPORTANT: FLIP REPRESENTATIONS IF CAT ODOR ON THE LEFT:
    if cat_odor == 'S':
        ax0.invert_xaxis()
        ax1.invert_xaxis()
        ax3.invert_xaxis()
        print('Cat odor is on the strides side (left). Figure has been flipped, so paired side is now on the right ')
        fig.suptitle('neuron {} video "{}" \n Cat odor is on the strides side (left). Figure has been flipped (paired side is on the right) \n Av.Hz per session = {} +- {} Hz'.format(neuron_name, video_name, round(av_Hz_per_interval,2),round(std_Hz_per_interval,2)))
    
    elif cat_odor == 'D':
        fig.suptitle('neuron {} video "{}" \n Cat odor is on the dots side (paired side is on the right) \n Av.Hz per session = {} +- {} Hz'.format(neuron_name, video_name, round(av_Hz_per_interval,2),round(std_Hz_per_interval,2)))

    else:
        sys.exit("ERROR: Define the cat_odor position!")

    plt.tight_layout()

    if save:
        plt.savefig('{}/time_spent_and_zscore_{}_{}_bins{}_frames{}.jpeg'.format(path,video_name,neuron_name,number_bins,minimum_n_frames_filter), dpi=600)
        
        



# In[3]:
           
# =============================================================================
# #### Import data and video from BEHAVIOR and NEURONAL RECORDINGS
# =============================================================================

log_file = open('log_bins{}_frames{}.txt'.format(num_bins,minimum_n_frames_filter),'w')
sys.stdout = log_file #redirect the standard output to the log file

fps_list = [14.99, 15.00, 14.62, 15.00, 14.58, 14.61, 15.00, 14.59, 14.42, 13.83, 14.58, 13.59, 13.31, 15.00, 13.95, 13.95, 13.67, 15.00, 14.64] #21

delay_list = [841.666865, 773.3983724, 727.5445321, 722.6872185, 817.3423601, 699.413735, 756.7929275,863.25299, 795.8903522, 717.7722145, 730.7924355, 805.7969429,
              819.293715, 794.6837295, 769.9164611, 821.0536723,  770.2042957, 846.4879696,759.407153]

paired_side_list = ['D','S','D','D','S','D','D','D','S','D','S','S','D','S','D','D','S','S','D','D','S']

ephys_files = ["C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/4A5 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/6P4 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/1A4 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/1A5 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/4W5 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/2A5 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/3P4 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/5A4 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/5W5 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/2P4 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/7A5 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/6W5 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/5P4 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/5A5 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/7W5 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/4P4 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/6A4 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/8W5 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/4U5 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/7U5 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/9W5 CPP (1 ms bins).csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/CPP_Ephys_Data/5M4 CPP (1 ms bins).csv"] 

video_csv = ["C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 3_4A5_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 4_6P4_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 5_1A4_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 5_1A5_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 5_4W5_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 7_2A5_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 7_3P4_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 7_5A4_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 8_5W5_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 9_2P4_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 9_7A5_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 10_6W5_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 11_5P4_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 12_5A5_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 12_7W5_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 13_4P4_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 13_6A4_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 14_8W5_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 16_4U5_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 18_7U5_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 20_9W5_CONFLICTDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 1_5M4_PREF AND CONFLICT_PREFDLC_resnet50_Test3_dlc_network_CPP_CanaMay2shuffle1_100000.csv"] #21

video_mp4 = ["C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 3_4A5_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 4_6P4_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 5_1A4_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 5_1A5_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 5_4W5_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 7_2A5_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 7_3P4_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 7_5A4_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 8_5W5_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 9_2P4_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 9_7A5_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 10_6W5_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 11_5P4_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 12_5A5_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 12_7W5_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 13_4P4_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 13_6A4_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 14_8W5_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 16_4U5_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 18_7U5_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Conflict_Videos/Test 20_9W5_CONFLICT.mp4",
"C:/Users/mmarinalejo/Documents/Test3_dlc_CPP_Cana/Cana CPP Conflict Analysis/Preference_Videos/Test 1_5M4_PREF AND CONFLICT_PREF.mp4"]

# In[4]:
# =============================================================================
# #### Run analysis
# =============================================================================

risk_takers = [3,4,13,16,1,10,20]
risk_avoiders = [5,7,8,11,19,14] 
saline = [2,9,6,0,18,12,21]

zscore_results = dict()
zscore_results['unpaired_side'] = dict()
zscore_results['unpaired_side']['excitatory'] = []
zscore_results['unpaired_side']['inhibitory'] = []
zscore_results['paired_side'] = dict()
zscore_results['paired_side']['excitatory'] = []
zscore_results['paired_side']['inhibitory'] = []
zscore_results['both_sides'] = dict()
zscore_results['both_sides']['excitatory'] = []
zscore_results['both_sides']['inhibitory'] = []
zscore_results['non_response'] = []


for i in range(0,len(ephys_files)): 
    np_path = ephys_files[i]
    video_path = video_csv[i]
    clips_path = video_mp4[i]
    video_name = clips_path[95:110]
    fps = fps_list[i]
    delay = delay_list[i]
    cat_odor = paired_side_list[i]
    interval_size_frames = round(temporal_bin_zscore * fps) # from 6 seconds to the specific quantity of frames for each video

    print('\n ########################################################')
    print(' ##########   {}  ##########'.format(video_name))
    print(' - Neuronal recording file: ',np_path)
    print(' - Video csv: ',video_path)
    print(' - Video mp4: ',clips_path)
    print(' - fps = ',fps)
    print(' - delay = ',delay, 's')
    print(' - paired side = ',cat_odor)
    
    #for the loop of videos
    nd = pd.read_csv("{}".format(np_path))
    behavior_data = pd.read_csv("{}".format(video_path)) #behavior_data has 3 headers. Consider when using an example dataset. 
    clip = vid.VideoFileClip("{}".format(clips_path))

    # Video properties
    width = clip.w
    height = clip.h
    duration = clip.duration
    reference_duration = 600
    
    #Create a folder for the specific video
    path = os.getcwd()
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print("\n A new directory is created for {}".format(path))    
        
    # Example image for background
    frame1 = clip.get_frame(0)
    
    #Calculate the bins for the heatmap
    if new_square == False:
            f = open('square_coords_{}_21bins_0frames.pkl'.format(video_name),'rb')
            square_coords = pickle.load(f)
            f.close()
            xbins_vector,ybins_vector,square_coords = grid_calculation(frame1,num_x_bins,num_y_bins,video_name,minimum_n_frames_filter, square_coords,new_square=False)
    else:
        xbins_vector,ybins_vector,square_coords = grid_calculation(frame1,num_x_bins,num_y_bins,video_name,minimum_n_frames_filter,square_coords='',new_square=True)
    
    #Filter the coordinates based on likelihood (and plot them if desired)
    x_head_good, y_head_good =extract_plot_x_y_likelihood(likelihood_threshold,'Head',behavior_data,delay,reference_duration,plotting=True, save=True)
    
    print('From {} frames ({}s), {} frames ({}s) are NaN after filtering by likelihood '.format(len(x_head_good),len(x_head_good)/fps,len(x_head_good[np.isnan(x_head_good)]),len(x_head_good[np.isnan(x_head_good)])/fps))
    
    #Neuronal recordings properties
    neurons = nd.columns[1:]
    print(neurons)
    
    #Extract info and plot spatial mapping: video, grid, and heatmap
    nan_positions,xedges,yedges = plot_spatial_mapping('Head', x_head_good,y_head_good,frame1,square_coords,xbins_vector,ybins_vector,cat_odor,title='',save=True,path=path,video_name = video_name,number_bins=num_bins,number_frames=minimum_n_frames_filter)
    
    #At this point, x_head_good and y_head_good have NaN in the positions with likelihood < 0.8.
    #Also, heatmap was represented with NaN where positions inside a bin is 0. These NaN positions are saved in nan_positions array. 
    
    video_index = i 
    
    for neuron in neurons:
        
        #Average the neuronal recording spikes from ms to s-bins, and starting from the delay time 
        x1_mod,y1_mod,nd_ms_mod,reference_duration = allign_behavior_and_neuronal_data(x_head_good,y_head_good,nd,fps,reference_duration,delay,neuron)
        nd_frames = nd_to_frames(nd_ms_mod,x1_mod,fps,path,'{}_{}_{}_{}bins'.format('Head',video_name,neuron,num_bins))
        #At this point, both x1_mod, y1_mod and nd_frames have the same frame length and same NaN positions for threshold filtering, in the same positions.

        bins_ms_times,bins_Hz_times = calculate_time_spent_and_Hz_per_bin(x1_mod,y1_mod,nd_frames,nan_positions,xedges,yedges,minimum_n_frames_filter=minimum_n_frames_filter)
        z_score_matrix, av_Hz_per_interval,std_Hz_per_interval = calculate_z_score_per_bin(bins_Hz_times,nd_frames,fps,interval_size_frames=interval_size_frames,log_file=log_file)
        plot_heatmaps_and_zscore(bins_ms_times,bins_Hz_times,minimum_n_frames_filter,num_bins,av_Hz_per_interval,std_Hz_per_interval,z_score_matrix,spatial_information=0.,neuron_name=neuron,cat_odor=cat_odor,video_name=video_name,plotting=True,save=True)

        #z score analysis: create a matrix with 1 for excitation and -1 for inhibition
        zscore_analysis_matrix = np.zeros(z_score_matrix.shape) #zscore_analysis_matrix[y,x] *row,column
        for j in np.arange(z_score_matrix.shape[1]): #column
            for i in np.arange(z_score_matrix.shape[0]): #row
                
                if len(xbins_vector)/2 % 1 == 0: #if number of columns is even
                    if j<= int(len(xbins_vector)/2):    #left side
                        if z_score_matrix[i,j]<=(-1.96):
                            zscore_analysis_matrix[i,j] = -1  
                        elif z_score_matrix[i,j]>=2.58:
                            zscore_analysis_matrix[i,j] = 1
                
                    if j > int(len(xbins_vector)/2):    #right side
                        if z_score_matrix[i,j]<=(-1.96):
                            zscore_analysis_matrix[i,j] = -1
                        elif z_score_matrix[i,j]>=2.58:
                            zscore_analysis_matrix[i,j] = 1
                
                else: #if number of columns is odd number
                    if j<= int(np.floor(len(xbins_vector)/2)):    #left side
                        if z_score_matrix[i,j]<=(-1.96):
                            zscore_analysis_matrix[i,j] = -1
                        elif z_score_matrix[i,j]>=2.58:
                            zscore_analysis_matrix[i,j] = 1
                
                    if j > int(np.floor(len(xbins_vector)/2))+1:    #right side including the center column
                        if z_score_matrix[i,j]<=(-1.96):
                            zscore_analysis_matrix[i,j] = -1
                        elif z_score_matrix[i,j]>=2.58:
                            zscore_analysis_matrix[i,j] = 1

        
        #the paired side is going to be at the right side now for every video
        if cat_odor =='S':
            zscore_analysis_matrix = np.flip(zscore_analysis_matrix, axis=1)
            print('z-score matrix (flipped): \n {}'.format(zscore_analysis_matrix.T))         
        elif cat_odor == 'D':
            print('z-score matrix: \n {}'.format(zscore_analysis_matrix.T))         
        else:
            sys.exit("ERROR: Define the cat_odor position!")

        #Transpose the matrix as the heatmap
        zscore_analysis_matrix = zscore_analysis_matrix.T
        
        
        #######classify in the dict
        ### excitatory
        if (np.any(zscore_analysis_matrix[:,0:int(len(xbins_vector)/2)] == 1)) and (np.any(zscore_analysis_matrix[:,int(len(xbins_vector)/2)-1:-1] == 1)):
            zscore_results['both_sides']['excitatory'].append('{}_{}'.format(video_name,neuron))
            
        elif (np.any(zscore_analysis_matrix[:,0:int(len(xbins_vector)/2)] == 1)) and not (np.any(zscore_analysis_matrix[:,int(len(xbins_vector)/2)-1:-1] == 1)):
            zscore_results['unpaired_side']['excitatory'].append('{}_{}'.format(video_name,neuron))
        
        elif not (np.any(zscore_analysis_matrix[:,0:int(len(xbins_vector)/2)] == 1)) and (np.any(zscore_analysis_matrix[:,int(len(xbins_vector)/2)-1:-1] == 1)):
            zscore_results['paired_side']['excitatory'].append('{}_{}'.format(video_name,neuron))
            
        ### inhibitory    
        elif (np.any(zscore_analysis_matrix[:,0:int(len(xbins_vector)/2)] == -1)) and (np.any(zscore_analysis_matrix[:,int(len(xbins_vector)/2)-1:-1] == -1)):
            zscore_results['both_sides']['inhibitory'].append('{}_{}'.format(video_name,neuron))
        
        elif (np.any(zscore_analysis_matrix[:,0:int(len(xbins_vector)/2)] == -1)) and not (np.any(zscore_analysis_matrix[:,int(len(xbins_vector)/2)-1:-1] == -1)):
            zscore_results['unpaired_side']['inhibitory'].append('{}_{}'.format(video_name,neuron))
            
        elif not (np.any(zscore_analysis_matrix[:,0:int(len(xbins_vector)/2)] == -1)) and (np.any(zscore_analysis_matrix[:,int(len(xbins_vector)/2)-1:-1] == -1)):
            zscore_results['paired_side']['inhibitory'].append('{}_{}'.format(video_name,neuron))            
        
        ##non_response
        else:
            zscore_results['non_response'].append('{}_{}'.format(video_name,neuron)) 
        
           
        ### RISK TAKERS GROUP
        if video_index in risk_takers:
            #Sum all the excitatory zscore from responsive neurons
            if 'RT_zscore_excitatory_responsive_matrix' not in dir():
                RT_zscore_excitatory_responsive_matrix= np.zeros(zscore_analysis_matrix.shape)
            #Sum all the inhibitory zscore from responsive neurons
            if 'RT_zscore_inhibitory_responsive_matrix' not in dir():
                RT_zscore_inhibitory_responsive_matrix= np.zeros(zscore_analysis_matrix.shape)
                
            for i in np.arange(zscore_analysis_matrix.shape[0]):
                for j in np.arange(zscore_analysis_matrix.shape[1]):
                    if zscore_analysis_matrix[i,j] == 1.:
                        RT_zscore_excitatory_responsive_matrix[i,j]+=1.0
                    if zscore_analysis_matrix[i,j] == -1.:
                        RT_zscore_inhibitory_responsive_matrix[i,j]+=1.0    
            
            #Save the zscore matrixes
            f = open('RT_zscore_excitatory_responsive_matrix_{}bins_{}frames.pkl'.format(str(num_bins),str(minimum_n_frames_filter)), 'wb')
            pickle.dump(RT_zscore_excitatory_responsive_matrix,f)
            f.close()
            
            f = open('RT_zscore_inhibitory_responsive_matrix_{}bins_{}frames.pkl'.format(str(num_bins),str(minimum_n_frames_filter)), 'wb')
            pickle.dump(RT_zscore_inhibitory_responsive_matrix,f)
            f.close()
            
        
        ### RISK AVOIDERS GROUP
        if video_index in risk_avoiders:
            #Sum all the excitatory zscore from responsive neurons
            if 'RA_zscore_excitatory_responsive_matrix' not in dir():
                RA_zscore_excitatory_responsive_matrix= np.zeros(zscore_analysis_matrix.shape)
            #Sum all the inhibitory zscore from responsive neurons
            if 'RA_zscore_inhibitory_responsive_matrix' not in dir():
                RA_zscore_inhibitory_responsive_matrix= np.zeros(zscore_analysis_matrix.shape)
                
            for i in np.arange(zscore_analysis_matrix.shape[0]): #row
                for j in np.arange(zscore_analysis_matrix.shape[1]): #column
                    if zscore_analysis_matrix[i,j] == 1.:
                        RA_zscore_excitatory_responsive_matrix[i,j]+=1.0
                    if zscore_analysis_matrix[i,j] == -1.:
                        RA_zscore_inhibitory_responsive_matrix[i,j]+=1.0    
            
            #Save the zscore matrixes
            f = open('RA_zscore_excitatory_responsive_matrix_{}bins_{}frames.pkl'.format(str(num_bins),str(minimum_n_frames_filter)), 'wb')
            pickle.dump(RA_zscore_excitatory_responsive_matrix,f)
            f.close()
            
            f = open('RA_zscore_inhibitory_responsive_matrix_{}bins_{}frames.pkl'.format(str(num_bins),str(minimum_n_frames_filter)), 'wb')
            pickle.dump(RA_zscore_inhibitory_responsive_matrix,f)
            f.close()
                
        
        ### SALINE GROUP
        if video_index in saline:
            #Sum all the excitatory zscore from responsive neurons
            if 'SAL_zscore_excitatory_responsive_matrix' not in dir():
                SAL_zscore_excitatory_responsive_matrix= np.zeros(zscore_analysis_matrix.shape)
            #Sum all the inhibitory zscore from responsive neurons
            if 'SAL_zscore_inhibitory_responsive_matrix' not in dir():
                SAL_zscore_inhibitory_responsive_matrix= np.zeros(zscore_analysis_matrix.shape)
                
            for i in np.arange(zscore_analysis_matrix.shape[0]):
                for j in np.arange(zscore_analysis_matrix.shape[1]):
                    if zscore_analysis_matrix[i,j] == 1.:
                        SAL_zscore_excitatory_responsive_matrix[i,j]+=1.0
                    if zscore_analysis_matrix[i,j] == -1.:
                        SAL_zscore_inhibitory_responsive_matrix[i,j]+=1.0    
            
            #Save the zscore matrixes
            f = open('SAL_zscore_excitatory_responsive_matrix_{}bins_{}frames.pkl'.format(str(num_bins),str(minimum_n_frames_filter)), 'wb')
            pickle.dump(SAL_zscore_excitatory_responsive_matrix,f)
            f.close()
            
            f = open('SAL_zscore_inhibitory_responsive_matrix_{}bins_{}frames.pkl'.format(str(num_bins),str(minimum_n_frames_filter)), 'wb')
            pickle.dump(SAL_zscore_inhibitory_responsive_matrix,f)
            f.close()

              
print('FINAL RESULTS: \n {}'.format(zscore_results)) 

print(' \n RESUME')
total_paired = len(zscore_results['paired_side']['excitatory'])+len(zscore_results['paired_side']['inhibitory'])
if total_paired == 0:
    print('There are no paired neurons')
else:
    print('Paired side excitatory = ',len(zscore_results['paired_side']['excitatory'])/total_paired*100)
    print('Paired side inhibitory = ',len(zscore_results['paired_side']['inhibitory'])/total_paired*100)

total_unpaired = len(zscore_results['unpaired_side']['excitatory'])+len(zscore_results['unpaired_side']['inhibitory'])
if total_unpaired == 0:
    print('There are no unpaired neurons')
else:
    print('Unpaired side excitatory = ',len(zscore_results['unpaired_side']['excitatory'])/total_unpaired*100)
    print('Unpaired side excitatory = ',len(zscore_results['unpaired_side']['inhibitory'])/total_unpaired*100)
    
total_both_sides = len(zscore_results['both_sides']['excitatory'])+len(zscore_results['both_sides']['inhibitory'])
if total_both_sides == 0:
    print('There are no neurons in both sides')
else:
    print('Both sides excitatory= ',len(zscore_results['both_sides']['excitatory'])/total_both_sides*100)
    print('Both sides inhibitory= ',len(zscore_results['both_sides']['inhibitory'])/total_both_sides*100)
print('non response= ',len(zscore_results['non_response'])/(total_paired+total_unpaired+total_both_sides+len(zscore_results['non_response']))*100)
print('total neurons = ',(total_paired+total_unpaired+total_both_sides+len(zscore_results['non_response'])))                         
print('#### End')   

log_file.close()
print('#### End')   
