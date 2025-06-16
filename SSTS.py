from matplotlib import pyplot as plt
from sklearn import preprocessing

import importlib
import pandas as pd
import numpy as np
import math as m
import random
import time 
import os

start = time.time()

import utils.gen_proc as gp
importlib.reload(gp)

import utils.piecewise_ss as pss  
importlib.reload(pss)

import utils.ft_proc as ft_proc
importlib.reload(ft_proc)

import utils.fft as fft
importlib.reload(fft)

init_station = 6
XX = str(init_station).zfill(2)
YY = str(init_station + 1).zfill(2)
data_dir        = 'val_raw_data/'   #raw data csv的folder
preproc_dir     = 'val/' #PSSS後儲存的folder
fault_types     = ['Type_A', 'Type_B', 'Type_C', 'Type_Normal']

# N
no_slices = 10
# C
channels = 3

segment_size    = 32
#no_slices       = 10

input_arr_size  = 64
fft_freq_range  = [5,0] 

ts_format       = 0
window_size     = 0
features        = ['mean', 'abs_mean', 'variance', 'rms']

print("[N, C] = [{}, {}]".format(no_slices, channels))
adj_channel = int(channels / 2)

isExist = os.path.exists(preproc_dir)
if not isExist:
    os.makedirs(preproc_dir)

#   Count number of subdirectories in all the classes.
#   Run once.

dirs = 0
for ftype in range(len(fault_types)):
    
    ftype_name = fault_types[ftype]
    dirs += len([x[1] for x in os.walk(data_dir + ftype_name)]) - 1

print('Total data: {}'.format(dirs))

#   堆疊通道

stack_l = 'X = np.stack(('
stack_vars = ''
stack_r = '), axis = 3)'

if channels > 1:
    for i in range(adj_channel):
        stack_vars += 'x_prev[{}], '.format(i)
    stack_vars += 'x, '
    for i in range(adj_channel):
        stack_vars += 'x_next[{}], '.format(i)

if window_size:
    for ft in features:
        stack_vars += 'x_{}, '.format(ft)

exec_stack = stack_l + stack_vars[:-2] + stack_r

print(exec_stack)
print('\n')

#   Iterating over all data dirs & calling SS funcs.

curr_dID = 1
adj_channel = int(channels / 2)
zero_fill = len(str(dirs))

#dir_id_arr   = []
slice_id_arr = []    

extract_start = time.time()
for ftype in range(4):
    
    ftype_name     = fault_types[ftype]
    class_dir_list = sorted([x[0] + '/' for x in os.walk(data_dir + ftype_name)][1:])
    print('Now sampling fault {}:'.format(ftype_name))

    for subdir in class_dir_list:
        if 'plots' in subdir: 
            continue
        
        if (curr_dID % 5 == 0 or curr_dID == dirs):
            print('{} of {}'.format(curr_dID, dirs))

        data_buffer = gp.join(subdir)
        data_buffer = gp.remove_stopseq(data_buffer)

        if len(fft_freq_range):
            data_buffer = fft.fft_denoiser(data_buffer, fft_freq_range)

        x, prev, next, slices = \
        pss.piecewise_ss(data_buffer, input_arr_size, no_slices, segment_size, channels)

        #dir_id_arr.extend([curr_dID] * len(x))
        slice_id_arr.extend(slices)

        curr_dID += 1

        #   Numpy-ize samples.
        x = np.array(x)
        prev = np.array(prev)
        next = np.array(next) 

        #   Extract features.
        if window_size:
            ft_init = ft_proc.ft_ext(x, window_size)

            for ft in features:
                exec('ft_{} = ft_init.{}()'.format(ft, ft))

        x_prev = []
        x_next = []
        if not ts_format:
            x = ft_proc.to_2D(x, input_arr_size)
            if channels > 1:
                for i in range(adj_channel):
                    x_prev.append(ft_proc.to_2D(prev[i], input_arr_size))
                    x_next.append(ft_proc.to_2D(next[i], input_arr_size))

            if window_size:
                for ft in features:
                    exec('x_{} = ft_proc.to_2D(ft_{}, input_arr_size)'.format(ft, ft))


        exec(exec_stack)
        print('{} {} {} {}'.format(curr_dID - 1, ftype+1, subdir[-11:-1], X.shape))

        #   Save data.

        np.save(preproc_dir + '{}_{}_{}'.format(str(curr_dID - 1).zfill(zero_fill), ftype + 1, subdir[-11:-1]), np.array(X))

        print(time.time() - start)
        print('\n')


end = time.time()
print('Time elapsed: {} seconds'.format(end - start))
print('Average time per path: {} seconds'.format((end - extract_start) / dirs))
