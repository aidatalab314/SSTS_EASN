import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt

import importlib

from . import fft
importlib.reload(fft)

from . import gen_proc as jd
importlib.reload(jd)

def hybrid_crop(X, k, min_avg):

    #front
    start = 0
    end = len(X)

    mid = int((end - start) / 2) + start
    prev = np.sum(np.abs(np.array(X[max(0, mid - k) : mid]))) / k
    next = np.sum(np.abs(np.array(X[mid : min(len(X), mid + k)]))) / k

    while not(prev < min_avg and next < min_avg):
        end = mid

        mid = int((end - start) / 2) + start
        prev = np.sum(np.abs(np.array(X[max(0, mid - k) : mid]))) / k
        next = np.sum(np.abs(np.array(X[mid : min(len(X), mid + k)]))) / k

    X = X[mid:]
    while(sum([abs(i) for i in X[:k]]) / k < min_avg):
        X = X[k:]


    #back
    start = 0
    end = len(X)

    mid = int((end - start) / 2) + start
    prev = np.sum(np.abs(np.array(X[max(0, mid - k) : mid]))) / k
    next = np.sum(np.abs(np.array(X[mid : min(len(X), mid + k)]))) / k

    while not(prev < min_avg and next < min_avg):
        start = mid

        mid = int((end - start) / 2) + start
        prev = np.sum(np.abs(np.array(X[max(0, mid - k) : mid]))) / k
        next = np.sum(np.abs(np.array(X[mid : min(len(X), mid + k)]))) / k
    
    X = X[:mid]
    while(sum([abs(i) for i in X[-1*k:]]) / k < min_avg):
        del X[-1*k:]
    return X

def piecewise_ss(input_data, matrix_size, slices, segment_size, channels):

    curr = []
    prev = []
    next = []

    adj_channel = int(channels / 2)
    for i in range(adj_channel):
        prev.append([])
        next.append([])


    data_len  = len(input_data)
    slice_seq = []
    sample_len = matrix_size * matrix_size
    
    #   Samples per slice
    samples_per_slice = int(data_len / (slices * sample_len))

    for i in range(slices):

        process_array = input_data[i * int(data_len / slices) : (i + 1) * int(data_len / slices)]
        skip_len = int(len(process_array) * segment_size / (sample_len))

        for j in range(samples_per_slice):

            if segment_size:

                #   Piecewise + Strengthened SS
                k = 0
                t_curr = []
                if channels > 1:
                    t_prev = []
                    t_next = []
                    for i in range(adj_channel):
                        t_prev.append([])
                        t_next.append([])

                while(len(t_curr) < (sample_len)): 
                    start = j * segment_size + k * skip_len
                    t_curr.extend(np.float16(process_array[start:start + segment_size]))

                    if channels > 1:
                        for i in range(adj_channel):
                            prev_start = max(0, start - (i + 1) * segment_size)
                            next_start = min(len(process_array) - segment_size, start + (i + 1) * segment_size)

                            t_prev[i].extend(np.float16(process_array[prev_start:prev_start + segment_size]))
                            t_next[i].extend(np.float16(process_array[next_start:next_start + segment_size]))
                        
                    k += 1
            else:
                if channels > 3:
                    raise Exception("Invalid channel count")
                else:
                    #   Piecewise + Standard SS
                
                    t_curr = process_array[j::samples_per_slice]

                    if channels > 1:
                        t_prev = process_array[min(j - 1 + samples_per_slice, len(process_array))::samples_per_slice]
                        t_next = process_array[max(j + 1 - samples_per_slice, 0)::samples_per_slice]

            curr.append(t_curr[:sample_len])
            if channels > 1:
                for i in range(adj_channel):
                    prev[i].append(t_prev[i][:sample_len])
                    next[i].append(t_next[i][:sample_len])

            slice_seq.append(i + 1)    

    return curr, prev, next, slice_seq