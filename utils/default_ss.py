import numpy as np
import pandas as pd
import os
import math as m
from fft import *

#   Trim front and back motionless sections.

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


def default_ss(target_dir, matrix_size, min_freq, segment_size, sample_adj):

    csv_list = list(os.listdir(target_dir))

    data_buffer = np.array([])
                
    check_step = 100
    min_avg = 0.2

    #   Iterate over all csv files in folder.
    for raw_data_file in csv_list:

        if raw_data_file[:3] == '.DS':
            continue 

        #   Create temporary DataFrame from csv and skip non-data rows.
        df_temp = pd.read_csv(target_dir + raw_data_file, skiprows = 3, names = ['timestamp', 'value'])

        #   In older csv files, the first data row has a string instead of data; drop it from the DataFrame.
        if str(type(df_temp['value'].iloc[0])) == "<class 'str'>":
            df_temp.drop(index = df_temp.index[0], axis = 0, inplace = True)
        
        data_buffer.extend([float(i) for i in df_temp['value']]) 

    if check_step:
        data_buffer = hybrid_crop(data_buffer, check_step, min_avg)

    if min_freq:
        data_buffer = fft_denoiser(data_buffer, min_freq)

    #                    #
    #   BEGIN SAMPLING   #
    #                    #

    #   Determine how many samples can be generated via integrated csv data.
    sample_len = matrix_size * matrix_size
    samples_for_folder = m.floor(len(data_buffer) / sample_len)
    skip_len = int(len(data_buffer) * segment_size / sample_len)

    #   Sample data with Systematic Sampling and Sliding Window.
    #   The iteration of variable i 'slides' the initial position,
    #   until no more samples can be generated.

    for i in range(samples_for_folder):
        temp = []
        if sample_adj:
            temp_p1 = []
            temp_n1 = []

        
        j = 0

        #   Systematic Sampling:
        #
        #   Obtain value at initial position i for the sample.
        #   Increment j positions, obtain another value,
        #   and repeat until the current sample reaches the required sample length.

        while(len(temp) < sample_len): 

            #   Strengthened SS
            if segment_size:
                start = i * segment_size + j * skip_len
                temp.extend(np.float16(data_buffer[start:start + segment_size]))

                #   Prev next
                if sample_adj:
                    prev_start = max(0, start - segment_size)
                    next_start = min(len(data_buffer) - segment_size, start + segment_size)

                    temp_p1.extend(np.float16(data_buffer[prev_start:prev_start + segment_size]))
                    temp_n1.extend(np.float16(data_buffer[next_start:next_start + segment_size]))

            #   Default SS
            else:
                temp.append(np.float16(data_buffer[i + j * samples_for_folder]))
                
                if sample_adj:
                    temp_p1.append(np.float16(data_buffer[max(0, i + j * samples_for_folder - 1)]))
                    temp_n1.append(np.float16(data_buffer[min(i + j * samples_for_folder + 1, len(data_buffer) - 1)]))



            j += 1

    if sample_adj:
        return temp, temp_p1, temp_n1
    else:
        return temp