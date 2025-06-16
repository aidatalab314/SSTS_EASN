import numpy as np
import pandas as pd
import os

def join(target_dir):

    csv_list = sorted(list(os.listdir(target_dir))) 

    data_buffer = np.array([])

    #   Iterate over every single csv file
    for raw_data_file in csv_list:
        if raw_data_file[:3] == '.DS' or '._' in raw_data_file:
            continue 
        
        else:
            try:  
                df_temp = pd.read_csv(target_dir + raw_data_file, names = ['timestamp', 'value'],\
                                        encoding = "utf-8", sep = ',', header=None, engine='python')[4:]
            except:
                print(target_dir + raw_data_file)
                quit()

            data_buffer = np.append(data_buffer, df_temp['value'].astype(float))

    return data_buffer

def remove_stopseq(input_data, filter_len = 50, min_peak = 0.01):

        #   Filter window length

        #   Min value for non-noise windows         

        filter = []

        for i in range(input_data.shape[0] - filter_len + 1):

            #   Get max val in filter window
            slide_peak = np.max(np.abs(input_data[i:i + filter_len]))

            if slide_peak > min_peak:
                #   Section is signal, not noise
                filter.append(i)

        return input_data[filter]