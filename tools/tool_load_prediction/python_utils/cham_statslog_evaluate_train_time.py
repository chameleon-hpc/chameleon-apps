from numpy.lib.npyio import load
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import sys
import os
import csv



"""Read cham-tool-logs with predicted-load values

Each rank holds an array of real-load and predicted load. For example,
    real_load_data=[R0_dat, R1_dat, ...]
        R0_dat = [] ...
    pred_load_data=[R0_dat, R1_dat, ...]
        R0_dat = [] ...
"""
def parse_statslog_results(filename):

    # open the logfile
    file = open(filename, 'r')
    num_ranks = 0
    for line in file:
        if "_num_overall_ranks" in line:
            tokens = line.split("\t")
            num_ranks = int(tokens[2])
            break

    # for storing runtime/iter
    ret_data = []
    tw_idle_arr = []
    train_time_arr = []
    infer_time_arr = []
    real_time_arr = []
    pred_time_arr = []
    mse_loss_arr = []
    for i in range(num_ranks):
        tw_idle_arr.append([])
        ret_data.append([])
        train_time_arr.append(0.0)
        infer_time_arr.append([])
        real_time_arr.append([])
        pred_time_arr.append([])
        mse_loss_arr.append([])

    # retrieve the info
    for line in file:
        # just get runtime per iteration
        if "_time_training_model_sum" in line:
            tokens = line.split("\t")
            rank_id = int(re.findall(r'\d+', ((tokens[0].split(" "))[1]))[0])
            train_time = float(tokens[3])
            if train_time != 0.0:
                train_time_arr[rank_id] = train_time

        # get real-load time
        if "_time_task_execution_overall_sum" in line:
            tokens = line.split("\t")
            rank_id = int(re.findall(r'\d+', ((tokens[0].split(" "))[1]))[0])
            real_load = float(tokens[3])
            real_time_arr[rank_id].append(real_load)

        # get pred_load time
        if "_time_task_execution_pred_sum" in line:
            tokens = line.split("\t")
            rank_id = int(re.findall(r'\d+', ((tokens[0].split(" "))[1]))[0])
            pred_load = float(tokens[3])
            pred_time_arr[rank_id].append(pred_load)

        # get teh inferencing time
        if "_time_inferenc_model_sum" in line:
            tokens = line.split("\t")
            rank_id = int(re.findall(r'\d+', ((tokens[0].split(" "))[1]))[0])
            infer_time = float(tokens[3])
            if infer_time != 0.0:
                infer_time_arr[rank_id].append(infer_time)
        
        # get the avg_taskwait time
        if "_time_taskwait_idling_sum" in line:
            tokens = line.split("\t")
            rank_id = int(re.findall(r'\d+', tokens[0])[0])
            idle_sum = float(tokens[3])
            count = float(tokens[5])
            avg_idle = idle_sum / count
            tw_idle_arr[rank_id].append(avg_idle)
    
    # calculate avg_mse loss
    num_iters = len(real_time_arr[0])
    for r in range(num_ranks):
        # print("Rank {}:".format(r))
        # print("\t {}".format(real_time_arr[r]))
        # print("\t {}".format(pred_time_arr[r]))
        # print("-------------------------------------------")
        for i in range(num_iters):
            real_val = real_time_arr[r][i]
            pred_val = pred_time_arr[r][i]
            if pred_val != 0.0:
                mse = (real_val - pred_val)**2
                mse_loss_arr[r].append(mse)
    
    # merge the ret_data array
    for i in range(num_ranks):
        avg_tw_idle_value = np.sum(tw_idle_arr[i]) / len(tw_idle_arr[i])
        ratio_train_tw_time = 100 * (train_time_arr[i] / avg_tw_idle_value)
        ret_data[i].append(i)                               # 0. rank_id
        ret_data[i].append(avg_tw_idle_value)               # 1. avg_tw
        ret_data[i].append(train_time_arr[i])               # 2. train_time
        ret_data[i].append(np.average(infer_time_arr[i]))   # 3. infer_time
        ret_data[i].append(ratio_train_tw_time)             # 4. ratio_tr_tw
        ret_data[i].append(np.average(mse_loss_arr[i]))     # 5. avg_mse_loss

    # return the result
    return ret_data


"""The main function

There are 3 mains phases in the boday of this source,
that could be reading-logs, visualizing, and prediction.
"""
if __name__ == "__main__":

    # array for storing avg_loss of all cases
    num_cases = 0
    avg_loss_arr = []

    # get folder-path of log-files
    folder = sys.argv[1]
    statslog_files = os.listdir(folder)
    
    # read err-file by err-file
    for f in statslog_files:

        # get prefix of log files
        logfile_tokens = f.split("_")
        f_prefix = logfile_tokens[0]

        # read out_ files to get dmin-dmax values:
        logfile_index = []
        if f_prefix == "out":
            job_id = int(logfile_tokens[1])
            out_file = open(os.path.join(folder, f), 'r')
            for line in out_file:
                if "mpirun" in line:
                    out_run_tokens = line.split(" ")
                    dmindmax = int(out_run_tokens[5])
                    logfile_index = [job_id, dmindmax]
                    break
            print("Outfile: {}".format(logfile_index))
            

        # read err_ files
        if f_prefix == "err":
            print("Reading file: {}...".format(f))
            file_path = os.path.join(folder, f)
            data = parse_statslog_results(file_path)
            np_data_arr = np.array(data)
            sorted_data = np_data_arr[np_data_arr[:,4].argsort()]
            # print(sorted_data)
            
            # get some avg_values
            avg_tw_idle_time = np.average(sorted_data[:,1])
            avg_train_time = np.average(sorted_data[:,2])
            avg_ratio = 100 * (avg_train_time / avg_tw_idle_time)
            avg_infer_time = np.average(sorted_data[:,3])
            avg_loss = np.average(sorted_data[:,5])
            print("AVG: {}(ms) {}(train-ms) {}(%) {}(infer-ms) {}(mse)".format(avg_tw_idle_time*1000, avg_train_time*1000, avg_ratio, avg_infer_time*1000, avg_loss))
            print("---------------------------------------------------")

    # plot box-plot
    
