from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
def parse_predicted_results(filename):

    # open the logfile
    file = open(filename, 'r')

    # for storing runtime/iter
    ret_data = []
    line_counter = 0
    for line in file:
        line_counter += 1
        # just get runtime per iteration
        if line_counter >= 20801:
            tmp = line.split("\t")
            real_load = float(tmp[0])
            pred_load = float(tmp[1])
            real_pred_tuple = [real_load, pred_load]
            ret_data.append(real_pred_tuple)

    # return the result
    return ret_data


"""Gather results of all cham-tool-log files in a folder.

Number of log-files depends on # ranks to run the application,
hence we process file-by-file, and storing results could be in a list.
"""
def gather_Results(folder, num_nodes):

    # create a list for storing
    num_ranks = num_nodes * 2
    results_per_rank = []
    for i in range(num_ranks):
        results_per_rank.append([])

    # loop for all files in the folder
    for f in os.listdir(folder):

        # get rank id
        filename = f.split("_")
        rank_id = int((filename[-1].split("."))[0])

        # read log-file
        filepath = os.path.join(folder,f)
        data = parse_predicted_results(filepath)
        results_per_rank[rank_id] = data

    return results_per_rank


"""Calculate MSE errors between real and pred load values.

Input is the array of real_ and pred_load per rank, MSE values are
calculated indepently for each iteration, but skip the previous iters
because of training phase.
"""
def calculate_MSE(chamtool_data, num_ranks):

    ret_arr = []
    for i in range(num_ranks):
        ret_arr.append([])

    for i in range(num_ranks):

        # get the data of rank i
        load_arr = chamtool_data[i]
        num_iters = len(load_arr)

        # array for storing mse_errors
        mse_loss = []
        for j in range(20, num_iters):
            real_load = (load_arr[j])[0]
            pred_load = (load_arr[j])[1]
            mse = np.square(np.subtract(real_load, pred_load))
            mse_loss.append(mse)

        # add to the parent arr
        ret_arr[i] = mse_loss    
    
    return ret_arr


"""Plot runtime-by-iters

Input is a list of runtime-data per rank. Use them to plot
a stacked-bar chart for easily to compare the load imbalance.
"""
def plot_pred_data_boxplot(avg_loss_arr, out_folder):

    # boxplot
    fig, ax = plt.subplots()
    ax.boxplot(avg_loss_arr)
    ax.set_xlabel("Scale (# nodes)")
    ax.set_ylabel("Loss (MSE)")
    ax.set_xticklabels(['2', '4', '8', '16'])

    plt.yscale('log')
    plt.grid(True)
    # plt.show()

    # save the figure
    fig_filename = "boxplot_avg_mse_loss" + ".pdf"
    plt.savefig(os.path.join(out_folder, fig_filename), bbox_inches='tight')



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
    dirs = os.listdir(folder)
    for d in dirs:
        # count num of the case traversed
        num_cases = num_cases + 1

        # get path to each directory
        path_to_dir = folder + "/" + d
        print(path_to_dir)

        # extract the folder name
        tokens = path_to_dir.split("/")
        sub_token   = tokens[len(tokens)-1].split("_")
        num_nodes = int(sub_token[0])

        # gather results in the folder of logs
        real_pred_load_data = gather_Results(path_to_dir, num_nodes)
    
        # calculate loss_errors per rank
        num_ranks = num_nodes * 2
        mse_loss_arr = calculate_MSE(real_pred_load_data, num_ranks)
        print("Case: {} nodes, len(mse_loss_arr)={}".format(num_nodes, len(mse_loss_arr)))

        # get avg_mse_arr for each case
        num_iters = len(mse_loss_arr[0])
        avg_mse_loss_arr = []
        for i in range(num_iters):
            tmp_arr = []
            for r in range(num_ranks):
                loss = mse_loss_arr[r][i]
                tmp_arr.append(loss)

            # get avg loss per iter of all ranks
            # print("\t Iter {}: {}".format(i, tmp_arr))
            avg_loss = np.average(tmp_arr) / len(tmp_arr)
            
            # store values to the par_array
            avg_mse_loss_arr.append(avg_loss)
        
        # append to the total array of loss
        avg_loss_arr.append(avg_mse_loss_arr)            

    # plot box-plot
    plot_pred_data_boxplot(avg_loss_arr, folder)
    
