import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import sys
import os
import subprocess
import re


# Some declarations
NUM_RANKS = 4
NUM_ITERATIONS = 20
NUM_OMPTHREADS = 23
SAMOA_SECTIONS = 16


#########################################################
###### Self-defined Classes
#########################################################
"""Task definition

Define a struct of task to save profile-info per task.
Todo: could be a lot of tasks in each log-file
"""
class Task:
    def __init__(self, idx, task_id, arg1, arg2, arg3, arg4, exe_time):
        self.idx = idx
        self.task_id = task_id
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        self.arg4 = arg4
        self.exe_time = exe_time


#########################################################
###### Util-functions
#########################################################
"""Read log-file and parse total_load per iter

As the layout of data for storing profile-data per rank. We define
results_per_rank    = [R0] [R1] ... [Rn]
    [R0]            = [it0_runtime] [it1_runtime] ... [itN_runtime]
    [R1]            = [it0_runtime] [it1_runtime] ... [itN_runtime]
"""
def parse_stats_iter_runtime(filename, num_ranks):

    # open the logfile
    file = open(filename, 'r')

    # for storing values
    data_per_rank = []
    for i in range(num_ranks):
        tmp_tuple = (i, [])
        data_per_rank.append(tmp_tuple)

    # getting total_load/iter
    for line in file:
        # just get the line with the content
        if "_time_task_execution_local_sum" in line:
            data_per_line = line.split("\t")
            get_rank = (data_per_line[0]).split(" ")[1]
            rank = int(re.findall(r'\d+', get_rank)[0])
            total_runtime = float(data_per_line[3])
            avg_iter_runt = total_runtime
            # append data to the list/rank
            data_per_rank[rank][1].append(avg_iter_runt)

    # return the result
    return data_per_rank


"""Plot runtime-by-iters

Input is a list of runtime-data per rank. Use them to plot
a stack-bar chart for easily to compare the load imbalance.
"""
def plot_runtime_by_iters(stats_data, output_folder):
    # for the chart information
    plt.xlabel("Iterations")
    plt.ylabel("Total_Load (in seconds)")
    plt.title("Total_Load per Iteration")

    # for x_index
    first_rank_data = stats_data[0]
    num_iters = len(first_rank_data[1])
    x_indices = np.arange(num_iters)

    # traverse the profile-data
    num_ranks = len(stats_data)
    bottom_layer = np.zeros(num_iters)
    for i in range(num_ranks):
        data_per_rank = stats_data[i]
        dat_label    = "R_" + str(data_per_rank[0])

        if i != 0:
            prev_data_per_rank = stats_data[i-1]
            prev_np_data_arr = np.array((prev_data_per_rank[1])[:num_iters])
            bottom_layer += prev_np_data_arr

        # convert data to numpy_arr
        if len(data_per_rank[1]) != 0:
            np_data_arr = np.array((data_per_rank[1])[:num_iters])
        else:
            np_data_arr = np.zeros(num_iters)
        
        # plot the line/bar
        # plt.plot(x_indices, np_data_arr, label=dat_label)
        if i == 0:
            plt.bar(x_indices, np_data_arr, label=dat_label)
        else:
            plt.bar(x_indices, np_data_arr, bottom=bottom_layer, label=dat_label)

    
    # plt.yscale('log')
    plt.grid(True)
    # plt.legend(loc='best', shadow=True, ncol=5, prop={'size': 5})
    # plt.show()

    # save the figure
    fig_filename = "runtime_per_iter_" + str(num_ranks) + "_ranks_from_chamstats_logs" + ".pdf"
    plt.savefig(os.path.join(output_folder, fig_filename), bbox_inches='tight')



""" Dataset generator 1
    Chain-on-Chain runtimes per iter dataset
    It means the runtimes of first iters could be the input
    for learning and predicting the next-iter runtime. """
def cc_runtime_dataset_generator(raw_data, rank, num_features):
    # get data by rank
    rank_data = raw_data[rank]
    runtime_list = rank_data[1]

    # identify num_features for the dataset
    num_iters = len(runtime_list)   # as the total num of points we have (named N)
    num_feat_points = num_features

    labels = []
    for i in range(num_feat_points):
        labels.append("a"+str(i))

    length = num_iters
    x_ds = []
    y_ds = []
    for i in range(num_feat_points, length):
        x_ds.append(runtime_list[(i-num_feat_points):i])
        y_ds.append(runtime_list[i])
    tmp_ds = pd.DataFrame(np.array(x_ds), columns=labels)
    final_ds = tmp_ds
    final_ds["target"] = y_ds

    # write to csv file
    final_ds.to_csv('./cc_runtime_dataset_r' + str(rank) + '.csv', index=False, header=False)

    return final_ds


#########################################################
###### Main Function
#########################################################
"""The main function

There are 3 mains phases in the body of this source,
that could be reading-logs, visualizing, and prediction.
"""
if __name__ == "__main__":

    # get folder of log-files
    cham_stats_file = sys.argv[1]

    # num ranks
    num_ranks = int(sys.argv[2])

    # read and parse values from the input
    stats_data = parse_stats_iter_runtime(cham_stats_file, num_ranks)

    # display stats_data
    for i in range(num_ranks):
        data_per_rank = stats_data[i]
        rank = data_per_rank[0]
        runtime_list = data_per_rank[1]
        statement = str(rank) + ": "
        for j in range(len(runtime_list)):
            formated_val = float("{:.4f}".format(runtime_list[j]))
            statement += str(formated_val) + "\t"
        print(statement)


    # generate dataset by chain-on-chain runtimes per iter
    if (len(sys.argv) <= 3):
        rank = 0
    else:
        rank = int(sys.argv[3])
    num_features = 6
    cc_runtime_dataset_generator(stats_data, rank, num_features)

    
    # plot a single-rank runtime-list
    # rank = 25
    # rank_data = stats_data[rank]
    # runtime_data = rank_data[1]
    # x_indices = np.arange(len(runtime_data))
    # plt.xlabel("Iterations")
    # plt.ylabel("Total_Load (in seconds)")
    # plt.plot(x_indices, runtime_data)
    # plt.grid(True)
    # plt.show()
    
    # plot the data
    # output_folder = "./"
    # plot_runtime_by_iters(stats_data, output_folder)