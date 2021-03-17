import torch
from torch.autograd import Variable
import torch.optim as optim
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
torch.manual_seed(1)    # reproducible


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


"""Prediction models

Define the net-structure for prediction models.
The inputs should be normalized or not, or this depends on
each kind of task-parallel applications. 
"""
class PredictionModel(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(LinearRegression, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)  # hidden layer
        self.hidden1 = torch.nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.zeros_(self.hidden.bias)
        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        torch.nn.init.zeros_(self.hidden1.bias)

        torch.nn.init.xavier_uniform_(self.predict.weight)
        torch.nn.init.zeros_(self.predict.bias)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))  # activation function for hidden layer
        x = torch.relu(self.hidden1(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


"""Read log-file and parse total_load per iter

As the layout of data for storing profile-data per rank. We define
results_per_rank          = [R0] [R1] ... [Rn]
    data_[R0]             = [iter0] [iter1] ... [iter_n]
        data_[iter0]      = [tot_runtime0]
        data_[iter1]      = [tot_runtime1]
        data_[iter2]      = [...         ]
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
            avg_iter_runt = total_runtime  / 3
            # append data to the list/rank
            data_per_rank[rank][1].append(avg_iter_runt)

    # return the result
    return data_per_rank


"""Plot runtime-by-iters

Input is a list of runtime-data per rank. Use them to plot
a line chart for easily to compare the load imbalance.
"""
def plot_runtime_by_iters(stats_data, output_folder):
    # for the chart information
    plt.xlabel("Iterations")
    plt.ylabel("Total_Load (in seconds)")
    plt.title("Total_Load per Iteration")

    # for x_index
    first_rank_data = stats_data[0]
    num_iters = 100 #len(first_rank_data[1])
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


"""The main function

There are 3 mains phases in the boday of this source,
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
        # print(statement)

    # plot the data
    output_folder = "./"
    plot_runtime_by_iters(stats_data, output_folder)