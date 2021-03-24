import torch
from torch.autograd import Variable
import torch.optim as optim
from mlpack import linear_regression
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import sys
import os
import subprocess
import csv


# Some declarations
NUM_ITERATIONS = 1000
NUM_OMPTHREADS = 23
SAMOA_SECTIONS = 16
torch.manual_seed(1)    # reproducible


"""Task definition

Define a struct of task to save profile-info per task.
Todo: could be a lot of tasks in each log-file
"""
class Task:
    def __init__(self, task_id, arg1, arg2, arg3, arg4, exe_time):
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
        super(PredictionModel, self).__init__()
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


"""Linear Regression model

Define the net-structure for linear-regression models.
The inputs should be normalized or not, or this depends on
what kind of apps. 
"""
class LinearRegression(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(LinearRegression, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x)) # activation function for hidden layer
        x = torch.relu(self.hidden2(x))
        x = self.predict(x) # linear output
        return x


"""Read log-file and parse runtime per task per iter

As the layout of data for storing profile-data per rank. We define
results_per_rank          = [R0] [R1] ... [Rn]
    data_[R0]             = [iter0] [iter1] ... [iter_n]
        data_[iter0]      = [t0] [t1] ...   ... [t_n]
            data_[t0]     = [arg0, arg1, arg2, ..., runtime]
            data_[t1]     = [arg0, arg1, arg2, ..., runtime]
            ...
"""
def parse_taskprof_data(filename, num_nodes):

    # open the logfile
    file = open(filename, 'r')

    # mark the begin_ & end_ line for
    line_end = SAMOA_SECTIONS * NUM_OMPTHREADS * NUM_ITERATIONS
    
    # for storing tasks/iter
    iter_data = []

    # scan the file
    line_counter = 0
    iter_idx = 0
    for line in file:
        # count the num of lines
        line_counter += 1

        if line_counter <= line_end:
            # create a new list of tasks
            # when starting a new iter
            if line_counter % 368 == 1:
                task_list = []

            # read line by line
            data = line.split("\t")
            task_id  = int(data[0])
            arg1     = int(data[3])
            arg2     = int(data[4])
            arg3     = int(data[5])
            arg4     = int(data[6])
            runtime  = 0.0
            task     = Task(task_id, arg1, arg2, arg3, arg4, runtime)

            # add task to the current list
            task_list.append(task)
            
            # add a list of tasks to iter_data
            if line_counter % 368 == 0:
                iter_data.append(task_list)
                iter_idx += 1

    # return the result
    return iter_data


"""Read log-file and parse total_load per iter

As the layout of data for storing profile-data per rank. We define
results_per_rank          = [R0] [R1] ... [Rn]
    data_[R0]             = [iter0] [iter1] ... [iter_n]
        data_[iter0]      = [tot_runtime0]
        data_[iter1]      = [tot_runtime1]
        data_[iter2]      = [...         ]
"""
def parse_iterruntime_data(filename, num_nodes):

    # open the logfile
    file = open(filename, 'r')

    # mark the begin_ & end_ line for
    # getting total_load/iter
    # don't know why nmax=100, dmax-min=20, then num_iters = 200
    beg_line = SAMOA_SECTIONS * NUM_OMPTHREADS * NUM_ITERATIONS + 1
    # end_line = beg_line + NUM_ITERATIONS - 1

    # for storing runtime/iter
    iter_runtime = []
    line_counter = 0
    for line in file:
        line_counter += 1
        # just get runtime per iteration
        if line_counter == beg_line:
            data = line.split("\t")
            for i in range(len(data)-1):
                iter_runtime.append(float(data[i]))

    # return the result
    return iter_runtime


"""Gather results of runtime/iter/rank

Number of log-files depends on # ranks to run the application,
hence we process file-by-file, and storing results could be in a list.
"""
def gather_results_iterruntime(folder, num_nodes):
    results_per_rank = []

    for f in os.listdir(folder):

        # to get the log of specific rank
        filename = f.split("_")
        rank_id  = int(filename[2])

        # read log-file
        filepath = os.path.join(folder,f)
        iterrunt_data = parse_iterruntime_data(filepath, num_nodes) # store just runtime/load of each iters

        rank_iterrunt_data_tuple = (rank_id, iterrunt_data)

        results_per_rank.append(rank_iterrunt_data_tuple)

    return results_per_rank


"""Gather results of taskprof/iter/rank

Number of log-files depends on # ranks to run the application,
hence we process file-by-file, and storing results could be in a list.
"""
def gather_results_taskprof(folder, num_nodes):
    results_per_rank = []

    for f in os.listdir(folder):

        # to get the log of specific rank
        filename = f.split("_")
        rank_id  = int(filename[2])

        # read log-file
        filepath = os.path.join(folder,f)
        taskprof_data = parse_taskprof_data(filepath, num_nodes)    # store task-args data of all tasks per iter

        rank_taskprof_data_tuple = (rank_id, taskprof_data)

        results_per_rank.append(rank_taskprof_data_tuple)

    return results_per_rank


"""Filter task-data-section on the grid

Each task holds a number of sections which will be traversed. An iter has # of tasks,
a rank holds # of iters. Hence, there could be that a rank just processed a fixed region
of sections.
"""
def filter_task_section(profile_data):
    # get data per rank
    num_ranks = len(profile_data)
    data_per_rank = []
    for i in range(num_ranks):
        rank_data       = profile_data[i]
        num_iters       = len(rank_data[1])
        iter_data       = rank_data[1]
        data_per_iter = []
        for j in range(num_iters):
            num_tasks_per_iter  = len(iter_data)
            tasks_list          = iter_data[j]
            tasks_per_iter      = []
            for k in range(num_tasks_per_iter):
                task    = tasks_list[k]
                t_arg1  = task.arg1
                t_arg2  = task.arg2
                t_arg3  = task.arg3
                t_arg4  = task.arg4
                section = [t_arg1, t_arg2, t_arg3, t_arg4]
                tasks_per_iter.append(section)
            data_per_iter.append(tasks_per_iter)
        data_per_rank.append(data_per_iter)

    # return data
    return data_per_rank


"""Plot runtime-by-iters

Input is a list of runtime-data per rank. Use them to plot
a stacked-bar chart for easily to compare the load imbalance.
"""
def plot_runtime_by_iters(profile_data, output_folder):
    # for the chart information
    plt.xlabel("Iterations")
    plt.ylabel("Load (runtime in s)")
    plt.title("Load per Iteration")

    # for x_index
    num_iters = 100 #len(profile_data[0])
    x_indices = np.arange(num_iters)

    # traverse the profile-data
    bottom_layer = np.zeros(num_iters)
    for i in range(len(profile_data)):
        data_per_rank = profile_data[i]
        dat_label    = "R_" + str(data_per_rank[0])

        if i != 0:
            prev_data_per_rank = profile_data[i-1]
            prev_np_data_arr = np.array((prev_data_per_rank[1])[:num_iters])
            bottom_layer += prev_np_data_arr

        # convert data to numpy_arr
        if len(data_per_rank[1]) != 0:
            np_data_arr = np.array((data_per_rank[1])[:num_iters])
        else:
            np_data_arr = np.zeros(num_iters)
        
        # plot the line/bar
        # print("Rank i = {}: {}".format(i, np_data_arr))
        # print("Bottom-layer = {}".format(bottom_layer))
        if i == 0:
            plt.bar(x_indices, np_data_arr, label=dat_label)
        else:
            plt.bar(x_indices, np_data_arr, bottom=bottom_layer, label=dat_label)

    
    # plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='best', shadow=True, ncol=4, prop={'size': 5})
    # plt.show()

    # save the figure
    num_ranks = len(profile_data)
    fig_filename = "runtime_per_iter_" + str(num_ranks) + "_ranks_from_toolprof_logs" + ".pdf"
    plt.savefig(os.path.join(output_folder, fig_filename), bbox_inches='tight')


"""Plot total-runtime-by-rank

Input is a list of runtime-data per rank. Use them to plot
a bar chart for easily to compare the total load per rank.
"""
def plot_total_runtime_by_rank(profile_data, output_folder):
    # remove old-figure files
    # subprocess.call(["rm", output_folder + "total_load_per_rank*"])

    # for the chart information
    plt.xlabel("Ranks")
    plt.ylabel("Load (runtime in s)")
    plt.title("Total Load per Rank")
    bar_width = 0.4

    # for x_index
    xticks_labels = []
    num_ranks = len(profile_data)
    x_indices = np.arange(num_ranks)
    for i in range(num_ranks):
        xticks_labels.append(str(i))

    # for the list of total_load
    total_load_per_rank = []

    # traverse the profile-data
    for i in range(len(profile_data)):
        data_per_rank = profile_data[i]

        # get total_load
        if len(data_per_rank[1]) != 0:
            total_load = sum(data_per_rank[1])
        else:
            total_load = 0.0

        # append the load
        total_load_per_rank.append(total_load)
        
    # convert to numpy_arr
    np_data_arr = np.array(total_load_per_rank)

    # plot the bar-chart
    plt.bar(x_indices, np_data_arr, bar_width, alpha=1)
    # plt.xticks(x_indices, xticks_labels)
    
    plt.grid(True)
    # plt.legend()
    # plt.show()

    # save the figure
    fig_filename = "total_load_per_rank_" + str(num_ranks) + ".pdf"
    plt.savefig(os.path.join(output_folder, fig_filename), bbox_inches='tight')


""" Plot the ground_truth

Input is a list of ground-truth vals. Use them to plot a line
chart of groud-truth vals for choosing a suitable model.
"""
def plot_groundtruth_by_iters(ground_truth, num_plot_iters):
    # for the chart information
    plt.xlabel("Iterations")
    plt.ylabel("Load (runtime in s)")
    plt.title("Load per Iteration")

    # for x_index
    num_iters = num_plot_iters # len(ground_truth)
    x_indices = np.arange(num_iters)
    np_data_arr = np.array(ground_truth[0:num_iters])

    # plot the data
    plt.plot(x_indices, np_data_arr)
    
    # plt.yscale('log')
    plt.grid(True)
    plt.show()


"""Dislay total_runtime per iter per rank

Input is a list of runtime-data per rank. Use them to print
and check on the screen.
"""
def display_iterruntime_per_rank(profile_data):
    print('[Rank] \t num_iters')
    for i in range(len(profile_data)):
        data_per_rank = profile_data[i]
        if len(data_per_rank[1]) != 0:
            rank_id             = data_per_rank[0]
            iter_data_per_rank  = data_per_rank[1]
            num_iters           = len(iter_data_per_rank)

            # show rank-info
            print('{:d} \t {:d}'.format(rank_id, num_iters))

            # a list of loads per rank
            loads_per_rank = []
            statement_load_per_rank = "\t"

            # gather tasks per iter, and then sum total_runtime/iter
            for j in range(num_iters):
                runtime_per_iter     = iter_data_per_rank[j]
                formated_value = float("{:.4f}".format(runtime_per_iter))
                loads_per_rank.append(formated_value)

                # add load-values to the print-statement
                # statement_load_per_rank += str(formated_value) + "\t"
            
            # print the statement
            # print(statement_load_per_rank)
        else:
            print('{:d} \t something\'s wrong here'.format(data_per_rank[0]))


"""Dislay task_prof data per iter per rank

Input is a list of taskprof_data (including task-args
with their runtime). Use them to print and check on
the screen.
"""
def display_taskprof_per_rank(profile_data):
    
    unique_args_set_list = []

    for i in range(len(profile_data)):
        data_per_rank = profile_data[i]
        if len(data_per_rank[1]) != 0:
            rank_id             = data_per_rank[0]
            iter_data_per_rank  = data_per_rank[1]
            num_iters           = len(iter_data_per_rank)
            for j in range(num_iters):
                tasklist = iter_data_per_rank[j]
                num_tasks = len(tasklist)
                taskargs_set = []
                for k in range(num_tasks):
                    arg1 = tasklist[k].arg1
                    arg2 = tasklist[k].arg2
                    arg3 = tasklist[k].arg3
                    arg4 = tasklist[k].arg4
                    # taskargs_set.extend([arg1, arg2, arg3, arg4])
                    args_set = {arg1, arg2, arg3, arg4}
                    taskargs_set.append(args_set)

                if j == 0:
                    forz_args_sets = set(frozenset(s) for s in taskargs_set)
                    unique_args_sets = set(forz_args_sets)
            unique_args_set_list.append(unique_args_sets)
            
            # print("Rank {}: {} | len = {}".format(rank_id, unique_args_sets, len(unique_args_sets)))
    
    return 


"""Train the prediction model

Input is a transformed list of data. Besides, we need to
delcare n_features, n_out, ...
"""
def train_model(X, Y, n_in, n_out):
    # convert into pytorch variable
    x, y = Variable(X), Variable(Y)

    learning_rate = 0.01
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # declare the network - using Nerual-Net model
    # net_model = PredictionModel(n_features=n_in, n_hidden=10, n_output=n_out)
    # optimizer = optim.SGD(net_model.parameters(), lr=learning_rate)

    # declare the network - using Linear-Regression model
    net_model = LinearRegression(n_feature=n_in, n_hidden1=200, n_hidden2=100, n_output=1)
    optimizer = torch.optim.Adam(net_model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(net_model.parameters(), lr=learning_rate)

    # training with visualizing the process
    n_epoch = 1000
    net_model.train()
    
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        prediction = net_model(x)
        loss = loss_fn(prediction, y)
        
        loss.backward()
        optimizer.step()
        
        print("[TRAINING] Epoch {} - Loss = {}".format(epoch, loss))

    return net_model


"""Validate the prediction model

Input is a transformed list of vali-data. Besides, we need to
transfer the trained-model, num_inputs to validate, ...
"""
def validate_model(net_model, X_valid, Y_valid, num_inputs):

    # convert valid_inputs to Variable_type
    x_val = Variable(X_valid)

    # validating the model
    print("Task ID: [inputs...] \t pred_val | real_val")
    for i in range(num_inputs):
        tid = int((x_valid[i])[0])
        pred_val = net_model(x_val[i]).item()
        real_val = float(Y_valid[i])
        print("{} ... \t {:4f} | {:4f}".format(tid, pred_val, real_val))

    return 0

"""The main function

There are 3 mains phases in the boday of this source,
that could be reading-logs, visualizing, and prediction.
"""
if __name__ == "__main__":

    # get folder-path of log-files
    folder = sys.argv[1]

    # extract the folder name
    folder_name = folder.split("/")
    sub_folder_name   = folder_name[len(folder_name)-1].split("-")
    num_nodes = int(sub_folder_name[2])

    # gather results in the folder of logs
    runtime_profile_data = gather_results_iterruntime(folder, num_nodes)
    task_profile_data = gather_results_taskprof(folder, num_nodes)

    # sorting the profile-data by rank
    task_profile_data.sort(key=lambda x:x[0])
    runtime_profile_data.sort(key=lambda x:x[0])

    """ -------- calculate and check total_runtime per iter """
    # display_iterruntime_per_rank(runtime_profile_data)

    """ -------- check task_prof data per iter per rank """
    # display_taskprof_per_rank(task_profile_data)

    # visualize the runtime per iter
    # output_folder = "./"
    # plot_runtime_by_iters(runtime_profile_data, output_folder)

    # visualize total_runtime per rank
    # plot_total_runtime_by_rank(runtime_profile_data, output_folder)
    
    """ -------- training based on args and iter-idx """
    # get the profile_data of the first rank
    rank = 2
    rank_task_profiled_data = task_profile_data[rank]
    rank_runtime_data = runtime_profile_data[rank]
    # rank = rank_task_profiled_data[0]
    num_iters = len(rank_task_profiled_data[1])
    num_tasks_per_iter = len((rank_task_profiled_data[1])[0])
    num_total_tasks = num_iters * num_tasks_per_iter
    print("Rank {}: {} iters, {} tasks in total".format(rank, num_iters, num_total_tasks))
    task_ars_data = rank_task_profiled_data[1]
    runtime_data = rank_runtime_data[1]
    filtered_data = []
    ground_truth = []
    for i in range(num_iters):
        task_args_per_iter = task_ars_data[i]
        args_list = []
        for j in range(num_tasks_per_iter):
            task = task_args_per_iter[j]
            arg1 = task.arg1
            arg2 = task.arg2
            arg3 = task.arg3
            arg4 = task.arg4
            args_list.append(arg1)
            args_list.append(arg2)
            args_list.append(arg3)
            args_list.append(arg4)
        uniqe_args_per_iter = set(args_list)
        sorted_uniqe_args_set = sorted(uniqe_args_per_iter)
        # print("Iter [{}, {}, {}]".format(i, uniqe_args_per_iter, runtime_data[i]))
        data_row = [i]
        for e in uniqe_args_per_iter:
            data_row.append(e)
        data_row.append(runtime_data[i])
        ground_truth.append(runtime_data[i])

        # put into a larger list
        filtered_data.append(data_row)
    

    """ -------- visualize the dataset before training """
    # num_plot_iters = 100
    # plot_groundtruth_by_iters(ground_truth, num_plot_iters)

    """ --------- write dataset oto csv file """
    csv_file = open("sample-dataset-r"+str(rank)+".csv", 'w')
    with csv_file:
        writer = csv.writer(csv_file, delimiter ='\t')
        for row in filtered_data:
            print(row)
            writer.writerow(row)
    

    """ -------- convert filtered_data to tensor_type """
    # tensor_dataset = torch.FloatTensor(filtered_data)
    # size = len(tensor_dataset)
    # num_elements = len(tensor_dataset[0])
    # # setting num_features for the input here
    # input_num    = num_elements - 1
    # output_num   = num_elements - input_num
    # NUM_TRAIN = int(size / 2)
    # NUM_VALID = int(NUM_TRAIN / 2)
    # NUM_TEST  = int(size - NUM_TRAIN - NUM_VALID)
    # print("Size: {} | num_inputs={}, num_outputs={}, NUM_TRAIN,_VALID,_TEST={}, {}, {}".format(size, input_num, output_num, NUM_TRAIN, NUM_VALID, NUM_TEST))


    """ ---------- refactor dataset for training & validating """
    # Case 1: get all features
    # num_features = input_num
    # num_out      = output_num
    # label_idx    = num_elements-1
    # x_train = tensor_dataset[0:NUM_TRAIN, 0:num_features].view(NUM_TRAIN, num_features)
    # y_train = tensor_dataset[0:NUM_TRAIN, label_idx].view(NUM_TRAIN, num_out)
    # x_valid = tensor_dataset[NUM_TRAIN:(NUM_TRAIN+NUM_VALID), 0:num_features].view(NUM_VALID, num_features)
    # y_valid = tensor_dataset[NUM_TRAIN:(NUM_TRAIN+NUM_VALID), label_idx].view(NUM_VALID, num_out)

    # Case 2: get 1 feature is iter_idx (it looks like a sim/cos function)
    # num_features = 1
    # num_out      = output_num
    # label_idx    = num_elements-1
    # x_train = tensor_dataset[0:NUM_TRAIN, 0].view(NUM_TRAIN, num_features)
    # y_train = tensor_dataset[0:NUM_TRAIN, label_idx].view(NUM_TRAIN, num_out)
    # x_valid = tensor_dataset[NUM_TRAIN:(NUM_TRAIN+NUM_VALID), 0].view(NUM_VALID, num_features)
    # y_valid = tensor_dataset[NUM_TRAIN:(NUM_TRAIN+NUM_VALID), label_idx].view(NUM_VALID, num_out)

    """ ---------- training & validating """
    # call training the model
    # net_model = train_model(x_train, y_train, num_features, num_out)

    # call validating the model
    # num_validate = 50
    # validate_model(net_model, x_valid, y_valid, num_validate)

    
