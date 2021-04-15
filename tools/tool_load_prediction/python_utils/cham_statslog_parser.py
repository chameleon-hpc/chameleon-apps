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


#########################################################
###### Util-functions
#########################################################
"""Read log-file and parse total_load per iter

As the layout of data for storing stats-data per rank. We define
total_load_list = [ [r0_i0_t, r0_i1_t, ..., r0_iK_t],
                    [... ],
                    [rN_i0_t, rN_i1_t, ..., rN_iK_t] ]
"""
def parse_stats_iter_runtime(filename, num_ranks):

    # open the logfile
    file = open(filename, 'r')

    # for storing values
    data_per_rank = []
    for i in range(num_ranks):
        data_per_rank.append([])

    # getting total_load/iter
    for line in file:
        # just get the line with the content
        if "_time_task_execution_overall_sum" in line:
            data_per_line = line.split("\t")
            get_rank = (data_per_line[0]).split(" ")[1]
            rank = int(re.findall(r'\d+', get_rank)[0])
            total_runtime = float(data_per_line[3])
            avg_iter_runt = total_runtime
            # append data to the list/rank
            data_per_rank[rank].append(avg_iter_runt)

    # return the result
    return data_per_rank


"""Read log-file and parse load per rank in detail

As the layout of data for storing stats-data per rank. We define 3 lists
local_load_list = [ [r0_i0_l, r0_i1_l, ..., r0_iK_l],... , [rN_i0_l, ..., rN_iK_l]]
stole_load_list = [ [r0_i0_s, r0_i1_s, ..., r0_iK_s],... , [rN_i0_s, ..., rN_iK_s]]
repli_load_list = [ [r0_i0_r, r0_i1_r, ..., r0_iK_r],... , [rN_i0_r, ..., rN_iK_r]]
"""
def parse_stats_load_indetail(filename, num_ranks):

    # open the logfile
    file = open(filename, 'r')

    # for storing values
    local_load_list = []
    stole_load_list = []
    repli_load_list = []
    for i in range(num_ranks):
        local_load_list.append([])
        stole_load_list.append([])
        repli_load_list.append([])

    # getting total_load/iter
    for line in file:
        # just get the line with the content
        if "_time_task_execution_local_sum" in line:
            line_local = line.split("\t")
            get_r_local = (line_local[0]).split(" ")[1]
            r_local = int(re.findall(r'\d+', get_r_local)[0])
            local_load = float(line_local[3])
            local_load_list[r_local].append(local_load)

        elif "_time_task_execution_stolen_sum" in line:
            line_stole = line.split("\t")
            get_r_stole = (line_stole[0]).split(" ")[1]
            r_stole = int(re.findall(r'\d+', get_r_stole)[0])
            stole_load = float(line_stole[3])
            stole_load_list[r_stole].append(stole_load)

        elif "_time_task_execution_replicated_sum" in line:
            line_repli = line.split("\t")
            get_r_repli = (line_repli[0]).split(" ")[1]
            r_repli = int(re.findall(r'\d+', get_r_repli)[0])
            repli_load = float(line_repli[3])
            (repli_load_list[r_repli]).append(repli_load)
    
    # check the data
    # for i in range(num_ranks):
    #     print("Local_Load R{}: {} iters".format(i, len(local_load_list[i])))
    #     print("Stole_Load R{}: {} iters".format(i, len(stole_load_list[i])))
    #     print("Repli_Load R{}: {} iters".format(i, len(repli_load_list[i])))

    # return the result
    return local_load_list, stole_load_list, repli_load_list


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
    num_iters = len(first_rank_data)
    x_indices = np.arange(num_iters)

    # traverse the profile-data
    num_ranks = len(stats_data)
    bottom_layer = np.zeros(num_iters)
    for i in range(num_ranks):
        data_per_rank = stats_data[i]
        dat_label    = "R_" + str(i)

        if i != 0:
            prev_data_per_rank = stats_data[i-1]
            prev_np_data_arr = np.array((prev_data_per_rank)[:num_iters])
            bottom_layer += prev_np_data_arr

        # convert data to numpy_arr
        if len(data_per_rank) != 0:
            np_data_arr = np.array((data_per_rank)[:num_iters])
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


"""Plot stacked-load types by ranks in a single iter

Input is a list of runtime-data per rank. Use them to plot
a stack-bar chart for easily to compare the load imbalance.
"""
def plot_stackedload_by_ranks(local_load, stole_load, repli_load, iter_idx, out_folder):
    
    # for the chart information
    plt.xlabel("Rank")
    plt.ylabel("Total_Load (in seconds)")
    plt.title("Load in detail by ranks [Iter-" + str(iter_idx) + "]")

    # for x_index
    num_ranks = len(local_load)
    x_indices = np.arange(num_ranks)
    bottom_layer = np.zeros(num_ranks)
    dat_label = ["local_l", "stole_l", "repli_l"]

    # traverse the profile-data & plot
    l_arr, s_arr, r_arr = [], [], []
    for i in range(num_ranks):
        # get local_load rank i at iter = iter_idx
        l_value = (local_load[i])[iter_idx]
        l_arr.append(l_value)
        # get stolen_load rank i at iter = iter_idx
        s_value = (stole_load[i])[iter_idx]
        s_arr.append(s_value)
        # get replic_load rank i at iter = iter_idx
        r_value = (repli_load[i])[iter_idx]
        r_arr.append(r_value)

    # convert to np_array
    np_l_arr = np.array(l_arr)
    np_s_arr = np.array(s_arr)
    np_r_arr = np.array(r_arr)

    # calculate total load
    np_total_arr = np_l_arr + np_s_arr + np_r_arr
    sort_np_total_arr = sorted((e, i) for i, e in enumerate(np_total_arr))
    pair_rank_by_load(sort_np_total_arr, iter_idx)

    for l in range(3):
        # because we have 3 types of task-load, so num_layers = 3
        if l == 0:
            plt.bar(x_indices, np_l_arr, label=dat_label[l])
        elif l == 1:
            bottom_layer += np_l_arr
            plt.bar(x_indices, np_s_arr, bottom=bottom_layer, label=dat_label[l])
        else:
            bottom_layer += np_s_arr
            plt.bar(x_indices, np_r_arr, bottom=bottom_layer, label=dat_label[l])
    
    # print("local_load: {}".format(np_l_arr))
    # print("stole_load: {}".format(np_s_arr))
    # print("repli_load: {}".format(np_r_arr))
    
    plt.grid(True)
    plt.legend(loc='best')
    # plt.show()

    # save the figure
    fig_filename = "load_diff_iter" + str(iter_idx) + "_" + str(num_ranks) + "_ranks_from_chamstats_logs" + ".pdf"
    plt.savefig(os.path.join(out_folder, fig_filename), bbox_inches='tight')


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


"""Pairing ranks by load

Input is a sorted list of load values which is associated with
the indices of ranks.
"""
def pair_rank_by_load(s_load_rank_info, iter):
    num_ranks = len(s_load_rank_info)
    n = int(num_ranks / 2)
    check_statement = "Iter{}: ".format(iter)
    for i in range(n):
        src = s_load_rank_info[i]
        vic = s_load_rank_info[(num_ranks-1)-i]
        src_r_idx = src[1]
        vic_r_idx = vic[1]
        load_src = src[0]
        load_vic = vic[0]
        diff = ((load_vic - load_src) / load_vic) * 100
        num_rec_mig_tasks = (load_vic - load_src) / (load_vic / 368)
        stats_info = "[R{}({:.2f}),R{}({:.2f}),{:.2f},{:.1f}] ".format(src_r_idx, load_src, vic_r_idx, load_vic, diff, num_rec_mig_tasks)
        check_statement += stats_info
    print(check_statement)


"""Display the statistic data for quickly checking

Input is num of ranks in the dataset, the raw stats_data
which we have read from the cham_log file.
"""
def display_stats_data(stats_data, num_ranks):
    for i in range(num_ranks):
        data_per_rank = stats_data[i]
        rank = data_per_rank[0]
        runtime_list = data_per_rank[1]
        statement = str(rank) + ": "
        for j in range(len(runtime_list)):
            formated_val = float("{:.4f}".format(runtime_list[j]))
            statement += str(formated_val) + "\t"
        print(statement)

"""Display the difference of load & num_mig_tasks recommendation

Input is num of ranks in the dataset, the raw stats_data
which we have read from the cham_log file.
"""
def display_load_diff_recomm(stats_data, num_ranks):
    num_iters = len((stats_data[0]))
    for i in range(num_iters):
        load_ranks_info = []
        for r in range(num_ranks):
            load = ((stats_data[r]))[i]   # load of rank-r at iter-i
            load_ranks_info.append(load)
        # sorting the load list
        s_load_rank_info = sorted((e, i) for i, e in enumerate(load_ranks_info))
        # pairing ranks and estimate num of tasks should be migrated
        pair_rank_by_load(s_load_rank_info, i) # pass a sorted list with idx


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
    # display_stats_data(stats_data, num_ranks)
    
    # display the diff of load & recommendation
    # display_load_diff_recomm(stats_data, num_ranks)

    # generate dataset by chain-on-chain runtimes per iter
    if (len(sys.argv) <= 3):
        rank = 0
    else:
        rank = int(sys.argv[3])
    # num_features = 6
    # cc_runtime_dataset_generator(stats_data, rank, num_features)
    
    # plot the load-data by stacked-rank per iter
    # stacked_rank_folder = "./figures/"
    # plot_runtime_by_iters(stats_data, stacked_rank_folder)

    # read and parse values from the input in detail
    loc_load_list,sto_load_list,rep_load_list = parse_stats_load_indetail(cham_stats_file, num_ranks)

    # plot the load-diff by stacked-load types in a single iter
    stacked_loadtypes_folder = "./figures/"
    iter_idx = 199
    plot_stackedload_by_ranks(loc_load_list, sto_load_list, rep_load_list, iter_idx, stacked_loadtypes_folder)

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