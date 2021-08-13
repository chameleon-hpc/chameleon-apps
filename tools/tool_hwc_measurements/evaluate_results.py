import os, sys
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import csv

script_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
path_src_folder = "F:\\repos\\hpc-projects\\chameleon\\chameleon-apps\\tools\\tool_hwc_measurements\\results"

class CFileMetaData():

    def __init__(self, file_path, file_name):
        # get file name and split
        file_name = file_name.split(".")[0]
        tmp_split = file_name.split("_")
        
        self.type           = tmp_split[1]
        self.m_size         = int(tmp_split[2])
        self.repetition     = int(tmp_split[4])
        self.time_overall   = -1
        self.avg_time_task  = -1

        with open(file_path) as f: lines = [x.strip() for x in list(f)]
        list_avg_task_times = []
        for line in lines:
            if "chameleon took" in line:
                tmp_split = line.split(" ")
                self.time_overall = float(tmp_split[-1].strip())
                continue
            if "_time_task_execution_local_sum" in line:
                tmp_split = line.split("\t")
                list_avg_task_times.append(float(tmp_split[-1].strip()))
                continue
        if len(list_avg_task_times) > 0:
            self.avg_time_task = st.mean(list_avg_task_times)


target_folder_data  = os.path.join(path_src_folder, "result_data")
if not os.path.exists(target_folder_data):
    os.makedirs(target_folder_data)

list_files = []

for file in os.listdir(path_src_folder):
    if file.endswith(".log") and "result_" in file:
        cur_file_path = os.path.join(path_src_folder, file)
        print(cur_file_path)

        # read file meta data
        file_meta = CFileMetaData(cur_file_path, file)
        list_files.append(file_meta)

# get unique combinations
unique_types        = sorted(list(set([x.type for x in list_files])))
unique_m_sizes      = sorted(list(set([x.m_size for x in list_files])))

target_file_path = os.path.join(target_folder_data, f"aggregated_data_overall.csv")
with open(target_file_path, mode="w", newline='') as f:
    writer = csv.writer(f, delimiter=';')
    
    # write header once
    header = ['Matrix Size', 'Data [KB]']
    for i in range(len(unique_types)):
        header.append("Time " + unique_types[i] + " [sec]")
        header.append("Overhead " + unique_types[i])
    writer.writerow(header)

    for cur_size in unique_m_sizes:
        cur_data_volume = cur_size * cur_size * 3 * 8 / 1000.0
        idx_baseline = None
        tmp_arr_data = [np.nan for x in range(2*len(unique_types))]

        for i in range(len(unique_types)):
            cur_type = unique_types[i]
            if cur_type == "no-tool":
                idx_baseline = i
            # get sub_list
            sub = [x for x in list_files if x.type == cur_type and x.m_size == cur_size]
            if len(sub) > 0:
                tmp_arr_data[i * 2] = st.mean([x.time_overall for x in sub])
        
        # calculate overheads / slow down
        for i in range(len(unique_types)):
            tmp_arr_data[i * 2 + 1] = tmp_arr_data[i * 2] / tmp_arr_data[idx_baseline * 2]
        # write to csv
        writer.writerow([cur_size, cur_data_volume] + tmp_arr_data)

target_file_path = os.path.join(target_folder_data, f"aggregated_data_per_task.csv")
with open(target_file_path, mode="w", newline='') as f:
    writer = csv.writer(f, delimiter=';')
    
    # write header once
    header = ['Matrix Size', 'Data [KB]']
    for i in range(len(unique_types)):
        header.append("Time " + unique_types[i] + " [sec]")
        header.append("Overhead " + unique_types[i])
    writer.writerow(header)

    for cur_size in unique_m_sizes:
        cur_data_volume = cur_size * cur_size * 3 * 8 / 1000.0
        idx_baseline = None
        tmp_arr_data = [np.nan for x in range(2*len(unique_types))]

        for i in range(len(unique_types)):
            cur_type = unique_types[i]
            if cur_type == "no-tool":
                idx_baseline = i
            # get sub_list
            sub = [x for x in list_files if x.type == cur_type and x.m_size == cur_size]
            if len(sub) > 0:
                tmp_arr_data[i * 2] = st.mean([x.avg_time_task for x in sub])
        
        # calculate overheads / slow down
        for i in range(len(unique_types)):
            tmp_arr_data[i * 2 + 1] = tmp_arr_data[i * 2] / tmp_arr_data[idx_baseline * 2]
        # write to csv
        writer.writerow([cur_size, cur_data_volume] + tmp_arr_data)
