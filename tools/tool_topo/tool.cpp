#ifndef _BSD_SOURCE
#define _BSD_SOURCE
#endif

#define _DEFAULT_SOURCE
#include <stdio.h>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

#include <mpi.h>
#include <mutex>
#include <fstream>
#include <cstring>
#include <atomic>
using namespace std;

#include "chameleon.h"
#include "chameleon_tools.h"

#include <unistd.h>
#include <sys/syscall.h>
#include <inttypes.h>
#include <assert.h>
#include <numeric>

#ifdef TRACE
#include "VT.h"
static int event_tool_task_create = -1;
static int event_tool_task_exec = -1;
#endif

#define TASK_TOOL_SAMPLE_DATA_SIZE 10

static cham_t_set_callback_t cham_t_set_callback;
static cham_t_get_callback_t cham_t_get_callback;
static cham_t_get_rank_data_t cham_t_get_rank_data;
static cham_t_get_thread_data_t cham_t_get_thread_data;
static cham_t_get_rank_info_t cham_t_get_rank_info;
static cham_t_get_task_param_info_t cham_t_get_task_param_info;
static cham_t_get_task_param_info_by_id_t cham_t_get_task_param_info_by_id;
static cham_t_get_task_data_t cham_t_get_task_data;

std::atomic<int> TOPO_MIGRATION_STRAT(0);

static uint8_t* topo_distances; // distance in hops from the current rank to the rank at the corresponding index (sorted after ranks) 
static uint8_t* topo_region_0; // all ranks in the same node as the current rank
static uint8_t* topo_region_2; // all ranks in the same rack but not the same node as the current rank
static uint8_t* topo_region_4; // all ranks not in the same rack
static uint8_t ranks_in_r0; // number of ranks in topo region 0
static uint8_t ranks_in_r2; // number of ranks in topo region 2
static uint8_t ranks_in_r4; // number of ranks in topo region 4

static const int nodename_length = 7; // 56=8*7, nodename has max 7 chars
static char my_nodename[nodename_length+1]; // +1 for the string terminator
std::mutex comp_topology_mutex;
static bool topo_distances_initialized = false;
MPI_Comm topotool_comm;

// sample struct to save some information for a task
typedef struct my_task_data_t {
    TYPE_TASK_ID task_id;
    size_t size_data;
    double * sample_data;    
} my_task_data_t;

void getEnvironment(){
    char *tmp = nullptr;
    tmp = nullptr;
    tmp = std::getenv("TOPO_MIGRATION_STRAT");
    if(tmp) {
         TOPO_MIGRATION_STRAT = std::atof(tmp);
    }
}

// return true if nodes are directly connected to the same switch
// (returns also false if node couldn't be found)
bool checkSameRack(char* node1_chars, char* node2_chars){
    // printf("CheckSameRack for %s %s\n",node1_chars,node2_chars);
    string node1 = node1_chars;
    string node2 = node2_chars;
    std::ifstream topo_file;
    // topo_file.open("testinput.log");
    topo_file.open("mapping_switches_nodes.log");
    int curLine = 0;
    int line1 = -1;
    bool found1 = false;
    std::string switch1;
    std::string switch2;
    int line2 = -1;
    bool found2 = false;
    std::string line;
    std::string currentSwitch;
    // search for the lines
    while(getline(topo_file, line)){
        // printf("Checking Line %d of topo_file.\n",curLine);
        if (line.find("Switch")!=string::npos){
            currentSwitch = line;
        }
        if (!found1 && line.find(node1)!=string::npos){
            // printf("Found Node1 in line %d!\n",curLine);
            line1 = curLine;
            switch1 = currentSwitch;
            found1 = true;
        }
        if (!found2 && line.find(node2)!=string::npos){
            // printf("Found Node2 in line %d!\n",curLine);
            line2 = curLine;
            switch2 = currentSwitch;
            found2 = true;
        }
        if (found1 && found2){
            // printf("Both nodes have been found!\n");
            break;
        }
        curLine++;
    }
    // printf("Found Node1 (%s) in line %d and Node2 (%s) in line %d\n",node1_chars,line1,node2_chars,line2);
    if (found1 && found2){
        // printf("Node1:%s Switch1:%s Node2:%s Switch2:%s\n",node1_chars,switch1.c_str(),node2_chars,switch2.c_str());
        if (switch1.compare(switch2) == 0){
            return true;
        }
    }
    topo_file.close();
    return false;
}

static void
on_cham_t_callback_thread_init(
    cham_t_data_t *thread_data)
{
    thread_data->value = syscall(SYS_gettid);
    // printf("on_cham_t_callback_thread_init ==> thread_id=%d\n", thread_data->value);
}

int compare( const void *pa, const void *pb )
{
    const int *a = (int*)pa;
    const int *b = (int*)pb;
    if(a[0] == b[0])
        return a[1] - b[1];
    else
        return a[0] - b[0];
}

static void
on_cham_t_callback_select_num_tasks_to_offload(
    int32_t* num_tasks_to_offload_per_rank,
    const int32_t* load_info_per_rank,
    int32_t num_tasks_local,
    int32_t num_tasks_stolen)
{
    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    //printf("on_cham_t_callback_select_num_tasks_to_offload ==> comm_rank=%d;comm_size=%d;num_tasks_to_offload_per_rank=" DPxMOD ";load_info_per_rank=" DPxMOD ";num_tasks_local=%d;num_tasks_stolen=%d\n", r_info->comm_rank, r_info->comm_size, DPxPTR(num_tasks_to_offload_per_rank), DPxPTR(load_info_per_rank), num_tasks_local, num_tasks_stolen);

    static double min_abs_imbalance_before_migration = -1;
    if(min_abs_imbalance_before_migration == -1) {
        // try to load it once
        char *min_abs_balance = getenv("MIN_ABS_LOAD_IMBALANCE_BEFORE_MIGRATION");
        if(min_abs_balance) {
            min_abs_imbalance_before_migration = atof(min_abs_balance);
        } else {
            min_abs_imbalance_before_migration = 2;
        }
        //printf("R#%d MIN_ABS_LOAD_IMBALANCE_BEFORE_MIGRATION=%f\n", r_info->comm_rank, min_abs_imbalance_before_migration);
    }

    static double min_rel_imbalance_before_migration = -1;
    if(min_rel_imbalance_before_migration == -1) {
        // try to load it once
        char *min_rel_balance = getenv("MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION");
        if(min_rel_balance) {
            min_rel_imbalance_before_migration = atof(min_rel_balance);
        } else {
            // default relative threshold
            min_rel_imbalance_before_migration = 0.0;
        }
        //printf("R#%d MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION=%f\n", r_info->comm_rank, min_rel_imbalance_before_migration);
    }

    // some variables used by multiple strategies
    int i;
    // int** tmp_sorted_array;

    switch(TOPO_MIGRATION_STRAT){
        // aggressive offload everything to the nearest ranks till they have average load
        case 1: { // without the bracket compiler is concerned about the tmp_sorted_arrays...
            // compute avg load
            int total_l = 0;
            total_l =std::accumulate(&load_info_per_rank[0], &load_info_per_rank[r_info->comm_size], total_l);  
            int avg_l = total_l / r_info->comm_size;

            // return if load of current task is lower than some threshold, dependending on avg load
            int myLoad = load_info_per_rank[r_info->comm_rank];
            if (myLoad-avg_l < min_abs_imbalance_before_migration){
                return;
            }

            // distribute number of tasks higher than avg to ranks with lower than avg load
            int moveable = myLoad-avg_l;
            int tmp_ldiff;
            int inc_l;
            // int i;
            int inspected_Rank_id;
            int inspected_Rank_ld;
            // int** tmp_sorted_array;

            // // iterate over the distance hierarchy
            // for (int i_dist = 0; i<3; i++){
            //     switch(i_dist){
            //         case 0:
            //             // Sort rank loads and keep track of ranks for distance 0
            //             tmp_sorted_array[ranks_in_r0][2];
            //             for (i = 0; i < ranks_in_r0; i++)
            //             {
            //                 tmp_sorted_array[i][0] = load_info_per_rank[topo_region_0[i]]; // load
            //                 tmp_sorted_array[i][1] = topo_region_0[i];      // rank
            //             }
            //             qsort(tmp_sorted_array, ranks_in_r0, sizeof tmp_sorted_array[0], compare);
            //         break;
            //         case 1:
            //             // Sort rank loads and keep track of ranks for distance 2
            //             tmp_sorted_array[ranks_in_r2][2];
            //             for (i = 0; i < ranks_in_r2; i++)
            //             {
            //                 tmp_sorted_array[i][0] = load_info_per_rank[topo_region_2[i]]; // load
            //                 tmp_sorted_array[i][1] = topo_region_2[i];      // rank
            //             }
            //             qsort(tmp_sorted_array, ranks_in_r2, sizeof tmp_sorted_array[0], compare);
            //         break;
            //         case 2:
            //             // Sort rank loads and keep track of ranks for distance 4
            //             tmp_sorted_array[ranks_in_r4][2];
            //             for (i = 0; i < ranks_in_r4; i++)
            //             {
            //                 tmp_sorted_array[i][0] = load_info_per_rank[topo_region_4[i]]; // load
            //                 tmp_sorted_array[i][1] = topo_region_4[i];      // rank
            //             }
            //             qsort(tmp_sorted_array, ranks_in_r4, sizeof tmp_sorted_array[0], compare);
            //         break;
            //     }

            //     // search for migration victims (first with distance 0, then 2, then 4)
            //     for (i=0; i<ranks_in_r0; i++){
            //         inspected_Rank_ld = tmp_sorted_array[i][0];
            //         inspected_Rank_id = tmp_sorted_array[i][1];
            //         tmp_ldiff = avg_l-inspected_Rank_ld;
            //         if(!(tmp_ldiff<min_rel_imbalance_before_migration)){
            //             inc_l = std::min( tmp_ldiff, moveable );
            //             num_tasks_to_offload_per_rank[inspected_Rank_id] = inc_l;
            //             moveable -= inc_l;
            //             if (moveable < min_abs_imbalance_before_migration){
            //                 return;
            //             }
            //         }
            //         else { 
            //             // load of inspected rank is not low enough
            //             // array is sortet, hence the next ranks wont have less load
            //             break;
            //         }
            //     }
            // }

            // Sort rank loads and keep track of ranks for distance 0
            int tmp_sorted_array0[ranks_in_r0][2];
            for (i = 0; i < ranks_in_r0; i++)
            {
                tmp_sorted_array0[i][0] = load_info_per_rank[topo_region_0[i]]; // load
                tmp_sorted_array0[i][1] = topo_region_0[i];      // rank
            }
            qsort(tmp_sorted_array0, ranks_in_r0, sizeof tmp_sorted_array0[0], compare);

            // search for migration victims (first with distance 0, then 2, then 4)
            for (i=0; i<ranks_in_r0; i++){
                inspected_Rank_ld = tmp_sorted_array0[i][0];
                inspected_Rank_id = tmp_sorted_array0[i][1];
                tmp_ldiff = avg_l-inspected_Rank_ld;
                if(!(tmp_ldiff<min_rel_imbalance_before_migration)){
                    inc_l = std::min( tmp_ldiff, moveable );
                    num_tasks_to_offload_per_rank[inspected_Rank_id] = inc_l;
                    moveable -= inc_l;
                    if (moveable < min_abs_imbalance_before_migration){
                        return;
                    }
                }
                else { 
                    // load of inspected rank is not low enough
                    // array is sortet, hence the next ranks wont have less load
                    break;
                }
            }

            // Sort rank loads and keep track of ranks for distance 2
            int tmp_sorted_array2[ranks_in_r2][2];
            for (i = 0; i < ranks_in_r2; i++)
            {
                tmp_sorted_array2[i][0] = load_info_per_rank[topo_region_2[i]]; // load
                tmp_sorted_array2[i][1] = topo_region_2[i];      // rank
            }
            qsort(tmp_sorted_array2, ranks_in_r2, sizeof tmp_sorted_array2[0], compare);

            // search for migration victims (first with distance 0, then 2, then 4)
            for (i=0; i<ranks_in_r0; i++){
                inspected_Rank_ld = tmp_sorted_array2[i][0];
                inspected_Rank_id = tmp_sorted_array2[i][1];
                tmp_ldiff = avg_l-inspected_Rank_ld;
                if(!(tmp_ldiff<min_rel_imbalance_before_migration)){
                    inc_l = std::min( tmp_ldiff, moveable );
                    num_tasks_to_offload_per_rank[inspected_Rank_id] = inc_l;
                    moveable -= inc_l;
                    if (moveable < min_abs_imbalance_before_migration){
                        return;
                    }
                }
                else { 
                    // load of inspected rank is not low enough
                    // array is sortet, hence the next ranks wont have less load
                    break;
                }
            }

            // Sort rank loads and keep track of ranks for distance 4
            int tmp_sorted_array4[ranks_in_r4][2];
            for (i = 0; i < ranks_in_r4; i++)
            {
                tmp_sorted_array4[i][0] = load_info_per_rank[topo_region_4[i]]; // load
                tmp_sorted_array4[i][1] = topo_region_4[i];      // rank
            }
            qsort(tmp_sorted_array4, ranks_in_r4, sizeof tmp_sorted_array4[0], compare);

            // search for migration victims (first with distance 0, then 2, then 4)
            for (i=0; i<ranks_in_r0; i++){
                inspected_Rank_ld = tmp_sorted_array4[i][0];
                inspected_Rank_id = tmp_sorted_array4[i][1];
                tmp_ldiff = avg_l-inspected_Rank_ld;
                if(!(tmp_ldiff<min_rel_imbalance_before_migration)){
                    inc_l = std::min( tmp_ldiff, moveable );
                    num_tasks_to_offload_per_rank[inspected_Rank_id] = inc_l;
                    moveable -= inc_l;
                    if (moveable < min_abs_imbalance_before_migration){
                        return;
                    }
                }
                else { 
                    // load of inspected rank is not low enough
                    // array is sortet, hence the next ranks wont have less load
                    break;
                }
            }

            // implementation without sorting the arrays by load

            // // start searching for migration victims with distance 0
            // for (i=0; i<ranks_in_r0; i++){
            //     tmp_ldiff = avg_l-load_info_per_rank[topo_region_0[i]];
            //     if(!(tmp_ldiff<min_rel_imbalance_before_migration)){
            //         inc_l = std::min( tmp_ldiff, moveable );
            //         num_tasks_to_offload_per_rank[topo_region_0[i]] = inc_l;
            //         moveable -= inc_l;
            //         if (moveable < min_abs_imbalance_before_migration){
            //             return;
            //         }
            //     }
            // }

            // // search for migration victims with distance 2
            // for (i=0; i<ranks_in_r2; i++){
            //     tmp_ldiff = avg_l-load_info_per_rank[topo_region_2[i]];
            //     if(!(tmp_ldiff<min_rel_imbalance_before_migration)){
            //         inc_l = std::min( tmp_ldiff, moveable );
            //         num_tasks_to_offload_per_rank[topo_region_2[i]] = inc_l;
            //         moveable -= inc_l;
            //         if (moveable < min_abs_imbalance_before_migration){
            //             return;
            //         }
            //     }
            // }

            // // search for migration victims with distance 4
            // for (i=0; i<ranks_in_r4; i++){
            //     tmp_ldiff = avg_l-load_info_per_rank[topo_region_4[i]];
            //     if(!(tmp_ldiff<min_rel_imbalance_before_migration)){
            //         inc_l = std::min( tmp_ldiff, moveable );
            //         num_tasks_to_offload_per_rank[topo_region_4[i]] = inc_l;
            //         moveable -= inc_l;
            //         if (moveable < min_abs_imbalance_before_migration){
            //             return;
            //         }
            //     }
            // }

        break;
        } // case 1

        // default non-topology-aware chameleon strat
        // sort after load per rank, migrate tasks from upper to lower ends
        case 0:
        default: 
            // Sort rank loads and keep track of indices
            int tmp_sorted_array[r_info->comm_size][2];
            // tmp_sorted_array[r_info->comm_size][2];
            // int i;
            for (i = 0; i < r_info->comm_size; i++)
            {
                tmp_sorted_array[i][0] = load_info_per_rank[i];
                tmp_sorted_array[i][1] = i;
            }

            qsort(tmp_sorted_array, r_info->comm_size, sizeof tmp_sorted_array[0], compare);

            double min_val      = (double) load_info_per_rank[tmp_sorted_array[0][1]];
            double max_val      = (double) load_info_per_rank[tmp_sorted_array[r_info->comm_size-1][1]];
            double cur_load     = (double) load_info_per_rank[r_info->comm_rank];

            double ratio_lb                 = 0.0; // 1 = high imbalance, 0 = no imbalance
            if (max_val > 0) {
                ratio_lb = (double)(max_val-min_val) / (double)max_val;
            }

            if((cur_load-min_val) < min_abs_imbalance_before_migration)
                return;
            
            if(ratio_lb >= min_rel_imbalance_before_migration) {
                int pos = 0;
                for(i = 0; i < r_info->comm_size; i++) {
                    if(tmp_sorted_array[i][1] == r_info->comm_rank) {
                        pos = i;
                        break;
                    }
                }

                // only offload if on the upper side
                if((pos) >= ((double)r_info->comm_size/2.0))
                {
                    int other_pos = r_info->comm_size-pos-1;
                    int other_idx = tmp_sorted_array[other_pos][1];
                    double other_val = (double) load_info_per_rank[other_idx];

                    double cur_diff = cur_load-other_val;
                    // check absolute condition
                    if(cur_diff < min_abs_imbalance_before_migration)
                        return;
                    double ratio = cur_diff / (double)cur_load;
                    if(other_val < cur_load && ratio >= min_rel_imbalance_before_migration) {
                        //printf("R#%d Migrating\t%d\ttasks to rank:\t%d\tload:\t%f\tload_victim:\t%f\tratio:\t%f\tdiff:\t%f\n", r_info->comm_rank, 1, other_idx, cur_load, other_val, ratio, cur_diff);
                        num_tasks_to_offload_per_rank[other_idx] = 1;
                    }
                }
            }
        break;
    }
}

// call with only one thread per rank
// get nodenames, distribute nodenames over all ranks, compute distance matrix
void compute_distance_matrix(){
    char hostname[1024];
    gethostname(hostname, 1024);
    // int nodename_length = 7; // 56=8*7, nodename has max 7 chars
    // char my_nodename[nodename_length]; 
    strncpy(my_nodename, hostname, 7);
    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    int myRank = r_info->comm_rank;
    int topotool_comm_rank;
    MPI_Comm_rank(topotool_comm, &topotool_comm_rank);
    if (topotool_comm_rank != myRank){
        printf("!!!!!!!!!\nDuplicated Communicator in tools assigned other rank order! Rank %d is now rank %d!\n!!!!!!!!!\n",myRank,topotool_comm_rank);
    }
    printf("Rank %d on node %s\n", myRank, my_nodename);
    
    // nodenames of all ranks sorted after rank
    // char** rank_nodes = (char**)malloc((r_info->comm_size) * (nodename_length+1) * sizeof(char)); // doesn't work
    char rank_nodes[r_info->comm_size][nodename_length+1];
    MPI_Allgather(
        my_nodename, //void* send_data,
        nodename_length+1, //int send_count, data sent by each process
        MPI_CHAR, //MPI_Datatype send_datatype,
        *rank_nodes, //void* recv_data,
        nodename_length+1, //int recv_count, data received from each process
        MPI_CHAR, //MPI_Datatype recv_datatype,
        topotool_comm //MPI_Comm communicator
    ); 
    // Definition of Allgather: The block of data sent from the jth process is received by every process and placed in the jth block of the buffer recvbuf.
    // Hence, rank_nodes[i] should have the node from rank i
    // for(int i = 0; i<r_info->comm_size; i++){
    //     printf("Rank %d thinks rank %d runs on node %s\n",myRank,i,rank_nodes[i]);
    // }

    topo_distances = (uint8_t*)malloc(r_info->comm_size * sizeof(uint8_t));

    // Calculate distances to other ranks
    for (int i = 0; i<r_info->comm_size; i++){
        // On the same node
        if (strcmp(my_nodename, rank_nodes[i]) == 0){
            topo_distances[i]=0;
        }
        // On the same rack
        else if (checkSameRack(my_nodename, rank_nodes[i])){
            topo_distances[i]=2;
        }
        // Not on the same rack
        else{
            topo_distances[i]=4;
        }
    }

    // print distance matrix
    string print_distances = "";
    for(int i = 0; i<r_info->comm_size; i++){
        // printf("Rank %d from node %s has distance %d to rank %d on node%s\n",myRank,my_nodename,topo_distances[i],i,rank_nodes[i]);
        print_distances += to_string(topo_distances[i])+" ";
    }
    printf("Distances from rank %d: %s\n",myRank,print_distances.c_str());
    
}

// iterate over the distance matrix and save the ranks of all regions to extra matrices seperated
void filter_distance_matrices(){
    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    topo_region_0 = (uint8_t*)malloc((r_info->comm_size+1) * sizeof(uint8_t));
    topo_region_2 = (uint8_t*)malloc((r_info->comm_size+1) * sizeof(uint8_t));
    topo_region_4 = (uint8_t*)malloc((r_info->comm_size+1) * sizeof(uint8_t));

    ranks_in_r0 = 0;
    ranks_in_r2 = 0;
    ranks_in_r4 = 0;

    for (int i = 0; i<r_info->comm_size; i++){
        if (topo_distances[i]==0){
            topo_region_0[ranks_in_r0]=i;
            ranks_in_r0++;
        }
        else if (topo_distances[i]==2){
            topo_region_2[ranks_in_r2]=i;
            ranks_in_r2++;
        }
        else {
            topo_region_4[ranks_in_r4]=i;
            ranks_in_r4++;
        }
    }
}

#define register_callback_t(name, type)                                         \
do{                                                                             \
    type f_##name = &on_##name;                                                 \
    if (cham_t_set_callback(name, (cham_t_callback_t)f_##name) == cham_t_set_never)   \
        printf("0: Could not register callback '" #name "'\n");                 \
} while(0)

#define register_callback(name) register_callback_t(name, name##_t)

int cham_t_initialize(
    cham_t_function_lookup_t lookup,
    cham_t_data_t *tool_data)
{
    cham_t_set_callback             = (cham_t_set_callback_t)               lookup("cham_t_set_callback");
    cham_t_get_callback             = (cham_t_get_callback_t)               lookup("cham_t_get_callback");
    cham_t_get_rank_data            = (cham_t_get_rank_data_t)              lookup("cham_t_get_rank_data");
    cham_t_get_thread_data          = (cham_t_get_thread_data_t)            lookup("cham_t_get_thread_data");
    cham_t_get_rank_info            = (cham_t_get_rank_info_t)              lookup("cham_t_get_rank_info");
    cham_t_get_task_param_info      = (cham_t_get_task_param_info_t)        lookup("cham_t_get_task_param_info");
    cham_t_get_task_param_info_by_id= (cham_t_get_task_param_info_by_id_t)  lookup("cham_t_get_task_param_info_by_id");
    cham_t_get_task_data            = (cham_t_get_task_data_t)              lookup("cham_t_get_task_data");

    // cham_t_get_unique_id = (cham_t_get_unique_id_t) lookup("cham_t_get_unique_id");
    // cham_t_get_num_procs = (cham_t_get_num_procs_t) lookup("cham_t_get_num_procs");

    // cham_t_get_state = (cham_t_get_state_t) lookup("cham_t_get_state");
    // cham_t_get_thread_data = (cham_t_get_thread_data_t) lookup("cham_t_get_thread_data");
    // cham_t_get_parallel_info = (cham_t_get_parallel_info_t) lookup("cham_t_get_parallel_info");
    // cham_t_get_num_places = (cham_t_get_num_places_t) lookup("cham_t_get_num_places");
    // cham_t_get_place_proc_ids = (cham_t_get_place_proc_ids_t) lookup("cham_t_get_place_proc_ids");
    // cham_t_get_place_num = (cham_t_get_place_num_t) lookup("cham_t_get_place_num");
    // cham_t_get_partition_place_nums = (cham_t_get_partition_place_nums_t) lookup("cham_t_get_partition_place_nums");
    // cham_t_get_proc_id = (cham_t_get_proc_id_t) lookup("cham_t_get_proc_id");
    // cham_t_enumerate_states = (cham_t_enumerate_states_t) lookup("cham_t_enumerate_states");
    // cham_t_enumerate_mutex_impls = (cham_t_enumerate_mutex_impls_t) lookup("cham_t_enumerate_mutex_impls");

    register_callback(cham_t_callback_thread_init);
    // register_callback(cham_t_callback_thread_finalize);
    // register_callback(cham_t_callback_task_create);
    // register_callback(cham_t_callback_task_schedule);
    // register_callback(cham_t_callback_encode_task_tool_data);
    // register_callback(cham_t_callback_decode_task_tool_data);
    // register_callback(cham_t_callback_sync_region);
    // register_callback(cham_t_callback_determine_local_load);

    // Priority is cham_t_callback_select_tasks_for_migration (fine-grained)
    // if not registered cham_t_callback_select_num_tasks_to_offload is used (coarse-grained)
    // register_callback(cham_t_callback_select_tasks_for_migration);
    register_callback(cham_t_callback_select_num_tasks_to_offload);
 
    // register_callback(cham_t_callback_select_num_tasks_to_replicate);

    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    cham_t_data_t * r_data      = cham_t_get_rank_data();
    r_data->value               = r_info->comm_rank;

    // compute topology distance matrix only with one thread per rank
    if (!topo_distances_initialized){
        comp_topology_mutex.lock();
        if (topo_distances_initialized){
            comp_topology_mutex.unlock();
        }
        else{
            // check whether MPI is initialized, otherwise do so
            int initialized, err;
            initialized = 0;
            err = MPI_Initialized(&initialized);
            if(!initialized) {
                int provided;
                MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
            }
            MPI_Comm_dup(MPI_COMM_WORLD, &topotool_comm);
            getEnvironment();
            compute_distance_matrix();
            filter_distance_matrices();
            topo_distances_initialized = true;
            comp_topology_mutex.unlock();
        }

    }

    return 1; //success
}

void cham_t_finalize(cham_t_data_t *tool_data)
{
    printf("0: cham_t_event_runtime_shutdown\n");
}

#ifdef __cplusplus
extern "C" {
#endif
cham_t_start_tool_result_t* cham_t_start_tool(unsigned int cham_version)
{
    printf("Starting tool with Chameleon Version: %d\n", cham_version);
    static cham_t_start_tool_result_t cham_t_start_tool_result = {&cham_t_initialize,&cham_t_finalize, 0};
    return &cham_t_start_tool_result;
}
#ifdef __cplusplus
}
#endif
