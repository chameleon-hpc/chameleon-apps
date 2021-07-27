#ifndef _BSD_SOURCE
#define _BSD_SOURCE
#endif

#define _DEFAULT_SOURCE
#include <stdio.h>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

// #include "tool.h"
#include <mpi.h>
#include <mutex>

#include "chameleon.h"
#include "chameleon_tools.h"
// #include "commthread.h"

#include <unistd.h>
#include <sys/syscall.h>
#include <inttypes.h>
#include <assert.h>

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

static uint8_t* topo_distances; // distance in hops from the current rank to the rank at the corresponding index (sorted after ranks) 
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

// returns true if the nodes are in the same rack
bool checkSameRack(char* node1, char* node2){

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

    // Sort rank loads and keep track of indices
    int tmp_sorted_array[r_info->comm_size][2];
    int i;
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
    for(int i = 0; i<r_info->comm_size; i++){
        printf("Rank %d thinks rank %d runs on node %s\n",myRank,i,rank_nodes[i]);
    }

    // testing Allgather: working correctly with int
    // int* rank_nodes = (int*)malloc(r_info->comm_size * sizeof(int));
    // MPI_Allgather(
    //     &myRank, //void* send_data,
    //     1, //int send_count,
    //     MPI_INT, //MPI_Datatype send_datatype,
    //     rank_nodes, //void* recv_data,
    //     1, //int recv_count,
    //     MPI_INT, //MPI_Datatype recv_datatype,
    //     topotool_comm //MPI_Comm communicator
    // ); 
    // for(int i = 0; i<r_info->comm_size; i++){
    //     printf("Rank %d has on array position %d the value: %d\n",myRank,i,rank_nodes[i]);
    // }

    // testing if MPI works: yes it does
    // if (myRank == 0){
    //     int testint = 42;
    //     MPI_Send(
    //         &testint, //void* data,
    //         1, //int count,
    //         MPI_INT, // MPI_Datatype datatype,
    //         1, //int destination,
    //         0, //int tag,
    //         topotool_comm //MPI_Comm communicator)
    //     );
    // }
    // else if (myRank == 1){
    //     int testint = 0;
    //     MPI_Recv(
    //         &testint, //void* data,
    //         1, //int count,
    //         MPI_INT, //MPI_Datatype datatype,
    //         0, //int source,
    //         MPI_ANY_TAG, //int tag,
    //         topotool_comm, //MPI_Comm communicator,
    //         MPI_STATUS_IGNORE //MPI_Status* status)
    //     );
    //     printf("Rank %d just received int %d\n",myRank,testint);
    // }

    // testing if MPI works with strings: "nullterminating char arrays" were the problem! Now it works:
    // if (myRank == 0){
    //     MPI_Send(
    //         my_nodename, //void* data,
    //         nodename_length+1, //int count,
    //         MPI_CHAR, // MPI_Datatype datatype,
    //         1, //int destination,
    //         0, //int tag,
    //         topotool_comm //MPI_Comm communicator)
    //     );
    //     printf("Rank 0 sent the string %s.\n",my_nodename);
    // }
    // else if (myRank == 1){
    //     char teststring[nodename_length+1];// = new char [nodename_length+1]();
    //     strncpy(teststring,"1234567",7);
    //     printf("Teststring before receiving anything: %s\n",teststring);
    //     MPI_Recv(
    //         teststring, //void* data,
    //         nodename_length+1, //int count,
    //         MPI_CHAR, //MPI_Datatype datatype,
    //         0, //int source,
    //         MPI_ANY_TAG, //int tag,
    //         topotool_comm, //MPI_Comm communicator,
    //         MPI_STATUS_IGNORE //MPI_Status* status)
    //     );
    //     printf("Rank %d just received string %s\n",myRank,teststring);
    // }

    // topo_distances = (uint8_t*)malloc(r_info->comm_size * sizeof(uint8_t));

    // // Calculate distances to other ranks
    // for (int i = 0; i<r_info->comm_size; i++){
    //     // On the same node
    //     if (strcmp(my_nodename, rank_nodes[i]) == 0){
    //         topo_distances[i]=0;
    //     }
    //     // On the same rack
    //     else if (checkSameRack(my_nodename, rank_nodes[i])){
    //         topo_distances[i]=2;
    //     }
    //     // Not on the same rack
    //     else{
    //         topo_distances[i]=4;
    //     }
    // }
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
    // register_callback(cham_t_callback_select_num_tasks_to_offload);
 
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
            compute_distance_matrix();
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
