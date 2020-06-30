#ifndef _BSD_SOURCE
#define _BSD_SOURCE
#endif

#define _DEFAULT_SOURCE
#include <stdio.h>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

#include "chameleon.h"
#include "chameleon_tools.h"

#include <unistd.h>
#include <sys/syscall.h>
#include <inttypes.h>
#include <assert.h>

#include <atomic>

#include "tool.h"

#define TASK_TOOL_SAMPLE_DATA_SIZE 10

static cham_t_set_callback_t cham_t_set_callback;
static cham_t_get_callback_t cham_t_get_callback;
static cham_t_get_rank_data_t cham_t_get_rank_data;
static cham_t_get_thread_data_t cham_t_get_thread_data;
static cham_t_get_rank_info_t cham_t_get_rank_info;
static cham_t_get_task_data_t cham_t_get_task_data;
// static cham_t_get_unique_id_t cham_t_get_unique_id;


static std::atomic<int> tasks_processed_sync_cycle (0);
static std::atomic<int> num_noise_generated(0);
static std::atomic<int> acc_noise (0);
static int noise = 0;
static double fraction = 0.5;

//================================================================
// Variables
//================================================================

//================================================================
// Additional functions
//================================================================

int compare( const void *pa, const void *pb ){
    const int *a = (int *) pa;
    const int *b = (int *) pb;
    printf("a[0] = %d, b[0] = %d\n", a[0], b[0]);
    if(a[0] == b[0])
        return a[0] - b[0];
    else
        return a[1] - b[1];
}

//================================================================
// Callback Functions
//================================================================ 
static void
on_cham_t_callback_thread_init(
    cham_t_data_t *thread_data)
{
    int rank_info       = cham_t_get_rank_info()->comm_rank;
    thread_data->value = syscall(SYS_gettid);
}

static void
on_cham_t_callback_thread_finalize(
    cham_t_data_t *thread_data)
{
    int rank_info       = cham_t_get_rank_info()->comm_rank;
    thread_data->value = syscall(SYS_gettid);
}

static void
on_cham_t_callback_task_create(
    cham_migratable_task_t * task,
    cham_t_data_t *task_data,
    const void *codeptr_ra)
{
    TYPE_TASK_ID internal_task_id        = chameleon_get_task_id(task);
}

static void
on_cham_t_callback_task_schedule(
    cham_migratable_task_t * task,                   // opaque data type for internal task
    cham_t_task_flag_t task_flag,
    cham_t_data_t *task_data,
    cham_t_task_schedule_type_t schedule_type,
    cham_migratable_task_t * prior_task,             // opaque data type for internal task
    cham_t_task_flag_t prior_task_flag,
    cham_t_data_t *prior_task_data)
{
    TYPE_TASK_ID task_id = chameleon_get_task_id(task);
    int rank = cham_t_get_rank_info()->comm_rank;

}

static void 
on_cham_t_callback_sync_region(
    cham_t_sync_region_type_t sync_region_type,
    cham_t_sync_region_status_t sync_region_status,
    cham_t_data_t *thread_data,
    const void *codeptr_ra)
{
    tasks_processed_sync_cycle = 0;
    num_noise_generated = 0;
    system("likwid-setFrequencies -f 2.3 -c 0,1");
}


static int32_t
on_cham_t_callback_determine_local_load(
    TYPE_TASK_ID* task_ids_local,
    int32_t num_tasks_local,
    TYPE_TASK_ID* task_ids_local_rep,
    int32_t num_tasks_local_rep,
    TYPE_TASK_ID* task_ids_stolen,
    int32_t num_tasks_stolen,
    TYPE_TASK_ID* task_ids_stolen_rep,
    int32_t num_tasks_stolen_rep)
{
   cham_t_rank_info_t *rank_info  = cham_t_get_rank_info();
   int rank = rank_info->comm_rank;   

   return num_tasks_local + num_tasks_local_rep + num_tasks_stolen;
}

static cham_t_migration_tupel_t*
on_cham_t_callback_select_tasks_for_migration(
    const int32_t* load_info_per_rank,
    TYPE_TASK_ID* task_ids_local,
    int32_t num_tasks_local,
    int32_t num_tasks_stolen,
    int32_t* num_tuples)
{
    cham_t_rank_info_t *rank_info  = cham_t_get_rank_info();
    cham_t_migration_tupel_t* task_migration_tuples = NULL;
    *num_tuples = 0;

    if (num_tasks_local > 0){
        task_migration_tuples = (cham_t_migration_tupel_t *)malloc(sizeof(cham_t_migration_tupel_t));   // allocate mem for the tuple
        // sort loads by rank
        int tmp_sorted_array[rank_info->comm_size][2];
        int i;
        for (i = 0; i < rank_info->comm_size; i++){
            tmp_sorted_array[i][0] = i; // contain rank index
            tmp_sorted_array[i][1] = load_info_per_rank[i]; // contain load per rank
        }

        qsort(tmp_sorted_array, rank_info->comm_size, sizeof tmp_sorted_array[0], compare);


        int min_val = load_info_per_rank[tmp_sorted_array[0][0]];   // load rank 0
        int max_val = load_info_per_rank[tmp_sorted_array[rank_info->comm_size-1][0]];  // load rank 1
        int load_this_rank = load_info_per_rank[rank_info->comm_rank];

        if (max_val > min_val){
            int pos = 0;
            for (i = 0; i < rank_info->comm_size; i++){
                if (tmp_sorted_array[i][0] == rank_info->comm_rank){
                    pos = i;
                    break;
                }
            }

            // only offload if on the upper side
            if((pos+1) >= ((double)rank_info->comm_size/2.0))
            {
                int other_pos = rank_info->comm_size-pos;
                // need to adapt in case of even number
                if(rank_info->comm_size % 2 == 0)
                    other_pos--;
                int other_idx = tmp_sorted_array[other_pos][0];
                int other_val = load_info_per_rank[other_idx];
                // calculate ration between those two and just move if over a certain threshold
                double ratio = (double)(load_this_rank-other_val) / (double)load_this_rank;
                if(other_val < load_this_rank && ratio > 0.5) {
                    double mig_time; TIMESTAMP(mig_time);
                    task_migration_tuples[0].task_id = task_ids_local[0];
                    task_migration_tuples[0].rank_id = other_idx;
                    *num_tuples = 1;
                }
            }
        }

        if(*num_tuples <= 0)
        {
            free(task_migration_tuples);
            task_migration_tuples = NULL;
        }
    }

    return task_migration_tuples;
}

static int32_t
on_cham_t_callback_change_freq_for_execution(
    cham_migratable_task_t * task,
    int32_t load_info_per_rank,
    int32_t total_created_tasks_per_rank
)
{
    int rank = cham_t_get_rank_info()->comm_rank;
    int start_tasks_processed_per_rank = total_created_tasks_per_rank * fraction; // 50% processed load

    if (rank%2==1 && tasks_processed_sync_cycle >= start_tasks_processed_per_rank && num_noise_generated==0){ // && load_info_per_rank != 0){
        //printf("setting clock frequency 1.2\n");
        system("likwid-setFrequencies -f 1.2 -c 0,1");
        //system("cat /proc/cpuinfo");

        num_noise_generated++;
    }
    tasks_processed_sync_cycle++;

    return 0;
}

//================================================================
// Start Tool & Register Callbacks
//================================================================

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
    printf("Calling register_callback...\n");
    cham_t_set_callback = (cham_t_set_callback_t) lookup("cham_t_set_callback");
    // cham_t_get_callback = (cham_t_get_callback_t) lookup("cham_t_get_callback");
    cham_t_get_rank_data = (cham_t_get_rank_data_t) lookup("cham_t_get_rank_data");
    cham_t_get_thread_data = (cham_t_get_thread_data_t) lookup("cham_t_get_thread_data");
    cham_t_get_rank_info = (cham_t_get_rank_info_t) lookup("cham_t_get_rank_info");
    // cham_t_get_task_data = (cham_t_get_task_data_t) lookup("cham_t_get_task_data");

    register_callback(cham_t_callback_thread_init);
    register_callback(cham_t_callback_thread_finalize);
    register_callback(cham_t_callback_task_create);
    register_callback(cham_t_callback_task_schedule);
    // register_callback(cham_t_callback_encode_task_tool_data);
    // register_callback(cham_t_callback_decode_task_tool_data);
    register_callback(cham_t_callback_sync_region);
    register_callback(cham_t_callback_determine_local_load);
    // register_callback(cham_t_callback_select_tasks_for_migration);
    register_callback(cham_t_callback_change_freq_for_execution);

    // Priority is cham_t_callback_select_tasks_for_migration (fine-grained)
    // if not registered cham_t_callback_select_num_tasks_to_offload is used (coarse-grained)
    // register_callback(cham_t_callback_select_num_tasks_to_offload);
    // register_callback(cham_t_callback_select_num_tasks_to_replicate);

    // cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    // cham_t_data_t * r_data      = cham_t_get_rank_data();
    // r_data->value               = r_info->comm_rank;
    char *env_p = getenv("NOISE");

    if(env_p) {
       noise = atoi(env_p);
       printf("Using noise = %d microseconds\n", noise);
    }
  
    env_p = getenv("FRACTION");
    if(env_p) {
       fraction = atof(env_p);
       printf("Using fraction = %f\n", fraction);
    }

    return 1; //success
}

void cham_t_finalize(cham_t_data_t *tool_data)
{
    printf("------------------------- Chameleon Statistics ---------------------\n");
     int rank_info       = cham_t_get_rank_info()->comm_rank;
     printf("R#%d accumulated generated noise = %d\n", rank_info, acc_noise.load());

    system("likwid-setFrequencies -f 2.3 -c 0,1");

    // if (rank_info == 0){
    //     printf("Task ID \t queued_time \t start_time \t mig_time\n");
    //     for (std::list<cham_t_task_info_t*>::iterator it=tool_task_list.task_list.begin(); it!=tool_task_list.task_list.end(); ++it) {
    //         printf("%d \t %.3f \t %.3f \t %.3f\n", (*it)->task_id, (*it)->queue_time, (*it)->start_time, (*it)->mig_time);
    //     }
    // }else{
    //     printf("Task ID \t queued_time \t start_time \t mig_time\n");
    //     for (std::list<cham_t_task_info_t*>::iterator it=tool_task_list.task_list.begin(); it!=tool_task_list.task_list.end(); ++it) {
    //         printf("%d \t %.3f \t %.3f \t %.3f\n", (*it)->task_id, (*it)->queue_time, (*it)->start_time, (*it)->mig_time);
    //     }
    // }
    
}

#ifdef __cplusplus
extern "C" {
#endif
cham_t_start_tool_result_t* cham_t_start_tool(unsigned int cham_version)
{
    printf("Starting tool with Chameleon Version: %d\n", cham_version);

    static cham_t_start_tool_result_t cham_t_start_tool_result = {&cham_t_initialize, &cham_t_finalize, 0};

    return &cham_t_start_tool_result;
}
#ifdef __cplusplus
}
#endif
