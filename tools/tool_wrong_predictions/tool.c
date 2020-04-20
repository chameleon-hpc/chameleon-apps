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
static cham_t_get_task_data_t cham_t_get_task_data;

// sample struct to save some information for a task
typedef struct my_task_data_t {
    TYPE_TASK_ID task_id;
    size_t size_data;
    double * sample_data;    
} my_task_data_t;

// static cham_t_get_state_t cham_t_get_state;
// static cham_t_get_task_info_t cham_t_get_task_info;
// static cham_t_get_thread_data_t cham_t_get_thread_data;
// static cham_t_get_parallel_info_t cham_t_get_parallel_info;
// static cham_t_get_unique_id_t cham_t_get_unique_id;
// static cham_t_get_num_procs_t cham_t_get_num_procs;
// static cham_t_get_num_places_t cham_t_get_num_places;
// static cham_t_get_place_proc_ids_t cham_t_get_place_proc_ids;
// static cham_t_get_place_num_t cham_t_get_place_num;
// static cham_t_get_partition_place_nums_t cham_t_get_partition_place_nums;
// static cham_t_get_proc_id_t cham_t_get_proc_id;
// static cham_t_enumerate_states_t cham_t_enumerate_states;
// static cham_t_enumerate_mutex_impls_t cham_t_enumerate_mutex_impls;

// static void print_ids(int level)
// {
//   int task_type, thread_num;
//   cham_t_frame_t *frame;
//   cham_t_data_t *task_parallel_data;
//   cham_t_data_t *task_data;
//   int exists_task = cham_t_get_task_info(level, &task_type, &task_data, &frame,
//                                        &task_parallel_data, &thread_num);
//   char buffer[2048];
//   format_task_type(task_type, buffer);
//   if (frame)
//     printf("%" PRIu64 ": task level %d: parallel_id=%" PRIu64
//            ", task_id=%" PRIu64 ", exit_frame=%p, reenter_frame=%p, "
//            "task_type=%s=%d, thread_num=%d\n",
//            cham_t_get_thread_data()->value, level,
//            exists_task ? task_parallel_data->value : 0,
//            exists_task ? task_data->value : 0, frame->exit_frame.ptr,
//            frame->enter_frame.ptr, buffer, task_type, thread_num);
// }

static void
on_cham_t_callback_thread_init(
    cham_t_data_t *thread_data)
{
    thread_data->value = syscall(SYS_gettid);
}

static void
on_cham_t_callback_thread_finalize(
    cham_t_data_t *thread_data)
{
    thread_data->value = syscall(SYS_gettid);
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
    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    
    
    if(r_info->comm_rank==0) {
      return 4000*(1+num_tasks_local+num_tasks_local_rep+num_tasks_stolen);
    } 
    else {
      return (num_tasks_local+num_tasks_local_rep+num_tasks_stolen);
    }
}

int compare( const void *pa, const void *pb )
{
    const int *a = pa;
    const int *b = pb;
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
   
    //printf("rank 0: %d\n", load_info_per_rank[0]);
    //printf("rank 1: %d\n", load_info_per_rank[1]);

    static double min_abs_imbalance_before_migration = -1;
    if(min_abs_imbalance_before_migration == -1) {
        // try to load it once
        char *min_abs_balance = getenv("MIN_ABS_LOAD_IMBALANCE_BEFORE_MIGRATION");
        if(min_abs_balance) {
            min_abs_imbalance_before_migration = atof(min_abs_balance);
        } else {
            min_abs_imbalance_before_migration = 2;
        }
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
                num_tasks_to_offload_per_rank[other_idx] = 1;
            }
            /*if( num_tasks_to_offload_per_rank[0] ) {
                printf("%d to rank 0 = %d\n", r_info->comm_rank, num_tasks_to_offload_per_rank[0]);
                printf("rank 0 load = %d\n", load_info_per_rank[0]);
                printf("rank 1 load = %d\n", load_info_per_rank[1]);
            }*/
        }
    }
}

static cham_t_replication_info_t*
on_cham_t_callback_select_num_tasks_to_replicate(
    const int32_t* load_info_per_rank,
    int32_t num_tasks_local,
    int32_t *num_replication_infos)
{
    
    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    int myLeft = r_info->comm_rank-1;
    int myRight = r_info->comm_rank+1;

    int neighbours = 0;

    if(myLeft>=0) neighbours++;
    if(myRight<r_info->comm_size) neighbours++;
    
    double alpha = 0.0;
    char* tmp;
    tmp = getenv("MAX_PERCENTAGE_REPLICATED_TASKS");
    if(tmp) {
        alpha = atof(tmp);
    } 
    else 
        alpha = 0.1;

    alpha = alpha/neighbours;

    cham_t_replication_info_t *replication_infos = (cham_t_replication_info_t*) malloc(sizeof(cham_t_replication_info_t)*neighbours);
    int32_t cnt = 0;

    if(myLeft>=0) {
        int num_tasks = num_tasks_local*alpha;
        int *replication_ranks = (int*) malloc(sizeof(int)*1);
	replication_ranks[0] = myLeft;
	cham_t_replication_info_t info = cham_t_replication_info_create(num_tasks, 1, replication_ranks);
	replication_infos[cnt++] = info;
    }
    if(myRight<r_info->comm_size) {
	int num_tasks = num_tasks_local*alpha;
	int *replication_ranks = (int*) malloc(sizeof(int)*1);
	replication_ranks[0] = myRight;
	cham_t_replication_info_t info = cham_t_replication_info_create(num_tasks, 1, replication_ranks);
	replication_infos[cnt++] = info;
    }
    *num_replication_infos = cnt;
    return replication_infos;
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
    cham_t_set_callback = (cham_t_set_callback_t) lookup("cham_t_set_callback");
    cham_t_get_callback = (cham_t_get_callback_t) lookup("cham_t_get_callback");
    cham_t_get_rank_data = (cham_t_get_rank_data_t) lookup("cham_t_get_rank_data");
    cham_t_get_thread_data = (cham_t_get_thread_data_t) lookup("cham_t_get_thread_data");
    cham_t_get_rank_info = (cham_t_get_rank_info_t) lookup("cham_t_get_rank_info");
    cham_t_get_task_data = (cham_t_get_task_data_t) lookup("cham_t_get_task_data");

    // cham_t_get_unique_id = (cham_t_get_unique_id_t) lookup("cham_t_get_unique_id");
    // cham_t_get_num_procs = (cham_t_get_num_procs_t) lookup("cham_t_get_num_procs");

    // cham_t_get_state = (cham_t_get_state_t) lookup("cham_t_get_state");
    // cham_t_get_task_info = (cham_t_get_task_info_t) lookup("cham_t_get_task_info");
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
    register_callback(cham_t_callback_thread_finalize);
    register_callback(cham_t_callback_determine_local_load);

    // Priority is cham_t_callback_select_tasks_for_migration (fine-grained)
    // if not registered cham_t_callback_select_num_tasks_to_offload is used (coarse-grained)
    // register_callback(cham_t_callback_select_tasks_for_migration);
    register_callback(cham_t_callback_select_num_tasks_to_offload);
    register_callback(cham_t_callback_select_num_tasks_to_replicate);

    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    cham_t_data_t * r_data      = cham_t_get_rank_data();
    r_data->value               = r_info->comm_rank;

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
