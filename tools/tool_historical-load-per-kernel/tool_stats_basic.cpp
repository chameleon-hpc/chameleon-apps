#ifndef _BSD_SOURCE
#define _BSD_SOURCE
#endif

#define _DEFAULT_SOURCE
#include <cstdio>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <list>
#include <numeric>
#include <vector>
#include <mutex>
#include <assert.h>

#ifndef USE_LIST
#define USE_LIST 0
#endif

// #ifndef NUM_EXCHANGES_SWITCH
// #define NUM_EXCHANGES_SWITCH 1000
// #endif

#if USE_LIST
#ifndef NUM_LAST_TASKS
#define NUM_LAST_TASKS 20
#endif
#endif

#include "chameleon.h"
#include "chameleon_tools.h"

#include <unistd.h>
#include <sys/syscall.h>
#include <inttypes.h>
#include <assert.h>

static cham_t_set_callback_t cham_t_set_callback;
static cham_t_get_callback_t cham_t_get_callback;
static cham_t_get_rank_data_t cham_t_get_rank_data;
static cham_t_get_thread_data_t cham_t_get_thread_data;
static cham_t_get_rank_info_t cham_t_get_rank_info;
static cham_t_get_task_param_info_t cham_t_get_task_param_info;
static cham_t_get_task_param_info_by_id_t cham_t_get_task_param_info_by_id;
static cham_t_get_task_meta_info_t cham_t_get_task_meta_info;
static cham_t_get_task_meta_info_by_id_t cham_t_get_task_meta_info_by_id;
static cham_t_get_task_data_t cham_t_get_task_data;

static std::mutex mtx;
#if USE_LIST
// consider only last NUM_LAST_TASKS tasks
static std::vector<double> last_task_runtimes(NUM_LAST_TASKS, 0);
static int idx = 0;
static int enough_data  = 0;
#else
// consider all tasks in current sync phase
static double sum_times = 0.0;
static double n_tasks   = 0.0;
#endif
// static long n_exchanges = 0;

static void
on_cham_t_callback_task_schedule(
    cham_migratable_task_t *task,
    cham_t_task_flag_t task_flag,
    cham_t_data_t *task_data,
    cham_t_task_schedule_type_t schedule_type,
    cham_migratable_task_t *prior_task,
    cham_t_task_flag_t prior_task_flag,
    cham_t_data_t *prior_task_data)
{
    if(schedule_type == cham_t_task_start) {
        double start_time = omp_get_wtime();
        memcpy(&(task_data->value), &start_time, sizeof(task_data->value));
    } else if(schedule_type == cham_t_task_end) {
        double elapsed = 0;
        memcpy(&elapsed, &(task_data->value), sizeof(elapsed));
        elapsed = omp_get_wtime() - elapsed;
        
        // add to rolling list
        mtx.lock();
        #if USE_LIST
        last_task_runtimes[idx++] = elapsed;
        if (!enough_data && idx >= NUM_LAST_TASKS) {
            enough_data = 1;
            // fprintf(stderr, "Pass TH at exchange: %lld\n", n_exchanges);
        }
        idx = idx % NUM_LAST_TASKS;
        #else
        sum_times += elapsed;
        n_tasks++;
        #endif        
        mtx.unlock();
    }
}

static void 
on_cham_t_callback_sync_region(
    cham_t_sync_region_type_t sync_region_type,
    cham_t_sync_region_status_t sync_region_status,
    cham_t_data_t *thread_data,
    const void *codeptr_ra)
{
    #if USE_LIST
    enough_data = 0;
    #else
    sum_times = 0.0;
    n_tasks = 0.0;
    #endif
    // n_exchanges = 0;
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
    // if (n_exchanges < NUM_EXCHANGES_SWITCH) {
    //     //printf("on_cham_t_callback_determine_local_load ==> task_ids_local=" DPxMOD ";num_tasks_local=%d;task_ids_stolen=" DPxMOD ";num_tasks_stolen=%d\n", DPxPTR(task_ids_local), num_tasks_local, DPxPTR(task_ids_stolen), num_tasks_stolen);
    //     return (num_tasks_local+num_tasks_local_rep+num_tasks_stolen);
    // } else {
        // determine mean execution time of last n tasks in microseconds
        double mean;
        #if USE_LIST
        // assert (enough_data == 1);
        mean = std::accumulate(last_task_runtimes.begin(), last_task_runtimes.end(), 0.0) / (double)NUM_LAST_TASKS;
        #else
        if (n_tasks == 0) {
            mean = 1.0;
        } else {
            mean = sum_times / n_tasks;
        }
        #endif
        return (int32_t) (1000 * mean * (num_tasks_local + num_tasks_stolen));
    // }
}

int compare( const void *pa, const void *pb )
{
    const int *a = (const int *)pa;
    const int *b = (const int *)pb;
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

    // // increment load exchange phases
    // n_exchanges++;
    // if (n_exchanges == NUM_EXCHANGES_SWITCH) {
    //     fprintf(stderr, "R#%d Switching to new statistics after %lld exchanges - n tasks executed: %f\n", r_info->comm_rank, n_exchanges, n_tasks);
    // }

    static double min_abs_imbalance_before_migration = -1;
    if(min_abs_imbalance_before_migration == -1) {
        // try to load it once
        char *min_abs_balance = getenv("MIN_ABS_LOAD_IMBALANCE_BEFORE_MIGRATION");
        if(min_abs_balance) {
            min_abs_imbalance_before_migration = atof(min_abs_balance);
        } else {
            min_abs_imbalance_before_migration = 2;
        }
        printf("R#%d MIN_ABS_LOAD_IMBALANCE_BEFORE_MIGRATION=%f\n", r_info->comm_rank, min_abs_imbalance_before_migration);
    }

    static double min_rel_imbalance_before_migration = -1;
    if(min_rel_imbalance_before_migration == -1) {
        // try to load it once
        char *min_rel_balance = getenv("MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION");
        if(min_rel_balance) {
            min_rel_imbalance_before_migration = atof(min_rel_balance);
        } else {
            // default relative threshold
            min_rel_imbalance_before_migration = 0.2;
        }
        printf("R#%d MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION=%f\n", r_info->comm_rank, min_rel_imbalance_before_migration);
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
    cham_t_get_task_meta_info       = (cham_t_get_task_meta_info_t)         lookup("cham_t_get_task_meta_info");
    cham_t_get_task_meta_info_by_id = (cham_t_get_task_meta_info_by_id_t)   lookup("cham_t_get_task_meta_info_by_id");
    cham_t_get_task_data            = (cham_t_get_task_data_t)              lookup("cham_t_get_task_data");

    register_callback(cham_t_callback_determine_local_load);
    register_callback(cham_t_callback_select_num_tasks_to_offload);
    register_callback(cham_t_callback_task_schedule);
    register_callback(cham_t_callback_sync_region);

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
