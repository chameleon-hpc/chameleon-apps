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

static cham_t_set_callback_t cham_t_set_callback;
static cham_t_get_callback_t cham_t_get_callback;
static cham_t_get_rank_data_t cham_t_get_rank_data;
static cham_t_get_thread_data_t cham_t_get_thread_data;
static cham_t_get_rank_info_t cham_t_get_rank_info;
static cham_t_get_task_data_t cham_t_get_task_data;

static int32_t 
on_cham_t_callback_determine_local_load(
    TYPE_TASK_ID* task_ids_local,
    int32_t num_tasks_local,
    TYPE_TASK_ID* task_ids_stolen,
    int32_t num_tasks_stolen)
{
    int i;
    int sum = 0;
    for(i = 0; i < num_tasks_local; i++)
    {
        chameleon_annotations_t* ann = chameleon_get_task_annotations(task_ids_local[i]);
        if(ann) {
            int num_cells;
            int found = chameleon_get_annotation_int(ann, "num_cells", &num_cells);
            if(found) {
                sum += num_cells;
            }
        }        
    }
    for(i = 0; i < num_tasks_stolen; i++)
    {
        chameleon_annotations_t* ann = chameleon_get_task_annotations(task_ids_stolen[i]);
        if(ann) {
            int num_cells;
            int found = chameleon_get_annotation_int(ann, "num_cells", &num_cells);
            if(found) {
                sum += num_cells;
            }
        }
    }
    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    //printf("R#%d\ton_cham_t_callback_determine_local_load ==> num_tasks_local=%d;num_tasks_stolen=%d;num_cells=%d\n", r_info->comm_rank, num_tasks_local, num_tasks_stolen, sum);
    return sum;
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

    // static double min_abs_imbalance_before_migration = -1;
    // if(min_abs_imbalance_before_migration == -1) {
    //     // try to load it once
    //     char *min_abs_balance = getenv("MIN_ABS_LOAD_IMBALANCE_BEFORE_MIGRATION");
    //     if(min_abs_balance) {
    //         min_abs_imbalance_before_migration = atof(min_abs_balance);
    //     } else {
    //         min_abs_imbalance_before_migration = 2;
    //     }
    //     printf("R#%d MIN_ABS_LOAD_IMBALANCE_BEFORE_MIGRATION=%f\n", r_info->comm_rank, min_abs_imbalance_before_migration);
    // }

    // ==== Calculate initial number of tasks at start of iteration based on environment variables
    static double initial_num_tasks = -1;
    static double omp_n_threads = -1;
    if(omp_n_threads == -1) {
        char *tmp = getenv("OMP_NUM_THREADS");
        if(tmp) {
            omp_n_threads = atof(tmp);
        }
    }
    static double n_tasks_per_threads = -1;
    if(n_tasks_per_threads == -1) {
        char *tmp = getenv("NUM_SECTIONS");
        if(tmp) {
            n_tasks_per_threads = atof(tmp);
            initial_num_tasks = n_tasks_per_threads*omp_n_threads;
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
        printf("R#%d MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION=%f\n", r_info->comm_rank, min_rel_imbalance_before_migration);
    }

    // Define treshold at upper bound for scaling
    double max_rel_imbalance_before_migration = 0.9;
    // Calculate scaled threshold based on initial number of tasks and current number of tasks
    double cur_rel_threshold = min_rel_imbalance_before_migration + (max_rel_imbalance_before_migration-min_rel_imbalance_before_migration) / initial_num_tasks * (initial_num_tasks-num_tasks_local);

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

    double ratio_lb     = 0.0;
    if(max_val > 0) {
        ratio_lb = (max_val - min_val) / max_val;
    }

    double avg_size_section_local = cur_load / (double) (num_tasks_local+num_tasks_stolen);
    double min_abs_imbalance_before_migration = avg_size_section_local*2;

    if((cur_load-min_val) < min_abs_imbalance_before_migration)
        return;
    
    if(ratio_lb >= cur_rel_threshold) {
        int pos = 0;
        for(i = 0; i < r_info->comm_size; i++) {
            if(tmp_sorted_array[i][1] == r_info->comm_rank) {
                pos = i;
                break;
            }
        }

        // only offload if on the upper side
        if((pos+1) >= ((double)r_info->comm_size/2.0))
        {
            int other_pos = r_info->comm_size-pos;
            // need to adapt in case of even number
            if(r_info->comm_size % 2 == 0)
                other_pos--;
            int other_idx = tmp_sorted_array[other_pos][1];
            double other_val = (double) load_info_per_rank[other_idx];

            double cur_diff = cur_load-other_val;
            // check absolute condition
            if(cur_diff < min_abs_imbalance_before_migration)
                return;
            double ratio = cur_diff / (double)cur_load;
            if(other_val < cur_load && ratio >= cur_rel_threshold) {
                // printf("R#%d Migrating\t%d\ttasks to rank:\t%d\tload:\t%f\tload_victim:\t%f\tratio:\t%f\tdiff:\t%f\tmin_abs_threshold:\t%f\tnum_tasks:\t%d\n", r_info->comm_rank, 1, other_idx, cur_load, other_val, ratio, cur_diff, (avg_size_section_local*2), (num_tasks_local+num_tasks_stolen));
                // chameleon_print(1, "ChameleonLib", r_info->comm_rank, "Migrating\t%d\ttasks to rank:\t%d\tload:\t%f\tload_victim:\t%f\tratio:\t%f\tdiff:\t%f\tmin_abs_threshold:\t%f\tnum_tasks:\t%d\n", 1, other_idx, cur_load, other_val, ratio, cur_diff, (avg_size_section_local*2), (num_tasks_local+num_tasks_stolen));
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
    cham_t_set_callback     = (cham_t_set_callback_t) lookup("cham_t_set_callback");
    cham_t_get_callback     = (cham_t_get_callback_t) lookup("cham_t_get_callback");
    cham_t_get_rank_data    = (cham_t_get_rank_data_t) lookup("cham_t_get_rank_data");
    cham_t_get_thread_data  = (cham_t_get_thread_data_t) lookup("cham_t_get_thread_data");
    cham_t_get_rank_info    = (cham_t_get_rank_info_t) lookup("cham_t_get_rank_info");
    cham_t_get_task_data    = (cham_t_get_task_data_t) lookup("cham_t_get_task_data");

    register_callback(cham_t_callback_determine_local_load);
    register_callback(cham_t_callback_select_num_tasks_to_offload);

    // cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    // cham_t_data_t * r_data      = cham_t_get_rank_data();
    // r_data->value               = r_info->comm_rank;

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