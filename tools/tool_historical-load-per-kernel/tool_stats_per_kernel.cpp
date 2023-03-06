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
#include <map>
#include <unordered_map>
#include <mutex>
#include <assert.h>

#ifndef QUERY_EVERYTIME
#define QUERY_EVERYTIME 0
#endif

#include "chameleon.h"
#include "chameleon_tools.h"

#include <unistd.h>
#include <sys/syscall.h>
#include <inttypes.h>
#include <assert.h>

typedef struct kernel_historical_data_t {
    double sum_times = 0;
    double n_tasks = 0;
    double mean = 0;
} kernel_historical_data_t;

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

static double sum_overall = 0;
static double n_tasks_overall = 0;
static std::mutex mtx;
std::map<double, kernel_historical_data_t*> m_historical;

double get_hash_for_task(cham_migratable_task_t * task) {
    // access to parameter information from task
    cham_t_task_param_info_t p_info = cham_t_get_task_param_info(task);
    cham_t_task_meta_info_t m_info  = cham_t_get_task_meta_info(task);

    // calculate hash for task 
    // (take offset of entry function as base and add literal values to it)
    double hash = (double) m_info.entry_image_offset;
    for(int i = 0; i < p_info.num_args; i++) {
        if ((p_info.arg_types[i] & CHAM_OMP_TGT_MAPTYPE_LITERAL) == CHAM_OMP_TGT_MAPTYPE_LITERAL) {
            int val = (int) (intptr_t) p_info.arg_pointers[i];
            hash = hash + val;
        }
    }
    return hash;
}

double get_hash_for_task_id(TYPE_TASK_ID task_id) {
    // access to parameter information from task
    cham_t_task_param_info_t p_info = cham_t_get_task_param_info_by_id(task_id);
    cham_t_task_meta_info_t m_info  = cham_t_get_task_meta_info_by_id(task_id);

    // calculate hash for task
    // (take offset of entry function as base and add literal values to it)
    double hash = (double) m_info.entry_image_offset;
    for(int i = 0; i < p_info.num_args; i++) {
        if ((p_info.arg_types[i] & CHAM_OMP_TGT_MAPTYPE_LITERAL) == CHAM_OMP_TGT_MAPTYPE_LITERAL) {
            int val = (int) (intptr_t) p_info.arg_pointers[i];
            hash = hash + val;
        }
    }
    return hash;
}

static void
on_cham_t_callback_task_create(
    cham_migratable_task_t * task,
    cham_t_data_t *task_data,
    const void *codeptr_ra)
{
#if QUERY_EVERYTIME == 0
    TYPE_TASK_ID internal_task_id        = chameleon_get_task_id(task);
    // access to parameter information from task
    cham_t_task_param_info_t p_info = cham_t_get_task_param_info(task);
    cham_t_task_meta_info_t m_info  = cham_t_get_task_meta_info(task);

    // calculate hash for task once and save it in task data
    double hash = get_hash_for_task(task);
    memcpy(&(task_data->value), &hash, sizeof(task_data->value));
    //printf("on_cham_t_callback_task_create ==> task_id=%" PRIu64 ";codeptr_ra=" DPxMOD ";entry_ptr=%lld;entry_image_offset=%lld;task_data=" DPxMOD ";num_args=%d;arg_sizes=" DPxMOD ";arg_types=" DPxMOD ";arg_pointers=" DPxMOD ";hash=%f\n", internal_task_id, DPxPTR(codeptr_ra), m_info.entry_ptr, m_info.entry_image_offset, DPxPTR(task_data->ptr), p_info.num_args, DPxPTR(p_info.arg_sizes), DPxPTR(p_info.arg_types), DPxPTR(p_info.arg_pointers), hash);
#endif
}

#if QUERY_EVERYTIME == 0
// Note: Only needed of hash has been saved in task_data and tasks might be migrated to other ranks
static void *
on_cham_t_callback_encode_task_tool_data(
    cham_migratable_task_t *task,
    cham_t_data_t *task_data,    
    int32_t *size) 
{
    // determine size of buffer
    *size = sizeof(double);
    void * cur_buf = malloc(*size);

    double hash;
    memcpy(&hash, &(task_data->value), sizeof(hash));
    ((double *) cur_buf)[0] = hash;
    return cur_buf;
}

static void 
on_cham_t_callback_decode_task_tool_data(
    cham_migratable_task_t *task,
    cham_t_data_t *task_data,
    void *buffer,
    int32_t size)
{
    // get hash again from buffer
    double hash = ((double *)buffer)[0];
    memcpy(&(task_data->value), &hash, sizeof(task_data->value));
}
#endif

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
    cham_t_data_t * thread_data = cham_t_get_thread_data();
    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();

    if(schedule_type == cham_t_task_start) {
        double start_time = omp_get_wtime();
        memcpy(&(thread_data->value), &start_time, sizeof(thread_data->value));
    } else if(schedule_type == cham_t_task_end) {
        double elapsed = 0;
        memcpy(&elapsed, &(thread_data->value), sizeof(elapsed));
        elapsed = omp_get_wtime() - elapsed;

#if QUERY_EVERYTIME == 0
        // retrieve hash from task_data struct
        double hash;
        memcpy(&hash, &(task_data->value), sizeof(hash));
#else
        // on the fly calculation
        double hash = get_hash_for_task(task);
#endif
        
        // increment or add to map
        mtx.lock();
        
        // global counters for all tasks (fallback-fallback)
        sum_overall += elapsed;
        n_tasks_overall++;

        if ( m_historical.find(hash) == m_historical.end() ) {
            fprintf(stderr, "R#%d Hash %f not found in map ==> Creating new entry with elapsed_sec = %f\n", r_info->comm_rank, hash, elapsed);
            kernel_historical_data_t* cur = (kernel_historical_data_t*) malloc(sizeof(kernel_historical_data_t));
            cur->sum_times = elapsed;
            cur->n_tasks = 1;
            cur->mean = elapsed;
            m_historical[hash] = cur;
        } else {
            kernel_historical_data_t* cur = m_historical[hash];
            cur->sum_times += elapsed;
            cur->n_tasks++;
            cur->mean = cur->sum_times / cur->n_tasks;
        }
        mtx.unlock();
    }
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
    std::map<double, int> counts;
    for(int i = 0; i < num_tasks_local; i++) {
        double hash = get_hash_for_task_id(task_ids_local[i]);
        if ( counts.find(hash) == counts.end() ) {
            counts[hash] = 1;
        } else {
            counts[hash]++;
        }
    }
    for(int i = 0; i < num_tasks_stolen; i++) {
        double hash = get_hash_for_task_id(task_ids_stolen[i]);
        if ( counts.find(hash) == counts.end() ) {
            counts[hash] = 1;
        } else {
            counts[hash]++;
        }
    }

    // IMPORTANT: Assumption is that every rank returns a value representing execution time (avoid mixing up time and nr of tasks)
    double overall_sum = 0.0;
    for (auto& it: counts) {
        double cur_mean;
        if ( m_historical.find(it.first) == m_historical.end() ) {
            if(n_tasks_overall == 0) {
                // if no tasks executed so far use dummy value (configurable?)
                // Note: might go very wrong in the first balancing step but should work fine after a while
                cur_mean = 1.0;
            } else {
                // use overall mean as fallback
                cur_mean = sum_overall / n_tasks_overall;
            }
        } else {
            cur_mean = m_historical[it.first]->mean;
        }
        overall_sum += cur_mean * it.second;
    }

    // milliseconds
    return (int32_t) (1000 * overall_sum);
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
    register_callback(cham_t_callback_task_create);
    register_callback(cham_t_callback_task_schedule);
#if QUERY_EVERYTIME == 0
    register_callback(cham_t_callback_encode_task_tool_data);
    register_callback(cham_t_callback_decode_task_tool_data);
#endif

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
