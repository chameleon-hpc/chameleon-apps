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
    printf("on_cham_t_callback_thread_init ==> thread_id=%d\n", thread_data->value);
}

static void
on_cham_t_callback_thread_finalize(
    cham_t_data_t *thread_data)
{
    thread_data->value = syscall(SYS_gettid);
    printf("on_cham_t_callback_thread_finalize ==> thread_id=%d\n", thread_data->value);
}

static void
on_cham_t_callback_task_create(
    cham_migratable_task_t * task,
    cham_t_data_t *task_data,
    const void *codeptr_ra)
{
    TYPE_TASK_ID internal_task_id        = chameleon_get_task_id(task);

    // create custom data structure and use task_data as pointer
    my_task_data_t * cur_task_data  = (my_task_data_t*) malloc(sizeof(my_task_data_t));
    cur_task_data->task_id          = internal_task_id;
    cur_task_data->size_data        = TASK_TOOL_SAMPLE_DATA_SIZE;
    cur_task_data->sample_data      = (double*) malloc(cur_task_data->size_data * sizeof(double));
    int i;
    for(i = 0; i < cur_task_data->size_data; i++) {
        cur_task_data->sample_data[i] = internal_task_id;
    }    
    task_data->ptr                  = (void*) cur_task_data;

    // late equipment of task with annotations
    chameleon_annotations_t* ann = chameleon_get_task_annotations_opaque(task);
    chameleon_set_annotation_int(ann, "TID", (int)internal_task_id);
    
    // access data containers for current rank and current thread
    cham_t_data_t * rank_data       = cham_t_get_rank_data();
    cham_t_data_t * thread_data     = cham_t_get_thread_data();

    printf("on_cham_t_callback_task_create ==> task_id=%" PRIu64 ";codeptr_ra=" DPxMOD ";rank_data=%" PRIu64 ";thread_data=%" PRIu64 ";task_data=" DPxMOD "\n", internal_task_id, DPxPTR(codeptr_ra), rank_data->value, thread_data->value, DPxPTR(task_data->ptr));
}

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
    TYPE_TASK_ID internal_task_id    = chameleon_get_task_id(task);

    char val_task_flag[50];
    char val_prior_task_flag[50];
    cham_t_task_flag_t_value(task_flag, val_task_flag);
    cham_t_task_flag_t_value(prior_task_flag, val_prior_task_flag);

    // validate that every task (also migrated tasks) keeps annotations
    chameleon_annotations_t* ann = chameleon_get_task_annotations_opaque(task);
    int tmp_validation_id;
    int found = chameleon_get_annotation_int(ann, "TID", &tmp_validation_id);
    assert(found == 1 && tmp_validation_id == (int)internal_task_id);

    if(prior_task_data) {
        TYPE_TASK_ID prior_internal_task_id    = chameleon_get_task_id(prior_task);
        printf("on_cham_t_callback_task_schedule ==> schedule_type=%s;task_id=%" PRIu64 ";task_flag=%s;task_data=" DPxMOD ";prior_task_id=%" PRIu64 ";prior_task_flag=%s;prior_task_data=" DPxMOD "\n", cham_t_task_schedule_type_t_values[schedule_type], internal_task_id, val_task_flag, DPxPTR(task_data->ptr), prior_internal_task_id, val_prior_task_flag, DPxPTR(prior_task_data->ptr));
    }
    else {
        printf("on_cham_t_callback_task_schedule ==> schedule_type=%s;task_id=%" PRIu64 ";task_flag=%s;task_data=" DPxMOD ";prior_task_id=%" PRIu64 ";prior_task_flag=%s;prior_task_data=" DPxMOD "\n", cham_t_task_schedule_type_t_values[schedule_type], internal_task_id, val_task_flag, DPxPTR(task_data->ptr), 0, val_prior_task_flag, DPxPTR(prior_task_data));
    }

    // verification that task tool data is correct
    my_task_data_t * cur_task_data = (my_task_data_t *) task_data->ptr;
    assert(cur_task_data->task_id == internal_task_id);
    assert(cur_task_data->size_data == TASK_TOOL_SAMPLE_DATA_SIZE);
    int i;
    for(i = 0; i < cur_task_data->size_data; i++) {
        assert(cur_task_data->sample_data[i] == (double)internal_task_id);
    }

    if(schedule_type == cham_t_task_end) {
        // dont need tool data any more ==> clean up
        free(task_data->ptr);
    }
}

static void *
on_cham_t_callback_encode_task_tool_data(
    cham_migratable_task_t *task,
    cham_t_data_t *task_data,    
    int32_t *size) 
{
    TYPE_TASK_ID internal_task_id    = chameleon_get_task_id(task);
    printf("on_cham_t_callback_encode_task_tool_data ==> task_id=%" PRIu64 "\n", internal_task_id);

    // determine size of buffer
    my_task_data_t *cur_task_data = (my_task_data_t *) task_data->ptr;
    *size = 
        sizeof(TYPE_TASK_ID) +   // task_id
        sizeof(size_t) +   // data size
        cur_task_data->size_data * sizeof(double); // data

    // create new buffer
    char * cur_buf = (char*) malloc(*size);
    void * buff_start = (void*)cur_buf;

    // set id
    ((TYPE_TASK_ID *) cur_buf)[0] = cur_task_data->task_id;
    cur_buf += sizeof(TYPE_TASK_ID);

    // set data size
    ((int32_t *) cur_buf)[0] = cur_task_data->size_data;
    cur_buf += sizeof(int32_t);

    // copy data
    memcpy(cur_buf, cur_task_data->sample_data, cur_task_data->size_data * sizeof(double));
    return buff_start;
}

static void 
on_cham_t_callback_decode_task_tool_data(
    cham_migratable_task_t *task,
    cham_t_data_t *task_data,
    void *buffer,
    int32_t size)
{
    TYPE_TASK_ID internal_task_id    = chameleon_get_task_id(task);
    printf("on_cham_t_callback_decode_task_tool_data ==> task_id=%" PRIu64 "\n", internal_task_id);

    my_task_data_t * cur_task_data  = (my_task_data_t*) malloc(sizeof(my_task_data_t));
    char * cur_buf = (char*) buffer;

    // task_id
    cur_task_data->task_id          = ((TYPE_TASK_ID *) cur_buf)[0];
    cur_buf += sizeof(TYPE_TASK_ID);

    // size of sample data
    cur_task_data->size_data        = ((int32_t *) cur_buf)[0];
    cur_buf += sizeof(int32_t);

    // sample data
    cur_task_data->sample_data      = (double*) malloc(cur_task_data->size_data * sizeof(double));
    memcpy(cur_task_data->sample_data, cur_buf, cur_task_data->size_data * sizeof(double));

    // need to set structure pointer again
    task_data->ptr = (void*) cur_task_data;
}

static void 
on_cham_t_callback_sync_region(
    cham_t_sync_region_type_t sync_region_type,
    cham_t_sync_region_status_t sync_region_status,
    cham_t_data_t *thread_data,
    const void *codeptr_ra)
{
    printf("on_cham_t_callback_sync_region ==> thread_id=%" PRIu64 ";region_type=%s;region_status=%s;codeptr_ra=" DPxMOD "\n", thread_data->value, cham_t_sync_region_type_t_values[sync_region_type], cham_t_sync_region_status_t_values[sync_region_status], DPxPTR(codeptr_ra));
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
    printf("on_cham_t_callback_determine_local_load ==> task_ids_local=" DPxMOD ";num_tasks_local=%d;task_ids_stolen=" DPxMOD ";num_tasks_stolen=%d\n", DPxPTR(task_ids_local), num_tasks_local, DPxPTR(task_ids_stolen), num_tasks_stolen);
    return (num_tasks_local+num_tasks_local_rep+num_tasks_stolen);
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
    printf("on_cham_t_callback_select_num_tasks_to_offload ==> comm_rank=%d;comm_size=%d;num_tasks_to_offload_per_rank=" DPxMOD ";load_info_per_rank=" DPxMOD ";num_tasks_local=%d;num_tasks_stolen=%d\n", r_info->comm_rank, r_info->comm_size, DPxPTR(num_tasks_to_offload_per_rank), DPxPTR(load_info_per_rank), num_tasks_local, num_tasks_stolen);

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
            min_rel_imbalance_before_migration = 0.0;
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
                printf("R#%d Migrating\t%d\ttasks to rank:\t%d\tload:\t%f\tload_victim:\t%f\tratio:\t%f\tdiff:\t%f\n", r_info->comm_rank, 1, other_idx, cur_load, other_val, ratio, cur_diff);
                num_tasks_to_offload_per_rank[other_idx] = 1;
            }
        }
    }
}

static cham_t_replication_info_t*
on_cham_t_callback_select_num_tasks_to_replicate(
    const int32_t* load_info_per_rank,
    int32_t num_tasks_local,
    int32_t *num_replication_infos)
{
    double alpha = 0.1;
    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    int myLeft = r_info->comm_rank-1;
    int myRight = r_info->comm_rank+1;

    int neighbours = 0;

    if(myLeft>=0) neighbours++;
    if(myRight<r_info->comm_size) neighbours++;

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

// static cham_t_migration_tupel_t*
// on_cham_t_callback_select_tasks_for_migration(
//     const int32_t* load_info_per_rank,
//     TYPE_TASK_ID* task_ids_local,
//     int32_t num_tasks_local,
//     int32_t num_tasks_stolen,
//     int32_t* num_tuples)
// {
//     cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
//     printf("on_cham_t_callback_select_tasks_for_migration ==> comm_rank=%d;comm_size=%d;load_info_per_rank=" DPxMOD ";task_ids_local=" DPxMOD ";num_tasks_local=%d;num_tasks_stolen=%d\n", r_info->comm_rank, r_info->comm_size, DPxPTR(load_info_per_rank), DPxPTR(task_ids_local), num_tasks_local, num_tasks_stolen);

//     if(num_tasks_local > 0) {
//         TYPE_TASK_ID tmp_id = task_ids_local[0];

//         // query/access task tool data
//         cham_t_data_t *tmp_task_data = cham_t_get_task_data(tmp_id);
//         // task is still present if valid pointer returns here
//         if(tmp_task_data) {
//             my_task_data_t* my_data = (my_task_data_t*)tmp_task_data->ptr;
//             assert(tmp_id == my_data->task_id);
//         }

//         // query/access annotations
//         chameleon_annotations_t* ann = chameleon_get_task_annotations(tmp_id);
//         int tmp_validation_id;
//         int found = chameleon_get_annotation_int(ann, "TID", &tmp_validation_id);
//         assert(found == 1 && tmp_validation_id == (int)tmp_id);
//     }

//     cham_t_migration_tupel_t* task_migration_tuples = NULL;
//     *num_tuples = 0;

//     if(num_tasks_local > 0) {
//         task_migration_tuples = malloc(sizeof(cham_t_migration_tupel_t));
    
//         // Sort rank loads and keep track of indices
//         int tmp_sorted_array[r_info->comm_size][2];
//         int i;
//         for (i = 0; i < r_info->comm_size; i++)
//         {
//             tmp_sorted_array[i][0] = load_info_per_rank[i];
//             tmp_sorted_array[i][1] = i;
//         }

//         // for(i = 0; i < r_info->comm_size;++i)
//         //     printf("%2d, %2d\n", tmp_sorted_array[i][0], tmp_sorted_array[i][1]);

//         qsort(tmp_sorted_array, r_info->comm_size, sizeof tmp_sorted_array[0], compare);

//         // for(i = 0; i < r_info->comm_size;++i)
//         //     printf("%2d, %2d\n", tmp_sorted_array[i][0], tmp_sorted_array[i][1]);

//         int min_val = load_info_per_rank[tmp_sorted_array[0][1]];
//         int max_val = load_info_per_rank[tmp_sorted_array[r_info->comm_size-1][1]];
        
//         int load_this_rank = load_info_per_rank[r_info->comm_rank];
        
//         if(max_val > min_val) {
//             int pos = 0;
//             for(i = 0; i < r_info->comm_size; i++) {
//                 if(tmp_sorted_array[i][1] == r_info->comm_rank) {
//                     pos = i;
//                     break;
//                 }
//             }

//             // only offload if on the upper side
//             if((pos+1) >= ((double)r_info->comm_size/2.0))
//             {
//                 int other_pos = r_info->comm_size-pos;
//                 // need to adapt in case of even number
//                 if(r_info->comm_size % 2 == 0)
//                     other_pos--;
//                 int other_idx = tmp_sorted_array[other_pos][1];
//                 int other_val = load_info_per_rank[other_idx];

//                 // calculate ration between those two and just move if over a certain threshold
//                 double ratio = (double)(load_this_rank-other_val) / (double)load_this_rank;
//                 if(other_val < load_this_rank && ratio > 0.5) {
                    
//                     task_migration_tuples[0].task_id = task_ids_local[0];
//                     task_migration_tuples[0].rank_id = other_idx;
//                     *num_tuples = 1;
//                 }
//             }
//         }

//         if(*num_tuples <= 0)
//         {
//             free(task_migration_tuples);
//             task_migration_tuples = NULL;
//         }
//     }
//     return task_migration_tuples;
// }

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
    register_callback(cham_t_callback_task_create);
    register_callback(cham_t_callback_task_schedule);
    register_callback(cham_t_callback_encode_task_tool_data);
    register_callback(cham_t_callback_decode_task_tool_data);
    register_callback(cham_t_callback_sync_region);
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
