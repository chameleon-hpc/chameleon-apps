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

static cham_t_set_callback_t cham_t_set_callback;
static cham_t_get_callback_t cham_t_get_callback;
static cham_t_get_rank_data_t cham_t_get_rank_data;
static cham_t_get_thread_data_t cham_t_get_thread_data;

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
on_cham_t_callback_task_create(
    cham_t_data_t *task_data,
    TargetTaskEntryTy * task)
{
    cham_t_data_t * rank_data       = cham_t_get_rank_data();
    cham_t_data_t * thread_data     = cham_t_get_thread_data();
    printf("on_cham_t_callback_task_create ==> rank_data=%" PRIu64 ";thread_data=%" PRIu64 ";task_data=%" PRIu64 "\n", rank_data->value, thread_data->value, task_data->value);
}

static void
on_cham_t_callback_task_schedule(
    cham_t_data_t *new_task_data,
    cham_t_task_flag_t new_task_flag,
    TargetTaskEntryTy * new_task,
    cham_t_task_schedule_type_t schedule_type,
    cham_t_data_t *prior_task_data,
    cham_t_task_flag_t prior_task_flag,
    TargetTaskEntryTy * prior_task)
{
    char val_new_task_flag[50];
    char val_prior_task_flag[50];
    cham_t_task_flag_t_value(new_task_flag, val_new_task_flag);
    cham_t_task_flag_t_value(prior_task_flag, val_prior_task_flag);

    printf("on_cham_t_callback_task_schedule ==> schedule_type=%s;new_task_data=%" PRIu64 ";new_task_flag=%s;prior_task_data=%" PRIu64 ";prior_task_flag=%s\n", cham_t_task_schedule_type_t_values[schedule_type], new_task_data->value, val_new_task_flag, prior_task_data->value, val_prior_task_flag);
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

//   register_callback(cham_t_callback_mutex_acquire);
//   register_callback_t(cham_t_callback_mutex_acquired, cham_t_callback_mutex_t);
//   register_callback_t(cham_t_callback_mutex_released, cham_t_callback_mutex_t);
//   register_callback(cham_t_callback_nest_lock);
//   register_callback(cham_t_callback_sync_region);
//   register_callback_t(cham_t_callback_sync_region_wait, cham_t_callback_sync_region_t);
//   register_callback(cham_t_callback_control_tool);
//   register_callback(cham_t_callback_flush);
//   register_callback(cham_t_callback_cancel);
//   register_callback(cham_t_callback_implicit_task);
//   register_callback_t(cham_t_callback_lock_init, cham_t_callback_mutex_acquire_t);
//   register_callback_t(cham_t_callback_lock_destroy, cham_t_callback_mutex_t);
//   register_callback(cham_t_callback_work);
//   register_callback(cham_t_callback_master);
//   register_callback(cham_t_callback_parallel_begin);
//   register_callback(cham_t_callback_parallel_end);
//   register_callback(cham_t_callback_task_create);
//   register_callback(cham_t_callback_task_schedule);
//   register_callback(cham_t_callback_dependences);
//   register_callback(cham_t_callback_task_dependence);
//   register_callback(cham_t_callback_thread_begin);

    register_callback(cham_t_callback_task_create);
    register_callback(cham_t_callback_task_schedule);

    printf("0: NULL_POINTER=%p\n", (void*)NULL);
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