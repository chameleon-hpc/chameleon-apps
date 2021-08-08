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
#include <papi.h>

// ==========================================================
// PAPI-related data structures and functions
// ==========================================================
int         num_papi_events;
int*        papi_events;
long long** thread_event_data_start;
long long** thread_event_data_end;
int*        thread_event_sets;

void handle_error (const char *msg) {
    fprintf(stderr, "PAPI Error: %s\n", msg);
    exit(1);
}

static cham_t_set_callback_t cham_t_set_callback;
static cham_t_get_rank_data_t cham_t_get_rank_data;
static cham_t_get_rank_info_t cham_t_get_rank_info;

static void
on_cham_t_callback_thread_init(
    cham_t_data_t *thread_data)
{
    PAPI_register_thread();

    // initialize data for current thread
    int cur_tid = omp_get_thread_num();
    int* my_set = &(thread_event_sets[16*cur_tid]);
    *my_set = PAPI_NULL;
    thread_event_data_start[cur_tid] = (long long*) malloc(sizeof(long long)*num_papi_events);
    thread_event_data_end[cur_tid] = (long long*) malloc(sizeof(long long)*num_papi_events);

    int perr;
    char errstring[PAPI_MAX_STR_LEN];

    if(PAPI_create_eventset(my_set) != PAPI_OK)
        handle_error("Could create event set");
    if (PAPI_add_events(*my_set, papi_events, num_papi_events) != PAPI_OK)
        handle_error("Could create event set");
    if (PAPI_start(*my_set) != PAPI_OK)
        handle_error("PAPI_start");
}

static void
on_cham_t_callback_post_init_serial(
    cham_t_data_t *thread_data)
{
    // printf("on_cham_t_callback_post_init_serial ==> thread_id=%d\n", thread_data->value);
}

static void
on_cham_t_callback_thread_finalize(
    cham_t_data_t *thread_data)
{
    int cur_tid = omp_get_thread_num();
    int* my_set = &(thread_event_sets[16*cur_tid]);

    if (PAPI_stop(*my_set, NULL) != PAPI_OK)
        handle_error("PAPI_stop");
    if (PAPI_cleanup_eventset(*my_set) != PAPI_OK)
        handle_error("PAPI_cleanup_eventset");
    if (PAPI_destroy_eventset(my_set) != PAPI_OK)
        handle_error("PAPI_destroy_eventset");

    PAPI_unregister_thread();
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
    int cur_tid = omp_get_thread_num();
    int* my_set = &(thread_event_sets[16*cur_tid]);

    if(schedule_type == cham_t_task_start) {

        if (PAPI_accum(*my_set, thread_event_data_start[cur_tid]) != PAPI_OK)
            handle_error("PAPI_accum");

    } else if (schedule_type == cham_t_task_end) {

        if (PAPI_read(*my_set, thread_event_data_end[cur_tid]) != PAPI_OK)
            handle_error("PAPI_read");
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
    cham_t_get_rank_data            = (cham_t_get_rank_data_t)              lookup("cham_t_get_rank_data");
    cham_t_get_rank_info            = (cham_t_get_rank_info_t)              lookup("cham_t_get_rank_info");

    register_callback(cham_t_callback_thread_init);
    register_callback(cham_t_callback_post_init_serial);
    register_callback(cham_t_callback_thread_finalize);
    register_callback(cham_t_callback_task_schedule);

    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    cham_t_data_t * r_data      = cham_t_get_rank_data();
    r_data->value               = r_info->comm_rank;

    // determine events and number of events
    // papi_events = getenv("PAPI_EVENTS");
    papi_events = (int*) malloc(sizeof(int) * 2);
    papi_events[0] = PAPI_TOT_INS;
    papi_events[1] = PAPI_L2_DCM;
    num_papi_events = 2;
    
    thread_event_data_start = (long long**) malloc(sizeof(long long*) * omp_get_max_threads());
    thread_event_data_end   = (long long**) malloc(sizeof(long long*) * omp_get_max_threads());
    thread_event_sets       = (int*) malloc(sizeof(int) * omp_get_max_threads() * 16);

    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
        handle_error("PAPI library init error!\n");

    if (PAPI_thread_init ((unsigned long (*)(void)) (omp_get_thread_num)) != PAPI_OK)
        handle_error("PAPI_thread_init failure\n");

    return 1; //success
}

void cham_t_finalize(cham_t_data_t *tool_data)
{
    printf("0: cham_t_event_runtime_shutdown\n");
    char cur_name[1000];
    cham_t_data_t * r_data      = cham_t_get_rank_data();

    for(int t = 0; t < omp_get_max_threads(); t++) {
        fprintf(stderr, "Printing results for Rank %d and thread %d:\n", r_data->value, t);
        for(int i = 0; i < num_papi_events; i++) {
            PAPI_event_code_to_name(papi_events[i], cur_name);
            fprintf(stderr, "%s = %lld\n", cur_name, thread_event_data_end[t][i]);
        }
    }

    free(thread_event_sets);
    for(int t = 0; t < omp_get_max_threads(); t++) {
        free(thread_event_data_start[t]);
        free(thread_event_data_end[t]);
    }
    free(thread_event_data_start);
    free(thread_event_data_end);
    free(papi_events);
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
