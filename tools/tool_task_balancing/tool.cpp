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

// @jusch includes
#include <map>
#include <string>
#include <iostream>
#include <fstream>

#include <ctime>
#include <chrono>
#include <iomanip>

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

// @jusch static variables
static std::map<int, double> runtimes;

// sample struct to save some information for a task
typedef struct my_task_data_t {
    TYPE_TASK_ID task_id;
    size_t size_data;
    double * sample_data;
} my_task_data_t;

typedef struct my_task_log_t {
    std::chrono::time_point<std::chrono::high_resolution_clock> time;
    TYPE_TASK_ID task_id;
} my_task_log_t;


// @jusch functionalities

// source: https://stackoverflow.com/questions/997946/how-to-get-current-time-and-date-in-c
const std::string currentTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%X", &tstruct);
    return buf;
}


static void
on_cham_t_callback_task_create(
        cham_migratable_task_t * task,
        cham_t_data_t *task_data,
        const void *codeptr_ra)
{
#ifdef TRACE
    if(event_tool_task_create == -1) {
        const char *event_tool_task_create_name = "tool_task_create";
        int ierr = VT_funcdef(event_tool_task_create_name, VT_NOCLASS, &event_tool_task_create);
    }
    VT_begin(event_tool_task_create);
#endif
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
    if(!ann) {
        ann = chameleon_create_annotation_container();
        chameleon_set_task_annotations(task, ann);
    }
    chameleon_set_annotation_int(ann, "TID", (int)internal_task_id);

#ifdef TRACE
    VT_end(event_tool_task_create);
#endif
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
#ifdef TRACE
    if(cham_t_task_start == schedule_type) {
        if(event_tool_task_exec == -1) {
            const char *event_tool_task_exec_name = "tool_task_exec";
            int ierr = VT_funcdef(event_tool_task_exec_name, VT_NOCLASS, &event_tool_task_exec);
        }
        VT_begin(event_tool_task_exec);
    }
#endif
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

    if(schedule_type == cham_t_task_start){
        my_task_log_t *curr_task_log = (my_task_log_t*) malloc(sizeof(my_task_log_t));
        curr_task_log->time = std::chrono::high_resolution_clock::now();
        curr_task_log->task_id          = internal_task_id;
        task_data->ptr                  = (void*) curr_task_log;
    }
    else{
        my_task_log_t *cur_task_data = (my_task_log_t *) task_data->ptr;

        auto beginn_time = cur_task_data->time;
        auto end_time = std::chrono::high_resolution_clock::now();
        runtimes[internal_task_id] = double(std::chrono::duration<double, std::milli>(end_time-beginn_time).count());

        // dont need tool data any more ==> clean up
        free(task_data->ptr);
#ifdef TRACE
        VT_end(event_tool_task_exec);
#endif
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
    cham_t_set_callback = (cham_t_set_callback_t) lookup("cham_t_set_callback");
    cham_t_get_callback = (cham_t_get_callback_t) lookup("cham_t_get_callback");
    cham_t_get_rank_data = (cham_t_get_rank_data_t) lookup("cham_t_get_rank_data");
    cham_t_get_thread_data = (cham_t_get_thread_data_t) lookup("cham_t_get_thread_data");
    cham_t_get_rank_info = (cham_t_get_rank_info_t) lookup("cham_t_get_rank_info");
    cham_t_get_task_data = (cham_t_get_task_data_t) lookup("cham_t_get_task_data");


    register_callback(cham_t_callback_task_create);
    register_callback(cham_t_callback_task_schedule);


    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    cham_t_data_t * r_data      = cham_t_get_rank_data();
    r_data->value               = r_info->comm_rank;
    return 1; //success
}

void cham_t_finalize(cham_t_data_t *tool_data)
{
    cham_t_data_t * rank_data       = cham_t_get_rank_data();

    // TODO: possible this section creates multiple files if minutes overlap
    std::string path = "/rwthfs/rz/cluster/home/ey186093/output/runtime_R" + std::to_string(rank_data->value) + '_' + currentTime() + ".txt";
    std::ofstream file (path, std::ios_base::app);

    file << "TASK_ID;RUNTIME_ms \n";
    for(auto it = runtimes.cbegin(); it != runtimes.cend(); ++it)
    {
        file << it->first << ";" << it->second<< "\n";
    }
    file << "\n";
    file.close();

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
