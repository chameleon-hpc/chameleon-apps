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

#include <mutex>

#include <vector>
#include <tuple>
#include <algorithm>

#include <ctime>
#include <chrono>
#include <iomanip>
#include <limits>

#ifdef TRACE
#include "VT.h"
static int event_tool_task_create = -1;
static int event_tool_task_exec = -1;
#endif

static cham_t_set_callback_t cham_t_set_callback;
static cham_t_get_callback_t cham_t_get_callback;
static cham_t_get_rank_data_t cham_t_get_rank_data;
static cham_t_get_thread_data_t cham_t_get_thread_data;
static cham_t_get_rank_info_t cham_t_get_rank_info;
static cham_t_get_task_param_info_t cham_t_get_task_param_info;
static cham_t_get_task_param_info_by_id_t cham_t_get_task_param_info_by_id;
static cham_t_get_task_data_t cham_t_get_task_data;

// @jusch static variables

static std::mutex rv, av;
static std::vector<long double> runtimes_v;
// true = TO, false = FROM
static std::vector<std::tuple<int, std::vector < long double>, std::vector<long double>, long double>>
args_v;

typedef struct my_task_log_t {
    std::chrono::time_point <std::chrono::high_resolution_clock> time;
} my_task_log_t;


// @jusch functionalities

// source: https://stackoverflow.com/questions/997946/how-to-get-current-time-and-date-in-c
const std::string currentTime() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%X", &tstruct);
    return buf;
}


static void
on_cham_t_callback_task_create(
        cham_migratable_task_t *task,
        cham_t_data_t *task_data,
        const void *codeptr_ra) {
#ifdef TRACE
    if(event_tool_task_create == -1) {
        const char *event_tool_task_create_name = "tool_task_create";
        int ierr = VT_funcdef(event_tool_task_create_name, VT_NOCLASS, &event_tool_task_create);
    }
    VT_begin(event_tool_task_create);
#endif
    TYPE_TASK_ID internal_task_id = chameleon_get_task_id(task);

    // late equipment of task with annotations
    chameleon_annotations_t *ann = chameleon_get_task_annotations_opaque(task);
    if (!ann) {
        ann = chameleon_create_annotation_container();
        chameleon_set_task_annotations(task, ann);
    }
    chameleon_set_annotation_int(ann, "TID", (int) internal_task_id);

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
        cham_t_data_t *prior_task_data) {
#ifdef TRACE
    if(cham_t_task_start == schedule_type) {
        if(event_tool_task_exec == -1) {
            const char *event_tool_task_exec_name = "tool_task_exec";
            int ierr = VT_funcdef(event_tool_task_exec_name, VT_NOCLASS, &event_tool_task_exec);
        }
        VT_begin(event_tool_task_exec);
    }
#endif
    TYPE_TASK_ID internal_task_id = chameleon_get_task_id(task);

    char val_task_flag[50];
    char val_prior_task_flag[50];
    cham_t_task_flag_t_value(task_flag, val_task_flag);
    cham_t_task_flag_t_value(prior_task_flag, val_prior_task_flag);

    // validate that every task (also migrated tasks) keeps annotations
    //chameleon_annotations_t* ann = chameleon_get_task_annotations_opaque(task);
    //int tmp_validation_id;
    //int found = chameleon_get_annotation_int(ann, "TID", &tmp_validation_id);
    //assert(found == 1 && tmp_validation_id == (int)internal_task_id);

    if (schedule_type == cham_t_task_start) {
        my_task_log_t *curr_task_log = (my_task_log_t *) malloc(sizeof(my_task_log_t));
        curr_task_log->time = std::chrono::high_resolution_clock::now();
        task_data->ptr = (void *) curr_task_log;

    } else {
        my_task_log_t *cur_task_data = (my_task_log_t *) task_data->ptr;

        auto beginn_time = cur_task_data->time;
        auto end_time = std::chrono::high_resolution_clock::now();

        rv.lock();
        runtimes_v.push_back(
                double(std::chrono::duration<long double, std::milli>(
                        end_time - beginn_time).count()));
        rv.unlock();

        cham_t_task_param_info_t p_info = cham_t_get_task_param_info(task);
        std::vector<long double> inSizes, outSizes;
        long double overallSize = 0;

        for (int i = 0; i < p_info.num_args; ++i) {

            overallSize += (long double) p_info.arg_sizes[i] /1000;

            if ((p_info.arg_types[i] & 1) == 1)
                inSizes.push_back((long double) p_info.arg_sizes[i]/1000);

            if ((p_info.arg_types[i] & 2) == 2)
                outSizes.push_back((long double) p_info.arg_sizes[i]/1000);

        }

        av.lock();
        args_v.push_back(std::tuple < int, std::vector < long double > , std::vector < long double > , long double >
                                                                                             {(int) p_info.num_args,
                                                                                              inSizes, outSizes,
                                                                                              overallSize});
        av.unlock();

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
        cham_t_data_t *tool_data) {
    cham_t_set_callback = (cham_t_set_callback_t) lookup("cham_t_set_callback");
    cham_t_get_callback = (cham_t_get_callback_t) lookup("cham_t_get_callback");
    cham_t_get_rank_data = (cham_t_get_rank_data_t) lookup("cham_t_get_rank_data");
    cham_t_get_thread_data = (cham_t_get_thread_data_t) lookup("cham_t_get_thread_data");
    cham_t_get_rank_info = (cham_t_get_rank_info_t) lookup("cham_t_get_rank_info");
    cham_t_get_task_param_info = (cham_t_get_task_param_info_t) lookup("cham_t_get_task_param_info");
    cham_t_get_task_param_info_by_id = (cham_t_get_task_param_info_by_id_t) lookup("cham_t_get_task_param_info_by_id");
    cham_t_get_task_data = (cham_t_get_task_data_t) lookup("cham_t_get_task_data");


    register_callback(cham_t_callback_task_create);
    register_callback(cham_t_callback_task_schedule);


    cham_t_rank_info_t *r_info = cham_t_get_rank_info();
    cham_t_data_t *r_data = cham_t_get_rank_data();
    r_data->value = r_info->comm_rank;
    return 1; //success
}

void cham_t_finalize(cham_t_data_t *tool_data) {
    cham_t_data_t *rank_data = cham_t_get_rank_data();

    //std::sort(runtimes_v.begin(), runtimes_v.end());
    //std::sort(args_v.begin(), args_v.end());

    std::vector<long double> inSize;
    std::vector<long double> outSize;

    int numberOfTasks;
    long double tmp;
    long double runtimeMin = std::numeric_limits<long double>::max(), runtimeMax = std::numeric_limits<long double>::min(), runtimeOverall = 0;
    long double inputMin = std::numeric_limits<long double>::max(), inputMax = std::numeric_limits<long double>::min(), inputOverall = 0;
    long double outputMin = std::numeric_limits<long double>::max(), outputMax = std::numeric_limits<long double>::min(), outputOverall = 0;
    long double runtimeMean, inputMean, outputMean;

    double numberOfArgsInput = 0, numberOfArgsOutput = 0;

    numberOfTasks = runtimes_v.size();

    for (int i = 0; i < runtimes_v.size(); ++i) {
        inSize = std::get<1>(args_v.at(i));
        outSize = std::get<2>(args_v.at(i));

        tmp = runtimes_v.at(i);
        if (tmp < runtimeMin)
            runtimeMin = tmp;
        else if (tmp > runtimeMax)
            runtimeMax = tmp;
        runtimeOverall += tmp;

        for (int j = 0; j < inSize.size(); j++) {
            tmp = inSize.at(j);
            if (tmp < inputMin)
                inputMin = tmp;
            else if (tmp > inputMax)
                inputMax = tmp;
            inputOverall += tmp;
            numberOfArgsInput++;
        }

        for (int j = 0; j < outSize.size(); j++) {
            tmp = inSize.at(j);
            if (tmp < outputMin)
                outputMin = tmp;
            else if (tmp > outputMax)
                outputMax = tmp;
            outputOverall += tmp;
            numberOfArgsOutput++;
        }

        inSize.clear();
        outSize.clear();
    }

    runtimeMean = (long double) runtimeOverall / numberOfTasks;
    inputMean = (long double) inputOverall / numberOfArgsInput;
    outputMean = (long double) outputOverall / numberOfArgsOutput;

    // Write to file
    // TODO: possible this section creates multiple files if seconds overlap (not likely)
    std::string path = "/rwthfs/rz/cluster/home/ey186093/output/runtime_R" + std::to_string(rank_data->value) + '_' +
                       currentTime() + ".csv";
    std::ofstream file(path, std::ios_base::app);

    if(rank_data->value == 0){
        std::string config = "/rwthfs/rz/cluster/home/ey186093/output/HEAD.csv";
        std::ofstream configFile(config, std::ios_base::app);

        configFile << "RUNTIME_ms;RMIN;RMAX;RMEAN;#ARGS;#IARGS;";
        for (int i = 0; i < std::get<1>(args_v.at(1)).size(); ++i)
            configFile << "SIARGS_" << i <<";";

        configFile << "SIMIN;SIMAX;SIMEAN;SIOVER;#OARGS;";

        for (int i = 0; i < std::get<2>(args_v.at(1)).size(); ++i)
            configFile << "SOARGS_" << i << ";";

        configFile << "SOMIN;SOMAX;SOMEAN;SOVER;OVSITA;#NOTA;ROVER\n";

        configFile.close();
    }


    for (int i = 0; i < runtimes_v.size(); ++i) {
        inSize = std::get<1>(args_v.at(i));
        outSize = std::get<2>(args_v.at(i));

        // runtime, min, max, mean
        file << std::scientific << runtimes_v.at(i) << ";" << runtimeMin << ";" << runtimeMax << ";" << runtimeMean << ";";

        // write #arguments, #input arguments, size of input arguments, min, max, sum, mean, overall
        file << std::scientific << std::get<0>(args_v.at(i)) << ";" << inSize.size() << ";";
        for (int j = 0; j < inSize.size(); j++)
            file << std::scientific   << inSize.at(j) << ";";
        file << std::scientific << inputMin << ";" << inputMax << ";" << inputMean << ";" << inputOverall << ";";

        // write #outout arguments, size of output arguments , min, max, sum, mean, overall
        file << std::scientific << outSize.size() << ";";
        for (int j = 0; j < outSize.size(); j++)
            file << std::scientific <<outSize.at(j) << ";";
        file << std::scientific << outputMin << ";" << outputMax << ";" << outputMean << ";" << outputOverall << ";";

        // write overall size of all arguments
        file << std::scientific << std::get<3>(args_v.at(i)) << ";";

        // #nodeTasks, runtimeOverall
        file << std::scientific << numberOfTasks << ";" << runtimeOverall << "\n";

        inSize.clear();
        outSize.clear();
    }
    file.close();


    printf("0: cham_t_event_runtime_shutdown\n");
}

#ifdef __cplusplus
extern "C" {
#endif

cham_t_start_tool_result_t *cham_t_start_tool(unsigned int cham_version) {
    printf("Starting tool with Chameleon Version: %d\n", cham_version);
    static cham_t_start_tool_result_t cham_t_start_tool_result = {&cham_t_initialize, &cham_t_finalize, 0};
    return &cham_t_start_tool_result;
}

#ifdef __cplusplus
}
#endif