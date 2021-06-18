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

#include <time.h>
#include <iomanip>
#include <limits>

#include "tool_likwid.h"
const int MAX_EVENT_SIZE = 10000;

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

static std::mutex m, mID;
static std::vector<long double> runtimes_v;
static std::vector<int> id_v;
static int ID=0;
// true = TO, false = FROM
static std::vector<std::tuple<int, std::vector < long double>, std::vector<long double>, long double>>
args_v;

typedef struct my_task_log_t {
    struct timespec start;
    int id;
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
    cham_t_data_t *rank_data = cham_t_get_rank_data();

    char val_task_flag[50];
    char val_prior_task_flag[50];
    cham_t_task_flag_t_value(task_flag, val_task_flag);
    cham_t_task_flag_t_value(prior_task_flag, val_prior_task_flag);

    // validate that every task (also migrated tasks) keeps annotations
    //chameleon_annotations_t* ann = chameleon_get_task_annotations_opaque(task);
    //int tmp_validation_id;
    //int found = chameleon_get_annotation_int(ann, "TID", &tmp_validation_id);
    //assert(found == 1 && tmp_validation_id == (int)internal_task_id);

    if (cham_t_task_start == schedule_type) {

        mID.lock();
        int tmpID = ID;
        ID++;
        mID.unlock();
        auto tag = std::to_string(rank_data->value) + "R" + std::to_string(tmpID);
        LIKWID_MARKER_START(tag.c_str());

        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);

        my_task_log_t *cur_task_data = (my_task_log_t *) malloc(sizeof(my_task_log_t));
        cur_task_data->start = t;
        cur_task_data->id = tmpID;
        task_data->ptr = (void *) cur_task_data;


    }
    else if (cham_t_task_end == schedule_type){
        struct timespec end_time;
        clock_gettime(CLOCK_MONOTONIC, &end_time);

        my_task_log_t *cur_task_data = (my_task_log_t *) task_data->ptr;
        auto start_time = cur_task_data->start;
        int tmpID = cur_task_data->id;

        auto tag = std::to_string(rank_data->value) + "R" + std::to_string(tmpID);
        LIKWID_MARKER_STOP(tag.c_str());

        cham_t_task_param_info_t p_info = cham_t_get_task_param_info(task);
        std::vector<long double> inSizes, outSizes;
        long double overallSize = 0;

        for (int i = 0; i < p_info.num_args; ++i) {

            overallSize += (long double) p_info.arg_sizes[i]/1000;

            if ((p_info.arg_types[i] & 1) == 1)
                inSizes.push_back((long double) p_info.arg_sizes[i]/1000);

            if ((p_info.arg_types[i] & 2) == 2)
                outSizes.push_back((long double) p_info.arg_sizes[i]/1000);

        }

        int nevents;
        int count;
        double time;
        double events[MAX_EVENT_SIZE];
        LIKWID_MARKER_GET(tag.c_str(), &nevents, events, &time, &count);

        if(nevents > 0){
            printf("Region tag - %i\n", (int) internal_task_id);
            printf("Region example measures %d events, total measurement time is %f\n", nevents, time);
            printf("The region was called %d times\n", count);
        }
        for (int i = 0; i < nevents; i++)
        {
            printf("Event %d: %f\n", i, events[i]);
        }

        m.lock();
        id_v.push_back(tmpID);
        runtimes_v.push_back(double(1000.0*end_time.tv_sec + 1e-6*end_time.tv_nsec - (1000.0*start_time.tv_sec + 1e-6*start_time.tv_nsec)));
        args_v.push_back(std::tuple < int, std::vector < long double > , std::vector < long double > , long double >
                                                                                                       {(int) p_info.num_args,
                                                                                                        inSizes, outSizes,
                                                                                                        overallSize});
        m.unlock();

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

    LIKWID_MARKER_INIT;

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

    LIKWID_MARKER_CLOSE;

    std::vector<long double> inSize;
    std::vector<long double> outSize;

    long double runtimeMin = std::numeric_limits<long double>::max(), runtimeMax = std::numeric_limits<long double>::min();
    long double runtimeOverall = 0;
    long double runtimeMean;

    std::vector<double> inputSize, outputSize, ovSize;
    std::vector<double> inputMean, outputMean;
    std::vector<double> inputMin, outputMin;
    std::vector<double> inputMax, outputMax;

    long double inSi=0, ouSi=0, ovSi=0;
    long double inMin=0, ouMin=0;
    long double inMax=0, ouMax=0;
    long double val;


    for (int i = 0; i < runtimes_v.size(); ++i) {
        inSize = std::get<1>(args_v.at(i));
        outSize = std::get<2>(args_v.at(i));

        // For each Task: input - min, max, mean,  Overall size in KB, output - min, max, mean, overall size in KB
        for (int j = 0; j < inSize.size(); ++j) {
            val = inSize.at(j);
            inSi += val;
            ovSi += val;

            if (inMin > val)
                inMin = val;
            if (inMax < val)
                inMax = val;
        }
        inputMin.push_back(inMin);
        inputMax.push_back(inMax);
        inputMean.push_back((long double) inSi/inSize.size());
        inputSize.push_back(inSi);

        // For each Task: output - min, max, mean,  Overall size in KB, output - min, max, mean, overall size in KB
        for (int j = 0; j < outSize.size(); ++j) {
            val = outSize.at(j);
            ouSi += val;
            ovSi +=0;

            if (ouMin > val)
                ouMin = val;
            if (ouMax < val)
                ouMax = val;
        }
        outputMin.push_back(ouMin);
        outputMax.push_back(ouMax);
        outputMean.push_back((long double) ouSi/outSize.size());
        outputSize.push_back(ouSi);
        ovSize.push_back((long double) ovSi/(inSize.size() + outSize.size()));

        // clear up all temporary values
        inSi=0;
        inMin=std::numeric_limits<long double>::max();
        inMax=std::numeric_limits<long double>::min();
        ouSi=0;
        ouMin=std::numeric_limits<long double>::max();
        ouMax=std::numeric_limits<long double>::min();


        // calculate node specific runtime - min, max, mean, overall
        val = runtimes_v.at(i);
        if (val < runtimeMin)
            runtimeMin = val;
        else if (val > runtimeMax)
            runtimeMax = val;
        runtimeOverall += val;

        inSize.clear();
        outSize.clear();
    }

    runtimeMean = (long double) runtimeOverall / runtimes_v.size();

    // Write to file
#ifdef UNIFORM
    // TODO: possible this section creates multiple files if seconds overlap (not likely)
    std::string path = "/rwthfs/rz/cluster/home/ey186093/output/.runtime/runtime_R" + std::to_string(rank_data->value) + '_' +
                       currentTime() + ".csv";
    std::ofstream file(path, std::ios_base::app);

    std::string path_N = "/rwthfs/rz/cluster/home/ey186093/output/.rank/rank_R" + std::to_string(rank_data->value) + '_' +
                         currentTime() + ".csv";
    std::ofstream file_N(path_N, std::ios_base::app);
#else
    // TODO: possible this section creates multiple files if seconds overlap (not likely)
    std::string path = "/rwthfs/rz/cluster/home/ey186093/output/.runtime_NU/runtime_R" + std::to_string(rank_data->value) + '_' +
                       currentTime() + ".csv";
    std::ofstream file(path, std::ios_base::app);

    std::string path_N = "/rwthfs/rz/cluster/home/ey186093/output/.rank_NU/rank_R" + std::to_string(rank_data->value) + '_' +
                       currentTime() + ".csv";
    std::ofstream file_N(path_N, std::ios_base::app);
#endif

    // TODO: 'RMIN', 'RMAX', 'RMEAN' not mentioned - maybe collect them until actual task
    // TODO: fix writing data into right files
    // TODO: First changes not tested

    if(rank_data->value == 0){
#ifdef UNIFORM
        std::string config = "/rwthfs/rz/cluster/home/ey186093/output/.head/HEAD.csv";
        std::ofstream configFile(config, std::ios_base::app);

        std::string config_N = "/rwthfs/rz/cluster/home/ey186093/output/.head/HEAD_N.csv";
        std::ofstream configFile_N(config_N, std::ios_base::app);
#else
        std::string config = "/rwthfs/rz/cluster/home/ey186093/output/.head_NU/HEAD.csv";
        std::ofstream configFile(config, std::ios_base::app);

        std::string config_N = "/rwthfs/rz/cluster/home/ey186093/output/.head_NU/HEAD_N.csv";
        std::ofstream configFile_N(config_N, std::ios_base::app);
#endif

        configFile << "TASK_ID;RUNTIME_ms;SIZE_IN_min;SIZE_IN_max;SIZE_IN_mean;SIZE_IN_overall;SIZE_OUT_min;SIZE_OUT_max;SIZE_OUT_mean;SIZE_OUT_overall;SIZE_overall\n";
        configFile_N << "TASK_ID;RUNTIME_min;RUNTIME_max;RUNTIME_mean;#TASKS;RUNTIME_overall\n";

        configFile.close();
        configFile_N.close();

    }

    for (int i = 0; i < runtimes_v.size(); ++i) {
        file << std::scientific << std::to_string(rank_data->value) << 'R' << id_v.at(i) << ";" << runtimes_v.at(i) << ";" << inputMin.at(i) << ";" << inputMax.at(i) << ";"\
        << inputMean.at(i) << ";" << std::to_string(rank_data->value) << inputSize.at(i) << ";" << outputMin.at(i) << ";" << outputMax.at(i) << ";"\
        << outputMean.at(i) << ";" << outputSize.at(i) << ";" << ovSize.at(i)<< "\n";

        file_N << std::scientific << std::to_string(rank_data->value) << 'R' << id_v.at(i) << ";" << runtimeMin << ";" << runtimeMax << ";" << runtimeMean << ";" \
        << runtimes_v.size() << ";" << runtimeOverall << "\n";
    }
    file.close();
    file_N.close();

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