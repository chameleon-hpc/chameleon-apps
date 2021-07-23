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
#include <ctime>

// @jusch includes
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include <mutex>
#include <atomic>

#include <vector>
#include <tuple>
#include <algorithm>

#include <time.h>
#include <iomanip>
#include <limits>

// #include "tool_likwid.h"
#include "likwid_metric_calculation.h"


#define UNIFORM
const int MAX_EVENT_SIZE = 10000;
const int MAX_NUMBER_TASKS = 150;

static cham_t_set_callback_t cham_t_set_callback;
static cham_t_get_callback_t cham_t_get_callback;
static cham_t_get_rank_data_t cham_t_get_rank_data;
static cham_t_get_thread_data_t cham_t_get_thread_data;
static cham_t_get_rank_info_t cham_t_get_rank_info;
static cham_t_get_task_param_info_t cham_t_get_task_param_info;
static cham_t_get_task_param_info_by_id_t cham_t_get_task_param_info_by_id;
static cham_t_get_task_data_t cham_t_get_task_data;

// @jusch static variables
static std::mutex m;
static std::vector<long double> runtimes_v;
static std::vector<int> id_v;
static std::vector<std::string> events_v;
static std::atomic<int> ID=0;
// true = TO, false = FROM
static std::vector<std::tuple<int, std::vector < long double>, std::vector<long double>, long double>>
args_v;

typedef struct my_task_log_t {
    struct timespec start;
    int id;
} my_task_log_t;

// Likwid Variables
static std::atomic<int> lkw_perfom_is_init = 0;
static std::mutex lkw_mtx_init;
static std::mutex lkw_mtx_output;
static std::mutex lkw_mtx_uncore;
static std::atomic<int> lkw_header_printed = 0;

static std::string estr("");
static std::vector<std::string> enames;
static int n_events = -1;
static int n_threads;
static int* cpu_ids;
static double inverseClock = -1;
static int lkw_grp_id = -1;
static std::vector<std::map<std::string, double>> maps_per_thread;
static std::vector<likwid_metric_info_t> metric_infos = lkw_init_metric_list();

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

double get_seconds(timespec spec) {
    double sec = spec.tv_sec + (double)spec.tv_nsec / 1e9;
    return sec;
}

static void
on_cham_t_callback_thread_init(
    cham_t_data_t *thread_data)
{
    int cur_proc_id = likwid_getProcessorId();
    cpu_ids[omp_get_thread_num()] = cur_proc_id;
}

static void
on_cham_t_callback_task_create(
        cham_migratable_task_t *task,
        cham_t_data_t *task_data,
        const void *codeptr_ra) {

    if(!lkw_perfom_is_init.load()) {
        // need lock here
        lkw_mtx_init.lock();
        // need to check again
        if(!lkw_perfom_is_init.load()) {
            int err;
            err = perfmon_init(n_threads, cpu_ids);
            if (err < 0) 
                fprintf(stderr, "Failed to initialize LIKWID's performance monitoring module\n");
            lkw_grp_id = perfmon_addEventSet(estr.c_str()); // Add eventset string to the perfmon module.
            if (lkw_grp_id < 0)
                fprintf(stderr, "Failed to add event string %s to LIKWID's performance monitoring module\n", estr.c_str());
            err = perfmon_setupCounters(lkw_grp_id);
            if (err < 0)
                fprintf(stderr, "Failed to setup group %d in LIKWID's performance monitoring module\n", lkw_grp_id);

            err = perfmon_startCounters();
            if (err < 0)
                fprintf(stderr, "Failed to start counters for group %d for thread %d\n", lkw_grp_id, (-1*err)-1);

            // mark as done
            lkw_perfom_is_init = 1;
        }
        lkw_mtx_init.unlock();
    }
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

    cham_t_data_t *rank_data = cham_t_get_rank_data();

    int cur_proc_id = likwid_getProcessorId();
    int thread_num  = omp_get_thread_num();
    int err;
    double result;

    if (cham_t_task_start == schedule_type) {

        auto tmpID = ++ID;
        auto tag = "R" + std::to_string(rank_data->value) + "T" + std::to_string(tmpID);
        
        err = perfmon_readCountersCpu(cur_proc_id);
        if (lkw_has_uncore_counters(enames, n_events)) {
            int tmp_first = lkw_find_first_thread_on_socket(thread_num, cpu_ids, n_threads);
            if (tmp_first != thread_num) {
                lkw_mtx_uncore.lock();
                err = perfmon_readCountersCpu(cpu_ids[tmp_first]);
                lkw_mtx_uncore.unlock();
            }
        }
        if (err < 0)
            fprintf(stderr, "Failed to read counters for group %d for thread %d\n", lkw_grp_id, (-1*err)-1);

        maps_per_thread[thread_num].clear();
        for (int j = 0; j < n_events; j++)
        {
            std::string cur_event(enames[j]);
            result = lkw_get_counter_result(lkw_grp_id, j, cur_event, thread_num, cpu_ids, n_threads);
            maps_per_thread[thread_num][cur_event] = result;
        }

        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);

        my_task_log_t *cur_task_data = (my_task_log_t *) malloc(sizeof(my_task_log_t));
        cur_task_data->start = t;
        cur_task_data->id = tmpID;
        task_data->ptr = (void *) cur_task_data;

    } else if (cham_t_task_end == schedule_type) {
        
        struct timespec end_time;
        clock_gettime(CLOCK_MONOTONIC, &end_time);

        my_task_log_t *cur_task_data = (my_task_log_t *) task_data->ptr;
        auto start_time = cur_task_data->start;
        int tmpID = cur_task_data->id;
        auto tag = "R" + std::to_string(rank_data->value) + "T" + std::to_string(tmpID);

        err = perfmon_readCountersCpu(cur_proc_id);
        if (lkw_has_uncore_counters(enames, n_events)) {
            int tmp_first = lkw_find_first_thread_on_socket(thread_num, cpu_ids, n_threads);
            if (tmp_first != thread_num) {
                lkw_mtx_uncore.lock();
                err = perfmon_readCountersCpu(cpu_ids[tmp_first]);
                lkw_mtx_uncore.unlock();
            }
        }
        if (err < 0)
            fprintf(stderr, "Failed to read counters for group %d for thread %d\n", lkw_grp_id, (-1*err)-1);

        for (int j = 0; j < n_events; j++)
        {
            std::string cur_event(enames[j]);
            result = lkw_get_counter_result(lkw_grp_id, j, cur_event, thread_num, cpu_ids, n_threads);
            maps_per_thread[thread_num][cur_event] = lkw_calculate_difference(cur_event, maps_per_thread[thread_num][cur_event], result);
        }
        maps_per_thread[thread_num]["time"] = get_seconds(end_time) - get_seconds(start_time);
        maps_per_thread[thread_num]["inverseClock"] = inverseClock;

        // calculate metrics
        std::vector<likwid_metric_value_t> results = lkw_calculate_metrics(metric_infos, maps_per_thread[thread_num]);

        // ========== DEBUG: Metric output
        std::stringstream ss_vals("");
        std::stringstream ss_header("");

        if(!lkw_header_printed.load()) {
            ss_header << "TaskTag;CPU;Rank;Thread;ExecTime;";
        }
        ss_vals << "Tag#" << tag << ";" << cur_proc_id << ";" << rank_data->value << ";" << thread_num << ";" << maps_per_thread[thread_num]["time"] << ";";
        for (const auto &entry : results) {
            if (!lkw_header_printed.load()) {
                ss_header << entry.metric_name << ";";
            }
            ss_vals << entry.value << ";";
        }
        lkw_mtx_output.lock();
        if (!lkw_header_printed.load()) {
            fprintf(stderr, "%s\n", ss_header.str().c_str());
            lkw_header_printed = 1;
        }
        fprintf(stderr, "%s\n", ss_vals.str().c_str());
        lkw_mtx_output.unlock();
        // ========== DEBUG: Metric output
        
        /*
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

        m.lock();
        events_v.push_back(tag);
        id_v.push_back(tmpID);
        runtimes_v.push_back(double(1000.0*end_time.tv_sec + 1e-6*end_time.tv_nsec - (1000.0*start_time.tv_sec + 1e-6*start_time.tv_nsec)));
        args_v.push_back(std::tuple < int, std::vector < long double > , std::vector < long double > , long double >
                                                                                             {(int) p_info.num_args,
                                                                                              inSizes, outSizes,
                                                                                              overallSize});
        m.unlock();
        */

        // dont need tool data any more ==> clean up
        free(task_data->ptr);
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
    
    timer_init();
    inverseClock = 1.0/timer_getCycleClock();

    // Read out env variable which events should be measured
    estr = std::string("INSTR_RETIRED_ANY:FIXC0,") +
        std::string("CPU_CLK_UNHALTED_CORE:FIXC1,") + 
        std::string("CPU_CLK_UNHALTED_REF:FIXC2,") +
        std::string("TEMP_CORE:TMP0,") +
        std::string("PWR_PKG_ENERGY:PWR0,") +
        std::string("PWR_PP0_ENERGY:PWR1,") +
        std::string("PWR_DRAM_ENERGY:PWR3");

    enames.push_back("INSTR_RETIRED_ANY:FIXC0");
    enames.push_back("CPU_CLK_UNHALTED_CORE:FIXC1"); 
    enames.push_back("CPU_CLK_UNHALTED_REF:FIXC2");
    enames.push_back("TEMP_CORE:TMP0");
    enames.push_back("PWR_PKG_ENERGY:PWR0");
    enames.push_back("PWR_PP0_ENERGY:PWR1");
    enames.push_back("PWR_DRAM_ENERGY:PWR3");

    // estr = std::string("INSTR_RETIRED_ANY:FIXC0,") +
    //     std::string("CPU_CLK_UNHALTED_CORE:FIXC1,") + 
    //     std::string("CPU_CLK_UNHALTED_REF:FIXC2,") +
    //     std::string("UOPS_EXECUTED_USED_CYCLES:PMC0,") +
    //     std::string("UOPS_EXECUTED_STALL_CYCLES:PMC1,") +
    //     std::string("CPU_CLOCK_UNHALTED_TOTAL_CYCLES:PMC2,") +
    //     std::string("UOPS_EXECUTED_STALL_CYCLES:PMC3:EDGEDETECT");

    // enames.push_back("INSTR_RETIRED_ANY:FIXC0");
    // enames.push_back("CPU_CLK_UNHALTED_CORE:FIXC1"); 
    // enames.push_back("CPU_CLK_UNHALTED_REF:FIXC2");
    // enames.push_back("UOPS_EXECUTED_USED_CYCLES:PMC0");
    // enames.push_back("UOPS_EXECUTED_STALL_CYCLES:PMC1");
    // enames.push_back("CPU_CLOCK_UNHALTED_TOTAL_CYCLES:PMC2");
    // enames.push_back("UOPS_EXECUTED_STALL_CYCLES:PMC3:EDGEDETECT");

    // allocate relevant data structures for likwid workaround
    n_events = enames.size();
    n_threads = omp_get_max_threads();
    cpu_ids = (int*)malloc(sizeof(int)*n_threads);
    maps_per_thread.resize(n_threads);

    cham_t_set_callback = (cham_t_set_callback_t) lookup("cham_t_set_callback");
    cham_t_get_callback = (cham_t_get_callback_t) lookup("cham_t_get_callback");
    cham_t_get_rank_data = (cham_t_get_rank_data_t) lookup("cham_t_get_rank_data");
    cham_t_get_thread_data = (cham_t_get_thread_data_t) lookup("cham_t_get_thread_data");
    cham_t_get_rank_info = (cham_t_get_rank_info_t) lookup("cham_t_get_rank_info");
    cham_t_get_task_param_info = (cham_t_get_task_param_info_t) lookup("cham_t_get_task_param_info");
    cham_t_get_task_param_info_by_id = (cham_t_get_task_param_info_by_id_t) lookup("cham_t_get_task_param_info_by_id");
    cham_t_get_task_data = (cham_t_get_task_data_t) lookup("cham_t_get_task_data");

    register_callback(cham_t_callback_thread_init);
    register_callback(cham_t_callback_task_create);
    register_callback(cham_t_callback_task_schedule);

    cham_t_rank_info_t *r_info = cham_t_get_rank_info();
    cham_t_data_t *r_data = cham_t_get_rank_data();
    r_data->value = r_info->comm_rank;

    return 1; //success
}

void cham_t_finalize(cham_t_data_t *tool_data) {
    cham_t_data_t *rank_data = cham_t_get_rank_data();

    /*
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

        std::string config_N = "/rwthfs/rz/cluster/home/ey186093/output/.head/HEAD_R.csv";
        std::ofstream configFile_N(config_N, std::ios_base::app);
#else
        std::string config = "/rwthfs/rz/cluster/home/ey186093/output/.head_NU/HEAD.csv";
        std::ofstream configFile(config, std::ios_base::app);

        std::string config_N = "/rwthfs/rz/cluster/home/ey186093/output/.head_NU/HEAD_R.csv";
        std::ofstream configFile_N(config_N, std::ios_base::app);
#endif
        auto eventNames = getEventnames();
        configFile << "TASK_ID;RUNTIME_ms;SIZE_IN_min;SIZE_IN_max;SIZE_IN_mean;SIZE_IN_overall;SIZE_OUT_min;SIZE_OUT_max;SIZE_OUT_mean;SIZE_OUT_overall;SIZE_overall";

        for (int i = 0; i < eventNames.size(); ++i)
            configFile << ";" << eventNames[i].c_str();

        configFile << "\n";
        configFile_N << "TASK_ID;RUNTIME_min;RUNTIME_max;RUNTIME_mean;#TASKS;RUNTIME_overall\n";

        configFile.close();
        configFile_N.close();

    }

    int nevents;
    double *events;
    double time;
    int count;

    for (int i = 0; i < runtimes_v.size(); ++i) {

        nevents = MAX_EVENT_SIZE;
        events = new double[MAX_EVENT_SIZE];
        time = 0;
        count = 0;

        LIKWID_MARKER_GET(events_v.at(i).c_str(), &nevents, events, &time, &count);

        file << std::scientific << std::to_string(rank_data->value) << 'R' << id_v.at(i) << ";" << runtimes_v.at(i) << ";" << inputMin.at(i) << ";" << inputMax.at(i) << ";"\
        << inputMean.at(i) << ";" << std::to_string(rank_data->value) << inputSize.at(i) << ";" << outputMin.at(i) << ";" << outputMax.at(i) << ";"\
        << outputMean.at(i) << ";" << outputSize.at(i) << ";" << ovSize.at(i);

        for (int j = 0; j < nevents; j++)
            file << ";" << events[j];

        file << "\n";

        file_N << std::scientific << std::to_string(rank_data->value) << 'R' << id_v.at(i) << ";" << runtimeMin << ";" << runtimeMax << ";" << runtimeMean << ";" \
        << runtimes_v.size() << ";" << runtimeOverall << "\n";
    }

    file.close();
    file_N.close();

    LIKWID_MARKER_CLOSE;
    */

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