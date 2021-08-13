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

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <inttypes.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <sys/syscall.h>
#include <time.h>
#include <tuple>
#include <unistd.h>
#include <vector>

#ifndef USE_LIKWID
#define USE_LIKWID 0
#endif

#ifndef USE_PAPI
#define USE_PAPI 0
#endif

#ifndef USE_PERF
#define USE_PERF 0
#endif

#include "metric_calculation_wrapper/likwid_metric_calculation.h"

#if USE_PAPI
#include "papi.h"
#endif

#if USE_PERF
#include <sys/ioctl.h>
#include <asm/unistd.h>
#endif

static cham_t_set_callback_t cham_t_set_callback;
static cham_t_get_callback_t cham_t_get_callback;
static cham_t_get_rank_data_t cham_t_get_rank_data;
static cham_t_get_thread_data_t cham_t_get_thread_data;
static cham_t_get_rank_info_t cham_t_get_rank_info;
static cham_t_get_task_param_info_t cham_t_get_task_param_info;
static cham_t_get_task_param_info_by_id_t cham_t_get_task_param_info_by_id;
static cham_t_get_task_data_t cham_t_get_task_data;

static std::mutex mtx;
static std::atomic<uint64_t> task_id_ctr = 0;

static std::vector<double>                                      vec_exec_times;
static std::vector <std::string>                                vec_task_tags;
static std::vector <std::vector<std::tuple < double, double>>>  vec_args;
static std::vector <std::map<std::string, double>>              vec_hwc_values;

static std::string path_result_file;
static int output_results = 0;

typedef struct my_task_log_t {
    struct timespec start;
    int id;
} my_task_log_t;

// ==========================================================
// General & partially Likwid-related data structures and functions
// ==========================================================
static std::atomic<int> lkw_perfom_is_init = 0;
static std::mutex lkw_mtx_init;
static std::mutex lkw_mtx_output;
static std::mutex lkw_mtx_uncore;

static int n_events     = -1;
static int n_threads    = -1;
static std::string event_list_string("");
static std::vector <std::string> event_list;
static double inverseClock = -1;
static std::vector <std::map<std::string, double>> maps_per_thread;
static std::vector <likwid_metric_info_t> metric_infos = lkw_init_metric_list();

// ==========================================================
// General & Likwid-related data structures and functions
// ==========================================================
#if USE_LIKWID
static int *cpu_ids;
static int lkw_grp_id = -1;
#endif

// ==========================================================
// PAPI-related data structures and functions
// ==========================================================
#if USE_PAPI
std::mutex mtx_papi;
int papi_events[100];
std::vector<std::string> papi_event_names;
long long** thread_event_data_end;
int*        thread_event_sets;

void handle_PAPI_error(const char *msg, int unlock_mtx = 0) {
    PAPI_perror((char*)msg);
    // fprintf(stderr, "PAPI Error: %s\n", msg);
    if(unlock_mtx) mtx_papi.unlock();
    exit(1);
}
#endif

// ==========================================================
// Perf-related data structures and functions
// ==========================================================
#if USE_PERF
typedef struct read_format_t {
    uint64_t  nr;           /* The number of events */
    struct {
        uint64_t  value;    /* The value of the event */
        uint64_t id;        /* if PERF_FORMAT_ID */
    } values[];
} read_format_t;

std::vector<perf_event_t> perf_event_list;
// file descriptors per thread and event
int **core_fds;
// data buffer per thread
char **data_buffers;
size_t data_buf_size;

int perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags){
    int ret = syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
    return ret;
}
#endif

double get_seconds(timespec spec) {
    double sec = spec.tv_sec + (double) spec.tv_nsec / 1e9;
    return sec;
}

static void
on_cham_t_callback_thread_init(
    cham_t_data_t *thread_data) {

    int cur_tid = omp_get_thread_num();
    cham_t_data_t *rank_data = cham_t_get_rank_data();

#if USE_LIKWID
    int cur_proc_id = likwid_getProcessorId();
    cpu_ids[cur_tid] = cur_proc_id;
#endif

#if USE_PAPI
    PAPI_register_thread();

    // initialize data for current thread
    int retval;
    int* my_set = &(thread_event_sets[16*cur_tid]);
    *my_set = PAPI_NULL;
    thread_event_data_end[cur_tid] = (long long*) malloc(sizeof(long long)*n_events);

    mtx_papi.lock();
    if(retval = PAPI_create_eventset(my_set) != PAPI_OK)
        handle_PAPI_error("Could not create event set", 1);
    if (retval = PAPI_add_events(*my_set, papi_events, n_events) != PAPI_OK) {
        for(int i = 0; i < n_events; i++) {
            fprintf(stderr, "T-IN --Rank R#%d T#%d -- Event Code[%d] = %d -- Err=%d\n", rank_data->value, cur_tid, i, papi_events[i], retval);
        }
        handle_PAPI_error("Could not add events to event set", 1);
    }
    if (retval = PAPI_start(*my_set) != PAPI_OK)
        handle_PAPI_error("PAPI_start", 1);
    mtx_papi.unlock();
#endif

#if USE_PERF
    // init event attr
    struct perf_event_attr pe, upe;
    memset(&pe, 0, sizeof(struct perf_event_attr));
	pe.size = sizeof(struct perf_event_attr);
	pe.disabled = 1;
	pe.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
    pe.exclude_kernel = 1; // might be necessary when paranoid level >= 2 on front-ends
    //pe.exclude_hv = 1;

    // initialize data for current thread
    core_fds[cur_tid] = (int*) malloc(sizeof(int) * n_events);
    data_buffers[cur_tid] = (char*) malloc(sizeof(uint64_t) + 2 * sizeof(uint64_t) * n_events);
    int fd_leader;

    for(int j = 0; j < n_events; j++) {
        pe.type     = perf_event_list[j].type;
        pe.config   = perf_event_list[j].config;

        if(j == 0){
            core_fds[cur_tid][j] = perf_event_open(&pe, 0, -1, -1, 0);
            fd_leader = core_fds[cur_tid][j];
        } else {
            core_fds[cur_tid][j] = perf_event_open(&pe, 0, -1, fd_leader, 0);
        }
        if (core_fds[cur_tid][j] == -1) {
            fprintf(stderr, "Error opening core event: core %d event %d type %d config %x: %s \n", cur_tid, j, pe.type, pe.config, strerror(core_fds[cur_tid][j]));
            exit(EXIT_FAILURE);
        }
    }
    // start and reset counters
    ioctl(core_fds[cur_tid][0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    ioctl(core_fds[cur_tid][0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
#endif
}

static void
on_cham_t_callback_post_init_serial(
    cham_t_data_t *thread_data )
{
#if USE_LIKWID
    if (!lkw_perfom_is_init.load()) {
        // need lock here
        lkw_mtx_init.lock();
        // need to check again
        if (!lkw_perfom_is_init.load()) {
            int err;
            err = perfmon_init(n_threads, cpu_ids);
            if (err < 0)
                fprintf(stderr, "Failed to initialize LIKWID's performance monitoring module\n");
            lkw_grp_id = perfmon_addEventSet(event_list_string.c_str()); // Add eventset string to the perfmon module.
            if (lkw_grp_id < 0)
                fprintf(stderr, "Failed to add event string %s to LIKWID's performance monitoring module\n",
                        event_list_string.c_str());
            err = perfmon_setupCounters(lkw_grp_id);
            if (err < 0)
                fprintf(stderr, "Failed to setup group %d in LIKWID's performance monitoring module\n", lkw_grp_id);

            err = perfmon_startCounters();
            if (err < 0)
                fprintf(stderr, "Failed to start counters for group %d for thread %d\n", lkw_grp_id, (-1 * err) - 1);

            // mark as done
            lkw_perfom_is_init = 1;
        }
        lkw_mtx_init.unlock();
    }
#endif
    timer_init();
    inverseClock = 1.0 / timer_getCycleClock();
}

static void
on_cham_t_callback_thread_finalize(
    cham_t_data_t *thread_data)
{
#if USE_PAPI
    int cur_tid = omp_get_thread_num();
    int* my_set = &(thread_event_sets[16*cur_tid]);

    mtx_papi.lock();
    if (PAPI_stop(*my_set, thread_event_data_end[cur_tid]) != PAPI_OK)
        handle_PAPI_error("PAPI_stop", 1);
    if (PAPI_cleanup_eventset(*my_set) != PAPI_OK)
        handle_PAPI_error("PAPI_cleanup_eventset", 1);
    if (PAPI_destroy_eventset(my_set) != PAPI_OK)
        handle_PAPI_error("PAPI_destroy_eventset", 1);
    PAPI_unregister_thread();
    mtx_papi.unlock();
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

    cham_t_data_t *rank_data = cham_t_get_rank_data();

    int cur_tid = omp_get_thread_num();

#if USE_LIKWID
    int err;
    double result;
    int cur_proc_id = likwid_getProcessorId();
#endif
#if USE_PAPI
    int* my_set = &(thread_event_sets[16*cur_tid]);
#endif

    if (cham_t_task_start == schedule_type) {

        auto tmp_id = ++task_id_ctr;
        auto tag = "R" + std::to_string(rank_data->value) + "T" + std::to_string(tmp_id);

#if USE_LIKWID
        err = perfmon_readCountersCpu(cur_proc_id);
        if (lkw_has_uncore_counters(event_list, n_events)) {
            int tmp_first = lkw_find_first_thread_on_socket(cur_tid, cpu_ids, n_threads);
            if (tmp_first != cur_tid) {
                lkw_mtx_uncore.lock();
                err = perfmon_readCountersCpu(cpu_ids[tmp_first]);
                lkw_mtx_uncore.unlock();
            }
        }
        if (err < 0)
            fprintf(stderr, "Failed to read counters for group %d for thread %d\n", lkw_grp_id, (-1 * err) - 1);

        for (int j = 0; j < n_events; j++) {
            std::string cur_event(event_list[j]);
            result = lkw_get_counter_result(lkw_grp_id, j, cur_event, cur_tid, cpu_ids, n_threads);
            maps_per_thread[cur_tid][cur_event] = result;
        }
#endif
#if USE_PAPI
        if (PAPI_reset(*my_set) != PAPI_OK)
            handle_PAPI_error("PAPI_reset");
#endif
#if USE_PERF
        ioctl(core_fds[cur_tid][0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
#endif
        struct timespec t_start;
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        my_task_log_t *cur_task_data    = (my_task_log_t *) malloc(sizeof(my_task_log_t));
        cur_task_data->start            = t_start;
        cur_task_data->id               = tmp_id;
        task_data->ptr                  = (void *) cur_task_data;

    } else if (cham_t_task_end == schedule_type) {

        struct timespec t_end;
        clock_gettime(CLOCK_MONOTONIC, &t_end);

        my_task_log_t *cur_task_data    = (my_task_log_t *) task_data->ptr;
        auto t_start                    = cur_task_data->start;
        int tmp_id                      = cur_task_data->id;
        auto tag                        = "R" + std::to_string(rank_data->value) + "T" + std::to_string(tmp_id);

#if USE_LIKWID
        err = perfmon_readCountersCpu(cur_proc_id);
        if (lkw_has_uncore_counters(event_list, n_events)) {
            int tmp_first = lkw_find_first_thread_on_socket(cur_tid, cpu_ids, n_threads);
            if (tmp_first != cur_tid) {
                lkw_mtx_uncore.lock();
                err = perfmon_readCountersCpu(cpu_ids[tmp_first]);
                lkw_mtx_uncore.unlock();
            }
        }
        if (err < 0)
            fprintf(stderr, "Failed to read counters for group %d for thread %d\n", lkw_grp_id, (-1 * err) - 1);

        for (int j = 0; j < n_events; j++) {
            std::string cur_event(event_list[j]);
            result = lkw_get_counter_result(lkw_grp_id, j, cur_event, cur_tid, cpu_ids, n_threads);
            maps_per_thread[cur_tid][cur_event] = lkw_calculate_difference(cur_event,
                                                                              maps_per_thread[cur_tid][cur_event],
                                                                              result);
        }
#endif
#if USE_PAPI
        if (PAPI_read(*my_set, thread_event_data_end[cur_tid]) != PAPI_OK)
            handle_PAPI_error("PAPI_read");
        
        for (int j = 0; j < n_events; j++) {
            std::string cur_event(event_list[j]);
            maps_per_thread[cur_tid][cur_event] = (double) thread_event_data_end[cur_tid][j];
        }
#endif
#if USE_PERF
        read(core_fds[cur_tid][0], data_buffers[cur_tid], data_buf_size);
        read_format_t *rf = (read_format_t*) data_buffers[cur_tid];
        for (int j = 0; j < n_events; j++) {
            std::string cur_event(event_list[j]);
            maps_per_thread[cur_tid][cur_event] = (double) rf->values[j].value;
        }
#endif

        cham_t_task_param_info_t p_info = cham_t_get_task_param_info(task);
        std::vector <std::tuple<double, double>> args;
        for (int i = 0; i < p_info.num_args; ++i) {
            args.push_back(std::make_tuple(p_info.arg_sizes[i], p_info.arg_types[i]));
        }

        double exec_time = get_seconds(t_end) - get_seconds(t_start);
        maps_per_thread[cur_tid]["time"]         = exec_time;
        maps_per_thread[cur_tid]["inverseClock"] = inverseClock;

        mtx.lock();
        vec_task_tags.push_back(tag);
        vec_args.push_back(args);
        vec_hwc_values.push_back(maps_per_thread[cur_tid]);
        vec_exec_times.push_back(exec_time * 1000);
        mtx.unlock();

        args.clear();
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

    // Read out env variable which events should be measured
    const char *LIKWID_EVENTS   = std::getenv("LIKWID_EVENTS");
    const char *RESULT_PATH     = std::getenv("RESULT_PATH");
    const char *OUTPUT_RESULTS  = std::getenv("OUTPUT_RESULTS");

    // Security checks for env vars
    if(!LIKWID_EVENTS || strlen(LIKWID_EVENTS) == 0) {
        fprintf(stderr, "Error: Environment variable \"LIKWID_EVENTS\" is not set correctly or empty\n");
        exit(42);
    }
    if(!RESULT_PATH || strlen(RESULT_PATH) == 0) {
        fprintf(stderr, "Error: Environment variable \"RESULT_PATH\" is not set correctly or empty\n");
        exit(42);
    }
    if(!OUTPUT_RESULTS || strlen(OUTPUT_RESULTS) == 0) {
        fprintf(stderr, "Error: Environment variable \"OUTPUT_RESULTS\" is not set correctly or empty\n");
        exit(42);
    }

    event_list_string   = LIKWID_EVENTS;
    event_list          = split_string(event_list_string, ',');
    output_results      = atoi(OUTPUT_RESULTS);

    // allocate relevant data structures for likwid workaround
    n_events    = event_list.size();
    n_threads   = omp_get_max_threads();
    maps_per_thread.resize(n_threads);

    cham_t_set_callback                 = (cham_t_set_callback_t) lookup("cham_t_set_callback");
    cham_t_get_callback                 = (cham_t_get_callback_t) lookup("cham_t_get_callback");
    cham_t_get_rank_data                = (cham_t_get_rank_data_t) lookup("cham_t_get_rank_data");
    cham_t_get_thread_data              = (cham_t_get_thread_data_t) lookup("cham_t_get_thread_data");
    cham_t_get_rank_info                = (cham_t_get_rank_info_t) lookup("cham_t_get_rank_info");
    cham_t_get_task_param_info          = (cham_t_get_task_param_info_t) lookup("cham_t_get_task_param_info");
    cham_t_get_task_param_info_by_id    = (cham_t_get_task_param_info_by_id_t) lookup("cham_t_get_task_param_info_by_id");
    cham_t_get_task_data                = (cham_t_get_task_data_t) lookup("cham_t_get_task_data");

    register_callback(cham_t_callback_thread_init);
    register_callback(cham_t_callback_thread_finalize);
    register_callback(cham_t_callback_post_init_serial);
    register_callback(cham_t_callback_task_schedule);

    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    cham_t_data_t *r_data       = cham_t_get_rank_data();
    r_data->value               = r_info->comm_rank;

#if USE_LIKWID
    cpu_ids     = (int *) malloc(sizeof(int) * n_threads);
#endif
#if USE_PAPI
    int err;
    int code = 0;
    std::vector<likwid_papi_mapping_entry_t> name_mapping = init_mapping_likwid_papi();
    papi_event_names = mapping_convert_likwid_to_papi(event_list, name_mapping);

    thread_event_data_end   = (long long**) malloc(sizeof(long long*) * n_threads);
    thread_event_sets       = (int*) malloc(sizeof(int) * n_threads * 16);

    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
        handle_PAPI_error("PAPI library init error!\n");

    if (PAPI_thread_init ((unsigned long (*)(void)) (omp_get_thread_num)) != PAPI_OK)
        handle_PAPI_error("PAPI_thread_init failure\n");

    for(int e = 0; e < n_events; e++) {
        err = PAPI_event_name_to_code((char*) papi_event_names[e].c_str(), &code);
        papi_events[e] = code;
    }
    // fprintf(stderr, "Outputting resulting Event Codes:\n");
    // for(int i = 0; i < n_events; i++) {
    //     fprintf(stderr, "IN -- Rank R#%d T#%d -- Event Code[%d] = %d -- Err=%d\n", r_data->value, 0, i, papi_events[i], 0);
    // }
#endif
#if USE_PERF
    std::vector<likwid_perf_mapping_entry_t> mapping_perf = init_mapping_likwid_perf();
    perf_event_list = mapping_convert_likwid_to_perf(event_list, mapping_perf);

    // malloc array for file descriptors
    core_fds        = (int**) malloc(sizeof(int*) * omp_get_max_threads());
    data_buffers    = (char**) malloc(sizeof(char*) * omp_get_max_threads());
    data_buf_size   = sizeof(uint64_t) + 2 * sizeof(uint64_t) * n_events;
#endif

    // if (r_info->comm_rank == 0) {
    path_result_file = std::string(RESULT_PATH) + "_R" + std::to_string(r_info->comm_rank) + ".txt";
    // }

    return 1; //success
}

void cham_t_finalize(cham_t_data_t *tool_data) {

    cham_t_rank_info_t *r_info  = cham_t_get_rank_info();

    // if (r_info->comm_rank == 0) {
    if (output_results) {
        int num_tasks = vec_exec_times.size();
        std::ofstream res_fstream   (path_result_file, std::ios_base::app);

        for (int i = 0; i < num_tasks; ++i) {
            // calculate metrics for task
            std::vector <likwid_metric_value_t> cur_metrics = lkw_calculate_metrics(metric_infos, vec_hwc_values.at(i));

            if (i == 0) {
                // print header first
                res_fstream  << "TASK_ID;RUNTIME_ms;INVERSE_CLCK;SIZE_IN_min;SIZE_IN_max;SIZE_IN_mean;SIZE_IN_overall;SIZE_OUT_min;SIZE_OUT_max;SIZE_OUT_mean;SIZE_OUT_overall;SIZE_overall";
                for (const auto &entry : cur_metrics)
                    res_fstream << ';' << entry.metric_name;
                res_fstream << "\n";
            }

            long double overall_size = 0, size_in = 0, size_out = 0;
            double num_in = 0, num_out = 0;
            long double in_min = std::numeric_limits<double>::max(), out_min = std::numeric_limits<double>::max();
            long double in_max = std::numeric_limits<double>::min(), out_max = std::numeric_limits<double>::min();

            for (int j = 0; j < vec_args.at(i).size(); ++j) {
                long double cur_size = (long double) std::get<0>(vec_args.at(i).at(j)) / 1000;
                int         cur_type = (int) std::get<1>(vec_args.at(i).at(j));
                overall_size += cur_size;

                if ((cur_type & CHAM_OMP_TGT_MAPTYPE_TO) == CHAM_OMP_TGT_MAPTYPE_TO) {
                    num_in++;
                    size_in += cur_size;
                    if (in_min > cur_size)
                        in_min = cur_size;
                    if (in_max < cur_size)
                        in_max = cur_size;
                }

                if ((cur_type & CHAM_OMP_TGT_MAPTYPE_FROM) == CHAM_OMP_TGT_MAPTYPE_FROM) {
                    num_out++;
                    size_out += cur_size;
                    if (out_min > cur_size)
                        out_min = cur_size;
                    if (out_max < cur_size)
                        out_max = cur_size;
                }
            }

            res_fstream << std::scientific << vec_task_tags.at(i) << ";" << vec_exec_times.at(i) << ";" << inverseClock << ";"\
            << in_min << ";"    << in_max << ";"    << size_in / num_in     << ";" << size_in << ";" \
            << out_min << ";"   << out_max << ";"   << size_out / num_out   << ";" << size_out << ";"\
            << overall_size;

            for (const auto &entry : cur_metrics) {
                res_fstream << ";" << entry.value ;

                if  (isnan(entry.value)){
                    std::cout << vec_task_tags.at(i) << " - NANEVENT: " << entry.metric_name << " - VALUE: " << entry.value << std::endl;
                    for (auto it = vec_hwc_values.at(i).cbegin(), next_it = it; it != vec_hwc_values.at(i).cend(); it = next_it)
                    {
                        ++next_it;
                        std::cout << vec_task_tags.at(i) << " - EVENT: " << it->first << " - VALUE: " << it->second << std::endl;
                    }
                }
            }

            res_fstream << "\n";
        }
        res_fstream.close();
    }
    // }
#if USE_LIKWID
    free(cpu_ids);
#endif
#if USE_PAPI
    free(thread_event_sets);
    for(int t = 0; t < omp_get_max_threads(); t++) {
        free(thread_event_data_end[t]);
    }
    free(thread_event_data_end);
#endif
#if USE_PERF
    for(int i = 0; i < omp_get_max_threads(); i++) {
        ioctl(core_fds[i][0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
		for(int j = 0; j < n_events; j++) {
        	close(core_fds[i][j]);
		}
        free(core_fds[i]);
        free(data_buffers[i]);
	}
    free(core_fds);
    free(data_buffers);
#endif
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