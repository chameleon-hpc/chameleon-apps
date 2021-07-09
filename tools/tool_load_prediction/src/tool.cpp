#include "tool.h"


//================================================================
// Variables
//================================================================
static cham_t_set_callback_t cham_t_set_callback;
static cham_t_get_rank_data_t cham_t_get_rank_data;
static cham_t_get_thread_data_t cham_t_get_thread_data;
static cham_t_get_rank_info_t cham_t_get_rank_info;


//================================================================
// Additional functions
//================================================================
int compare( const void *pa, const void *pb ){
    const int *a = (int *) pa;
    const int *b = (int *) pb;

    if(a[0] == b[0])
        return a[0] - b[0];
    else
        return a[1] - b[1];
}

//================================================================
// Callback Functions
//================================================================ 

/**
 * Callback task create.
 *
 * @param task: a pointer to the migration task object at Chameleon-side.
 * @param arg_sizes: list of argument sizes
 * @param queue_time: could be measured at the time a task is added to the queue
 * @param codeptr_ra: the code pointer of the task-entry (function)
 * @param taskwait_counter: id of the current iteration (cycle)
 */
static void
on_cham_t_callback_task_create(cham_migratable_task_t * task, std::vector<int64_t> arg_sizes,
    double queued_time, intptr_t codeptr_ra, int taskwait_counter)
{
    int rank_id = cham_t_get_rank_info()->comm_rank;
    TYPE_TASK_ID cham_task_id = chameleon_get_task_id(task);

    // get num of args per task
    const int num_args = arg_sizes.size();
    int num_cycle = taskwait_counter;
    
    // create custom data structure and use task_data as pointer
    prof_task_info_t *cur_task  = new prof_task_info_t;
    if (rank_id != 0){
        int shift_rank_id       = rank_id << NBITS_SHIFT;
        cur_task->tool_tid      = (profiled_task_list.ntasks_per_rank * num_cycle) + (cham_task_id - shift_rank_id - 1);
    } else {
        cur_task->tool_tid      = (profiled_task_list.ntasks_per_rank * num_cycle) + (cham_task_id - 1);
    }
    cur_task->cham_tid          = cham_task_id;
    cur_task->rank_belong       = rank_id;
    cur_task->num_args          = num_args;
    cur_task->que_time          = queued_time;
    cur_task->code_ptr          = codeptr_ra;

    // try to get cpu-core frequency here
    int core_id = sched_getcpu();
    double freq = get_core_freq(core_id);
    cur_task->core_freq = freq;

    // get arg_sizes
    cur_task->args_list.resize(num_args);
    for (int i = 0; i < num_args; i++){
        cur_task->args_list[i] = arg_sizes[i];
    }

    // add task to the list
    // int thread_id = omp_get_thread_num();
    // printf("[TOOL] R%d T%d: callback create task cham_tid=%d, tool_tid=%d, tw_counter=%d\n",
    //                     rank_id, thread_id, cur_task->cham_tid, cur_task->tool_tid, num_cycle);
    profiled_task_list.push_back(cur_task);

    // increase tool_tasks_counter
    tool_tasks_count++;
}

/**
 * Callback get stats_load info after a cycle (iteration) is done.
 *
 * @param taskwait_counter: id of the current iteration (cycle).
 * @param thread_id: the last thread calls this callback.
 * @param taskwait_load: the loac value per this cycle.
 */
static void
on_cham_t_callback_get_load_stats_per_taskwait(int32_t taskwait_counter,
    int32_t thread_id, double taskwait_load)
{
    int rank = cham_t_get_rank_info()->comm_rank;
    int iter = taskwait_counter;
    profiled_task_list.add_avgload(taskwait_load, iter);

    // get the wallclock execution time per iteration
    avg_load_per_iter_list[iter] = taskwait_load;

}

/**
 * Callback get stats_load info after a cycle (iteration) is done.
 *
 * @param taskwait_counter: id of the current iteration (cycle).
 * @param thread_id: the last thread calls this callback.
 * @param taskwait_load: the loac value per this cycle.
 */
static void
on_cham_t_callback_get_task_wallclock_time(int32_t taskwait_counter,
    int32_t thread_id, int task_id, double wallclock_time)
{
    int idx;
    int num_cycle = taskwait_counter;
    int rank_id = cham_t_get_rank_info()->comm_rank;
    if (rank_id != 0){
        int shift_rank_id = rank_id << NBITS_SHIFT;
        idx = (profiled_task_list.ntasks_per_rank * num_cycle) + (task_id - shift_rank_id - 1);
    } else {
        idx = (profiled_task_list.ntasks_per_rank * num_cycle) + (task_id - 1);
    }

    // add the wallclock time per task
    profiled_task_list.add_wallclock_time(wallclock_time, task_id, num_cycle, thread_id, idx);
    // printf("[CHAMTOOL] T%d: passed add_wallclock_time(), task_id=%d, tool_idx=%d\n", thread_id, task_id, idx);
}

/**
 * Callback get the trigger from comm_thread, then training the pred-model.
 *
 * @param taskwait_counter: id of the current iteration (cycle), now trigger this
 *        callback by the num of passed iters.
 * @return is_trained: a bool flag to notice the prediction model that has been trained.
 */
static bool
on_cham_t_callback_train_prediction_model(int32_t taskwait_counter, int prediction_mode)
{
    bool is_trained = false;
    int rank = cham_t_get_rank_info()->comm_rank;

    /* Mode 1&2: prediction by time-series load as the patterns */
    if (prediction_mode == 1 || prediction_mode == 2){
        printf("[CHAM_TOOL] R%d: starts training pred_model at iter-%d\n", rank, taskwait_counter);
        int num_points = 6;
        int num_finished_iters = taskwait_counter-1;
        is_trained = online_mlpack_training_iterative_load(profiled_task_list, num_points, num_finished_iters);
    }
    /* Mode 3: prediction by task-characterization as the patterns */
    else if (prediction_mode == 3){
        printf("[CHAM_TOOL] R%d: starts training pred_model at iter-%d\n", rank, taskwait_counter);
        is_trained = online_mlpack_training_task_features(profiled_task_list, taskwait_counter);
    }

    return is_trained;
}


/**
 * Callback get the trigger from comm_thread, then calling the trained pred-model.
 *
 * This callback is used to validate or get the predicted values when cham_lib requests.
 * @param taskwait_counter: id of the current iteration (cycle).
 * @return predicted_value: for the load of the corresponding iter.
 */
static std::vector<double>
on_cham_t_callback_load_prediction_model(int32_t taskwait_counter, int prediction_mode)
{
    // prepare the input
    int rank = cham_t_get_rank_info()->comm_rank;
    int num_features = 6;
    int s_point = taskwait_counter - num_features;
    int e_point = taskwait_counter;
    std::vector<double> pred_load_vec(1, 0.0);

    /* Mode 1: predict iter-by-iter, by time-series patterns */
    if (prediction_mode == 1) {
        get_iter_by_iter_prediction(s_point, e_point, pred_load_vec);
    }
    /* Mode 2: predict the whole future, by time-series patterns and predicted values */
    else if (prediction_mode == 2) {
        pred_load_vec.resize(pre_load_per_iter_list.size(), 0.0);
        get_whole_future_prediction(rank, s_point, e_point, pred_load_vec);
    }
    /* Mode 3: predict iter-by-iter, by task-characterization */
    else if (prediction_mode == 3) {
        pred_load_vec.resize(profiled_task_list.ntasks_per_rank, 0.0);
        get_iter_by_iter_prediction_by_task_characs(taskwait_counter, pred_load_vec);
    }

    // TODO: consider the size of vector
    return pred_load_vec;
}

//================================================================
// Start Tool & Register Callbacks
//================================================================

#define register_callback_t(name, type)                                             \
do{                                                                                 \
    type f_##name = &on_##name;                                                     \
    if (cham_t_set_callback(name, (cham_t_callback_t)f_##name) == cham_t_set_never) \
        printf("0: Could not register callback '" #name "'\n");                     \
} while(0)

#define register_callback(name) register_callback_t(name, name##_t)


/**
 * Initializing the cham-tool callbacks.
 *
 * @param lookup: search the name of activated callbacks.
 * @param tool_data: return the profile data of the tool. But temporarily have
 *        not used this param, just return directly the profile data or use
 *        directly in memory.
 */
int cham_t_initialize(
    cham_t_function_lookup_t lookup,
    cham_t_data_t *tool_data)
{
    printf("Calling register_callback...\n");
    cham_t_set_callback     = (cham_t_set_callback_t) lookup("cham_t_set_callback");
    cham_t_get_rank_data    = (cham_t_get_rank_data_t) lookup("cham_t_get_rank_data");
    cham_t_get_thread_data  = (cham_t_get_thread_data_t) lookup("cham_t_get_thread_data");
    cham_t_get_rank_info    = (cham_t_get_rank_info_t) lookup("cham_t_get_rank_info");

    register_callback(cham_t_callback_task_create);
    register_callback(cham_t_callback_get_load_stats_per_taskwait);
    register_callback(cham_t_callback_get_task_wallclock_time);
    register_callback(cham_t_callback_train_prediction_model);
    register_callback(cham_t_callback_load_prediction_model);
    

    // get info about the number of iterations
    int max_num_iters = DEFAULT_NUM_ITERS;
    int max_num_tasks_per_rank = DEFAULT_NUM_TASKS_PER_RANK;
    char *program_num_iters = std::getenv("EST_NUM_ITERS");
    char *program_num_tasks = std::getenv("TASKS_PER_RANK");
    if (program_num_iters != NULL){
        max_num_iters = atoi(program_num_iters);
    }
    if (program_num_tasks != NULL){
        max_num_tasks_per_rank = atoi(program_num_tasks);
    }

    // resize vectors inside profiled_task_list
    profiled_task_list.ntasks_per_rank = max_num_tasks_per_rank;
    profiled_task_list.avg_load_list.resize(max_num_iters, 0.0);
    profiled_task_list.task_list.resize(max_num_iters * max_num_tasks_per_rank);
    
    // resize the arma::vectors
    tool_tasks_count = 0;
    avg_load_per_iter_list.resize(max_num_iters);
    pre_load_per_iter_list.resize(max_num_iters);

    return 1;
}

/**
 * Finalizing the cham-tool.
 *
 * This callback is to finalize the whole callback tool, after the
 * chameleon finished. This is called from chameleon-lib side.
 * @param tool_data
 */
void cham_t_finalize(cham_t_data_t *tool_data)
{
    // writing per rank
    int rank = cham_t_get_rank_info()->comm_rank;

    // write tool-logs
    chameleon_t_write_logs(profiled_task_list, rank);

    // clear profiled-data task list
    clear_prof_tasklist();

}

/**
 * Starting the cham-tool.
 *
 * This is to start the chameleon tool, then the functions, i.e.,
 * cham_t_initialize() and cham_t_finalize() would be pointed to call.
 * @param cham_version.
 * @return as a main function of the callback took, would init
 *         cham_t_initialize and cham_t_finalize.
 */
#ifdef __cplusplus
extern "C" {
#endif
cham_t_start_tool_result_t* cham_t_start_tool(unsigned int cham_version)
{
    printf("Starting tool with Chameleon Version: %d\n", cham_version);

    static cham_t_start_tool_result_t cham_t_start_tool_result = {&cham_t_initialize, &cham_t_finalize, 0};

    return &cham_t_start_tool_result;
}
#ifdef __cplusplus
}
#endif