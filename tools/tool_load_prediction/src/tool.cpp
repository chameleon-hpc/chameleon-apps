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
    const int num_args          = arg_sizes.size();
    
    // create custom data structure and use task_data as pointer
    prof_task_info_t *cur_task  = new prof_task_info_t;
    cur_task->tid               = cham_task_id;
    cur_task->rank_belong       = rank_id;
    cur_task->num_args          = num_args;
    cur_task->que_time          = queued_time;
    cur_task->code_ptr          = codeptr_ra;

    // get arg_sizes
    cur_task->args_list.resize(num_args);
    for (int i = 0; i < num_args; i++){
        cur_task->args_list[i] = arg_sizes[i];
    }

    // add task to the list
    profiled_task_list.push_back(cur_task);
}


/**
 * Callback task begin.
 *
 * @param thread_id: the current thread is proceeding this task.
 * @param task: a pointer to the migration task object at Chameleon-side.
 * @param start_time: could be measured at the time a task is started.
 * @param arg_sizes: list of argument sizes.
 * @param taskwait_counter: id of the current iteration (cycle).
 */
static void
on_cham_t_callback_task_begin(int thread_id, cham_migratable_task_t * task, double start_time,
    std::vector<int64_t> arg_sizes, int taskwait_counter)
{
    /**
     * Not using yet.
     */

}


/**
 * Callback task end.
 *
 * @param thread_id: the current thread is proceeding this task.
 * @param task: a pointer to the migration task object at Chameleon-side.
 * @param end_time: could be measured at the time a task is finished.
 * @param taskwait_counter: id of the current iteration (cycle).
 */
static void
on_cham_t_callback_task_end(int thread_id, cham_migratable_task_t * task,
    double end_time, int32_t taskwait_counter)
{
    
#if TRACE==1
    static int cb_event_taskend = -1;
    std::string cb_event_taskend_name = "cb_taskend";
    if(cb_event_taskend == -1) 
        int ierr = VT_funcdef(cb_event_taskend_name.c_str(), VT_NOCLASS, &cb_event_taskend);
    VT_BEGIN_CONSTRAINED(cb_event_taskend);
#endif

    TYPE_TASK_ID task_id = chameleon_get_task_id(task);
    double exetime = 0.0;

#if TRACE==1
    VT_END_W_CONSTRAINED(cb_event_taskend);
#endif
}


/**
 * Callback get stats_load info after a cycle (iteration) is done.
 *
 * @param taskwait_counter: id of the current iteration (cycle).
 * @param thread_id: the last thread calls this callback.
 * @param taskwait_load: the loac value per this cycle.
 */
static double
on_cham_t_callback_get_load_stats_per_taskwait(int32_t taskwait_counter,
    int32_t thread_id, double taskwait_load)
{
    int rank = cham_t_get_rank_info()->comm_rank;
    int iter = taskwait_counter;
    profiled_task_list.add_avgload(taskwait_load);

    /**
     * Do something / try to add it to another arma::vector
     */
    avg_load_per_iter_list[iter] = taskwait_load;

    return taskwait_load;
}

/**
 * Callback get the trigger from comm_thread, then training the pred-model.
 *
 * @param taskwait_counter: id of the current iteration (cycle),
 *      now trigger this callack by the num of passed iters.
 */
static bool
on_cham_t_callback_train_prediction_model(int32_t taskwait_counter)
{
    bool is_trained = false;
    int rank = cham_t_get_rank_info()->comm_rank;

    printf("[CHAM_TOOL] R%d: starts training pred_model at iter-%d\n", rank, taskwait_counter);
    int num_points = 6;
    int num_finished_iters = taskwait_counter-1;
    is_trained = gather_training_data(profiled_task_list, num_points, num_finished_iters);

    return is_trained;
}


/**
 * Callback get the trigger from comm_thread, then calling the trained pred-model.
 *
 * @param taskwait_counter: id of the current iteration (cycle).
 * @return predicted_value: for the load of the corresponding iter.
 */
static double
on_cham_t_callback_valid_prediction_model(int32_t taskwait_counter)
{
    // prepare the input
    int rank = cham_t_get_rank_info()->comm_rank;
    int num_features = 6;
    int s_point = taskwait_counter - num_features;
    int e_point = taskwait_counter;
    double pred_load = 0.0;
    arma::vec input_vec(num_features);
    for (int i = s_point; i < e_point; i++){
        input_vec[i-s_point] = double(profiled_task_list.avg_load_list[i]);
    }
    std::cout << "[CHAM_TOOL] R" << rank << " valid_pred_model: input_vec from iter-"
              << s_point << " to iter-" << e_point-1 << std::endl;
    std::cout << input_vec.t();

    // call the pred_model
    arma::mat x_mat(1, num_features);
    x_mat = input_vec;
    arma::rowvec p_load;
    lr.Predict(x_mat, p_load);
    pred_load = p_load[0];

    // store to the arma::vector for writing logs
    pre_load_per_iter_list[taskwait_counter] = pred_load;

    return pred_load;
}

//================================================================
// Start Tool & Register Callbacks
//================================================================

#define register_callback_t(name, type)                                         \
do{                                                                             \
    type f_##name = &on_##name;                                                 \
    if (cham_t_set_callback(name, (cham_t_callback_t)f_##name) == cham_t_set_never)   \
        printf("0: Could not register callback '" #name "'\n");                 \
} while(0)

#define register_callback(name) register_callback_t(name, name##_t)


/**
 * Initializing the cham-tool callbacks.
 *
 * @param lookup: search the name of activated callbacks.
 * @param tool_data
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
    // register_callback(cham_t_callback_task_begin);
    // register_callback(cham_t_callback_task_end);
    register_callback(cham_t_callback_get_load_stats_per_taskwait);
    register_callback(cham_t_callback_train_prediction_model);
    register_callback(cham_t_callback_valid_prediction_model);

    // get num_samples_env_var value
    char* num_samples_env_var = std::getenv("NUM_SAMPLE");
    if (num_samples_env_var != NULL)
        num_samples = std::atoi(num_samples_env_var);

    // get num_epochs_env_var value
    char* num_epochs_env_var = std::getenv("NUM_EPOCH");
    if (num_epochs_env_var != NULL)
        num_epochs = std::atoi(num_epochs_env_var);
    
    // free memory
    delete num_samples_env_var;
    delete num_epochs_env_var;

    return 1;
}

/**
 * Finalizing the cham-tool.
 *
 * @param tool_data
 */
void cham_t_finalize(cham_t_data_t *tool_data)
{
    int rank = cham_t_get_rank_info()->comm_rank;

    chameleon_t_write_logs(profiled_task_list, rank);

    // clear profiled-data task list
    clear_prof_tasklist();

}

/**
 * Starting the cham-tool.
 *
 * @param cham_version.
 * @return as a main function of the callback took,
 *      would init cham_t_initialize and cham_t_finalize.
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