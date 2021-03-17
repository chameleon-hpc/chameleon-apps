#include "tool.h"

static cham_t_set_callback_t cham_t_set_callback;
static cham_t_get_rank_data_t cham_t_get_rank_data;
static cham_t_get_thread_data_t cham_t_get_thread_data;
static cham_t_get_rank_info_t cham_t_get_rank_info;

//================================================================
// Variables
//================================================================


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

/* interfere creating-tasks */
static void
on_cham_t_callback_task_create(
    cham_migratable_task_t * task,
    std::vector<int64_t> arg_sizes,
    double queued_time,
    intptr_t codeptr_ra,
    int taskwait_counter)
{
    int rank_id                 = cham_t_get_rank_info()->comm_rank;
    TYPE_TASK_ID cham_task_id   = chameleon_get_task_id(task);

    // get num of args per task
    const int num_args          = arg_sizes.size();
    
    // create custom data structure and use task_data as pointer
    // malloc doesn't work here???
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

/* interfere processing-tasks */
static void
on_cham_t_callback_task_begin(int thread_id, cham_migratable_task_t * task, double start_time,
    std::vector<int64_t> arg_sizes, int taskwait_counter)
{
    // get rank
    int rank_info = cham_t_get_rank_info()->comm_rank;

    // get and set core_freq_info
    // int core_id = sched_getcpu();
    // double core_freq = get_core_freq(core_id);
}

/* interfere ending-tasks */
static void
on_cham_t_callback_task_end(int thread_id, cham_migratable_task_t * task, double end_time, int32_t taskwait_counter)
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

/* interfere distributed-taskwaits */
static double
on_cham_t_callback_get_load_stats_per_taskwait(int32_t taskwait_counter, int32_t thread_id, double taskwait_load)
{
    int rank = cham_t_get_rank_info()->comm_rank;
    int iter = taskwait_counter;
    double total_load = taskwait_load;
    double load_avg = total_load / NUM_THREADS;
    profiled_task_list.add_avgload(load_avg);

    // do something

    return load_avg;
}

static bool
on_cham_t_callback_train_prediction_model(int32_t taskwait_counter)
{
    bool is_trained = false;
    int rank = cham_t_get_rank_info()->comm_rank;

    sleep(3);

    // printf("[CHAM_TOOL] R%d: starts training pred_model at iter-%d\n", rank, taskwait_counter);

    return is_trained;
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


/* cham_tool init and calling callbacks */
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

/* cham_tool finalize */
void cham_t_finalize(cham_t_data_t *tool_data)
{
    int rank = cham_t_get_rank_info()->comm_rank;

    chameleon_t_write_logs(profiled_task_list, rank);

    // clear prof-data
    clear_prof_tasklist();

}

/* start the tool */
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