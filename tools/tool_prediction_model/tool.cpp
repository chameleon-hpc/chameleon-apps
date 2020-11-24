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
    intptr_t codeptr_ra)
{
    int rank_info                   = cham_t_get_rank_info()->comm_rank;
    TYPE_TASK_ID internal_task_id   = chameleon_get_task_id(task);
    double q_time                   = omp_get_wtime();
    
    // create custom data structure and use task_data as pointer
    cham_t_task_info_t * cur_task = new cham_t_task_info_t;
    cur_task->task_id           = internal_task_id;
    cur_task->rank_belong       = rank_info;
    cur_task->queue_time        = q_time;
    cur_task->codeptr_ra        = codeptr_ra;
    cur_task->arg_num           = chameleon_get_arg_num(task);

    // get arg_sizes
    for (std::vector<int64_t>::iterator it=arg_sizes.begin(); it!=arg_sizes.end(); ++it){
        cur_task->add_arg_size(*it);
    }

    // add task to the list
    tool_task_list.push_back(cur_task);
}

/* interfere processing-tasks */
static void
on_cham_t_callback_task_processed(
    cham_migratable_task_t * task,
    std::vector<int64_t> arg_sizes)
{
    TYPE_TASK_ID task_id = chameleon_get_task_id(task);
    double start_time = omp_get_wtime();
    tool_task_list.set_start_time(task_id, start_time);

    // get core info
    int core_id = sched_getcpu();
    double core_freq = get_core_freq(core_id);
    tool_task_list.set_processed_freq(task_id, core_freq);

    // predict task-runtime when the model is trained successfully
    if (is_model_trained == true)
    {
        // create a vector of inputs
        std::vector<float> input;

        // get num_features
        int num_args = chameleon_get_arg_num(task);
        int num_features = num_args + 1;

        // normalize the input
        for (int i = 0; i < num_args; i++)
        {
            float norm_val = 0.0;
            if (max_vec[i] != min_vec[i])
                norm_val = (arg_sizes[i] - min_vec[i]) / (max_vec[i] - min_vec[i]);
            input.push_back(norm_val);
        }

        // add norm_freq
        float norm_freq = 0.0;
        if (max_vec[num_features-1] != min_vec[num_features-1])
            norm_freq = (core_freq - min_vec[num_features-1]) / (max_vec[num_features-1] - min_vec[num_features-1]);
        input.push_back(norm_freq);

        // convert input to tensor_type
        auto tensor_input = torch::from_blob(input.data(), {1,num_features});

        // get the norm_pred_val
        auto norm_pred_val = net->forward(tensor_input);
        float pred_val = norm_pred_val.item<float>() * (max_vec[num_features] - min_vec[num_features])
                        + min_vec[num_features];    // num_features is the index of min-max val for exe_time
        pred_runtime_list.push_back(pred_val);
    }
    else
    {
        // have not yet the pred-model
        pred_runtime_list.push_back(0.0);
    }
}

/* interfere ending-tasks */
static void
on_cham_t_callback_task_end(
    cham_migratable_task_t * task)
{
    TYPE_TASK_ID task_id = chameleon_get_task_id(task);
    double end_time = omp_get_wtime();
    tool_task_list.set_end_time(task_id, end_time);

    float runtime = tool_task_list.get_exe_time(task_id);
    runtime_list.push_back(runtime);

    int list_finished_tasks = runtime_list.size();

    // check dataset for calling train_model
    if (list_finished_tasks == num_samples)
    {
        // normalize input
        normalize_input(&tool_task_list, norm_input);

        // normalize ground_truth
        normalize_ground_truth(norm_ground_truth);

        net = online_training_model(norm_input, norm_ground_truth);
    }
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
    cham_t_set_callback = (cham_t_set_callback_t) lookup("cham_t_set_callback");
    cham_t_get_rank_data = (cham_t_get_rank_data_t) lookup("cham_t_get_rank_data");
    cham_t_get_thread_data = (cham_t_get_thread_data_t) lookup("cham_t_get_thread_data");
    cham_t_get_rank_info = (cham_t_get_rank_info_t) lookup("cham_t_get_rank_info");

    register_callback(cham_t_callback_task_create);
    register_callback(cham_t_callback_task_end);
    register_callback(cham_t_callback_task_processed);

    // get num_samples_env_var value
    char* num_samples_env_var = std::getenv("NUM_SAMPLE");
    if (num_samples_env_var != NULL)
        num_samples = std::atoi(num_samples_env_var);

    // get num_epochs_env_var value
    char* num_epochs_env_var = std::getenv("NUM_EPOCH");
    if (num_epochs_env_var != NULL)
        num_epochs = std::atoi(num_epochs_env_var);

    // test bcl-fast-queue here
    // int rank_info = cham_t_get_rank_info()->comm_rank;
    // int num_ranks = cham_t_get_rank_info()->comm_size;
    // test_bcl_fast_queue(rank_info, num_ranks);

    return 1;
}

/* cham_tool finalize */
void cham_t_finalize(cham_t_data_t *tool_data)
{
    // get rank_num
    int rank = cham_t_get_rank_info()->comm_rank;

    // writing logs
    chameleon_t_write_logs(&tool_task_list, rank);
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