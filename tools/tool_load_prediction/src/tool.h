#include "chameleon.h"
#include "chameleon_tools.h"
#include <unistd.h>
#include <sys/syscall.h>
#include <inttypes.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <sched.h>
#include <numeric>
#include <mpi.h>

// for using torch
#include <torch/torch.h>
#include <iostream>
#include <cstddef>
#include <iomanip>

// for using mlpack
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <ensmallen.hpp>

#ifndef TRACE
#define TRACE 0
#endif

#if TRACE==1
#include "VT.h"
static int event_tool_task_create = -1;
static int event_tool_task_exec = -1;
static int _tracing_enabled = 1;
#ifndef VT_BEGIN_CONSTRAINED
#define VT_BEGIN_CONSTRAINED(event_id) if (_tracing_enabled) VT_begin(event_id);
#endif

#ifndef VT_END_W_CONSTRAINED
#define VT_END_W_CONSTRAINED(event_id) if (_tracing_enabled) VT_end(event_id);
#endif
#endif

#ifndef NUM_SAMPLE
#define DEF_NUM_SAMPLE 100
#endif

#ifndef NUM_EPOCH
#define DEF_NUM_EPOCH 2000
#endif

#ifndef NUM_ITERATIONS
#define DEF_ITERS 100
#endif

#ifndef NUM_OMP_THREADS
#define DEF_N_THREADS 2
#endif

#ifndef MXM_EXAMPLE
#define MXM_EXAMPLE 0
#endif

#ifndef SAMOA_EXAMPLE
#define SAMOA_EXAMPLE 1
#endif

#ifndef LOG_DIR
#define DEF_LOG_DIR "./logs"
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 3
#endif

// ================================================================================
// Declare Struct
// ================================================================================

/**
 * The cham-tool task struct for storing profiled-data.
 *
 * This struct helps collect task-by-task info
 * @attributes: tid, rank, num_args, args_list, ...
 */
typedef struct prof_task_info_t {
    TYPE_TASK_ID tid;   // task id
    int rank_belong;    // rank created it
    int num_args;       // num arguments
    std::vector<int64_t> args_list;   // list of arguments 
    double que_time;    // queued time
    double sta_time;    // started time
    double end_time;    // end time
    double mig_time;    // migrated time
    double exe_time;    // runtime
    intptr_t code_ptr;  // code pointer
} prof_task_info_t;


/**
 * The cham-tool profile-task data class.
 *
 * This class helps holding a list of prof-tasks.
 * @attributes: tasks_list, ...
 */
class prof_task_list_t {
    public:
        std::vector<prof_task_info_t *> task_list;
        std::vector<double> avg_load_list;
        std::mutex m;
        std::atomic<size_t> tasklist_size;
        
        /* duplicate to avoid contention on single atomic
        from comm thread and worker threads */
        std::atomic<size_t> dup_list_size;

        prof_task_list_t() { tasklist_size = 0; dup_list_size = 0; }

        size_t dup_size() {
            return this->dup_list_size.load();
        }

        size_t size() {
            return this->tasklist_size.load();
        }

        bool empty() {
            return this->tasklist_size <= 0;
        }

        void push_back(prof_task_info_t* task) {
            this->m.lock();
            this->task_list.push_back(task);
            this->tasklist_size++;
            this->dup_list_size++;
            this->m.unlock();
        }

        prof_task_info_t* pop_back(){
            if(this->empty())
                return nullptr;
            
            prof_task_info_t* ret_val = nullptr;
            this->m.lock();
            if(!this->empty()){
                this->tasklist_size--;
                this->dup_list_size--;
                ret_val = this->task_list.back();
                this->task_list.pop_back();
            }
            this->m.unlock();
            return ret_val;
        }

        void add_avgload(double avg_value){
            this->m.lock();
            // this works for torch, but no need for mlpack from now
            // float f_value = (float) avg_value;
            this->avg_load_list.push_back(avg_value);
            this->m.unlock();
        }
};


/**
 * The general regression model struct based on Torch-CXX
 *
 * @param num_features, num hidden layers, num_outputs
 * @return a corresponding prediction model
 */
struct SimpleRegression:torch::nn::Module {

    SimpleRegression(int in_dim, int n_hidden, int out_dim){
        hidden1 = register_module("hidden1", torch::nn::Linear(in_dim, n_hidden));
        hidden2 = register_module("hidden2", torch::nn::Linear(n_hidden, n_hidden));
        predict = register_module("predict", torch::nn::Linear(n_hidden, out_dim));

        // for optimizing the model
        torch::nn::init::xavier_uniform_(hidden1->weight);
        torch::nn::init::zeros_(hidden1->bias);
        torch::nn::init::xavier_uniform_(hidden2->weight);
        torch::nn::init::zeros_(hidden2->bias);

        torch::nn::init::xavier_uniform_(predict->weight);
        torch::nn::init::zeros_(predict->bias);
    }

    torch::Tensor forward(const torch::Tensor& input) {
        auto x = torch::tanh(hidden1(input));
        x = torch::relu(hidden2(x));
        x = predict(x);
        return x;
    }

    torch::nn::Linear hidden1{nullptr}, hidden2{nullptr}, predict{nullptr};
};

/**
 * Example: how to declare the torch-based regression model.
 *  + auto net = std::make_shared<SimpleRegression>(4, 10, 1);
 *  + ...
 */


// ================================================================================
// Global Variables
// ================================================================================
prof_task_list_t profiled_task_list;
std::vector<float> min_vec; // the last element is min_ground_truth
std::vector<float> max_vec; // the last element is max_ground_truth
int num_samples = DEF_NUM_SAMPLE;
int num_epochs = DEF_NUM_EPOCH;

// Declare a general mlpack regression model
mlpack::regression::LinearRegression lr;

// Declare 2 vectors for storing avg_load & pre_load (predicted loads)
const int REF_NUM_ITERS = 100;
arma::vec avg_load_per_iter_list(REF_NUM_ITERS);
arma::vec pre_load_per_iter_list(REF_NUM_ITERS);


// ================================================================================
// Help Functions
// ================================================================================
#pragma region Local Helpers
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

	// initialize original index locations
	std::vector<size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	std::sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

	return idx;
}

/**
 * Paring the index numbers
 *
 * @param a, b
 * @return an unique number 
 */
int pairing_function(int a, int b){
    int result = ((a + b) * (a + b + 1) / 2) + b;
    return result;
}

#pragma endregion Local Helpers


// ================================================================================
// Util Functions
// ================================================================================

/**
 * Writing logs for all tasks.
 *
 * @param task_list_ref with profiled-info, mpi_rank
 * @return .csv logfiles by rank
 */
void chameleon_t_write_logs(prof_task_list_t& tasklist_ref, int mpi_rank){
    // get log_dir_env
    char* log_dir_env = std::getenv("LOG_DIR");
    std::string log_dir;
    if (log_dir_env != NULL)
        log_dir = log_dir_env;
    else
        log_dir = DEF_LOG_DIR;

    // declare output file
    std::ofstream outfile;
    outfile.open(log_dir + "/logfile_rank_" + std::to_string(mpi_rank) + "_" + std::to_string(num_samples) + "samples_" + std::to_string(num_epochs) + "epochs" + ".csv");

    // create an iterator to traverse the list
    std::vector<prof_task_info_t *>::iterator it;

    // get through the list of profiled tasks
    for (it = tasklist_ref.task_list.begin(); it != tasklist_ref.task_list.end(); it++){

        // get a list of prob_sizes per task
        int num_args = (*it)->num_args;

        // string for storing prob_sizes
        std::string prob_sizes_statement;

#if SAMOA_EXAMPLE==1
        const int selected_arg = 8;  // the last 4 args affecting much on the taks runtime
        for (int arg = selected_arg; arg < num_args; arg++){
            prob_sizes_statement += std::to_string((*it)->args_list[arg]) + "\t";
        }
#else
        const int selected_arg = 0;
        prob_sizes_statement += std::to_string((*it)->arg_sizes.at(selected_arg)) + "\t";
#endif

        std::string line = std::to_string((*it)->tid) + "\t"
                            + prob_sizes_statement + "\n";
                            // + std::to_string((*it)->exe_time) + "\n";
        
        // writing logs
        outfile << line;
    }

    // write avg_load per iterations
    int num_iterations = tasklist_ref.avg_load_list.size();
    for (int i = 0; i < num_iterations; i++){
        double load_val = tasklist_ref.avg_load_list[i];
        double pred_val = pre_load_per_iter_list[i];
        std::string load_str = std::to_string(load_val) + "\t" + std::to_string(pred_val) + "\n";
        outfile << load_str;
    }

    // close file
    outfile.close();
}


/**
 * Free memory of each task.
 *
 * @param a pointer to a task object (struct prof_task_info)
 * @return free mem
 */
static void free_prof_task(prof_task_info_t* task){
    if (task){
        delete task;
        task = nullptr;
    }
}

/**
 * Free memory of a whole tasks-list.
 *
 * @param a ref to a list of tasks (class prof_task_list)
 * @return free mem
 */
void clear_prof_tasklist() {
    while(!profiled_task_list.empty()) {
        prof_task_info_t *task = profiled_task_list.pop_back();
        free_prof_task(task);
    }
}

/**
 * Get CPU frequency (Hz).
 *
 * @param CPU core ID
 * @return freq_value
 */
double get_core_freq(int core_id){
	std::string line;
	std::ifstream file ("/proc/cpuinfo");

	double freq = 0.0;
	int i = 0;

	if (file.is_open()){
		while (getline(file, line)){
			if (line.substr(0,7) == "cpu MHz"){
				if (i == core_id){
					std::string::size_type sz;
					freq = std::stod (line.substr(11,21), &sz);
					return freq;
				}
				else	i++;
			}
		}

		file.close();

	} else{
		printf("Unable to open file!\n");
	}

	return freq;
}

/**
 * Printing tensor values in Torch-CXX.
 *
 * @param tensor_arr, num_vals are an array of tensor values,
 *      and num of values we want to print.
 * @return I/O
 */
void print_tensor(torch::Tensor tensor_arr, int num_vals)
{
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < num_vals; i++){
        std::cout << tensor_arr[i].item<float>() << std::endl;
    }
}


/**
 * Normalizing 2D-vector by column for training with Torch.
 *
 * @param vec: a ref to the 2D vector
 */
void normalize_2dvector_by_column(std::vector<std::vector<float>> &vec)
{
    int num_rows = vec.size();
    int num_cols = (vec[0]).size();

    // find min-max vectors by each column
    for (int i = 0; i < num_cols; i++) {
        min_vec.push_back(10000000.0);
        max_vec.push_back(0.0);
    }

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            // check min
            if (vec[i][j] < min_vec[j])
                min_vec[j] = vec[i][j];

            // check max
            if (vec[i][j] > max_vec[j])
                max_vec[j] = vec[i][j];
        }
    }

    // normalize the whole vector
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if (min_vec[j] == max_vec[j])
                vec[i][j] = 0.0;
            else
                vec[i][j] = (vec[i][j] - min_vec[j]) / (max_vec[j] - min_vec[j]) * 2 - 1;
        }
    }
}


/**
 * Normalizing the groundtruth for training with Torch.
 *
 * @param vec: a ref to the groundtruth vector
 */
void normalize_ground_truth(std::vector<float> &norm_ground_output)
{
    // // find min-max values
    // const auto [min, max] = std::minmax_element(runtime_list.begin(), runtime_list.end());

    // // add min-max to the global min-max_vec
    // min_vec.push_back(*min);
    // max_vec.push_back(*max);

    // // normalize the vector
    // int vec_size = runtime_list.size();
    // for (int i = 0; i < vec_size; i++) {
    //     float norm_val = (runtime_list[i] - *min) / (*max - *min);
    //     norm_ground_truth.push_back(norm_val);
    // }
}


/**
 * Normalizing the features_input for training with Torch
 *
 * @param tasklist_ref, result: a ref to the tasklist_ref & a 2D vector
        for saving the result.
 */
void normalize_input(prof_task_list_t& tasklist_ref, std::vector<std::vector<float>> result)
{
//     // find min-max in arg_sizes
//     int num_args = 0;

//     // create an iterator to traverse the list
//     std::list<cham_tool_profiled_task_t *>::iterator it = tasklist_ptr->task_list.begin();

//     // get through the list of profiled_tasks
//     for (int i = 0; i < num_samples; i++) {

//         // create a vector of args
//         std::vector<float> task_arg_sizes;

// #if SAMOA_EXAMPLE==1
//         num_args = 14;
//         int selected_arg = 10;
//         for (int j = selected_arg; j < num_args; j++)
//             task_arg_sizes.push_back(float((*it)->arg_sizes.at(j)));
// #else
//         num_args = 5;
//         const int selected_arg = 0;
//         task_arg_sizes.push_back(float((*it)->arg_sizes.at(selected_arg)));
// #endif

//         // add core_freq
//         // task_arg_sizes.push_back(float(list_of_tasks[i]->processed_freq));
        
//         // push back the vector of args to a norm_input vector
//         norm_input.push_back(task_arg_sizes);

//         ++it;
//     }

//     // normalized 2d-vector by column
//     normalize_2dvector_by_column(norm_input);
}


/**
 * Gathering profiled-data for training with mlpack.
 *
 * @param tasklist_ref: a ref to the list of profiled tasks.
        But currently work for the runtime list, no need arguments
        of each task.
 * @return a dataset matrix
 */
bool gather_training_data(prof_task_list_t& tasklist_ref, int num_input_points, int num_iters)
{
    // the current list of finished tasks
    int size_tasklist = tasklist_ref.tasklist_size;
    std::cout << "[CHAM_TOOL] gather_train_data: num finished tasks=" << size_tasklist
              << ", num finished iters=" << num_iters <<std::endl;

    // preparing data
    arma::vec runtime_list(num_iters);
    for (int i = 0; i < num_iters; i++){
        runtime_list[i] = double(tasklist_ref.avg_load_list[i]);
    }
    // std::cout << "[CHAM_TOOL] gather_train_data: runtime_list" << std::endl;
    // std::cout << runtime_list << std::endl;

    // generate a std mat dataset for training
    int n_rows_X = num_input_points;
    int n_cols_X = num_iters - num_input_points;
    int n_rows_Y = 1;
    int n_cols_Y = n_cols_X;
    std::cout << "[CHAM_TOOL] gather_train_data: trainX_size=" << n_rows_X << "x" << n_cols_X
              << ", trainY_size=" << n_rows_Y << "x" << n_cols_Y << std::endl;
    arma::mat trainX(n_rows_X, n_cols_X);
    arma::mat trainY(n_rows_Y, n_cols_Y);
    for (int i = num_input_points; i < num_iters; i++){
        trainX.col(i-num_input_points) = runtime_list.subvec((i-num_input_points), (i-1));
        trainY.col(i-num_input_points) = runtime_list[i];
    }
    // std::cout << "[CHAM_TOOL] gather_train_data: trainX" << std::endl;
    // std::cout << trainX << std::endl;
    // std::cout << "[CHAM_TOOL] gather_train_data: trainY" << std::endl;
    // std::cout << trainY << std::endl;

    // declare and generate the regression model here
    mlpack::regression::LinearRegression lr_pred_model(trainX, trainY);
    lr = lr_pred_model;

    std::cout << "[CHAM_TOOL] the training is done." << std::endl;

    return true;
}


/**
 * Offline regression training.
 *
 * @param mat
 */


/**
 * Online regression training.
 *
 * @param mat
 */
auto online_training_model(std::vector<std::vector<float>> input, std::vector<float> ground_truth)
{
    // // measure time
    // double train_start_time = omp_get_wtime();

    // // convert input to tensor type
    // const int num_rows = input.size();
    // const int num_cols = input[0].size();
    // auto tensor_input = torch::from_blob(input.data(), {num_rows,num_cols});
    // auto ground_truth_tensor = torch::from_blob(ground_truth.data(), {num_rows,1});

    // // generate seed & learning_rate
    // torch::manual_seed(1);
    // const double learning_rate = 0.01;
    // auto final_loss = 0.0;

    // // create the model
    // torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(learning_rate));

    // for (int epoch = 1; epoch <= num_epochs; epoch++) {
    //     auto out = net->forward(tensor_input);

    //     auto loss = torch::nn::functional::mse_loss(out, ground_truth_tensor);

    //     optimizer.zero_grad();

    //     loss.backward();

    //     optimizer.step();

    //     final_loss = loss.item<float>();
    // }

    // // change the global flag, the pre-model was trained
    // is_model_trained = true;

    // // get the measured time
    // double train_end_time = omp_get_wtime();
    // printf("Elapsed Time for training Pred-Model with N_SAMPLES=%d, N_EPOCH=%d: %.5f (s), LOSS= %.5f\n", num_rows, num_epochs, (train_end_time-train_start_time), final_loss);

    // return net;
}