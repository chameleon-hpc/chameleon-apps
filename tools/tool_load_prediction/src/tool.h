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
#include <iostream>
#include <cstddef>
#include <iomanip>
#include <mpi.h>

// for using torch
#ifndef ENABLE_PYTORCH_CXX
#define ENABLE_PYTORCH_CXX 0
#endif

#if ENABLE_PYTORCH_CXX == 1
#include <torch/torch.h>
#endif

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

#ifndef NUM_ITERATIONS
#define DEFAULT_NUM_ITERS 100
#endif

#ifndef MXM_EXAMPLE
#define MXM_EXAMPLE 0
#endif

#ifndef JACOBI_EXAMPLE
#define JACOBI_EXAMPLE 0
#endif

#ifndef SAMOA_EXAMPLE
#define SAMOA_EXAMPLE 0
#endif

#ifndef LOG_DIR
#define DEF_LOG_DIR "./logs"
#endif


// ================================================================================
// Declare Struct
// ================================================================================

/**
 * The cham-tool task struct for storing profiled-data.
 *
 * This struct helps collecting task-by-task info
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
    double core_freq;   // cpu-core frequency
} prof_task_info_t;


/**
 * The cham-tool profile-task data class.
 *
 * This class helps holding a list of prof-tasks.
 * @attributes: task_list, list for storing avg load per iter,
 *              size of the list, ...
 */
class prof_task_list_t {
    public:
        std::vector<prof_task_info_t *> task_list;
        std::vector<double> avg_load_list;
        std::atomic<size_t> tasklist_size;
        std::mutex m;
        
        // duplicate to avoid contention on single atomic 
        // from comm thread and worker threads
        std::atomic<size_t> dup_list_size;

        // constructor 1
        prof_task_list_t() {
            // set tasklist sizes
            tasklist_size = 0;
            dup_list_size = 0;

            // get the total num of iterations
            int n_iters;
            char *total_num_iters = std::getenv("EST_NUM_ITERS");
            if (total_num_iters != NULL)
                n_iters = atoi(total_num_iters);
            else
                n_iters = DEFAULT_NUM_ITERS;

            // resize and fill the lists
            avg_load_list.resize(n_iters, 0.0);
        }

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

        void add_avgload(double avg_value, int iter){
            this->m.lock();
            this->avg_load_list[iter] = avg_value;
            this->m.unlock();
        }
};


/**
 * The general regression model struct based on Torch-CXX
 *
 * @param num_features, num hidden layers, num_outputs
 * @return a corresponding prediction model
 */
#if ENABLE_PYTORCH_CXX == 1
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
#endif

/**
 * Example: how to declare the torch-based regression model.
 *      + auto net = std::make_shared<SimpleRegression>(4, 10, 1);
 *      + ...
 */


// ================================================================================
// Global Variables
// ================================================================================
prof_task_list_t profiled_task_list = prof_task_list_t();
std::vector<float> min_vec; // the last element is min_ground_truth
std::vector<float> max_vec; // the last element is max_ground_truth

// Declare a general mlpack regression model
mlpack::regression::LinearRegression lr;

// Declare 2 vectors for storing avg_load & pre_load (predicted loads)
arma::vec avg_load_per_iter_list;
arma::vec pre_load_per_iter_list;


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
	std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2)
    {
        return v[i1] < v[i2];
    });

	return idx;
}

/**
 * Paring the index numbers.
 *
 * This function give a unique number index from 2 input numbers.
 * For example, we have a=3, b=4, the unique index = 32.
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
 * The function will generate the profile-data log in .csv files. One file
 * per rank is written. TODO: which information could be written, should think about
 * this or somehow describe them at the application side (user-defined).
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

    // check output-path for writing logfile
    // printf("[TOOL] log_dir: %s\n", log_dir.c_str());

    // declare output file
    std::ofstream outfile;
    outfile.open(log_dir + "/chamtool_logfile_rank_" + std::to_string(mpi_rank) + ".csv");

    // create an iterator to traverse the list
    std::vector<prof_task_info_t *>::iterator it;

    // get through the list of profiled tasks
    for (it = tasklist_ref.task_list.begin(); it != tasklist_ref.task_list.end(); it++){

        // get a list of prob_sizes per task
        int num_args = (*it)->num_args;

        // string for storing prob_sizes
        std::string prob_sizes_statement;

#if SAMOA_EXAMPLE==1
        // the last 4 args affecting much on the taks runtime
        const int selected_arg = 8;
        for (int arg = selected_arg; arg < num_args; arg++){
            prob_sizes_statement += std::to_string((*it)->args_list[arg]) + "\t";
        }
#endif

#if MXM_EXAMPLE==1
        const int selected_arg = 0;
        prob_sizes_statement += std::to_string((*it)->arg_sizes.at(selected_arg)) + "\t";
#endif

        // cast profile-data to string for logging
        std::string line = std::to_string((*it)->tid) + "\t"
                            + prob_sizes_statement + "\t"
                            + std::to_string((*it)->core_freq) + "\n";
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
 * The task type is prof_task_info_t, each task, after being pushed into
 * the list (prof_task_list), would be allocated the memory. So, this function should
 * remove the allocated mem.
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
 * This function would pop all tasks in the profile-tasks list out, then
 * free the allocated memory of tasks one by one.
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
 * This function is simply to get the frequency value per core. The value 
 * is obtained from the system file, i.e., /proc/cpuinfo.
 * @param CPU core_id: the current core was executing the task.
 * @return freq_value: the frequency value from that corresponding core.
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
 * Such the tensor objects in Torch-CXX, this function is used to access
 * and print those values for checking or debugging.
 * @param tensor_arr, num_vals are an array of tensor values,
 *        and num of values we want to print.
 * @return I/O: tensor values
 */
#if ENABLE_PYTORCH_CXX == 1
void print_tensor(torch::Tensor tensor_arr, int num_vals)
{
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < num_vals; i++){
        std::cout << tensor_arr[i].item<float>() << std::endl;
    }
}
#endif

/**
 * Normalizing 2D-vector by column for training with Torch.
 *
 * The normalized 2D vector are retuned back to the argument by passing
 * references in this fucntion. To this end, this needs to find min and
 * max values of the input vector.
 * @return vec: a ref to the 2D vector
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
 * Online training the prediction model.
 *
 * @param tasklist_ref: a ref to the list of profiled tasks. But currently
 *        work for the runtime list, no need arguments of each task.
 * @param num_input_points: represent for num of input features
 * @param num_iters: the ended iter to train the prediction model
 * @return a dataset matrix
 */
bool online_mlpack_training(prof_task_list_t& tasklist_ref, int num_input_points, int num_iters)
{
    // the current list of finished tasks
    int size_tasklist = tasklist_ref.tasklist_size;
    // std::cout << "[CHAM_TOOL] gather_train_data: num finished tasks=" << size_tasklist
    //           << ", num finished iters=" << num_iters <<std::endl;

    // preparing data
    arma::vec runtime_list(num_iters);
    for (int i = 0; i < num_iters; i++){
        runtime_list[i] = tasklist_ref.avg_load_list[i];
    }
    // std::cout << "[CHAM_TOOL] gather_train_data: runtime_list" << std::endl;
    // std::cout << runtime_list << std::endl;

    // generate a std mat dataset for training
    int n_rows_X = num_input_points;
    int n_cols_X = num_iters - num_input_points;
    int n_rows_Y = 1;
    int n_cols_Y = n_cols_X;
    // std::cout << "[CHAM_TOOL] gather_train_data: trainX_size=" << n_rows_X << "x" << n_cols_X
    //           << ", trainY_size=" << n_rows_Y << "x" << n_cols_Y << std::endl;
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
 * Get the predicted values for the whole future.
 *
 * This means that with the input-start/end-points, the first iter of the main loop
 * in this function could return the predicted result of the expected current
 * cham-taskwait cycle/iter. Then, it will be used to continously create the input
 * and predict for the next iters until an end.
 * 
 * @param rank: the current rank is running this func, just for checking
 * @param start_point: the start chameleon-taskwait iter/cycle to make an input vector
 * @param end_point: the end chameleon-taskwait iter/cycle to make an input vector
 * @return &predicted_load_vec: a reference to the return-vector
 */
void get_whole_future_prediction(int rank, int start_point, int end_point, std::vector<double> &predicted_load_vec)
{
    int num_features = 6;   // TODO: get it by the setup-time or env-variables
    int end_program_iter = pre_load_per_iter_list.size(); // TODO: get this by the env-variable

    // the main loop for the whole future prediction with the first start point
    for (int i = end_point; i < end_program_iter; i++){
        // prepare the input vector for the model
        arma::vec input_vec(num_features);

        // assert (num_features back from the end_point, the value is exist)
        if (i == end_point){
            assert((i-num_features) >= 0);
            assert((avg_load_per_iter_list[i-num_features]) >= 0);
        }

        // get the previous loads as the input for predicting the load at end_point
        int s = i - num_features;
        int e = i;
        // std::string idx_iter_str_log = "(s" + std::to_string(s) + ",e" + std::to_string(e) + ")";
        // std::string val_iter_str_log = "";
        for (int j = s; j < e; j++){
            input_vec[j-s] = avg_load_per_iter_list[j];
            
            // for checking or logging
            // idx_iter_str_log += std::to_string(j) + " ";
            // val_iter_str_log += std::to_string(input_vec[j-s]) + " ";
        }

        // call the pred_model
        arma::mat x_mat(1, num_features);   // 1 row, num_features cols
        arma::rowvec p_load;                // for the result
        x_mat = input_vec;
        lr.Predict(x_mat, p_load);          // load the trained model
        predicted_load_vec[i] = p_load[0];  // save the result to return back to chameleon-lib side
        pre_load_per_iter_list[i] = p_load[0];

        // update the pred-value into the list of avg_load
        avg_load_per_iter_list[i] = p_load[0];

        // check this flow
        // std::cout << "[CHAM_TOOL] R" << rank << "-Iter" << end_point
        //           << ": input [" << idx_iter_str_log.c_str() << "]"
        //           << "= [" << val_iter_str_log.c_str() << "]"
        //           << "| output(i" << e << ") = " << p_load[0] << std::endl;
    }
}

/**
 * Get the predicted values by iter-by-iter
 *
 * This means that with the input-start/end-points, from the real-load of these input
 * points, the function would load the trained model and return the predicted result.
 * 
 * @param start_point: the start chameleon-taskwait iter/cycle to make an input vector
 * @param end_point: the end chameleon-taskwait iter/cycle to make an input vector
 * @return &predicted_load_vec: a reference to the return-vector
 */
void get_iter_by_iter_prediction(int start_point, int end_point, std::vector<double> &predicted_load_vec)
{

}
