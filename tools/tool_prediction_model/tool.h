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

#include <torch/torch.h>
#include <iostream>
#include <cstddef>
#include <iomanip>

// instead of using mpi.h, we use bcl.h
#include <mpi.h>
// #include <bcl/bcl.hpp>
// #include <bcl/containers/FastQueue.hpp>

// #ifdef TRACE
#include "VT.h"
// static int event_tool_task_create = -1;
// static int event_tool_task_exec = -1;
// #endif

#ifndef NUM_SAMPLE
#define DEF_NUM_SAMPLE 100
#endif

#ifndef NUM_EPOCH
#define DEF_NUM_EPOCH 2000
#endif

/* #define TIMESTAMP(time_) 			\
	do {								\
		struct timespec ts;				\
		clock_gettime(CLOCK_MONOTONIC, &ts);							\
		time_ = ((double)ts.tv_sec) + (1.0e-9)*((double)ts.tv_nsec);	\
	} while(0) */


/**************** global variables *****************/
cham_t_task_list_t tool_task_list;
std::vector<float> min_vec; // the last element is min_ground_truth
std::vector<float> max_vec; // the last element is max_ground_truth
std::vector<std::vector<float>> norm_input;
std::vector<float> norm_ground_truth;
std::vector<float> runtime_list;
std::vector<float> pred_runtime_list;
bool is_model_trained = false;
int num_samples = DEF_NUM_SAMPLE;
int num_epochs = DEF_NUM_EPOCH;
int num_features = 1;


/**************** regression model definition *****/
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


/**************** create a global model ***********/
// TODO: get num of features here
auto net = std::make_shared<SimpleRegression>(6, 20, 1);


/**************** util-functions ******************/
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
#pragma endregion Local Helpers


/**************** writing logs ******************/
void chameleon_t_write_logs(cham_t_task_list_t *tool_task_list, int mpi_rank){
    // declare output file
    int i = 0;
    std::ofstream outfile;
    outfile.open("./logs/logfile_rank_" + std::to_string(mpi_rank) + "_" + std::to_string(num_samples) + "samples_" + std::to_string(num_epochs) + "epochs" + ".csv");
  
    // writing logs
    for (std::list<cham_t_task_info_t*>::iterator it=tool_task_list->task_list.begin(); it!=tool_task_list->task_list.end(); ++it) {
		// get waiting time
        float w_time = (*it)->start_time - (*it)->queue_time;
        if (w_time < 0) w_time = 0.0; // make sure the error values (still 0 but e^-oo)

        // get a list of prob_sizes per task
        int num_args = (*it)->arg_num;
        std::string prob_sizes_statement;
        for (int p = 0; p < num_args; p++){
            prob_sizes_statement += std::to_string((*it)->arg_sizes.at(p)) + "\t";
        }

    	// writing logs
        std::string line = std::to_string((*it)->task_id) + "\t"
                        + std::to_string((*it)->arg_num) + "\t"
                        + prob_sizes_statement
                        + std::to_string((*it)->processed_freq) + "\t"
                        + std::to_string((*it)->exe_time) + "\t"
                        + std::to_string(pred_runtime_list[i]) + "\n";
        outfile << line;

        // increase i
        i++;
    }

    // close file
    outfile.close();
}

/**************** get core frequencies ***************/
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
	}
	else
	{
		printf("Unable to open file!\n");
	}

	return freq;
}


/**************** check tensor values ***************/
void print_tensor(torch::Tensor tensor_arr, int num_vals)
{
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < num_vals; i++){
        std::cout << tensor_arr[i].item<float>() << std::endl;
    }
}


/**************** normalize 2d-vector by column *****/
void normalize_2dvector_by_column(std::vector<std::vector<float>> &vec)
{
    int num_rows = vec.size();
    int num_cols = (vec[0]).size();

    // find min-max vectors by each column
    for (int i = 0; i < num_cols; i++)
    {
        min_vec.push_back(10000000.0);
        max_vec.push_back(0.0);
    }

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            // check min
            if (vec[i][j] < min_vec[j])
                min_vec[j] = vec[i][j];

            // check max
            if (vec[i][j] > max_vec[j])
                max_vec[j] = vec[i][j];
        }
    }

    // normalize the whole vector
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            if (min_vec[j] == max_vec[j])
                vec[i][j] = 0.0;
            else
                vec[i][j] = (vec[i][j] - min_vec[j]) / (max_vec[j] - min_vec[j]);
        }
    }
}


/**************** normalize ground_truth *************/
void normalize_ground_truth(std::vector<float> &norm_ground_output)
{
    // find min-max values
    const auto [min, max] = std::minmax_element(runtime_list.begin(), runtime_list.end());

    // add min-max to the global min-max_vec
    min_vec.push_back(*min);
    max_vec.push_back(*max);

    // normalize the vector
    int vec_size = runtime_list.size();
    for (int i = 0; i < vec_size; i++)
    {
        float norm_val = (runtime_list[i] - *min) / (*max - *min);
        norm_ground_truth.push_back(norm_val);
    }
}


/**************** normalize input ********************/
void normalize_input(cham_t_task_list_t *tool_task_list, std::vector<std::vector<float>> result)
{
    // find min-max in arg_sizes
    int count = 0;
    int num_args = 0;
    std::list<cham_t_task_info_t*>::iterator it=tool_task_list->task_list.begin();
    for (int i = 0; i < num_samples; i++)
    {
        num_args = (*it)->arg_num;
        
        // create a vector of args
        std::vector<float> task_arg_sizes;
        for (int j = 0; j < num_args; j++)
            task_arg_sizes.push_back(float((*it)->arg_sizes.at(j)));

        // add core_freq
        task_arg_sizes.push_back(float((*it)->processed_freq));
        
        // push back the vector of args to a norm_input vector
        norm_input.push_back(task_arg_sizes);

        it++;       // move to the next task
        count++;    // increase count
    }

    // normalized 2d-vector by column
    normalize_2dvector_by_column(norm_input);
}

/**************** online training model **************/
auto online_training_model(std::vector<std::vector<float>> input, std::vector<float> ground_truth)
{
    // measure time
    double train_start_time = omp_get_wtime();

    // convert input to tensor type
    const int num_rows = input.size();
    const int num_cols = input[0].size();
    auto tensor_input = torch::from_blob(input.data(), {num_rows,num_cols});
    auto ground_truth_tensor = torch::from_blob(ground_truth.data(), {num_rows,1});

    // generate seed & learning_rate
    torch::manual_seed(1);
    const double learning_rate = 0.01;
    auto final_loss = 0.0;

    // create the model
    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(learning_rate));

    for (int epoch = 1; epoch <= num_epochs; epoch++) {
        auto out = net->forward(tensor_input);

        auto loss = torch::nn::functional::mse_loss(out, ground_truth_tensor);

        optimizer.zero_grad();

        loss.backward();

        optimizer.step();

        final_loss = loss.item<float>();
    }

    // change the global flag, the pre-model was trained
    is_model_trained = true;

    // get the measured time
    double train_end_time = omp_get_wtime();
    printf("Elapsed Time for training Pred-Model with N_SAMPLES=%d, N_EPOCH=%d: %.5f (s), LOSS= %.5f\n", num_rows, num_epochs, (train_end_time-train_start_time), final_loss);

    return net;
}