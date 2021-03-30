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
#include <iostream>
#include <fstream>

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;

/* Reading and loading a matrix from csv file
    - input: is file path & delimeter
    - output: the mat type from armadillo-lib */
arma::mat read_mat_csv(const std::string &filename, const std::string &delimeter = ","){

    // file name & data
    std::ifstream csv_file(filename);
    std::vector<std::vector<double>> data_batch;

    // read file line by line
    for(std::string line; std::getline(csv_file, line); ) {
        // a vector to store data_points
        std::vector<double> datapoint_vector;

        // split string by delimeter
        auto start = 0U;
        auto end = line.find(delimeter);

        // get through a line
        while (end != std::string::npos) {
            datapoint_vector.push_back(std::stod(line.substr(start, end - start)));
            start = end + delimeter.length();
            end = line.find(delimeter, start);
        }
        datapoint_vector.push_back(std::stod(line.substr(start, end)));
        data_batch.push_back(datapoint_vector);
    }

    arma::mat data_mat = arma::zeros<arma::mat>(data_batch.size(), data_batch[0].size());

    int n = data_batch.size();
    for (int i = 0; i < n; i++) {
        arma::mat r(data_batch[i]);
        data_mat.row(i) = r.t();
    }

    return data_mat;
}

/* Main function
    - there're 3 phases: reading data, training, validating
    - output: could store the trained model somewhere */
int main(int argc, char* argv[])
{
    // Path to the dataset used for training and testing.
    std::string datasetPath;
    if (argc < 2) {
        std::cout << "Usage: ./lr_regression <path_to_dataset_file>" << std::endl;
        exit(1);
    } else
        datasetPath = argv[1];

    // File for saving the model.
    const std::string modelFile = "./linear_regressor.bin";

    arma::mat dataset; // The dataset itself.
    arma::vec responses; // The responses, one row for each row in data.

    // In Armadillo rows represent features, columns represent data points.
    std::cout << "1. Reading data." << std::endl;
    bool loadedDataset = data::Load(datasetPath, dataset);
    // If dataset is not loaded correctly, exit.
    if (!loadedDataset)
        return -1;
    
    std::cout << "\t Size of the whole dataset: " << dataset.size()
            << " | " << dataset.n_rows << "x" << dataset.n_cols << std::endl;

    // So, plit it by myself, note that armadillo stores data by row-major

    arma::mat trainData, validData;
    const int T_FROM_ROW = 0, T_TO_ROW = 6;
    const int T_FROM_COL = 0, T_TO_COL = 49;
    const int V_FROM_ROW = 0, V_TO_ROW = 6;
    const int V_FROM_COL = 50, V_TO_COL = 193;
    trainData = dataset.submat(T_FROM_ROW, T_FROM_COL, T_TO_ROW, T_TO_COL);
    validData = dataset.submat(V_FROM_ROW, V_FROM_COL, V_TO_ROW, V_TO_COL);
    std::cout << "\t Checking train-valid/data: "
            << "trainData.size=" << trainData.n_rows << "x" << trainData.n_cols
            << ", validData.size=" << validData.n_rows << "x" << validData.n_cols
            << std::endl;

    // std::cout << "------------------------------" << std::endl;
    // std::cout << trainData.row(0) << std::endl;
    // std::cout << validData.row(0) << std::endl;
    // std::cout << "------------------------------" << std::endl;

    // The train and valid datasets contain both - the features as well as the
    // prediction. Split these into separate matrices.
    arma::mat trainX =
        trainData.submat(0, 0, trainData.n_rows-2, trainData.n_cols-1);
    arma::mat validX =
        validData.submat(0, 0, validData.n_rows-2, validData.n_cols-1);
    std::cout << "\t Checking train/valid_X: "
            << "trainX.size=" << trainX.n_rows << "x" << trainX.n_cols
            << ", validX.size=" << validX.n_rows << "x" << validX.n_cols
            << std::endl;

    // Create prediction data for training and validatiion datasets.
    arma::rowvec trainY = trainData.row(trainData.n_rows-1);
    arma::mat validY = validData.row(validData.n_rows-1);
    std::cout << "\t Checking train/valid_Y: "
            << "trainY.size=" << trainY.n_rows << "x" << trainY.n_cols
            << ", validY.size=" << validY.n_rows << "x" << validY.n_cols
            << std::endl;
    
    // showing trainY, validY
    // std::cout << "---------------------------------" << std::endl;
    // std::cout << "Trying to show trainY, validY:" << std::endl;
    // std::cout << trainY << std::endl;
    // std::cout << validY << std::endl;

    // declare the built-in model
    mlpack::regression::LinearRegression lr(trainX, trainY);

    // Get the parameters, or coefficients.
    // arma::vec parameters = lr.Parameters();

    // check prediction
    arma::rowvec pred_values;
    lr.Predict(validX, pred_values);
    std::cout << "2. Checking the prediction model." << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    std::cout << "--------------Prediction---------------" << std::endl;
    std::cout << pred_values << std::endl;

    std::cout << "---------------------------------------" << std::endl;
    std::cout << "--------------Groundtruth--------------" << std::endl;
    std::cout << validY << std::endl;

    // try with a single value
    arma::rowvec single_value;
    const int col_idx = 5;
    lr.Predict(validX.col(col_idx), single_value);
    std::cout << "3. Checking with a single input." << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    std::cout << "\t Input: " << validX.col(col_idx).t() << std::endl;
    std::cout << "\t Groundtruth: " << validY.col(col_idx) << std::endl;
    std::cout << "\t Prediction: " << single_value << std::endl;

    return 0;
}