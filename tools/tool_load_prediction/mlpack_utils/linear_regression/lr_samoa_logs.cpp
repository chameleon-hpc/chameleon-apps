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

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;

int main()
{
    // Path to the dataset used for training and testing.
    const std::string datasetPath = "./log_samoa_r0.tsv";
    // File for saving the model.
    const std::string modelFile = "./linear_regressor.bin";

    // Testing data is taken from the dataset in this ratio.
    constexpr double RATIO = 0.1; // 10%

    arma::mat dataset; // The dataset itself.
    arma::vec responses; // The responses, one row for each row in data.

    // In Armadillo rows represent features, columns represent data points.
    std::cout << "Reading data." << std::endl;
    bool loadedDataset = data::Load(datasetPath, dataset, true);
    // If dataset is not loaded correctly, exit.
    if (!loadedDataset)
        return -1;
    
    // Split the dataset into training and validation sets.
    arma::mat trainData, validData;
    data::Split(dataset, trainData, validData, RATIO);
    std::cout << "Checking train-valid/data: trainData.size=" << trainData.n_rows << ", validData.size=" << validData.n_rows << std::endl;

    // The train and valid datasets contain both - the features as well as the
    // prediction. Split these into separate matrices.
    arma::mat trainX =
        trainData.submat(1, 0, trainData.n_rows - 1, trainData.n_cols - 1);
    arma::mat validX =
        validData.submat(1, 0, validData.n_rows - 1, validData.n_cols - 1);

    // declare the model
    // mlpack::regression::LinearRegression lr(data, responses);

    // Get the parameters, or coefficients.
    // arma::vec parameters = lr.Parameters();

    return 0;
}