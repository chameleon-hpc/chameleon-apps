## Chameleon Prediction Tool
This is a callback tool for predicting task-runtime with an online/offline trained model. Depending on each kind of application, we select the features for the model. For example, the current use-case is iterative simulation, named Sam(oa)2, the input features could be boundary-sizes per task, or list of boundary-sizes based on execution oders per iteration, or simply the previous points of load as time progress. The working flow of Chameleon and the tool, for example, as follow:
<p align="left">
  <img src="./figures/cham-tool-workflow.png" alt="The working flow of the prediction model" width="700">
</p>

## Dependencies
At the current status, there're 2 options for machine-learning librabies:
* Pytorch C++ (https://pytorch.org/cppdocs/installing.html)
  * Don't need to install, just download and point to where is it at when compiling the tool.
  * E.g., as the sample compile-script in build/
  * Temporarily, I prefer mlpack c++ below, and turn this off.
* Mlpack C++ (https://www.mlpack.org/getstarted.html)
  * Need to install it with the dependencies (Armadillo, Boost, ensmallen)
  * Could follow here https://www.mlpack.org/doc/mlpack-3.4.2/doxygen/build.html

## Package organization
The code is simply organized with its utils as follow:
* build/: a sample script to link and build the tool with Chameleon. TODO: need to adapt dependencies at your side.
* chameleon_patch/: simply just the src-code of the Chameleon lib (latest version) with some changes to fit the prediction tool. To avoid hurting the original version of Chameleon, so leave them here temporarily. Note: need to replace them with the original one when we compile.
* mlpack_utils/: some examples with mlpack library to build the regression models.
* python_utils/: some examples with scikit-learn/mlpack lib in Python, to build and test the regression models.

## How it works
As the diagram above, the tool works as the plugin of Chameleon lib (sounds like the event-based working flow). When we need a callback event, we need to define it, determine when it's called and what should it return back to the chameleon-lib side or not. Therefore, it could be managed as:
* Define the callback event associated with its function (action).
* When it's called, should control who calls it (comm_thread or execution threads).
* What should it give back to the cham-lib side.

## Compiling the tool
For example, could follow the sample compile-script in build/ folder, and need to adapt the dependencies that are declared in CMakeLists.txt (at the src/ folder).

## Compiling Chameleon & Linking with the tool
There could be a sample script to compile Chameleon with the callback tools (to be updated). Btw, there're some steps:
* Copy and replace the original version of Chameleon src-code with the files in chameleon_patch/, could use another separate folder quickly.
* Loading dependencies: libffi, hwloc, and the corresponding compiler (e.g., Intel).
* Set env-flags for Chameleon tool:
  * CHAMELEON_TOOL=1
  * CHAMELEON_TOOL_LIBRARIES=/path/to/the-compiled-tool (.so)
* Regarding the Chameleon-lib internal flags (as migration, replication modes), please set:
  * -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DCHAMELEON_TOOL_SUPPORT
  * A new variable is CHAM_PRED_MIGRATION (as another mode of migration), it should be turned into 1, e.g., -DCHAM_PRED_MIGRATION=1
* If everything is fine, then compile Chameleon.

## Test the tool & Chameleon
Currently, the testcase is samoa-osc (the aderdg-opt version). TODO: merge the example of mxm.

## Evaluate the prediction tool
The current usecase is Samoa-ADERDG-OPT (https://gitlab.lrz.de/samoa/samoa/-/tree/ADER-DG-opt) with Oscillating-Lake scenario. The following test was performed on CoolMUC2 (LRZ), 16 nodes, 2 ranks per node, 14 threads per rank. The line charts show the comparison between real-load and predicted-load by the tool. Note: the simulation was running with 100 time-steps, i.e., R8 to R11 are shown below, the results of other ranks could find in /python_utils/figures/.
<p align="left">
  <img src="./figures/osc_samoa_pred_load.png" alt="Predicted load with real load" width="700">
</p>