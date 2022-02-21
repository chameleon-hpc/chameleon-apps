## Chameleon Load Prediction Tool & Proactive Task Offloading Strategy
The document shows how the online prediction scheme and proactive task offloading work. We design the module as a callback tool outside Chameleon depending on domain-specific applications. The examples are illustrated through a synthetic test case (MxM) and an iterative simulation named Sam(oa)2 (adaptive mesh refinement with solving partial differential equations - PDEs).

## Working Flow
<p align="left">
  <img src="./figures/react_and_proact_loadbalancing.svg" alt="The working flow" width="700">
</p>

Figure (A) shows the reactive load balancing approach, where the dedicated thread ($Tcomm$) is used to continuously monitor the execution speed/rank and react migrating tasks when the imbalance happens. In contrast, Figure (B) reveals the proactive approach. $Tcomm$ characterizes task and runtime in several first iterations ($char$, $data\_collect$), then train the prediction model ($trainM$) and load it afterwards ($loadM$). When the model is loaded, the prediction results of load per rank will input to the proactive algorithm (proact\_mig), which can guide task migration/offloading.

## An Example with Proactive Task Offloading Algorithm
<p align="left">
  <img src="./figures/proact_task_mig_algorithm.svg" alt="Proactive Alogrithm Illustration" width="650">
</p>

After the prediction phase, its result is transferred to the algorithm. The first step is sorting the involved ranks by the exchanged values of load prediction (where $load$ accounts for the wall clock execution time of a specific iteration, e.g., $W10_{1}$ means the wall clock execution time of Rank 1 in Iteration 10). We need an array to record the total load of tasks executed at the local rank (\textit{local tasks}), and another one for tasks executed at the remote rank (\textit{remote tasks}). Furthermore, a table is used to track the number of local and remote tasks.

In general, the algorithm consists of two $for$-loops. The first loop will go through the victims ($v_{i}$) which have $L < T_{opt}$. In the second loop, each offloader ($`r_{j}`$), who has $L > T_{opt}$, will be traversed. $\delta_{overloaded}$ is the load difference between an offloader and $T_{opt}$, while $\delta_{underloaded}$ is the one between a victim and $T_{opt}$. After that, we can compute the number of tasks that should be offloaded to fill the gap of $\delta_{underloaded}$ at the victim side. If the current offloader does not have enough tasks to fill $\delta_{underloaded}$ up, the next one will be processed.

The figure above shows an imbalance case of 8 ranks with uniform tasks. The number of tasks per rank causes the given imbalance. We assume that the prediction information is ready here; the inputs of the proactive algorithm are total load ($W_{p,M}$) and the number of tasks in the queue per rank. As we can see in the first step (\textit{Init step}), $LoadArr(local)$ holds the total load, $TrackingTable(8,8)$ indicates 8 ranks involved, and the diagonal line points to the number of local tasks associated with a corresponding rank. After sorting the predicted load (descending), the order of ranks is $R0, R1, R6, R7, R2, R3, R4, R5$.
The first loop goes to the victim - $R5$, and the first offloader is $R0$. Next, we estimate that $R0$ should offload 113 tasks to $R5$ based on the values of $T_{opt}$, $\delta_{overloaded}$, $\delta_{underloaded}$. The new load values of $local$ \& $remote$ are updated; the tracking table also needs to update the number of migrated tasks at the current offloader row (0 for $R0$). Finally, the output shows that $R0$ should migrate 64, 96, ... tasks to the corresponding victims ($R1$, $R2$, ...).

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
* src/: the src-code of the tool.

## How it works And Configurations
As the diagram above, the tool works as the plugin of Chameleon lib (sounds like the event-based working flow). When we need a callback event, we need to define it, determine when it's called and what should it return back to the chameleon-lib side or not. Therefore, it could be managed as:
* Define the callback event associated with its function (action).
* When it's called, should control who calls it (comm_thread or execution threads).
* What should it give back to the cham-lib side.

As some changes in the folder - chameleon_patch/, there are 2 ENV-variables (defined in `chameleon_patch/chameleon_common.h`) to define prediction modes and migration modes along with the prediction.
``` CXX
// specify the method for predicting load by the callback tool
#ifndef CHAM_PREDICTION_MODE
#define CHAM_PREDICTION_MODE 0      // no prediction
// #define CHAM_PREDICTION_MODE 1   // time-series load as the patterns for prediction
// #define CHAM_PREDICTION_MODE 2   // time-series load as the patterns, use predicted values to predict the whole future
// #define CHAM_PREDICTION_MODE 3   // task-characterization, args as the patterns for prediction
#endif

// specify the strategy of work-stealing with prediction tool
#ifndef CHAM_PROACT_MIGRATION
#define CHAM_PROACT_MIGRATION 0   // predict iter-by-iter, no migration action
// #define CHAM_PROACT_MIGRATION 1    // predict iter-by-iter, then migrate-actions 
// #define CHAM_PROACT_MIGRATION 2    // predict for the whole future, then migrate-actions
#endif
```

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
  * A new variable is CHAM_PROACT_MIGRATION (as another mode of migration), it should be turned into 1, e.g., -DCHAM_PROACT_MIGRATION=1
* If everything is fine, then compile Chameleon.

## Test the tool & Chameleon
Currently, the testcase is samoa-osc (the aderdg-opt version). TODO: merge the example of mxm.

## Evaluate the prediction tool
The current usecase is Samoa-ADERDG-OPT (https://gitlab.lrz.de/samoa/samoa/-/tree/ADER-DG-opt) with Oscillating-Lake scenario. The following test was performed on CoolMUC2 (LRZ), 16 nodes, 2 ranks per node, 14 threads per rank. The line charts show the comparison between real-load and predicted-load by the tool. Note: the simulation was running with 100 time-steps, i.e., R8 to R11 are shown below, the results of other ranks could find in /python_utils/figures/.
<p align="left">
  <img src="./figures/osc_samoa_pred_load.png" alt="Predicted load with real load" width="700">
</p>
