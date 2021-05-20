#include "chameleon_strategies.h"
#include "commthread.h" 
#include "chameleon_common.h"
 
#include <numeric>
#include <algorithm>
#include <cassert>
#include <iostream>

#pragma region Local Helpers
template <typename T>

/**
 * Sort the list of rank-orders by load
 *
 * This function will sort the order of ranks by current load, then
 * return a new list of ordered values with their indices.
 * @param &v: a reference to a vector of values with the type (std::vector<T>)
 * @return a sorted vector: std::vector<size_t>
 */
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


#pragma region Strategies

/**
 * Compute num tasks to offload
 *
 * This function will sort the order of ranks by current load, then
 * pairing ranks and estimate the number of tasks for offloading at once.
 * @param tasksToOffloadPerRank: contains the results, a list of tasks-num for offloading
 * @param loadInfoRanks: monitor-info about load per rank at runtime
 * @param num_tasks_local: the current amount of tasks for the corresponding rank
 * @param num_tasks_stolen: the amount of tasks in the stolen queue (or the remote queue)
 */
void compute_num_tasks_to_offload(std::vector<int32_t>& tasksToOffloadPerRank,
                                    std::vector<int32_t>& loadInfoRanks,
                                    int32_t num_tasks_local,
                                    int32_t num_tasks_stolen) {

#if OFFLOADING_STRATEGY_AGGRESSIVE
    int input_r = 0, input_l = 0;
    int output_r = 0, output_l = 0;

    int total_l = 0;
    total_l =std::accumulate(&loadInfoRanks[0], &loadInfoRanks[chameleon_comm_size], total_l);  
    int avg_l = total_l / chameleon_comm_size;
  
    input_l = loadInfoRanks[input_r];
    output_l = loadInfoRanks[output_r];

    while(output_r<chameleon_comm_size) {
        // TODO: maybe use this to compute load dependent on characteristics
        // of target rank (slow node..)
        int target_load_out = avg_l;
        int target_load_in = avg_l;

        while(output_l<target_load_out) {
            int diff_l = target_load_out-output_l;

            if(output_r==input_r) {
                input_r++; 
                input_l = loadInfoRanks[input_r];
                continue;
            }

            int moveable = input_l-target_load_in;
            if(moveable>0) {
                int inc_l = std::min( diff_l, moveable );
                output_l += inc_l;
                input_l -= inc_l;
 
                if(input_r==chameleon_comm_rank) {
                    tasksToOffloadPerRank[output_r]= inc_l;
                }
            }
       
            if(input_l <=target_load_in ) {
                input_r++;
                if(input_r<chameleon_comm_size) {
                    input_l = loadInfoRanks[input_r];
                    target_load_in = avg_l;
                }
            }
        }
        output_r++;
        if(output_r<chameleon_comm_size)
            output_l = loadInfoRanks[output_r];
    }
#else
    // Sort the order of rank indices by load
    // the output is a list of ranks sorted by load, e.g.,
    // [R1, R0, R3, R2] (Load_R1 <= Load R0 <= Load_R3 <= Load_R2)
    //  + min_val = load(R1)
    //  + max_val = load(R2)
    std::vector<size_t> tmp_sorted_idx = sort_indexes(loadInfoRanks);
    double min_val                  = (double) loadInfoRanks[tmp_sorted_idx[0]];
    double max_val                  = (double) loadInfoRanks[tmp_sorted_idx[chameleon_comm_size-1]];
    double cur_load                 = (double) loadInfoRanks[chameleon_comm_rank];
    double ratio_lb                 = 0.0; // 1 = high imbalance, 0 = no imbalance
    if (max_val > 0) {
        ratio_lb = (double)(max_val - min_val) / (double)max_val;
    }

#if !FORCE_MIGRATION
    // check absolute condition
    if((cur_load - min_val) < MIN_ABS_LOAD_IMBALANCE_BEFORE_MIGRATION)
        return;

    if(ratio_lb >= MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION) {

#else
    if(true) {
#endif
        // determine the index sorted by load of the current rank
        int pos = std::find(tmp_sorted_idx.begin(), tmp_sorted_idx.end(), chameleon_comm_rank) - tmp_sorted_idx.begin();

#if !FORCE_MIGRATION
        // only offload if on the upper side
        if((pos) >= ((double)chameleon_comm_size/2.0))
        {
#endif
            int other_pos       = chameleon_comm_size-pos-1;
            int other_idx       = tmp_sorted_idx[other_pos];
            double other_val    = (double) loadInfoRanks[other_idx];
            double cur_diff     = cur_load-other_val;

            // check pos what is this rank?
            int rank_pos        = tmp_sorted_idx[pos];

#if !FORCE_MIGRATION
            // check absolute condition
            if(cur_diff < MIN_ABS_LOAD_IMBALANCE_BEFORE_MIGRATION)
                return;

            // calculate the ratio for estimating num_tasks to migrate
            double ratio = cur_diff / (double)cur_load;

            if(other_val < cur_load && ratio >= MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION) {
#endif
                int num_tasks = (int)(cur_diff * PERCENTAGE_DIFF_TASKS_TO_MIGRATE);
                if(num_tasks < 1)
                    num_tasks = 1;
                
                // RELP("Migrating\t%d\ttasks from R%d to R%d\tload:\t%f\tload_victim:\t%f\tratio:\t%f\tdiff:\t%f\n",
                //                             num_tasks, rank_pos, other_idx, cur_load, other_val, ratio, cur_diff);
                tasksToOffloadPerRank[other_idx] = num_tasks;
#if !FORCE_MIGRATION
            }
        }
#endif
    }
#endif
}


/**
 * Pairing and computing the number of tasks for offloading, based on the prediction tool
 *
 * This function will sort the order of ranks by the predicted load in future, then
 * pairing ranks and estimate the number of tasks for offloading at once.
 * @param tasks_to_offload_per_rank: contains the results, a list of tasks-num for offloading
 * @param load_info_ranks: monitor-info about load per rank at runtime
 * @param predicted_load_info_ranks: predicted info about load per rank for the current exe-cycle
 * @param num_tasks_local: the current amount of tasks for the corresponding rank
 * @param num_tasks_stolen: the amount of tasks in the stolen queue (or the remote queue)
 */
void pair_num_tasks_to_offload(std::vector<int32_t>& tasks_to_offload_per_rank,
                                std::vector<int32_t>& load_info_ranks,
                                std::vector<double>& predicted_load_info_ranks,
                                int32_t num_tasks_local,
                                int32_t num_tasks_stolen) {

    // check pred_info list before sorting it
    int num_ranks = predicted_load_info_ranks.size();
    int tw_idx = _commthread_time_taskwait_count.load();

    // sort ranks by predicted load
    std::vector<size_t> tmp_sorted_idx = sort_indexes(predicted_load_info_ranks);
    double pred_min_load = predicted_load_info_ranks[tmp_sorted_idx[0]];
    double pred_max_load = predicted_load_info_ranks[tmp_sorted_idx[chameleon_comm_size-1]];
    double pred_cur_load = predicted_load_info_ranks[chameleon_comm_rank];

    // check pred-log info
    std::string rank_orders = "[";
    std::string pred_load_arr = "[";
    for (int i = 0; i < num_ranks; i++){
        int r = tmp_sorted_idx[i];
        rank_orders += "R" + std::to_string(r) + " ";
        pred_load_arr += std::to_string(predicted_load_info_ranks[r]) + " ";
    }
    rank_orders += "]";
    pred_load_arr += "]";
    // RELP("[PAIR_DBG] Iter%d: %s = %s\n", tw_idx, rank_orders.c_str(), pred_load_arr.c_str());

    // init and compute the ratio of load-balancing
    double ratio_lb = 0.0;  // 1.0 is high, 0.0 is no imbalance
    if (pred_max_load > 0)
        ratio_lb = (pred_max_load - pred_min_load) / pred_max_load;

    if (ratio_lb >= MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION){
        // determine the postition of the current rank in progress
        int cur_pos = std::find(tmp_sorted_idx.begin(), tmp_sorted_idx.end(), chameleon_comm_rank) - tmp_sorted_idx.begin();

        // decide to offload, but for safe, only offload the upper side
        if ( cur_pos >= ((double)chameleon_comm_size / 2.0) ) {
            // identify the victim to offload tasks there
            int vic_pos = chameleon_comm_size - cur_pos - 1;
            int vic_rank = tmp_sorted_idx[vic_pos];
            double vic_pred_load = predicted_load_info_ranks[vic_rank];
            double load_diff = pred_cur_load - vic_pred_load;
            double load_diff_ratio = load_diff / pred_cur_load;
            double load_diff_percentage = load_diff_ratio * 100;

            // check the estimation
            // RELP("[PAIR_TASKS_OFFLOAD] vic_pos=%d, vic_rank=%d: vic_pred_load=%.3f | load_diff=%.2f%\n",
            //                            vic_pos, vic_rank, vic_pred_load, load_diff_percentage);

            // check the absolute condition with the constant of diff_load to migrate tasks
            // RELP("[PAIR_TASKS_OFFLOAD] load_diff between src=R%d & vic=R%d is %.3f %\n",
            //                         chameleon_comm_rank, vic_rank, load_diff_percentage);
            if (load_diff_ratio < MIN_LOAD_DIFF_PERCENTAGE_TO_MIGRATION)
                return;

            // if satify the absolute condition, then estimate num tasks to offload
            if (vic_pred_load < pred_cur_load && load_diff_ratio >= MIN_LOAD_DIFF_PERCENTAGE_TO_MIGRATION) {
                // compute the avg_load of a single task per rank
                // TODO: adapt dynamically when running with the large-scale experiments
                //       + currently, this is being checked on a note with 2 ranks, 3 threads / rank
                //       + the use-case: samoa-osc (aderdg-opt version), section=16, num time-steps = 20/25
                //       + so, the total amount of tasks per rank is 48, total num iters <= 100
                const int max_tasks_per_rank = MAX_TASKS_PER_RANK;
                double avg_load_per_task_on_src = pred_cur_load / max_tasks_per_rank;

                // from the load_diff, compute the max number of tasks could be migrated at the victim side
                int max_num_tasks = (int)(load_diff / avg_load_per_task_on_src);

                // set a threshold, an approriate number of tasks should be migrated
                //      + currently, try to set 40% of the diff-load
                //      + TODO: which % should be good?
                int appro_num_tasks = (int)(max_num_tasks * 0.4);
                if (appro_num_tasks >= max_tasks_per_rank)
                    appro_num_tasks = 1;

                // check the estimation
                RELP("[PAIR_DBG] Iter%d: src(R%d) [max_migtasks=%d] should migrate %d tasks to victim(R%d)\n",
                                            tw_idx, chameleon_comm_rank, max_num_tasks, appro_num_tasks, vic_rank);

                // prevent the num_tasks < 1
                if (appro_num_tasks < 1)
                    appro_num_tasks = 1;
                
                // return the results
                tasks_to_offload_per_rank[vic_rank] = appro_num_tasks; // app_num_tasks;
            }
        }
    }
}


/**
 * Compute the number of tasks for replication
 *
 * This function will implement the default replication strategy where neighbouring ranks
 * logically have some "overlapping tasks".
 * @param loadInfoRanks: monitor-info about load per rank at runtime
 * @param num_tasks_local: the current amount of tasks for the corresponding rank
 * @param num_replication_infos: the current info about the number of replicated tasks
 */
 cham_t_replication_info_t * compute_num_tasks_to_replicate(  std::vector<int32_t>& loadInfoRanks,
                                int32_t num_tasks_local,
                                int32_t *num_replication_infos ) {

    double alpha = 0.0;
    int myLeft = (chameleon_comm_rank-1 + chameleon_comm_size)%chameleon_comm_size;
    int myRight = (chameleon_comm_rank+1 + chameleon_comm_size)%chameleon_comm_size;
    
    assert(num_tasks_local>=0);

    int num_neighbours = 0;
    if(myLeft>=0) num_neighbours++;
    if(myRight<chameleon_comm_size) num_neighbours++;
    cham_t_replication_info_t *replication_infos = (cham_t_replication_info_t*) malloc(sizeof(cham_t_replication_info_t)*num_neighbours);

    alpha = MAX_PERCENTAGE_REPLICATED_TASKS/num_neighbours;
    assert(alpha>=0);

    int32_t cnt = 0;

	if(myLeft>=0) {
	    //printf("alpha %f, num_tasks_local %d\n", alpha, num_tasks_local);
	    int num_tasks = num_tasks_local*alpha;
            assert(num_tasks>=0);
	    int *replication_ranks = (int*) malloc(sizeof(int)*1);
	    replication_ranks[0] = myLeft;
		cham_t_replication_info_t info = cham_t_replication_info_create(num_tasks, 1, replication_ranks);
		replication_infos[cnt++] = info;
	}
	if(myRight<chameleon_comm_size) {
	    int num_tasks = num_tasks_local*alpha;
            assert(num_tasks>=0);
	    int *replication_ranks = (int*) malloc(sizeof(int)*1);
	    replication_ranks[0] = myRight;
	    cham_t_replication_info_t info = cham_t_replication_info_create(num_tasks, 1, replication_ranks);
	    replication_infos[cnt++] = info;
	}
	*num_replication_infos = cnt;
	return replication_infos;
}


/**
 * Get the default load information of the rank
 *
 * This function will simply return number of tasks in queue.
 * @param local_task_ids:
 * @param num_tasks_local:
 * @param local_rep_task_ids:
 * @param num_tasks_local_rep:
 * @param stolen_task_ids:
 * @param num_tasks_stolen:
 * @param stolen_task_ids_rep:
 * @param num_tasks_stolen_rep:
 */
int32_t get_default_load_information_for_rank(TYPE_TASK_ID* local_task_ids, int32_t num_tasks_local,
                                        TYPE_TASK_ID* local_rep_task_ids, int32_t num_tasks_local_rep,
                                        TYPE_TASK_ID* stolen_task_ids, int32_t num_tasks_stolen,
                                        TYPE_TASK_ID* stolen_task_ids_rep, int32_t num_tasks_stolen_rep) {
    int32_t num_ids;
    assert(num_tasks_stolen_rep>=0);
    assert(num_tasks_stolen>=0);
    assert(num_tasks_local_rep>=0);
    assert(num_tasks_local>=0);

    //Todo: include replicated tasks which are "in flight"
    num_ids = num_tasks_local + num_tasks_local_rep;

#if CHAM_REPLICATION_MODE==1
    num_ids += num_tasks_stolen + num_tasks_stolen_rep;
#else
    num_ids += num_tasks_stolen;
#endif

    return num_ids;
}
#pragma endregion Strategies
