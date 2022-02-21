#include "chameleon_strategies.h"
#include "commthread.h" 
#include "chameleon_common.h"
 
#include <numeric>
#include <algorithm>
#include <cassert>

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

std::list<std::string> split(const std::string& s, char delimiter)
{
    std::list<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
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

        // TODO: maybe use this to compute load dependent on characteristics of target rank (slow node..)
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
    double min_val = (double) loadInfoRanks[tmp_sorted_idx[0]];
    double max_val = (double) loadInfoRanks[tmp_sorted_idx[chameleon_comm_size-1]];
    double cur_load = (double) loadInfoRanks[chameleon_comm_rank];
    double ratio_lb = 0.0; // 1 = high imbalance, 0 = no imbalance
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
        // determine the index of the current rank sorted by load
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

            // check pos what is this rank
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

                // check the sorted load at the moment
                // std::string qload_logs = "Check load-progress: ";
                // for(int i = 0; i < chameleon_comm_size; i++){
                //     qload_logs += "R" + std::to_string(tmp_sorted_idx[i]) + "," + std::to_string(loadInfoRanks[tmp_sorted_idx[i]]) + "\t";
                // }
                // RELP("%s\n", qload_logs.c_str());
                
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

void compute_num_tasks_to_steal(std::vector<int32_t>& tasks_to_offload_per_rank,
                                    std::vector<int32_t>& load_info_ranks,
                                    int32_t num_tasks_local,
                                    int32_t num_tasks_stolen)
{
    // calculate sum of load
    int sum_load = 0.0;
    for (int i = 0; i < load_info_ranks.size(); i++){
        sum_load += load_info_ranks[i];
    }

    // find min - max load
    std::vector<size_t> tmp_sorted_idx = sort_indexes(load_info_ranks);
    double min_load = (double)load_info_ranks[tmp_sorted_idx[0]];
    double max_load = (double)load_info_ranks[tmp_sorted_idx[chameleon_comm_size-1]];

    // calculate average load
    double avg_load = (double) sum_load / chameleon_comm_size;

    // calculate imb ratio
    double imb_ratio = (max_load / avg_load) - 1;
    // RELP("R%d: max_load=%.2f, avg_load=%.2f, imb_ratio=%.2f\n", chameleon_comm_rank, max_load, avg_load, imb_ratio);
    // just use to check for 4 ranks
    // RELP("R[%d,%d,%d,%d] = [%d,%d,%d,%d], max_load=%.2f, avg_load=%.2f, imb_ratio=%.2f\n",
    //         0, 1, 2, 3, load_info_ranks[0], load_info_ranks[1], load_info_ranks[2], load_info_ranks[3],
    //         max_load, avg_load, imb_ratio);

    if (imb_ratio > 0.2 && max_load >= (avg_load+2) && min_load == 0) {

        // init a vector of overloaded ranks
        std::vector<int> overloaded_ranks;
        std::vector<int> empty_ranks;
        for (int i = 0; i < load_info_ranks.size(); i++){
            if ((double)load_info_ranks[i] > (avg_load+2))
                overloaded_ranks.push_back(i);
            else if (load_info_ranks[i] == 0)
                empty_ranks.push_back(i);
        }

        // for empty ranks
        if (num_tasks_local == 0 && overloaded_ranks.size() > 0){
            // choose victim randomly for stealing tasks
            int victim_idx = 0;
            int victim_rank = overloaded_ranks[victim_idx];
            if (overloaded_ranks.size() > 1){
                srand(time(NULL));
                victim_idx = rand() % (overloaded_ranks.size()-1) + 0;
                victim_rank = overloaded_ranks[victim_idx];
            }
            // RELP("R%d(L%d)-RANDVICTIM to steal tasks: R%d(L%d)\n", chameleon_comm_rank, num_tasks_local, victim_rank, load_info_ranks[victim_rank]);
        } // for overloaded ranks
        else if ((double)num_tasks_local >= (avg_load+2) && empty_ranks.size() > 0){
            // choose recv randomly for sending tasks
            int recv_idx = 0;
            int recv_rank = empty_ranks[recv_idx];
            if (empty_ranks.size() > 1){
                srand(time(NULL));
                recv_idx = rand() % (empty_ranks.size()-1) + 0;
                recv_rank = empty_ranks[recv_idx];
            }
            // RELP("R%d(L%d)-RANDRECV to send tasks: R%d(L%d)\n", chameleon_comm_rank, num_tasks_local, recv_rank, load_info_ranks[recv_rank]);
            // RELP("R[%d,%d,%d,%d] = [%d,%d,%d,%d], max_load=%.2f, avg_load=%.2f, imb_ratio=%.2f\n",
            //             0, 1, 2, 3, load_info_ranks[0], load_info_ranks[1], load_info_ranks[2], load_info_ranks[3],
            //             max_load, avg_load, imb_ratio);

            // decide num tasks to steal
            int num_task = 1;
            tasks_to_offload_per_rank[recv_rank] = 1;
        }
    }
}


/**
 * Pairing and computing the number of tasks for offloading, based on the prediction tool
 *
 * This function will sort the order of ranks by the predicted load in future, then
 * pairing ranks and estimate the number of tasks for offloading at once.
 * @param proact_tasks_to_offload_table: a tracking table for proactive task migration algorithm
 * @param tasks_to_offload_per_rank: contains the results, a list of tasks-num for offloading
 * @param load_info_ranks: monitor-info about load per rank at runtime
 * @param predicted_load_info_ranks: predicted info about load per rank for the current exe-cycle
 * @param num_tasks_local: the current amount of tasks for the corresponding rank
 * @param num_tasks_stolen: the amount of tasks in the stolen queue (or the remote queue)
 */
void compute_num_tasks_to_proact_offload(std::vector<int32_t>& proact_tasks_to_offload_table,
                                std::vector<int32_t>& tasks_to_offload_per_rank,
                                std::vector<int32_t>& load_info_ranks,
                                std::vector<double>& predicted_load_info_ranks,
                                int32_t num_tasks_local,
                                int32_t num_tasks_stolen) {

#if CHAMELEON_TOOL_SUPPORT && CHAM_PREDICTION_MODE > 0 && CHAM_PROACT_MIGRATION > 0

    // check pred_info list before sorting it
    int num_ranks = predicted_load_info_ranks.size();
    int tw_idx = _commthread_time_taskwait_count.load();

    #if DEBUG_PROACTIVE_MIGRATION == 1
    // get debug_load per rank
    std::vector<double> debug_predload_arr1{0.0186,2.9951,7.5228,0.0161,0.0171,13.4808,13.4969,0.0189,0.0226,7.4983,2.9926,0.0168,0.0171,1.5873,1.5882,0.0159};
    std::vector<double> debug_predload_arr2{0.0554,0.7705,1.7219,0.0538,0.0548,2.8979,2.8976,0.0565,0.0611,1.7169,0.7654,0.0522,0.0559,0.4604,0.4634,0.0587};
    // char *debug_predload_arr1 = std::getenv("DBG_PRELOAD_ARR1");
    // char *debug_predload_arr2 = std::getenv("DBG_PRELOAD_ARR2");
    // if (debug_predload_arr1 != NULL && debug_predload_arr2 != NULL){
    //     std::string str1_dbg_loads(debug_predload_arr1);
    //     std::string str2_dbg_loads(debug_predload_arr2);
    //     std::list<std::string> split_dbgloads1 = split(str1_dbg_loads, ',');
    //     std::list<std::string> split_dbgloads2 = split(str2_dbg_loads, ',');
    //     std::list<std::string>::iterator it = split_dbgloads1.begin();
        // if (tw_idx % 2 != 0)
        //     it = split_dbgloads2.begin();
        // for (int i = 0; i < num_ranks; i++){
        //     advance(it, i);
        //     double dbg_load = std::atof((*it).c_str());
        //     predicted_load_info_ranks[i] = dbg_load;
        // }
    // }
    std::vector<double> pred_value = debug_predload_arr1;
    if (tw_idx % 2 != 0)
        pred_value = debug_predload_arr2;
    for (int i = 0; i < num_ranks; i++){
        predicted_load_info_ranks[i] = pred_value[i];
    }
    #endif

    // sort ranks by predicted load
    std::vector<size_t> tmp_sorted_idx = sort_indexes(predicted_load_info_ranks);
    double pred_min_load = predicted_load_info_ranks[tmp_sorted_idx[0]];
    double pred_max_load = predicted_load_info_ranks[tmp_sorted_idx[chameleon_comm_size-1]];
    double pred_cur_load = predicted_load_info_ranks[chameleon_comm_rank];

    // calculate average load of the pred_load_vector
    double pred_sum_load = 0.0;
    for (int i = 0; i < num_ranks; i++)
        pred_sum_load += predicted_load_info_ranks[i];
    double pred_avg_load = pred_sum_load / num_ranks;

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
    RELP("[PROACT_MIGRATION] Iter%d: %s = %s\n", tw_idx, rank_orders.c_str(), pred_load_arr.c_str());

    // init and compute the ratio of load-balancing
    double ratio_lb = 0.0;  // 1.0 is high, 0.0 is no imbalance
    double imb_ratio = 0.0;
    if (pred_max_load > 0) {
        // imb ration by max and min load
        ratio_lb = (pred_max_load - pred_min_load) / pred_max_load;

        // imb ration by max and avg load
        imb_ratio = (pred_max_load - pred_avg_load) / pred_avg_load;
    } 
    // RELP("[PROACT_MIGRATION] Iter%d: orig_imb_ratio=%.3f, threshold_avg_load_per_rank=%.3f\n", tw_idx, imb_ratio, pred_avg_load);

    
    // refill the tracking table and init load_per_rank vectors
    std::vector<double> local_load_vector(num_ranks, 0.0);
    std::vector<double> remote_load_vector(num_ranks, 0.0);
    std::vector<double> sum_load_vector(num_ranks, 0.0);
    for (int i = 0;  i < num_ranks; i++){
        for (int j = 0; j < num_ranks; j++){
            if (i == j){
                proact_tasks_to_offload_table[i*num_ranks + j] = num_tasks_local;
                local_load_vector[i] = predicted_load_info_ranks[i];
            }
            else {
                proact_tasks_to_offload_table[i*num_ranks + j] = 0;
            }
        }
    }

    // std::string tracking_table_str = "[";
    // for (int i = 0; i < num_ranks; i++){
    //     for (int j = 0; j < num_ranks; j++){
    //         tracking_table_str += std::to_string(proact_tasks_to_offload_table[i*num_ranks+j]) + " ";
    //     }
    // }
    // tracking_table_str += "]";
    // RELP("[PROACT_MIGRATION] Check the tracking table: %s\n", tracking_table_str.c_str());

    /* DIST_GREADY_KNAPSACK ALGORITHM */
    for (int i = 0; i < num_ranks; i++){

        // from the most underloaded rank
        int victim_rank = tmp_sorted_idx[i];
        double victim_load = local_load_vector[victim_rank] + remote_load_vector[victim_rank];
        if (victim_load < pred_avg_load){
            double underload = pred_avg_load - victim_load;
            // RELP("[PROACT_MIGRATION] VICTIM: R%d, load=%.3f, underload=%.3f\n", victim_rank, victim_load, underload);

            // from the most overloaded rank
            for (int j = num_ranks-1; j >= 0; j--){
                int offloader_rank = tmp_sorted_idx[j];
                double offloader_load = local_load_vector[offloader_rank] + remote_load_vector[offloader_rank];
                double load_per_task = offloader_load / proact_tasks_to_offload_table[offloader_rank*num_ranks + offloader_rank];
                double overload = offloader_load - pred_avg_load;
                // RELP("[PROACT_MIGRATION] OFFLOADER: R%d, load=%.3f (load_per_task=%.3f), overload=%.3f\n", offloader_rank, offloader_load, load_per_task, overload);

                if (offloader_load > pred_avg_load && overload >= load_per_task){

                    // RELP("[PROACT_MIGRATION] OFFLOADER: R%d, load=%.3f (load_per_task=%.3f), overload=%.3f\n", offloader_rank, offloader_load, load_per_task, overload);

                    // calculate num of tasks that can be migrated to the victim
                    int numtasks_can_migrate = 0;
                    double migrated_load = 0.0;
                    if (overload >= underload) {
                        numtasks_can_migrate = int(underload / load_per_task);
                        migrated_load = numtasks_can_migrate * load_per_task;
                    }else{
                        numtasks_can_migrate = int(overload / load_per_task);
                        migrated_load = numtasks_can_migrate * load_per_task;
                    }

                    // update the underload and overload 
                    underload -= migrated_load;
                    local_load_vector[offloader_rank] -= migrated_load;
                    remote_load_vector[victim_rank] += migrated_load;
                    sum_load_vector[offloader_rank] = local_load_vector[offloader_rank] + remote_load_vector[offloader_rank];
                    sum_load_vector[victim_rank] = local_load_vector[victim_rank] + remote_load_vector[victim_rank];

                    // update num of local and remote tasks
                    proact_tasks_to_offload_table[offloader_rank*num_ranks + offloader_rank] -= numtasks_can_migrate;
                    proact_tasks_to_offload_table[offloader_rank*num_ranks + victim_rank] += numtasks_can_migrate;

                    // check the break-condition with underload
                    double abs_upd_underload = abs(sum_load_vector[victim_rank] - pred_avg_load);
                    if (abs_upd_underload < load_per_task)
                        break;
                }
            }
        }
    }
    /* END DIST_GREADY_KNAPSACK ALGORITHM */

#endif /* CHAMELEON_TOOL_SUPPORT && CHAM_PREDICTION_MODE > 0 && CHAM_PROACT_MIGRATION > 0 */

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
