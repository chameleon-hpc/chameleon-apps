#include "../common/ch_common.h"
#include "../timing.h"
<<<<<<< HEAD
// #include "../timing_override.h"

void cholesky_mpi(const int ts, const int nt, double * SPEC_RESTRICT A[nt][nt], double * SPEC_RESTRICT B, double * SPEC_RESTRICT C[nt], int *block_rank)
=======

void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt], int *block_rank)
>>>>>>> a6fe163e5c2c3b760678780248eabfa1d604c9df
{
    TASK_DEPTH_INIT;
    #if defined(CHAMELEON) || defined(CHAMELEON_TARGET)
    #pragma omp parallel
    {
        chameleon_thread_init();
    }
    // necessary to be aware of binary base addresses to calculate offset for target entry functions
    chameleon_determine_base_addresses((void *)&cholesky_mpi);
<<<<<<< HEAD
    void* literal_ts            = *(void**)(&ts);
=======
    void* literal_ts = *(void**)(&ts);
>>>>>>> a6fe163e5c2c3b760678780248eabfa1d604c9df
    #endif

    #pragma omp parallel
    #pragma omp master
    INIT_TIMING(omp_get_num_threads());

    START_TIMING(TIME_TOTAL);

    #pragma omp parallel
    {
    #pragma omp single nowait
    {
    {
    START_TIMING(TIME_CREATE);
    
    for (int k = 0; k < nt; k++) {
        double * SPEC_RESTRICT tmp_a_k_k = A[k][k];

        if (block_rank[k*nt+k] == mype) {
            #pragma omp task depend(out: A[k][k]) firstprivate(k)
            {
                TASK_DEPTH_INCR;
                DEBUG_PRINT("Computing omp_potrf[%03d][%03d] - Start\n", k, k);
                #if CHAMELEON
                chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(3*sizeof(chameleon_map_data_entry_t));
                args[0] = chameleon_map_data_entry_create(tmp_a_k_k, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                args[1] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                args[2] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_potrf, 3, args);
                int32_t res = chameleon_add_task(cur_task);
                free(args);
                TYPE_TASK_ID tmp_id = chameleon_get_last_local_task_id_added();
                
                while(!chameleon_local_task_has_finished(tmp_id)) {
                    chameleon_taskyield();
                }
                #else
                omp_potrf(tmp_a_k_k, ts, ts);
                #endif
                DEBUG_PRINT("Computing omp_potrf[%03d][%03d] - End\n", k, k);
                TASK_DEPTH_DECR;
            }
        }

        int comm_sentinel; // <-- sentinel, never actual referenced

        if (block_rank[k*nt+k] == mype && np != 1) {
            // use comm_sentinel to make sure this task runs before the communication tasks below
            #pragma omp task depend(in: A[k][k], comm_sentinel) firstprivate(k) untied
            {
                TASK_DEPTH_INCR;
                DEBUG_PRINT("Sending diagonal[%03d][%03d] - Start\n", k, k);
                START_TIMING(TIME_COMM);
                MPI_Request *reqs = NULL;
                int nreqs = 0;
                char send_flags[np];
                reset_send_flags(send_flags);
                for (int kk = k+1; kk < nt; kk++) {
                    if (!send_flags[block_rank[k*nt+kk]]) {
                    ++nreqs;
                    send_flags[block_rank[k*nt+kk]] = 1;
                    }
                }
                reqs = malloc(sizeof(MPI_Request)*nreqs);
                nreqs = 0;
                for (int dst = 0; dst < np; dst++) {
                    if (send_flags[dst] && dst != mype) {
                    MPI_Request send_req;
                    MPI_Isend(A[k][k], ts*ts, MPI_DOUBLE, dst, k*nt+k, MPI_COMM_WORLD, &send_req);
                    reqs[nreqs++] = send_req;
                    }
                }
                waitall(reqs, nreqs);
                free(reqs);
                END_TIMING(TIME_COMM);
                DEBUG_PRINT("Sending diagonal[%03d][%03d] - End\n", k, k);
                TASK_DEPTH_DECR;
            }
        } else if (block_rank[k*nt+k] != mype) {
            // use comm_sentinel to make sure this task runs before the communication tasks below
            #pragma omp task depend(out: B) depend(in:comm_sentinel) firstprivate(k) untied
            {
                TASK_DEPTH_INCR;
                DEBUG_PRINT("Receiving diagonal[%03d][%03d] - Start\n", k, k);
                START_TIMING(TIME_COMM);
                int recv_flag = 0;
                for (int i = k + 1; i < nt; i++) {
                    if (block_rank[k*nt+i] == mype) {
                    recv_flag = 1;
                    break;
                    }
                }
                if (recv_flag) {
                    MPI_Request recv_req;
                    MPI_Irecv(B, ts*ts, MPI_DOUBLE, block_rank[k*nt+k], k*nt+k, MPI_COMM_WORLD, &recv_req);
                    waitall(&recv_req, 1);
                }
                END_TIMING(TIME_COMM);
                DEBUG_PRINT("Receiving diagonal[%03d][%03d] - End\n", k, k);
                TASK_DEPTH_DECR;
            }
        }

<<<<<<< HEAD
=======
        double * SPEC_RESTRICT tmp_b = B;

>>>>>>> a6fe163e5c2c3b760678780248eabfa1d604c9df
        for (int i = k + 1; i < nt; i++) {
            double * SPEC_RESTRICT tmp_a_k_i = A[k][i];

            if (block_rank[k*nt+i] == mype) {
                if (block_rank[k*nt+k] == mype) {
                    #pragma omp task depend(in: A[k][k], comm_sentinel) depend(out: A[k][i]) firstprivate(k, i)
                    {
                        TASK_DEPTH_INCR;
                        DEBUG_PRINT("Computing omp_trsm[%03d][%03d] - Start\n", k, i);
                        #if CHAMELEON
                        chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(4*sizeof(chameleon_map_data_entry_t));
                        args[0] = chameleon_map_data_entry_create(tmp_a_k_k, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                        args[1] = chameleon_map_data_entry_create(tmp_a_k_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                        args[2] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                        args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                        cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_trsm, 4, args);
                        int32_t res = chameleon_add_task(cur_task);
                        free(args);
                        TYPE_TASK_ID tmp_id = chameleon_get_last_local_task_id_added();
                        
                        while(!chameleon_local_task_has_finished(tmp_id)) {
                            chameleon_taskyield();
                        }
                        #else
                        omp_trsm(tmp_a_k_k, tmp_a_k_i, ts, ts);
                        #endif
                        DEBUG_PRINT("Computing omp_trsm[%03d][%03d] - End\n", k, i);
                        TASK_DEPTH_DECR;
                    }
                } else {
                    #pragma omp task depend(in: B, comm_sentinel) depend(out: A[k][i]) firstprivate(k, i)
                    {
                        TASK_DEPTH_INCR;
                        DEBUG_PRINT("Computing omp_trsm[%03d][%03d] - Start\n", k, i);
                        #if CHAMELEON
                        chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(4*sizeof(chameleon_map_data_entry_t));
<<<<<<< HEAD
                        args[0] = chameleon_map_data_entry_create(B, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
=======
                        args[0] = chameleon_map_data_entry_create(tmp_b, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
>>>>>>> a6fe163e5c2c3b760678780248eabfa1d604c9df
                        args[1] = chameleon_map_data_entry_create(tmp_a_k_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                        args[2] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                        args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                        cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_trsm, 4, args);
                        int32_t res = chameleon_add_task(cur_task);
                        free(args);
                        TYPE_TASK_ID tmp_id = chameleon_get_last_local_task_id_added();
                        
                        while(!chameleon_local_task_has_finished(tmp_id)) {
                            chameleon_taskyield();
                        }
                        #else
<<<<<<< HEAD
                        omp_trsm(B, tmp_a_k_i, ts, ts);
=======
                        omp_trsm(tmp_b, tmp_a_k_i, ts, ts);
>>>>>>> a6fe163e5c2c3b760678780248eabfa1d604c9df
                        #endif
                        DEBUG_PRINT("Computing omp_trsm[%03d][%03d] - End\n", k, i);
                        TASK_DEPTH_DECR;
                    }
                }
            }
        }

        #pragma omp task depend(inout: comm_sentinel) firstprivate(k) shared(A) untied
        {
            TASK_DEPTH_INCR;
            DEBUG_PRINT("Send/Recv omp_trsm[%03d] - Start\n", k);
            START_TIMING(TIME_COMM);
            char send_flags[np];
            reset_send_flags(send_flags);
            int nreqs = 0;
            // upper bound in case all our blocks have to be sent
            int max_req = (nt-k)*(np-1);
            MPI_Request *reqs = malloc(sizeof(*reqs)*max_req);
            for (int i = k + 1; i < nt; i++) {
                if (block_rank[k*nt+i] == mype && np != 1) {
                    for (int ii = k + 1; ii < i; ii++) {
                        if (!send_flags[block_rank[ii*nt+i]]) {
                            send_flags[block_rank[ii*nt+i]] = 1;
                        }
                    }
                    for (int ii = i + 1; ii < nt; ii++) {
                        if (!send_flags[block_rank[i*nt+ii]]) {
                            send_flags[block_rank[i*nt+ii]] = 1;
                        }
                    }
                    if (!send_flags[block_rank[i*nt+i]]) send_flags[block_rank[i*nt+i]] = 1;
                    for (int dst = 0; dst < np; dst++) {
                        if (send_flags[dst] && dst != mype) {
                            MPI_Request send_req;
                            MPI_Isend(A[k][i], ts*ts, MPI_DOUBLE, dst, k*nt+i, MPI_COMM_WORLD, &send_req);
                            reqs[nreqs++] = send_req;
                        }
                    }
                    reset_send_flags(send_flags);
                }
                if (block_rank[k*nt+i] != mype) {
                    int recv_flag = 0;
                    for (int ii = k + 1; ii < i; ii++) {
                        if (block_rank[ii*nt+i] == mype) recv_flag = 1;
                    }
                    for (int ii = i + 1; ii < nt; ii++) {
                        if (block_rank[i*nt+ii] == mype) recv_flag = 1;
                    }
                    if (block_rank[i*nt+i] == mype) recv_flag = 1;
                    if (recv_flag) {
                        MPI_Request recv_req;
                        MPI_Irecv(C[i], ts*ts, MPI_DOUBLE, block_rank[k*nt+i], k*nt+i, MPI_COMM_WORLD, &recv_req);
                        reqs[nreqs++] = recv_req;
                    }
                }
            }

            waitall(reqs, nreqs);
            free(reqs);
            END_TIMING(TIME_COMM);
            DEBUG_PRINT("Send/Recv omp_trsm[%03d] - End\n", k);
            TASK_DEPTH_DECR;
        }

        for (int i = k + 1; i < nt; i++) {
            double * SPEC_RESTRICT tmp_a_k_i = A[k][i];
            double * SPEC_RESTRICT tmp_a_i_i = A[i][i];
            double * SPEC_RESTRICT tmp_c_i   = C[i];

            for (int j = k + 1; j < i; j++) {
                double * SPEC_RESTRICT tmp_a_k_j = A[k][j];
                double * SPEC_RESTRICT tmp_a_j_i = A[j][i];
                double * SPEC_RESTRICT tmp_c_j   = C[j];

                if (block_rank[j*nt+i] == mype) {
                    if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] == mype) {
                        #pragma omp task depend(in: A[k][i], A[k][j]) depend(out: A[j][i]) firstprivate(k, j, i)
                        {
                            TASK_DEPTH_INCR;
                            DEBUG_PRINT("Computing omp_gemm[%03d][%03d] - Start\n", j, i);
                            #if CHAMELEON
                            chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(5*sizeof(chameleon_map_data_entry_t));
                            args[0] = chameleon_map_data_entry_create(tmp_a_k_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[1] = chameleon_map_data_entry_create(tmp_a_k_j, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[2] = chameleon_map_data_entry_create(tmp_a_j_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                            args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            args[4] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_gemm, 5, args);
                            int32_t res = chameleon_add_task(cur_task);
                            free(args);
                            TYPE_TASK_ID tmp_id = chameleon_get_last_local_task_id_added();
                            
                            while(!chameleon_local_task_has_finished(tmp_id)) {
                                chameleon_taskyield();
                            }
                            #else
                            omp_gemm(tmp_a_k_i, tmp_a_k_j, tmp_a_j_i, ts, ts);
                            #endif
                            DEBUG_PRINT("Computing omp_gemm[%03d][%03d] - End\n", j, i);
                            TASK_DEPTH_DECR;
                        }
                    } else if (block_rank[k*nt+i] != mype && block_rank[k*nt+j] == mype) {
                        #pragma omp task depend(in: A[k][j], comm_sentinel) depend(out: A[j][i]) firstprivate(k, j, i)
                        {
                            TASK_DEPTH_INCR;
                            DEBUG_PRINT("Computing omp_gemm[%03d][%03d] - Start\n", j, i);
                            #if CHAMELEON
                            chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(5*sizeof(chameleon_map_data_entry_t));
                            args[0] = chameleon_map_data_entry_create(tmp_c_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[1] = chameleon_map_data_entry_create(tmp_a_k_j, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[2] = chameleon_map_data_entry_create(tmp_a_j_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                            args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            args[4] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_gemm, 5, args);
                            int32_t res = chameleon_add_task(cur_task);
                            free(args);
                            TYPE_TASK_ID tmp_id = chameleon_get_last_local_task_id_added();
                            
                            while(!chameleon_local_task_has_finished(tmp_id)) {
                                chameleon_taskyield();
                            }
                            #else
                            omp_gemm(tmp_c_i, tmp_a_k_j, tmp_a_j_i, ts, ts);
                            #endif
                            DEBUG_PRINT("Computing omp_gemm[%03d][%03d] - End\n", j, i);
                            TASK_DEPTH_DECR;
                        }
                    } else if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] != mype) {
                        #pragma omp task depend(in: A[k][i], comm_sentinel) depend(out: A[j][i]) firstprivate(k, j, i)
                        {
                            TASK_DEPTH_INCR;
                            DEBUG_PRINT("Computing omp_gemm[%03d][%03d] - Start\n", j, i);
                            #if CHAMELEON
                            chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(5*sizeof(chameleon_map_data_entry_t));
                            args[0] = chameleon_map_data_entry_create(tmp_a_k_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[1] = chameleon_map_data_entry_create(tmp_c_j, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[2] = chameleon_map_data_entry_create(tmp_a_j_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                            args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            args[4] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_gemm, 5, args);
                            int32_t res = chameleon_add_task(cur_task);
                            free(args);
                            TYPE_TASK_ID tmp_id = chameleon_get_last_local_task_id_added();
                            
                            while(!chameleon_local_task_has_finished(tmp_id)) {
                                chameleon_taskyield();
                            }
                            #else
                            omp_gemm(tmp_a_k_i, tmp_c_j, tmp_a_j_i, ts, ts);
                            #endif
                            DEBUG_PRINT("Computing omp_gemm[%03d][%03d] - End\n", j, i);
                            TASK_DEPTH_DECR;
                        }
                    } else {
                        #pragma omp task depend(in: comm_sentinel) depend(out: A[j][i]) firstprivate(k, j, i)
                        {
                            TASK_DEPTH_INCR;
                            DEBUG_PRINT("Computing omp_gemm[%03d][%03d] - Start\n", j, i);
                            #if CHAMELEON
                            chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(5*sizeof(chameleon_map_data_entry_t));
                            args[0] = chameleon_map_data_entry_create(tmp_c_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[1] = chameleon_map_data_entry_create(tmp_c_j, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[2] = chameleon_map_data_entry_create(tmp_a_j_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                            args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            args[4] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_gemm, 5, args);
                            int32_t res = chameleon_add_task(cur_task);
                            free(args);
                            TYPE_TASK_ID tmp_id = chameleon_get_last_local_task_id_added();
                            
                            while(!chameleon_local_task_has_finished(tmp_id)) {
                                chameleon_taskyield();
                            }
                            #else
                            omp_gemm(tmp_c_i, tmp_c_j, tmp_a_j_i, ts, ts);
                            #endif
                            DEBUG_PRINT("Computing omp_gemm[%03d][%03d] - End\n", j, i);
                            TASK_DEPTH_DECR;
                        }
                    }
                }
            }

            if (block_rank[i*nt+i] == mype) {
                if (block_rank[k*nt+i] == mype) {
                    #pragma omp task depend(in: A[k][i]) depend(out: A[i][i]) firstprivate(k, i)
                    {
                        TASK_DEPTH_INCR;
                        DEBUG_PRINT("Computing omp_syrk[%03d][%03d] - Start\n", i, i);
                        #if CHAMELEON
                        chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(4*sizeof(chameleon_map_data_entry_t));
                        args[0] = chameleon_map_data_entry_create(tmp_a_k_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                        args[1] = chameleon_map_data_entry_create(tmp_a_i_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                        args[2] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                        args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                        cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_syrk, 4, args);
                        int32_t res = chameleon_add_task(cur_task);
                        free(args);
                        TYPE_TASK_ID tmp_id = chameleon_get_last_local_task_id_added();
                        
                        while(!chameleon_local_task_has_finished(tmp_id)) {
                            chameleon_taskyield();
                        }
                        #else
                        omp_syrk(tmp_a_k_i, tmp_a_i_i, ts, ts);
                        #endif
                        DEBUG_PRINT("Computing omp_syrk[%03d][%03d] - End\n", i, i);
                        TASK_DEPTH_DECR;
                    }
                } else {
                    #pragma omp task depend(in: comm_sentinel) depend(out: A[i][i]) firstprivate(k, i)
                    {
                        TASK_DEPTH_INCR;
                        DEBUG_PRINT("Computing omp_syrk[%03d][%03d] - Start\n", i, i);
                        #if CHAMELEON
                        chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(4*sizeof(chameleon_map_data_entry_t));
                        args[0] = chameleon_map_data_entry_create(tmp_c_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                        args[1] = chameleon_map_data_entry_create(tmp_a_i_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                        args[2] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                        args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                        cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_syrk, 4, args);
                        int32_t res = chameleon_add_task(cur_task);
                        free(args);
                        TYPE_TASK_ID tmp_id = chameleon_get_last_local_task_id_added();
                        
                        while(!chameleon_local_task_has_finished(tmp_id)) {
                            chameleon_taskyield();
                        }
                        #else
                        omp_syrk(tmp_c_i, tmp_a_i_i, ts, ts);
                        #endif
                        DEBUG_PRINT("Computing omp_syrk[%03d][%03d] - End\n", i, i);
                        TASK_DEPTH_DECR;
                    }
                }
            }
        }
    }
    END_TIMING(TIME_CREATE);
    }
    }// pragma omp single

    #if defined(CHAMELEON) || defined(CHAMELEON_TARGET)
    chameleon_distributed_taskwait(0);
    #endif
    
    #pragma omp single
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    }// pragma omp parallel

    END_TIMING(TIME_TOTAL);

    #pragma omp parallel
    #pragma omp master
    PRINT_TIMINGS(omp_get_num_threads());

    FREE_TIMING();

    #if defined(CHAMELEON) || defined(CHAMELEON_TARGET)
    chameleon_finalize();
    #endif

    TASK_DEPTH_FINALIZE;
}

