#include "../common/ch_common.h"
#include "../timing.h"
#include "../timing_override.h"

void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt], int *block_rank)
{
    TASK_DEPTH_INIT;
    #if defined(CHAMELEON) || defined(CHAMELEON_TARGET)
    #pragma omp parallel
    {
        chameleon_thread_init();
    }
    // necessary to be aware of binary base addresses to calculate offset for target entry functions
    chameleon_determine_base_addresses((void *)&cholesky_mpi);
    chameleon_post_init_serial();
    void* literal_ts = *(void**)(&ts);
    #endif
    double * SPEC_RESTRICT tmp_b = B;

    #ifdef USE_TIMING
#if !defined(OMPSS_VER)
    #pragma omp parallel
    #pragma omp master
#endif
    INIT_TIMING(omp_get_num_threads());
    #endif
    
    int send_cnt = 0;
    char recv_flag = 0, *send_flags = malloc(sizeof(char) * np);
    MPI_Request recv_req, *send_reqs = malloc(sizeof(MPI_Request) * np);

    #ifdef TRACE
    static int event_communication = -1;
    char* event_name = "communication";
    if(event_communication == -1) {
        int ierr;
        ierr = VT_funcdef(event_name, VT_NOCLASS, &event_communication);
    }
    #endif

    START_TIMING(TIME_TOTAL);
#if !defined(OMPSS_VER)
    #pragma omp parallel
    {
#endif
    for (int k = 0; k < nt; k++) {
        DEBUG_PRINT("Iteration [%03d][000] --> 0 Starting new loop iter\n", k);
        if (block_rank[k*nt+k] == mype) {
#if !defined(OMPSS_VER)
            #pragma omp single
#endif
            {
                // first calculate diagonal element
                omp_potrf(A[k][k], ts, ts);

                START_TIMING(TIME_COMM);
                send_cnt = 0;
                reset_send_flags(send_flags);
                send_cnt = get_send_flags(send_flags, block_rank, k, k, k+1, nt-1, nt);
                if (send_cnt != 0) {
                    #ifdef TRACE
                    VT_begin(event_communication);
                    #endif
                    int exec_wait = 0;
                    send_cnt = 0;
                    for (int dst = 0; dst < np; dst++) {
                        if (send_flags[dst] && dst != mype) {
                            DEBUG_PRINT("Iteration [%03d][000] --> 0 Sending A-k-k to R#%d\n", k, dst);
                            exec_wait = 1;
                            MPI_Isend(A[k][k], ts*ts, MPI_DOUBLE, dst, k*nt+k, MPI_COMM_WORLD,
                                    &send_reqs[send_cnt++]);
                        }
                    }
                    if(exec_wait)
                        waitall(send_reqs, send_cnt);
                    #ifdef TRACE
                    VT_end(event_communication);
                    #endif
                }
                END_TIMING(TIME_COMM);
            }
        } else {
#if !defined(OMPSS_VER)
            #pragma omp single
#endif
            {
                START_TIMING(TIME_COMM);
                recv_flag = 0;
                get_recv_flag(&recv_flag, block_rank, k, k, k+1, nt-1, nt);

                if (recv_flag) {
                    #ifdef TRACE
                    VT_begin(event_communication);
                    #endif
                    DEBUG_PRINT("Iteration [%03d][000] --> 0 Recv A-k-k from R#%d\n", k, block_rank[k*nt+k]);
                    MPI_Irecv(B, ts*ts, MPI_DOUBLE, block_rank[k*nt+k], k*nt+k, MPI_COMM_WORLD,
                            &recv_req);
                    wait(&recv_req);
                    #ifdef TRACE
                    VT_end(event_communication);
                    #endif
                }
                END_TIMING(TIME_COMM);
            }
        }

        #pragma omp for nowait
        for (int i = k + 1; i < nt; i++) {
            if (block_rank[k*nt+i] == mype) {
                if (block_rank[k*nt+k] == mype) {
                    double * SPEC_RESTRICT tmp_a_k_k = A[k][k];
                    double * SPEC_RESTRICT tmp_a_k_i = A[k][i];
                    
                    #ifdef CHAMELEON_TARGET
                    #pragma omp target map(to: tmp_a_k_k[0:ts*ts]) map(tofrom: tmp_a_k_i[0:ts*ts]) device(1002)
                    {
                        omp_trsm(tmp_a_k_k, tmp_a_k_i, ts, ts);
                    }
                    #elif CHAMELEON
                    chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(4*sizeof(chameleon_map_data_entry_t));
                    args[0] = chameleon_map_data_entry_create(tmp_a_k_k, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                    args[1] = chameleon_map_data_entry_create(tmp_a_k_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                    args[2] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                    args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                    cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_trsm, 4, args);
                    int32_t res = chameleon_add_task(cur_task);
                    free(args);
                    #else
                    #pragma omp task
                    {
                        omp_trsm(tmp_a_k_k, tmp_a_k_i, ts, ts);
                    }
                    #endif
                } else {
                    double * SPEC_RESTRICT tmp_a_k_i   = A[k][i];
                    #ifdef CHAMELEON_TARGET
                    #pragma omp target map(to: B[0:ts*ts]) map(tofrom: tmp_a_k_i[0:ts*ts]) device(1002)
                    {
                        omp_trsm(tmp_b, tmp_a_k_i, ts, ts);
                    }
                    #elif CHAMELEON
                    chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(4*sizeof(chameleon_map_data_entry_t));
                    args[0] = chameleon_map_data_entry_create(tmp_b, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                    args[1] = chameleon_map_data_entry_create(tmp_a_k_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                    args[2] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                    args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                    cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_trsm, 4, args);
                    int32_t res = chameleon_add_task(cur_task);
                    free(args);
                    #else
                    #pragma omp task
                    {
                        omp_trsm(tmp_b, tmp_a_k_i, ts, ts);
                    }
                    #endif
                }
            }
        }

        #if defined(CHAMELEON) || defined(CHAMELEON_TARGET)
        chameleon_distributed_taskwait(0);
        #else
        #pragma omp barrier
        #endif

#if !defined(OMPSS_VER)
        #pragma omp single nowait
#endif
        {
            for (int i = k + 1; i < nt; i++) {
                DEBUG_PRINT("Iteration [%03d][%03d] --> 0 Begin\n", k, i);
                if (block_rank[k*nt+i] != mype) {
                    START_TIMING(TIME_COMM);
                    recv_flag = 0;
                    get_recv_flag(&recv_flag, block_rank, k+1, i-1, i, i, nt);
                    get_recv_flag(&recv_flag, block_rank, i, i, i+1, nt-1, nt);
                    get_recv_flag(&recv_flag, block_rank, i, i, i, i, nt);

                    if (recv_flag) {
                        #ifdef TRACE
                        VT_begin(event_communication);
                        #endif
                        DEBUG_PRINT("Iteration [%03d][%03d] --> 1 Recieving from R#%d - Start\n", k, i, block_rank[k*nt+i]);
                        MPI_Irecv(C[i], ts*ts, MPI_DOUBLE, block_rank[k*nt+i], k*nt+i, MPI_COMM_WORLD, &recv_req);
                        wait(&recv_req);
                        DEBUG_PRINT("Iteration [%03d][%03d] --> 2 Recieving from R#%d - zComplete\n", k, i, block_rank[k*nt+i]);
                        #ifdef TRACE
                        VT_end(event_communication);
                        #endif
                    }
                    END_TIMING(TIME_COMM);
                } else {
                    START_TIMING(TIME_COMM);
                    send_cnt = 0;
                    reset_send_flags(send_flags);
                    send_cnt += get_send_flags(send_flags, block_rank, k+1, i-1, i, i, nt);
                    send_cnt += get_send_flags(send_flags, block_rank, i, i, i+1, nt-1, nt);
                    send_cnt += get_send_flags(send_flags, block_rank, i, i, i, i, nt);

                    if (send_cnt != 0) {
                        #ifdef TRACE
                        VT_begin(event_communication);
                        #endif
                        send_cnt = 0;
                        int exec_wait = 0;
                        for (int dst = 0; dst < np; dst++) {
                            if (send_flags[dst] && dst != mype) {
                                DEBUG_PRINT("Iteration [%03d][%03d] --> 1 Sending to R#%d - Start\n", k, i, dst);
                                exec_wait = 1;
                                MPI_Isend(A[k][i], ts*ts, MPI_DOUBLE, dst, k*nt+i,
                                        MPI_COMM_WORLD, &send_reqs[send_cnt++]);
                            }
                        }
                        if(exec_wait) {
                            waitall(send_reqs, send_cnt);
                            DEBUG_PRINT("Iteration [%03d][%03d] --> 2 Sending zComplete\n", k, i);
                        }
                        #ifdef TRACE
                        VT_end(event_communication);
                        #endif
                    }
                    END_TIMING(TIME_COMM);
                }
                DEBUG_PRINT("Iteration [%03d][%03d] --> 3 Comm finished\n", k, i);
                // temporary pointers to be able to define slices for offloading
                double * SPEC_RESTRICT tmp_a_k_i   = A[k][i];
                double * SPEC_RESTRICT tmp_a_i_i   = A[i][i];
                double * SPEC_RESTRICT tmp_c_i     = C[i];
                {
                START_TIMING(TIME_CREATE);
                for (int j = k + 1; j < i; j++) {
                    // temporary pointers to be able to define slices for offloading
                    double * SPEC_RESTRICT tmp_a_k_j   = A[k][j];
                    double * SPEC_RESTRICT tmp_a_j_i   = A[j][i];
                    double * SPEC_RESTRICT tmp_c_j     = C[j];
                    if (block_rank[j*nt+i] == mype) {
                        if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] == mype) {
                            #ifdef CHAMELEON_TARGET
                            #pragma omp target map(to: tmp_a_k_i[0:ts*ts], tmp_a_k_j[0:ts*ts]) map(tofrom: tmp_a_j_i[0:ts*ts]) device(1002)
                            {
                                omp_gemm(tmp_a_k_i, tmp_a_k_j, tmp_a_j_i, ts, ts);
                            }
                            #elif CHAMELEON
                            chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(5*sizeof(chameleon_map_data_entry_t));
                            args[0] = chameleon_map_data_entry_create(tmp_a_k_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[1] = chameleon_map_data_entry_create(tmp_a_k_j, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[2] = chameleon_map_data_entry_create(tmp_a_j_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                            args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            args[4] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_gemm, 5, args);
                            int32_t res = chameleon_add_task(cur_task);
                            free(args);
                            #else
                            #pragma omp task
                            {
                                omp_gemm(tmp_a_k_i, tmp_a_k_j, tmp_a_j_i, ts, ts);
                            }
                            #endif
                        } else if (block_rank[k*nt+i] != mype && block_rank[k*nt+j] == mype) {
                            #ifdef CHAMELEON_TARGET
                            #pragma omp target map(to: tmp_c_i[0:ts*ts], tmp_a_k_j[0:ts*ts]) map(tofrom: tmp_a_j_i[0:ts*ts]) device(1002)
                            {
                                omp_gemm(tmp_c_i, tmp_a_k_j, tmp_a_j_i, ts, ts);
                            }
                            #elif CHAMELEON
                            chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(5*sizeof(chameleon_map_data_entry_t));
                            args[0] = chameleon_map_data_entry_create(tmp_c_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[1] = chameleon_map_data_entry_create(tmp_a_k_j, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[2] = chameleon_map_data_entry_create(tmp_a_j_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                            args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            args[4] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_gemm, 5, args);
                            int32_t res = chameleon_add_task(cur_task);
                            free(args);
                            #else
                            #pragma omp task
                            {
                                omp_gemm(tmp_c_i, tmp_a_k_j, tmp_a_j_i, ts, ts);
                            }
                            #endif
                        } else if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] != mype) {
                            #ifdef CHAMELEON_TARGET
                            #pragma omp target map(to: tmp_a_k_i[0:ts*ts], tmp_c_j[0:ts*ts]) map(tofrom: tmp_a_j_i[0:ts*ts]) device(1002)
                            {
                                omp_gemm(tmp_a_k_i, tmp_c_j, tmp_a_j_i, ts, ts);
                            }
                            #elif CHAMELEON
                            chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(5*sizeof(chameleon_map_data_entry_t));
                            args[0] = chameleon_map_data_entry_create(tmp_a_k_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[1] = chameleon_map_data_entry_create(tmp_c_j, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[2] = chameleon_map_data_entry_create(tmp_a_j_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                            args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            args[4] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_gemm, 5, args);
                            int32_t res = chameleon_add_task(cur_task);
                            free(args);
                            #else
                            #pragma omp task
                            {
                                omp_gemm(tmp_a_k_i, tmp_c_j, tmp_a_j_i, ts, ts);
                            }
                            #endif
                        } else {
                            #ifdef CHAMELEON_TARGET
                            #pragma omp target map(to: tmp_c_i[0:ts*ts], tmp_c_j[0:ts*ts]) map(tofrom: tmp_a_j_i[0:ts*ts]) device(1002)
                            {
                                omp_gemm(tmp_c_i, tmp_c_j, tmp_a_j_i, ts, ts);
                            }
                            #elif CHAMELEON
                            chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(5*sizeof(chameleon_map_data_entry_t));
                            args[0] = chameleon_map_data_entry_create(tmp_c_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[1] = chameleon_map_data_entry_create(tmp_c_j, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                            args[2] = chameleon_map_data_entry_create(tmp_a_j_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                            args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            args[4] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                            cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_gemm, 5, args);
                            int32_t res = chameleon_add_task(cur_task);
                            free(args);
                            #else
                            #pragma omp task
                            {
                                omp_gemm(tmp_c_i, tmp_c_j, tmp_a_j_i, ts, ts);
                            }
                            #endif
                        }
                    }
                }
                DEBUG_PRINT("Iteration [%03d][%03d] --> 4 Gemm Tasks created\n", k, i);
                if (block_rank[i*nt+i] == mype) {
                    if (block_rank[k*nt+i] == mype) {
                        #ifdef CHAMELEON_TARGET
                        #pragma omp target map(tofrom: tmp_a_k_i[0:ts*ts]) map(to: tmp_a_i_i[0:ts*ts]) device(1002)
                        {
                            omp_syrk(tmp_a_k_i, tmp_a_i_i, ts, ts);
                        }
                        #elif CHAMELEON
                        chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(4*sizeof(chameleon_map_data_entry_t));
                        args[0] = chameleon_map_data_entry_create(tmp_a_k_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                        args[1] = chameleon_map_data_entry_create(tmp_a_i_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                        args[2] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                        args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                        cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_syrk, 4, args);
                        int32_t res = chameleon_add_task(cur_task);
                        free(args);
                        #else
                        #pragma omp task
                        {
                            omp_syrk(tmp_a_k_i, tmp_a_i_i, ts, ts);
                        }
                        #endif
                    } else {
                        #ifdef CHAMELEON_TARGET
                        #pragma omp target map(tofrom: tmp_c_i[0:ts*ts]) map(to: tmp_a_i_i[0:ts*ts]) device(1002)
                        {
                            omp_syrk(tmp_c_i, tmp_a_i_i, ts, ts);
                        }
                        #elif CHAMELEON
                        chameleon_map_data_entry_t* args = (chameleon_map_data_entry_t*) malloc(4*sizeof(chameleon_map_data_entry_t));
                        args[0] = chameleon_map_data_entry_create(tmp_c_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                        args[1] = chameleon_map_data_entry_create(tmp_a_i_i, ts*ts*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_FROM);
                        args[2] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                        args[3] = chameleon_map_data_entry_create(literal_ts, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                        cham_migratable_task_t *cur_task = chameleon_create_task((void *)&omp_syrk, 4, args);
                        int32_t res = chameleon_add_task(cur_task);
                        free(args);
                        #else
                        #pragma omp task
                        {
                            omp_syrk(tmp_c_i, tmp_a_i_i, ts, ts);
                        }
                        #endif
                    }
                }
                DEBUG_PRINT("Iteration [%03d][%03d] --> 5 Syrk Tasks created\n", k, i);
                END_TIMING(TIME_CREATE);
                }
            }
        }
        DEBUG_PRINT("Iteration [%03d][998] --> 6 Proceeding to chameleon_distributed_taskwait(...)/barrier\n", k);
        
        #if defined(CHAMELEON) || defined(CHAMELEON_TARGET)
        chameleon_distributed_taskwait(0);
        #else
        #pragma omp barrier
        #endif
        DEBUG_PRINT("Iteration [%03d][999] --> 7 Finished chameleon_distributed_taskwait(...)/barrier\n", k);

        // #if !defined(CHAMELEON) && !defined(CHAMELEON_TARGET)
        // #pragma omp single
        // {
        // PRINT_INTERMEDIATE_TIMINGS(omp_get_num_threads());
        // }
        // #endif
    }
#if !defined(OMPSS_VER)
} /* end omp parallel */
#endif
    END_TIMING(TIME_TOTAL);
    MPI_Barrier(MPI_COMM_WORLD);

#if !defined(OMPSS_VER)    
    #pragma omp parallel
    #pragma omp master
#endif
    PRINT_TIMINGS(omp_get_num_threads());

	FREE_TIMING();

    #if defined(CHAMELEON) || defined(CHAMELEON_TARGET)
    chameleon_finalize();
    #endif
    free(send_flags);
    TASK_DEPTH_FINALIZE;
}

