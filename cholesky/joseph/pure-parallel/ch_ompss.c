
#include "ch_common.h"
#include "../timing.h"
#include "../timing_override.h"

void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt], int *block_rank)
{
#ifdef CHAMELEON
    chameleon_init();
#endif

#ifdef USE_TIMING
#pragma omp parallel
#pragma omp master
    INIT_TIMING(omp_get_num_threads());
#endif
    
    int send_cnt = 0;
    char recv_flag = 0, *send_flags = malloc(sizeof(char) * np);
    MPI_Request recv_req, *send_reqs = malloc(sizeof(MPI_Request) * np);

    START_TIMING(TIME_TOTAL);
#pragma omp parallel
{
    for (int k = 0; k < nt; k++) {
        my_print("Iteration [%03d][000]\tR#%d T#%d (OS_TID:%ld): --> 0 Starting new loop iter\n", k, mype, omp_get_thread_num(), syscall(SYS_gettid));
        if (block_rank[k*nt+k] == mype) {
            #pragma omp single 
            {
                // first calculate diagonal element
                omp_potrf(A[k][k], ts, ts);
            }

            #pragma omp master 
            {
                START_TIMING(TIME_COMM);
                send_cnt = 0;
                reset_send_flags(send_flags);
                send_cnt = get_send_flags(send_flags, block_rank, k, k, k+1, nt-1, nt);
                if (send_cnt != 0) {
                    int exec_wait = 0;
                    send_cnt = 0;
                    for (int dst = 0; dst < np; dst++) {
                        if (send_flags[dst] && dst != mype) {
                            exec_wait = 1;
                            MPI_Isend(A[k][k], ts*ts, MPI_DOUBLE, dst, k*nt+k, MPI_COMM_WORLD,
                                    &send_reqs[send_cnt++]);
                        }
                    }
                    if(exec_wait)
                        waitall(send_reqs, send_cnt);
                }
                END_TIMING(TIME_COMM);
            }
        } else {
            #pragma omp single
            {
                START_TIMING(TIME_COMM);
                recv_flag = 0;
                get_recv_flag(&recv_flag, block_rank, k, k, k+1, nt-1, nt);

                if (recv_flag) {
                    MPI_Irecv(B, ts*ts, MPI_DOUBLE, block_rank[k*nt+k], k*nt+k, MPI_COMM_WORLD,
                            &recv_req);
                    wait(&recv_req);
                }
                END_TIMING(TIME_COMM);
            }
        }

        // temporary pointers to be able to define slices for offloading
        #pragma omp for
        for (int i = k + 1; i < nt; i++) {
            if (block_rank[k*nt+i] == mype) {
                if (block_rank[k*nt+k] == mype) {
                    double *tmp_a_k_k = A[k][k];
                    double *tmp_a_k_i = A[k][i];
                    // printf("R#%d T#%d (OS_TID:%ld): --> Before Task: AKK = " DPxMOD ", AKI = " DPxMOD "\n", mype, omp_get_thread_num(), syscall(SYS_gettid), DPxPTR(tmp_a_k_k), DPxPTR(tmp_a_k_i));
// #ifdef CHAMELEON
//                     #pragma omp target map(to: tmp_a_k_k[0:ts*ts]) map(tofrom: tmp_a_k_i[0:ts*ts]) device(1002)
//                     {
//                         // printf("In Task: AKK = " DPxMOD ", AKI = " DPxMOD "\n", DPxPTR(tmp_a_k_k), DPxPTR(tmp_a_k_i));
//                         omp_trsm(tmp_a_k_k, tmp_a_k_i, ts, ts);
//                     }
// #else
                    #pragma omp task
                    {
                        // printf("R#%d T#%d (OS_TID:%ld): --> In Task: AKK = " DPxMOD ", AKI = " DPxMOD "\n", mype, omp_get_thread_num(), syscall(SYS_gettid), DPxPTR(tmp_a_k_k), DPxPTR(tmp_a_k_i));
                        omp_trsm(tmp_a_k_k, tmp_a_k_i, ts, ts);
                    }
// #endif
                } else {
                    double *tmp_a_k_i   = A[k][i];
                    // printf("R#%d T#%d (OS_TID:%ld): --> Before Task: B = " DPxMOD ", AKI = " DPxMOD "\n", mype, omp_get_thread_num(), syscall(SYS_gettid), DPxPTR(B), DPxPTR(tmp_a_k_i));
// #ifdef CHAMELEON
//                     #pragma omp target map(to: B[0:ts*ts]) map(tofrom: tmp_a_k_i[0:ts*ts]) device(1002)
//                     {
//                         // printf("In Task: B = " DPxMOD ", AKI = " DPxMOD "\n", DPxPTR(B), DPxPTR(tmp_a_k_i));
//                         omp_trsm(B, tmp_a_k_i, ts, ts);
//                     }
// #else
                    #pragma omp task
                    {
                        // printf("R#%d T#%d (OS_TID:%ld): --> In Task: B = " DPxMOD ", AKI = " DPxMOD "\n", mype, omp_get_thread_num(), syscall(SYS_gettid), DPxPTR(B), DPxPTR(tmp_a_k_i));
                        omp_trsm(B, tmp_a_k_i, ts, ts);
                    }
// #endif
                }
            }
        }

// #ifdef CHAMELEON
//         chameleon_distributed_taskwait(1);
// #endif

        #pragma omp single
        {
// #ifdef CHAMELEON
//             // chameleon call to start/wake up communication threads
//             chameleon_wake_up_comm_threads();
// #endif
            for (int i = k + 1; i < nt; i++) {
                my_print("Iteration [%03d][%03d]\tR#%d T#%d (OS_TID:%ld): --> 0 Begin\n", k, i, mype, omp_get_thread_num(), syscall(SYS_gettid));
                if (block_rank[k*nt+i] != mype) {
                    START_TIMING(TIME_COMM);
                    recv_flag = 0;
                    get_recv_flag(&recv_flag, block_rank, k+1, i-1, i, i, nt);
                    get_recv_flag(&recv_flag, block_rank, i, i, i+1, nt-1, nt);
                    get_recv_flag(&recv_flag, block_rank, i, i, i, i, nt);

                    if (recv_flag) {
                        my_print("Iteration [%03d][%03d]\tR#%d T#%d (OS_TID:%ld): --> 1 Recieving from R#%d - Start\n", k, i,mype, omp_get_thread_num(), syscall(SYS_gettid), block_rank[k*nt+i]);
                        MPI_Irecv(C[i], ts*ts, MPI_DOUBLE, block_rank[k*nt+i], k*nt+i, MPI_COMM_WORLD, &recv_req);
                        wait(&recv_req);
                        my_print("Iteration [%03d][%03d]\tR#%d T#%d (OS_TID:%ld): --> 2 Recieving from R#%d - zComplete\n", k, i,mype, omp_get_thread_num(), syscall(SYS_gettid), block_rank[k*nt+i]);
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
                        send_cnt = 0;
                        int exec_wait = 0;
                        for (int dst = 0; dst < np; dst++) {
                            if (send_flags[dst] && dst != mype) {
                                my_print("Iteration [%03d][%03d]\tR#%d T#%d (OS_TID:%ld): --> 1 Sending to R#%d - Start\n", k, i,mype, omp_get_thread_num(), syscall(SYS_gettid), dst);
                                exec_wait = 1;
                                MPI_Isend(A[k][i], ts*ts, MPI_DOUBLE, dst, k*nt+i,
                                        MPI_COMM_WORLD, &send_reqs[send_cnt++]);
                            }
                        }
                        if(exec_wait) {
                            waitall(send_reqs, send_cnt);
                            my_print("Iteration [%03d][%03d]\tR#%d T#%d (OS_TID:%ld): --> 2 Sending zComplete\n", k, i,mype, omp_get_thread_num(), syscall(SYS_gettid));
                        }
                    }
                    END_TIMING(TIME_COMM);
                }
                my_print("Iteration [%03d][%03d]\tR#%d T#%d (OS_TID:%ld): --> 3 Comm finished\n", k, i,mype, omp_get_thread_num(), syscall(SYS_gettid));
#ifdef CHAMELEON
                // temporary pointers to be able to define slices for offloading
                double *tmp_a_k_i   = A[k][i];
                double *tmp_a_i_i   = A[i][i];
                double *tmp_c_i     = C[i];
#endif
                {
                START_TIMING(TIME_CREATE);
                for (int j = k + 1; j < i; j++) {
#ifdef CHAMELEON
                    // temporary pointers to be able to define slices for offloading
                    double *tmp_a_k_j   = A[k][j];
                    double *tmp_a_j_i   = A[j][i];
                    double *tmp_c_j     = C[j];
#endif
                    if (block_rank[j*nt+i] == mype) {
                        if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] == mype) {
#ifdef CHAMELEON
                            #pragma omp target map(to: tmp_a_k_i[0:ts*ts], tmp_a_k_j[0:ts*ts]) map(tofrom: tmp_a_j_i[0:ts*ts]) device(1002)
                            {
                                omp_gemm(tmp_a_k_i, tmp_a_k_j, tmp_a_j_i, ts, ts);
                            }
#else
                            #pragma omp task
                            {
                                omp_gemm(A[k][i], A[k][j], A[j][i], ts, ts);
                            }
#endif
                        } else if (block_rank[k*nt+i] != mype && block_rank[k*nt+j] == mype) {
#ifdef CHAMELEON
                            #pragma omp target map(to: tmp_c_i[0:ts*ts], tmp_a_k_j[0:ts*ts]) map(tofrom: tmp_a_j_i[0:ts*ts]) device(1002)
                            {
                                omp_gemm(tmp_c_i, tmp_a_k_j, tmp_a_j_i, ts, ts);
                            }
#else
                            #pragma omp task
                            {
                                omp_gemm(C[i], A[k][j], A[j][i], ts, ts);
                            }
#endif
                        } else if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] != mype) {
#ifdef CHAMELEON
                            #pragma omp target map(to: tmp_a_k_i[0:ts*ts], tmp_c_j[0:ts*ts]) map(tofrom: tmp_a_j_i[0:ts*ts]) device(1002)
                            {
                                omp_gemm(tmp_a_k_i, tmp_c_j, tmp_a_j_i, ts, ts);
                            }
#else
                            #pragma omp task
                            {
                                omp_gemm(A[k][i], C[j], A[j][i], ts, ts);
                            }
#endif
                        } else {
#ifdef CHAMELEON
                            #pragma omp target map(to: tmp_c_i[0:ts*ts], tmp_c_j[0:ts*ts]) map(tofrom: tmp_a_j_i[0:ts*ts]) device(1002)
                            {
                                omp_gemm(tmp_c_i, tmp_c_j, tmp_a_j_i, ts, ts);
                            }
#else
                            #pragma omp task
                            {
                                omp_gemm(C[i], C[j], A[j][i], ts, ts);
                            }
#endif
                        }
                    }
                }

                my_print("Iteration [%03d][%03d]\tR#%d T#%d (OS_TID:%ld): --> 4 Gemm Tasks created\n", k, i,mype, omp_get_thread_num(), syscall(SYS_gettid));

                if (block_rank[i*nt+i] == mype) {
                    if (block_rank[k*nt+i] == mype) {
#ifdef CHAMELEON
                        #pragma omp target map(tofrom: tmp_a_k_i[0:ts*ts]) map(to: tmp_a_i_i[0:ts*ts]) device(1002)
                        {
                            omp_syrk(tmp_a_k_i, tmp_a_i_i, ts, ts);
                        }
#else
                        #pragma omp task
                        {
                            omp_syrk(A[k][i], A[i][i], ts, ts);
                        }
#endif
                    } else {
#ifdef CHAMELEON
                        #pragma omp target map(tofrom: tmp_c_i[0:ts*ts]) map(to: tmp_a_i_i[0:ts*ts]) device(1002)
                        {
                            omp_syrk(tmp_c_i, tmp_a_i_i, ts, ts);
                        }
#else
                        #pragma omp task
                        {
                            omp_syrk(C[i], A[i][i], ts, ts);
                        }
#endif
                    }
                }

                my_print("Iteration [%03d][%03d]\tR#%d T#%d (OS_TID:%ld): --> 5 Syrk Tasks created\n", k, i,mype, omp_get_thread_num(), syscall(SYS_gettid));
                END_TIMING(TIME_CREATE);
                }
            }
        }
#ifdef CHAMELEON
        my_print("Iteration [%03d][998]\tR#%d T#%d (OS_TID:%ld): --> 6 Proceeding to chameleon_distributed_taskwait(...)\n", k, mype, omp_get_thread_num(), syscall(SYS_gettid));
        chameleon_distributed_taskwait(1);
        my_print("Iteration [%03d][999]\tR#%d T#%d (OS_TID:%ld): --> 7 Finished chameleon_distributed_taskwait(...)\n", k, mype, omp_get_thread_num(), syscall(SYS_gettid));
#endif
    }
} /* end omp parallel */
    END_TIMING(TIME_TOTAL);
    MPI_Barrier(MPI_COMM_WORLD);
#pragma omp parallel
#pragma omp master
    PRINT_TIMINGS(omp_get_num_threads());

	FREE_TIMING();

#ifdef CHAMELEON
    chameleon_finalize();
#endif

    free(send_flags);
}

