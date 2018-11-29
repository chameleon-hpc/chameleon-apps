
#include "ch_common.h"
#include "../timing.h"

void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt], int *block_rank)
{
#ifdef CHAMELEON
    chameleon_init();
#endif

    int send_cnt = 0;
    char recv_flag = 0, *send_flags = malloc(sizeof(char) * np);
    MPI_Request recv_req, *send_reqs = malloc(sizeof(MPI_Request) * np);

#pragma omp parallel
{
    for (int k = 0; k < nt; k++) {

        if (block_rank[k*nt+k] == mype) {
            #pragma omp single
            omp_potrf(A[k][k], ts, ts);

            #pragma omp master
            {
                send_cnt = 0;
                reset_send_flags(send_flags);
                send_cnt = get_send_flags(send_flags, block_rank, k, k, k+1, nt-1, nt);

                if (send_cnt != 0) {

                    send_cnt = 0;
                    for (int dst = 0; dst < np; dst++) {
                        if (send_flags[dst] && dst != mype) {
                            MPI_Isend(A[k][k], ts*ts, MPI_DOUBLE, dst, k*nt+k, MPI_COMM_WORLD,
                                    &send_reqs[send_cnt++]);
                        }
                    }
                    waitall(send_reqs, send_cnt);
                }
            }
        } else {
            #pragma omp single
            {
                recv_flag = 0;
                get_recv_flag(&recv_flag, block_rank, k, k, k+1, nt-1, nt);

                if (recv_flag) {
                    MPI_Irecv(B, ts*ts, MPI_DOUBLE, block_rank[k*nt+k], k*nt+k, MPI_COMM_WORLD,
                            &recv_req);
                    wait(&recv_req);
                }
            }
        }

        #pragma omp for
        for (int i = k + 1; i < nt; i++) {
            if (block_rank[k*nt+i] == mype) {
                if (block_rank[k*nt+k] == mype) {
                    #pragma omp task
                    {
                        omp_trsm(A[k][k], A[k][i], ts, ts);
                    }
                } else {
                    #pragma omp task
                    {
                        omp_trsm(B, A[k][i], ts, ts);
                    }
                }
            }
        }

        #pragma omp single
        {
#ifdef CHAMELEON
            // chameleon call to start/wake up communication threads
            wake_up_comm_threads();
#endif
            for (int i = k + 1; i < nt; i++) {

                if (block_rank[k*nt+i] != mype) {

                    recv_flag = 0;
                    get_recv_flag(&recv_flag, block_rank, k+1, i-1, i, i, nt);
                    get_recv_flag(&recv_flag, block_rank, i, i, i+1, nt-1, nt);
                    get_recv_flag(&recv_flag, block_rank, i, i, i, i, nt);

                    if (recv_flag) {

                        MPI_Irecv(C[i], ts*ts, MPI_DOUBLE, block_rank[k*nt+i], k*nt+i,
                                MPI_COMM_WORLD, &recv_req);

                        wait(&recv_req);
                    }

                } else {

                    send_cnt = 0;
                    reset_send_flags(send_flags);
                    send_cnt += get_send_flags(send_flags, block_rank, k+1, i-1, i, i, nt);
                    send_cnt += get_send_flags(send_flags, block_rank, i, i, i+1, nt-1, nt);
                    send_cnt += get_send_flags(send_flags, block_rank, i, i, i, i, nt);

                    if (send_cnt != 0) {

                        send_cnt = 0;
                        for (int dst = 0; dst < np; dst++) {
                            if (send_flags[dst] && dst != mype) {
                                MPI_Isend(A[k][i], ts*ts, MPI_DOUBLE, dst, k*nt+i,
                                        MPI_COMM_WORLD, &send_reqs[send_cnt++]);
                            }
                        }
                        waitall(send_reqs, send_cnt);
                    }
                }

#ifdef CHAMELEON
                // temporary pointers to be able to define slices for offloading
                double *tmp_a_k_i   = A[k][i];
                double *tmp_a_i_i   = A[i][i];
                double *tmp_c_i     = C[i];
#endif

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
            }
        }
    }
#ifdef CHAMELEON
        // TODO: distributed taskwait should go here
        // TODO: revise taskwait and prohibit breakout if commthreads actived or taskwait has not been called everywhere
        // TODO: otherwise race condition might occur if thread is faster than the one that is creating the first task and breakout occurs
        chameleon_distributed_taskwait(1);
#endif
} /* end omp parallel */
    MPI_Barrier(MPI_COMM_WORLD);

#ifdef CHAMELEON
    chameleon_finalize();
#endif

    free(send_flags);
}

