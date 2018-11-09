
#include "common.h"
#include "chameleon.h"

static void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt],
                         int *block_rank);

double do_cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt],
                       int *block_rank)
{
    double time;
    chameleon_init();

    time = get_time();
    cholesky_mpi(ts, nt, (double* (*)[nt])A, B, C, block_rank);
    time = get_time() - time;

    chameleon_finalize();
    return time;
}

static void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt],
                         int *block_rank)
{
    int send_cnt = 0; 
    char recv_flag = 0, *send_flags = malloc(sizeof(char) * np);
    MPI_Request recv_req, *send_reqs = malloc(sizeof(MPI_Request) * np);

#pragma omp parallel
{
    for (int k = 0; k < nt; k++) {
        
        if (block_rank[k*nt+k] == mype) {
            #pragma omp single 
            {
                do_potrf(A[k][k], ts, ts);
            }

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
                    testall_and_yield(send_cnt, send_reqs);
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

                    test_and_yield(&recv_req, 1, block_rank[k*nt+k], k*nt+k);
                }
            }
        }

        #pragma omp for
        for (int i = k + 1; i < nt; i++) {

            if (block_rank[k*nt+i] == mype) {
                if (block_rank[k*nt+k] == mype) {
                    #pragma omp task 
                    {
                        do_trsm(A[k][k], A[k][i], ts, ts);
                    }
                } else {
                    #pragma omp task 
                    {
                        do_trsm(B, A[k][i], ts, ts);
                    }
                }
            }
        }

        // #pragma omp barrier

        // TODO: start communication threads or active them if already present

        #pragma omp single
        {
            // chameleon call to wake up threads
            wake_up_comm_threads();

            for (int i = k + 1; i < nt; i++) {

                if (block_rank[k*nt+i] != mype) {

                    recv_flag = 0;
                    get_recv_flag(&recv_flag, block_rank, k+1, i-1, i, i, nt);
                    get_recv_flag(&recv_flag, block_rank, i, i, i+1, nt-1, nt);
                    get_recv_flag(&recv_flag, block_rank, i, i, i, i, nt);

                    if (recv_flag) {

                        MPI_Irecv(C[i], ts*ts, MPI_DOUBLE, block_rank[k*nt+i], k*nt+i,
                                MPI_COMM_WORLD, &recv_req);

                        // TODO: we need a custom taskyield here
                        test_and_yield(&recv_req, 1, block_rank[k*nt+i], k*nt+i);
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

                        // TODO: we need a custom taskyield here
                        testall_and_yield(send_cnt, send_reqs);
                    }
                }

                for (int j = k + 1; j < i; j++) {

                    if (block_rank[j*nt+i] == mype) {
                        if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] == mype) {
                            #pragma omp task
                            {
                                do_gemm(A[k][i], A[k][j], A[j][i], ts, ts);
                            }
                        } else if (block_rank[k*nt+i] != mype && block_rank[k*nt+j] == mype) {
                            #pragma omp task
                            {
                                do_gemm(C[i], A[k][j], A[j][i], ts, ts);
                            }
                        } else if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] != mype) {
                            #pragma omp task
                            {
                                do_gemm(A[k][i], C[j], A[j][i], ts, ts);
                            }
                        } else {
                            #pragma omp task
                            {
                                do_gemm(C[i], C[j], A[j][i], ts, ts);
                            }
                        }
                    }
                }

                if (block_rank[i*nt+i] == mype) {
                    if (block_rank[k*nt+i] == mype) {
                        #pragma omp task
                        {
                            do_syrk(A[k][i], A[i][i], ts, ts);
                        }
                    } else {
                        #pragma omp task
                        {
                            do_syrk(C[i], A[i][i], ts, ts);
                        }
                    }
                }
            }
        }

        // TODO: distributed taskwait should go here
        // TODO: revise taskwait and prohibit breakout if commthreads actived or taskwait has not been called everywhere
        // TODO: otherwise race condition might occur if thread is faster than the one that is creating the first task and breakout occurs
        chameleon_distributed_taskwait(1);
    }
} /* end omp parallel */
    MPI_Barrier(MPI_COMM_WORLD);
    free(send_flags);
}

