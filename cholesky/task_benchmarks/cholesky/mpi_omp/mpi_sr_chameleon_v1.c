
#include "common.h"
#include "chameleon.h"

long syscall(long number, ...);
int32_t chameleon_taskyield();
void testall_and_yield_cham(int comm_cnt, MPI_Request *comm_reqs);
void test_and_yield_cham(MPI_Request *comm_req, int c_type, int src_dst, int tag, int i, int j);

inline void testall_and_yield_cham(int comm_cnt, MPI_Request *comm_reqs)
{
    if (!comm_cnt) return;

    int comm_comp = 0;


    MPI_Testall(comm_cnt, comm_reqs, &comm_comp, MPI_STATUSES_IGNORE);
    while (!comm_comp) {
        // call specific chameleon taskyield
        int32_t res = chameleon_taskyield();
        #pragma omp taskyield
        MPI_Testall(comm_cnt, comm_reqs, &comm_comp, MPI_STATUSES_IGNORE);
    }
}

inline void test_and_yield_cham(MPI_Request *comm_req, int c_type, int src_dst, int tag, int i, int j)
{
    int comm_comp = 0;
#ifdef DEBUG
    int printed = 0;
#endif
    MPI_Test(comm_req, &comm_comp, MPI_STATUS_IGNORE);
    while (!comm_comp) {
        // call specific chameleon taskyield
        int32_t res = chameleon_taskyield();
        #pragma omp taskyield
#ifdef DEBUG
        if (!printed) {
            if(c_type == 0)
                fprintf(stderr, "[%0*d][%0*d]    #R%d (OS_TID:%ld):    SEND    Wait    to      %*d with tag %*d    [%*d][%*d]\n",tmp_width, i, tmp_width, j, mype, syscall(SYS_gettid), tmp_width, src_dst, tmp_width, tag, tmp_width, i, tmp_width, j);
            else if(c_type == 1)
                fprintf(stderr, "[%0*d][%0*d]    #R%d (OS_TID:%ld):    RECV    Wait    from    %*d with tag %*d    [%*d][%*d]\n",tmp_width, i, tmp_width, j, mype, syscall(SYS_gettid), tmp_width, src_dst, tmp_width, tag, tmp_width, i, tmp_width, j);
            printed = 1;
        }
#endif
        MPI_Test(comm_req, &comm_comp, MPI_STATUS_IGNORE);
    }
#ifdef DEBUG
    if(c_type == 0)
        fprintf(stderr, "[%0*d][%0*d]    #R%d (OS_TID:%ld):    SEND    End     to      %*d with tag %*d    [%*d][%*d]\n",tmp_width, i, tmp_width, j, mype, syscall(SYS_gettid), tmp_width, src_dst, tmp_width, tag, tmp_width, i, tmp_width, j);
    else if(c_type == 1)
        fprintf(stderr, "[%0*d][%0*d]    #R%d (OS_TID:%ld):    RECV    End     from    %*d with tag %*d    [%*d][%*d]\n",tmp_width, i, tmp_width, j, mype, syscall(SYS_gettid), tmp_width, src_dst, tmp_width, tag, tmp_width, i, tmp_width, j);
#endif
}

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
                    testall_and_yield_cham(send_cnt, send_reqs);
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

                    test_and_yield_cham(&recv_req, 1, block_rank[k*nt+k], k*nt+k, k, k);
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

        #pragma omp single nowait
        {
            // chameleon call to start/wake up communication threads
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

                        test_and_yield_cham(&recv_req, 1, block_rank[k*nt+i], k*nt+i, i, k);
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
                        testall_and_yield_cham(send_cnt, send_reqs);
                    }
                }

                double *tmp_a_k_i   = A[k][i];
                double *tmp_a_i_i   = A[i][i];
                double *tmp_c_i     = C[i];

                for (int j = k + 1; j < i; j++) {

                    double *tmp_a_k_j   = A[k][j];
                    double *tmp_a_j_i   = A[j][i];
                    double *tmp_c_j     = C[j];

                    if (block_rank[j*nt+i] == mype) {
                        if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] == mype) {                            
                            // #pragma omp task
                            #pragma omp target map(to: tmp_a_k_i[0:ts*ts], tmp_a_k_j[0:ts*ts]) map(tofrom: tmp_a_j_i[0:ts*ts]) device(1002)
                            {
                                // do_gemm(A[k][i], A[k][j], A[j][i], ts, ts);
                                do_gemm(tmp_a_k_i, tmp_a_k_j, tmp_a_j_i, ts, ts);
                            }
                        } else if (block_rank[k*nt+i] != mype && block_rank[k*nt+j] == mype) {
                            // #pragma omp task
                            #pragma omp target map(to: tmp_c_i[0:ts*ts], tmp_a_k_j[0:ts*ts]) map(tofrom: tmp_a_j_i[0:ts*ts]) device(1002)
                            {
                                do_gemm(tmp_c_i, tmp_a_k_j, tmp_a_j_i, ts, ts);
                                // do_gemm(C[i], A[k][j], A[j][i], ts, ts);
                            }
                        } else if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] != mype) {
                            // #pragma omp task
                            #pragma omp target map(to: tmp_a_k_i[0:ts*ts], tmp_c_j[0:ts*ts]) map(tofrom: tmp_a_j_i[0:ts*ts]) device(1002)
                            {
                                do_gemm(tmp_a_k_i, tmp_c_j, tmp_a_j_i, ts, ts);
                                // do_gemm(A[k][i], C[j], A[j][i], ts, ts);
                            }
                        } else {
                            // #pragma omp task
                            #pragma omp target map(to: tmp_c_i[0:ts*ts], tmp_c_j[0:ts*ts]) map(tofrom: tmp_a_j_i[0:ts*ts]) device(1002)
                            {
                                do_gemm(tmp_c_i, tmp_c_j, tmp_a_j_i, ts, ts);
                                // do_gemm(C[i], C[j], A[j][i], ts, ts);
                            }
                        }
                    }
                }

                if (block_rank[i*nt+i] == mype) {
                    if (block_rank[k*nt+i] == mype) {
                        // #pragma omp task
                        #pragma omp target map(tofrom: tmp_a_k_i[0:ts*ts]) map(to: tmp_a_i_i[0:ts*ts]) device(1002)
                        {
                            do_syrk(tmp_a_k_i, tmp_a_i_i, ts, ts);
                            // do_syrk(A[k][i], A[i][i], ts, ts);
                        }
                    } else {
                        // #pragma omp task
                        #pragma omp target map(tofrom: tmp_c_i[0:ts*ts]) map(to: tmp_a_i_i[0:ts*ts]) device(1002)
                        {
                            do_syrk(tmp_c_i, tmp_a_i_i, ts, ts);
                            // do_syrk(C[i], A[i][i], ts, ts);
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

