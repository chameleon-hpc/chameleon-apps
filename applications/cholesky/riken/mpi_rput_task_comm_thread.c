
#include "common.h"

static struct _comm_req *_send_reqs_pool[STACK_SIZE];
static struct _comm_req *_recv_reqs_pool[STACK_SIZE];
static struct _comm_req *_put_reqs_pool[STACK_SIZE];

static void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt],
                         int *block_rank, MPI_Win *winB, MPI_Win *winC);

double do_cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt],
                       int *block_rank)
{
    double time;
    MPI_Win *winB, *winC;

    MPI_Alloc_mem(1  * sizeof(MPI_Win), MPI_INFO_NULL, &winB);
    MPI_Alloc_mem(nt * sizeof(MPI_Win), MPI_INFO_NULL, &winC);

    MPI_Win_create(B, ts*ts*sizeof(double), sizeof(double), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &winB[0]);
    for (int i = 0; i < nt; i++) {
        MPI_Win_create(C[i], ts*ts*sizeof(double), sizeof(double), MPI_INFO_NULL,
                      MPI_COMM_WORLD, &winC[i]);
    }

    MPI_Win_lock_all(0, winB[0]);
    for (int i = 0; i < nt; i++) {
        MPI_Win_lock_all(0, winC[i]);
    }

    time = get_time();
    cholesky_mpi(ts, nt, (double* (*)[nt])A, B, C, block_rank, winB, winC);
    time = get_time() - time;

    MPI_Win_unlock_all(winB[0]);
    for (int i = 0; i < nt; i++) {
        MPI_Win_unlock_all(winC[i]);
    }

    free(winB);
    free(winC);

    return time;
}

static void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt],
                         int *block_rank, MPI_Win *winB, MPI_Win *winC)
{
    char recv_flag = 0, *send_flags = malloc(sizeof(char) * np);
    int finish_flag = 0;

    top_send_req = top_recv_req = top_put_req = NULL;

#pragma omp parallel
{
/* Comm thread */
#pragma omp master
{
    int send_req_num, recv_req_num, put_req_num, comm_comp;
    struct _comm_req *req;

    send_req_num = recv_req_num = put_req_num = comm_comp = 0;

    while (1) {
        /* Send */
        while ((req = pop_comm_req(&top_send_req)) != NULL) {
            if (send_req_num == STACK_SIZE) {
                push_comm_req(req, &top_send_req);
                break;
            }
            MPI_Isend(req->buf, req->size, req->data_type, req->target,
                      req->tag, MPI_COMM_WORLD, &req->req);
            _send_reqs_pool[send_req_num++] = req;
        }
        /* Recv */
        while ((req = pop_comm_req(&top_recv_req)) != NULL) {
            if (recv_req_num == STACK_SIZE) {
                push_comm_req(req, &top_recv_req);
                break;
            }
            MPI_Irecv(req->buf, req->size, req->data_type, req->target,
                      req->tag, MPI_COMM_WORLD, &req->req);
            _recv_reqs_pool[recv_req_num++] = req;
        }
        /* Rput */
        while ((req = pop_comm_req(&top_put_req)) != NULL) {
            if (put_req_num == STACK_SIZE) {
                push_comm_req(req, &top_put_req);
                break;
            }
            MPI_Rput(req->buf, req->size, req->data_type, req->target, req->disp,
                     req->size, req->data_type, req->win, &req->req);
            _put_reqs_pool[put_req_num++] = req;
        }
        for (int j = send_req_num-1; j >= 0; j--) {
            comm_comp = 0;
            MPI_Test(&(_send_reqs_pool[j]->req), &comm_comp, MPI_STATUS_IGNORE);
            if (comm_comp) {
                while (!__sync_bool_compare_and_swap(&_send_reqs_pool[j]->comm_comp,
                       0, 1));
                for (int k = j+1; k < send_req_num; k++) {
                    _send_reqs_pool[k-1] = _send_reqs_pool[k];
                }
                send_req_num = (send_req_num == 0) ? 0 : send_req_num - 1;
            }
        }
        for (int j = recv_req_num-1; j >= 0; j--) {
            comm_comp = 0;
            MPI_Test(&(_recv_reqs_pool[j]->req), &comm_comp, MPI_STATUS_IGNORE);
            if (comm_comp) {
                while (!__sync_bool_compare_and_swap(&_recv_reqs_pool[j]->comm_comp,
                       0, 1));
                for (int k = j+1; k < recv_req_num; k++) {
                    _recv_reqs_pool[k-1] = _recv_reqs_pool[k];
                }
                recv_req_num = (recv_req_num == 0) ? 0 : recv_req_num - 1;
            }
        }
        for (int j = put_req_num-1; j >= 0; j--) {
            comm_comp = 0;
            MPI_Test(&(_put_reqs_pool[j]->req), &comm_comp, MPI_STATUS_IGNORE);
            if (comm_comp) {
                MPI_Win_flush(_put_reqs_pool[j]->target, _put_reqs_pool[j]->win);
                while (!__sync_bool_compare_and_swap(&_put_reqs_pool[j]->comm_comp,
                       0, 1));
                for (int k = j+1; k < put_req_num; k++) {
                    _put_reqs_pool[k-1] = _put_reqs_pool[k];
                }
                put_req_num = (put_req_num == 0) ? 0 : put_req_num - 1;
            }
        }
        if (finish_flag == 1 && send_req_num == 0 && recv_req_num == 0 && put_req_num == 0) break;
    }
} /* end omp master */

#pragma omp single
{
    for (int k = 0; k < nt; k++) {
        if (block_rank[k*nt+k] == mype) {
#pragma omp task depend(out:A[k][k]) firstprivate(k) 
            do_potrf(A[k][k], ts, ts);
        }

        if (block_rank[k*nt+k] == mype && np != 1) {

            reset_send_flags(send_flags);
            get_send_flags(send_flags, block_rank, k, k, k+1, nt-1, nt);

            for (int dst = 0; dst < np; dst++) {
                if (send_flags[dst] && dst != mype) {
#pragma omp task depend(in:A[k][k]) firstprivate(k, dst) 
{
                    do_mpi_recv_on_comm_thread(NULL, 0, MPI_CHAR, dst, k*nt+k);
                    do_mpi_rput_on_comm_thread(A[k][k], ts*ts, MPI_DOUBLE, dst, 0, winB[0]);
                    do_mpi_send_on_comm_thread(NULL, 0, MPI_CHAR, dst, k*nt+k);
}
                }
            }
        }

        if (block_rank[k*nt+k] != mype) {

            recv_flag = 0;
            get_recv_flag(&recv_flag, block_rank, k, k, k+1, nt-1, nt);

            if (recv_flag) {
#pragma omp task depend(out:B) firstprivate(k) 
{
                do_mpi_send_on_comm_thread(NULL, 0, MPI_CHAR, block_rank[k*nt+k], k*nt+k);
                do_mpi_recv_on_comm_thread(NULL, 0, MPI_CHAR, block_rank[k*nt+k], k*nt+k);
}
            }
        }
    
        for (int i = k + 1; i < nt; i++) {

            if (block_rank[k*nt+i] == mype) {
                if (block_rank[k*nt+k] == mype) {
#pragma omp task depend(in:A[k][k]) depend(out:A[k][i]) firstprivate(k, i) 
                    do_trsm(A[k][k], A[k][i], ts, ts);
                } else {
#pragma omp task depend(in:B) depend(out:A[k][i]) firstprivate(k, i) 
                    do_trsm(B, A[k][i], ts, ts);
                }
            }

            if (block_rank[k*nt+i] == mype && np != 1) {

                reset_send_flags(send_flags);
                get_send_flags(send_flags, block_rank, k+1, i-1, i, i, nt);
                get_send_flags(send_flags, block_rank, i, i, i+1, nt-1, nt);
                get_send_flags(send_flags, block_rank, i, i, i, i, nt);

                for (int dst = 0; dst < np; dst++) {
                    if (send_flags[dst] && dst != mype) {
#pragma omp task depend(in:A[k][i]) firstprivate(k, i, dst) 
{
                        do_mpi_recv_on_comm_thread(NULL, 0, MPI_CHAR, dst, k*nt+i);
                        do_mpi_rput_on_comm_thread(A[k][i], ts*ts, MPI_DOUBLE, dst, 0, winC[i]);
                        do_mpi_send_on_comm_thread(NULL, 0, MPI_CHAR, dst, k*nt+i);
}
                    }
                }
            }
            if (block_rank[k*nt+i] != mype) {

                recv_flag = 0;
                get_recv_flag(&recv_flag, block_rank, k+1, i-1, i, i, nt);
                get_recv_flag(&recv_flag, block_rank, i, i, i+1, nt-1, nt);
                get_recv_flag(&recv_flag, block_rank, i, i, i, i, nt);

                if (recv_flag) {
#pragma omp task depend(out:C[i]) firstprivate(k, i) 
{
                    do_mpi_send_on_comm_thread(NULL, 0, MPI_CHAR, block_rank[k*nt+i], k*nt+i);
                    do_mpi_recv_on_comm_thread(NULL, 0, MPI_CHAR, block_rank[k*nt+i], k*nt+i);
}
                }
            }
        }

        for (int i = k + 1; i < nt; i++) {

            for (int j = k + 1; j < i; j++) {
                if (block_rank[j*nt+i] == mype) {
                    if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] == mype) {
#pragma omp task depend(in:A[k][i], A[k][j]) depend(out:A[j][i]) firstprivate(k, j, i) 
                        do_gemm(A[k][i], A[k][j], A[j][i], ts, ts);
                    } else if (block_rank[k*nt+i] != mype && block_rank[k*nt+j] == mype) {
#pragma omp task depend(in:C[i], A[k][j]) depend(out:A[j][i]) firstprivate(k, j, i) 
                        do_gemm(C[i], A[k][j], A[j][i], ts, ts);
                    } else if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] != mype) {
#pragma omp task depend(in:A[k][i], C[j]) depend(out:A[j][i]) firstprivate(k, j, i) 
                        do_gemm(A[k][i], C[j], A[j][i], ts, ts);
                    } else {
#pragma omp task depend(in:C[i], C[j]) depend(out:A[j][i]) firstprivate(k, j, i) 
                        do_gemm(C[i], C[j], A[j][i], ts, ts);
                    }
                }
            }

            if (block_rank[i*nt+i] == mype) {
                if (block_rank[k*nt+i] == mype) {
#pragma omp task depend(in:A[k][i]) depend(out:A[i][i]) firstprivate(k, i) 
                    do_syrk(A[k][i], A[i][i], ts, ts);
                } else {
#pragma omp task depend(in:C[i]) depend(out:A[i][i]) firstprivate(k, i) 
                    do_syrk(C[i], A[i][i], ts, ts);
                }
            }
        }
    }
#pragma omp taskwait
    finish_flag = 1;
} /* end omp single */
} /* end omp parallel */
    MPI_Barrier(MPI_COMM_WORLD);
    free(send_flags);
}

