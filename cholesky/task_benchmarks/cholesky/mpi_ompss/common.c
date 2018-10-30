
#include "common.h"

/* Computation */
void do_potrf(double * const A, int ts, int ld)
{
    static int INFO;
    static const char L = 'L';

    LAPACKE_dpotrf(LAPACK_COL_MAJOR, L, ts, A, ts);
}

void do_trsm(double *A, double *B, int ts, int ld)
{
    static char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
    static double DONE = 1.0;

    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                ts, ts, DONE, A, ld, B, ld);
}

void do_gemm(double *A, double *B, double *C, int ts, int ld)
{
    static const char TR = 'T', NT = 'N';
    static double DONE = 1.0, DMONE = -1.0;

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, ts, ts, ts, DMONE, A, ld,
                B, ld, DONE, C, ld);
}

void do_syrk(double *A, double *B, int ts, int ld)
{
    static char LO = 'L', NT = 'N';
    static double DONE = 1.0, DMONE = -1.0;

    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, ts, ts, DMONE, A, ld,
                DONE, B, ld);
}


/* Communication */
void do_mpi_send(double *buf, int size, MPI_Datatype data_type, int dst, int tag)
{
    MPI_Request send_req;

    MPI_Isend(buf, size, data_type, dst, tag, MPI_COMM_WORLD, &send_req);

    test_and_yield(&send_req);
}

void do_mpi_recv(double *buf, int size, MPI_Datatype data_type, int src, int tag)
{
    MPI_Request recv_req;

    MPI_Irecv(buf, size, data_type, src, tag, MPI_COMM_WORLD, &recv_req);

    test_and_yield(&recv_req);
}

void do_mpi_rput(double *buf, int size, MPI_Datatype data_type, int target, int disp, MPI_Win win)
{
    MPI_Request rput_req;

    MPI_Rput(buf, size, data_type, target, disp, size, data_type, win, &rput_req);

    test_and_yield(&rput_req);

    MPI_Win_flush(target, win);
}

inline void test_and_yield(MPI_Request *comm_req)
{
    int comm_comp = 0;

    MPI_Test(comm_req, &comm_comp, MPI_STATUS_IGNORE);
    while (!comm_comp) {
#pragma omp taskyield
        MPI_Test(comm_req, &comm_comp, MPI_STATUS_IGNORE);
    }
}

inline void testall_and_yield(int comm_cnt, MPI_Request *comm_reqs)
{
    if (!comm_cnt) return;

    int comm_comp = 0;

    MPI_Testall(comm_cnt, comm_reqs, &comm_comp, MPI_STATUSES_IGNORE);
    while (!comm_comp) {
#pragma omp taskyield
        MPI_Testall(comm_cnt, comm_reqs, &comm_comp, MPI_STATUSES_IGNORE);
    }
}

inline void reset_send_flags(char *send_flags)
{
    for (int i = 0; i < np; i++) send_flags[i] = 0;
}

inline int get_send_flags(char *send_flags, int *block_rank, int itr1_str,
                          int itr1_end, int itr2_str, int itr2_end, int n)
{
    int send_cnt = 0;
    for (int i = itr1_str; i <= itr1_end; i++) {
        for (int j = itr2_str; j <= itr2_end; j++) {
            if (!send_flags[block_rank[i*n+j]]) {
                send_flags[block_rank[i*n+j]] = 1;
                send_cnt++;
            }
        }
    }
    return send_cnt;
}

inline void get_recv_flag(char *recv_flag, int *block_rank, int itr1_str,
                          int itr1_end, int itr2_str, int itr2_end, int n)
{
    if (*recv_flag == 1) return;

    for (int i = itr1_str; i <= itr1_end; i++) {
        for (int j = itr2_str; j <= itr2_end; j++) {
            if (block_rank[i*n+j] == mype) {
                *recv_flag = 1;
            }
        }
    }
}

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec * 1.0e-6 + (double)tv.tv_sec;
}
