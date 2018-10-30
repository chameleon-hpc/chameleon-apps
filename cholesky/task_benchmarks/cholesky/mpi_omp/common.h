#ifndef _BENCH_CHOLESKY_COMMON_
#define _BENCH_CHOLESKY_COMMON_

#include <mpi.h>
#include <mkl.h>
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <sys/time.h>

//#include "timer.h"

#ifdef _USE_HBW
#include <hbwmalloc.h>
#define _MALLOC(a,b) hbw_posix_memalign((void **)&a, 64, b)
#define _FREE(a) hbw_free(a)
#else
#define _MALLOC(a,b) MPI_Alloc_mem(b, MPI_INFO_NULL, &a)
#define _FREE(a) free(a)
#endif

#define STACK_SIZE 512

struct _comm_req {
    double *buf;
    int size;
    int target;
    int tag;
    MPI_Datatype data_type;
    int disp;
    MPI_Win win;
    MPI_Request req;
    int comm_comp;
    struct _comm_req *next;
};

/* common.c */
void do_potrf(double * const A, int ts, int ld);
void do_trsm(double *A, double *B, int ts, int ld);
void do_gemm(double *A, double *B, double *C, int ts, int ld);
void do_syrk(double *A, double *B, int ts, int ld);

void do_mpi_send(double *buf, int size, MPI_Datatype data_type, int dst, int tag);
void do_mpi_recv(double *buf, int size, MPI_Datatype data_type, int src, int tag);
void do_mpi_rput(double *buf, int size, MPI_Datatype data_type, int target, int disp, MPI_Win win);

void do_mpi_send_on_comm_thread(double *buf, int size, MPI_Datatype data_type, int dst, int tag);
void do_mpi_recv_on_comm_thread(double *buf, int size, MPI_Datatype data_type, int src, int tag);
void do_mpi_rput_on_comm_thread(double *buf, int size, MPI_Datatype data_type, int target,
                                int disp, MPI_Win win);

void test_and_yield(MPI_Request *comm_req);
void testall_and_yield(int comm_cnt, MPI_Request *comm_reqs);

void reset_send_flags(char *send_flags);
int get_send_flags(char *send_flags, int *block_rank, int itr1_str, int itr1_end, int itr2_str,
                   int itr2_end, int n);
void get_recv_flag(char *recv_flag, int *block_rank, int itr1_str, int itr1_end,  int itr2_str,
                   int itr2_end, int n);

void check_comm_status(struct _comm_req *req);
void push_comm_req(struct _comm_req *req, struct _comm_req **top);
struct _comm_req* pop_comm_req(struct _comm_req **top);
void set_comm_req(struct _comm_req *req, double *buf, int size, int target, int tag,
                  MPI_Datatype data_type, int disp, MPI_Win win);

double get_time();

/* omp_task.c */
double do_cholesky_ser(const int ts, const int nt, double* A[nt][nt]);

/* mpi_*.c */
double do_cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt],
                       int *block_rank);


#ifdef MAIN
int np;
int mype;
int num_threads;
struct _comm_req *top_send_req;
struct _comm_req *top_recv_req;
struct _comm_req *top_put_req;
#else
extern int np;
extern int mype;
extern int num_threads;
extern struct _comm_req *top_send_req;
extern struct _comm_req *top_recv_req;
extern struct _comm_req *top_put_req;
#endif

#endif
