#ifndef _BENCH_CHOLESKY_COMMON_
#define _BENCH_CHOLESKY_COMMON_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/syscall.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include <mkl.h>
#include <mpi.h>
#include <omp.h>

#ifdef TRACE
#include "VT.h"
#endif

#ifdef MAIN
int np;
int mype;
int num_threads;
#else
extern int np;
extern int mype;
extern int num_threads;
#endif

#if defined(USE_TIMING)
void helper_start_timing(int tt);
void helper_end_timing(int tt, double elapsed);
#endif

// #define SPEC_RESTRICT __restrict__
#define SPEC_RESTRICT restrict

#if defined(CHAMELEON) || defined(CHAMELEON_TARGET)
#include "chameleon.h"
#endif

#ifndef DPxMOD
#define DPxMOD "0x%0*" PRIxPTR
#endif

#ifndef DPxPTR
#define DPxPTR(ptr) ((int)(2*sizeof(uintptr_t))), ((uintptr_t) (ptr))
#endif

#ifdef DEBUG
int* __task_depth_counter;
#define TASK_DEPTH_OFFSET 32
#define TASK_DEPTH_INIT do { \
    __task_depth_counter = (int*) malloc((omp_get_max_threads()+1)*TASK_DEPTH_OFFSET*sizeof(int)); \
    for (int i = 0; i < omp_get_max_threads(); i++) { \
        __task_depth_counter[i*TASK_DEPTH_OFFSET] = 0; \
    } \
} while(0)
#define TASK_DEPTH_FINALIZE \
    if(__task_depth_counter) \
        free(__task_depth_counter);

#define TASK_DEPTH_INCR ++__task_depth_counter[omp_get_thread_num()*TASK_DEPTH_OFFSET]
#define TASK_DEPTH_DECR --__task_depth_counter[omp_get_thread_num()*TASK_DEPTH_OFFSET]
#define TASK_DEPTH_GET __task_depth_counter[omp_get_thread_num()*TASK_DEPTH_OFFSET]
#define DEBUG_PRINT(STR, ...) do { \
    char* tmp_str = malloc(sizeof(char)*512); \
    tmp_str[0] = '\0'; \
    strcat(tmp_str,"R#%02d T#%02d (OS_TID:%06ld) Task-Depth:%02d : --> "); \
    strcat(tmp_str,STR); \
    fprintf(stderr, tmp_str, mype, omp_get_thread_num(), syscall(SYS_gettid), TASK_DEPTH_GET, __VA_ARGS__); \
    free(tmp_str); \
} while(0)
#else
#define TASK_DEPTH_INIT
#define TASK_DEPTH_FINALIZE
#define TASK_DEPTH_INCR
#define TASK_DEPTH_DECR
#define TASK_DEPTH_GET 0
#define DEBUG_PRINT(STR, ...)
#endif

#ifdef _USE_HBW
#include <hbwmalloc.h>
#endif

void dgemm_ (const char *transa, const char *transb, int *l, int *n, int *m, double *alpha,
             const void *a, int *lda, void *b, int *ldb, double *beta, void *c, int *ldc);

void dtrsm_ (char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha,
             double *a, int *lda, double *b, int *ldb);

void dsyrk_ (char *uplo, char *trans, int *n, int *k, double *alpha, double *a, int *lda,
             double *beta, double *c, int *ldc);

void cholesky_single(const int ts, const int nt, double* A[nt][nt]);
void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt], int *block_rank);

void omp_potrf(double * SPEC_RESTRICT const A, int ts, int ld);
void omp_trsm(double * SPEC_RESTRICT A, double * SPEC_RESTRICT B, int ts, int ld);
void omp_gemm(double * SPEC_RESTRICT A, double * SPEC_RESTRICT B, double * SPEC_RESTRICT C, int ts, int ld);
void omp_syrk(double * SPEC_RESTRICT A, double * SPEC_RESTRICT B, int ts, int ld);

int get_send_flags(char *send_flags, int *block_rank, int itr1_str, int itr1_end, int itr2_str, int itr2_end, int n);
void get_recv_flag(char *recv_flag, int *block_rank, int itr1_str, int itr1_end, int itr2_str, int itr2_end, int n);

void wait(MPI_Request *comm_req);

inline static void waitall(MPI_Request *comm_req, int n)
{
    #ifdef TRACE
    static int event_waitall = -1;
    if(event_waitall == -1) {
        char* event_name = "waitall";
        int ierr;
        ierr = VT_funcdef(event_name, VT_NOCLASS, &event_waitall);
    }
    VT_begin(event_waitall);
    #endif
    #ifdef DISABLE_TASKYIELD
    MPI_Waitall(n, comm_req, MPI_STATUSES_IGNORE);
    #else
    while (1) {
        int flag = 0;
        MPI_Testall(n, comm_req, &flag, MPI_STATUSES_IGNORE);
        if (flag) break;
        (void)flag; // <-- make the Cray compiler happy
        #if defined(CHAMELEON) || defined(CHAMELEON_TARGET)
            int32_t res = chameleon_taskyield();
        #else
            #pragma omp taskyield
        #endif
    }
    #endif
    #ifdef TRACE
    VT_end(event_waitall);
    #endif
}
void reset_send_flags(char *send_flags);
#endif
