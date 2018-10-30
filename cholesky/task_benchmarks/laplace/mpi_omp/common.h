#ifndef _BENCH_LAPLACE_COMMON_
#define _BENCH_LAPLACE_COMMON_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <sys/time.h>

#include "timer.h"

#ifdef _USE_HBW
#include <hbwmalloc.h>
#define _MALLOC(a,b) hbw_posix_memalign((void **)&a, 64, b)
#define _FREE(a) hbw_free(a)
#else
#define _MALLOC(a,b) MPI_Alloc_mem(b, MPI_INFO_NULL, &a)
#define _FREE(a) free(a)
#endif

#define STACK_SIZE 512
#define PI M_PI

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

double lap_main(void);
void init_comm(void);
double verify(void);
void print_header(void);
void test_and_yield(MPI_Request *comm_req);

void copy(int xs, int xe, int ys, int ye);
void calc(int xs, int xe, int ys, int ye);
void pack(double *buf, int x_str, int x_end, int y_ind);
void unpack(double *buf, int x_str, int x_end, int y_ind);

double get_time();

void check_comm_status(struct _comm_req *req);
void push_comm_req(struct _comm_req *req, struct _comm_req **top);
struct _comm_req* pop_comm_req(struct _comm_req **top);
void set_comm_req(struct _comm_req *req, double *buf, int size, int target, int tag,
                  MPI_Datatype data_type, int disp, MPI_Win win);

void do_mpi_send(double *buf, int size, MPI_Datatype data_type, int dst, int tag);
void do_mpi_recv(double *buf, int size, MPI_Datatype data_type, int src, int tag);
void do_mpi_rput(double *buf, int size, MPI_Datatype data_type, int target, int disp, MPI_Win win);

void do_mpi_send_on_comm_thread(double *buf, int size, MPI_Datatype data_type, int dst, int tag);
void do_mpi_recv_on_comm_thread(double *buf, int size, MPI_Datatype data_type, int src, int tag);
void do_mpi_rput_on_comm_thread(double *buf, int size, MPI_Datatype data_type, int target,
                                int disp, MPI_Win win);

#ifdef MAIN
double **u, **uu;
double *uu_left_pack, *uu_right_pack, *uu_left_unpack, *uu_right_unpack;
int xrank, yrank, rank, np, niter;
int ndx, ndy;
int xmax, ymax;
int xoffset, yoffset;
int size, bsize;
MPI_Request req[8];
MPI_Comm cart_comm;
struct _comm_req *top_send_req, *top_recv_req, *top_put_req;
#else
extern double **u, **uu;
extern double *uu_left_pack, *uu_right_pack, *uu_left_unpack, *uu_right_unpack;
extern int xrank, yrank, rank, np, niter;
extern int ndx, ndy;
extern int xmax, ymax;
extern int xoffset, yoffset;
extern int size, bsize;
extern MPI_Request req[8];
extern MPI_Comm cart_comm;
extern struct _comm_req *top_send_req, *top_recv_req, *top_put_req;
#endif

#endif
