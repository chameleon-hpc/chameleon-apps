
#include "common.h"

inline void test_and_yield(MPI_Request *comm_req)
{
    int comm_comp = 0;
    MPI_Test(comm_req, &comm_comp, MPI_STATUS_IGNORE);
    while (!comm_comp) {
#pragma omp taskyield
        comm_comp = 0;
        MPI_Test(comm_req, &comm_comp, MPI_STATUS_IGNORE);
    }
}

inline void copy(int xs, int xe, int ys, int ye)
{
    int x, y;
    for (x = xs; x < xe; x++) {
        for (y = ys; y < ye; y++) {
            uu[x][y] = u[x][y];
        }
    }
}

inline void calc(int xs, int xe, int ys, int ye)
{
    int x, y;
    for (x = xs; x < xe; x++) {
        for (y = ys; y < ye; y++) {
            u[x][y] = (uu[x-1][y] + uu[x+1][y] + uu[x][y-1] + uu[x][y+1]) / 4.0;
        }
    }
}

inline void pack(double *buf, int x_str, int x_end, int y_ind)
{
    int x;
    for (x = x_str; x < x_end; x++) {
        buf[x] = uu[x][y_ind];
    }
}

inline void unpack(double *buf, int x_str, int x_end, int y_ind)
{
    int x;
    for (x = x_str; x < x_end; x++) {
        uu[x][y_ind] = buf[x];
    }
}

inline double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_usec * 1.0e-6 + tv.tv_sec;
}

void init_comm()
{
    int dims[2] = {0};
    int periods[2] = {0};
    int cart_rank[2];

    MPI_Dims_create(np, 2, dims);
    ndx = dims[0];
    ndy = dims[1];

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, cart_rank);

    xrank = cart_rank[0];
    yrank = cart_rank[1];

    xmax = size / ndx;
    xoffset = xmax * xrank;
    if (xrank == ndx - 1) xmax = size - xoffset;
    ymax = size / ndy;
    yoffset = ymax * yrank;
    if (yrank == ndy - 1) ymax = size - yoffset;

    if(xrank != 0) xmax++;
    if(yrank != 0) ymax++;
    if(xrank != ndx - 1) xmax++;
    if(yrank != ndy - 1) ymax++;

    if(xrank != 0) xoffset--;
    if(yrank != 0) yoffset--;
}

double verify()
{
    int x, y;
    double lsum = 0.0;
    double sum = 0.0;

#pragma omp parallel for private(x, y) reduction(+:lsum)
    for (x = 1; x < xmax - 1; x++) {
        for (y = 1; y < ymax - 1; y++) {
            lsum += uu[x][y] - u[x][y];
        }
    }

    MPI_Reduce(&lsum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return sum;
}

void check_comm_status(struct _comm_req *req)
{
    while (!__sync_bool_compare_and_swap(&req->comm_comp, 1, 1)) {
#pragma omp taskyield
    }
}

void push_comm_req(struct _comm_req *req, struct _comm_req **top)
{
    struct _comm_req *old_top;
    do {
        req->next = old_top = *top;
    } while (!__sync_bool_compare_and_swap(top, old_top, req));
}

struct _comm_req* pop_comm_req(struct _comm_req **top)
{
    struct _comm_req *old_top, *next_top;
    do {
        old_top = *top;
        if (old_top == NULL) return NULL;
        next_top = old_top->next;
    } while (!__sync_bool_compare_and_swap(top, old_top, next_top));
    return old_top;
}

void set_comm_req(struct _comm_req *req, double *buf, int size, int target, int tag,
                  MPI_Datatype data_type, int disp, MPI_Win win)
{
    req->buf        = buf;
    req->size       = size;
    req->target     = target;
    req->tag        = tag;
    req->data_type  = data_type;
    req->comm_comp  = 0;
    req->disp       = disp;
    req->win        = win;
    req->next       = NULL;
}

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

void do_mpi_send_on_comm_thread(double *buf, int size, MPI_Datatype data_type, int dst, int tag)
{
    struct _comm_req *req = malloc(sizeof(struct _comm_req));
    set_comm_req(req, buf, size, dst, tag, data_type, 0, MPI_WIN_NULL);
    push_comm_req(req, &top_send_req);
    check_comm_status(req);
    free(req);
}

void do_mpi_recv_on_comm_thread(double *buf, int size, MPI_Datatype data_type, int src, int tag)
{
    struct _comm_req *req = malloc(sizeof(struct _comm_req));
    set_comm_req(req, buf, size, src, tag, data_type, 0, MPI_WIN_NULL);
    push_comm_req(req, &top_recv_req);
    check_comm_status(req);
    free(req);
}
void do_mpi_rput_on_comm_thread(double *buf, int size, MPI_Datatype data_type, int target,
                                int disp, MPI_Win win)
{
    struct _comm_req *req = malloc(sizeof(struct _comm_req));
    set_comm_req(req, buf, size, target, 0, data_type, disp, win);
    push_comm_req(req, &top_put_req);
    check_comm_status(req);
    free(req);
}
