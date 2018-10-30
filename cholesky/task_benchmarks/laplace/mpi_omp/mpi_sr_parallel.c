
#include "common.h"

static void lap_mpi_sr();

double lap_main()
{
    double time;

    MPI_Barrier(MPI_COMM_WORLD);

    time = get_time();
    lap_mpi_sr();
    time = get_time() - time;

    return time;
}

static void lap_mpi_sr()
{
    int x, y, k, xs, xe, ys, ye;
    int upper, lower, left, right;
    int bx, by, max_bx, max_by;

    max_bx = (xmax / bsize);
    max_by = (ymax / bsize);

    MPI_Cart_shift(cart_comm, 0, 1, &upper, &lower);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

#pragma omp parallel private(k, x, y, xs, xe, ys, ye, bx, by) shared(u, uu)
{
    for (k = 0; k < niter; k++) {

#pragma omp for collapse(2) schedule(dynamic)
        for (bx = 0; bx < max_bx; bx++) {
            for (by = 0; by < max_by; by++) {
                xs = (bx != 0) ? bx * bsize : 1;
                xe = (bx != max_bx - 1) ? (bx + 1) * bsize : xmax - 1;
                ys = (by != 0) ? by * bsize : 1;
                ye = (by != max_by - 1) ? (by + 1) * bsize : ymax - 1;
                copy(xs, xe, ys, ye);
            }
        }
#pragma omp single
{
        MPI_Isend(&uu[1][1],           ymax - 2, MPI_DOUBLE, upper, 0, cart_comm, req);
        MPI_Isend(&uu[xmax-2][1],      ymax - 2, MPI_DOUBLE, lower, 1, cart_comm, req+1);

        if (left  != MPI_PROC_NULL)
            pack(uu_left_pack, 0, xmax, 1);
        if (right != MPI_PROC_NULL)
            pack(uu_right_pack, 0, xmax, ymax-2);

        MPI_Isend(&uu_left_pack[1],    xmax - 2, MPI_DOUBLE, left,  2, cart_comm, req+2);
        MPI_Isend(&uu_right_pack[1],   xmax - 2, MPI_DOUBLE, right, 3, cart_comm, req+3);

        MPI_Irecv(&uu[xmax-1][1],      ymax - 2, MPI_DOUBLE, lower, 0, cart_comm, req+4);
        MPI_Irecv(&uu[0][1],           ymax - 2, MPI_DOUBLE, upper, 1, cart_comm, req+5);
        MPI_Irecv(&uu_right_unpack[1], xmax - 2, MPI_DOUBLE, right, 2, cart_comm, req+6);
        MPI_Irecv(&uu_left_unpack[1],  xmax - 2, MPI_DOUBLE, left,  3, cart_comm, req+7);

        MPI_Waitall(8, req, MPI_STATUSES_IGNORE);

        if (right != MPI_PROC_NULL)
            unpack(uu_right_unpack, 0, xmax, ymax-1);
        if (left  != MPI_PROC_NULL)
            unpack(uu_left_unpack, 0, xmax, 0);
}

#pragma omp for collapse(2) schedule(dynamic)
        for (bx = 0; bx < max_bx; bx++) {
            for (by = 0; by < max_by; by++) {
                xs = (bx != 0) ? bx * bsize : 1;
                xe = (bx != max_bx - 1) ? (bx + 1) * bsize : xmax - 1;
                ys = (by != 0) ? by * bsize : 1;
                ye = (by != max_by - 1) ? (by + 1) * bsize : ymax - 1;
                calc(xs, xe, ys, ye);
            }
        }
    }
} /* end omp parallel */
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_header()
{
#pragma omp parallel
#pragma omp single
    if (rank == 0) {
        fprintf(stderr, "Laplace BLOCK OMP FOR version\n");
        fprintf(stderr, "Procs(%d, %d), num_threads:%d\n", ndx, ndy, omp_get_num_threads());
        fprintf(stderr, "The number of iterations = %d\n", niter);
        fprintf(stderr, "Matrix Size = %d x %d\n", size, size);
    }
}
