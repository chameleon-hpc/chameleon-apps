
#include "common.h"

static void lap_mpi_put(MPI_Win *upper_win, MPI_Win *lower_win, MPI_Win *left_win,
                        MPI_Win *right_win);

double lap_main()
{
    double time;
    MPI_Win *upper_win, *lower_win, *left_win, *right_win;

    MPI_Alloc_mem(sizeof(MPI_Win), MPI_INFO_NULL, &upper_win);
    MPI_Alloc_mem(sizeof(MPI_Win), MPI_INFO_NULL, &lower_win);
    MPI_Alloc_mem(sizeof(MPI_Win), MPI_INFO_NULL, &left_win);
    MPI_Alloc_mem(sizeof(MPI_Win), MPI_INFO_NULL, &right_win);

    MPI_Win_create(&uu[xmax-1][0],  ymax * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &upper_win[0]);
    MPI_Win_create(&uu[0][0],       ymax * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &lower_win[0]);
    MPI_Win_create(uu_left_unpack,  xmax * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &left_win[0]);
    MPI_Win_create(uu_right_unpack, xmax * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &right_win[0]);

    MPI_Win_lock_all(0, upper_win[0]);
    MPI_Win_lock_all(0, lower_win[0]);
    MPI_Win_lock_all(0, left_win[0]);
    MPI_Win_lock_all(0, right_win[0]);

    MPI_Barrier(MPI_COMM_WORLD);

    time = get_time();
    lap_mpi_put(upper_win, lower_win, left_win, right_win);
    time = get_time() - time;

    MPI_Win_unlock_all(upper_win[0]);
    MPI_Win_unlock_all(lower_win[0]);
    MPI_Win_unlock_all(left_win[0]);
    MPI_Win_unlock_all(right_win[0]);

    free(upper_win);
    free(lower_win);
    free(left_win);
    free(right_win);

    return time;
}

static void lap_mpi_put(MPI_Win *upper_win, MPI_Win *lower_win, MPI_Win *left_win,
                        MPI_Win *right_win)
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
        MPI_Barrier(MPI_COMM_WORLD);

        if (left  != MPI_PROC_NULL)
            pack(uu_left_pack, 0, xmax, 1);
        if (right != MPI_PROC_NULL)
            pack(uu_right_pack, 0, xmax, ymax-2);

        MPI_Put(&uu[xmax-2][1],    ymax-2, MPI_DOUBLE, lower, 1, ymax-2, MPI_DOUBLE, *lower_win);
        MPI_Put(&uu[1][1],         ymax-2, MPI_DOUBLE, upper, 1, ymax-2, MPI_DOUBLE, *upper_win);
        MPI_Put(&uu_left_pack[1],  xmax-2, MPI_DOUBLE, left,  1, xmax-2, MPI_DOUBLE, *left_win);
        MPI_Put(&uu_right_pack[1], xmax-2, MPI_DOUBLE, right, 1, xmax-2, MPI_DOUBLE, *right_win);

        MPI_Win_flush_all(*upper_win);
        MPI_Win_flush_all(*lower_win);
        MPI_Win_flush_all(*left_win);
        MPI_Win_flush_all(*right_win);

        MPI_Barrier(MPI_COMM_WORLD);

        if (right != MPI_PROC_NULL)
            unpack(uu_left_unpack, 0, xmax, ymax-1);
        if (left  != MPI_PROC_NULL)
            unpack(uu_right_unpack, 0, xmax, 0);
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
