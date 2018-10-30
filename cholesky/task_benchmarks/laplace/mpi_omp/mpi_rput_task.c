
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
    int k;
    int upper, lower, left, right;
    int bx, by, max_bx, max_by;
    int upper_send, lower_send, left_send, right_send;
    int upper_recv, lower_recv, left_recv, right_recv;

    max_bx = (xmax / bsize);
    max_by = (ymax / bsize);

    MPI_Cart_shift(cart_comm, 0, 1, &upper, &lower);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

#pragma omp parallel private(k) shared(u, uu)
{
#pragma omp single
    for (k = 0; k < niter; k++) {

        for (by = 1; by < max_by - 1; by++) {
#pragma omp task depend(in:u[0][by], upper_send) \
                 depend(out:uu[0][by]) firstprivate(by)
            copy(1, bsize, by * bsize, (by + 1) * bsize);
#pragma omp task depend(in:u[max_bx-1][by], lower_send) \
                 depend(out:uu[max_bx-1][by]) firstprivate(by)
            copy((max_bx - 1) * bsize, xmax - 1, by * bsize, (by + 1) * bsize);
        }

        for (bx = 1; bx < max_bx - 1; bx++) {
#pragma omp task depend(in:u[bx][0], left_send) \
                 depend(out:uu[bx][0]) firstprivate(bx)
            copy(bx * bsize, (bx + 1) * bsize, 1, bsize);
#pragma omp task depend(in:u[bx][max_by-1], right_send) \
                 depend(out:uu[bx][max_by-1]) firstprivate(bx)
            copy(bx * bsize, (bx + 1) * bsize, (max_by - 1) * bsize, ymax - 1);
        }

#pragma omp task depend(in:u[0][0], upper_send, left_send) \
                 depend(out:uu[0][0])
        copy(1, bsize, 1, bsize);

#pragma omp task depend(in:u[0][max_by-1], upper_send, right_send) \
                 depend(out:uu[0][max_by-1])
        copy(1, bsize, (max_by - 1) * bsize, ymax - 1);

#pragma omp task depend(in:u[max_bx-1][0], lower_send, left_send) \
                 depend(out:uu[max_bx-1][0])
        copy((max_bx - 1) * bsize, xmax - 1, 1, bsize);

#pragma omp task depend(in:u[max_bx-1][max_by-1], lower_send, right_send) \
                 depend(out:uu[max_bx-1][max_by-1])
        copy((max_bx - 1) * bsize, xmax - 1, (max_by - 1) * bsize, ymax - 1);

        for (bx = 1; bx < max_bx - 1; bx++) {
            for (by = 1; by < max_by - 1; by++) {
#pragma omp task depend(in:u[bx][by]) \
                 depend(out:uu[bx][by]) firstprivate(bx, by)
                copy(bx * bsize, (bx + 1) * bsize, by * bsize, (by + 1) * bsize);
            }
        }

#pragma omp task depend(out:upper_send)
{
        if (upper != MPI_PROC_NULL) {
            do_mpi_recv(NULL, 0, MPI_CHAR, upper, 0);
            do_mpi_rput(&uu[1][1], ymax-2, MPI_DOUBLE, upper, 1, *upper_win);
            do_mpi_send(NULL, 0, MPI_CHAR, upper, 0);
        }
}
#pragma omp task depend(out:lower_recv)
{
        if (lower != MPI_PROC_NULL) {
            do_mpi_send(NULL, 0, MPI_CHAR, lower, 0);
            do_mpi_recv(NULL, 0, MPI_CHAR, lower, 0);
        }
}

#pragma omp task depend(out:lower_send)
{
        if (lower != MPI_PROC_NULL) {
            do_mpi_recv(NULL, 0, MPI_CHAR, lower, 1);
            do_mpi_rput(&uu[xmax-2][1], ymax-2, MPI_DOUBLE, lower, 1, *lower_win);
            do_mpi_send(NULL, 0, MPI_CHAR, lower, 1);
        }
}
#pragma omp task depend(out:upper_recv)
{
        if (upper != MPI_PROC_NULL) {
            do_mpi_send(NULL, 0, MPI_CHAR, upper, 1);
            do_mpi_recv(NULL, 0, MPI_CHAR, upper, 1);
        }
}

#pragma omp task depend(out:left_send)
{
        if (left != MPI_PROC_NULL) {
            pack(uu_left_pack, 0, xmax, 1);

            do_mpi_recv(NULL, 0, MPI_CHAR, left, 2);
            do_mpi_rput(&uu_left_pack[1], xmax-2, MPI_DOUBLE, left, 1, *left_win);
            do_mpi_send(NULL, 0, MPI_CHAR, left, 2);
        }
}
#pragma omp task depend(out:right_recv)
{
        if (right != MPI_PROC_NULL) {
            do_mpi_send(NULL, 0, MPI_CHAR, right, 2);
            do_mpi_recv(NULL, 0, MPI_CHAR, right, 2);

            unpack(uu_left_unpack, 0, xmax, ymax-1);
        }
}

#pragma omp task depend(out:right_send)
{
        if (right != MPI_PROC_NULL) {
            pack(uu_right_pack, 0, xmax, ymax-2);

            do_mpi_recv(NULL, 0, MPI_CHAR, right, 3);
            do_mpi_rput(&uu_right_pack[1], xmax-2, MPI_DOUBLE, right, 1, *right_win);
            do_mpi_send(NULL, 0, MPI_CHAR, right, 3);
        }
}
#pragma omp task depend(out:left_recv)
{
        if (left != MPI_PROC_NULL) {
            do_mpi_send(NULL, 0, MPI_CHAR, left, 3);
            do_mpi_recv(NULL, 0, MPI_CHAR, left, 3);

            unpack(uu_right_unpack, 0, xmax, 0);
        }
}

        for (by = 1; by < max_by - 1; by++) {
#pragma omp task depend(in:uu[0][by], \
                           uu[1][by], \
                           uu[0][by-1], \
                           uu[0][by+1], upper_recv) \
                 depend(out:u[0][by]) firstprivate(by)
            calc(1, bsize, by * bsize, (by + 1) * bsize);
#pragma omp task depend(in:uu[max_bx-1][by], \
                           uu[max_bx-2][by], \
                           uu[max_bx-1][by-1], \
                           uu[max_bx-1][by+1], lower_recv) \
                 depend(out:u[max_bx-1][by]) firstprivate(by)
            calc((max_bx - 1) * bsize, xmax - 1, by * bsize, (by + 1) * bsize);
        }

        for (bx = 1; bx < max_bx - 1; bx++) {
#pragma omp task depend(in:uu[bx][0], \
                           uu[bx][1], \
                           uu[bx-1][0], \
                           uu[bx+1][0], left_recv) \
                 depend(out:u[bx][0]) firstprivate(bx)
            calc(bx * bsize, (bx + 1) * bsize, 1, bsize);
#pragma omp task depend(in:uu[bx][max_by-1], \
                           uu[bx][max_by-2], \
                           uu[bx-1][max_by-1], \
                           uu[bx+1][max_by-1], right_recv) \
                 depend(out:u[bx][max_by-1]) firstprivate(bx)
            calc(bx * bsize, (bx + 1) * bsize, (max_by - 1) * bsize, ymax - 1);
        }

#pragma omp task depend(in:uu[0][0], \
                           uu[1][0], \
                           uu[0][1], upper_recv, left_recv) \
                 depend(out:u[0][0])
        calc(1, bsize, 1, bsize);

#pragma omp task depend(in:uu[0][max_by-1], \
                           uu[1][max_by-1], \
                           uu[0][max_by-2], upper_recv, right_recv) \
                 depend(out:u[0][max_by-1])
        calc(1, bsize, (max_by - 1) * bsize, ymax - 1);

#pragma omp task depend(in:uu[max_bx-1][0], \
                           uu[max_bx-2][0], \
                           uu[max_bx-1][1], lower_recv, left_recv) \
                 depend(out:u[max_bx-1][0])
        calc((max_bx - 1) * bsize, xmax - 1, 1, bsize);

#pragma omp task depend(in:uu[max_bx-1][max_by-1], \
                           uu[max_bx-2][max_by-1], \
                           uu[max_bx-1][max_by-2], lower_recv, right_recv) \
                 depend(out:u[max_bx-1][max_by-1])
        calc((max_bx - 1) * bsize, xmax - 1, (max_by - 1) * bsize, ymax - 1);

        for (bx = 1; bx < max_bx - 1; bx++) {
            for (by = 1; by < max_by - 1; by++) {
#pragma omp task depend(in:uu[bx][by], \
                           uu[bx+1][by], \
                           uu[bx-1][by], \
                           uu[bx][by+1], \
                           uu[bx][by-1]) \
                 depend(out:u[bx][by]) firstprivate(bx, by)
                calc(bx * bsize, (bx + 1) * bsize, by * bsize, (by + 1) * bsize);
            }
        }
    }
} /* end omp paralle */
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_header()
{
#pragma omp parallel
#pragma omp single
    if (rank == 0) {
        fprintf(stderr, "Laplace BLOCK OMP TASK version (bsize:%d)\n", bsize);
        fprintf(stderr, "Procs(%d, %d), num_threads:%d\n", ndx, ndy, omp_get_num_threads());
        fprintf(stderr, "The number of iterations = %d\n", niter);
        fprintf(stderr, "Matrix Size = %d x %d\n", size, size);
    }
}
