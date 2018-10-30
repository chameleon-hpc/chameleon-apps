
#include "common.h"

static void lap_mpi_sr();

double lap_main()
{
    double time_start, time_end;

    MPI_Barrier(MPI_COMM_WORLD);

    time_start = get_time();
    lap_mpi_sr();
    time_end   = get_time() - time_start;

    return time_end;
}

static void lap_mpi_sr()
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

    for (k = 0; k < niter; k++) {

        for (by = 1; by < max_by - 1; by++) {
#pragma omp task in(u[0][by], upper_send) \
                 out(uu[0][by]) firstprivate(by)
            copy(1, bsize, by * bsize, (by + 1) * bsize);
#pragma omp task in(u[max_bx-1][by], lower_send) \
                 out(uu[max_bx-1][by]) firstprivate(by)
            copy((max_bx - 1) * bsize, xmax - 1, by * bsize, (by + 1) * bsize);
        }

        for (bx = 1; bx < max_bx - 1; bx++) {
#pragma omp task in(u[bx][0], left_send) \
                 out(uu[bx][0]) firstprivate(bx)
            copy(bx * bsize, (bx + 1) * bsize, 1, bsize);
#pragma omp task in(u[bx][max_by-1], right_send) \
                 out(uu[bx][max_by-1]) firstprivate(bx)
            copy(bx * bsize, (bx + 1) * bsize, (max_by - 1) * bsize, ymax - 1);
        }

#pragma omp task in(u[0][0], upper_send, left_send) \
                 out(uu[0][0])
        copy(1, bsize, 1, bsize);

#pragma omp task in(u[0][max_by-1], upper_send, right_send) \
                 out(uu[0][max_by-1])
        copy(1, bsize, (max_by - 1) * bsize, ymax - 1);

#pragma omp task in(u[max_bx-1][0], lower_send, left_send) \
                 out(uu[max_bx-1][0])
        copy((max_bx - 1) * bsize, xmax - 1, 1, bsize);

#pragma omp task in(u[max_bx-1][max_by-1], lower_send, right_send) \
                 out(uu[max_bx-1][max_by-1])
        copy((max_bx - 1) * bsize, xmax - 1, (max_by - 1) * bsize, ymax - 1);

        for (bx = 1; bx < max_bx - 1; bx++) {
            for (by = 1; by < max_by - 1; by++) {
#pragma omp task in(u[bx][by]) \
                 out(uu[bx][by]) firstprivate(bx, by)
                copy(bx * bsize, (bx + 1) * bsize, by * bsize, (by + 1) * bsize);
            }
        }

#pragma omp task out(upper_send)
{
        if (upper != MPI_PROC_NULL) {
            do_mpi_send(&uu[1][1], ymax-2, MPI_DOUBLE, upper, 0);
        }
}
#pragma omp task out(lower_send)
{
        if (lower != MPI_PROC_NULL) {
            do_mpi_send(&uu[xmax-2][1], ymax-2, MPI_DOUBLE, lower, 1);
        }
}
#pragma omp task out(left_send)
{
        if (left  != MPI_PROC_NULL) {
            pack(uu_left_pack, 0, xmax, 1);
            do_mpi_send(&uu_left_pack[1], xmax-2, MPI_DOUBLE, left, 2);
        }
}
#pragma omp task out(right_send)
{
        if (right != MPI_PROC_NULL) {
            pack(uu_right_pack, 0, xmax, ymax-2);
            do_mpi_send(&uu_right_pack[1], xmax-2, MPI_DOUBLE, right, 3);
        }
}


#pragma omp task out(lower_recv)
{
        if (lower != MPI_PROC_NULL) {
            do_mpi_recv(&uu[xmax-1][1], ymax-2, MPI_DOUBLE, lower, 0);
        }
}
#pragma omp task out(upper_recv)
{
        if (upper != MPI_PROC_NULL) {
            do_mpi_recv(&uu[0][1], ymax-2, MPI_DOUBLE, upper, 1);
        }
}
#pragma omp task out(right_recv)
{
        if (right != MPI_PROC_NULL) {
            do_mpi_recv(&uu_right_unpack[1], xmax-2, MPI_DOUBLE, right, 2);
            unpack(uu_right_unpack, 0, xmax, ymax-1);
        }
}
#pragma omp task out(left_recv)
{
        if (left  != MPI_PROC_NULL) {
            do_mpi_recv(&uu_left_unpack[1], xmax-2, MPI_DOUBLE, left, 3);
            unpack(uu_left_unpack, 0, xmax, 0);
        }
}

        for (by = 1; by < max_by - 1; by++) {
#pragma omp task in(uu[0][by], \
                    uu[1][by], \
                    uu[0][by-1], \
                    uu[0][by+1], upper_recv) \
                 out(u[0][by]) firstprivate(by)
            calc(1, bsize, by * bsize, (by + 1) * bsize);
#pragma omp task in(uu[max_bx-1][by], \
                    uu[max_bx-2][by], \
                    uu[max_bx-1][by-1], \
                    uu[max_bx-1][by+1], lower_recv) \
                 out(u[max_bx-1][by]) firstprivate(by)
            calc((max_bx - 1) * bsize, xmax - 1, by * bsize, (by + 1) * bsize);
        }

        for (bx = 1; bx < max_bx - 1; bx++) {
#pragma omp task in(uu[bx][0], \
                    uu[bx][1], \
                    uu[bx-1][0], \
                    uu[bx+1][0], left_recv) \
                 out(u[bx][0]) firstprivate(bx)
            calc(bx * bsize, (bx + 1) * bsize, 1, bsize);
#pragma omp task in(uu[bx][max_by-1], \
                    uu[bx][max_by-2], \
                    uu[bx-1][max_by-1], \
                    uu[bx+1][max_by-1], right_recv) \
                 out(u[bx][max_by-1]) firstprivate(bx)
            calc(bx * bsize, (bx + 1) * bsize, (max_by - 1) * bsize, ymax - 1);
        }

#pragma omp task in(uu[0][0], \
                    uu[1][0], \
                    uu[0][1], upper_recv, left_recv) \
                 out(u[0][0])
        calc(1, bsize, 1, bsize);

#pragma omp task in(uu[0][max_by-1], \
                    uu[1][max_by-1], \
                    uu[0][max_by-2], upper_recv, right_recv) \
                 out(u[0][max_by-1])
        calc(1, bsize, (max_by - 1) * bsize, ymax - 1);

#pragma omp task in(uu[max_bx-1][0], \
                    uu[max_bx-2][0], \
                    uu[max_bx-1][1], lower_recv, left_recv) \
                 out(u[max_bx-1][0])
        calc((max_bx - 1) * bsize, xmax - 1, 1, bsize);

#pragma omp task in(uu[max_bx-1][max_by-1], \
                    uu[max_bx-2][max_by-1], \
                    uu[max_bx-1][max_by-2], lower_recv, right_recv) \
                 out(u[max_bx-1][max_by-1])
        calc((max_bx - 1) * bsize, xmax - 1, (max_by - 1) * bsize, ymax - 1);

        for (bx = 1; bx < max_bx - 1; bx++) {
            for (by = 1; by < max_by - 1; by++) {
#pragma omp task in(uu[bx][by], \
                    uu[bx+1][by], \
                    uu[bx-1][by], \
                    uu[bx][by+1], \
                    uu[bx][by-1]) \
                 out(u[bx][by]) firstprivate(bx, by)
                calc(bx * bsize, (bx + 1) * bsize, by * bsize, (by + 1) * bsize);
            }
        }
    }
#pragma omp taskwait
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_header()
{
    if (rank == 0) {
        fprintf(stderr, "Laplace BLOCK OMP TASK DEPEND version (bsize:%d)\n", bsize);
        fprintf(stderr, "Procs(%d, %d), num_threads:%d\n", ndx, ndy, omp_get_num_threads());
        fprintf(stderr, "The number of iterations = %d\n", niter);
        fprintf(stderr, "Matrix Size = %d x %d\n", size, size);
    }
}
