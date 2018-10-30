
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
    int k, x, xs, xe, ys, ye;
    int xl, xu, yl, yu;
    int upper, lower, left, right;
    int bx, by, max_bx, max_by;
    int upper_send, lower_send, left_send, right_send;
    int upper_recv, lower_recv, left_recv, right_recv;

    max_bx = xmax / bsize;
    max_by = ymax / bsize;

    MPI_Cart_shift(cart_comm, 0, 1, &upper, &lower);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

    for (k = 0; k < niter; k++) {

        for (bx = 0; bx < max_bx; bx++) {
            for (by = 0; by < max_by; by++) {
#pragma omp task in(u[bx][by]) \
                 out(uu[bx][by]) \
                 firstprivate(bx, by) 
{
                xs = (bx != 0) ? bx * bsize : 1;
                xe = (bx != max_bx - 1) ? (bx + 1) * bsize : xmax - 1;
                ys = (by != 0) ? by * bsize : 1;
                ye = (by != max_by - 1) ? (by + 1) * bsize : ymax - 1;
                copy(xs, xe, ys, ye);
}
            }
        }

        for (bx = 0; bx < max_bx; bx++) {
#pragma omp task in(uu[bx][0]) \
                 firstprivate(bx) 
{
            if (left  != MPI_PROC_NULL) {
                pack(uu_left_pack, bx*bsize, (bx+1)*bsize, 1);
                do_mpi_send(&uu_left_pack[bx*bsize], bsize, MPI_DOUBLE, left, bx);
            }
}
#pragma omp task in(uu[bx][max_by-1]) \
                 firstprivate(bx) 
{
            if (right != MPI_PROC_NULL) {
                pack(uu_right_pack, bx*bsize, (bx+1)*bsize, ymax-2);
                do_mpi_send(&uu_right_pack[bx*bsize], bsize, MPI_DOUBLE, right, bx+max_bx);
            }
}
        }
        for (by = 0; by < max_by; by++) {
#pragma omp task in(uu[0][by]) \
                 firstprivate(by) 
{
            if (upper != MPI_PROC_NULL) {
                do_mpi_send(&uu[1][by*bsize], bsize, MPI_DOUBLE, upper, by);
            }
}
#pragma omp task in(uu[max_bx-1][by]) \
                 firstprivate(by) 
{
            if (lower != MPI_PROC_NULL) {
                do_mpi_send(&uu[xmax-2][by*bsize], bsize, MPI_DOUBLE, lower, by+max_by);
            }
}
	}

        for (bx = 0; bx < max_bx; bx++) {
#pragma omp task out(uu[bx][max_by-1]) \
                 firstprivate(bx) 
{
            if (right != MPI_PROC_NULL) {
                do_mpi_recv(&uu_right_unpack[bx*bsize], bsize, MPI_DOUBLE, right, bx);
                unpack(uu_right_unpack, bx*bsize, (bx+1)*bsize, ymax-1);
            }
}
#pragma omp task out(uu[bx][0]) \
                 firstprivate(bx) 
{
            if (left != MPI_PROC_NULL) {
                do_mpi_recv(&uu_left_unpack[bx*bsize], bsize, MPI_DOUBLE, left, bx+max_bx);
                unpack(uu_left_unpack, bx*bsize, (bx+1)*bsize, 0);
            }
}
        }
        for (by = 0; by < max_by; by++) {
#pragma omp task out(uu[max_bx-1][by]) \
                 firstprivate(by) 
{
            if (lower != MPI_PROC_NULL) {
                do_mpi_recv(&uu[xmax-1][by*bsize], bsize, MPI_DOUBLE, lower, by);
            }
}
#pragma omp task out(uu[0][by]) \
                 firstprivate(by) 
{
            if (upper != MPI_PROC_NULL) {
                do_mpi_recv(&uu[0][by*bsize], bsize, MPI_DOUBLE, upper, by+max_by);
            }
}
        }

        for (bx = 0; bx < max_bx; bx++) {
            for (by = 0; by < max_by; by++) {
                xl = (bx == 0) ? 0 : 1;
                xu = (bx == max_bx - 1) ? 0 : 1;
                yl = (by == 0) ? 0 : 1;
                yu = (by == max_by - 1) ? 0 : 1;
#pragma omp task in(uu[bx-xl][by], \
                    uu[bx+xu][by], \
                    uu[bx][by-yl], \
                    uu[bx][by+yu], \
                    uu[bx][by]) \
                 out(u[bx][by]) \
                 firstprivate(bx, by) 
{
                xs = (bx != 0) ? bx * bsize : 1;
                xe = (bx != max_bx - 1) ? (bx + 1) * bsize : xmax - 1;
                ys = (by != 0) ? by * bsize : 1;
                ye = (by != max_by - 1) ? (by + 1) * bsize : ymax - 1;
                calc(xs, xe, ys, ye);
}
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
