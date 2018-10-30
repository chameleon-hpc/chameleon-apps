
#define MAIN

#include "common.h"

int main(int argc, char *argv[]) 
{
    int x, y, provided, num_threads;
    int num_windows, window_size, lock_all_flag;
    double time, value;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        exit(0);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (argc < 3) {
        size = 1024;
        bsize = 128;
    } else {
        size = atoi(argv[1]);
        bsize = atoi(argv[2]);
    }
    niter = 100;

    init_comm();
    print_header();

    _MALLOC(u,  xmax * sizeof(double *));
    _MALLOC(uu, xmax * sizeof(double *));
    for (x = 0; x < xmax; x++) {
        _MALLOC(u[x],   ymax * sizeof(double));
        _MALLOC(uu[x],  ymax * sizeof(double));
    }
    _MALLOC(uu_left_pack,    xmax * sizeof(double));
    _MALLOC(uu_right_pack,   xmax * sizeof(double));
    _MALLOC(uu_left_unpack,  xmax * sizeof(double));
    _MALLOC(uu_right_unpack, xmax * sizeof(double));

#pragma omp for private(x, y)
    for (x = 0; x < xmax; x++) {
        for (y = 0; y < ymax; y++) {
            u[x][y] = 0.;
            uu[x][y] = 0.;
        }
    }
#pragma omp for private(x, y)
    for (x = 1; x < xmax - 1; x++) {
        for (y = 1; y < ymax - 1; y++) {
            u[x][y] = sin((double)((x + xoffset) - 1) / size * PI)
                    + cos((double)((y + yoffset) - 1) / size * PI);
        }
    }
    num_threads = omp_get_num_threads();

    time = lap_main();

    value = verify();

    double maxtime;
    MPI_Reduce(&time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("test:%s-%d-%d-%d-%d:mype:%2d:np:%2d:threads:%2d:ver:%.9f:gflops:%f:time:%f\n",
               argv[0], size, bsize, niter, num_threads, rank, np, num_threads,
               value, (double)niter*(size-2)*(size-2)*4/maxtime/1000/1000/1000, maxtime);
    }

#ifdef _USE_HBW
    for (x = 0; x < xmax; x++) {
        hbw_free(u[x]);
        hbw_free(uu[x]);
    }
    hbw_free(u);
    hbw_free(uu);
    hbw_free(uu_left_pack);
    hbw_free(uu_right_pack);
    hbw_free(uu_left_unpack);
    hbw_free(uu_right_unpack);
#else
    for (x = 0; x < xmax; x++) {
        free(u[x]);
        free(uu[x]);
    }
    free(u);
    free(uu);
    free(uu_left_pack);
    free(uu_right_pack);
    free(uu_left_unpack);
    free(uu_right_unpack);
#endif

    MPI_Finalize();

    return 0;
}
