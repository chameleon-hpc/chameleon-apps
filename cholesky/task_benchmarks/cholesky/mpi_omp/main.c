
#define MAIN

#include "common.h"
#include "cholesky.h"

static void get_block_rank(int *block_rank, int nt);

int main(int argc, char *argv[])
{
    /* MPI Initialize */
    int provided, check_id = 0;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

#pragma omp parallel
#pragma omp single
    num_threads = omp_get_num_threads();

    /* cholesky init */
    char *result[3] = {"n/a","successful","UNSUCCESSFUL"};
    const double eps = BLAS_dfpinfo(blas_eps);

    if (argc < 4) {
        printf("cholesky matrix_size block_size check\n");
        exit(-1);
    }
    const int  n = atoi(argv[1]); /* matrix size */
    const int ts = atoi(argv[2]); /* tile size */
    int check    = atoi(argv[3]); /* check result? */

    double * const matrix = (double *) malloc(n * n * sizeof(double));
    assert(matrix != NULL);
    initialize_matrix(n, ts, matrix);

    const int nt = n / ts;

    if (mype == 0) 
        printf("nt = %d, ts = %d\n", nt, ts);

    /* Set block rank */
    int *block_rank = malloc(nt * nt * sizeof(int));
    get_block_rank(block_rank, nt);

#ifdef DEBUG
    if (mype == 0) {
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                printf("%d ", block_rank[i * nt + j]);
            }
            printf("\n");
        }
    }
#endif

    double *A[nt][nt], *B, *C[nt], *Ans[nt][nt];

    if (check != 0) {

        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                MPI_Alloc_mem(ts * ts * sizeof(double), MPI_INFO_NULL, &Ans[i][j]);
            }
        }

        convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ans);
        free(matrix);

        _MALLOC(B, ts * ts * sizeof(double));
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                if (block_rank[i*nt+j] == mype) {
                    _MALLOC(A[i][j], ts * ts * sizeof(double));
                    for (int k = 0; k < ts * ts; k++) {
                        A[i][j][k] = Ans[i][j][k];
                    }
                }
            }
            _MALLOC(C[i], ts * ts * sizeof(double));
        }

    } else {

        _MALLOC(B, ts * ts * sizeof(double));
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                if (block_rank[i*nt+j] == mype) {
                    _MALLOC(A[i][j], ts * ts * sizeof(double));
                }
            }
            _MALLOC(C[i], ts * ts * sizeof(double));
        }
        convert_to_blocks_with_rank(ts, nt, n, (double(*)[n]) matrix, A, block_rank, mype);
        free(matrix);
    }

    /* Warm up */
    double time_ser = 0.0, time_mpi = 0.0;
    if (check) time_ser = do_cholesky_ser(ts, nt, (double* (*)[nt]) Ans);

    MPI_Barrier(MPI_COMM_WORLD);

    time_mpi = do_cholesky_mpi(ts, nt, (double* (*)[nt])A, B, C, block_rank);

    /* Verification */
    if (check) {
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                if (block_rank[i * nt + j] == mype) {
                    for (int k = 0; k < ts*ts; k++) {
                        if (Ans[i][j][k] != A[i][j][k]) {
                            check_id = 2;
                            break;
                        }
                    }
                    if(check_id != 0)
                        break;
                }
            }
            if(check_id != 0)
                break;
        }
        if (check_id == 0) check_id = 1;
    }

    float gflops_ser = (((1.0 / 3.0) * n * n * n) / ((time_ser) * 1.0e+9));
    float gflops_mpi = (((1.0 / 3.0) * n * n * n) / ((time_mpi) * 1.0e+9));

    if (mype == 0 || check_id == 2)
        printf("test:%s-%d-%d-%d:mype:%2d:np:%2d:threads:%2d:result:%s:gflops:%f:time:%f:gflops_ser:%f:time_ser:%f\n", 
               argv[0], n, ts, num_threads, mype, np, num_threads, result[check_id], gflops_mpi, time_mpi, gflops_ser, time_ser);

    for (int i = 0; i < nt; i++) {
        for (int j = 0; j < nt; j++) {
            if (block_rank[i*nt+j] == mype) {
                _FREE(A[i][j]);
            }
            if (check != 0) free(Ans[i][j]);
        }
        _FREE(C[i]);
    }
    _FREE(B);

    free(block_rank);

    /* MPI Finalize */
    MPI_Finalize();

    return 0;
}

/* 2D Block-cyclic distribution */
static void get_block_rank(int *block_rank, int nt)
{
    int row, col;
    row = col = np;

    if (np != 1) {
        while (1) {
            row = row / 2;
            if (row * col == np) break;
            col = col / 2;
            if (row * col == np) break;
        }
    }
    if (mype == 0) printf("row = %d, col = %d\n", row, col);

    int i, j, tmp_rank = 0, offset = 0;
    for (i = 0; i < nt; i++) {
        for (j = 0; j < nt; j++) {
            block_rank[i*nt + j] = tmp_rank + offset;
            tmp_rank++;
            if (tmp_rank >= col) tmp_rank = 0;
        }
        tmp_rank = 0;
        offset = (offset + col >= np) ? 0 : offset + col;
    }
}
