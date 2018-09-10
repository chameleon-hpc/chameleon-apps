// size of the double arrays for complex scenario
#ifndef ARR_SIZE
#define ARR_SIZE 400000
//#define ARR_SIZE 200
#endif
// number of tasks for complex scenario
#ifndef NR_TASKS
#define NR_TASKS 200
// #define NR_TASKS 10
#endif
// whether or not to print intermediate results for debugging
#ifndef PRINT_DATA_VERBOSE
#define PRINT_DATA_VERBOSE 0
#endif

#include <mpi.h>
#include "chameleon.h"
#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef DEV_NR
//#define DEV_NR 1001 // CHAMELEON_HOST
#define DEV_NR CHAMELEON_MPI // CHAMELEON_MPI
#endif

int main(int argc, char **argv)
{
    int n = ARR_SIZE;
    double fTimeStart, fTimeEnd;
    int i;
    int step = 0;

    // double   array_a_dbl[n];
    // int      array_b_int[n];
    double  *array_a_dbl = (double *)   malloc(n* sizeof(double));
    int     *array_b_int = (int *)      malloc(n* sizeof(int));

    int iMyRank, iNumProcs;
    int provided;
    int reqeuested = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, reqeuested, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);

    chameleon_init();
   
    if (iMyRank == 0)
    {
        printf("Host: Values of array_a_dbl initially set to 5.0\n");
        printf("Host: Values of array_b_int initially set to 3\n");
        printf("In Target Region: Values of array_a_dbl will be set to 13.37\n");
        printf("In Target Region: Values of array_b_int will be set to 42\n\n");

        // data initialization
        #pragma omp parallel for
        for(i = 0; i < n; i++)
        {
            array_a_dbl[i] = 5.0;
            array_b_int[i] = 3;
        }

        // calculate offsets to decompose array into single tasks
        step = ARR_SIZE / NR_TASKS;
        if (step * NR_TASKS < ARR_SIZE) step++;
        
        #if PRINT_DATA_VERBOSE
        for(i = 0; i < NR_TASKS; i++) {
            int tmp_idx_start = i * step;
            printf("Master: array_a_dbl[%3d] = %f at (" DPxMOD ")\n", tmp_idx_start, array_a_dbl[tmp_idx_start], DPxPTR(&array_a_dbl[tmp_idx_start]));
            printf("Master: array_b_int[%3d] = %d at (" DPxMOD ")\n", tmp_idx_start, array_b_int[tmp_idx_start], DPxPTR(&array_b_int[tmp_idx_start]));
        }
        #endif

        // measure start time here
        fTimeStart = omp_get_wtime();

        // run multiple threads where each is creating target tasks
        #pragma omp parallel
        {
            #pragma omp for
            for(i = 0; i < NR_TASKS; i++) {
                // determine start and end index as well as length
                int idx_start = i * step;
                int cur_len = step;
                int idx_end = i*(step+1)-1;
                if(idx_end > ARR_SIZE-1) {
                    cur_len = ARR_SIZE-1-idx_start;
                }

                #pragma omp target map(tofrom:array_a_dbl[idx_start:cur_len], array_b_int[idx_start:cur_len]) device(DEV_NR)
                {
                    #if PRINT_DATA_VERBOSE
                    printf("OS_TID:%ld Running task with start idx %d and end idx %d\n", syscall(SYS_gettid), idx_start, idx_start+cur_len-1);
                    for(int j = 0; j < cur_len; j++) {
                        printf("Device: array_a_dbl[%3d] = %f at (" DPxMOD ")\n", idx_start+j, array_a_dbl[idx_start+j], DPxPTR(&array_a_dbl[idx_start+j]));
                    }
                    for(int j = 0; j < cur_len; j++) {
                        printf("Device: array_b_int[%3d] = %d at (" DPxMOD ")\n", idx_start+j, array_b_int[idx_start+j], DPxPTR(&array_b_int[idx_start+j]));
                    }
                    #endif

                    // generate enough work here
                    for(int o = 0; o < 20000; o++) {
                        for(int j = 0; j < cur_len; j++) {
                            array_a_dbl[idx_start+j] = (13.37 * 20.0 / 20.0) + 1.0 - 1.0;
                            array_b_int[idx_start+j] = 42 * 2 / 2;
                        }
                    }
                }
            }

            // work on tasks as long as there are tasks
            int res = chameleon_distributed_taskwait();
        }
    } else {
        // second rank is only working on data that is speculatively send from master rank
        #pragma omp parallel
        {
            // work on tasks as long as there are tasks
            int res = chameleon_distributed_taskwait();
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    if (iMyRank == 0)
    {
        fTimeEnd = omp_get_wtime() - fTimeStart;
        printf("\nElapsed computation time: %.3f\n", fTimeEnd);

        // Verification
        int succ = 1;
        for(i = 0; i < ARR_SIZE; i++) {
            if(array_a_dbl[i] != 13.37 || array_b_int[i] != 42) {
                succ = 0;
                break;
            }                
        }
        printf("Verification: %s\n", succ ? "SUCCESS" : "FAILED");
        
        #if PRINT_DATA_VERBOSE
        printf("Results:\n");
        for(i = 0; i < ARR_SIZE; i++) {
            printf("Master: array_a_dbl[%3d] = %f at (" DPxMOD ")\n", i, array_a_dbl[i], DPxPTR(&array_a_dbl[i]));
        }
        for(i = 0; i < ARR_SIZE; i++) {
            printf("Master: array_b_int[%3d] = %d at (" DPxMOD ")\n", i, array_b_int[i], DPxPTR(&array_b_int[i]));
        }
        #endif
    }

    // clean up
    free(array_a_dbl);
    free(array_b_int);

    chameleon_finalize();
    MPI_Finalize();
    return 0;
}
