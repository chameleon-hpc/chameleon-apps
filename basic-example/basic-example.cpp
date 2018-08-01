#ifndef USE_MPI
#define USE_MPI 0
#endif
// whether or not to offload tasks
#ifndef USE_OFFLOADING
#define USE_OFFLOADING 0
#endif
// complex scenario that might use multiple OpenMP threads to create multiple target tasks
#ifndef USE_COMPLEX
#define USE_COMPLEX 0
#endif
// size of the double arrays for complex scenario
#ifndef ARR_SIZE
#define ARR_SIZE 10
#endif
// number of tasks for complex scenario
#ifndef NR_TASKS
#define NR_TASKS 2
#endif
#ifndef DEV_NR
#if USE_MPI
#define DEV_NR 1002 // CHAMELEON_MPI
#else
// use standard host offloading in case no MPI is used
#define DEV_NR 1001 // CHAMELEON_HOST
#endif
#endif

#define MAX(a,b) (((a)>(b))?(a):(b))

#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#if USE_MPI
#include <mpi.h>
#include "chameleon.h"
#endif

int main(int argc, char **argv)
{
    int n = ARR_SIZE;
    const double fPi25DT = 3.141592653589793238462643;
    double fTimeStart, fTimeEnd;
    int i;
    int step = 0;

#if USE_COMPLEX
    double a[n];
    int b[n];
#else
    int scalar_A;
    double scalar_B;
    int scalar_C;
#endif

#if USE_MPI
    int iMyRank, iNumProcs;
    /* MPI Initialization */
    // MPI_Init(&argc, &argv);
    int provided;
    // int reqeuested = MPI_THREAD_FUNNELED;
    int reqeuested = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, reqeuested, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);

    chameleon_init();

    #pragma omp target device(1001) // 1001 = CHAMELEON_HOST
    {
       DBP("chameleon_init - dummy region\n");
    }
   
    if (iMyRank == 0)
    {
#endif
        // data initialization
#if USE_COMPLEX
        for(i = 0; i < n; i++)
        {
            a[i] = 5.0;
            b[i] = 3;
        }
        // calculate offsets to decompose array into single tasks
        step = ARR_SIZE / NR_TASKS;
        if (step * NR_TASKS < ARR_SIZE) step++;
        // for(i = 0; i < NR_TASKS; i++) {
        //     int tmp_idx_start = i * step;
        //     printf("Master: array_a_dbl[%d] = %f at (" DPxMOD ")\n", tmp_idx_start, a[tmp_idx_start], DPxPTR(&a[tmp_idx_start]));
        //     printf("Master: array_b_int[%d] = %d at (" DPxMOD ")\n", tmp_idx_start, b[tmp_idx_start], DPxPTR(&b[tmp_idx_start]));
        // }
        for(i = 0; i < ARR_SIZE; i++) {
            printf("Master: array_a_dbl[%02d] = %f at (" DPxMOD ")\n", i, a[i], DPxPTR(&a[i]));
        }
        for(i = 0; i < ARR_SIZE; i++) {
            printf("Master: array_b_int[%02d] = %d at (" DPxMOD ")\n", i, b[i], DPxPTR(&b[i]));
        }
#else
        scalar_A = 1;
        scalar_B = 2;
        scalar_C = 3;        
        printf("Master: scalar_A_int = %d at (" DPxMOD ")\n", scalar_A, DPxPTR(&scalar_A));
        printf("Master: scalar_B_dbl = %f at (" DPxMOD ")\n", scalar_B, DPxPTR(&scalar_B));
        printf("Master: scalar_c_int = %d at (" DPxMOD ")\n", scalar_C, DPxPTR(&scalar_C));
#endif //USE_COMPLEX
        // measure start time here
        fTimeStart = omp_get_wtime();
#if USE_COMPLEX
        // run multiple threads where each is creating target tasks
        #pragma omp parallel for
        for(i = 0; i < NR_TASKS; i++) {
            int idx_start = i * step;
            int cur_len = step;
            int idx_end = i*(step+1)-1;
            if(idx_end > ARR_SIZE-1) {
                cur_len = ARR_SIZE-1-idx_start;
            }
#if USE_OFFLOADING
            #pragma omp target map(tofrom:a[idx_start:cur_len], b[idx_start:cur_len]) device(DEV_NR)
            {
#endif // USE_OFFLOADING
                printf("Device: array_a_dbl[%d] = %f at (" DPxMOD ")\n", idx_start, a[idx_start], DPxPTR(&a[idx_start]));
                printf("Device: array_b_int[%d] = %d at (" DPxMOD ")\n", idx_start, b[idx_start], DPxPTR(&b[idx_start]));
                printf("Device: setting array_a_dbl = 13.37\n");
                printf("Device: setting array_b_int = 42\n");
                for(int j = 0; j < cur_len; j++) {
                    a[idx_start+j] = 13.37;
                    b[idx_start+j] = 42;
                }
#if USE_OFFLOADING
            }
#endif // USE_OFFLOADING
        }

#else // USE_COMPLEX

#if USE_OFFLOADING
        #pragma omp target map(tofrom:scalar_A) device(DEV_NR)
        {
#endif // USE_OFFLOADING
            printf("Device: Implict mapped scalar_B_dbl = %f at (" DPxMOD ")\n", scalar_B, DPxPTR(&scalar_B));
            printf("Device: Implict mapped scalar_C_int = %d at (" DPxMOD ")\n", scalar_C, DPxPTR(&scalar_C));
            printf("Device: scalar_A_int = %d at (" DPxMOD ")\n", scalar_A, DPxPTR(&scalar_A));
            printf("Device: Setting scalar_A_int = 42\n");
            scalar_A = 42; 
#if USE_OFFLOADING
        }
#endif // USE_OFFLOADING
#endif // USE_COMPLEX
#if USE_MPI
    }
    
    // work on tasks as long as there are tasks
    int res = chameleon_distributed_taskwait();
    MPI_Barrier(MPI_COMM_WORLD);

    if (iMyRank == 0)
    {
#endif
        fTimeEnd = omp_get_wtime() - fTimeStart;
        printf("Elapsed computation time: %.3f\n", fTimeEnd);
        printf("Results:\n");
#if USE_COMPLEX
        // for(i = 0; i < NR_TASKS; i++) {
        //     int tmp_idx_start = i*step;
        //     printf("Master: array_a_dbl[%d] = %f at (" DPxMOD ")\n", tmp_idx_start, a[tmp_idx_start], DPxPTR(&a[tmp_idx_start]));
        //     printf("Master: array_b_int[%d] = %d at (" DPxMOD ")\n", tmp_idx_start, b[tmp_idx_start], DPxPTR(&b[tmp_idx_start]));
        // }
        for(i = 0; i < ARR_SIZE; i++) {
            printf("Master: array_a_dbl[%02d] = %f at (" DPxMOD ")\n", i, a[i], DPxPTR(&a[i]));
        }
        for(i = 0; i < ARR_SIZE; i++) {
            printf("Master: array_b_int[%02d] = %d at (" DPxMOD ")\n", i, b[i], DPxPTR(&b[i]));
        }
#else
        printf("Master: scalar_A_int = %d at (" DPxMOD ")\n", scalar_A, DPxPTR(&scalar_A));
        printf("Master: scalar_B_dbl = %f at (" DPxMOD ")\n", scalar_B, DPxPTR(&scalar_B));
#endif
#if USE_MPI
    }    
    chameleon_finalize();
    MPI_Finalize();
#endif
    return 0;
}
