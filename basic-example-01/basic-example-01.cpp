#ifndef USE_MPI
#define USE_MPI 1
#endif

#if USE_MPI
#include <mpi.h>
#include "chameleon.h"
#endif

#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef DEV_NR
#if USE_MPI
#define DEV_NR CHAMELEON_MPI // CHAMELEON_MPI
#else
// use standard host offloading in case no MPI is used
#define DEV_NR CHAMELEON_HOST // CHAMELEON_HOST
#endif
#endif

#ifndef DPxMOD
#define DPxMOD "0x%0*" PRIxPTR
#endif

#ifndef DPxPTR
#define DPxPTR(ptr) ((int)(2*sizeof(uintptr_t))), ((uintptr_t) (ptr))
#endif

int main(int argc, char **argv)
{
    double fTimeStart, fTimeEnd;

    int scalar_A;
    double scalar_B;
    int scalar_C;

#if USE_MPI
    int iMyRank, iNumProcs;
    
    int provided;
    int reqeuested = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, reqeuested, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
    // MPI_Init(&argc, &argv);

    chameleon_init();

    #pragma omp target device(1001) // 1001 = CHAMELEON_HOST
    {
       DBP("chameleon_init - dummy region\n");
    }
   
    if (iMyRank == 0)
    {
#endif
        scalar_A = 1;
        scalar_B = 2;
        scalar_C = 3;
        printf("Master: scalar_A_int = %d at (" DPxMOD ")\n", scalar_A, DPxPTR(&scalar_A));
        printf("Master: scalar_B_dbl = %f at (" DPxMOD ")\n", scalar_B, DPxPTR(&scalar_B));
        printf("Master: scalar_C_int = %d at (" DPxMOD ")\n", scalar_C, DPxPTR(&scalar_C));
        
        // measure start time here
        fTimeStart = omp_get_wtime();

        #pragma omp target map(tofrom:scalar_A) device(DEV_NR)
        {
            printf("Device: Mapped scalar_A_int = %d at (" DPxMOD ")\n", scalar_A, DPxPTR(&scalar_A));
            printf("Device: Implict mapped scalar_B_dbl = %f at (" DPxMOD ")\n", scalar_B, DPxPTR(&scalar_B));
            printf("Device: Implict mapped scalar_C_int = %d at (" DPxMOD ")\n", scalar_C, DPxPTR(&scalar_C));
            printf("... Device: Setting scalar_A_int = 42\n");
            scalar_A = 42; 
        }
#if USE_MPI
        // work on tasks as long as there are tasks
        int res = chameleon_distributed_taskwait();
    } else {
        // work on tasks as long as there are tasks
        int res = chameleon_distributed_taskwait();
    }
    
    // wait until all is finished
    MPI_Barrier(MPI_COMM_WORLD);

    if (iMyRank == 0)
    {
#endif
        fTimeEnd = omp_get_wtime() - fTimeStart;
        printf("Elapsed computation time: %.3f\n\n", fTimeEnd);
        printf("Results:\n");
        printf("Master: scalar_A_int = %d at (" DPxMOD ")\n", scalar_A, DPxPTR(&scalar_A));
        printf("Master: scalar_B_dbl = %f at (" DPxMOD ")\n", scalar_B, DPxPTR(&scalar_B));
        printf("Master: scalar_C_int = %d at (" DPxMOD ")\n", scalar_C, DPxPTR(&scalar_C));
#if USE_MPI
    }
    chameleon_finalize();
    MPI_Finalize();
#endif
    return 0;
}
