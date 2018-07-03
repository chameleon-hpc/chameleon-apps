#ifndef USE_MPI
#define USE_MPI 0
#endif
#ifndef USE_OFFLOADING
#define USE_OFFLOADING 0
#endif
#ifndef USE_COMPLEX
#define USE_COMPLEX 0
#endif
#ifndef SECOND_TARGET_REGION
#define SECOND_TARGET_REGION 0
#endif

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

#define N 15

#ifndef DEV_NR
//#define DEV_NR 1001 // CHAMELEON_HOST
#define DEV_NR 1002 // CHAMELEON_MPI
#endif

int main(int argc, char **argv)
{
    int n = N;
    const double fPi25DT = 3.141592653589793238462643;
    double fTimeStart, fTimeEnd;
    int i;
    int scalar_A;
    double scalar_B;
    int scalar_C;
    double a[n], b[n], c[n];
    //double *a = (double*) malloc(sizeof(double)*n);
    //double *b = (double*) malloc(sizeof(double)*n);
    //double *c = (double*) malloc(sizeof(double)*n);

#if USE_MPI
    int iMyRank, iNumProcs;
    /* MPI Initialization */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);

    chameleon_init();
   
    if (iMyRank == 0)
    {
#endif
        // data initialization
        scalar_A = 1;
        scalar_B = 2;
        scalar_C = 3;
        for(i = 0; i < n; i++)
        {
            a[i] = 1.0 / sin(i);
            b[i] = 1.0 / cos(i);
        }
        
        printf("Master: scalar_A_int = %d at (" DPxMOD ")\n", scalar_A, DPxPTR(&scalar_A));
        printf("Master: scalar_B_dbl = %f at (" DPxMOD ")\n", scalar_B, DPxPTR(&scalar_B));
        printf("Master: scalar_c_int = %d at (" DPxMOD ")\n", scalar_C, DPxPTR(&scalar_C));

        fTimeStart = omp_get_wtime();
#if USE_OFFLOADING
        // first test: calculate complete block in target region
#if USE_COMPLEX
        #pragma omp target map(tofrom:b[0:N], a[0:N], c[0:N]) device(DEV_NR)
#else
        #pragma omp target map(tofrom:scalar_A) device(DEV_NR)
#endif
        {
#endif
#if USE_COMPLEX
            int ii;
            for(ii = 0; ii < n; ii++)
            {
                c[ii] = ((4.0*fPi25DT / (1.0 + a[ii]*b[ii])) + b[ii]);
            }
#else
            printf("Device: Implict mapped scalar_B_dbl = %f at (" DPxMOD ")\n", scalar_B, DPxPTR(&scalar_B));
            printf("Device: Implict mapped scalar_C_int = %d at (" DPxMOD ")\n", scalar_C, DPxPTR(&scalar_C));
            printf("Device: scalar_A_int = %d at (" DPxMOD ")\n", scalar_A, DPxPTR(&scalar_A));
            printf("Device: Setting scalar_A_int = 42\n");
            scalar_A = 42; 
#endif
#if USE_OFFLOADING
        }
#endif
        //usleep(2000);
        fTimeEnd = omp_get_wtime() - fTimeStart;
        printf("Master: scalar_A_int = %d at (" DPxMOD ")\n", scalar_A, DPxPTR(&scalar_A));
        printf("Master: scalar_B_dbl = %f at (" DPxMOD ")\n", scalar_B, DPxPTR(&scalar_B));

#if USE_OFFLOADING && SECOND_TARGET_REGION
        #pragma omp target map(tofrom:scalar_A) device(DEV_NR)
        {
            printf("Device: scalar_A_int = %d at (" DPxMOD ")\n", scalar_A, DPxPTR(&scalar_A));
            printf("Device: scalar_B_dbl = %f at (" DPxMOD ")\n", scalar_B, DPxPTR(&scalar_B));
        }
#endif
#if USE_MPI
    }
    else 
    {
#if USE_OFFLOADING
        // TODO:
        // 1. receive MPI requests + hand shake
        // 2. work on item
        // 3. send back results
#else
        // don't do anything: reference version single threaded
#endif
    }
    
    // work on tasks as long as there are tasks
    int res = chameleon_distributed_taskwait();
    MPI_Barrier(MPI_COMM_WORLD);

    if (iMyRank == 0)
    {
#endif
        printf("Elapsed computation time: %.3f\n", fTimeEnd);
#if USE_COMPLEX
        printf("Results:\n");
        for(i = 0; i < 5; i++)
        {
            printf("c[%d] = %f\n", i, c[i]);
        }
        printf("...\n");
#endif
#if USE_MPI
    }
    
    //chameleon_finalize();
    MPI_Finalize();
#endif
    return 0;
}
