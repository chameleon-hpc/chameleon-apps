#ifndef USE_MPI
#define USE_MPI 0
#endif
#ifndef USE_OFFLOADING
#define USE_OFFLOADING 0
#endif
#ifndef COMPLEX
#define COMPLEX 0
#endif
#ifndef SECOND_TARGET_REGION
#define SECOND_TARGET_REGION 1
#endif


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>
#if USE_MPI
#include <mpi.h>
#include "chameleon.h"
#endif

#define N 15

int main(int argc, char **argv)
{
    int n = N;
    const double fPi25DT = 3.141592653589793238462643;
    double fTimeStart, fTimeEnd;
	int i;
  int scalar, scalar2;
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
    scalar = 1;
		scalar2 = 2;
		for(i = 0; i < n; i++)
		{
			a[i] = 1.0 / sin(i);
			b[i] = 1.0 / cos(i);
		}

    int dev = 1001; // CHAMELEON_HOST
    //int dev = 1002; // CHAMELEON_MPI
		
    printf("Host: n = %d at (%p)\n", n, &n);
    printf("Host: scalar = %d at (%p)\n", scalar, &scalar);
		fTimeStart = omp_get_wtime();
#if USE_OFFLOADING
		// first test: calculate complete block in target region
#if COMPLEX
    #pragma omp target map(tofrom:b[0:N], a[0:N], c[0:N]) device(dev)
#else
    #pragma omp target map(tofrom:scalar) device(dev)
#endif
		{
#endif
#if COMPLEX
      int ii;
		  for(ii = 0; ii < n; ii++)
		  {
			  c[ii] = ((4.0*fPi25DT / (1.0 + a[ii]*b[ii])) + b[ii]);
		  }
#else
      printf("Implict mapped n = %d at (%p)\n", n, &n);
      printf("Device: scalar = %d at (%p)\n", scalar, &scalar);
      printf("Device: Setting scalar = 3\n");
      scalar = 3;      
#endif
#if USE_OFFLOADING
		}
#endif
    	//usleep(2000);
		fTimeEnd = omp_get_wtime() - fTimeStart;
    printf("Host: scalar = %d at (%p)\n", scalar, &scalar);
    //printf("Measured: %f\n", fTimeEnd);

#if USE_OFFLOADING && SECOND_TARGET_REGION
	#pragma omp target map(tofrom:scalar2) device(dev)
	{
		printf("Device: scalar2 = %d at (%p)\n", scalar2, &scalar2);
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
	MPI_Barrier(MPI_COMM_WORLD);

	if (iMyRank == 0)
	{
#endif
		printf("Elapsed computation time: %.3f\n", fTimeEnd);
#if COMPLEX
			printf("Results:\n");
			for(i = 0; i < 5; i++)
			{
				printf("c[%d] = %f\n", i, c[i]);
			}
			printf("...\n");
#endif
#if USE_MPI
	}
	MPI_Finalize();
#endif
	return 0;
}
