#ifdef USE_MPI
#include <mpi.h>
#endif
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    int n = 150000;
    const double fPi25DT = 3.141592653589793238462643;
    double fTimeStart, fTimeEnd;
	int i;
	double a[n], b[n], c[n];

#ifdef USE_MPI
	int iMyRank, iNumProcs;
    /* MPI Initialization */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
   
	if (iMyRank == 0)
  	{
#endif
		// data initialization
		for(i = 0; i < n; i++)
		{
			a[i] = 1.0 / sin(i);
			b[i] = 1.0 / cos(i);
		}
		
		fTimeStart = omp_get_wtime();

#ifdef USE_OFFLOADING
		// first test: calculate complete block in target region
		#pragma omp target map(to: a, b) map(from: c) //device(chameleon)
		{
#else
		for(i = 0; i < n; i++)
		{
			c[i] = (4.0*fPi25DT / (1.0 + a[i]*b[i]));
		}
#endif
#ifdef USE_OFFLOADING
		}
#endif
    	//usleep(2000);
		fTimeEnd = omp_get_wtime() - fTimeStart;
    	//printf("Measured: %f\n", fTimeEnd);
#ifdef USE_MPI
	}
	else 
	{
#ifdef USE_OFFLOADING
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
			printf("Results:\n");
			for(i = 0; i < 5; i++)
			{
				printf("c[%d] = %f\n", i, c[i]);
			}
			printf("...\n");
#ifdef USE_MPI
	}
	MPI_Finalize();
#endif
	return 0;
}
