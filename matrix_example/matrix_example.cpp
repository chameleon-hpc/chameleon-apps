// whether or not to offload tasks
#ifndef USE_OFFLOADING
#define USE_OFFLOADING 0
#endif

// size of the double arrays for complex scenario
#ifndef MATRIX_SIZE
#define MATRIX_SIZE 400
#endif

// number of tasks for complex scenario
#ifndef NR_TASKS
#define NR_TASKS 10
#endif

#ifndef RANDOMINIT
#define RANDOMINIT 0
#endif

#ifndef RANDOMDIST
#define RANDOMDIST 1
#endif

#ifndef DEV_NR
#define DEV_NR 1002 // CHAMELEON_MPI
#endif

//#define LOG(rank, str) fprintf(stderr, "#R%d: %s\n", rank, str)
#define LOG(rank, str) printf("#R%d: %s\n", rank, str)

#include <mpi.h>
#include "chameleon.h"
#include <random>
#include <iostream>
#include <string>

void initialize_matrix_rnd(double *mat) {
	double lower_bound = 0;
	double upper_bound = 10000;
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::default_random_engine re;

	for(int i=0; i<MATRIX_SIZE*MATRIX_SIZE; i++) {
		mat[i]= unif(re);
	}
}

void initialize_matrix_zero(double *mat) {
	for(int i=0; i<MATRIX_SIZE*MATRIX_SIZE; i++) {
		mat[i]= 0;
	}
}

void initialize_matrix_test_A(double *mat) {
	for(int i=0; i<MATRIX_SIZE*MATRIX_SIZE; i++) {
			mat[i]= 1;
    }
}

#pragma omp declare target
void compute_matrix_matrix(double *a, double *b, double *c) {
	for(int i=0;i<MATRIX_SIZE;i++) {
		for(int j=0;j<MATRIX_SIZE;j++) {
			c[i*MATRIX_SIZE+j]=0;
			for(int k=0;k<MATRIX_SIZE;k++) {
				c[i*MATRIX_SIZE+j] += a[i*MATRIX_SIZE+k] * b[k*MATRIX_SIZE+j];
			}
		}
	}
}
#pragma omp end declare target

#pragma omp declare target
bool check_test_matrix(double *c, double val) {
	int iMyRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
	for(int i=0;i<MATRIX_SIZE;i++) {
		for(int j=0;j<MATRIX_SIZE;j++) {
			if(c[i*MATRIX_SIZE+j]!=val) {
				printf("#R%d: Error in matrix entry (%d,%d) expected:%f but value is %f\n", iMyRank,i,j,val,c[i*MATRIX_SIZE+j]);
                //fprintf(stderr, "#R%d: Error in matrix entry (%d,%d) expected:%f but value is %f\n", iMyRank,i,j,val,c[i*MATRIX_SIZE+j]);
				return false;
			}
		}
	}
	return true;
}
#pragma omp end declare target

void compute_random_task_distribution(int *dist, int nRanks) {
	double *weights = new double[nRanks];
	
	double lower_bound = 0;
	double upper_bound = 1;
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::default_random_engine re;
	double sum = 0;

	for(int i=0; i<nRanks; i++) {
		weights[i]= unif(re);
		sum += weights[i];
	}

	for(int i=0; i<nRanks; i++) {
		weights[i]= weights[i]/sum;
		dist[i] = weights[i]*NR_TASKS;
	}

	delete[] weights;
}

int main(int argc, char **argv)
{
	int iMyRank, iNumProcs;
	int provided;
	int requested = MPI_THREAD_MULTIPLE;
	MPI_Init_thread(&argc, &argv, requested, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
       int numberOfTasks;

	chameleon_init();

	double fTimeStart, fTimeEnd;

	if(argc==iNumProcs+1){
		if(iMyRank==0) {
			LOG(iMyRank, "using user-defined initial load distribution...");
		}
              numberOfTasks = atoi( argv[iMyRank+1] );
	}
       else {
#if RANDOMDIST
        int *dist = new int[iNumProcs];
		if( iMyRank==0 ) {
 			compute_random_task_distribution(dist, iNumProcs);
		}
        MPI_Bcast(dist, iNumProcs, MPI_INTEGER, 0, MPI_COMM_WORLD);
        numberOfTasks = dist[iMyRank];
        delete[] dist;
#else
		numberOfTasks = NR_TASKS;
#endif
        std::string msg = "will create "+std::to_string(numberOfTasks)+" tasks";
        LOG(iMyRank, msg.c_str());
	}

	double **matrices_a, **matrices_b, **matrices_c;

	matrices_a = new double*[numberOfTasks];
	matrices_b = new double*[numberOfTasks];
	matrices_c = new double*[numberOfTasks];

	//allocate and initialize matrices
	for(int i=0; i<numberOfTasks; i++) {
 		matrices_a[i] = new double[MATRIX_SIZE*MATRIX_SIZE];
    		matrices_b[i] = new double[MATRIX_SIZE*MATRIX_SIZE];
    		matrices_c[i] = new double[MATRIX_SIZE*MATRIX_SIZE];
    	if(RANDOMINIT) {
    		initialize_matrix_rnd(matrices_a[i]);
    		initialize_matrix_rnd(matrices_b[i]);
    		initialize_matrix_zero(matrices_c[i]);
    	}
    	else {
    		initialize_matrix_test_A(matrices_a[i]);
    		initialize_matrix_test_A(matrices_b[i]);
    		initialize_matrix_zero(matrices_c[i]);
    	}
    }

    MPI_Barrier(MPI_COMM_WORLD);
    fTimeStart=MPI_Wtime();

    #pragma omp parallel
    {
    	// if(iMyRank==0) {
		#pragma omp for
    		for(int i=0; i<numberOfTasks; i++) {
				double *A = matrices_a[i];
		        double *B = matrices_b[i];
		        double *C = matrices_c[i];	
				#pragma omp target map(tofrom:A[0:MATRIX_SIZE*MATRIX_SIZE],B[0:MATRIX_SIZE*MATRIX_SIZE], C[0:MATRIX_SIZE*MATRIX_SIZE]) device(DEV_NR)
				{
					//check_test_matrix(A, 1);
					//check_test_matrix(B, 1);
					//check_test_matrix(C, 0);
					compute_matrix_matrix(A, B, C);
					//check_test_matrix(C, MATRIX_SIZE);
				}
				//LOG(iMyRank, "offloading to chameleon");
    		}

    	//LOG(iMyRank, "entering taskwait");
    	int res = chameleon_distributed_taskwait();
    	//LOG(iMyRank, "leaving taskwait");
    }
    fTimeEnd=MPI_Wtime();
    if( iMyRank==0 ) {
        printf("#R%d: Computations with chameleon offloading took %.2f\n", iMyRank, fTimeEnd-fTimeStart);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    fTimeStart=MPI_Wtime();

    #pragma omp parallel
    {
    	// if(iMyRank==0) {
		#pragma omp for
    		for(int i=0; i<numberOfTasks; i++) {
				double *A = matrices_a[i];
		        double *B = matrices_b[i];
		        double *C = matrices_c[i];	
				#pragma omp target map(tofrom:A[0:MATRIX_SIZE*MATRIX_SIZE],B[0:MATRIX_SIZE*MATRIX_SIZE], C[0:MATRIX_SIZE*MATRIX_SIZE]) device(1001)
				{
					//check_test_matrix(A, 1);
					//check_test_matrix(B, 1);
					//check_test_matrix(C, 0);
					compute_matrix_matrix(A, B, C);
					//check_test_matrix(C, MATRIX_SIZE);
				}
				//LOG(iMyRank, "offloading to chameleon");
    		}

    	//LOG(iMyRank, "entering taskwait");
    	int res = chameleon_distributed_taskwait();
    	//LOG(iMyRank, "leaving taskwait");
    }
    fTimeEnd=MPI_Wtime();
    if( iMyRank==0 ) {
        printf("#R%d: Computations with host offloading took %.2f\n", iMyRank, fTimeEnd-fTimeStart);
    }  
//    LOG(iMyRank, "Validation:");
//    bool pass = check_test_matrix(matrices_c[numberOfTasks-1], MATRIX_SIZE);
//    if(pass)
//        LOG(iMyRank, "TEST SUCESS");
//    else
//        LOG(iMyRank, "TEST FAILED");

    //deallocate matrices
    for(int i=0; i<numberOfTasks; i++) {
    	delete[] matrices_a[i];
    	delete[] matrices_b[i];
    	delete[] matrices_c[i];
    }

    delete[] matrices_a;
    delete[] matrices_b;
    delete[] matrices_c;

    MPI_Barrier(MPI_COMM_WORLD);
    chameleon_finalize();
    MPI_Finalize();
//#endif
    return 0;
}
