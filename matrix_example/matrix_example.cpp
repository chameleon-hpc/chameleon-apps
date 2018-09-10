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
#define NR_TASKS 2
#endif

#ifndef RANDOMINIT
#define RANDOMINIT 0
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

int main(int argc, char **argv)
{
	int iMyRank, iNumProcs;
	int provided;
	int requested = MPI_THREAD_MULTIPLE;
	MPI_Init_thread(&argc, &argv, requested, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);

	chameleon_init();

    double fTimeStart, fTimeEnd;

    double *matrices_a[NR_TASKS], *matrices_b[NR_TASKS], *matrices_c[NR_TASKS];

    //allocate and initialize matrices
    for(int i=0; i<NR_TASKS; i++) {
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

    double A[MATRIX_SIZE*MATRIX_SIZE],B[MATRIX_SIZE*MATRIX_SIZE],C[MATRIX_SIZE*MATRIX_SIZE];
    if(RANDOMINIT) {
        		initialize_matrix_rnd(A);
        		initialize_matrix_rnd(B);
        		initialize_matrix_zero(C);
	}
	else {
		initialize_matrix_test_A(A);
		initialize_matrix_test_A(B);
		initialize_matrix_zero(C);
	}

	#pragma omp parallel
    {
    	// if(iMyRank==0) {
		#pragma omp single
    	{
//			#pragma omp target map(tofrom:matrices_a[0][0:MATRIX_SIZE*MATRIX_SIZE],matrices_b[0][0:MATRIX_SIZE*MATRIX_SIZE], matrices_c[0][0:MATRIX_SIZE*MATRIX_SIZE]) device(DEV_NR)
//    		{
//    			check_test_matrix(matrices_a[0], 1);
//    			check_test_matrix(matrices_b[0], 1);
//    			check_test_matrix(matrices_c[0], 0);
//    			compute_matrix_matrix(matrices_a[0], matrices_b[0], matrices_c[0]);
//    			check_test_matrix(matrices_c[0], 0);
//    		}
    		LOG(iMyRank, "offloading to chameleon");
            #pragma omp target map(tofrom:A[0:MATRIX_SIZE*MATRIX_SIZE],B[0:MATRIX_SIZE*MATRIX_SIZE], C[0:MATRIX_SIZE*MATRIX_SIZE]) device(DEV_NR)
            {
                    // LOG(iMyRank, "executing");
                    check_test_matrix(A, 1);
                    check_test_matrix(B, 1);
                    check_test_matrix(C, 0);
                    compute_matrix_matrix(A, B, C);
                    check_test_matrix(C, MATRIX_SIZE);
            }
    	}
    	// }
    	LOG(iMyRank, "entering taskwait");
    	int res = chameleon_distributed_taskwait();
    	LOG(iMyRank, "leaving taskwait");
    }

    LOG(iMyRank, "Validation:");
    bool pass = check_test_matrix(C, MATRIX_SIZE);
    if(pass)
        LOG(iMyRank, "TEST SUCESS");
    else
        LOG(iMyRank, "TEST FAILED");

    //deallocate matrices
    for(int i=0; i<NR_TASKS; i++) {
    	delete[] matrices_a[i];
    	delete[] matrices_b[i];
    	delete[] matrices_c[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    chameleon_finalize();
    MPI_Finalize();
//#endif
    return 0;
}
