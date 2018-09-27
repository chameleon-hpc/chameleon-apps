// size of the matrices
//#ifndef MATRIX_SIZE
//#define MATRIX_SIZE 500
//#endif

// number of tasks 
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

void initialize_matrix_rnd(double *mat, int matrixSize) {
	double lower_bound = 0;
	double upper_bound = 10000;
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::default_random_engine re;

	for(int i=0; i<matrixSize*matrixSize; i++) {
		mat[i]= unif(re);
	}
}

void initialize_matrix_zero(double *mat, int matrixSize) {
	for(int i=0; i<matrixSize*matrixSize; i++) {
		mat[i]= 0;
	}
}

void initialize_matrix_test_A(double *mat, int matrixSize) {
	for(int i=0; i<matrixSize*matrixSize; i++) {
			mat[i]= 1;
    }
}

#pragma omp declare target
void compute_matrix_matrix(double *a, double *b, double *c, int matrixSize) {
	for(int i=0;i<matrixSize;i++) {
		for(int j=0;j<matrixSize;j++) {
			c[i*matrixSize+j]=0;
			for(int k=0;k<matrixSize;k++) {
				c[i*matrixSize+j] += a[i*matrixSize+k] * b[k*matrixSize+j];
			}
		}
	}
}
#pragma omp end declare target

#pragma omp declare target
bool check_test_matrix(double *c, double val, int matrixSize) {
	int iMyRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
	for(int i=0;i<matrixSize;i++) {
		for(int j=0;j<matrixSize;j++) {
			if(c[i*matrixSize+j]!=val) {
				printf("#R%d: Error in matrix entry (%d,%d) expected:%f but value is %f\n", iMyRank,i,j,val,c[i*matrixSize+j]);
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

void printHelpMessage() {
    std::cout<<"Usage: mpiexec -n np ./matrixExample matrixSize [nt_(0) ... nt_(np-1)] "<<std::endl;
    std::cout<<"    Arguments: "<<std::endl;
    std::cout<<"        matrixSize: number of elements of the matrixSize x matrixSize matrices"<<std::endl;
    std::cout<<"        nt_(i): number of tasks for process i "<<std::endl;
    std::cout<<"If the number of tasks is not specified for every process, the application will generate an initial task distribution"<<std::endl; 
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
	int matrixSize;
	double fTimeStart, fTimeEnd;
	double wTimeCham, wTimeHost;
	bool pass = true;

	chameleon_init();

    if(argc==2) {
            matrixSize = atoi( argv[1] );
            if(RANDOMDIST) {
                int *dist = new int[iNumProcs];
            	if( iMyRank==0 ) {
 			        compute_random_task_distribution(dist, iNumProcs);
		        }
                MPI_Bcast(dist, iNumProcs, MPI_INTEGER, 0, MPI_COMM_WORLD);
                numberOfTasks = dist[iMyRank];
                delete[] dist;
            }
            else {
		        numberOfTasks = NR_TASKS;
            }
    }
    else if(argc==iNumProcs+2) {
            if(iMyRank==0) {
    			LOG(iMyRank, "using user-defined initial load distribution...");    
    		} 
            matrixSize = atoi( argv[1] ); 
            numberOfTasks = atoi( argv[iMyRank+2] ); 
    }
    else{ 
            printHelpMessage();
            return 0;     
    }
	
    std::string msg = "will create "+std::to_string(numberOfTasks)+" tasks";
    LOG(iMyRank, msg.c_str());

	double **matrices_a, **matrices_b, **matrices_c;

	matrices_a = new double*[numberOfTasks];
	matrices_b = new double*[numberOfTasks];
	matrices_c = new double*[numberOfTasks];

	//allocate and initialize matrices
	for(int i=0; i<numberOfTasks; i++) {
 		matrices_a[i] = new double[matrixSize*matrixSize];
    		matrices_b[i] = new double[matrixSize*matrixSize];
    		matrices_c[i] = new double[matrixSize*matrixSize];
    	if(RANDOMINIT) {
    		initialize_matrix_rnd(matrices_a[i], matrixSize);
    		initialize_matrix_rnd(matrices_b[i], matrixSize);
    		initialize_matrix_zero(matrices_c[i], matrixSize);
    	}
    	else {
    		initialize_matrix_test_A(matrices_a[i], matrixSize);
    		initialize_matrix_test_A(matrices_b[i], matrixSize);
    		initialize_matrix_zero(matrices_c[i], matrixSize);
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
				#pragma omp target map(tofrom: C[0:matrixSize*matrixSize]) map(to:matrixSize, A[0:matrixSize*matrixSize], B[0:matrixSize*matrixSize]) device(DEV_NR)
				{
					//check_test_matrix(A, 1);
					//check_test_matrix(B, 1);
					//check_test_matrix(C, 0);
					compute_matrix_matrix(A, B, C, matrixSize);
					//check_test_matrix(C, MATRIX_SIZE);
				}
				//LOG(iMyRank, "offloading to chameleon");
    		}

    	//LOG(iMyRank, "entering taskwait");
    	int res = chameleon_distributed_taskwait();
    	//LOG(iMyRank, "leaving taskwait");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    fTimeEnd=MPI_Wtime();
    wTimeCham = fTimeEnd-fTimeStart;
    if( iMyRank==0 ) {
        printf("#R%d: Computations with chameleon offloading took %.2f\n", iMyRank, wTimeCham);
    }
    LOG(iMyRank, "Validation:");
    if(numberOfTasks>0) {
        for(int i=0; i<numberOfTasks; i++) {
            pass &= check_test_matrix(matrices_c[i], matrixSize, matrixSize);
        }
        if(pass)
            LOG(iMyRank, "TEST SUCCESS");
        else
            LOG(iMyRank, "TEST FAILED");
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
				#pragma omp target map(tofrom:C[0:matrixSize*matrixSize]) map(to:matrixSize, A[0:matrixSize*matrixSize], B[0:matrixSize*matrixSize]) device(1001)
				{
					//check_test_matrix(A, 1);
					//check_test_matrix(B, 1);
					//check_test_matrix(C, 0);
					compute_matrix_matrix(A, B, C, matrixSize);
					//check_test_matrix(C, MATRIX_SIZE);
				}
				//LOG(iMyRank, "offloading to chameleon");
    		}

    	//LOG(iMyRank, "entering taskwait");
    	//int res = chameleon_distributed_taskwait();
    	//LOG(iMyRank, "leaving taskwait");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    fTimeEnd=MPI_Wtime();
    wTimeHost = fTimeEnd-fTimeStart;
    if( iMyRank==0 ) {
        printf("#R%d: Computations with host offloading took %.2f\n", iMyRank, wTimeHost);
        printf("#R%d: This corresponds to a speedup of %.2f!\n", iMyRank, wTimeHost/wTimeCham);
    }  
    LOG(iMyRank, "Validation:");
    if(numberOfTasks>0) {
        for(int i=0; i<numberOfTasks; i++) {
            pass &= check_test_matrix(matrices_c[i], matrixSize, matrixSize);
        }
        if(pass)
            LOG(iMyRank, "TEST SUCCESS");
        else
            LOG(iMyRank, "TEST FAILED");
    }

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
