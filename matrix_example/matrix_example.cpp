// size of the matrices
//#ifndef MATRIX_SIZE
//#define MATRIX_SIZE 500
//#endif

// number of tasks 
#ifndef NR_TASKS
#define NR_TASKS 200
#endif

#ifndef RANDOMINIT
#define RANDOMINIT 0
#endif

#ifndef RANDOMDIST
#define RANDOMDIST 1
#endif

#ifndef VERY_VERBOSE
#define VERY_VERBOSE 0
#endif

#ifndef CHECK_GENERATED_TASK_ID
#define CHECK_GENERATED_TASK_ID 0
#endif

#ifndef SIMULATE_CONST_WORK
#define SIMULATE_CONST_WORK 0
#endif

#ifndef CALC_SPEEDUP
#define CALC_SPEEDUP 1
#endif

#ifndef ITERATIVE_VERSION
#define ITERATIVE_VERSION 1
#endif

#ifndef NUM_ITERATIONS
#define NUM_ITERATIONS 10
#endif

#ifndef DEV_NR
#define DEV_NR 1002 // CHAMELEON_MPI
#endif

//#define LOG(rank, str) fprintf(stderr, "#R%d: %s\n", rank, str)
#define LOG(rank, str) printf("#R%d: %s\n", rank, str)

#include <mpi.h>
#include "chameleon.h"
#include <cstdlib>
#include <cstdio>
#include <random>
#include <iostream>
#include <string>
#include <sstream>
#include "math.h"
#include <cmath>
#include <unistd.h>
#include <sys/syscall.h>

#if CHECK_GENERATED_TASK_ID
#include <mutex>
#include <list>
#endif


#define SPEC_RESTRICT __restrict__
//#define SPEC_RESTRICT restrict

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
void compute_matrix_matrix(double * SPEC_RESTRICT a, double * SPEC_RESTRICT b, double * SPEC_RESTRICT c, int matrixSize) {
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
			if(fabs(c[i*matrixSize+j] - val) > 1e-3) {
				printf("#R%d (OS_TID:%ld): Error in matrix entry (%d,%d) expected:%f but value is %f\n", iMyRank, syscall(SYS_gettid),i,j,val,c[i*matrixSize+j]);
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

#if CHECK_GENERATED_TASK_ID
    std::mutex mtx_t_ids;
    std::list<int32_t> t_ids;
#endif

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
	#pragma omp parallel for
	for(int i=0; i<numberOfTasks; i++) {
 		matrices_a[i] = new double[(long)matrixSize*matrixSize];
    	matrices_b[i] = new double[(long)matrixSize*matrixSize];
    	matrices_c[i] = new double[(long)matrixSize*matrixSize];
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
#if VERY_VERBOSE
        printf("#R%d (OS_TID:%ld): Master A[%d] at (" DPxMOD ")\n", iMyRank, syscall(SYS_gettid), i, DPxPTR(&matrices_a[i][0]));
        printf("#R%d (OS_TID:%ld): Master B[%d] at (" DPxMOD ")\n", iMyRank, syscall(SYS_gettid), i, DPxPTR(&matrices_b[i][0]));
        printf("#R%d (OS_TID:%ld): Master C[%d] at (" DPxMOD ")\n", iMyRank, syscall(SYS_gettid), i, DPxPTR(&matrices_c[i][0]));
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);
    fTimeStart=MPI_Wtime();

    #pragma omp parallel
    {
#if ITERATIVE_VERSION
        for(int iter = 0; iter < NUM_ITERATIONS; iter++) {
            if(iMyRank == 0) {
                #pragma omp master
                printf("Executing iteration %d ...\n", iter);
            }
#endif
		#pragma omp for
    		for(int i=0; i<numberOfTasks; i++) {
				double * SPEC_RESTRICT A = matrices_a[i];
		        double * SPEC_RESTRICT B = matrices_b[i];
		        double * SPEC_RESTRICT C = matrices_c[i];
#if VERY_VERBOSE
                printf("#R%d (OS_TID:%ld): Itermediate A[%d] at (" DPxMOD ")\n", iMyRank, syscall(SYS_gettid), i, DPxPTR(&A[0]));
                printf("#R%d (OS_TID:%ld): Itermediate B[%d] at (" DPxMOD ")\n", iMyRank, syscall(SYS_gettid), i, DPxPTR(&B[0]));
                printf("#R%d (OS_TID:%ld): Itermediate C[%d] at (" DPxMOD ")\n", iMyRank, syscall(SYS_gettid), i, DPxPTR(&C[0]));	
#endif
                #pragma omp target map(tofrom: C[0:matrixSize*matrixSize]) map(to: A[0:matrixSize*matrixSize], B[0:matrixSize*matrixSize]) device(DEV_NR)
				{
#if VERY_VERBOSE
                    int iMyRank2;
	                MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank2);
                    printf("#R%d (OS_TID:%ld): A[%d] at (" DPxMOD ")\n", iMyRank2, syscall(SYS_gettid), i, DPxPTR(&A[0]));
                    printf("#R%d (OS_TID:%ld): B[%d] at (" DPxMOD ")\n", iMyRank2, syscall(SYS_gettid), i, DPxPTR(&B[0]));
                    printf("#R%d (OS_TID:%ld): C[%d] at (" DPxMOD ")\n", iMyRank2, syscall(SYS_gettid), i, DPxPTR(&C[0]));
#endif

#if SIMULATE_CONST_WORK
                    // simulate work by just touching the arrays and wait for 50 ms here
                    C[matrixSize] = A[matrixSize] * B[matrixSize];
                    usleep(50000);
#else
					compute_matrix_matrix(A, B, C, matrixSize);
#endif
				}

#if CHECK_GENERATED_TASK_ID
                TYPE_TASK_ID last_t_id = chameleon_get_last_local_task_id_added();
                printf("#R%d (OS_TID:%ld): last task that has been created: %ld\n", iMyRank, syscall(SYS_gettid), last_t_id);
                mtx_t_ids.lock();
                t_ids.push_back(last_t_id);
                mtx_t_ids.unlock();
#endif
    		}

#if CHECK_GENERATED_TASK_ID
        #pragma omp barrier
        #pragma omp master
        {
            printf("Before Running Tasks\n");
            for (std::list<int32_t>::iterator it=t_ids.begin(); it!=t_ids.end(); ++it) {
                printf("R#%d Task with id %d finished?? ==> %d\n", iMyRank, *it, chameleon_local_task_has_finished(*it));
            }
        }
        #pragma omp barrier
#endif
    	int res = chameleon_distributed_taskwait(0);
#if ITERATIVE_VERSION
        }
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);

#if CHECK_GENERATED_TASK_ID
    printf("After Running Tasks\n");
    for (std::list<int32_t>::iterator it=t_ids.begin(); it!=t_ids.end(); ++it) {
        printf("R#%d Task with id %d finished?? ==> %d\n", iMyRank, *it, chameleon_local_task_has_finished(*it));
    }
#endif

    fTimeEnd=MPI_Wtime();
    wTimeCham = fTimeEnd-fTimeStart;
    if( iMyRank==0 ) {
        printf("#R%d: Computations with chameleon offloading took %.2f\n", iMyRank, wTimeCham);
    }
    LOG(iMyRank, "Validation:");
    if(numberOfTasks>0) {
        for(int t=0; t<numberOfTasks; t++) {
            pass &= check_test_matrix(matrices_c[t], matrixSize, matrixSize);
        }
        if(pass)
            LOG(iMyRank, "TEST SUCCESS");
        else
            LOG(iMyRank, "TEST FAILED");
    }

    MPI_Barrier(MPI_COMM_WORLD);

#if CALC_SPEEDUP
    fTimeStart=MPI_Wtime();
    #pragma omp parallel
    {
#if ITERATIVE_VERSION
        for(int iter = 0; iter < NUM_ITERATIONS; iter++) {
            if(iMyRank == 0) {
                #pragma omp master
                printf("Executing iteration %d ...\n", iter);
            }
#endif
		#pragma omp for
        for(int i=0; i<numberOfTasks; i++) {
            double *A = matrices_a[i];
            double *B = matrices_b[i];
            double *C = matrices_c[i];	
            // somehow target offloading is very slow when performaing more that one iteration
            // #pragma omp target map(from: C[0:matrixSize*matrixSize]) map(to:matrixSize, A[0:matrixSize*matrixSize], B[0:matrixSize*matrixSize]) device(1001)
            // uses normal tasks to have a fair comparison
            #pragma omp task default(shared) firstprivate(A,B,C)
            {
                compute_matrix_matrix(A, B, C, matrixSize);
            }
        }
#if ITERATIVE_VERSION
        }
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);
    fTimeEnd=MPI_Wtime();
    wTimeHost = fTimeEnd-fTimeStart;
    if( iMyRank==0 ) {
        printf("#R%d: Computations with host offloading took %.2f\n", iMyRank, wTimeHost);
        printf("#R%d: This corresponds to a speedup of %.2f!\n", iMyRank, wTimeHost/wTimeCham);
    }
    LOG(iMyRank, "Validation:");
    pass = true;
    if(numberOfTasks>0) {
        for(int t=0; t<numberOfTasks; t++) {
            pass &= check_test_matrix(matrices_c[t], matrixSize, matrixSize);
        }
        if(pass)
            LOG(iMyRank, "TEST SUCCESS");
        else
            LOG(iMyRank, "TEST FAILED");
    }
#endif

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
