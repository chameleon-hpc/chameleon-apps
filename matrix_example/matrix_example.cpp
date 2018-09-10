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

#define LOG(rank, str) std::cout<<"#R"<<rank<<":"<<str<<std::endl

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

void initialize_matrix_test_B(double *mat){
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
				LOG(iMyRank, "Error in matrix entry ("<<i<<","<<j<<") expected:"<<val<<" but value is "<<c[i*MATRIX_SIZE+j]);
				return false;
			}
		}
	}
	return true;
}

int main(int argc, char **argv)
{
	int iMyRank, iNumProcs;
	/* MPI Initialization */
	// MPI_Init(&argc, &argv);
	int provided;
	// int reqeuested = MPI_THREAD_FUNNELED;
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
    		initialize_matrix_test_B(matrices_b[i]);
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
		initialize_matrix_test_B(B);
		initialize_matrix_zero(C);
	}

	#pragma omp parallel
    {
    	if(iMyRank==0) {
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
			#pragma omp target device(DEV_NR) //map(tofrom:A[0:MATRIX_SIZE*MATRIX_SIZE-1],B[0:MATRIX_SIZE*MATRIX_SIZE-1], C[0:MATRIX_SIZE*MATRIX_SIZE-1]) device(DEV_NR)
			{
    			//LOG(iMyRank, "executing");
				//check_test_matrix(A, 1);
				//check_test_matrix(B, 1);
				//check_test_matrix(C, 0);
				//compute_matrix_matrix(A, B, C);
				//check_test_matrix(C, 0);
			}
    	}
    	}
    	LOG(iMyRank, "entering taskwait");
    	int res = chameleon_distributed_taskwait();
    	LOG(iMyRank, "leaving taskwait");
    	//check_test_matrix(matrices_c[0], MATRIX_SIZE);
    }
    //bool pass = check_test_matrix(matrices_c[0], MATRIX_SIZE);

    //if(pass)  LOG(iMyRank, "PASSED TEST");
    //else LOG(iMyRank, "FAILED TEST");

    //deallocate matrices
    for(int i=0; i<NR_TASKS; i++) {
    	delete[] matrices_a[i];
    	delete[] matrices_b[i];
    	delete[] matrices_c[i];
    }
//    int i;
//    int step = 0;
//
//#if USE_COMPLEX
//    double a[n];
//    int b[n];
//#else
//    int scalar_A;
//    double scalar_B;
//    int scalar_C;
//#endif
//
//#if USE_MPI
//    int iMyRank, iNumProcs;
//    /* MPI Initialization */
//    // MPI_Init(&argc, &argv);
//    int provided;
//    // int reqeuested = MPI_THREAD_FUNNELED;
//    int reqeuested = MPI_THREAD_MULTIPLE;
//    MPI_Init_thread(&argc, &argv, reqeuested, &provided);
//    MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
//    MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
//
//    chameleon_init();
//
//    #pragma omp target device(1001) // 1001 = CHAMELEON_HOST
//    {
//       DBP("chameleon_init - dummy region\n");
//    }
//
//    if (iMyRank == 0)
//    {
//#endif
//        // data initialization
//#if USE_COMPLEX
//        for(i = 0; i < n; i++)
//        {
//            a[i] = 5.0;
//            b[i] = 3;
//        }
//        // calculate offsets to decompose array into single tasks
//        step = ARR_SIZE / NR_TASKS;
//        if (step * NR_TASKS < ARR_SIZE) step++;
//        // for(i = 0; i < NR_TASKS; i++) {
//        //     int tmp_idx_start = i * step;
//        //     printf("Master: array_a_dbl[%d] = %f at (" DPxMOD ")\n", tmp_idx_start, a[tmp_idx_start], DPxPTR(&a[tmp_idx_start]));
//        //     printf("Master: array_b_int[%d] = %d at (" DPxMOD ")\n", tmp_idx_start, b[tmp_idx_start], DPxPTR(&b[tmp_idx_start]));
//        // }
//        // for(i = 0; i < ARR_SIZE; i++) {
//        //     printf("Master: array_a_dbl[%02d] = %f at (" DPxMOD ")\n", i, a[i], DPxPTR(&a[i]));
//        // }
//        // for(i = 0; i < ARR_SIZE; i++) {
//        //     printf("Master: array_b_int[%02d] = %d at (" DPxMOD ")\n", i, b[i], DPxPTR(&b[i]));
//        // }
//#else
//        scalar_A = 1;
//        scalar_B = 2;
//        scalar_C = 3;
//        printf("Master: scalar_A_int = %d at (" DPxMOD ")\n", scalar_A, DPxPTR(&scalar_A));
//        printf("Master: scalar_B_dbl = %f at (" DPxMOD ")\n", scalar_B, DPxPTR(&scalar_B));
//        printf("Master: scalar_c_int = %d at (" DPxMOD ")\n", scalar_C, DPxPTR(&scalar_C));
//#endif //USE_COMPLEX
//        // measure start time here
//        fTimeStart = omp_get_wtime();
//#if USE_COMPLEX
//        // run multiple threads where each is creating target tasks
//        #pragma omp parallel
//        {
//            #pragma omp for
//            for(i = 0; i < NR_TASKS; i++) {
//                int idx_start = i * step;
//                int cur_len = step;
//                int idx_end = i*(step+1)-1;
//                if(idx_end > ARR_SIZE-1) {
//                    cur_len = ARR_SIZE-1-idx_start;
//                }
//#if USE_OFFLOADING
//                #pragma omp target map(tofrom:a[idx_start:cur_len], b[idx_start:cur_len]) device(DEV_NR)
//                {
//#endif // USE_OFFLOADING
//                    // for(int j = 0; j < cur_len; j++) {
//                    //     printf("Device: array_a_dbl[%d] = %f at (" DPxMOD ")\n", idx_start+j, a[idx_start+j], DPxPTR(&a[idx_start+j]));
//                    // }
//                    // for(int j = 0; j < cur_len; j++) {
//                    //     printf("Device: array_b_int[%d] = %d at (" DPxMOD ")\n", idx_start+j, b[idx_start+j], DPxPTR(&b[idx_start+j]));
//                    // }
//                    printf("Device: setting array_a_dbl = 13.37\n");
//                    printf("Device: setting array_b_int = 42\n");
//                    for(int o = 0; o < 2000; o++) {
//                    for(int j = 0; j < cur_len; j++) {
//                        a[idx_start+j] = 13.37 * 20.0 / 3.0;
//                        b[idx_start+j] = 42;
//                    }
//                    }
//#if USE_OFFLOADING
//                }
//#endif // USE_OFFLOADING
//            }
//#if USE_MPI
//            // work on tasks as long as there are tasks
//            int res = chameleon_distributed_taskwait();
//#endif // USE_MPI
//        }
//
//#else // USE_COMPLEX
//
//#if USE_OFFLOADING
//        #pragma omp target map(tofrom:scalar_A) device(DEV_NR)
//        {
//#endif // USE_OFFLOADING
//            printf("Device: Implict mapped scalar_B_dbl = %f at (" DPxMOD ")\n", scalar_B, DPxPTR(&scalar_B));
//            printf("Device: Implict mapped scalar_C_int = %d at (" DPxMOD ")\n", scalar_C, DPxPTR(&scalar_C));
//            printf("Device: scalar_A_int = %d at (" DPxMOD ")\n", scalar_A, DPxPTR(&scalar_A));
//            printf("Device: Setting scalar_A_int = 42\n");
//            scalar_A = 42;
//#if USE_OFFLOADING
//        }
//#endif // USE_OFFLOADING
//#if USE_MPI
//        // work on tasks as long as there are tasks
//        int res = chameleon_distributed_taskwait();
//#endif // USE_MPI
//#endif // USE_COMPLEX
//#if USE_MPI
//    } else {
//        #pragma omp parallel
//        {
//            // work on tasks as long as there are tasks
//            int res = chameleon_distributed_taskwait();
//        }
//    }
//
//    MPI_Barrier(MPI_COMM_WORLD);
//
//    if (iMyRank == 0)
//    {
//#endif
//        fTimeEnd = omp_get_wtime() - fTimeStart;
//        printf("Elapsed computation time: %.3f\n", fTimeEnd);
//        printf("Results:\n");
//#if USE_COMPLEX
//        // for(i = 0; i < NR_TASKS; i++) {
//        //     int tmp_idx_start = i*step;
//        //     printf("Master: array_a_dbl[%d] = %f at (" DPxMOD ")\n", tmp_idx_start, a[tmp_idx_start], DPxPTR(&a[tmp_idx_start]));
//        //     printf("Master: array_b_int[%d] = %d at (" DPxMOD ")\n", tmp_idx_start, b[tmp_idx_start], DPxPTR(&b[tmp_idx_start]));
//        // }
//        // for(i = 0; i < ARR_SIZE; i++) {
//        //     printf("Master: array_a_dbl[%02d] = %f at (" DPxMOD ")\n", i, a[i], DPxPTR(&a[i]));
//        // }
//        // for(i = 0; i < ARR_SIZE; i++) {
//        //     printf("Master: array_b_int[%02d] = %d at (" DPxMOD ")\n", i, b[i], DPxPTR(&b[i]));
//        // }
//#else
//        printf("Master: scalar_A_int = %d at (" DPxMOD ")\n", scalar_A, DPxPTR(&scalar_A));
//        printf("Master: scalar_B_dbl = %f at (" DPxMOD ")\n", scalar_B, DPxPTR(&scalar_B));
//#endif
//#if USE_MPI
//    }
    MPI_Barrier(MPI_COMM_WORLD);
    chameleon_finalize();
    MPI_Finalize();
//#endif
    return 0;
}
