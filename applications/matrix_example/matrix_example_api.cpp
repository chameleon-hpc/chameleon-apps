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

#ifndef PARALLEL_INIT
#define PARALLEL_INIT 1
#endif

#ifndef VERBOSE_MSG
#define VERBOSE_MSG 0
#endif

#ifndef VERBOSE_MATRIX
#define VERBOSE_MATRIX 0
#endif

#ifndef CHECK_GENERATED_TASK_ID
#define CHECK_GENERATED_TASK_ID 0
#endif

#ifndef SIMULATE_CONST_WORK
#define SIMULATE_CONST_WORK 0
#endif

#ifndef COMPILE_CHAMELEON
#define COMPILE_CHAMELEON 1
#endif

#ifndef COMPILE_TASKING
#define COMPILE_TASKING 1
#endif

#ifndef USE_TASK_ANNOTATIONS
#define USE_TASK_ANNOTATIONS 0
#endif

#ifndef USE_REPLICATION
#define USE_REPLICATION 0
#endif

#ifndef ITERATIVE_VERSION
#define ITERATIVE_VERSION 1
#endif

#ifndef NUM_ITERATIONS
#define NUM_ITERATIONS 1
#endif

#ifndef NUM_REPETITIONS
#define NUM_REPETITIONS 1
#endif

#ifndef USE_EXTERNAL_CALLBACK
#define USE_EXTERNAL_CALLBACK 0
#endif

#if !COMPILE_CHAMELEON
#undef USE_EXTERNAL_CALLBACK
#define USE_EXTERNAL_CALLBACK 0
#endif

#ifndef USE_ALIGNMENT
#define USE_ALIGNMENT 1
#endif

#ifndef USE_HUGE_PAGES
#define USE_HUGE_PAGES 0
#endif

//#define LOG(rank, str) fprintf(stderr, "#R%d: %s\n", rank, str)
#define LOG(rank, str) printf("#R%d: %s\n", rank, str)

#define SPEC_RESTRICT __restrict__
//#define SPEC_RESTRICT restrict

#include <assert.h>
#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <inttypes.h>
#include <iostream>
#include <string.h>
#include <sstream>
#include "math.h"
#include <malloc.h>
#include <cmath>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <atomic>
#include <list>
#include <vector>
#include <algorithm>
#include <functional>
#include "util_string.h"

#if CHECK_GENERATED_TASK_ID
#include <mutex>
#endif

#ifndef DPxMOD
#define DPxMOD "0x%0*" PRIxPTR
#endif

#ifndef DPxPTR
#define DPxPTR(ptr) ((int)(2*sizeof(uintptr_t))), ((uintptr_t) (ptr))
#endif

#if COMPILE_CHAMELEON
#include "chameleon.h"
#include "chameleon_pre_init.h"
#endif

// static rank id that can also be used in other functions except main
static int my_rank_id = 0;
static int num_procs = -1;

typedef enum matrix_size_mode_t {
    matrix_size_mode_normal = 0,
    matrix_size_mode_non_uniform = 1
} matrix_size_mode_t;

matrix_size_mode_t matrix_size_mode = matrix_size_mode_normal;
int numberOfTasks = 0;

// mode: normal
int matrixSize = 100;

// mode: non-uniform
typedef enum non_uniform_ordering_t {
    non_uniform_ordering_high_to_low = 0,
    non_uniform_ordering_low_to_high = 1
} non_uniform_ordering_t;

typedef struct non_uniform_matrix_settings_t {
    int matrix_size;
    int number_tasks;
} non_uniform_matrix_settings_t;

non_uniform_ordering_t non_uniform_ordering = non_uniform_ordering_high_to_low;
std::vector<non_uniform_matrix_settings_t> non_uniform_matrix_settings;
std::vector<int> non_uniform_full_array_matrix_sizes;

#if USE_EXTERNAL_CALLBACK
typedef struct my_custom_params_t {
    int val1;
    TYPE_TASK_ID task_id;
} my_custom_params_t;

void print_finish_message(void *param) {
    // parse parameter again to regular data type
    my_custom_params_t *mydata = (my_custom_params_t *) param;
    printf("#R%d (OS_TID:%ld): External task finish callback for task with id %d with value %d\n", my_rank_id, syscall(SYS_gettid), mydata->task_id, mydata->val1);
    // clean up
    free(mydata);
}
#endif

#define MEM_ALIGNMENT 4096

static inline void* alloc(size_t size)
{
#if USE_ALIGNMENT
    void* p = memalign(MEM_ALIGNMENT, size);
#else
    void* p = malloc(size);
#endif
#if !USE_HUGE_PAGES
    madvise(p, size, MADV_NOHUGEPAGE);
#endif
    return p;
}

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

void compute_matrix_matrix(double * SPEC_RESTRICT a, double * SPEC_RESTRICT b, double * SPEC_RESTRICT c, int matrixSize) {
    // make the tasks more computational expensive by repeating this operation several times to better see effects 
    for(int iter=0;iter<NUM_REPETITIONS;iter++) {
        for(int i=0;i<matrixSize;i++) {
            for(int j=0;j<matrixSize;j++) {
                c[i*matrixSize+j]=0;
                for(int k=0;k<matrixSize;k++) {
                    c[i*matrixSize+j] += a[i*matrixSize+k] * b[k*matrixSize+j];
                }
            }
        }
    }
}

bool check_test_matrix(double *c, int matrix_idx, double val, int matrixSize) {
    if (NUM_REPETITIONS > 0) {
        for(int i=0;i<matrixSize;i++) {
            for(int j=0;j<matrixSize;j++) {
                if(fabs(c[i*matrixSize+j] - val) > 1e-3) {
                    printf("#R%d (OS_TID:%ld): Error in matrix %03d entry (%d,%d) expected:%f but value is %f\n", my_rank_id, syscall(SYS_gettid),matrix_idx,i,j,val,c[i*matrixSize+j]);
                    return false;
                }
            }
        }
    }
	return true;
}

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
    if(my_rank_id == 0) {
        std::cout << "Usage (mode=normal): mpiexec -n np ./matrixExample matrixSize [nt_(0) ... nt_(np-1)] " << std::endl;
        std::cout << "    Arguments: " << std::endl;
        std::cout << "        matrixSize:   Number of elements of the matrixSize x matrixSize matrices" << std::endl;
        std::cout << "        nt_(i):       Number of tasks for process i " << std::endl;
        std::cout << "    If the number of tasks is not specified for every process, the application will generate an initial task distribution" << std::endl << std::endl;

        std::cout << "Usage (mode=non-uniform): mpiexec -n np ./matrixExample non-uniform matrixSizes numberTasks [order_(0) ... order_(np-1)] " << std::endl;
        std::cout << "    Arguments: " << std::endl;
        std::cout << "        matrixSizes:  Comma separated list of different matrix sizes for non-uniform task creation" << std::endl;
        std::cout << "        numberTasks:  Comma separated list defining number of tasks for each matrix size" << std::endl;
        std::cout << "        order_(i):    Ordering of tasks using matrix sizes for rank/process i; 0=\"high to low\" (default); 1=\"low to high\"" << std::endl << std::endl;
    }
}

void printArray(int rank, double * SPEC_RESTRICT array, char* arr_name, int n) {
    printf("#R%d (OS_TID:%ld): %s[0-%d] at (" DPxMOD "): ", rank, syscall(SYS_gettid), arr_name, n, DPxPTR(&array[0]));
    for(int i = 0; i < n; i++) {
        printf("%f ", array[i]);
    }
    printf("\n");
}

void matrixMatrixKernel(double * SPEC_RESTRICT A, double * SPEC_RESTRICT B, double * SPEC_RESTRICT C, int matrixSize, int i) {
#if VERBOSE_MATRIX
    int iMyRank2;
    MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank2);
    printArray(iMyRank2, A, "A", 10);
    printArray(iMyRank2, B, "B", 10);
    printArray(iMyRank2, C, "C", 10);
#endif

#if SIMULATE_CONST_WORK
    // simulate work by just touching the arrays and wait for 50 ms here
    C[matrixSize] = A[matrixSize] * B[matrixSize];
    usleep(50000);
#else
    compute_matrix_matrix(A, B, C, matrixSize);
#endif
}

int parse_command_line_args(int argc, char **argv) {
    if(argc>=2 && strcmp(argv[1], "non-uniform")==0) {
        matrix_size_mode = matrix_size_mode_non_uniform;

        if(argc < 4) {
            std::cout << "Error: Insufficient number parameters" << std::endl;
            printHelpMessage();
            return 1;
        }

        // parse matrix sizes and number of tasks
        std::string str_msizes(argv[2]);
        std::list<std::string> cur_split_msizes = split(str_msizes, ',');
        std::string str_ntasks(argv[3]);
        std::list<std::string> cur_split_ntasks = split(str_ntasks, ',');
        if(cur_split_msizes.size() != cur_split_ntasks.size()) {
            std::cout << "Error: Number of matrix sizes and number of tasks does not match!" << std::endl;
            return 1;
        }

        for (std::string s : cur_split_msizes) {
            non_uniform_matrix_settings_t new_obj;
            new_obj.matrix_size = std::atoi(s.c_str());
            non_uniform_matrix_settings.push_back(new_obj);
        }

        numberOfTasks = 0;
        int count = 0;
        for (std::string s : cur_split_ntasks) {
            int tmp_num = std::atoi(s.c_str());
            non_uniform_matrix_settings[count].number_tasks = tmp_num;
            numberOfTasks += tmp_num;
            count++;
        }

        // parse ordering
        if(argc > 4) {
            if(argc != 4+num_procs) {
                std::cout << "Error: Number of matrix ordering values does not match number of processes/ranks!" << std::endl;
                return 1;
            }
            int tmp_order = std::atoi(argv[4+my_rank_id]);
            non_uniform_ordering = (non_uniform_ordering_t)tmp_order;
        }

        // apply ordering
        if (non_uniform_ordering == non_uniform_ordering_high_to_low) {
            std::sort(
                non_uniform_matrix_settings.begin(), 
                non_uniform_matrix_settings.end(),
                [](const non_uniform_matrix_settings_t & a, const non_uniform_matrix_settings_t & b) -> bool
                { 
                    return a.matrix_size > b.matrix_size;
                }
            );
        } else {
            std::sort(
                non_uniform_matrix_settings.begin(), 
                non_uniform_matrix_settings.end(),
                [](const non_uniform_matrix_settings_t & a, const non_uniform_matrix_settings_t & b) -> bool
                { 
                    return b.matrix_size > a.matrix_size;
                }
            );
        }

        non_uniform_full_array_matrix_sizes.clear();
        for (non_uniform_matrix_settings_t s : non_uniform_matrix_settings) {
            for (int i = 0; i < s.number_tasks; i++) {
                non_uniform_full_array_matrix_sizes.push_back(s.matrix_size);
            }
        }

        // ===== DEBUG
        // printf("Rank#%d - Ordering: %d\n", my_rank_id, non_uniform_ordering);
        // for (non_uniform_matrix_settings_t s : non_uniform_matrix_settings) {
        //     printf("Rank#%d - MatrixSize: %d, NumTasks: %d\n", my_rank_id, s.matrix_size, s.number_tasks);
        // }
        // printf("Rank#%d - Size Array: ", my_rank_id);
        // for (int s : non_uniform_full_array_matrix_sizes) {
        //     printf("%d,", s);
        // }
        // printf("\n");
        // ===== DEBUG

    } else if(argc==2) {
        matrix_size_mode = matrix_size_mode_normal;
        matrixSize = atoi( argv[1] );
        if(RANDOMDIST) {
            int *dist = new int[num_procs];
            if( my_rank_id==0 ) {
                compute_random_task_distribution(dist, num_procs);
            }
            MPI_Bcast(dist, num_procs, MPI_INTEGER, 0, MPI_COMM_WORLD);
            numberOfTasks = dist[my_rank_id];
            delete[] dist;
        } else {
            numberOfTasks = NR_TASKS;
        }
    } else if(argc==num_procs+2) {
        matrix_size_mode = matrix_size_mode_normal;
        if(my_rank_id==0) {
            LOG(my_rank_id, "using user-defined initial load distribution...");    
        }
        matrixSize = atoi( argv[1] ); 
        numberOfTasks = atoi( argv[my_rank_id+2] ); 
    } else { 
        printHelpMessage();
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
	int iMyRank, iNumProcs;
	int provided;
	int requested = MPI_THREAD_MULTIPLE;
	MPI_Init_thread(&argc, &argv, requested, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
    my_rank_id = iMyRank;
    num_procs = iNumProcs;
	double fTimeStart, fTimeEnd;
	double wTimeCham, wTimeHost;
	bool pass = true;

#if COMPILE_CHAMELEON
    chameleon_pre_init();
    #pragma omp parallel
    {
        chameleon_thread_init();
    }
    // necessary to be aware of binary base addresses to calculate offset for target entry functions
    chameleon_determine_base_addresses((void *)&main);
    chameleon_post_init_serial();
    #pragma omp barrier

#if USE_EXTERNAL_CALLBACK
    chameleon_external_callback_t call_back_fcn = print_finish_message;
#endif
#endif /* COMPILE_CHAMELEON */

    int ret_code = parse_command_line_args(argc, argv);
    if (ret_code != 0) {
        return ret_code;
    }

    if(iMyRank == 0) {
        if(matrix_size_mode == matrix_size_mode_normal) {
            printf("Mode: Normal Task Distribution\n");
        } else if (matrix_size_mode == matrix_size_mode_non_uniform) {
            printf("Mode: Non-Uniform Task Distribution\n");
        }
    }

#if CHECK_GENERATED_TASK_ID
    std::mutex mtx_t_ids;
    std::list<int32_t> t_ids;
#endif
	
    std::string msg = "will create "+std::to_string(numberOfTasks)+" tasks";
    LOG(iMyRank, msg.c_str());

	double **matrices_a, **matrices_b, **matrices_c;
	matrices_a = new double*[numberOfTasks];
	matrices_b = new double*[numberOfTasks];
	matrices_c = new double*[numberOfTasks];

#if PARALLEL_INIT
    if(iMyRank == 0) {
        printf("Executing parallel init\n");
    }
    #pragma omp parallel for
#endif
	for(int i=0; i<numberOfTasks; i++) {

        int cur_size = matrixSize;
        if(matrix_size_mode == matrix_size_mode_non_uniform) {
            cur_size = non_uniform_full_array_matrix_sizes[i];
        }

 		matrices_a[i] = (double*) alloc((long)cur_size*cur_size*sizeof(double));
    	matrices_b[i] = (double*) alloc((long)cur_size*cur_size*sizeof(double));
    	matrices_c[i] = (double*) alloc((long)cur_size*cur_size*sizeof(double));
    	if(RANDOMINIT) {
    		initialize_matrix_rnd(matrices_a[i], cur_size);
    		initialize_matrix_rnd(matrices_b[i], cur_size);
    		initialize_matrix_zero(matrices_c[i], cur_size);
    	}
    	else {
    		initialize_matrix_test_A(matrices_a[i], cur_size);
    		initialize_matrix_test_A(matrices_b[i], cur_size);
    		initialize_matrix_zero(matrices_c[i], cur_size);
    	}
#if VERBOSE_MATRIX
        printArray(iMyRank, matrices_a[i], "A", 10);
        printArray(iMyRank, matrices_b[i], "B", 10);
        printArray(iMyRank, matrices_c[i], "C", 10);
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);

#if COMPILE_CHAMELEON
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
    	// if(iMyRank==0) {
#if USE_REPLICATION
        std::atomic<int> replicated_cnt = 0;
        int num_to_replicate            = 10;
#endif
		#pragma omp for
    		for(int i=0; i<numberOfTasks; i++) {
                int cur_size = matrixSize;
                if(matrix_size_mode == matrix_size_mode_non_uniform) {
                    cur_size = non_uniform_full_array_matrix_sizes[i];
                }
#if VERBOSE_MSG
                printf("R#%d: Chameleon: MxM multiplication %03d working with matrix size %d\n", iMyRank, i, cur_size);
#endif
				double * SPEC_RESTRICT A = matrices_a[i];
		        double * SPEC_RESTRICT B = matrices_b[i];
		        double * SPEC_RESTRICT C = matrices_c[i];
#if VERBOSE_MATRIX
                printArray(iMyRank, A, "A", 10);
                printArray(iMyRank, B, "B", 10);
                printArray(iMyRank, C, "C", 10);
#endif
                // here we need to call library function to add task entry point and parameters by hand
                void* literal_matrix_size   = *(void**)(&cur_size);
                void* literal_i             = *(void**)(&i);

                chameleon_map_data_entry_t* args = new chameleon_map_data_entry_t[5];
                args[0] = chameleon_map_data_entry_create(A, cur_size*cur_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                args[1] = chameleon_map_data_entry_create(B, cur_size*cur_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                args[2] = chameleon_map_data_entry_create(C, cur_size*cur_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_FROM);
                args[3] = chameleon_map_data_entry_create(literal_matrix_size, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                args[4] = chameleon_map_data_entry_create(literal_i, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);

                // create opaque task here
                cham_migratable_task_t *cur_task = chameleon_create_task((void *)&matrixMatrixKernel, 5, args);
#if USE_TASK_ANNOTATIONS
                chameleon_annotations_t* annotations = chameleon_get_task_annotations_opaque(cur_task);
                if(!annotations) {
                    annotations = chameleon_create_annotation_container();
                    chameleon_set_task_annotations(cur_task, annotations);
                }
                chameleon_set_annotation_int(annotations, "Int", 42);
                chameleon_set_annotation_double(annotations, "Dbl", 42.1345);
                chameleon_set_annotation_string(annotations, "Str", "Test123");
#endif
#if USE_REPLICATION
                if(iMyRank==0) {
                    if(replicated_cnt++<num_to_replicate) {
                        int num_replication = 1;
                        int *replication_ranks = new int[num_replication];
                        replication_ranks[0] = 1;
                        chameleon_set_task_replication_info(cur_task, num_replication, replication_ranks);
                        delete[] replication_ranks;
                    }
                }
#endif
                // get the id of the last task added
                TYPE_TASK_ID last_t_id = chameleon_get_task_id(cur_task);
#if CHECK_GENERATED_TASK_ID
                printf("#R%d (OS_TID:%ld): last task that has been created: %ld\n", iMyRank, syscall(SYS_gettid), last_t_id);
                mtx_t_ids.lock();
                t_ids.push_back(last_t_id);
                mtx_t_ids.unlock();
#endif
#if USE_EXTERNAL_CALLBACK
                my_custom_params_t* tmp_struct = (my_custom_params_t*) malloc(sizeof(my_custom_params_t));
                tmp_struct->val1 = 42;
                tmp_struct->task_id = last_t_id;
                chameleon_set_callback_task_finish(cur_task, call_back_fcn, (void*)tmp_struct);
#endif
                int32_t res = chameleon_add_task(cur_task);
                // clean up again
                delete[] args;
#if USE_TASK_ANNOTATIONS
                chameleon_annotations_t* tmp_annotations = chameleon_get_task_annotations(last_t_id);
                int     found = 0; 
                int     val_annotation_int;
                double  val_annotation_dbl;
                char*   val_annotation_string;
                
                found = chameleon_get_annotation_int(tmp_annotations, "Int", &val_annotation_int);
                assert(found == 1 && val_annotation_int == 42);
                found = chameleon_get_annotation_double(tmp_annotations, "Dbl", &val_annotation_dbl);
                assert(found == 1 && val_annotation_dbl == 42.1345);
                found = chameleon_get_annotation_string(tmp_annotations, "Str", &val_annotation_string);
                assert(found == 1 && strcmp("Test123", val_annotation_string) == 0);
#endif
    		}

#if CHECK_GENERATED_TASK_ID
        #pragma omp barrier
        #pragma omp single
        {
            printf("Before Running Tasks\n");
            for (std::list<int32_t>::iterator it=t_ids.begin(); it!=t_ids.end(); ++it) {
                printf("R#%d Task with id %d finished?? ==> %d\n", iMyRank, *it, chameleon_local_task_has_finished(*it));
            }
        }
#endif
    	int res = chameleon_distributed_taskwait(0);

        #pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);
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
        printf("#R%d: Computations with chameleon took %.5f\n", iMyRank, wTimeCham);
    }
    if(numberOfTasks>0) {
        for(int t=0; t<numberOfTasks; t++) {
            int cur_size = matrixSize;
            if(matrix_size_mode == matrix_size_mode_non_uniform) {
                cur_size = non_uniform_full_array_matrix_sizes[t];
            }
#if VERBOSE_MSG
            printf("R#%d: Chameleon: Validating resulting matrix %03d with matrix size %d\n", iMyRank, t, cur_size);
#endif
            pass &= check_test_matrix(matrices_c[t], t, cur_size, cur_size);
        }
        if(pass)
            LOG(iMyRank, "Validation: TEST SUCCESS");
        else
            LOG(iMyRank, "Validation: TEST FAILED");
    }
#endif /* COMPILE_CHAMELEON */

    MPI_Barrier(MPI_COMM_WORLD);

#if COMPILE_TASKING
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
            int cur_size = matrixSize;
            if(matrix_size_mode == matrix_size_mode_non_uniform) {
                cur_size = non_uniform_full_array_matrix_sizes[i];
            }
#if VERBOSE_MSG
            printf("R#%d: OpenMP Tasking: MxM multiplication %03d working with matrix size %d\n", iMyRank, i, cur_size);
#endif
            // double *A = matrices_a[i];
            // double *B = matrices_b[i];
            // double *C = matrices_c[i];
            
            // somehow target offloading is very slow when performing more that one iteration
            // #pragma omp target map(from: C[0:cur_size*cur_size]) map(to:cur_size, A[0:cur_size*cur_size], B[0:cur_size*cur_size]) device(1001)
            
            // uses normal tasks to have a fair comparison
            #pragma omp task default(shared) firstprivate(i,cur_size)
            {
                compute_matrix_matrix(matrices_a[i], matrices_b[i], matrices_c[i], cur_size);
            }
        }

        #pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);
#if ITERATIVE_VERSION
        }
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);
    fTimeEnd=MPI_Wtime();
    wTimeHost = fTimeEnd-fTimeStart;

    if( iMyRank==0 ) {
        printf("#R%d: Computations with normal tasking took %.5f\n", iMyRank, wTimeHost);
    }
    
    pass = true;
    if(numberOfTasks>0) {
        for(int t=0; t<numberOfTasks; t++) {
            int cur_size = matrixSize;
            if(matrix_size_mode == matrix_size_mode_non_uniform) {
                cur_size = non_uniform_full_array_matrix_sizes[t];
            }
#if VERBOSE_MSG
            printf("R#%d: OpenMP Tasking: Validating resulting matrix %03d with matrix size %d\n", iMyRank, t, cur_size);
#endif
            pass &= check_test_matrix(matrices_c[t], t, cur_size, cur_size);
        }
        if(pass)
            LOG(iMyRank, "Validation: TEST SUCCESS");
        else
            LOG(iMyRank, "Validation: TEST FAILED");
    }
#endif /* COMPILE_TASKING */

#if COMPILE_TASKING && COMPILE_CHAMELEON
    if( iMyRank==0 ) {
        printf("#R%d: This corresponds to a speedup of %.5f!\n", iMyRank, wTimeHost/wTimeCham);
    }
#endif /* COMPILE_TASKING && COMPILE_CHAMELEON */  

    //deallocate matrices
    for(int i=0; i<numberOfTasks; i++) {
    	free(matrices_a[i]);
    	free(matrices_b[i]);
    	free(matrices_c[i]);
    }

    delete[] matrices_a;
    delete[] matrices_b;
    delete[] matrices_c;

    MPI_Barrier(MPI_COMM_WORLD);
#if COMPILE_CHAMELEON
    #pragma omp parallel
    {
        chameleon_thread_finalize();
    }
    chameleon_finalize();
#endif
    MPI_Finalize();
    return 0;
}
