#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#ifndef NUM_ITERATIONS
#define NUM_ITERATIONS 100
#endif

#ifndef MAX_MSG_SIZE_EXPONENT
#define MAX_MSG_SIZE_EXPONENT 28
#endif

#ifndef BENCHMARK_TYPE
#define BENCHMARK_TYPE 0 // regular ping pong benchmark using cached, contiguous data
// #define BENCHMARK_TYPE 1 // ping pong that always uses a different chunk of data (also contiguous but might not be cached)
// #define BENCHMARK_TYPE 2 // using custom MPI data types like done in Chameleon (strided data access but with cached data)
#endif

#ifndef USE_PAPI
#define USE_PAPI 1
#endif

// define number of PAPI events
#define NUM_EVENTS 4
#if USE_PAPI
#include <papi.h>

void handle_error (const char *msg)
{
    fprintf(stderr, "PAPI Error: %s\n", msg);
    exit(1);
}
#endif

int main(int argc, char *argv[]) {
    int iMyRank, iNumProcs;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);

    int msg_size_start  = 2;
    int msg_size_end    = MAX_MSG_SIZE_EXPONENT;
    int msg_size;
    int iter;
    int b;
    int tmp_err;

    if(iNumProcs != 2) {
        fprintf(stderr, "Wrong number of processes. Has to be 2.\n");
        return 1;
    }

    long long papi_cntr_values_start[NUM_EVENTS];
    long long papi_cntr_values_end[NUM_EVENTS];
    long long papi_cntr_values_stop[NUM_EVENTS];
    for(iter = 0; iter < NUM_EVENTS; iter++) {
        papi_cntr_values_start[iter] = 0;
        papi_cntr_values_end[iter] = 0;
    }

    #if USE_PAPI
    tmp_err = PAPI_library_init(PAPI_VER_CURRENT);
    int papi_events[NUM_EVENTS] = {PAPI_L3_TCA, PAPI_L3_TCM, PAPI_L3_LDM, PAPI_TOT_INS};
    #endif

    #if BENCHMARK_TYPE == 1
    int **tmp_buffers_send = malloc(sizeof(int*)*NUM_ITERATIONS);
    int **tmp_buffers_recv = malloc(sizeof(int*)*NUM_ITERATIONS);
    #elif BENCHMARK_TYPE == 2
    int **tmp_buffers_send = (int **) malloc(sizeof(int*)*4);
    int **tmp_buffers_recv = (int **) malloc(sizeof(int*)*4);
    #endif

    for(msg_size = msg_size_start; msg_size <= msg_size_end; msg_size++) {
        int cur_size_bytes = pow(2.0, msg_size);

        #if BENCHMARK_TYPE == 0
        int *tmp_buffer_send = (int*) malloc(cur_size_bytes);
        int *tmp_buffer_recv = (int*) malloc(cur_size_bytes);
        for(b = 0; b < cur_size_bytes/sizeof(int); b++) {
            tmp_buffer_send[b] = 1;
            tmp_buffer_send[b] = 0;
        }
        #elif BENCHMARK_TYPE == 1
        for(iter = 0; iter < NUM_ITERATIONS; iter++) {
            tmp_buffers_send[iter]  = (int*) malloc(cur_size_bytes);
            tmp_buffers_recv[iter]  = (int*) malloc(cur_size_bytes);
            int *cur_send           = tmp_buffers_send[iter];
            int *cur_recv           = tmp_buffers_recv[iter];
            for(b = 0; b < cur_size_bytes/sizeof(int); b++) {
                cur_send[b] = 1;
                cur_recv[b] = 0;
            }
        }
        #elif BENCHMARK_TYPE == 2
        for(iter = 0; iter < 4; iter++) {
            tmp_buffers_send[iter]  = (int*) malloc(cur_size_bytes);
            tmp_buffers_recv[iter]  = (int*) malloc(cur_size_bytes);
            int *cur_send           = tmp_buffers_send[iter];
            int *cur_recv           = tmp_buffers_recv[iter];
            for(b = 0; b < cur_size_bytes/sizeof(int); b++) {
                cur_send[b] = 1;
                cur_recv[b] = 0;
            }
        }
        #endif

        #if BENCHMARK_TYPE == 2
        MPI_Datatype    cur_type_send;
        MPI_Datatype    dt_send[4];
        int             blocklen_send[4];
        MPI_Aint        disp_send[4];

        MPI_Datatype    cur_type_recv;
        MPI_Datatype    dt_recv[4];
        int             blocklen_recv[4];
        MPI_Aint        disp_recv[4];

        int chunk, ierr;
        for(chunk = 0; chunk < 4; chunk++) {
            dt_send[chunk]          = MPI_BYTE;
            blocklen_send[chunk]    = cur_size_bytes/4;
            MPI_Get_address(tmp_buffers_send[chunk], &(disp_send[chunk]));

            dt_recv[chunk]          = MPI_BYTE;
            blocklen_recv[chunk]    = cur_size_bytes/4;
            MPI_Get_address(tmp_buffers_recv[chunk], &(disp_recv[chunk]));
        }

        ierr = MPI_Type_create_struct(4, blocklen_send, disp_send, dt_send, &cur_type_send);
        ierr = MPI_Type_commit(&cur_type_send);
        ierr = MPI_Type_create_struct(4, blocklen_recv, disp_recv, dt_recv, &cur_type_recv);
        ierr = MPI_Type_commit(&cur_type_recv);
        #endif
        
        double time = 0.;
        #if USE_PAPI
        if (PAPI_start_counters(papi_events, NUM_EVENTS) != PAPI_OK)
            handle_error("Starting PAPI counters failed");
        if (PAPI_read_counters(papi_cntr_values_start, NUM_EVENTS) != PAPI_OK)
            handle_error("Reading PAPI counters failed");
        #endif
        if(iMyRank == 0) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            time -= omp_get_wtime();
            for(iter = 0; iter < NUM_ITERATIONS; iter++) {
                #if BENCHMARK_TYPE == 0
                MPI_Send(tmp_buffer_send, cur_size_bytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(tmp_buffer_recv, cur_size_bytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                #elif BENCHMARK_TYPE == 1
                MPI_Send(tmp_buffers_send[iter], cur_size_bytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(tmp_buffers_recv[iter], cur_size_bytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                #elif BENCHMARK_TYPE == 2
                MPI_Send(MPI_BOTTOM, 1, cur_type_send, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(MPI_BOTTOM, 1, cur_type_recv, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                #endif
            }
            time += omp_get_wtime();
            time /= NUM_ITERATIONS;
        } else {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            time -= omp_get_wtime();
            for(iter = 0; iter < NUM_ITERATIONS; iter++) {                
                #if BENCHMARK_TYPE == 0
                MPI_Recv(tmp_buffer_recv, cur_size_bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(tmp_buffer_send, cur_size_bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                #elif BENCHMARK_TYPE == 1
                MPI_Recv(tmp_buffers_recv[iter], cur_size_bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(tmp_buffers_send[iter], cur_size_bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                #elif BENCHMARK_TYPE == 2
                MPI_Recv(MPI_BOTTOM, 1, cur_type_recv, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(MPI_BOTTOM, 1, cur_type_send, 0, 0, MPI_COMM_WORLD);
                #endif
            }
            time += omp_get_wtime();
            time /= NUM_ITERATIONS;
        }
        #if USE_PAPI
        if (PAPI_read_counters(papi_cntr_values_end, NUM_EVENTS) != PAPI_OK)
            handle_error("Reading PAPI counters failed");
        if (PAPI_stop_counters(papi_cntr_values_stop, NUM_EVENTS) != PAPI_OK)
            handle_error("Stopping PAPI counters failed");
        #endif

        // calculate cache miss ratios
        double miss_ratio_L3        = 0;
        double miss_ratio_L3_load   = 0;
        if(papi_cntr_values_end[0] != 0) {
            miss_ratio_L3       = (double) papi_cntr_values_end[1] / (double) papi_cntr_values_end[0];
            miss_ratio_L3_load  = (double) papi_cntr_values_end[2] / (double) papi_cntr_values_end[0];
        }

        if(iMyRank == 0) {
            double cur_size_kb      = cur_size_bytes/1024.0;
            double cur_micro_secs   = time*1e06;
            double cur_thoughput    = ((double)cur_size_bytes*2.0/(1e06*time));

            fprintf(stderr, "PingPong with msg_size:\t%d\t(\t%.3f\tKB) took\t%.3f\tus with a throughput of\t%.3f\tMB/s\t", cur_size_bytes, cur_size_kb, cur_micro_secs, cur_thoughput);
            fprintf(stderr, "L3_accesses\t%lld\tL3_misses\t%lld\tL3_miss_ratio\t%.3f\t", papi_cntr_values_end[0], papi_cntr_values_end[1], miss_ratio_L3);
            fprintf(stderr, "L3_load_misses\t%lld\tL3_load_miss_ratio\t%.3f\t", papi_cntr_values_end[2], miss_ratio_L3_load);
            fprintf(stderr, "PAPI_TOT_INS\t%lld\n", papi_cntr_values_end[3]);
        }

        #if BENCHMARK_TYPE == 0
        free(tmp_buffer_send);
        free(tmp_buffer_recv);
        #elif BENCHMARK_TYPE == 1
        for(iter = 0; iter < NUM_ITERATIONS; iter++) {
            free(tmp_buffers_send[iter]);
            free(tmp_buffers_recv[iter]);
        }
        #elif BENCHMARK_TYPE == 2
        for(iter = 0; iter < 4; iter++) {
            free(tmp_buffers_send[iter]);
            free(tmp_buffers_recv[iter]);
        }
        MPI_Type_free(&cur_type_send);
        MPI_Type_free(&cur_type_recv);
        #endif
    }
    
    #if BENCHMARK_TYPE == 1 || BENCHMARK_TYPE == 2
    free(tmp_buffers_send);
    free(tmp_buffers_recv);
    #endif
    
    MPI_Finalize();
    return 0;
}