//#include <mpi.h>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <omp.h>

#define MB 1024*1024

double omp_get_wtime() {
    struct timeval currentTime;
	gettimeofday(&currentTime, NULL);
	return (double)(currentTime.tv_sec) +  (1.0E-06)*(double)(currentTime.tv_usec);
}

char **split(char *str, char div, int num_args);
char *space = "\n-------------------------------------------------\n";

typedef enum disturbance_mode_t {
    compute = 0,
    memory = 1,
    communication = 2,
} disturbance_mode;

disturbance_mode d_mode     = compute;
long window_us_comp         = 1000*1000;    // default 1 sec
long window_us_pause        = 1000*50;      // default 500 ms 
long window_us_size_min     = 1000*1000;    // default 1 sec
long window_us_size_max     = 1000*3000;    // default 3 sesc
bool use_random             = false;
int rank_number             = 0;
int use_multiple_cores      = 1;
int use_ram                 = 1;
int seed                    = 0;

#ifdef DEBUB
static void DEBUG_PRINT(const char * format, ... ) {
    fprintf(stderr, "Disturbance -- R#%02d T#%02d (OS_TID:%06ld): --> ", rank_number, omp_get_thread_num(), syscall(SYS_gettid));
    va_list argptr;
    va_start(argptr, format);
    vfprintf(stderr, format, argptr);
    va_end(argptr);
}
#else
static void DEBUG_PRINT(const char * format, ... ) { }
#endif

void compute_kernel()
{
    int volatile i, j;
    int v;
    volatile int n = 50000000;
    const int length = 8;
    float a[length];
    float c[length];

    for (i = 0; i < length; i++)
    {
        a[i] = i;
        c[i] = 1.0+i*0.01;
    }

    if (window_us_pause <= 0) {
        long count = 0;
        double start_time = omp_get_wtime();
        while(true)
        {
            for (i = 0; i < n; i++) {
                int tmp = i;
                #pragma omp simd
                for (v = 0; v < length; v++) {
                    a[v]=a[v]+c[v]*tmp;
                }
            }

            // #pragma omp master
            // {
            //     count++;
            //     DEBUG_PRINT("Running step %ld\n", count);
            //     if (count % 1000 == 0) {
            //         double elapsed = omp_get_wtime() - start_time;
            //         DEBUG_PRINT("Elasped time (sec) = %f\n", elapsed);
            //     }
            // }

            usleep(10); // sleep for 10 us that OS is able to reschedule resources
        }
    } else {
        double start_time = omp_get_wtime();
        long passed_time = 0;
        int count = 0;
        while(true)
        {
            for (i = 0; i < n; i++) {
                int tmp = i;
                #pragma omp simd
                for (v = 0; v < length; v++) {
                    a[v]=a[v]+c[v]*tmp;
                }
            }

            count++;
            if(count % 10 == 0) {
                count = 0;
                passed_time = (long)((omp_get_wtime()-start_time) * 1e6);
                if(passed_time > window_us_comp)
                {
                    #pragma omp master
                    {
                        DEBUG_PRINT("passed time: %ld us - going to sleep\n", passed_time);
                    }                    
                    usleep(window_us_pause);
                    start_time = omp_get_wtime();
                }
            }
        }
    }
}

void memory_kernel()
{

    long long size_of_ram_usage = use_ram/use_multiple_cores/2;
    //printf ("Allocating %dMB of RAM\n",size_of_ram_usage*2);

    void *p = malloc(size_of_ram_usage*MB);
    void *q = malloc(size_of_ram_usage*MB);
    int i = 0;

    if (window_us_pause <= 0) {

        memset(p, 1, size_of_ram_usage*MB);
        memcpy(q, p, size_of_ram_usage*MB);

        while(true)
        {
            memset(p, i, size_of_ram_usage*MB);
            memcpy(q, p,  size_of_ram_usage*MB);
            usleep(10);

            if (i == 0)
                i = 1;
            else
                i = 0;
            
        }
        return;
    }

    double start_time = omp_get_wtime();
    long passed_time = 0;
    int count = 0;
    while(true)
    {
        memset(p, i, size_of_ram_usage*MB);
        memcpy(q, p,  size_of_ram_usage*MB);

        if (i == 0)
            i = 1;
        else
            i = 0;

        count++;
        if(count % 10 == 0) {
            count = 0;
            passed_time = (long)((omp_get_wtime()-start_time) * 1e6);
            if(passed_time > window_us_comp)
            {
                free(p);
                free(q);
                usleep(window_us_pause);
                p = malloc(size_of_ram_usage*MB);
                q = malloc(size_of_ram_usage*MB);
            }
        }
    }
}

void communication_kernel()
{
    while(true)
    {

    }
}

int main(int argc, char *argv[])
{
    int i, k;

    int iMyRank, iNumProcs;
	int provided;
	/*int requested = MPI_THREAD_MULTIPLE;
	MPI_Init_thread(&argc, &argv, requested, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);*/
    
    for (i = 1; i < argc; i++)
    {
        char **s = split(argv[i], '=', 2);

        if(strcmp(s[0], "--type") == 0)
        {
            if(strcmp(s[1], "compute") == 0)
            {
                d_mode = 0;
            } 
            else if(strcmp(s[1], "memory") == 0)
            {
                d_mode = 1;
            } 
            else if(strcmp(s[1], "communication") == 0)
            {
                d_mode = 2;
            } else
            {
                fprintf(stderr,space);
                fprintf(stderr,"--type was not set correctly!\n");
                fprintf(stderr,"please try one of the following:\n");
                fprintf(stderr,"\tcompute\n\tmemory\n\tcommunication\n");
                fprintf(stderr,space);
                continue;
            }
        }
        else if(strcmp(s[0], "--window_us_comp") == 0)
        {
            long a = atol(s[1]);
            if (a == 0 && s[1] != "0" || a < 0)
            {
                fprintf(stderr,space);
                fprintf(stderr,"--window_us_comp was not set correctly!\n");
                fprintf(stderr,"please only use positiv integers!\n");
                fprintf(stderr,space);
                continue;
            }
            window_us_comp = a;
        }
        else if(strcmp(s[0], "--window_us_size_max") == 0)
        {
            long a = atol(s[1]);
            if (a == 0 && s[1] != "0" || a < 0)
            {
                fprintf(stderr,space);
                fprintf(stderr,"--window_us_size_max was not set correctly!\n");
                fprintf(stderr,"please only use positiv integers!\n");
                fprintf(stderr,space);
                continue;
            }
            window_us_size_max = a;
        }
        else if(strcmp(s[0], "--window_us_size_min") == 0)
        {
            long a = atol(s[1]);
            if (a == 0 && s[1] != "0" || a < 0)
            {
                fprintf(stderr,space);
                fprintf(stderr,"--window_us_size_min was not set correctly!\n");
                fprintf(stderr,"please only use positiv integers!\n");
                fprintf(stderr,space);
                continue;
            }
            window_us_size_min = a;
        }
        else if(strcmp(s[0], "--window_us_pause") == 0)
        {
            long a = atol(s[1]);
            if (a == 0 && s[1] != "0")
            {
                fprintf(stderr,space);
                fprintf(stderr,"--window_us_pause was not set correctly!\n");
                fprintf(stderr,"please only use positiv integers!\n");
                fprintf(stderr,space);
                continue;
            }
            window_us_pause = a;
        }
        else if(strcmp(s[0], "--use_multiple_cores") == 0)
        {
            int a = atoi(s[1]);
            if (a == 0 && s[1] != "0" || a < 0)
            {
                fprintf(stderr,space);
                fprintf(stderr,"--use_multiple_cores was not set correctly!\n");
                fprintf(stderr,"please only use positiv integers!\n");
                fprintf(stderr,space);
                continue;
            }
            use_multiple_cores = a;
        }
        else if(strcmp(s[0], "--use_random") == 0)
        {if(strcmp(s[1], "true") == 0)
            {
                use_random = true;
            } 
            else if(strcmp(s[1], "false") == 0){
                use_random = false;
            } 
            else
            {
                fprintf(stderr,space);
                fprintf(stderr,"--use_random was not set correctly!\n");
                fprintf(stderr,"please try one of the following:\n");
                fprintf(stderr,"\ttrue\n\tfalse");
                fprintf(stderr,space);
                continue;
            }
        }
        else if(strcmp(s[0], "--rank_number") == 0)
        {
            int a = atoi(s[1]);
            if (a == 0 && s[1] != "0" || a < 0)
            {
                fprintf(stderr,space);
                fprintf(stderr,"--rank_number was not set correctly!\n");
                fprintf(stderr,"please only use positiv integers!\n");
                fprintf(stderr,space);
                continue;
            }
            rank_number = a;
        }
        else if(strcmp(s[0], "--use_ram") == 0)
        {
            int a = atoi(s[1]);
            if (a == 0 && s[1] != "0" || a < 0)
            {
                fprintf(stderr,space);
                fprintf(stderr,"--use_ram was not set correctly!\n");
                fprintf(stderr,"please only use positiv integers!\n");
                fprintf(stderr,space);
                continue;
            }
            use_ram = a;
        }
        else if(strcmp(argv[i], "--help") == 0)
        {
            fprintf(stderr,"Possible arguments are:\n");
            fprintf(stderr,"\t--type\n");
            fprintf(stderr,"\t--window_us_comp\t in micro seconds\n");
            fprintf(stderr,"\t--window_us_pause\t in micro seconds\n");
            fprintf(stderr,"\t--window_us_size_min\t in micro seconds\n");
            fprintf(stderr,"\t--window_us_size_max\t in micro seconds\n");
            fprintf(stderr,"\t--use_random\n");
            fprintf(stderr,"\t--use_multiple_cores\n");
            fprintf(stderr,"\t--rank_number\n");
            fprintf(stderr,"\t--use_ram\t  in MB\n");
            return 0;
        }
    }

    DEBUG_PRINT("--type is set to: %d\n", d_mode);
    DEBUG_PRINT("--window_us_comp is set to: %ld\n", window_us_comp);
    DEBUG_PRINT("--window_us_size_max is set to: %ld\n", window_us_size_max);
    DEBUG_PRINT("--window_us_size_min is set to: %ld\n", window_us_size_min);
    DEBUG_PRINT("--window_us_pause is set to: %ld\n", window_us_pause);
    DEBUG_PRINT("--use_multiple_cores is set to: %ld\n", use_multiple_cores);
    DEBUG_PRINT("--use_random is set to: %d\n", use_random);
    DEBUG_PRINT("--rank_number is set to: %d\n", rank_number);
    DEBUG_PRINT("--use_ram is set to: %d\n", use_ram);

    if (use_random)
    {
        seed = (rank_number+1)*42;
        srand(seed);
        double random = ((double) rand() / (RAND_MAX));
        DEBUG_PRINT("Seed = %d, random = %f\n", seed, random);

        window_us_comp = random*(window_us_size_max-window_us_size_min) + window_us_size_min;
        DEBUG_PRINT("New window comp set to: %d\n", window_us_comp);
    }


    if (window_us_comp <= 0)
    {
        DEBUG_PRINT("window_us_comp is negative or zero, program will terminate\n");
        return 0;
    }
    int num_threads = use_multiple_cores;

#pragma omp parallel num_threads(use_multiple_cores)
{
    // DEBUG_PRINT("started thread: %d\n", omp_get_thread_num());
    switch (d_mode)
    {
    case compute:
        compute_kernel();
        break;
    case memory:
        memory_kernel();
        break;
    case communication:
        communication_kernel();
        break;
    default:
        break;
    }
}

    //MPI_Finalize();
}

char **split(char *str, char div, int num_args)
{
    char **result = (char**)malloc(sizeof(char)*num_args);

    int i;
    for (i = 0; i < num_args; i++)
    {
        result[i] = (char*)malloc(strlen(str)*sizeof(char));
    }

    int j = 0;
    int k = 0;

    for(i = 0; i < strlen(str); i++)
    {
        if (str[i] == div)
        {
            j = j + 1;
            k = 0;
            if (j >= num_args) {
               return result;
            }
        }
        else
        {
            result[j][k] = str[i];
            k = k + 1;
        }
    }

    return result;
}