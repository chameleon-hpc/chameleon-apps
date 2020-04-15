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
#include <mpi.h>
#include <signal.h>

#ifdef TRACE
#include "VT.h"
#endif

#define MB 1024*1024

double omp_get_wtime() {
    struct timeval currentTime;
	gettimeofday(&currentTime, NULL);
	return (double)(currentTime.tv_sec) +  (1.0E-06)*(double)(currentTime.tv_usec);
}

char **split(char *str, char *div, int num_args, int *length);
int get_integer(char *str, char* id, int normal);
void communicate(int* data, int transmitter, int reciver);

char *space = "\n-------------------------------------------------\n";

typedef enum disturbance_mode_t {
    compute = 0,
    memory = 1,
    communication = 2,
} disturbance_mode;

typedef enum comunication_mode_t {
    pingpong = 0,
    roundtrip = 1,
} comunication_mode;

disturbance_mode d_mode     = compute;
comunication_mode c_mode    = roundtrip;

long window_us_comp         = 1000*1000;    // default 1 sec
long window_us_pause        = 1000*500;     // default 500 ms 
// long window_us_pause        = -1;        // constant work
long window_us_size_min     = 1000*1000;    // default 1 sec
long window_us_size_max     = 1000*3000;    // default 3 sesc
bool use_random             = false;
int rank_number             = -1;
int use_multiple_cores      = 1;
int use_ram                 = 1;
int *ranks_to_disturb;
int com_size                = 1000;
int num_partner             = 0;
int seed                    = 0;
int abort_program           = 0;


int iNumProcs;

#ifdef DEBUG
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

void sig_handler(int signo)
{
    if (signo == SIGINT || signo == SIGABRT || signo == SIGTERM) {
        DEBUG_PRINT("received signo=%d\n", signo);
        abort_program = 1;
    }
}

void compute_kernel()
{
    int volatile i, j;
    int v;
    volatile int n = 5000000;
    const int length = 8;
    float a[length];
    float c[length];
    int ierr;

    for (i = 0; i < length; i++)
    {
        a[i] = i;
        c[i] = 1.0+i*0.01;
    }

    if (window_us_pause <= 0) {
        // long count = 0;
        double start_time = omp_get_wtime();
#ifdef TRACE
        static int event_compute = -1;
        const char *event_compute_name = "disturbance_compute";
        if(event_compute == -1) {
            ierr = VT_funcdef(event_compute_name, VT_NOCLASS, &event_compute);
        }
        VT_begin(event_compute);
#endif
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
#ifdef TRACE
            VT_end(event_compute);
#endif
            if (abort_program) {
                break;
            }
#ifdef TRACE
            VT_begin(event_compute);
#endif
        }
    } else {
        double start_time = omp_get_wtime();
        long passed_time = 0;
#ifdef TRACE
        static int event_compute = -1;
        const char *event_compute_name = "disturbance_compute";
        if(event_compute == -1) {
            ierr = VT_funcdef(event_compute_name, VT_NOCLASS, &event_compute);
        }
        VT_begin(event_compute);
#endif
        while(true)
        {
            for (i = 0; i < n; i++) {
                int tmp = i;
                #pragma omp simd
                for (v = 0; v < length; v++) {
                    a[v]=a[v]+c[v]*tmp;
                }
            }

            passed_time = (long)((omp_get_wtime()-start_time) * 1e6);
            if(passed_time > window_us_comp)
            {
#ifdef TRACE
                VT_end(event_compute);
#endif
#ifdef DEBUG
                #pragma omp master
                {
                    DEBUG_PRINT("passed time: %ld us - going to sleep\n", passed_time);
                }
#endif
                usleep(window_us_pause);
                if (abort_program) {
                    break;
                }
                start_time = omp_get_wtime();
#ifdef TRACE
                VT_begin(event_compute);
#endif
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
                start_time = omp_get_wtime();
                p = malloc(size_of_ram_usage*MB);
                q = malloc(size_of_ram_usage*MB);
            }
        }
    }
}

void communication_kernel()
{
    if (num_partner < 2) 
        return;

    int i ,id;
    int *data = (int*)malloc(sizeof(int)*com_size*MB);

    int total_num_partner = num_partner;

    for (i = 0; i < num_partner; i++) {
        if (ranks_to_disturb[i] == rank_number)
            id = i;

        
    }

    int previous_rank_id = (i - 1)%2;
    int target_rank_id = (i + 1)%2;

    if (c_mode == pingpong) {
        if (id >= 2)
            return;
        total_num_partner = 2;
        if (previous_rank_id >= 2)
            previous_rank_id = 0;
    }

    MPI_Status status;

    if (id == 0) {
        memset(data, rank_number, com_size*MB);
        MPI_Send(&data,com_size*MB, MPI_INT, ranks_to_disturb[target_rank_id], 1, MPI_COMM_WORLD);
    }

    if (window_us_pause == 0) {
        while (true)
        {
            MPI_Recv(&data,com_size,MPI_INT,ranks_to_disturb[previous_rank_id],1,MPI_COMM_WORLD,&status);
            memset(data, rank_number, com_size*MB);
            MPI_Send(&data,com_size*MB, MPI_INT, ranks_to_disturb[target_rank_id], 1, MPI_COMM_WORLD);
            DEBUG_PRINT("Rank %d\t Recived Message from %d, send to %d", rank_number, ranks_to_disturb[previous_rank_id], ranks_to_disturb[target_rank_id]);
            
            if (abort_program) {
                break;
            }
        }
        
        return;
    }
    
    double start_time = 0;
    long passed_time = 0;

    if (id == 0) {
        start_time = omp_get_wtime();
    }

    while (true)
    {

        MPI_Recv(&data,com_size,MPI_INT,ranks_to_disturb[previous_rank_id],1,MPI_COMM_WORLD,&status);
        memset(data, rank_number, com_size*MB);

        if (id == 0) {
            passed_time = (long)((omp_get_wtime()-start_time) * 1e6);
            if(passed_time > window_us_comp)
            {
                usleep(window_us_pause);
                start_time = omp_get_wtime();
            }
        }
        
        MPI_Send(&data,com_size*MB, MPI_INT, ranks_to_disturb[target_rank_id], 1, MPI_COMM_WORLD);
        DEBUG_PRINT("Rank %d\t Recived Message from %d, send to %d", rank_number, ranks_to_disturb[previous_rank_id], ranks_to_disturb[target_rank_id]);
        if (abort_program) {
            break;
        }
    }
    
}

int main(int argc, char *argv[])
{
    // catch signals to allow controlled way to end application
    if (signal(SIGINT, sig_handler) == SIG_ERR)
        printf("\ncan't catch SIGINT\n");
    if (signal(SIGTERM, sig_handler) == SIG_ERR)
        printf("\ncan't catch SIGTERM\n");
    if (signal(SIGABRT, sig_handler) == SIG_ERR)
        printf("\ncan't catch SIGABRT\n");

    int i, k;
    int *length = (int*)malloc(sizeof(int));
    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_number);

#ifdef TRACE
    int ierr;
    static int event_main = -1;
    const char *event_main_name = "disturbance_main";
    if(event_main == -1) {
        ierr = VT_funcdef(event_main_name, VT_NOCLASS, &event_main);
    }
    VT_begin(event_main);
#endif
    
    for (i = 1; i < argc; i++)
    {
        // DEBUG_PRINT("Param %d: Len=%d - %s\n", i, strlen(argv[i]), argv[i]);
        char **s = split(argv[i], "=", 2, length);
        // DEBUG_PRINT("Param %d: s[0]=%s s[1]=%s\n", i, s[0], s[1]);

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
            window_us_comp = get_integer(s[1], s[0], window_us_comp);
        }
        else if(strcmp(s[0], "--window_us_size_max") == 0)
        {
            window_us_size_max = get_integer(s[1], s[0], window_us_size_max);
        }
        else if(strcmp(s[0], "--window_us_size_min") == 0)
        {
            window_us_size_min = get_integer(s[1], s[0], window_us_size_min);
        }
        else if(strcmp(s[0], "--window_us_pause") == 0)
        {
            window_us_pause = get_integer(s[1], s[0], window_us_pause);
        }
        else if(strcmp(s[0], "--use_multiple_cores") == 0)
        {
            use_multiple_cores = get_integer(s[1], s[0], use_multiple_cores);
        }
        else if(strcmp(s[0], "--use_random") == 0)
        {
            if(strcmp(s[1], "true") == 0)
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
            rank_number = get_integer(s[1], s[0], rank_number);
        }
        else if(strcmp(s[0], "--use_ram") == 0)
        {
            use_ram = get_integer(s[1], s[0], use_ram);
        }
        else if(strcmp(s[0], "--ranks_to_disturb") == 0)
        {
            //char* tmp = split(s[1],'(',iNumProcs)[1];
            //tmp = split(tmp,')',iNumProcs)[0];
            //char** partner_list = split(tmp,',',iNumProcs);  
            char** partner_list = split(s[1], ",", iNumProcs, length);  
            num_partner = *length;

            int *tmp_com_partner = (int*)malloc(sizeof(int)*num_partner);

            for (k = 0; k < (*length); k++) {
                if(strcmp(partner_list[k], "") == 1) {
                    if(atoi(partner_list[k]) <= num_partner){
                        tmp_com_partner[k] = atoi(partner_list[k]);
                    }
                }
            }

            ranks_to_disturb = (int*)malloc(sizeof(int)*num_partner);
            char* partners = (char*)malloc(sizeof(char)*num_partner*2);

            for (k = 0; k < (*length); k++) {
                ranks_to_disturb[k] = tmp_com_partner[k];
                partners[2*k] =  (char)(tmp_com_partner[k] + 48);
                if(2*k >=  (*length))
                    partners[2*k + 1] = ',';
                else partners[2*k + 1] = 0;
            }
            
            if(iNumProcs > 1)
                for (i = 0; i < num_partner; i++) {
                    if (ranks_to_disturb[i] >= iNumProcs || ranks_to_disturb[i] < 0) {
                        printf("--ranks_to_disturb %d is not set correctly\n", i);
                        printf("please use only available ranks\n");
                        return 0;
                    }
                }


            printf("--ranks_to_disturb are set to: %s\t number of ranks to disturb: %d\n", partners, *length);

            free(tmp_com_partner);
        }
        else if(strcmp(s[0], "--com_type") == 0)
        {
            if(strcmp(s[1], "pingpong") == 0)
            {
                c_mode = 0;
            } 
            else if(strcmp(s[1], "roundtrip") == 0)
            {
                c_mode = 1;
            } else
            {
                printf(space);
                printf("--type was not set correctly!\n");
                printf("please try one of the following:\n");
                printf("\tpingpong\n\troundtrip");
                printf(space);
                continue;
            }
            
            printf("--com_type is set to: %s\n", s[1]);
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
            fprintf(stderr,"\t--com_type\n");
            fprintf(stderr,"\t--ranks_to_disturb\t in form 1,2,3\n");
            fprintf(stderr,"\t--com_size\t in MB\n");
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
    DEBUG_PRINT("--com_type is set to: %d\n", c_mode);
    DEBUG_PRINT("--ranks_to_disturb is set to: %d\n", ranks_to_disturb);
    DEBUG_PRINT("--com_size is set to: %d\n", com_size);

    if (use_random)
    {
        seed = (rank_number+1)*42;
        srand(seed);
        double random = ((double) rand() / (RAND_MAX));
        DEBUG_PRINT("Seed = %d, random = %f\n", seed, random);

        window_us_comp = random*(window_us_size_max-window_us_size_min) + window_us_size_min;
        DEBUG_PRINT("New window comp set to: %d\n", window_us_comp);
    }

    bool not_disturb = true;
    for (int i = 0; i < num_partner; i++) {
        if (rank_number == ranks_to_disturb[i])
        {
            not_disturb = false;
        }
    }

    printf(space);
    if (not_disturb) {
        printf("Rank %d will not be disturbed\nExit disturbance", rank_number);
        printf(space);
        MPI_Finalize();
        return 1;
    }
    else 
    {
        printf("Rank %d will be disturbed\n", rank_number);
    }
    printf(space);
    
    
    if (window_us_comp <= 0)
    {
        DEBUG_PRINT("window_us_comp is negative or zero, program will terminate\n");
        return 0;
    }
    int num_threads = use_multiple_cores;
    free(length);

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
#ifdef TRACE
    VT_end(event_main);
#endif

    MPI_Finalize();
}



char **split(char *str, char *div, int num_args, int *length)
{
    char **result = (char**)malloc(sizeof(char)*num_args);
    
    char * token = strtok(str, div);
    int count = 0;
    // loop through the string to extract all other tokens
    while( token != NULL) {
        int tmp_size = strlen(token);
        result[count] = (char*) malloc(sizeof(char)*tmp_size);
        strcpy(result[count], token);
        //printf( "%sEND\n", token); //printing each token
        token = strtok(NULL, div);
        count++;
        if(count >= num_args) {
            break;    
        }            
    }

    *length = count;
    return result;
}

int get_integer(char *str, char *id, int normal)
{
    int a = atoi(str);
    if (a == 0 && str != "0" || a < 0)
    {
        printf(space);
        printf("%s was not set correctly!\n", id);
        printf("please only use positiv integers!");
        printf(space);
        return normal;
    }
    printf("%s is set to: %d\n", id, a);
    return a;
}