# compilation and linking
CC      		= mpiicc
# CFLAGS  		= -O3 -I./ -std=c99  -mt_mpi -qopenmp -ipo -DDEBUG
CFLAGS  		= -O0 -g -I./ -std=c99  -mt_mpi -qopenmp -ipo -DDEBUG
LDFLAGS 		= -std=c99 -lrt -lpthread -lm -mt_mpi -qopenmp -ipo -DDEBUG 
LDFLAGS 		+= -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl

# settings for execution and linking
RUN_SETTINGS	= OMP_PLACES=cores OMP_PROC_BIND=spread I_MPI_PIN=1 I_MPI_PIN_DOMAIN=auto

# application parameters
PROG			?= exec_mpi_sr_task
M_SIZE 			?= 5000
BLOCK_SIZE 		?= 200
P_CHECK			?= 1

all : 	exec_mpi_sr_parallel \
	  	exec_mpi_sr_task \
	  	exec_mpi_sr_task_comm_thread \
	  	exec_mpi_put_parallel \
	  	exec_mpi_rput_task \
	  	exec_mpi_rput_task_comm_thread \
		exec_omp_only

#OBJ = main.o common.o omp_task.o timer.o
OBJ = main.o common.o omp_task.o

exec_mpi_sr_parallel: mpi_sr_parallel.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_sr_parallel.o $(OBJ) $(LDFLAGS)
exec_mpi_sr_task: mpi_sr_task.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_sr_task.o $(OBJ) $(LDFLAGS)
exec_mpi_sr_task_comm_thread: mpi_sr_task_comm_thread.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_sr_task_comm_thread.o $(OBJ) $(LDFLAGS)
exec_mpi_put_parallel: mpi_put_parallel.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_put_parallel.o $(OBJ) $(LDFLAGS)
exec_mpi_rput_task: mpi_rput_task.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_rput_task.o $(OBJ) $(LDFLAGS)
exec_mpi_rput_task_comm_thread: mpi_rput_task_comm_thread.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_rput_task_comm_thread.o $(OBJ) $(LDFLAGS)
exec_omp_only:
	$(CC) $(CFLAGS) -DOMP_ONLY -c -o main.o main.c
	$(CC) $(CFLAGS) -DOMP_ONLY -c -o common.o common.c
	$(CC) $(CFLAGS) -DOMP_ONLY -c -o omp_task.o omp_task.c

	$(CC) $(CFLAGS) -DOMP_ONLY -o $@ $(OBJ) $(LDFLAGS)

run: $(PROG)
	$(RUN_SETTINGS) mpiexec.hydra -np 2 -genvall ./$(PROG) $(M_SIZE) $(BLOCK_SIZE) $(P_CHECK)

clean:
	rm -f *.o core.* trace.log exec_mpi* exec_omp*
