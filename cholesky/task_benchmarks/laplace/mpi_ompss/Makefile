
CC = mpiicc
CFLAGS 	= -O3 -fopenmp -I ./ -mt_mpi -std=gnu99 --ompss
LDFLAGS = -fopenmp --ompss

all : exec_mpi_sr_task \
	  exec_mpi_sr_task_v2 \
	  exec_mpi_rput_task
OBJ = main.o common.o timer.o

exec_mpi_sr_task: mpi_sr_task.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_sr_task.o $(OBJ) $(LDFLAGS)
exec_mpi_sr_task_v2: mpi_sr_task_v2.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_sr_task_v2.o $(OBJ) $(LDFLAGS)

exec_mpi_rput_task: mpi_rput_task.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_rput_task.o $(OBJ) $(LDFLAGS)

clean:
	rm -f *.o core.* trace.log
clean-all:
	rm -f *.o core.* trace.log exec_* Laplace_*

