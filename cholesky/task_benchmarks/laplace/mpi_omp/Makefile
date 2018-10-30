
CC = mpiicc
CFLAGS 	= -O3 -qopenmp -I ./ -mt_mpi -ipo -std=gnu99
LDFLAGS = -qopenmp -ipo

all : exec_mpi_sr_parallel \
	  exec_mpi_sr_task \
	  exec_mpi_sr_task_comm_thread \
	  exec_mpi_sr_task_v2 \
	  exec_mpi_put_parallel \
	  exec_mpi_rput_task \
	  exec_mpi_rput_task_comm_thread
OBJ = main.o common.o timer.o

exec_mpi_sr_parallel: mpi_sr_parallel.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_sr_parallel.o $(OBJ) $(LDFLAGS)
exec_mpi_sr_task: mpi_sr_task.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_sr_task.o $(OBJ) $(LDFLAGS)
exec_mpi_sr_task_comm_thread: mpi_sr_task_comm_thread.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_sr_task_comm_thread.o $(OBJ) $(LDFLAGS)
exec_mpi_sr_task_v2: mpi_sr_task_v2.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_sr_task_v2.o $(OBJ) $(LDFLAGS)

exec_mpi_put_parallel: mpi_put_parallel.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_put_parallel.o $(OBJ) $(LDFLAGS)
exec_mpi_rput_task: mpi_rput_task.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_rput_task.o $(OBJ) $(LDFLAGS)
exec_mpi_rput_task_comm_thread: mpi_rput_task_comm_thread.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_rput_task_comm_thread.o $(OBJ) $(LDFLAGS)

clean:
	rm -f *.o core.* trace.log
