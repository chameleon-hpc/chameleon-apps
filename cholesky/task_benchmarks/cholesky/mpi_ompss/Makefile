
CC      = mpiicc
CFLAGS  = -O3 -I./ -std=gnu99  -mt_mpi -qopenmp --ompss
LDFLAGS = -std=gnu99 -lrt -lpthread -lm -mt_mpi -qopenmp --ompss

LDFLAGS += -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl

all : exec_mpi_sr_task \
	  exec_mpi_rput_task
OBJ = main.o common.o ompss_task.o timer.o

exec_mpi_sr_task: mpi_sr_task.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_sr_task.o $(OBJ) $(LDFLAGS)
exec_mpi_rput_task: mpi_rput_task.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ mpi_rput_task.o $(OBJ) $(LDFLAGS)

clean:
	rm -f *.o core.* trace.log
