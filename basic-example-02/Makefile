.PHONY: clean default run run_debug plain

PROG=basic-example-02

#ADD_ARGS=
ADD_ARGS=-g -O0
#ADD_ARGS=-g -O0 -DPRINT_DATA_VERBOSE=1
#ADD_ARGS=-g -O0 -rdynamic

RUN_SETTINGS=OMP_PLACES=cores OMP_PROC_BIND=spread I_MPI_PIN=1 I_MPI_PIN_DOMAIN=auto

default: # MPI+Offloading
	$(MPICXX) -I../../src -L../../src -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu -DUSE_MPI=1 $(ADD_ARGS) -o $(PROG).exe $(PROG).cpp -lm -lchameleon

run:
	$(RUN_SETTINGS) mpiexec.hydra -np 2 -genvall $(PROG).exe

trace:
	$(RUN_SETTINGS) mpiexec.hydra -trace -np 2 -genvall $(PROG).exe

run-mpi-debug:
	$(RUN_SETTINGS) LIBOMPTARGET_DEBUG=1 mpiexec.hydra -np 2 $(PROG).exe

clean:
	rm -f *.exe
