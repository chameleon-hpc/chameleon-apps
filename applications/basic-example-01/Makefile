.PHONY: clean default run run_debug plain

PROG=basic-example-01

#ADD_ARGS=
ADD_ARGS=-g -O0
#ADD_ARGS=-g -O0 -rdynamic

default: # MPI+Offloading
	$(MPICXX) -I../../src -L../../src -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu -DUSE_MPI=1 $(ADD_ARGS) -o $(PROG).exe $(PROG).cpp -lm -lchameleon

plain:
	clang -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu -DUSE_MPI=0 $(ADD_ARGS) -o $(PROG).exe $(PROG).cpp -lm -lstdc++

run-mpi:
	mpiexec.hydra -np 2 $(PROG).exe

run-mpi-debug:
	LIBOMPTARGET_DEBUG=1 mpiexec.hydra -np 2 $(PROG).exe

clean:
	rm -f *.exe
