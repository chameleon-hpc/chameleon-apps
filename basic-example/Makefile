.PHONY: clean default run run_debug plain offload mpi

#ADD_ARGS=-DCOMPLEX=1
#ADD_ARGS=-g -O0 -rdynamic
ADD_ARGS=-g -O0
# ADD_ARGS=-g -O0 -DUSE_COMPLEX=1
#ADD_ARGS=

default: # MPI+Offloading
	$(MPICXX) -I../../src -L../../src -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu -DUSE_MPI=1 -DUSE_OFFLOADING=1 $(ADD_ARGS) -o basic-example.exe basic-example.cpp -lm -lchameleon

mpi:
	$(MPICXX) -I../../src -L../../src -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu -DUSE_MPI=1 $(ADD_ARGS) -o basic-example.exe ../../src/chameleon.cpp basic-example.cpp -lm

offload:
	clang -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu -DUSE_OFFLOADING=1 $(ADD_ARGS) -o basic-example.exe basic-example.c -lm

plain:
	clang -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu $(ADD_ARGS) -o basic-example.exe basic-example.c -lm

run-mpi:
	$(eval LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):/work/jk869269/repos/chameleon/chameleon-lib/src)
	@export LD_LIBRARY_PATH
	mpiexec.hydra -np 2 basic-example.exe

run-mpi-debug:
	$(eval LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):/work/jk869269/repos/chameleon/chameleon-lib/src)
	@export LD_LIBRARY_PATH
	LIBOMPTARGET_DEBUG=1 mpiexec.hydra -np 2 basic-example.exe

clean:
	rm -f *.exe
