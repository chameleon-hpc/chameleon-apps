.PHONY: clean default debug simulate-work run trace run-mpi-debug

#FILE_NAMES=matrix_example.cpp
FILE_NAMES=matrix_example_api.cpp
PROG=main
#MATRIX_SIZE=20
MATRIX_SIZE=300

RUN_SETTINGS=OMP_PLACES=cores OMP_PROC_BIND=spread I_MPI_PIN=1 I_MPI_PIN_DOMAIN=auto
CXX_COMPILE_FLAGS=-std=c++11
LINKER_FLAGS=-lm -lchameleon -lstdc++

ifeq ($(CXX),clang++)
    MPICXX=I_MPI_CXX=clang++ mpiicpc
    CXX_COMPILE_FLAGS+=-fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu # used with target offloading
    #CXX_COMPILE_FLAGS+=-fopenmp # used with manual API
else ifeq ($(CXX),icpc)
    MPICXX=I_MPI_CXX=icpc mpiicpc
    MPIF90=I_MPI_F90=ifort mpiifort
    CXX_COMPILE_FLAGS+=-qopenmp -qno-openmp-offload -Wno-unknown-pragmas
endif

default: 
	${MPICXX} -g -O3 -o $(PROG) $(CXX_COMPILE_FLAGS) $(FILE_NAMES) $(LINKER_FLAGS)
	
debug: 
	${MPICXX} -g -O0 -o $(PROG) $(CXX_COMPILE_FLAGS) $(FILE_NAMES) $(LINKER_FLAGS)

simulate-work:
	${MPICXX} -O3 -o $(PROG) $(CXX_COMPILE_FLAGS) -DSIMULATE_CONST_WORK=1 -DCALC_SPEEDUP=0 $(FILE_NAMES) $(LINKER_FLAGS)

run:
	$(RUN_SETTINGS) mpiexec.hydra -np 4 -genvall $(PROG) $(MATRIX_SIZE) 70 50 30 20

trace:
	$(RUN_SETTINGS) mpiexec.hydra -trace -np 2 -genvall $(PROG) $(MATRIX_SIZE)

run-mpi-debug:
	$(RUN_SETTINGS) LIBOMPTARGET_DEBUG=1 mpiexec.hydra -np 2 $(PROG) $(MATRIX_SIZE)

clean:
	rm -f $(PROG) *.o 
