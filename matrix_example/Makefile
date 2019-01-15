.PHONY: clean default debug simulate-work run trace run-mpi-debug

#FILE_NAMES=matrix_example.cpp
FILE_NAMES=matrix_example_api.cpp
PROG=main
#MATRIX_SIZE=20
MATRIX_SIZE=200

RUN_SETTINGS=OMP_PLACES=cores OMP_PROC_BIND=spread I_MPI_PIN=1 I_MPI_PIN_DOMAIN=auto

MPICXX=I_MPI_CXX=clang++ mpiicpc
FLAGS_OPENMP=-fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu
#MPICXX=mpiicpc
#FLAGS_OPENMP=-qopenmp

default: 
	${MPICXX} -O3 -o $(PROG) -std=c++11 $(FLAGS_OPENMP) $(FILE_NAMES) -lm -lchameleon -lstdc++ 
	
debug: 
	${MPICXX} -g -O0 -o $(PROG) -std=c++11 $(FLAGS_OPENMP) $(FILE_NAMES) -lm -lchameleon -lstdc++ 

simulate-work:
	${MPICXX} -O3 -o $(PROG) -std=c++11 $(FLAGS_OPENMP) -DSIMULATE_CONST_WORK=1 -DCALC_SPEEDUP=0 $(FILE_NAMES) -lm -lchameleon -lstdc++ 

run:
	$(RUN_SETTINGS) mpiexec.hydra -np 4 -genvall $(PROG) $(MATRIX_SIZE) 70 50 30 20

trace:
	$(RUN_SETTINGS) mpiexec.hydra -trace -np 2 -genvall $(PROG) $(MATRIX_SIZE)

run-mpi-debug:
	$(RUN_SETTINGS) LIBOMPTARGET_DEBUG=1 mpiexec.hydra -np 2 $(PROG) $(MATRIX_SIZE)

clean:
	rm -f $(PROG) *.o 
