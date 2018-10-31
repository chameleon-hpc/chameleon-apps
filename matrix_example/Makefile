.PHONY: clean default debug run trace run-mpi-debug

PROG=main
#MATRIX_SIZE=20
MATRIX_SIZE=200

RUN_SETTINGS=OMP_PLACES=cores OMP_PROC_BIND=spread I_MPI_PIN=1 I_MPI_PIN_DOMAIN=auto

default: 
	${MPICXX} -O3 -o main -std=c++11 -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu matrix_example.cpp -lm -lchameleon -lstdc++ 
	
debug: 
	${MPICXX} -g -O0 -o main -std=c++11 -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu matrix_example.cpp -lm -lchameleon -lstdc++ 

run:
	$(RUN_SETTINGS) mpiexec.hydra -np 4 -genvall $(PROG) $(MATRIX_SIZE) 70 50 30 20

trace:
	$(RUN_SETTINGS) mpiexec.hydra -trace -np 2 -genvall $(PROG) $(MATRIX_SIZE)

run-mpi-debug:
	$(RUN_SETTINGS) LIBOMPTARGET_DEBUG=1 mpiexec.hydra -np 2 $(PROG) $(MATRIX_SIZE)

clean:
	rm -f main *.o 
