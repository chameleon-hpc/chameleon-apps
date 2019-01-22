.PHONY: clean default debug 

FILE_NAMES=fortran_testcase.f90
PROG=main

RUN_SETTINGS=OMP_PLACES=cores OMP_PROC_BIND=spread I_MPI_PIN=1 I_MPI_PIN_DOMAIN=auto
LINKER_FLAGS=-lm -lchameleon


# ===== Intel compiler
MPIF90=I_MPI_F90=ifort mpiifort
FLAGS_OPENMP=-qopenmp 
#-qno-openmp-offload -Wno-unknown-pragmas
#FLAGS_OPENMP=-qopenmp -qopenmp-offload=host -Wno-unknown-pragmas

default: 
	${MPIF90} -g -O3 -o $(PROG) $(FLAGS_OPENMP) $(FILE_NAMES) $(LINKER_FLAGS)
	
debug: 
	${MPIF90} -g -O0 -o $(PROG) $(FLAGS_OPENMP) $(FILE_NAMES) $(LINKER_FLAGS)

clean:
	rm -f $(PROG) *.o 
