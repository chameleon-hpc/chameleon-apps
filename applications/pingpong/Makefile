ADDITIONAL_COMPILE_FLAGS ?=

compile:
	I_MPI_CC=icc mpiicc -qopenmp -mcmodel=medium -O3 ${ADDITIONAL_COMPILE_FLAGS} -o pingpong.exe pingpong.c -lm -lpapi

run-sm-socket:
	I_MPI_DEBUG=5 I_MPI_FABRICS="shm:tmi" I_MPI_PIN=1 I_MPI_PIN_DOMAIN=socket mpiexec.hydra -np 2 -hosts localhost pingpong.exe

run-sm-cache2:
	I_MPI_DEBUG=5 I_MPI_FABRICS="shm:tmi" I_MPI_PIN=1 I_MPI_PIN_DOMAIN=cache2 mpiexec.hydra -np 2 -hosts localhost pingpong.exe

run-dm:
	I_MPI_DEBUG=5 I_MPI_FABRICS="shm:tmi" I_MPI_PIN=1 I_MPI_PIN_DOMAIN=auto mpiexec -np 2 pingpong.exe
