.PHONY: clean default debug simulate-work run trace run-mpi-debug run-paper-experiments

FILE_NAMES ?= matrix_example_api.cpp
PROG ?= matrix_app.exe
MATRIX_SIZE ?= 300
ITERATIVE_VERSION ?= 1
NUM_ITERATIONS ?= 2
NUM_REPETITIONS ?= 1
PARALLEL_INIT ?= 1
COMPILE_CHAMELEON ?= 1
COMPILE_TASKING ?= 1
USE_REPLICATION ?= 0
USE_EXTERNAL_CALLBACK ?= 0
USE_LIKWID ?= 0

ifneq ($(TARGET),)
include ../../flags_$(TARGET).def
else
include ../../flags_claix_intel.def
endif
include ../../flags_replace.def

NUM_TASKS ?= $(shell echo $$(( $(OMP_NUM_THREADS) * 100 )))

APP_COMPILE_FLAGS=-std=c++11 $(CH_OMP_FLAGS) -DITERATIVE_VERSION=${ITERATIVE_VERSION} -DNUM_ITERATIONS=${NUM_ITERATIONS} -DPARALLEL_INIT=${PARALLEL_INIT} -DCOMPILE_CHAMELEON=${COMPILE_CHAMELEON} -DCOMPILE_TASKING=${COMPILE_TASKING} -DNUM_REPETITIONS=${NUM_REPETITIONS} -DUSE_REPLICATION=${USE_REPLICATION} -DUSE_EXTERNAL_CALLBACK=${USE_EXTERNAL_CALLBACK}
APP_LINKER_FLAGS=-lm -lstdc++

ifeq (${COMPILE_CHAMELEON}, 1)
APP_LINKER_FLAGS += -lchameleon
endif

# default is release
FLAGS_OPTIMIZATION=-g -O3

default: release

release: FLAGS_OPTIMIZATION=-g -O3
release: generic

debug: FLAGS_OPTIMIZATION=-g -O0
debug: generic

simulate-work: FLAGS_OPTIMIZATION=-g -O3 -DSIMULATE_CONST_WORK=1 -DCALC_SPEEDUP=0
simulate-work: generic

likwid: LIKWID_LIB=$(shell echo $$(dirname $$(which likwid-perfctr)))/../lib/
likwid: LIKWID_INCLUDE=$(shell echo $$(dirname $$(which likwid-perfctr)))/../include/
likwid: FLAGS_OPTIMIZATION=-g -O3
likwid: APP_COMPILE_FLAGS+= -DUSE_LIKWID=1
likwid: 
	${MPICXX}  $(FLAGS_OPTIMIZATION) -DLIKWID_PERFMON -L${LIKWID_LIB} -I${LIKWID_INCLUDE} -o $(PROG) $(APP_COMPILE_FLAGS) $(FILE_NAMES) $(APP_LINKER_FLAGS) -llikwid

generic: clean
	${MPICXX}  $(FLAGS_OPTIMIZATION) -o $(PROG) $(APP_COMPILE_FLAGS) $(FILE_NAMES) $(APP_LINKER_FLAGS)

run:
	$(RUN_SETTINGS) mpiexec.hydra -np 1 -genvall no_numa_balancing ./$(PROG) $(MATRIX_SIZE) 120

run-batch:
	$(RUN_SETTINGS_SLURM) ${MPIEXEC} ${FLAGS_MPI_BATCH} ${MPI_EXPORT_VARS_SLURM} ./$(PROG) ${MATRIX_SIZE} ${NUM_TASKS} ${NUM_TASKS} ${NUM_TASKS}

clean:
	rm -f $(PROG) *.o
