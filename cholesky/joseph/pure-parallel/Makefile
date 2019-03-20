
ifneq ($(TARGET),)
include ../flags_$(TARGET).def
else
TARGET=generic
CFLAGS=-fopenmp
LDFLAGS=-fopenmp
endif

# parameters and settings for execution
RUN_SETTINGS    = OMP_PLACES=cores OMP_PROC_BIND=spread I_MPI_PIN=1 I_MPI_PIN_DOMAIN=auto
NUM_RANKS       ?= 2 
PROG_EXE        ?= ch_$(TARGET)_par
MATRIX_SIZE     ?= 15360
BLOCK_SIZE      ?= 512
BOOL_CHECK      ?= 0
# CFLAGS          += -DDEBUG -DTRACE -I$(VT_ROOT)/include
# LDFLAGS         += -DDEBUG -trace

all : ch_$(TARGET)_par

ch_$(TARGET)_par: ch_ompss.c ../common/ch_common.c Makefile
	$(CC) $(CFLAGS) -o $@ ch_ompss.c -g ../common/ch_common.c $(LDFLAGS)

ch_$(TARGET)_par_timing: ch_ompss.c ../common/ch_common.c Makefile
	$(CC) $(CFLAGS) -o $@ ch_ompss.c -g ../common/ch_common.c $(LDFLAGS) -DUSE_TIMING

run:
	$(RUN_SETTINGS) mpiexec.hydra -np $(NUM_RANKS) -genvall ./$(PROG_EXE) $(MATRIX_SIZE) $(BLOCK_SIZE) $(BOOL_CHECK)

clean:
	rm -f *.o ch_$(TARGET)_par ch_$(TARGET)_par_timing

cleanall: 
	rm -f *.o ch_*_par ch_*_par_extrae ch_*_par_timing
