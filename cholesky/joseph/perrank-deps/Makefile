
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
PROG_EXE        ?= ch_ompss_perrank
MATRIX_SIZE     ?= 5120
BLOCK_SIZE      ?= 512
BOOL_CHECK      ?= 1
# CFLAGS          += -DDEBUG
# LDFLAGS         += -DDEBUG

all : ch_ompss_perrank ch_ompss_perrank_timing

ch_ompss_perrank: ch_ompss.c ../common/ch_common.c
	$(CC) $(CFLAGS) -g -gdwarf-3 -o $@ ch_ompss.c --keep-all-files ../common/ch_common.c $(LDFLAGS)

ch_ompss_perrank_extrae: ch_ompss.c ../common/ch_common.c
	$(CC) $(CFLAGS) -g -gdwarf-3 -o $@ ch_ompss.c --keep-all-files ../common/ch_common.c $(LDFLAGS) -DUSE_EXTRAE $(EXTRAE_FLAGS)

ch_ompss_perrank_timing: ch_ompss.c ../common/ch_common.c
	$(CC) $(CFLAGS) -g -gdwarf-3 -o $@ ch_ompss.c --keep-all-files ../common/ch_common.c $(LDFLAGS) -DUSE_TIMING

run:
	$(RUN_SETTINGS) mpiexec -np $(NUM_RANKS) -genvall ./$(PROG_EXE) $(MATRIX_SIZE) $(BLOCK_SIZE) $(BOOL_CHECK)

clean:
	rm -f *.o  ch_ompss ch_ompss_*
