
ifneq ($(TARGET),)
include ../flags_$(TARGET).def
else
TARGET=generic
CFLAGS=-fopenmp
LDFLAGS=-fopenmp
endif

# parameters and settings for execution
NUM_RANKS       ?= 2 
PROG_EXE        ?= ch-$(TARGET)-single-timing
MATRIX_SIZE     ?= 15360
BLOCK_SIZE      ?= 512
BOOL_CHECK      ?= 0
RUN_SETTINGS    ?= OMP_PLACES=cores OMP_PROC_BIND=spread I_MPI_PIN=1 I_MPI_PIN_DOMAIN=auto
MPI_EXEC_CMD    ?= mpiexec.hydra -np $(NUM_RANKS)
#CFLAGS          += -DTRACE -I$(VT_ROOT)/include
#LDFLAGS         += -trace

.PHONY: ch-$(TARGET)-single ch-$(TARGET)-single-noyield ch-$(TARGET)-single-timing clean cleanall

all : ch-$(TARGET)-single ch-$(TARGET)-single-noyield ch-$(TARGET)-single-timing

ch-$(TARGET)-single:
	$(CC) $(CFLAGS) -g -o $@ ch_ompss.c ../common/ch_common.c $(LDFLAGS)

ch-$(TARGET)-single-noyield:
	$(CC) $(CFLAGS) -g -o $@ ch_ompss.c ../common/ch_common.c $(LDFLAGS) -DDISABLE_TASKYIELD -DUSE_TIMING

ch-$(TARGET)-single-timing:
	$(CC) $(CFLAGS) -g -o $@ ch_ompss.c ../common/ch_common.c $(LDFLAGS) -DUSE_TIMING

run:
	$(RUN_SETTINGS) ${MPI_EXEC_CMD} ./$(PROG_EXE) $(MATRIX_SIZE) $(BLOCK_SIZE) $(BOOL_CHECK)

trace:
	$(RUN_SETTINGS) ${MPI_EXEC_CMD} -trace ./$(PROG_EXE) $(MATRIX_SIZE) $(BLOCK_SIZE) $(BOOL_CHECK)

clean:
	rm -f *.o ch-$(TARGET)-single ch-$(TARGET)-single-noyield ch-$(TARGET)-single-timing

cleanall: 
	rm -f *.o ch-*-single ch-*-single-noyield ch-*-single-timing
