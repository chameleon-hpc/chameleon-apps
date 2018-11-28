
ifneq ($(TARGET),)
include ../$(TARGET).def
else
TARGET=generic
CFLAGS=-fopenmp
LDFLAGS=-fopenmp
endif

all : ch_ompss_perrank ch_ompss_perrank_extrae ch_ompss_perrank_timing
ch_ompss_perrank: ch_ompss.c ch_common.c
	$(CC) $(CFLAGS) -g -gdwarf-3 -o $@ ch_ompss.c --keep-all-files ch_common.c $(LDFLAGS)

ch_ompss_perrank_extrae: ch_ompss.c ch_common.c
	$(CC) $(CFLAGS) -g -gdwarf-3 -o $@ ch_ompss.c --keep-all-files ch_common.c $(LDFLAGS) -DUSE_EXTRAE $(EXTRAE_FLAGS)

ch_ompss_perrank_timing: ch_ompss.c ch_common.c
	$(CC) $(CFLAGS) -g -gdwarf-3 -o $@ ch_ompss.c --keep-all-files ch_common.c $(LDFLAGS) -DUSE_TIMING

clean:
	rm -f *.o  ch_ompss ch_ompss_*
