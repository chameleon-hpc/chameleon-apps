
ifneq ($(TARGET),)
include ../flags_$(TARGET).def
else
TARGET=generic
CFLAGS=-fopenmp
LDFLAGS=-fopenmp
endif

all : ch_$(TARGET)_single ch_$(TARGET)_single_noyield ch_$(TARGET)_single_timing Makefile
ch_$(TARGET)_single: ch_ompss.c ch_common.c Makefile
	$(CC) $(CFLAGS) -o $@ ch_ompss.c ch_common.c $(LDFLAGS)

ch_$(TARGET)_single_noyield: ch_ompss.c ch_common.c Makefile
	$(CC) $(CFLAGS) -DDISABLE_TASKYIELD -o $@ ch_ompss.c ch_common.c $(LDFLAGS)

ch_$(TARGET)_single_timing: ch_ompss.c ch_common.c Makefile
	$(CC) $(CFLAGS) -DDISABLE_TASKYIELD -o $@ ch_ompss.c ch_common.c $(LDFLAGS) -DUSE_TIMING

clean:
	rm -f *.o ch_$(TARGET)_single ch_$(TARGET)_single_noyield ch_$(TARGET)_single_timing

cleanall: 
	rm -f *.o ch_*_single ch_*_single_noyield ch_*_single_timing
