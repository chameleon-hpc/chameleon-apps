TOOL_NAME=tool

release:
	$(CC) -O3 -shared -fPIC -std=gnu++11 -o $(TOOL_NAME)_UNI.so $(TOOL_NAME)_UNI.cpp
	$(CC) -O3 -shared -fPIC -std=gnu++11 -o $(TOOL_NAME)_NU.so $(TOOL_NAME)_NU.cpp

trace:
	$(CC) -O3 -shared -fPIC -o $(TOOL_NAME)_UNI.so $(TOOL_NAME)_UNI.cpp -DTRACE -I$(VT_ROOT)/include -trace
	$(CC) -O3 -shared -fPIC -o $(TOOL_NAME)_NU.so $(TOOL_NAME)_NU.cpp -DTRACE -I$(VT_ROOT)/include -trace

debug:
	$(CC) -g -O0 -shared -fPIC -o $(TOOL_NAME)_UNI.so $(TOOL_NAME)_UNI.cpp
	$(CC) -g -O0 -shared -fPIC -o $(TOOL_NAME)_NU.so $(TOOL_NAME)_NU.cpp

likwid: LIKWID_LIB=$(shell echo $$(dirname $$(which likwid-perfctr)))/../lib/
likwid: LIKWID_INCLUDE=$(shell echo $$(dirname $$(which likwid-perfctr)))/../include/
likwid:
	icc -fPIC -O3 -c -o tinyexpr.o tinyexpr.c
	icpc -shared -fPIC -qopenmp -O3 -std=gnu++11 -pthread -L/opt/likwid/bin/../lib/ -I/opt/likwid/bin/../include/ -DLIKWID_PERFMON -o $(TOOL_NAME)_UNI.so $(TOOL_NAME)_UNI.cpp tinyexpr.o -llikwid

###### -std=gnu++11