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
	$(CC) -g -O0 -shared -fPIC -std=gnu++11 -pthread -DLIKWID_PERFMON -L${LIKWID_LIB} -I${LIKWID_INCLUDE} -o $(TOOL_NAME)_UNI.so $(TOOL_NAME)_UNI.cpp -llikwid
	$(CC) -g -O0 -shared -fPIC -std=gnu++11 -pthread -DLIKWID_PERFMON -L${LIKWID_LIB} -I${LIKWID_INCLUDE} -o $(TOOL_NAME)_NU.so $(TOOL_NAME)_NU.cpp -llikwid


###### -std=gnu++11