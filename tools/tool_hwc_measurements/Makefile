TOOL_NAME=tool

LIKWID_LIB=$(shell echo $$(dirname $$(which likwid-perfctr)))/../lib/
LIKWID_INCLUDE=$(shell echo $$(dirname $$(which likwid-perfctr)))/../include/

default: release

release: tiny likwid papi perf

likwid:
	icpc -pthread -fPIC  -shared -qopenmp -O3 -std=c++11 -L${LIKWID_LIB} -I${LIKWID_INCLUDE} -o $(TOOL_NAME)_likwid.so 	$(TOOL_NAME).cpp metric_calculation_wrapper/tinyexpr.o -DLIKWID_PERFMON -DUSE_LIKWID=1 -llikwid

papi:
	icpc -pthread -fPIC  -shared -qopenmp -O3 -std=c++11 -L${LIKWID_LIB} -I${LIKWID_INCLUDE} -o $(TOOL_NAME)_papi.so 	$(TOOL_NAME).cpp metric_calculation_wrapper/tinyexpr.o -DUSE_PAPI=1 -lpapi -llikwid

perf:
	icpc -pthread -fPIC  -shared -qopenmp -O3 -std=c++11 -L${LIKWID_LIB} -I${LIKWID_INCLUDE} -o $(TOOL_NAME)_perf.so 	$(TOOL_NAME).cpp metric_calculation_wrapper/tinyexpr.o -DUSE_PERF=1 -llikwid

tiny:
	icc -fPIC -w -O3 -c -o metric_calculation_wrapper/tinyexpr.o metric_calculation_wrapper/tinyexpr.c

