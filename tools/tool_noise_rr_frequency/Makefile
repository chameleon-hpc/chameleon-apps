TOOL_NAME=tool

release:
	$(CXX) -O3 -std=c++11 -fopenmp -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).c

debug:
	$(CXX) -g -std=c++11 -fopenmp -O0 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).c
