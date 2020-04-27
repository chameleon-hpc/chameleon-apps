TOOL_NAME=tool

release:
	$(CXX) -O3 -std=c++11 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).c

debug:
	$(CXX) -g -std=c++11 -O0 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).c
