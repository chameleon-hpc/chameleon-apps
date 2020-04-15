TOOL_NAME=tool

release:
	$(CXX) -O3 -shared -fPIC -std=c++11 -I${INCLUDE} -o $(TOOL_NAME).so $(TOOL_NAME).cpp

debug:
	$(CXX) -g -O0 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).cpp
