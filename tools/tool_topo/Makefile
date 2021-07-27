TOOL_NAME=tool

# release:
# 	$(CC) -O3 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).c

# trace:
# 	$(CC) -O3 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).c -DTRACE -I$(VT_ROOT)/include -trace

# debug:
# 	$(CC) -g -O0 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).c

release:
	$(MPICXX) -std=c++11 -O3 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).cpp

trace:
	$(MPICXX) -std=c++11 -O3 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).cpp -DTRACE -I$(VT_ROOT)/include -trace

debug:
	$(MPICXX) -std=c++11 -g -O0 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).cpp
