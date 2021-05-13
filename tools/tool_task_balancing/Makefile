TOOL_NAME=tool

release:
	$(CC) -O3 -shared -fPIC -std=gnu++11 -o $(TOOL_NAME).so $(TOOL_NAME).cpp

trace:
	$(CC) -O3 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).cpp -DTRACE -I$(VT_ROOT)/include -trace

debug:
	$(CC) -g -O0 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).cpp


###### -std=gnu++11