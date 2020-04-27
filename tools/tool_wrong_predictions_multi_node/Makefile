TOOL_NAME=tool

release:
	$(CC) -O3 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).c

trace:
	$(CC) -O3 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).c -DTRACE -I$(VT_ROOT)/include -trace

debug:
	$(CC) -g -O0 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).c
