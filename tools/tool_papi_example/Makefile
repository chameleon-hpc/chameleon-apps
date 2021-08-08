TOOL_NAME=tool

release:
	$(CC) -O3 -g -shared -fPIC -std=c11 -o $(TOOL_NAME).so $(TOOL_NAME).c -lpapi

trace:
	$(CC) -O3 -g -shared -fPIC -std=c11 -o $(TOOL_NAME).so $(TOOL_NAME).c -DTRACE -I$(VT_ROOT)/include -trace -lpapi

debug:
	$(CC) -O0 -g -shared -fPIC -std=c11 -o $(TOOL_NAME).so $(TOOL_NAME).c -lpapi
