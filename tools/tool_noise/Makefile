TOOL_NAME=tool

release:
	$(CC) -O3 -shared -fPIC -I${INCLUDE} -o $(TOOL_NAME).so $(TOOL_NAME).c

debug:
	$(CC) -g -O0 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).c
