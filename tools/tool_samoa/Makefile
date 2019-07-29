TOOL_NAME=tool_annotation_cells

release:
	$(CC) -O3 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).c

debug:
	$(CC) -g -O0 -shared -fPIC -o $(TOOL_NAME).so $(TOOL_NAME).c
