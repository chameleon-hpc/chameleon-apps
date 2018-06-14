#ifndef __PRINT_ARRAY_H__
#define __PRINT_ARRAY_H__

#pragma omp declare target
void print_array(int* b, char* name, int size, int is_host);
#pragma omp end declare target

#endif
