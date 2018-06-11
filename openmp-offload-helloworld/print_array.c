#include <stdio.h>

#include "print_array.h"

void print_array(int* b, char* name, int size, int is_host){
  int i;

  printf("%s: %s = ", is_host ? "Host":"Device", name);
  for(i = 0; i < size; i++) {
    printf("%d ", b[i]);
  }
  printf("(at %p)\n", b);

}


