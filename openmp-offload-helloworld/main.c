#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <omp.h>

#define N 10

#include "print_array.h"

struct C {
  uint64_t d;
  int e;
}; 


int main (){
  int i;
  int a = 42;
  int* b = (int*)malloc(N*sizeof(int));

  int init_dev;

  //int b[N];
  struct C c;
  c.d = 7;
  c.e= 8;

  for(i = 0; i < N; i++) {
    b[i] = -1;
  }

  printf("Host: a = %d at %p\n", a, &a);
  print_array(b, "b", N, 1);

  //init_dev = omp_get_initial_device();
  init_dev = 0;

  printf("omp_get_num_devices: %d\n", omp_get_num_devices());
  printf("omp_get_default_device: %d\n", omp_get_default_device());
  printf("omp_get_initial_device: %d\n", omp_get_initial_device());
  printf("omp_is_initial_device: %d\n", omp_is_initial_device());

  printf("Finally using device : %d\n", init_dev);

  //#pragma omp target map(tofrom:b[0:N], a)
  #pragma omp target map(tofrom:b[0:N], a) device(init_dev)
  {
    int i;
    for(i = 0; i < 10; i++) {
    //for(i = 0; i < N; i++) { //TODO: macros not working right now
      b[i] = i;
    }
   
    printf("Device: a = %d at %p\n", a, &a);
    print_array(b, "b", 10, 0);
  
    char hostname[100];
    gethostname(hostname, 100);
  
    a = 23;
    printf("Device: hostname = %s, Setting a = %d (%p).\n", hostname, a, &a);

    #pragma omp parallel num_threads(4)
    {
      printf("Device: This is thread %d of %d\n", omp_get_thread_num(), omp_get_max_threads());
    }
  }

  printf("Host: a = %d at %p\n", a, &a);
  print_array(b, "b", N, 1);

  //#pragma omp target map(to:a) map(tofrom:c)
  #pragma omp target map(to:a) map(tofrom:c) device(init_dev)
  {
    FILE* fd = fopen("test.out", "w+");
    uint64_t z = 99;
    fprintf(fd, "%ld, %ld, %d\n", z, c.d, c.e);
    fclose(fd);

    printf("This is a second target region, a is still %d (%p), wrote file test.out.\n", a, &a);
  }

  return 0;
}

