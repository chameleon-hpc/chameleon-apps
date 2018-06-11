default: main

main: main.c print_array.c
	clang -v -o main -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu main.c print_array.c
	#clang -v -o main -fopenmp -fopenmp-targets=chameleon-unknown-linux-gnu main.c print_array.c
	#clang -v -o main -fopenmp main.c print_array.c
	#clang -S -emit-llvm -fopenmp main.c print_array.c

clean:
	rm -f main *.o *.ll test.out
