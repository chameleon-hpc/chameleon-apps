default: main

main: main.c print_array.c

	@#####  We don't really need a spearate triple since we to cross compile target regions for the host
	@#####  Instead we switch to a different RTL plugin in the runtime by using a env variable
	@#clang -### -o main -fopenmp -fopenmp-targets=chameleon-unknown-linux-gnu main.c print_array.c
	@#clang -o main -fopenmp -fopenmp-targets=chameleon-unknown-linux-gnu main.c print_array.c

	@#####  Use host triple for compilation
	@#clang -### -o main -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu main.c print_array.c
	clang -o main -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu main.c print_array.c
	
	@#####  Show intermediat representation
	@#clang -S -emit-llvm -fopenmp main.c print_array.c

clean:
	rm -f main *.o *.ll test.out
