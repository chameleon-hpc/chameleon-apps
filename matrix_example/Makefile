.PHONY: clean default debug

default: 
	${MPICXX} -o main -std=c++11 -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu matrix_example.cpp -lm -lchameleon -lstdc++ 
	
debug: 
	${MPICXX} -g -o main -std=c++11 -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu matrix_example.cpp -lm -lchameleon -lstdc++ 

clean:
	rm -f main *.o 
