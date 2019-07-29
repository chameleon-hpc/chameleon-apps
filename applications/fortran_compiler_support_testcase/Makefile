.PHONY: gcc clean flang


default: flang

flang:
	#$(FC) -fopenmp -fopenmp -o main case.f90
	$(FC) -v -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu -o main fortran_testcase.f90

gcc:
	#$(FC) -fopenmp -foffload=nvptx-none -o main case.f90 
	#$(FC) -fopenmp -foffload=x86_64-pc-linux-gnu -o main case.f90 
	$(FC) -fopenmp -foffload=x86_64-pc-linux-gnu --verbose -o main case.f90 
	#$(FC) -fopenmp -foffload=x86_64-pc-linux-gnu="-Wall,--verbose" -Wall --verbose -fno-stack-protector -o main case.f90 
	#$(FC) -fopenmp -foffload=anything -o main case.f90 

clean:
	rm -f main
