# very simple makefile


default: bitonic-openmp  bitonic-cilk

bitonic-openmp: main.c
	gcc -DBITONIC_PARTYPE=1 -fopenmp main.c -o bitonic-openmp

bitonic-cilk: main.c
	gcc -DBITONIC_PARTYPE=2 -fcilkplus main.c -o bitonic-cilk

clean:
	-rm -f bitonic-openmp
	-rm -f bitonic-cilk
