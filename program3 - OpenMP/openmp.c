// gcc -fopenmp openmp.c

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void main () {
	int nthrds, num_steps = 100000; 
	double step, pi; 
	step = 1.0/(double) num_steps;

	#pragma omp parallel 
	{
		int i, id;
		double x, sum = 0.0;
		id = omp_get_thread_num();
		nthrds = omp_get_num_threads();
		printf("%d\n", nthrds);
		for(i = id; i < num_steps; i += nthrds) {
			x = (i+0.5)*step;
			sum = sum + 4.0/(1.0+x*x);
		}
		#pragma omp critical
		{
			pi += step * sum; 		
		}
	}

	printf("%f\n", pi);
}