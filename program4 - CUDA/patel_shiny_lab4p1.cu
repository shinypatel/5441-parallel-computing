#include <stdio.h>
#include <stdlib.h>

__global__ void MatrixMulDevice(double *A, double *C, int size)
{
    int row = (blockIdx.x * blockDim.x + threadIdx.x)/size; // row = blockIdx.x should work!
    int col = threadIdx.x;
    int k;
    double sum = 0.0;
    for (k = 0; k < size; k++){
        sum +=  A[row * size + k] * A[col * size + k];
    }
    C[row * size + col] = sum;
}

int main(int argc, char* argv[]){
    printf("--------------------------------------------------------------------\n");
	int size = 1024;
	int totalFlops = 2 * size * size * size;
	static double A[1024][1024], C[1024][1024]; 

	int i, j; 
	for(i=0; i<size; i++){
		for(j=0; j<size; j++){
			A[i][j] = 1.0 + ((double)rand() / (double)RAND_MAX); 
		}
	}

	int nblocks = size, tpb = size;

	// allocate device memory
	size_t memSize;
	memSize = size * size * sizeof(double);

	double *d_A, *d_C; 
	cudaMalloc( (void**) &d_A, memSize);
	cudaMalloc( (void**) &d_C, memSize);

	// initialize device memory
	cudaMemcpy(d_A, A, memSize, cudaMemcpyHostToDevice);
	
	// launch kernel
	dim3 dimGrid(nblocks); 
	dim3 dimBlock(tpb);
	
    float diff;
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	// perform MatrixMul on Device
	MatrixMulDevice<<< dimGrid, dimBlock >>>(d_A, d_C, size);

	// retrieve results
	cudaMemcpy( C, d_C, memSize, cudaMemcpyDeviceToHost); 
	
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&diff, start, stop);
    
    cudaFree(d_A);
    cudaFree(d_C);
    
    printf("Time taken for parallel version: %3.1f ms \n", diff);
    float gigaFlops_s = (totalFlops / diff) / (1000 * 1000);
    printf("Gigaflops/s: %.2f\n\n", fabs(gigaFlops_s));
    
    
    //Serial
    int k;
    clock_t clockStart;
    clockStart = clock();

	for(i=0; i<size; i++){
		for(j=0; j<size; j++){
			for(k=0; k<size; k++){
				C[i][j] += A[i][k] * A[j][k];
			}
		}
	}

    diff = (clock() - clockStart) / 1000;
    printf("Time taken for serial version: %3.1f ms\n", diff);
    gigaFlops_s = (totalFlops / diff) / (1000 * 1000);
    printf("Gigaflops/s: %.2f\n", fabs(gigaFlops_s));

    printf("--------------------------------------------------------------------\n");
}

