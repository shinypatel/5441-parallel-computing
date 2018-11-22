#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ALLOWED_CMD_ARGS  4

#define HELP_MESSAGE 	"Check the command line arguments \n"

extern "C"
{
    #include "read_bmp.h"
}

__global__ void sobelDevice(uint8_t *bmp_data, uint8_t *new_bmp_img, uint32_t wd, uint32_t threshold, 
        uint32_t *percent_black_cells)
{
    uint32_t i = blockIdx.x + 1;
    uint32_t j = threadIdx.x + 1;
    
	uint32_t sum1, sum2, mag;
	sum1 = bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i-1)*wd + (j-1) ] \
			+ 2*bmp_data[ (i)*wd + (j+1) ] - 2*bmp_data[ (i)*wd + (j-1) ] \
			+ bmp_data[ (i+1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ];
		
	sum2 = bmp_data[ (i-1)*wd + (j-1) ] + 2*bmp_data[ (i-1)*wd + (j) ] \
			+ bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ] \
			- 2*bmp_data[ (i+1)*wd + (j) ] - bmp_data[ (i+1)*wd + (j+1) ];
		
	mag = sqrt((double)(sum1 * sum1 + sum2 * sum2));
	
	if(mag > threshold)
	{
		new_bmp_img[ i*wd + j] = 255;
	}
	else
	{
		new_bmp_img[ i*wd + j] = 0;
		atomicAdd(percent_black_cells, 1);
	}
	
}

int main(int argc, char* argv[])
{
    printf("--------------------------------------------------------------------\n");
    
	int cmd_arg;
	uint8_t *bmp_data;
	uint8_t *new_bmp_img;
	uint32_t wd, ht;
	uint32_t threshold;
	FILE *serial_out_file, *parallel_out_file, *inFile;
	uint32_t percent_black_cells = 0;
	uint32_t total_cells;
	
	/*First Check if no of arguments is permissible to execute*/
	if (argc > MAX_ALLOWED_CMD_ARGS)
	{
		perror(HELP_MESSAGE);
		exit(-1);
	}
	
	/*Roll over the command line arguments and obtain the values*/
	for (cmd_arg = 1; cmd_arg < argc; cmd_arg++)
	{

		/*Switch execution based on the pass of commad line arguments*/
		switch (cmd_arg)
		{
			case 1: 
                inFile = fopen(argv[cmd_arg], "rb");
				break;
			
			case 2: 
                serial_out_file = fopen(argv[cmd_arg], "wb");
				break;
				
			case 3: 
                parallel_out_file = fopen(argv[cmd_arg], "wb");
				break;
		}
	}
	
	//Read the binary bmp file into buffer
	bmp_data = (uint8_t *)read_bmp_file(inFile);	
	
	new_bmp_img = (uint8_t *)malloc(get_num_pixel());
	wd = get_image_width();    
	ht = get_image_height();
	
	//start measurement for sobel loop only
    float timeDiff;
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
	
	int nblocks = ht-2, tpb = wd-2;

	size_t memSize;
	memSize = get_num_pixel() * sizeof(uint8_t);
	
    uint32_t *d_percent_black_cells;
	uint8_t *d_bmp_data;
	uint8_t *d_new_bmp_img;
	
	cudaMalloc( (void**) &d_bmp_data, memSize);
	cudaMalloc( (void**) &d_new_bmp_img, memSize);
    
	cudaMemcpy(d_bmp_data, bmp_data, memSize, cudaMemcpyHostToDevice);

	threshold = 0;
	total_cells = wd * ht;
	while(percent_black_cells < 75)
	{	
		percent_black_cells = 0;
		threshold += 1;		
        
    	cudaMalloc( (void**) &d_percent_black_cells, sizeof(uint32_t));
    	
    	// initialize device memory
    	cudaMemcpy(d_percent_black_cells, &percent_black_cells, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    	// launch kernel
    	dim3 dimGrid(nblocks); 
    	dim3 dimBlock(tpb);
		
		sobelDevice<<< dimGrid, dimBlock >>>(d_bmp_data, d_new_bmp_img, wd, threshold, d_percent_black_cells);
		
    	// retrieve results
    	cudaMemcpy(new_bmp_img, d_new_bmp_img, memSize, cudaMemcpyDeviceToHost);
    	cudaMemcpy(&percent_black_cells, d_percent_black_cells, sizeof(uint32_t), cudaMemcpyDeviceToHost);

		percent_black_cells = (percent_black_cells * 100) / total_cells;
	}
	
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeDiff, start, stop);
    
    printf("Time taken for Cuda Sobel Operation: %3.1f ms\n", timeDiff);
	printf("Threshold: %d\n\n", threshold);
	
	cudaFree(d_bmp_data);
	cudaFree(d_new_bmp_img);
	cudaFree(d_percent_black_cells);
	
	write_bmp_file(parallel_out_file, new_bmp_img);
	
	
	// Serial
	uint32_t i, j;
	uint32_t sum1, sum2, mag;
	new_bmp_img = (uint8_t *)malloc(get_num_pixel());
	
	//start measurement for sobel loop only
    clock_t clockStart = clock();
	
	threshold = 0;
	percent_black_cells = 0;
	while(percent_black_cells < 75)
	{	

		percent_black_cells = 0;
		threshold += 1;
		
		for(i=1; i < (ht-1); i++)
		{
			for(j=1; j < (wd-1); j++)
			{
				sum1 = bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i-1)*wd + (j-1) ] \
						+ 2*bmp_data[ (i)*wd + (j+1) ] - 2*bmp_data[ (i)*wd + (j-1) ] \
						+ bmp_data[ (i+1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ];
						
				sum2 = bmp_data[ (i-1)*wd + (j-1) ] + 2*bmp_data[ (i-1)*wd + (j) ] \
						+ bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ] \
						- 2*bmp_data[ (i+1)*wd + (j) ] - bmp_data[ (i+1)*wd + (j+1) ];
						
				mag = sqrt(sum1 * sum1 + sum2 * sum2);
				if(mag > threshold)
				{
					new_bmp_img[ i*wd + j] = 255;
				}
				else
				{
					new_bmp_img[ i*wd + j] = 0;
					percent_black_cells++;
				}
			}
		}
		percent_black_cells = (percent_black_cells * 100) / total_cells;
	}
		
	double diff = (clock() - clockStart) / 1000;
	printf("Time taken for Serial Sobel Operation: %3.1f ms\n", diff);
	printf("Threshold: %d\n", threshold);
	
	//Write the buffer into the bmp file
	write_bmp_file(serial_out_file, new_bmp_img);
	
    printf("--------------------------------------------------------------------\n");
    return 0;
}
