#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#define MAX_ALLOWED_CMD_ARGS  3

#define HELP_MESSAGE 	"Check the command line arguments \n"

#include "read_bmp.h"

int main(int argc, char* argv[])
{
    printf("--------------------------------------------------------------------\n");
    
	int cmd_arg;
	uint8_t *bmp_data;
	uint8_t *new_bmp_data, *new_bmp_img;
	uint32_t wd, ht;
	uint32_t threshold;
	uint32_t i, j;
	uint32_t sum1, sum2, mag;
	uint32_t percent_black_cells = 0;
	FILE *inFile;
	uint32_t total_cells;
	
	if (argc > MAX_ALLOWED_CMD_ARGS)
	{
		perror(HELP_MESSAGE);
		exit(-1);
	}
	
    inFile = fopen(argv[1], "rb");
	bmp_data = (uint8_t *)read_bmp_file(inFile);
	new_bmp_data = (uint8_t *)malloc(get_num_pixel());
	new_bmp_img = (uint8_t *)malloc(get_num_pixel());
	wd = get_image_width();    
	ht = get_image_height();

	threshold = 0;
	total_cells = wd * ht;
	
	int rank, size;
    MPI_Init( &argc, &argv);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Status status;
    
    int offset, quotient = (ht-2)/size;
	if(rank+1 == size){
        offset = ht-2; 
    }else{
        offset = (rank + 1) * quotient; 
    }
    
    double start = MPI_Wtime();
	while(percent_black_cells < 75)
	{	
	    percent_black_cells = 0;
		threshold += 1;	
		
		int m, black_cells = 0;
        for(m = (rank * quotient); m < offset; m++){
            i = m + 1;
            
            #pragma omp parallel for reduction(+ : black_cells) num_threads(2)
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
    				new_bmp_data[ i*wd + j] = 255;
    			}
    			else
    			{
    				new_bmp_data[ i*wd + j] = 0;
    				black_cells++;
    			}
    			
           	}
           	
        }
        
		MPI_Reduce( &black_cells, &percent_black_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD );
        percent_black_cells = (percent_black_cells * 100) / total_cells;
        MPI_Bcast( &percent_black_cells, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}
	
    double elapsedTime = MPI_Wtime() - start;
    double totalTime;
    MPI_Reduce( &elapsedTime, &totalTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
    
    int total = quotient;    
    if(rank != 0){
        
        if(size != 1 && rank + 1 == size){  
            /*
              ht - 2 = maximum offset value (see above)
              total = last offset for last process - last offset for prev process
            */
            total = (ht-2) - (quotient * rank); 
        }
        
        int i = rank * quotient + 1;
        int idx = i * wd + 1;
        // printf("%d %d %d\n", rank, idx, wd*total);
        MPI_Send( &new_bmp_data[idx], wd*total, MPI_INT, 0, 0, MPI_COMM_WORLD);
        
	}else {
	    int r;  // r - rank of process
	    for(r = 1; r < size; r++){
            
            if(r + 1 == size){  
                total = (ht-2) - (quotient * r); // or (ht-1) - (quotient * r) + 1
            }

            int i = r * quotient + 1;
	        int idx = i * wd + 1;   // 1 - initial j val in serial version
    	    MPI_Recv( &new_bmp_data[idx], wd*total, MPI_INT, r, 0, MPI_COMM_WORLD, &status); 
	    }

        printf("\nTime taken by master process during convergence loop: %.2f s\n", elapsedTime);
        printf("Sum of time taken by all MPI processes during convergence loop: %.2f s\n", totalTime);
    	printf("Threshold: %d\n", threshold);
        printf("--------------------------------------------------------------------\n");
    	
        FILE *out_file = fopen(argv[2], "wb");
    	write_bmp_file(out_file, new_bmp_data);
	    
	}
	
	free(bmp_data);
	free(new_bmp_data);
	free(new_bmp_img);
	
    MPI_Finalize();
    return 0;
}
