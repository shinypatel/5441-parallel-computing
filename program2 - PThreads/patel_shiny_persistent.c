#include "def.h"

int n, num_threads, count, quotient, converge; 
struct box boxes[MAX];
double max, min, diff, epsilon, affectRate, waat[MAX];
pthread_barrier_t barr;

void *newTemp(void *param){
	int i, offset, thread_num = *((int *)param);
	if(thread_num+1 == num_threads || num_threads == n){
		offset = n ; 
	}else{
		offset = (thread_num + 1) * quotient; 
	}

	while(converge == 0){
		for(i = (thread_num * quotient); i < offset; i++){
			int j, perimeter = 0;
			int height = boxes[i].info[2];
			int width = boxes[i].info[3];
			
			for(j = 0; j < 4; j++){		// consider its each side
				int num, *ids;
				int contactDist = 1;
				
				if(j == 0){	
					num = boxes[i].top;
					ids = boxes[i].t_id;
				}else if(j == 1){
					num = boxes[i].bottom;
					ids = boxes[i].b_id;
				}else if(j == 2){
					num = boxes[i].left;
					ids = boxes[i].l_id;
				}else if(j == 3){
					num = boxes[i].right;
					ids = boxes[i].r_id;
				}

				if(num == 0){	// if there are no neighbors on a side, use the current box's temp
					if(j == 0 || j == 1){	// top or bottom
						contactDist = width;
					}else{
						contactDist = height;
					}
					waat[i] += (boxes[i].temp * contactDist);
					perimeter += contactDist;
				}else{
					int k, a, b, c, d;
					for(k = 0; k < num; k++){ 	
						int id = ids[k];
						
						if(j == 2 || j == 3){	// left or right
							a = boxes[id].info[0];	// upper-left x of neighbor
							b = boxes[id].info[4];	// upper-right x of neighbor
							c = boxes[i].info[0];	// upper-left x of current box
							d = boxes[i].info[4];	// upper-right x of current box
							contactDist = contactDistance(height, a, b, c, d);
						}else{
							a = boxes[id].info[1];	// similar for y
							b = boxes[id].info[5];
							c = boxes[i].info[1];	
							d = boxes[i].info[5];
							contactDist = contactDistance(width, a, b, c, d);
						}
						waat[i] += (boxes[id].temp * contactDist);
						perimeter += contactDist;
					}
				}
			}
			waat[i] /= perimeter;
		}
		pthread_barrier_wait(&barr);

		for(i = (thread_num * quotient); i < offset; i++){		// update box temp (DSVs)
			if(boxes[i].temp > waat[i]){
				boxes[i].temp -= ((boxes[i].temp - waat[i]) * affectRate);
			}else{
				boxes[i].temp += ((waat[i] - boxes[i].temp) * affectRate);
			}	
		}
		pthread_barrier_wait(&barr);

		if(thread_num == 0){
			for(i = 0; i < n; i++){
				max = maximum(i, max, boxes[i].temp);	// find new max and min 
				min = minimum(i, min, boxes[i].temp);
			}
			diff = max - min;

			if((diff/max) > epsilon){
				memset(waat, 0, n * sizeof(double));
			}else{
				converge = 1; 
			}
			count++; 
		}
		pthread_barrier_wait(&barr);
	}
	free(param);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) { 

	if(argc == 4){
		affectRate = atof(argv[1]);
		epsilon = atof(argv[2]);
		num_threads = atoi(argv[3]);
	}else{
		printf("Invalid number of arguments!\n");
		exit(-1);
	}	

	FILE* file = stdin;
	char line[256];
	
	int i, j, idx;
	if(!fgets(line, sizeof(line), file)){
		printf("Error reading!\n");
		exit(-1);
	}

    int *size = split(line);
	char *token = strtok(line, " \t");

    n = size[0];	// total number of boxes
    if(num_threads > n){
    	num_threads = n - 1; 
    }
    quotient = n/num_threads;

	for(i = 0; i < n; i++){		// for each box
		j = 0;
		while(j < 7){		// read its corresponding 7 lines
			if(!fgets(line, sizeof(line), file)){
				printf("Error reading file!\n");
			}

			if(line[0] == '\n' || line[0] == '\t'){		// skip empty lines
				continue;
			}

			if(j == 1){
				token = strtok(line, " \t");
				for(idx = 0; token != NULL; idx++) {
					boxes[i].info[idx] = atoi(token);
					token = strtok(NULL, " \t");	
				}
				// calculate upper-right x from upper-left x and height
				boxes[i].info[4] = boxes[i].info[0] + boxes[i].info[2] - 1;		
				// calculate upper-right y from upper-left y and width
				boxes[i].info[5] = boxes[i].info[1] + boxes[i].info[3] - 1; 	
			}else if(j == 2){
				int *arr = split(line);
				boxes[i].top = arr[0];
				boxes[i].t_id = splice(boxes[i].top, arr);
			}else if(j == 3){
				int *arr = split(line);
				boxes[i].bottom = arr[0];
				boxes[i].b_id = splice(boxes[i].bottom, arr);
			}else if(j == 4){
				int *arr = split(line);
				boxes[i].left = arr[0];
				boxes[i].l_id = splice(boxes[i].left, arr);
			}else if(j == 5){
				int *arr = split(line);
				boxes[i].right = arr[0];
				boxes[i].r_id = splice(boxes[i].right, arr);
			}else if(j == 6){
				boxes[i].temp = atof(line);
				max = maximum(i, max, boxes[i].temp);
				min = minimum(i, min, boxes[i].temp);
			} 
			j++;
		}
	}

	diff = max - min;
	count = 0, converge = 0; 

    int *param;
    pthread_t threads[num_threads];
	void *th_status[num_threads];
    pthread_barrier_init(&barr, NULL, num_threads); 

    time_t timeStart, timeEnd;
    time(&timeStart);
	clock_t clockk = clock();
	clock_gettime (CLOCK_REALTIME, &start);

	for(i = 0; i < num_threads; i++){
		param = malloc(sizeof(int));
		*param = i;
    	pthread_create(&threads[i], NULL, newTemp, (void *)param);
	}

	for(i = 0; i < num_threads; i++){
		pthread_join(threads[i], &th_status[i]);
	}

    pthread_barrier_destroy(&barr);

	clock_gettime (CLOCK_REALTIME, &end);
	double posixClock = (double)(((end.tv_sec - start.tv_sec) * CLOCKS_PER_SEC) + ((end.tv_nsec - start.tv_nsec) / NS_PER_US));
	printf("------------------------------------------------------------------\n");
  	printf ("elapsed convergence loop time (posix clock): %.2f\n", posixClock);

	clockk = clock() - clockk;
  	printf ("elapsed convergence loop time (clock ticks): %d\n", (int)clockk);
  	printf("elapsed convergence loop time (clock seconds): %.2f\n", (double)clockk/CLOCKS_PER_SEC);

    time(&timeEnd);
    int timeDiff = timeEnd - timeStart;
    printf("elapsed convergence loop time (time): %d\n", timeDiff);

	printf("\nnumber of iterations: %d\n", count);
	printf("max DSV: %f and min DSV: %f\n", max, min);
	printf("epsilon: %.2f and affect rate: %.2f\n", epsilon, affectRate); 
	printf("------------------------------------------------------------------\n");

	for(i = 0; i < n; i++){	
		free(boxes[i].t_id);
		free(boxes[i].b_id);
		free(boxes[i].l_id);
		free(boxes[i].r_id);
	}

}
