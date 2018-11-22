#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void *say_hello(void* i){
	printf("%d\n", *((int *) i));
	free(i);
	pthread_exit((void *) NULL);
}

int main(){
	int i;
	int *param;
	pthread_t *threads;
	void *th_status;

	for(i = 0; i < 10; i++){
		param = malloc(sizeof(int));
		*param = i;
    	pthread_create(&threads[i], NULL, say_hello, (void *)param);
	}

	for(i = 0; i < 10; i++){
		pthread_join(threads[i], &th_status);
	}
}