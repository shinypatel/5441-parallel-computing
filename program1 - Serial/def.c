#include "def.h"

int* split(char line[]){
	int idx;
	static int arr[100];
	char *token = strtok(line, " \t");
    for(idx = 0; token != NULL; idx++) {
        arr[idx] = atoi(token);
        token = strtok(NULL, " \t");
    }
    return arr;
}

int* splice(int size, int *arr){
	int i, j, *ids = (int *) malloc(sizeof(int) * size);
	for(i = 0, j = 1; i < size; i++, j++){
		ids[i] = arr[j];
	}
	return ids;
}

double maximum(int i, double max, double temp){
	if(i == 0){		// initial max temp
		max = temp;
	}
	else{
		if(temp > max){
			max = temp;
		}
	}
	return max;
}


double minimum(int i, double min, double temp){
	if(i == 0){
		min = temp;
	}
	else{
		if(temp < min){
			min = temp;
		}
	}
	return min;
}

int contactDistance(int x, int a, int b, int c, int d){
	int distance;
	if(a > c && b > d){
		distance = (d - a) + 1;
	}else if(a < c && b < d){
		distance = (b - c) + 1;
	}else if(a >= c && b <= d){
		distance = (a - c) + (d - b);
		return x - distance;
	}else{
		distance = x;
	}
	return distance;
}