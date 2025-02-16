#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int findMin(int arr[], int n) {
	int min = arr[0];
	for (int i = 1; i < n; i++) {
		if (arr[i] < min) {
			min = arr[i];
		}
	}
	return min;
}

int main(){

	int arr[100];
	srand(time(NULL));

	for(int i = 0; i < 100; i++){
		arr[i] = rand() % 100;
		printf("%d ", arr[i]);
	}
	
	printf("\n");

	int min = findMin(arr, 100);
	printf("Min number: %d\n", min);
	

	return 0;
}
