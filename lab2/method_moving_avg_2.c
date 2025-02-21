#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "files.h"

double method_moving_average(double* values, int index, int count, int win) {
    double sum = 0.0;
    int measurements = win + 1;
    int start = index - win/2;
    int end = index + win/2;
    
    if (start < 0) start = 0;
    if (end >= count) end = count - 1;
    
    for (int k = start; k <= end; k++) {
        sum += values[k];
    }
    
    return sum / measurements;
}

int main() {
    int count = 0;
    const int window_size = 2;

    double* x_values = (double*)malloc(BUFFER_SIZE * sizeof(double));
    double* y_values = (double*)malloc(BUFFER_SIZE * sizeof(double));

    read_values_from_file("files/Funct_with_GausseNoise.txt", x_values, y_values, &count);

    FILE *Moving_average;
    open_file(&Moving_average, "files/method_moving_average.txt", "w");

    for (int i = 0; i < count; i++) {
        double avg = method_moving_average(y_values, i, count, window_size);
        if (fprintf(Moving_average, "%.2f %.4f\n", x_values[i], avg) < 0) {
            perror("Error writing to file");
            fclose(Moving_average);
            free(x_values);
            free(y_values);
            return 1;
        }
    }

    fclose(Moving_average);
    free(x_values);
    free(y_values);

    return 0;
}