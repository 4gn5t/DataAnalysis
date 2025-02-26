#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "files.h"

typedef struct {
    double mae;
    double mse;
} Metrics;

Metrics calculate_metrics(double* original, double* smoothed, int count) {
    Metrics metrics = {0.0, 0.0};
    
    for (int i = 0; i < count; i++) {
        double error = original[i] - smoothed[i];
        metrics.mae += fabs(error);
        metrics.mse += error * error;
    }
    
    metrics.mae /= count;
    metrics.mse /= count;
    
    return metrics;
}


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
    int window_sizes[] = {2, 4, 6, 8, 10};
    int num_windows = sizeof(window_sizes) / sizeof(window_sizes[0]);

    double* x_values = (double*)malloc(BUFFER_SIZE * sizeof(double));
    double* y_values = (double*)malloc(BUFFER_SIZE * sizeof(double));
    double* smoothed_y_values = (double*)malloc(BUFFER_SIZE * sizeof(double)); 

    read_values_from_file("files/Funct_with_GausseNoise.txt", x_values, y_values, &count);

    for (int w = 0; w < num_windows; w++) {
        int window_size = window_sizes[w];

        FILE *Moving_average;
        char filename[100];
        sprintf(filename, "files/method_moving_average_%d.txt", window_size);
        open_file(&Moving_average, filename, "w");

        for (int i = 0; i < count; i++) {
            double avg = method_moving_average(y_values, i, count, window_size);
            smoothed_y_values[i] = avg;
            if (fprintf(Moving_average, "%.2f %.4f\n", x_values[i], avg) < 0) {
                perror("Error writing to file");
                fclose(Moving_average);
                free(x_values);
                free(y_values);
                free(smoothed_y_values);
                return 1;
            }
        }

        fclose(Moving_average);

        Metrics metrics = calculate_metrics(y_values, smoothed_y_values, count);
        printf("Metrics with window %d:\n", window_size);
        printf("MAE: %.6f\n", metrics.mae);
        printf("MSE: %.6f\n", metrics.mse);
    }

    free(x_values);
    free(y_values);
    free(smoothed_y_values);

    return 0;
}