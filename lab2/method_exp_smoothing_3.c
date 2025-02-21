#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "files.h"

typedef struct {
    double mae;
    double mse;
} Metrics;

double calculate_updated_average_smoothing(double prev_avg, double new_value, double alpha) {
    return alpha * new_value + (1 - alpha) * prev_avg;
}

double method_exp_smoothing(double* values, int index, double alpha) {
    static double prev_avg = 0.0;
    double updated_avg = calculate_updated_average_smoothing(prev_avg, values[index], alpha);
    prev_avg = updated_avg;
    return updated_avg;
}

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

void test_alpha(double* x_values, double* y_values, int count, double alpha) {
    double* smoothed_values = (double*)malloc(count * sizeof(double));
    method_exp_smoothing(y_values, 0, alpha);
    
    for (int i = 0; i < count; i++) {
        smoothed_values[i] = method_exp_smoothing(y_values, i, alpha);
    }
    
    Metrics metrics = calculate_metrics(y_values, smoothed_values, count);
    printf("Alpha: %.2f, MAE: %.6f, MSE: %.6f\n", alpha, metrics.mae, metrics.mse);
    
    free(smoothed_values);
}

int main() {
    int count = 0;

    double* x_values = (double*)malloc(BUFFER_SIZE * sizeof(double));
    double* y_values = (double*)malloc(BUFFER_SIZE * sizeof(double));

    read_values_from_file("files/Funct_with_GausseNoise.txt", x_values, y_values, &count);

    double alpha_values[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    int num_alphas = sizeof(alpha_values) / sizeof(alpha_values[0]);

    for (int i = 0; i < num_alphas; i++) {
        test_alpha(x_values, y_values, count, alpha_values[i]);
    }
    
    const double best_alpha = 0.1;

    FILE *Exp_smoothing = NULL;
    open_file(&Exp_smoothing, "files/Exp_smoothing.txt", "w");

    for (int i = 0; i < count; i++) {
        double smoothed = method_exp_smoothing(y_values, i, best_alpha);
        if (fprintf(Exp_smoothing, "%.2f %.4f\n", x_values[i], smoothed) < 0) {
            perror("Error writing to file");
            fclose(Exp_smoothing);
            free(x_values);
            free(y_values);
            return 1;
        }
    }

    fclose(Exp_smoothing);
    free(x_values);
    free(y_values);

    return 0;
}