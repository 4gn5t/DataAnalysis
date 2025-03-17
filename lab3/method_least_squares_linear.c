#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <time.h>
#include "files.h"

struct LinearCoefficients {
    double a;  
    double b;  
};

typedef struct {
    double mae;
    double mse;
    double determination;
} Metrics;

Metrics calculate_metrics(double* x_values, double* y_values, struct LinearCoefficients coeffs, int count) {
    Metrics metrics = {0.0, 0.0};
    
    for (int i = 0; i < count; i++) {
        double predicted = coeffs.a * x_values[i] + coeffs.b;
        double error = predicted - y_values[i];

        metrics.determination += (y_values[i] - predicted) * (y_values[i] - predicted);
        metrics.mae += fabs(error);
        metrics.mse += error * error;
    }
    
    metrics.mae /= count;
    metrics.mse /= count;
    
    return metrics;
}

struct LinearCoefficients method_least_squares_linear(double* x_values, double* y_values, int count) {
    struct LinearCoefficients coeffs;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_x2 = 0.0;
    double sum_xy = 0.0;

    for (int i = 0; i < count; i++) {
        sum_x += x_values[i];
        sum_y += y_values[i];
        sum_x2 += x_values[i] * x_values[i];
        sum_xy += x_values[i] * y_values[i];
    }

    coeffs.a = (count * sum_xy - sum_x * sum_y) / (count * sum_x2 - sum_x * sum_x);
    coeffs.b = (sum_y - coeffs.a * sum_x) / count;

    return coeffs;
}

int main() {
    const double step = 0.1;
    const double max_value = 5.0;
    int count = (int)((max_value - (-5.0)) / step) + 1;

    double* x_values_noise = (double*)malloc(BUFFER_SIZE * sizeof(double));
    double* y_values_noise = (double*)malloc(BUFFER_SIZE * sizeof(double));

    if (x_values_noise == NULL || y_values_noise == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    read_values_from_file("files/Funct_with_GausseNoise.txt", x_values_noise, y_values_noise, &count);

    struct LinearCoefficients lin_coeffs = method_least_squares_linear(x_values_noise, y_values_noise, count);
    
    FILE *least_squares_file = fopen("files/Least_squares_linear.txt", "w");
    FILE *gradient_file = fopen("files/Least_squares_gradient.txt", "w");

    for (double x = -5.0; x < max_value; x += step) {
        double y_ls = lin_coeffs.a * x + lin_coeffs.b;
        
        fprintf(least_squares_file, "%.2f %.4f\n", x, y_ls);
    }

    Metrics ls_metrics = calculate_metrics(x_values_noise, y_values_noise, lin_coeffs, count);

    printf("\n ERRORS::\n");
    printf("Coefficients: a = %.4f, b = %.4f\n", lin_coeffs.a, lin_coeffs.b);
    printf("MAE: %.4f\n", ls_metrics.mae);
    printf("MSE: %.4f\n", ls_metrics.mse);
    printf("RMSE: %.4f\n\n", sqrt(ls_metrics.mse));
    printf("Determination: %.4f\n", ls_metrics.determination);

    fclose(least_squares_file);
    fclose(gradient_file);
    free(x_values_noise);
    free(y_values_noise);

    return 0;
}

