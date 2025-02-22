#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <time.h>
#include "files.h"

struct LinearCoefficients {
    double a;  
    double b;  
};

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

struct LinearCoefficients gradient_descent_optimization(double* x_values, double* y_values, int count, 
    double learning_rate, int max_iter, double eps) {
    struct LinearCoefficients coeffs = {0.0, 0.0};
    double prev_cost = INFINITY;

    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    for (int i = 0; i < count; i++) {
        sum_x += x_values[i];
        sum_y += y_values[i];
        sum_xy += x_values[i] * y_values[i];
        sum_x2 += x_values[i] * x_values[i];
    }
    coeffs.a = (count * sum_xy - sum_x * sum_y) / (count * sum_x2 - sum_x * sum_x);
    coeffs.b = (sum_y - coeffs.a * sum_x) / count;

    double learning_rate_a = learning_rate;
    double learning_rate_b = learning_rate;

    for (int iter = 0; iter < max_iter; iter++) {
        double a_grad = 0.0;
        double b_grad = 0.0;
        double current_cost = 0.0;

        for (int j = 0; j < count; j++) {
            double error = (coeffs.a * x_values[j] + coeffs.b) - y_values[j];
            a_grad += error * x_values[j];
            b_grad += error;
            current_cost += error * error;
        }
        current_cost /= count;

        a_grad = (2.0/count) * a_grad;
        b_grad = (2.0/count) * b_grad;

        coeffs.a -= learning_rate_a * a_grad;
        coeffs.b -= learning_rate_b * b_grad;

        if (current_cost > prev_cost) {
            learning_rate_a *= 0.5;
            learning_rate_b *= 0.5;
        }

        if (fabs(prev_cost - current_cost) < eps) {
            break;
        }
        prev_cost = current_cost;
    }

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
    struct LinearCoefficients grad_coeffs = gradient_descent_optimization(x_values_noise, y_values_noise, count, 
                                                                        0.01, 50000, 1e-12);
    
    FILE *least_squares_file = fopen("files/Least_squares_linear.txt", "w");
    if (least_squares_file == NULL) {
        perror("Error opening least squares file");
        free(x_values_noise);
        free(y_values_noise);
        return 1;
    }

    FILE *gradient_file = fopen("files/Least_squares_gradient.txt", "w");
    if (gradient_file == NULL) {
        perror("Error opening gradient file");
        fclose(least_squares_file);
        free(x_values_noise);
        free(y_values_noise);
        return 1;
    }

    for (double x = -5.0; x < max_value; x += step) {
        double y_ls = lin_coeffs.a * x + lin_coeffs.b;
        double y_grad = grad_coeffs.a * x + grad_coeffs.b;
        
        fprintf(least_squares_file, "%.2f %.4f\n", x, y_ls);
        fprintf(gradient_file, "%.2f %.4f\n", x, y_grad);
    }

    fclose(least_squares_file);
    fclose(gradient_file);
    free(x_values_noise);
    free(y_values_noise);

    return 0;
}

