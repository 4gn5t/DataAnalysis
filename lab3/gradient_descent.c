#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "files.h"

struct LinearCoefficients {
    double a;  
    double b;  
};


struct LinearCoefficients gradient_descent(double* x_values, double* y_values, int count, 
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
    double alfa = 10.0;
    int count = 0;
    
    count = (int)((max_value - (-5.0)) / step) + 1;
    
    double* x_values_noise = (double*)malloc(count * sizeof(double));
    double* y_values_noise = (double*)malloc(count * sizeof(double));


    read_values_from_file("files/Funct_with_GausseNoise.txt", x_values_noise, y_values_noise, &count);

    FILE* gradient_descent_file = fopen("files/Gradient_decent.txt", "w");
    if (gradient_descent_file == NULL) {
        fprintf(stderr, "Error opening output file\n");
        free(x_values_noise);
        free(y_values_noise);
        return 1;
    }

    double learning_rate = 0.01;  
    int max_iter = 50000;     
    double eps = 1e-12; 
        
    struct LinearCoefficients grad_coeffs = gradient_descent(x_values_noise, y_values_noise, count, 
                                                           learning_rate, max_iter, eps);
    
    
    for (double x = -5.0; x < max_value + step/2; x += step) {
        double y = grad_coeffs.a * x + grad_coeffs.b;
        fprintf(gradient_descent_file, "%.2f %.4f\n", x, y);
    }
    
    fclose(gradient_descent_file);
    free(x_values_noise);
    free(y_values_noise);

    return 0; 
}