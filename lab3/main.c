#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <time.h>

#define MEAN 0.0
#define SIGMA 0.2

struct LinearCoefficients {
    double a;  
    double b;  
};

struct QuadraticCoefficients {
    double a;  
    double b; 
    double c; 
};

double Fun_2(double x) {
    return 8 - 2*x;
}

double gaussian_random() {
    double u1, u2;
    do {
        u1 = (double)rand() / RAND_MAX;
    } while (u1 == 0);
    u2 = (double)rand() / RAND_MAX;
    
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return MEAN + SIGMA * z;
}

double Fun_with_Gausse_noise(double x, double alfa) {
    return 8 - 2*x + alfa * gaussian_random();
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

struct QuadraticCoefficients method_least_squares_quadratic(double* x_values, double* y_values, int count) {
    struct QuadraticCoefficients coeffs;
    double sum_x = 0.0, sum_y = 0.0;
    double sum_x2 = 0.0, sum_x3 = 0.0, sum_x4 = 0.0;
    double sum_xy = 0.0, sum_x2y = 0.0;

    for (int i = 0; i < count; i++) {
        double x = x_values[i];
        double y = y_values[i];
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_x3 += x * x * x;
        sum_x4 += x * x * x * x;
        sum_xy += x * y;
        sum_x2y += x * x * y;
    }

    double d = count * sum_x2 * sum_x4 + 2 * sum_x * sum_x2 * sum_x3 
               - sum_x2 * sum_x2 * sum_x2 - count * sum_x3 * sum_x3 
               - sum_x * sum_x * sum_x4;

    coeffs.a = (sum_x2y * (count * sum_x2 - sum_x * sum_x) 
                - sum_xy * (count * sum_x3 - sum_x * sum_x2) 
                + sum_y * (sum_x * sum_x3 - sum_x2 * sum_x2)) / d;

    coeffs.b = (count * sum_xy * sum_x4 - sum_x * sum_y * sum_x4 
                - count * sum_x2y * sum_x3 + sum_x2 * sum_y * sum_x3 
                + sum_x * sum_x2y * sum_x2 - sum_x2 * sum_xy * sum_x2) / d;

    coeffs.c = (sum_y * sum_x2 * sum_x4 - sum_xy * sum_x3 * sum_x2 
                + sum_x2y * sum_x2 * sum_x2 - sum_y * sum_x3 * sum_x3 
                - sum_x * sum_x2y * sum_x3 + sum_x * sum_xy * sum_x4) / d;

    return coeffs;
}

double calculate_error_procent(double expected, double actual) {
    return fabs(expected - actual) / expected * 100.0;
}

int main(){
    srand((unsigned int)time(NULL));

    const double step = 0.1;
    const double max_value = 5.0;
    double alfa = 10.0;
    int count = 0;
    for (double x = -5.0; x < max_value; x += step) {
        count++;
    }
    double* x_values = (double*)malloc(count * sizeof(double));
    double* y_values = (double*)malloc(count * sizeof(double));
    //double alfa = ((double)rand() / RAND_MAX) * 10.0;

    int i = 0;
    for (double x = -5.0; x < max_value; x += step) {
        x_values[i] = x;
        y_values[i] = Fun_with_Gausse_noise(x, alfa);
        i++;
    }

    FILE *Funct_2;
    Funct_2 = fopen("files/Funct_2.txt", "w");
    if (Funct_2 == NULL) {
        perror("Error opening file");
        return 1;
    }
    for (double x = -5.0; x < max_value; x += step) {
        if (fprintf(Funct_2, "%.2f %.4f\n", x, Fun_2(x)) < 0) {
            perror("Error writing to file");
            fclose(Funct_2);
            return 1;
        }
    }
    fclose(Funct_2);

    FILE *Funct_with_GausseNoise;
    Funct_with_GausseNoise = fopen("files/Funct_with_GausseNoise.txt", "w");
    if (Funct_with_GausseNoise == NULL) {
        perror("Error opening file");
        return 1;
    }
    for (double x = -5.0; x < max_value; x += step) {
        if (fprintf(Funct_with_GausseNoise, "%.2f %.4f\n", x, Fun_with_Gausse_noise(x, alfa)) < 0) {
            perror("Error writing to file");
            fclose(Funct_with_GausseNoise);
            return 1;
        }
    }
    fclose(Funct_with_GausseNoise);

    struct LinearCoefficients lin_coeffs = method_least_squares_linear(x_values, y_values, count);
    
    FILE *Least_squares_linear = fopen("files/Least_squares_linear.txt", "w");
    if (Least_squares_linear == NULL) {
        perror("Error opening file");
        return 1;
    }
    
    for (double x = -5.0; x < max_value; x += step) {
        double y = lin_coeffs.a * x + lin_coeffs.b;  
        if (fprintf(Least_squares_linear, "%.2f %.4f\n", x, y) < 0) {
            perror("Error writing to file");
            fclose(Least_squares_linear);
            return 1;
        }
    }
    fclose(Least_squares_linear);

    struct QuadraticCoefficients quad_coeffs = method_least_squares_quadratic(x_values, y_values, count);
    
    FILE *Least_squares_quadratic = fopen("files/Least_squares_quadratic.txt", "w");
    if (Least_squares_quadratic == NULL) {
        perror("Error opening file");
        return 1;
    }
    
    for (double x = -5.0; x < max_value; x += step) {
        double y = quad_coeffs.a * x * x + quad_coeffs.b * x + quad_coeffs.c;  
        if (fprintf(Least_squares_quadratic, "%.2f %.4f\n", x, y) < 0) {
            perror("Error writing to file");
            fclose(Least_squares_quadratic);
            return 1;
        }
    }
    fclose(Least_squares_quadratic);

    free(x_values);
    free(y_values);

    FILE* error = fopen("files/Error.txt", "w");
    fprintf(error, "Alpha: %.1f\n", alfa);
    fprintf(error, "Linear approximation:\n");
    fprintf(error, "a1 = %.4f\n", lin_coeffs.b);  
    fprintf(error, "a2 = %.4f\n", lin_coeffs.a); 
    fprintf(error, "\nQuadratic approximation:\n");
    fprintf(error, "a1 = %.4f\n", quad_coeffs.c);  
    fprintf(error, "a2 = %.4f\n", quad_coeffs.b); 
    fprintf(error, "a3 = %.4f\n", quad_coeffs.a); 

    return 0;
}