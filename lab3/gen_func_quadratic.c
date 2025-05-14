#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "files.h"

#define MEAN 0.0
#define SIGMA 1.0

double Fun_2(double x) {
    return 9 * pow(x, 2) - 8 * x + 0.1;
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
    return 9 * pow(x, 2) - 8 * x + 0.1 + alfa * gaussian_random();
}

int main() {
    const double step = 0.1;
    const double min_value = -5.0;
    const double max_value = 5.0;
    double alfa = 1.0;

    int count = (int)((max_value - min_value) / step) + 1; 
    double* x_values = (double*)malloc(count * sizeof(double));
    double* y_values = (double*)malloc(count * sizeof(double));

    if (!x_values || !y_values) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    // Generate noisy quadratic data
    FILE *Funct_with_GausseNoise = fopen("files/Funct_quadratic_with_GausseNoise.txt", "w");
    if (!Funct_with_GausseNoise) {
        perror("Error opening file");
        return 1;
    }

    int i = 0;
    for (double x = min_value; x <= max_value; x += step) {
        x_values[i] = x;
        y_values[i] = Fun_with_Gausse_noise(x, alfa);
        fprintf(Funct_with_GausseNoise, "%.2f %.4f\n", x, y_values[i]);
        i++;
    }
    fclose(Funct_with_GausseNoise);

    // Generate true quadratic function data (without noise)
    FILE *Funct_2 = fopen("files/Funct_2_quadratic.txt", "w");
    if (!Funct_2) {
        perror("Error opening file");
        return 1;
    }

    for (double x = min_value; x <= max_value; x += step) {
        fprintf(Funct_2, "%.2f %.4f\n", x, Fun_2(x));
    }
    fclose(Funct_2);

    free(x_values);
    free(y_values);
    return 0;
}
