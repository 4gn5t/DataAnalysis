#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "files.h"

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

int main(){
    const double step = 0.1;
    const double max_value = 5.0;
    double alfa = 10.0;
    int count = 0;
    for (double x = -5.0; x < max_value; x += step) {
        count++;
    }
    double* x_values = (double*)malloc(count * sizeof(double));
    double* y_values = (double*)malloc(count * sizeof(double));

    int i = 0;
    for (double x = -5.0; x < max_value; x += step) {
        x_values[i] = x;
        y_values[i] = Fun_with_Gausse_noise(x, alfa);
        i++;
    }

    FILE *Funct_2;
    Funct_2 = fopen("files/Funct_2.txt", "w");


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

    return 0;
}