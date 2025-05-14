#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "files.h"

#define MEAN 0.0 
#define SIGMA 0.1
#define BUFFER_SIZE 1000

double gaussian_random() {
    double u1, u2;
    do {
        u1 = (double)rand() / RAND_MAX;
    } while (u1 == 0);
    u2 = (double)rand() / RAND_MAX;
    
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return MEAN + SIGMA * z;
}

double Fun_with_Gausse_noise(double x) {
    return sin(0.1 * x) * cos(x) + gaussian_random();
}

double Fun_2(double x) {
    return sin(0.1 * x) * cos(x);
}

int main(){
    srand((unsigned int)time(NULL));
    const double step = 0.1;
    const double max_value = 40;
    double* x_values = (double*)malloc(BUFFER_SIZE * sizeof(double));
    double* y_values = (double*)malloc(BUFFER_SIZE * sizeof(double));
    int count = 0;

    FILE *Funct_with_GausseNoise;
    open_file(&Funct_with_GausseNoise, "files/Funct_with_GausseNoise.txt", "w");

    for (double x = 0.0; x < max_value; x += step) {
        if (fprintf(Funct_with_GausseNoise, "%.2f %.4f\n", x, Fun_with_Gausse_noise(x)) < 0) {
            perror("Error writing to file");
            fclose(Funct_with_GausseNoise);
            return 1;
        }
    }
    fclose(Funct_with_GausseNoise);

    FILE *Funct_2;
    open_file(&Funct_2, "files/Funct_2.txt", "w");
    for (double x = 0.0; x < max_value; x += step) {
        if (fprintf(Funct_2, "%.2f %.4f\n", x, Fun_2(x)) < 0) {
            perror("Error writing to file");
            fclose(Funct_2);
            return 1;
        }
    }
    fclose(Funct_2);    
    return 0;
}