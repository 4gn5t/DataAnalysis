#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MEAN 0.0 
#define SIGMA 0.2
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

double method_moving_average(double x, int l, double (*f)(double)) {
    double sum = 0.0;
    int measurements = l + 1;  
    double step = 0.1; 
    
    for (int k = -l/2; k <= l/2; k++) {
        double x_k = x + k * step;
        sum += f(x_k);
    }
    
    return sum / measurements;
}

double calculate_updated_average(double prev_avg, double new_value, int n) {
    return prev_avg + (1.0 / n) * (new_value - prev_avg);
}

double calculate_updated_average_smoothing(double prev_avg, double new_value, int n, double alpha) {
    return alpha * new_value + (1 - alpha) * prev_avg;
}

double* smooth_values(double* values, int size) {
    if (size <= 0 || values == NULL) {
        return NULL;
    }

    double* smoothed = (double*)malloc(size * sizeof(double));
    if (smoothed == NULL) {
        return NULL;
    }

    smoothed[0] = values[0];
    for (int i = 1; i < size; i++) {
        smoothed[i] = calculate_updated_average(smoothed[i-1], values[i], i+1);
    }

    return smoothed;
}

double method_exp_smoothing(double x, double alpha, double (*f)(double)) {
    static double prev_avg = 0.0;
    static int n = 0;
    n++;
    double new_value = f(x);
    double updated_avg = calculate_updated_average_smoothing(prev_avg, new_value, n, alpha);
    prev_avg = updated_avg;
    return updated_avg;
}

void open_file(FILE **file, const char *filename, const char *mode) {
    *file = fopen(filename, mode);
    if (*file == NULL) {
        perror("Error opening file");
        exit(1);
    }
}

int main() {
    srand((unsigned int)time(NULL));
    const double step = 0.1;
    const double max_value = 30.7;
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

    FILE *Moving_average;
    open_file(&Moving_average, "files/Moving_average.txt", "w");
    for (double x = 0.0; x < max_value; x += step) {
        if (fprintf(Moving_average, "%.2f %.4f\n", x, method_moving_average(x, 2, Fun_with_Gausse_noise)) < 0) {
            perror("Error writing to file");
            fclose(Moving_average);
            return 1;
        }
    }
    fclose(Moving_average);

    for (double x = 0.0; x < max_value && count < BUFFER_SIZE; x += step) {
        x_values[count] = x;
        y_values[count] = Fun_with_Gausse_noise(x);
        count++;
    }

    double* smoothed = smooth_values(y_values, count);
    if (smoothed == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(x_values);
        free(y_values);
        return 1;
    }

    FILE *Moving_average_next;
    open_file(&Moving_average_next, "files/Moving_average_next.txt", "w");
    
    for (int i = 0; i < count; i++) {
        if (fprintf(Moving_average_next, "%.2f %.4f\n", x_values[i], smoothed[i]) < 0) {
            perror("Error writing to file");
            fclose(Moving_average_next);
            free(x_values);
            free(y_values);
            free(smoothed);
            return 1;
        }
    }
    
    fclose(Moving_average_next);
    free(x_values);
    free(y_values);
    free(smoothed);

    FILE *Exp_smoothing;
    open_file(&Exp_smoothing, "files/Exp_smoothing.txt", "w");
    for (double x = 0.0; x < max_value; x += step) {
        if (fprintf(Exp_smoothing, "%.2f %.4f\n", x, method_exp_smoothing(x, 0.1, Fun_with_Gausse_noise)) < 0) {
            perror("Error writing to file");
            fclose(Exp_smoothing);
            return 1;
        }
    }
    fclose(Exp_smoothing);

    return 0;
}