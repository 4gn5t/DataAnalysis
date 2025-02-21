#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "files.h"

#define MEAN 0.0 
#define SIGMA 0.2
#define BUFFER_SIZE 1000

double calculate_updated_avg(double prev_val, double current_val, int n) {
    return prev_val + (1.0 / n) * (current_val - prev_val);
}

double* _calculate(double* values, int size) {
    double* data = (double*)malloc(size * sizeof(double));
    data[0] = values[0];
    for (int i = 1; i < size; i++) {
        data[i] = calculate_updated_avg(data[i-1], values[i], i+1);
    }

    return data;
}

int main(){
    srand((unsigned int)time(NULL));
    int count = 0;
    double* x_values = (double*)malloc(BUFFER_SIZE * sizeof(double));
    double* y_values = (double*)malloc(BUFFER_SIZE * sizeof(double));

    read_values_from_file("files/Funct_with_GausseNoise.txt", x_values, y_values, &count);
    double* data = _calculate(y_values, count);
    
    FILE *Method_update_avg_;
    open_file(&Method_update_avg_, "files/Moving_update_avg.txt", "w");
    
    for (int i = 0; i < count; i++) {
        if (fprintf(Method_update_avg_, "%.2f %.4f\n", x_values[i], data[i]) < 0) {
            perror("Error writing to file");
            fclose(Method_update_avg_);
            free(x_values);
            free(y_values);
            free(data);
            return 1;
        }
    }
    
    fclose(Method_update_avg_);

    free(x_values);
    free(y_values);
    free(data);

    return 0;

}