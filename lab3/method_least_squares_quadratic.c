#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "files.h"

#define BUFFER_SIZE 1000


void solve_system(double A[3][3], double b[3], double x[3]) {
    int n = 3;
    for (int i = 0; i < n; i++) {
        int max_i = i;
        for (int j = i + 1; j < n; j++)
            if (fabs(A[j][i]) > fabs(A[max_i][i]))
                max_i = j;
        
        if (max_i != i) {
            for (int j = 0; j < n; j++) {
                double temp = A[i][j];
                A[i][j] = A[max_i][j];
                A[max_i][j] = temp;
            }
            double temp = b[i];
            b[i] = b[max_i];
            b[max_i] = temp;
        }

        for (int j = i + 1; j < n; j++) {
            double factor = A[j][i] / A[i][i];
            b[j] -= factor * b[i];
            for (int k = i; k < n; k++)
                A[j][k] -= factor * A[i][k];
        }
    }

    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++)
            x[i] -= A[i][j] * x[j];
        x[i] /= A[i][i];
    }
}

int main() {
    double x_values[BUFFER_SIZE], y_values[BUFFER_SIZE];
    int count = 0;

    if (read_values_from_file("files/Funct_quadratic_with_GausseNoise.txt", 
                             x_values, y_values, &count) != 0) {
        return 1;
    }

    double sum_x = 0, sum_x2 = 0, sum_x3 = 0, sum_x4 = 0;
    double sum_y = 0, sum_xy = 0, sum_x2y = 0;

    for (int i = 0; i < count; i++) {
        double x = x_values[i];
        double x2 = x * x;
        double y = y_values[i];

        sum_x += x;
        sum_x2 += x2;
        sum_x3 += x2 * x;
        sum_x4 += x2 * x2;
        sum_y += y;
        sum_xy += x * y;
        sum_x2y += x2 * y;
    }

    double A[3][3] = {
        {sum_x4, sum_x3, sum_x2},
        {sum_x3, sum_x2, sum_x},
        {sum_x2, sum_x, count}
    };
    double b[3] = {sum_x2y, sum_xy, sum_y};
    double coeffs[3];

    solve_system(A, b, coeffs);

    FILE *output = fopen("files/Least_squares_quadratic.txt", "w");
    
    double min_x = x_values[0];
    double max_x = x_values[count-1];
    double step = (max_x - min_x) / 100;

    for (double x = min_x; x <= max_x; x += step) {
        double y = coeffs[0] * x * x + coeffs[1] * x + coeffs[2];
        fprintf(output, "%.2f %.4f\n", x, y);
    }

    fclose(output);
    return 0;
}
