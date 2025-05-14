#include <stdio.h>

#define MEAN 0.0 
#define SIGMA 0.1
#define BUFFER_SIZE 1000

void open_file(FILE **file, const char *filename, const char *mode);
void read_values_from_file(const char* filename, double* x_values, double* y_values, int* count);
