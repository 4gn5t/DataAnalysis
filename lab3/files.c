#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "files.h"

int read_values_from_file(const char* filename, double* x_values, double* y_values, int* count) {
    FILE* file;
    file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file for reading");
        return 1;
    }

    // If we're just counting items
    if (x_values == NULL || y_values == NULL) {
        *count = 0;
        char line[256];
        while (fgets(line, sizeof(line), file)) {
            (*count)++;
        }
        fclose(file);
        return 0;
    }

    *count = 0;
    while (fscanf(file, "%lf %lf", &x_values[*count], &y_values[*count]) == 2) {
        (*count)++;
        if (*count >= BUFFER_SIZE) {
            printf("Warning: Buffer full\n");
            break;
        }
    }

    fclose(file);
    return 0;
}


void open_file(FILE **file, const char *filename, const char *mode) {
    *file = fopen(filename, mode);
    if (*file == NULL) {
        perror("Error opening file");
        exit(1);
    }
}
