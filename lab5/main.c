#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string.h>

#define BUFFER_SIZE 50000
#define MEAN 5.0
#define SIGMA 3.5
#define NUM_EXPERIMENTS 100

typedef struct Statistics_values {
    double mean;
    double median;
    double mode;
    double variance;
    double standard_deviation;
} Statistics_values;

static uint64_t xoshiro256_state[4];

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t xoshiro256_next(void) {
    const uint64_t result = rotl(xoshiro256_state[1] * 5, 7) * 9;
    const uint64_t t = xoshiro256_state[1] << 17;
    xoshiro256_state[2] ^= xoshiro256_state[0];
    xoshiro256_state[3] ^= xoshiro256_state[1];
    xoshiro256_state[1] ^= xoshiro256_state[2];
    xoshiro256_state[0] ^= xoshiro256_state[3];
    xoshiro256_state[2] ^= t;
    xoshiro256_state[3] = rotl(xoshiro256_state[3], 45);
    return result;
}

void init_random_state() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t seed = ((uint64_t)tv.tv_sec << 32) | tv.tv_usec;
    for(int i = 0; i < 4; i++) {
        xoshiro256_state[i] = seed + i;
        seed = xoshiro256_next();
    }
}

double random_double() {
    return (xoshiro256_next() >> 11) * 0x1.0p-53;
}

double generate_normal(double mean, double std_dev) {
    double u1, u2;
    do {
        u1 = random_double();
    } while (u1 <= 0.0);
    u2 = random_double();
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + std_dev * z;
}

void generate_random_sequence(double* values, int count) {
    #pragma omp parallel 
    {
        unsigned int thread_num = omp_get_thread_num();
        uint64_t local_state[4];
        for(int i = 0; i < 4; i++) {
            local_state[i] = xoshiro256_state[i] ^ thread_num;
        }
        #pragma omp for
        for (int i = 0; i < count; i++) {
            values[i] = generate_normal(MEAN, SIGMA);
        }
    }
}

int compare_doubles(const void* a, const void* b) {
    double arg1 = *(const double*)a;
    double arg2 = *(const double*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

double calculate_median(double* sorted_values, int count) {
    if (count % 2 == 0) {
        return (sorted_values[count/2 - 1] + sorted_values[count/2]) / 2.0;
    }
    return sorted_values[count / 2];
}

double calculate_mode(double* sorted_values, int count) {
    double mode = sorted_values[0];
    int max_count = 1;
    int current_count = 1;
    for (int i = 1; i < count; i++) {
        if (fabs(sorted_values[i] - sorted_values[i-1]) < 1e-10) {
            current_count++;
        } else {
            if (current_count > max_count) {
                max_count = current_count;
                mode = sorted_values[i - 1];
            }
            current_count = 1;
        }
    }
    if (current_count > max_count) {
        mode = sorted_values[count - 1];
    }
    return mode;
}

Statistics_values calculate_statistics(double* values, int count) {
    Statistics_values stats;
    double* sorted_values = (double*)malloc(count * sizeof(double));
    memcpy(sorted_values, values, count * sizeof(double));
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < count; i++) {
        sum += values[i];
    }
    stats.mean = sum / count;
    double sum_sq_diff = 0.0;
    #pragma omp parallel for reduction(+:sum_sq_diff)
    for (int i = 0; i < count; i++) {
        double diff = values[i] - stats.mean;
        sum_sq_diff += diff * diff;
    }
    stats.variance = sum_sq_diff / count;
    stats.standard_deviation = sqrt(stats.variance);
    qsort(sorted_values, count, sizeof(double), compare_doubles);
    stats.median = calculate_median(sorted_values, count);
    stats.mode = calculate_mode(sorted_values, count);
    free(sorted_values);
    return stats;
}

int main() {
    init_random_state();
    omp_set_num_threads(omp_get_num_procs());
    double* random_values = (double*)malloc(BUFFER_SIZE * sizeof(double));
    if (!random_values) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }
    generate_random_sequence(random_values, BUFFER_SIZE);
    FILE* random_sequence = fopen("files/Random_sequence.txt", "w");
    if (!random_sequence) {
        perror("Error opening Random_sequence.txt");
        free(random_values);
        return 1;
    }
    for (int i = 0; i < BUFFER_SIZE; i++) {
        fprintf(random_sequence, "%.6f\n", random_values[i]);
    }
    fclose(random_sequence);
    FILE* errors = fopen("files/errors.txt", "w");
    if (!errors) {
        perror("Error opening errors.txt");
        free(random_values);
        return 1;
    }
    fprintf(errors, "experiment,mean_error,std_dev_error\n");
    for (int experiment = 1; experiment <= NUM_EXPERIMENTS; experiment++) {
        generate_random_sequence(random_values, BUFFER_SIZE);
        Statistics_values stats = calculate_statistics(random_values, BUFFER_SIZE);
        fprintf(errors, "%d,%.6f,%.6f\n", experiment, fabs(stats.mean - MEAN), fabs(stats.standard_deviation - SIGMA));
        if (experiment % 10 == 0) {
            printf("Progress: %d%%\n", experiment);
        }
    }
    fclose(errors);
    Statistics_values final_stats = calculate_statistics(random_values, BUFFER_SIZE);
    printf("\nFinal Statistics:\n");
    printf("Target Mean: %.6f, Actual Mean: %.6f, Error: %.6f\n", MEAN, final_stats.mean, fabs(final_stats.mean - MEAN));
    printf("Target Std Dev: %.6f, Actual Std Dev: %.6f, Error: %.6f\n", SIGMA, final_stats.standard_deviation, fabs(final_stats.standard_deviation - SIGMA));
    printf("Median: %.6f\n", final_stats.median);
    printf("Mode: %.6f\n", final_stats.mode);
    printf("Variance: %.6f\n", final_stats.variance);
    free(random_values);
    return 0;
}