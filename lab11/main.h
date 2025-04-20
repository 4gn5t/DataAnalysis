#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE_LENGTH 1024
#define MAX_FEATURES 100
#define MAX_SAMPLES 10000

#define CSV_EXT ".csv"
#define XLS_EXT ".xls"
#define XLSX_EXT ".xlsx"

typedef enum {
    SIGMOID,
    TANH,
    STEP,
    RELU
} ActivationFunction;

typedef struct {
    double* weights;
    double bias;
    double* delta_weights_prev;
    double delta_bias_prev;
    int num_features;
    ActivationFunction activation;
    
    int* epochs;
    double* train_errors;
    double* val_errors;
    int log_size;
    int log_capacity;
} SingleNeuronClassifier;

typedef struct {
    double** X;       
    int* y;           
    int num_samples;
    int num_features;
    
    double** X_train;
    int* y_train;
    int num_train;
    
    double** X_val;
    int* y_val;
    int num_val;
    
    double** X_test;
    int* y_test;
    int num_test;
    
    double* mean;
    double* std_dev;
} Dataset;

SingleNeuronClassifier* create_neuron(int num_features, ActivationFunction activation);
void free_neuron(SingleNeuronClassifier* neuron);
double sigmoid(double x);
double sigmoid_derivative(double x);
double tanh_activation(double x);
double tanh_derivative(double x);
double step_activation(double x);
double step_derivative(double x);
void initialize_weights(SingleNeuronClassifier* neuron);
double forward(SingleNeuronClassifier* neuron, double* x);
double compute_error(int* y_true, double* y_pred, int n);
void add_to_log(SingleNeuronClassifier* neuron, int epoch, double train_error, double val_error);
int fit(SingleNeuronClassifier* neuron, Dataset* data, double alpha, double nu, double E0, int max_epochs);
void predict(SingleNeuronClassifier* neuron, double** X, int n, int* predictions, double threshold);
double compute_accuracy(int* y_true, int* y_pred, int n);

Dataset* load_data(const char* filename, int has_header, int target_col);
void normalize_data(Dataset* data);
void split_data(Dataset* data, double train_ratio, double val_ratio);
void free_dataset(Dataset* data);

void print_vector(double* vec, int n);
void print_training_info(SingleNeuronClassifier* neuron, int epochs, double train_error, double val_error);
void print_neuron_info(SingleNeuronClassifier* neuron);
void print_data_info(Dataset* data);
void print_usage();
int is_excel_file(const char* filename);
int convert_excel_to_csv(const char* excel_file, const char* csv_file);

void free_dataset(Dataset* data) {
    for (int i = 0; i < data->num_samples; i++) {
        free(data->X[i]);
    }
    free(data->X);
    free(data->y);
    
    free(data->y_train);
    free(data->y_val);
    free(data->y_test);
    
    free(data->mean);
    free(data->std_dev);
    
    free(data);
}

void print_vector(double* vec, int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%.4f", vec[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

void print_training_info(SingleNeuronClassifier* neuron, int epochs, double train_error, double val_error) {
    printf("Training completed after %d epochs\n", epochs);
    printf("Final training error: %.6f\n", train_error);
    printf("Final validation error: %.6f\n", val_error);
    printf("\n");
}

void print_neuron_info(SingleNeuronClassifier* neuron) {
    printf("Neuron weights: ");
    print_vector(neuron->weights, neuron->num_features);
    printf("Bias: %.4f\n", neuron->bias);
    printf("Activation function: ");
    switch (neuron->activation) {
        case SIGMOID: printf("sigmoid\n"); break;
        case TANH: printf("tanh\n"); break;
        case STEP: printf("step\n"); break;
    }
}

void print_data_info(Dataset* data) {
    printf("Dataset information:\n");
    printf("Total samples: %d\n", data->num_samples);
    printf("Number of features: %d\n", data->num_features);
    printf("\n");
}

void print_usage() {
    printf("Usage:\n");
    printf("  train <data_file> <has_header> <target_col> <train_ratio> <val_ratio> <activation> [alpha] [nu] [E0] [max_epochs]\n");
    printf("Example:\n");
    printf("  ./single_neuron_cli train Лаб1_6.xlsx 1 0 0.6 0.2 sigmoid 0.01 0.0 0.01 2000\n");
    printf("  ./single_neuron_cli all Лаб1_6.xlsx\n");
}

int convert_excel_to_csv(const char* excel_file, const char* csv_file) {
    char command[512];
    
    snprintf(command, sizeof(command), 
             "python3 -c \"import pandas as pd; pd.read_excel('%s').to_csv('%s', index=False)\"",
             excel_file, csv_file);
    
    int result = system(command);
    if (result != 0) {
        printf("Error: Failed to convert Excel file.\n");
        return 0;
    }
    
    return 1;
}

void free_neuron(SingleNeuronClassifier* neuron) {
    free(neuron->weights);
    free(neuron->delta_weights_prev);
    free(neuron->epochs);
    free(neuron->train_errors);
    free(neuron->val_errors);
    free(neuron);
}


void save_predictions_to_csv(int* y_true, int* y_pred, double** X, int n, int num_features, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot open file %s for writing predictions\n", filename);
        return;
    }
    
    fprintf(file, "actual,predicted");
    for (int i = 0; i < num_features; i++) {
        fprintf(file, ",feature_%d", i+1);
    }
    fprintf(file, "\n");
    
    for (int i = 0; i < n; i++) {
        fprintf(file, "%d,%d", y_true[i], y_pred[i]);
        for (int j = 0; j < num_features; j++) {
            fprintf(file, ",%f", X[i][j]);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
    printf("Predictions saved to %s\n", filename);
}