import numpy as np 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x_values_f2, y_values_f2 = [], []
    x_values_noise, y_values_noise = [], []

    with open('/Users/zborivskyi/Documents/3course/Data analysis /lab3/files/Funct_with_GausseNoise.txt', 'r') as file:
        for line in file:
            x, y = map(float, line.split())
            x_values_noise.append(x)
            y_values_noise.append(y)

    
    with open('/Users/zborivskyi/Documents/3course/Data analysis /lab3/files/Funct_2.txt', 'r') as file:
        for line in file:
            x, y = map(float, line.split())
            x_values_f2.append(x)
            y_values_f2.append(y)

    plt.plot(x_values_f2, y_values_f2, label='Funct_2')
    plt.plot(x_values_noise, y_values_noise, label='Funct_with_GausseNoise')
    plt.legend()
    plt.show()

    x_least_quad, y_least_quad = [], []
    x_lest_linear, y_lest_linear = [], []

    with open('/Users/zborivskyi/Documents/3course/Data analysis /lab3/files/Least_squares_quadratics.txt', 'r') as file:
        for line in file:
            x, y = map(float, line.split())
            x_least_quad.append(x)
            y_least_quad.append(y)

    with open('/Users/zborivskyi/Documents/3course/Data analysis /lab3/files/Least_squares_linear.txt', 'r') as file:
        for line in file:
            x, y = map(float, line.split())
            x_lest_linear.append(x)
            y_lest_linear.append(y)

    plt.plot(x_values_f2, y_values_f2, label='Funct_2')
    plt.plot(x_values_noise, y_values_noise, label='Funct_with_GausseNoise')
    plt.plot(x_least_quad, y_least_quad, label='Least_squares_quad')
    plt.plot(x_lest_linear, y_lest_linear, label='Least_squares_linear')
    plt.legend()
    plt.show()

