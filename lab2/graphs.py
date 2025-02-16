import numpy as np
import matplotlib.pyplot as plt    

if __name__ == '__main__':
    x_values_f2, y_values_f2 = [], []
    x_values_noise, y_values_noise = [], []

    with open('/Users/zborivskyi/Documents/3course/Data analysis /lab2/files/Funct_with_GausseNoise.txt', 'r') as file:
        for line in file:
            x, y = map(float, line.split())
            x_values_noise.append(x)
            y_values_noise.append(y)

    
    with open('/Users/zborivskyi/Documents/3course/Data analysis /lab2/files/Funct_2.txt', 'r') as file:
        for line in file:
            x, y = map(float, line.split())
            x_values_f2.append(x)
            y_values_f2.append(y)

    plt.plot(x_values_f2, y_values_f2, label='Funct_2')
    plt.plot(x_values_noise, y_values_noise, label='Funct_with_GausseNoise')
    plt.legend()
    plt.show()

    x_values_avg, y_values_avg = [], []

    with open('/Users/zborivskyi/Documents/3course/Data analysis /lab2/files/Moving_average.txt', 'r') as file:
        for line in file:
            x, y = map(float, line.split())
            x_values_avg.append(x)
            y_values_avg.append(y)
    
    plt.plot(x_values_f2, y_values_f2, label='Funct_2')
    plt.plot(x_values_noise, y_values_noise, label='Funct_with_GausseNoise')
    plt.plot(x_values_avg, y_values_avg, label='Moving_average')
    plt.legend()
    plt.show()

    x_values_avg_next, y_values_avg_next = [], []

    with open('/Users/zborivskyi/Documents/3course/Data analysis /lab2/files/Moving_average_next.txt', 'r') as file:
        for line in file:
            x, y = map(float, line.split())
            x_values_avg_next.append(x)
            y_values_avg_next.append(y)
    
    plt.plot(x_values_f2, y_values_f2, label='Funct_2')
    plt.plot(x_values_noise, y_values_noise, label='Funct_with_GausseNoise')
    plt.plot(x_values_avg_next, y_values_avg_next, label='Moving_average_next')
    plt.legend()
    plt.show() 

    x_values_exp, y_values_exp = [], []

    with open('/Users/zborivskyi/Documents/3course/Data analysis /lab2/files/Exp_smoothing.txt', 'r') as file:
        for line in file:
            x, y = map(float, line.split())
            x_values_exp.append(x)
            y_values_exp.append(y)
    
    #plt.plot(x_values_f2, y_values_f2, label='Funct_2')
    plt.plot(x_values_noise, y_values_noise, label='Funct_with_GausseNoise')
    plt.plot(x_values_exp, y_values_exp, label='Exp_smoothing')
    plt.legend()
    plt.show()



