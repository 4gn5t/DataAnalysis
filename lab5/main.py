import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def plot_analysis():
    random_values = np.loadtxt('/Users/zborivskyi/Documents/3course/Data analysis /lab5/files/Random_sequence.txt')
        
    errors_df = pd.read_csv('/Users/zborivskyi/Documents/3course/Data analysis /lab5/files/errors.txt')


    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    counts, bins, _ = plt.hist(random_values, bins=50, density=True, 
                              alpha=0.7, color='blue', 
                              label='Згенеровані значення')
    

    x = np.linspace(min(bins), max(bins), 100)
    pdf = stats.norm.pdf(x, 5.0, 3.5) 
    plt.plot(x, pdf, 'r-', lw=2, label='Теоретичний розподіл')
    plt.title('Розподіл згенерованих випадкових чисел')
    plt.xlabel('Значення')
    plt.ylabel('Щільність')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(errors_df['experiment'], errors_df['mean_error'], 
             label='Похибка середнього')
    plt.title('Похибка середнього значення')
    plt.xlabel('Номер експерименту')
    plt.ylabel('Абсолютна похибка')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(errors_df['experiment'], errors_df['std_dev_error'], 
             color='red', label='Похибка станд. відхилення')
    plt.title('Похибка стандартного відхилення')
    plt.xlabel('Номер експерименту')
    plt.ylabel('Абсолютна похибка')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.scatter(errors_df['mean_error'], errors_df['std_dev_error'], 
               alpha=0.6, color='purple')
    plt.title('Залежність між похибками')
    plt.xlabel('Похибка середнього')
    plt.ylabel('Похибка стандартного відхилення')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_analysis()