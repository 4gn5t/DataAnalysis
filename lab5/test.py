import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# Constants
BUFFER_SIZE = 50000
MEAN = 5.0
SIGMA = 3.5
NUM_EXPERIMENTS = 100

class StatisticsValues:
    def __init__(self, mean, median, mode, variance, standard_deviation):
        self.mean = mean
        self.median = median
        self.mode = mode
        self.variance = variance
        self.standard_deviation = standard_deviation

def generate_random_sequence(size):
    """Generate random numbers using Box-Muller transform"""
    return np.random.normal(MEAN, SIGMA, size)

def calculate_statistics(values):
    """Calculate statistical measures for the given sequence"""
    mean = np.mean(values)
    median = np.median(values)
    # Calculate mode by rounding to 4 decimal places first
    rounded_values = np.round(values, 4)
    mode_result = stats.mode(rounded_values)
    # Handle both old and new scipy versions
    mode = mode_result.mode if isinstance(mode_result.mode, np.float64) else mode_result.mode[0]
    variance = np.var(values)
    std_dev = np.std(values)
    
    return StatisticsValues(mean, median, mode, variance, std_dev)

def save_sequence(values, filename):
    """Save the generated sequence to a file"""
    np.savetxt(filename, values, fmt='%.4f')

def run_experiments():
    """Run multiple experiments and track errors"""
    errors = []
    
    for experiment in range(1, NUM_EXPERIMENTS + 1):
        values = generate_random_sequence(BUFFER_SIZE)
        stats = calculate_statistics(values)
        
        errors.append({
            'experiment': experiment,
            'mean_error': abs(stats.mean - MEAN),
            'std_dev_error': abs(stats.standard_deviation - SIGMA)
        })
        
        if experiment % 10 == 0:
            print(f"Progress: {experiment}%")
    
    return pd.DataFrame(errors)

def plot_analysis(random_values, errors_df):
    """Create visualization plots"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Distribution of random numbers
    plt.subplot(2, 2, 1)
    counts, bins, _ = plt.hist(random_values, bins=50, density=True, 
                              alpha=0.7, color='blue', 
                              label='Generated Values')
    
    x = np.linspace(min(bins), max(bins), 100)
    pdf = stats.norm.pdf(x, MEAN, SIGMA)
    plt.plot(x, pdf, 'r-', lw=2, label='Theoretical Distribution')
    plt.title('Distribution of Generated Random Numbers')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Mean error over experiments
    plt.subplot(2, 2, 2)
    plt.plot(errors_df['experiment'], errors_df['mean_error'],
             label='Mean Error')
    plt.title('Mean Value Error')
    plt.xlabel('Experiment Number')
    plt.ylabel('Absolute Error')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Standard deviation error over experiments
    plt.subplot(2, 2, 3)
    plt.plot(errors_df['experiment'], errors_df['std_dev_error'],
             color='red', label='Std Dev Error')
    plt.title('Standard Deviation Error')
    plt.xlabel('Experiment Number')
    plt.ylabel('Absolute Error')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Error correlation
    plt.subplot(2, 2, 4)
    plt.scatter(errors_df['mean_error'], errors_df['std_dev_error'],
               alpha=0.6, color='purple')
    plt.title('Error Correlation')
    plt.xlabel('Mean Error')
    plt.ylabel('Standard Deviation Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # Create output directory if it doesn't exist
    os.makedirs('files', exist_ok=True)
    
    # Generate and save initial sequence
    random_values = generate_random_sequence(BUFFER_SIZE)
    save_sequence(random_values, 'files/Random_sequence.txt')
    
    # Run experiments and save errors
    errors_df = run_experiments()
    errors_df.to_csv('files/errors.txt', index=False)
    
    # Calculate and display final statistics
    final_stats = calculate_statistics(random_values)
    print("\nFinal Statistics:")
    print(f"Target Mean: {MEAN:.4f}, Actual Mean: {final_stats.mean:.4f}, "
          f"Error: {abs(final_stats.mean - MEAN):.4f}")
    print(f"Target Std Dev: {SIGMA:.4f}, Actual Std Dev: {final_stats.standard_deviation:.4f}, "
          f"Error: {abs(final_stats.standard_deviation - SIGMA):.4f}")
    print(f"Median: {final_stats.median:.4f}")
    print(f"Mode: {final_stats.mode:.4f}")
    print(f"Variance: {final_stats.variance:.4f}")
    
    # Create visualization plots
    plot_analysis(random_values, errors_df)

if __name__ == "__main__":
    main()