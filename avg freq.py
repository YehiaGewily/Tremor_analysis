import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import welch
from scipy import stats
from concurrent.futures import ProcessPoolExecutor

###############################################################################
#                            Configuration Settings                           #
###############################################################################
# File paths
OUTPUT_FOLDER = r'E:\Tremor\visualizations'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Signal processing parameters
SAMPLING_RATE = 30  # Hz
SEGMENT_LENGTH = 256
OVERLAP = 192
FREQ_RANGE = (3, 12)  # Define the frequency range of interest

# Define file paths manually for each day
# Replace these with your actual file paths
before_files = [
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay1Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay2Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay3Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay4Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay5Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay6Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay7Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay8Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay9Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay10Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay11Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay12Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay13Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay14Before_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data\NewDay15Before_euclidean_distances.csv'
]

after_files = [
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay1After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay2After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay3After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay4After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay5After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay6After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay7After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay8After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay9After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay10After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay11After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay12After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay13After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay14After_euclidean_distances.csv',
    r'E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data\NewDay15After_euclidean_distances.csv'
]


###############################################################################
#                            Analysis Functions                               #
###############################################################################
def calculate_weighted_frequency(file_path):
    """
    Compute power-weighted average frequency within a given range.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing tremor data
    
    Returns:
    --------
    float
        Power-weighted average frequency or NaN if processing fails
    """
    try:
        # Read data file
        df = pd.read_csv(file_path, header=None, skiprows=1)
        
        # Basic validation
        if df.empty:
            print(f"Warning: Empty file {file_path}")
            return np.nan
            
        # Extract signal column and convert to float
        signal = df.iloc[:, 1].astype(float).values
        
        # Check for valid signal
        if signal.size < SEGMENT_LENGTH:
            print(f"Warning: Signal too short in {file_path}")
            return np.nan
        
        # Calculate power spectral density
        freq, psd = welch(signal, SAMPLING_RATE, nperseg=SEGMENT_LENGTH, noverlap=OVERLAP)
        
        # Filter only desired frequency range
        mask = (freq >= FREQ_RANGE[0]) & (freq <= FREQ_RANGE[1])
        freq, psd = freq[mask], psd[mask]
        
        # Calculate weighted average frequency
        if np.sum(psd) > 0:
            return np.sum(freq * psd) / np.sum(psd)
        else:
            print(f"Warning: No power in frequency range for {file_path}")
            return np.nan
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.nan

def process_files(file_list):
    """
    Parallel processing of files for efficiency.
    
    Parameters:
    -----------
    file_list : list
        List of file paths to process
        
    Returns:
    --------
    list
        List of weighted frequencies for each file
    """
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(calculate_weighted_frequency, file_list))
    return results

def perform_statistical_analysis(before_avg, after_avg):
    """
    Perform statistical analysis on before and after data.
    
    Parameters:
    -----------
    before_avg : array
        Average frequencies before stimulation
    after_avg : array
        Average frequencies after stimulation
        
    Returns:
    --------
    dict
        Dictionary containing statistical results
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(before_avg, after_avg)
    
    # Effect size (Cohen's d)
    effect_size = np.mean(after_avg - before_avg) / np.std(before_avg) if np.std(before_avg) > 0 else np.nan
    
    # Mean and standard deviation
    mean_diff = np.mean(after_avg - before_avg)
    std_diff = np.std(after_avg - before_avg)
    
    # Percent change
    percent_change = 100 * np.mean((after_avg - before_avg) / before_avg)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'percent_change': percent_change
    }

###############################################################################
#                             Main Analysis                                   #
###############################################################################
def main():
    """Main analysis function."""
    print("Starting tremor analysis...")
    
    # Calculate averages in parallel
    print(f"Processing {len(before_files)} files for before stimulation...")
    before_avg = process_files(before_files)
    
    print(f"Processing {len(after_files)} files for after stimulation...")
    after_avg = process_files(after_files)
    
    # Filter out NaN values
    valid_indices = [i for i, (b, a) in enumerate(zip(before_avg, after_avg)) if not (np.isnan(b) or np.isnan(a))]
    
    if not valid_indices:
        print("Error: No valid data points after processing")
        return
        
    before_avg = [before_avg[i] for i in valid_indices]
    after_avg = [after_avg[i] for i in valid_indices]
    
    # Create day numbers
    days = np.arange(1, len(before_avg) + 1)
    
    # Convert to numpy arrays
    before_avg = np.array(before_avg)
    after_avg = np.array(after_avg)
    
    # Statistical analysis
    stats_results = perform_statistical_analysis(before_avg, after_avg)
    
    ###########################################################################
    #                             Visualization                               #
    ###########################################################################
    # Timeline plot
    plt.figure(figsize=(12, 6))
    plt.plot(days, before_avg, 'bo-', markersize=8, label='Before Stimulation')
    plt.plot(days, after_avg, 'rs--', markersize=8, label='After Stimulation')
    plt.title('Tremor Frequency Timeline', fontsize=14)
    plt.xlabel('Day Number', fontsize=12)
    plt.ylabel('Average Frequency (Hz)', fontsize=12)
    plt.xticks(days)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Add value labels
    for d, b, a in zip(days, before_avg, after_avg):
        plt.text(d, b, f'{b:.2f}', ha='center', va='bottom', color='blue')
        plt.text(d, a, f'{a:.2f}', ha='center', va='top', color='red')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'Frequency_Timeline.png'), dpi=300)
    plt.close()
    
    # Bar chart for differences
    plt.figure(figsize=(12, 5))
    diff = after_avg - before_avg
    bars = plt.bar(days, diff, color=['g' if d < 0 else 'r' for d in diff])
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Frequency Change After Stimulation', fontsize=14)
    plt.xlabel('Day Number', fontsize=12)
    plt.ylabel('Change in Frequency (Hz)', fontsize=12)
    plt.xticks(days)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, diff):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + 0.1 if height > 0 else height - 0.2,
                f'{value:.2f}', 
                ha='center', va='bottom' if height > 0 else 'top',
                color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'Frequency_Change.png'), dpi=300)
    plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Day': days,
        'Before_Stimulation': before_avg,
        'After_Stimulation': after_avg,
        'Difference': after_avg - before_avg,
        'Percent_Change': 100 * (after_avg - before_avg) / before_avg
    })
    results_df.to_csv(os.path.join(OUTPUT_FOLDER, 'frequency_results.csv'), index=False)
    
    # Save statistical results
    with open(os.path.join(OUTPUT_FOLDER, 'statistical_analysis.txt'), 'w') as f:
        f.write("Statistical Analysis Results\n")
        f.write("===========================\n\n")
        f.write(f"Number of samples: {len(before_avg)}\n")
        f.write(f"T-statistic: {stats_results['t_statistic']:.4f}\n")
        f.write(f"P-value: {stats_results['p_value']:.4f}\n")
        f.write(f"Effect size (Cohen's d): {stats_results['effect_size']:.4f}\n")
        f.write(f"Mean difference: {stats_results['mean_difference']:.4f} Hz\n")
        f.write(f"Standard deviation of difference: {stats_results['std_difference']:.4f} Hz\n")
        f.write(f"Average percent change: {stats_results['percent_change']:.2f}%\n")
        f.write("\nInterpretation:\n")
        if stats_results['p_value'] < 0.05:
            f.write("There is a statistically significant difference between before and after stimulation.\n")
        else:
            f.write("There is no statistically significant difference between before and after stimulation.\n")
            
    print(f"Analysis complete. Results saved to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()