import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from scipy.signal import savgol_filter
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle

# Set Seaborn style for more attractive plots
sns.set(style="darkgrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def load_tremor_data(data_dir, file_pattern="*.csv"):
    """
    Load tremor data files with enhanced error handling and display.
    """
    all_data = []
    file_paths = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    
    print(f"Loading {len(file_paths)} tremor data files...")
    
    for i, file_path in enumerate(file_paths):
        try:
            df = pd.read_csv(file_path)
            if len(df.columns) >= 2:
                df = df.iloc[:, 0:2]
                df.columns = ['Frame', 'Amplitude']
                df['Day'] = i + 1
                all_data.append(df)
                print(f"  Loaded day {i+1}: {os.path.basename(file_path)} - {len(df)} frames")
            else:
                print(f"  Warning: File {file_path} doesn't have enough columns")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    return all_data

def create_fancy_tremor_plots(historical_data, synthetic_data, output_dir, output_filename="fancy_tremor_visualization.png"):
    """
    Create fancy visualizations of tremor data with advanced styling.
    
    Parameters:
    - historical_data: List of DataFrames containing historical tremor data
    - synthetic_data: List of DataFrames containing synthetic tremor data
    - output_dir: Directory to save output plots
    - output_filename: Filename for the output visualization
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define number of plots to show
    num_historical = min(3, len(historical_data))
    num_synthetic = min(5, len(synthetic_data))
    
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(num_historical + num_synthetic + 2, 1, height_ratios=[0.7] + [1] * (num_historical + num_synthetic) + [0.5])
    
    # Create a title subplot
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, "Tremor Amplitude Analysis & Prediction", 
                  fontsize=24, ha='center', va='center', fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.6, edgecolor='blue'))
    
    # Create color palettes
    historical_cmap = cm.Blues
    synthetic_cmap = cm.Reds
    
    # Keep track of global stats for colormap normalization
    all_means = []
    all_maxes = []
    all_stds = []
    
    # Collect statistics from all datasets
    for df in historical_data[:num_historical] + synthetic_data[:num_synthetic]:
        all_means.append(df['Amplitude'].mean())
        all_maxes.append(df['Amplitude'].max())
        all_stds.append(df['Amplitude'].std())
    
    mean_range = (min(all_means) * 0.9, max(all_means) * 1.1)
    max_range = (min(all_maxes) * 0.9, max(all_maxes) * 1.1)
    std_range = (min(all_stds) * 0.9, max(all_stds) * 1.1)
    
    # Plot historical data
    for i in range(num_historical):
        df = historical_data[i]
        day = i + 1
        
        # Create subplot
        ax = fig.add_subplot(gs[i+1])
        
        # Calculate statistics
        mean_amp = df['Amplitude'].mean()
        max_amp = df['Amplitude'].max()
        std_amp = df['Amplitude'].std()
        
        # Determine color based on mean amplitude
        color_intensity = (mean_amp - mean_range[0]) / (mean_range[1] - mean_range[0])
        color_intensity = min(max(color_intensity, 0.2), 0.9)  # Keep within reasonable range
        line_color = historical_cmap(color_intensity)
        
        # Create a smoothed line for trend
        window_size = min(51, len(df) // 10 * 2 + 1)  # Ensure window size is odd
        if window_size >= 3:
            try:
                smoothed = savgol_filter(df['Amplitude'], window_size, 3)
                ax.plot(df['Frame'], smoothed, color='navy', alpha=0.5, linewidth=1.5)
            except Exception as e:
                print(f"Error creating smoothed trend for day {day}: {e}")
        
        # Plot main data
        ax.plot(df['Frame'], df['Amplitude'], color=line_color, linewidth=0.8, alpha=0.8)
        
        # Fill between x-axis and line
        ax.fill_between(df['Frame'], df['Amplitude'], color=line_color, alpha=0.2)
        
        # Add statistics box
        stats_text = f"Mean: {mean_amp:.2f}   |   Max: {max_amp:.2f}   |   StdDev: {std_amp:.2f}"
        ax.text(0.02, 0.93, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.8, edgecolor=line_color))
        
        # Add horizontal line at mean
        ax.axhline(mean_amp, color='black', linestyle='--', alpha=0.3)
        
        # Customize appearance
        ax.set_title(f'Historical Data - Day {day}', fontweight='bold', color='navy')
        ax.set_ylabel('Amplitude')
        
        # Show y-axis only on left side, x-axis only on bottom
        if i < num_historical - 1:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Frame')
        
        # Set y-limits with some padding
        ax.set_ylim(bottom=0, top=max_amp * 1.1)
        
        # Grid styling
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add background shading for visual separation
        ax.patch.set_facecolor('aliceblue')
        ax.patch.set_alpha(0.3)
    
    # Plot synthetic data
    for i in range(num_synthetic):
        df = synthetic_data[i]
        day = len(historical_data) + i + 1
        
        # Extract method from filename if available
        method = "model"  # Default
        if 'method' in df.columns:
            method = df['method'].iloc[0]
        
        # Create subplot
        ax = fig.add_subplot(gs[i+num_historical+1])
        
        # Calculate statistics
        mean_amp = df['Amplitude'].mean()
        max_amp = df['Amplitude'].max()
        std_amp = df['Amplitude'].std()
        
        # Determine color based on mean amplitude
        color_intensity = (mean_amp - mean_range[0]) / (mean_range[1] - mean_range[0])
        color_intensity = min(max(color_intensity, 0.2), 0.9)  # Keep within reasonable range
        line_color = synthetic_cmap(color_intensity)
        
        # Create a smoothed line for trend
        window_size = min(51, len(df) // 10 * 2 + 1)  # Ensure window size is odd
        if window_size >= 3:
            try:
                smoothed = savgol_filter(df['Amplitude'], window_size, 3)
                ax.plot(df['Frame'], smoothed, color='darkred', alpha=0.5, linewidth=1.5)
            except Exception as e:
                print(f"Error creating smoothed trend for day {day}: {e}")
                
        # Plot main data
        ax.plot(df['Frame'], df['Amplitude'], color=line_color, linewidth=0.8, alpha=0.8)
        
        # Fill between x-axis and line
        ax.fill_between(df['Frame'], df['Amplitude'], color=line_color, alpha=0.2)
        
        # Add statistics box
        stats_text = f"Mean: {mean_amp:.2f}   |   Max: {max_amp:.2f}   |   StdDev: {std_amp:.2f}"
        ax.text(0.02, 0.93, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.8, edgecolor=line_color))
        
        # Add horizontal line at mean
        ax.axhline(mean_amp, color='black', linestyle='--', alpha=0.3)
        
        # Customize appearance
        ax.set_title(f'Synthetic Data - Day {day} ({method})', fontweight='bold', color='darkred')
        ax.set_ylabel('Amplitude')
        
        # Show y-axis only on left side, x-axis only on bottom
        if i < num_synthetic - 1:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Frame')
        
        # Set y-limits with some padding
        ax.set_ylim(bottom=0, top=max_amp * 1.1)
        
        # Grid styling
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add background shading for visual separation
        ax.patch.set_facecolor('mistyrose')
        ax.patch.set_alpha(0.3)
    
    # Add legend/explanation subplot at the bottom
    ax_legend = fig.add_subplot(gs[-1])
    ax_legend.axis('off')
    
    # Create explanation text
    legend_text = """
    Visualization Details:
    • Historical data (blue): Original tremor recordings from patients
    • Synthetic data (red): Predicted tremor patterns for future days
    • Solid lines: Raw tremor amplitude measurements
    • Faint lines: Smoothed trend lines showing general tremor patterns
    • Dashed lines: Mean amplitude for each recording session
    
    Statistical measures shown for each day help compare tremor characteristics across time.
    """
    
    # Add the explanation text
    ax_legend.text(0.5, 0.5, legend_text, ha='center', va='center', fontsize=12,
                  bbox=dict(boxstyle="round,pad=1.0", facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    
    # Save figure
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Fancy visualization saved to {output_path}")
    
    plt.show()
    
    # Create a second statistical comparison plot
    create_statistical_comparison(historical_data, synthetic_data, output_dir)

def create_statistical_comparison(historical_data, synthetic_data, output_dir):
    """
    Create a statistical comparison visualization of tremor metrics.
    """
    # Extract key metrics from all data
    metrics = []
    
    for i, df in enumerate(historical_data):
        metrics.append({
            'Day': i + 1,
            'Type': 'Historical',
            'Mean': df['Amplitude'].mean(),
            'Max': df['Amplitude'].max(),
            'StdDev': df['Amplitude'].std(),
            'PeakFrequency': np.sum(df['Amplitude'] > df['Amplitude'].mean() * 1.5) / len(df)
        })
    
    for i, df in enumerate(synthetic_data):
        day_num = len(historical_data) + i + 1
        metrics.append({
            'Day': day_num,
            'Type': 'Synthetic',
            'Mean': df['Amplitude'].mean(),
            'Max': df['Amplitude'].max(),
            'StdDev': df['Amplitude'].std(),
            'PeakFrequency': np.sum(df['Amplitude'] > df['Amplitude'].mean() * 1.5) / len(df)
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Create a nice visualization of these metrics
    plt.figure(figsize=(14, 10))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Set a clean background style
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Add a title to the entire figure
    fig.suptitle('Tremor Metrics Comparison: Historical vs. Synthetic Data', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Plot mean amplitude
    ax1 = axes[0, 0]
    sns.barplot(x='Day', y='Mean', hue='Type', data=metrics_df, ax=ax1,
               palette={'Historical': 'royalblue', 'Synthetic': 'crimson'})
    ax1.set_title('Mean Tremor Amplitude', fontweight='bold')
    ax1.set_xlabel('Day Number')
    ax1.set_ylabel('Mean Amplitude')
    
    # Add a trend line
    days = metrics_df['Day'].unique()
    for data_type in ['Historical', 'Synthetic']:
        type_data = metrics_df[metrics_df['Type'] == data_type]
        if len(type_data) >= 2:  # Need at least 2 points for a line
            sns.regplot(x='Day', y='Mean', data=type_data, 
                      scatter=False, ax=ax1, color='navy' if data_type == 'Historical' else 'darkred',
                      line_kws={'linestyle': '--', 'linewidth': 1})
    
    # Plot max amplitude
    ax2 = axes[0, 1]
    sns.barplot(x='Day', y='Max', hue='Type', data=metrics_df, ax=ax2,
               palette={'Historical': 'royalblue', 'Synthetic': 'crimson'})
    ax2.set_title('Maximum Tremor Amplitude', fontweight='bold')
    ax2.set_xlabel('Day Number')
    ax2.set_ylabel('Maximum Amplitude')
    
    # Plot standard deviation
    ax3 = axes[1, 0]
    sns.barplot(x='Day', y='StdDev', hue='Type', data=metrics_df, ax=ax3,
               palette={'Historical': 'royalblue', 'Synthetic': 'crimson'})
    ax3.set_title('Tremor Variability (Standard Deviation)', fontweight='bold')
    ax3.set_xlabel('Day Number')
    ax3.set_ylabel('Standard Deviation')
    
    # Plot peak frequency
    ax4 = axes[1, 1]
    sns.barplot(x='Day', y='PeakFrequency', hue='Type', data=metrics_df, ax=ax4,
               palette={'Historical': 'royalblue', 'Synthetic': 'crimson'})
    ax4.set_title('Tremor Peak Frequency', fontweight='bold')
    ax4.set_xlabel('Day Number')
    ax4.set_ylabel('Proportion of Peaks')
    ax4.set_ylim(0, min(1.0, metrics_df['PeakFrequency'].max() * 1.2))
    
    # Adjust layout and legends
    for ax in axes.flatten():
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Remove redundant legends and keep only one
        if ax != ax1:
            ax.get_legend().remove()
    
    # Adjust the position and title of the remaining legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, title='Data Type', loc='upper right', frameon=True, 
              framealpha=0.9, edgecolor='gray')
    
    # Add a separator line between historical and synthetic data
    for ax in axes.flatten():
        ylim = ax.get_ylim()
        ax.plot([len(historical_data) + 0.5, len(historical_data) + 0.5], ylim, 
                '--', color='gray', linewidth=1.5, alpha=0.7)
        ax.text(len(historical_data) + 0.5, ylim[1] * 0.95, 'Predictions →', 
               rotation=90, va='top', ha='center', color='gray', fontsize=10)
    
    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.3, hspace=0.3)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'tremor_metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Statistical comparison saved to {output_path}")
    
    plt.show()

def load_and_visualize(historical_dir, prediction_dir, output_dir):
    """
    Load all data and create visualizations.
    """
    # Load historical data
    historical_data = load_tremor_data(historical_dir)
    
    # Load synthetic data (from prediction directory)
    # First find all prediction files
    synthetic_files = sorted(glob.glob(os.path.join(prediction_dir, "predicted_tremor_day_*.csv")))
    
    # Load each synthetic file
    synthetic_data = []
    for file_path in synthetic_files:
        try:
            df = pd.read_csv(file_path)
            
            # Extract method from filename if available
            if '_model' in os.path.basename(file_path):
                df['method'] = 'model'
            elif '_bootstrap' in os.path.basename(file_path):
                df['method'] = 'bootstrap'
            else:
                df['method'] = 'unknown'
                
            synthetic_data.append(df)
            print(f"Loaded synthetic data: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Create fancy visualizations
    if historical_data and synthetic_data:
        create_fancy_tremor_plots(historical_data, synthetic_data, output_dir)
    else:
        print("Not enough data to create visualizations.")

if __name__ == "__main__":
    historical_dir = "E:/Tremor/after_data"  # Directory with original CSV files
    prediction_dir = "E:/Tremor/predictions"  # Directory with prediction results
    output_dir = "E:/Tremor/visualizations"   # Directory to save visualizations
    
    load_and_visualize(historical_dir, prediction_dir, output_dir)