import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, List, Tuple, Optional

class DataVisualizer:
    def __init__(self, file_pairs: Dict[str, Tuple[str, str]], output_folder='visualizations'):
        """
        Initialize the data visualizer with pairs of original and predicted file paths.
        
        Args:
            file_pairs: Dictionary with day/label as key and tuple of (original_file_path, predicted_file_path) as value
            output_folder: Path to save visualization outputs
        """
        self.file_pairs = file_pairs
        self.output_folder = output_folder
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # Set default plot style
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
        })
    
    def load_data(self, day_label: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load original and predicted data for a given day/label."""
        if day_label not in self.file_pairs:
            print(f"No file pair found for day/label: {day_label}")
            return None, None
            
        original_path, predicted_path = self.file_pairs[day_label]
        
        try:
            original_df = pd.read_csv(original_path)
            predicted_df = pd.read_csv(predicted_path)
            return original_df, predicted_df
        except Exception as e:
            print(f"Error loading data for {day_label}: {e}")
            return None, None
    
    def apply_smoothing(self, data, window_length=11, polyorder=3):
        """Apply Savitzky-Golay filter for smoothing."""
        # Make sure window_length is odd and less than data length
        if len(data) <= window_length:
            window_length = min(len(data) - (len(data) % 2) - 1, window_length)
            if window_length <= polyorder:
                # If data is too short, use simple moving average instead
                return data.rolling(window=max(3, len(data)//5), center=True).mean().fillna(data)
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
            
        try:
            smoothed = savgol_filter(data, window_length, polyorder)
            return pd.Series(smoothed, index=data.index)
        except Exception as e:
            print(f"Smoothing error: {e}, using original data")
            return data
    
    def align_datasets(self, original_df, predicted_df, index_col):
        """
        Align original and predicted datasets to have the same length and indices.
        This handles the case where datasets have different numbers of rows.
        """
        print(f"Original dataset shape: {original_df.shape}, Predicted dataset shape: {predicted_df.shape}")
        
        # Create copies to avoid modifying originals
        original_df = original_df.copy()
        predicted_df = predicted_df.copy()
        
        # If no common index column exists or it's the artificial 'index' we created,
        # we'll just resample both datasets to the same length
        if index_col == 'index' and 'index' not in original_df.columns.tolist()[:2]:
            print("Using uniform resampling for alignment (no common index)")
            # Use the shorter dataset length
            min_length = min(len(original_df), len(predicted_df))
            
            # Resample both datasets to this length
            original_resampled = pd.DataFrame()
            predicted_resampled = pd.DataFrame()
            
            # Process each column
            for col in original_df.columns:
                if col == index_col:
                    # Use range for index column
                    original_resampled[col] = np.arange(min_length)
                    continue
                    
                if pd.api.types.is_numeric_dtype(original_df[col]):
                    # Resample numeric columns
                    x_orig = np.linspace(0, 1, len(original_df))
                    x_new = np.linspace(0, 1, min_length)
                    y_orig = original_df[col].values
                    original_resampled[col] = np.interp(x_new, x_orig, y_orig)
                else:
                    # For non-numeric, just take the first min_length items
                    original_resampled[col] = original_df[col].iloc[:min_length].values
            
            for col in predicted_df.columns:
                if col == index_col:
                    # Use range for index column
                    predicted_resampled[col] = np.arange(min_length)
                    continue
                    
                if pd.api.types.is_numeric_dtype(predicted_df[col]):
                    # Resample numeric columns
                    x_pred = np.linspace(0, 1, len(predicted_df))
                    x_new = np.linspace(0, 1, min_length)
                    y_pred = predicted_df[col].values
                    predicted_resampled[col] = np.interp(x_new, x_pred, y_pred)
                else:
                    # For non-numeric, just take the first min_length items
                    predicted_resampled[col] = predicted_df[col].iloc[:min_length].values
            
            return original_resampled, predicted_resampled
            
        # If index columns exist, try to find matching indices
        elif index_col in original_df.columns and index_col in predicted_df.columns:
            # Convert to numeric if not already numeric
            if not pd.api.types.is_numeric_dtype(original_df[index_col]):
                try:
                    original_df[index_col] = pd.to_numeric(original_df[index_col], errors='coerce')
                except:
                    pass
                    
            if not pd.api.types.is_numeric_dtype(predicted_df[index_col]):
                try:
                    predicted_df[index_col] = pd.to_numeric(predicted_df[index_col], errors='coerce')
                except:
                    pass
            
            # Try to find common indices
            try:
                orig_indices = set(original_df[index_col])
                pred_indices = set(predicted_df[index_col])
                common_indices = orig_indices.intersection(pred_indices)
                
                if len(common_indices) > 0:
                    print(f"Aligning datasets using common indices (found {len(common_indices)} matching points)")
                    original_df = original_df[original_df[index_col].isin(common_indices)]
                    predicted_df = predicted_df[predicted_df[index_col].isin(common_indices)]
                    
                    # Sort both datasets by the index column
                    original_df = original_df.sort_values(by=index_col).reset_index(drop=True)
                    predicted_df = predicted_df.sort_values(by=index_col).reset_index(drop=True)
                    
                    return original_df, predicted_df
            except:
                print("Error finding common indices, falling back to resampling")
        
        # If the above doesn't work or isn't applicable, use interpolation to make datasets the same length
        print("Using interpolation to align datasets of different lengths")
        
        # Approach 1: Match the length by interpolating to the shorter length
        min_length = min(len(original_df), len(predicted_df))
        
        # Create new indices based on the min length
        new_indices = np.linspace(0, 1, min_length)
        old_orig_indices = np.linspace(0, 1, len(original_df))
        old_pred_indices = np.linspace(0, 1, len(predicted_df))
        
        # Create new dataframes with aligned lengths
        aligned_original = pd.DataFrame()
        aligned_predicted = pd.DataFrame()
        
        # Add the index column if it exists
        if index_col in original_df.columns:
            # Create interpolated index column
            if pd.api.types.is_numeric_dtype(original_df[index_col]):
                aligned_original[index_col] = np.interp(new_indices, old_orig_indices, original_df[index_col])
            else:
                # For non-numeric indices, use the first min_length items
                aligned_original[index_col] = original_df[index_col].iloc[:min_length].values
        
        if index_col in predicted_df.columns:
            if pd.api.types.is_numeric_dtype(predicted_df[index_col]):
                aligned_predicted[index_col] = np.interp(new_indices, old_pred_indices, predicted_df[index_col])
            else:
                aligned_predicted[index_col] = predicted_df[index_col].iloc[:min_length].values
        
        # Handle other numeric columns with interpolation
        for col in original_df.columns:
            if col != index_col and pd.api.types.is_numeric_dtype(original_df[col]):
                aligned_original[col] = np.interp(new_indices, old_orig_indices, original_df[col])
            elif col != index_col:
                aligned_original[col] = original_df[col].iloc[:min_length].values
        
        for col in predicted_df.columns:
            if col != index_col and pd.api.types.is_numeric_dtype(predicted_df[col]):
                aligned_predicted[col] = np.interp(new_indices, old_pred_indices, predicted_df[col])
            elif col != index_col:
                aligned_predicted[col] = predicted_df[col].iloc[:min_length].values
        
        print(f"Aligned dataset shape: {aligned_original.shape}")
        return aligned_original, aligned_predicted
    
    def visualize_single_day(self, day_label: str, smoothing_window=31, show_plot=False):
        """Create visualization for a single day/label with improved plot clarity."""
        original_df, predicted_df = self.load_data(day_label)
        if original_df is None or predicted_df is None:
            return False
            
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f'Comparison for Day: {day_label}', fontsize=18)
        
        # Index column - try to find a good candidate for x-axis
        index_col = None
        for col_candidate in ['time', 'date', 'timestamp', 'index']:
            if col_candidate in original_df.columns:
                index_col = col_candidate
                break
                
        # If no index column found, use dataframe index
        if index_col is None:
            original_df['index'] = original_df.index
            predicted_df['index'] = predicted_df.index
            index_col = 'index'
        
        # Get numerical columns (exclude datetime columns)
        numerical_cols = original_df.select_dtypes(include=np.number).columns
        
        # Define columns to exclude (add 'Frame' to this list)
        exclude_cols = [index_col, 'Frame']
        
        # Filter out excluded columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        n_cols = len(numerical_cols)
        
        if n_cols == 0:
            print(f"No numerical columns found for day {day_label}")
            return False
        
        # Align datasets before visualization
        original_df, predicted_df = self.align_datasets(original_df, predicted_df, index_col)
        
        # Identify viable columns for plotting
        viable_cols = [col for col in numerical_cols if col in original_df.columns and col in predicted_df.columns]
        n_plots = len(viable_cols)
        
        # Calculate grid layout - we need to be careful about number of subplots
        if n_plots == 0:
            print(f"No viable columns found for plotting in day {day_label}")
            return False
                
        # If only 1 plot (plus summary), use 1 row with 2 columns
        # If 2 plots (plus summary), use 2 rows with 2 columns
        # If 3+ plots (plus summary), use ceil(n/2) rows with 2 columns
        n_rows = max(1, (n_plots + 1) // 2)  # Calculate rows needed
        
        # Plot each numerical column
        plot_idx = 1
        for col in numerical_cols:
            # Skip columns not in both datasets
            if col not in original_df.columns or col not in predicted_df.columns:
                print(f"Column {col} not found in both datasets, skipping")
                continue
                
            ax = fig.add_subplot(n_rows, 2, plot_idx)
            plot_idx += 1
            
            # Apply stronger smoothing
            if col in original_df.columns:
                orig_data = original_df[col]
                orig_smoothed = self.apply_smoothing(orig_data, smoothing_window)
                
                # Use fewer markers for scatter (only show every 20th point)
                marker_indices = np.arange(0, len(orig_data), 20)
                ax.scatter(original_df.iloc[marker_indices][index_col], orig_data.iloc[marker_indices], 
                        alpha=0.4, s=20, color='blue', label='Original (sample)')
                
                # Make the smoothed line more prominent
                ax.plot(original_df[index_col], orig_smoothed, 
                        linewidth=2.5, color='darkblue', label='Original (trend)')
            
            # Predicted data - similar improvements
            if col in predicted_df.columns:
                pred_data = predicted_df[col]
                pred_smoothed = self.apply_smoothing(pred_data, smoothing_window)
                
                # Use fewer markers, offset from original to avoid overlap
                marker_indices = np.arange(10, len(pred_data), 20) 
                ax.scatter(predicted_df.iloc[marker_indices][index_col], pred_data.iloc[marker_indices], 
                        alpha=0.4, s=20, color='red', label='Predicted (sample)')
                
                # Make the smoothed line more prominent
                ax.plot(predicted_df[index_col], pred_smoothed, 
                        linewidth=2.5, color='darkred', label='Predicted (trend)')
            
            # Use a lighter, more transparent error range
            if col in original_df.columns and col in predicted_df.columns:
                # Calculate error
                error = np.abs(original_df[col] - predicted_df[col])
                
                # Use a more transparent fill_between
                ax.fill_between(
                    original_df[index_col], 
                    np.minimum(original_df[col], predicted_df[col]),
                    np.maximum(original_df[col], predicted_df[col]),
                    color='gray', alpha=0.1, label='Error range'
                )
                
                # Add RMSE to title
                rmse = np.sqrt(np.mean(np.square(original_df[col] - predicted_df[col])))
                ax.set_title(f'{col} (RMSE: {rmse:.2f})', fontsize=14)
            else:
                ax.set_title(col, fontsize=14)
                    
            # Improve axis labels and formatting
            ax.set_xlabel(index_col, fontsize=12)
            ax.set_ylabel(col, fontsize=12)
            
            # Use a smaller, more readable legend
            ax.legend(loc='upper right', fontsize=10)
            
            # Use lighter grid
            ax.grid(True, alpha=0.2, linestyle='--')
            
            # Add subtle spines
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#dddddd')

        # Add a summary plot with improved formatting
        common_cols = [col for col in numerical_cols if col in original_df.columns and col in predicted_df.columns]
        
        # Only add summary if we have viable columns and haven't exceeded grid
        if common_cols and len(common_cols) > 0 and plot_idx <= n_rows * 2:
            try:
                # Calculate RMSE values first
                rmse_values = []
                for col in common_cols:
                    rmse = np.sqrt(np.mean(np.square(original_df[col] - predicted_df[col])))
                    rmse_values.append(rmse)
                
                # Only create plot if we have values
                if rmse_values:
                    ax_summary = fig.add_subplot(n_rows, 2, plot_idx)
                    
                    # Create bar chart of RMSE values with better colors
                    bars = ax_summary.bar(common_cols, rmse_values, color='#1f77b4', alpha=0.8)
                    ax_summary.set_title('RMSE Summary', fontsize=14)
                    ax_summary.set_xlabel('Feature', fontsize=12)
                    ax_summary.set_ylabel('RMSE', fontsize=12)
                    ax_summary.tick_params(axis='x', rotation=45)
                    
                    # Add values on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax_summary.text(
                            bar.get_x() + bar.get_width()/2.,
                            height + 0.02 * max(rmse_values),
                            f'{height:.2f}',
                            ha='center', va='bottom', rotation=0,
                            fontsize=10
                        )
                    
                    # Use lighter grid for summary too
                    ax_summary.grid(True, alpha=0.2, linestyle='--', axis='y')
                    for spine in ax_summary.spines.values():
                        spine.set_visible(True)
                        spine.set_color('#dddddd')
            except Exception as e:
                print(f"Error creating summary plot: {e}")
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)
        
        # Save figure
        output_path = os.path.join(self.output_folder, f"{day_label}_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
        return True

# Also update the heatmap function to exclude the Frame column
    def create_heatmap_comparison(self, show_plot=False):
        """Create a heatmap comparing RMSE across all days."""
        # Dictionary to store RMSE values for each day and column
        all_rmse = {}
        
        # Process each day
        for day_label in self.file_pairs.keys():
            original_df, predicted_df = self.load_data(day_label)
            if original_df is None or predicted_df is None:
                continue
                
            # Index column - try to find a good candidate for alignment
            index_col = None
            for col_candidate in ['time', 'date', 'timestamp', 'index']:
                if col_candidate in original_df.columns:
                    index_col = col_candidate
                    break
                    
            # If no index column found, use dataframe index
            if index_col is None:
                original_df['index'] = original_df.index
                predicted_df['index'] = predicted_df.index
                index_col = 'index'
                
            # Align datasets
            original_df, predicted_df = self.align_datasets(original_df, predicted_df, index_col)
                
            # Get common numerical columns, excluding the 'Frame' column
            numerical_cols = [
                col for col in original_df.select_dtypes(include=np.number).columns
                if col in predicted_df.columns and col != index_col and col != 'Frame'
            ]
            
            # Calculate RMSE for each column
            rmse_dict = {}
            for col in numerical_cols:
                rmse = np.sqrt(np.mean(np.square(original_df[col] - predicted_df[col])))
                rmse_dict[col] = rmse
                
            all_rmse[day_label] = rmse_dict
        
        # Convert to DataFrame
        if not all_rmse:
            print("No valid RMSE values to create heatmap")
            return
            
        heatmap_df = pd.DataFrame(all_rmse).T
        
        # Fill any missing values with NaN
        heatmap_df = heatmap_df.fillna(np.nan)
        
        # If there are no common columns across all days, skip drawing the heatmap
        if heatmap_df.empty or heatmap_df.shape[1] == 0:
            print("No common columns across datasets for heatmap")
            return
            
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_df, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('RMSE Comparison Across Days')
        plt.ylabel('Day')
        plt.xlabel('Feature')
        
        # Save figure
        output_path = os.path.join(self.output_folder, "rmse_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            

            
    def visualize_all_days(self, smoothing_window=11, show_plots=False):
        """Process all days with data file pairs."""
        days = list(self.file_pairs.keys())
        print(f"Found {len(days)} days with data pairs")
        
        for i, day_label in enumerate(days):
            print(f"Processing {i+1}/{len(days)}: {day_label}")
            self.visualize_single_day(day_label, smoothing_window, show_plots)
            
        # Create summary heatmap
        self.create_heatmap_comparison(show_plots)
        
    def create_pdf_report(self, smoothing_window=11):
        """Create a single PDF report with all visualizations."""
        days = list(self.file_pairs.keys())
        if not days:
            return
            
        pdf_path = os.path.join(self.output_folder, "visualization_report.pdf")
        
        with PdfPages(pdf_path) as pdf:
            # Add a title page
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, 'Data Visualization Report', 
                     ha='center', va='center', fontsize=24)
            plt.text(0.5, 0.4, f'Number of days analyzed: {len(days)}', 
                     ha='center', va='center', fontsize=14)
            plt.text(0.5, 0.3, f'Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', 
                     ha='center', va='center', fontsize=14)
            plt.axis('off')
            pdf.savefig()
            plt.close()
            
            # Process each day
            for i, day_label in enumerate(days):
                print(f"Adding to PDF {i+1}/{len(days)}: {day_label}")
                
                # Check if visualization image exists
                img_path = os.path.join(self.output_folder, f"{day_label}_comparison.png")
                if not os.path.exists(img_path):
                    self.visualize_single_day(day_label, smoothing_window, False)
                
                # Add visualization to PDF
                try:
                    plt.figure(figsize=(12, 8))
                    img = plt.imread(img_path)
                    plt.imshow(img)
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()
                except Exception as e:
                    print(f"Error adding {img_path} to PDF: {e}")
            
            # Add heatmap as the last page
            heatmap_path = os.path.join(self.output_folder, "rmse_heatmap.png")
            if not os.path.exists(heatmap_path):
                self.create_heatmap_comparison(False)
                
            if os.path.exists(heatmap_path):
                try:
                    plt.figure(figsize=(12, 8))
                    img = plt.imread(heatmap_path)
                    plt.imshow(img)
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()
                except Exception as e:
                    print(f"Error adding heatmap to PDF: {e}")
                
        print(f"PDF report saved to {pdf_path}")


def main():
    """
    Main function to run the visualization.
    Configure your data file pairs here.
    """
    # Define file pairs as a dictionary:
    # Key: A label for the day/dataset
    # Value: Tuple of (original_file_path, predicted_file_path)
    #
    # MODIFY THIS DICTIONARY WITH YOUR ACTUAL FILE PATHS
    file_pairs = {
        "Day1": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay1After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_16_model.csv"),
        "Day2": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay2After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_17_model.csv"),
        "Day3": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay3After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_18_model.csv"),
        "Day4": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay4After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_19_model.csv"),
        "Day5": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay5After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_20_model.csv"),
        "Day6": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay6After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_21_model.csv"),
        "Day7": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay7After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_22_model.csv"),
        "Day8": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay8After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_23_bootstrap.csv"),
        "Day9": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay9After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_24_bootstrap.csv"),
        "Day10": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay10After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_25_bootstrap.csv"),
        "Day11": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay11After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_26_bootstrap.csv"),
        "Day12": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay12After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_27_bootstrap.csv"),
        "Day13": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay13After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_28_bootstrap.csv"),
        "Day14": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay14After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_29_bootstrap.csv"),
        "Day15": (r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg\NewDay15After_euclidean_distances_avg_tremor.csv", r"E:\Tremor\predictions_after\predicted_tremor_day_30_bootstrap.csv"),
    }
    
    # Set output folder
    output_folder = r"E:\Tremor\vis_predicited"
    
    # Create visualizer
    visualizer = DataVisualizer(file_pairs, output_folder)
    
    # Visualize all days with moderate smoothing
    visualizer.visualize_all_days(smoothing_window=11, show_plots=False)
    
    # Create PDF report
    visualizer.create_pdf_report(smoothing_window=11)
    
    print("Visualization complete!")
    

if __name__ == "__main__":
    main()