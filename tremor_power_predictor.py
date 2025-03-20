import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.ar_model import AutoReg
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TremorPowerPredictor:
    """Class to predict tremor band power values."""
    
    def __init__(self, file_path):
        """Initialize the predictor with data from the given file path."""
        self.file_path = file_path
        self.df = None
        self.ar_model = None
        self.poly_model = None
        self.poly_features = None
        
        # Create output directory for plots
        self.plots_dir = os.path.join(os.path.dirname(os.path.abspath(file_path)), "plots")
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            print(f"Created directory: {self.plots_dir}")
        
        # Load the data
        self._load_data()
        
        # Train the models
        self._train_models()
    
    def _load_data(self):
        """Load data from the CSV file."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded data from {self.file_path}")
            print(f"Dataset shape: {self.df.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _train_models(self):
        """Train the predictive models."""
        # AR model for pre_band_power
        ar_model = AutoReg(self.df['pre_band_power'], lags=1)
        self.ar_model = ar_model.fit()
        
        # AR model for differenced series
        diff_series = self.df['pre_band_power'].diff().dropna()
        ar_diff_model = AutoReg(diff_series, lags=1)
        self.ar_diff_model = ar_diff_model.fit()
        
        # Polynomial model for post_band_power
        X = self.df[['pre_band_power']]
        y = self.df['post_band_power']
        
        self.poly_features = PolynomialFeatures(degree=2)
        X_poly = self.poly_features.fit_transform(X)
        
        self.poly_model = LinearRegression()
        self.poly_model.fit(X_poly, y)
        
        print("Models trained successfully")
    
    def predict_future(self, days=1, plot=True):
        """Predict future tremor band power values."""
        last_day = self.df['day'].max()
        future_days = range(last_day + 1, last_day + days + 1)
        
        # Initialize results DataFrame
        future_df = pd.DataFrame({'day': future_days})
        
        try:
            # AR model prediction for pre_band_power
            ar_predictions = self.ar_model.predict(start=len(self.df), end=len(self.df) + days - 1)
            future_df['pre_band_power_ar'] = ar_predictions
            
            # AR with differencing (ARIMA-like) approach
            try:
                ar_diff_pred = self.ar_diff_model.predict(start=len(self.df['pre_band_power'].diff().dropna()), 
                                                         end=len(self.df['pre_band_power'].diff().dropna()) + days - 1)
                future_df['pre_band_power_arima'] = self.df['pre_band_power'].iloc[-1] + ar_diff_pred
            except Exception as e:
                print(f"Warning: Error in ARIMA-like prediction: {e}")
                future_df['pre_band_power_arima'] = future_df['pre_band_power_ar']
            
            # Check for NaN values and handle them
            if future_df['pre_band_power_ar'].isnull().any():
                print("Warning: NaN values detected in AR predictions. Using last known value.")
                future_df['pre_band_power_ar'] = future_df['pre_band_power_ar'].fillna(self.df['pre_band_power'].iloc[-1])
                
            if future_df['pre_band_power_arima'].isnull().any():
                print("Warning: NaN values detected in ARIMA predictions. Using AR predictions instead.")
                future_df['pre_band_power_arima'] = future_df['pre_band_power_arima'].fillna(future_df['pre_band_power_ar'])
            
            # Ensemble prediction for pre_band_power
            future_df['pre_band_power'] = (future_df['pre_band_power_ar'] + future_df['pre_band_power_arima']) / 2
            
            # Polynomial prediction for post_band_power
            future_pre = future_df[['pre_band_power']]
            future_pre_poly = self.poly_features.transform(future_pre)
            future_df['post_band_power'] = self.poly_model.predict(future_pre_poly)
            
            # Calculate reduction percentage
            future_df['band_power_reduction_percent'] = ((future_df['pre_band_power'] - future_df['post_band_power']) / 
                                                       future_df['pre_band_power']) * 100
            
            # Handle potential infinity values in reduction percentage (division by zero)
            future_df['band_power_reduction_percent'] = future_df['band_power_reduction_percent'].replace([np.inf, -np.inf], np.nan)
            if future_df['band_power_reduction_percent'].isnull().any():
                print("Warning: NaN values detected in reduction percentage. Using direct calculation.")
                for idx, row in future_df.iterrows():
                    if pd.isna(row['band_power_reduction_percent']):
                        if row['pre_band_power'] < 0.0001:  # Very small pre_band_power
                            if row['post_band_power'] > row['pre_band_power']:
                                future_df.at[idx, 'band_power_reduction_percent'] = -100.0  # Worsening
                            else:
                                future_df.at[idx, 'band_power_reduction_percent'] = 0.0     # No change
            
            # Print predictions
            print("\nPredictions for future days:")
            for _, row in future_df.iterrows():
                print(f"\nDay {int(row['day'])}:")
                print(f"  Pre Band Power: {row['pre_band_power']:.4f}")
                print(f"  Post Band Power: {row['post_band_power']:.4f}")
                print(f"  Reduction Percentage: {row['band_power_reduction_percent']:.2f}%")
            
            # Plot predictions if requested
            if plot:
                self._plot_predictions(future_df)
            
            return future_df
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            # Return a simple prediction using last day's trend
            print("\nFalling back to simple trend-based prediction...")
            
            # Simple trend-based prediction
            last_pre = self.df['pre_band_power'].iloc[-1]
            last_post = self.df['post_band_power'].iloc[-1]
            
            # If more than one data point, calculate trend
            if len(self.df) > 1:
                pre_trend = self.df['pre_band_power'].iloc[-1] - self.df['pre_band_power'].iloc[-2]
                post_trend = self.df['post_band_power'].iloc[-1] - self.df['post_band_power'].iloc[-2]
            else:
                pre_trend = 0
                post_trend = 0
            
            # Create simple predictions
            future_df['pre_band_power'] = [last_pre + (i+1)*pre_trend for i in range(days)]
            future_df['post_band_power'] = [last_post + (i+1)*post_trend for i in range(days)]
            future_df['band_power_reduction_percent'] = ((future_df['pre_band_power'] - future_df['post_band_power']) / 
                                                        future_df['pre_band_power']) * 100
            
            # Handle potential infinity or nan
            future_df['band_power_reduction_percent'] = future_df['band_power_reduction_percent'].replace([np.inf, -np.inf], np.nan)
            future_df['band_power_reduction_percent'] = future_df['band_power_reduction_percent'].fillna(0)
            
            return future_df
    
    def _plot_predictions(self, future_df):
        """Plot historical data and predictions."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Pre and Post Band Power
            ax1.plot(self.df['day'], self.df['pre_band_power'], 'o-', color='blue', label='Historical Pre Band Power')
            ax1.plot(self.df['day'], self.df['post_band_power'], 'o-', color='green', label='Historical Post Band Power')
            
            ax1.plot(future_df['day'], future_df['pre_band_power'], 'o--', color='darkblue', label='Predicted Pre Band Power')
            ax1.plot(future_df['day'], future_df['post_band_power'], 'o--', color='darkgreen', label='Predicted Post Band Power')
            
            # Add uncertainty range
            ar_std = np.std(self.df['pre_band_power'] - self.ar_model.fittedvalues)
            ax1.fill_between(future_df['day'], 
                             future_df['pre_band_power'] - 1.96*ar_std,
                             future_df['pre_band_power'] + 1.96*ar_std,
                             color='blue', alpha=0.2)
            
            ax1.set_xlabel('Day')
            ax1.set_ylabel('Band Power')
            ax1.set_title('Tremor Band Power Prediction')
            ax1.grid(True)
            ax1.legend()
            
            # Plot 2: Reduction Percentage
            bars = ax2.bar(self.df['day'], self.df['band_power_reduction_percent'], color='orange', alpha=0.7, label='Historical')
            pred_bars = ax2.bar(future_df['day'], future_df['band_power_reduction_percent'], color='red', alpha=0.7, label='Predicted')
            
            # Color the bars based on positive/negative values
            for i, bar in enumerate(bars):
                if self.df['band_power_reduction_percent'].iloc[i] < 0:
                    bar.set_color('tomato')
                else:
                    bar.set_color('skyblue')
                    
            for i, bar in enumerate(pred_bars):
                if future_df['band_power_reduction_percent'].iloc[i] < 0:
                    bar.set_color('darkred')
                else:
                    bar.set_color('darkblue')
            
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Day')
            ax2.set_ylabel('Reduction Percentage (%)')
            ax2.set_title('Band Power Reduction Percentage')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            
            # Save plot to file
            plot_path = os.path.join(self.plots_dir, 'tremor_power_predictions.png')
            plt.savefig(plot_path, dpi=300)
            print(f"Saved plot to {plot_path}")
            plt.show()
        except Exception as e:
            print(f"Error during plotting: {e}")
    
    def analyze_relationships(self):
        """Analyze relationships in the data."""
        try:
            # Group by positive/negative reduction
            pos_reduction = self.df[self.df['band_power_reduction_percent'] > 0]
            neg_reduction = self.df[self.df['band_power_reduction_percent'] <= 0]
            
            avg_pre_pos = pos_reduction['pre_band_power'].mean()
            avg_pre_neg = neg_reduction['pre_band_power'].mean()
            
            print("\nRelationship between pre_band_power and reduction outcome:")
            print(f"Days with positive reduction: {len(pos_reduction)}")
            print(f"Days with negative reduction: {len(neg_reduction)}")
            print(f"Average pre_band_power for positive reduction days: {avg_pre_pos:.4f}")
            print(f"Average pre_band_power for negative reduction days: {avg_pre_neg:.4f}")
            
            # Plot relationships
            plt.figure(figsize=(12, 10))
            
            # Create scatter plot of pre vs post band power
            plt.subplot(2, 1, 1)
            plt.scatter(self.df['pre_band_power'], self.df['post_band_power'], c=self.df['band_power_reduction_percent'], 
                        cmap='coolwarm', s=100, alpha=0.7)
            
            # Add regression line
            x_range = np.linspace(self.df['pre_band_power'].min(), self.df['pre_band_power'].max(), 100)
            x_range_poly = self.poly_features.transform(x_range.reshape(-1, 1))
            y_pred = self.poly_model.predict(x_range_poly)
            plt.plot(x_range, y_pred, color='green', linestyle='--')
            
            plt.colorbar(label='Reduction Percentage')
            plt.xlabel('Pre Band Power')
            plt.ylabel('Post Band Power')
            plt.title('Relationship between Pre and Post Band Power')
            plt.grid(True)
            
            # Create box plot comparing pre_band_power for positive and negative reduction
            plt.subplot(2, 1, 2)
            data = [pos_reduction['pre_band_power'], neg_reduction['pre_band_power']]
            plt.boxplot(data, labels=['Positive Reduction', 'Negative Reduction'])
            plt.ylabel('Pre Band Power')
            plt.title('Pre Band Power Distribution by Reduction Outcome')
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save plot to file
            plot_path = os.path.join(self.plots_dir, 'tremor_relationships.png')
            plt.savefig(plot_path, dpi=300)
            print(f"Saved plot to {plot_path}")
            plt.show()
            
            return {
                'pos_reduction_count': len(pos_reduction),
                'neg_reduction_count': len(neg_reduction),
                'avg_pre_pos': avg_pre_pos,
                'avg_pre_neg': avg_pre_neg
            }
        except Exception as e:
            print(f"Error during relationship analysis: {e}")
            return {
                'pos_reduction_count': 0,
                'neg_reduction_count': 0,
                'avg_pre_pos': 0,
                'avg_pre_neg': 0
            }
    
    def model_performance(self):
        """Evaluate the performance of the models."""
        try:
            # Evaluate AR model
            ar_predictions = self.ar_model.fittedvalues
            ar_mse = ((self.df['pre_band_power'].iloc[1:] - ar_predictions) ** 2).mean()
            
            # Evaluate polynomial model
            X_poly = self.poly_features.transform(self.df[['pre_band_power']])
            poly_predictions = self.poly_model.predict(X_poly)
            poly_mse = ((self.df['post_band_power'] - poly_predictions) ** 2).mean()
            
            print("\nModel Performance:")
            print(f"AR model MSE for pre_band_power: {ar_mse:.4f}")
            print(f"Polynomial model MSE for post_band_power: {poly_mse:.4f}")
            
            # Plot actual vs predicted
            plt.figure(figsize=(14, 6))
            
            # Plot pre_band_power actual vs predicted
            plt.subplot(1, 2, 1)
            plt.scatter(self.df['pre_band_power'].iloc[1:], ar_predictions, alpha=0.7)
            plt.plot([self.df['pre_band_power'].min(), self.df['pre_band_power'].max()], 
                     [self.df['pre_band_power'].min(), self.df['pre_band_power'].max()], 
                     'r--')
            plt.xlabel('Actual Pre Band Power')
            plt.ylabel('Predicted Pre Band Power')
            plt.title('AR Model Performance')
            plt.grid(True)
            
            # Plot post_band_power actual vs predicted
            plt.subplot(1, 2, 2)
            plt.scatter(self.df['post_band_power'], poly_predictions, alpha=0.7)
            plt.plot([self.df['post_band_power'].min(), self.df['post_band_power'].max()], 
                     [self.df['post_band_power'].min(), self.df['post_band_power'].max()], 
                     'r--')
            plt.xlabel('Actual Post Band Power')
            plt.ylabel('Predicted Post Band Power')
            plt.title('Polynomial Model Performance')
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save plot to file
            plot_path = os.path.join(self.plots_dir, 'model_performance.png')
            plt.savefig(plot_path, dpi=300)
            print(f"Saved plot to {plot_path}")
            plt.show()
            
            return {
                'ar_mse': ar_mse,
                'poly_mse': poly_mse
            }
        except Exception as e:
            print(f"Error during model performance evaluation: {e}")
            return {
                'ar_mse': 0,
                'poly_mse': 0
            }

# Example usage
if __name__ == "__main__":
    try:
        # Get file path (use current directory if not specified)
        import sys
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        else:
            # Try to find CSV files in the current directory
            import glob
            csv_files = glob.glob("*.csv")
            if not csv_files:
                print("No CSV files found in current directory.")
                print("Please specify a file path: python tremor_power_predictor.py path/to/your/file.csv")
                sys.exit(1)
            
            file_path = csv_files[0]
            print(f"Using first CSV file found: {file_path}")
        
        # Create predictor instance
        predictor = TremorPowerPredictor(file_path)
        
        # Analyze relationships in the data
        relationship_stats = predictor.analyze_relationships()
        
        # Evaluate model performance
        performance_stats = predictor.model_performance()
        
        # Predict future values (next 3 days)
        future_predictions = predictor.predict_future(days=3)
        
        # Save predictions to CSV
        output_file = os.path.join(os.path.dirname(os.path.abspath(file_path)), "predictions.csv")
        future_predictions.to_csv(output_file, index=False)
        print(f"\nSaved predictions to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()