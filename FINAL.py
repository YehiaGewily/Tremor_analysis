import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def load_tremor_data(data_dir):
    """
    Load all tremor CSV files from a directory.
    """
    all_data = []
    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    
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

def extract_daily_features(daily_data):
    """
    Extract meaningful features from each day's tremor data.
    """
    features = []
    for day, df in enumerate(daily_data, 1):
        mean_amp = df['Amplitude'].mean()
        max_amp = df['Amplitude'].max()
        min_amp = df['Amplitude'].min()
        std_amp = df['Amplitude'].std()
        threshold = mean_amp * 1.5
        peaks = np.sum(df['Amplitude'] > threshold)
        duration = len(df) / 22.12 if len(df) > 1 else 0
        q25 = df['Amplitude'].quantile(0.25)
        q50 = df['Amplitude'].quantile(0.50)
        q75 = df['Amplitude'].quantile(0.75)
        
        features.append({
            'Day': day,
            'MeanAmplitude': mean_amp,
            'MaxAmplitude': max_amp,
            'MinAmplitude': min_amp,
            'StdAmplitude': std_amp,
            'NumPeaks': peaks,
            'NumFrames': len(df),
            'Duration': duration,
            'Q25': q25,
            'Q50': q50,
            'Q75': q75,
            'PeakRatio': peaks / len(df) if len(df) > 0 else 0
        })
    
    return pd.DataFrame(features)

def train_prediction_model(features_df, daily_data, num_future_days):
    """
    Train models with forced downward trend for future predictions.
    This implementation explicitly creates a downward trend regardless of
    what the automatic trend detection finds.
    """
    X = features_df[['Day']].values
    models = {}
    metrics = ['MeanAmplitude', 'MaxAmplitude', 'StdAmplitude', 'NumPeaks', 'PeakRatio']
    
    plt.figure(figsize=(15, 10))
    all_predictions = {}
    
    max_day = len(daily_data) + num_future_days
    all_days = np.arange(1, max_day + 1).reshape(-1, 1)
    
    for i, metric in enumerate(metrics):
        y = features_df[metric].values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        models[metric] = model
        
        plt.subplot(len(metrics), 1, i+1)
        plt.plot(features_df['Day'], y, 'o-', label=f'Actual {metric}')
        
        # Make base predictions (but we won't use these directly)
        base_predictions = model.predict(all_days)
        
        # Start with a copy
        future_predictions = base_predictions.copy()
        
        # Get last historical value
        last_value = features_df[metric].iloc[-1]
        
        # Calculate average of last 5 days to get a baseline
        recent_avg = features_df[metric].iloc[-5:].mean()
        
        # EXPLICITLY FORCE A DOWNWARD TREND
        # Calculate a reasonable decline rate (10-30% decline over 15 days)
        # Adjust multiplier to control steepness of decline
        decline_multiplier = 0.02  # 2% decline per day
        
        # Override all future predictions with a downward trend
        for j in range(len(daily_data), len(all_days)):
            days_from_last = j - len(daily_data) + 1
            
            # Calculate diminishing factor (starts at 1.0 and decreases)
            decline_factor = 1.0 - (decline_multiplier * days_from_last)
            decline_factor = max(0.5, decline_factor)  # Don't let it go below 50%
            
            # Set the prediction to decline from the last historical value
            future_predictions[j] = last_value * decline_factor
            
            # Add some controlled randomness (less than before)
            if len(y) > 1:
                std_dev = np.std(y) * 0.15  # Just 15% of historical std dev
                noise = np.random.normal(0, std_dev)
                future_predictions[j] += noise
            
            # Ensure we don't go below reasonable minimums
            if metric in ['MeanAmplitude', 'MaxAmplitude', 'StdAmplitude']:
                min_value = min(y) * 0.7  # Don't go below 70% of historical minimum
                future_predictions[j] = max(future_predictions[j], min_value)
            
            if metric == 'PeakRatio':
                future_predictions[j] = max(min(future_predictions[j], 1.0), 0.05)
        
        all_predictions[metric] = future_predictions
        
        # Plot with explicit annotation
        plt.plot(all_days, future_predictions, 'r--', label=f'Predicted {metric}')
        plt.axvline(x=len(daily_data) + 0.5, color='g', linestyle='--', label='Future predictions')
        
        # Add annotation about forced downward trend
        if i == 0:  # Only add this annotation to the first subplot
            plt.annotate('Forced downward trend',
                        xy=(len(daily_data), future_predictions[len(daily_data)-1]),
                        xytext=(len(daily_data)+2, future_predictions[len(daily_data)-1] * 1.2),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=9)
        
        plt.title(f'{metric} - Actual vs Predicted')
        plt.legend()
        
        print(f"\nPredictions for {metric}:")
        for day in range(len(daily_data) + 1, max_day + 1):
            print(f"  Day {day}: {future_predictions[day-1]:.2f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_metrics.png'))
    plt.show()
    
    return models, all_predictions

def generate_synthetic_tremor(mean_amp, max_amp, std_amp, num_frames, peak_ratio):
    """
    Generate synthetic tremor data based on predicted characteristics.
    """
    # Create baseline values with normal distribution around the mean
    baseline = np.random.normal(mean_amp, std_amp / 2, num_frames)
    
    # Determine number of peaks based on peak ratio
    num_peaks = int(num_frames * peak_ratio)
    
    # Randomly choose which frames will contain peaks
    peak_indices = np.random.choice(num_frames, num_peaks, replace=False)
    
    # Generate peak values between 1.5x mean and max amplitude
    peak_values = np.random.uniform(mean_amp * 1.5, max_amp, num_peaks)
    
    # Insert peaks into the baseline
    for i, peak_value in zip(peak_indices, peak_values):
        baseline[i] = peak_value
    
    # Create DataFrame with frame numbers and amplitude values
    synthetic_df = pd.DataFrame({
        'Frame': range(1, num_frames + 1),
        'Amplitude': baseline
    })
    
    return synthetic_df

def bootstrap_tremor_data(daily_data, day_number):
    """
    Create a synthetic day by bootstrapping from historical data with variation.
    """
    # Randomly select one day as a template
    template_idx = np.random.randint(0, len(daily_data))
    template_df = daily_data[template_idx].copy()
    
    print(f"  Using day {template_idx + 1} as template for day {day_number}")
    
    # Add some random noise to create variations (15% noise factor)
    noise_factor = 0.15
    amplitude_noise = template_df['Amplitude'] * np.random.normal(0, noise_factor, len(template_df))
    
    # Create new DataFrame with modified amplitude values
    synthetic_df = pd.DataFrame({
        'Frame': template_df['Frame'].values,
        'Amplitude': template_df['Amplitude'].values + amplitude_noise,
        'Day': day_number
    })
    
    # Ensure no negative values
    synthetic_df['Amplitude'] = synthetic_df['Amplitude'].clip(lower=0)
    
    return synthetic_df

def predict_future_tremors(data_dir, output_dir, num_future_days=15):
    """
    Main function to predict tremors for future days, using a combined approach.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load all data
    daily_data = load_tremor_data(data_dir)
    
    if len(daily_data) == 0:
        print("No data loaded. Check your file paths.")
        return
    
    # Extract features
    features_df = extract_daily_features(daily_data)
    features_df.to_csv(os.path.join(output_dir, 'tremor_features.csv'), index=False)
    print(f"Features saved to {os.path.join(output_dir, 'tremor_features.csv')}")
    
    # Train model and predict
    # Train model and predict
    models, all_predictions = train_prediction_model(features_df, daily_data, num_future_days)

    
    # Generate synthetic data for future days
    future_days = range(len(daily_data) + 1, len(daily_data) + num_future_days + 1)
    
    # We'll use two approaches:
    # 1. Model-based generation for days 16-22
    # 2. Bootstrap-based generation for days 23-30
    for day in future_days:
        idx = day - 1  # Convert to 0-based index
        
        if day <= len(daily_data) + 7:  # First week of predictions: model-based
            print(f"Generating model-based synthetic data for day {day}")
            
            # Get predictions for this day
            mean_amp = all_predictions['MeanAmplitude'][idx]
            max_amp = all_predictions['MaxAmplitude'][idx]
            std_amp = all_predictions['StdAmplitude'][idx]
            peak_ratio = all_predictions['PeakRatio'][idx]
            
            # Determine number of frames (average from existing data)
            avg_frames = int(np.mean([len(df) for df in daily_data]))
            
            # Generate synthetic data
            synthetic_df = generate_synthetic_tremor(
                mean_amp=mean_amp,
                max_amp=max_amp,
                std_amp=std_amp,
                num_frames=avg_frames,
                peak_ratio=peak_ratio
            )
            
            # Add day identifier
            synthetic_df['Day'] = day
            
            # Save note about method
            method = "model"
            
        else:  # Second week of predictions: bootstrap-based
            print(f"Generating bootstrap-based synthetic data for day {day}")
            synthetic_df = bootstrap_tremor_data(daily_data, day)
            method = "bootstrap"
        
        # Save to CSV
        output_file = os.path.join(output_dir, f'predicted_tremor_day_{day}_{method}.csv')
        synthetic_df.to_csv(output_file, index=False)
        print(f"  Generated synthetic data for day {day} saved to {output_file}")
    
    # Plot a sample of generated data
    plot_synthetic_samples(output_dir, daily_data, future_days)
    
    print("Prediction complete!")

def plot_synthetic_samples(output_dir, daily_data, future_days):
    """
    Plot a sample of original data and generated synthetic data for comparison.
    """
    # Number of plots to show
    num_historical = min(3, len(daily_data))
    num_future = min(5, len(future_days))
    
    fig, axes = plt.subplots(num_historical + num_future, 1, figsize=(15, 4*(num_historical + num_future)))
    
    # Plot some historical days first
    for i in range(num_historical):
        day = i + 1
        df = daily_data[i]
        
        axes[i].plot(df['Frame'], df['Amplitude'], color='blue')
        axes[i].set_title(f'Original Data - Day {day}')
        axes[i].set_xlabel('Frame')
        axes[i].set_ylabel('Amplitude')
        
        # Add statistics info
        stats_text = (f"Mean: {df['Amplitude'].mean():.2f}, "
                     f"Max: {df['Amplitude'].max():.2f}, "
                     f"StdDev: {df['Amplitude'].std():.2f}")
        axes[i].text(0.02, 0.9, stats_text, transform=axes[i].transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot future days
    for i, day in enumerate(list(future_days)[:num_future]):
        # Try to find the file
        model_file = os.path.join(output_dir, f'predicted_tremor_day_{day}_model.csv')
        bootstrap_file = os.path.join(output_dir, f'predicted_tremor_day_{day}_bootstrap.csv')
        
        if os.path.exists(model_file):
            syn_df = pd.read_csv(model_file)
            method = "model"
        elif os.path.exists(bootstrap_file):
            syn_df = pd.read_csv(bootstrap_file)
            method = "bootstrap"
        else:
            continue
        
        axes[i + num_historical].plot(syn_df['Frame'], syn_df['Amplitude'], color='red')
        axes[i + num_historical].set_title(f'Synthetic Data - Day {day} ({method})')
        axes[i + num_historical].set_xlabel('Frame')
        axes[i + num_historical].set_ylabel('Amplitude')
        
        # Add statistics info
        stats_text = (f"Mean: {syn_df['Amplitude'].mean():.2f}, "
                     f"Max: {syn_df['Amplitude'].max():.2f}, "
                     f"StdDev: {syn_df['Amplitude'].std():.2f}")
        axes[i + num_historical].text(0.02, 0.9, stats_text, transform=axes[i + num_historical].transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'synthetic_tremor_comparison.png'))
    plt.show()

# Usage
if __name__ == "__main__":
    data_dir = r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg"  # Change to your directory with CSV files
    output_dir = "E:/Tremor/predictions_after"
    num_future_days = 15
    predict_future_tremors(data_dir, output_dir, num_future_days=num_future_days)