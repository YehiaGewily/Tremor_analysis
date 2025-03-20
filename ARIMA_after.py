import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates

def load_and_prepare_data(file_path, date_col=None, value_col=0, 
                         segment_info=None, sampling_rate=None):
    """
    Load and prepare time series data for analysis, accounting for separate recording sessions.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    date_col : str, optional
        Name of the date column (if exists)
    value_col : int or str, default 0
        Column index or name containing the values to analyze
    segment_info : dict, optional
        Information about data segments, with keys:
        - 'n_segments': Number of segments (e.g., 15 days)
        - 'segment_duration': Duration of each segment in seconds (e.g., 60)
    sampling_rate : float, optional
        Sampling rate in Hz, used to calculate time intervals
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with DatetimeIndex and amplitude values
    list
        List of segment boundaries (start, end) indices
    """
    print(f"Loading data from: {file_path}")
    
    df = pd.read_csv(file_path)
    series = df.iloc[:, value_col] if isinstance(value_col, int) else df[value_col]
    
    if date_col and date_col in df.columns:
        # Use existing date column
        df[date_col] = pd.to_datetime(df[date_col])
        data = df.set_index(date_col)
        amplitude_col = value_col if isinstance(value_col, str) else df.columns[value_col]
    else:
        # Create appropriate time indices based on segment information
        if segment_info and sampling_rate:
            n_segments = segment_info.get('n_segments', 1)
            segment_duration = segment_info.get('segment_duration', 60)  # in seconds
            
            # Calculate points per segment
            points_per_segment = int(segment_duration * sampling_rate)
            
            # Check if data length matches expected length
            expected_length = n_segments * points_per_segment
            if len(series) != expected_length:
                print(f"Warning: Data length ({len(series)}) doesn't match expected length ({expected_length})")
                print(f"Adjusting segment duration to match data length")
                points_per_segment = len(series) // n_segments
            
            # Create time index reflecting the segmented nature of the data
            time_indices = []
            segment_boundaries = []
            
            for i in range(n_segments):
                segment_start = i * points_per_segment
                segment_end = (i + 1) * points_per_segment - 1
                segment_boundaries.append((segment_start, segment_end))
                
                # Create time index for this segment
                segment_times = pd.date_range(
                    start=pd.Timestamp(f"2025-01-{i+1}"),  # Use different dates for each segment
                    periods=points_per_segment,
                    freq=f"{1000/sampling_rate:.6f}ms"  # Convert sampling rate to milliseconds
                )
                time_indices.extend(segment_times)
            
            # Create DataFrame with the proper time index
            if len(time_indices) > len(series):
                time_indices = time_indices[:len(series)]
            elif len(time_indices) < len(series):
                # Extend time indices if needed
                last_time = time_indices[-1]
                freq = time_indices[1] - time_indices[0]
                additional_times = [last_time + freq * (i+1) for i in range(len(series) - len(time_indices))]
                time_indices.extend(additional_times)
            
            data = pd.DataFrame({'Amplitude': series.values}, index=time_indices)
        else:
            # Default approach if no segment info provided
            sampling_freq = '100ms'  # Default if not specified
            date_range = pd.date_range(start='2025-01-01', periods=len(series), freq=sampling_freq)
            data = pd.DataFrame({'Amplitude': series.values}, index=date_range)
            segment_boundaries = []
        
        amplitude_col = 'Amplitude'
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    if segment_boundaries:
        print(f"Data divided into {len(segment_boundaries)} segments")
    
    return data, amplitude_col, segment_boundaries

def check_stationarity(series, significance=0.05):
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to check
    significance : float, default 0.05
        Significance level for the test
        
    Returns:
    --------
    bool
        True if series is stationary, False otherwise
    int
        Recommended differencing order
    """
    result = adfuller(series.dropna())
    
    print('\n=== Stationarity Check ===')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    is_stationary = result[1] < significance
    
    if is_stationary:
        print("Series is STATIONARY (no differencing needed)")
        return True, 0
    else:
        # Check first difference
        diff1 = series.diff().dropna()
        result_diff1 = adfuller(diff1)
        
        if result_diff1[1] < significance:
            print("Series becomes stationary after first differencing")
            return False, 1
        else:
            print("Series needs second differencing or transformation")
            return False, 2

def determine_arima_order(series, max_p=5, max_q=5, d=1):
    """
    Determine optimal ARIMA orders using ACF and PACF plots.
    This is a simple approach - for more rigorous selection, consider
    using auto_arima from pmdarima or grid search.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    max_p, max_q : int
        Maximum AR and MA orders to consider
    d : int
        Differencing order
        
    Returns:
    --------
    tuple
        Suggested ARIMA order (p, d, q)
    """
    print('\n=== Determining ARIMA Parameters ===')
    
    # Create a figure for ACF and PACF plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Apply differencing if needed
    if d > 0:
        series_to_analyze = series.diff(d).dropna()
    else:
        series_to_analyze = series
    
    # Plot ACF and PACF
    plot_acf(series_to_analyze, ax=ax1, lags=max_q)
    plot_pacf(series_to_analyze, ax=ax2, lags=max_p)
    
    plt.tight_layout()
    plt.show()
    
    # This function just provides the plots for manual inspection
    # For a simple heuristic, we could suggest:
    print("Based on ACF/PACF plots, suggest visual inspection for parameter selection.")
    print("Using AIC for multiple models would be more robust.")
    
    # Return default parameters - in practice, choose based on the plots or AIC
    suggested_p = 1  # Placeholder - should be determined from PACF
    suggested_q = 1  # Placeholder - should be determined from ACF
    
    return (suggested_p, d, suggested_q)

def fit_arima_model(series, order, verbose=True):
    """
    Fit ARIMA model to time series data.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    order : tuple
        ARIMA order (p, d, q)
    verbose : bool, default True
        Whether to print model summary
        
    Returns:
    --------
    fitted model
    """
    print(f'\n=== Fitting ARIMA({order[0]}, {order[1]}, {order[2]}) Model ===')
    
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    
    if verbose:
        print(model_fit.summary())
    
    return model_fit

def evaluate_model(model_fit, series, test_size=0):
    """
    Evaluate ARIMA model performance.
    
    Parameters:
    -----------
    model_fit : ARIMAResults
        Fitted ARIMA model
    series : pd.Series
        Original time series
    test_size : int, default 0
        Number of observations to use for testing
        
    Returns:
    --------
    pd.Series
        Predicted values
    float
        RMSE
    """
    print('\n=== Model Evaluation ===')
    
    if test_size > 0:
        # Split into train/test
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]
        
        # Fit on training data
        model = ARIMA(train, order=model_fit.model.order)
        model_fit = model.fit()
        
        # Forecast test period
        predictions = model_fit.forecast(steps=test_size)
        rmse = np.sqrt(mean_squared_error(test, predictions))
        
        print(f"Out-of-sample RMSE: {rmse:.4f}")
        
        # Refit on full dataset
        model = ARIMA(series, order=model_fit.model.order)
        model_fit = model.fit()
    
    # In-sample evaluation
    predictions = model_fit.predict(start=0, end=len(series)-1)
    rmse = np.sqrt(mean_squared_error(series, predictions))
    
    print(f"In-sample RMSE: {rmse:.4f}")
    
    return predictions, rmse

def forecast_future(model_fit, steps=30, plot=True):
    """
    Generate future forecasts from ARIMA model.
    
    Parameters:
    -----------
    model_fit : ARIMAResults
        Fitted ARIMA model
    steps : int, default 30
        Number of steps to forecast
    plot : bool, default True
        Whether to plot the forecast
        
    Returns:
    --------
    pd.Series
        Forecasted values
    """
    print(f'\n=== Forecasting {steps} steps ahead ===')
    
    # Get forecast
    forecast = model_fit.forecast(steps=steps)
    
    # Confidence intervals
    pred_ci = model_fit.get_forecast(steps=steps).conf_int()
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(forecast.index, forecast, color='red', label='Forecast')
        plt.fill_between(pred_ci.index, 
                         pred_ci.iloc[:, 0], 
                         pred_ci.iloc[:, 1], 
                         color='pink', alpha=0.3)
        plt.title('ARIMA Forecast with Confidence Intervals')
        plt.xlabel('Date')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return forecast

def plot_results(original_series, predictions, segment_boundaries=None, window_size=30):
    """
    Plot original data and predictions with various visualizations,
    highlighting different recording segments if available.
    
    Parameters:
    -----------
    original_series : pd.Series
        Original time series data
    predictions : pd.Series
        Predicted values from model
    segment_boundaries : list, optional
        List of (start, end) indices for each segment
    window_size : int, default 30
        Window size for moving average smoothing
    """
    print('\n=== Visualizing Results ===')
    
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Raw data vs predictions with segment highlights if available
    axes[0, 0].plot(original_series.index, original_series, 
                   label='Original', color='gray', alpha=0.6)
    axes[0, 0].plot(predictions.index, predictions, 
                   label='Predicted', color='red', linewidth=1.5)
    
    # Highlight different segments if boundaries are provided
    if segment_boundaries:
        for i, (start, end) in enumerate(segment_boundaries):
            # Get the corresponding dates for the segment
            segment_start_date = original_series.index[start]
            segment_end_date = original_series.index[min(end, len(original_series.index)-1)]
            
            # Add segment highlight
            axes[0, 0].axvspan(segment_start_date, segment_end_date, 
                              alpha=0.1, color=f'C{i%10}')
            
            # Add segment label to the first subplot only
            if i == 0:
                axes[0, 0].text(
                    segment_start_date, 
                    original_series.max() * 0.95,
                    f"Segment {i+1}",
                    fontsize=8,
                    ha='center'
                )
            
    axes[0, 0].set_title('Original Data vs Predictions (60s segments from different days)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Smoothed data vs smoothed predictions (Moving Average)
    smoothed_original = original_series.rolling(window=window_size, min_periods=1).mean()
    smoothed_predictions = predictions.rolling(window=window_size, min_periods=1).mean()
    
    axes[0, 1].plot(smoothed_original.index, smoothed_original, 
                   label=f'Original (MA-{window_size})', color='blue')
    axes[0, 1].plot(smoothed_predictions.index, smoothed_predictions, 
                   label=f'Predicted (MA-{window_size})', color='orange')
    axes[0, 1].set_title(f'Smoothed Trends (Window: {window_size})')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Amplitude (Smoothed)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals
    residuals = original_series - predictions
    
    axes[1, 0].plot(residuals.index, residuals, color='green')
    axes[1, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[1, 0].set_title('Residuals (Original - Predicted)')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Residual Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals histogram
    axes[1, 1].hist(residuals, bins=30, color='green', alpha=0.7)
    axes[1, 1].set_title('Residuals Distribution')
    axes[1, 1].set_xlabel('Residual Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create one more separate plot for the overall comparison
    plt.figure(figsize=(12, 6))
    
    # Plot original with lighter color
    plt.plot(original_series.index, original_series, 
             label='Historical Data', color='blue', alpha=0.4)
    
    # Plot smoothed original with darker color
    plt.plot(smoothed_original.index, smoothed_original, 
             label=f'Historical Trend (MA-{window_size})', color='blue', linewidth=2)
    
    # Plot smoothed predictions
    plt.plot(smoothed_predictions.index, smoothed_predictions, 
             label=f'Model Fit (MA-{window_size})', color='red', linewidth=2)
    
    plt.title('Tremor Amplitude: Historical Data and Model Fit')
    plt.xlabel('Date')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis with dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

def save_results(original_series, predictions, predictions_future=None, output_dir='.'):
    """
    Save original data, predictions, and future forecasts to CSV.
    
    Parameters:
    -----------
    original_series : pd.Series
        Original time series data
    predictions : pd.Series
        Predicted values from model
    predictions_future : pd.Series, optional
        Future forecasted values
    output_dir : str, default '.'
        Directory to save results
    """
    print('\n=== Saving Results ===')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Combine original data and predictions
    results_df = pd.DataFrame({
        'Date': original_series.index,
        'Original': original_series.values,
        'Predicted': predictions.values,
        'Residuals': original_series.values - predictions.values
    })
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'tremor_analysis_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # Save future predictions if available
    if predictions_future is not None:
        future_df = pd.DataFrame({
            'Date': predictions_future.index,
            'Forecast': predictions_future.values
        })
        
        future_path = os.path.join(output_dir, 'tremor_forecast.csv')
        future_df.to_csv(future_path, index=False)
        print(f"Forecast saved to: {future_path}")

def analyze_segments_separately(data, amplitude_col, segment_boundaries):
    """
    Analyze each segment (day) separately and compare results.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Complete dataset
    amplitude_col : str
        Column name for amplitude values
    segment_boundaries : list
        List of (start, end) indices for each segment
        
    Returns:
    --------
    dict
        Dictionary with analysis results for each segment
    """
    print('\n=== Analyzing Segments Separately ===')
    
    segment_results = {}
    
    # Set up a figure for comparing segments
    plt.figure(figsize=(15, 10))
    
    for i, (start, end) in enumerate(segment_boundaries):
        # Extract segment data
        segment_data = data.iloc[start:end+1].copy()
        
        print(f"\nAnalyzing Segment {i+1} ({len(segment_data)} data points)")
        
        # Quick stationarity check
        is_stationary, diff_order = check_stationarity(segment_data[amplitude_col])
        
        # Fit ARIMA model (simplified parameters)
        arima_order = (2, diff_order, 2)  # Simplified model for each segment
        try:
            model_fit = fit_arima_model(segment_data[amplitude_col], order=arima_order, verbose=False)
            predictions = model_fit.predict(start=0, end=len(segment_data)-1)
            rmse = np.sqrt(mean_squared_error(segment_data[amplitude_col], predictions))
            
            # Store results
            segment_results[i] = {
                'data': segment_data,
                'predictions': predictions,
                'rmse': rmse,
                'model': model_fit
            }
            
            # Plot this segment
            plt.plot(segment_data.index, segment_data[amplitude_col], 
                    alpha=0.4, label=f"Segment {i+1}")
            
        except Exception as e:
            print(f"Error analyzing segment {i+1}: {e}")
    
    plt.title('Comparison of All Segments (15 Days, 60s Each)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Compare RMSEs
    if segment_results:
        rmses = [res['rmse'] for res in segment_results.values()]
        segment_nums = list(segment_results.keys())
        
        plt.figure(figsize=(10, 6))
        plt.bar(segment_nums, rmses)
        plt.title('RMSE Comparison Across Segments')
        plt.xlabel('Segment Number')
        plt.ylabel('RMSE')
        plt.xticks(segment_nums)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return segment_results

def main():
    """Main function to run the entire analysis pipeline."""
    # Setup parameters
    file_path = r'E:\Tremor\combined_after_stimulation_data.csv'  # Use absolute path
    output_dir = 'tremor_analysis_output'
    
    # Define segment information based on your description
    segment_info = {
        'n_segments': 15,           # 15 different days
        'segment_duration': 60      # 60 seconds per day
    }
    sampling_rate = 22.12          # Estimated based on approx 19909 datapoints over 15*60 seconds
    
    # 1. Load and prepare data with segment information
    data, amplitude_col, segment_boundaries = load_and_prepare_data(
        file_path, 
        date_col=None,             # No date column in the original data
        value_col=0,               # First column contains amplitude values
        segment_info=segment_info,
        sampling_rate=sampling_rate
    )
    
    # 2. Check stationarity and determine differencing order
    is_stationary, diff_order = check_stationarity(data[amplitude_col])
    
    # 3. Determine ARIMA order
    suggested_order = determine_arima_order(data[amplitude_col], d=diff_order)
    
    # Allow custom order if needed
    # arima_order = (4, 1, 1)  # Manual override if needed
    arima_order = suggested_order
    
    # 4. Fit ARIMA model
    model_fit = fit_arima_model(data[amplitude_col], order=arima_order)
    
    # 5. Evaluate model
    predictions, rmse = evaluate_model(model_fit, data[amplitude_col], test_size=0)
    
    # 6. Analyze individual segments
    segment_results = analyze_segments_separately(data, amplitude_col, segment_boundaries)
    
    # 7. Generate future forecast
    # Note: Forecasting with this type of segmented data is complex
    # This is only meaningful if we're predicting future behavior within a segment
    forecast_seconds = 10  # Forecast 10 seconds ahead instead of days
    steps = int(forecast_seconds * sampling_rate)
    future_forecast = forecast_future(model_fit, steps=steps)
    
    # 8. Visualize results
    plot_results(data[amplitude_col], predictions, segment_boundaries, window_size=30)
    
    # 9. Save results
    save_results(data[amplitude_col], predictions, future_forecast, output_dir)
    
    print("\nAnalysis completed successfully.")
    print("\nNote: Since your data consists of 15 separate 60-second recordings,")
    print("traditional time series forecasting across days may not be meaningful.")
    print("Consider whether you're looking for patterns within each 60-second session")
    print("or trying to detect changes across different days.")

if __name__ == "__main__":
    main()