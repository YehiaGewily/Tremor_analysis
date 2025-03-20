import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# --- Data Preparation ---
file_location_arima = r'E:\Tremor\combined_before_stimulation_data.csv'
df = pd.read_csv(file_location_arima)
df_series = df.iloc[:, 0]  # Assuming a single column


total_days = 15
total_seconds = total_days * 24 * 60 * 60
frequency_in_seconds = total_seconds / len(df_series)

# Create a date range spanning 15 days with the calculated frequency
date_range = pd.date_range(start='01/01/2025', periods=19265, freq='5S')

# Create DataFrame with the correct date range
data = pd.DataFrame({'Date': date_range, 'Amplitude': df_series})
data.set_index('Date', inplace=True)

# --- Stationarity Check ---
result = adfuller(data['Amplitude'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# For this example, we assume the series is stationary.
if result[1] > 0.05:
    print("Time series is not stationary. Differencing will be applied.")
    data = data.diff().dropna()
else:
    print("Time series is stationary.")

# --- Fit ARIMA Model on the Entire Dataset ---
# Adjust ARIMA order as needed; here we use ARIMA(1, 0, 1) as an example.
model = ARIMA(data['Amplitude'], order=(4, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# --- In-Sample Prediction ---
predictions = model_fit.predict(start=0, end=len(data)-1)

# Optionally compute RMSE (for reference)
rmse = np.sqrt(mean_squared_error(data['Amplitude'], predictions))
print(f"RMSE: {rmse}")

# --- Save the Predicted Data ---
predicted_df = pd.DataFrame({
    'Date': data.index,
    'Predicted_Amplitude': predictions
})
predicted_df.to_csv(r'E:\Tremor\predicted_data.csv', index=False)
print("Predicted data saved to 'E:\\Tremor\\predicted_data.csv'")

# --- Option 1: Visualize as Trend Lines (Raw Data) ---
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Amplitude'], label='Original Data', color='gray', alpha=0.6)
plt.plot(data.index, predictions, label='Predicted Trend', color='orange', linewidth=2)
plt.title('Trend Comparison: Original Data vs Predicted Trend')
plt.xlabel('Date')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# --- Option 2: Visualize Smoothed Trends using Moving Averages ---
# Create a 30-day moving average (adjust the window size as needed)
data['Trend'] = data['Amplitude'].rolling(window=30, min_periods=1).mean()
predicted_df['Trend'] = predictions.rolling(window=30, min_periods=1).mean()

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Trend'], label='Original Trend (15-day MA)', color='blue')
plt.plot(data.index, predicted_df['Trend'], label='Predicted Trend (15-day MA)', color='orange')
plt.title('Smoothed Trend Comparison: Original vs Predicted')
plt.xlabel('Date')
plt.ylabel('Amplitude (Smoothed)')
plt.legend()
plt.show()


# Assume your predicted trend (smoothed) is in predicted_df['Trend']
# and your original data is in data with index as dates

# Append the predicted trend to the origi


predicted_trend = predicted_df['Trend']
predicted_trend.name = 'Amplitude'

# Concatenate the two series. This creates one continuous Series.
combined_series = pd.concat([data['Amplitude'], predicted_trend])
combined_series.sort_index(inplace=True)


# Optional: If there are overlapping dates, you might want to drop duplicates:
combined_series = combined_series[~combined_series.index.duplicated(keep='first')]

# Plot the combined series
plt.figure(figsize=(12, 6))
plt.plot(combined_series.index, combined_series, label='Amplitude + Predicted Trend', color='orange')
plt.title('Combined Amplitude and Predicted Trend')
plt.xlabel('Date')
plt.ylabel('Amplitude')
plt.legend()
plt.show()



combined_series = pd.concat([data['Amplitude'], predicted_df['Trend']])
combined_series.sort_index(inplace=True)


# Calculate a rolling average over the combined series (e.g., 30-day moving average)
combined_rolling = combined_series.rolling(window=30, min_periods=1).mean()

# Plot the combined rolling average trend
plt.figure(figsize=(12, 6))
plt.plot(combined_rolling.index, combined_rolling, label='Combined 30-day Moving Average', color='green')
plt.title('Stock Trend-like Visualization: Amplitude + Predicted Trend')
plt.xlabel('Date')
plt.ylabel('Amplitude (Smoothed)')
plt.legend()
plt.show()


combined_series = pd.concat([data['Amplitude'], predicted_df['Trend']])
combined_series.sort_index(inplace=True)


# Apply a rolling average with a larger window (e.g., 60 days) for smoothing.
smoothed_series = combined_series.rolling(window=60, min_periods=1).mean()

# Plot the smoothed series
plt.figure(figsize=(12, 6))
plt.plot(smoothed_series.index, smoothed_series, label='Smoothed Trend (30-day MA)', color='blue', linewidth=2)
plt.title('Smoothed Trend: Combined Amplitude and Predicted Trend')
plt.xlabel('Date')
plt.ylabel('Amplitude (Smoothed)')
plt.legend()
plt.show()


combined_series = pd.concat([data['Amplitude'], predicted_df['Trend']])
combined_series.sort_index(inplace=True)  # Ensure chronological order
window_size = 60  # or any other window for smoothing
smoothed_series = combined_series.rolling(window=window_size, min_periods=1).mean()
# Get the last timestamp of the historical data
last_historical_date = data.index.max()

# Slice the rolling series into two parts:
historical_rolling = smoothed_series.loc[:last_historical_date]
predicted_rolling = smoothed_series.loc[last_historical_date:]
plt.figure(figsize=(12, 6))

# Plot historical portion
plt.plot(historical_rolling.index, historical_rolling,
         label='Historical (Smoothed)', color='blue')

# Plot predicted portion
plt.plot(predicted_rolling.index, predicted_rolling,
         label='Predicted (Smoothed)', color='orange')

plt.title('Combined Smoothed Series: Different Colors for Historical vs. Predicted')
plt.xlabel('Date')
plt.ylabel('Amplitude (Smoothed)')
plt.legend()
plt.show()
