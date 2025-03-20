import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the prepared data
data_file = r'E:\Tremor\combined_after_stimulation_seconds.csv'
data = pd.read_csv(data_file)

# Display the first few rows to verify
print("First 5 rows of the loaded data:")
print(data.head())

# Fix the 'ds' column based on frame rate
frame_rate = 30  # Adjust if different
data['ds'] = data.index / frame_rate  # Rebuild 'ds' as continuous time in seconds

# Remove duplicates and sort by 'ds' to ensure proper time sequence
data = data.drop_duplicates(subset='ds').sort_values('ds').reset_index(drop=True)

# Display fixed data sample
print("\nFixed 'ds' column (time in seconds):")
print(data.head())

# Initialize the Prophet model
model = Prophet()

# Fit the model
print("\nTraining the model, please wait...")
model.fit(data)

# Forecast for the same number of rows as the original dataset
original_length = len(data)
print(f"\nForecasting for the same number of rows ({original_length})...")
future = model.make_future_dataframe(periods=original_length, freq='S')

# Predict future values
forecast = model.predict(future)

# Save the forecast to a CSV file
forecast_file = r'E:\Tremor\forecast_after_stimulation_fixed_time.csv'
forecast.to_csv(forecast_file, index=False)
print(f"\nForecast saved as '{forecast_file}'")

# Display a sample of the forecasted data
print("\nSample of forecasted data:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the forecasted data
plt.figure(figsize=(14, 7))
model.plot(forecast)
plt.title('Tremor Amplitude Forecast (Fixed Time in Seconds)')
plt.xlabel('Time (in seconds)')
plt.ylabel('Tremor Amplitude')
plt.grid(True)
plt.show()

# Plot the forecast components (trends, seasonality)
plt.figure(figsize=(14, 10))
model.plot_components(forecast)
plt.show()

# Plot Actual vs. Predicted Data
plt.figure(figsize=(14, 7))
plt.plot(data['ds'], data['y'], label='Actual Data', linestyle='-', marker='o', markersize=2)
plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Data', linestyle='--')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.3)
plt.title('Actual vs. Forecasted Tremor Amplitude (Fixed Time)')
plt.xlabel('Time (in seconds)')
plt.ylabel('Tremor Amplitude')
plt.legend()
plt.grid(True)
plt.show()
