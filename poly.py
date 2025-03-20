import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline

# -------------------------------
# Data Preparation
# -------------------------------
file_location_arima = r'E:\Tremor\combined_before_stimulation_data.csv'
df = pd.read_csv(file_location_arima)
df_series = df.iloc[:, 0]  # Assuming the CSV has a single column with amplitude values

# Calculate the frequency based on total seconds over 15 days and number of observations
total_days = 15
total_seconds = total_days * 24 * 60 * 60
frequency_in_seconds = total_seconds / len(df_series)

# Create a date range spanning 15 days using a proper timedelta frequency
freq_offset = pd.to_timedelta(frequency_in_seconds, unit='s')
date_range = pd.date_range(start='01/01/2025', periods=len(df_series), freq=freq_offset)

# Create DataFrame with the correct date range and set the date as index
data = pd.DataFrame({'Date': date_range, 'Amplitude': df_series})
data.set_index('Date', inplace=True)

# -------------------------------
# Stationarity Check using ADF Test
# -------------------------------
result = adfuller(data['Amplitude'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# For ARIMA, we need a stationary series.
if result[1] > 0.05:
    print("Time series is not stationary. Applying first-order differencing.")
    data_arima = data.diff().dropna()
else:
    print("Time series is stationary.")
    data_arima = data.copy()

# -------------------------------
# ARIMA Model
# -------------------------------
# Adjust the ARIMA order as needed; here we use ARIMA(4, 1, 1) as an example.
model_arima = ARIMA(data_arima['Amplitude'], order=(4, 1, 1))
model_arima_fit = model_arima.fit()
print(model_arima_fit.summary())

# In-sample prediction using ARIMA
predictions_arima = model_arima_fit.predict(start=0, end=len(data_arima)-1)
rmse_arima = np.sqrt(mean_squared_error(data_arima['Amplitude'], predictions_arima))
print(f"ARIMA RMSE: {rmse_arima}")

# Save the ARIMA predicted data
predicted_arima_df = pd.DataFrame({
    'Date': data_arima.index,
    'Predicted_Amplitude': predictions_arima
})
predicted_arima_df.to_csv(r'E:\Tremor\predicted_data_arima.csv', index=False)
print("Predicted ARIMA data saved to 'E:\\Tremor\\predicted_data_arima.csv'")

# Plot the ARIMA predictions
plt.figure(figsize=(12, 6))
plt.plot(data_arima.index, data_arima['Amplitude'], label='Original Data', color='gray', alpha=0.6)
plt.plot(data_arima.index, predictions_arima, label='ARIMA Predicted Trend', color='orange', linewidth=2)
plt.title('ARIMA: Original Data vs Predicted Trend')
plt.xlabel('Date')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# -------------------------------
# Bayesian Polynomial Regression
# -------------------------------
# For Bayesian polynomial regression, we use the original (non-differenced) data.
# Create a time index (0 to n-1) as the independent variable.
X = np.arange(len(data)).reshape(-1, 1)
y = data['Amplitude'].values

# Define the polynomial degree (e.g., 3 for a cubic polynomial)
degree = 4

# Create a pipeline that first generates polynomial features then applies Bayesian Ridge Regression.
model_poly = make_pipeline(PolynomialFeatures(degree), BayesianRidge())

# Fit the model to the data
model_poly.fit(X, y)

# Generate predictions with the Bayesian polynomial regression model
predictions_poly = model_poly.predict(X)

# Plot the Bayesian polynomial regression trend
plt.figure(figsize=(12, 6))
plt.plot(data.index, y, label='Original Data', color='gray', alpha=0.6)
plt.plot(data.index, predictions_poly, label='Bayesian Polynomial Regression Trend', color='red', linewidth=2)
plt.title('Bayesian Polynomial Regression: Original Data vs Trend')
plt.xlabel('Date')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# -------------------------------
# Comparison Plot: ARIMA vs Bayesian Polynomial Regression
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Amplitude'], label='Original Data', color='gray', alpha=0.6)
# For ARIMA, we use the differenced data's index; ensure alignment if needed.
plt.plot(data_arima.index, predictions_arima, label='ARIMA Trend', color='orange', linewidth=2)
plt.plot(data.index, predictions_poly, label='Bayesian Poly Trend', color='red', linewidth=2)
plt.title('Comparison: ARIMA vs Bayesian Polynomial Regression Trends')
plt.xlabel('Date')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
