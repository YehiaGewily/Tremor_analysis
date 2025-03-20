import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch
from scipy.ndimage import gaussian_filter1d
import os

# STEP 1: Define the sampling frequency
fs = 30 #ps

# STEP 2: Load your CSV data
file_location_for_day1_before = r'E:\Tremor\before_data\Day_1_before_tre_amp.csv'


file_location_for_day15_after = r'E:\Tremor\after_data\Day_15_after.MOV_tremor_amplitudes.csv'




data_day1_before = pd.read_csv(file_location_for_day1_before, header=None, skiprows=1)
data_day15_after = pd.read_csv(file_location_for_day15_after, header=None, skiprows=1)


# Extract data
amplitude_1 = data_day1_before.iloc[1:, 1].values  # Tremor amplitudes
amplitude_day_15_after = data_day15_after.iloc[1:, 1].values  # Tremor amplitudes

# STEP 3: Compute the PSD using Welch's method for smoother results
# Use longer segments and more overlap for smoother results
nperseg = 256  # Length of each segment (increase for smoother results)
noverlap = 192  # Overlap between segments (75% overlap)

frequencies_day_1, power_day_1 = welch(amplitude_1, fs, nperseg=nperseg, noverlap=noverlap)
frequencies_day_15, power_day_15 = welch(amplitude_day_15_after, fs, nperseg=nperseg, noverlap=noverlap)

# STEP 4: Plot with improved styling
plt.figure(figsize=(15, 10))

# Subplot 1: Day 1 (Before Stimulation)
plt.subplot(2, 2, 1)
plt.semilogy(frequencies_day_1, np. sqrt(power_day_1), linewidth=2, color='blue')
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Power Spectral Density', fontsize=12)
plt.title('Day 1 (Before Stimulation) Patient #1', fontsize=14)
plt.xlim(0, 15)
plt.grid(True, linestyle='--', alpha=0.7)

# Find peaks and annotate them
from scipy.signal import find_peaks
peaks, _ = find_peaks(power_day_1, height=np.max(power_day_1)*0.3, distance=5)
for peak in peaks:
    if frequencies_day_1[peak] > 0.1:  # Ignore very low frequencies
        plt.annotate(f'{frequencies_day_1[peak]:.2f} Hz', 
                    xy=(frequencies_day_1[peak], power_day_1[peak]),
                    xytext=(0, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                    fontsize=10, color='red')

# Subplot 2: Day 15 (After Stimulation)
plt.subplot(2, 2, 2)
plt.semilogy(frequencies_day_15, np. sqrt(power_day_15), linewidth=2, color='green')
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Power Spectral Density', fontsize=12)
plt.title('Day 15 (After Stimulation) Patient 1', fontsize=14)
plt.xlim(0, 15)
plt.grid(True, linestyle='--', alpha=0.7)

# Find peaks and annotate them
peaks, _ = find_peaks(power_day_15, height=np.max(power_day_15)*0.3, distance=5)
for peak in peaks:
    if frequencies_day_15[peak] > 0.1:  # Ignore very low frequencies
        plt.annotate(f'{frequencies_day_15[peak]:.2f} Hz', 
                    xy=(frequencies_day_15[peak], power_day_15[peak]),
                    xytext=(0, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                    fontsize=10, color='red')

# Subplot 3: Combined comparison with smoothed lines
plt.subplot(2, 1, 2)
plt.semilogy(frequencies_day_1, np. sqrt(power_day_1), linewidth=2, label='Day 1 (Before)', color='blue', alpha=0.7)
plt.semilogy(frequencies_day_15, np.sqrt(power_day_15), linewidth=2, label='Day 15 (After)', color='green', alpha=0.7)

# Highlight typical tremor frequency ranges
plt.axvspan(3, 8, alpha=0.15, color='gray', label='Typical Physiological Tremor (3-8 Hz)')
plt.axvspan(8, 12, alpha=0.1, color='red', label='Potential Pathological Tremor (8-12 Hz)')

plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Power Spectral Density', fontsize=12)
plt.title('Tremor PSD: Before vs After Stimulation (patient #1)', fontsize=14)
plt.xlim(0, 15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, loc='upper right')

plt.tight_layout()

# STEP 5: Create an additional smoothed plot for even clearer visualization
plt.figure(figsize=(12, 6))

# Apply additional Gaussian smoothing to the power spectrum
smooth_power_1 = gaussian_filter1d(power_day_1, sigma=2)
smooth_power_15 = gaussian_filter1d(power_day_15, sigma=2)

plt.semilogy(frequencies_day_1, smooth_power_1, linewidth=2.5, label='Day 1 (Before)', color='blue')
plt.semilogy(frequencies_day_15, smooth_power_15, linewidth=2.5, label='Day 15 (After)', color='green')

# Add shaded confidence regions (optional)
plt.fill_between(frequencies_day_1, smooth_power_1*0.8, smooth_power_1*1.2, color='blue', alpha=0.2)
plt.fill_between(frequencies_day_15, smooth_power_15*0.8, smooth_power_15*1.2, color='green', alpha=0.2)

# Highlight key frequency ranges
plt.axvspan(3, 8, alpha=0.15, color='gray', label='Typical Physiological Tremor')

plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Frequency of the Amplitude', fontsize=12)
plt.title('Smoothed Tremor PSD: Before vs After Stimulation (Patient #1)', fontsize=14)
plt.xlim(0, 15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()