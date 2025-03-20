import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.ndimage import gaussian_filter1d
import os

###############################################################################
#                          User-Specified Output Folder                       #
###############################################################################
output_folder = r'E:\Tremor\visualizations'  # <-- CHANGE THIS TO YOUR DESIRED DIRECTORY
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

###############################################################################
#                          Sampling Frequency & Data                          #
###############################################################################
fs = 30  # Hz

file_location_for_day1_before = r'E:\Tremor\before_data\Day_1_before_tre_amp.csv'
file_location_for_day15_after = r'E:\Tremor\after_data\Day_15_after.MOV_tremor_amplitudes.csv'
file_location_for_day30_after = r'E:\Tremor\predictions\after\predicted_tremor_day_30_bootstrap.csv'

data_day1_before = pd.read_csv(file_location_for_day1_before, header=None, skiprows=1)
data_day15_after = pd.read_csv(file_location_for_day15_after, header=None, skiprows=1)
data_day30_after = pd.read_csv(file_location_for_day30_after, header=None, skiprows=1)

# Extract amplitude data (time-domain)
amplitude_1 = data_day1_before.iloc[1:, 1].values            # Day 1 (Before)
amplitude_day_15_after = data_day15_after.iloc[1:, 1].values # Day 15 (After)
amplitude_day_30_after = data_day30_after.iloc[1:, 1].values # Day 30 (After)

###############################################################################
#                           Compute Welch PSD                                 #
###############################################################################
nperseg = 256
noverlap = 192

frequencies_day_1, power_day_1 = welch(amplitude_1, fs, nperseg=nperseg, noverlap=noverlap)
frequencies_day_15, power_day_15 = welch(amplitude_day_15_after, fs, nperseg=nperseg, noverlap=noverlap)
frequencies_day_30, power_day_30 = welch(amplitude_day_30_after, fs, nperseg=nperseg, noverlap=noverlap)

###############################################################################
#                Figure 1: PSD with 3 Subplots (Day 1, 15, 30)                #
###############################################################################
fig1 = plt.figure(figsize=(15, 15))

# Subplot 1: Day 1
ax1 = fig1.add_subplot(3, 1, 1)
ax1.semilogy(frequencies_day_1, power_day_1, linewidth=2, color='blue')
ax1.set_xlabel('Frequency (Hz)', fontsize=12)
ax1.set_ylabel('Power Spectral Density', fontsize=12)
ax1.set_title('Day 1 (Before Stimulation)', fontsize=14)
ax1.set_xlim(0, 15)
ax1.grid(True, linestyle='--', alpha=0.7)

# Find peaks
peaks, _ = find_peaks(power_day_1, height=np.max(power_day_1)*0.3, distance=5)
for peak in peaks:
    if frequencies_day_1[peak] > 0.1:
        ax1.annotate(f'{frequencies_day_1[peak]:.2f} Hz', 
                     xy=(frequencies_day_1[peak], power_day_1[peak]),
                     xytext=(0, 20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                     fontsize=10, color='red')

# Subplot 2: Day 15
ax2 = fig1.add_subplot(3, 1, 2)
ax2.semilogy(frequencies_day_15, power_day_15, linewidth=2, color='green')
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Power Spectral Density', fontsize=12)
ax2.set_title('Day 15 (After Stimulation)', fontsize=14)
ax2.set_xlim(0, 15)
ax2.grid(True, linestyle='--', alpha=0.7)

peaks, _ = find_peaks(power_day_15, height=np.max(power_day_15)*0.3, distance=5)
for peak in peaks:
    if frequencies_day_15[peak] > 0.1:
        ax2.annotate(f'{frequencies_day_15[peak]:.2f} Hz', 
                     xy=(frequencies_day_15[peak], power_day_15[peak]),
                     xytext=(0, 20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                     fontsize=10, color='red')

# Subplot 3: Day 30
ax3 = fig1.add_subplot(3, 1, 3)
ax3.semilogy(frequencies_day_30, power_day_30, linewidth=2, color='purple')
ax3.set_xlabel('Frequency (Hz)', fontsize=12)
ax3.set_ylabel('Power Spectral Density', fontsize=12)
ax3.set_title('Day 30 (Predicted Data, After Stimulation)', fontsize=14)
ax3.set_xlim(0, 15)
ax3.grid(True, linestyle='--', alpha=0.7)

peaks, _ = find_peaks(power_day_30, height=np.max(power_day_30)*0.3, distance=5)
for peak in peaks:
    if frequencies_day_30[peak] > 0.1:
        ax3.annotate(f'{frequencies_day_30[peak]:.2f} Hz', 
                     xy=(frequencies_day_30[peak], power_day_30[peak]),
                     xytext=(0, 20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                     fontsize=10, color='red')

plt.tight_layout()
fig1.savefig(os.path.join(output_folder, 'Fig1_PSD_Subplots.png'), dpi=300)

###############################################################################
#               Figure 2: Combined PSD Comparison (Day 1, 15, 30)             #
###############################################################################
fig2 = plt.figure(figsize=(15, 10))
plt.semilogy(frequencies_day_1, power_day_1, linewidth=2, label='Day 1 (Before)', color='blue', alpha=0.7)
plt.semilogy(frequencies_day_15, power_day_15, linewidth=2, label='Day 15 (After)', color='green', alpha=0.7)
plt.semilogy(frequencies_day_30, power_day_30, linewidth=2, label='Day 30 (After)', color='purple', alpha=0.7)

plt.axvspan(3, 8, alpha=0.15, color='gray', label='Typical Physiological Tremor (3-8 Hz)')
plt.axvspan(8, 12, alpha=0.1, color='red', label='Potential Pathological Tremor (8-12 Hz)')

plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Power Spectral Density', fontsize=12)
plt.title('Tremor PSD: Before vs After Stimulation (Day 15 & Day 30)', fontsize=14)
plt.xlim(0, 15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, loc='upper right')
plt.tight_layout()
fig2.savefig(os.path.join(output_folder, 'Fig2_Combined_PSD.png'), dpi=300)

###############################################################################
#     Figure 3: Smoothed PSD Comparison (Gaussian Filter, Day 1, 15, 30)      #
###############################################################################
fig3 = plt.figure(figsize=(12, 6))

smooth_power_1 = gaussian_filter1d(power_day_1, sigma=2)
smooth_power_15 = gaussian_filter1d(power_day_15, sigma=2)
smooth_power_30 = gaussian_filter1d(power_day_30, sigma=2)

plt.semilogy(frequencies_day_1, smooth_power_1, linewidth=2.5, label='Day 1 (Before)', color='blue')
plt.semilogy(frequencies_day_15, smooth_power_15, linewidth=2.5, label='Day 15 (After)', color='green')
plt.semilogy(frequencies_day_30, smooth_power_30, linewidth=2.5, label='Day 30 (After)', color='purple')

plt.fill_between(frequencies_day_1, smooth_power_1*0.8, smooth_power_1*1.2, color='blue', alpha=0.2)
plt.fill_between(frequencies_day_15, smooth_power_15*0.8, smooth_power_15*1.2, color='green', alpha=0.2)
plt.fill_between(frequencies_day_30, smooth_power_30*0.8, smooth_power_30*1.2, color='purple', alpha=0.2)

plt.axvspan(3, 8, alpha=0.15, color='gray', label='Typical Physiological Tremor')

plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Power Spectral Density (smoothed)', fontsize=12)
plt.title('Smoothed Tremor PSD: Multiple Days', fontsize=14)
plt.xlim(0, 15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
fig3.savefig(os.path.join(output_folder, 'Fig3_Smoothed_PSD.png'), dpi=300)

###############################################################################
#                  NEW: Plot the Amplitude Spectrum (Day 1, 15, 30)           #
###############################################################################
# Convert PSD to amplitude by taking the sqrt
amplitude_spectrum_day1 = np.sqrt(power_day_1)
amplitude_spectrum_day15 = np.sqrt(power_day_15)
amplitude_spectrum_day30 = np.sqrt(power_day_30)

###############################################################################
#        Figure 4: Amplitude Spectrum with Subplots (Day 1, 15, 30)           #
###############################################################################
fig4 = plt.figure(figsize=(15, 10))

# Subplot 1: Day 1
ax4_1 = fig4.add_subplot(3, 1, 1)
ax4_1.plot(frequencies_day_1, amplitude_spectrum_day1, linewidth=2, color='blue')
ax4_1.set_xlabel('Frequency (Hz)', fontsize=12)
ax4_1.set_ylabel('Amplitude', fontsize=12)
ax4_1.set_title('Amplitude Spectrum (Day 1 - Before Stimulation)', fontsize=14)
ax4_1.set_xlim(0, 15)
ax4_1.grid(True, linestyle='--', alpha=0.7)

# Subplot 2: Day 15
ax4_2 = fig4.add_subplot(3, 1, 2)
ax4_2.plot(frequencies_day_15, amplitude_spectrum_day15, linewidth=2, color='green')
ax4_2.set_xlabel('Frequency (Hz)', fontsize=12)
ax4_2.set_ylabel('Amplitude', fontsize=12)
ax4_2.set_title('Amplitude Spectrum (Day 15 - After Stimulation)', fontsize=14)
ax4_2.set_xlim(0, 15)
ax4_2.grid(True, linestyle='--', alpha=0.7)

# Subplot 3: Day 30
ax4_3 = fig4.add_subplot(3, 1, 3)
ax4_3.plot(frequencies_day_30, amplitude_spectrum_day30, linewidth=2, color='purple')
ax4_3.set_xlabel('Frequency (Hz)', fontsize=12)
ax4_3.set_ylabel('Amplitude', fontsize=12)
ax4_3.set_title('Amplitude Spectrum (Day 30 - Predicted, After Stimulation)', fontsize=14)
ax4_3.set_xlim(0, 15)
ax4_3.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
fig4.savefig(os.path.join(output_folder, 'Fig4_Amplitude_Subplots.png'), dpi=300)

###############################################################################
#           Figure 5: Combined Amplitude Spectrum (Day 1, 15, 30)             #
###############################################################################
fig5 = plt.figure(figsize=(12, 6))
plt.plot(frequencies_day_1, amplitude_spectrum_day1, linewidth=2, label='Day 1 (Before)', color='blue', alpha=0.7)
plt.plot(frequencies_day_15, amplitude_spectrum_day15, linewidth=2, label='Day 15 (After)', color='green', alpha=0.7)
plt.plot(frequencies_day_30, amplitude_spectrum_day30, linewidth=2, label='Day 30 (After)', color='purple', alpha=0.7)

plt.axvspan(3, 8, alpha=0.15, color='gray', label='Typical Tremor Range (3-8 Hz)')

plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.title('Amplitude Spectrum Comparison', fontsize=14)
plt.xlim(0, 15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
fig5.savefig(os.path.join(output_folder, 'Fig5_Combined_Amplitude.png'), dpi=300)

###############################################################################
#                          Show all figures (optional)                        #
###############################################################################
plt.show()
