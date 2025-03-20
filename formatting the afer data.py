import pandas as pd
import os

# Define the folder path containing your CSV files
folder_path = r'E:\Tremor\before_data'  # Replace with your folder path

# Get a list of all CSV files in the folder
all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# Create an empty list to store individual DataFrames
data_list = []

# Define frame rate (adjust if different)
frame_rate = 30  # Frames per second

# Loop through each file, read it, and append to the data_list
for file in sorted(all_files):
    df = pd.read_csv(file)
    
    # Convert frames to seconds
    df['Time'] = df['Frame'] / frame_rate
    
    # Rename columns for Prophet compatibility
    df = df.rename(columns={'Time': 'ds', 'Tremor_Amplitude': 'y'})
    
    # Append to data_list
    data_list.append(df[['ds', 'y']])

# Concatenate all DataFrames into one
after_data = pd.concat(data_list, ignore_index=True)

# Sort by time to ensure chronological order
after_data = after_data.sort_values('ds')

# Save the combined data to a new CSV file
output_file = r'E:\Tremor\combined_before_stimulation_data.csv'  # Replace if you want a different path
after_data.to_csv(output_file, index=False)

# Display confirmation and a sample of the data
print(f"Dataset saved as '{output_file}'")
print("First 5 rows of combined data in seconds:")
print(after_data.head())
print("\nLast 5 rows of combined data in seconds:")
print(after_data.tail())
