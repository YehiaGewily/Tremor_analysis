import pandas as pd
import os
import glob

def process_csv_file(file_path):
    """
    Process a single CSV file by calculating the average of all columns except 'Frame'
    Returns a dataframe with Frame and AverageTremor columns
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Create a new dataframe with just the Frame column
    result_df = pd.DataFrame()
    result_df['Frame'] = df['Frame']
    
    # Calculate the average for each row, excluding the Frame column
    columns_to_average = [col for col in df.columns if col != 'Frame']
    result_df['AverageTremor'] = df[columns_to_average].mean(axis=1)
    
    return result_df

def process_folder(input_folder, output_folder):
    """
    Process all CSV files in a folder and save results in a separate output folder
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Get a list of all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files in {input_folder}")
    
    # Process each CSV file
    for file_path in csv_files:
        # Get the filename without extension
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Process the file
        result_df = process_csv_file(file_path)
        
        # Create output filename in the output folder
        output_file = os.path.join(output_folder, f"{base_name}_avg_tremor.csv")
        
        # Save the result to a new CSV file
        result_df.to_csv(output_file, index=False)
        
        print(f"Processed {file_path} -> {output_file}")

# Main execution
if __name__ == "__main__":
    # Define input and output folders
    folder1 = r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data"  # Replace with your actual input folder path
    folder2 = r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data"  # Replace with your actual input folder path
    
    output_folder1 = r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\before_data_avg"  # Replace with your desired output folder path
    output_folder2 = r"E:\Tremor\Euclidean Distances of Keypoints for Patient #1\after_data_avg"  # Replace with your desired output folder path
    
    # Process each folder
    if os.path.exists(folder1):
        print(f"\nProcessing folder: {folder1}")
        process_folder(folder1, output_folder1)
    else:
        print(f"Folder not found: {folder1}")
    
    if os.path.exists(folder2):
        print(f"\nProcessing folder: {folder2}")
        process_folder(folder2, output_folder2)
    else:
        print(f"Folder not found: {folder2}")
    
    print("\nAll processing completed!")