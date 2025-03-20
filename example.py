import os
import sys
import argparse

# This makes it so that the script can be run from any directory
if __name__ == "__main__":
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict tremor band power values')
    parser.add_argument('--file', type=str, 
                        default=os.path.join(script_dir, 'tremor_band_power_results0.csv'),
                        help='Path to the CSV file containing tremor data')
    parser.add_argument('--days', type=int, default=5,
                        help='Number of days to predict')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("TREMOR BAND POWER PREDICTION TOOL")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' does not exist.")
        sys.exit(1)
    
    # Import the TremorPowerPredictor class
    try:
        from tremor_power_predictor import TremorPowerPredictor
    except ImportError:
        print("Error: Could not import TremorPowerPredictor.")
        print("Make sure the file 'tremor_power_predictor.py' is in the same directory.")
        sys.exit(1)
    
    # Create predictor
    print(f"\nLoading data from {args.file}...")
    predictor = TremorPowerPredictor(args.file)
    
    # Print info about the data
    data = predictor.df
    print(f"\nDataset summary:")
    print(f"  - Number of days: {len(data)}")
    print(f"  - Date range: Day {data['day'].min()} to Day {data['day'].max()}")
    print(f"  - Average pre band power: {data['pre_band_power'].mean():.4f}")
    print(f"  - Average post band power: {data['post_band_power'].mean():.4f}")
    print(f"  - Average reduction percentage: {data['band_power_reduction_percent'].mean():.2f}%")
    
    # Analyze relationships
    print("\nAnalyzing data relationships...")
    relationships = predictor.analyze_relationships()
    
    # Check model performance
    print("\nEvaluating model performance...")
    performance = predictor.model_performance()
    
    # Make predictions
    print(f"\nPredicting tremor band power for the next {args.days} days...")
    predictions = predictor.predict_future(days=args.days)
    
    # Save predictions to CSV
    output_file = os.path.join(os.path.dirname(os.path.abspath(args.file)), "predictions.csv")
    predictions.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY OF FINDINGS")
    print("="*60)
    print(f"1. There are {relationships['pos_reduction_count']} positive vs {relationships['neg_reduction_count']} negative reduction days in the dataset.")
    print(f"2. Higher pre-band power ({relationships['avg_pre_pos']:.4f}) is associated with positive outcomes.")
    print(f"3. Lower pre-band power ({relationships['avg_pre_neg']:.4f}) is associated with negative outcomes.")
    print(f"4. AR model accuracy for pre-band power: MSE = {performance['ar_mse']:.4f}")
    print(f"5. Polynomial model accuracy for post-band power: MSE = {performance['poly_mse']:.4f}")
    
    # Prediction summary
    print("\nPREDICTIONS SUMMARY:")
    for _, row in predictions.iterrows():
        outcome = "improvement" if row['band_power_reduction_percent'] > 0 else "worsening"
        confidence = "high" if abs(row['band_power_reduction_percent']) > 50 else "moderate"
        
        print(f"Day {int(row['day'])}: {row['band_power_reduction_percent']:.2f}% ({outcome}, {confidence} confidence)")
    
    print("\nPlots have been saved to the 'plots' folder in the same directory as your data file.")
    print("="*60)