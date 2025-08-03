import os
import pandas as pd
from model import run_enhanced_analysis # Import your main function from app.py

def run_full_dataset_validation():
    """
    Runs the RUL prediction backtest on all .mat files in the current directory
    and calculates the average performance.
    """
    # Find all battery .mat files in the current folder
    battery_files = [f for f in os.listdir('.') if f.endswith('.mat')]
    
    if not battery_files:
        print("❌ Error: No .mat files found in this directory.")
        return

    print(f"Found {len(battery_files)} battery datasets to test: {battery_files}")
    
    all_results = []
    
    # Loop through each file and run the analysis
    for file_name in battery_files:
        # The run_enhanced_analysis function will print its own progress
        mae = run_enhanced_analysis(file_name)
        
        # We only collect a result if the backtesting was successful
        if mae is not None:
            all_results.append({'battery': file_name, 'mae': mae})
        else:
            print(f"ℹ️ Skipping {file_name} as it did not have a valid backtesting result (likely never reached EOL).")

    # --- Final Summary ---
    if not all_results:
        print("\nNo batteries had enough data for a complete backtest.")
        return
        
    results_df = pd.DataFrame(all_results)
    average_mae = results_df['mae'].mean()
    
    print("\n\n" + "="*60)
    print("🏆 FULL DATASET VALIDATION SUMMARY 🏆")
    print("="*60)
    print("Individual Battery Performance (MAE in cycles):")
    print(results_df.to_string(index=False))
    print("-" * 60)
    print(f"🎯 Average MAE Across All Testable Batteries: {average_mae:.2f} cycles")
    print("="*60)

# Run the full validation
if __name__ == "__main__":
    run_full_dataset_validation()