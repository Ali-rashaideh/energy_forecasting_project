import os
import pandas as pd
import pickle
from src.data_preparation import main as prepare_data
from src.feature_engineering import prepare_features
from src.modeling import model_pipeline, train_test_split
from src.evaluation import create_report

# Create directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('results', exist_ok=True)

# MODIFIED: Run data preparation without arguments
print("=== DATA PREPARATION ===")
prepare_data()  # Now handles download and processing internally

# Run feature engineering
print("\n=== FEATURE ENGINEERING ===")
# MODIFIED: Use the hourly data from the new preparation code
df = pd.read_csv('data/processed/hourly_data.csv', index_col='datetime', parse_dates=True)
df_featured = prepare_features(df)
df_featured.to_csv('data/processed/hourly_data_featured.csv')

# MODIFIED: Add memory optimization by clearing unused objects
del df, df_featured

# Run modeling
print("\n=== MODELING ===")
# MODIFIED: Reload featured data to ensure clean state
df_featured = pd.read_csv('data/processed/hourly_data_featured.csv', 
                          index_col='datetime', parse_dates=True)
results_df, predictions = model_pipeline(df_featured)

# Save results
results_df.to_csv('results/model_performance.csv')
with open('results/predictions.pkl', 'wb') as f:
    pickle.dump(predictions, f)
    
# Get y_test for evaluation
# MODIFIED: Pass the reloaded dataframe
train, test = train_test_split(df_featured)
test['Global_active_power'].to_csv('results/y_test.csv')

# Create final report
print("\n=== EVALUATION & REPORTING ===")
create_report(
    results_df,
    predictions,
    test['Global_active_power'],
    'results/Energy_Forecasting_Report.pdf'
)

print("\n=== PIPELINE COMPLETE ===")
print("Results saved in 'results' directory")