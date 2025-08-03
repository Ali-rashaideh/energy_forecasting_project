import pandas as pd
import numpy as np
import os
import zipfile
import requests
from tqdm import tqdm
import shutil

def download_and_extract_data(url, extract_dir):
    """Download and extract dataset from URL"""
    os.makedirs(extract_dir, exist_ok=True)
    zip_path = os.path.join(extract_dir, 'household_power_consumption.zip')
    
    # Download the file with progress bar
    print(f"Downloading dataset from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))
    
    # Extract the zip file
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Find the extracted CSV file
    csv_path = os.path.join(extract_dir, 'household_power_consumption.txt')
    if not os.path.exists(csv_path):
        # Try alternative name
        csv_path = os.path.join(extract_dir, 'household_power_consumption.csv')
    
    # Rename to consistent filename
    target_path = os.path.join(extract_dir, 'household_power_consumption.csv')
    if os.path.exists(csv_path) and csv_path != target_path:
        shutil.move(csv_path, target_path)
    
    return target_path

def load_data(file_path):
    """Load and preprocess raw data"""
    print("Loading data...")
    df = pd.read_csv(
        file_path,
        sep=';',
        parse_dates={'datetime': ['Date', 'Time']},
        infer_datetime_format=True,
        low_memory=False,
        na_values=['?', 'nan'],
        dtype={
            'Global_active_power': 'float32',
            'Global_reactive_power': 'float32',
            'Voltage': 'float32',
            'Global_intensity': 'float32',
            'Sub_metering_1': 'float32',
            'Sub_metering_2': 'float32',
            'Sub_metering_3': 'float32'
        }
    )
    df.set_index('datetime', inplace=True)
    return df

def handle_missing_values(df):
    """Handle missing values using multiple strategies"""
    print("Handling missing values...")
    # Forward fill for short gaps (<3 hours)
    df_ffill = df.ffill(limit=3)
    
    # Linear interpolation for medium gaps
    df_interp = df_ffill.interpolate(method='linear', limit_direction='both')
    
    # Flag missing values
    df_interp['missing'] = df.isnull().any(axis=1).astype(int)
    return df_interp

def detect_outliers(df, window=24, threshold=3):
    """Detect and handle outliers using rolling statistics"""
    print("Detecting outliers...")
    df = df.copy()
    numeric_cols = [col for col in df.columns if df[col].dtype in ['float32', 'float64']]
    
    for col in tqdm(numeric_cols, desc="Processing columns"):
        # Calculate rolling statistics
        rolling_median = df[col].rolling(window=window, center=True, min_periods=1).median()
        rolling_std = df[col].rolling(window=window, center=True, min_periods=1).std().replace(0, 1)
        
        # Identify outliers
        outlier_mask = np.abs(df[col] - rolling_median) > (threshold * rolling_std)
        
        # Replace outliers with rolling median
        df.loc[outlier_mask, col] = rolling_median[outlier_mask]
        
    return df

def aggregate_data(df):
    """Create aggregated datasets at different time frequencies"""
    print("Aggregating data...")
    aggregations = {
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'missing': 'max'
    }
    
    hourly = df.resample('H').agg(aggregations)
    daily = df.resample('D').agg(aggregations)
    weekly = df.resample('W').agg(aggregations)
    
    return {
        'hourly': hourly,
        'daily': daily,
        'weekly': weekly
    }

def save_processed_data(data_dict, output_dir):
    """Save processed data to CSV files"""
    print("Saving processed data...")
    os.makedirs(output_dir, exist_ok=True)
    for freq, df in data_dict.items():
        path = os.path.join(output_dir, f'{freq}_data.csv')
        df.to_csv(path)
        print(f"Saved {freq} data to {path}")

def main():
    """Full data preparation pipeline"""
    # Configuration
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    raw_dir = './data/raw'
    processed_dir = './data/processed'
    
    # Download and extract data
    raw_file_path = os.path.join(raw_dir, 'household_power_consumption.csv')
    if not os.path.exists(raw_file_path):
        raw_file_path = download_and_extract_data(data_url, raw_dir)
    
    # Run processing pipeline
    df = load_data(raw_file_path)
    df_clean = handle_missing_values(df)
    df_outliers_removed = detect_outliers(df_clean)
    aggregated_data = aggregate_data(df_outliers_removed)
    save_processed_data(aggregated_data, processed_dir)
    
    print("Data processing complete!")
    return aggregated_data

if __name__ == "__main__":
    main()