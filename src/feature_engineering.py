import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler

def create_time_features(df):
    """Create time-based features from datetime index"""
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    return df

def create_lag_features(df, target_col, lags=[24, 48, 168]):
    """Create lag features for specified time intervals"""
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col, windows=[24, 168]):
    """Create rolling statistical features"""
    df = df.copy()
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = (
            df[target_col].rolling(window=window).mean())
        df[f'{target_col}_rolling_std_{window}'] = (
            df[target_col].rolling(window=window).std())
    return df

def add_holidays(df, country='FR'):
    """Add holiday indicators for specified country"""
    fr_holidays = holidays.France()
    df["is_holiday"] = pd.Index(df.index.date).isin(fr_holidays).astype(int)
    return df

def add_weather_features(df):
    """Add weather features (placeholder for API integration)"""
    # In practice, integrate with weather API/historical data
    df['temperature'] = 15  # Placeholder
    df['humidity'] = 70     # Placeholder
    return df

def scale_features(df, exclude_cols):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    numeric_cols = [col for col in df.columns if col not in exclude_cols]
    scaled_features = scaler.fit_transform(df[numeric_cols])
    df_scaled = pd.DataFrame(scaled_features, columns=numeric_cols, index=df.index)
    return pd.concat([df_scaled, df[exclude_cols]], axis=1)

def prepare_features(df, target_col='Global_active_power'):
    """Full feature engineering pipeline"""
    print("Creating time features...")
    df = create_time_features(df)
    
    print("Creating lag features...")
    df = create_lag_features(df, target_col)
    
    print("Creating rolling features...")
    df = create_rolling_features(df, target_col)
    
    print("Adding holidays...")
    df = add_holidays(df)
    
    print("Adding weather features...")
    df = add_weather_features(df)
    
    print("Handling missing values from feature creation...")
    df = df.dropna()
    
    print("Scaling features...")
    df = scale_features(df, exclude_cols=[target_col, 'is_holiday', 'is_weekend'])
    
    return df

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('../data/processed/hourly_data.csv', index_col='datetime', parse_dates=True)
    df_featured = prepare_features(df)
    df_featured.to_csv('../data/processed/hourly_data_featured.csv')
