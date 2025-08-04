import os
import joblib
import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_time_features(df):
    idx = df.index
    df['hour'] = idx.hour.astype('int8')
    df['day_of_week'] = idx.dayofweek.astype('int8')
    df['day_of_month'] = idx.day.astype('int8')
    df['week_of_year'] = idx.isocalendar().week.astype('int16')
    df['month'] = idx.month.astype('int8')
    df['quarter'] = idx.quarter.astype('int8')
    df['year'] = idx.year.astype('int16')
    df['is_weekend'] = (idx.dayofweek >= 5).astype('int8')
    return df

def create_lag_features(df, cols, lags):
    for col in cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def create_rolling_features(df, cols, windows):
    for col in cols:
        for window in windows:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
    return df

def add_holidays(df, country_code):
    df.index = pd.to_datetime(df.index)
    years = df.index.year.unique().tolist()
    hol = holidays.country_holidays(country_code, years=years)
    holiday_dates = set(hol.keys())
    weekend_days = [4, 5] if country_code == 'JO' else [5, 6]
    df['is_holiday_or_weekend'] = (
        np.isin(df.index.date, list(holiday_dates)) | 
        df.index.dayofweek.isin(weekend_days)
    ).astype('int8')
    
    return df

def cap_outliers(df, cols, q_low=0.001, q_high=0.999):
    for col in cols:
        low, high = df[col].quantile([q_low, q_high])
        df[col] = df[col].clip(low, high)
    return df

def fill_small_gaps(df, col, max_gap_hours=48):
    df[col] = (df[col]
               .fillna(method='ffill', limit=max_gap_hours)
               .fillna(method='bfill', limit=max_gap_hours))
    return df

def scale_features(df, exclude, train_idx):
    scaler = StandardScaler().fit(df.loc[train_idx, [c for c in df.columns if c not in exclude]])
    num_cols = [c for c in df.columns if c not in exclude]
    df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler

def prepare_features(df, target='Global_active_power', country_code='JO'):
    df = fill_small_gaps(df, target)
    df = create_time_features(df)
    df = create_lag_features(df, [target, 'Sub_metering_3', 'Voltage'], [24, 48, 168])
    df = create_rolling_features(df, [target, 'Voltage'], [24, 168])
    df = add_holidays(df, country_code)
    df = cap_outliers(df, [target])
    df.dropna(inplace=True)
    train_idx, _ = train_test_split(df.index, test_size=0.2, shuffle=False)
    df, scaler = scale_features(df, [target, 'is_holiday_or_weekend'], train_idx)
    return df, scaler

def run_pipeline(input_csv, output_csv, scaler_path):
    df = pd.read_csv(input_csv, index_col='datetime', parse_dates=True)
    featured_df, scaler = prepare_features(df)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    featured_df.to_csv(output_csv)
    joblib.dump(scaler, scaler_path)

if __name__ == '__main__':
    run_pipeline(
        input_csv='./data/processed/hourly_data.csv',
        output_csv='./data/processed/hourly_featured.csv',
        scaler_path='./data/processed/feature_scaler.joblib'
    )
