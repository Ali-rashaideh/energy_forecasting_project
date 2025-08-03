import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

def train_test_split(df, test_size=0.2):
    """Time-based train-test split"""
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

def train_sarima(train, test, target_col='Global_active_power'):
    """Train SARIMA model"""
    print("Training SARIMA model...")
    model = SARIMAX(
        train[target_col],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24)
    )
    results = model.fit(disp=False)
    
    # Forecast
    forecast = results.get_forecast(steps=len(test))
    y_pred = forecast.predicted_mean
    ci = forecast.conf_int()
    ci.columns = ['lower', 'upper']
    return y_pred, ci

def train_prophet(train, test, target_col='Global_active_power'):
    """Train Facebook Prophet model"""
    print("Training Prophet model...")
    # Prepare DataFrame with required columns
    df_prophet = train[[target_col]].reset_index()
    df_prophet = df_prophet.rename(columns={'datetime': 'ds', target_col: 'y'})
    
    model = Prophet(
        interval_width=0.95,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    model.add_country_holidays(country_name='FR')
    model.fit(df_prophet)
    
    # Create future dataframe
    future = model.make_future_dataframe(
        periods=len(test), 
        freq='H', 
        include_history=False
    )
    forecast = model.predict(future)
    
    y_pred = forecast['yhat'].values
    ci = forecast[['yhat_lower', 'yhat_upper']].rename(
        columns={'yhat_lower': 'lower', 'yhat_upper': 'upper'}
    )
    return y_pred, ci

def train_xgboost(train, test, target_col='Global_active_power'):
    """Train XGBoost model with uncertainty estimation"""
    print("Training XGBoost model...")
def train_xgboost(train, test, target_col='Global_active_power'):
    # Drop rows with missing values in features
    X_train = train.drop(columns=[target_col]).dropna()
    y_train = train.loc[X_train.index][target_col]
    
    X_test = test.drop(columns=[target_col]).dropna()
    y_test = test.loc[X_test.index][target_col]
    
    # Point prediction model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Uncertainty estimation using residuals
    train_pred = model.predict(X_train)
    residuals = y_train - train_pred
    std_residual = residuals.std()
    ci_lower = y_pred - 1.96 * std_residual
    ci_upper = y_pred + 1.96 * std_residual
    ci = pd.DataFrame({
        'lower': ci_lower,
        'upper': ci_upper
    }, index=test.index)
    return y_pred, ci

def model_pipeline(df, target_col='Global_active_power'):
    """Full modeling pipeline"""
    df = df.dropna(subset=[target_col]).copy()
    train, test = train_test_split(df)
    
    # Train models
    sarima_pred, sarima_ci = train_sarima(train, test, target_col)
    prophet_pred, prophet_ci = train_prophet(train, test, target_col)
    xgb_pred, xgb_ci = train_xgboost(train, test, target_col)
    
    # Evaluate models
    y_test = test[target_col]
    results = [
        evaluate_model(y_test, sarima_pred, 'SARIMA'),
        evaluate_model(y_test, prophet_pred, 'Prophet'),
        evaluate_model(y_test, xgb_pred, 'XGBoost')
    ]
    
    # Package predictions
    predictions = {
        'SARIMA': {'point': sarima_pred, 'interval': sarima_ci},
        'Prophet': {'point': prophet_pred, 'interval': prophet_ci},
        'XGBoost': {'point': xgb_pred, 'interval': xgb_ci}
    }
    
    return pd.DataFrame(results), predictions

if __name__ == "__main__":
    df = pd.read_csv('../data/processed/hourly_data_featured.csv', index_col='datetime', parse_dates=True)
    results_df, predictions = model_pipeline(df)
    results_df.to_csv('../results/model_performance.csv')