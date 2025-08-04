import os, pickle, pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

def train_test_split(df, test_size=0.2):            
    idx = int(len(df) * (1 - test_size))
    return df.iloc[:idx], df.iloc[idx:]

def _metrics(y, y_hat):
    mae = mean_absolute_error(y, y_hat)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mape = np.mean(np.abs((y - y_hat) / y)) * 100
    return mae, rmse, mape

def _sarima(train, test, col):
    mod = SARIMAX(train[col], order=(1,1,1), seasonal_order=(1,1,1,24)).fit(disp=False)
    fc = mod.get_forecast(len(test))
    ci = fc.conf_int()
    ci_df = pd.DataFrame({
        'lower': ci.iloc[:, 0],
        'upper': ci.iloc[:, 1]
    }, index=ci.index)
    return fc.predicted_mean, ci_df

def _prophet(train, test, col):
    dfp = train[[col]].reset_index().rename(columns={'datetime':'ds', col:'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    m.add_country_holidays('FR')
    m.fit(dfp)
    fut = m.make_future_dataframe(len(test), freq='H', include_history=False)
    fc = m.predict(fut)
    y_hat = fc['yhat'].values
    ci = fc[['yhat_lower','yhat_upper']].rename(columns={'yhat_lower':'lower','yhat_upper':'upper'}).set_index(test.index)
    return y_hat, ci

def _xgb(train, test, col):
    X_tr = train.drop(columns=[col]).dropna()
    y_tr = train.loc[X_tr.index, col]
    X_te = test.drop(columns=[col]).dropna()
    y_te = test.loc[X_te.index, col]
    model = XGBRegressor(objective='reg:squarederror', n_estimators=200)
    model.fit(X_tr, y_tr)
    y_hat = pd.Series(model.predict(X_te), index=X_te.index)
    resid = y_tr - model.predict(X_tr)
    s = resid.std()
    ci = pd.DataFrame({'lower': y_hat - 1.96 * s, 'upper': y_hat + 1.96 * s})
    return y_hat, ci.reindex(test.index)

def model_pipeline(df, target_col='Global_active_power'):  
    """
    Takes a *featured* dataframe, trains SARIMA-Prophet-XGB, and
    returns:
        1. metrics_df : DataFrame with MAE/RMSE/MAPE
        2. preds_dict : {model: {'point': Series, 'interval': DataFrame}}
    """
    df = df.dropna(subset=[target_col]).copy()
    train, test = train_test_split(df)
    y_test = test[target_col]

    sar_y, sar_ci = _sarima(train, test, target_col)
    pro_y, pro_ci = _prophet(train, test, target_col)
    xgb_y, xgb_ci = _xgb(train, test, target_col)

    res = [
        ('SARIMA',)  + _metrics(y_test, sar_y),
        ('Prophet',) + _metrics(y_test, pro_y),
        ('XGBoost',) + _metrics(y_test, xgb_y)
    ]
    metrics_df = pd.DataFrame(res, columns=['Model', 'MAE', 'RMSE', 'MAPE'])

    preds = {
        'SARIMA':  {'point': sar_y, 'interval': sar_ci},
        'Prophet': {'point': pro_y, 'interval': pro_ci},
        'XGBoost': {'point': xgb_y, 'interval': xgb_ci}
    }

    return metrics_df, preds

def main():                                           # unchanged
    os.makedirs('./results', exist_ok=True)
    df = pd.read_csv('./data/processed/hourly_featured.csv',
                     index_col='datetime', parse_dates=True)

    metrics_df, preds = model_pipeline(df)
    metrics_df.to_csv('./results/model_performance.csv', index=False)
    with open('./results/predictions.pkl', 'wb') as f:
        pickle.dump(preds, f)
    _, test = train_test_split(df)
    test['Global_active_power'].to_csv('./results/y_test.csv')

if __name__ == '__main__':
    main()