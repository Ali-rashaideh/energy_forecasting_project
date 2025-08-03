import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def calculate_interval_metrics(y_true, interval):
    """
    Calculate prediction interval metrics with index alignment
    
    Args:
        y_true: True values (Series with index)
        interval: DataFrame with prediction intervals
        
    Returns:
        coverage: Proportion of true values within the interval
        width: Average width of the interval
    """
    # Align indices by merging on index
    combined = pd.concat([y_true, interval], axis=1)
    
    # Drop rows with missing values
    combined = combined.dropna()
    
    # Extract aligned values
    y_aligned = combined.iloc[:, 0]
    
    # Handle different column names
    if 'lower' in combined and 'upper' in combined:
        lower = combined['lower']
        upper = combined['upper']
    elif 'yhat_lower' in combined and 'yhat_upper' in combined:
        lower = combined['yhat_lower']
        upper = combined['yhat_upper']
    else:
        # Assume first and second columns are lower and upper bounds
        lower = combined.iloc[:, 1]
        upper = combined.iloc[:, 2]
    
    # Calculate metrics
    coverage = np.mean((y_aligned >= lower) & (y_aligned <= upper))
    width = np.mean(upper - lower)
    
    return coverage, width

def plot_forecasts(y_true, predictions, title, filename):
    """Plot forecasts with uncertainty intervals"""
    plt.figure(figsize=(15, 7))
    plt.plot(y_true.index, y_true, 'k-', label='Actual')
    
    for model, preds in predictions.items():
        plt.plot(y_true.index, preds['point'], label=f'{model} Forecast')
        
        if 'lower' in preds['interval'] and 'upper' in preds['interval']:
            plt.fill_between(
                y_true.index,
                preds['interval']['lower'],
                preds['interval']['upper'],
                alpha=0.2
            )
        elif 'yhat_lower' in preds['interval'] and 'yhat_upper' in preds['interval']:
            plt.fill_between(
                y_true.index,
                preds['interval']['yhat_lower'],
                preds['interval']['yhat_upper'],
                alpha=0.2
            )
        else:
            plt.fill_between(
                y_true.index,
                preds['interval'].iloc[:, 0],
                preds['interval'].iloc[:, 1],
                alpha=0.2
            )
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Global Active Power (kW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_residuals(y_true, y_pred, model_name, filename):
    """Analyze and plot residuals"""
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Residual Analysis - {model_name}')
    
    # Residuals over time
    ax[0, 0].plot(residuals)
    ax[0, 0].set_title('Residuals over Time')
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Residual')
    
    # Histogram
    ax[0, 1].hist(residuals, bins=30)
    ax[0, 1].set_title('Residual Distribution')
    ax[0, 1].set_xlabel('Residual')
    
    # ACF plot
    pd.plotting.autocorrelation_plot(residuals, ax=ax[1, 0])
    ax[1, 0].set_title('Autocorrelation')
    
    # QQ plot
    from scipy import stats
    stats.probplot(residuals, plot=ax[1, 1])
    ax[1, 1].set_title('QQ Plot')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_report(results_df, predictions, y_test, output_path):
    """Create comprehensive PDF report"""
    with PdfPages(output_path) as pdf:
        # Summary metrics
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        plt.table(
            cellText=results_df.values,
            colLabels=results_df.columns,
            cellLoc='center',
            loc='center'
        )
        plt.title('Model Performance Summary')
        pdf.savefig()
        plt.close()
        
        # Forecast plots
        plot_forecasts(
            y_test, 
            predictions, 
            'Model Comparison: Actual vs Forecasted Power Consumption',
            'temp_forecast_plot.png'
        )
        plt.figure(figsize=(15, 7))
        plt.imshow(plt.imread('temp_forecast_plot.png'))
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Residual analysis for each model
        for model in predictions.keys():
            plot_residuals(
                y_test,
                predictions[model]['point'],
                model,
                f'temp_residuals_{model}.png'
            )
            plt.figure(figsize=(15, 7))
            plt.imshow(plt.imread(f'temp_residuals_{model}.png'))
            plt.axis('off')
            pdf.savefig()
            plt.close()
        
        # Uncertainty metrics
        uncertainty_metrics = []
        for model, preds in predictions.items():
            coverage, width = calculate_interval_metrics(
                y_test, 
                preds['interval']  # Pass only the interval dataframe
            )
            uncertainty_metrics.append({
                'Model': model,
                'Coverage': coverage,
                'Interval Width': width
            })
        
        uncertainty_df = pd.DataFrame(uncertainty_metrics)
        plt.figure(figsize=(10, 4))
        plt.axis('off')
        plt.table(
            cellText=uncertainty_df.values,
            colLabels=uncertainty_df.columns,
            cellLoc='center',
            loc='center'
        )
        plt.title('Uncertainty Quantification Metrics')
        pdf.savefig()
        plt.close()

def main():
    results_df = pd.read_csv('../results/model_performance.csv')
    predictions = pd.read_pickle('../results/predictions.pkl')
    y_test = pd.read_csv('../results/y_test.csv', index_col='datetime', parse_dates=True)
    
    create_report(
        results_df,
        predictions,
        y_test['Global_active_power'],
        '../results/Energy_Forecasting_Report.pdf'
    )

if __name__ == "__main__":
    main()