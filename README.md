# Energy Forecasting Project

## Overview
The Energy Forecasting Project is a complete data science solution created to study and predict household energy usage patterns for ProgressSoft.  This project uses a full machine learning process that includes EDA, advanced feature engineering, with 3 forecasts models to give accurate energy consumption predictions .

The project analyzes household power consumption data using three different modeling approaches:
- **SARIMA**
- **Facebook Prophet**
- **XGBoost**

## Dataset
This project uses the Individual home Electric Power Consumption dataset, which contains measurements of electric power consumption in one home with a one-minute sampling rate over a time of almost 4 years.  The file includes different electrical numbers and sub-metering values.

| Feature                     | Description                                                                      | Units         |
| --------------------------- | -------------------------------------------------------------------------------- | ------------- |
| **Global\_active\_power**   | Whole-house real (active) power consumed, averaged over one minute               | kW            |
| **Global\_reactive\_power** | Whole-house reactive power (non-working component), averaged over one minute     | kVar          |
| **Voltage**                 | Mains supply voltage measured at the house                                       | V             |
| **Global\_intensity**       | Total electric current drawn by the house, averaged over one minute              | A             |
| **Sub\_metering\_1**        | Energy usage recorded on the kitchen circuit (e.g., dishwasher, microwave)       | Wh per minute |
| **Sub\_metering\_2**        | Energy usage recorded on the laundry-room circuit (e.g., washing machine, dryer) | Wh per minute |
| **Sub\_metering\_3**        | Energy usage recorded on the water-heater / HVAC circuit                         | Wh per minute |



## Project Structure

```
energy_forecasting_project/
├── data/                          # Data storage directory
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── EDA.ipynb                  # Exploratory Data Analysis
│   ├── Comprehensive_Energy_Forecasting_Report.ipynb  # Complete analysis report
│   └── data/                      # Notebook-specific data files
├── src/                           # Source code modules
│   ├── data_preparation.py        # Data loading and preprocessing
│   ├── feature_engineering.py     # Feature creation and scaling
│   ├── modeling.py                # Model training and prediction
│   └── evaluation.py              # Model evaluation and reporting
├── results/                       # Output directory for analysis results
├── run_pipeline.py                # Main pipeline execution script
├── requirements.txt               # Python package dependencies
├── README.md                      # Project documentation
├── Final model comparsion report.pdf  # Model comparison analysis
└── Introduction with EDA.pdf      # EDA summary report
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ali-rashaideh/energy_forecasting_project.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd energy_forecasting_project
   ```

3. **Create and activate the Python virtual environment:**
   
   **On Windows (PowerShell):**
   ```powershell
   python -m venv env
   .\eng\Scripts\Activate.ps1
   ```
   
   **On macOS/Linux:**
   ```bash
   python -m venv eng
   source eng/bin/activate
   ```

4. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Quick Start
To run the complete analysis pipeline:
```bash
python run_pipeline.py
```

This will execute all stages of the pipeline and generate a comprehensive analysis report.

### Individual Components

#### Jupyter Notebooks
The project includes interactive notebooks for detailed analysis:

- **`EDA.ipynb`**: Comprehensive exploratory data analysis including:
   - Data quality review and missing value analysis
   - Statistical reports and spread analysis
   - Time series display and pattern spotting
   - Correlation analysis and feature links

- **`Comprehensive_Energy_Forecasting_Report.ipynb`**: Complete project analysis featuring:
  - End-to-end pipeline demonstration  Model comparison and performance review
  - Uncertainty measurement analysis
  - Business thoughts and suggestions

#### Python Scripts

**Core Pipeline Components:**

- `data_preparation.py`: This script handles the preprocessing of raw data. It performs the following tasks:
  - Downloads and extracts the dataset from a specified URL.
  - Loads the raw data into a pandas DataFrame and preprocesses it by handling missing values and detecting outliers.
  - Aggregates the data into different time frequencies (hourly, daily, weekly).
  - Saves the processed data into CSV files for further analysis.

- `feature_engineering.py`: This script is responsible for creating new features from the existing dataset. It includes:
  - Generating time-based features such as hour, day of the week, and month.
  - Creating lag features to capture temporal dependencies.
  - Adding rolling statistical features like mean and standard deviation.
  - Incorporating holiday indicators and placeholder weather features.
  - Scaling numerical features using StandardScaler for better model performance.

- `modeling.py`: This script focuses on building and training machine learning models for the forecasting. It includes:
  - Splitting the dataset into training and testing sets.
  - Training three models: SARIMA, Facebook Prophet, and XGBoost.
  - Evaluating the models using metrics such as MAE, RMSE, and MAPE.
  - Generating point predictions and uncertainty intervals for each model.

- `evaluation.py`: This script evaluates the performance of the trained models. It performs:
  - Calculation of prediction interval metrics such as coverage and interval width.
  - Visualization of forecasts with uncertainty intervals.
  - Residual analysis using plots like histograms, autocorrelation, and QQ plots.
  - Creation of a comprehensive PDF report summarizing model performance and uncertainty metrics.

- `run_pipeline.py`: This script orchestrates the entire pipeline. It:
  - Executes the data preparation, feature engineering, modeling, and evaluation scripts sequentially.
  - Saves the results, including model performance metrics and predictions.
  - Generates a final report in PDF format summarizing the entire analysis.

### Running Scripts

**Execute the complete pipeline:**
```bash
python run_pipeline.py
```



## Key Features

### Data Processing
- **Automated data download** and extraction from source URLs
- **Comprehensive data cleaning** including missing value imputation and outlier detection
- **Multi-frequency aggregation** (minute, hourly, daily, weekly)
- **Data quality reporting** with detailed statistics

### Feature Engineering
- **Time-based features**: Hour, day of week, month, season
- **Lag features**: Historical values for trend capture
- **Rolling statistics**: Moving averages and standard deviations
- **Holiday indicators**: Integration with holidays library
- **Feature scaling**: StandardScaler normalization for ML models

### Modeling Approaches
- **SARIMA**: Seasonal Auto-Regressive Integrated Moving Average for time series
- **Facebook Prophet**: Robust forecasting with automatic seasonality detection
- **XGBoost**: Gradient boosting for complex pattern recognition

### Evaluation Metrics
- **Point accuracy**: MAE, RMSE, MAPE
- **Uncertainty quantification**: Prediction intervals and coverage analysis
- **Residual analysis**: Autocorrelation and normality tests
- **Visual diagnostics**: Comprehensive plotting and reporting

## Results and Outputs

The analysis generates comprehensive results stored in the `results/` directory:

### Generated Files
- **Model predictions**: CSV files with forecasts and uncertainty intervals
- **Performance metrics**: JSON files with detailed evaluation statistics
- **Visualization outputs**: PNG/PDF plots for model comparison and diagnostics
- **Comprehensive PDF report**: Executive summary with key findings and recommendations

### Key Findings
- Comparative analysis of three forecasting approaches
- Uncertainty quantification for risk assessment
- Seasonal and temporal pattern identification
- Model performance benchmarking and recommendations

## Dependencies

### Core Libraries
The project uses the following essential Python packages:

**Data Processing & Analysis:**
- `pandas==2.3.1` - Data manipulation and analysis
- `numpy==2.3.2` - Numerical computing
- `scipy==1.16.1` - Scientific computing

**Machine Learning:**
- `scikit-learn==1.7.1` - Machine learning algorithms and utilities
- `xgboost` - Gradient boosting framework
- `prophet==1.1.7` - Facebook's forecasting tool
- `statsmodels` - Statistical modeling

**Visualization:**
- `matplotlib==3.10.5` - Plotting library
- `seaborn==0.13.2` - Statistical data visualization
- `missingno==0.5.2` - Missing data visualization

**Time Series & Utilities:**
- `holidays==0.77` - Holiday calendar integration
- `joblib==1.5.1` - Serialization and parallel computing
- `cmdstanpy==1.2.5` - Stan interface for Prophet



**Ali Rashaideh**
- Email: alirashaideh@yahoo.com
- GitHub: [@Ali-rashaideh](https://github.com/Ali-rashaideh)

