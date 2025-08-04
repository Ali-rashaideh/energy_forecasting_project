# Energy Forecasting Project

## Overview
The Energy Forecasting Project is designed to analyze and predict energy consumption  using  data science techniques. This project leverages exploratory data analysis (EDA), feature engineering, machine learning modeling to provide energy forecasts.

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
├── data/                # Raw and processed data files will be generated using data preperation 
├── notebooks/           # IPYNB notebooks for analysis and reporting
├── results/             # Outputs and results of the analysis
├── src/                 # Source code for data preparation, feature engineering, modeling, evaluation
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
├── run_pipeline.py      # Script to execute the full pipeline
├── requirements.txt     # Python dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Ali-rashaideh/energy_forecasting_project.git
   ```

2. Navigate to the project directory:
   ```
   cd energy_forecasting_project
   ```

3. Set up the Python virtual environment:
   ```
   .venv\Scripts\activate.ps1
   ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Notebooks
- `EDA.ipynb`: Perform EDA on the energy dataset.
- `Energy_Forecasting_Report.ipynb`: a detailed report on the project.

### Scripts

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
To run the full pipeline, use the following command:
```
python run_pipeline.py
```

## Results
The results of the analysis and forecasting are stored in the `results/` directory. This includes model outputs, visualizations, evaluation metrics, and a comprehensive PDF report.

## Dependencies
The project requires the following Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- holidays
- cmdstanpy
- statsmodels
- prophet
- xgboost


## Contact
For questions or feedback, alirashaideh@yahoo.com.