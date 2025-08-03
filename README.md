# Energy Forecasting Project

## Overview
The Energy Forecasting Project is designed to analyze and predict energy consumption patterns using advanced data science techniques. This project leverages exploratory data analysis (EDA), feature engineering, and machine learning modeling to provide accurate energy forecasts.

## Project Structure

```
├── data/                # Raw and processed data files
├── eng/                 # Python virtual environment
├── notebooks/           # Jupyter notebooks for analysis and reporting
├── results/             # Outputs and results of the analysis
├── src/                 # Source code for data preparation, feature engineering, modeling, and evaluation
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
├── run_pipeline.py      # Script to execute the full pipeline
├── README.md            # Project documentation
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
   eng\Scripts\activate
   ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Notebooks
- `EDA.ipynb`: Perform exploratory data analysis on the energy dataset.
- `Energy_Forecasting_Report.ipynb`: Generate a detailed report on energy forecasting.

### Scripts

- `data_preparation.py`: This script handles the preprocessing of raw data. It includes tasks such as downloading, extracting, cleaning, handling missing values, detecting outliers, and aggregating data at different time frequencies.

- `feature_engineering.py`: This script is responsible for creating new features from the existing dataset. It applies techniques like creating time-based features, lag features, rolling statistical features, holiday indicators, and weather placeholders. It also scales the features for modeling.

- `modeling.py`: This script focuses on building and training machine learning models for energy forecasting. The models used include SARIMA, Facebook Prophet, and XGBoost. It trains these models using default parameters without hyperparameter tuning.

- `evaluation.py`: This script evaluates the performance of the trained models. It calculates metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE). It also generates visualizations for forecasts, residuals, and uncertainty metrics.

- `run_pipeline.py`: This script orchestrates the entire pipeline, including data preparation, feature engineering, modeling, and evaluation. It saves the results and generates a comprehensive report.

### Running Scripts
To run the full pipeline, use the following command:
```
python run_pipeline.py
```

To run individual scripts, use:
```
python src/<script_name>.py
```
Replace `<script_name>` with the desired script (e.g., `data_preparation.py`).

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

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or feedback, please contact Ali Rashaideh at [email@example.com].