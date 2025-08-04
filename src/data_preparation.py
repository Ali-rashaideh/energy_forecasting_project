import os
import zipfile
import shutil
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler

#extract data from a zip file and handle the CSV file
def download_and_extract_data(url, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    zip_file = os.path.join(extract_dir, "household_power_consumption.zip")
    r = requests.get(url, stream=True)
    with open(zip_file, "wb") as f, tqdm(total=int(r.headers.get("content-length", 0)), unit="B", unit_scale=True, unit_divisor=1024) as bar:
        for chunk in r.iter_content(1024):
            f.write(chunk)
            bar.update(len(chunk))
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(extract_dir)
    src = next((os.path.join(extract_dir, n) for n in ("household_power_consumption.txt", "household_power_consumption.csv") if os.path.exists(os.path.join(extract_dir, n))), None)
    dst = os.path.join(extract_dir, "household_power_consumption.csv")
    if src and src != dst:
        shutil.move(src, dst)
    return dst

# Load, clean, and prepare the data
def load_data(file_path):
    df = pd.read_csv(
        file_path,
        sep=";",
        parse_dates={"datetime": ["Date", "Time"]},
        low_memory=False,
        na_values=["?", "nan"],
        dtype={
            "Global_active_power": "float32",
            "Global_reactive_power": "float32",
            "Voltage": "float32",
            "Global_intensity": "float32",
            "Sub_metering_1": "float32",
            "Sub_metering_2": "float32",
            "Sub_metering_3": "float32",
        },
    )
    df.set_index("datetime", inplace=True)
    return df

# Handle missing values
def handle_missing_values(df, method="ffill"):
    if method == "ffill":
        return df.ffill()
    if method == "bfill":
        return df.bfill()
    if method == "interpolate":
        return df.interpolate(limit_direction="both")
    return df

# Detect and handle outliers
def detect_outliers(df, threshold=3.0):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lb = q1 - threshold * iqr
        ub = q3 + threshold * iqr
        if any(x in col.lower() for x in ["power", "sub", "intensity"]):
            lb = max(lb, 0.0)
        df[col] = df[col].clip(lb, ub)
    return df

# Aggregate data into different timeframes
def aggregate(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    agg = {
        'Global_active_power':   'sum',
        'Global_reactive_power': 'sum',
        'Voltage':               'mean',
        'Global_intensity':      'mean',
        'Sub_metering_1':        'sum',
        'Sub_metering_2':        'sum',
        'Sub_metering_3':        'sum',
    }
    return {
        'hourly':  df.resample('H').agg(agg),
        'daily':   df.resample('D').agg(agg),
        'weekly':  df.resample('W').agg(agg),
    }

def main(save_csv: bool = True) -> None:
    url      = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    raw_dir  = "./data/raw"
    out_dir  = "./data/processed"

    csv_file = os.path.join(raw_dir, "household_power_consumption.csv")
    if not os.path.exists(csv_file):
        csv_file = download_and_extract_data(url, raw_dir)

    df_raw   = load_data(csv_file)
    df_clean = detect_outliers(handle_missing_values(df_raw))

    agg = aggregate(df_clean)

    if save_csv:
        os.makedirs(out_dir, exist_ok=True)
        agg['hourly'].to_csv(os.path.join(out_dir, "hourly_data.csv"))
        agg['daily'].to_csv(os.path.join(out_dir, "daily_data.csv"))
        agg['weekly'].to_csv(os.path.join(out_dir, "weekly_data.csv"))

    return df_clean, agg


if __name__ == "__main__":
    main(save_csv=True)
