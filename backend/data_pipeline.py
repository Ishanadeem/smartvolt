"""
Data cleaning and merging pipeline. Turns the two raw Spanish CSVs
(energy_dataset.csv + weather_features.csv) into one clean hourly file
(clean_merged.csv). Used by the Data page's Upload & Clean button and by
build_data.py.
"""
import os
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "clean_merged.csv")

# Columns the uploaded files MUST contain (Spanish-market schema) so the trained
# N-BEATSx model has every input it needs. Checked up front so a bad file gives a
# clear "which column is missing" message instead of a crash or a wrong result.
REQUIRED_ENERGY = [
    "time", "price actual", "total load forecast", "forecast solar day ahead",
    "forecast wind onshore day ahead", "total load actual", "generation solar",
    "generation wind onshore", "generation nuclear", "generation fossil gas",
    "generation fossil hard coal",
]
REQUIRED_WEATHER = ["dt_iso", "temp", "pressure", "humidity", "wind_speed", "clouds_all"]


def _check_columns(energy_path, weather_path):
    e_cols = pd.read_csv(energy_path, nrows=0).columns
    w_cols = pd.read_csv(weather_path, nrows=0).columns
    miss = [f"energy: {c}" for c in REQUIRED_ENERGY if c not in e_cols]
    miss += [f"weather: {c}" for c in REQUIRED_WEATHER if c not in w_cols]
    if miss:
        raise ValueError("Missing required columns → " + "; ".join(miss)
                         + ". Use the same column names as the Spanish-market "
                           "energy_dataset.csv and weather_features.csv files.")


def clean_energy(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    dead = [c for c in df.columns if c != "time" and (df[c].isna().all() or
            (pd.api.types.is_numeric_dtype(df[c]) and (df[c].fillna(0) == 0).all()))]
    df = df.drop(columns=dead)
    num = df.select_dtypes("number").columns
    df[num] = df[num].interpolate(method="linear", limit_direction="both")
    return df.sort_values("time").reset_index(drop=True)


def clean_weather(path):
    df = pd.read_csv(path)
    df["dt_iso"] = pd.to_datetime(df["dt_iso"], utc=True)
    df["city_name"] = df["city_name"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["dt_iso", "city_name"], keep="first")
    if "pressure" in df:
        df.loc[(df["pressure"] < 900) | (df["pressure"] > 1100), "pressure"] = np.nan
    if "wind_speed" in df:
        df.loc[(df["wind_speed"] < 0) | (df["wind_speed"] > 50), "wind_speed"] = np.nan
    for c in ["temp", "temp_min", "temp_max"]:
        if c in df:
            df[c] = df[c] - 273.15
    keep = [c for c in ["temp", "pressure", "humidity", "wind_speed",
                        "rain_1h", "snow_3h", "clouds_all"] if c in df]
    agg = df.groupby("dt_iso")[keep].mean().reset_index().rename(columns={"dt_iso": "time"})
    agg[keep] = agg[keep].interpolate(method="linear", limit_direction="both")
    return agg.sort_values("time").reset_index(drop=True)


def add_calendar(df):
    t = df["time"].dt
    df["hour"] = t.hour
    df["dayofweek"] = t.dayofweek
    df["month"] = t.month
    df["is_weekend"] = (t.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def build_merged(energy_path, weather_path, out=OUT):
    _check_columns(energy_path, weather_path)
    energy = clean_energy(energy_path)
    weather = clean_weather(weather_path)
    merged = add_calendar(pd.merge(energy, weather, on="time", how="inner"))
    merged = merged.sort_values("time").reset_index(drop=True)
    if len(merged) == 0:
        raise ValueError("No overlapping hours between the two files. "
                         "Check that both files cover the same dates.")
    merged.to_csv(out, index=False)
    return merged
