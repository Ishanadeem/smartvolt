"""
Build data/clean_merged.csv from the two raw Spanish CSVs.
Put energy_dataset.csv and weather_features.csv in the data/ folder, then run:
    python build_data.py
"""
import os, pandas as pd, numpy as np
D = os.path.join(os.path.dirname(__file__), "data")

def clean_energy(path):
    df = pd.read_csv(path); df["time"] = pd.to_datetime(df["time"], utc=True)
    dead = [c for c in df.columns if c != "time" and (df[c].isna().all() or
            (pd.api.types.is_numeric_dtype(df[c]) and (df[c].fillna(0) == 0).all()))]
    df = df.drop(columns=dead)
    num = df.select_dtypes("number").columns
    df[num] = df[num].interpolate(method="linear", limit_direction="both")
    return df.sort_values("time").reset_index(drop=True)

def clean_weather(path):
    df = pd.read_csv(path); df["dt_iso"] = pd.to_datetime(df["dt_iso"], utc=True)
    df["city_name"] = df["city_name"].str.strip()
    df = df.drop_duplicates(subset=["dt_iso", "city_name"], keep="first")
    df.loc[(df["pressure"] < 900) | (df["pressure"] > 1100), "pressure"] = np.nan
    df.loc[(df["wind_speed"] < 0) | (df["wind_speed"] > 50), "wind_speed"] = np.nan
    for c in ["temp", "temp_min", "temp_max"]:
        if c in df: df[c] = df[c] - 273.15
    keep = [c for c in ["temp","pressure","humidity","wind_speed","rain_1h","snow_3h","clouds_all"] if c in df]
    agg = df.groupby("dt_iso")[keep].mean().reset_index().rename(columns={"dt_iso":"time"})
    agg[keep] = agg[keep].interpolate(method="linear", limit_direction="both")
    return agg.sort_values("time").reset_index(drop=True)

def add_calendar(df):
    t = df["time"].dt
    df["hour"]=t.hour; df["dayofweek"]=t.dayofweek; df["month"]=t.month
    df["is_weekend"]=(t.dayofweek>=5).astype(int)
    df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24); df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
    return df

if __name__ == "__main__":
    e = clean_energy(os.path.join(D,"energy_dataset.csv"))
    w = clean_weather(os.path.join(D,"weather_features.csv"))
    m = add_calendar(pd.merge(e, w, on="time", how="inner")).sort_values("time").reset_index(drop=True)
    out = os.path.join(D,"clean_merged.csv"); m.to_csv(out, index=False)
    print(f"Saved {len(m)} rows -> {out}")
