"""
Forecast service.

Loads the trained N-BEATSx model once and exposes the project's AI functionalities:
  1. N-BEATSx deep-learning day-ahead forecasting (the point forecast).
  2. Monte Carlo Dropout uncertainty — dropout is kept ON at inference and the
     model is run MC_SAMPLES times; the spread of those runs (plus a calibrated
     noise term) gives the risk band. This is real Bayesian-approximation code,
     not a fixed number.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import logging
import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast

from .constants import (FUTR_EXOG, HIST_EXOG, INPUT_SIZE, HORIZON,
                        MC_SAMPLES, TAU_INV)

# quieten the per-call prediction logs
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, "nbeatsx_model")
# Built-in Spanish demo dataset (never overwritten) and the user-uploaded set.
BUILTIN_FILE = os.path.join(HERE, "data", "clean_merged.csv")
USER_FILE = os.path.join(HERE, "data", "user_merged.csv")
DATA_FILE = BUILTIN_FILE   # kept for backward compatibility


def _keep_dropout_on(module, _inp):
    """forward-pre-hook: force a Dropout layer to stay in train mode at inference."""
    module.train()


def _add_calendar(df):
    t = df["ds"].dt
    df["hour"] = t.hour
    df["dayofweek"] = t.dayofweek
    df["month"] = t.month
    df["is_weekend"] = (t.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


import threading


class ForecastService:
    def __init__(self):
        # One shared model instance is not safe to run from two requests at
        # once (PyTorch Lightning's predict loop is not reentrant), and FastAPI
        # serves sync endpoints from a thread pool. This lock serialises
        # predictions so concurrent calls queue instead of crashing (was a 500).
        self._predict_lock = threading.Lock()
        self.nf = NeuralForecast.load(path=MODEL_DIR)
        self.model = self.nf.models[0]
        # keep every dropout layer active during inference -> enables MC Dropout
        self.n_dropout = 0
        for mod in self.model.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.register_forward_pre_hook(_keep_dropout_on)
                self.n_dropout += 1
        self.active_file = BUILTIN_FILE      # "builtin" | "user" source
        self.reload_data()

    @property
    def source(self):
        return "user" if self.active_file == USER_FILE else "builtin"

    def use_builtin(self):
        """Switch back to the built-in Spanish demo dataset."""
        self.active_file = BUILTIN_FILE
        return self.reload_data()

    def use_user(self):
        """Switch to the last uploaded-and-cleaned user dataset."""
        if not os.path.exists(USER_FILE):
            raise ValueError("No uploaded data yet — upload your CSV files first.")
        self.active_file = USER_FILE
        return self.reload_data()

    def reload_data(self):
        self.frame = self._load_frame()
        self.min_day = self.frame["ds"].min().normalize()
        self.max_day = self.frame["ds"].max().normalize()
        return self.frame

    def _load_frame(self):
        df = pd.read_csv(self.active_file)
        df["ds"] = pd.to_datetime(df["time"], utc=True)
        df = df.rename(columns={"price actual": "y"})
        df["unique_id"] = "ES"
        df = _add_calendar(df)
        needed = ["unique_id", "ds", "y"] + FUTR_EXOG + HIST_EXOG
        df = df[[c for c in needed if c in df.columns]].sort_values("ds").reset_index(drop=True)
        return df

    def data_status(self):
        return {
            "rows": int(len(self.frame)),
            "source": self.source,
            "start": str(self.frame["ds"].min()),
            "end": str(self.frame["ds"].max()),
            "min_forecast_day": str(self.min_day.date()),
            "max_forecast_day": str(self.max_day.date()),
            "mc_samples": MC_SAMPLES,
            "dropout_layers": self.n_dropout,
        }

    def _windows(self, day):
        day_start = pd.Timestamp(day, tz="UTC").normalize()
        day_end = day_start + pd.Timedelta(hours=HORIZON)
        hist = self.frame[self.frame["ds"] < day_start].tail(INPUT_SIZE)
        futr = self.frame[(self.frame["ds"] >= day_start) & (self.frame["ds"] < day_end)]
        if len(hist) < INPUT_SIZE:
            raise ValueError(f"Not enough history before {day}. Pick a later day.")
        if len(futr) < HORIZON:
            raise ValueError(f"No future regressors for {day}. Pick a day inside the data range.")
        return day_start, hist, futr

    def _mc_predict(self, hist, futr, samples):
        """Run the model `samples` times with dropout ON. Returns (samples x 24)."""
        hist_cols = ["unique_id", "ds", "y"] + FUTR_EXOG + HIST_EXOG
        futr_cols = ["unique_id", "ds"] + FUTR_EXOG
        h, f = hist[hist_cols], futr[futr_cols]
        runs = []
        # Serialise all model.predict calls (see note in __init__).
        with self._predict_lock:
            for i in range(samples):
                seed = 1000 + i
                self.model.random_seed = seed          # override neuralforecast's fixed seed
                torch.manual_seed(seed)
                runs.append(self.nf.predict(df=h, futr_df=f)["NBEATSx"].to_numpy())
        return np.vstack(runs)

    def forecast(self, day, samples=MC_SAMPLES):
        day_start, hist, futr = self._windows(day)
        A = self._mc_predict(hist, futr, samples)      # (samples, 24)

        mu = A.mean(axis=0)                            # MC mean forecast
        mc_var = A.var(axis=0)                         # epistemic variance (from dropout)
        mc_std = np.sqrt(mc_var)
        sigma = np.sqrt(TAU_INV + mc_var)              # total predictive std
        lower, upper = mu - 2 * sigma, mu + 2 * sigma

        actual = futr["y"].to_numpy() if "y" in futr else [None] * HORIZON
        rows = []
        for h in range(HORIZON):
            ts = day_start + pd.Timedelta(hours=h)
            av = actual[h]
            rows.append({
                "hour": h,
                "time": ts.strftime("%H:%M"),
                "predicted": round(float(mu[h]), 2),
                "actual": (round(float(av), 2) if av is not None and not pd.isna(av) else None),
                "lower": round(float(lower[h]), 2),
                "upper": round(float(upper[h]), 2),
                "band_width": round(float(upper[h] - lower[h]), 2),
                "mc_std": round(float(mc_std[h]), 3),
            })
        return {"day": day, "horizon": HORIZON, "mc_samples": samples, "rows": rows}
