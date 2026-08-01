"""
Lightweight forecast service for the free cloud deployment.

Serves PRE-COMPUTED N-BEATSx forecasts (computed offline by the real model) so
the hosted version needs no PyTorch / neural-forecast and fits a 512 MB free
tier. The genuine deep-learning model still lives in model_service.py and runs
on localhost; this just replays its saved outputs.
"""
import os
import json
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRECOMP = os.path.join(HERE, "data", "precomputed_forecasts.json")
HORIZON = 24


class ForecastService:
    def __init__(self):
        with open(PRECOMP, encoding="utf-8") as f:
            self._fc = json.load(f)
        self.reload_data()

    def reload_data(self):
        days = sorted(self._fc.keys())
        self.min_day = pd.Timestamp(days[0]) if days else None
        self.max_day = pd.Timestamp(days[-1]) if days else None
        self.frame = self._build_frame(days)
        return self.frame

    def _build_frame(self, days):
        """Reconstruct a price history (ds, y) from the pre-computed actuals so
        the price_history table can still be populated. No CSV needed."""
        rows = []
        for d in days:
            for r in self._fc[d]:
                if r.get("actual") is not None:
                    rows.append((pd.Timestamp(d + " " + r["time"], tz="UTC"), r["actual"]))
        if not rows:
            return pd.DataFrame(columns=["ds", "y"])
        df = pd.DataFrame(rows, columns=["ds", "y"]).sort_values("ds").reset_index(drop=True)
        return df

    def data_status(self):
        return {
            "rows": int(len(self.frame)),
            "start": str(self.frame["ds"].min()) if len(self.frame) else None,
            "end": str(self.frame["ds"].max()) if len(self.frame) else None,
            "min_forecast_day": str(self.min_day.date()) if self.min_day is not None else None,
            "max_forecast_day": str(self.max_day.date()) if self.max_day is not None else None,
            "mc_samples": 25,
            "dropout_layers": 9,
            "mode": "cloud",   # pre-computed forecasts (desktop version runs the live model)
        }

    def forecast(self, day, samples=25):
        rows = self._fc.get(str(day))
        if not rows:
            raise ValueError(f"No forecast available for {day}. Pick a day inside the data range.")
        return {"day": day, "horizon": HORIZON, "mc_samples": samples, "rows": rows}
