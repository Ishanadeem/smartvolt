"""
Populate the SQLite database and export a .sql dump for submission.
Loads the cleaned data into price_history, seeds the metrics, runs a handful of
day-ahead forecasts (filling forecast + decision), then writes
data/forecasting_system_dump.sql.

Run:  python build_database.py
"""
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from backend.model_service import ForecastService
from backend.decision_service import decide
from backend.database import Database

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "data", "forecasting_system_dump.sql")


def main():
    fs = ForecastService()
    db = Database()
    db.populate_history(fs.frame)
    print(f"price_history populated with {len(fs.frame)} rows")

    # run forecasts for the last 5 whole days that have future regressors
    last_day = fs.frame["ds"].max().normalize() - pd.Timedelta(days=1)
    days = [(last_day - pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    done = 0
    for day in days:
        try:
            fc = fs.forecast(day)
            dec = decide(fc)
            db.save_forecast(day, dec["rows"], dec["estimated_saving"])
            done += 1
        except Exception as e:
            print(f"  skip {day}: {e}")
    print(f"stored {done} day-ahead forecasts (forecast + decision tables)")

    db.dump_sql(DUMP)
    size = os.path.getsize(DUMP)
    print(f"SQL dump written -> {DUMP}  ({size} bytes)")


if __name__ == "__main__":
    main()
