"""
SQLite persistence. Schema matches the ERD in the documentation:
  PRICE_HISTORY, FORECAST, DECISION, METRIC.
"""
import os
import json
import hashlib
import secrets
import sqlite3

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(HERE, "data", "forecasting_system.db")


def _hash(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 100000).hex()

# Real test-set metrics (30% held-out backtest on the full 2015-2018 data).
DEFAULT_METRICS = [
    ("Naive persistence", 5.43, 7.98, 10.30, 0.5841),
    ("ARX (linear)", 4.70, 6.39, 8.90, 0.7328),
    ("LSTM", 4.44, 6.57, 8.72, 0.7176),
    ("N-BEATSx (ours)", 4.31, 6.04, 8.02, 0.7613),
]


class Database:
    def __init__(self, path=DB):
        self.path = path
        self._init()
        self._seed_metrics()

    def _conn(self):
        return sqlite3.connect(self.path)

    def _init(self):
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT UNIQUE, price_actual REAL,
                total_load REAL, temperature REAL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS forecast (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT, hour INTEGER, timestamp TEXT,
                predicted_price REAL, actual_price REAL,
                lower_bound REAL, upper_bound REAL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS decision (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_id INTEGER, action TEXT, trade_signal TEXT,
                estimated_saving REAL,
                FOREIGN KEY(forecast_id) REFERENCES forecast(id))""")
            c.execute("""CREATE TABLE IF NOT EXISTS metric (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT UNIQUE, mae REAL, rmse REAL, mape REAL, r2 REAL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE, salt TEXT, pw_hash TEXT,
                appliances TEXT DEFAULT '[]', prefs TEXT DEFAULT '{}')""")
            c.commit()

    # ---- user accounts (login) ----
    def create_user(self, username, password):
        salt = secrets.token_hex(8)
        try:
            with self._conn() as c:
                c.execute("INSERT INTO users(username,salt,pw_hash,appliances) VALUES (?,?,?,?)",
                          (username.strip().lower(), salt, _hash(password, salt),
                           json.dumps(["washing_machine", "dishwasher", "ev", "water_heater"])))
                c.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # username already taken

    def check_user(self, username, password):
        with self._conn() as c:
            row = c.execute("SELECT salt,pw_hash FROM users WHERE username=?",
                            (username.strip().lower(),)).fetchone()
        return bool(row) and _hash(password, row[0]) == row[1]

    def get_appliances(self, username):
        with self._conn() as c:
            row = c.execute("SELECT appliances FROM users WHERE username=?",
                            (username.strip().lower(),)).fetchone()
        try:
            return json.loads(row[0]) if row else []
        except Exception:
            return []

    def set_appliances(self, username, appliances):
        with self._conn() as c:
            c.execute("UPDATE users SET appliances=? WHERE username=?",
                      (json.dumps(appliances), username.strip().lower()))
            c.commit()

    def get_prefs(self, username):
        with self._conn() as c:
            row = c.execute("SELECT prefs FROM users WHERE username=?",
                            (username.strip().lower(),)).fetchone()
        try:
            return json.loads(row[0]) if row and row[0] else {}
        except Exception:
            return {}

    def set_prefs(self, username, prefs):
        with self._conn() as c:
            c.execute("UPDATE users SET prefs=? WHERE username=?",
                      (json.dumps(prefs), username.strip().lower()))
            c.commit()

    def change_password(self, username, old_pw, new_pw):
        if not self.check_user(username, old_pw):
            return False
        salt = secrets.token_hex(8)
        with self._conn() as c:
            c.execute("UPDATE users SET salt=?, pw_hash=? WHERE username=?",
                      (salt, _hash(new_pw, salt), username.strip().lower()))
            c.commit()
        return True

    def delete_user(self, username):
        with self._conn() as c:
            c.execute("DELETE FROM users WHERE username=?", (username.strip().lower(),))
            c.commit()

    def _seed_metrics(self):
        with self._conn() as c:
            if c.execute("SELECT COUNT(*) FROM metric").fetchone()[0] == 0:
                c.executemany(
                    "INSERT INTO metric(model_name,mae,rmse,mape,r2) VALUES (?,?,?,?,?)",
                    DEFAULT_METRICS)
                c.commit()

    def get_metrics(self):
        with self._conn() as c:
            rows = c.execute(
                "SELECT model_name,mae,rmse,mape,r2 FROM metric ORDER BY id").fetchall()
        return [dict(model=r[0], mae=r[1], rmse=r[2], mape=r[3], r2=r[4]) for r in rows]

    def populate_history(self, frame):
        """frame: the cleaned dataset (ds, y, total load actual, temp)."""
        rows = []
        for _, r in frame.iterrows():
            rows.append((str(r["ds"]), float(r["y"]),
                         float(r.get("total load actual", 0) or 0),
                         float(r.get("temp", 0) or 0)))
        with self._conn() as c:
            c.execute("DELETE FROM price_history")
            c.executemany("""INSERT OR IGNORE INTO price_history
                (timestamp,price_actual,total_load,temperature) VALUES (?,?,?,?)""", rows)
            c.commit()

    def save_forecast(self, run_date, rows, saving):
        """rows: list of dicts with hour,time,predicted,actual,lower,upper,battery,trade."""
        with self._conn() as c:
            c.execute("DELETE FROM decision WHERE forecast_id IN "
                      "(SELECT id FROM forecast WHERE run_date=?)", (run_date,))
            c.execute("DELETE FROM forecast WHERE run_date=?", (run_date,))
            for r in rows:
                cur = c.execute("""INSERT INTO forecast
                    (run_date,hour,timestamp,predicted_price,actual_price,lower_bound,upper_bound)
                    VALUES (?,?,?,?,?,?,?)""",
                    (run_date, r["hour"], r["time"], r["predicted"],
                     r.get("actual"), r["lower"], r["upper"]))
                fid = cur.lastrowid
                c.execute("""INSERT INTO decision
                    (forecast_id,action,trade_signal,estimated_saving) VALUES (?,?,?,?)""",
                    (fid, r.get("battery", "HOLD"), r.get("trade", "WAIT"), saving))
            c.commit()

    def get_history(self, limit=15):
        with self._conn() as c:
            rows = c.execute("""
                SELECT f.run_date,
                       ROUND(AVG(f.predicted_price),1) AS avg_price,
                       ROUND(AVG(ABS(f.actual_price-f.predicted_price)/f.actual_price)*100,1) AS mape,
                       MAX(d.estimated_saving) AS saving
                FROM forecast f LEFT JOIN decision d ON d.forecast_id=f.id
                WHERE f.actual_price IS NOT NULL
                GROUP BY f.run_date ORDER BY f.run_date DESC LIMIT ?""", (limit,)).fetchall()
        return [dict(day=r[0], avg_price=r[1], mape=r[2],
                     saving=(r[3] or 0),   # numeric household euro saving the savings page sums & charts
                     actions=(f"est. saving EUR {r[3]}" if r[3] else "")) for r in rows]

    def dump_sql(self, out_path):
        with self._conn() as c:
            with open(out_path, "w", encoding="utf-8") as f:
                for line in c.iterdump():
                    f.write(line + "\n")
