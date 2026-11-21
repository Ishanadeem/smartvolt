"""
Greenest Hours service — computes REAL low-carbon / clean-energy information for a
given day directly from the generation columns in the cleaned Spanish market data
(generation solar, wind, nuclear, fossil gas, fossil hard coal). Nothing is
hard-coded: every number here comes from the same dataset the model is trained on.

Carbon intensity is estimated from the generation mix using standard life-cycle
emission factors (gCO2 per kWh):
    solar/wind/nuclear ~ 0 (low-carbon)   gas ~ 490   coal ~ 820
"""
import numpy as np
import pandas as pd

GAS_GCO2 = 490.0
COAL_GCO2 = 820.0

LOW = ["generation solar", "generation wind onshore", "generation nuclear"]
FOSSIL = ["generation fossil gas", "generation fossil hard coal"]
ALL_SRC = LOW + FOSSIL


def _cls(intensity, lo, hi):
    """Classify an hour's carbon intensity into 4 clean-ness bands."""
    if intensity <= lo:
        return "veryclean"
    if intensity <= (lo + hi) / 2:
        return "clean"
    if intensity <= hi:
        return "average"
    return "high"


def compute_green(frame, day):
    """Return a real clean-energy summary for `day` (YYYY-MM-DD) from the dataset."""
    cutoff = pd.Timestamp(day, tz="UTC")
    d = frame[(frame["ds"] >= cutoff) & (frame["ds"] < cutoff + pd.Timedelta(days=1))].copy()
    if len(d) < 24:
        raise ValueError(f"No full day of generation data for {day}.")
    d = d.sort_values("ds").reset_index(drop=True)

    for c in ALL_SRC:
        d[c] = pd.to_numeric(d[c], errors="coerce").clip(lower=0).fillna(0.0)

    d["low"] = d[LOW].sum(axis=1)
    d["gas"] = d["generation fossil gas"]
    d["coal"] = d["generation fossil hard coal"]
    d["total"] = d["low"] + d["gas"] + d["coal"]
    d["total"] = d["total"].replace(0, np.nan)

    d["intensity"] = (d["gas"] * GAS_GCO2 + d["coal"] * COAL_GCO2) / d["total"]
    d["clean_pct"] = d["low"] / d["total"] * 100.0
    d["intensity"] = d["intensity"].fillna(0.0)
    d["clean_pct"] = d["clean_pct"].fillna(0.0)

    inten = d["intensity"].values
    lo, hi = float(np.percentile(inten, 25)), float(np.percentile(inten, 75))

    hours = []
    for i, r in d.iterrows():
        hh = int(pd.Timestamp(r["ds"]).hour)
        hours.append({
            "hour": hh,
            "time": f"{hh:02d}:00",
            "intensity": round(float(r["intensity"]), 0),
            "clean_pct": round(float(r["clean_pct"]), 0),
            "cls": _cls(float(r["intensity"]), lo, hi),
        })

    imin = int(np.argmin(inten))
    imax = int(np.argmax(inten))

    # day-average energy mix (share of each real source)
    tot = float(d[ALL_SRC].values.sum())
    mix = []
    if tot > 0:
        for src, kind in [("generation solar", "solar"), ("generation wind onshore", "wind"),
                          ("generation nuclear", "nuclear"), ("generation fossil gas", "gas"),
                          ("generation fossil hard coal", "coal")]:
            pct = round(float(d[src].values.sum()) / tot * 100.0, 1)
            mix.append({"source": kind, "pct": pct,
                        "low_carbon": kind in ("solar", "wind", "nuclear")})

    clean_share = round(float(d["low"].sum()) / tot * 100.0, 0) if tot > 0 else 0
    # CO2 a 2 kWh appliance avoids by running at the cleanest vs the dirtiest hour
    co2_saved_kg = round(2.0 * (float(inten[imax]) - float(inten[imin])) / 1000.0, 2)

    return {
        "day": str(day),
        "hours": hours,
        "cleanest": {"time": hours[imin]["time"], "intensity": hours[imin]["intensity"]},
        "dirtiest": {"time": hours[imax]["time"], "intensity": hours[imax]["intensity"]},
        "min_intensity": hours[imin]["intensity"],
        "clean_share_pct": clean_share,
        "co2_saved_kg": co2_saved_kg,
        "mix": mix,
    }
