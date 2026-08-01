"""
SmartVolt recommendation engine (pure Python, no LLM needed).
Turns a 24-hour price forecast into money-saving advice: the best hours to run
each appliance, when to charge an EV, when to charge/discharge a home battery,
solar self-use guidance, and per-hour grid signals (BUY / SELL / HOLD).
"""
import numpy as np

# Typical appliances: power in kW and how many hours they run.
APPLIANCES = {
    "washing_machine": {"label": "Washing machine", "kw": 1.0, "hours": 2},
    "dishwasher":      {"label": "Dishwasher",      "kw": 1.2, "hours": 2},
    "dryer":           {"label": "Tumble dryer",    "kw": 2.5, "hours": 2},
    "water_heater":    {"label": "Water heater",    "kw": 2.0, "hours": 3},
    "ev":              {"label": "EV charging",     "kw": 7.0, "hours": 6},
    "air_conditioner": {"label": "Air conditioner", "kw": 1.5, "hours": 4},
}


def _window_cost(prices, start, dur, kw):
    """Cost (EUR) of running a kw appliance for `dur` hours starting at `start`.
    prices are EUR/MWh; energy per hour = kw/1000 MWh."""
    seg = prices[start:start + dur]
    return float(np.sum(seg) * (kw / 1000.0))


def _best_window(prices, dur, kw):
    """Return (best_start, best_cost, worst_cost) over all start hours."""
    n = len(prices)
    costs = [(_window_cost(prices, s, dur, kw), s) for s in range(0, n - dur + 1)]
    best_cost, best_start = min(costs)
    worst_cost, _ = max(costs)
    return best_start, best_cost, worst_cost


def _grid_signals(prices, times):
    """Per-hour BUY / SELL / HOLD signal based on where the price sits in the day.
    BUY  = cheap third  -> import from grid / charge battery / run appliances.
    SELL = dear third   -> use stored power / discharge or sell, avoid grid.
    HOLD = middle third.
    """
    lo = float(np.percentile(prices, 33))
    hi = float(np.percentile(prices, 66))
    sig = []
    for i, p in enumerate(prices):
        if p <= lo:
            s = "BUY"
        elif p >= hi:
            s = "SELL"
        else:
            s = "HOLD"
        sig.append({"time": times[i], "hour": i,
                    "price": round(float(p), 2), "signal": s})
    return sig, round(lo, 2), round(hi, 2)


def recommend(forecast, appliances=None, custom=None):
    """
    forecast: dict from ForecastService.forecast (has rows with 'predicted').
    appliances: list of built-in appliance keys the user owns.
    custom: optional list of dicts {label, kw, hours} for user-added devices.
    Returns a structured recommendation dict.
    """
    rows = forecast["rows"]
    prices = np.array([r["predicted"] for r in rows], dtype=float)
    times = [r["time"] for r in rows]
    if appliances is None:
        appliances = ["washing_machine", "dishwasher", "ev", "water_heater"]

    avg = float(prices.mean())
    cheapest_idx = sorted(np.argsort(prices)[:3].tolist())
    dearest_idx = sorted(np.argsort(prices)[-3:].tolist())

    # Build the full appliance list: built-ins + any custom devices.
    specs = []
    for key in appliances:
        a = APPLIANCES.get(key)
        if a:
            specs.append({"key": key, "label": a["label"],
                          "kw": a["kw"], "hours": a["hours"]})
    for c in (custom or []):
        try:
            kw = float(c.get("kw", 1.0))
            hrs = int(float(c.get("hours", 1)))
            label = str(c.get("label", "My device")).strip()[:40] or "My device"
            hrs = max(1, min(hrs, len(times) - 1))
            kw = max(0.05, min(kw, 100.0))
            specs.append({"key": "custom", "label": label, "kw": kw, "hours": hrs})
        except Exception:
            continue

    recs = []
    total_saving = 0.0
    for a in specs:
        start, best_cost, worst_cost = _best_window(prices, a["hours"], a["kw"])
        saving = round(worst_cost - best_cost, 2)
        total_saving += saving
        end = min(start + a["hours"], len(times) - 1)
        recs.append({
            "appliance": a["label"],
            "key": a["key"],
            "best_start": times[start],
            "best_end": times[end],
            "best_cost_eur": round(best_cost, 2),
            "saving_eur": saving,
            "advice": f"Run the {a['label'].lower()} at {times[start]} "
                      f"to save about EUR {saving:.2f} versus the worst time.",
        })

    # Grid signals + battery/solar strategy
    signals, lo_thr, hi_thr = _grid_signals(prices, times)
    charge_h = cheapest_idx
    discharge_h = dearest_idx
    battery_saving = round(float(prices[discharge_h].mean() - prices[charge_h].mean()), 2)

    # Solar owners: solar peaks around midday (10:00-16:00). Guidance depends on
    # whether those midday hours are expensive (use your own solar) or the price
    # is high later (store solar in the battery to use in the evening peak).
    solar_hours = [i for i in range(len(times)) if 10 <= i <= 16]
    solar_avg = round(float(prices[solar_hours].mean()), 2) if solar_hours else avg
    evening_peak = [i for i in dearest_idx if i >= 17]
    solar_note = ("Your solar panels produce most around midday. "
                  + ("Prices are high in the evening, so store the extra solar in your "
                     "battery during the day and use it at "
                     + ", ".join(times[i] for i in evening_peak) + " instead of the grid."
                     if evening_peak else
                     "Use your solar directly during the day and top up the battery for later."))

    return {
        "day": forecast["day"],
        "average_price": round(avg, 2),
        "cheapest_hours": [times[i] for i in cheapest_idx],
        "most_expensive_hours": [times[i] for i in dearest_idx],
        "appliance_recommendations": recs,
        "total_appliance_saving_eur": round(total_saving, 2),
        "grid_signals": signals,
        "signal_thresholds": {"buy_below": lo_thr, "sell_above": hi_thr},
        "battery_advice": {
            "charge_hours": [times[i] for i in charge_h],
            "discharge_hours": [times[i] for i in discharge_h],
            "saving_eur_per_mwh": battery_saving,
        },
        "solar_advice": {
            "midday_avg_price": solar_avg,
            "note": solar_note,
        },
        "ev_advice": next((r for r in recs if r["appliance"] == "EV charging"), None),
    }
