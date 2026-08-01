"""
Decision service: turns the 24-hour forecast into simple actions.
Charge/buy when price is low, discharge/sell when high, hold otherwise.
The three hours the model is LEAST sure about (highest Monte Carlo Dropout
spread) are raised as risk alerts, so uncertainty drives an operational warning.
"""
import numpy as np


def decide(forecast):
    rows = forecast["rows"]
    prices = np.array([r["predicted"] for r in rows], dtype=float)
    stds = np.array([r["mc_std"] for r in rows], dtype=float)
    q_low = np.percentile(prices, 30)
    q_high = np.percentile(prices, 70)

    # the 3 most uncertain hours (highest epistemic std from MC Dropout)
    uncertain_idx = set(np.argsort(stds)[-3:].tolist())

    out, charge_prices, discharge_prices, alerts = [], [], [], []
    for i, r in enumerate(rows):
        p = r["predicted"]
        if p <= q_low:
            battery, trade = "CHARGE", "BUY"
            charge_prices.append(p)
        elif p >= q_high:
            battery, trade = "DISCHARGE", "SELL"
            discharge_prices.append(p)
        else:
            battery, trade = "HOLD", "WAIT"
        wide = i in uncertain_idx
        if wide:
            alerts.append(r["time"])
        out.append({**r, "battery": battery, "trade": trade, "wide_band": wide})

    order = np.argsort(prices)
    cheapest = sorted(order[:2].tolist())
    dearest = sorted(order[-2:].tolist())
    best_charge = f"{rows[cheapest[0]]['time']}-{rows[cheapest[-1]]['time']}"
    best_discharge = f"{rows[dearest[0]]['time']}-{rows[dearest[-1]]['time']}"

    saving = 0.0
    if charge_prices and discharge_prices:
        saving = round(float(np.mean(discharge_prices) - np.mean(charge_prices)), 1)

    return {
        "rows": out,
        "best_charge_window": best_charge,
        "best_discharge_window": best_discharge,
        "estimated_saving": saving,
        "alerts": sorted(alerts),
    }
