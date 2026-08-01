"""Model input definition — must match how the N-BEATSx model was trained."""

# Future-known regressors (available before the forecast day)
FUTR_EXOG = [
    "hour_sin", "hour_cos", "dayofweek", "month", "is_weekend",
    "total load forecast", "forecast solar day ahead",
    "forecast wind onshore day ahead",
]

# Historically-observed regressors (weather + realised generation/load)
HIST_EXOG = [
    "temp", "pressure", "humidity", "wind_speed", "clouds_all",
    "total load actual", "generation solar", "generation wind onshore",
    "generation nuclear", "generation fossil gas", "generation fossil hard coal",
]

INPUT_SIZE = 168   # hours of history the model needs
HORIZON = 24       # hours the model predicts

# ---- Monte Carlo Dropout (AI functionality #2) ----
# Number of stochastic forward passes with dropout kept ON at inference.
MC_SAMPLES = 25
# Aleatoric variance term (EUR/MWh^2). Total predictive variance is
#   sigma^2 = TAU_INV + MC_variance   (epistemic from dropout + data noise),
# following Gal & Ghahramani (2016). TAU_INV is calibrated from the backtest
# residuals so that ~94% of real prices fall inside the +/- 2 sigma band.
# Calibrated on the full 2015-2018 held-out backtest (residual std ~6.0 EUR/MWh).
TAU_INV = 32.63
