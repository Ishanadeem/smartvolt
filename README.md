# Electricity Price Forecasting using Deep Learning — SmartVolt

Final Year Project, University of Management and Technology, Lahore.

**SmartVolt** forecasts day-ahead electricity prices for the Spanish market with an
**N-BEATSx** deep-learning model, attaches a calibrated risk band to each forecast,
and uses a team of AI agents (built with the **Agno** framework and a **Groq** LLM)
to turn the forecast into plain-language, money-saving advice for a household:
the cheapest hours to run appliances, charge an EV, and use a home battery.

## Features
- Day-ahead price forecast (24 hours) with a Monte-Carlo Dropout risk band
- Money-saving plan: best times to run each appliance, plus add-your-own-device
- Battery / solar advice and hour-by-hour BUY / SELL / HOLD grid signals
- Multi-agent AI assistant (Forecast Analyst + Energy Strategy Advisor)
- Model accuracy vs. naive, ARX and LSTM baselines
- Forecast history, user accounts, settings and dark mode

## Tech stack
FastAPI · PyTorch / NeuralForecast (N-BEATSx) · Agno · Groq (Llama 3.3) · SQLite ·
HTML/CSS/JS

## Running it
Locally (full model): install the requirements and run
`uvicorn backend.main:app --host 127.0.0.1 --port 8000`, then open
http://127.0.0.1:8000

Cloud (free): this repo deploys on Render — it serves pre-computed forecasts, so it
needs no heavy libraries. Set `SMARTVOLT_LITE=1` and (optionally) a `GROQ_API_KEY`
secret to enable the live AI assistant.

## Team
Isha Nadeem (F2022065028) · Izhar Ahmed (F2022065030) · Faizan Ahmed (F2022065025)
Supervisor: Dr. Muhammad Adnan
