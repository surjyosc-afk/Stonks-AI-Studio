# 🇮🇳 STONKS — NSE/BSE AI Stock Analysis & Forecasting System

A production-grade AI system for Indian equity markets, combining **LSTM price forecasting**, **FinBERT sentiment analysis**, **technical pattern detection**, **corporate filings intelligence**, **NSE universe scanning**, and **2FA-secured REST API** — all served through a FastAPI backend with a glassmorphism frontend UI.

---

## 📐 Architecture

```text
┌─────────────────────────────────────────────────────────┐
│              notebooks/model_pipeline.ipynb             │
│  • Data preprocessing  • Feature engineering            │
│  • LSTM training       • FinBERT sentiment pipeline     │
│  • Artefact saving (model.pt, scaler.pkl, CSVs)         │
└───────────────────────┬─────────────────────────────────┘
                        │ artefacts
        ┌───────────────▼───────────────┐
        │      models/saved_models/     │
        │  lstm_model.pt  scaler.pkl    │
        └───────────────┬───────────────┘
                        │ load
┌───────────────────────▼──────────────────────────────────┐
│                    src/ modules                          │
│                                                          │
│  api/app.py                      ← FastAPI REST + 2FA   │
│  ingestion/data_loader.py        ← Alpha Vantage OHLCV  │
│  features/feature_utils.py       ← Scaler + windowing   │
│  inference/predictor.py          ← LSTM inference       │
│  nlp/finbert_sentiment.py        ← FinBERT sentiment    │
│  technicals/pattern_detection.py ← S/R, breakouts, H&S │
│  signals/opportunity_radar.py    ← Composite alerts     │
│  signals/nse_universe.py         ← NSE universe scan    │
│  filings/filings.py              ← Insider trades       │
│  backtesting/backtester.py       ← Pattern win rates    │
└──────────────────────────────────────────────────────────┘
```

**Strict separation of concerns:**

| Concern | Location |
|---|---|
| Data preprocessing, feature engineering, model training | `notebooks/model_pipeline.ipynb` **only** |
| Inference, API, caching, auth, signal logic | `src/` modules **only** |

---

## 📁 Project Structure

```text
Stonks/
├── .env                              ← API keys (never commit)
├── config/
│   └── config.yaml                   ← Central configuration
├── notebooks/
│   └── model_pipeline.ipynb          ← Full training pipeline
├── models/
│   └── saved_models/
│       ├── lstm_model.pt             ← Trained LSTM weights
│       └── scaler.pkl                ← Fitted MinMaxScaler
├── data/
│   ├── raw/                          ← Raw OHLCV CSVs
│   └── processed/
│       ├── processed_stock_data.csv  ← Feature-engineered data
│       └── sentiment_output.csv      ← FinBERT batch output
├── src/
│   ├── api/app.py                    ← FastAPI application + 2FA auth
│   ├── ingestion/data_loader.py      ← Alpha Vantage data fetcher
│   ├── features/feature_utils.py     ← Scaler + sequence prep
│   ├── inference/predictor.py        ← LSTM model loader & inference
│   ├── nlp/finbert_sentiment.py      ← FinBERT sentiment engine
│   ├── technicals/pattern_detection.py ← Chart pattern detection
│   ├── signals/
│   │   ├── opportunity_radar.py      ← Composite signal scoring
│   │   └── nse_universe.py           ← NSE-wide watchlist scanner
│   ├── filings/filings.py            ← Corporate filings & insider trades
│   ├── backtesting/backtester.py     ← Pattern back-testing engine
│   └── utils/helpers.py              ← Config, logging, date utils
├── frontend/
│   ├── index.html                    ← Main dashboard UI
│   ├── terminal.html                 ← Terminal/login view
│   ├── app.js                        ← API integration & chart logic
│   └── styles.css                    ← Glassmorphism theme
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/surjyosc-afk/Stonks-AI-Studio.git
cd Stonks-AI-Studio/Stonks-main

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up credentials

Create a `.env` file in the project root:

```env
# Alpha Vantage — free key at https://www.alphavantage.co/support/#api-key
ALPHAVANTAGE_API_KEY=your_api_key_here
```

### 3. Train the model (headless mode)

This downloads historical data, engineers features, trains the PyTorch LSTM, runs the FinBERT sentiment pipeline, and saves all artefacts to `models/saved_models/`:

```bash
jupyter nbconvert --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  notebooks/model_pipeline.ipynb
```

After this step you will have:
- `models/saved_models/lstm_model.pt` — trained LSTM weights
- `models/saved_models/scaler.pkl` — fitted MinMaxScaler
- `data/processed/processed_stock_data.csv`
- `data/processed/sentiment_output.csv`

### 4. Start the API

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs: **http://localhost:8000/docs**

### 5. Open the frontend

Open `frontend/index.html` in your browser. Enter a ticker (e.g. `RELIANCE.BSE`) and click **Analyze**.

---

## 🔐 Authentication & Security

The API is protected by **TOTP-based 2FA** (Google Authenticator compatible) via `pyotp`.

| Field | Value |
|---|---|
| Demo email | `analyst@stonks.com` |
| Demo password | `admin` |
| 2FA bypass (demo only) | `000000` |

> ⚠️ The `000000` master bypass is for live demos only. Disable it in production by removing that branch from `src/api/app.py`.

---

## 🌐 API Reference

### `GET /health`
Liveness check — confirms the API is running.

---

### `POST /predict`
LSTM next-close-price prediction. Returns the predicted price alongside 30-day historical arrays for frontend charting. Falls back to a heuristic algorithm if `lstm_model.pt` has not been trained yet.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "ZOMATO.BSE", "period": "6mo", "use_mock": false}'
```

**Response includes:** `current_price`, `predicted_price`, `historical_dates[]`, `historical_prices[]`

---

### `POST /sentiment`
FinBERT sentiment classification on one or more financial headlines. Returns label (`positive` / `negative` / `neutral`), numerical score (±1), confidence, and full probability distribution.

```bash
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Reliance Industries reports record quarterly profit",
      "Nifty50 crashes 3% amid global sell-off"
    ]
  }'
```

---

### `GET /signals/{ticker}`
Composite **Opportunity Radar** alert — aggregates price momentum, volume spikes, FinBERT sentiment, and technical pattern signals into a single score and `BUY` / `SELL` / `WATCH` action.

```bash
curl "http://localhost:8000/signals/RELIANCE.BSE?period=6mo&include_news=true"
```

**Scoring breakdown:**

| Signal | Weight |
|---|---|
| FinBERT sentiment | ±0.30 |
| Price momentum | ±0.25 |
| Volume spike | +0.15 |
| Bullish breakout | +0.20 |
| Bullish reversal (RSI) | +0.10 |

Composite ≥ 0.30 → `BUY` · ≤ -0.20 → `SELL` · otherwise → `WATCH`

---

### `GET /patterns/{ticker}`
Chart pattern detection with optional back-tested success rates.

Detects:
- **Support & Resistance** — rolling-extrema clustering
- **Breakouts** — confirmed close above resistance / below support (≥ 3% penetration, 2-bar confirmation)
- **Trend Reversals** — RSI crossing below 30 (bullish) or above 70 (bearish)
- **Head & Shoulders** — simplified pivot-high detection

```bash
curl "http://localhost:8000/patterns/TCS.BSE?period=1y&include_backtest=true"
```

---

### `GET /filings/{ticker}`
Corporate filings intelligence — bulk/block deals, insider trades, and regulatory announcements.

```bash
curl "http://localhost:8000/filings/RELIANCE.BSE"
```

---

### `GET /backtest/{ticker}`
Pattern back-test — historical win rate and average return per pattern type for a specific stock.

```bash
curl "http://localhost:8000/backtest/RELIANCE.BSE?period=2y&holding_days=10"
```

---

### `GET /universe/scan`
Scan the full NSE universe and return all tickers ranked by absolute composite signal score (only tickers with `|score| ≥ 0.20` are included).

---

### `GET /universe/sector/{sector}`
Sector-level opportunity summary — bullish/bearish breakdown with top picks for the given sector.

---

## 🧠 Model Details

### LSTM Price Predictor

| Parameter | Value |
|---|---|
| Architecture | Multi-layer LSTM (PyTorch) |
| Input features | 10 (Close, Open, High, Low, Volume, RSI, MACD, MACD Signal, BB Upper, BB Lower) |
| Hidden size | 128 |
| Layers | 2 |
| Dropout | 0.2 |
| Look-back window | 60 bars |
| Output | Next-day Close price |
| Training epochs | 50 (patience = 10 early stopping) |
| Optimiser | Adam (lr = 0.001) |

Training and scaler fitting happen **exclusively** in `notebooks/model_pipeline.ipynb`. The `src/` modules load saved artefacts at inference time — they never retrain.

### FinBERT Sentiment Engine

Uses `ProsusAI/finbert` (HuggingFace) to classify financial headlines as `positive`, `negative`, or `neutral`. Individual predictions are aggregated into a single composite score using confidence-weighted averaging, clamped to `[-1, 1]`.

---

## ⚙️ Configuration

All tuneable parameters live in `config/config.yaml`:

```yaml
data:
  window_size: 60          # LSTM look-back window (bars)
  test_split: 0.2

model:
  input_size: 10           # feature count
  hidden_size: 128
  num_layers: 2
  dropout: 0.2

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  patience: 10             # early stopping

signals:
  volume_spike_threshold: 2.0   # ×rolling average
  price_breakout_pct: 0.03      # 3% penetration required

patterns:
  support_resistance_window: 20
  breakout_confirmation_bars: 2
  trend_reversal_rsi_oversold: 30
  trend_reversal_rsi_overbought: 70

api:
  host: "0.0.0.0"
  port: 8000
```

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `torch==2.3.0` | LSTM training & inference |
| `transformers==4.41.2` | ProsusAI/FinBERT |
| `fastapi==0.111.0` + `uvicorn` | REST API |
| `alpha-vantage==2.3.1` | NSE/BSE historical OHLCV |
| `ta==0.11.0` | RSI, MACD, Bollinger Bands, ATR |
| `scikit-learn==1.4.2` | MinMaxScaler |
| `pyotp>=2.9.0` | TOTP 2FA |
| `nbconvert>=7.16.0` | Headless notebook training |
| `joblib==1.4.2` | Scaler serialisation |
| `pandas==2.2.2` + `numpy==1.26.4` | Data manipulation |
| `plotly==5.22.0` | Interactive charting |
| `beautifulsoup4==4.12.3` | Filings scraping |

Full list in `requirements.txt`.

---

## 🖥️ Frontend

The frontend (`frontend/`) is a single-page dashboard built with **Tailwind CSS**, **Chart.js**, and **Font Awesome**. It connects directly to the FastAPI backend.

**Features:**
- Ticker search with live API calls (sequential fetching to respect Alpha Vantage rate limits)
- Curved 30-day historical price chart with LSTM prediction point
- FinBERT sentiment bar (bullish / neutral / bearish)
- Composite signal badge (`BUY` / `SELL` / `WATCH`) with locked reveal interaction
- Technical patterns panel (breakouts, reversals, H&S)
- Support & resistance levels in INR
- Glassmorphism dark theme with video background and animated loader

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Do not make real investment decisions based on model output. Equity markets carry substantial risk.

---

## 📜 Licence

MIT — free to use, modify, and distribute.
