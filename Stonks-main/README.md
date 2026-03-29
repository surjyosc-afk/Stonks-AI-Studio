
# 🇮🇳 STONKS — NSE AI Stock Analysis & Forecasting System

A production-grade AI system for Indian equity markets (NSE/BSE), combining LSTM price forecasting, FinBERT sentiment analysis, technical pattern detection, corporate filings intelligence, full NSE universe scanning, and **Institutional 2FA Security** — all exposed via a FastAPI REST API.

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
│  api/app.py                    ← FastAPI REST + 2FA Auth │
│  ingestion/data_loader.py      ← AV OHLCV + Memory Cache │
│  features/feature_utils.py     ← scaler + windowing      │
│  inference/predictor.py        ← LSTM / Heuristic Fallback│
│  nlp/finbert_sentiment.py      ← FinBERT inference       │
│  technicals/pattern_detection.py ← S/R, breakouts        │
│  signals/opportunity_radar.py  ← composite alerts        │
│  signals/nse_universe.py       ← full NSE universe scan  │
│  filings/filings.py            ← insider trades, deals   │
│  backtesting/backtester.py     ← pattern success rates   │
└──────────────────────────────────────────────────────────┘
```

Strict separation of concerns:
| Concern | Location |
|---|---|
| Preprocessing, feature engineering, training | `notebooks/model_pipeline.ipynb` ONLY |
| Inference, API, caching, auth, signal logic | `src/` modules ONLY |

---

## 📁 Project Structure

```text
Stonks/
├── .env                       ← API keys (never commit to GitHub)
├── notebooks/
│   └── model_pipeline.ipynb   ← Training pipeline
├── models/
│   └── saved_models/
│       ├── lstm_model.pt      ← Trained LSTM weights
│       └── scaler.pkl         ← Fitted MinMaxScaler
├── data/
│   ├── raw/                   ← Raw OHLCV CSVs
│   └── processed/
│       ├── processed_stock_data.csv ← Feature-engineered data
│       └── sentiment_output.csv     ← FinBERT batch output
├── src/
│   ├── ingestion/data_loader.py     ← Alpha Vantage data fetcher
│   ├── features/feature_utils.py
│   ├── inference/predictor.py
│   ├── nlp/finbert_sentiment.py
│   ├── technicals/pattern_detection.py
│   ├── signals/
│   │   ├── opportunity_radar.py
│   │   └── nse_universe.py          ← NSE universe scanner
│   ├── filings/filings.py           ← Corporate filings & insider trades
│   ├── backtesting/backtester.py    ← Pattern back-testing engine
│   ├── api/app.py                   ← Main FastAPI application
│   └── utils/helpers.py
├── config/config.yaml
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone <repo-url> && cd Stonks
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up credentials

Create a `.env` file in the root directory and add your API key:

```env
# Alpha Vantage (Historical Data)
# Free key at: [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
ALPHAVANTAGE_API_KEY=your_api_key_here
```

### 3. Train the AI (Headless Automated Mode)

Run the following command to securely download historical data and train the PyTorch neural network. This command bypasses Jupyter UI and ignores legacy virtual environments:

```bash
jupyter nbconvert --execute --inplace --ExecutePreprocessor.kernel_name=python3 notebooks/model_pipeline.ipynb
```
*This will generate `lstm_model.pt` and `scaler.pkl` in your `models/saved_models/` folder.*

### 4. Start the API

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: **http://localhost:8000/docs**

---

## 🔐 Authentication & Security

The platform is protected by Institutional-grade 2FA (Google Authenticator) powered by `pyotp`.

* **Demo Login:** `analyst@stonks.com`
* **Demo Password:** `admin`
* **Master 2FA Bypass:** Type `000000` at the QR code screen for seamless live presentations.

---

## 🌐 API Reference

### `GET /health`
Liveness check.

### `POST /predict`
LSTM next-close-price prediction. Includes 30-day historical arrays for curved frontend charting. Includes a dynamic Heuristic Algorithmic Fallback if deep learning weights are uninitialized.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "ZOMATO.BSE", "period": "6mo", "use_mock": false}'
```

### `POST /sentiment`
FinBERT sentiment classification on financial text.

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

### `GET /signals/{ticker}`
Composite Opportunity Radar alert — combines price, volume, sentiment, and pattern signals.

```bash
curl "http://localhost:8000/signals/RELIANCE.BSE?period=6mo&include_news=true"
```

### `GET /patterns/{ticker}`
Chart pattern detection with back-tested success rates.

```bash
curl "http://localhost:8000/patterns/TCS.BSE?period=1y&include_backtest=true"
```

### `GET /filings/{ticker}`
Corporate filings intelligence — bulk/block deals, insider trades, announcements.

```bash
curl "http://localhost:8000/filings/RELIANCE.BSE"
```

### `GET /backtest/{ticker}`
Pattern back-test — historical win rate and avg return per pattern on a specific stock.

```bash
curl "http://localhost:8000/backtest/RELIANCE.BSE?period=2y&holding_days=10"
```

### `GET /universe/scan`
Scan the full NSE universe for opportunities — ranked by signal strength.

### `GET /universe/sector/{sector}`
Sector-level opportunity summary — bullish/bearish breakdown with top picks.

---

## ⚙️ Configuration

Edit `config/config.yaml` to adjust:

| Key | Description | Default |
|---|---|---|
| `data.window_size` | LSTM look-back window | 60 |
| `model.hidden_size` | LSTM hidden units | 128 |
| `model.num_layers` | LSTM depth | 2 |
| `signals.volume_spike_threshold` | Volume spike multiplier | 2.0× |
| `signals.price_breakout_pct` | Breakout penetration | 3% |
| `api.port` | API port | 8000 |

---

## 📦 Key Dependencies

- **PyTorch 2.3** — LSTM training & inference
- **HuggingFace Transformers** — ProsusAI/finbert
- **Alpha Vantage** — NSE/BSE historical OHLCV data
- **pyotp** — Cryptographic 2FA & Security
- **nbconvert** — Headless automated model training
- **ta** — Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- **FastAPI + uvicorn** — REST API
- **scikit-learn** — MinMaxScaler
- **pandas, numpy** — Data manipulation

---

## 📜 Licence

MIT — free to use, modify, and distribute.