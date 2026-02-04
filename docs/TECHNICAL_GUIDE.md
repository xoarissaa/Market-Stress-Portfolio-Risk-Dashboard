# Developer Guide & Technical Architecture 📐

**Project**: Market Stress & Portfolio Risk Dashboard
**Version**: 1.0.0
**Language**: Python 3.10+
**Framework**: Streamlit

---

## 🏗️ 1. Project Structure

The project follows a modular "Model-View-Controller" (MVC) adaptation for Streamlit.

```text
root/
├── app.py                  # [View/Controller] Main entry point. Handles UI layout and state.
├── requirements.txt        # Dependency lockfile (plotly, yfinance, sklearn, chrome).
├── data/
│   ├── raw/                # Temporary storage for raw fetch.
│   └── processed_market_regimes.parquet  # [Model] The "Brain". Pre-computed regimes.
├── outputs/
│   └── models/             # Serialized ML models (GMM, Scalers).
└── src/
    ├── definitions.py      # [Config] Text, labels, and color constants.
    ├── data_loader.py      # [ETL] Fetches live data from Yahoo Finance.
    ├── market_regime.py    # [ML] Trains/Infers the Gaussian Mixture Model.
    ├── portfolio.py        # [Logic] ETF Portfolio Object (Weights, Returns).
    ├── risk_engine.py      # [Logic] Monte Carlo & Advanced Risk Math.
    └── metrics.py          # [Utils] Math helpers (CAGR, Sharpe, Drawdown).
```

---

## 🔌 2. API & Data Inputs

The system relies on **Yahoo Finance (`yfinance`)** as the sole data source.

### Key Tickers Fetched (`src/data_loader.py`)
*   **Market Signal**: `SPY` (S&P 500)
*   **Volatility**: `^VIX` (CBOE VIX)
*   **Credit Risk**: `HYG` (High Yield Bonds), `IEF` (7-10Y Treasuries)
*   **Yield Curve**: `^TNX` (10Y Yield), `^IRX` (13W Bill Yield)
*   **Reference ETFs**: The "Universe" of ~15 tickers (e.g., `VTI`, `XIC.TO`) defined in `app.py`.

**Note**: The app caches data in `data/processed_market_regimes.parquet` to avoid hitting API limits.

---

## 🤖 3. Machine Learning Details (`src/market_regime.py`)

The core "Market Stress" engine involves an Unsupervised Learning pipeline.

### Step A: Feature Engineering
1.  **Trend**: `(Price / 200d_MA) - 1` (positive = uptrend).
2.  **Volatility**: `VIX / 60-day_Moving_Avg` (identifies volatility spikes).
3.  **Credit spread**: `HYG / IEF` ratio (falling = rising credit stress).
4.  **Yield Curve**: `10Y - 3M` (negative = inversion/recession risk).

### Step B: The Model
*   **Algorithm**: Gaussian Mixture Model (GMM).
*   **Components**: 3 (Calm, Choppy, Stress).
*   **Training**: It fits on history (2000–Present) to learn the statistical "shape" of each regime.
*   **Inference**: Assigns a probability to today's data (e.g., "80% prob of Stress").

---

## 🎲 4. Risk Engine Logic (`src/risk_engine.py`)

### A. Regime-Weighted Monte Carlo
Standard MC implies "all history is equally likely". We use **Conditional Sampling**:
1.  User inputs probabilities: $P_{calm}, P_{choppy}, P_{stress}$.
2.  Engine separates historical returns into 3 buckets: `Pool_Calm`, `Pool_Choppy`, `Pool_Stress`.
3.  For each day in the future (T+1...T+N):
    *   Roll a die based on User Probabilities to pick a Regime.
    *   Randomly sample **one day's return** from that specific Regime's history.
4.  Repeat 1,000 times to build probability cones.

### B. Downside Capture Ratio
$$ DCR = \frac{\text{Average Portfolio Return (when Market < 0)}}{\text{Average Market Return (when Market < 0)}} $$
*   Goal: $< 1.0$ (Portfolio falls less than the market).

---

## 🛠️ 5. How to Extend

### Adding a new feature?
1.  **New Math**: Add function to `src/metrics.py` or `src/risk_engine.py`.
2.  **New Data**: Add ticker to `src/data_loader.py` and delete `data/` to force re-fetch.
3.  **New UI**: Add a widget to `app.py`.

### Changing the Model?
1.  Edit `src/market_regime.py` (e.g., change `n_components` to 4).
2.  Run `python src/market_regime.py` to re-train and save the pickle files.
