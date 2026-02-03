# Market Stress & Portfolio Risk Monitor 📉

A professional-grade dashboard for monitoring market regimes, stress testing portfolios, and projecting future wealth using context-aware analytics.

**Goal**: To answer the question: *"How does my portfolio behave specifically during market stress?"*

---

## 🌟 Key Features

### 1. 🌪️ Market Regime Detection (The "Weather Report")
Instead of predicting crashes, this system classifies the current market environment into three "Regimes" using a **Gaussian Mixture Model (GMM)**:
*   **🟢 Stable Growth**: Low volatility, positive trend.
*   **🟡 Elevated Uncertainty**: Transition periods, high noise.
*   **🔴 High Stress**: Bear markets, liquidity shocks (e.g., 2008, 2020).

### 2. 💼 Context-Aware Portfolio Analysis
*   **Interactive Builder**: Construct portfolios using ETFs from your universe.
*   **Conditional Performance**: See your portfolio's CAGR, Volatility, and Sharpe Ratio *specifically* for each regime.
*   **Fragility Gauge**: A rolling **Downside Capture Ratio** to measure how much pain you take when the market drops.
*   **Hedge Monitor**: Tracks rolling correlations (e.g., Stock/Bond) to alert when diversification is failing.

### 3. 🔮 Regime-Weighted Capital Projection
*   **Advanced Monte Carlo**: Project your wealth over 5–30 years.
*   **Scenario Planning**: Input your own probability assumptions (e.g., *"What if the next decade is 50% Stress?"*).
*   **Survival Cones**: Visualize the 10th, 50th, and 90th percentile outcomes.

---

## 🛠️ Technology Stack
*   **Python 3.10+**
*   **Streamlit**: Interactive Web UI.
*   **Scikit-Learn**: Gaussian Mixture Models for regime detection.
*   **Pandas/NumPy**: Vectorized financial calculations.
*   **Plotly**: Interactive charting.
*   **YFinance**: Live market data feed.

---

## 🚀 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/market-stress-monitor.git
    cd market-stress-monitor
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Data Pipeline** (First Run Only)
    ```bash
    # 1. Fetch latest market data
    python src/data_loader.py
    
    # 2. Train Regime Model
    python src/market_regime.py
    ```

5.  **Launch Dashboard**
    ```bash
    streamlit run app.py
    ```

---

## 📊 Methodology

### The Regime Model
We use 4 key indicators to detect stress:
1.  **Trend**: Price vs. 200-day Moving Average (SPY).
2.  **Volatility**: VIX Index (`^VIX`).
3.  **Credit Stress**: High Yield vs. Treasury Ratio (`HYG` / `IEF`).
4.  **Yield Curve**: 10Y Treasury - 3-Month T-Bill (`10Y-3M`).

The GMM clusters these proprietary signals into the 3 regimes described above.

---

## ⚠️ Disclaimer
This tool is for **informational and research purposes only**. It does not constitute financial advice. All projections are probabilistic and based on historical data.
