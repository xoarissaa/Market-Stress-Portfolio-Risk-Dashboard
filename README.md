# Market Stress & Portfolio Risk Monitor 📉

A professional-grade financial analytics dashboard for monitoring market regimes, stress-testing portfolios, and projecting future wealth using context-aware machine learning.

**Goal**: To answer the definitive portfolio question: *"How does my strategy behave specifically during market stress, and is it built to survive the next decade?"*

---

## 🌟 Key Features

### 🌪️ 1. Market Weather Station (Regime Detection)
Stop treating all market days the same. This system uses a **Gaussian Mixture Model (GMM)** to classify the market into three distinct "Weather" states:
*   **🟢 Stable Growth**: Low volatility, positive trend. The "Normal" state.
*   **🟡 Elevated Uncertainty**: Transition periods, high noise, and trend breaks.
*   **🔴 High Stress**: Bear markets, liquidity shocks, and high-volatility crashes.

**Indicators Tracked**:
- **Trend**: Price vs. 200-day Moving Average (SPY).
- **Fear Index**: VIX Volatility (`^VIX`).
- **Credit Stress**: High Yield vs. Treasury spreads (`HYG`/`IEF`).
- **Yield Curve**: 10Y Treasury - 3-Month T-Bill spread.

### ⚖️ 2. Portfolio Laboratory (Side-by-Side Comparison)
Compare two portfolios or individual ETFs with institutional-grade forensics:
*   **Interactive Builder**: Construct complex portfolios using a library of 50+ ETFs across US Equity, Bonds, International, Sectors, and Commodities.
*   **Quick Presets**: Instantly load benchmarks like "60/40 Classic", "All Weather", or "Nasdaq 100".
*   **Regime-Conditional Breakdown**: View performance (CAGR, Sharpe, etc.) filtered by each market state to see who wins when the "weather" turns bad.
*   **Decision Summary**: Automated plain-English insights identifying which portfolio is more defensive, which has better growth potential, and which handles stress more efficiently.

### 🎲 3. Regime-Aware Monte Carlo Simulation
Traditional simulations assume the future is a random walk. Ours assumes the future is a sequence of regimes:
*   **Custom Market Outlook**: Set your own probabilities for the next decade (e.g., *"What if we spend 40% of the time in High Stress?"*).
*   **Survival Cones**: Visualize 10th (Conservative), 50th (Median), and 90th (Optimistic) percentile wealth paths.
*   **Cloning & Scenarios**: Locally modify portfolio weights to see how minor adjustments impact long-term survival without changing your main comparison.

### 📊 4. Deep Risk Analytics
Every calculation is performed with regime-context in mind:
*   **CAGR & Volatility**: Annualized growth and risk.
*   **Sharpe & Sortino Ratios**: Risk-adjusted and downside-risk-adjusted returns.
*   **Downside Capture Ratio**: Measures exactly how much of the market's pain you take.
*   **Value at Risk (VaR)**: Daily 95% confidence loss projections.
*   **Max Drawdown**: Historical peak-to-trough pain metrics.

---

## 🛠️ Technology Stack
*   **Python 3.10+**
*   **Streamlit**: Advanced interactive Web UI.
*   **Scikit-Learn**: Gaussian Mixture Models (Unsupervised Learning).
*   **Pandas/NumPy**: High-performance financial vectorization.
*   **Plotly**: Interactive institutional-quality charting.
*   **YFinance**: Real-time market data integration.

---

## 🌎 Supported Universe
Built-in support for a broad range of assets:
- **US Equities**: SPY, QQQ, VTI, VTV, VUG, SCHD, etc.
- **Fixed Income**: BND, AGG, TLT, IEF, SHY, LQD, HYG.
- **International**: VEA, VWO, IEFA, IEMG, VXUS.
- **Sectors**: Technology (XLK), Healthcare (XLV), Finance (XLF), Energy (XLE), etc.
- **Alternatives**: Gold (GLD/IAU), Silver (SLV), Commodities (DBC).
- **Canada**: XIC.TO, VCN.TO, VDY.TO, VFV.TO, ZLB.TO.

---

## 🚀 Installation & Usage

1.  **Clone & Setup**
    ```bash
    git clone https://github.com/your-username/market-stress-monitor.git
    cd market-stress-monitor
    python -m venv .venv
    source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

2.  **Initialize Data**
    The dashboard allows you to refresh and retrain directly from the UI, but you can also run:
    ```bash
    python src/data_loader.py   # Fetch data
    python src/market_regime.py # Train Model
    ```

3.  **Launch**
    ```bash
    streamlit run app.py
    ```

---

## 🧠 Methodology: The Intelligence Layer
The core of this system is the **GMM Regime Model**. Unlike simple thresholding (e.g., "VIX > 30"), the GMM looks at the *joint distribution* of Trend, Volatility, and Credit Stress. This allows it to identify "Hidden" stress states where one indicator might look fine but the system as a whole is fragile.

---

## ⚠️ Disclaimer
This tool is for **informational and research purposes only**. It does not constitute financial advice. Past performance is not indicative of future results. All projections are probabilistic.

