# Feature Specification 📋

## Current Features

### Tab 1: Market Stress Monitor 🌪️
*   **Live Regime Status**: Displays the current market state (Stable Growth, Elevated Uncertainty, High Stress).
*   **Key Indicators**: Visualizes the 4 underlying signals driving the model:
    *   Trend (SPY 200d MA)
    *   Volatility (VIX)
    *   Credit Risk (High Yield Spreads)
    *   Yield Curve (10Y-3M)
*   **Historical Context**: Interactive chart showing SPY price colored by historical regimes.

### Tab 2: Context-Aware Portfolio 💼
*   **Interactive Portfolio Builder**: Interface to select ETFs and assign weights dynamically (summing to 100%).
*   **Regime-Conditional Performance**: Table showing CAGR, Volatility, and Sharpe Ratio filtered by each regime (e.g., "How did I do in 'High Stress'?").
*   **Risk Diagnostics**:
    *   **Hedge Monitor**: 60-day Rolling Correlation chart (Portfolio vs. SPY) to detect when diversification fails.
    *   **Fragility Gauge**: Downside Capture Ratio metric (target < 1.0).

### Tab 3: Future Wealth Projection 🔮
*   **Regime-Weighted Monte Carlo**: 500-iteration simulation that samples returns based on user-defined regime probabilities.
*   **Scenario Inputs**: Sliders to set "Outlook" (e.g., 50% Stress) and Time Horizon (5–30 Years).
*   **Survival Cones**: Visualizes 10th (Conservative), 50th (Median), and 90th (Optimistic) percentile outcomes.

### Tab 4: User Guide 📘
*   **Methodology**: Explains the GMM model and regime definitions.
*   **FAQ**: Common questions about interpretation and data sources.

---

## Metrics Calculated 🧮
*   **CAGR** (Compound Annual Growth Rate)
*   **Annualized Volatility** (Standard Deviation * sqrt(252))
*   **Sharpe Ratio** (Risk-adjusted return)
*   **Sortino Ratio** (Downside risk-adjusted return)
*   **Max Drawdown** (Peak-to-Trough decline)
*   **Downside Capture Ratio** (Performance relative to benchmark when benchmark < 0)
*   **Rolling Correlation** (60-day window vs. SPY)
*   **Value at Risk (VaR)** (Daily 95% confidence level)

---

## Models Used 🤖
*   **Gaussian Mixture Model (GMM)**:
    *   **Library**: `scikit-learn`
    *   **Components**: 3 (Calm, Choppy, Stress)
    *   **Covariance Type**: 'full'
    *   **Training Data**: 2000–Present (Daily)

---

## Inputs Supported ⌨️
*   **Ticker Universe**:
    *   **US**: SPY, VTI, SCHD, BND, SPLV, IAU, QUAL, MOAT, QUS, TLT, IEF, GLD.
    *   **Canada**: XIC.TO, VCN.TO, VDY.TO, ZAG.TO, VFV.TO, ZLB.TO.
*   **User Parameters**:
    *   Portfolio Weights (0–100%)
    *   Initial Investment Amount ($)
    *   Time Horizon (Years)
    *   Future Regime Probabilities (%)
