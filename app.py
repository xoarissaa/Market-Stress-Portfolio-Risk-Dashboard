import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from src.market_regime import MarketRegimeDetector
from src.portfolio import Portfolio
from datetime import datetime

# Set Page Config
st.set_page_config(page_title="Market Stress & Portfolio Monitor", page_icon="📉", layout="wide")

# Title
st.title("📉 Market Stress & Portfolio Context")
st.markdown("""
**System Status**: Unified Dashboard Active.
*Monitor market 'weather' and check how your portfolio ships sail in these conditions.*
""")

# --- Ticker Universe (From GitHub) ---
US_ETFS = ["SPY", "VTI", "SCHD", "BND", "SPLV", "IAU", "QUAL", "MOAT", "QUS"]
CAD_ETFS = ["XIC.TO", "VCN.TO", "VDY.TO", "ZAG.TO", "VFV.TO", "ZLB.TO"]
ALL_TICKERS = sorted(list(set(US_ETFS + CAD_ETFS + ['TLT', 'IEF', 'GLD']))) # Add common ones

# --- Load Market Data ---
@st.cache_data
def load_market_data():
    try:
        df = pd.read_parquet("data/processed_market_regimes.parquet")
        return df
    except FileNotFoundError:
        return None

market_df = load_market_data()

# --- Load Definitions ---
from src.definitions import RegimeDefinitions

# --- Tabs ---
# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🌪️ Market Stress Monitor", "💼 Context-Aware Portfolio", "🔮 Future Wealth Projection", "📘 User Guide"])

with tab1:
    if market_df is None:
        st.error("Market Data not found. Run pipeline first.")
    else:
        # Latest Data Point
        latest = market_df.iloc[-1]
        regime_label = latest['Regime_Label']
        
        # Get Text Assets
        r_name, r_desc, r_context = RegimeDefinitions.get_regime_info(regime_label)
        r_color = RegimeDefinitions.COLORS.get(regime_label, "off")
        
        # --- Top Row: Status & Gauge ---
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.metric(label="Market Condition", value=r_name, delta="Regime Status", delta_color=r_color)

        with col2:
            st.metric(label="Market Fear (VIX)", value=f"{latest['^VIX']:.1f}", delta=f"{latest['^VIX'] - market_df.iloc[-2]['^VIX']:.2f}", delta_color="inverse")
        
        with col3:
            st.info(f"**Insight:** {r_desc}\n\n{r_context}")

        # --- Middle: Historical Regime Chart ---
        st.subheader("Historical Market Regimes")
        fig = px.scatter(market_df.reset_index(), x='Date', y='SPY', color='Regime_Label', 
                         color_continuous_scale=['green', 'yellow', 'red'],
                         title="S&P 500 Price Colored by Market Stress Regime")
        st.plotly_chart(fig, use_container_width=True)

        # --- Bottom: Core Indicators ---
        st.subheader("Core Risk Indicators")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.line_chart(market_df['Trend_Signal'].tail(252)); st.caption("Trend (200d MA)")
        with c2: st.line_chart(market_df['VIX_Signal'].tail(252)); st.caption("Volatility (VIX)")
        with c3: st.line_chart(market_df['Credit_Signal'].tail(252)); st.caption("Credit Risk (HYG/IEF)")
        with c4: st.line_chart(market_df['Yield_Curve'].tail(252)); st.caption("Yield Curve (10Y-13W)")

with tab2:
    st.header("Context-Aware Portfolio Analysis")
    
    col_config, col_res = st.columns([1, 3])
    
    with col_config:
        st.subheader("1. Build Portfolio")
        selected_tickers = st.multiselect("Select Assets", ALL_TICKERS, default=["SPY", "TLT"])
        
        weights = {}
        st.write("Assign Weights:")
        total_weight = 0
        for t in selected_tickers:
            w = st.slider(f"{t} %", 0, 100, int(100/len(selected_tickers)) if len(selected_tickers) > 0 else 0, key=f"w_{t}")
            weights[t] = w / 100.0
            total_weight += w
            
        if total_weight != 100:
            st.warning(f"Total Weight: {total_weight}%. Please sum to 100%.")
            
        run_analysis = st.button("Run Context Analysis")

    with col_res:
        if run_analysis and market_df is not None:
            st.subheader("2. Performance vs Regimes")
            
            with st.spinner("Fetching Portfolio Data & Analyzing..."):
                # 1. Init Portfolio
                port = Portfolio(weights)
                
                # 2. Fetch Data (Align date range with market_df)
                start_date = market_df.index.min().strftime('%Y-%m-%d')
                port.fetch_data(start_date=start_date)
                
                # 3. Merge Portfolio Returns with Regime Labels
                # Ensure indices match
                p_rets = port.portfolio_returns.to_frame(name='Portfolio_Ret')
                merged = p_rets.join(market_df['Regime_Label'], how='inner')
                
                if merged.empty:
                    st.error("No overlapping data between Portfolio and Regimes.")
                else:
                    # 4. Total Stats
                    stats = port.get_performance_summary()
                    
                    # 5. Conditional Stats (The "Context")
                    regime_stats = []
                    
                    for r_code in sorted(market_df['Regime_Label'].unique()):
                        r_name = RegimeDefinitions.LABELS.get(r_code, f"Regime {r_code}")
                        
                        subset = merged[merged['Regime_Label'] == r_code]['Portfolio_Ret']
                        if not subset.empty:
                            ann_ret = subset.mean() * 252
                            ann_vol = subset.std() * np.sqrt(252)
                            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
                            regime_stats.append({
                                "Regime": r_name,
                                "Ann Return": f"{ann_ret:.1%}",
                                "Ann Volatility": f"{ann_vol:.1%}",
                                "Sharpe": f"{sharpe:.2f}",
                                "Daily VaR (95%)": f"{subset.quantile(0.05):.1%}"
                            })
                    
                    # Display
                    r1, r2 = st.columns(2)
                    with r1:
                        st.markdown("#### Overall Performance")
                        st.dataframe(pd.DataFrame([stats]).T)
                        
                    with r2:
                        st.markdown("#### Performance by Market Regime")
                        st.table(pd.DataFrame(regime_stats))
                    
                    # --- 3. Risk Diagnostics ---
                    st.subheader("3. Risk Diagnostics")
                    
                    # Prepare Benchmark Data
                    spy_prices = market_df['SPY']
                    spy_rets = spy_prices.pct_change().dropna()
                    
                    # Align Portfolio and Benchmark
                    params_diagnostics = pd.concat([port.portfolio_returns, spy_rets], axis=1).dropna()
                    params_diagnostics.columns = ['Portfolio', 'SPY']
                    
                    # A. Rolling Correlation (Hedge Monitor)
                    roll_corr = RiskEngine.rolling_correlation(params_diagnostics['Portfolio'], params_diagnostics['SPY'], window=60)
                    
                    # B. Downside Capture
                    dcr = RiskEngine.calculate_downside_capture(params_diagnostics['Portfolio'], params_diagnostics['SPY'])
                    
                    # Display
                    d_col1, d_col2 = st.columns([2, 1])
                    
                    with d_col1:
                        st.markdown("**🛡️ Hedge Monitor (Rolling 60d Correlation vs SPY)**")
                        st.caption("Lower is better for hedging. >0.8 means your portfolio moves exactly like the market.")
                        st.line_chart(roll_corr)
                        
                    with d_col2:
                        st.markdown("**📉 Fragility Gauge**")
                        st.metric(
                            label="Downside Capture Ratio", 
                            value=f"{dcr:.2f}", 
                            delta="< 1.0 is Good" if dcr < 1.0 else "High Fragility",
                            delta_color="normal" if dcr < 1.0 else "inverse"
                        )
                        st.info(f"""
                        **Interpretation:**
                        A value of **{dcr:.2f}** means that for every -1% the market drops, 
                        your portfolio captures **-{dcr:.2f}%**.
                        """)

                    # 6. Cumulative Return Chart
                    st.markdown("#### Cumulative Growth (Log Scale)")
                    cum_ret = (1 + merged['Portfolio_Ret']).cumprod()
                    fig_growth = px.line(cum_ret, title="Portfolio Growth")
                    st.plotly_chart(fig_growth, use_container_width=True)

# --- Tab 3: Future Risk Lab ---
from src.risk_engine import RiskEngine

with tab3:
    st.header("🔮 Future Wealth Projection")
    st.markdown("""
    **"How might my investment grow?"**
    Project capital outcomes using regime-aware simulations.
    *Note: These are probabilistic scenarios, not guarantees.*
    """)
    
    if market_df is None:
        st.error("Market Data required.")
    else:
        mc_col1, mc_col2 = st.columns([1, 2])
        
        with mc_col1:
            st.subheader("1. Simulation Inputs")
            
            # Capital & Horizon
            initial_inv = st.number_input("Initial Investment ($)", min_value=1000, value=20000, step=1000, format="%d")
            sim_years = st.slider("Time Horizon (Years)", 5, 30, 20, step=10)
            
            st.markdown("---")
            st.markdown("**Regime Outlook (Next 5 Years)**")
            st.caption("Define the market 'Weather' probabilities.")
            
            # User Inputs for Regime Probabilities
            prob_calm = st.slider("Calm / Bull (Green) %", 0, 100, 50, key="p_calm")
            prob_choppy = st.slider("Choppy / Mixed (Yellow) %", 0, 100, 30, key="p_chop")
            prob_stress = st.slider("Stress / Bear (Red) %", 0, 100, 20, key="p_stress")
            
            total_prob = prob_calm + prob_choppy + prob_stress
            if total_prob != 100:
                st.warning(f"Total Probability: {total_prob}%. Please adjust to 100%.")
            
            sim_button = st.button("Run Projection")

        with mc_col2:
            if sim_button and total_prob == 100:
                # Need valid portfolio first
                if 'port' not in locals():
                     # Re-init if not in memory (simplified for now, ideally use session state)
                     weights = {t: st.session_state.get(f"w_{t}", 0)/100.0 for t in ALL_TICKERS if st.session_state.get(f"w_{t}", 0) > 0}
                     if not weights: 
                         st.error("Please build a portfolio in Tab 2 first.")
                     else:
                        port = Portfolio(weights)
                        start_date = market_df.index.min().strftime('%Y-%m-%d')
                        port.fetch_data(start_date=start_date)
                        port_rets = port.portfolio_returns
                        port_labels = market_df.loc[port_rets.index, 'Regime_Label']
                
                if 'port' in locals() and not port.portfolio_returns.empty:
                    with st.spinner(f"Simulating {sim_years} Years of Market Conditions..."):
                        # Run MC
                        regime_probs = {0: prob_calm/100, 1: prob_choppy/100, 2: prob_stress/100}
                        
                        paths = RiskEngine.monte_carlo_regime_aware(
                            port.portfolio_returns, 
                            port_labels, 
                            sim_years=sim_years, 
                            n_sims=500, 
                            regime_probs=regime_probs
                        )
                        
                        # Convert to Wealth
                        wealth_paths = paths * initial_inv
                        
                        # Get quantiles
                        final_vals = wealth_paths[-1, :]
                        p10 = np.percentile(final_vals, 10)
                        p50 = np.percentile(final_vals, 50)
                        p90 = np.percentile(final_vals, 90)
                        
                        # Stats Row
                        st.subheader(f"Projected Value in {sim_years} Years")
                        c1, c2, c3 = st.columns(3)
                        
                        # Formatting helpers
                        def fmt_curr(x): return f"${x:,.0f}"
                        gain_pct = lambda final: (final - initial_inv) / initial_inv
                        
                        c1.metric("Optimistic (90th)", fmt_curr(p90), f"+{gain_pct(p90):.0%}")
                        c2.metric("Median (50th)", fmt_curr(p50), f"+{gain_pct(p50):.0%}")
                        c3.metric("Conservative (10th)", fmt_curr(p10), f"+{gain_pct(p10):.0%}", delta_color="inverse")

                        # Plot
                        fig_mc = go.Figure()
                        x_axis = np.arange(wealth_paths.shape[0])
                        
                        # Plot grey lines (Subset)
                        for i in range(0, 500, 10): 
                            fig_mc.add_trace(go.Scatter(x=x_axis, y=wealth_paths[:, i], mode='lines', line=dict(color='grey', width=0.5), opacity=0.2, showlegend=False))
                            
                        # Plot Percentiles
                        # (We could plot specific paths that end near the percentiles, but mean is simpler for now)
                        mean_path = np.mean(wealth_paths, axis=1)
                        fig_mc.add_trace(go.Scatter(x=x_axis, y=mean_path, mode='lines', line=dict(color='blue', width=3), name="Average Path"))
                        
                        # Add initial investment line
                        fig_mc.add_hline(y=initial_inv, line_dash="dash", line_color="red", annotation_text="Principal")

                        fig_mc.update_layout(
                            title=f"Monte Carlo: {sim_years}-Year Outcomes ({total_prob}% Prob. Checked)", 
                            xaxis_title="Trading Days", 
                            yaxis_title="Portfolio Value ($)",
                            yaxis_tickformat="$,.0f"
                        )
                        st.plotly_chart(fig_mc, use_container_width=True)
                        
                        # Disclaimer
                        st.caption(f"""
                        **Methodology:** 500 simulations sampling from historical regimes based on your defined probabilities: 
                        (Calm: {prob_calm}%, Choppy: {prob_choppy}%, Stress: {prob_stress}%).
                        Past performance of the regime model does not guarantee future results.
                        """)

# --- Tab 4: Documentation ---
with tab4:
    st.header("📘 User Guide & Methodology")
    
    st.markdown("""
    ### 1. What is a "Market Regime"?
    Markets do not move in a straight line. They cycle through different "states" or "regimes" driven by investor psychology, economic data, and volatility.
    
    This dashboard uses a **Machine Learning algorithm (Gaussian Mixture Model)** to automatically classify history into three distinct regimes based on:
    *   **Trend**: Is the price moving up or down? (200-day Moving Average)
    *   **Fear**: How expensive is insurance? (VIX Index)
    *   **Credit Stress**: Are lenders nervous? (High Yield Spread)
    
    #### The Three Regimes
    *   **🟢 Stable Growth**: Low volatility, positive trends. The "Normal" state of the market.
    *   **🟡 Elevated Uncertainty**: Transition periods. Volatility spikes, trends break. Often precedes a recovery OR a crash.
    *   **🔴 High Stress**: Bear markets, recessions, or liquidity shocks. Correlations go to 1.0 (everything falls together).
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 2. Feature Overview
    
    #### 🌪️ Tab 1: Market Stress Monitor
    *   **Live Weather Report**: Shows the current regime today.
    *   **Indicators**: Tracks the input signals (VIX, Trend, Yield Curve) so you can see *why* the model is worried (or calm).
    
    #### 💼 Tab 2: Context-Aware Portfolio
    *   **Interactive Builder**: Build your own ETF portfolio (e.g., 60% Stocks, 40% Bonds).
    *   **Regime Breakdown**: Instead of just "Average Return" (which hides risks), we show you exactly how your portfolio performs *during* Stress.
    *   **Fragility Gauge**: A quick check on your downside. A value < 1.0 means you are safer than the market.
    
    #### 🔮 Tab 3: Future Wealth Projection
    *   **Regime-Weighted Monte Carlo**: Standard projections assume the future is random. We let you ask: *"What if the next 10 years are 50% High Stress?"*
    *   **Survival Cones**: We show the 10th percentile outcome (Pessimistic) so you can plan for the worst case, not just the average.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 3. FAQ
    *   **Is this financial advice?**
        *   No. This is a data visualization tool. It helps you understand historical patterns, but the future is never guaranteed.
    *   **Why use "Regimes" instead of just Returns?**
        *   Averages lie. A portfolio might make +8% per year *on average*, but if it drops -50% during a Stress Regime, you might panic sell. This tool highlights that specific risk.
    """)
