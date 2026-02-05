import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from src.market_regime import MarketRegimeDetector
from src.portfolio import Portfolio
from src.risk_engine import RiskEngine
from datetime import datetime

# Set Page Config
st.set_page_config(page_title="Market Stress & Portfolio Monitor", page_icon="📉", layout="wide")

# Title
st.title("📉 Market Stress & Portfolio Monitor")
st.caption("**System Status**: Unified Dashboard Active. *Monitor market 'weather' and check how your portfolio ships sail in these conditions.*")

# --- Expanded Ticker Universe ---
# US Equity ETFs
US_EQUITY = ["SPY", "VTI", "QQQ", "IWM", "VTV", "VUG", "VOO", "SCHD", "SPLV", "QUAL", "MOAT", "QUS", "VIG", "DGRO"]

# US Bond ETFs
US_BONDS = ["BND", "AGG", "TLT", "IEF", "SHY", "LQD", "HYG", "MUB", "TIP"]

# International ETFs
INTERNATIONAL = ["VEA", "VWO", "IEFA", "IEMG", "EFA", "EEM", "VXUS"]

# Sector ETFs
SECTORS = ["XLK", "XLV", "XLF", "XLE", "XLY", "XLP", "XLI", "XLU", "XLRE", "XLC", "XLB"]

# Commodity & Alternative ETFs
COMMODITIES = ["GLD", "IAU", "SLV", "DBC", "USO", "UNG"]

# Canadian ETFs
CAD_ETFS = ["XIC.TO", "VCN.TO", "VDY.TO", "ZAG.TO", "VFV.TO", "ZLB.TO", "XEF.TO", "XEC.TO"]

# Combine all
ALL_TICKERS = sorted(list(set(US_EQUITY + US_BONDS + INTERNATIONAL + SECTORS + COMMODITIES + CAD_ETFS)))

# --- Load Market Data ---
@st.cache_data
def load_market_data():
    try:
        df = pd.read_parquet("data/processed_market_regimes.parquet")
        return df
    except FileNotFoundError:
        return None

market_df = load_market_data()

# --- Sidebar: Data Management ---
with st.sidebar:
    st.header("⚙️ Data Management")
    if st.button("🔄 Refresh Market Data", help="Fetch latest data from yfinance and re-train regime model"):
        with st.spinner("Fetching latest data from yfinance..."):
            from src.data_loader import MarketDataLoader
            from src.market_regime import MarketRegimeDetector
            
            try:
                # 1. Fetch
                loader = MarketDataLoader()
                loader.fetch_data()
                
                # 2. Detect Regimes
                detector = MarketRegimeDetector()
                feat_df = detector.load_and_engineer_features()
                detector.train_model(feat_df)
                
                st.cache_data.clear()
                st.success("Data refreshed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to refresh data: {e}")
    
    st.divider()
    st.caption("Last data point: " + (market_df.index.max().strftime('%Y-%m-%d') if market_df is not None else "None found"))


# --- Load Definitions ---
from src.definitions import RegimeDefinitions
from src.comparison_engine import ComparisonEngine, format_comparison_table
from src.regime_context import RegimeContext

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🌪️ Market Stress Monitor", "⚖️ Compare Portfolios / ETFs", "🎲 Monte Carlo Simulation", "📘 User Guide"])

with tab1:
    if market_df is None:
        st.error("Market Data not found. Please click 'Refresh Market Data' in the sidebar to download data from yfinance.")
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
    st.header("⚖️ Compare Portfolios / ETFs")
    st.caption("Compare two portfolios side-by-side with regime-conditional analysis. Individual ETFs are treated as 100% portfolios.")
    
    if market_df is None:
        st.error("Market Data not found. Run pipeline first.")
    else:
        # --- Side-by-Side Portfolio Builders ---
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("📊 Portfolio A")
            
            # Quick presets
            preset_a = st.selectbox(
                "Quick Preset A",
                ["Custom", "SPY (100%)", "60/40 SPY/TLT", "All Weather", "QQQ (100%)"],
                key="preset_a"
            )
            
            if preset_a == "SPY (100%)":
                selected_a = ["SPY"]
                weights_a = {"SPY": 100}
            elif preset_a == "60/40 SPY/TLT":
                selected_a = ["SPY", "TLT"]
                weights_a = {"SPY": 60, "TLT": 40}
            elif preset_a == "All Weather":
                selected_a = ["SPY", "TLT", "IEF", "GLD", "DBC"]
                weights_a = {"SPY": 30, "TLT": 40, "IEF": 15, "GLD": 7.5, "DBC": 7.5}
            elif preset_a == "QQQ (100%)":
                selected_a = ["QQQ"]
                weights_a = {"QQQ": 100}
            else:
                selected_a = st.multiselect("Select Assets A", ALL_TICKERS, default=["SPY"], key="assets_a")
                weights_a = {}
                if selected_a:
                    st.write("Assign Weights:")
                    for t in selected_a:
                        w = st.slider(f"{t} %", 0, 100, int(100/len(selected_a)), key=f"w_a_{t}")
                        weights_a[t] = w
            
            total_a = sum(weights_a.values()) if weights_a else 0
            if total_a != 100 and weights_a:
                st.warning(f"Total: {total_a}% (will auto-normalize)")
            else:
                st.success(f"✅ Total: {total_a}%")
        
        with col_b:
            st.subheader("📊 Portfolio B")
            
            # Quick presets
            preset_b = st.selectbox(
                "Quick Preset B",
                ["Custom", "SPY (100%)", "60/40 SPY/TLT", "All Weather", "QQQ (100%)"],
                key="preset_b"
            )
            
            if preset_b == "SPY (100%)":
                selected_b = ["SPY"]
                weights_b = {"SPY": 100}
            elif preset_b == "60/40 SPY/TLT":
                selected_b = ["SPY", "TLT"]
                weights_b = {"SPY": 60, "TLT": 40}
            elif preset_b == "All Weather":
                selected_b = ["SPY", "TLT", "IEF", "GLD", "DBC"]
                weights_b = {"SPY": 30, "TLT": 40, "IEF": 15, "GLD": 7.5, "DBC": 7.5}
            elif preset_b == "QQQ (100%)":
                selected_b = ["QQQ"]
                weights_b = {"QQQ": 100}
            else:
                selected_b = st.multiselect("Select Assets B", ALL_TICKERS, default=["TLT"], key="assets_b")
                weights_b = {}
                if selected_b:
                    st.write("Assign Weights:")
                    for t in selected_b:
                        w = st.slider(f"{t} %", 0, 100, int(100/len(selected_b)), key=f"w_b_{t}")
                        weights_b[t] = w
            
            total_b = sum(weights_b.values()) if weights_b else 0
            if total_b != 100 and weights_b:
                st.warning(f"Total: {total_b}% (will auto-normalize)")
            else:
                st.success(f"✅ Total: {total_b}%")
        
        # --- Compare Button ---
        st.markdown("---")
        compare_btn = st.button("🔍 Compare Portfolios", type="primary", use_container_width=True)
        
        if compare_btn and weights_a and weights_b:
            with st.spinner("Fetching data and analyzing..."):
                try:
                    # Create portfolios
                    port_a = Portfolio(weights_a)
                    port_b = Portfolio(weights_b)
                    
                    # Fetch data
                    start_date = market_df.index.min().strftime('%Y-%m-%d')
                    
                    try:
                        port_a.fetch_data(start_date=start_date)
                    except ValueError as e:
                        st.error(f"Failed to fetch data for Portfolio A: {e}")
                        st.stop()
                    
                    try:
                        port_b.fetch_data(start_date=start_date)
                    except ValueError as e:
                        st.error(f"Failed to fetch data for Portfolio B: {e}")
                        st.stop()
                    
                    # Store in session state for other tabs (like Monte Carlo)
                    st.session_state.port_a = port_a
                    st.session_state.port_b = port_b

                    
                    # Debug: Show date ranges
                    with st.expander("🛠️ Debug Data Info", expanded=False):
                        st.write(f"- Portfolio A dates: {port_a.portfolio_returns.index.min()} to {port_a.portfolio_returns.index.max()} ({len(port_a.portfolio_returns)} days)")
                        st.write(f"- Portfolio B dates: {port_b.portfolio_returns.index.min()} to {port_b.portfolio_returns.index.max()} ({len(port_b.portfolio_returns)} days)")
                        st.write(f"- Market data dates: {market_df.index.min()} to {market_df.index.max()} ({len(market_df)} days)")
                    
                    # Get regime labels - try RegimeContext first, fallback to market_df
                    try:
                        regime_ctx = RegimeContext()
                        regime_ctx.load_regime_data()
                        regime_labels = regime_ctx.get_regime_labels()
                    except (FileNotFoundError, Exception) as e:
                        st.warning(f"Using regime labels from market data (RegimeContext unavailable: {e})")
                        regime_labels = market_df['Regime_Label']
                    
                    # Get SPY benchmark
                    spy_prices = market_df['SPY']
                    spy_rets = spy_prices.pct_change().dropna()
                    
                    # Create comparison engine
                    engine = ComparisonEngine(port_a, port_b, regime_labels, spy_rets)
                    
                    # --- Overall Comparison ---
                    st.markdown("---")
                    st.subheader("📊 Overall Comparison")
                    
                    overall = engine.compare_overall_metrics()
                    overall_df = format_comparison_table(overall)
                    
                    # Format the dataframe for display
                    def highlight_winner(row):
                        if row['Winner'] == 'A':
                            return ['background-color: #d4edda'] * 2 + [''] * 2
                        elif row['Winner'] == 'B':
                            return [''] * 2 + ['background-color: #d4edda'] * 2
                        else:
                            return [''] * 4
                    
                    # Display with formatting
                    st.dataframe(
                        overall_df.style.apply(highlight_winner, axis=1),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # --- Regime-Conditional Comparison ---
                    st.markdown("---")
                    st.subheader("🌍 Performance by Market Regime")
                    
                    regime_tabs = st.tabs(["🟢 Stable Growth", "🟡 Elevated Uncertainty", "🔴 High Stress"])
                    
                    for idx, regime_code in enumerate([0, 1, 2]):
                        with regime_tabs[idx]:
                            regime_comp = engine.compare_by_regime(regime_code)
                            regime_df = format_comparison_table(regime_comp)
                            
                            st.markdown(f"**{regime_comp['regime_name']}**")
                            st.dataframe(
                                regime_df.style.apply(highlight_winner, axis=1),
                                use_container_width=True,
                                hide_index=True
                            )
                    
                    # --- Decision Summary ---
                    st.markdown("---")
                    st.subheader("💡 Decision Summary")
                    
                    summary = engine.generate_decision_summary()
                    st.info(summary)
                    
                    # --- Cumulative Performance Chart ---
                    st.markdown("---")
                    st.subheader("📈 Cumulative Performance Comparison")
                    
                    # Calculate cumulative returns
                    cum_a = (1 + port_a.portfolio_returns).cumprod()
                    cum_b = (1 + port_b.portfolio_returns).cumprod()
                    
                    # Align dates
                    comparison_df = pd.DataFrame({
                        'Portfolio A': cum_a,
                        'Portfolio B': cum_b
                    }).dropna()
                    
                    fig = px.line(
                        comparison_df,
                        title="Cumulative Growth Comparison",
                        labels={'value': 'Growth ($1 invested)', 'index': 'Date'}
                    )
                    fig.update_layout(hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during comparison: {str(e)}")
                    st.exception(e)


# --- Tab 3: Monte Carlo Simulation ---
with tab3:
    st.header("🎲 Monte Carlo Simulation")
    st.caption("Analytical projection of portfolio outcomes based on historical regime distributions and future outlook.")

    if market_df is None:
        st.error("Market Data not found. Please refresh data in sidebar.")
    else:
        mc_col1, mc_col2 = st.columns([1, 2])
        
        with mc_col1:
            st.subheader("Simulation Parameters")
            
            # 1. Portfolio Selection
            selected_sim_port = st.radio(
                "Select Portfolio Source",
                ["Portfolio A", "Portfolio B"],
                help="Start with the assets and weights defined in the Comparison tab."
            )
            
            # --- Clone & Edit Logic ---
            source_port = st.session_state.get('port_a' if selected_sim_port == "Portfolio A" else 'port_b')
            
            do_clone = st.checkbox("🛠️ Clone & Edit Scenario", help="Modify weights locally for this simulation only. Does not affect the comparison tab.")
            local_weights = {}
            
            if do_clone:
                if source_port:
                    st.info("💡 Adjust weights below. Total must equal 100%.")
                    for ticker, weight in source_port.weights.items():
                        new_w = st.number_input(f"{ticker} Weight (%)", min_value=0, max_value=100, value=int(weight*100), key=f"mc_edit_{ticker}")
                        local_weights[ticker] = new_w / 100.0
                    
                    total_w = sum(local_weights.values())
                    if abs(total_w - 1.0) > 0.001:
                        st.warning(f"Scenario Total: {total_w*100:.1f}%. Please adjust to 100%.")
                else:
                    st.warning(f"Define {selected_sim_port} in Comparison tab first.")
            
            # 2. Capital & Horizon
            initial_inv = st.number_input("Initial Investment ($)", min_value=1000, value=10000, step=1000, format="%d")
            sim_years = st.slider("Time Horizon (Years)", 1, 30, 10)
            
            st.markdown("---")
            st.subheader("Market Regime Outlook")
            st.caption("Define the probability of each market 'weather' state.")
            
            # User Inputs for Regime Probabilities
            prob_calm = st.slider("Calm / Bull %", 0, 100, 50, key="mc_p_calm")
            prob_choppy = st.slider("Choppy / Mixed %", 0, 100, 30, key="mc_p_chop")
            prob_stress = st.slider("High Stress / Bear %", 0, 100, 20, key="mc_p_stress")
            
            total_prob = prob_calm + prob_choppy + prob_stress
            if total_prob != 100:
                st.warning(f"Totals must equal 100% (Current: {total_prob}%)")
            
            sim_button = st.button("Run Monte Carlo Simulation", use_container_width=True, type="primary")

        with mc_col2:
            if sim_button and total_prob == 100:
                target_port = None
                try:
                    if do_clone and local_weights:
                        if abs(sum(local_weights.values()) - 1.0) < 0.001:
                            # Create temporary local portfolio
                            target_port = Portfolio(local_weights)
                            start_date = market_df.index.min().strftime('%Y-%m-%d')
                            with st.spinner("Fetching data for local scenario..."):
                                target_port.fetch_data(start_date=start_date)
                        else:
                            st.error("Scenario weights must total 100%.")
                    else:
                        target_port = source_port
                            
                    if target_port is None:
                        st.error(f"Please define {selected_sim_port} in the 'Compare Portfolios' tab first.")
                    elif target_port.portfolio_returns.empty:
                        st.error(f"No data for selection. Please ensure data fetch was successful.")
                    else:

                        with st.spinner(f"Simulating {selected_sim_port} over {sim_years} years..."):
                            # Run MC
                            regime_probs = {0: prob_calm/100, 1: prob_choppy/100, 2: prob_stress/100}
                            port_rets = target_port.portfolio_returns
                            port_labels = market_df.loc[port_rets.index, 'Regime_Label']
                            
                            n_sims = 1000
                            paths = RiskEngine.monte_carlo_regime_aware(
                                port_rets, 
                                port_labels, 
                                sim_years=sim_years, 
                                n_sims=n_sims, 
                                regime_probs=regime_probs
                            )
                            
                            wealth_paths = paths * initial_inv
                            
                            # Analytics
                            final_vals = wealth_paths[-1, :]
                            p10 = np.percentile(final_vals, 10)
                            p50 = np.percentile(final_vals, 50)
                            p90 = np.percentile(final_vals, 90)
                            
                            st.subheader(f"Outcomes for {selected_sim_port}")
                            c1, c2, c3 = st.columns(3)
                            
                            gain = lambda x: (x - initial_inv) / initial_inv
                            
                            c1.metric("Optimistic (90th)", f"${p90:,.0f}", f"{gain(p90):+.1%}")
                            c2.metric("Median (50th)", f"${p50:,.0f}", f"{gain(p50):+.1%}")
                            c3.metric("Conservative (10th)", f"${p10:,.0f}", f"{gain(p10):+.1%}", delta_color="inverse")

                            # Plotting with Shaded Areas (Survival Cones)
                            fig_mc = go.Figure()
                            x_axis = np.arange(wealth_paths.shape[0])
                            
                            # Percentiles for shading
                            p10_path = np.percentile(wealth_paths, 10, axis=1)
                            p50_path = np.percentile(wealth_paths, 50, axis=1)
                            p90_path = np.percentile(wealth_paths, 90, axis=1)
                            
                            # Add Shading (10th to 90th)
                            fig_mc.add_trace(go.Scatter(
                                x=np.concatenate([x_axis, x_axis[::-1]]),
                                y=np.concatenate([p90_path, p10_path[::-1]]),
                                fill='toself',
                                fillcolor='rgba(0,100,80,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                hoverinfo="skip",
                                name="10th-90th Percentile Range"
                            ))
                            
                            # Add Individual Paths (Lightly)
                            for i in range(0, n_sims, 50):
                                fig_mc.add_trace(go.Scatter(
                                    x=x_axis, y=wealth_paths[:, i], 
                                    mode='lines', line=dict(color='grey', width=0.5), 
                                    opacity=0.1, showlegend=False
                                ))
                            
                            # Highlight Median
                            fig_mc.add_trace(go.Scatter(
                                x=x_axis, y=p50_path, 
                                mode='lines', line=dict(color='blue', width=2), 
                                name="Median (50th)"
                            ))
                            
                            fig_mc.add_hline(y=initial_inv, line_dash="dash", line_color="red", name="Principal")
                            
                            fig_mc.update_layout(
                                title=f"Monte Carlo Simulation: {selected_sim_port}",
                                xaxis_title="Trading Days",
                                yaxis_title="Wealth ($)",
                                template="plotly_white",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig_mc, use_container_width=True)
                            
                            st.info(f"""
                            **Simulation Context:** This model draws from historical returns of {selected_sim_port} 
                            specifically as they occurred during Calm, Choppy, and Stress regimes. It then projects {sim_years} years 
                            based on your weighted 'weather' outlook.
                            """)
                            
                except Exception as e:
                    st.error(f"Simulation Error: {e}")
            else:
                st.info("💡 Adjust parameters and click 'Run Monte Carlo Simulation' to view projections.")

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
    
    #### 🎲 Tab 3: Monte Carlo Simulation
    *   **Regime-Weighted Simulation**: Unlike standard tools that assume the future is random, we allow you to stress-test your specific portfolio against custom scenarios (e.g., *"What if the next decade has 40% High Stress?"*).
    *   **Percentile Bands**: We visualize the 10th (pessimistic), 50th (median), and 90th (optimistic) percentile paths to show the range of probable outcomes.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 3. FAQ
    *   **Is this financial advice?**
        *   No. This is a data visualization tool. It helps you understand historical patterns, but the future is never guaranteed.
    *   **Why use "Regimes" instead of just Returns?**
        *   Averages lie. A portfolio might make +8% per year *on average*, but if it drops -50% during a Stress Regime, you might panic sell. This tool highlights that specific risk.
    """)
