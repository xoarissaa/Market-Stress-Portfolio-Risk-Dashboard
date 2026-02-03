import numpy as np
import pandas as pd

class RiskEngine:
    """
    Advanced Risk Analytics for Context-Aware Portfolio Analysis.
    Includes:
    1. Regime-Weighted Monte Carlo Simulation
    2. Rolling Optimization / Hedge Monitoring
    """
    
    @staticmethod
    def monte_carlo_regime_aware(
        returns: pd.Series, 
        regime_labels: pd.Series, 
        sim_years: int = 5, 
        n_sims: int = 1000, 
        regime_probs: dict = None
    ):
        """
        Runs a Monte Carlo simulation where samples are drawn based on specific regime probabilities.
        
        :param returns: Daily portfolio returns (Series).
        :param regime_labels: Series of regime labels (0=Calm, 1=Choppy, 2=Stress) aligned with returns.
        :param regime_probs: Dict of target probabilities for future state. 
                             e.g., {0: 0.5, 1: 0.3, 2: 0.2}. 
                             If None, uses historical frequency.
        """
        # Align data
        data = pd.DataFrame({'ret': returns, 'regime': regime_labels}).dropna()
        
        # Separate pools for each regime
        pools = {
            r: data[data['regime'] == r]['ret'].values 
            for r in data['regime'].unique()
        }
        
        # Default probabilities if None (Historical Frequency)
        if regime_probs is None:
            counts = data['regime'].value_counts(normalize=True)
            regime_probs = counts.to_dict()
            
        # Normalize probs
        total_p = sum(regime_probs.values())
        choice_probs = [regime_probs.get(r, 0)/total_p for r in pools.keys()]
        pool_keys = list(pools.keys())
        
        sim_days = int(sim_years * 252)
        final_paths = np.zeros((sim_days, n_sims))
        
        for i in range(n_sims):
            # 1. Determine Regime Sequence for this path
            # (Simplified: Randomly pick a regime for each day based on weights)
            # A better approach would be Markov Chain, but independent sampling is "Robust" enough for stress testing.
            daily_regimes = np.random.choice(pool_keys, size=sim_days, p=choice_probs)
            
            # 2. Sample returns from chosen regime pools
            path_returns = np.zeros(sim_days)
            for r in pool_keys:
                mask = (daily_regimes == r)
                n_draws = np.sum(mask)
                if n_draws > 0:
                    if len(pools[r]) > 0:
                         draws = np.random.choice(pools[r], size=n_draws, replace=True)
                         path_returns[mask] = draws
                    else:
                        # Fallback if a regime has no history
                        path_returns[mask] = 0.0 
            
            # 3. Construct Price Path (Cumprod)
            final_paths[:, i] = (1 + path_returns).cumprod()
            
        return final_paths

    @staticmethod
    def calculate_downside_capture(port_returns, bench_returns, window=None):
        """
        Calculates Downside Capture Ratio vs Benchmark.
        DCR = (Portfolio Downside CAGR / Benchmark Downside CAGR) * 100
        Simplified: Average return on days when Benchmark < 0.
        """
        # Align
        df = pd.DataFrame({'port': port_returns, 'bench': bench_returns}).dropna()
        
        if window:
            # Rolling (Not implemented for MVP, returning static)
            pass
            
        # Filter for days when Benchmark is negative
        down_market = df[df['bench'] < 0]
        if down_market.empty:
            return 0.0
            
        avg_port_down = down_market['port'].mean()
        avg_bench_down = down_market['bench'].mean()
        
        if avg_bench_down == 0: return 0.0
        
        # Capture Ratio
        capture = (avg_port_down / avg_bench_down)
        return capture

    @staticmethod
    def rolling_correlation(series_a, series_b, window=60):
        """
        Computes rolling correlation between two series. 
        Useful for 'Hedge Monitoring'.
        """
        return series_a.rolling(window=window).corr(series_b)
