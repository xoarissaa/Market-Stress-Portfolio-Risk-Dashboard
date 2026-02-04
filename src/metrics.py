import numpy as np
import pandas as pd

class PerformanceMetrics:
    """
    Standard Logic for Portfolio KPIs.
    Assumes input is a pandas Series of daily returns.
    """
    
    @staticmethod
    def total_return(returns):
        """Calculates total cumulative return."""
        return (1 + returns).prod() - 1

    @staticmethod
    def cagr(returns, periods_per_year=252):
        """Compound Annual Growth Rate."""
        n_years = len(returns) / periods_per_year
        if n_years == 0: return 0
        cum_ret = (1 + returns).prod()
        return (cum_ret ** (1 / n_years)) - 1

    @staticmethod
    def volatility(returns, periods_per_year=252):
        """Annualized Volatility (Standard Deviation)."""
        return returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
        """Annualized Sharpe Ratio."""
        excess_ret = returns - risk_free_rate/periods_per_year
        if excess_ret.std() == 0: return 0
        return (excess_ret.mean() / excess_ret.std()) * np.sqrt(periods_per_year)

    @staticmethod
    def sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
        """Annualized Sortino Ratio (Downside Deviation only)."""
        excess_ret = returns - risk_free_rate/periods_per_year
        downside_ret = excess_ret[excess_ret < 0]
        if len(downside_ret) == 0 or downside_ret.std() == 0: return 0
        
        downside_dev = downside_ret.std() * np.sqrt(periods_per_year)
        return (excess_ret.mean() * periods_per_year) / downside_dev

    @staticmethod
    def max_drawdown(returns):
        """Maximum Drawdown from Peak."""
        wealth_index = (1 + returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns.min()

    @staticmethod
    def calmar_ratio(returns, periods_per_year=252):
        """CAGR / Max Drawdown."""
        cagr_val = PerformanceMetrics.cagr(returns, periods_per_year)
        mdd = abs(PerformanceMetrics.max_drawdown(returns))
        if mdd == 0: return 0
        return cagr_val / mdd

    @staticmethod
    def downside_capture_ratio(portfolio_returns, benchmark_returns):
        """
        Calculates Downside Capture Ratio vs Benchmark.
        Measures how much of the benchmark's downside the portfolio captures.
        
        DCR < 1.0 means the portfolio falls less than the benchmark during down markets (good)
        DCR > 1.0 means the portfolio falls more than the benchmark during down markets (bad)
        
        :param portfolio_returns: Portfolio daily returns (pd.Series)
        :param benchmark_returns: Benchmark daily returns (pd.Series)
        :return: Downside capture ratio
        """
        # Align the two series
        df = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        # Filter for days when benchmark is negative
        down_market = df[df['benchmark'] < 0]
        
        if down_market.empty or len(down_market) < 2:
            return 0.0
        
        # Average returns during down markets
        avg_portfolio_down = down_market['portfolio'].mean()
        avg_benchmark_down = down_market['benchmark'].mean()
        
        if avg_benchmark_down == 0:
            return 0.0
        
        # Capture ratio (both are negative, so ratio is positive)
        capture = avg_portfolio_down / avg_benchmark_down
        return capture

