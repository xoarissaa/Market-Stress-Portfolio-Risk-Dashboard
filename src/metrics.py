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
