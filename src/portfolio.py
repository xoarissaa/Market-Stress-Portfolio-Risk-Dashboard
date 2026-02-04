import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from src.metrics import PerformanceMetrics

class Portfolio:
    """
    Manages a collection of assets capabilities to fetch data and calculate performance.
    """
    def __init__(self, positions, initial_capital=10000):
        """
        :param positions: Dictionary {ticker: weight}, e.g., {'SPY': 0.6, 'TLT': 0.4}
                         Weights can be percentages (60, 40) or decimals (0.6, 0.4)
        :param initial_capital: Starting value for equity curve (default 10k)
        """
        self.validate_portfolio(positions)
        self.positions = self.normalize_weights(positions)
        self.tickers = list(self.positions.keys())
        self.weights = pd.Series(self.positions)
        self.initial_capital = initial_capital
        self.data = None
        self.returns = None
        self.portfolio_returns = None
    
    def validate_portfolio(self, positions):
        """
        Validates portfolio configuration.
        Raises ValueError if portfolio is invalid.
        """
        if not positions:
            raise ValueError("Portfolio cannot be empty. Please provide at least one ticker.")
        
        if not isinstance(positions, dict):
            raise TypeError("Positions must be a dictionary of {ticker: weight}")
        
        for ticker, weight in positions.items():
            if not isinstance(ticker, str) or not ticker.strip():
                raise ValueError(f"Invalid ticker: {ticker}")
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"Weight for {ticker} must be a non-negative number, got {weight}")
    
    def normalize_weights(self, positions):
        """
        Normalizes weights to sum to 1.0.
        Handles both percentage (60, 40) and decimal (0.6, 0.4) inputs.
        
        :param positions: Dictionary {ticker: weight}
        :return: Normalized dictionary with weights summing to 1.0
        """
        total = sum(positions.values())
        
        if total == 0:
            raise ValueError("Total weight cannot be zero")
        
        # Normalize
        normalized = {ticker: weight / total for ticker, weight in positions.items()}
        
        # Warn if original weights were far from 1.0 or 100
        if not (0.99 <= total <= 1.01 or 99 <= total <= 101):
            print(f"⚠️  Weights normalized: {total:.2f} → 1.0")
        
        return normalized

    def fetch_data(self, start_date="2010-01-01", end_date=None):
        """
        Fetches adjusted close prices for portfolio assets.
        """
        print(f"Fetching portfolio data for: {self.tickers}")
        df = yf.download(
            self.tickers, 
            start=start_date, 
            end=end_date, 
            group_by='ticker', 
            auto_adjust=True,
            threads=True
        )
        
        prices = pd.DataFrame()
        for t in self.tickers:
            # Handle Single Ticker case vs Multi Ticker result structure
            try:
                if len(self.tickers) == 1:
                    series = df['Close']
                elif isinstance(df.columns, pd.MultiIndex):
                    series = df[t]['Close']
                else:
                    series = df[t] # Fallback
                prices[t] = series
            except Exception as e:
                print(f"Error extracting {t}: {e}")
                
        self.data = prices.dropna()
        self.calculate_returns()
        return self.data

    def calculate_returns(self):
        """
        Computes daily returns and weighted portfolio returns.
        Assumes daily rebalancing (simplified) or static weights for now.
        """
        if self.data is None:
            raise ValueError("Data not fetched. Call fetch_data() first.")
            
        self.returns = self.data.pct_change().dropna()
        
        # Weighted Return = w1*r1 + w2*r2 ...
        # Align weights to columns
        weighted_rets = self.returns.mul(self.weights, axis=1)
        self.portfolio_returns = weighted_rets.sum(axis=1)
        
    def get_performance_summary(self, start_date=None, end_date=None):
        """
        Returns a dictionary of metrics for the portfolio in a specific date range.
        Useful for 'Regime Analysis'.
        
        :param start_date: Start date (str 'YYYY-MM-DD', datetime, or pd.Timestamp)
        :param end_date: End date (str 'YYYY-MM-DD', datetime, or pd.Timestamp)
        :return: Dictionary of performance metrics
        """
        if self.portfolio_returns is None:
            return {}
        
        # Convert dates to pandas Timestamp for robust comparison
        if start_date is not None:
            start_date = pd.Timestamp(start_date)
        if end_date is not None:
            end_date = pd.Timestamp(end_date)
            
        # Filter range
        f_rets = self.portfolio_returns
        if start_date:
            f_rets = f_rets[f_rets.index >= start_date]
        if end_date:
            f_rets = f_rets[f_rets.index <= end_date]
            
        if len(f_rets) == 0:
            return {"Error": "No data in range"}

        metrics = {
            "Total Return": PerformanceMetrics.total_return(f_rets),
            "CAGR": PerformanceMetrics.cagr(f_rets),
            "Volatility": PerformanceMetrics.volatility(f_rets),
            "Sharpe Ratio": PerformanceMetrics.sharpe_ratio(f_rets),
            "Max Drawdown": PerformanceMetrics.max_drawdown(f_rets),
            "Sortino Ratio": PerformanceMetrics.sortino_ratio(f_rets)
        }
        return metrics
    
    def get_regime_conditional_metrics(self, regime_series, regime_code):
        """
        Calculate performance metrics for periods matching a specific regime.
        
        :param regime_series: pd.Series with regime labels, indexed by date
        :param regime_code: The regime code to filter (e.g., 0, 1, 2)
        :return: Dictionary of metrics for the specified regime periods
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated. Call fetch_data() first.")
        
        # Align regime data with portfolio returns
        aligned = pd.DataFrame({
            'returns': self.portfolio_returns,
            'regime': regime_series
        }).dropna()
        
        # Filter by regime
        regime_returns = aligned[aligned['regime'] == regime_code]['returns']
        
        if len(regime_returns) == 0:
            return {"Error": f"No data for regime {regime_code}"}
        
        # Calculate metrics for this regime only
        metrics = {
            "Total Return": PerformanceMetrics.total_return(regime_returns),
            "CAGR": PerformanceMetrics.cagr(regime_returns),
            "Volatility": PerformanceMetrics.volatility(regime_returns),
            "Sharpe Ratio": PerformanceMetrics.sharpe_ratio(regime_returns),
            "Max Drawdown": PerformanceMetrics.max_drawdown(regime_returns),
            "Sortino Ratio": PerformanceMetrics.sortino_ratio(regime_returns),
            "Days in Regime": len(regime_returns)
        }
        return metrics

if __name__ == "__main__":
    # Test
    port = Portfolio({'SPY': 0.6, 'TLT': 0.4})
    port.fetch_data(start_date="2020-01-01")
    stats = port.get_performance_summary()
    print("Portfolio Stats (2020-Now):")
    for k, v in stats.items():
        print(f"{k}: {v:.2%}" if "Ratio" not in k else f"{k}: {v:.2f}")
