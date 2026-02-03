import pandas as pd
import yfinance as yf
from src.metrics import PerformanceMetrics

class Portfolio:
    """
    Manages a collection of assets capabilities to fetch data and calculate performance.
    """
    def __init__(self, positions, initial_capital=10000):
        """
        :param positions: Dictionary {ticker: weight}, e.g., {'SPY': 0.6, 'TLT': 0.4}
        :param initial_capital: Starting value for equity curve (default 10k)
        """
        self.positions = positions
        self.tickers = list(positions.keys())
        self.weights = pd.Series(positions)
        self.initial_capital = initial_capital
        self.data = None
        self.returns = None
        self.portfolio_returns = None

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
        """
        if self.portfolio_returns is None:
            return {}
            
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

if __name__ == "__main__":
    # Test
    port = Portfolio({'SPY': 0.6, 'TLT': 0.4})
    port.fetch_data(start_date="2020-01-01")
    stats = port.get_performance_summary()
    print("Portfolio Stats (2020-Now):")
    for k, v in stats.items():
        print(f"{k}: {v:.2%}" if "Ratio" not in k else f"{k}: {v:.2f}")
