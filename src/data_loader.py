import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

class MarketDataLoader:
    """
    Fetches and aligns market data for the Stress Detection Dashboard.
    """
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Tickers for MVP
        # SPY: Market Proxy
        # ^VIX: Volatility
        # HYG: High Yield Bond (Credit Risk)
        # IEF: 7-10 Year Treasury (Safe Asset / Duration)
        # ^TNX: 10 Year Treasury Yield
        # ^IRX: 13 Week Treasury Bill Yield (Short End)
        self.tickers = ['SPY', '^VIX', 'HYG', 'IEF', '^TNX', '^IRX']
        self.start_date = "2005-01-01" # Capture 2008 crash

    def fetch_data(self):
        """Downloads data from Yahoo Finance."""
        print(f"Fetching data for: {self.tickers}...")
        
        # Download all at once
        data = yf.download(
            self.tickers, 
            start=self.start_date, 
            end=datetime.today().strftime('%Y-%m-%d'),
            group_by='ticker',
            auto_adjust=True,
            threads=True
        )
        
        processed_data = pd.DataFrame()
        
        # Extract Close prices for each ticker
        # Note: yfname structure can be multi-index. 
        # Structure: (Price, Ticker) or (Ticker, Price) depending on version.
        # We will iterate safely.
        
        for ticker in self.tickers:
            try:
                # Handle multi-index columns if necessary
                if isinstance(data.columns, pd.MultiIndex):
                     # Try to get Close, if not available, use 'Adj Close' (auto_adjust=True usually returns Close as Adjusted)
                    if (ticker, 'Close') in data.columns:
                        series = data[(ticker, 'Close')]
                    elif 'Close' in data[ticker].columns:
                        series = data[ticker]['Close']
                    else:
                        print(f"Warning: No 'Close' found for {ticker}")
                        continue
                else:
                    # Single level (unlikely with multiple tickers but possible)
                    series = data['Close'] # risky if multiple
                
                processed_data[ticker] = series
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        # Basic Cleaning
        processed_data = processed_data.ffill() # Fill forward for holidays
        processed_data = processed_data.dropna() # Drop initial NaNs
        
        output_path = os.path.join(self.data_dir, "market_data.parquet")
        processed_data.to_parquet(output_path)
        print(f"Data saved to {output_path}. Shape: {processed_data.shape}")
        return processed_data

if __name__ == "__main__":
    loader = MarketDataLoader()
    df = loader.fetch_data()
    print(df.tail())
