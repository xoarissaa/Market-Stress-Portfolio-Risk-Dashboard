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
        print(f"Fetching market data for: {self.tickers}...")
        
        # Download all at once
        try:
            data = yf.download(
                self.tickers, 
                start=self.start_date, 
                end=datetime.today().strftime('%Y-%m-%d'),
                group_by='ticker',
                auto_adjust=True,
                threads=False, # Reliability
                progress=False
            )
        except Exception as e:
            print(f"Batch fetch failed: {e}")
            data = pd.DataFrame()
        
        processed_data = pd.DataFrame()
        
        for ticker in self.tickers:
            series = None
            try:
                # 1. Try batch
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker in data.columns.levels[0]:
                            ticker_df = data[ticker]
                            for col in ['Close', 'Adj Close', 'Price']:
                                if col in ticker_df.columns:
                                    series = ticker_df[col]
                                    break
                    elif ticker in data.columns:
                        series = data[ticker]
                
                # 2. Try individual history fallback
                if series is None or series.empty or series.isna().all():
                    print(f"⚠️  Refreshing {ticker} individually...")
                    t_obj = yf.Ticker(ticker)
                    hist = t_obj.history(start=self.start_date, auto_adjust=True)
                    if not hist.empty:
                        for col in ['Close', 'Adj Close', 'Price']:
                            if col in hist.columns:
                                series = hist[col]
                                break
                
                if series is not None:
                    processed_data[ticker] = series
                    print(f"✓ {ticker} loaded")
                else:
                    print(f"❌ Failed to load {ticker}")
                    
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        # Cleaning & Normalization
        processed_data = processed_data.ffill() 
        processed_data = processed_data.dropna()
        
        # Normalize Index
        if processed_data.index.tz is not None:
            processed_data.index = processed_data.index.tz_localize(None)
        processed_data.index = pd.to_datetime(processed_data.index).normalize()
        
        output_path = os.path.join(self.data_dir, "market_data.parquet")
        processed_data.to_parquet(output_path)
        print(f"Data saved to {output_path}. Shape: {processed_data.shape}")
        return processed_data

if __name__ == "__main__":
    loader = MarketDataLoader()
    df = loader.fetch_data()
    print(df.tail())
