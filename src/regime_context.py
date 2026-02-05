import pandas as pd
import os

class RegimeContext:
    """
    Clean interface to load and query regime data from Phase 1 (Market Stress Module).
    This module does NOT re-compute regimes - it only loads and provides access to them.
    """
    
    def __init__(self, regime_data_path="data/processed_market_regimes.parquet"):
        """
        Initialize RegimeContext with path to processed regime data.
        
        :param regime_data_path: Path to parquet file with regime labels
        """
        self.regime_data_path = regime_data_path
        self.regime_data = None
        
    def load_regime_data(self):
        """
        Loads regime data from Phase 1 output.
        
        :return: DataFrame with regime labels and market indicators
        :raises FileNotFoundError: If regime data file doesn't exist
        """
        if not os.path.exists(self.regime_data_path):
            raise FileNotFoundError(
                f"Regime data not found at {self.regime_data_path}. "
                "Please run Phase 1 (Market Regime Detection) first."
            )
        
        self.regime_data = pd.read_parquet(self.regime_data_path)
        
        # Normalize Index: Remove timezone and time for alignment
        if self.regime_data.index.tz is not None:
            self.regime_data.index = self.regime_data.index.tz_localize(None)
        self.regime_data.index = pd.to_datetime(self.regime_data.index).normalize()
        
        # Validate required columns
        required_cols = ['Regime_Label']
        missing = [col for col in required_cols if col not in self.regime_data.columns]
        if missing:
            raise ValueError(f"Regime data missing required columns: {missing}")
        
        return self.regime_data
    
    def get_regime_labels(self, start_date=None, end_date=None):
        """
        Get regime labels for a specific date range.
        
        :param start_date: Start date (str, datetime, or pd.Timestamp), optional
        :param end_date: End date (str, datetime, or pd.Timestamp), optional
        :return: pd.Series of regime labels indexed by date
        """
        if self.regime_data is None:
            self.load_regime_data()
        
        labels = self.regime_data['Regime_Label']
        
        # Filter by date range if provided
        if start_date is not None:
            start_date = pd.Timestamp(start_date)
            labels = labels[labels.index >= start_date]
        
        if end_date is not None:
            end_date = pd.Timestamp(end_date)
            labels = labels[labels.index <= end_date]
        
        return labels
    
    def filter_dates_by_regime(self, dates, regime_code):
        """
        Filter a DatetimeIndex to only include dates matching a specific regime.
        
        :param dates: DatetimeIndex or list of dates
        :param regime_code: Regime code to filter (0, 1, or 2)
        :return: Filtered DatetimeIndex
        """
        if self.regime_data is None:
            self.load_regime_data()
        
        # Get regime labels for the provided dates
        regime_labels = self.regime_data.loc[dates, 'Regime_Label']
        
        # Filter to matching regime
        matching_dates = regime_labels[regime_labels == regime_code].index
        
        return matching_dates
    
    def get_regime_statistics(self):
        """
        Get distribution statistics for regimes in the dataset.
        
        :return: Dictionary with regime distribution info
        """
        if self.regime_data is None:
            self.load_regime_data()
        
        regime_counts = self.regime_data['Regime_Label'].value_counts().sort_index()
        total_days = len(self.regime_data)
        
        stats = {
            'total_days': total_days,
            'regime_counts': regime_counts.to_dict(),
            'regime_percentages': (regime_counts / total_days * 100).to_dict(),
            'date_range': {
                'start': self.regime_data.index.min(),
                'end': self.regime_data.index.max()
            }
        }
        
        return stats
    
    def get_regime_for_date(self, date):
        """
        Get the regime label for a specific date.
        
        :param date: Date to query (str, datetime, or pd.Timestamp)
        :return: Regime code (0, 1, or 2) or None if date not found
        """
        if self.regime_data is None:
            self.load_regime_data()
        
        date = pd.Timestamp(date)
        
        if date in self.regime_data.index:
            return self.regime_data.loc[date, 'Regime_Label']
        else:
            return None
    
    def get_regime_periods(self, regime_code):
        """
        Get all date periods for a specific regime.
        
        :param regime_code: Regime code (0, 1, or 2)
        :return: DatetimeIndex of all dates in that regime
        """
        if self.regime_data is None:
            self.load_regime_data()
        
        regime_mask = self.regime_data['Regime_Label'] == regime_code
        return self.regime_data[regime_mask].index


if __name__ == "__main__":
    # Test
    ctx = RegimeContext()
    
    try:
        ctx.load_regime_data()
        print("✅ Regime data loaded successfully")
        
        stats = ctx.get_regime_statistics()
        print(f"\n📊 Regime Statistics:")
        print(f"  Total days: {stats['total_days']}")
        print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"\n  Distribution:")
        for regime, pct in stats['regime_percentages'].items():
            print(f"    Regime {regime}: {pct:.1f}%")
        
        # Test getting labels
        labels = ctx.get_regime_labels()
        print(f"\n✅ Retrieved {len(labels)} regime labels")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
