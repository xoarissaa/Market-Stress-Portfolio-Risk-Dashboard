import pandas as pd
import numpy as np
import joblib
import os
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class MarketRegimeDetector:
    """
    Calculates market stress indicators and fits a GMM to detect regimes.
    """
    def __init__(self, data_path="data/raw/market_data.parquet", output_dir="outputs/models"):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_and_engineer_features(self):
        """Loads raw data and calculates the 4 core MVP indicators."""
        df = pd.read_parquet(self.data_path)
        
        # 1. Trend (SPY Price vs 200D MA)
        # Using a simple ratio: Price / MA - 1. Positive = Bull, Negative = Bear.
        df['MA200'] = df['SPY'].rolling(window=200).mean()
        df['Trend_Signal'] = (df['SPY'] / df['MA200']) - 1
        
        # 2. VIX (Market Fear)
        # Use raw VIX but we might want to normalize it later for the model
        df['VIX_Signal'] = df['^VIX']
        
        # 3. Credit Risk (HYG / IEF Ratio)
        # Higher is Better (Risk On). Lower is Stress (Risk Off).
        # We calculate the 50-day change in this ratio to catch momentum shifts.
        df['Credit_Ratio'] = df['HYG'] / df['IEF']
        df['Credit_Signal'] = df['Credit_Ratio'].pct_change(periods=20) # 1 month momentum
        
        # 4. Yield Curve (^TNX - ^IRX)
        # 10Y Yield - 13 Week Yield (Steepness). Inverted (<0) is Recession Warning.
        df['Yield_Curve'] = df['^TNX'] - df['^IRX']
        
        # Drop NaNs created by rolling windows
        df = df.dropna()
        
        return df

    def train_model(self, df):
        """Trains GMM to cluster market into 3 regimes."""
        # Features for the model
        # We standarize them so VIX (10-80) doesn't overpower Trend (-0.2 to 0.2)
        features = ['Trend_Signal', 'VIX_Signal', 'Credit_Signal', 'Yield_Curve']
        X = df[features].copy()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # GMM with 3 Components (Calm, Choppy, Stress)
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        gmm.fit(X_scaled)
        
        # Predict Regimes
        regimes = gmm.predict(X_scaled)
        df['Regime'] = regimes
        
        # We need to map the meaningless cluster IDs (0, 1, 2) to human logic.
        # We assume "Stress" regime has the Highest VIX average.
        # We assume "Calm" regime has the Lowest VIX average.
        
        regime_stats = df.groupby('Regime')['^VIX'].mean().sort_values()
        
        # Mapping: Lowest VIX -> 0 (Low Stress), Middle -> 1, Highest -> 2 (High Stress)
        mapping = {old_label: new_label for new_label, old_label in enumerate(regime_stats.index)}
        
        df['Regime_Label'] = df['Regime'].map(mapping)
        
        # Save artifacts
        joblib.dump(gmm, os.path.join(self.output_dir, "gmm_model.pkl"))
        joblib.dump(scaler, os.path.join(self.output_dir, "scaler.pkl"))
        joblib.dump(mapping, os.path.join(self.output_dir, "regime_mapping.pkl"))
        
        # Save processed data with regimes
        df.to_parquet("data/processed_market_regimes.parquet")
        
        print("Model trained and data saved.")
        print("Regime Distribution:\n", df['Regime_Label'].value_counts())
        print("Average VIX per Regime:\n", df.groupby('Regime_Label')['^VIX'].mean())
        
        return df

if __name__ == "__main__":
    detector = MarketRegimeDetector()
    df = detector.load_and_engineer_features()
    detector.train_model(df)
