import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
import joblib
from datetime import datetime, timedelta
import os
from sklearn.model_selection import train_test_split

from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Constants
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'macd', 'rsi', 'bb_bbm', 'bb_bbh', 'bb_bbl',
    'ema_20', 'sma_20', 'return_1d', 'return_2d', 'volatility_5d'
]

def engineer_features(df):
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    close_series = df["Close"].astype(float).squeeze()
    
    # MACD
    df["macd"] = MACD(close=close_series).macd_diff()
    
    # RSI
    df["rsi"] = RSIIndicator(close=close_series).rsi()
    
    # Bollinger Bands
    bb = BollingerBands(close=close_series)
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()
    
    # Moving Averages
    df["ema_20"] = EMAIndicator(close=close_series, window=20).ema_indicator()
    df["sma_20"] = SMAIndicator(close=close_series, window=20).sma_indicator()
    
    # Lag features
    df["return_1d"] = df["Close"].pct_change()
    df["return_2d"] = df["Close"].pct_change(2)
    df["volatility_5d"] = df["Close"].rolling(window=5).std()
    
    # Target
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    
    df.dropna(inplace=True)
    return df[FEATURE_COLS + ['Target']]

# Main execution
if __name__ == "__main__":
    # Download data
    ticker = "RELIANCE.NS"
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 2)
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    df.dropna(inplace=True)

    # Feature engineering
    df = engineer_features(df)
    
    # Train model
    X = df[FEATURE_COLS]
    y = df["Target"]
    
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X, y)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump((model, FEATURE_COLS), "models/xgb_model.pkl")
    print("✅ Model and feature list saved")