import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
import joblib
from datetime import datetime, timedelta
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Constants - Using GENERIC feature names
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'macd', 'rsi', 'bb_bbm', 'bb_bbh', 'bb_bbl',
    'ema_20', 'sma_20', 'return_1d', 'return_2d', 'volatility_5d'
]

def engineer_features(df):
    """Create features with consistent naming"""
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLS + ['Target'])
    
    # Create copy and force standard column names
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # CRITICAL: Force standard names
    df.dropna(inplace=True)
    
    if len(df) < 30:  # Need enough data for indicators
        return pd.DataFrame(columns=FEATURE_COLS + ['Target'])
    
    close_series = df["Close"].astype(float).squeeze()
    
    # Technical indicators
    df["macd"] = MACD(close=close_series).macd_diff()
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

def get_nifty50_tickers():
    """Get current Nifty 50 tickers"""
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "ITC.NS", "KOTAKBANK.NS", "SBIN.NS", "ASIANPAINT.NS",
        "AXISBANK.NS", "LT.NS", "MARUTI.NS", "BAJFINANCE.NS", "WIPRO.NS",
        "ONGC.NS", "SUNPHARMA.NS", "BHARTIARTL.NS", "NESTLEIND.NS", "ULTRACEMCO.NS",
        "TITAN.NS", "POWERGRID.NS", "NTPC.NS", "INDUSINDBK.NS", "BAJAJFINSV.NS",
        "ADANIPORTS.NS", "TECHM.NS", "JSWSTEEL.NS", "HCLTECH.NS", "DRREDDY.NS",
        "GRASIM.NS", "BRITANNIA.NS", "DIVISLAB.NS", "BAJAJ-AUTO.NS", "SHREECEM.NS",
        "HINDALCO.NS", "UPL.NS", "CIPLA.NS", "TATASTEEL.NS", "COALINDIA.NS",
        "BPCL.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "IOC.NS", "SBILIFE.NS",
        "GAIL.NS", "HDFCLIFE.NS", "ADANIENT.NS", "TATAMOTORS.NS"
    ]

if __name__ == "__main__":
    # Get all Nifty 50 tickers
    tickers = get_nifty50_tickers()
    all_data = []
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 3)  # 3 years of data
    
    print("Downloading and processing data for Nifty 50 stocks...")
    for ticker in tqdm(tickers):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            if not df.empty and len(df) > 100:
                df = engineer_features(df)
                all_data.append(df)
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
    
    if not all_data:
        print("No data processed. Exiting.")
        exit()
    
    # Combine all data
    full_df = pd.concat(all_data)
    
    # Verify we have the correct structure
    print(f"Final dataset shape: {full_df.shape}")
    print(f"Columns: {full_df.columns.tolist()}")
    
    # Train model
    X = full_df[FEATURE_COLS]
    y = full_df["Target"]
    
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\nTraining Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump((model, FEATURE_COLS), "models/xgb_model.pkl")
    print("✅ Model trained on all Nifty 50 stocks and saved")