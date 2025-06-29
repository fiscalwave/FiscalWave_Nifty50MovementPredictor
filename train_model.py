import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
import joblib
from datetime import datetime, timedelta
import os
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator
from scipy.fft import rfft
import pywt

# Enhanced feature set
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'macd', 'rsi', 'bb_bbm', 'bb_bbh', 'bb_bbl',
    'ema_20', 'sma_20', 'return_1d', 'return_2d', 'return_3d',
    'return_5d', 'return_10d', 'volatility_5d', 'price_bb_diff',
    'macd_hist', 'adi', 'obv', 'atr'
]

def engineer_features(df):
    """Create features with consistent naming"""
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLS + ['Target'])
    
    # Create copy and force standard column names
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.dropna(inplace=True)
    
    if len(df) < 30:  # Need enough data for indicators
        return pd.DataFrame(columns=FEATURE_COLS + ['Target'])
    
    close_series = df["Close"].astype(float).squeeze()
    
    # Technical indicators
    macd = MACD(close=close_series)
    df["macd"] = macd.macd_diff()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    df["rsi"] = RSIIndicator(close=close_series).rsi()
    
    bb = BollingerBands(close=close_series)
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()
    
    df["ema_20"] = EMAIndicator(close=close_series, window=20).ema_indicator()
    df["sma_20"] = SMAIndicator(close=close_series, window=20).sma_indicator()
    
    # Lag features
    df["return_1d"] = df["Close"].pct_change()
    df["return_2d"] = df["Close"].pct_change(2)
    df["return_3d"] = df["Close"].pct_change(3)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["volatility_5d"] = df["Close"].rolling(window=5).std()
    
    # Volume indicators
    df["adi"] = AccDistIndexIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume']
    ).acc_dist_index()
    
    df["obv"] = OnBalanceVolumeIndicator(
        close=df['Close'],
        volume=df['Volume']
    ).on_balance_volume()
    
    # Volatility indicator
    df["atr"] = AverageTrueRange(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14
    ).average_true_range()
    
    # Additional features
    df["price_bb_diff"] = (df["Close"] - df["bb_bbl"]) / (df["bb_bbh"] - df["bb_bbl"])
    
    # Target - predict next day return
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    
    # Fourier transforms for cyclical patterns
    if len(df) > 50:
        try:
            close_values = df['Close'].values
            fft = rfft(close_values)
            for i in range(1, 4):  # First 3 components
                df[f'fft_real_{i}'] = np.real(fft[i])
                df[f'fft_imag_{i}'] = np.imag(fft[i])
        except:
            pass
    
    # Wavelet transforms
    if len(df) > 100:
        try:
            coeffs = pywt.wavedec(df['Close'], 'db4', level=3)
            for i, coeff in enumerate(coeffs):
                df[f'wavelet_{i}'] = np.concatenate([coeff, np.zeros(len(df) - len(coeff))])
        except:
            pass
    
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
    start_date = end_date - timedelta(days=365 * 5)  # 5 years of data
    
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
    
    # Prepare data
    X = full_df[FEATURE_COLS]
    y = full_df["Target"]
    
    # Create model ensemble with faster alternatives
    base_models = [
        ('xgb', XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1
        )),
        ('logreg', LogisticRegression(max_iter=1000, n_jobs=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
    ]
    
    meta_model = LogisticRegression()
    ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    # Time-series cross validation
    tscv = TimeSeriesSplit(n_splits=3)  # Reduced from 5
    scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        scores.append({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        })
    
    # Print average scores
    avg_precision = np.mean([s['precision'] for s in scores])
    avg_recall = np.mean([s['recall'] for s in scores])
    avg_f1 = np.mean([s['f1'] for s in scores])
    avg_roc_auc = np.mean([s['roc_auc'] for s in scores])
    
    print(f"\nAverage Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average ROC AUC: {avg_roc_auc:.4f}")
    
    # Final training on all data
    print("\nTraining final model on entire dataset...")
    ensemble.fit(X, y)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump((ensemble, FEATURE_COLS), "models/ensemble_model.pkl")
    print("✅ Model trained on all Nifty 50 stocks and saved")