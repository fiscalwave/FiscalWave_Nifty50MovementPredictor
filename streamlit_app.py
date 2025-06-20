import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# App Title
st.title("📈 Nifty 50 Stock Price Movement Predictor")

@st.cache_resource
def load_model():
    model, feature_cols = joblib.load("models/xgb_model.pkl")
    return model, feature_cols

def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    return df.dropna()

def engineer_features(df, feature_cols):
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
    
    df.dropna(inplace=True)
    return df[feature_cols]

def main():
    model, feature_cols = load_model()
    
    stock_list = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    ticker = st.selectbox("Select stock:", stock_list)
    days = st.slider("Analysis period (days):", 100, 730, 365)
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    df = fetch_data(ticker, start_date, end_date)
    
    if df.empty:
        st.error("No data found")
        return
    
    df = engineer_features(df, feature_cols)
    X = df[feature_cols].tail(1)
    
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][int(pred)]
    
    st.subheader(f"Prediction: {'UP' if pred else 'DOWN'} ({(prob*100):.1f}% confidence)")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Price", color='blue')
    ax2 = ax.twinx()
    ax2.plot(df.index, df["rsi"], label="RSI", color='orange', alpha=0.7)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    st.pyplot(fig)

if __name__ == "__main__":
    main()