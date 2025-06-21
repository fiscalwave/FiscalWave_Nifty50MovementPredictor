import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
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
    return df

def engineer_features(df):
    """Create features with consistent naming"""
    if df.empty:
        return pd.DataFrame()

    # Create a copy and force standard column names
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # CRITICAL: Force standard names
    df.dropna(inplace=True)

    if len(df) < 30:  # Need minimum data points for indicators
        return pd.DataFrame()

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
    return df

def get_nifty50_tickers():
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

def main():
    try:
        model, feature_cols = load_model()
        
        stock_list = get_nifty50_tickers()
        ticker = st.selectbox("Select stock:", stock_list, index=0)
        days = st.slider("Analysis period (days):", 100, 1095, 365)
        
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)
        
        with st.spinner(f"Fetching data for {ticker}..."):
            df = fetch_data(ticker, start_date, end_date)
        
        if df.empty:
            st.error(f"No data found for {ticker}")
            return
        
        st.info(f"Retrieved {len(df)} data points for {ticker}")
        
        with st.spinner("Calculating technical indicators..."):
            df_features = engineer_features(df)
        
        if df_features.empty:
            st.error("Not enough historical data to calculate indicators. Try a longer time period.")
            return
        
        # Verify we have the required features
        if not all(col in df_features.columns for col in feature_cols):
            missing = set(feature_cols) - set(df_features.columns)
            st.error(f"Missing features: {', '.join(missing)}")
            st.error("Please retrain your model with the updated feature engineering")
            return
        
        # Get the latest data point for prediction
        X = df_features[feature_cols].tail(1)
        
        with st.spinner("Making prediction..."):
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0][int(pred)]
        
        st.subheader(f"Prediction for {ticker.split('.')[0]}:")
        st.success(f"{'UP' if pred else 'DOWN'} ({(prob*100):.1f}% confidence)")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart
        ax1.plot(df.index, df["Close"], label="Price", color='blue')
        ax1.set_title(f"{ticker} Price Chart")
        ax1.set_ylabel("Price")
        ax1.grid(True)
        
        # RSI
        ax2.plot(df_features.index, df_features["rsi"], label="RSI", color='orange')
        ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax2.set_title("RSI Indicator")
        ax2.set_ylabel("RSI")
        ax2.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show feature values
        with st.expander("View technical details"):
            st.subheader("Latest Feature Values")
            st.dataframe(X)
            
            st.subheader("Recent Price Data")
            st.dataframe(df.tail(10)[["Open", "High", "Low", "Close", "Volume"]])
    
    except Exception as e:
        st.error("An error occurred. Please try again or select a different stock.")
        st.error(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()