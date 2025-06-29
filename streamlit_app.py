import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator
import requests
import os
from PIL import Image
import time
import json
import altair as alt
from scipy.fft import rfft
import pywt

# Set page config
st.set_page_config(
    page_title="FiscalWave Pro - AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CONSTANTS ----------
NIFTY50_TICKERS = [
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

TIME_OPTIONS = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730,
    "Max": 365*5
}

# ---------- UTILITY FUNCTIONS ----------
def to_scalar(value):
    """Convert any value to a scalar (non-Series) type with robust handling"""
    if value is None:
        return None
    
    # Handle pandas Series
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        return value.iloc[0]  # Return first element
    
    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        # Extract first element for 1D arrays
        if value.ndim == 1 and value.size > 0:
            return value[0]
        # For multi-dimensional arrays, take first element
        return value.flat[0]
    
    # Handle numpy scalars
    if hasattr(value, 'item'):
        return value.item()
    
    # Handle Python sequences
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) > 0 else None
    
    return value  # Return as-is if already scalar

def safe_compare(value, op, threshold):
    """Safely compare a value to a threshold after converting to scalar"""
    try:
        scalar_value = to_scalar(value)
        if scalar_value is None:
            return False
            
        if op == 'gt':
            return scalar_value > threshold
        elif op == 'lt':
            return scalar_value < threshold
        elif op == 'eq':
            return scalar_value == threshold
        return False
    except:
        return False

# ---------- CACHED FUNCTIONS ----------
@st.cache_data(ttl=300, show_spinner=False)
def get_nifty50_performance():
    """New approach: Fetch all data at once with proper error handling"""
    try:
        # Download all tickers at once
        data = yf.download(NIFTY50_TICKERS, period="2d", group_by='ticker', progress=False)
        
        performance = {}
        for ticker in NIFTY50_TICKERS:
            try:
                ticker_data = data[ticker] if len(NIFTY50_TICKERS) > 1 else data
                if len(ticker_data) < 2:
                    performance[ticker.split('.')[0]] = 0.0
                    continue
                    
                prev_close = ticker_data['Close'].iloc[-2]
                current_close = ticker_data['Close'].iloc[-1]
                
                # Calculate percentage change with rounding
                pct_change = round(((current_close - prev_close) / prev_close * 100), 2)
                performance[ticker.split('.')[0]] = pct_change
                
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                performance[ticker.split('.')[0]] = 0.0
                
        return performance
        
    except Exception as e:
        print(f"Global fetch error: {str(e)}")
        # Fallback to individual requests
        return get_nifty50_performance_fallback()

def get_nifty50_performance_fallback():
    """Fallback method if bulk download fails"""
    performance = {}
    for ticker in NIFTY50_TICKERS:
        try:
            data = yf.download(ticker, period="2d", progress=False)
            if len(data) < 2:
                performance[ticker.split('.')[0]] = 0.0
                continue
                
            prev_close = data['Close'].iloc[-2]
            current_close = data['Close'].iloc[-1]
            pct_change = round(((current_close - prev_close) / prev_close * 100), 2)
            performance[ticker.split('.')[0]] = pct_change
            
        except Exception:
            performance[ticker.split('.')[0]] = 0.0
            
    return performance

@st.cache_resource
def load_model():
    """Load model with validation"""
    model_path = "models/ensemble_model.pkl"  # Updated to new model
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()
    
    try:
        model, feature_cols = joblib.load(model_path)
        return model, feature_cols
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def load_logo():
    """Load logo with multiple fallback paths"""
    paths = [
        "FiscalWaveOriginal.png",
        "/workspaces/blank-app/assets/FiscalWaveOriginal.png",
        "./assets/FiscalWaveOriginal.png"
    ]
    
    for path in paths:
        if os.path.exists(path):
            return Image.open(path)
    return None

def fetch_data(ticker, start_date, end_date):
    """Fetch data with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
            if not df.empty:
                return df
        except Exception:
            if attempt == max_retries - 1:
                return pd.DataFrame()
            time.sleep(1)
    return pd.DataFrame()

def engineer_features(df):
    """Create features with robust error handling"""
    if df.empty or len(df) < 30:
        return pd.DataFrame()

    try:
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.dropna(inplace=True)
        
        close_series = df["Close"].astype(float).squeeze()

        # MACD
        macd = MACD(close=close_series)
        df["macd"] = macd.macd_diff()
        df["macd_signal"] = macd.macd_signal()
        
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
        df["return_3d"] = df["Close"].pct_change(3)  # New
        df["return_5d"] = df["Close"].pct_change(5)  # New
        df["return_10d"] = df["Close"].pct_change(10)  # New
        df["volatility_5d"] = df["Close"].rolling(window=5).std()
        
        # Volume-based indicators
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
        
        # Volatility indicators
        df["atr"] = AverageTrueRange(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=14
        ).average_true_range()
        
        # Additional features
        df["price_bb_diff"] = (df["Close"] - df["bb_bbl"]) / (df["bb_bbh"] - df["bb_bbl"])
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
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
        return df
    except Exception as e:
        st.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

# ========== ENHANCED NEWS FUNCTIONS ==========
@st.cache_data(ttl=3600, show_spinner="Fetching market news...")  # Cache for 1 hour
def get_stock_news(stock_name):
    """Enhanced news fetching with multiple sources and fallbacks"""
    try:
        # Use the actual stock name without .NS suffix
        query_name = stock_name.split('.')[0] if '.' in stock_name else stock_name
        
        # Get API key from secrets - simplified approach
        api_key = None
        try:
            api_key = st.secrets.get("NEWS_API_KEY", st.secrets.get("news_api_key"))
        except Exception as e:
            st.warning(f"Secrets access issue: {str(e)}")
        
        if not api_key:
            return {
                "status": "error",
                "message": "News API key not configured. Please add your API key to secrets.toml"
            }
        
        # Try different endpoints and parameters
        endpoints = [
            f"https://newsapi.org/v2/everything?q={query_name}&apiKey={api_key}&pageSize=5&language=en",
            f"https://newsapi.org/v2/top-headlines?q={query_name}&apiKey={api_key}&pageSize=5&language=en&category=business",
            f"https://newsapi.org/v2/everything?q=NSE:{query_name}&apiKey={api_key}&pageSize=5&language=en"
        ]
        
        for url in endpoints:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                news_data = response.json()
                
                if news_data.get('status') == 'ok' and news_data.get('totalResults', 0) > 0:
                    return {
                        "status": "success",
                        "articles": news_data.get('articles', [])
                    }
            except requests.exceptions.RequestException as e:
                print(f"News API error ({url}): {str(e)}")
                continue  # Try next endpoint
        
        return {"status": "no_results", "message": "No news found for this stock"}
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error fetching news: {str(e)}"
        }

def format_news_date(raw_date):
    """Safely format news date with fallback"""
    try:
        if not raw_date:
            return "Date not available"
        date_obj = datetime.strptime(raw_date, "%Y-%m-%dT%H:%M:%SZ")
        return date_obj.strftime("%b %d, %Y %I:%M %p")
    except:
        return raw_date[:10] if raw_date else "Date not available"

def analyze_sentiment(text):
    """Simple sentiment analysis (placeholder for actual model)"""
    positive_words = ['bullish', 'growth', 'profit', 'gain', 'buy', 'strong']
    negative_words = ['bearish', 'loss', 'sell', 'weak', 'decline', 'drop']
    
    if not text:
        return 0.5  # Neutral
    
    text_lower = text.lower()
    positive_count = sum(word in text_lower for word in positive_words)
    negative_count = sum(word in text_lower for word in negative_words)
    
    total = positive_count + negative_count
    if total == 0:
        return 0.5
    
    return positive_count / total

def render_news_section(news_result, selected_stock):
    """Render news section with sentiment analysis"""
    st.subheader(f"📰 Latest News for {selected_stock}")
    
    if news_result["status"] == "success":
        articles = news_result["articles"]
        for i, article in enumerate(articles):
            text = f"{article.get('title', '')}. {article.get('description', '')}"
            sentiment = analyze_sentiment(text)
            sentiment_color = "#4CAF50" if sentiment > 0.6 else "#F44336" if sentiment < 0.4 else "#FFC107"
            sentiment_label = "Positive" if sentiment > 0.6 else "Negative" if sentiment < 0.4 else "Neutral"
            
            with st.expander(f"{i+1}. {article.get('title', 'No title available')}", expanded=(i==0)):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if article.get('urlToImage'):
                        st.image(article['urlToImage'], width=200)
                    else:
                        st.image("https://via.placeholder.com/200x100?text=No+Image", width=200)
                
                with col2:
                    st.markdown(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
                    st.markdown(f"**Published:** {format_news_date(article.get('publishedAt'))}")
                    st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}; font-weight:bold;'>{sentiment_label} ({sentiment:.2f})</span>", 
                               unsafe_allow_html=True)
                    st.markdown(f"**Description:** {article.get('description', 'No description available')}")
                    st.markdown(f"[Read full article]({article.get('url', '#')})")
        
        # Add refresh button
        if st.button("🔄 Refresh News", key="refresh_news"):
            st.cache_data.clear()
    
    elif news_result["status"] == "no_results":
        st.info("📭 No recent news articles found. Try again later.")
    
    else:  # Error case
        st.error(f"❌ {news_result['message']}")

# ========== HISTORICAL DATA ==========
def render_historical_data(df, selected_stock):
    """Render historical data section with interactive features"""
    st.subheader("📊 Historical Price Data")
    
    if df.empty:
        st.info("📭 No historical data available for this time period")
        st.markdown("""
        <div style="background-color:#e8f4f8; padding:15px; border-radius:10px; margin-top:20px;">
            <h4>💡 Tips for historical data:</h4>
            <ul>
                <li>Try a longer time period (1 Year or Max)</li>
                <li>Check if the stock trades on Indian markets</li>
                <li>Verify the stock symbol is correct</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Add date range selector
    min_date = df.index.min().to_pydatetime()
    max_date = df.index.max().to_pydatetime()
    
    if min_date != max_date:
        date_range = st.slider(
            "Select date range:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
        
        # Filter data based on selected range
        filtered_df = df.loc[date_range[0]:date_range[1]]
    else:
        filtered_df = df
    
    # Display data with sorting and search
    st.dataframe(
        filtered_df[["Open", "High", "Low", "Close", "Volume"]].style.format({
            "Open": "{:.2f}",
            "High": "{:.2f}",
            "Low": "{:.2f}",
            "Close": "{:.2f}",
            "Volume": "{:,.0f}"
        }).background_gradient(cmap="Blues", subset=["Close"])
        .set_properties(**{'text-align': 'center'}),
        height=400
    )
    
    # Add chart visualization
    st.subheader("Price Trend")
    
    # Add chart type selection
    chart_type = st.radio("Chart Type:", ["Line", "Candlestick"], horizontal=True)
    
    # Create a copy of the filtered data and reset index
    plot_df = filtered_df.copy().reset_index()
    plot_df['Date'] = plot_df['Date'].dt.strftime('%Y-%m-%d')
    
    if chart_type == "Line":
        # Create line chart using Altair
        import altair as alt
        
        line_chart = alt.Chart(plot_df).mark_line(
            color='#1f77b4', 
            size=3
        ).encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Close:Q', title='Price (₹)'),
            tooltip=['Date:T', 'Close:Q']
        ).properties(
            title=f"{selected_stock} Price Trend",
            width=800,
            height=400
        ).interactive()
        
        st.altair_chart(line_chart, use_container_width=True)
        
    else:  # Candlestick chart
        # Create candlestick chart using Altair
        import altair as alt
        
        # Create base chart
        base = alt.Chart(plot_df).encode(
            x='Date:T',
            color=alt.condition(
                "datum.Open <= datum.Close",
                alt.value("#06982d"),  # Green for up
                alt.value("#ae1325")   # Red for down
            )
        )
        
        # Create candlesticks
        rule = base.mark_rule().encode(
            y='Low:Q',
            y2='High:Q'
        )
        
        bar = base.mark_bar().encode(
            y='Open:Q',
            y2='Close:Q'
        )
        
        # Combine and configure
        chart = (rule + bar).properties(
            title=f"{selected_stock} Candlestick Chart",
            width=800,
            height=400
        ).configure_axis(
            grid=False
        ).configure_view(
            strokeWidth=0
        )
        
        st.altair_chart(chart, use_container_width=True)
    
    # Add download options
    st.markdown("---")
    st.subheader("Download Data")
    
    col1, col2 = st.columns(2)
    with col1:
        csv = filtered_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"{selected_stock}_historical_data.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = filtered_df.to_json(orient="records")
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"{selected_stock}_historical_data.json",
            mime="application/json"
        )
		
# ========== TECHNICAL ANALYSIS ==========
def get_technical_insight(rsi, macd_hist, price, bb_bbl, bb_bbh, adi, obv):
    """Generate technical insights with enhanced safety checks"""
    insights = []
    
    try:
        # Convert all values to scalars first
        rsi = to_scalar(rsi)
        macd_hist = to_scalar(macd_hist)
        price = to_scalar(price)
        bb_bbl = to_scalar(bb_bbl)
        bb_bbh = to_scalar(bb_bbh)
        adi = to_scalar(adi)
        obv = to_scalar(obv)
        
        # RSI Insights
        if rsi is not None:
            if rsi > 70:
                insights.append("RSI > 70 (Overbought)")
            elif rsi < 30:
                insights.append("RSI < 30 (Oversold)")
        
        # MACD Insights
        if macd_hist is not None:
            if macd_hist > 0:
                insights.append("MACD Bullish")
            else:
                insights.append("MACD Bearish")
        
        # Bollinger Bands Insights
        if all(v is not None for v in [price, bb_bbl, bb_bbh]):
            if price < bb_bbl:
                insights.append("Price below Lower Bollinger Band (Oversold)")
            elif price > bb_bbh:
                insights.append("Price above Upper Bollinger Band (Overbought)")
        
        # Volume Insights
        if adi is not None and adi > 0:
            insights.append("Accumulation/Distribution trending up")
        
        if obv is not None and obv > 0:
            insights.append("On-Balance Volume positive")
                
    except Exception as e:
        print(f"Insight generation error: {str(e)}")
    
    return insights

def safe_format(value, format_str=".2f", default="N/A"):
    """Safely format values with type checking"""
    if value is None or pd.isna(value):
        return default
    try:
        scalar_value = to_scalar(value)
        return format(float(scalar_value), format_str)
    except (TypeError, ValueError):
        return default

def detect_market_regime(df):
    """Detect market regime using clustering"""
    from sklearn.cluster import KMeans
    
    if len(df) < 30:
        return "Normal"
    
    features = df[['volatility_5d', 'return_1d', 'Volume']].dropna().tail(30)
    
    if len(features) < 10:
        return "Normal"
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    features['regime'] = kmeans.fit_predict(features)
    
    # Map to human-readable labels
    regime_map = {
        0: "Low Volatility",
        1: "High Volatility",
        2: "Trending"
    }
    
    latest_regime = features['regime'].iloc[-1]
    return regime_map.get(latest_regime, "Normal")

def create_performance_marquee(performance_data):
    """Create animated marquee content for all 50 tickers"""
    ticker_items = []
    for stock, change in performance_data.items():
        # Ensure change is a scalar value
        change_value = to_scalar(change)
        
        # Format the percentage change properly
        if change_value == 0:
            display_change = "0.00%"
            arrow = "→"
        else:
            display_change = f"{abs(change_value):.2f}%"
            arrow = "↑" if change_value > 0 else "↓"
        
        color_class = "ticker-up" if change_value > 0 else "ticker-down" if change_value < 0 else "ticker-neutral"
        
        ticker_items.append(
            f"<span class='{color_class}'>{stock} {arrow} {display_change}</span>"
        )
    
    # Combine all 50 stocks in a single marquee with proper spacing
    marquee_content = " 🔔 NIFTY 50 LIVE &nbsp;&nbsp;|&nbsp;&nbsp; " + " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(ticker_items)
    
    return marquee_content

def create_top_performers(performance_data, top_n=5):
    """Create top gainers and losers sections"""
    # Sort stocks by performance
    sorted_stocks = sorted(performance_data.items(), key=lambda x: x[1], reverse=True)
    
    # Get top gainers and losers
    top_gainers = sorted_stocks[:top_n]
    top_losers = sorted_stocks[-top_n:][::-1]  # Reverse to show worst first
    
    return top_gainers, top_losers

def render_performance_sections(top_gainers, top_losers):
    """Render top gainers and losers in columns"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div style="background-color:#e6f4ea; padding:10px; border-radius:10px; margin-bottom:15px;">
                <h4 style="color:#2e8b57; margin-bottom:10px; text-align:center;">🏆 Top Gainers</h4>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    {''.join([
                        f'<div style="display: flex; justify-content: space-between; padding: 3px 10px; border-radius: 5px; background-color: rgba(46, 139, 87, 0.1);">'
                        f'<span style="font-weight: bold;">{stock}</span>'
                        f'<span style="color: #2e8b57;">↑ {change:.2f}%</span>'
                        f'</div>'
                        for stock, change in top_gainers
                    ])}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="background-color:#fce8e6; padding:10px; border-radius:10px; margin-bottom:15px;">
                <h4 style="color:#d93025; margin-bottom:10px; text-align:center;">💣 Top Losers</h4>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    {''.join([
                        f'<div style="display: flex; justify-content: space-between; padding: 3px 10px; border-radius: 5px; background-color: rgba(217, 48, 37, 0.1);">'
                        f'<span style="font-weight: bold;">{stock}</span>'
                        f'<span style="color: #d93025;">↓ {abs(change):.2f}%</span>'
                        f'</div>'
                        for stock, change in top_losers
                    ])}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
	
# ---------- UI COMPONENTS ----------
def render_header(performance_data):
    """Render application header with logo, ticker, and performance sections"""
    logo = load_logo()
    marquee_content = create_performance_marquee(performance_data)
    
    # Create a container for the header
    header_container = st.container()
    
    with header_container:
        # New row: Logo and Title
        col_title, col_logo = st.columns([4, 1])
        
        with col_title:
            # Moved title to this position
            st.markdown(
                """
                <style>
                .animated-gradient {
                    font-size: 30px;
                    font-weight: bold;
                    background: linear-gradient(
                        270deg,
                        #FF6666,
                        #66FF66,
                        #FF0000,
                        #00CC00,
                        #FF6666
                    );
                    background-size: 1000% 100%;
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    animation: gradientMove 5s ease infinite;
                    white-space: nowrap;
                }

                @keyframes gradientMove {
                    0% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                    100% { background-position: 0% 50%; }
                }
                </style>

                <h2 class='animated-gradient' style='text-align: center; margin-top: 0; margin-bottom: 0;'>
                    FiscalWave Pro: AI-Powered Stock Predictor
                </h2>
                """,
                unsafe_allow_html=True
            )
        
        with col_logo:
            if logo:
                st.image(logo, width=150)
            else:
                st.empty()
        
        # Marquee row (below title and logo)
        st.markdown(
            f"""
            <div style="width: 100%; overflow: hidden; white-space: nowrap; margin-top: 10px;">
                <marquee behavior="scroll" direction="left" scrollamount="8" style="font-weight:bold;">
                    {marquee_content}
                </marquee>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Add spacing
    st.write("")
    
    # Add Top Gainers and Losers sections BELOW the header
    top_gainers, top_losers = create_top_performers(performance_data)
    render_performance_sections(top_gainers, top_losers)
    
    # Add spacing
    st.write("")

def render_prediction_ui(pred, confidence, uncertainty=None):
    """Render prediction result UI with uncertainty"""
    try:
        # Simple boolean check
        if pred:
            confidence_display = f"{confidence:.1f}%" if confidence is not None else "N/A"
            if uncertainty is not None:
                confidence_display += f" ±{uncertainty:.1f}%"
                
            st.markdown(
                f"<div class='prediction-up'>"
                f"<h2>📈 BUY Recommendation</h2>"
                f"<h3>Confidence: {confidence_display}</h3>"
                f"</div>", 
                unsafe_allow_html=True
            )
        else:
            confidence_display = f"{confidence:.1f}%" if confidence is not None else "N/A"
            if uncertainty is not None:
                confidence_display += f" ±{uncertainty:.1f}%"
                
            st.markdown(
                f"<div class='prediction-down'>"
                f"<h2>📉 SELL Recommendation</h2>"
                f"<h3>Confidence: {confidence_display}</h3>"
                f"</div>", 
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Could not render prediction: {str(e)}")

def render_technical_metrics(latest):
    """Render technical metric cards"""
    st.subheader("Key Technical Metrics")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        rsi_value = to_scalar(latest.get('rsi', None))
        rsi_display = safe_format(rsi_value, '.2f')
        status = 'Overbought' if safe_compare(rsi_value, 'gt', 70) else 'Oversold' if safe_compare(rsi_value, 'lt', 30) else 'Neutral'
        st.markdown(f"""
        <div class='metric-card rsi-card'>
            <h4>RSI</h4>
            <h3>{rsi_display}</h3>
            <p>{status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        adi_value = to_scalar(latest.get('adi', None))
        adi_display = safe_format(adi_value, '.2f')
        trend = 'Up' if safe_compare(adi_value, 'gt', 0) else 'Down' if safe_compare(adi_value, 'lt', 0) else 'Neutral'
        st.markdown(f"""
        <div class='metric-card adi-card'>
            <h4>Accumulation/Dist</h4>
            <h3>{adi_display}</h3>
            <p>{trend}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        atr_value = to_scalar(latest.get('atr', None))
        atr_display = safe_format(atr_value, '.2f')
        volatility = 'High' if safe_compare(atr_value, 'gt', 0.05) else 'Low' if safe_compare(atr_value, 'lt', 0.01) else 'Moderate'
        st.markdown(f"""
        <div class='metric-card atr-card'>
            <h4>ATR (Volatility)</h4>
            <h3>{atr_display}</h3>
            <p>{volatility}</p>
        </div>
        """, unsafe_allow_html=True)

# ---------- MAIN APPLICATION ----------
def main():
    # Custom CSS
    st.markdown(
        """
        <style>
        /* Remove top whitespace */
        .block-container {
            padding-top: 1rem !important;
        }

        .main {
            padding-top: 0rem !important;
            padding-bottom: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        /* Prediction styling */
        .prediction-up {
            background-color: #e6f4ea;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: green;
            box-shadow: 0 4px 8px rgba(0,100,0,0.2);
            border: 1px solid #c3e6cb;
        }
        
        .prediction-down {
            background-color: #fce8e6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: #d93025;
            box-shadow: 0 4px 8px rgba(139,0,0,0.2);
            border: 1px solid #f5c6cb;
        }
        
        /* Metric card styling */
        .metric-card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        .rsi-card {
            border-top: 4px solid #FF6666;
        }
        
        .adi-card {
            border-top: 4px solid #2196F3;
        }
        
        .atr-card {
            border-top: 4px solid #9C27B0;
        }
        
        /* Ticker styling */
        .ticker-up {
            color: green;
            font-weight: bold;
        }
        
        .ticker-down {
            color: red;
            font-weight: bold;
        }
        
        .ticker-neutral {
            color: #007BFF;
            font-weight: bold;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            border-radius: 8px 8px 0 0;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #e6f4ea;
            font-weight: bold;
        }
        
        /* Top performers styling */
        .top-gainer-item, .top-loser-item {
            transition: all 0.3s ease;
        }
        .top-gainer-item:hover, .top-loser-item:hover {
            transform: scale(1.02);
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Load performance data
    with st.spinner("Loading market data..."):
        performance_data = get_nifty50_performance()
        
        if len(set(performance_data.values())) == 1:
            st.warning("Performance data appears identical for all stocks - may indicate stale data")
    
    # Render header
    render_header(performance_data)
    
    try:
        # Load model
        model, feature_cols = load_model()
        
        # Stock selection
        stock_list = [t.split('.')[0] for t in NIFTY50_TICKERS]
        selected_stock = st.selectbox("Select stock:", stock_list, index=0)
        ticker = f"{selected_stock}.NS"
        
        # Timeframe selection
        selected_time = st.select_slider("Analysis period:", options=list(TIME_OPTIONS.keys()), value="1 Year")
        days = TIME_OPTIONS[selected_time]
        
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data
        with st.spinner(f"Fetching data for {selected_stock}..."):
            df = fetch_data(ticker, start_date, end_date)
        
        if df.empty:
            st.error(f"No data found for {selected_stock}. Please try a different stock or timeframe.")
            return
        
        st.success(f"Retrieved {len(df)} trading days of data for {selected_stock}")
        
        # Engineer features
        with st.spinner("Calculating technical indicators..."):
            df_features = engineer_features(df)
        
        if df_features.empty:
            st.error("Not enough historical data to calculate indicators. Try a longer time period.")
            return
        
        # Verify required features
        missing_features = set(feature_cols) - set(df_features.columns)
        if missing_features:
            st.error(f"Missing features: {', '.join(missing_features)}")
            st.error("Please retrain your model with the updated feature engineering")
            return
        
        # Get latest data for prediction
        latest = df_features.iloc[-1]
        X = df_features[feature_cols].tail(1)
        
        # Detect market regime
        market_regime = detect_market_regime(df_features)
        st.markdown(f"**Market Regime:** <span style='color:#2196F3;'>{market_regime}</span>", unsafe_allow_html=True)
        
        # Make prediction - FIXED SECTION
        with st.spinner("Analyzing market patterns..."):
            # Get raw prediction value
            pred_value = model.predict(X)[0]
            
            # Convert to scalar and boolean
            pred_scalar = to_scalar(pred_value)
            bool_pred = bool(pred_scalar)
            
            # Get confidence value
            confidence_val = None
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)
                confidence_val = float(np.max(pred_proba[0])) * 100
            
            # Get uncertainty value
            uncertainty_val = None
            if hasattr(model, 'estimators_'):
                predictions = []
                for estimator in model.estimators_:
                    if hasattr(estimator, 'predict_proba'):
                        pred = estimator.predict_proba(X)
                        predictions.append(pred)
                if predictions:
                    predictions_array = np.array(predictions)
                    std_pred = np.std(predictions_array, axis=0)
                    uncertainty_val = float(std_pred[0][0]) * 100
        
            # Render prediction
            render_prediction_ui(bool_pred, confidence_val, uncertainty_val)
        
        # Get technical insights
        insights = get_technical_insight(
            latest.get('rsi', None), 
            latest.get('macd_hist', None),
            latest.get('Close', None),
            latest.get('bb_bbl', None),
            latest.get('bb_bbh', None),
            latest.get('adi', None),
            latest.get('obv', None)
        )
        
        # Main layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Technical insights
            if insights:
                st.subheader("Technical Insights")
                for insight in insights:
                    if "Bullish" in insight or "Oversold" in insight or "Up" in insight:
                        st.success(f"✅ {insight}")
                    elif "Bearish" in insight or "Overbought" in insight or "Down" in insight:
                        st.warning(f"⚠️ {insight}")
                    else:
                        st.info(f"ℹ️ {insight}")
            else:
                st.warning("No technical insights available")
            
            # Technical metrics
            render_technical_metrics(latest)
            
            # Performance metrics
            st.subheader("Performance Metrics")
            col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
            
            with col_perf1:
                return_1d = to_scalar(latest.get('return_1d', None))
                st.metric("1D Return", 
                         f"{safe_format(return_1d*100, '.2f')}%" if return_1d is not None else "N/A",
                         delta_color="inverse")
            
            with col_perf2:
                return_5d = to_scalar(latest.get('return_5d', None))
                st.metric("5D Return", 
                         f"{safe_format(return_5d*100, '.2f')}%" if return_5d is not None else "N/A",
                         delta_color="inverse")
            
            with col_perf3:
                volatility = to_scalar(latest.get('volatility_5d', None))
                st.metric("5D Volatility", 
                         safe_format(volatility, '.4f') if volatility is not None else "N/A")
            
            with col_perf4:
                try:
                    current_price = to_scalar(df['Close'].iloc[-1])
                    prev_price = to_scalar(df['Close'].iloc[-2])
                    price_change = current_price - prev_price
                    change_percent = (price_change / prev_price) * 100
                    st.metric("Latest Price", 
                             safe_format(current_price, '.2f'),
                             f"{safe_format(price_change, '.2f')} ({safe_format(change_percent, '.2f')}%)")
                except:
                    st.metric("Latest Price", safe_format(df['Close'].iloc[-1], '.2f'))
        
        with col2:
            # Visualization
            with st.spinner("Generating charts..."):
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), 
                                                   gridspec_kw={'height_ratios': [3, 1, 1]})
                
                # Price chart with Bollinger Bands
                ax1.plot(df.index, df["Close"], label="Price", color='#1f77b4', linewidth=2)
                ax1.plot(df_features.index, df_features["bb_bbh"], label="Upper BB", 
                        color='#d62728', alpha=0.7, linestyle='--')
                ax1.plot(df_features.index, df_features["bb_bbl"], label="Lower BB", 
                        color='#2ca02c', alpha=0.7, linestyle='--')
                ax1.fill_between(df_features.index, df_features["bb_bbl"], df_features["bb_bbh"], 
                                color='#e0e0e0', alpha=0.2)
                ax1.set_title(f"{selected_stock} Price with Bollinger Bands", fontsize=14, fontweight='bold')
                ax1.set_ylabel("Price", fontsize=12)
                ax1.legend(loc='upper left', frameon=False)
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.tick_params(axis='x', rotation=45)
                
                # MACD
                ax2.plot(df_features.index, df_features["macd"], label="MACD", color='#ff7f0e')
                ax2.plot(df_features.index, df_features["macd_signal"], label="Signal", color='#17becf')
                ax2.bar(df_features.index, df_features["macd_hist"], 
                       label="Histogram", color=np.where(df_features["macd_hist"] > 0, '#2ca02c', '#d62728'), alpha=0.7)
                ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
                ax2.set_title("MACD Indicator", fontsize=14, fontweight='bold')
                ax2.set_ylabel("MACD", fontsize=12)
                ax2.legend(loc='upper left', frameon=False)
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.tick_params(axis='x', rotation=45)
                
                # Volume Indicators
                ax3.plot(df_features.index, df_features["adi"], label="Accumulation/Dist", color='#9467bd')
                ax3.plot(df_features.index, df_features["obv"], label="OBV", color='#8c564b')
                ax3.set_title("Volume Indicators", fontsize=14, fontweight='bold')
                ax3.set_ylabel("Value", fontsize=12)
                ax3.grid(True, linestyle='--', alpha=0.7)
                ax3.tick_params(axis='x', rotation=45)
                ax3.legend(loc='upper left', frameon=False)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # Tabs for additional information
        tab1, tab2, tab3 = st.tabs(["Technical Details", "Market News", "Historical Data"])
        
        with tab1:
            st.subheader("Latest Feature Values")
            formatted_df = X.T.applymap(lambda x: safe_format(x, '.4f'))
            st.dataframe(formatted_df, height=400)
            
        with tab2:
            # Get market news
            news_result = get_stock_news(selected_stock)
            render_news_section(news_result, selected_stock)
        
        with tab3:
            # Render historical data
            render_historical_data(df, selected_stock)
    
    except Exception as e:
        st.error("An unexpected error occurred. Please try again or select a different stock.")
        st.error(f"Error details: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()