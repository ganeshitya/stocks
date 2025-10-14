# V32 Screener - Enhanced (Embedded static tickers + Balanced Scoring + Z-score + Backtest + Radar)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import ta
from ta.volatility import AverageTrueRange
from scipy.stats import zscore

# ---------------------------
# Page config & CSS (unchanged)
# ---------------------------
st.set_page_config(page_title="Indian Stock Screener", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .buy-signal {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .sell-signal {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .neutral-signal {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Static ticker lists (Option-A)
# ---------------------------
# Large Cap tickers (NIFTY50 subset/static)
LARGE_CAP_TICKERS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","BHARTIARTL.NS",
    "HINDUNILVR.NS","ITC.NS","LT.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS",
    "ASIANPAINT.NS","BAJFINANCE.NS","SUNPHARMA.NS","TITAN.NS","MARUTI.NS",
    "ULTRACEMCO.NS","HCLTECH.NS","WIPRO.NS","POWERGRID.NS","NTPC.NS","ADANIENT.NS",
    "ADANIPORTS.NS","TATASTEEL.NS","JSWSTEEL.NS","M&M.NS","COALINDIA.NS","ONGC.NS",
    "TECHM.NS","NESTLEIND.NS","BRITANNIA.NS","HDFCLIFE.NS","SBILIFE.NS",
    "BAJAJFINSV.NS","CIPLA.NS","DRREDDY.NS","DIVISLAB.NS","HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS","EICHERMOT.NS","TATAMOTORS.NS","BPCL.NS","IOC.NS","INDUSINDBK.NS",
    "LTIM.NS","GRASIM.NS","TATACONSUM.NS","SHREECEM.NS","APOLLOHOSP.NS"
]

# Mid Cap tickers (subset)
MID_CAP_TICKERS = [
    "BANKBARODA.NS","CANBK.NS","PFC.NS","GAIL.NS","SIEMENS.NS","INDIGO.NS",
    "PIDILITIND.NS","ADANIPOWER.NS","DMART.NS","BEL.NS","LUPIN.NS","VOLTAS.NS",
    "POLYCAB.NS","TRENT.NS","IRCTC.NS","TORNTPHARM.NS","HAVELLS.NS","CHOLAFIN.NS",
    "BOSCHLTD.NS","DABUR.NS","PAGEIND.NS","ABB.NS","BERGEPAINT.NS","DLF.NS",
    "COLPAL.NS","AUROPHARMA.NS","TATAPOWER.NS","INDUSTOWER.NS","MPHASIS.NS",
    "COFORGE.NS","MUTHOOTFIN.NS","PERSISTENT.NS","PIIND.NS","LTTS.NS",
    "BALKRISIND.NS","ASTRAL.NS","CROMPTON.NS","CONCOR.NS","SUNTV.NS","ICICIPRULI.NS",
    "JINDALSTEL.NS","ESCORTS.NS","AMBUJACEM.NS","CANFINHOME.NS","TATACOMM.NS","ZOMATO.NS",
    "INDHOTEL.NS","SAIL.NS","GODREJPROP.NS"
]

# Small Cap tickers (subset)
SMALL_CAP_TICKERS = [
    "BATAINDIA.NS","RADICO.NS","CERA.NS","KEI.NS","KPRMILL.NS","DEEPAKNTR.NS",
    "IIFL.NS","JUBLFOOD.NS","KIRLOSENG.NS","SYMPHONY.NS","RITES.NS","SUNDARMFIN.NS",
    "GRANULES.NS","FINEORG.NS","METROPOLIS.NS","NAVINFLUOR.NS","TATACOFFEE.NS",
    "TIMKEN.NS","TTKPRESTIG.NS","VINATIORGA.NS","WHIRLPOOL.NS","TRIVENI.NS",
    "LEMONTREE.NS","HINDCOPPER.NS","JKTYRE.NS","RAJESHEXPO.NS","TEAMLEASE.NS",
    "SANOFI.NS","FDC.NS","AARTIIND.NS","CYIENT.NS","AMARAJABAT.NS","TEXRAIL.NS"
]

# Master universe
ALL_STATIC_STOCKS = list(set(LARGE_CAP_TICKERS + MID_CAP_TICKERS + SMALL_CAP_TICKERS))

# ---------------------------
# Helper indicator & fetch functions (unchanged behavior)
# ---------------------------
def calculate_sma(data, period):
    return data['Close'].rolling(window=period).mean()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def safe_yf_fetch(ticker, retries=3, pause=1.0):
    for i in range(retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if info and (('longName' in info) or ('shortName' in info) or ('regularMarketPrice' in info)):
                return stock, info
        except Exception:
            time.sleep(pause)
    return None, {}

# ---------------------------
# Scoring model, z-score & helpers (new)
# ---------------------------
def safe_get_float(row, keys):
    for k in keys:
        try:
            if isinstance(row, dict):
                v = row.get(k)
            else:
                v = row[k] if k in row else None
        except Exception:
            v = None
        try:
            if v is None:
                continue
            return float(v)
        except Exception:
            continue
    return None

def compute_sector_stats(df, metric_map=None):
    metric_map = metric_map or {
        'pe': 'P/E Ratio',
        'roe': 'ROE (%)',
        'profit_growth': '3Y Profit Growth (%)',
        'de': 'Debt to Equity',
        'volatility': 'Volatility (%)'
    }
    stats = {}
    for key, col in metric_map.items():
        if col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if not series.empty:
                stats[key] = (series.mean(), series.std(ddof=0))
    return stats

def calculate_scores_for_stock(data_row, sector_stats=None):
    """
    Balanced scoring:
      - Fundamental Strength (40%) [max 4]
      - Technical Momentum (40%) [max 4]
      - Sector & Risk (20%) [max 2]
    Returns dict with component scores, total 0-10 and recommendation.
    Uses z-score normalization via sector_stats if available.
    """
    # fundamental inputs
    pe = safe_get_float(data_row, ['P/E Ratio','P/E','pe_ratio'])
    roe = safe_get_float(data_row, ['ROE (%)','ROE','roe'])
    profit_growth = safe_get_float(data_row, ['3Y Profit Growth (%)','3Y Profit Growth','profit_growth'])
    debt_equity = safe_get_float(data_row, ['Debt to Equity','debtToEquity','Debt/Equity'])
    dividend = safe_get_float(data_row, ['Dividend Yield (%)','Dividend Yield','dividend_yield'])
    # technical inputs
    rsi = safe_get_float(data_row, ['RSI','rsi'])
    macd = safe_get_float(data_row, ['MACD','macd'])
    macd_signal = safe_get_float(data_row, ['MACD_Signal','macd_signal','Signal'])
    adx = safe_get_float(data_row, ['ADX','adx'])
    price = safe_get_float(data_row, ['Current Price','Current_Price','currentPrice'])
    sma50 = safe_get_float(data_row, ['SMA_50','SMA50'])
    beta = safe_get_float(data_row, ['Beta','beta'])
    vol = safe_get_float(data_row, ['Volatility (%)','volatility'])

    def normalized(metric_name, value):
        try:
            if not sector_stats or metric_name not in sector_stats:
                return None
            mean, std = sector_stats[metric_name]
            if std and std > 0:
                return (value - mean) / std
        except Exception:
            return None
        return None

    # Fundamental points (0-4 approx)
    f_points = 0.0
    # P/E (lower better): allocate up to 1.5
    if pe and pe > 0:
        z = normalized('pe', pe)
        if z is not None:
            pe_score = max(0.0, min(1.5, 1.5 - 0.3 * z))  # reduce when z positive
        else:
            pe_score = 1.0 if pe < 25 else 0.6 if pe < 40 else 0.2
        f_points += max(0, pe_score)
    # ROE up to 1.0
    if roe is not None:
        z = normalized('roe', roe)
        if z is not None:
            roe_score = max(0.0, min(1.0, 0.5 + 0.15 * z))
        else:
            roe_score = 1.0 if roe >= 15 else 0.6 if roe >= 8 else 0.0
        f_points += max(0, roe_score)
    # Profit growth up to 1.0
    if profit_growth is not None:
        z = normalized('profit_growth', profit_growth)
        if z is not None:
            pg_score = max(0.0, min(1.0, 0.5 + 0.12 * z))
        else:
            pg_score = 1.0 if profit_growth >= 10 else 0.5 if profit_growth >= 0 else 0.0
        f_points += max(0, pg_score)
    # Debt/equity up to 1.0 (lower better)
    if debt_equity is not None:
        z = normalized('de', debt_equity)
        if z is not None:
            de_score = max(0.0, min(1.0, 0.6 - 0.12 * z))
        else:
            de_score = 1.0 if debt_equity < 0.5 else 0.6 if debt_equity < 1.5 else 0.0
        f_points += max(0, de_score)
    # dividend micro-bonus
    if dividend is not None and dividend > 0:
        f_points += 0.2

    f_score = min(4.0, f_points)

    # Technical points (0-4)
    t_points = 0.0
    if rsi is not None:
        if rsi < 30:
            t_points += 1.5
        elif 30 <= rsi <= 70:
            t_points += 0.8
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            t_points += 1.2
    if adx is not None:
        if adx > 25:
            t_points += 0.8
        elif adx > 15:
            t_points += 0.4
    if price is not None and sma50 is not None:
        if price > sma50:
            t_points += 0.5

    t_score = min(4.0, t_points)

    # Sector / Risk (0-2)
    s_points = 0.0
    if beta is not None:
        s_points += 0.5 if beta < 1.2 else 0.2
    if vol is not None:
        s_points += 1.0 if vol < 30 else 0.3 if vol < 60 else 0.0

    s_score = min(2.0, s_points)

    # Normalize components and combine to 0-10
    fundamental_norm = (f_score / 4.0)
    technical_norm = (t_score / 4.0)
    sector_norm = (s_score / 2.0)

    total = (fundamental_norm * 0.4 + technical_norm * 0.4 + sector_norm * 0.2) * 10.0
    total_score = round(float(total), 2)

    if total_score >= 7.5:
        rec = "STRONG BUY"
    elif total_score >= 5.0:
        rec = "BUY"
    elif total_score >= 3.0:
        rec = "HOLD"
    else:
        rec = "SELL"

    return {
        "Fundamental_Score": round(f_score, 2),
        "Technical_Score": round(t_score, 2),
        "Sector_Score": round(s_score, 2),
        "Total_Score": total_score,
        "Balanced_Recommendation": rec
    }

# ---------------------------
# Technical analysis & data enrichment (adapted from your V32 functions)
# ---------------------------
def analyze_technical_signals_for_row(row):
    """
    Given a row (dict) with Ticker, fetch history and attach RSI, MACD, ADX, SMA50, Volatility etc.
    """
    try:
        ticker = row['Ticker']
        t = yf.Ticker(ticker)
        hist = None
        try:
            hist = t.history(period="1y", interval="1d", auto_adjust=False)
        except Exception:
            hist = None

        # default blanks
        row['RSI'] = None
        row['MACD'] = None
        row['MACD_Signal'] = None
        row['ADX'] = None
        row['SMA_50'] = None
        row['Volatility (%)'] = None

        if hist is not None and not hist.empty:
            hist = hist.sort_index()
            try:
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['RSI'] = ta.momentum.RSIIndicator(hist['Close']).rsi()
                macd = ta.trend.MACD(hist['Close'])
                hist['MACD'] = macd.macd()
                hist['MACD_Signal'] = macd.macd_signal()
                hist['ADX'] = ta.trend.ADXIndicator(hist['High'], hist['Low'], hist['Close']).adx()
                last = hist.iloc[-1]
                row['RSI'] = float(last.get('RSI')) if pd.notna(last.get('RSI')) else None
                row['MACD'] = float(last.get('MACD')) if pd.notna(last.get('MACD')) else None
                row['MACD_Signal'] = float(last.get('MACD_Signal')) if pd.notna(last.get('MACD_Signal')) else None
                row['ADX'] = float(last.get('ADX')) if pd.notna(last.get('ADX')) else None
                row['SMA_50'] = float(last.get('SMA_50')) if pd.notna(last.get('SMA_50')) else None
                row['Volatility (%)'] = float(hist['Close'].pct_change().std() * np.sqrt(252) * 100)
            except Exception:
                pass

        # Yahoo recommendations snapshot (best-effort)
        try:
            t_info = t.info
            row['Yahoo_RecommendationKey'] = t_info.get('recommendationKey')
            row['Yahoo_RecommendationMean'] = t_info.get('recommendationMean')
            row['Yahoo_RecommendationCount'] = t_info.get('numberOfAnalystOpinions')
        except Exception:
            pass

        return row
    except Exception:
        return row

# ---------------------------
# Core fetch + scoring pipeline (parallel)
# ---------------------------
def get_stock_core_data(ticker):
    try:
        t, info = safe_yf_fetch(ticker)
        if not info:
            return None
        data = {
            'Ticker': ticker,
            'Name': info.get('longName', info.get('shortName', ticker)),
            'Sector': info.get('sector', None),
            'Industry': info.get('industry', None),
            'Market Cap (Cr)': (info.get('marketCap')/10000000) if info.get('marketCap') else None,
            'Current Price': info.get('currentPrice') or info.get('regularMarketPrice'),
            'P/E Ratio': info.get('trailingPE'),
            'Forward P/E': info.get('forwardPE'),
            'Debt to Equity': info.get('debtToEquity'),
            'Dividend Yield (%)': (info.get('dividendYield')*100) if info.get('dividendYield') else None,
            'ROE (%)': (info.get('returnOnEquity')*100) if info.get('returnOnEquity') else None,
            'Profit Margin (%)': (info.get('profitMargins')*100) if info.get('profitMargins') else None,
            'Revenue Growth (%)': (info.get('revenueGrowth')*100) if info.get('revenueGrowth') else None,
            'EPS (Trailing)': info.get('trailingEps'),
            'Book Value': info.get('bookValue'),
            'Price to Book (P/B)': info.get('priceToBook'),
            'Beta': info.get('beta'),
            '52W High': info.get('fiftyTwoWeekHigh'),
            '52W Low': info.get('fiftyTwoWeekLow'),
            '1Y Target Price': info.get('targetMeanPrice'),
            'totalInfo': info
        }
        return data
    except Exception:
        return None

def fetch_and_score_universe(stock_list, max_workers=10):
    all_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_stock_core_data, t): t for t in stock_list}
        for f in as_completed(futures):
            try:
                r = f.result()
                if r:
                    all_data.append(r)
            except Exception:
                pass

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    # compute sector stats (use scanned universe)
    sector_stats = compute_sector_stats(df)

    # enrich with technicals and score (parallel)
    enriched = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_technical_signals_for_row, row): row for row in all_data}
        for f in as_completed(futures):
            try:
                r = f.result()
                enriched.append(r)
            except Exception:
                pass

    if not enriched:
        return df

    enriched_df = pd.DataFrame(enriched)

    # compute scores row-wise using sector_stats
    scores = []
    for idx, row in enriched_df.iterrows():
        try:
            score_obj = calculate_scores_for_stock(row.to_dict(), sector_stats)
            for k, v in score_obj.items():
                enriched_df.at[idx, k] = v
        except Exception:
            enriched_df.at[idx, 'Fundamental_Score'] = None
            enriched_df.at[idx, 'Technical_Score'] = None
            enriched_df.at[idx, 'Sector_Score'] = None
            enriched_df.at[idx, 'Total_Score'] = None
            enriched_df.at[idx, 'Balanced_Recommendation'] = None

    # compute simple Filter_Score & Expert_Score placeholders if you use earlier filters
    # (we won't change your filter flow — these are additional KPIs)
    return enriched_df

# ---------------------------
# Backtesting helpers (forward quick-check)
# ---------------------------
def compute_future_return(ticker, start_date, days_forward=30):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start_date - timedelta(days=7), end=start_date + timedelta(days=days_forward+7), interval="1d")
        if hist is None or hist.empty:
            return None
        hist = hist.sort_index()
        close_on = None
        for dt, row in hist.iterrows():
            if dt.date() >= start_date.date():
                close_on = row['Close']
                break
        if close_on is None:
            return None
        target_date = start_date + timedelta(days=days_forward)
        close_target = None
        for dt, row in hist.iterrows():
            if dt.date() >= target_date.date():
                close_target = row['Close']
                break
        if close_target is None:
            return None
        return (close_target - close_on) / close_on * 100.0
    except Exception:
        return None

def backtest_current_picks(df, days_forward_list=[30, 90]):
    picks = df[df['Balanced_Recommendation'] == 'STRONG BUY']
    results = []
    start_date = datetime.utcnow()
    for _, row in picks.iterrows():
        tkr = row['Ticker']
        entry = {'Ticker': tkr}
        for d in days_forward_list:
            entry[f'Forward_{d}d_Return_%'] = compute_future_return(tkr, start_date, days_forward=d)
        results.append(entry)
    return pd.DataFrame(results)

# ---------------------------
# Radar chart helper
# ---------------------------
def normalize_value(val, vmin, vmax, invert=False):
    try:
        if val is None or np.isnan(val):
            return None
        x = max(vmin, min(val, vmax))
        norm = (x - vmin) / (vmax - vmin)
        if invert:
            return 1 - norm
        return norm
    except Exception:
        return None

def radar_chart_compare(stock_row, peer_df=None):
    metrics = {
        'PE': ('P/E Ratio', False),
        'ROE': ('ROE (%)', True),
        'ProfitGrowth': ('3Y Profit Growth (%)', True),
        'DebtEq': ('Debt to Equity', False),
        'Volatility': ('Volatility (%)', False)
    }
    stock_vals = []
    peer_vals = []
    for k, (col, higher_is_better) in metrics.items():
        s_val = None
        try:
            s_val = float(stock_row.get(col)) if col in stock_row and stock_row.get(col) is not None else None
        except Exception:
            s_val = None
        peer_mean = None
        if peer_df is not None and col in peer_df.columns:
            try:
                peer_mean = pd.to_numeric(peer_df[col], errors='coerce').dropna().mean()
            except Exception:
                peer_mean = None

        if col == 'P/E Ratio':
            s_norm = normalize_value(s_val, 0, 50, invert=(not higher_is_better))
            p_norm = normalize_value(peer_mean, 0, 50, invert=(not higher_is_better))
        elif col == 'ROE (%)':
            s_norm = normalize_value(s_val, -10, 40, invert=(not higher_is_better))
            p_norm = normalize_value(peer_mean, -10, 40, invert=(not higher_is_better))
        elif col == '3Y Profit Growth (%)':
            s_norm = normalize_value(s_val, -50, 100, invert=(not higher_is_better))
            p_norm = normalize_value(peer_mean, -50, 100, invert=(not higher_is_better))
        elif col == 'Debt to Equity':
            s_norm = normalize_value(s_val, 0, 3, invert=(not higher_is_better))
            p_norm = normalize_value(peer_mean, 0, 3, invert=(not higher_is_better))
        elif col == 'Volatility (%)':
            s_norm = normalize_value(s_val, 0, 100, invert=(not higher_is_better))
            p_norm = normalize_value(peer_mean, 0, 100, invert=(not higher_is_better))
        else:
            s_norm = normalize_value(s_val, 0, 1)
            p_norm = normalize_value(peer_mean, 0, 1)

        stock_vals.append(0 if s_norm is None else s_norm)
        peer_vals.append(0 if p_norm is None else p_norm)

    categories = list(metrics.keys())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=stock_vals, theta=categories, fill='toself', name='Stock'))
    fig.add_trace(go.Scatterpolar(r=peer_vals, theta=categories, fill='toself', name='Peers (avg)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, height=480)
    return fig

# ---------------------------
# Main app flow (based on your V32 UI, unchanged look & behavior)
# ---------------------------
def main():
    # Session state defaults
    if 'page' not in st.session_state:
        st.session_state.page = 'setup'
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    if 'tech_analysis' not in st.session_state:
        st.session_state.tech_analysis = None
    if 'screening_timestamp' not in st.session_state:
        st.session_state.screening_timestamp = None
    if 'tech_timestamp' not in st.session_state:
        st.session_state.tech_timestamp = None
    if 'individual_refresh_times' not in st.session_state:
        st.session_state.individual_refresh_times = {}

    st.markdown('<h1 class="main-header">📈 Indian Stock Screener with Technical Analysis</h1>', unsafe_allow_html=True)

    # Setup Page
    if st.session_state.page == 'setup':
        st.markdown("Filter Indian stocks (Large / Mid / Small) with fundamental & technical analysis")

        # Sidebar Filters (keeps original UI)
        st.sidebar.header("🎯 Filtering Criteria")
        st.sidebar.subheader("✅ Mandatory Filters")

        filters = {}
        filters['roce'] = st.sidebar.checkbox("ROCE Filter", value=True)
        filters['roce_min'] = st.sidebar.slider("Minimum ROCE (%)", 10, 30, 15, 1) if filters['roce'] else 15
        filters['fcf'] = st.sidebar.checkbox("Positive FCF (3/4 Quarters)", value=True)
        filters['pe'] = st.sidebar.checkbox("P/E Ratio Filter", value=True)
        filters['pe_max'] = st.sidebar.slider("Maximum P/E", 10, 100, 40, 5) if filters['pe'] else 40
        filters['profit_growth'] = st.sidebar.checkbox("Profit Growth Filter", value=True)
        filters['profit_growth_min'] = st.sidebar.slider("Min 3Y Profit Growth (%)", 0, 30, 10, 1) if filters['profit_growth'] else 10
        filters['debt_equity'] = st.sidebar.checkbox("Debt/Equity Filter", value=True)
        filters['debt_equity_max'] = st.sidebar.slider("Max Debt/Equity", 0.0, 2.0, 1.0, 0.1) if filters['debt_equity'] else 1.0

        st.sidebar.subheader("🎯 Preferred Filters")
        filters['dividend'] = st.sidebar.checkbox("Dividend Payer", value=False)
        filters['roe'] = st.sidebar.checkbox("ROE Filter", value=False)
        filters['roe_min'] = st.sidebar.slider("Minimum ROE (%)", 10, 30, 15, 1) if filters['roe'] else 15

        st.sidebar.subheader("⚖ Risk Filters (Optional)")
        filters['risk'] = st.sidebar.checkbox("Apply Volatility Filter", value=False)
        if filters['risk']:
            filters['max_vol'] = st.sidebar.slider("Max Annual Volatility (%)", 10, 80, 40, 5)

        st.sidebar.subheader("⚙ Filter Mode")
        filter_mode = st.sidebar.radio(
            "Select Filtering Mode",
            ["Flexible (Recommended)", "Strict (All criteria must pass)"]
        )
        filters['strict_mode'] = (filter_mode == "Strict (All criteria must pass)")

        if not filters['strict_mode']:
            filters['min_score'] = st.sidebar.slider("Minimum Score (out of 8.5)", 1.0, 8.5, 3.0, 0.5)

        st.sidebar.subheader("📊 Stock Universe")
        stock_category = st.sidebar.multiselect(
            "Select Categories",
            ["Large Cap", "Mid Cap", "Small Cap"],
            default=["Large Cap"]
        )

        stock_list = []
        if "Large Cap" in stock_category:
            stock_list.extend(LARGE_CAP_TICKERS)
        if "Mid Cap" in stock_category:
            stock_list.extend(MID_CAP_TICKERS)
        if "Small Cap" in stock_category:
            stock_list.extend(SMALL_CAP_TICKERS)
        stock_list = list(set(stock_list))

        # Custom stocks option
        use_custom = st.sidebar.checkbox("Add custom tickers")
        if use_custom:
            custom_stocks = st.sidebar.text_area(
                "Enter ticker symbols (one per line, with .NS suffix)",
                placeholder="Example:\nRELIANCE.NS\nTCS.NS\nINFY.NS",
                height=100
            )
            if custom_stocks:
                custom_list = [s.strip().upper() for s in custom_stocks.split('\n') if s.strip()]
                stock_list.extend(custom_list)
                stock_list = list(set(stock_list))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks to Scan", len(stock_list))
        with col2:
            st.metric("Large Cap", len([s for s in stock_list if s in LARGE_CAP_TICKERS]))
        with col3:
            st.metric("Mid+Small Cap", len([s for s in stock_list if s not in LARGE_CAP_TICKERS]))

        if st.button("🔍 Start Screening", type="primary", use_container_width=True):
            st.info("Fetching stock data... This may take a while depending on universe size.")
            progress_bar = st.progress(0)
            status_text = st.empty()

            all_data = []
            try:
                # use parallel fetch and scoring
                results = fetch_and_score_universe(stock_list, max_workers=10)
                if isinstance(results, pd.DataFrame) and not results.empty:
                    filtered_df = results.copy()
                else:
                    filtered_df = pd.DataFrame()
            except Exception:
                filtered_df = pd.DataFrame()

            progress_bar.empty()
            status_text.empty()

            if not filtered_df.empty:
                # store for results page
                st.session_state.filtered_data = filtered_df
                st.session_state.screening_timestamp = datetime.now()
                st.session_state.page = 'results'
                st.rerun()
            else:
                st.error("❌ Unable to fetch or score data. Try smaller universe or check connectivity.")

    # Results Page
    elif st.session_state.page == 'results':
        st.sidebar.header("📊 Screening Complete")

        if st.session_state.screening_timestamp:
            time_elapsed = datetime.now() - st.session_state.screening_timestamp
            minutes_ago = int(time_elapsed.total_seconds() / 60)
            hours_ago = int(minutes_ago / 60)
            if hours_ago > 0:
                freshness_text = f"🕐 Data fetched {hours_ago}h {minutes_ago % 60}m ago"
            else:
                freshness_text = f"🕐 Data fetched {minutes_ago}m ago"
            st.sidebar.info(freshness_text)
            st.sidebar.caption(f"Last updated: {st.session_state.screening_timestamp.strftime('%I:%M %p, %d %b %Y')}")

        if st.sidebar.button("🔄 New Screening", type="primary", use_container_width=True):
            st.session_state.page = 'setup'
            st.session_state.filtered_data = None
            st.session_state.tech_analysis = None
            st.session_state.screening_timestamp = None
            st.session_state.tech_timestamp = None
            st.session_state.individual_refresh_times = {}
            st.rerun()

        filtered_df = st.session_state.filtered_data

        if filtered_df is not None and not filtered_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Stocks Passing Filters (scanned)", len(filtered_df))
            with col2:
                if st.button("📊 Generate Technical Analysis for All Stocks", use_container_width=True):
                    st.session_state.page = 'tech_analysis'
                    st.rerun()

            if st.session_state.screening_timestamp:
                time_elapsed = datetime.now() - st.session_state.screening_timestamp
                minutes_ago = int(time_elapsed.total_seconds() / 60)
                if minutes_ago < 60:
                    st.caption(f"⏱ Fundamental data age: {minutes_ago} minutes old")
                else:
                    hours_ago = int(minutes_ago / 60)
                    st.caption(f"⏱ Fundamental data age: {hours_ago}h {minutes_ago % 60}m old")

            st.subheader("✅ Scored Stocks (Top first)")
            display_df = filtered_df.copy()
            # ensure our new columns exist
            extra_cols = ['Fundamental_Score','Technical_Score','Sector_Score','Total_Score','Balanced_Recommendation']
            for c in extra_cols:
                if c not in display_df.columns:
                    display_df[c] = None

            cols_order = ['Ticker','Name','Sector','Current Price','P/E Ratio','ROE (%)','Debt to Equity',
                          'Volatility (%)'] + extra_cols + ['Yahoo_RecommendationKey','Yahoo_RecommendationMean']
            # ensure present
            display_df = display_df[[c for c in cols_order if c in display_df.columns]]

            st.dataframe(display_df.sort_values('Total_Score', ascending=False).reset_index(drop=True), height=420)

            csv = display_df.to_csv(index=False)
            st.download_button("📥 Download scanned results CSV", csv, f"scanned_results_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        else:
            st.warning("No stocks scanned yet. Run a screening to see results.")

    # Technical Analysis Page (keeps existing flow)
    elif st.session_state.page == 'tech_analysis':
        st.sidebar.header("📊 Technical Analysis")
        if st.sidebar.button("⬅ Back to Results", use_container_width=True):
            st.session_state.page = 'results'
            st.rerun()
        if st.sidebar.button("🔄 New Screening", use_container_width=True):
            st.session_state.page = 'setup'
            st.session_state.filtered_data = None
            st.session_state.tech_analysis = None
            st.session_state.screening_timestamp = None
            st.session_state.tech_timestamp = None
            st.session_state.individual_refresh_times = {}
            st.rerun()
        if st.sidebar.button("🔄 Refresh All Technical Data", use_container_width=True):
            st.session_state.tech_analysis = None
            st.session_state.tech_timestamp = None
            st.session_state.individual_refresh_times = {}
            st.rerun()

        filtered_df = st.session_state.filtered_data
        if filtered_df is None or filtered_df.empty:
            st.warning("No scanned data available for technical analysis.")
            return

        if st.session_state.tech_analysis is None:
            st.info("Analyzing technical signals for all scanned stocks...")
            progress_bar = st.progress(0)
            tech_results = []
            rows = filtered_df.to_dict('records')
            total = len(rows)
            for idx, row in enumerate(rows):
                ticker = row['Ticker']
                tech = analyze_technical_signals_for_row(row)
                if tech:
                    # minimal summary
                    tech_results.append({
                        'Ticker': ticker,
                        'Name': tech.get('Name',''),
                        'Current Price': tech.get('Current Price'),
                        'RSI': tech.get('RSI'),
                        'Signal': tech.get('Balanced_Recommendation',''),
                        'Recommendation': tech.get('Balanced_Recommendation',''),
                        'Score': tech.get('Total_Score', 0),
                        'data': None  # large data omitted here
                    })
                progress_bar.progress((idx + 1) / max(1,total))
                time.sleep(0.02)
            progress_bar.empty()
            st.session_state.tech_analysis = pd.DataFrame(tech_results)
            st.session_state.tech_timestamp = datetime.now()
            st.success("✅ Technical analysis complete!")

        tech_df = st.session_state.tech_analysis

        # show age
        if st.session_state.tech_timestamp:
            time_elapsed = datetime.now() - st.session_state.tech_timestamp
            minutes_ago = int(time_elapsed.total_seconds() / 60)
            if minutes_ago < 1:
                age_text = "⏱ Technical data: Less than 1 minute old (Live)"
            elif minutes_ago < 5:
                age_text = f"⏱ Technical data: {minutes_ago} minutes old (Recent)"
            else:
                age_text = f"⏱ Technical data: {minutes_ago} minutes old"
            st.markdown(f"**{age_text}**")

        st.subheader("📊 Technical Analysis Summary - All Stocks")
        summary_df = tech_df[['Ticker','Name','Current Price','RSI','Signal','Recommendation','Score']].copy()
        st.dataframe(summary_df.style.format({'Current Price':'₹{:.2f}','RSI':'{:.1f}','Score':'{:.0f}'}), height=420)

        # Download
        csv_metadata = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nStocks screened: {len(filtered_df)}\n"
        csv_body = summary_df.to_csv(index=False)
        csv = f"# Indian Stock Screener (Expert Mode)\n# {csv_metadata}\n{csv_body}"
        st.download_button("📥 Download Technical Analysis", csv, f"technical_analysis_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

        # Detailed Analysis selection
        st.markdown("---")
        st.subheader("📈 Detailed Stock Analysis")
        stock_names = [f"{row['Ticker']} - {row['Name']}" for _, row in tech_df.iterrows()]
        col1, col2 = st.columns([3,1])
        with col1:
            selected = st.selectbox("Select stock for detailed view:", stock_names, key='stock_detail_selector')

        if selected:
            ticker = selected.split(' - ')[0]
            # refresh button
            with col2:
                st.write("")
                if st.button(f"🔄 Refresh {ticker}", use_container_width=True, key=f'refresh_{ticker}'):
                    with st.spinner(f"Refreshing {ticker}..."):
                        # find row in filtered_df and update
                        try:
                            row = next(r for r in filtered_df.to_dict('records') if r['Ticker']==ticker)
                        except StopIteration:
                            row = None
                        if row:
                            updated = analyze_technical_signals_for_row(row)
                            # recompute sector stats using filtered_df
                            sector_stats = compute_sector_stats(filtered_df)
                            score_obj = calculate_scores_for_stock(updated, sector_stats)
                            updated.update(score_obj)
                            # update session_state.filtered_data
                            idx = filtered_df[filtered_df['Ticker']==ticker].index
                            if len(idx)>0:
                                for k,v in updated.items():
                                    filtered_df.at[idx[0], k] = v
                                st.session_state.filtered_data = filtered_df
                                st.session_state.individual_refresh_times[ticker] = datetime.now()
                                st.success(f"✅ {ticker} refreshed!")
                                st.rerun()

            # pull the current data row
            try:
                stock_fund = filtered_df[filtered_df['Ticker']==ticker].iloc[0]
            except Exception:
                stock_fund = None

            # show last refresh times
            if ticker in st.session_state.individual_refresh_times:
                last_refresh = st.session_state.individual_refresh_times[ticker]
                seconds_ago = int((datetime.now() - last_refresh).total_seconds())
                if seconds_ago < 60:
                    st.success(f"🕐 {ticker} data: {seconds_ago} seconds old (Just refreshed!)")
                else:
                    st.info(f"🕐 {ticker} data: {int(seconds_ago/60)} minutes old | Last refreshed: {last_refresh.strftime('%I:%M:%S %p')}")

            rec_class = "neutral-signal"
            if stock_fund is not None:
                rec = stock_fund.get('Balanced_Recommendation','')
                if rec and 'BUY' in rec:
                    rec_class = "buy-signal"
                elif rec and 'SELL' in rec:
                    rec_class = "sell-signal"
                st.markdown(f'<div class="{rec_class}">Recommendation: {rec} (Score: {stock_fund.get("Total_Score")})</div>', unsafe_allow_html=True)

            st.markdown("---")
            # metrics
            if stock_fund is not None:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"₹{stock_fund.get('Current Price',0):.2f}")
                with col2:
                    st.metric("P/E Ratio", f"{stock_fund.get('P/E Ratio', 'N/A')}")
                with col3:
                    st.metric("ROE (%)", f"{stock_fund.get('ROE (%)','N/A')}")
                with col4:
                    st.metric("ROCE (%)", f"{stock_fund.get('ROCE (%)','N/A') if 'ROCE (%)' in stock_fund else 'N/A'}")

                # radar vs peers
                peer_df = None
                if stock_fund.get('Sector') and 'filtered_df' in st.session_state and st.session_state.filtered_data is not None:
                    try:
                        peer_df = st.session_state.filtered_data[st.session_state.filtered_data['Sector']==stock_fund.get('Sector')]
                    except Exception:
                        peer_df = st.session_state.filtered_data
                else:
                    peer_df = st.session_state.filtered_data

                try:
                    radar = radar_chart_compare(stock_fund.to_dict(), peer_df)
                    st.plotly_chart(radar, use_container_width=True)
                except Exception:
                    pass

                # price chart & indicators (pull history for chart)
                tkn, info = safe_yf_fetch(ticker)
                hist = None
                if tkn:
                    try:
                        hist = tkn.history(period="1y", interval="1d")
                        hist = hist.sort_index()
                        hist['SMA_50'] = hist['Close'].rolling(50).mean()
                    except Exception:
                        hist = None
                if hist is not None and not hist.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'))
                    if 'SMA_50' in hist.columns:
                        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50'))
                    fig.update_layout(title=f"{ticker} Price & SMA50", height=500, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

    # Info sections (unchanged)
    with st.expander("ℹ Understanding Technical Indicators"):
        st.markdown("""
        ### Key Technical Indicators
        - RSI <30: Oversold (potential buy), >70: Overbought (potential sell)
        - SMA 20/50/200 cross indicates trend shifts
        - MACD cross: momentum signal
        - ADX > 25: strong trend
        """)

    with st.expander("ℹ About Fundamental Filtering"):
        st.markdown("""
        ### Filtering Criteria Summary
        - ROCE, Positive FCF, P/E, Profit Growth, Debt/Equity used as filters
        - Scoring system (Flexible) awards points for each criterion
        - New Balanced Score: Fundamental (40%) + Technical (40%) + Sector (20%) -> 0-10
        """)

if __name__ == "__main__":
    main()
