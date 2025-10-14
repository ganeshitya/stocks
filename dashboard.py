# advanced_screener_app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import ta
from ta.volatility import AverageTrueRange
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore

# -----------------------
# Config / Constants
# -----------------------
API_URL = "http://localhost:8000"  # your MCP server (unchanged)
FINNHUB_API_KEY = ""  # Optional: set your Finnhub API key here
ALPHAVANTAGE_API_KEY = ""  # Optional
MAX_WORKERS = 8

st.set_page_config(page_title="Advanced Indian Stock Screener", layout="wide")

# --- helper: MCP call (keeps your original) ---
def call_mcp(method: str, params: dict = None):
    payload = {"method": method}
    if params:
        payload["params"] = params
    try:
        resp = requests.post(f"{API_URL}/mcp", json=payload, timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

# -----------------------
# Data fetch helpers
# -----------------------
def safe_yf_ticker(ticker, retries=2, pause=0.8):
    for i in range(retries):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if info:
                return t, info
        except Exception:
            time.sleep(pause)
    return None, {}

def get_yahoo_recommendations(ticker):
    """
    Uses yfinance's 'recommendations' and 'info' fields to capture analyst recommendations.
    Returns a dict with counts and a normalized score.
    """
    try:
        t, info = safe_yf_ticker(ticker)
        rec = {}
        # analyst recommendation trends (yfinance Ticker.recommendations)
        try:
            tmp = t.recommendations
            if tmp is not None and not tmp.empty:
                # Most recent analyst action counts
                last_analysts = tmp.tail(12)
                counts = last_analysts['To Grade'].value_counts().to_dict() if 'To Grade' in last_analysts.columns else {}
                rec['recent_counts'] = counts
        except Exception:
            rec['recent_counts'] = {}

        # info fields: recommendationKey/recommendationMean
        rec['recommendationKey'] = info.get('recommendationKey')  # e.g., 'buy', 'hold'
        rec['recommendationMean'] = info.get('recommendationMean')  # numeric mean (1-5)
        rec['recommendationCount'] = info.get('numberOfAnalystOpinions')

        return rec
    except Exception:
        return {}

def get_finnhub_recommendation(ticker, apikey=FINNHUB_API_KEY):
    """
    Optional: Query Finnhub for analyst recommendations if API key provided.
    ticker should be in symbol format (e.g., NSE:RELIANCE or RELIANCE.NS); Finnhub uses 'RELIANCE.NS' etc depending on coverage.
    """
    if not apikey:
        return {}
    try:
        # Finnhub endpoint: /stock/recommendation?symbol=
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={apikey}"
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

# -----------------------
# Core scoring model
# -----------------------
def calculate_scores_for_stock(data_row, sector_stats=None):
    """
    Compute Fundamental (40%), Technical (40%), Sector/Risk (20%) based score (0-10),
    with z-score normalization option via `sector_stats` (dict of series mean/std)
    data_row: dictionary-like with fields used below
    sector_stats: dict containing mean/std for metrics (optional)
    """
    # extract raw metrics (tolerant)
    pe = safe_get_float(data_row, ['P/E Ratio', 'P/E', 'pe_ratio'])
    roe = safe_get_float(data_row, ['ROE (%)', 'ROE', 'roe'])
    profit_growth = safe_get_float(data_row, ['3Y Profit Growth (%)', '3Y Profit Growth', 'profit_growth'])
    debt_equity = safe_get_float(data_row, ['Debt to Equity', 'debtToEquity', 'Debt/Equity'])
    dividend = safe_get_float(data_row, ['Dividend Yield (%)', 'Dividend Yield', 'dividend_yield'])
    rsi = safe_get_float(data_row, ['RSI', 'rsi'])
    macd = safe_get_float(data_row, ['MACD', 'macd'])
    macd_signal = safe_get_float(data_row, ['MACD_Signal', 'macd_signal', 'Signal'])
    adx = safe_get_float(data_row, ['ADX', 'adx'])
    price = safe_get_float(data_row, ['Current Price', 'currentPrice', 'Current_Price'])
    sma50 = safe_get_float(data_row, ['SMA_50', 'SMA50'])

    # Optionally z-score normalize each metric vs sector stats to remove scale differences
    def normalized(metric_name, value):
        if sector_stats and metric_name in sector_stats:
            mean, std = sector_stats[metric_name]
            try:
                if std and std > 0:
                    z = (value - mean) / std
                    return z
            except Exception:
                return None
        return None

    # --- Fundamental subscore (max 4 points) ---
    f_points = 0.0
    # lower P/E relative to sector is better -> we interpret using zscore if available
    pe_score = None
    if pe and pe > 0:
        z_pe = normalized('pe', pe)
        # if z_pe available, convert to score: lower z (below mean) = better
        if z_pe is not None:
            pe_score = max(0, 1.5 - (z_pe * 0.5))  # scaled
        else:
            pe_score = 1.0 if pe < 25 else 0.5 if pe < 40 else 0.0
        f_points += min(1.5, max(0, pe_score))

    # ROE
    if roe is not None:
        z_roe = normalized('roe', roe)
        if z_roe is not None:
            roe_score = max(0, min(1.0, 0.5 + (z_roe * 0.2)))  # scale
        else:
            roe_score = 1.0 if roe >= 15 else 0.5 if roe >= 8 else 0.0
        f_points += roe_score

    # profit growth
    if profit_growth is not None:
        z_pg = normalized('profit_growth', profit_growth)
        if z_pg is not None:
            pg_score = max(0, min(1.0, 0.5 + (z_pg * 0.15)))
        else:
            pg_score = 1.0 if profit_growth >= 10 else 0.5 if profit_growth >= 0 else 0.0
        f_points += pg_score

    # Debt to Equity (lower is better)
    if debt_equity is not None:
        z_de = normalized('de', debt_equity)
        if z_de is not None:
            de_score = max(0, min(1.0, 0.5 - (z_de * 0.15)))
        else:
            de_score = 1.0 if debt_equity < 0.5 else 0.5 if debt_equity < 1.5 else 0.0
        f_points += de_score

    # dividend small boost
    if dividend is not None and dividend > 0:
        f_points += 0.2

    # f_points is roughly 0-4.2; normalize to 0-4.0
    f_score = min(4.0, f_points)

    # --- Technical subscore (max 4 points) ---
    t_points = 0.0
    # RSI: oversold (<30) gives points; neutral gives medium points
    if rsi is not None:
        if rsi < 30:
            t_points += 1.5
        elif 30 <= rsi <= 70:
            t_points += 0.8
        else:
            t_points += 0.0

    # MACD cross: if MACD > signal -> positive
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            t_points += 1.2
    # ADX strength
    if adx is not None:
        if adx > 25:
            t_points += 0.8
        elif adx > 15:
            t_points += 0.4

    # price above sma50
    if price is not None and sma50 is not None:
        if price > sma50:
            t_points += 0.5

    t_score = min(4.0, t_points)

    # --- Sector / Risk (max 2 points) ---
    s_points = 0.0
    beta = safe_get_float(data_row, ['Beta', 'beta'])
    vol = safe_get_float(data_row, ['Volatility (%)', 'volatility', 'Volatility'])

    if beta is not None:
        s_points += 0.5 if beta < 1.2 else 0.2
    if vol is not None:
        # lower volatility better
        s_points += 1.0 if vol < 30 else 0.3 if vol < 60 else 0.0

    s_score = min(2.0, s_points)

    # Weighted total: Fundamental 40%, Technical 40%, Sector 20% but normalized to 0-10
    # scale each to its weight: f_score (0-4) maps to 0-4*0.4? simpler: compute ratio and weight
    fundamental_norm = (f_score / 4.0) if f_score is not None else 0
    technical_norm = (t_score / 4.0) if t_score is not None else 0
    sector_norm = (s_score / 2.0) if s_score is not None else 0

    total = (fundamental_norm * 0.4 + technical_norm * 0.4 + sector_norm * 0.2) * 10.0
    total_score = round(float(total), 2)

    # Recommendation mapping
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

def safe_get_float(row, keys):
    for k in keys:
        if isinstance(row, dict):
            v = row.get(k)
        else:
            try:
                v = row[k]
            except Exception:
                v = None
        try:
            if v is None:
                continue
            return float(v)
        except Exception:
            continue
    return None

# -----------------------
# Sector peer stats (Z-score)
# -----------------------
def compute_sector_stats(df, metric_map=None):
    """
    df: dataframe of stocks (rows) containing core metrics.
    metric_map: map of canonical metric names to df columns e.g. {'pe': 'P/E Ratio', ...}
    returns: dict {metric: (mean, std)}
    """
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

# -----------------------
# Aggregation functions
# -----------------------
def fetch_and_score_universe(stock_list):
    """
    Fetch data (yfinance) for each ticker in stock_list concurrently,
    compute fundamental fields, compute scores with sector zscore, return dataframe.
    """
    all_data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_stock_core_data, t): t for t in stock_list}
        for f in as_completed(futures):
            try:
                res = f.result()
                if res:
                    all_data.append(res)
            except Exception:
                pass
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    # compute sector stats (we'll use entire list as 'peer group' unless sector present)
    sector_stats = compute_sector_stats(df)
    # now compute per-row technicals via yfinance history
    # compute basic indicators and update score
    scored_rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(add_technical_and_score, row, sector_stats): row['Ticker'] for row in all_data}
        for f in as_completed(futures):
            try:
                r = f.result()
                if r:
                    scored_rows.append(r)
            except Exception:
                pass
    if scored_rows:
        return pd.DataFrame(scored_rows)
    return pd.DataFrame()

def get_stock_core_data(ticker):
    """
    Lightweight fetch from yfinance: returns dictionary with core fundamental fields.
    """
    try:
        t, info = safe_yf_ticker(ticker)
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
            'Debt to Equity': info.get('debtToEquity') or info.get('debtToEquity'),
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

def add_technical_and_score(row, sector_stats):
    """
    Adds technical indicators (RSI, MACD, ADX, SMA50), updates row with scores & recommendation
    """
    try:
        ticker = row['Ticker']
        t = yf.Ticker(ticker)
        hist = None
        try:
            hist = t.history(period="1y", interval="1d", auto_adjust=False)
        except Exception:
            hist = None
        # set defaults
        row['RSI'] = None
        row['MACD'] = None
        row['MACD_Signal'] = None
        row['ADX'] = None
        row['SMA_50'] = None
        if hist is not None and not hist.empty:
            hist = hist.sort_index()
            # compute moving averages and RSI
            try:
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                rsi_series = ta.momentum.RSIIndicator(hist['Close']).rsi()
                hist['RSI'] = rsi_series
                macd = ta.trend.MACD(hist['Close'])
                hist['MACD'] = macd.macd()
                hist['MACD_Signal'] = macd.macd_signal()
                adx_series = ta.trend.ADXIndicator(hist['High'], hist['Low'], hist['Close']).adx()
                hist['ADX'] = adx_series
                # attach latest
                last = hist.iloc[-1]
                row['RSI'] = float(last.get('RSI')) if pd.notna(last.get('RSI')) else None
                row['MACD'] = float(last.get('MACD')) if pd.notna(last.get('MACD')) else None
                row['MACD_Signal'] = float(last.get('MACD_Signal')) if pd.notna(last.get('MACD_Signal')) else None
                row['ADX'] = float(last.get('ADX')) if pd.notna(last.get('ADX')) else None
                row['SMA_50'] = float(last.get('SMA_50')) if pd.notna(last.get('SMA_50')) else None
                # annualized volatility
                row['Volatility (%)'] = float(hist['Close'].pct_change().std() * np.sqrt(252) * 100)
            except Exception:
                pass
        else:
            # fallback: attempt to get 6mo
            pass

        # compute z-score based scores and final total
        score_obj = calculate_scores_for_stock(row, sector_stats)
        row.update(score_obj)

        # attach yahoo recommendations too
        try:
            rec = get_yahoo_recommendations(ticker)
            row['Yahoo_RecommendationKey'] = rec.get('recommendationKey')
            row['Yahoo_RecommendationMean'] = rec.get('recommendationMean')
            row['Yahoo_RecommendationCount'] = rec.get('recommendationCount')
        except Exception:
            pass

        return row
    except Exception:
        return row

# -----------------------
# Backtesting helpers
# -----------------------
def compute_future_return(ticker, start_date, days_forward=30):
    """
    For ticker, compute forward % return from start_date to start_date + days_forward
    returns None if insufficient data.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start_date - timedelta(days=7), end=start_date + timedelta(days=days_forward+7), interval="1d")
        if hist is None or hist.empty:
            return None
        hist = hist.sort_index()
        # find first close on/after start_date
        close_on = None
        for dt, row in hist.iterrows():
            if dt.date() >= start_date.date():
                close_on = row['Close']
                break
        if close_on is None:
            return None
        # find close at target day
        target_date = start_date + timedelta(days=days_forward)
        close_target = None
        # find first available after target_date
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
    """
    For current Strong Buy picks in df, compute their forward returns for days_forward_list
    Returns a dataframe with results and simple summary metrics (avg, median).
    """
    picks = df[df['Balanced_Recommendation'] == 'STRONG BUY']
    results = []
    for _, row in picks.iterrows():
        ticker = row['Ticker']
        # use today's date as start, compute forward returns (this is forward-looking; for historical backtest
        # you'd want to simulate scanning in the past and compute forward returns — heavy)
        start_date = datetime.utcnow()
        entry = {'Ticker': ticker}
        for d in days_forward_list:
            r = compute_future_return(ticker, start_date, days_forward=d)
            entry[f'Forward_{d}d_Return_%'] = r
        results.append(entry)
    return pd.DataFrame(results)

# -----------------------
# Radar chart visualization
# -----------------------
def radar_chart_compare(stock_row, peer_df=None):
    """
    Creates a radar chart comparing key normalized metrics of stock vs peer averages.
    stock_row: dict/Series
    peer_df: dataframe of peers (if None uses entire df)
    """
    metrics = {
        'PE': ('P/E Ratio', False),  # lower better -> invert
        'ROE': ('ROE (%)', True),
        'ProfitGrowth': ('3Y Profit Growth (%)', True),
        'DebtEq': ('Debt to Equity', False),  # lower better -> invert
        'Volatility': ('Volatility (%)', False)  # lower better -> invert
    }

    stock_vals = []
    peer_vals = []
    for k, (col, higher_is_better) in metrics.items():
        s_val = safe_get_float(stock_row, [col])
        peer_mean = None
        if peer_df is not None and col in peer_df.columns:
            peer_mean = pd.to_numeric(peer_df[col], errors='coerce').dropna().mean()
        # normalize to 0-1 scale by clipping reasonable ranges
        # define metric-specific normalization windows:
        if col in ['P/E Ratio']:
            s_norm = normalize_value(s_val, 0, 50, invert=(not higher_is_better))
            p_norm = normalize_value(peer_mean, 0, 50, invert=(not higher_is_better))
        elif col in ['ROE (%)']:
            s_norm = normalize_value(s_val, -10, 40, invert=(not higher_is_better))
            p_norm = normalize_value(peer_mean, -10, 40, invert=(not higher_is_better))
        elif col in ['3Y Profit Growth (%)']:
            s_norm = normalize_value(s_val, -50, 100, invert=(not higher_is_better))
            p_norm = normalize_value(peer_mean, -50, 100, invert=(not higher_is_better))
        elif col in ['Debt to Equity']:
            s_norm = normalize_value(s_val, 0, 3, invert=(not higher_is_better))
            p_norm = normalize_value(peer_mean, 0, 3, invert=(not higher_is_better))
        elif col in ['Volatility (%)']:
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

def normalize_value(val, vmin, vmax, invert=False):
    try:
        if val is None or np.isnan(val):
            return None
        # clip
        x = max(vmin, min(val, vmax))
        norm = (x - vmin) / (vmax - vmin)
        if invert:
            return 1 - norm
        return norm
    except Exception:
        return None

# -----------------------
# Streamlit UI
# -----------------------
st.title("🔬 Advanced Indian Stock Screener (MCP + Multi-source Recommendations + Scoring)")

# Left: MCP controls (preserve original)
st.header("🔌 MCP Server (unchanged)")
st.write("This panel keeps your original MCP calls.")
if st.button("Check MCP Status"):
    st.json(call_mcp("get_alpha_vantage_status"))

st.header("💼 MCP Portfolio / Recommendations")
if st.button("Get MCP Portfolio Summary"):
    st.json(call_mcp("get_portfolio_summary", {"limit": 10, "summary": True}))
if st.button("Get MCP Technical / Recommendations (example)"):
    st.json(call_mcp("get_stock_recommendations", {"criteria":"growth", "limit": 5}))

st.markdown("---")

# Right: new aggregator
st.header("📥 Aggregate Buy Recommendations & Scoring")

universe_input = st.text_area("Enter tickers to scan (one per line). Use .NS suffix. Example: RELIANCE.NS", height=120,
                             value="RELIANCE.NS\nTCS.NS\nINFY.NS\nDRREDDY.NS")
stock_list = [s.strip().upper() for s in universe_input.splitlines() if s.strip()]

col1, col2, col3 = st.columns(3)
with col1:
    run_scan = st.button("🔍 Fetch & Score Universe", key="scan")
with col2:
    run_recs = st.button("📥 Fetch Buy Recommendations (Yahoo + Finnhub) ", key="recs")
with col3:
    run_backtest = st.button("🧪 Backtest Strong Buys (30/90d forward)", key="backtest")

# Placeholders
scan_placeholder = st.empty()
recs_placeholder = st.empty()
backtest_placeholder = st.empty()

if run_scan and stock_list:
    with st.spinner("Fetching fundamentals and computing scores..."):
        df = fetch_and_score_universe(stock_list)
    if df is None or df.empty:
        st.error("No data returned for provided tickers.")
    else:
        # Display sortable table with new KPIs
        display_cols = ['Ticker', 'Name', 'Current Price', 'P/E Ratio',
                        'ROE (%)', '3Y Profit Growth (%)', 'Debt to Equity',
                        'Volatility (%)', 'Fundamental_Score', 'Technical_Score',
                        'Sector_Score', 'Total_Score', 'Balanced_Recommendation',
                        'Yahoo_RecommendationKey', 'Yahoo_RecommendationMean']
        for c in display_cols:
            if c not in df.columns:
                df[c] = None
        df = df[display_cols]
        st.subheader("Scored Universe")
        st.dataframe(df.sort_values('Total_Score', ascending=False).reset_index(drop=True), height=420)

        # Add download button
        csv = df.to_csv(index=False)
        st.download_button("📥 Download scored universe CSV", csv, f"scored_universe_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

if run_recs and stock_list:
    st.info("Fetching analyst recommendations from Yahoo and Finnhub (if API key set)...")
    recs = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_yahoo_recommendations, t): t for t in stock_list}
        for f in as_completed(futures):
            t = futures[f]
            try:
                rec = f.result()
                rec['Ticker'] = t
                # fetch finnhub if key provided
                if FINNHUB_API_KEY:
                    try:
                        rec['Finnhub'] = get_finnhub_recommendation(t, FINNHUB_API_KEY)
                    except Exception:
                        rec['Finnhub'] = None
                recs.append(rec)
            except Exception:
                recs.append({'Ticker': t, 'error': 'failed'})

    recs_df = pd.DataFrame(recs)
    st.subheader("Aggregated Recommendations")
    st.dataframe(recs_df, height=360)
    st.download_button("📥 Download recommendations CSV", recs_df.to_csv(index=False), f"recommendations_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

if run_backtest and stock_list:
    st.info("Running quick forward-return backtest for current STRONG BUY picks (note: forward returns require future data; this will compute forward returns starting today — for true historical backtest you'd run the scoring for past dates).")
    # Ensure we have scored df - run scan if not present
    try:
        df
    except NameError:
        df = fetch_and_score_universe(stock_list)
    if df is None or df.empty:
        st.error("No scored data available. Click 'Fetch & Score Universe' first.")
    else:
        backtest_df = backtest_current_picks(df)
        if backtest_df.empty:
            st.write("No STRONG BUY picks found to backtest.")
        else:
            st.subheader("Backtest (forward returns from today)")
            st.dataframe(backtest_df, height=300)
            st.download_button("📥 Download backtest CSV", backtest_df.to_csv(index=False), f"backtest_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

# Detailed per-stock analysis area (non-invasive)
st.markdown("---")
st.header("🔎 Detailed Stock Analysis (select one of scanned tickers)")

selected_ticker = st.selectbox("Select ticker (from last scanned list)", options=(stock_list if stock_list else [None]))
if selected_ticker:
    t, info = safe_yf_ticker(selected_ticker)
    st.subheader(f"{selected_ticker} — {info.get('longName', '')}")
    # show MCP quick tech if desired
    if st.button(f"Get MCP technical for {selected_ticker}"):
        st.json(call_mcp("get_technical_analysis", {"symbol": selected_ticker}))

    # fetch scoreboard row if df exists
    try:
        scored_df  # check existence
    except NameError:
        pass

    # fetch fresh detailed data & technicals
    with st.spinner("Fetching detailed data and indicators..."):
        detailed = get_stock_core_data(selected_ticker)
        # compute technical series for chart
        t_kn, _ = safe_yf_ticker(selected_ticker)
        hist = t_kn.history(period="1y", interval="1d") if t_kn else None
        if hist is not None and not hist.empty:
            hist = hist.sort_index()
            hist['SMA_50'] = hist['Close'].rolling(50).mean()
            hist['RSI'] = ta.momentum.RSIIndicator(hist['Close']).rsi()
            macd = ta.trend.MACD(hist['Close'])
            hist['MACD'] = macd.macd()
            hist['MACD_Signal'] = macd.macd_signal()
            # compute final score using sector stats (if df present)
            try:
                current_df = df if 'df' in locals() else pd.DataFrame([detailed])
                sector_stats = compute_sector_stats(current_df)
            except Exception:
                sector_stats = None
            detailed_tech = add_technical_and_score(detailed, sector_stats)
            # present summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"₹{detailed_tech.get('Current Price'):.2f}" if detailed_tech.get('Current Price') else "N/A")
                st.metric("P/E", f"{detailed_tech.get('P/E Ratio'):.2f}" if detailed_tech.get('P/E Ratio') else "N/A")
            with col2:
                st.metric("ROE (%)", f"{detailed_tech.get('ROE (%)'):.2f}" if detailed_tech.get('ROE (%)') else "N/A")
                st.metric("ROCE (%)", f"{detailed_tech.get('ROCE (%)'):.2f}" if detailed_tech.get('ROCE (%)') else "N/A")
            with col3:
                st.metric("Filter Score", f"{detailed_tech.get('Filter_Score', 'N/A')}")
                st.metric("Debt/Equity", f"{detailed_tech.get('Debt to Equity'):.2f}" if detailed_tech.get('Debt to Equity') else "N/A")
            with col4:
                st.metric("Total Score (0-10)", f"{detailed_tech.get('Total_Score')}")
                st.metric("Balanced Recommendation", f"{detailed_tech.get('Balanced_Recommendation')}")

            # radar chart vs peers (use df subset same sector if available)
            peer_df = None
            if 'df' in locals() and not df.empty:
                if detailed_tech.get('Sector'):
                    peer_df = df[df['Sector'] == detailed_tech.get('Sector')]
                else:
                    peer_df = df
            try:
                radar = radar_chart_compare(detailed_tech, peer_df)
                st.plotly_chart(radar, use_container_width=True)
            except Exception:
                pass

            # price & indicators chart
            try:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'))
                if 'SMA_50' in hist.columns:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50'))
                fig.update_layout(title=f"{selected_ticker} Price & SMA50", height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.write("Chart unavailable")

# Info & suggestions
st.markdown("---")
st.subheader("🏁 How this aggregates with your MCP dashboard")
st.markdown("""
- This app keeps your MCP calls intact (see top-left) and **augments** them with live analyst data and a balanced scoring model.
- The scoring uses z-score normalization (peer vs universe) to reduce scale bias.
- Backtesting is a simple forward-return module; for a true historical backtest you'd run the scoring engine on historical snapshots (possible but heavier).
""")

st.subheader("📈 Ways to improve accuracy (practical roadmap)")
st.markdown("""
**Data & Sources**
- Use multiple data sources for redundancy: Yahoo (yfinance), Finnhub, IEX/Polygon/IEX Cloud, AlphaVantage, or paid vendors (Refinitiv, FactSet) if available.
- Validate key fields (PE, Debt/Equity, Dividend Yield) with at least one alternate source (NSE/BSE data or company filings).

**News & Sentiment**
- Integrate per-stock news feeds and compute sentiment (Finnhub / NewsAPI / GDELT). Use sentiment as a short-term factor for momentum and event risk.
- Prioritize earnings surprise/ guidance changes, management commentary, regulatory actions.

**Modeling**
- Backtest the scoring model: simulate scanning on historical dates and compute forward returns (30/90/180 days). Use that to tune weights.
- Use feature importance (tree-based models) to find which metrics predict future returns and adjust weights accordingly.
- Consider ensemble approach: combine rule-based scoring (this app) with a supervised ML model trained on historical successes.

**Operational**
- Build data quality checks (clamp absurd values, flag missing, compare across sources).
- Add "confidence" to each recommendation: incorporate data completeness & consensus from multiple APIs.
- Add position sizing rules and transaction cost estimates for realistic backtest.

**UX**
- Show a confidence meter and expose the main drivers of the score (explainability).
- Show alerts on data anomalies (e.g., extremely high dividend yield).

**Regulatory & Ethics**
- Provide clear disclaimers and never claim financial advice. Keep logs for reproducibility.

""")

# End of app
