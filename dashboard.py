import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore
import time
import json
import os

# --- Configuration ---
st.set_page_config(
    page_title="Advanced Hybrid Stock Screener V3.2",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- GLOBAL VARIABLES & DATA STORAGE ---
# File for storing backtesting picks (Simple JSON for Streamlit Cloud persistence)
PICK_FILE = "strong_buy_picks.json"

# Sector-specific metrics storage (for Z-score calculation)
SECTOR_DATA = {}

# --- 1. Data Fetching (Hybrid Approach: Stooq/YF for Price, YF for Fundamentals) ---

@st.cache_data(show_spinner="Fetching and validating stock data (This may take a minute)...")
def fetch_data(tickers):
    """
    Fetches price history (for technicals) and fundamental data (for fundamentals)
    using yfinance as the primary source, simplifying the original hybrid.
    """
    if not tickers:
        return {}

    end_date = datetime.now()
    # Request data starting 18 months ago
    start_date = end_date - timedelta(days=18 * 30) 
    
    data = {}
    
    for full_ticker in tickers:
        try:
            # --- Fetch Price & Fundamentals via yfinance ---
            stock = yf.Ticker(full_ticker)
            info = stock.info
            yf_hist = stock.history(start=start_date, end=end_date)
            
            # --- CRITICAL CHECK: Data Quality ---
            price_hist = yf_hist['Close'].dropna()
            valid_days = price_hist.shape[0]
            
            # Check for minimum required data
            if valid_days < 200:
                st.warning(f"Skipping {full_ticker}: Insufficient historical data (found only {valid_days} days).")
                continue
            
            # Get current data points
            market_cap = info.get('marketCap', np.nan)
            current_price = price_hist.iloc[-1]
            sector = info.get('sector', 'N/A')
            
            data[full_ticker] = {
                # Fundamental Metrics
                'PE_Ratio': info.get('trailingPE', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'DebtToEquity': info.get('debtToEquity', np.nan),
                'Growth': info.get('earningsQuarterlyGrowth', np.nan), # Proxy for Growth
                
                # Technical/Risk Metrics
                'Beta': info.get('beta', np.nan),
                'Volume_20D_Avg': yf_hist['Volume'].tail(20).mean() if 'Volume' in yf_hist.columns else np.nan,
                
                # Metadata
                'MarketCap': market_cap,
                'CurrentPrice': current_price,
                'Sector': sector,
                'PriceHistory': price_hist,
            }

        except Exception as e:
            st.error(f"Skipping {full_ticker}: Failed to fetch data: {e}")
            continue
            
    return data

# ----------------------------------------------------------------------

# --- 2. Metric Calculation (Fundamentals, Technicals, Risk) ---

def calculate_technical_metrics(price_hist):
    """Calculates RSI, MACD, ADX, and SMA-based signals."""
    if price_hist.empty or len(price_hist) < 200:
        return {'RSI': np.nan, 'MACD_Signal': np.nan, 'ADX': np.nan, 'SMA_Signal': np.nan}

    # 1. RSI (14 days)
    delta = price_hist.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    # 2. MACD (12, 26, 9)
    exp12 = price_hist.ewm(span=12, adjust=False).mean()
    exp26 = price_hist.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_signal = (macd.iloc[-1] - signal.iloc[-1]) # Positive is bullish
    
    # 3. SMA Signal (50-day vs 200-day)
    sma50 = price_hist.rolling(window=50).mean().iloc[-1]
    sma200 = price_hist.rolling(window=200).mean().iloc[-1]
    # Simple ratio: > 1 is bullish
    sma_signal = sma50 / sma200 
    
    # 4. Volatility (Annualized log returns)
    log_returns = np.log(price_hist / price_hist.shift(1)).dropna()
    volatility = log_returns.std() * np.sqrt(252)
    
    # ADX is complex and often requires separate data. Use a placeholder or simplified trend strength
    # For simplification, we'll use Volatility as a key "technical"/risk metric.
    
    return {
        'RSI': rsi,
        'MACD_Signal': macd_signal,
        'SMA_Signal': sma_signal,
        'Volatility': volatility
    }

def compute_all_metrics(data):
    """Calculates all fundamental, technical, and risk metrics."""
    processed_data = {}
    for ticker, d in data.items():
        price_hist = d['PriceHistory']
        
        tech_metrics = calculate_technical_metrics(price_hist)
        
        d.update(tech_metrics)
        d.pop('PriceHistory')
        
        # Calculate 1-Year Price Change
        lookback_period = min(252, len(price_hist)) 
        start_price = price_hist.iloc[-lookback_period]
        end_price = price_hist.iloc[-1]
        d['1Y_Change'] = ((end_price - start_price) / start_price) if start_price != 0 else np.nan
        
        processed_data[ticker] = d
        
    return pd.DataFrame.from_dict(processed_data, orient='index')

# ----------------------------------------------------------------------

# --- 3. Z-Score Normalization (Vs Sector Peers) ---

def z_score_normalize_by_sector(df, metric, direction):
    """Calculates Z-score for a metric relative to its sector group."""
    scored_metric_name = f'{metric}_ZScore'
    
    # Calculate Mean and Std Dev for each sector
    sector_stats = df.groupby('Sector')[metric].agg(['mean', 'std']).reset_index()
    sector_stats.columns = ['Sector', f'{metric}_mean', f'{metric}_std']
    
    # Merge stats back to the main DataFrame
    df_merged = df.merge(sector_stats, on='Sector', how='left')
    
    # Calculate Z-score: Z = (X - Mu) / Sigma
    # Handle zero standard deviation (should only happen with very few stocks per sector)
    def calculate_z(row):
        mu = row[f'{metric}_mean']
        sigma = row[f'{metric}_std']
        val = row[metric]
        
        if pd.isna(val) or pd.isna(mu):
            return 0 # Neutral score for missing data
        if sigma == 0 or pd.isna(sigma):
            return 0 # Cannot calculate Z-score, return neutral
        
        z = (val - mu) / sigma
        
        # Apply direction:
        # 'higher' is better -> Positive Z-score is good.
        # 'lower' is better -> Negative Z-score is good (flip the sign).
        return -z if direction == 'lower' else z

    df_merged[scored_metric_name] = df_merged.apply(calculate_z, axis=1)
    
    # Return only the Z-score column (ready for min-max scaling to 0-4)
    return df_merged[[scored_metric_name]]

# ----------------------------------------------------------------------

# --- 4. Scoring Logic (KPIS & Final Score) ---

def calculate_kpis_and_total_score(df):
    """
    Calculates the 3 main KPI scores (0-4, 0-4, 0-2) and the final 0-10 score.
    Uses Z-score normalization vs sector peers.
    """
    if df.empty:
        return df
    
    scored_df = df.copy()
    
    # 4.1. --- FUNDAMENTAL STRENGTH (40% / 0-4 points) ---
    fund_metrics = {
        'PE_Ratio': 'lower',      # Value factor
        'ROE': 'higher',          # Quality/Profitability
        'DebtToEquity': 'lower',  # Financial Health
        'Growth': 'higher',       # Growth
    }
    
    z_score_cols = []
    for metric, direction in fund_metrics.items():
        # Calculate Z-scores vs Sector Peers
        z_scores = z_score_normalize_by_sector(scored_df, metric, direction)
        z_col = z_scores.columns[0]
        scored_df = scored_df.merge(z_scores, left_index=True, right_index=True)
        z_score_cols.append(z_col)
        
    # Combine Z-scores for Fundamental Score (Simple Mean)
    # The result is normalized Z-score space. Now, normalize mean Z-score to 0-4 scale.
    scored_df['Fundamental_Mean_Z'] = scored_df[z_score_cols].mean(axis=1)
    
    # Min-Max scale the mean Z-score to a 0-4 range
    min_val = scored_df['Fundamental_Mean_Z'].min()
    max_val = scored_df['Fundamental_Mean_Z'].max()
    
    if max_val != min_val:
        scored_df['Fundamental_Score'] = 4 * (scored_df['Fundamental_Mean_Z'] - min_val) / (max_val - min_val)
    else:
        scored_df['Fundamental_Score'] = 2.0 # Neutral score if all are equal


    # 4.2. --- TECHNICAL MOMENTUM (40% / 0-4 points) ---
    tech_metrics = {
        'RSI': 'higher',       # High RSI (momentum)
        'MACD_Signal': 'higher', # Bullish crossover
        'SMA_Signal': 'higher',  # Long-term trend up
        '1Y_Change': 'higher'    # Price momentum
    }
    
    tech_z_score_cols = []
    for metric, direction in tech_metrics.items():
        z_scores = z_score_normalize_by_sector(scored_df, metric, direction)
        z_col = f'{metric}_Tech_ZScore' # Differentiate name
        scored_df = scored_df.merge(z_scores.rename(columns={z_scores.columns[0]: z_col}), left_index=True, right_index=True)
        tech_z_score_cols.append(z_col)

    scored_df['Technical_Mean_Z'] = scored_df[tech_z_score_cols].mean(axis=1)
    
    min_val_tech = scored_df['Technical_Mean_Z'].min()
    max_val_tech = scored_df['Technical_Mean_Z'].max()
    
    if max_val_tech != min_val_tech:
        scored_df['Technical_Score'] = 4 * (scored_df['Technical_Mean_Z'] - min_val_tech) / (max_val_tech - min_val_tech)
    else:
        scored_df['Technical_Score'] = 2.0 # Neutral score

    # 4.3. --- SECTOR & RISK FACTORS (20% / 0-2 points) ---
    risk_metrics = {
        'Beta': 'lower',      # Low market risk
        'Volatility': 'lower', # Low intrinsic risk
    }
    
    risk_z_score_cols = []
    for metric, direction in risk_metrics.items():
        z_scores = z_score_normalize_by_sector(scored_df, metric, direction)
        z_col = f'{metric}_Risk_ZScore'
        scored_df = scored_df.merge(z_scores.rename(columns={z_scores.columns[0]: z_col}), left_index=True, right_index=True)
        risk_z_score_cols.append(z_col)
        
    scored_df['Risk_Mean_Z'] = scored_df[risk_z_score_cols].mean(axis=1)
    
    min_val_risk = scored_df['Risk_Mean_Z'].min()
    max_val_risk = scored_df['Risk_Mean_Z'].max()
    
    if max_val_risk != min_val_risk:
        # Scale to 0-2 range (20% weight)
        scored_df['Sector_Score'] = 2 * (scored_df['Risk_Mean_Z'] - min_val_risk) / (max_val_risk - min_val_risk)
    else:
        scored_df['Sector_Score'] = 1.0 # Neutral score

    # 4.4. --- TOTAL SCORE (0-10) and RECOMMENDATION ---
    scored_df['Total_Score'] = scored_df['Fundamental_Score'] + scored_df['Technical_Score'] + scored_df['Sector_Score']
    
    # 4.5. --- Recommendation ---
    def assign_recommendation(score):
        if score >= 9.0:
            return "STRONG BUY"
        elif score >= 7.0:
            return "BUY"
        elif score >= 4.0:
            return "HOLD"
        else:
            return "SELL"
    
    scored_df['Balanced_Recommendation'] = scored_df['Total_Score'].apply(assign_recommendation)
    
    # Final cleanup and sort
    scored_df = scored_df.sort_values('Total_Score', ascending=False)
    
    return scored_df

# ----------------------------------------------------------------------

# --- 5. Backtesting Module (Data Capture & Reporting) ---

def load_picks():
    """Loads historical strong buy picks from JSON file."""
    if os.path.exists(PICK_FILE):
        with open(PICK_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_picks(picks):
    """Saves historical strong buy picks to JSON file."""
    with open(PICK_FILE, 'w') as f:
        json.dump(picks, f, indent=4)

def capture_strong_buy_picks(scored_df):
    """
    Captures 'STRONG BUY' picks with their entry price and date.
    This runs once per execution to track potential picks.
    """
    current_picks = load_picks()
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    strong_buys = scored_df[scored_df['Balanced_Recommendation'] == 'STRONG BUY']
    
    if not strong_buys.empty:
        if today_str not in current_picks:
            current_picks[today_str] = []
            
        for ticker in strong_buys.index:
            entry_price = strong_buys.loc[ticker, 'CurrentPrice']
            if ticker not in [p['ticker'] for p in current_picks[today_str]]:
                current_picks[today_str].append({
                    'ticker': ticker,
                    'entry_price': entry_price,
                    'status': 'NEW',
                    'date': today_str
                })
        
        save_picks(current_picks)

def run_backtest_summary():
    """
    Calculates the 30-day and 90-day returns for historical 'STRONG BUY' picks.
    """
    picks = load_picks()
    test_results = []
    
    for date_str, picks_list in picks.items():
        pick_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Only evaluate picks old enough for 30-day test
        if (datetime.now() - pick_date).days < 30:
            continue
            
        for pick in picks_list:
            ticker = pick['ticker']
            entry_price = pick['entry_price']
            
            try:
                # Fetch price data from entry date to now
                stock = yf.Ticker(ticker)
                # Fetch for 95 days to cover the 90-day period
                history = stock.history(start=pick_date.strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))['Close']
                
                if history.empty:
                    continue
                
                # Calculate 30-day return
                date_30d = pick_date + timedelta(days=30)
                price_30d = history.asof(date_30d)
                return_30d = ((price_30d - entry_price) / entry_price) * 100 if not pd.isna(price_30d) else np.nan

                # Calculate 90-day return
                date_90d = pick_date + timedelta(days=90)
                price_90d = history.asof(date_90d)
                # Only calculate if the 90-day period has passed
                return_90d = ((price_90d - entry_price) / entry_price) * 100 if not pd.isna(price_90d) and (datetime.now() >= date_90d) else np.nan

                test_results.append({
                    'Ticker': ticker,
                    'Pick Date': date_str,
                    'Entry Price': entry_price,
                    '30D Return (%)': return_30d,
                    '90D Return (%)': return_90d
                })
                
            except Exception:
                # Silently skip failed fetches for backtesting
                continue

    if not test_results:
        return None

    results_df = pd.DataFrame(test_results)
    # Aggregate summary
    summary = results_df[['30D Return (%)', '90D Return (%)']].agg(['mean', 'median', 'count']).T
    summary['Mean'] = summary['mean'].round(2)
    summary['Median'] = summary['median'].round(2)
    summary['Count'] = summary['count'].astype(int)
    
    return summary[['Count', 'Mean', 'Median']]

# ----------------------------------------------------------------------

# --- 6. Visualization & UI ---

def create_radar_chart(df, ticker):
    """Creates a radar chart comparing stock scores against sector averages."""
    sector = df.loc[ticker, 'Sector']
    sector_avg = df.groupby('Sector')[['Fundamental_Score', 'Technical_Score', 'Sector_Score']].mean().loc[sector]
    stock_scores = df.loc[ticker, ['Fundamental_Score', 'Technical_Score', 'Sector_Score']]

    categories = ['Fundamental Strength', 'Technical Momentum', 'Sector & Risk']
    
    # Scale scores from max (4, 4, 2) to 100 for easy plotting comparison
    max_scores = [4, 4, 2]
    stock_scaled = [(stock_scores.iloc[i] / max_scores[i]) * 100 for i in range(3)]
    sector_scaled = [(sector_avg.iloc[i] / max_scores[i]) * 100 for i in range(3)]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=stock_scaled,
        theta=categories,
        fill='toself',
        name=f'{ticker} Score'
    ))
    fig.add_trace(go.Scatterpolar(
        r=sector_scaled,
        theta=categories,
        fill='toself',
        name=f'{sector} Average'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100] # Standardized range
            )),
        showlegend=True,
        title=f'{ticker} Performance vs. {sector} Peers (Scaled to 100%)'
    )
    return fig


def results_page(scored_df):
    """Displays the main results and visualizations."""
    
    # --- Capture Picks (Must run first) ---
    capture_strong_buy_picks(scored_df)
    
    st.title("🏆 Advanced Hybrid Stock Screener Results (V3.2)")
    st.markdown(f"**Screening Date:** {datetime.now().strftime('%Y-%m-%d')}")

    if scored_df.empty:
        st.warning("No stocks passed the data validation checks.")
        return

    # --- 1. Ranked Stock Summary (Table) ---
    st.header("1. Ranked Stock Summary")
    
    display_cols = ['Total_Score', 'Balanced_Recommendation', 'CurrentPrice', 
                    'Fundamental_Score', 'Technical_Score', 'Sector_Score',
                    'PE_Ratio', 'ROE', 'DebtToEquity', 'RSI', 'Beta', 'Sector']
    
    formatted_df = scored_df[display_cols].copy()
    formatted_df.columns = ['Total Score (0-10)', 'Recommendation', 'Price (INR)', 
                            'Fund. Score (0-4)', 'Tech. Score (0-4)', 'Risk Score (0-2)',
                            'P/E', 'ROE', 'D/E', 'RSI', 'Beta', 'Sector']
    
    # Apply formatting
    for col in ['Total Score (0-10)', 'Fund. Score (0-4)', 'Tech. Score (0-4)', 'Risk Score (0-2)', 'P/E', 'D/E', 'RSI', 'Beta']:
        formatted_df[col] = formatted_df[col].round(2)
        
    formatted_df['Price (INR)'] = formatted_df['Price (INR)'].apply(lambda x: f"₹{x:,.2f}")
    formatted_df['ROE'] = (scored_df['ROE'] * 100).round(2).astype(str) + '%'

    st.dataframe(formatted_df, use_container_width=True)
    
    # --- 2. Backtesting Summary ---
    st.header("2. Backtesting Performance (STRONG BUY Picks)")
    backtest_summary = run_backtest_summary()
    
    if backtest_summary is not None:
        st.markdown("**Historical Model Precision (Strong Buy Picks):**")
        st.dataframe(backtest_summary, use_container_width=True)
    else:
        st.info("No historical 'STRONG BUY' picks are currently old enough (>= 30 days) to run backtesting.")
        
    # --- 3. Detailed Stock Analysis & Radar Chart ---
    st.header("3. Detailed Stock Analysis & Sector Comparison")
    selected_ticker = st.selectbox("Select a Ticker for Detailed View", scored_df.index)
    
    if selected_ticker:
        st.subheader(f"Analysis for {selected_ticker} ({scored_df.loc[selected_ticker, 'Sector']})")
        
        # Display Radar Chart
        st.plotly_chart(create_radar_chart(scored_df, selected_ticker), use_container_width=True)

        # Display Detailed Metrics (As requested: "extra lines under Fundamental Metrics")
        
        # Prepare the metrics for display
        detail_data = {
            "Total Score (0-10)": scored_df.loc[selected_ticker, 'Total_Score'].round(2),
            "Recommendation": scored_df.loc[selected_ticker, 'Balanced_Recommendation'],
            "--- SCORE BREAKDOWN ---": "",
            "Fundamental Strength (0-4)": scored_df.loc[selected_ticker, 'Fundamental_Score'].round(2),
            "Technical Momentum (0-4)": scored_df.loc[selected_ticker, 'Technical_Score'].round(2),
            "Sector & Risk (0-2)": scored_df.loc[selected_ticker, 'Sector_Score'].round(2),
            "--- KEY METRICS (Raw) ---": "",
            "P/E Ratio": scored_df.loc[selected_ticker, 'PE_Ratio'].round(2),
            "ROE": f"{(scored_df.loc[selected_ticker, 'ROE'] * 100):.2f}%",
            "D/E Ratio": scored_df.loc[selected_ticker, 'DebtToEquity'].round(2),
            "1Y Price Change": f"{(scored_df.loc[selected_ticker, '1Y_Change'] * 100):.2f}%",
            "RSI (14)": scored_df.loc[selected_ticker, 'RSI'].round(2),
            "Beta": scored_df.loc[selected_ticker, 'Beta'].round(2),
        }
        
        detail_df = pd.DataFrame(detail_data.items(), columns=['Metric', 'Value'])
        detail_df = detail_df.set_index('Metric')
        st.table(detail_df)


# ----------------------------------------------------------------------
# --- 7. Main Execution ---

def main():
    """The main function to run the Streamlit application."""
    st.sidebar.title("📈 Advanced Screener")
    st.sidebar.markdown("Hybrid model using **40% Fundamental, 40% Technical, 20% Risk**.")
    st.sidebar.markdown("Metrics are **Z-score normalized** vs. sector peers.")

    # Get user inputs (Scoring weights are fixed by the 40/40/20 model, so we only need tickers)
    default_tickers = ["TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS", "TATAMOTORS.NS"]
    
    ticker_input = st.sidebar.text_area(
        "Enter NSE Tickers (one per line, ensure '.NS' suffix is used):",
        value="\n".join(default_tickers)
    )
    tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]

    if not tickers:
        st.error("Please enter at least one NSE ticker.")
        return

    # Fetch Data
    raw_data = fetch_data(tickers)
    
    if not raw_data:
        st.error("No valid data was returned for the selected tickers.")
        return

    # Compute additional metrics (Technicals, 1Y Change)
    metrics_df = compute_all_metrics(raw_data)
    
    # Apply scoring and ranking
    scored_df = calculate_kpis_and_total_score(metrics_df)

    # Display results
    results_page(scored_df)

if __name__ == "__main__":
    # Ensure yfinance is configured to use the right session for stability
    yf.pdr_override() 
    main()
