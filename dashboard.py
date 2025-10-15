import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.stats import zscore
import json
import os
import io

# --- Configuration ---
st.set_page_config(
    page_title="Advanced Hybrid Stock Screener V3.3",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- GLOBAL VARIABLES & DATA STORAGE ---
PICK_FILE = "strong_buy_picks.json"

# --- Function to parse Nifty 500 symbols from CSV ---
def parse_nifty_symbols():
    nifty_csv_content = """Company Name,Industry,Symbol,Series,ISIN Code
360 ONE WAM Ltd.,Financial Services,360ONE,EQ,INE466L01038
3M India Ltd.,Diversified,3MINDIA,EQ,INE470A01017
ABB India Ltd.,Capital Goods,ABB,EQ,INE117A01022
ACC Ltd.,Construction Materials,ACC,EQ,INE012A01025
ACME Solar Holdings Ltd.,Power,ACMESOLAR,EQ,INE622W01025
AIA Engineering Ltd.,Capital Goods,AIAENG,EQ,INE212H01026
APL Apollo Tubes Ltd.,Capital Goods,APLAPOLLO,EQ,INE702C01027
AU Small Finance Bank Ltd.,Financial Services,AUBANK,EQ,INE949L01017
AWL Agri Business Ltd.,Fast Moving Consumer Goods,AWL,EQ,INE699H01024
Aadhar Housing Finance Ltd.,Financial Services,AADHARHFC,EQ,INE883F01010
Aarti Industries Ltd.,Chemicals,AARTIIND,EQ,INE769A01020
Aavas Financiers Ltd.,Financial Services,AAVAS,EQ,INE216P01012
Abbott India Ltd.,Healthcare,ABBOTINDIA,EQ,INE358A01014
Action Construction Equipment Ltd.,Capital Goods,ACE,EQ,INE731H01025
Adani Energy Solutions Ltd.,Power,ADANIENSOL,EQ,INE423A01021
Adani Enterprises Ltd.,Diversified,ADANIENT,EQ,INE423A01020
Adani Ports and Special Economic Zone Ltd.,Services,ADANIPORTS,EQ,INE742F01042
Amara Raja Batteries Ltd.,Automobile and Auto Components,AMARARAJA,EQ,INE885A01032
Ambuja Cements Ltd.,Construction Materials,AMBUJACEM,EQ,INE079A01024
Ashok Leyland Ltd.,Automobile and Auto Components,ASHOKLEY,EQ,INE208A01039
Axis Bank Ltd.,Financial Services,AXISBANK,EQ,INE238A01026
Bank of Baroda,Financial Services,BANKBARODA,EQ,INE011A01026
Canara Bank,Financial Services,CANBK,EQ,INE476A01014
HDFC Bank Ltd.,Financial Services,HDFCBANK,EQ,INE040A01026
ICICI Bank Ltd.,Financial Services,ICICIBANK,EQ,INE090A01021
Infosys Ltd.,Information Technology,INFY,EQ,INE009A01021
IRB Infrastructure Developers Ltd.,Construction,IRB,EQ,INE828L01016
Jammu & Kashmir Bank Ltd.,Financial Services,J&KBANK,EQ,INE162A01018
Larsen & Toubro Ltd.,Capital Goods,LT,EQ,INE018A01030
Maruti Suzuki India Ltd.,Automobile and Auto Components,MARUTI,EQ,INE045A01017
Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018
Tata Consultancy Services Ltd.,Information Technology,TCS,EQ,INE467A01029
Titan Company Ltd.,Consumer Durables,TITAN,EQ,INE280A01028
Vedanta Ltd.,Metals & Mining,VEDL,EQ,INE205A01025
Voltas Ltd.,Consumer Durables,VOLTAS,EQ,INE226A01021
Wipro Ltd.,Information Technology,WIPRO,EQ,INE075A01022
""" 
    try:
        df = pd.read_csv(io.StringIO(nifty_csv_content))
        df['YF_Symbol'] = df['Symbol'].apply(lambda x: f"{x}.NS")
        df['Display_Name'] = df['Symbol'] + " (" + df['Company Name'] + ")"
        
        symbol_map = pd.Series(df.YF_Symbol.values, index=df.Display_Name).to_dict()
        display_names = df['Display_Name'].dropna().unique().tolist()
        
        return display_names, symbol_map
    except Exception:
        return [], {}

# --- 1. Enhanced Data Fetching with Analyst Recommendations ---

@st.cache_data(show_spinner="Fetching stock data, analyst ratings, and news...")
def fetch_data(tickers):
    """
    Fetches price history, fundamental data, analyst recommendations,
    and news using yfinance.
    """
    if not tickers:
        return {}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=18 * 30)
    
    data = {}
    
    for full_ticker in tickers:
        try:
            stock = yf.Ticker(full_ticker)
            info = stock.info
            yf_hist = stock.history(start=start_date, end=end_date)
            
            # --- CRITICAL CHECK: Data Quality ---
            price_hist = yf_hist['Close'].dropna()
            
            if price_hist.shape[0] < 200 or pd.isna(price_hist.iloc[-1]):
                continue
            
            current_price = price_hist.iloc[-1]
            sector = info.get('sector', 'N/A')
            
            if sector == 'N/A':
                 continue
            
            # --- NEW: Fetch Analyst Recommendations ---
            try:
                recommendations = stock.recommendations
                analyst_ratings = extract_analyst_ratings(recommendations)
            except:
                analyst_ratings = {
                    'total_analysts': 0,
                    'strong_buy': 0,
                    'buy': 0,
                    'hold': 0,
                    'sell': 0,
                    'strong_sell': 0,
                    'weighted_score': 0,
                    'consensus': 'N/A'
                }
            
            # --- NEW: Fetch Recent News ---
            try:
                news = stock.news[:5] if hasattr(stock, 'news') and stock.news else []
                news_items = extract_news_items(news)
            except:
                news_items = []
            
            data[full_ticker] = {
                # Fundamental Metrics
                'PE_Ratio': info.get('trailingPE', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'DebtToEquity': info.get('debtToEquity', np.nan),
                'Growth': info.get('earningsQuarterlyGrowth', np.nan), 
                
                # Technical/Risk Metrics
                'Beta': info.get('beta', np.nan),
                
                # External Recommendation Consensus
                'External_Recommendation': info.get('recommendationKey', 'N/A').upper().replace('STRONG_BUY', 'STRONG BUY'),
                
                # --- NEW: Analyst Ratings Data ---
                'Analyst_Ratings': analyst_ratings,
                
                # Target Price
                'Target_Mean': info.get('targetMeanPrice', np.nan),
                'Target_High': info.get('targetHighPrice', np.nan),
                'Target_Low': info.get('targetLowPrice', np.nan),
                
                # --- NEW: News Items ---
                'News': news_items,
                
                # Metadata
                'CurrentPrice': current_price,
                'Sector': sector,
                'PriceHistory': price_hist,
                'Company_Name': info.get('longName', full_ticker),
            }

        except Exception as e:
            continue
            
    return data

def extract_analyst_ratings(recommendations_df):
    """
    Extracts and calculates weighted analyst ratings from recommendations DataFrame.
    Returns a dictionary with rating counts and consensus.
    """
    if recommendations_df is None or recommendations_df.empty:
        return {
            'total_analysts': 0,
            'strong_buy': 0,
            'buy': 0,
            'hold': 0,
            'sell': 0,
            'strong_sell': 0,
            'weighted_score': 0,
            'consensus': 'N/A'
        }
    
    # Get the most recent recommendations (last 3 months)
    recent_date = datetime.now() - timedelta(days=90)
    recent_recs = recommendations_df[recommendations_df.index >= recent_date]
    
    if recent_recs.empty:
        recent_recs = recommendations_df.tail(10)  # Use last 10 if no recent data
    
    # Count ratings
    ratings = {
        'strong_buy': 0,
        'buy': 0,
        'hold': 0,
        'sell': 0,
        'strong_sell': 0
    }
    
    for col in recent_recs.columns:
        col_lower = col.lower()
        if 'strongbuy' in col_lower or 'strong buy' in col_lower:
            ratings['strong_buy'] = recent_recs[col].sum()
        elif 'buy' in col_lower and 'strong' not in col_lower:
            ratings['buy'] = recent_recs[col].sum()
        elif 'hold' in col_lower:
            ratings['hold'] = recent_recs[col].sum()
        elif 'sell' in col_lower and 'strong' not in col_lower:
            ratings['sell'] = recent_recs[col].sum()
        elif 'strongsell' in col_lower or 'strong sell' in col_lower:
            ratings['strong_sell'] = recent_recs[col].sum()
    
    total = sum(ratings.values())
    
    # Calculate weighted score (5 = Strong Buy, 1 = Strong Sell)
    if total > 0:
        weighted_score = (
            ratings['strong_buy'] * 5 +
            ratings['buy'] * 4 +
            ratings['hold'] * 3 +
            ratings['sell'] * 2 +
            ratings['strong_sell'] * 1
        ) / total
    else:
        weighted_score = 0
    
    # Determine consensus
    if weighted_score >= 4.5:
        consensus = 'STRONG BUY'
    elif weighted_score >= 3.5:
        consensus = 'BUY'
    elif weighted_score >= 2.5:
        consensus = 'HOLD'
    elif weighted_score >= 1.5:
        consensus = 'SELL'
    else:
        consensus = 'STRONG SELL'
    
    return {
        'total_analysts': int(total),
        'strong_buy': int(ratings['strong_buy']),
        'buy': int(ratings['buy']),
        'hold': int(ratings['hold']),
        'sell': int(ratings['sell']),
        'strong_sell': int(ratings['strong_sell']),
        'weighted_score': round(weighted_score, 2),
        'consensus': consensus
    }

def extract_news_items(news_list):
    """
    Extracts relevant information from news items.
    """
    news_items = []
    for item in news_list[:5]:  # Limit to 5 most recent
        try:
            news_items.append({
                'title': item.get('title', 'N/A'),
                'publisher': item.get('publisher', 'Unknown'),
                'link': item.get('link', '#'),
                'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M') if item.get('providerPublishTime') else 'N/A'
            })
        except:
            continue
    return news_items

# --- 2. Metric Calculation (Technicals and Risk) ---

def calculate_technical_metrics(price_hist):
    """Calculates RSI, MACD, SMA-based signals, and Volatility."""
    if price_hist.empty or len(price_hist) < 200:
        return {'RSI': np.nan, 'MACD_Signal': np.nan, 'SMA_Signal': np.nan, 'Volatility': np.nan}

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
    macd_signal = (macd.iloc[-1] - signal.iloc[-1])
    
    # 3. SMA Signal (50-day vs 200-day ratio)
    sma50 = price_hist.rolling(window=50).mean().iloc[-1]
    sma200 = price_hist.rolling(window=200).mean().iloc[-1]
    sma_signal = sma50 / sma200 
    
    # 4. Volatility
    log_returns = np.log(price_hist / price_hist.shift(1)).dropna()
    volatility = log_returns.std() * np.sqrt(252)
    
    return {
        'RSI': rsi,
        'MACD_Signal': macd_signal,
        'SMA_Signal': sma_signal,
        'Volatility': volatility
    }

def compute_all_metrics(data):
    """Calculates all metrics into a single DataFrame."""
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
        
        # Calculate upside potential
        if not pd.isna(d['Target_Mean']) and d['CurrentPrice'] > 0:
            d['Upside_Potential'] = ((d['Target_Mean'] - d['CurrentPrice']) / d['CurrentPrice']) * 100
        else:
            d['Upside_Potential'] = np.nan
        
        processed_data[ticker] = d
        
    return pd.DataFrame.from_dict(processed_data, orient='index')

# --- 3. Z-Score Normalization ---

def z_score_normalize_by_sector(df, metric, direction):
    """Calculates Z-score for a metric relative to its sector group."""
    scored_metric_name = f'{metric}_ZScore'
    
    sector_stats = df.groupby('Sector')[metric].agg(['mean', 'std']).reset_index()
    sector_stats.columns = ['Sector', f'{metric}_mean', f'{metric}_std']
    
    df_merged = df.merge(sector_stats, on='Sector', how='left')
    
    def calculate_z(row):
        mu = row[f'{metric}_mean']
        sigma = row[f'{metric}_std']
        val = row[metric]
        
        if pd.isna(val) or pd.isna(mu) or sigma == 0 or pd.isna(sigma):
            return 0 
        
        z = (val - mu) / sigma
        
        return -z if direction == 'lower' else z

    df_merged[scored_metric_name] = df_merged.apply(calculate_z, axis=1)
    return df_merged[[scored_metric_name]]

# --- 4. Scoring Logic ---

def calculate_kpis_and_total_score(df):
    """Calculates the 3 main KPI scores and the final score."""
    if df.empty:
        return df
    
    scored_df = df.copy()
    
    def calculate_and_scale_score(df, metrics, weight_max):
        z_score_cols = []
        for metric, direction in metrics.items():
            z_scores = z_score_normalize_by_sector(df, metric, direction)
            z_col = f'{metric}_{weight_max}_ZScore'
            df = df.merge(z_scores.rename(columns={z_scores.columns[0]: z_col}), left_index=True, right_index=True)
            z_score_cols.append(z_col)
            
        mean_z_col = 'Mean_Z_' + str(weight_max)
        df[mean_z_col] = df[z_score_cols].mean(axis=1)

        min_val = df[mean_z_col].min()
        max_val = df[mean_z_col].max()
        
        score_col = 'Score_' + str(weight_max)
        if max_val != min_val:
            df[score_col] = weight_max * (df[mean_z_col] - min_val) / (max_val - min_val)
        else:
            df[score_col] = weight_max / 2.0 
            
        return df, score_col

    # Fundamental Strength (40%)
    fund_metrics = {'PE_Ratio': 'lower', 'ROE': 'higher', 'DebtToEquity': 'lower', 'Growth': 'higher'}
    scored_df, fund_score_col = calculate_and_scale_score(scored_df, fund_metrics, 4)
    scored_df = scored_df.rename(columns={fund_score_col: 'Fundamental_Score'})

    # Technical Momentum (40%)
    tech_metrics = {'RSI': 'higher', 'MACD_Signal': 'higher', 'SMA_Signal': 'higher', '1Y_Change': 'higher'}
    scored_df, tech_score_col = calculate_and_scale_score(scored_df, tech_metrics, 4)
    scored_df = scored_df.rename(columns={tech_score_col: 'Technical_Score'})

    # Sector & Risk Factors (20%)
    risk_metrics = {'Beta': 'lower', 'Volatility': 'lower'}
    scored_df, risk_score_col = calculate_and_scale_score(scored_df, risk_metrics, 2)
    scored_df = scored_df.rename(columns={risk_score_col: 'Sector_Score'})

    # Total Score (0-10)
    scored_df['Total_Score'] = scored_df['Fundamental_Score'] + scored_df['Technical_Score'] + scored_df['Sector_Score']
    
    # Recommendation
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
    
    return scored_df.sort_values('Total_Score', ascending=False)

# --- 5. Backtesting Module ---

def load_picks():
    if os.path.exists(PICK_FILE):
        try:
            with open(PICK_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
             return {}
    return {}

def save_picks(picks):
    with open(PICK_FILE, 'w') as f:
        json.dump(picks, f, indent=4)

def capture_strong_buy_picks(scored_df):
    current_picks = load_picks()
    today_str = datetime.now().strftime('%Y-%m-%d')
    strong_buys = scored_df[scored_df['Balanced_Recommendation'] == 'STRONG BUY']
    
    if not strong_buys.empty:
        if today_str not in current_picks:
            current_picks[today_str] = []
            
        today_pickers = {p['ticker'] for p in current_picks[today_str]}
            
        for ticker in strong_buys.index:
            entry_price = strong_buys.loc[ticker, 'CurrentPrice']
            if ticker not in today_pickers:
                current_picks[today_str].append({
                    'ticker': ticker,
                    'entry_price': entry_price,
                    'date': today_str
                })
        save_picks(current_picks)

def run_backtest_summary():
    picks = load_picks()
    test_results = []
    all_picks = [p for day in picks.values() for p in day]
    
    for pick in all_picks:
        ticker = pick['ticker']
        date_str = pick['date']
        entry_price = pick['entry_price']
        pick_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        if (datetime.now().date() - pick_date).days < 30:
            continue
            
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(start=pick_date, end=datetime.now())['Close']
            
            if history.empty: continue
            
            date_30d = pick_date + timedelta(days=30)
            price_30d = history.asof(date_30d) 
            return_30d = ((price_30d - entry_price) / entry_price) * 100 if not pd.isna(price_30d) else np.nan

            date_90d = pick_date + timedelta(days=90)
            return_90d = np.nan
            
            if datetime.now().date() >= date_90d:
                price_90d = history.asof(date_90d)
                return_90d = ((price_90d - entry_price) / entry_price) * 100 if not pd.isna(price_90d) else np.nan
            
            test_results.append({'30D Return (%)': return_30d, '90D Return (%)': return_90d})
            
        except Exception:
            continue

    if not test_results: return None

    results_df = pd.DataFrame(test_results)
    summary = results_df[['30D Return (%)', '90D Return (%)']].agg(['mean', 'median', 'count']).T
    summary['Mean'] = summary['mean'].round(2)
    summary['Median'] = summary['median'].round(2)
    summary['Count'] = summary['count'].astype(int)
    
    return summary[['Count', 'Mean', 'Median']]

# --- 6. Visualization & UI ---

def create_radar_chart(df, ticker):
    """Creates a radar chart comparing stock scores against sector averages."""
    sector = df.loc[ticker, 'Sector']
    sector_avg = df.groupby('Sector')[['Fundamental_Score', 'Technical_Score', 'Sector_Score']].mean().loc[sector]
    stock_scores = df.loc[ticker, ['Fundamental_Score', 'Technical_Score', 'Sector_Score']]

    categories = ['Fundamental Strength', 'Technical Momentum', 'Sector & Risk']
    max_scores = [4.0, 4.0, 2.0] 
    
    stock_scaled = [(stock_scores.iloc[i] / max_scores[i]) * 100 if max_scores[i] > 0 else 0 for i in range(3)]
    sector_scaled = [(sector_avg.iloc[i] / max_scores[i]) * 100 if max_scores[i] > 0 else 0 for i in range(3)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=stock_scaled, theta=categories, fill='toself', name=f'{ticker} Score'))
    fig.add_trace(go.Scatterpolar(r=sector_scaled, theta=categories, fill='toself', name=f'{sector} Average'))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title=f'{ticker} Performance vs. {sector} Peers')
    return fig

def create_analyst_distribution_chart(analyst_ratings):
    """Creates a bar chart showing analyst rating distribution."""
    categories = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
    values = [
        analyst_ratings['strong_buy'],
        analyst_ratings['buy'],
        analyst_ratings['hold'],
        analyst_ratings['sell'],
        analyst_ratings['strong_sell']
    ]
    
    colors = ['#00CC00', '#66FF66', '#FFD700', '#FF9900', '#FF0000']
    
    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, marker_color=colors, text=values, textposition='auto')
    ])
    
    fig.update_layout(
        title=f"Analyst Ratings Distribution (Total: {analyst_ratings['total_analysts']} Analysts)",
        xaxis_title="Rating",
        yaxis_title="Number of Analysts",
        showlegend=False,
        height=350
    )
    
    return fig

def sidebar_inputs():
    """Defines the sidebar layout and retrieves user selected tickers."""
    st.sidebar.title("📈 Hybrid Screener V3.3")
    st.sidebar.markdown("---")
    
    st.sidebar.header("⚖️ Model Weighting (Fixed)")
    st.sidebar.markdown("""
        - **Fundamental Strength:** **40%** (0-4 pts)
        - **Technical Momentum:** **40%** (0-4 pts)
        - **Sector & Risk Factors:** **20%** (0-2 pts)
        ---
        *Metrics are Z-score normalized vs. sector peers.*
    """)
    st.sidebar.markdown("---")
    
    st.sidebar.header("🆕 New Features")
    st.sidebar.markdown("""
        - ✅ Weighted Analyst Ratings
        - ✅ Expert Recommendations
        - ✅ Latest News & Insights
        - ✅ Target Price Analysis
    """)
    st.sidebar.markdown("---")
    
    display_names, symbol_map = parse_nifty_symbols()
    
    default_names = [
        "RELIANCE (Reliance Industries Ltd.)", 
        "TCS (Tata Consultancy Services Ltd.)", 
        "ASHOKLEY (Ashok Leyland Ltd.)", 
        "CANBK (Canara Bank)", 
        "IRB (IRB Infrastructure Developers Ltd.)", 
        "J&KBANK (Jammu & Kashmir Bank Ltd.)"
    ]
    
    initial_selection = [name for name in default_names if name in display_names]
    
    selected_names = st.sidebar.multiselect(
        "Select Stocks for Analysis (NSE)",
        options=display_names,
        default=initial_selection,
        help="Choose stocks from the list to analyze their scores, analyst ratings, and news."
    )
    
    selected_tickers = [symbol_map[name] for name in selected_names]
    
    return selected_tickers 

def results_page(scored_df):
    """Displays the main results and visualizations."""
    
    capture_strong_buy_picks(scored_df)
    
    st.title("🏆 Hybrid Stock Analysis Results (V3.3)")
    st.subheader("Fixed Model Weighting: 40% Fundamental / 40% Technical / 20% Risk")
    st.markdown(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}")

    if scored_df.empty:
        st.error("No stocks were selected or passed the data validation checks.")
        return

    # --- 1. Enhanced Ranked Stock Summary with Analyst Ratings ---
    st.header("1. Scores, Recommendations & Analyst Ratings")
    
    display_cols = ['Total_Score', 'Balanced_Recommendation', 'External_Recommendation',
                    'Fundamental_Score', 'Technical_Score', 'Sector_Score',
                    'CurrentPrice', 'Target_Mean', 'Upside_Potential', 'Sector']
    
    formatted_df = scored_df[display_cols].copy()
    
    # Add analyst consensus
    formatted_df['Analyst_Consensus'] = scored_df['Analyst_Ratings'].apply(lambda x: x['consensus'] if isinstance(x, dict) else 'N/A')
    formatted_df['Num_Analysts'] = scored_df['Analyst_Ratings'].apply(lambda x: x['total_analysts'] if isinstance(x, dict) else 0)
    
    formatted_df.columns = ['Total Score (0-10)', 'App Recommendation', 'YF Consensus',
                            'Fund. Score (0-4)', 'Tech. Score (0-4)', 'Risk Score (0-2)',
                            'Current Price', 'Target Price', 'Upside %', 'Sector', 
                            'Analyst Consensus', '# Analysts']
    
    # Apply formatting
    for col in ['Total Score (0-10)', 'Fund. Score (0-4)', 'Tech. Score (0-4)', 'Risk Score (0-2)']:
        formatted_df[col] = formatted_df[col].round(2)
        
    formatted_df['Current Price'] = formatted_df['Current Price'].apply(lambda x: f"₹{x:,.2f}")
    formatted_df['Target Price'] = formatted_df['Target Price'].apply(lambda x: f"₹{x:,.2f}" if not pd.isna(x) else 'N/A')
    formatted_df['Upside %'] = formatted_df['Upside %'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else 'N/A')

    st.dataframe(formatted_df, use_container_width=True)
    
    st.markdown("""
        **Recommendation Sources:**
        * **App Recommendation:** Based on 40/40/20 Model Score (≥9.0: STRONG BUY, ≥7.0: BUY, ≥4.0: HOLD).
        * **YF Consensus:** Yahoo Finance aggregated recommendation from various data providers.
        * **Analyst Consensus:** Weighted average from recent analyst ratings (last 3 months).
    """)
    
    # --- 2. Detailed Stock Analysis with Analyst Ratings & News ---
    st.header("2. Detailed Analysis with Expert Insights")
    selected_ticker = st.selectbox("Select a Ticker for Detailed View", scored_df.index)
    
    if selected_ticker:
        company_name = scored_df.loc[selected_ticker, 'Company_Name']
        st.subheader(f"Analysis for {selected_ticker} - {company_name}")
        st.caption(f"Sector: {scored_df.loc[selected_ticker, 'Sector']}")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Scores & Metrics", "👥 Analyst Ratings", "📰 Latest News", "📈 Visual Analysis"])
        
        # TAB 1: Scores & Metrics
        with tab1:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### 📊 Scoring Breakdown")
                
                detail_data = {
                    "Total Score (0-10)": formatted_df.loc[selected_ticker, 'Total Score (0-10)'],
                    "App Recommendation": formatted_df.loc[selected_ticker, 'App Recommendation'],
                    "YF Consensus": formatted_df.loc[selected_ticker, 'YF Consensus'],
                    "Analyst Consensus": formatted_df.loc[selected_ticker, 'Analyst Consensus'],
                    "--- SCORE BREAKDOWN ---": "",
                    "Fundamental Strength (0-4)": formatted_df.loc[selected_ticker, 'Fund. Score (0-4)'],
                    "Technical Momentum (0-4)": formatted_df.loc[selected_ticker, 'Tech. Score (0-4)'],
                    "Sector & Risk (0-2)": formatted_df.loc[selected_ticker, 'Risk Score (0-2)'],
                }
                detail_df = pd.DataFrame(detail_data.items(), columns=['Metric', 'Value'])
                detail_df = detail_df.set_index('Metric')
                st.table(detail_df)
                
            with col2:
                st.markdown("### 📈 Key Raw Metrics")
                
                current_price = scored_df.loc[selected_ticker, 'CurrentPrice']
                target_mean = scored_df.loc[selected_ticker, 'Target_Mean']
                upside = scored_df.loc[selected_ticker, 'Upside_Potential']
                
                raw_metrics = {
                    "Current Price": f"₹{current_price:,.2f}",
                    "Target Price (Mean)": f"₹{target_mean:,.2f}" if not pd.isna(target_mean) else 'N/A',
                    "Upside Potential": f"{upside:.1f}%" if not pd.isna(upside) else 'N/A',
                    "--- FUNDAMENTALS ---": "",
                    "P/E Ratio": f"{scored_df.loc[selected_ticker, 'PE_Ratio']:.2f}" if not pd.isna(scored_df.loc[selected_ticker, 'PE_Ratio']) else 'N/A',
                    "ROE": f"{(scored_df.loc[selected_ticker, 'ROE'] * 100):.2f}%" if not pd.isna(scored_df.loc[selected_ticker, 'ROE']) else 'N/A',
                    "D/E Ratio": f"{scored_df.loc[selected_ticker, 'DebtToEquity']:.2f}" if not pd.isna(scored_df.loc[selected_ticker, 'DebtToEquity']) else 'N/A',
                    "--- TECHNICALS ---": "",
                    "1Y Price Change": f"{(scored_df.loc[selected_ticker, '1Y_Change'] * 100):.2f}%",
                    "RSI (14)": f"{scored_df.loc[selected_ticker, 'RSI']:.2f}",
                    "Beta": f"{scored_df.loc[selected_ticker, 'Beta']:.2f}" if not pd.isna(scored_df.loc[selected_ticker, 'Beta']) else 'N/A',
                }
                raw_df = pd.DataFrame(raw_metrics.items(), columns=['Metric', 'Value'])
                raw_df = raw_df.set_index('Metric')
                st.table(raw_df)
        
        # TAB 2: Analyst Ratings
        with tab2:
            analyst_ratings = scored_df.loc[selected_ticker, 'Analyst_Ratings']
            
            if isinstance(analyst_ratings, dict) and analyst_ratings['total_analysts'] > 0:
                st.markdown("### 👥 Analyst Ratings Breakdown")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.plotly_chart(create_analyst_distribution_chart(analyst_ratings), use_container_width=True)
                
                with col2:
                    st.markdown("#### 📊 Rating Summary")
                    
                    # Consensus with color coding
                    consensus = analyst_ratings['consensus']
                    if consensus == 'STRONG BUY':
                        consensus_color = '🟢'
                    elif consensus == 'BUY':
                        consensus_color = '🟡'
                    elif consensus == 'HOLD':
                        consensus_color = '🟠'
                    else:
                        consensus_color = '🔴'
                    
                    st.metric("Consensus Rating", f"{consensus_color} {consensus}")
                    st.metric("Weighted Score", f"{analyst_ratings['weighted_score']:.2f} / 5.0")
                    st.metric("Total Analysts", analyst_ratings['total_analysts'])
                    
                    st.markdown("---")
                    st.markdown("#### 📋 Distribution")
                    st.write(f"**Strong Buy:** {analyst_ratings['strong_buy']}")
                    st.write(f"**Buy:** {analyst_ratings['buy']}")
                    st.write(f"**Hold:** {analyst_ratings['hold']}")
                    st.write(f"**Sell:** {analyst_ratings['sell']}")
                    st.write(f"**Strong Sell:** {analyst_ratings['strong_sell']}")
                
                # Target Price Analysis
                st.markdown("---")
                st.markdown("### 🎯 Target Price Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                target_low = scored_df.loc[selected_ticker, 'Target_Low']
                target_high = scored_df.loc[selected_ticker, 'Target_High']
                
                with col1:
                    st.metric("Current Price", f"₹{current_price:,.2f}")
                with col2:
                    st.metric("Target Low", f"₹{target_low:,.2f}" if not pd.isna(target_low) else 'N/A')
                with col3:
                    st.metric("Target Mean", f"₹{target_mean:,.2f}" if not pd.isna(target_mean) else 'N/A')
                with col4:
                    st.metric("Target High", f"₹{target_high:,.2f}" if not pd.isna(target_high) else 'N/A')
                
                if not pd.isna(upside):
                    if upside > 0:
                        st.success(f"📈 Potential upside of **{upside:.1f}%** based on mean analyst target price")
                    else:
                        st.warning(f"📉 Current price is **{abs(upside):.1f}%** above mean analyst target price")
                        
            else:
                st.info("No analyst ratings data available for this stock.")
        
        # TAB 3: Latest News
        with tab3:
            news_items = scored_df.loc[selected_ticker, 'News']
            
            if news_items and len(news_items) > 0:
                st.markdown("### 📰 Latest News & Market Insights")
                st.caption("Recent news articles and analysis from various sources")
                
                for i, news in enumerate(news_items, 1):
                    with st.expander(f"📌 {news['title']}", expanded=(i==1)):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Publisher:** {news['publisher']}")
                            st.markdown(f"**Published:** {news['published']}")
                        with col2:
                            st.markdown(f"[🔗 Read Article]({news['link']})")
                        
                st.markdown("---")
                st.info("💡 **Tip:** News sentiment and expert opinions can provide valuable context alongside quantitative metrics.")
            else:
                st.info("No recent news available for this stock.")
        
        # TAB 4: Visual Analysis
        with tab4:
            st.markdown("### 🎯 Stock vs. Sector Average Comparison (Radar Chart)")
            st.plotly_chart(create_radar_chart(scored_df, selected_ticker), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 Recommendation Comparison")
            
            # Create a comparison table
            rec_comparison = pd.DataFrame({
                'Source': ['App Model', 'Yahoo Finance', 'Analyst Consensus'],
                'Recommendation': [
                    scored_df.loc[selected_ticker, 'Balanced_Recommendation'],
                    scored_df.loc[selected_ticker, 'External_Recommendation'],
                    analyst_ratings['consensus'] if isinstance(analyst_ratings, dict) else 'N/A'
                ],
                'Basis': [
                    '40/40/20 Quantitative Model',
                    'Aggregated Data Provider Consensus',
                    f"{analyst_ratings['total_analysts']} Analyst Ratings" if isinstance(analyst_ratings, dict) else 'N/A'
                ]
            })
            
            st.dataframe(rec_comparison, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Interpretation Guide:**
            - **Strong Agreement:** All sources align → Higher confidence
            - **Moderate Agreement:** 2 out of 3 align → Moderate confidence  
            - **Disagreement:** Sources diverge → Consider multiple perspectives
            """)

    # --- 3. Backtesting Summary ---
    st.header("3. Backtesting Performance (STRONG BUY Picks)")
    backtest_summary = run_backtest_summary()
    
    if backtest_summary is not None:
        st.markdown("Historical performance of 'STRONG BUY' recommendations (Mean & Median returns):")
        st.dataframe(backtest_summary, use_container_width=True)
        st.caption("⚠️ Past performance does not guarantee future results. Use as one of many decision factors.")
    else:
        st.info("No historical 'STRONG BUY' picks are currently old enough (≥ 30 days) to run backtesting.")

# --- 7. Main Execution ---

def main():
    """The main function to run the Streamlit application."""
    
    # 1. Get user selected tickers from sidebar
    tickers = sidebar_inputs()
    
    if not tickers:
        st.info("👈 Please select stocks in the sidebar to run the analysis.")
        st.markdown("---")
        st.markdown("""
        ### 🎯 What's New in V3.3?
        
        This enhanced version includes:
        
        1. **📊 Weighted Analyst Ratings**
           - Aggregates recommendations from multiple analysts
           - Calculates consensus ratings (Strong Buy to Strong Sell)
           - Shows rating distribution and weighted scores
        
        2. **🎯 Target Price Analysis**
           - Displays analyst target prices (Low, Mean, High)
           - Calculates potential upside/downside
           - Compares current price against analyst expectations
        
        3. **📰 Latest News Integration**
           - Fetches recent news articles about selected stocks
           - Shows publisher, publication date, and article links
           - Provides market context and expert opinions
        
        4. **🔍 Multi-Source Recommendations**
           - App's quantitative model (40/40/20 weighting)
           - Yahoo Finance consensus
           - Analyst consensus from recent ratings
        
        5. **📈 Enhanced Visualizations**
           - Analyst rating distribution charts
           - Recommendation comparison tables
           - Interactive tabs for organized information
        
        Select stocks from the sidebar to begin your analysis!
        """)
        return
    
    # 2. Fetch Data
    raw_data = fetch_data(tickers)
    
    if not raw_data:
        st.error("The selected stocks failed data validation checks or could not be fetched.")
        return

    # 3. Compute Metrics & 4. Apply Scoring
    metrics_df = compute_all_metrics(raw_data)
    scored_df = calculate_kpis_and_total_score(metrics_df)

    # 5. Display results
    results_page(scored_df)

if __name__ == "__main__":
    main()
