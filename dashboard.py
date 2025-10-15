import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as pdr
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import percentileofscore
import time 

# --- Configuration ---
st.set_page_config(
    page_title="NSE Hybrid Stock Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 1. Data Fetching (Hybrid Approach: Stooq for Price, yfinance for Fundamentals) ---

@st.cache_data(show_spinner="Fetching and validating stock data (This may take a minute)...")
def fetch_data(tickers):
    """
    Fetches price history using pandas-datareader (Stooq) and
    key fundamental data using yfinance. Includes a yfinance fallback.
    """
    if not tickers:
        return None

    end_date = datetime.now()
    # Request data starting 18 months ago to ensure at least 12 months (200 days) of history
    start_date = end_date - timedelta(days=18 * 30) 

    data = {}
    
    for full_ticker in tickers:
        # Ticker cleaning for Stooq: Use the symbol without the '.NS' suffix
        # Note: Stooq often expects clean tickers, but its reliability for NSE is variable.
        ticker_clean = full_ticker.replace('.NS', '')
        
        try:
            # --- 1. Fetch Price Data via pandas-datareader (Stooq) ---
            # Using 'stooq' as the source
            price_data = pdr.DataReader(ticker_clean, 'stooq', start=start_date, end=end_date)
            
            # Stooq returns date as index, and columns are often capitalized.
            price_data = price_data.rename(columns={
                'Close': 'Close', 
                'Open': 'Open', 
                'High': 'High', 
                'Low': 'Low', 
                'Volume': 'Volume'
            }).sort_index() # Sort by date ascending
            
            # We focus on the Close price for calculations
            price_hist = price_data['Close']
            
            # --- 2. Fetch Fundamental Data via yfinance (using the full .NS ticker) ---
            stock = yf.Ticker(full_ticker)
            info = stock.info
            
            # CRITICAL CHECK: Require at least 200 days of valid historical data
            valid_days = price_hist.dropna().shape[0] if price_hist is not None else 0
            
            if price_hist is None or price_hist.empty or valid_days < 200:
                # If Stooq data is bad, raise an error to trigger the YF Fallback
                raise ValueError(f"Stooq data incomplete (found only {valid_days} days).")
            
            # --- 3. Build Data Structure ---
            data[full_ticker] = {
                'MarketCap': info.get('marketCap', np.nan),
                'PE_Ratio': info.get('trailingPE', np.nan),
                'PS_Ratio': info.get('priceToSalesTrailing12Months', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'DebtToEquity': info.get('debtToEquity', np.nan),
                'Beta': info.get('beta', np.nan),
                'CurrentPrice': price_hist.iloc[-1],
                'PriceHistory': price_hist,
            }

        except Exception as e:
            # --- Yfinance Fallback for Price History ---
            st.warning(f"Failed to fetch data for {full_ticker} via Stooq/YF Info: {e}. Trying YF Historical Fallback.")
            try:
                # Retry fetching price and fundamentals using yfinance for robustness
                stock = yf.Ticker(full_ticker)
                info = stock.info
                # Fetch 15 months of historical data
                yf_hist = stock.history(period="15mo")['Close']
                
                valid_days = yf_hist.dropna().shape[0] if yf_hist is not None else 0
                if valid_days < 200:
                    st.error(f"Skipping {full_ticker}: YF Fallback failed. Found only {valid_days} days. (Required > 200 days).")
                    continue
                    
                # Use the YF fetched data for the final output
                data[full_ticker] = {
                    'MarketCap': info.get('marketCap', np.nan),
                    'PE_Ratio': info.get('trailingPE', np.nan),
                    'PS_Ratio': info.get('priceToSalesTrailing12Months', np.nan),
                    'ROE': info.get('returnOnEquity', np.nan),
                    'DebtToEquity': info.get('debtToEquity', np.nan),
                    'Beta': info.get('beta', np.nan),
                    'CurrentPrice': yf_hist.iloc[-1],
                    'PriceHistory': yf_hist,
                }
                st.info(f"Successfully used YF historical fallback for {full_ticker}")
            except Exception as e_yf:
                st.error(f"Skipping {full_ticker}: YF fallback also failed: {e_yf}")
            continue
            
    return data

# ----------------------------------------------------------------------

# --- 2. Metric Calculation ---

def compute_metrics(data):
    """Calculates Price Change, Volatility, and cleans data."""
    processed_data = {}
    for ticker, d in data.items():
        price_hist = d['PriceHistory'].dropna()
        
        if price_hist.empty:
            continue
        
        # 1-Year Price Change
        # Use a rolling 252-day window or the whole history if less than 252 points are available
        lookback_period = min(252, len(price_hist)) 
        
        start_price = price_hist.iloc[-lookback_period]
        end_price = price_hist.iloc[-1]
        price_change = ((end_price - start_price) / start_price) * 100 if start_price != 0 else np.nan
        
        # 1-Year Volatility (Standard Deviation of Daily Log Returns)
        log_returns = np.log(price_hist / price_hist.shift(1)).dropna()
        # Annualize the volatility (252 trading days)
        volatility = log_returns.std() * np.sqrt(252) 

        processed_data[ticker] = {
            'Price Change (%)': price_change,
            'Volatility (Annual)': volatility,
            'MarketCap': d['MarketCap'],
            'PE_Ratio': d['PE_Ratio'],
            'PS_Ratio': d['PS_Ratio'],
            'ROE': d['ROE'],
            'DebtToEquity': d['DebtToEquity'],
            'Beta': d['Beta'],
            'CurrentPrice': d['CurrentPrice'],
            'PriceHistory': price_hist,
        }
    return pd.DataFrame.from_dict(processed_data, orient='index')

# ----------------------------------------------------------------------

# --- 3. Scoring and Ranking ---

def normalize_and_score(df, criteria):
    """Normalizes metrics and calculates a composite score."""
    if df.empty:
        return df

    scored_df = df.copy()
    
    # Apply percentile-based scoring for robustness
    for metric, direction in criteria.items():
        if metric in scored_df.columns and not scored_df[metric].isnull().all():
            
            # Calculate percentile rank (0 to 100)
            percentile_ranks = scored_df[metric].apply(lambda x: percentileofscore(scored_df[metric].dropna(), x))
            
            # Apply direction: Higher is better (P/C), Lower is better (100 - P/C)
            if direction == 'lower':
                scored_df[f'{metric} Score'] = 100 - percentile_ranks
            else: # 'higher'
                scored_df[f'{metric} Score'] = percentile_ranks
        else:
            scored_df[f'{metric} Score'] = 0

    # Calculate Composite Score (Simple average of component scores)
    score_columns = [col for col in scored_df.columns if 'Score' in col]
    if score_columns:
        scored_df['Composite Score'] = scored_df[score_columns].mean(axis=1)
        scored_df = scored_df.sort_values('Composite Score', ascending=False)
    
    return scored_df

# ----------------------------------------------------------------------

# --- 4. Streamlit UI Functions ---

def sidebar_inputs():
    """Defines the sidebar layout and retrieves user inputs."""
    st.sidebar.header("🎯 Screener Parameters")

    # Example NSE Tickers (replace with your desired list)
    default_tickers = ["TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS", "TATAMOTORS.NS"]
    
    ticker_input = st.sidebar.text_area(
        "Enter NSE Tickers (one per line, ensure '.NS' suffix is used):",
        value="\n".join(default_tickers)
    )
    
    tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]

    st.sidebar.header("📊 Scoring Criteria")
    
    # Define metrics and the direction for a 'good' score
    metric_criteria = {
        'PE_Ratio': ('lower', 'P/E Ratio (Lower is Better)'),
        'ROE': ('higher', 'ROE (Return on Equity)'),
        'Price Change (%)': ('higher', '1-Year Price Change (%)'),
        'Volatility (Annual)': ('lower', 'Volatility (Risk)'),
        'DebtToEquity': ('lower', 'Debt to Equity'),
    }

    # The weights for this simple implementation are binary (0 or 1), 
    # as the current scoring function uses a simple mean.
    # We use the slider only to determine which metrics are included (weight > 0).
    selected_criteria = {}
    st.sidebar.markdown("_Set weight to 1 to include the metric in scoring._")
    for metric, (direction, label) in metric_criteria.items():
        weight = st.sidebar.slider(
            label,
            min_value=0,
            max_value=1, # Change max to 1 for simplicity with current scoring
            value=1, # Default to 1 (included)
            step=1,
            key=f"weight_{metric}"
        )
        if weight > 0:
            selected_criteria[metric] = {'direction': direction, 'weight': weight}
            
    # Convert weights structure to the format expected by normalize_and_score
    scoring_criteria = {m: d['direction'] for m, d in selected_criteria.items()}
    
    if not tickers:
        st.error("Please enter at least one NSE ticker.")
        st.stop()
        
    return tickers, scoring_criteria

def results_page(scored_df):
    """Displays the main results and visualizations."""
    st.title("🏆 NSE Stock Screener Results")

    if scored_df.empty:
        st.warning("No stocks passed the data validation checks. Please check tickers/data history.")
        return

    # --- Summary Table ---
    st.header("1. Ranked Stock Summary")
    
    # Select columns for display
    display_cols = ['Composite Score', 'CurrentPrice', 'Price Change (%)', 
                    'PE_Ratio', 'ROE', 'DebtToEquity', 'Volatility (Annual)', 'MarketCap']
    
    # Format and display
    formatted_df = scored_df[display_cols].copy()
    formatted_df.columns = ['Composite Score', 'Price (INR)', '1Y Change (%)', 
                            'P/E', 'ROE', 'D/E', 'Volatility', 'Mkt Cap']
    
    # Apply formatting
    formatted_df['Composite Score'] = formatted_df['Composite Score'].round(2)
    formatted_df['Price (INR)'] = formatted_df['Price (INR)'].apply(lambda x: f"₹{x:,.2f}")
    formatted_df['1Y Change (%)'] = formatted_df['1Y Change (%)'].round(2).astype(str) + '%'
    # Convert Market Cap (in USD or INR) from raw value to Crore (1 Crore = 10^7)
    # Note: Market Cap data from yfinance for .NS stocks is often in USD, 
    # but we display it as a large number and label it in Crores for context.
    formatted_df['Mkt Cap'] = (formatted_df['Mkt Cap'] / 10**7).round(2).apply(lambda x: f"₹{x:,.2f} Cr")
    formatted_df['ROE'] = (formatted_df['ROE'] * 100).round(2).astype(str) + '%'
    formatted_df['P/E'] = formatted_df['P/E'].round(2)
    formatted_df['D/E'] = formatted_df['D/E'].round(2)
    formatted_df['Volatility'] = formatted_df['Volatility'].round(3)

    st.dataframe(
        formatted_df,
        use_container_width=True,
        column_config={
            "Composite Score": st.column_config.NumberColumn("Composite Score", format="%.2f", help="Average of all weighted percentile scores (0-100)."),
        }
    )

    # --- Scatter Plot (Risk vs. Reward) ---
    st.header("2. Risk vs. Reward (Volatility vs. Price Change)")
    fig = px.scatter(
        scored_df,
        x='Volatility (Annual)',
        y='Price Change (%)',
        color='Composite Score',
        text=scored_df.index,
        size='MarketCap',
        hover_data={'PE_Ratio': ':.2f', 'ROE': ':.2%'},
        title="Stock Performance: Risk (Volatility) vs. Return (Price Change)",
        labels={'Volatility (Annual)': 'Annualized Volatility (Risk)', 'Price Change (%)': '1-Year Price Change (%)'}
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

    # --- Detailed Price Charts ---
    st.header("3. Detailed Price History")
    selected_ticker = st.selectbox("Select a Ticker to View Price Chart", scored_df.index)
    
    if selected_ticker:
        price_hist = scored_df.loc[selected_ticker, 'PriceHistory']
        fig_price = go.Figure(data=[
            go.Scatter(x=price_hist.index, y=price_hist, mode='lines', name='Price')
        ])
        fig_price.update_layout(
            title=f"Price History for {selected_ticker}",
            xaxis_title="Date",
            yaxis_title="Closing Price (INR)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_price, use_container_width=True)

# ----------------------------------------------------------------------

# --- 5. Main Execution ---

def main():
    """The main function to run the Streamlit application."""
    st.sidebar.title("📈 Hybrid Screener")
    st.sidebar.markdown("Fetches data using **Stooq** (Price) and **yfinance** (Fundamentals) for resilience.")

    # Get user inputs
    tickers, scoring_criteria = sidebar_inputs()
    
    # Fetch Data
    raw_data = fetch_data(tickers)
    
    if not raw_data:
        st.error("No valid data was returned for the selected tickers.")
        return

    # Compute additional metrics (Price Change, Volatility)
    metrics_df = compute_metrics(raw_data)

    # Apply scoring and ranking
    scored_df = normalize_and_score(metrics_df, scoring_criteria)

    # Display results
    results_page(scored_df)

if __name__ == "__main__":
    main()
