import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.express as px
from scipy.stats import percentileofscore

# --- Configuration ---
st.set_page_config(
    page_title="NSE Fundamental Stock Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 1. Data Fetching (Current Fundamentals Only) ---

@st.cache_data(show_spinner="Fetching current fundamental data...")
def fetch_data(tickers):
    """
    Fetches only current fundamental data and price using yfinance.
    Historical data fetching is removed as requested.
    """
    if not tickers:
        return None

    data = {}
    
    for full_ticker in tickers:
        try:
            stock = yf.Ticker(full_ticker)
            info = stock.info
            
            # CRITICAL CHECK: Ensure we get a current price
            current_price = info.get('currentPrice', info.get('regularMarketPrice', np.nan))
            
            if pd.isna(current_price):
                 st.warning(f"Skipping {full_ticker}: Could not find a valid current price or essential data.")
                 continue

            data[full_ticker] = {
                'MarketCap': info.get('marketCap', np.nan),
                'PE_Ratio': info.get('trailingPE', np.nan),
                'PS_Ratio': info.get('priceToSalesTrailing12Months', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'DebtToEquity': info.get('debtToEquity', np.nan),
                'Beta': info.get('beta', np.nan),
                'CurrentPrice': current_price,
            }

        except Exception as e:
            st.error(f"Skipping {full_ticker}: Failed to fetch data via yfinance: {e}")
            
    return pd.DataFrame.from_dict(data, orient='index')


# ----------------------------------------------------------------------
# --- 2. Metric Calculation (Simplified - No History) ---

def compute_metrics(df):
    """Placeholder: No historical metrics needed. Returns the dataframe directly."""
    # Ensure numerical columns are handled correctly
    for col in ['MarketCap', 'PE_Ratio', 'PS_Ratio', 'ROE', 'DebtToEquity', 'Beta', 'CurrentPrice']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ----------------------------------------------------------------------

# --- 3. Scoring and Ranking (Only using Fundamentals) ---

def normalize_and_score(df, criteria):
    """Normalizes fundamental metrics and calculates a composite score."""
    if df.empty:
        return df

    scored_df = df.copy()
    
    # Apply percentile-based scoring
    for metric, direction in criteria.items():
        if metric in scored_df.columns and not scored_df[metric].isnull().all():
            
            # Calculate percentile rank (0 to 100)
            valid_data = scored_df[metric].dropna()
            if valid_data.empty:
                 scored_df[f'{metric} Score'] = 0
                 continue
                 
            percentile_ranks = scored_df[metric].apply(lambda x: percentileofscore(valid_data, x) if not pd.isna(x) else 0)
            
            # Apply direction
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

    # Example NSE Tickers
    default_tickers = ["TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS", "TATAMOTORS.NS"]
    
    ticker_input = st.sidebar.text_area(
        "Enter NSE Tickers (one per line, ensure '.NS' suffix is used):",
        value="\n".join(default_tickers)
    )
    
    tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]

    st.sidebar.header("📊 Scoring Criteria (Fundamentals Only)")
    
    # Define metrics and the direction for a 'good' score
    metric_criteria = {
        'PE_Ratio': ('lower', 'P/E Ratio (Lower is Better)'),
        'ROE': ('higher', 'ROE (Return on Equity)'),
        'DebtToEquity': ('lower', 'Debt to Equity'),
        'PS_Ratio': ('lower', 'P/S Ratio (Lower is Better)'),
    }

    selected_criteria = {}
    st.sidebar.markdown("_Set weight to 1 to include the metric in scoring._")
    for metric, (direction, label) in metric_criteria.items():
        weight = st.sidebar.slider(
            label,
            min_value=0,
            max_value=1, 
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
    st.title("🏆 NSE Fundamental Stock Screener Results")

    if scored_df.empty:
        st.warning("No stocks passed the data fetching checks.")
        return

    # --- Summary Table ---
    st.header("1. Ranked Stock Summary (By Fundamental Score)")
    
    # Select columns for display
    display_cols = ['Composite Score', 'CurrentPrice', 'PE_Ratio', 'ROE', 
                    'DebtToEquity', 'PS_Ratio', 'MarketCap', 'Beta']
    
    # Format and display
    formatted_df = scored_df[display_cols].copy()
    formatted_df.columns = ['Composite Score', 'Price (INR)', 'P/E', 'ROE', 
                            'D/E', 'P/S', 'Mkt Cap', 'Beta']
    
    # Apply formatting
    formatted_df['Composite Score'] = formatted_df['Composite Score'].round(2)
    formatted_df['Price (INR)'] = formatted_df['Price (INR)'].apply(lambda x: f"₹{x:,.2f}")
    
    # Market Cap conversion (using 10^7 for Crores)
    formatted_df['Mkt Cap'] = (formatted_df['Mkt Cap'] / 10**7).round(2).apply(lambda x: f"₹{x:,.2f} Cr")
    
    # Percentage format for ROE
    formatted_df['ROE'] = (formatted_df['ROE'] * 100).round(2).astype(str) + '%'
    
    # Rounding for ratios/beta
    formatted_df['P/E'] = formatted_df['P/E'].round(2)
    formatted_df['D/E'] = formatted_df['D/E'].round(2)
    formatted_df['P/S'] = formatted_df['P/S'].round(2)
    formatted_df['Beta'] = formatted_df['Beta'].round(2)


    st.dataframe(
        formatted_df,
        use_container_width=True,
        column_config={
            "Composite Score": st.column_config.NumberColumn("Composite Score", format="%.2f", help="Average of all percentile scores (0-100)."),
        }
    )

    # --- Scatter Plot (Value vs. Growth/Risk Proxy) ---
    st.header("2. Valuation vs. Risk (P/E vs. Beta)")
    
    # Use P/E (Valuation) and Beta (Risk Proxy) since volatility is removed
    # Filter out NaNs for plot
    plot_df = scored_df.dropna(subset=['PE_Ratio', 'Beta', 'Composite Score'])

    fig = px.scatter(
        plot_df,
        x='PE_Ratio',
        y='Beta',
        color='Composite Score',
        text=plot_df.index,
        size='MarketCap',
        hover_data={'ROE': ':.2%'},
        title="Valuation (P/E) vs. Market Risk (Beta)",
        labels={'PE_Ratio': 'P/E Ratio (Lower is better)', 'Beta': 'Beta (Market Risk)'},
        color_continuous_scale=px.colors.sequential.Viridis_r # Reverse scale for better visualization
    )
    
    # Add quadrants to help interpretation (e.g., median lines)
    median_pe = plot_df['PE_Ratio'].median()
    median_beta = plot_df['Beta'].median()
    
    fig.add_vline(x=median_pe, line_width=1, line_dash="dash", line_color="red", name=f"Median P/E ({median_pe:.2f})")
    fig.add_hline(y=median_beta, line_width=1, line_dash="dash", line_color="blue", name=f"Median Beta ({median_beta:.2f})")
    
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# --- 5. Main Execution ---

def main():
    """The main function to run the Streamlit application."""
    st.sidebar.title("📈 Fundamental Screener")
    st.sidebar.markdown("Fetches **current data only** using **yfinance**.")

    # Get user inputs
    tickers, scoring_criteria = sidebar_inputs()
    
    # Fetch Data
    metrics_df = fetch_data(tickers)
    
    if metrics_df.empty:
        st.error("No valid data was returned for the selected tickers after filtering.")
        return

    # Compute additional metrics (currently does nothing but clean data)
    metrics_df = compute_metrics(metrics_df)

    # Apply scoring and ranking
    scored_df = normalize_and_score(metrics_df, scoring_criteria)

    # Display results
    results_page(scored_df)

if __name__ == "__main__":
    main()
