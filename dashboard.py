import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import percentileofscore
import time # Used for simulated loading in the UI

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Integrated Expert Screener")

# --- Global State & Session Variables (Mimicking Existing App Flow) ---
if 'tickers' not in st.session_state:
    st.session_state['tickers'] = []
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'
if 'selected_stock' not in st.session_state:
    st.session_state['selected_stock'] = None
if 'analysis_df' not in st.session_state:
    st.session_state['analysis_df'] = pd.DataFrame()
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None

# --- HELPER FUNCTIONS ---

@st.cache_data
def fetch_data(tickers):
    """Fetches price history and key fundamental data using yfinance."""
    if not tickers:
        return None

    # Fetch price data for the last 1 year (for returns and technicals)
    # Fetch 1 year + a buffer for accurate 12M lookbacks
    price_data = yf.download(tickers, period="15mo", interval="1d", group_by='ticker', progress=False)

    data = {}
    for ticker in tickers:
        try:
            # Get current info (P/E, Market Cap, etc.)
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get 1-year price series
            if len(tickers) == 1:
                price_hist = price_data['Close']
            else:
                # Handle multi-index if fetching multiple tickers
                if ticker in price_data.columns.get_level_values(0):
                    price_hist = price_data[ticker]['Close']
                else:
                    st.warning(f"Price data structure error for {ticker}")
                    continue

            # Ensure price_hist is not empty before processing
            if price_hist.empty:
                st.warning(f"No price data found for {ticker}")
                continue
            
            # Basic data structure
            data[ticker] = {
                'MarketCap': info.get('marketCap', np.nan),
                'PE_Ratio': info.get('trailingPE', np.nan),
                'PS_Ratio': info.get('priceToSalesTrailing12Months', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'DebtToEquity': info.get('debtToEquity', np.nan),
                'Beta': info.get('beta', np.nan),
                'CurrentPrice': price_hist.iloc[-1] if not price_hist.empty else np.nan,
                'PriceHistory': price_hist,
            }
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            continue
            
    return data

# --- EXPERT ANALYSIS CORE FUNCTIONS ---

def compute_metrics(data):
    """
    Computes all individual metrics for the Expert Analysis.
    """
    metrics_data = {}
    
    # Days in: 3M ~ 63, 6M ~ 126, 12M ~ 252 trading days
    DAYS_3M = 63
    DAYS_6M = 126
    DAYS_12M = 252

    for ticker, d in data.items():
        price_hist = d.get('PriceHistory')
        if price_hist is None or price_hist.empty:
            metrics_data[ticker] = {}
            continue
        
        # 1. Fundamental Metrics
        funda_metrics = {
            'Value_PE': d.get('PE_Ratio', np.nan),         # Lower is better (L)
            'Value_PS': d.get('PS_Ratio', np.nan),         # Lower is better (L)
            'Profitability_ROE': d.get('ROE', np.nan),     # Higher is better (H)
        }

        # 2. Momentum Metrics 
        ret_3m = (price_hist.iloc[-1] / price_hist.iloc[-DAYS_3M] - 1) * 100 if len(price_hist) >= DAYS_3M else np.nan
        ret_12m = (price_hist.iloc[-1] / price_hist.iloc[-DAYS_12M] - 1) * 100 if len(price_hist) >= DAYS_12M else np.nan

        momentum_metrics = {
            'Momentum_3M': ret_3m,                        # Higher is better (H)
            'Momentum_12M': ret_12m,                      # Higher is better (H)
        }

        # 3. Quality/Stability Metrics
        quality_metrics = {
            'Stability_D_E': d.get('DebtToEquity', np.nan), # Lower is better (L)
            'Stability_Beta': d.get('Beta', np.nan),        # Lower is better (L)
        }

        # 4. Technical Metrics
        try:
            # Simple technical score: Current price relative to 200-day MA
            MA200 = price_hist.rolling(window=200).mean().iloc[-1]
            Tech_MA200_Diff = ((d['CurrentPrice'] / MA200) - 1) * 100
        except:
            Tech_MA200_Diff = np.nan
            
        technical_metrics = {
            'Tech_MA200_Proximity': Tech_MA200_Diff, # Higher is better (H)
        }

        # --- Retrospective Validation Metrics (Historical Returns) ---
        # Compares TODAY's score against PAST performance (retrospective)
        ret_past_3m = (price_hist.iloc[-1] / price_hist.iloc[-DAYS_3M] - 1) * 100 if len(price_hist) >= DAYS_3M else np.nan
        ret_past_6m = (price_hist.iloc[-1] / price_hist.iloc[-DAYS_6M] - 1) * 100 if len(price_hist) >= DAYS_6M else np.nan
        ret_past_12m = (price_hist.iloc[-1] / price_hist.iloc[-DAYS_12M] - 1) * 100 if len(price_hist) >= DAYS_12M else np.nan
        
        validation_metrics = {
            'Hist_Return_3M': ret_past_3m,
            'Hist_Return_6M': ret_past_6m,
            'Hist_Return_12M': ret_past_12m,
        }

        metrics_data[ticker] = {**funda_metrics, **momentum_metrics, **quality_metrics, **technical_metrics, **validation_metrics}

    return pd.DataFrame.from_dict(metrics_data, orient='index')

def normalize_and_score(metrics_df):
    """
    Normalizes metrics to percentile ranks (0-100) and computes final scores.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    analysis_df = metrics_df.copy()
    
    # Define how each raw metric should be scored (Higher is Better 'H', Lower is Better 'L')
    scoring_logic = {
        'Value_PE': 'L',
        'Value_PS': 'L',
        'Profitability_ROE': 'H',
        'Momentum_3M': 'H',
        'Momentum_12M': 'H',
        'Stability_D_E': 'L',
        'Stability_Beta': 'L',
        'Tech_MA200_Proximity': 'H',
    }
    
    # 1. Cross-sectional Normalization (Percentile Rank 0-100)
    for col, logic in scoring_logic.items():
        if col in analysis_df.columns:
            temp_series = analysis_df[col].dropna()
            
            # To handle extreme outliers, you could winsorize/cap values here before percentiling.
            # For simplicity, we use the raw values for percentile rank calculation.
            
            if not temp_series.empty and len(temp_series) > 1:
                # Compute percentile ranks for non-NaN values
                scores = [
                    percentileofscore(temp_series, x, 'weak') 
                    for x in analysis_df[col]
                ]
                
                # Invert score for 'Lower is Better'
                if logic == 'L':
                    scores = [100 - s for s in scores]
                
                # Re-apply NaNs where the original data was NaN
                analysis_df[f'{col}_Score'] = scores
                analysis_df.loc[analysis_df[col].isna(), f'{col}_Score'] = np.nan
            else:
                # Assign NaN or a neutral score if not enough data for cross-section
                analysis_df[f'{col}_Score'] = np.nan 


    # 2. Category Aggregation
    
    # Define the weights for the final score and components
    weights = {
        'Fundamental': 0.35, 
        'Technical': 0.15, 
        'Momentum': 0.30, 
        'Quality/Stability': 0.20
    }
    
    # a. Fundamental Score (Mean of Value_PE_Score, Value_PS_Score, Profitability_ROE_Score)
    funda_cols = [c for c in analysis_df.columns if c.startswith('Value_') or c.startswith('Profitability_')]
    analysis_df['Fundamental_Score'] = analysis_df[[c for c in funda_cols if c.endswith('_Score')]].mean(axis=1)
    
    # b. Technical Score
    analysis_df['Technical_Score'] = analysis_df['Tech_MA200_Proximity_Score']

    # c. Momentum Score (Mean of Momentum_3M_Score, Momentum_12M_Score)
    momentum_cols = [c for c in analysis_df.columns if c.startswith('Momentum_')]
    analysis_df['Momentum_Score'] = analysis_df[[c for c in momentum_cols if c.endswith('_Score')]].mean(axis=1)
    
    # d. Quality/Stability Score (Mean of Stability_D_E_Score, Stability_Beta_Score)
    quality_cols = [c for c in analysis_df.columns if c.startswith('Stability_')]
    analysis_df['Quality/Stability_Score'] = analysis_df[[c for c in quality_cols if c.endswith('_Score')]].mean(axis=1)

    # 3. Final Weighted Score
    score_cols = {
        'Fundamental_Score': weights['Fundamental'],
        'Technical_Score': weights['Technical'],
        'Momentum_Score': weights['Momentum'],
        'Quality/Stability_Score': weights['Quality/Stability']
    }
    
    final_score = 0
    total_weight = 0
    for col, weight in score_cols.items():
        # Multiply score by weight
        score_component = analysis_df[col] * weight
        # Ensure we only add the component if the score is not NaN
        final_score += score_component.fillna(0)
        
        # Calculate the *effective* total weight (excluding NaNs)
        total_weight += np.where(analysis_df[col].isna(), 0, weight)
        
    # Normalize by the effective total weight sum to get a score out of 100
    analysis_df['Final_Weighted_Score'] = np.where(total_weight > 0, final_score / total_weight, np.nan)
    
    # Keep only the relevant scores, returns, and raw metrics
    score_return_cols = [c for c in analysis_df.columns if 'Score' in c or c.startswith('Hist_Return')]
    return analysis_df[score_return_cols + list(metrics_df.columns)].drop_duplicates()


@st.cache_data(show_spinner="Fetching data and running Expert Analysis...")
def run_expert_analysis(tickers):
    """
    Main function to run the entire expert analysis pipeline.
    """
    if not tickers:
        return pd.DataFrame()
        
    # 1. Fetch Raw Data
    raw_data = fetch_data(tickers)
    st.session_state['raw_data'] = raw_data
    
    # 2. Compute Individual Metrics
    metrics_df = compute_metrics(raw_data)
    
    # 3. Normalize and Score
    analysis_df = normalize_and_score(metrics_df)
    
    return analysis_df.sort_values(by='Final_Weighted_Score', ascending=False)

# --- PLOTLY VISUALIZATIONS ---

def plot_radar_chart(df, ticker):
    """Radar chart of category scores for the selected stock."""
    if ticker not in df.index:
        return go.Figure().add_annotation(text="Data not available for selected stock.", showarrow=False)

    scores = df.loc[ticker, ['Fundamental_Score', 'Technical_Score', 'Momentum_Score', 'Quality/Stability_Score']].fillna(0)
    
    # Ensure scores is a series of 4 values (for the 4 axes)
    if len(scores) < 4:
        return go.Figure().add_annotation(text="Incomplete data for radar chart.", showarrow=False)

    fig = go.Figure(data=[
        go.Scatterpolar(
            r=scores.values,
            theta=scores.index.str.replace('_Score', ''),
            fill='toself',
            name=ticker,
            line_color='#007BFF',
            marker_color='#007BFF'
        )
    ])

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                linecolor='gray',
                gridcolor='lightgray'
            ),
            bgcolor='white'
        ),
        showlegend=False,
        title=f"Expert Score Categories for **{ticker}**",
        height=400,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig

def plot_final_score_bar_chart(df):
    """Bar chart of final scores across the filtered universe."""
    df_plot = df.reset_index().rename(columns={'index': 'Ticker'}).dropna(subset=['Final_Weighted_Score'])
    
    fig = px.bar(df_plot, x='Ticker', y='Final_Weighted_Score',
                 title="Final Expert Score Across Universe",
                 color='Final_Weighted_Score',
                 color_continuous_scale=px.colors.sequential.Viridis,
                 height=400)
    fig.update_yaxes(range=[0, 100], title="Final Score (0-100)")
    fig.update_layout(margin=dict(l=30, r=30, t=50, b=30))
    return fig

def plot_score_vs_returns_scatter(df, return_type='Hist_Return_6M'):
    """Scatter plot of Final Score vs historical returns (3M/6M/12M)."""
    df_plot = df.reset_index().rename(columns={'index': 'Ticker'}).dropna(subset=['Final_Weighted_Score', return_type])
    
    if df_plot.empty:
        return go.Figure().add_annotation(text="Not enough valid data for scatter plot.", showarrow=False)

    title_map = {
        'Hist_Return_3M': '3-Month Historical Return (%)',
        'Hist_Return_6M': '6-Month Historical Return (%)',
        'Hist_Return_12M': '12-Month Historical Return (%)',
    }
    
    fig = px.scatter(df_plot, 
                     x='Final_Weighted_Score', y=return_type,
                     hover_data=['Ticker'],
                     title=f"Score vs. {title_map.get(return_type, return_type)} (Retrospective Validation)",
                     labels={'Final_Weighted_Score': 'Final Score (0-100)', return_type: title_map.get(return_type, return_type)},
                     color_discrete_sequence=['#FF4B4B'], # Streamlit red color
                     height=450)
    
    fig.update_xaxes(range=[0, 100])
    fig.update_layout(margin=dict(l=30, r=30, t=50, b=30))
    return fig

def plot_returns_by_quartile_bar(df, return_type='Hist_Return_6M'):
    """Bar chart showing average historical returns by score quartile."""
    
    df_temp = df.copy().dropna(subset=['Final_Weighted_Score', return_type])
    if df_temp.empty or len(df_temp) < 4:
        return go.Figure().add_annotation(text="Not enough data to compute meaningful quartiles (need at least 4 stocks).", showarrow=False)

    # Determine Quartiles
    try:
        df_temp['Score_Quartile'] = pd.qcut(df_temp['Final_Weighted_Score'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')
    except ValueError:
        return go.Figure().add_annotation(text="Not enough unique score values to create 4 quartiles.", showarrow=False)
        
    # Calculate Mean Return per Quartile
    quartile_performance = df_temp.groupby('Score_Quartile')[return_type].mean().reset_index()
    
    title_map = {
        'Hist_Return_3M': '3-Month Avg. Return (%)',
        'Hist_Return_6M': '6-Month Avg. Return (%)',
        'Hist_Return_12M': '12-Month Avg. Return (%)',
    }
    
    fig = px.bar(quartile_performance, x='Score_Quartile', y=return_type,
                 title=f"{title_map.get(return_type, return_type)} by Score Quartile",
                 labels={return_type: title_map.get(return_type, return_type), 'Score_Quartile': 'Final Score Quartile'},
                 color=return_type,
                 color_continuous_scale=px.colors.sequential.Plasma,
                 height=450)
                 
    fig.update_layout(margin=dict(l=30, r=30, t=50, b=30))
    return fig

# --- UI COMPONENTS (Mimicking Existing Flow) ---

def sidebar_menu():
    """Sets up the sidebar with navigation buttons."""
    st.sidebar.header("Navigation")
    
    # Home/Screener Page
    if st.sidebar.button("🏠 Screener Input", key='nav_home'):
        st.session_state['page'] = 'Home'
        
    # Results/Expert Analysis Page
    if st.sidebar.button("📊 Results & Expert Analysis", key='nav_results'):
        if st.session_state['tickers']:
            st.session_state['page'] = 'Results'
        else:
            st.sidebar.warning("Please enter tickers first.")

    # Technical Analysis Page (Can also contain detailed analysis)
    if st.sidebar.button("📈 Technical Details", key='nav_tech'):
        if st.session_state['tickers']:
            st.session_state['page'] = 'Technical'
        else:
            st.sidebar.warning("Please enter tickers first.")

    # Selection for detailed view (used on Results/Technical pages)
    if st.session_state['analysis_df'] is not None and not st.session_state['analysis_df'].empty:
        st.sidebar.markdown("---")
        
        # Sort tickers by Final Score for intuitive selection
        sorted_tickers = st.session_state['analysis_df'].index.tolist()
        
        # Determine the initial selection
        if st.session_state['selected_stock'] not in sorted_tickers:
             initial_selection = sorted_tickers[0] if sorted_tickers else None
        else:
             initial_selection = st.session_state['selected_stock']
             
        st.session_state['selected_stock'] = st.sidebar.selectbox(
            "Select Stock for Detail View:", 
            options=sorted_tickers,
            index=sorted_tickers.index(initial_selection) if initial_selection in sorted_tickers else 0
        )
    elif st.session_state['tickers']:
         st.sidebar.markdown("---")
         st.session_state['selected_stock'] = st.sidebar.selectbox(
            "Select Stock for Detail View:", 
            st.session_state['tickers']
        )


def home_page():
    """The main input page."""
    st.title("Screener Input: Ticker Universe")
    st.info("Enter tickers separated by commas (e.g., AAPL, GOOGL, MSFT, AMZN, TSLA)")

    # User Input for Tickers
    ticker_input = st.text_input(
        "Enter Tickers:", 
        value=", ".join(st.session_state['tickers']),
        key='ticker_input'
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Run Screener & Analysis"):
            raw_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
            
            if not raw_tickers:
                st.error("Please enter valid tickers.")
                return

            # Update session state with new tickers
            st.session_state['tickers'] = raw_tickers
            st.session_state['analysis_df'] = pd.DataFrame() # Clear previous run
            
            # Run the new analysis pipeline
            analysis_df = run_expert_analysis(raw_tickers)
            st.session_state['analysis_df'] = analysis_df
            
            # Automatically navigate to Results page upon completion
            if not analysis_df.empty:
                st.session_state['page'] = 'Results'
                st.success("Analysis Complete!")
                st.experimental_rerun()
            else:
                 st.error("Analysis failed or returned no valid data.")
            
    with col2:
        # Keep placeholder or original content intact
        st.empty()

def results_page():
    """Displays the aggregated results and Expert Analysis summaries."""
    st.title("📊 Expert Analysis Results & Scores")

    df = st.session_state['analysis_df']
    if df.empty:
        st.error("No valid data available. Please run the screener first.")
        st.session_state['page'] = 'Home'
        return

    # --- Section 1: Final Score Table ---
    st.header("1. Final Expert Scores & Components")
    display_cols = [
        'Final_Weighted_Score', 'Fundamental_Score', 'Technical_Score', 
        'Momentum_Score', 'Quality/Stability_Score', 
        'Hist_Return_6M' # Include a key validation metric here
    ]
    st.dataframe(
        df[display_cols].style.format({
            'Final_Weighted_Score': "{:.1f}", 
            'Fundamental_Score': "{:.1f}", 
            'Technical_Score': "{:.1f}", 
            'Momentum_Score': "{:.1f}", 
            'Quality/Stability_Score': "{:.1f}",
            'Hist_Return_6M': "{:.1f}%",
        }).background_gradient(cmap='viridis', subset=['Final_Weighted_Score']),
        use_container_width=True
    )
    
    st.markdown("---")

    # --- Section 2: Final Score Distribution and Validation Plots ---
    st.header("2. Universe-Wide Insights (Backtest-Style Validation)")
    
    col_score_bar, col_spacer = st.columns([1, 0.01])
    with col_score_bar:
        st.subheader("Final Score Distribution")
        st.plotly_chart(plot_final_score_bar_chart(df), use_container_width=True)

    # 2.2 Score vs. Historical Returns (Validation)
    st.subheader("Retrospective Validation: Score vs. Historical Returns")
    
    return_option = st.selectbox(
        "Select Historical Return Period for Validation:",
        options=['Hist_Return_3M', 'Hist_Return_6M', 'Hist_Return_12M'],
        format_func=lambda x: x.replace('Hist_Return_', '')
    )
    
    col_scatter, col_quartile = st.columns(2)
    
    with col_scatter:
        st.plotly_chart(plot_score_vs_returns_scatter(df, return_option), use_container_width=True)

    with col_quartile:
        st.plotly_chart(plot_returns_by_quartile_bar(df, return_option), use_container_width=True)
        
    st.caption("**:red[Retrospective Validation Note:]** Compares TODAY's score against PAST returns. For a robust backtest, scores must be calculated on historical dates and forward returns measured.")
    
    st.markdown("---")
    
    # --- Section 3: Detailed Stock View (Radar Chart) ---
    st.header("3. Detailed Stock Analysis")
    selected_stock = st.session_state['selected_stock']
    
    if selected_stock and selected_stock in df.index:
        st.subheader(f"Category Breakdown for {selected_stock}")
        
        col_radar, col_detail = st.columns([1, 2])
        
        with col_radar:
            st.plotly_chart(plot_radar_chart(df, selected_stock), use_container_width=True)
            
        with col_detail:
            st.subheader("Raw Metrics and Normalized Scores")
            stock_data = df.loc[[selected_stock]].T.rename(columns={selected_stock: 'Value'})
            
            # Filter the display to key metrics + scores
            key_metrics_and_scores = [
                'Final_Weighted_Score', 'Fundamental_Score', 'Momentum_Score', 'Technical_Score', 'Quality/Stability_Score',
                'Value_PE', 'Value_PE_Score', 'Profitability_ROE', 'Profitability_ROE_Score',
                'Momentum_3M', 'Momentum_3M_Score', 'Tech_MA200_Proximity', 'Tech_MA200_Proximity_Score',
                'Stability_Beta', 'Stability_Beta_Score', 'Hist_Return_6M'
            ]
            
            # Prioritize the display order
            display_order = [idx for idx in key_metrics_and_scores if idx in stock_data.index]
            stock_data = stock_data.reindex(display_order)
            stock_data = stock_data[stock_data['Value'].notna()]
            
            # Format display
            def format_metric(index, value):
                if '_Score' in index or index.startswith('Final_'):
                    return f"{value:.1f}"
                elif 'Hist_Return' in index or 'Momentum' in index or 'Proximity' in index:
                    return f"{value:.1f}%"
                elif 'D_E' in index:
                    return f"{value:.0f}"
                else:
                    return f"{value:,.2f}"

            styled_df = stock_data.style.apply(lambda x: [
                'background-color: #e6f7ff; font-weight: bold' if 'Score' in idx or idx.startswith('Final_') else '' 
                for idx in stock_data.index
            ], axis=0).format(format_metric)
            
            st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning("Please select a stock in the sidebar for detailed analysis.")


def technical_page():
    """The technical details page, also enhanced with score details."""
    st.title("📈 Technical Analysis & Score Components")
    df = st.session_state['analysis_df']
    selected_stock = st.session_state['selected_stock']
    raw_data = st.session_state.get('raw_data', {})
    
    if df.empty or not selected_stock or selected_stock not in df.index:
        st.error("No data available or stock not selected. Please run the screener first.")
        return

    st.subheader(f"Detailed Technical and Momentum Metrics for **{selected_stock}**")

    # Display key metrics related to Technical/Momentum
    metrics_to_show = [
        'Technical_Score', 'Tech_MA200_Proximity', 'Tech_MA200_Proximity_Score',
        'Momentum_Score', 'Momentum_3M', 'Momentum_3M_Score', 'Momentum_12M', 'Momentum_12M_Score'
    ]
    
    # Filter to only show relevant rows and format
    stock_df = df.loc[[selected_stock]].T
    tech_data = stock_df.loc[stock_df.index.intersection(metrics_to_show)]
    
    def format_tech_metric(index, value):
        if 'Score' in index:
            return f"{value:.1f}"
        elif 'Momentum' in index or 'Proximity' in index:
            return f"{value:.1f}%"
        else:
            return f"{value:,.2f}"

    styled_tech_df = tech_data.style.apply(lambda x: [
        'background-color: #e6f7ff; font-weight: bold' if 'Score' in idx else '' 
        for idx in tech_data.index
    ], axis=0).format(format_tech_metric)

    st.dataframe(styled_tech_df, use_container_width=True)

    # Simple Price Plot (as a placeholder for a complex technical chart)
    st.subheader("Price History (Last 1 Year)")
    
    try:
        if selected_stock in raw_data and 'PriceHistory' in raw_data[selected_stock]:
            price_hist = raw_data[selected_stock]['PriceHistory']
            
            # Calculate 200-day MA for technical context
            price_hist = pd.DataFrame(price_hist).rename(columns={'Close': 'Price'})
            price_hist['MA200'] = price_hist['Price'].rolling(window=200).mean()
            
            fig = px.line(price_hist, 
                          y=['Price', 'MA200'], 
                          title=f"{selected_stock} Price History vs 200-Day MA",
                          color_discrete_map={'Price': '#007BFF', 'MA200': '#FF4B4B'})
            
            fig.update_layout(yaxis_title="Price ($)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Price history data not available in session state.")
            
    except Exception as e:
        st.error(f"Could not load price history for chart: {e}")

# --- MAIN APP EXECUTION ---

def main():
    """The main function to run the application and handle navigation."""
    
    # 1. Sidebar (Always present)
    sidebar_menu()

    # 2. Main Content (Page Switcher)
    if st.session_state['page'] == 'Home':
        home_page()
    elif st.session_state['page'] == 'Results':
        results_page()
    elif st.session_state['page'] == 'Technical':
        technical_page()
        
    st.markdown("---")
    st.caption("Integrated Expert Screener: Multi-Factor Scoring Engine")

if __name__ == "__main__":
    main()
