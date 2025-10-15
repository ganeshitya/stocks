import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import json
import os
import io

# --- Configuration ---
st.set_page_config(
    page_title="Stock Recommendation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# --- Data Fetching with Recommendations ---

@st.cache_data(show_spinner="Fetching stock data and recommendations...")
def fetch_stock_recommendations(tickers):
    """
    Fetches recommendations and key data from yfinance.
    """
    if not tickers:
        return {}

    data = {}
    
    for full_ticker in tickers:
        try:
            stock = yf.Ticker(full_ticker)
            info = stock.info
            
            # Get current price
            current_price = info.get('currentPrice', info.get('regularMarketPrice', np.nan))
            
            if pd.isna(current_price) or current_price == 0:
                continue
            
            sector = info.get('sector', 'N/A')
            
            # --- Fetch Analyst Recommendations ---
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
            
            # --- Fetch Recent News ---
            try:
                news = stock.news[:5] if hasattr(stock, 'news') and stock.news else []
                news_items = extract_news_items(news)
            except:
                news_items = []
            
            data[full_ticker] = {
                # Basic Info
                'Company_Name': info.get('longName', full_ticker),
                'Sector': sector,
                'CurrentPrice': current_price,
                
                # Yahoo Finance Recommendation
                'YF_Recommendation': info.get('recommendationKey', 'N/A').upper().replace('STRONG_BUY', 'STRONG BUY').replace('_', ' '),
                
                # Analyst Ratings
                'Analyst_Ratings': analyst_ratings,
                
                # Target Prices
                'Target_Mean': info.get('targetMeanPrice', np.nan),
                'Target_High': info.get('targetHighPrice', np.nan),
                'Target_Low': info.get('targetLowPrice', np.nan),
                
                # Additional Metrics for Context
                'PE_Ratio': info.get('trailingPE', np.nan),
                'Market_Cap': info.get('marketCap', np.nan),
                'Dividend_Yield': info.get('dividendYield', np.nan),
                '52Week_High': info.get('fiftyTwoWeekHigh', np.nan),
                '52Week_Low': info.get('fiftyTwoWeekLow', np.nan),
                
                # News
                'News': news_items,
            }

        except Exception as e:
            continue
            
    return data

def extract_analyst_ratings(recommendations_df):
    """
    Extracts and calculates weighted analyst ratings from recommendations DataFrame.
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
    
    # Get the most recent recommendations
    recent_date = datetime.now() - timedelta(days=90)
    recent_recs = recommendations_df[recommendations_df.index >= recent_date]
    
    if recent_recs.empty:
        recent_recs = recommendations_df.tail(10)
    
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
    for item in news_list[:5]:
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

# --- Visualization ---

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

def get_recommendation_icon(recommendation):
    """Returns an icon based on recommendation."""
    if recommendation in ['STRONG BUY']:
        return '🟢'
    elif recommendation in ['BUY']:
        return '🟡'
    elif recommendation in ['HOLD']:
        return '🟠'
    else:
        return '🔴'

# --- Sidebar ---

def sidebar_inputs():
    """Defines the sidebar layout and retrieves user selected tickers."""
    st.sidebar.title("📊 Stock Recommendations")
    st.sidebar.markdown("---")
    
    st.sidebar.header("ℹ️ About This Dashboard")
    st.sidebar.markdown("""
        Get **Buy, Sell, or Hold** recommendations from multiple expert sources:
        
        - 📊 **Yahoo Finance Consensus**
        - 👥 **Analyst Ratings** (Aggregated)
        - 🎯 **Target Price Analysis**
        - 📰 **Latest Market News**
    """)
    st.sidebar.markdown("---")
    
    display_names, symbol_map = parse_nifty_symbols()
    
    default_names = [
        "RELIANCE (Reliance Industries Ltd.)", 
        "TCS (Tata Consultancy Services Ltd.)", 
        "INFY (Infosys Ltd.)",
        "HDFCBANK (HDFC Bank Ltd.)",
    ]
    
    initial_selection = [name for name in default_names if name in display_names]
    
    selected_names = st.sidebar.multiselect(
        "Select Stocks (NSE)",
        options=display_names,
        default=initial_selection,
        help="Choose stocks to view their Buy/Sell/Hold recommendations"
    )
    
    selected_tickers = [symbol_map[name] for name in selected_names]
    
    return selected_tickers 

# --- Results Page ---

def results_page(stock_data):
    """Displays recommendations and analysis."""
    
    st.title("📊 Stock Recommendations Dashboard")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.markdown("---")

    if not stock_data:
        st.error("No stock data available. Please select stocks from the sidebar.")
        return

    # Convert to DataFrame for display
    df_list = []
    for ticker, data in stock_data.items():
        analyst_ratings = data['Analyst_Ratings']
        
        df_list.append({
            'Ticker': ticker.replace('.NS', ''),
            'Company': data['Company_Name'],
            'Sector': data['Sector'],
            'Current Price': data['CurrentPrice'],
            'YF Recommendation': data['YF_Recommendation'],
            'Analyst Consensus': analyst_ratings['consensus'],
            'Num Analysts': analyst_ratings['total_analysts'],
            'Target Price': data['Target_Mean'],
            'Upside %': ((data['Target_Mean'] - data['CurrentPrice']) / data['CurrentPrice'] * 100) if not pd.isna(data['Target_Mean']) and data['CurrentPrice'] > 0 else np.nan,
        })
    
    display_df = pd.DataFrame(df_list)
    
    # --- 1. Summary Table ---
    st.header("1. 📋 Recommendations Summary")
    
    # Format the display
    formatted_df = display_df.copy()
    formatted_df['Current Price'] = formatted_df['Current Price'].apply(lambda x: f"₹{x:,.2f}")
    formatted_df['Target Price'] = formatted_df['Target Price'].apply(lambda x: f"₹{x:,.2f}" if not pd.isna(x) else 'N/A')
    formatted_df['Upside %'] = formatted_df['Upside %'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else 'N/A')
    
    # Add icons
    formatted_df['YF Rec'] = display_df['YF Recommendation'].apply(lambda x: f"{get_recommendation_icon(x)} {x}")
    formatted_df['Analyst Rec'] = display_df['Analyst Consensus'].apply(lambda x: f"{get_recommendation_icon(x)} {x}")
    
    # Select columns for display
    display_cols = ['Ticker', 'Company', 'Current Price', 'YF Rec', 'Analyst Rec', 'Num Analysts', 'Target Price', 'Upside %']
    st.dataframe(formatted_df[display_cols], use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Legend:**
    - 🟢 **Strong Buy** / Buy with high confidence
    - 🟡 **Buy** / Positive outlook
    - 🟠 **Hold** / Neutral stance
    - 🔴 **Sell** / Negative outlook
    """)
    
    # --- 2. Detailed Analysis ---
    st.header("2. 🔍 Detailed Stock Analysis")
    
    selected_ticker = st.selectbox(
        "Select a stock for detailed analysis",
        options=list(stock_data.keys()),
        format_func=lambda x: f"{x.replace('.NS', '')} - {stock_data[x]['Company_Name']}"
    )
    
    if selected_ticker:
        data = stock_data[selected_ticker]
        analyst_ratings = data['Analyst_Ratings']
        
        st.subheader(f"📈 {data['Company_Name']} ({selected_ticker.replace('.NS', '')})")
        st.caption(f"Sector: {data['Sector']}")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Recommendations", "👥 Analyst Ratings", "📰 Latest News"])
        
        # TAB 1: Recommendations Overview
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"₹{data['CurrentPrice']:,.2f}")
            with col2:
                target_mean = data['Target_Mean']
                st.metric("Target Price", f"₹{target_mean:,.2f}" if not pd.isna(target_mean) else 'N/A')
            with col3:
                upside = ((target_mean - data['CurrentPrice']) / data['CurrentPrice'] * 100) if not pd.isna(target_mean) and data['CurrentPrice'] > 0 else np.nan
                st.metric("Upside Potential", f"{upside:.1f}%" if not pd.isna(upside) else 'N/A')
            with col4:
                st.metric("52W Range", f"₹{data['52Week_Low']:,.0f} - ₹{data['52Week_High']:,.0f}" if not pd.isna(data['52Week_Low']) else 'N/A')
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Expert Recommendations")
                
                yf_rec = data['YF_Recommendation']
                analyst_rec = analyst_ratings['consensus']
                
                rec_data = {
                    'Source': ['Yahoo Finance', 'Analyst Consensus'],
                    'Recommendation': [
                        f"{get_recommendation_icon(yf_rec)} {yf_rec}",
                        f"{get_recommendation_icon(analyst_rec)} {analyst_rec}"
                    ],
                    'Details': [
                        'Aggregated market consensus',
                        f"Based on {analyst_ratings['total_analysts']} analysts"
                    ]
                }
                
                rec_df = pd.DataFrame(rec_data)
                st.dataframe(rec_df, use_container_width=True, hide_index=True)
                
                # Interpretation
                if yf_rec == analyst_rec:
                    st.success("✅ **Strong Agreement**: Both sources align on the recommendation")
                else:
                    st.warning("⚠️ **Mixed Signals**: Sources show different recommendations. Consider additional research.")
            
            with col2:
                st.markdown("### 💰 Key Metrics")
                
                metrics_data = {
                    'Metric': ['P/E Ratio', 'Market Cap', 'Dividend Yield', 'Target Low', 'Target High'],
                    'Value': [
                        f"{data['PE_Ratio']:.2f}" if not pd.isna(data['PE_Ratio']) else 'N/A',
                        f"₹{data['Market_Cap']/10000000:.2f} Cr" if not pd.isna(data['Market_Cap']) else 'N/A',
                        f"{data['Dividend_Yield']*100:.2f}%" if not pd.isna(data['Dividend_Yield']) else 'N/A',
                        f"₹{data['Target_Low']:,.2f}" if not pd.isna(data['Target_Low']) else 'N/A',
                        f"₹{data['Target_High']:,.2f}" if not pd.isna(data['Target_High']) else 'N/A',
                    ]
                }
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # TAB 2: Analyst Ratings
        with tab2:
            if analyst_ratings['total_analysts'] > 0:
                st.markdown("### 👥 Analyst Ratings Breakdown")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.plotly_chart(create_analyst_distribution_chart(analyst_ratings), use_container_width=True)
                
                with col2:
                    st.markdown("#### 📊 Rating Summary")
                    
                    consensus = analyst_ratings['consensus']
                    consensus_icon = get_recommendation_icon(consensus)
                    
                    st.metric("Consensus", f"{consensus_icon} {consensus}")
                    st.metric("Weighted Score", f"{analyst_ratings['weighted_score']:.2f} / 5.0")
                    st.metric("Total Analysts", analyst_ratings['total_analysts'])
                    
                    st.markdown("---")
                    st.markdown("#### 📋 Distribution")
                    st.write(f"🟢 **Strong Buy:** {analyst_ratings['strong_buy']}")
                    st.write(f"🟡 **Buy:** {analyst_ratings['buy']}")
                    st.write(f"🟠 **Hold:** {analyst_ratings['hold']}")
                    st.write(f"🟤 **Sell:** {analyst_ratings['sell']}")
                    st.write(f"🔴 **Strong Sell:** {analyst_ratings['strong_sell']}")
                
                st.markdown("---")
                
                # Target Price Analysis
                st.markdown("### 🎯 Price Target Analysis")
                
                current = data['CurrentPrice']
                low = data['Target_Low']
                mean = data['Target_Mean']
                high = data['Target_High']
                
                if not pd.isna(mean):
                    upside_low = ((low - current) / current * 100) if not pd.isna(low) else np.nan
                    upside_mean = ((mean - current) / current * 100)
                    upside_high = ((high - current) / current * 100) if not pd.isna(high) else np.nan
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"₹{current:,.2f}")
                    with col2:
                        st.metric("Target Low", f"₹{low:,.2f}" if not pd.isna(low) else 'N/A', 
                                 delta=f"{upside_low:.1f}%" if not pd.isna(upside_low) else None)
                    with col3:
                        st.metric("Target Mean", f"₹{mean:,.2f}", delta=f"{upside_mean:.1f}%")
                    with col4:
                        st.metric("Target High", f"₹{high:,.2f}" if not pd.isna(high) else 'N/A',
                                 delta=f"{upside_high:.1f}%" if not pd.isna(upside_high) else None)
                    
                    if upside_mean > 15:
                        st.success(f"📈 **Strong Upside Potential:** Analysts expect {upside_mean:.1f}% upside from current levels")
                    elif upside_mean > 5:
                        st.info(f"📊 **Moderate Upside:** Analysts expect {upside_mean:.1f}% upside from current levels")
                    elif upside_mean > -5:
                        st.warning(f"➡️ **Near Fair Value:** Current price is close to analyst targets (±5%)")
                    else:
                        st.error(f"📉 **Downside Risk:** Current price is {abs(upside_mean):.1f}% above analyst targets")
                        
            else:
                st.info("No analyst ratings data available for this stock.")
        
        # TAB 3: News
        with tab3:
            news_items = data['News']
            
            if news_items and len(news_items) > 0:
                st.markdown("### 📰 Latest News & Market Insights")
                st.caption(f"Recent news articles about {data['Company_Name']}")
                
                for i, news in enumerate(news_items, 1):
                    with st.expander(f"📌 {news['title']}", expanded=(i==1)):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Publisher:** {news['publisher']}")
                            st.markdown(f"**Published:** {news['published']}")
                        with col2:
                            st.markdown(f"[🔗 Read Full Article]({news['link']})")
                
                st.markdown("---")
                st.info("💡 **Tip:** Review news sentiment alongside quantitative recommendations for better decision-making.")
            else:
                st.info("No recent news available for this stock.")

# --- Main Execution ---

def main():
    """Main function to run the Streamlit application."""
    
    # Get user selected tickers from sidebar
    tickers = sidebar_inputs()
    
    if not tickers:
        st.info("👈 **Please select stocks from the sidebar to view recommendations**")
        st.markdown("---")
        st.markdown("""
        ### 📊 Welcome to Stock Recommendations Dashboard
        
        This dashboard provides **Buy, Sell, and Hold recommendations** from expert sources:
        
        #### 📈 Features:
        
        1. **Multi-Source Recommendations**
           - Yahoo Finance market consensus
           - Aggregated analyst ratings from multiple firms
           - Weighted scoring system (1-5 scale)
        
        2. **Price Target Analysis**
           - Analyst target prices (Low, Mean, High)
           - Upside/downside potential calculations
           - 52-week price ranges
        
        3. **Detailed Analyst Breakdown**
           - Distribution of Buy/Sell/Hold recommendations
           - Number of analysts covering each stock
           - Visual charts and metrics
        
        4. **Latest Market News**
           - Recent news articles and expert opinions
           - Publisher information and timestamps
           - Direct links to full articles
        
        5. **Key Financial Metrics**
           - P/E Ratio, Market Cap, Dividend Yield
           - Current price vs target comparisons
           - Sector classification
        
        ---
        
        **Get Started:** Select one or more stocks from the sidebar to view their recommendations!
        """)
        return
    
    # Fetch recommendations
    stock_data = fetch_stock_recommendations(tickers)
    
    if not stock_data:
        st.error("Unable to fetch data for the selected stocks. Please try again or select different stocks.")
        return

    # Display results
    results_page(stock_data)

if __name__ == "__main__":
    main()
