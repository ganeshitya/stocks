import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.stats import zscore
import json
import os

# --- Configuration ---
st.set_page_config(
    page_title="Advanced Hybrid Stock Screener V3.2",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- GLOBAL VARIABLES & DATA STORAGE ---
PICK_FILE = "strong_buy_picks.json"

# --- 1. Data Fetching (yfinance Only) ---

@st.cache_data(show_spinner="Fetching and validating stock data (This may take a minute)...")
def fetch_data(tickers):
    """
    Fetches price history (for technicals) and fundamental data (for fundamentals)
    using yfinance exclusively.
    """
    if not tickers:
        return {}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=18 * 30) # 1.5 years for 200-day SMA and 1Y change
    
    data = {}
    
    for full_ticker in tickers:
        try:
            stock = yf.Ticker(full_ticker)
            info = stock.info
            yf_hist = stock.history(start=start_date, end=end_date)
            
            # --- CRITICAL CHECK: Data Quality ---
            price_hist = yf_hist['Close'].dropna()
            
            if price_hist.shape[0] < 200 or pd.isna(price_hist.iloc[-1]):
                # st.warning(f"Skipping {full_ticker}: Insufficient historical data or missing price.")
                continue
            
            current_price = price_hist.iloc[-1]
            sector = info.get('sector', 'N/A')
            
            if sector == 'N/A':
                 # st.warning(f"Skipping {full_ticker}: Missing sector data.")
                 continue
            
            data[full_ticker] = {
                # Fundamental Metrics
                'PE_Ratio': info.get('trailingPE', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'DebtToEquity': info.get('debtToEquity', np.nan),
                'Growth': info.get('earningsQuarterlyGrowth', np.nan), 
                
                # Technical/Risk Metrics
                'Beta': info.get('beta', np.nan),
                
                # Metadata
                'CurrentPrice': current_price,
                'Sector': sector,
                'PriceHistory': price_hist,
            }

        except Exception:
            # st.error(f"Skipping {full_ticker}: Failed to fetch data.")
            continue
            
    return data

# ----------------------------------------------------------------------

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
    
    # 2. MACD (12, 26, 9) - Crossover Distance
    exp12 = price_hist.ewm(span=12, adjust=False).mean()
    exp26 = price_hist.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_signal = (macd.iloc[-1] - signal.iloc[-1]) # Positive is bullish
    
    # 3. SMA Signal (50-day vs 200-day ratio)
    sma50 = price_hist.rolling(window=50).mean().iloc[-1]
    sma200 = price_hist.rolling(window=200).mean().iloc[-1]
    sma_signal = sma50 / sma200 
    
    # 4. Volatility (Annualized log returns)
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
        
        processed_data[ticker] = d
        
    return pd.DataFrame.from_dict(processed_data, orient='index')

# ----------------------------------------------------------------------

# --- 3. Z-Score Normalization (Vs Sector Peers) ---

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
            return 0 # Neutral score for missing or non-calculable data
        
        z = (val - mu) / sigma
        
        # 'lower' is better -> Negative Z-score is good (flip the sign).
        return -z if direction == 'lower' else z

    df_merged[scored_metric_name] = df_merged.apply(calculate_z, axis=1)
    return df_merged[[scored_metric_name]]

# ----------------------------------------------------------------------

# --- 4. Scoring Logic (KPIS & Final Score) ---

def calculate_kpis_and_total_score(df):
    """Calculates the 3 main KPI scores (0-4, 0-4, 0-2) and the final 0-10 score."""
    if df.empty:
        return df
    
    scored_df = df.copy()
    
    # Helper to calculate and scale a composite score
    def calculate_and_scale_score(df, metrics, weight_max):
        z_score_cols = []
        for metric, direction in metrics.items():
            z_scores = z_score_normalize_by_sector(df, metric, direction)
            z_col = f'{metric}_{weight_max}_ZScore'
            # Rename Z-score column before merge to ensure uniqueness
            df = df.merge(z_scores.rename(columns={z_scores.columns[0]: z_col}), left_index=True, right_index=True)
            z_score_cols.append(z_col)
            
        mean_z_col = 'Mean_Z_' + str(weight_max)
        df[mean_z_col] = df[z_score_cols].mean(axis=1)

        min_val = df[mean_z_col].min()
        max_val = df[mean_z_col].max()
        
        score_col = 'Score_' + str(weight_max)
        if max_val != min_val:
            # Min-Max Scale to the desired weight_max (e.g., 4 or 2)
            df[score_col] = weight_max * (df[mean_z_col] - min_val) / (max_val - min_val)
        else:
            df[score_col] = weight_max / 2.0 # Neutral score
            
        return df, score_col

    # 4.1. --- FUNDAMENTAL STRENGTH (40% / 0-4 points) ---
    fund_metrics = {'PE_Ratio': 'lower', 'ROE': 'higher', 'DebtToEquity': 'lower', 'Growth': 'higher'}
    scored_df, fund_score_col = calculate_and_scale_score(scored_df, fund_metrics, 4)
    scored_df = scored_df.rename(columns={fund_score_col: 'Fundamental_Score'})

    # 4.2. --- TECHNICAL MOMENTUM (40% / 0-4 points) ---
    tech_metrics = {'RSI': 'higher', 'MACD_Signal': 'higher', 'SMA_Signal': 'higher', '1Y_Change': 'higher'}
    scored_df, tech_score_col = calculate_and_scale_score(scored_df, tech_metrics, 4)
    scored_df = scored_df.rename(columns={tech_score_col: 'Technical_Score'})


    # 4.3. --- SECTOR & RISK FACTORS (20% / 0-2 points) ---
    risk_metrics = {'Beta': 'lower', 'Volatility': 'lower'}
    scored_df, risk_score_col = calculate_and_scale_score(scored_df, risk_metrics, 2)
    scored_df = scored_df.rename(columns={risk_score_col: 'Sector_Score'})


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
    
    return scored_df.sort_values('Total_Score', ascending=False)

# ----------------------------------------------------------------------

# --- 5. Backtesting Module (Data Capture & Reporting) ---

def load_picks():
    """Loads historical strong buy picks from JSON file."""
    if os.path.exists(PICK_FILE):
        try:
            with open(PICK_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
             return {}
    return {}

def save_picks(picks):
    """Saves historical strong buy picks to JSON file."""
    with open(PICK_FILE, 'w') as f:
        json.dump(picks, f, indent=4)

def capture_strong_buy_picks(scored_df):
    """Captures 'STRONG BUY' picks with their entry price and date."""
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
    """Calculates the 30-day and 90-day returns for historical 'STRONG BUY' picks."""
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
            
            # --- 30-Day Return ---
            date_30d = pick_date + timedelta(days=30)
            price_30d = history.asof(date_30d) 
            return_30d = ((price_30d - entry_price) / entry_price) * 100 if not pd.isna(price_30d) else np.nan

            # --- 90-Day Return ---
            date_90d = pick_date + timedelta(days=90)
            return_90d = np.nan
            
            if datetime.now().date() >= date_90d:
                price_90d = history.asof(date_90d)
                return_90d = ((price_90d - entry_price) / entry_price) * 100 if not pd.isna(price_90d) else np.nan
            
            test_results.append({
                'Ticker': ticker,
                'Pick Date': date_str,
                '30D Return (%)': return_30d,
                '90D Return (%)': return_90d
            })
            
        except Exception:
            continue

    if not test_results:
        return None

    results_df = pd.DataFrame(test_results)
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
    max_scores = [4.0, 4.0, 2.0] 
    
    stock_scaled = [(stock_scores.iloc[i] / max_scores[i]) * 100 if max_scores[i] > 0 else 0 for i in range(3)]
    sector_scaled = [(sector_avg.iloc[i] / max_scores[i]) * 100 if max_scores[i] > 0 else 0 for i in range(3)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=stock_scaled, theta=categories, fill='toself', name=f'{ticker} Score'))
    fig.add_trace(go.Scatterpolar(r=sector_scaled, theta=categories, fill='toself', name=f'{sector} Average'))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title=f'{ticker} Performance vs. {sector} Peers (Scaled to 100%)')
    return fig


def sidebar_inputs():
    """Defines the sidebar layout and retrieves user inputs with the Nifty 500 list."""
    st.sidebar.title("📈 Hybrid Screener V3.2")
    st.sidebar.markdown("Advanced model using **40% Fundamental, 40% Technical, 20% Risk**.")
    
    # --- Tickers extracted from ind_nifty500list.csv (503 Tickers with .NS) ---
    all_available_tickers = ['360ONE.NS', '3MINDIA.NS', 'AADHARHFC.NS', 'AARTIIND.NS', 'AAVAS.NS', 'ABB.NS', 'ABBOTINDIA.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'ABLBL.NS', 'ABREL.NS', 'ABSLAMC.NS', 'ACC.NS', 'ACE.NS', 'ACMESOLAR.NS', 'ADANIENSOL.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANIPOWER.NS', 'AEGISLOG.NS', 'AEGISVOPAK.NS', 'AFCONS.NS', 'AFFLE.NS', 'AGARWALEYE.NS', 'AIAENG.NS', 'AIIL.NS', 'AJANTPHARM.NS', 'AKUMS.NS', 'AKZOINDIA.NS', 'ALKEM.NS', 'ALKYLAMINE.NS', 'ALOKINDS.NS', 'AMBER.NS', 'AMBUJACEM.NS', 'ANANDRATHI.NS', 'ANANTRAJ.NS', 'ANGELONE.NS', 'APARINDS.NS', 'APLAPOLLO.NS', 'APLLTD.NS', 'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'APTUS.NS', 'ARE&M.NS', 'ASAHIINDIA.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTERDM.NS', 'ASTRAL.NS', 'ASTRAZEN.NS', 'ATGL.NS', 'ATHERENERG.NS', 'ATUL.NS', 'AUBANK.NS', 'AUROPHARMA.NS', 'AWL.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJAJHFL.NS', 'BAJAJHLDNG.NS', 'BAJFINANCE.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BANKINDIA.NS', 'BASF.NS', 'BATAINDIA.NS', 'BAYERCROP.NS', 'BBTC.NS', 'BDL.NS', 'BEL.NS', 'BEML.NS', 'BERGEPAINT.NS', 'BHARATFORG.NS', 'BHARTIARTL.NS', 'BHARTIHEXA.NS', 'BHEL.NS', 'BIKAJI.NS', 'BIOCON.NS', 'BLS.NS', 'BLUEDART.NS', 'BLUEJET.NS', 'BLUESTARCO.NS', 'BOSCHLTD.NS', 'BPCL.NS', 'BRIGADE.NS', 'BRITANNIA.NS', 'BSE.NS', 'BSOFT.NS', 'CAMPUS.NS', 'CAMS.NS', 'CANBK.NS', 'CANFINHOME.NS', 'CAPLIPOINT.NS', 'CARBORUNIV.NS', 'CASTROLIND.NS', 'CCL.NS', 'CDSL.NS', 'CEATLTD.NS', 'CENTRALBK.NS', 'CENTURYPLY.NS', 'CERA.NS', 'CESC.NS', 'CGCL.NS', 'CGPOWER.NS', 'CHALET.NS', 'CHAMBLFERT.NS', 'CHENNPETRO.NS', 'CHOICEIN.NS', 'CHOLAFIN.NS', 'CHOLAHLDNG.NS', 'CIPLA.NS', 'CLEAN.NS', 'COALINDIA.NS', 'COCHINSHIP.NS', 'COFORGE.NS', 'COHANCE.NS', 'COLPAL.NS', 'CONCOR.NS', 'CONCORDBIO.NS', 'COROMANDEL.NS', 'CRAFTSMAN.NS', 'CREDITACC.NS', 'CRISIL.NS', 'CROMPTON.NS', 'CUB.NS', 'CUMMINSIND.NS', 'CYIENT.NS', 'DABUR.NS', 'DALBHARAT.NS', 'DATAPATTNS.NS', 'DBREALTY.NS', 'DCMSHRIRAM.NS', 'DEEPAKFERT.NS', 'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DEVYANI.NS', 'DIVISLAB.NS', 'DIXON.NS', 'DLF.NS', 'DMART.NS', 'DOMS.NS', 'DRREDDY.NS', 'DUMMYDBRLT.NS', 'DUMMYSKFIN.NS', 'DUMMYTATAM.NS', 'ECLERX.NS', 'EICHERMOT.NS', 'EIDPARRY.NS', 'EIHOTEL.NS', 'ELECON.NS', 'ELGIEQUIP.NS', 'EMAMILTD.NS', 'EMCURE.NS', 'ENDURANCE.NS', 'ENGINERSIN.NS', 'ENRIN.NS', 'ERIS.NS', 'ESCORTS.NS', 'ETERNAL.NS', 'EXIDEIND.NS', 'FACT.NS', 'FEDERALBNK.NS', 'FINCABLES.NS', 'FINPIPE.NS', 'FIRSTCRY.NS', 'FIVESTAR.NS', 'FLUOROCHEM.NS', 'FORCEMOT.NS', 'FORTIS.NS', 'FSL.NS', 'GAIL.NS', 'GESHIP.NS', 'GICRE.NS', 'GILLETTE.NS', 'GLAND.NS', 'GLAXO.NS', 'GLENMARK.NS', 'GMDCLTD.NS', 'GMRAIRPORT.NS', 'GODFRYPHLP.NS', 'GODIGIT.NS', 'GODREJAGRO.NS', 'GODREJCP.NS', 'GODREJIND.NS', 'GODREJPROP.NS', 'GPIL.NS', 'GRANULES.NS', 'GRAPHITE.NS', 'GRASIM.NS', 'GRAVITA.NS', 'GRSE.NS', 'GSPL.NS', 'GUJGASLTD.NS', 'GVT&D.NS', 'HAL.NS', 'HAPPSTMNDS.NS', 'HAVELLS.NS', 'HBLENGINE.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEG.NS', 'HEROMOTOCO.NS', 'HEXT.NS', 'HFCL.NS', 'HINDALCO.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS', 'HINDZINC.NS', 'HOMEFIRST.NS', 'HONASA.NS', 'HONAUT.NS', 'HSCL.NS', 'HUDCO.NS', 'HYUNDAI.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'IDBI.NS', 'IDEA.NS', 'IDFCFIRSTB.NS', 'IEX.NS', 'IFCI.NS', 'IGIL.NS', 'IGL.NS', 'IIFL.NS', 'IKS.NS', 'INDGN.NS', 'INDHOTEL.NS', 'INDIACEM.NS', 'INDIAMART.NS', 'INDIANB.NS', 'INDIGO.NS', 'INDUSINDBK.NS', 'INDUSTOWER.NS', 'INFY.NS', 'INOXINDIA.NS', 'INOXWIND.NS', 'INTELLECT.NS', 'IOB.NS', 'IOC.NS', 'IPCALAB.NS', 'IRB.NS', 'IRCON.NS', 'IRCTC.NS', 'IREDA.NS', 'IRFC.NS', 'ITC.NS', 'ITCHOTELS.NS', 'ITI.NS', 'J&KBANK.NS', 'JBCHEPHARM.NS', 'JBMA.NS', 'JINDALSAW.NS', 'JINDALSTEL.NS', 'JIOFIN.NS', 'JKCEMENT.NS', 'JKTYRE.NS', 'JMFINANCIL.NS', 'JPPOWER.NS', 'JSL.NS', 'JSWENERGY.NS', 'JSWINFRA.NS', 'JSWSTEEL.NS', 'JUBLFOOD.NS', 'JUBLINGREA.NS', 'JUBLPHARMA.NS', 'JWL.NS', 'JYOTHYLAB.NS', 'JYOTICNC.NS', 'KAJARIACER.NS', 'KALYANKJIL.NS', 'KARURVYSYA.NS', 'KAYNES.NS', 'KEC.NS', 'KEI.NS', 'KFINTECH.NS', 'KIMS.NS', 'KIRLOSBROS.NS', 'KIRLOSENG.NS', 'KOTAKBANK.NS', 'KPIL.NS', 'KPITTECH.NS', 'KPRMILL.NS', 'KSB.NS', 'LALPATHLAB.NS', 'LATENTVIEW.NS', 'LAURUSLABS.NS', 'LEMONTREE.NS', 'LICHSGFIN.NS', 'LICI.NS', 'LINDEINDIA.NS', 'LLOYDSME.NS', 'LODHA.NS', 'LT.NS', 'LTF.NS', 'LTFOODS.NS', 'LTIM.NS', 'LTTS.NS', 'LUPIN.NS', 'M&M.NS', 'M&MFIN.NS', 'MAHABANK.NS', 'MAHSCOOTER.NS', 'MAHSEAMLES.NS', 'MANAPPURAM.NS', 'MANKIND.NS', 'MANYAVAR.NS', 'MAPMYINDIA.NS', 'MARICO.NS', 'MARUTI.NS', 'MAXHEALTH.NS', 'MAZDOCK.NS', 'MCX.NS', 'MEDANTA.NS', 'METROPOLIS.NS', 'MFSL.NS', 'MGL.NS', 'MINDACORP.NS', 'MMTC.NS', 'MOTHERSON.NS', 'MOTILALOFS.NS', 'MPHASIS.NS', 'MRF.NS', 'MRPL.NS', 'MSUMI.NS', 'MUTHOOTFIN.NS', 'NAM-INDIA.NS', 'NATCOPHARM.NS', 'NATIONALUM.NS', 'NAUKRI.NS', 'NAVA.NS', 'NAVINFLUOR.NS', 'NBCC.NS', 'NCC.NS', 'NESTLEIND.NS', 'NETWEB.NS', 'NEULANDLAB.NS', 'NEWGEN.NS', 'NH.NS', 'NHPC.NS', 'NIACL.NS', 'NIVABUPA.NS', 'NLCINDIA.NS', 'NMDC.NS', 'NSLNISP.NS', 'NTPC.NS', 'NTPCGREEN.NS', 'NUVAMA.NS', 'NUVOCO.NS', 'NYKAA.NS', 'OBEROIRLTY.NS', 'OFSS.NS', 'OIL.NS', 'OLAELEC.NS', 'OLECTRA.NS', 'ONESOURCE.NS', 'ONGC.NS', 'PAGEIND.NS', 'PATANJALI.NS', 'PAYTM.NS', 'PCBL.NS', 'PERSISTENT.NS', 'PETRONET.NS', 'PFC.NS', 'PFIZER.NS', 'PGEL.NS', 'PGHH.NS', 'PHOENIXLTD.NS', 'PIDILITIND.NS', 'PIIND.NS', 'PNB.NS', 'PNBHOUSING.NS', 'POLICYBZR.NS', 'POLYCAB.NS', 'POLYMED.NS', 'POONAWALLA.NS', 'POWERGRID.NS', 'POWERINDIA.NS', 'PPLPHARMA.NS', 'PRAJIND.NS', 'PREMIERENE.NS', 'PRESTIGE.NS', 'PTCIL.NS', 'PVRINOX.NS', 'RADICO.NS', 'RAILTEL.NS', 'RAINBOW.NS', 'RAMCOCEM.NS', 'RBLBANK.NS', 'RCF.NS', 'RECLTD.NS', 'REDINGTON.NS', 'RELIANCE.NS', 'RELINFRA.NS', 'RHIM.NS', 'RITES.NS', 'RKFORGE.NS', 'RPOWER.NS', 'RRKABEL.NS', 'RVNL.NS', 'SAGILITY.NS', 'SAIL.NS', 'SAILIFE.NS', 'SAMMAANCAP.NS', 'SAPPHIRE.NS', 'SARDAEN.NS', 'SAREGAMA.NS', 'SBFC.NS', 'SBICARD.NS', 'SBILIFE.NS', 'SBIN.NS', 'SCHAEFFLER.NS', 'SCHNEIDER.NS', 'SCI.NS', 'SHREECEM.NS', 'SHRIRAMFIN.NS', 'SHYAMMETL.NS', 'SIEMENS.NS', 'SIGNATURE.NS', 'SJVN.NS', 'SKFINDIA.NS', 'SOBHA.NS', 'SOLARINDS.NS', 'SONACOMS.NS', 'SONATSOFTW.NS', 'SRF.NS', 'STARHEALTH.NS', 'SUMICHEM.NS', 'SUNDARMFIN.NS', 'SUNDRMFAST.NS', 'SUNPHARMA.NS', 'SUNTV.NS', 'SUPREMEIND.NS', 'SUZLON.NS', 'SWANCORP.NS', 'SWIGGY.NS', 'SYNGENE.NS', 'SYRMA.NS', 'TARIL.NS', 'TATACHEM.NS', 'TATACOMM.NS', 'TATACONSUM.NS', 'TATAELXSI.NS', 'TATAINVEST.NS', 'TATAMOTORS.NS', 'TATAPOWER.NS', 'TATASTEEL.NS', 'TATATECH.NS', 'TBOTEK.NS', 'TCS.NS', 'TECHM.NS', 'TECHNOE.NS', 'TEJASNET.NS', 'THELEELA.NS', 'THERMAX.NS', 'TIINDIA.NS', 'TIMKEN.NS', 'TITAGARH.NS', 'TITAN.NS', 'TORNTPHARM.NS', 'TORNTPOWER.NS', 'TRENT.NS', 'TRIDENT.NS', 'TRITURBINE.NS', 'TRIVENI.NS', 'TTML.NS', 'TVSMOTOR.NS', 'UBL.NS', 'UCOBANK.NS', 'ULTRACEMCO.NS', 'UNIONBANK.NS', 'UNITDSPR.NS', 'UNOMINDA.NS', 'UPL.NS', 'USHAMART.NS', 'UTIAMC.NS', 'VBL.NS', 'VEDL.NS', 'VENTIVE.NS', 'VGUARD.NS', 'VIJAYA.NS', 'VMM.NS', 'VOLTAS.NS', 'VTL.NS', 'WAAREEENER.NS', 'WELCORP.NS', 'WELSPUNLIV.NS', 'WHIRLPOOL.NS', 'WIPRO.NS', 'WOCKPHARMA.NS', 'YESBANK.NS', 'ZEEL.NS', 'ZENSARTECH.NS', 'ZENTEC.NS', 'ZFCVINDIA.NS', 'ZYDUSLIFE.NS']
    
    st.sidebar.header("🎯 Screener Parameters")

    # Set default selections (Using a few popular names from the list)
    default_selection = [t for t in ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"] if t in all_available_tickers]
    
    # --- Multi-Select Dropdown Input ---
    tickers = st.sidebar.multiselect(
        "Select NSE Tickers (up to 10 recommended for speed):",
        options=all_available_tickers,
        default=default_selection
    )

    st.sidebar.header("⚖️ Model Weighting (Fixed)")
    st.sidebar.markdown("""
        - **Fundamental Strength:** 40% (0-4 pts)
        - **Technical Momentum:** 40% (0-4 pts)
        - **Sector & Risk Factors:** 20% (0-2 pts)
        ---
        *Metrics are Z-score normalized vs. sector peers.*
    """)
            
    if not tickers:
        st.error("Please select at least one NSE ticker to run the analysis.")
        st.stop()
        
    return tickers, {} 


def results_page(scored_df):
    """Displays the main results and visualizations."""
    
    capture_strong_buy_picks(scored_df)
    
    st.title("🏆 Advanced Hybrid Stock Screener Results (V3.2)")
    st.markdown(f"**Screening Date:** {datetime.now().strftime('%Y-%m-%d')}")

    if scored_df.empty:
        st.error("No stocks passed the data validation checks.")
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
    
    for col in ['Total Score (0-10)', 'Fund. Score (0-4)', 'Tech. Score (0-4)', 'Risk Score (0-2)', 'P/E', 'D/E', 'RSI', 'Beta']:
        formatted_df[col] = formatted_df[col].round(2)
        
    formatted_df['Price (INR)'] = formatted_df['Price (INR)'].apply(lambda x: f"₹{x:,.2f}")
    formatted_df['ROE'] = (scored_df['ROE'] * 100).round(2).astype(str) + '%'

    st.dataframe(formatted_df, use_container_width=True)
    
    # --- 2. Backtesting Summary ---
    st.header("2. Backtesting Performance (STRONG BUY Picks)")
    backtest_summary = run_backtest_summary()
    
    if backtest_summary is not None:
        st.markdown("Metrics show the **Mean** and **Median** returns of historical picks after 30 and 90 days:")
        st.dataframe(backtest_summary, use_container_width=True)
    else:
        st.info("No historical 'STRONG BUY' picks are currently old enough (>= 30 days) to run backtesting.")
        
    # --- 3. Detailed Stock Analysis & Radar Chart ---
    st.header("3. Detailed Stock Analysis & Sector Comparison")
    selected_ticker = st.selectbox("Select a Ticker for Detailed View", scored_df.index)
    
    if selected_ticker:
        st.subheader(f"Analysis for {selected_ticker} ({scored_df.loc[selected_ticker, 'Sector']})")
        
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📊 Scoring Breakdown")
            
            detail_data = {
                "Total Score (0-10)": scored_df.loc[selected_ticker, 'Total_Score'].round(2),
                "Recommendation": scored_df.loc[selected_ticker, 'Balanced_Recommendation'],
                "--- SCORE BREAKDOWN ---": "",
                "Fundamental Strength (0-4)": scored_df.loc[selected_ticker, 'Fundamental_Score'].round(2),
                "Technical Momentum (0-4)": scored_df.loc[selected_ticker, 'Technical_Score'].round(2),
                "Sector & Risk (0-2)": scored_df.loc[selected_ticker, 'Sector_Score'].round(2),
            }
            detail_df = pd.DataFrame(detail_data.items(), columns=['Metric', 'Value'])
            detail_df = detail_df.set_index('Metric')
            st.table(detail_df)
            
        with col2:
            st.markdown("### 📈 Raw Metrics")
            raw_metrics = {
                "P/E Ratio": scored_df.loc[selected_ticker, 'PE_Ratio'].round(2),
                "ROE": f"{(scored_df.loc[selected_ticker, 'ROE'] * 100):.2f}%",
                "D/E Ratio": scored_df.loc[selected_ticker, 'DebtToEquity'].round(2),
                "1Y Price Change": f"{(scored_df.loc[selected_ticker, '1Y_Change'] * 100):.2f}%",
                "RSI (14)": scored_df.loc[selected_ticker, 'RSI'].round(2),
                "Beta": scored_df.loc[selected_ticker, 'Beta'].round(2),
            }
            raw_df = pd.DataFrame(raw_metrics.items(), columns=['Metric', 'Value'])
            raw_df = raw_df.set_index('Metric')
            st.table(raw_df)
            
        st.markdown("---")
        st.markdown("### 🎯 Stock vs. Sector Average Comparison (Radar Chart)")
        st.plotly_chart(create_radar_chart(scored_df, selected_ticker), use_container_width=True)


# ----------------------------------------------------------------------
# --- 7. Main Execution ---

def main():
    """The main function to run the Streamlit application."""
    
    # 1. Get user inputs
    tickers, _ = sidebar_inputs()
    
    # 2. Fetch Data
    raw_data = fetch_data(tickers)
    
    if not raw_data:
        # If the multi-select had tickers but none returned valid data.
        st.error("The selected tickers failed data validation (e.g., missing price/sector data). Please try different tickers.")
        return

    # 3. Compute Metrics
    metrics_df = compute_all_metrics(raw_data)
    
    # 4. Apply Scoring
    scored_df = calculate_kpis_and_total_score(metrics_df)

    # 5. Display results
    results_page(scored_df)

if __name__ == "__main__":
    main()
