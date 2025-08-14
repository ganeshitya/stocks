import streamlit as st
import requests

# URL where your MCP server is running
API_URL = "http://localhost:8000"  # adjust port if needed

def call_mcp(method: str, params: dict = None):
    payload = {"method": method}
    if params:
        payload["params"] = params
    resp = requests.post(f"{API_URL}/mcp", json=payload)
    return resp.json()

st.title("Indian Stock MCP Dashboard")

# 1. API Status
st.header("🔌 API Status")
status = call_mcp("get_alpha_vantage_status")
st.json(status)

# 2. Portfolio Summary
st.header("💼 Portfolio Summary")
summary = call_mcp("get_portfolio_summary", {"limit": 10, "summary": True})
st.json(summary)

# 3. Portfolio Analysis
st.header("📈 Portfolio Analysis (Segment 1)")
analysis = call_mcp("portfolio_analysis", {"segment": 1, "segment_size": 5, "include_details": False})
st.json(analysis)

# Sidebar: stock recommendations
st.sidebar.header("Recommendations & Trends")
if st.sidebar.button("Get Stock Recommendations"):
    recs = call_mcp("get_stock_recommendations", {"criteria": "growth", "limit": 5})
    st.sidebar.json(recs)

if st.sidebar.button("Get Market Trends"):
    trends = call_mcp("get_market_trend_recommendations", {"limit": 5})
    st.sidebar.json(trends)

# Technical Analysis Example
st.header("📊 Technical Analysis")
symbol = st.text_input("Enter NSE Symbol (e.g. NSE:RELIANCE)", "NSE:RELIANCE")
if st.button("Get Technical Indicators"):
    tech = call_mcp("get_technical_analysis", {"symbol": symbol})
    st.json(tech)
