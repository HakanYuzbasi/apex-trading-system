"""
monitoring/dashboard.py - Streamlit Dashboard
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="APEX Trading", layout="wide")

st.title("🚀 APEX Trading System Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Portfolio Value", "$100,000", "+2.5%")

with col2:
    st.metric("Daily P&L", "+$2,500", "2.5%")

with col3:
    st.metric("Positions", "5", "")

st.subheader("Equity Curve")
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1,2,3], y=[100000, 101000, 102500], mode='lines'))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Current Positions")
st.dataframe({
    'Symbol': ['AAPL', 'GOOGL', 'MSFT'],
    'Shares': [50, 30, 40],
    'Value': ['$8,500', '$4,200', '$16,000']
})
