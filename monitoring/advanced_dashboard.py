"""
monitoring/advanced_dashboard.py - Advanced Visualization Dashboard

Provides interactive Streamlit visualizations for complex market data.

Features:
- 3D Volatility Surface (Strike x Expiry x IV) - Simulated or Real
- Real-time Correlation Heatmap
- Portfolio Risk Decomposition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import json

# Setup page config (must be first streamlit command if run directly)
# st.set_page_config(layout="wide", page_title="APEX Advanced Dashboard")

class AdvancedDashboard:
    """
    Advanced Dashboard Components.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def render_volatility_surface(self, symbol: str, options_data: Optional[pd.DataFrame] = None):
        """
        Render interactive 3D Volatility Surface.
        
        Args:
            symbol: Ticker symbol
            options_data: DataFrame with columns [strike, days_to_expiry, implied_volatility]
        """
        st.subheader(f"3D Volatility Surface: {symbol}")
        
        if options_data is None or options_data.empty:
            # Generate dummy surface for demonstration if no data
            st.info("Simulating volatility surface (No live options data connected)")
            strikes = np.linspace(80, 120, 20)
            expiries = np.linspace(7, 90, 10)
            X, Y = np.meshgrid(strikes, expiries)
            
            # Volatility smile + Term structure
            # Smirk: Higher vol at lower strikes (skew)
            # Term: Higher vol at longer expiries (contango) or backwardation
            skew = (100 - X) * 0.005
            term = np.log(Y) * 0.02
            Z = 0.20 + skew + term + np.random.normal(0, 0.005, X.shape)
            Z = np.maximum(0.10, Z)
            
        else:
            # Pivot real data
            # Requires pivoting to grid format
            # This is complex with sparse real data, usually requires interpolation (e.g. cubic spline)
            st.warning("Real options data rendering requires interpolation. Showing Simulation.")
            strikes = np.linspace(80, 120, 20)
            expiries = np.linspace(7, 90, 10)
            X, Y = np.meshgrid(strikes, expiries)
            Z = 0.20 + (100 - X) * 0.005 + np.log(Y) * 0.02

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
        
        fig.update_layout(
            title=f'{symbol} Implied Volatility Surface',
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Days to Expiry',
                zaxis_title='Implied Volatility'
            ),
            width=800,
            height=600,
            margin=dict(l=65, r=50, b=65, t=90)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_correlation_matrix(self, returns_data: pd.DataFrame):
        """
        Render dynamic correlation heatmap.
        """
        st.subheader("Real-Time Regime Correlation")
        
        if returns_data.empty:
            st.warning("No returns data available")
            return
            
        # Calculate rolling correlation (e.g. last 30 days)
        corr = returns_data.corr()
        
        # Mask upper triangle? No, heatmap looks better full
        
        fig = px.imshow(
            corr,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title="Asset Correlation Matrix (30D Rolling)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Analysis (Dendrogram logic simplified)
        # Identify highly correlated clusters
        s = corr.unstack()
        so = s.sort_values(kind="quicksort", ascending=False)
        
        high_corr = so[(so > 0.8) & (so < 1.0)]
        if not high_corr.empty:
            st.markdown("### ⚠️ Concentration Risks (Corr > 0.8)")
            seen = set()
            for idx, val in high_corr.items():
                pair = tuple(sorted(idx))
                if pair not in seen:
                    st.write(f"**{pair[0]} - {pair[1]}**: {val:.2f}")
                    seen.add(pair)

    def render_risk_decomposition(self, positions: Dict, equity: float):
        """
        Render Sunburst chart of risk allocation.
        """
        st.subheader("Risk Allocation Analysis")
        
        if not positions:
            st.info("No active positions")
            return
            
        # Prepare data for sunburst
        data = []
        
        for symbol, qty in positions.items():
            # In real system, fetch sector/industry/asset_class
            # Here we simulate categories
            asset_class = 'Equity'
            sector = 'Technology' if symbol in ['AAPL', 'MSFT', 'NVDA'] else 'ETF'
            if symbol in ['GLD', 'SLV']: sector = 'Commodity'
            
            value = abs(qty * 100) # Placeholder price
            exposure_pct = value / equity
            
            data.append({
                'Asset Class': asset_class,
                'Sector': sector,
                'Symbol': symbol,
                'Value': value,
                'Exposure': exposure_pct
            })
            
        df = pd.DataFrame(data)
        
        fig = px.sunburst(
            df,
            path=['Asset Class', 'Sector', 'Symbol'],
            values='Value',
            color='Exposure',
            color_continuous_scale='OrRd',
            title="Portfolio Risk Exposure Hierarchy"
        )
        
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    # Test execution
    st.set_page_config(layout="wide", page_title="APEX Advanced Dashboard")
    st.title("APEX Advanced Analytics Module")
    
    dashboard = AdvancedDashboard(".")
    
    # 1. Vol Surface
    dashboard.render_volatility_surface("SPX")
    
    # 2. Correlation
    dates = pd.date_range(end=datetime.now(), periods=100)
    dummy_returns = pd.DataFrame(np.random.normal(0, 0.01, (100, 5)), columns=['AAPL', 'MSFT', 'GOOG', 'GLD', 'TLT'], index=dates)
    # Induce correlation
    dummy_returns['MSFT'] = dummy_returns['AAPL'] * 0.8 + np.random.normal(0, 0.005, 100)
    
    dashboard.render_correlation_matrix(dummy_returns)
    
    # 3. Risk
    dashboard.render_risk_decomposition({'AAPL': 100, 'MSFT': 50, 'GLD': 200}, 50000)
