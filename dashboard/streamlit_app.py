"""
dashboard/streamlit_app.py - APEX Trading System Live Dashboard
Professional dark theme with consistent styling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ApexConfig
from data.market_data import MarketDataFetcher

# Page config
st.set_page_config(
    page_title="APEX Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with consistent dark theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-purple: #667eea;
        --secondary-purple: #764ba2;
        --dark-bg: #0e1117;
        --card-bg: #1e2130;
        --text-primary: #fafafa;
        --text-secondary: #b0b0b0;
        --success-green: #00d4aa;
        --danger-red: #ff6b6b;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Enhanced metric cards */
    .stMetric {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    .stMetric label {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--card-bg);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 1rem;
        padding: 0 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.1);
        color: var(--primary-purple);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-purple), var(--secondary-purple));
        color: white !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2130 0%, #0e1117 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: rgba(0, 212, 170, 0.1);
        border-left: 4px solid var(--success-green);
        border-radius: 4px;
    }
    
    .stError {
        background-color: rgba(255, 107, 107, 0.1);
        border-left: 4px solid var(--danger-red);
        border-radius: 4px;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        border-radius: 4px;
    }
    
    .stInfo {
        background-color: rgba(102, 126, 234, 0.1);
        border-left: 4px solid var(--primary-purple);
        border-radius: 4px;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    .status-online {
        background: linear-gradient(135deg, #00d4aa, #00b894);
        color: white;
        box-shadow: 0 2px 8px rgba(0, 212, 170, 0.3);
    }
    
    .status-offline {
        background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
        color: white;
        box-shadow: 0 2px 8px rgba(255, 107, 107, 0.3);
    }
    
    /* Section divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary-purple), transparent);
        margin: 2rem 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)


class ApexDashboard:
    """Real-time dashboard for APEX Trading System."""
    
    # Professional color scheme
    COLORS = {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'accent': '#f093fb',
        'success': '#00d4aa',
        'danger': '#ff6b6b',
        'warning': '#ffc107',
        'info': '#4dabf7',
        'bg_dark': '#0e1117',
        'bg_card': '#1e2130',
        'text': '#fafafa',
        'text_muted': '#b0b0b0'
    }
    
    def __init__(self):
        self.data_file = Path("data/trading_state.json")
        self.trades_file = Path("data/trades.csv")
        self.equity_file = Path("data/equity_curve.csv")
        self.market_data = MarketDataFetcher()
        
        # Create data directory
        self.data_file.parent.mkdir(exist_ok=True)
    
    def load_state(self) -> dict:
        """Load current trading state."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    if data.get('capital', 0) > 0:
                        return data
            except Exception as e:
                st.error(f"❌ Error loading state: {e}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'capital': 0.0,
            'starting_capital': 0.0,
            'positions': {},
            'daily_pnl': 0.0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'open_positions': 0
        }
    
    def load_trades(self) -> pd.DataFrame:
        """Load trade history."""
        if self.trades_file.exists():
            try:
                return pd.read_csv(self.trades_file)
            except Exception as e:
                st.error(f"❌ Error loading trades: {e}")
        
        return pd.DataFrame(columns=[
            'timestamp', 'symbol', 'side', 'quantity', 'price', 'pnl'
        ])
    
    def load_equity_curve(self) -> pd.DataFrame:
        """Load equity curve."""
        if self.equity_file.exists():
            try:
                df = pd.read_csv(self.equity_file, parse_dates=['timestamp'])
                return df
            except Exception as e:
                st.error(f"❌ Error loading equity: {e}")
        
        return pd.DataFrame(columns=['timestamp', 'equity', 'drawdown'])
    
    def get_chart_layout(self, title: str = "", height: int = 400):
        """Get consistent chart layout."""
        return {
            'template': None,
            'title': {
                'text': title,
                'font': {'size': 18, 'color': self.COLORS['text'], 'family': 'system-ui'},
                'x': 0.5,
                'xanchor': 'center'
            },
            'height': height,
            'paper_bgcolor': self.COLORS['bg_card'],
            'plot_bgcolor': self.COLORS['bg_dark'],
            'font': {'color': self.COLORS['text'], 'family': 'system-ui'},
            'hovermode': 'x unified',
            'showlegend': True,
            'legend': {
                'bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': self.COLORS['text_muted']}
            },
            'margin': {'l': 60, 'r': 40, 't': 60, 'b': 50},
            'xaxis': {
                'gridcolor': 'rgba(102, 126, 234, 0.1)',
                'showgrid': True,
                'zeroline': False,
                'color': self.COLORS['text_muted']
            },
            'yaxis': {
                'gridcolor': 'rgba(102, 126, 234, 0.1)',
                'showgrid': True,
                'zeroline': False,
                'color': self.COLORS['text_muted']
            }
        }
    
    def create_equity_chart(self, equity_df):
        """Create equity and drawdown chart with professional styling."""
        if equity_df.empty:
            fig = go.Figure()
            fig.update_layout(**self.get_chart_layout("Waiting for trading data...", 400))
            fig.add_annotation(
                text="No equity data yet<br>Start trading to see your portfolio growth",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font={'size': 14, 'color': self.COLORS['text_muted']}
            )
            return fig
        
        try:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    '<b>Portfolio Value</b>',
                    '<b>Drawdown</b>'
                )
            )
            
            # Equity curve with gradient fill
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=equity_df['equity'],
                    mode='lines',
                    name='Portfolio Value',
                    line={'color': self.COLORS['primary'], 'width': 3},
                    fill='tozeroy',
                    fillcolor=f'rgba(102, 126, 234, 0.15)',
                    hovertemplate='<b>%{x}</b><br>Value: $%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Drawdown
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=equity_df['drawdown'] * 100,
                    mode='lines',
                    name='Drawdown',
                    line={'color': self.COLORS['danger'], 'width': 2},
                    fill='tozeroy',
                    fillcolor=f'rgba(255, 107, 107, 0.15)',
                    hovertemplate='<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Update layout
            layout = self.get_chart_layout("", 600)
            layout['showlegend'] = True
            
            fig.update_layout(**layout)
            
            # Update axes
            fig.update_xaxes(title_text="<b>Time</b>", row=2, col=1, 
                           gridcolor='rgba(102, 126, 234, 0.1)')
            fig.update_yaxes(title_text="<b>Value ($)</b>", row=1, col=1,
                           gridcolor='rgba(102, 126, 234, 0.1)')
            fig.update_yaxes(title_text="<b>Drawdown (%)</b>", row=2, col=1,
                           gridcolor='rgba(102, 126, 234, 0.1)')
            
            # Update subplot titles
            for annotation in fig['layout']['annotations']:
                annotation['font'] = {'size': 14, 'color': self.COLORS['text']}
            
            return fig
        
        except Exception as e:
            st.error(f"❌ Chart error: {e}")
            fig = go.Figure()
            fig.update_layout(**self.get_chart_layout("Chart Error", 400))
            return fig
    
    def create_positions_chart(self, positions: dict):
        """Create positions pie chart with gradient colors."""
        if not positions:
            return None
        
        symbols = list(positions.keys())
        values = [pos['qty'] * pos['current_price'] for pos in positions.values()]
        
        # Generate gradient colors
        n = len(symbols)
        colors = []
        for i in range(n):
            # Interpolate between primary and secondary purple
            t = i / max(n - 1, 1)
            r = int(102 + (118 - 102) * t)
            g = int(126 + (75 - 126) * t)
            b = int(234 + (162 - 234) * t)
            colors.append(f'rgb({r},{g},{b})')
        
        fig = go.Figure(data=[go.Pie(
            labels=symbols,
            values=values,
            hole=0.5,
            marker={
                'colors': colors,
                'line': {'color': self.COLORS['bg_dark'], 'width': 2}
            },
            textfont={'size': 14, 'color': 'white'},
            hovertemplate='<b>%{label}</b><br>Value: $%{value:,.0f}<br>%{percent}<extra></extra>'
        )])
        
        # Add center text
        total_value = sum(values)
        fig.add_annotation(
            text=f'<b>${total_value:,.0f}</b><br><span style="font-size:12px; color:{self.COLORS["text_muted"]}">Total Value</span>',
            x=0.5, y=0.5,
            font={'size': 20, 'color': self.COLORS['text']},
            showarrow=False
        )
        
        layout = self.get_chart_layout("<b>Position Allocation</b>", 450)
        layout['showlegend'] = True
        layout['legend'] = {
            'orientation': 'v',
            'x': 1.05,
            'y': 0.5,
            'bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': self.COLORS['text']}
        }
        
        fig.update_layout(**layout)
        
        return fig
    
    def create_pnl_chart(self, trades_df: pd.DataFrame):
        """Create P&L chart with professional styling."""
        if trades_df.empty:
            return None
        
        trades_df = trades_df.copy()
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        
        fig = go.Figure()
        
        # P&L bars with conditional colors (SIMPLIFIED)
        colors = [self.COLORS['success'] if x >= 0 else self.COLORS['danger'] 
                for x in trades_df['pnl']]
        
        fig.add_trace(go.Bar(
            x=trades_df['timestamp'],
            y=trades_df['pnl'],
            name='Trade P&L',
            marker_color=colors,  # ← SIMPLIFIED: Just use marker_color
            hovertemplate='<b>%{x}</b><br>P&L: $%{y:+,.2f}<extra></extra>',
            opacity=0.8
        ))
        
        # Cumulative P&L line
        fig.add_trace(go.Scatter(
            x=trades_df['timestamp'],
            y=trades_df['cumulative_pnl'],
            name='Cumulative P&L',
            line={'color': self.COLORS['accent'], 'width': 3},
            yaxis='y2',
            hovertemplate='<b>%{x}</b><br>Cumulative: $%{y:+,.2f}<extra></extra>'
        ))
        
        layout = self.get_chart_layout("<b>Trade P&L History</b>", 450)
        layout['yaxis'] = {
            'title': '<b>Trade P&L ($)</b>',
            'gridcolor': 'rgba(102, 126, 234, 0.1)',
            'color': self.COLORS['text_muted'],
            'zeroline': True,
            'zerolinecolor': 'rgba(255, 255, 255, 0.2)',
            'zerolinewidth': 1
        }
        layout['yaxis2'] = {
            'title': '<b>Cumulative P&L ($)</b>',
            'overlaying': 'y',
            'side': 'right',
            'gridcolor': 'rgba(118, 75, 162, 0.1)',
            'color': self.COLORS['text_muted']
        }
        layout['xaxis']['title'] = '<b>Time</b>'
        
        fig.update_layout(**layout)
        
        return fig
    
    def render_header(self):
        """Render professional header."""
        st.markdown('<h1 class="main-header">🚀 APEX TRADING SYSTEM</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f'<p class="subtitle">Live Dashboard • Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} CET</p>',
                unsafe_allow_html=True
            )
    
    def render_sidebar(self, state: dict):
        """Render enhanced sidebar."""
        with st.sidebar:
            # Logo/Title area
            st.markdown("""
                <div style="text-align: center; padding: 1rem 0 2rem 0;">
                    <div style="font-size: 3rem;">📊</div>
                    <h2 style="margin: 0; background: linear-gradient(135deg, #667eea, #764ba2); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        APEX
                    </h2>
                    <p style="color: #b0b0b0; font-size: 0.9rem; margin: 0;">Algorithmic Trading</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # System status
            st.markdown("### 🔔 System Status")
            has_data = state['capital'] > 0
            
            if has_data:
                status_html = '<span class="status-badge status-online">🟢 ONLINE</span>'
            else:
                status_html = '<span class="status-badge status-offline">🔴 OFFLINE</span>'
            
            st.markdown(status_html, unsafe_allow_html=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Controls
            st.markdown("### ⚙️ Controls")
            refresh_interval = st.slider("Auto-refresh (seconds)", 5, 60, 10, key="refresh")
            show_trades = st.checkbox("Show Trade Log", value=True, key="show_trades")
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # System Info
            st.markdown("### 📊 Configuration")
            
            info_items = [
                ("Universe", f"{len(ApexConfig.SYMBOLS)} symbols"),
                ("Max Positions", str(ApexConfig.MAX_POSITIONS)),
                ("Position Size", f"${ApexConfig.POSITION_SIZE_USD:,}"),
                ("Strategy", "ML Ensemble")
            ]
            
            for label, value in info_items:
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; 
                                border-bottom: 1px solid rgba(102, 126, 234, 0.1);">
                        <span style="color: #b0b0b0;">{label}</span>
                        <span style="color: #fafafa; font-weight: 600;">{value}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Alerts
            st.markdown("### 🔔 Alerts")
            
            if has_data:
                if state['max_drawdown'] < -0.10:
                    st.error("⚠️ High Drawdown Alert!")
                if state['daily_pnl'] < -5000:
                    st.warning("📉 Significant Daily Loss")
                if state['daily_pnl'] < 0:
                    st.info("📊 Negative Daily P&L")
                
                if state['max_drawdown'] >= -0.05 and state['daily_pnl'] >= 0:
                    st.success("✅ All Systems Optimal")
            else:
                st.info("📡 Waiting for trading data...")
            
            return refresh_interval, show_trades
    
    def create_signal_strength_chart(self, state: dict):
        """Create signal strength chart for current positions."""
        if not state.get('positions'):
            return None
        
        symbols = []
        signals = []
        colors = []
        
        for symbol, pos in state['positions'].items():
            signal = pos.get('current_signal', 0)
            symbols.append(symbol)
            signals.append(signal)
            
            # Color based on signal
            if signal > 0.40:
                colors.append(self.COLORS['success'])
            elif signal > 0:
                colors.append('#90EE90')  # Light green
            elif signal > -0.40:
                colors.append('#FFA500')  # Orange
            else:
                colors.append(self.COLORS['danger'])
        
        fig = go.Figure(go.Bar(
            x=symbols,
            y=signals,
            marker_color=colors,
            text=[f"{s:+.3f}" for s in signals],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Signal: %{y:+.3f}<extra></extra>'
        ))
        
        layout = self.get_chart_layout("<b>Current Signal Strength</b>", 350)
        layout['yaxis'] = {
            'title': '<b>Signal Strength</b>',
            'gridcolor': 'rgba(102, 126, 234, 0.1)',
            'color': self.COLORS['text_muted'],
            'zeroline': True,
            'zerolinecolor': 'rgba(255, 255, 255, 0.5)',
            'zerolinewidth': 2,
            'range': [-1, 1]
        }
        layout['xaxis']['title'] = '<b>Symbol</b>'
        
        # Add threshold lines
        fig.add_hline(y=0.40, line_dash="dash", line_color="green", 
                    annotation_text="BUY Threshold", annotation_position="right")
        fig.add_hline(y=-0.40, line_dash="dash", line_color="red",
                    annotation_text="SELL Threshold", annotation_position="right")
        fig.add_hline(y=0.20, line_dash="dot", line_color="gray",
                    annotation_text="Weak BUY", annotation_position="right")
        fig.add_hline(y=-0.20, line_dash="dot", line_color="gray",
                    annotation_text="Weak SELL", annotation_position="right")
        
        fig.update_layout(**layout)
        
        return fig

    def render(self):
        """Render the complete dashboard."""
        # Header
        self.render_header()
        
        # Load data
        state = self.load_state()
        trades_df = self.load_trades()
        equity_df = self.load_equity_curve()
        
        # Check if data exists
        has_data = state['capital'] > 0
        
        # Sidebar
        refresh_interval, show_trades = self.render_sidebar(state)
        
        # Main content
        if not has_data:
            st.warning("⚠️ **No trading data available yet!**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info("""
                    **🚀 Getting Started:**
                    
                    1. Make sure `main.py` is running in another terminal
                    2. Wait for the system to generate signals and place trades
                    3. This dashboard will auto-refresh every 10 seconds
                    
                    The system exports data to:
                    - `data/trading_state.json` (portfolio state)
                    - `data/trades.csv` (trade history)
                    - `data/equity_curve.csv` (performance)
                """)
            
            with col2:
                st.code("""
# Start trading system:
cd ~/apex-trading
python main.py
                """, language="bash")
            
            time.sleep(10)
            st.rerun()
            return
        
        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            delta_color = "normal" if state['daily_pnl'] >= 0 else "inverse"
            st.metric(
                "💼 Portfolio Value",
                f"${state['capital']:,.2f}",
                f"{state['daily_pnl']:+,.2f}",
                delta_color=delta_color
            )
        
        with col2:
            if state['starting_capital'] > 0:
                total_return = (state['capital'] - state['starting_capital']) / state['starting_capital']
            else:
                total_return = 0
            
            st.metric(
                "📈 Total Return",
                f"{total_return*100:+.2f}%",
                f"${state['total_pnl']:+,.2f}"
            )
        
        with col3:
            sharpe_color = "normal" if state['sharpe_ratio'] >= 1.5 else "inverse"
            st.metric(
                "📊 Sharpe Ratio",
                f"{state['sharpe_ratio']:.2f}",
                "Target: >1.5"
            )
        
        with col4:
            st.metric(
                "🎯 Win Rate",
                f"{state['win_rate']*100:.1f}%",
                f"{state['total_trades']} trades"
            )
        
        with col5:
            st.metric(
                "📉 Max Drawdown",
                f"{state['max_drawdown']*100:.2f}%",
                f"{state['open_positions']} positions",
                delta_color="inverse"
            )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2, tab3 = st.tabs([
            "📈 Portfolio Overview",
            "💼 Current Positions",
            "📜 Trade History"
        ])
        
        with tab1:
            st.markdown("### Portfolio Performance")
            equity_chart = self.create_equity_chart(equity_df)
            st.plotly_chart(equity_chart, use_container_width=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # ✅ ADD SIGNAL STRENGTH CHART
            if state['positions']:
                st.markdown("### 📡 Current Signal Strength")
                signal_chart = self.create_signal_strength_chart(state)
                if signal_chart:
                    st.plotly_chart(signal_chart, use_container_width=True)
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 💰 Trade P&L")
                if len(trades_df) > 0:
                    pnl_chart = self.create_pnl_chart(trades_df)
                    if pnl_chart:
                        st.plotly_chart(pnl_chart, use_container_width=True)
                else:
                    st.info("📊 No trades executed yet")
            
            with col2:
                st.markdown("### 📊 Position Allocation")
                if state['positions']:
                    positions_chart = self.create_positions_chart(state['positions'])
                    if positions_chart:
                        st.plotly_chart(positions_chart, use_container_width=True)
                else:
                    st.info("📦 No open positions")
        
        with tab2:
            st.markdown("### 💼 Current Holdings")
            
            if state['positions']:
                position_data = []
                for symbol, pos in state['positions'].items():
                    pnl = pos.get('pnl', (pos['current_price'] - pos['avg_price']) * pos['qty'])
                    pnl_pct = pos.get('pnl_pct', (pos['current_price'] / pos['avg_price'] - 1) * 100 if pos['avg_price'] > 0 else 0)
                    
                    # Get signal data
                    signal = pos.get('current_signal', 0)
                    confidence = pos.get('signal_confidence', 0)
                    direction = pos.get('signal_direction', 'UNKNOWN')
                    strength = pos.get('signal_strength', 0)
                    
                    # Signal emoji
                    if signal > 0.40:
                        signal_emoji = "🟢"
                    elif signal > 0.20:
                        signal_emoji = "🟡"
                    elif signal > -0.20:
                        signal_emoji = "⚪"
                    elif signal > -0.40:
                        signal_emoji = "🟠"
                    else:
                        signal_emoji = "🔴"
                    
                    position_data.append({
                        'Symbol': symbol,
                        'Quantity': f"{pos['qty']:,}",
                        'Avg Price': f"${pos['avg_price']:.2f}",
                        'Current Price': f"${pos['current_price']:.2f}",
                        'Market Value': f"${pos['qty'] * pos['current_price']:,.2f}",
                        'Unrealized P&L': f"${pnl:+,.2f}",
                        'P&L %': f"{pnl_pct:+.2f}%",
                        'Signal': f"{signal_emoji} {direction}",  # ← NEW
                        'Strength': f"{signal:+.3f}",  # ← NEW
                        'Confidence': f"{confidence:.1%}",  # ← NEW
                        'Sector': ApexConfig.get_sector(symbol) if hasattr(ApexConfig, 'get_sector') else 'N/A'
                    })
                
                df = pd.DataFrame(position_data)
                
                # Style the dataframe
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=min(len(df) * 40 + 50, 600)
                )
                
                # Summary stats
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                total_value = sum([pos['qty'] * pos['current_price'] for pos in state['positions'].values()])
                total_pnl = sum([pos.get('pnl', 0) for pos in state['positions'].values()])
                avg_pnl_pct = (total_pnl / (total_value - total_pnl)) * 100 if (total_value - total_pnl) > 0 else 0
                avg_signal = sum([pos.get('current_signal', 0) for pos in state['positions'].values()]) / len(state['positions'])
                avg_confidence = sum([pos.get('signal_confidence', 0) for pos in state['positions'].values()]) / len(state['positions'])
                
                col1.metric("Total Positions", len(state['positions']))
                col2.metric("Total Value", f"${total_value:,.0f}")
                col3.metric("Total Unrealized P&L", f"${total_pnl:+,.2f}")
                col4.metric("Avg Signal Strength", f"{avg_signal:+.3f}")  # ← NEW
                col5.metric("Avg Confidence", f"{avg_confidence:.1%}")  # ← NEW
                
            else:
                st.info("📦 No open positions currently")
        
        with tab3:
            st.markdown("### 📜 Recent Trades")
            
            if show_trades and len(trades_df) > 0:
                trades_display = trades_df.copy().tail(100)
                trades_display['timestamp'] = pd.to_datetime(trades_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                trades_display['price'] = trades_display['price'].apply(lambda x: f"${x:.2f}")
                trades_display['pnl'] = trades_display['pnl'].apply(lambda x: f"${x:+,.2f}")
                
                trades_display = trades_display.rename(columns={
                    'timestamp': 'Time',
                    'symbol': 'Symbol',
                    'side': 'Side',
                    'quantity': 'Quantity',
                    'price': 'Price',
                    'pnl': 'P&L'
                })
                
                st.dataframe(
                    trades_display,
                    use_container_width=True,
                    height=500
                )
                
                # Trade statistics
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                total_trades = len(trades_df)
                buys = len(trades_df[trades_df['side'] == 'BUY'])
                sells = len(trades_df[trades_df['side'] == 'SELL'])
                total_pnl_trades = trades_df['pnl'].sum()
                
                col1.metric("Total Trades", total_trades)
                col2.metric("Buys / Sells", f"{buys} / {sells}")
                col3.metric("Total Realized P&L", f"${total_pnl_trades:+,.2f}")
                col4.metric("Avg Trade P&L", f"${total_pnl_trades/max(sells,1):+,.2f}")
                
            else:
                st.info("📊 No trades executed yet")
        
        # Auto-refresh
        time.sleep(refresh_interval)
        st.rerun()


def main():
    """Main entry point."""
    dashboard = ApexDashboard()
    dashboard.render()


if __name__ == "__main__":
    main()
