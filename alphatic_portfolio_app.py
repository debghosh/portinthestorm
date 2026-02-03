"""
Alphatic Portfolio Analyzer - MODULAR VERSION
A comprehensive portfolio analysis platform with advanced features for sophisticated investors

NEW FEATURES:
- Visual enhancements (modern gradient backgrounds, professional typography)
- Educational features (detailed metric explanations with tooltips)
- Market Regime Analysis (5 regime types with historical classification)
- Forward-Looking Risk Analysis (Monte Carlo simulations, VaR, CVaR)
- Enhanced interpretations for every chart
- Complete PyFolio integration

MODULAR STRUCTURE:
- helper_functions.py: All calculation and utility functions
- sidebar_panel.py: Portfolio builder and management
- tabs/: Individual tab modules (11 tabs)
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import all helper functions
from helper_functions import *

# Import sidebar and tab modules
import sidebar_panel
from tabs import (
    tab_00_education,
    tab_01_overview,
    tab_02_detailed_analysis,
    tab_03_sleeves,
    tab_04_pyfolio,
    tab_05_backtesting,
    tab_06_market_regimes,
    tab_07_forward_risk,
    tab_08_compare_benchmarks,
    tab_09_optimization,
    tab_10_trading_signals,
    tab_11_technical_charts
)


# OpenBB Platform (optional - for advanced features)
try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False
    # Will show warning in sidebar if needed


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Alphatic Portfolio Analyzer ✨",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ENHANCED CUSTOM CSS - MODERN GRADIENT THEME
# =============================================================================

st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Modern Gradient Background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Main Header */
    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .tagline {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-out;
    }
    
    /* Sub Headers */
    .sub-header {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 5px solid #667eea;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Color-Coded Metric Boxes */
    .metric-excellent {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .metric-good {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .metric-fair {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .metric-poor {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Success/Warning/Info Boxes */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: slideInLeft 0.5s ease-out;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ffc107;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: slideInRight 0.5s ease-out;
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #17a2b8;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Interpretation Boxes */
    .interpretation-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
        border-left: 5px solid #2196f3;
    }
    
    .interpretation-title {
        font-weight: 600;
        color: #1976d2;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================

if 'portfolios' not in st.session_state:
    st.session_state.portfolios = {}
if 'current_portfolio' not in st.session_state:
    st.session_state.current_portfolio = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}

# =============================================================================
# RENDER SIDEBAR
# =============================================================================

sidebar_panel.render()

# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

# Header
st.markdown('<h1 class="main-header">Alphatic Portfolio Analyzer ✨</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Sophisticated analysis for the educated investor</p>', unsafe_allow_html=True)

# =============================================================================
# TABS STRUCTURE  
# =============================================================================

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "📚 Portfolio Education",
    "📊 Overview",
    "📈 Detailed Analysis",
    "🎯 Sleeves",
    "📉 PyFolio Analysis",
    "⚔️ Backtesting",
    "🌡️ Market Regimes",
    "🔮 Forward Risk",
    "⚖️ Compare Benchmarks",
    "🎯 Optimization",
    "📡 Trading Signals",
    "📉 Technical Charts"
])

# =============================================================================
# RENDER TAB 0: PORTFOLIO EDUCATION (Always visible)
# =============================================================================

tab_00_education.render(tab0)

# =============================================================================
# CHECK IF PORTFOLIO EXISTS
# =============================================================================

if not st.session_state.portfolios or not st.session_state.current_portfolio:
    # No portfolio message for other tabs
    no_portfolio_msg = """
        <div style="text-align: center; padding: 4rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                    border-radius: 15px; margin: 2rem auto; max-width: 600px;">
            <h2 style="color: #667eea; margin-bottom: 1rem;">📊 No Portfolio Created Yet</h2>
            <p style="font-size: 1.1rem; color: #6c757d; margin-bottom: 2rem;">
                Start by creating a portfolio to see detailed analysis!
            </p>
            <ol style="text-align: left; display: inline-block; color: #6c757d; font-size: 1rem;">
                <li>Go to <strong>📚 Portfolio Education</strong> tab</li>
                <li>Choose a model (e.g., ⚖️ Classic 60/40)</li>
                <li>Click "📥 Load"</li>
                <li>Go to <strong>sidebar</strong> → "🚀 Build Portfolio"</li>
            </ol>
        </div>
    """
    with tab1:
        st.markdown(no_portfolio_msg, unsafe_allow_html=True)
    with tab2:
        st.markdown(no_portfolio_msg, unsafe_allow_html=True)
    with tab3:
        st.markdown(no_portfolio_msg, unsafe_allow_html=True)
    with tab4:
        st.markdown(no_portfolio_msg, unsafe_allow_html=True)
    with tab5:
        st.markdown(no_portfolio_msg, unsafe_allow_html=True)
    with tab6:
        st.markdown(no_portfolio_msg, unsafe_allow_html=True)
    with tab7:
        st.markdown(no_portfolio_msg, unsafe_allow_html=True)
    with tab8:
        st.markdown(no_portfolio_msg, unsafe_allow_html=True)
    with tab9:
        st.markdown(no_portfolio_msg, unsafe_allow_html=True)
    with tab10:
        st.markdown(no_portfolio_msg, unsafe_allow_html=True)
else:
    # Portfolio exists - define all variables needed by tabs
    current = st.session_state.portfolios[st.session_state.current_portfolio]
    portfolio_returns = current['returns']
    prices = current['prices']
    weights = current['weights']
    tickers = current['tickers']
    metrics = calculate_portfolio_metrics(portfolio_returns)
    
    # =============================================================================
    # RENDER ALL TABS WITH PORTFOLIO DATA
    # =============================================================================
    
    tab_01_overview.render(tab1, portfolio_returns, prices, weights, tickers, metrics, current)
    tab_02_detailed_analysis.render(tab2, portfolio_returns, prices, weights, tickers, metrics, current)
    tab_03_sleeves.render(tab3, portfolio_returns, prices, weights, tickers, metrics, current)
    tab_04_pyfolio.render(tab4, portfolio_returns, prices, weights, tickers, metrics, current)
    tab_05_backtesting.render(tab5, portfolio_returns, prices, weights, tickers, metrics, current)
    tab_06_market_regimes.render(tab6, portfolio_returns, prices, weights, tickers, metrics, current)
    tab_07_forward_risk.render(tab7, portfolio_returns, prices, weights, tickers, metrics, current)
    tab_08_compare_benchmarks.render(tab8, portfolio_returns, prices, weights, tickers, metrics, current)
    tab_09_optimization.render(tab9, portfolio_returns, prices, weights, tickers, metrics, current)
    tab_10_trading_signals.render(tab10, portfolio_returns, prices, weights, tickers, metrics, current)
    tab_11_technical_charts.render(tab11, portfolio_returns, prices, weights, tickers, metrics, current)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 2rem;">
        <p style="font-size: 1.1rem;">
            <strong>Alphatic Portfolio Analyzer ✨</strong><br>
            Sophisticated analysis for the educated investor
        </p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            Built with ❤️ for affluent non-experts who want to understand their investments<br>
            Remember: Past performance does not guarantee future results. Invest responsibly.
        </p>
    </div>
""", unsafe_allow_html=True)
