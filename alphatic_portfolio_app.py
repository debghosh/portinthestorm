"""
Alphatic Portfolio Analyzer - ENHANCED VERSION
A comprehensive portfolio analysis platform with advanced features for sophisticated investors

NEW FEATURES:
- Visual enhancements (modern gradient backgrounds, professional typography)
- Educational features (detailed metric explanations with tooltips)
- Market Regime Analysis (5 regime types with historical classification)
- Forward-Looking Risk Analysis (Monte Carlo simulations, VaR, CVaR)
- Enhanced interpretations for every chart
- Complete PyFolio integration
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import pyfolio as pf
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# TECHNICAL ANALYSIS FUNCTIONS (AlphaPy-Inspired)
# =============================================================================

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_support_resistance(prices, window=20):
    """
    Identify key support and resistance levels
    Uses rolling highs/lows and pivot points
    """
    # Recent highs and lows
    rolling_high = prices.rolling(window=window).max()
    rolling_low = prices.rolling(window=window).min()
    
    # Calculate pivot points
    high = prices.rolling(window=3).max()
    low = prices.rolling(window=3).min()
    close = prices
    
    pivot = (high + low + close) / 3
    resistance1 = 2 * pivot - low
    support1 = 2 * pivot - high
    resistance2 = pivot + (high - low)
    support2 = pivot - (high - low)
    
    return {
        'resistance_1': resistance1.iloc[-1],
        'resistance_2': resistance2.iloc[-1],
        'support_1': support1.iloc[-1],
        'support_2': support2.iloc[-1],
        'pivot': pivot.iloc[-1],
        'recent_high': rolling_high.iloc[-1],
        'recent_low': rolling_low.iloc[-1]
    }

def generate_trading_signal(prices):
    """
    Generate comprehensive trading signal with confidence
    """
    # Calculate indicators
    rsi = calculate_rsi(prices)
    macd, macd_signal, macd_hist = calculate_macd(prices)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
    sma_50 = calculate_sma(prices, 50)
    sma_200 = calculate_sma(prices, 200)
    
    # Get current values
    current_price = prices.iloc[-1]
    current_rsi = rsi.iloc[-1]
    current_macd = macd.iloc[-1]
    current_macd_signal = macd_signal.iloc[-1]
    current_macd_hist = macd_hist.iloc[-1]
    prev_macd_hist = macd_hist.iloc[-2] if len(macd_hist) > 1 else 0
    
    # Scoring system
    score = 0
    signals = []
    
    # RSI signals
    if current_rsi < 30:
        score += 2
        signals.append("RSI Oversold (Bullish)")
    elif current_rsi > 70:
        score -= 2
        signals.append("RSI Overbought (Bearish)")
    elif current_rsi < 40:
        score += 1
        signals.append("RSI Bullish Lean")
    elif current_rsi > 60:
        score -= 1
        signals.append("RSI Bearish Lean")
    
    # MACD signals
    if current_macd > current_macd_signal and prev_macd_hist < 0 < current_macd_hist:
        score += 2
        signals.append("MACD Bullish Crossover")
    elif current_macd < current_macd_signal and prev_macd_hist > 0 > current_macd_hist:
        score -= 2
        signals.append("MACD Bearish Crossover")
    elif current_macd > current_macd_signal:
        score += 1
        signals.append("MACD Bullish")
    else:
        score -= 1
        signals.append("MACD Bearish")
    
    # Trend signals
    if not pd.isna(sma_50.iloc[-1]) and not pd.isna(sma_200.iloc[-1]):
        if current_price > sma_50.iloc[-1] > sma_200.iloc[-1]:
            score += 2
            signals.append("Strong Uptrend")
        elif current_price < sma_50.iloc[-1] < sma_200.iloc[-1]:
            score -= 2
            signals.append("Strong Downtrend")
        elif current_price > sma_200.iloc[-1]:
            score += 1
            signals.append("Above 200 SMA")
        else:
            score -= 1
            signals.append("Below 200 SMA")
    
    # Bollinger Band signals
    if current_price < bb_lower.iloc[-1]:
        score += 1
        signals.append("Below Lower BB")
    elif current_price > bb_upper.iloc[-1]:
        score -= 1
        signals.append("Above Upper BB")
    
    # Determine overall signal
    if score >= 4:
        signal = "STRONG BUY"
        action = "Accumulate"
    elif score >= 2:
        signal = "BUY"
        action = "Accumulate"
    elif score <= -4:
        signal = "STRONG SELL"
        action = "Distribute"
    elif score <= -2:
        signal = "SELL"
        action = "Distribute"
    else:
        signal = "HOLD"
        action = "Hold"
    
    confidence = min(abs(score) * 15, 100)
    
    return {
        'signal': signal,
        'action': action,
        'score': score,
        'confidence': confidence,
        'signals': signals,
        'rsi': current_rsi,
        'macd': current_macd,
        'macd_signal': current_macd_signal,
        'price_vs_sma50': ((current_price / sma_50.iloc[-1]) - 1) * 100 if not pd.isna(sma_50.iloc[-1]) else None,
        'price_vs_sma200': ((current_price / sma_200.iloc[-1]) - 1) * 100 if not pd.isna(sma_200.iloc[-1]) else None
    }

def detect_market_regime_enhanced(returns, prices):
    """
    Enhanced market regime detection with 5 regimes and actionable recommendations
    """
    # Calculate metrics
    vol_window = min(60, len(returns))
    volatility = returns.tail(vol_window).std() * np.sqrt(252)
    recent_return_20d = (prices.iloc[-1] / prices.iloc[-20] - 1) * 100 if len(prices) >= 20 else 0
    recent_return_60d = (prices.iloc[-1] / prices.iloc[-60] - 1) * 100 if len(prices) >= 60 else 0
    momentum_60d = recent_return_60d / 100
    
    # Calculate SMAs
    sma_50 = prices.rolling(window=50).mean()
    sma_200 = prices.rolling(window=200).mean()
    
    # Price relative to SMAs
    price_vs_sma200 = None
    if len(sma_200) > 0 and not pd.isna(sma_200.iloc[-1]):
        price_vs_sma200 = ((prices.iloc[-1] / sma_200.iloc[-1]) - 1) * 100
    
    # Trend determination
    if len(sma_50) >= 50 and len(sma_200) >= 200:
        if sma_50.iloc[-1] > sma_200.iloc[-1]:
            trend = "Bullish"
        elif sma_50.iloc[-1] < sma_200.iloc[-1]:
            trend = "Bearish"
        else:
            trend = "Neutral"
    else:
        trend = "Insufficient Data"
    
    # Regime classification
    signals = []
    
    if volatility > 0.35:  # High volatility
        regime = "⚠️ High Volatility / Crisis"
        confidence = "High"
        action = "Reduce equity exposure to 40-50%. Increase cash and defensive positions. Avoid new positions until volatility subsides."
        allocation = {'stocks': 45, 'bonds': 45, 'cash': 10}
        color = 'error'
        signals.append(f"Volatility extremely high: {volatility*100:.1f}%")
        
    elif momentum_60d < -0.10 and volatility > 0.25:  # Negative momentum + elevated vol
        regime = "🐻 Bear Market"
        confidence = "High" if abs(momentum_60d) > 0.15 else "Medium"
        action = "Reduce equity to 50-60%. Focus on quality, dividend-paying stocks. Consider defensive sectors."
        allocation = {'stocks': 55, 'bonds': 40, 'cash': 5}
        color = 'error'
        signals.append(f"Negative momentum: {momentum_60d*100:.1f}%")
        if trend == "Bearish":
            signals.append("Death Cross: 50-day below 200-day SMA")
            
    elif momentum_60d > 0.15 and volatility < 0.20:  # Strong positive momentum + low vol
        regime = "🐂 Bull Market"
        confidence = "High"
        action = "Maintain 70-80% equity allocation. This is accumulation phase. Focus on growth and momentum."
        allocation = {'stocks': 75, 'bonds': 22, 'cash': 3}
        color = 'success'
        signals.append(f"Strong positive momentum: {momentum_60d*100:.1f}%")
        if trend == "Bullish":
            signals.append("Golden Cross: 50-day above 200-day SMA")
            
    elif momentum_60d > 0 and recent_return_20d > 0:  # Recovering
        regime = "📈 Recovery"
        confidence = "Medium"
        action = "Gradually increase equity to 60-70%. Good time to add positions. Monitor for continued strength."
        allocation = {'stocks': 65, 'bonds': 30, 'cash': 5}
        color = 'warning'
        signals.append(f"Recovery in progress: {momentum_60d*100:.1f}% momentum")
        
    else:  # Neutral / Choppy
        regime = "➡️ Neutral / Consolidation"
        confidence = "Medium"
        action = "Maintain balanced 60/40 portfolio. Wait for clearer directional signals before making changes."
        allocation = {'stocks': 60, 'bonds': 35, 'cash': 5}
        color = 'info'
        signals.append("Market lacking clear direction")
    
    return {
        'regime': regime,
        'confidence': confidence,
        'action': action,
        'allocation': allocation,
        'color': color,
        'signals': signals,
        'metrics': {
            'volatility': volatility,
            'momentum_60d': momentum_60d,
            'recent_return_20d': recent_return_20d,
            'recent_return_60d': recent_return_60d,
            'trend': trend,
            'price_vs_sma200': price_vs_sma200
        }
    }


# OpenBB Platform (optional - for advanced features)
try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False
    st.sidebar.warning("⚠️ OpenBB not installed. Some advanced features disabled. Install with: pip install openbb --break-system-packages")

# Configure page
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

# Initialize session state
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = {}
if 'current_portfolio' not in st.session_state:
    st.session_state.current_portfolio = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}


# =============================================================================
# EDUCATIONAL CONTENT - METRIC EXPLANATIONS
# =============================================================================

METRIC_EXPLANATIONS = {
    'annual_return': {
        'simple': 'Average yearly gain/loss of your investment',
        'detailed': '''
        **Annual Return** is the average yearly percentage gain or loss on your investment.
        
        **Real-World Example:** If your portfolio has a 10% annual return:
        - $100,000 grows to $110,000 in one year
        - Over 10 years, it grows to approximately $259,000 (with compounding)
        
        **What's Normal:**
        - S&P 500 long-term average: ~10% per year
        - Conservative portfolios: 4-6% per year
        - Aggressive growth: 12-15%+ per year
        
        **Important:** Past returns don't guarantee future results!
        ''',
        'thresholds': {
            'excellent': (15, 'Above 15% - Outstanding performance'),
            'good': (10, '10-15% - Very good performance'),
            'fair': (5, '5-10% - Moderate performance'),
            'poor': (0, 'Below 5% - Consider alternatives')
        }
    },
    
    'sharpe_ratio': {
        'simple': 'Risk-adjusted returns - higher is better',
        'detailed': '''
        **Sharpe Ratio** measures how much extra return you get for the extra risk you take.
        
        **Real-World Example:** 
        - Portfolio A: 12% return, Sharpe = 0.5 (volatile)
        - Portfolio B: 10% return, Sharpe = 1.5 (smooth)
        - Portfolio B is better! More consistent returns with less anxiety.
        
        **Professional Benchmarks:**
        - Below 1.0: Not great - too much risk for the return
        - 1.0-2.0: Good to excellent - solid risk-adjusted performance
        - Above 2.0: Outstanding - very rare, often unsustainable
        - Above 3.0: Exceptional - used by top hedge funds
        
        **Why It Matters:** Would you rather have a bumpy 15% return or a smooth 12%? 
        Sharpe Ratio helps you decide.
        ''',
        'thresholds': {
            'excellent': (2.0, 'Above 2.0 - Outstanding risk-adjusted returns'),
            'good': (1.0, '1.0-2.0 - Good risk-adjusted returns'),
            'fair': (0.5, '0.5-1.0 - Acceptable but could be better'),
            'poor': (0, 'Below 0.5 - Poor risk-adjusted returns')
        }
    },
    
    'max_drawdown': {
        'simple': 'Largest peak-to-trough decline',
        'detailed': '''
        **Maximum Drawdown** is the biggest drop from a peak to a trough in your portfolio value.
        
        **Real-World Example:**
        - Your portfolio peaks at $200,000
        - It drops to $150,000 during a market crash
        - Maximum Drawdown = 25% ($50,000 loss)
        - This is the worst pain you experienced
        
        **Can You Handle It?**
        - -10%: Mild correction, happens often
        - -20%: Significant drop, happens every few years
        - -30%: Severe bear market, very painful
        - -40%+: Crisis level, many investors panic sell (DON'T!)
        
        **2008 Crisis Reference:**
        - S&P 500: -56% drawdown
        - Conservative 60/40: -30% drawdown
        - Cash: 0% (but lost to inflation)
        
        **Key Question:** If your portfolio drops by this much, will you sell in panic or stay invested?
        ''',
        'thresholds': {
            'excellent': (-10, 'Above -10% - Very low drawdown'),
            'good': (-20, '-10% to -20% - Moderate drawdown'),
            'fair': (-30, '-20% to -30% - Significant drawdown'),
            'poor': (-40, 'Below -30% - Severe drawdown')
        }
    },
    
    'volatility': {
        'simple': 'How much your portfolio value fluctuates',
        'detailed': '''
        **Volatility (Standard Deviation)** measures how much your portfolio bounces around.
        
        **Real-World Example:**
        - Low volatility (10%): $100K portfolio typically moves $10K up/down yearly
        - Medium volatility (20%): $100K portfolio typically moves $20K up/down yearly
        - High volatility (30%+): $100K portfolio might move $30K+ yearly
        
        **Sleep Well Test:**
        - Below 10%: Very stable, good for retirees
        - 10-15%: Moderate, most can handle this
        - 15-20%: Elevated, need strong stomach
        - Above 20%: High, prepare for wild swings
        
        **Benchmark:**
        - S&P 500: ~15-20% volatility
        - Bonds: ~5-8% volatility
        - Bitcoin: 70-100% volatility (!)
        
        **Important:** Lower volatility = Better sleep at night
        ''',
        'thresholds': {
            'excellent': (10, 'Below 10% - Very low volatility'),
            'good': (15, '10-15% - Moderate volatility'),
            'fair': (20, '15-20% - Elevated volatility'),
            'poor': (25, 'Above 20% - High volatility')
        }
    },
    
    'sortino_ratio': {
        'simple': 'Like Sharpe but only penalizes downside risk',
        'detailed': '''
        **Sortino Ratio** is similar to Sharpe Ratio, but smarter: it only cares about bad volatility (drops), not good volatility (gains).
        
        **Why It's Better Than Sharpe:**
        - Sharpe penalizes you for BOTH ups and downs
        - Sortino only penalizes you for downs
        - Example: A portfolio that goes up 20%, up 25%, up 15% has high Sharpe volatility
        - But that's GOOD volatility! Sortino recognizes this.
        
        **Real-World Comparison:**
        - Portfolio A: Smooth 10% return, Sortino = 1.5
        - Portfolio B: Volatile 12% (mostly up), Sortino = 2.0
        - Portfolio B is better! Higher return AND better downside protection.
        
        **Professional Standards:**
        - Below 1.0: Excessive downside risk
        - 1.0-2.0: Good downside protection
        - Above 2.0: Excellent downside management
        - Above 3.0: Elite downside protection
        
        **Use This When:** You don't mind upside volatility, but you hate losses.
        ''',
        'thresholds': {
            'excellent': (2.0, 'Above 2.0 - Excellent downside protection'),
            'good': (1.0, '1.0-2.0 - Good downside protection'),
            'fair': (0.5, '0.5-1.0 - Moderate downside risk'),
            'poor': (0, 'Below 0.5 - High downside risk')
        }
    },
    
    'calmar_ratio': {
        'simple': 'Return relative to worst drawdown',
        'detailed': '''
        **Calmar Ratio** = Annual Return ÷ Maximum Drawdown
        
        **Real-World Example:**
        - Portfolio A: 12% return, -30% max drawdown → Calmar = 0.4
        - Portfolio B: 10% return, -15% max drawdown → Calmar = 0.67
        - Portfolio B is better! Less risk for similar return.
        
        **What It Means:**
        - A Calmar of 0.5 means you get 0.5% return for every 1% of max drawdown
        - Higher is better - more return for less pain
        
        **Professional Standards:**
        - Below 0.5: High risk for the return
        - 0.5-1.0: Good balance
        - 1.0-2.0: Excellent risk-adjusted returns
        - Above 2.0: Outstanding - rare
        
        **Use Case:** Comparing strategies with different risk profiles.
        ''',
        'thresholds': {
            'excellent': (1.5, 'Above 1.5 - Outstanding return vs drawdown'),
            'good': (0.75, '0.75-1.5 - Good return vs drawdown'),
            'fair': (0.5, '0.5-0.75 - Acceptable'),
            'poor': (0, 'Below 0.5 - High risk for the return')
        }
    },
    
    'alpha': {
        'simple': 'Returns above/below expected (vs benchmark)',
        'detailed': '''
        **Alpha** measures if your portfolio beat the market (benchmark) after accounting for risk.
        
        **Real-World Example:**
        - Benchmark (SPY) returns 10%
        - Your portfolio returns 12% with same risk → Alpha = +2%
        - You added 2% of value through smart selection!
        
        **What Positive Alpha Means:**
        - +2% Alpha = You beat the market by 2% per year
        - Over 10 years, that's 22% more wealth!
        - On $1M, that's an extra $220,000
        
        **Reality Check:**
        - Most professional managers have NEGATIVE alpha (after fees)
        - Getting positive alpha consistently is very hard
        - Even +1% alpha is considered excellent
        
        **Professional Standards:**
        - Positive: You're beating the market - great job!
        - Negative but close to 0: You're matching the market
        - Significantly negative: Consider index funds instead
        
        **Important:** Alpha can be due to skill OR luck. Longer time periods = more reliable.
        ''',
        'thresholds': {
            'excellent': (3, 'Above 3% - Outstanding value added'),
            'good': (1, '1-3% - Good value added'),
            'fair': (-1, '-1% to 1% - Matching benchmark'),
            'poor': (-3, 'Below -1% - Underperforming')
        }
    },
    
    'beta': {
        'simple': 'How much your portfolio moves with the market',
        'detailed': '''
        **Beta** measures how much your portfolio moves compared to the market (benchmark).
        
        **Real-World Example:**
        - Beta = 1.0: Your portfolio moves exactly like the market
          - Market up 10% → Your portfolio up 10%
        - Beta = 1.5: Your portfolio is 50% more volatile
          - Market up 10% → Your portfolio up 15%
          - Market down 10% → Your portfolio down 15%
        - Beta = 0.5: Your portfolio is 50% less volatile
          - Market up 10% → Your portfolio up 5%
          - Market down 10% → Your portfolio down 5%
        
        **What's Right for You?**
        - Beta < 0.8: Conservative, defensive portfolio
        - Beta 0.8-1.2: Similar to market
        - Beta > 1.2: Aggressive, amplified moves
        
        **Life Stage Guide:**
        - Young (20-40): Beta 1.0-1.3 (ride the growth)
        - Mid-career (40-55): Beta 0.8-1.1 (moderate)
        - Near retirement (55-65): Beta 0.6-0.9 (defensive)
        - Retired (65+): Beta 0.5-0.7 (preserve capital)
        
        **Important:** High beta = Higher risk AND higher potential reward
        ''',
        'thresholds': {
            'excellent': (0.8, '0.8-1.2 - Well-balanced market exposure'),
            'good': (0.6, '0.6-0.8 or 1.2-1.4 - Moderate deviation'),
            'fair': (0.5, '0.5-0.6 or 1.4-1.6 - Significant deviation'),
            'poor': (0, 'Below 0.5 or above 1.6 - Extreme deviation')
        }
    },
    
    'win_rate': {
        'simple': 'Percentage of profitable periods',
        'detailed': '''
        **Win Rate** is the percentage of time periods (days, months, etc.) where your portfolio made money.
        
        **Real-World Example:**
        - 65% daily win rate = 65% of days are green (up)
        - 75% monthly win rate = 3 out of 4 months are positive
        
        **Interpretation:**
        - Above 60%: Very consistent, good for confidence
        - 50-60%: Typical for good strategies
        - Below 50%: More losing periods than winning
        
        **Psychology Matters:**
        - Higher win rate = Better emotional experience
        - Lower win rate can still work if wins are bigger than losses
        - Example: 40% win rate but wins average +5% and losses average -1%
        
        **Benchmark:**
        - S&P 500: ~55% daily win rate
        - Good trend-following: 45-50% win rate (but big wins)
        - Mean reversion: 60-70% win rate (but smaller wins)
        
        **Use This To:** Assess if you can emotionally handle the strategy.
        ''',
        'thresholds': {
            'excellent': (65, 'Above 65% - Highly consistent'),
            'good': (55, '55-65% - Good consistency'),
            'fair': (50, '50-55% - Acceptable'),
            'poor': (45, 'Below 50% - More losing than winning periods')
        }
    }
}


# =============================================================================
# OPENBB HELPER FUNCTIONS - PHASE 1 FEATURES
# =============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_etf_info_openbb(symbol):
    """
    Get comprehensive ETF information using OpenBB
    Returns dict with info, holdings, sectors, or None if unavailable
    """
    if not OPENBB_AVAILABLE:
        return None
    
    try:
        # Note: OpenBB API structure - adjust based on actual OpenBB version
        # This is a template - actual implementation depends on OpenBB 4.x API
        result = {
            'symbol': symbol,
            'basic_info': {
                'name': f"{symbol} ETF",  # Placeholder
                'expense_ratio': 0.0,
                'aum': 0.0,
                'inception_date': 'N/A',
                'dividend_yield': 0.0
            },
            'holdings': pd.DataFrame(),  # Top holdings
            'sectors': {}  # Sector allocation
        }
        
        # Try to get real data from OpenBB
        # Note: Actual OpenBB 4.x API calls would go here
        # For now, return placeholder structure
        
        return result
    except Exception as e:
        st.warning(f"Could not fetch OpenBB data for {symbol}: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_economic_data_openbb():
    """
    Get current economic indicators using OpenBB
    Returns dict with GDP, unemployment, inflation, etc.
    """
    if not OPENBB_AVAILABLE:
        return None
    
    try:
        # Placeholder structure for economic data
        economic_data = {
            'gdp_growth': 2.5,  # Percentage
            'unemployment': 3.8,  # Percentage
            'inflation_cpi': 2.8,  # Percentage
            'fed_funds_rate': 5.25,  # Percentage
            'treasury_10y': 4.15,  # Percentage
            'vix': 14.5,  # VIX level
            'yield_curve': -0.15,  # 10Y-2Y spread
            'last_updated': datetime.now()
        }
        
        # Try to get real data from OpenBB
        # Note: Actual OpenBB 4.x API calls would go here
        # Example: economic_data['gdp_growth'] = obb.economy.gdp().to_df()
        
        return economic_data
    except Exception as e:
        st.warning(f"Could not fetch economic data: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_benchmark_data_openbb(benchmark_symbol, start_date, end_date):
    """
    Get benchmark data using OpenBB (fallback to yfinance if unavailable)
    """
    # For now, use yfinance as it's more reliable
    # OpenBB can be integrated later for additional benchmarks
    try:
        data = download_ticker_data([benchmark_symbol], start_date, end_date)
        return data
    except Exception as e:
        st.warning(f"Could not fetch benchmark {benchmark_symbol}: {str(e)}")
        return None


def get_cheaper_etf_alternatives(symbol, expense_ratio):
    """
    Find cheaper alternatives to an ETF
    Returns list of similar ETFs with lower expense ratios
    """
    # Common ETF alternatives database
    alternatives = {
        'SPY': [
            {'symbol': 'VOO', 'name': 'Vanguard S&P 500', 'expense_ratio': 0.0003, 'tracking': 'Perfect'},
            {'symbol': 'IVV', 'name': 'iShares Core S&P 500', 'expense_ratio': 0.0003, 'tracking': 'Perfect'}
        ],
        'QQQ': [
            {'symbol': 'QQQM', 'name': 'Invesco NASDAQ 100', 'expense_ratio': 0.0015, 'tracking': 'Perfect'}
        ],
        'IWM': [
            {'symbol': 'VTWO', 'name': 'Vanguard Russell 2000', 'expense_ratio': 0.0010, 'tracking': 'Very Good'}
        ],
        'AGG': [
            {'symbol': 'BND', 'name': 'Vanguard Total Bond', 'expense_ratio': 0.0003, 'tracking': 'Excellent'}
        ],
        'VTI': [
            {'symbol': 'ITOT', 'name': 'iShares Core S&P Total', 'expense_ratio': 0.0003, 'tracking': 'Excellent'}
        ]
    }
    
    return alternatives.get(symbol, [])


def interpret_economic_regime(econ_data):
    """
    Interpret economic data into regime classification
    Returns regime name and description
    """
    if econ_data is None:
        return "Unknown", "Economic data unavailable"
    
    gdp = econ_data.get('gdp_growth', 0)
    inflation = econ_data.get('inflation_cpi', 0)
    unemployment = econ_data.get('unemployment', 0)
    
    # Goldilocks: Strong growth, low inflation, low unemployment
    if gdp > 2.0 and inflation < 3.5 and unemployment < 4.5:
        return "Goldilocks", "Strong growth + Low inflation + Low unemployment = Best for stocks"
    
    # Stagflation: Weak growth, high inflation
    elif gdp < 1.5 and inflation > 4.0:
        return "Stagflation", "Weak growth + High inflation = Bad for stocks and bonds"
    
    # Recession: Negative/very low growth, rising unemployment
    elif gdp < 0.5 or unemployment > 5.5:
        return "Recession", "Weak/negative growth = Defensive positioning needed"
    
    # Overheating: Strong growth, high inflation
    elif gdp > 3.0 and inflation > 3.5:
        return "Overheating", "Strong growth + High inflation = Fed likely to raise rates"
    
    # Moderate: Balanced conditions
    else:
        return "Moderate Growth", "Balanced economic conditions = Stable environment"


def get_upcoming_economic_events():
    """
    Get upcoming high-impact economic events
    Returns list of events with dates and impact levels
    """
    # For now, return common recurring events
    # In production, would fetch from economic calendar API
    today = datetime.now()
    
    events = []
    
    # Fed meetings (8 per year, roughly every 6 weeks)
    # Next meeting dates (these would come from API in production)
    fed_meetings = [
        datetime(2026, 1, 29),
        datetime(2026, 3, 19),
        datetime(2026, 5, 7),
        datetime(2026, 6, 18),
        datetime(2026, 7, 30),
        datetime(2026, 9, 17),
        datetime(2026, 11, 5),
        datetime(2026, 12, 17)
    ]
    
    for meeting in fed_meetings:
        if meeting > today and meeting < today + timedelta(days=90):
            events.append({
                'date': meeting,
                'event': 'Fed Meeting',
                'impact': 'HIGH',
                'description': 'FOMC rate decision and policy statement'
            })
    
    # Monthly jobs reports (first Friday of month)
    # CPI reports (mid-month)
    # GDP reports (quarterly)
    
    return sorted(events, key=lambda x: x['date'])[:5]  # Return next 5 events


def calculate_expense_ratio_savings(current_ratio, new_ratio, portfolio_value):
    """
    Calculate annual savings from switching to cheaper ETF
    """
    current_cost = portfolio_value * current_ratio
    new_cost = portfolio_value * new_ratio
    annual_savings = current_cost - new_cost
    
    # Calculate 20-year savings with compound effect
    years = 20
    annual_return = 0.08  # Assume 8% annual return
    
    # Future value of savings invested at 8% annually
    fv_savings = sum(annual_savings * ((1 + annual_return) ** (years - i)) for i in range(years))
    
    return {
        'annual_savings': annual_savings,
        'savings_20y': fv_savings,
        'percent_cheaper': ((current_ratio - new_ratio) / current_ratio * 100) if current_ratio > 0 else 0
    }


def get_smart_benchmarks(tickers, weights):
    """
    Auto-select relevant benchmarks based on portfolio composition
    Returns list of benchmark symbols with reasoning
    """
    benchmarks = []
    reasons = []
    
    # Always include S&P 500
    benchmarks.append('SPY')
    reasons.append('Core US large cap benchmark')
    
    # Check for tech-heavy portfolios
    tech_etfs = ['QQQ', 'XLK', 'VGT', 'SOXX']
    if any(ticker in tech_etfs for ticker in tickers):
        if 'QQQ' not in benchmarks:
            benchmarks.append('QQQ')
            reasons.append('Tech exposure warrants Nasdaq comparison')
    
    # Check for small cap exposure
    small_cap_etfs = ['IWM', 'VB', 'IJR']
    if any(ticker in small_cap_etfs for ticker in tickers):
        if 'IWM' not in benchmarks:
            benchmarks.append('IWM')
            reasons.append('Small cap exposure present')
    
    # Check for international exposure
    intl_etfs = ['VT', 'VXUS', 'EFA', 'VEA', 'IEFA']
    if any(ticker in intl_etfs for ticker in tickers):
        if 'VT' not in benchmarks:
            benchmarks.append('VT')
            reasons.append('International holdings present')
    
    # Check for bond exposure
    bond_etfs = ['AGG', 'BND', 'TLT', 'IEF', 'SHY']
    if any(ticker in bond_etfs for ticker in tickers):
        if 'AGG' not in benchmarks:
            benchmarks.append('AGG')
            reasons.append('Fixed income component')
    
    # Always add 60/40 for risk-adjusted comparison
    # We'll calculate this synthetically
    
    return list(zip(benchmarks, reasons))



def render_metric_explanation(metric_key):
    """
    Render an educational explanation for a metric in an expander
    """
    if metric_key in METRIC_EXPLANATIONS:
        info = METRIC_EXPLANATIONS[metric_key]
        
        with st.expander(f"ℹ️ Learn More About This Metric"):
            st.markdown(f"**Quick Summary:** {info['simple']}")
            st.markdown("---")
            st.markdown(info['detailed'])
            
            if 'thresholds' in info:
                st.markdown("---")
                st.markdown("**📊 How to Interpret:**")
                for level, (threshold, description) in info['thresholds'].items():
                    if level == 'excellent':
                        st.markdown(f"🟢 **Excellent:** {description}")
                    elif level == 'good':
                        st.markdown(f"🟡 **Good:** {description}")
                    elif level == 'fair':
                        st.markdown(f"🟠 **Fair:** {description}")
                    elif level == 'poor':
                        st.markdown(f"🔴 **Poor:** {description}")


def get_metric_color_class(metric_key, value):
    """
    Determine the CSS class for a metric based on its value
    """
    if metric_key not in METRIC_EXPLANATIONS:
        return 'metric-card'
    
    thresholds = METRIC_EXPLANATIONS[metric_key].get('thresholds', {})
    
    # Handle metrics where higher is better
    if metric_key in ['annual_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'alpha', 'win_rate']:
        if value >= thresholds.get('excellent', (float('inf'), ''))[0]:
            return 'metric-excellent'
        elif value >= thresholds.get('good', (float('inf'), ''))[0]:
            return 'metric-good'
        elif value >= thresholds.get('fair', (float('inf'), ''))[0]:
            return 'metric-fair'
        else:
            return 'metric-poor'
    
    # Handle max_drawdown (lower absolute value is better)
    elif metric_key == 'max_drawdown':
        if value >= thresholds.get('excellent', (-float('inf'), ''))[0]:
            return 'metric-excellent'
        elif value >= thresholds.get('good', (-float('inf'), ''))[0]:
            return 'metric-good'
        elif value >= thresholds.get('fair', (-float('inf'), ''))[0]:
            return 'metric-fair'
        else:
            return 'metric-poor'
    
    # Handle volatility (lower is better)
    elif metric_key == 'volatility':
        if value <= thresholds.get('excellent', (float('inf'), ''))[0]:
            return 'metric-excellent'
        elif value <= thresholds.get('good', (float('inf'), ''))[0]:
            return 'metric-good'
        elif value <= thresholds.get('fair', (float('inf'), ''))[0]:
            return 'metric-fair'
        else:
            return 'metric-poor'
    
    # Handle beta (closer to 1.0 is better)
    elif metric_key == 'beta':
        abs_deviation = abs(value - 1.0)
        if abs_deviation <= 0.2:
            return 'metric-excellent'
        elif abs_deviation <= 0.4:
            return 'metric-good'
        elif abs_deviation <= 0.6:
            return 'metric-fair'
        else:
            return 'metric-poor'
    
    return 'metric-card'


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def get_earliest_start_date(tickers):
    """
    Determine the earliest common start date for all tickers
    """
    earliest_dates = []
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, period='max', progress=False, auto_adjust=True)
            if not data.empty:
                earliest_dates.append(data.index[0])
        except Exception as e:
            st.warning(f"Could not fetch history for {ticker}: {str(e)}")
    
    if earliest_dates:
        return max(earliest_dates)
    return None


def download_ticker_data(tickers, start_date, end_date=None):
    """
    Download historical price data for multiple tickers with DIVIDENDS REINVESTED
    
    This function uses auto_adjust=True which automatically adjusts for:
    - Dividends (assumes reinvestment)
    - Stock splits
    - Other corporate actions
    
    This gives you TOTAL RETURN performance, not just price appreciation.
    """
    if end_date is None:
        end_date = datetime.now()
    
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True  # Automatically adjusts for dividends and splits
        )
        
        if len(tickers) == 1:
            data = pd.DataFrame(data['Close'])
            data.columns = tickers
        else:
            data = data['Close']
        
        return data
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None


# =============================================================================
# PORTFOLIO OPTIMIZATION FUNCTIONS
# =============================================================================

def calculate_portfolio_returns(prices, weights):
    """
    Calculate portfolio returns given prices and weights
    """
    returns = prices.pct_change().dropna()
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Ensure it's a Series with a name for consistency
    if not isinstance(portfolio_returns, pd.Series):
        portfolio_returns = pd.Series(portfolio_returns)
    
    # Give it a default name if it doesn't have one
    if portfolio_returns.name is None:
        portfolio_returns.name = 'returns'
    
    return portfolio_returns


def optimize_portfolio(prices, method='max_sharpe'):
    """
    Optimize portfolio weights
    """
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    num_assets = len(prices.columns)
    
    def portfolio_stats(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_std
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def neg_sharpe(weights):
        return -portfolio_stats(weights)[2]
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    if method == 'max_sharpe':
        result = minimize(neg_sharpe, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
    
    return result.x if result.success else initial_guess


def calculate_efficient_frontier(prices, num_portfolios=100):
    """
    Calculate efficient frontier for visualization
    """
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    num_assets = len(prices.columns)
    results = np.zeros((3, num_portfolios))
    weights_array = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_array.append(weights)
        
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = portfolio_return / portfolio_std
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std
        results[2,i] = sharpe
    
    return results, weights_array


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def calculate_portfolio_metrics(returns, benchmark_returns=None, risk_free_rate=0.02):
    """
    Calculate comprehensive portfolio metrics
    """
    # Ensure returns are a pandas Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else 0
    
    # Downside metrics
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (ann_return - risk_free_rate) / downside_std if downside_std != 0 else 0
    
    # Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns)
    
    metrics = {
        'Total Return': total_return,
        'Annual Return': ann_return,
        'Annual Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar,
        'Win Rate': win_rate
    }
    
    # Alpha and Beta (if benchmark provided)
    if benchmark_returns is not None:
        if isinstance(benchmark_returns, pd.DataFrame):
            benchmark_returns = benchmark_returns.iloc[:, 0]
        
        # Align the series
        aligned_data = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) > 0:
            covariance = aligned_data.cov().iloc[0, 1] * 252
            benchmark_variance = aligned_data['benchmark'].var() * 252
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
            
            benchmark_return = (1 + aligned_data['benchmark']).prod() - 1
            benchmark_ann_return = (1 + benchmark_return) ** (252 / len(aligned_data)) - 1
            
            alpha = ann_return - (risk_free_rate + beta * (benchmark_ann_return - risk_free_rate))
            
            metrics['Alpha'] = alpha
            metrics['Beta'] = beta
    
    return metrics


def detect_market_regimes(returns, lookback=60):
    """
    Detect market regimes based on volatility and returns
    
    Regimes:
    1. Bull Market (Low Vol) - Positive returns, low volatility
    2. Bull Market (High Vol) - Positive returns, high volatility  
    3. Sideways/Choppy - Returns near zero, any volatility
    4. Bear Market (Low Vol) - Negative returns, low volatility
    5. Bear Market (High Vol) - Negative returns, high volatility (crisis)
    """
    # Ensure returns is a Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    # Calculate rolling metrics
    rolling_returns = returns.rolling(lookback).mean() * 252  # Annualized
    rolling_vol = returns.rolling(lookback).std() * np.sqrt(252)  # Annualized
    
    # Calculate percentiles for thresholds
    vol_median = rolling_vol.median()
    return_positive = rolling_returns > 0.02  # Above 2% annualized
    return_negative = rolling_returns < -0.02  # Below -2% annualized
    vol_high = rolling_vol > vol_median
    
    # Classify regimes
    regimes = pd.Series(index=returns.index, dtype='object')
    regimes[:] = 'Sideways/Choppy'  # Default
    
    # Bull markets
    regimes[return_positive & ~vol_high] = 'Bull Market (Low Vol)'
    regimes[return_positive & vol_high] = 'Bull Market (High Vol)'
    
    # Bear markets
    regimes[return_negative & ~vol_high] = 'Bear Market (Low Vol)'
    regimes[return_negative & vol_high] = 'Bear Market (High Vol)'
    
    return regimes


def analyze_regime_performance(returns, regimes):
    """
    Analyze portfolio performance by market regime
    """
    df = pd.DataFrame({'returns': returns, 'regime': regimes})
    
    regime_stats = []
    for regime in df['regime'].unique():
        regime_returns = df[df['regime'] == regime]['returns']
        
        if len(regime_returns) > 0:
            stats = {
                'Regime': regime,
                'Occurrences': len(regime_returns),
                'Avg Daily Return': regime_returns.mean(),
                'Volatility': regime_returns.std() * np.sqrt(252),
                'Best Day': regime_returns.max(),
                'Worst Day': regime_returns.min(),
                'Win Rate': (regime_returns > 0).sum() / len(regime_returns)
            }
            regime_stats.append(stats)
    
    return pd.DataFrame(regime_stats)


def monte_carlo_simulation(returns, days_forward=252, num_simulations=1000):
    """
    Run Monte Carlo simulation for forward-looking risk analysis
    """
    # Ensure returns is a Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    # Calculate parameters from historical returns
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Run simulations
    last_price = 1.0  # Normalized starting point
    simulations = np.zeros((days_forward, num_simulations))
    
    for i in range(num_simulations):
        daily_returns = np.random.normal(mean_return, std_return, days_forward)
        price_path = last_price * (1 + daily_returns).cumprod()
        simulations[:, i] = price_path
    
    return simulations


def calculate_forward_risk_metrics(returns, confidence_level=0.95):
    """
    Calculate forward-looking risk metrics
    """
    # Ensure returns is a Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    # Expected return and volatility
    expected_return = returns.mean() * 252
    expected_vol = returns.std() * np.sqrt(252)
    
    # Value at Risk (VaR)
    var_95 = returns.quantile(1 - 0.95)
    var_99 = returns.quantile(1 - 0.99)
    
    # Conditional VaR (CVaR / Expected Shortfall)
    cvar_95 = returns[returns <= var_95].mean()
    cvar_99 = returns[returns <= var_99].mean()
    
    # Probability of daily loss
    prob_loss = (returns < 0).sum() / len(returns)
    
    # Estimated maximum drawdown (based on historical)
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - running_max) / running_max
    estimated_max_dd = drawdowns.min()
    
    return {
        'Expected Annual Return': expected_return,
        'Expected Volatility': expected_vol,
        'VaR (95%)': var_95,
        'VaR (99%)': var_99,
        'CVaR (95%)': cvar_95,
        'CVaR (99%)': cvar_99,
        'Probability of Daily Loss': prob_loss,
        'Estimated Max Drawdown': estimated_max_dd
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_cumulative_returns(returns, title='Cumulative Returns', benchmark_returns=None):
    """
    Plot cumulative returns over time with enhanced styling
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Ensure returns is a Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    cum_returns = (1 + returns).cumprod()
    cum_returns.plot(ax=ax, linewidth=2.5, label='Portfolio', color='#667eea')
    
    if benchmark_returns is not None:
        if isinstance(benchmark_returns, pd.DataFrame):
            benchmark_returns = benchmark_returns.iloc[:, 0]
        
        cum_bench = (1 + benchmark_returns).cumprod()
        cum_bench.plot(ax=ax, linewidth=2, label='Benchmark', 
                      color='#ff6b6b', linestyle='--', alpha=0.7)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    return fig


def plot_drawdown(returns, title='Drawdown Over Time'):
    """
    Plot drawdown over time with enhanced styling
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Ensure returns is a Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    
    ax.fill_between(drawdown.index, 0, drawdown.values, 
                    color='#dc3545', alpha=0.3, label='Drawdown')
    drawdown.plot(ax=ax, linewidth=2, color='#dc3545')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    return fig


def plot_monthly_returns_heatmap(returns, title='Monthly Returns Heatmap'):
    """
    Plot monthly returns as a heatmap with enhanced styling
    """
    # Ensure returns is a Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    # Calculate monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Convert to DataFrame with explicit column name
    monthly_returns_df = pd.DataFrame({'returns': monthly_returns})
    monthly_returns_df['Year'] = monthly_returns_df.index.year
    monthly_returns_df['Month'] = monthly_returns_df.index.month
    
    # Pivot the data
    monthly_returns_pivot = monthly_returns_df.pivot(
        index='Year', columns='Month', values='returns'
    )
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_returns_pivot.columns = [month_names[i-1] for i in monthly_returns_pivot.columns]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(monthly_returns_pivot * 100, annot=True, fmt='.1f', 
                cmap='RdYlGn', center=0, ax=ax, cbar_kws={'label': 'Return (%)'})
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Year', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_rolling_metrics(returns, window=60, title='Rolling Metrics'):
    """
    Plot rolling Sharpe and Sortino ratios with enhanced styling
    """
    # Ensure returns is a Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    rolling_return = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_return / rolling_vol
    
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    rolling_downside_vol = downside_returns.rolling(window).std() * np.sqrt(252)
    rolling_sortino = rolling_return / rolling_downside_vol
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Sharpe Ratio
    rolling_sharpe.plot(ax=ax1, linewidth=2, color='#667eea', label='Rolling Sharpe')
    ax1.axhline(y=1, color='#28a745', linestyle='--', alpha=0.7, label='Good (1.0)')
    ax1.axhline(y=0, color='#dc3545', linestyle='--', alpha=0.7)
    ax1.set_title(f'Rolling Sharpe Ratio ({window}-day)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    
    # Sortino Ratio
    rolling_sortino.plot(ax=ax2, linewidth=2, color='#764ba2', label='Rolling Sortino')
    ax2.axhline(y=1, color='#28a745', linestyle='--', alpha=0.7, label='Good (1.0)')
    ax2.axhline(y=0, color='#dc3545', linestyle='--', alpha=0.7)
    ax2.set_title(f'Rolling Sortino Ratio ({window}-day)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Sortino Ratio', fontsize=11, fontweight='bold')
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#f8f9fa')
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig


def plot_regime_chart(regimes, returns):
    """
    Plot market regime timeline with returns
    """
    # Ensure returns is a Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Color map for regimes
    regime_colors = {
        'Bull Market (Low Vol)': '#28a745',
        'Bull Market (High Vol)': '#17a2b8',
        'Sideways/Choppy': '#ffc107',
        'Bear Market (Low Vol)': '#fd7e14',
        'Bear Market (High Vol)': '#dc3545'
    }
    
    # Plot returns
    cum_returns = (1 + returns).cumprod()
    cum_returns.plot(ax=ax1, linewidth=2, color='#667eea', label='Portfolio Value')
    ax1.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
    ax1.set_title('Portfolio Performance Across Market Regimes', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    
    # Plot regimes as colored background
    for regime, color in regime_colors.items():
        mask = regimes == regime
        if mask.any():
            ax1.fill_between(returns.index, 0, 1, where=mask, 
                            transform=ax1.get_xaxis_transform(),
                            alpha=0.2, color=color, label=regime)
    
    # Create regime timeline
    regime_numeric = pd.Series(index=regimes.index, dtype=float)
    regime_map = {regime: i for i, regime in enumerate(regime_colors.keys())}
    for regime, value in regime_map.items():
        regime_numeric[regimes == regime] = value
    
    ax2.plot(regime_numeric.index, regime_numeric.values, linewidth=0)
    for regime, color in regime_colors.items():
        mask = regimes == regime
        if mask.any():
            ax2.fill_between(regimes.index, 0, 5, where=mask,
                            alpha=0.6, color=color, label=regime)
    
    ax2.set_yticks(range(len(regime_colors)))
    ax2.set_yticklabels(list(regime_colors.keys()))
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Market Regime', fontsize=12, fontweight='bold')
    ax2.set_title('Market Regime Classification', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax2.set_facecolor('#f8f9fa')
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig


def plot_monte_carlo_simulation(simulations, title='Monte Carlo Simulation - 1 Year Forward'):
    """
    Plot Monte Carlo simulation results
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot individual simulations (subset for performance)
    num_to_plot = min(100, simulations.shape[1])
    for i in range(0, num_to_plot):
        ax.plot(simulations[:, i], color='#667eea', alpha=0.1, linewidth=0.5)
    
    # Calculate and plot percentiles
    percentiles = [5, 25, 50, 75, 95]
    percentile_values = np.percentile(simulations, percentiles, axis=1)
    
    colors = ['#dc3545', '#fd7e14', '#28a745', '#17a2b8', '#6c757d']
    labels = ['5th %ile (Worst Case)', '25th %ile', '50th %ile (Median)', 
              '75th %ile', '95th %ile (Best Case)']
    
    for i, (pct, color, label) in enumerate(zip(percentile_values, colors, labels)):
        ax.plot(pct, color=color, linewidth=2.5, label=label, alpha=0.9)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Starting Value')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Trading Days Forward', fontsize=12, fontweight='bold')
    ax.set_ylabel('Portfolio Value (Normalized)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    return fig


def plot_efficient_frontier(results, optimal_weights, portfolio_return, portfolio_std):
    """
    Plot efficient frontier with enhanced styling
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(results[1,:], results[0,:], c=results[2,:], 
                        cmap='viridis', marker='o', s=50, alpha=0.6)
    ax.scatter(portfolio_std, portfolio_return, marker='*', color='red', 
              s=500, label='Current Portfolio', edgecolors='black', linewidths=2)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20, fontweight='bold')
    
    ax.set_title('Efficient Frontier', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Volatility (Standard Deviation)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return', fontsize=12, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    return fig


# =============================================================================
# SIDEBAR - PORTFOLIO BUILDER
# =============================================================================

st.sidebar.markdown("## 📊 Alphatic Portfolio Analyzer ✨")
st.sidebar.markdown("---")

# Portfolio Builder Section
st.sidebar.markdown("### 🔨 Build Portfolio")

# Input for new portfolio name
portfolio_name = st.sidebar.text_input("Portfolio Name", value="My Portfolio")

# Ticker input
ticker_input = st.sidebar.text_area(
    "Enter Tickers (one per line or comma-separated)",
    value="SPY\nQQQ\nAGG",
    height=100
)

# Parse tickers
if ticker_input:
    tickers_list = [t.strip().upper() for t in ticker_input.replace(',', '\n').split('\n') if t.strip()]
else:
    tickers_list = []

# Allocation method
allocation_method = st.sidebar.radio(
    "Allocation Method",
    ["Equal Weight", "Custom Weights", "Optimize (Max Sharpe)"]
)

# Custom weights if selected
custom_weights = {}
if allocation_method == "Custom Weights" and tickers_list:
    st.sidebar.markdown("**Set Custom Weights (must sum to 100%):**")
    for ticker in tickers_list:
        weight = st.sidebar.number_input(
            f"{ticker} %",
            min_value=0.0,
            max_value=100.0,
            value=100.0 / len(tickers_list),
            step=1.0,
            key=f"weight_{ticker}"
        )
        custom_weights[ticker] = weight / 100.0
    
    weight_sum = sum(custom_weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        st.sidebar.warning(f"⚠️ Weights sum to {weight_sum*100:.1f}% (should be 100%)")

# Date range selection
st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Date Range")

date_method = st.sidebar.radio(
    "Start Date Method",
    ["Auto (Earliest Available)", "Custom Date"]
)

if date_method == "Custom Date":
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime(2020, 1, 1)
    )
else:
    start_date = None

end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.now()
)

# Build Portfolio Button
if st.sidebar.button("🚀 Build Portfolio", type="primary"):
    if not tickers_list:
        st.sidebar.error("Please enter at least one ticker!")
    else:
        with st.spinner("Building portfolio..."):
            # Determine start date if auto
            if start_date is None:
                st.info("Determining earliest available start date...")
                auto_start_date = get_earliest_start_date(tickers_list)
                if auto_start_date:
                    start_date = auto_start_date
                    st.success(f"✅ Using earliest start date: {start_date.strftime('%Y-%m-%d')}")
                else:
                    st.error("Could not determine start date. Please use custom date.")
                    st.stop()
            
            # Download data
            prices = download_ticker_data(tickers_list, start_date, end_date)
            
            if prices is not None and not prices.empty:
                # Determine weights
                if allocation_method == "Equal Weight":
                    weights = {ticker: 1/len(tickers_list) for ticker in tickers_list}
                elif allocation_method == "Custom Weights":
                    weights = custom_weights
                else:  # Optimize
                    optimal_weights = optimize_portfolio(prices)
                    weights = {ticker: w for ticker, w in zip(tickers_list, optimal_weights)}
                
                # Calculate portfolio returns
                weights_array = np.array([weights[ticker] for ticker in prices.columns])
                portfolio_returns = calculate_portfolio_returns(prices, weights_array)
                
                # Store in session state
                st.session_state.portfolios[portfolio_name] = {
                    'tickers': tickers_list,
                    'weights': weights,
                    'prices': prices,
                    'returns': portfolio_returns,
                    'start_date': start_date,
                    'end_date': end_date
                }
                st.session_state.current_portfolio = portfolio_name
                
                st.sidebar.success(f"✅ Portfolio '{portfolio_name}' created successfully!")
                st.sidebar.info("📊 Returns include dividends reinvested (Total Return)")
            else:
                st.sidebar.error("Failed to download price data. Please check tickers and dates.")

# Portfolio Management
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Manage Portfolios")

if st.session_state.portfolios:
    # Select portfolio
    selected_portfolio = st.sidebar.selectbox(
        "Select Portfolio",
        list(st.session_state.portfolios.keys()),
        index=list(st.session_state.portfolios.keys()).index(st.session_state.current_portfolio) 
        if st.session_state.current_portfolio else 0
    )
    st.session_state.current_portfolio = selected_portfolio
    
    # Delete portfolio
    if st.sidebar.button("🗑️ Delete Selected Portfolio"):
        del st.session_state.portfolios[selected_portfolio]
        st.session_state.current_portfolio = list(st.session_state.portfolios.keys())[0] if st.session_state.portfolios else None
        st.sidebar.success("Portfolio deleted!")
        st.rerun()
    
    # Export/Import
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Export/Import")
    
    if st.sidebar.button("📥 Export All Portfolios"):
        export_data = {}
        for name, portfolio in st.session_state.portfolios.items():
            export_data[name] = {
                'tickers': portfolio['tickers'],
                'weights': portfolio['weights'],
                'start_date': portfolio['start_date'].isoformat(),
                'end_date': portfolio['end_date'].isoformat()
            }
        
        json_str = json.dumps(export_data, indent=2)
        st.sidebar.download_button(
            label="Download portfolios.json",
            data=json_str,
            file_name="alphatic_portfolios.json",
            mime="application/json"
        )


# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

# Header
st.markdown('<h1 class="main-header">Alphatic Portfolio Analyzer ✨</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Sophisticated analysis for the educated investor</p>', unsafe_allow_html=True)

# Check if portfolio exists
if not st.session_state.current_portfolio:
    st.markdown("""
        <div class="info-box">
            <h3>👋 Welcome to Alphatic Portfolio Analyzer!</h3>
            <p style="font-size: 1.1rem;">
                Get started by building your first portfolio using the sidebar on the left.
            </p>
            <p><strong>Features:</strong></p>
            <ul>
                <li>📊 Comprehensive portfolio analysis with detailed metrics</li>
                <li>🎯 Portfolio optimization (Maximum Sharpe Ratio)</li>
                <li>📈 PyFolio integration for professional-grade analytics</li>
                <li>🌡️ <strong>NEW:</strong> Market regime analysis across 5 conditions</li>
                <li>🔮 <strong>NEW:</strong> Forward-looking risk analysis with Monte Carlo</li>
                <li>💡 <strong>NEW:</strong> Educational tooltips for every metric</li>
                <li>⚖️ Multi-portfolio and benchmark comparisons</li>
            </ul>
            <p style="margin-top: 1rem; padding: 1rem; background-color: #e3f2fd; border-radius: 8px;">
                <strong>📊 Total Return Analysis:</strong> All performance metrics include dividends 
                reinvested and are adjusted for stock splits. This represents real-world total returns 
                you would achieve with a buy-and-hold strategy.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Get current portfolio
current = st.session_state.portfolios[st.session_state.current_portfolio]
portfolio_returns = current['returns']
prices = current['prices']
weights = current['weights']
tickers = current['tickers']

# Calculate metrics for current portfolio
metrics = calculate_portfolio_metrics(portfolio_returns)

# =============================================================================
# TABS STRUCTURE - 7 TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📈 Overview",
    "📊 Detailed Analysis", 
    "📬 PyFolio Analysis",
    "🌡️ Market Regimes",
    "🔮 Forward Risk",
    "⚖️ Compare Benchmarks",
    "🎯 Optimization",
    "🚦 Trading Signals",
    "📉 Technical Charts"
])

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================


# =============================================================================
# REDESIGNED OVERVIEW TAB - "INVESTING AS COOKING"
# Replace the existing Overview tab (with tab1:) with this version
# =============================================================================


# =============================================================================
# COMPLETE OVERVIEW TAB - All Metrics + Cooking Metaphor (Compact)
# Replace existing tab1 with this version
# =============================================================================

with tab1:
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin-bottom: 1.5rem;">
            <h1 style="margin: 0; font-size: 2rem;">👨‍🍳 Your Investment Kitchen</h1>
            <p style="font-size: 1rem; margin-top: 0.3rem; opacity: 0.9;">
                Goal • Ingredients • Recipe • Timing • Results
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # =============================================================================
    # SECTION 1: COMPACT KEY INFO (4 columns, tighter spacing)
    # =============================================================================
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate values
    start_date = pd.to_datetime(current['start_date'])
    end_date = pd.to_datetime(current['end_date'])
    days_invested = (end_date - start_date).days
    years_invested = days_invested / 365.25
    total_return = metrics['Total Return']
    final_value = 100000 * (1 + total_return)
    volatility = metrics['Annual Volatility']
    sharpe = metrics['Sharpe Ratio']
    
    # Overall quality
    if sharpe > 1.5:
        quality = "Excellent"
        quality_emoji = "⭐⭐⭐⭐⭐"
        quality_color = "#28a745"
    elif sharpe > 1.0:
        quality = "Very Good"
        quality_emoji = "⭐⭐⭐⭐"
        quality_color = "#20c997"
    elif sharpe > 0.5:
        quality = "Good"
        quality_emoji = "⭐⭐⭐"
        quality_color = "#ffc107"
    elif sharpe > 0:
        quality = "Fair"
        quality_emoji = "⭐⭐"
        quality_color = "#fd7e14"
    else:
        quality = "Needs Work"
        quality_emoji = "⭐"
        quality_color = "#dc3545"
    
    # Risk profile
    if volatility < 0.10:
        risk_profile = "Conservative"
        risk_emoji = "🛡️"
        risk_color = "#28a745"
    elif volatility < 0.15:
        risk_profile = "Moderate"
        risk_emoji = "🎯"
        risk_color = "#ffc107"
    elif volatility < 0.20:
        risk_profile = "Aggressive"
        risk_emoji = "🚀"
        risk_color = "#fd7e14"
    else:
        risk_profile = "Very Aggressive"
        risk_emoji = "⚡"
        risk_color = "#dc3545"
    
    with col1:
        st.markdown(f"""
            <div style="background: {quality_color}; color: white; padding: 0.8rem; border-radius: 8px; text-align: center;">
                <h5 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Quality</h5>
                <h3 style="margin: 0.2rem 0; font-size: 1.2rem;">{quality_emoji}</h3>
                <p style="margin: 0; font-size: 0.85rem; font-weight: bold;">{quality}</p>
                <p style="margin: 0; font-size: 0.7rem; opacity: 0.9;">Sharpe: {sharpe:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="background: white; padding: 0.8rem; border-radius: 8px; border-left: 4px solid #667eea;">
                <h5 style="margin: 0; font-size: 0.8rem; color: #666;">⏱️ Time Period</h5>
                <h3 style="margin: 0.2rem 0; color: #667eea; font-size: 1.3rem;">{years_invested:.1f} yrs</h3>
                <p style="margin: 0; font-size: 0.75rem; color: #888;">{start_date.strftime('%b %Y')} → {end_date.strftime('%b %Y')}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div style="background: white; padding: 0.8rem; border-radius: 8px; border-left: 4px solid #28a745;">
                <h5 style="margin: 0; font-size: 0.8rem; color: #666;">💰 $100k Grows To</h5>
                <h3 style="margin: 0.2rem 0; color: #28a745; font-size: 1.3rem;">${final_value:,.0f}</h3>
                <p style="margin: 0; font-size: 0.75rem; color: #888;">{metrics['Annual Return']*100:.1f}% annual</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div style="background: {risk_color}; color: white; padding: 0.8rem; border-radius: 8px; text-align: center;">
                <h5 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Risk Level</h5>
                <h3 style="margin: 0.2rem 0; font-size: 1.5rem;">{risk_emoji}</h3>
                <p style="margin: 0; font-size: 0.85rem; font-weight: bold;">{risk_profile}</p>
                <p style="margin: 0; font-size: 0.7rem; opacity: 0.9;">{volatility*100:.1f}% vol</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =============================================================================
    # SECTION 2: PORTFOLIO COMPOSITION
    # =============================================================================
    st.markdown("## 🥘 Your Ingredients")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Build enhanced composition table
        ingredients_data = []
        for ticker in weights.keys():
            weight = weights[ticker]
            
            if ticker in prices.columns:
                signal_data = generate_trading_signal(prices[ticker])
                action = signal_data['action']
                
                ticker_returns = prices[ticker].pct_change().dropna()
                ticker_annual_return = (1 + ticker_returns.mean()) ** 252 - 1
                
                # Categorize
                if ticker in ['SPY', 'VTI', 'QQQ', 'VOO', 'VUG']:
                    ingredient_type = "🥩 Core"
                elif ticker in ['AGG', 'BND', 'TLT', 'IEF', 'SHY']:
                    ingredient_type = "🥗 Bonds"
                elif ticker in ['VEA', 'VWO', 'EFA', 'IEMG', 'VXUS']:
                    ingredient_type = "🌶️ Intl"
                elif ticker in ['GLD', 'IAU']:
                    ingredient_type = "🧂 Gold"
                elif ticker in ['VYM', 'SCHD', 'DVY']:
                    ingredient_type = "💰 Dividend"
                else:
                    ingredient_type = "🥄 Other"
                
                ingredients_data.append({
                    'Ticker': ticker,
                    'Type': ingredient_type,
                    'Weight': f"{weight*100:.1f}%",
                    'Return': f"{ticker_annual_return*100:+.1f}%",
                    'Action': action
                })
        
        ingredients_df = pd.DataFrame(ingredients_data)
        
        def style_action(val):
            if val == 'Accumulate':
                return 'background-color: #d4edda; color: #155724; font-weight: bold'
            elif val == 'Distribute':
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
            elif val == 'Hold':
                return 'background-color: #fff3cd; color: #856404; font-weight: bold'
            return ''
        
        styled_ingredients = ingredients_df.style.applymap(style_action, subset=['Action'])
        st.dataframe(styled_ingredients, use_container_width=True, hide_index=True)

        # Ingredient Guide
        with st.expander("🧾 Ingredient Guide - What Each Type Does"):
            st.markdown("""
            **🥩 Main Course (Core Growth)** - Large-cap stocks (SPY, VTI, QQQ)  
            → Provides primary growth. Like the protein in your meal.
            
            **🥗 Stabilizer (Bonds)** - Fixed income (AGG, BND, TLT)  
            → Reduces volatility, provides steady income. Like vegetables that balance the meal.
            
            **🌶️ Spice (International)** - Foreign stocks (VEA, VWO, EFA)  
            → Adds diversification and growth from other economies. Enhances flavor.
            
            **🧂 Preservative (Gold)** - Precious metals (GLD, IAU)  
            → Inflation hedge, crisis insurance. Preserves value when market sours.
            
            **🥄 Specialty** - Sector-specific or thematic ETFs  
            → Targeted exposure to specific themes. Special seasoning.
            """)
    
    with col2:

        # st.markdown("### 🥧 Ingredient Proportions")

        fig, ax = plt.subplots(figsize=(6, 6))
        colors = plt.cm.Set3(range(len(weights)))
        wedges, texts, autotexts = ax.pie(
            weights.values(), 
            labels=weights.keys(), 
            autopct='%1.1f%%',
            colors=colors, 
            startangle=90,
            textprops={'fontsize': 9}
        )
        ax.set_title('### 🥧 Ingredient Proportions', fontsize=12, fontweight='bold', pad=10)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # =============================================================================
    # SECTION 3: ALL PERFORMANCE METRICS
    # =============================================================================
    st.markdown("### 🎯 Performance Metrics vs S&P 500")
    
    # Calculate SPY
    try:
        spy_data = download_ticker_data(['SPY'], current['start_date'], current['end_date'])
        if spy_data is not None:
            spy_returns = spy_data.pct_change().dropna()
            spy_metrics = calculate_portfolio_metrics(spy_returns)
        else:
            spy_metrics = None
    except:
        spy_metrics = None
    
    def get_comparison_indicator(portfolio_value, spy_value, metric_type='higher_better'):
        if spy_metrics is None:
            return "", "white"
        if metric_type == 'higher_better':
            if portfolio_value > spy_value:
                return "🟢", "#28a745"
            elif portfolio_value < spy_value:
                return "🔴", "#dc3545"
            else:
                return "⚪", "#ffc107"
        else:
            if portfolio_value < spy_value:
                return "🟢", "#28a745"
            elif portfolio_value > spy_value:
                return "🔴", "#dc3545"
            else:
                return "⚪", "#ffc107"
    
    # First row: Core metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_class = get_metric_color_class('annual_return', metrics['Annual Return'])
        arrow, color = get_comparison_indicator(metrics['Annual Return'], 
                                               spy_metrics['Annual Return'] if spy_metrics else 0, 
                                               'higher_better')
        st.markdown(f"""
            <div class="{metric_class}">
                <h4>Annual Return {arrow}</h4>
                <h2>{metrics['Annual Return']:.2%}</h2>
                <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Annual Return']:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
        render_metric_explanation('annual_return')
    
    with col2:
        metric_class = get_metric_color_class('sharpe_ratio', metrics['Sharpe Ratio'])
        arrow, color = get_comparison_indicator(metrics['Sharpe Ratio'], 
                                               spy_metrics['Sharpe Ratio'] if spy_metrics else 0, 
                                               'higher_better')
        st.markdown(f"""
            <div class="{metric_class}">
                <h4>Sharpe Ratio {arrow}</h4>
                <h2>{metrics['Sharpe Ratio']:.2f}</h2>
                <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Sharpe Ratio']:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        render_metric_explanation('sharpe_ratio')
    
    with col3:
        metric_class = get_metric_color_class('max_drawdown', metrics['Max Drawdown'])
        arrow, color = get_comparison_indicator(metrics['Max Drawdown'], 
                                               spy_metrics['Max Drawdown'] if spy_metrics else 0, 
                                               'lower_better')
        st.markdown(f"""
            <div class="{metric_class}">
                <h4>Max Drawdown {arrow}</h4>
                <h2>{metrics['Max Drawdown']:.2%}</h2>
                <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Max Drawdown']:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
        render_metric_explanation('max_drawdown')
    
    with col4:
        metric_class = get_metric_color_class('volatility', metrics['Annual Volatility'])
        arrow, color = get_comparison_indicator(metrics['Annual Volatility'], 
                                               spy_metrics['Annual Volatility'] if spy_metrics else 0, 
                                               'lower_better')
        st.markdown(f"""
            <div class="{metric_class}">
                <h4>Volatility {arrow}</h4>
                <h2>{metrics['Annual Volatility']:.2%}</h2>
                <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Annual Volatility']:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
        render_metric_explanation('volatility')
    
    # Second row: Additional metrics
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_class = get_metric_color_class('sortino_ratio', metrics['Sortino Ratio'])
        arrow, color = get_comparison_indicator(metrics['Sortino Ratio'], 
                                               spy_metrics['Sortino Ratio'] if spy_metrics else 0, 
                                               'higher_better')
        st.markdown(f"""
            <div class="{metric_class}">
                <h4>Sortino Ratio {arrow}</h4>
                <h2>{metrics['Sortino Ratio']:.2f}</h2>
                <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Sortino Ratio']:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        render_metric_explanation('sortino_ratio')
    
    with col2:
        metric_class = get_metric_color_class('calmar_ratio', metrics['Calmar Ratio'])
        arrow, color = get_comparison_indicator(metrics['Calmar Ratio'], 
                                               spy_metrics['Calmar Ratio'] if spy_metrics else 0, 
                                               'higher_better')
        st.markdown(f"""
            <div class="{metric_class}">
                <h4>Calmar Ratio {arrow}</h4>
                <h2>{metrics['Calmar Ratio']:.2f}</h2>
                <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Calmar Ratio']:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        render_metric_explanation('calmar_ratio')
    
    with col3:
        metric_class = get_metric_color_class('win_rate', metrics['Win Rate'])
        arrow, color = get_comparison_indicator(metrics['Win Rate'], 
                                               spy_metrics['Win Rate'] if spy_metrics else 0, 
                                               'higher_better')
        st.markdown(f"""
            <div class="{metric_class}">
                <h4>Win Rate {arrow}</h4>
                <h2>{metrics['Win Rate']:.2%}</h2>
                <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Win Rate']:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
        render_metric_explanation('win_rate')
    
    with col4:
        arrow, color = get_comparison_indicator(metrics['Total Return'], 
                                               spy_metrics['Total Return'] if spy_metrics else 0, 
                                               'higher_better')
        st.markdown(f"""
            <div class="metric-card">
                <h4>Total Return {arrow}</h4>
                <h2>{metrics['Total Return']:.2%}</h2>
                <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Total Return']:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Legend
    st.markdown("""
        <div style="text-align: center; padding: 8px; margin-top: 8px; background: #f8f9fa; border-radius: 5px;">
            <small><strong>Legend:</strong> 🟢 = Better | 🔴 = Worse | ⚪ = Equal vs SPY</small>
        </div>
    """, unsafe_allow_html=True)
    
    # Performance Chart
    st.markdown("---")
    st.markdown("### 📈 Performance Over Time")
    fig = plot_cumulative_returns(portfolio_returns, f'{st.session_state.current_portfolio} - Cumulative Returns')
    st.pyplot(fig)
    
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 What This Chart Means</div>
            <p><strong>How to Read:</strong> Shows how $1 grows over time. Value of 1.5 = 50% gain.</p>
            <p><strong>Look For:</strong> Steady upward = good. Sharp drops = drawdowns. Flat = stagnation.</p>
            <p><strong>Action:</strong> 6+ months downtrend = review strategy.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Drawdown Chart
    st.markdown("---")
    st.markdown("### 📉 Drawdown Analysis")
    fig = plot_drawdown(portfolio_returns, 'Portfolio Drawdown')
    st.pyplot(fig)
    
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 Understanding Drawdowns</div>
            <p><strong>What This Shows:</strong> How much underwater from peak value.</p>
            <p><strong>Red Flag:</strong> >20% drawdown = bear market. Don't panic-sell!</p>
            <p><strong>Psychology Check:</strong> Can you handle the max drawdown without selling?</p>
        </div>
    """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: DETAILED ANALYSIS
# =============================================================================

with tab2:
    st.markdown("## 📊 Detailed Analysis")
    
    # Monthly Returns Heatmap
    st.markdown("### 📅 Monthly Returns Heatmap")
    fig = plot_monthly_returns_heatmap(portfolio_returns, 'Monthly Returns (%)')
    st.pyplot(fig)
    
    # Heatmap interpretation
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 How to Use This Heatmap</div>
            <p><strong>What This Shows:</strong> Each cell shows the return for that month. 
            Green = gains, Red = losses.</p>
            <p><strong>Patterns to Look For:</strong></p>
            <ul>
                <li>Seasonal trends: Some months consistently better/worse?</li>
                <li>Streaks: 3+ consecutive red months = review needed</li>
                <li>Year comparisons: Are recent years better or worse than historical?</li>
            </ul>
            <p><strong>Red Flags:</strong></p>
            <ul>
                <li>Entire rows of red (bad years - what happened?)</li>
                <li>Consistent December losses (tax-loss harvesting season)</li>
                <li>Recent months all red (time to re-evaluate strategy)</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Monthly Income/Gains Table
    st.markdown("---")
    st.markdown("### 💰 Monthly Income Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Calculate dollar gains/losses per month based on portfolio value**")
    
    with col2:
        initial_capital = st.number_input(
            "Initial Portfolio Value ($)", 
            min_value=1000, 
            max_value=100000000, 
            value=100000, 
            step=10000,
            help="Enter your starting portfolio value to see dollar gains/losses"
        )
    
    # Calculate monthly dollar gains with dividend breakdown
    returns_series = portfolio_returns if isinstance(portfolio_returns, pd.Series) else portfolio_returns.iloc[:, 0]
    monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Estimate dividend component (approximate - based on typical dividend yields)
    # For more accuracy, would need separate dividend data
    # Rough estimate: ~2% annual dividend yield for typical stock portfolio
    # Distributed across months based on return
    monthly_data = []
    cumulative_value = initial_capital
    annual_dividend_yield = 0.018  # Approximate 1.8% annual yield for diversified portfolio
    monthly_dividend_rate = annual_dividend_yield / 12
    
    for date, monthly_return in monthly_returns.items():
        month_start_value = cumulative_value
        
        # Estimate dividend portion (rough approximation)
        # Dividends are roughly consistent, capital gains vary
        estimated_dividend = month_start_value * monthly_dividend_rate
        
        # Total dollar gain
        total_dollar_gain = month_start_value * monthly_return
        
        # Capital gain = Total gain - Dividends
        capital_gain = total_dollar_gain - estimated_dividend
        
        # Update cumulative value
        cumulative_value = month_start_value + total_dollar_gain
        
        monthly_data.append({
            'Date': date.strftime('%Y-%m'),
            'Month': date.strftime('%B'),
            'Year': date.year,
            'Return %': monthly_return * 100,
            'Total Gain/Loss': total_dollar_gain,
            'Capital Gain/Loss': capital_gain,
            'Dividend Income': estimated_dividend,
            'Portfolio Value': cumulative_value
        })
    
    monthly_df = pd.DataFrame(monthly_data)
    
    # Add note about dividend estimation
    st.info("""
        **📊 Dividend Estimation:**  
        Dividends are estimated at ~1.8% annually (0.15% monthly) based on typical portfolio yields.  
        For exact dividend amounts, you would need dividend-specific data from your broker.  
        Capital gains = Total gains minus estimated dividends.
    """)
    
    # Display options
    view_option = st.radio(
        "View:",
        ["Last 12 Months", "Current Year", "All Time", "By Year"],
        horizontal=True
    )
    
    if view_option == "Last 12 Months":
        display_df = monthly_df.tail(12).copy()
    elif view_option == "Current Year":
        current_year = datetime.now().year
        display_df = monthly_df[monthly_df['Year'] == current_year].copy()
    elif view_option == "By Year":
        selected_year = st.selectbox("Select Year:", sorted(monthly_df['Year'].unique(), reverse=True))
        display_df = monthly_df[monthly_df['Year'] == selected_year].copy()
    else:  # All Time
        display_df = monthly_df.copy()
    
    # Format for display
    display_df['Return %'] = display_df['Return %'].apply(lambda x: f"{x:+.2f}%")
    display_df['Total Gain/Loss'] = display_df['Total Gain/Loss'].apply(lambda x: f"${x:+,.2f}")
    display_df['Capital Gain/Loss'] = display_df['Capital Gain/Loss'].apply(lambda x: f"${x:+,.2f}")
    display_df['Dividend Income'] = display_df['Dividend Income'].apply(lambda x: f"${x:,.2f}")
    display_df['Portfolio Value'] = display_df['Portfolio Value'].apply(lambda x: f"${x:,.2f}")
    
    st.dataframe(
        display_df[['Date', 'Month', 'Return %', 'Capital Gain/Loss', 'Dividend Income', 'Total Gain/Loss', 'Portfolio Value']],
        use_container_width=True,
        hide_index=True
    )
    
    # Summary statistics with dividend breakdown
    st.markdown("#### 📊 Income Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_gain = monthly_df['Total Gain/Loss'].sum()
    total_dividends = monthly_df['Dividend Income'].sum()
    total_capital_gains = monthly_df['Capital Gain/Loss'].sum()
    positive_months = (monthly_df['Total Gain/Loss'] > 0).sum()
    negative_months = (monthly_df['Total Gain/Loss'] < 0).sum()
    avg_monthly_gain = monthly_df['Total Gain/Loss'].mean()
    
    with col1:
        st.metric(
            "Total Gain/Loss",
            f"${total_gain:,.2f}",
            f"{((cumulative_value - initial_capital) / initial_capital * 100):+.2f}%"
        )
    
    with col2:
        st.metric(
            "Total Dividends",
            f"${total_dividends:,.2f}",
            f"{(total_dividends / total_gain * 100 if total_gain > 0 else 0):.1f}% of total"
        )
    
    with col3:
        st.metric(
            "Capital Gains",
            f"${total_capital_gains:,.2f}",
            f"{(total_capital_gains / total_gain * 100 if total_gain > 0 else 0):.1f}% of total"
        )
    
    with col4:
        st.metric(
            "Positive Months",
            f"{positive_months}",
            f"{positive_months / len(monthly_df) * 100:.1f}%"
        )
    
    with col5:
        st.metric(
            "Avg Monthly Gain",
            f"${avg_monthly_gain:,.2f}"
        )
    
    # Tax planning insights with dividend focus
    st.markdown("---")
    st.info("""
        **💡 Tax Planning Tips (Capital Gains vs Dividends):**
        
        **Dividends:**
        - **Qualified dividends**: 0%, 15%, or 20% tax rate (held >60 days)
        - **Ordinary dividends**: Taxed as ordinary income (10-37%)
        - **Steady income**: Dividends provide consistent monthly income
        - **Tax efficient**: Qualified dividends taxed lower than wages
        
        **Capital Gains:**
        - **Short-term** (held <1 year): Taxed as ordinary income (10-37%)
        - **Long-term** (held >1 year): Lower rates (0%, 15%, or 20%)
        - **Tax-loss harvesting**: Negative months can offset gains
        - **Wash sale rule**: Can't repurchase same security within 30 days
        
        **Strategy Tips:**
        - Hold dividend stocks in tax-advantaged accounts (401k, IRA) to defer taxes
        - Harvest losses in taxable accounts to offset capital gains
        - In retirement, qualified dividends are tax-efficient income source
        - **Consult a CPA**: This is for planning only - not tax advice!
    """)
    
    # Monthly income interpretation
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 How to Use Monthly Income Data</div>
            <p><strong>For Retirement Planning:</strong></p>
            <ul>
                <li>Look at average monthly gain - is it enough to live on?</li>
                <li>Check volatility - can you handle the negative months?</li>
                <li>Win rate above 60% = more consistent income</li>
            </ul>
            <p><strong>For Tax Planning:</strong></p>
            <ul>
                <li>December losses? Good time to harvest for tax deduction</li>
                <li>Big gains in one month? Might push you into higher bracket</li>
                <li>Spread gains over multiple years if possible</li>
            </ul>
            <p><strong>For Strategy Evaluation:</strong></p>
            <ul>
                <li>Are monthly gains getting bigger or smaller over time?</li>
                <li>Do gains cluster in certain months (seasonality)?</li>
                <li>Can you emotionally handle the worst months?</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Rolling Metrics
    st.markdown("---")
    st.markdown("### 📈 Rolling Risk-Adjusted Performance")
    window = st.slider("Rolling Window (days)", min_value=20, max_value=252, value=60, step=10)
    fig = plot_rolling_metrics(portfolio_returns, window=window)
    st.pyplot(fig)
    
    # Rolling metrics interpretation
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 Understanding Rolling Metrics</div>
            <p><strong>What This Shows:</strong> How your risk-adjusted performance changes over time.</p>
            <p><strong>Sharpe Ratio:</strong> Measures returns vs ALL volatility</p>
            <ul>
                <li>Above 1.0 (green line) = Good risk-adjusted returns</li>
                <li>Consistently above 1.0 = Sustainable strategy</li>
                <li>Dropping toward 0 = Strategy losing effectiveness</li>
            </ul>
            <p><strong>Sortino Ratio:</strong> Measures returns vs DOWNSIDE volatility only</p>
            <ul>
                <li>Higher than Sharpe = Good! Means upside volatility is high</li>
                <li>Much lower than Sharpe = Too many down days</li>
            </ul>
            <p><strong>Action Items:</strong></p>
            <ul>
                <li>If both metrics trend down for 3+ months, consider rebalancing</li>
                <li>Sudden spikes after crashes = good recovery</li>
                <li>Steady improvement = strategy working</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Distribution Analysis
    st.markdown("---")
    st.markdown("### 📊 Returns Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        portfolio_returns.hist(bins=50, ax=ax, color='#667eea', alpha=0.7, edgecolor='black')
        ax.axvline(portfolio_returns.mean(), color='#28a745', linestyle='--', 
                   linewidth=2, label=f'Mean: {portfolio_returns.mean():.4f}')
        ax.axvline(portfolio_returns.median(), color='#ffc107', linestyle='--', 
                   linewidth=2, label=f'Median: {portfolio_returns.median():.4f}')
        ax.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Daily Return', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    with col2:
        # QQ Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(portfolio_returns.dropna(), dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normal Distribution Test)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    # Distribution interpretation
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 What Distribution Analysis Tells You</div>
            <p><strong>Histogram (Left):</strong></p>
            <ul>
                <li>Centered around 0? Good, means positive and negative days balance</li>
                <li>Long left tail (fat negative side)? Portfolio has crash risk</li>
                <li>Long right tail (fat positive side)? Portfolio captures big gains</li>
            </ul>
            <p><strong>Q-Q Plot (Right):</strong></p>
            <ul>
                <li>Points follow red line closely? Returns are "normal" (predictable)</li>
                <li>Points curve away at ends? "Fat tails" = more extreme events than expected</li>
                <li>Lower-left points below line? More severe crashes than normal distribution predicts</li>
            </ul>
            <p><strong>Why It Matters:</strong> Standard risk models assume normal distribution. 
            If your returns aren't normal, you might have more risk than you think!</p>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 3: PYFOLIO COMPREHENSIVE ANALYSIS
# =============================================================================

with tab3:
    st.markdown("## 📬 PyFolio Professional Analysis")
    
    # What is PyFolio section
    st.markdown("""
        <div class="info-box">
            <h3>🎓 What is PyFolio?</h3>
            <p><strong>PyFolio is the institutional-grade analytics library used by hedge funds, 
            asset managers, and professional traders.</strong></p>
            <p><strong>Created by Quantopian</strong> (a professional quant hedge fund platform), 
            PyFolio is the SAME tool used by:</p>
            <ul>
                <li>📊 Hedge fund managers to evaluate their strategies</li>
                <li>💼 Institutional investors to analyze fund performance</li>
                <li>🏦 Asset management firms for client reporting</li>
                <li>📈 Quantitative researchers for strategy validation</li>
            </ul>
            <p><strong>Why is this powerful?</strong> You're getting the EXACT same analytics 
            that professional money managers pay thousands for. This is not "investor-lite" – 
            this is the real deal.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # PyFolio vs Detailed Analysis
    st.markdown("---")
    st.markdown("### 🔬 PyFolio vs. Detailed Analysis Tab")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h4>📊 Detailed Analysis Tab</h4>
                <p><strong>Focus:</strong> Easy-to-understand metrics</p>
                <p><strong>Best For:</strong></p>
                <ul>
                    <li>Quick performance check</li>
                    <li>Understanding basic patterns</li>
                    <li>Educational tooltips</li>
                    <li>Non-expert friendly</li>
                </ul>
                <p><strong>Metrics:</strong> Standard risk/return metrics with explanations</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card" style="border-left: 5px solid #764ba2;">
                <h4>📬 PyFolio Analysis Tab</h4>
                <p><strong>Focus:</strong> Professional validation</p>
                <p><strong>Best For:</strong></p>
                <ul>
                    <li>Comparing to professionals</li>
                    <li>Institutional-grade reporting</li>
                    <li>Deep statistical analysis</li>
                    <li>Due diligence on strategies</li>
                </ul>
                <p><strong>Metrics:</strong> Comprehensive tear sheets used by hedge funds</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 When to Use Each Tab</div>
            <p><strong>Use Detailed Analysis when:</strong></p>
            <ul>
                <li>You want quick, easy-to-understand insights</li>
                <li>You're learning about portfolio metrics</li>
                <li>You need to make a quick decision</li>
                <li>You want clear action items</li>
            </ul>
            <p><strong>Use PyFolio Analysis when:</strong></p>
            <ul>
                <li>You want to validate your strategy like a professional</li>
                <li>You're comparing your performance to fund managers</li>
                <li>You need comprehensive statistics for serious money decisions</li>
                <li>You want to see if your strategy has institutional-quality metrics</li>
                <li>You're presenting performance to sophisticated investors (family office, etc.)</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # What PyFolio Adds
    st.markdown("---")
    st.markdown("### 🎯 What PyFolio Adds Beyond Basic Analysis")
    
    st.markdown("""
        <div class="success-box">
            <h4>📊 Unique PyFolio Features:</h4>
            <ol>
                <li><strong>Rolling Beta & Sharpe:</strong> See how your market exposure changes over time</li>
                <li><strong>Rolling Volatility:</strong> Track when your strategy gets risky</li>
                <li><strong>Top Drawdown Periods:</strong> Identify your worst periods with exact dates</li>
                <li><strong>Underwater Plot:</strong> Visualize how long you stayed in drawdown</li>
                <li><strong>Monthly & Annual Returns Table:</strong> Complete historical breakdown</li>
                <li><strong>Distribution Analysis:</strong> Advanced statistical validation</li>
                <li><strong>Worst Drawdown Timing:</strong> Understand when pain happens</li>
            </ol>
            <p style="margin-top: 1rem;"><strong>The Bottom Line:</strong> PyFolio tells you if your 
            strategy would pass institutional due diligence. If hedge funds would invest in your 
            strategy, PyFolio will show it. If they wouldn't, PyFolio will reveal why.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Practical Decision Making Guide
    st.markdown("---")
    st.markdown("### 🎓 How to Use PyFolio for Real Portfolio Decisions")
    
    st.markdown("#### 💼 Real-World Decision Framework")
    
    # Scenario 1
    st.markdown("**Scenario 1: Should I Keep This Strategy?**")
    st.markdown("**Look for:**")
    st.markdown("""
    - **Rolling Sharpe Ratio:** Is it consistently above 0.5? Good sign.
    - **Drawdown Periods:** Do you recover within 6-12 months? Acceptable.
    - **Annual Returns Table:** More green than red years? Keep going.
    """)
    st.markdown("**Red Flags:**")
    st.markdown("""
    - Rolling Sharpe consistently below 0.3 → Strategy isn't working
    - Drawdowns last 2+ years → Too slow to recover
    - More losing years than winning years → Fundamental problem
    """)
    
    # Scenario 2
    st.markdown("**Scenario 2: Is My Strategy Better Than Just Buying SPY?**")
    st.markdown("**Look for:**")
    st.markdown("""
    - **Compare Rolling Sharpe to SPY:** Are you consistently higher? Yes = Worth it.
    - **Check Worst Drawdowns:** Are yours shallower than SPY's -30% to -50%? Good!
    - **Recovery Time:** Do you bounce back faster than SPY? Excellent.
    """)
    st.markdown("**Decision Rule:**")
    st.markdown("""
    - If Rolling Sharpe less than SPY for 2+ years → Just buy SPY (simpler, cheaper)
    - If max drawdown worse than SPY but returns aren't higher → Just buy SPY
    - If you beat SPY on risk-adjusted basis → Keep your strategy!
    """)
    
    # Scenario 3
    st.markdown("**Scenario 3: Can I Handle More Risk?**")
    st.markdown("**Look for:**")
    st.markdown("""
    - **Underwater Plot:** How long were you "underwater" (below peak)?
    - **Top 5 Drawdowns:** Look at duration (days underwater)
    - **Rolling Volatility:** Is it stable or spiky?
    """)
    st.markdown("**Decision Framework:**")
    st.markdown("""
    - If typical drawdown recovery is less than 6 months → You have capacity for more risk
    - If rolling volatility is very stable → Can add more aggressive positions
    - If you're never underwater more than 1 year → Portfolio is quite conservative
    """)
    
    # Scenario 4
    st.markdown("**Scenario 4: Presenting Performance to Financial Advisor**")
    st.markdown("**Your advisor will look at:**")
    st.markdown("""
    - **Cumulative Returns vs Drawdown:** Shows risk-adjusted growth
    - **Rolling Metrics:** Proves consistency, not luck
    - **Worst Drawdown Periods:** Shows you survived crises
    - **Annual Returns Table:** Detailed historical track record
    """)
    st.markdown("**What impresses advisors:**")
    st.markdown("""
    - Positive Sharpe in 2008, 2020, 2022 (crisis years)
    - Consistent rolling Sharpe above 1.0
    - Maximum drawdown less than 25%
    - Fast recovery from drawdowns (under 12 months)
    """)
    
    # Key Metrics to Watch
    st.markdown("---")
    st.markdown("### 📋 PyFolio Metrics Decoder")
    
    with st.expander("📊 Complete Guide to Reading PyFolio Output"):
        st.markdown("""
            <h4>Section 1: Cumulative Returns</h4>
            <ul>
                <li><strong>What it shows:</strong> Portfolio value over time (normalized to start at 1.0)</li>
                <li><strong>Look for:</strong> Steady upward trend with controlled drawdowns</li>
                <li><strong>Red flag:</strong> Long flat periods or severe drops</li>
            </ul>
            
            <h4>Section 2: Rolling Sharpe (6-month)</h4>
            <ul>
                <li><strong>What it shows:</strong> Risk-adjusted returns over time</li>
                <li><strong>Look for:</strong> Line consistently above 0.5, ideally above 1.0</li>
                <li><strong>Red flag:</strong> Frequent dips below 0 (negative risk-adjusted returns)</li>
                <li><strong>Pro tip:</strong> If this trends down over time, your strategy is degrading</li>
            </ul>
            
            <h4>Section 3: Rolling Beta</h4>
            <ul>
                <li><strong>What it shows:</strong> How much your portfolio moves with the market</li>
                <li><strong>Look for:</strong> Stability (beta doesn't swing wildly)</li>
                <li><strong>Interpretation:</strong> 
                    <ul>
                        <li>Beta increasing over time = Taking more market risk</li>
                        <li>Beta decreasing = Becoming more defensive</li>
                        <li>Stable beta = Consistent strategy</li>
                    </ul>
                </li>
            </ul>
            
            <h4>Section 4: Rolling Volatility</h4>
            <ul>
                <li><strong>What it shows:</strong> How much your returns fluctuate</li>
                <li><strong>Look for:</strong> Stable line, spikes during known crisis periods only</li>
                <li><strong>Red flag:</strong> Volatility increasing over time = Strategy becoming riskier</li>
            </ul>
            
            <h4>Section 5: Top 5 Drawdown Periods</h4>
            <ul>
                <li><strong>What it shows:</strong> Your worst losing periods with exact dates</li>
                <li><strong>Look for:</strong> 
                    <ul>
                        <li>Drawdowns aligning with known crises (2008, 2020, 2022) = Expected</li>
                        <li>Recovery time < 12 months = Good resilience</li>
                    </ul>
                </li>
                <li><strong>Red flag:</strong> 
                    <ul>
                        <li>Drawdowns during bull markets = Strategy problem</li>
                        <li>Recovery time > 24 months = Very painful</li>
                    </ul>
                </li>
            </ul>
            
            <h4>Section 6: Underwater Plot</h4>
            <ul>
                <li><strong>What it shows:</strong> How far below your peak you are at any time</li>
                <li><strong>How to read:</strong> 
                    <ul>
                        <li>0% = At new peak (best possible)</li>
                        <li>-20% = 20% below your previous high</li>
                    </ul>
                </li>
                <li><strong>Look for:</strong> Frequent returns to 0% (making new highs)</li>
                <li><strong>Red flag:</strong> Long periods deep underwater = Slow recovery</li>
            </ul>
            
            <h4>Section 7: Monthly Returns (%)</h4>
            <ul>
                <li><strong>What it shows:</strong> Returns for every month, year by year</li>
                <li><strong>Look for:</strong> More green (positive) than red (negative) months</li>
                <li><strong>Pattern analysis:</strong>
                    <ul>
                        <li>Seasonal patterns? Some strategies work better certain times of year</li>
                        <li>Recent years vs early years? Is performance degrading?</li>
                        <li>Consistent bad Decembers? Could be tax-loss harvesting effect</li>
                    </ul>
                </li>
            </ul>
            
            <h4>Section 8: Annual Returns (%)</h4>
            <ul>
                <li><strong>What it shows:</strong> Total return each year</li>
                <li><strong>Look for:</strong> Majority of years positive</li>
                <li><strong>Key benchmark:</strong> 
                    <ul>
                        <li>70%+ winning years = Very good</li>
                        <li>50-70% winning years = Good</li>
                        <li>Below 50% = Questionable</li>
                    </ul>
                </li>
            </ul>
            
            <h4>Section 9: Distribution Analysis</h4>
            <ul>
                <li><strong>What it shows:</strong> Statistical properties of your returns</li>
                <li><strong>Look for:</strong> Relatively normal distribution (bell curve)</li>
                <li><strong>Red flag:</strong> 
                    <ul>
                        <li>Fat left tail = More severe crashes than expected</li>
                        <li>High kurtosis = More extreme events than normal</li>
                    </ul>
                </li>
            </ul>
        """, unsafe_allow_html=True)
    
    # Generate PyFolio Analysis
    st.markdown("---")
    st.markdown("### 📊 Portfolio Report Card")
    st.markdown("""
        **Your portfolio graded against market benchmarks.** Grading is calibrated so the S&P 500 
        earns a solid **B grade** (since SPY beats 80% of professionals long-term). Each metric shows where  you excel and where you need improvement.
        
        **Key:** A = Beating SPY significantly | B = SPY-level (excellent!) | C = Below SPY | D/F = Poor
    """)
    
    # Calculate comprehensive metrics for grading
    def calculate_all_metrics(returns, benchmark_returns=None):
        """Calculate all metrics needed for grading"""
        metrics = calculate_portfolio_metrics(returns, benchmark_returns)
        
        # Add additional metrics for grading
        returns_series = returns if isinstance(returns, pd.Series) else returns.iloc[:, 0]
        
        # Win rate
        win_rate = (returns_series > 0).sum() / len(returns_series)
        
        # Best and worst month
        monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
        best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0
        worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0
        
        # Recovery time (average days to recover from drawdown)
        cum_returns = (1 + returns_series).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            # Calculate average recovery time
            recovery_periods = []
            start_dd = None
            for i, (date, is_dd) in enumerate(in_drawdown.items()):
                if is_dd and start_dd is None:
                    start_dd = date
                elif not is_dd and start_dd is not None:
                    recovery_periods.append((date - start_dd).days)
                    start_dd = None
            avg_recovery_days = np.mean(recovery_periods) if recovery_periods else 0
        else:
            avg_recovery_days = 0
        
        return {
            'Annual Return': metrics['Annual Return'],
            'Sharpe Ratio': metrics['Sharpe Ratio'],
            'Sortino Ratio': metrics['Sortino Ratio'],
            'Max Drawdown': metrics['Max Drawdown'],
            'Volatility': metrics['Annual Volatility'],
            'Calmar Ratio': metrics['Calmar Ratio'],
            'Win Rate': win_rate,
            'Best Month': best_month,
            'Worst Month': worst_month,
            'Alpha': metrics.get('Alpha', 0),
            'Beta': metrics.get('Beta', 1),
            'Avg Recovery Days': avg_recovery_days
        }
    
    def grade_metric(metric_name, value):
        """
        Grade a metric A through F based on REALISTIC market benchmarks
        Calibrated so S&P 500 (SPY) earns a solid B grade
        
        Grading Philosophy:
        - A grade = Beating S&P 500 significantly (top 20% of all strategies)
        - B grade = S&P 500 level (market benchmark - already beats 80% of professionals!)
        - C grade = Below market but positive
        - D grade = Barely positive or slightly negative
        - F grade = Significantly negative or terrible risk-adjusted returns
        
        Returns: (grade, explanation)
        """
        grading_criteria = {
            'Annual Return': {
                'ranges': 'A: >12%, B: 8-12%, C: 4-8%, D: 0-4%, F: <0%',
                'A': (0.12, float('inf')),
                'B': (0.08, 0.12),
                'C': (0.04, 0.08),
                'D': (0.00, 0.04),
                'F': (-float('inf'), 0.00)
            },
            'Sharpe Ratio': {
                'ranges': 'A: >1.0, B: 0.5-1.0, C: 0.2-0.5, D: 0-0.2, F: <0',
                'A': (1.0, float('inf')),
                'B': (0.5, 1.0),
                'C': (0.2, 0.5),
                'D': (0.0, 0.2),
                'F': (-float('inf'), 0.0)
            },
            'Sortino Ratio': {
                'ranges': 'A: >1.5, B: 0.9-1.5, C: 0.5-0.9, D: 0.2-0.5, F: <0.2',
                'A': (1.5, float('inf')),
                'B': (0.9, 1.5),
                'C': (0.5, 0.9),
                'D': (0.2, 0.5),
                'F': (-float('inf'), 0.2)
            },
            'Max Drawdown': {
                'ranges': 'A: >-15%, B: -15% to -25%, C: -25% to -35%, D: -35% to -50%, F: <-50%',
                'A': (-0.15, 0),
                'B': (-0.25, -0.15),
                'C': (-0.35, -0.25),
                'D': (-0.50, -0.35),
                'F': (-float('inf'), -0.50)
            },
            'Volatility': {
                'ranges': 'A: <12%, B: 12-16%, C: 16-20%, D: 20-25%, F: >25%',
                'A': (0, 0.12),
                'B': (0.12, 0.16),
                'C': (0.16, 0.20),
                'D': (0.20, 0.25),
                'F': (0.25, float('inf'))
            },
            'Calmar Ratio': {
                'ranges': 'A: >1.0, B: 0.5-1.0, C: 0.25-0.5, D: 0.1-0.25, F: <0.1',
                'A': (1.0, float('inf')),
                'B': (0.5, 1.0),
                'C': (0.25, 0.5),
                'D': (0.1, 0.25),
                'F': (-float('inf'), 0.1)
            },
            'Win Rate': {
                'ranges': 'A: >60%, B: 55-60%, C: 50-55%, D: 45-50%, F: <45%',
                'A': (0.60, 1.0),
                'B': (0.55, 0.60),
                'C': (0.50, 0.55),
                'D': (0.45, 0.50),
                'F': (0, 0.45)
            },
            'Best Month': {
                'ranges': 'A: >12%, B: 8-12%, C: 4-8%, D: 1-4%, F: <1%',
                'A': (0.12, float('inf')),
                'B': (0.08, 0.12),
                'C': (0.04, 0.08),
                'D': (0.01, 0.04),
                'F': (-float('inf'), 0.01)
            },
            'Worst Month': {
                'ranges': 'A: >-8%, B: -8% to -12%, C: -12% to -16%, D: -16% to -20%, F: <-20%',
                'A': (-0.08, 0),
                'B': (-0.12, -0.08),
                'C': (-0.16, -0.12),
                'D': (-0.20, -0.16),
                'F': (-float('inf'), -0.20)
            },
            'Alpha': {
                'ranges': 'A: >2%, B: 0.5-2%, C: -0.5% to 0.5%, D: -2% to -0.5%, F: <-2%',
                'A': (0.02, float('inf')),
                'B': (0.005, 0.02),
                'C': (-0.005, 0.005),
                'D': (-0.02, -0.005),
                'F': (-float('inf'), -0.02)
            },
            'Beta': {
                'ranges': 'A: 0.85-1.15, B: 0.7-0.85 or 1.15-1.3, C: 0.5-0.7 or 1.3-1.5, D: 0.3-0.5 or 1.5-1.7, F: <0.3 or >1.7',
                'A': [(0.85, 1.15)],
                'B': [(0.7, 0.85), (1.15, 1.3)],
                'C': [(0.5, 0.7), (1.3, 1.5)],
                'D': [(0.3, 0.5), (1.5, 1.7)],
                'F': [(0, 0.3), (1.7, float('inf'))]
            },
            'Avg Recovery Days': {
                'ranges': 'A: <120 days, B: 120-240 days, C: 240-365 days, D: 365-540 days, F: >540 days',
                'A': (0, 120),
                'B': (120, 240),
                'C': (240, 365),
                'D': (365, 540),
                'F': (540, float('inf'))
            }
        }
        
        if metric_name not in grading_criteria:
            return 'N/A', grading_criteria.get(metric_name, {}).get('ranges', 'N/A')
        
        criteria = grading_criteria[metric_name]
        ranges_explanation = criteria['ranges']
        
        # Special handling for Beta (multiple ranges per grade)
        if metric_name == 'Beta':
            for grade in ['A', 'B', 'C', 'D', 'F']:
                for low, high in criteria[grade]:
                    if low <= value < high:
                        return grade, ranges_explanation
            return 'F', ranges_explanation
        
        # Standard handling for other metrics
        for grade in ['A', 'B', 'C', 'D', 'F']:
            low, high = criteria[grade]
            if low <= value < high:
                return grade, ranges_explanation
        
        return 'F', ranges_explanation
    
    def calculate_overall_grade(grades):
        """
        Calculate overall grade with weighting (hedge fund emphasis)
        
        Weighting:
        - Sharpe Ratio: 25% (most important - risk-adjusted return)
        - Alpha: 20% (value added vs benchmark)
        - Max Drawdown: 15% (downside protection)
        - Annual Return: 15% (absolute performance)
        - Sortino Ratio: 10% (downside risk)
        - Calmar Ratio: 5%
        - Volatility: 5%
        - Win Rate: 3%
        - Beta: 2%
        - Others: 5% combined
        """
        grade_points = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0, 'N/A': 2.0}
        
        weights = {
            'Sharpe Ratio': 0.25,
            'Alpha': 0.20,
            'Max Drawdown': 0.15,
            'Annual Return': 0.15,
            'Sortino Ratio': 0.10,
            'Calmar Ratio': 0.05,
            'Volatility': 0.05,
            'Win Rate': 0.03,
            'Beta': 0.02
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for metric, grade in grades.items():
            weight = weights.get(metric, 0.005)  # Small weight for others
            weighted_sum += grade_points.get(grade, 2.0) * weight
            total_weight += weight
        
        gpa = weighted_sum / total_weight if total_weight > 0 else 2.0
        
        # Convert GPA to letter grade
        if gpa >= 3.5:
            return 'A', gpa
        elif gpa >= 2.5:
            return 'B', gpa
        elif gpa >= 1.5:
            return 'C', gpa
        elif gpa >= 0.5:
            return 'D', gpa
        else:
            return 'F', gpa
    
    # Calculate all metrics
    try:
        # Get benchmark for Alpha/Beta if available
        benchmark_returns = None
        try:
            spy_data = download_ticker_data(['SPY'], current['start_date'], current['end_date'])
            if spy_data is not None:
                benchmark_returns = spy_data.pct_change().dropna().iloc[:, 0]
        except:
            pass
        
        all_metrics = calculate_all_metrics(portfolio_returns, benchmark_returns)
        
        # Build grading table
        grading_data = []
        grades_dict = {}
        
        for metric_name, value in all_metrics.items():
            grade, ranges = grade_metric(metric_name, value)
            grades_dict[metric_name] = grade
            
            # Format value based on metric type
            if metric_name in ['Annual Return', 'Volatility', 'Best Month', 'Worst Month', 'Alpha']:
                formatted_value = f"{value:.2%}"
            elif metric_name in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Beta']:
                formatted_value = f"{value:.2f}"
            elif metric_name == 'Max Drawdown':
                formatted_value = f"{value:.2%}"
            elif metric_name == 'Win Rate':
                formatted_value = f"{value:.1%}"
            elif metric_name == 'Avg Recovery Days':
                formatted_value = f"{value:.0f} days"
            else:
                formatted_value = f"{value:.2f}"
            
            # Color code the grade
            grade_color = {
                'A': '🟢',
                'B': '🟡', 
                'C': '🟠',
                'D': '🔴',
                'F': '⛔'
            }
            
            grading_data.append({
                'Metric': metric_name,
                'Grading Scale': ranges,
                'Your Value': formatted_value,
                'Grade': f"{grade_color.get(grade, '')} {grade}"
            })
        
        # Calculate overall grade
        overall_letter, gpa = calculate_overall_grade(grades_dict)
        
        # Display the table
        grading_df = pd.DataFrame(grading_data)
        
        # Style the dataframe
        st.dataframe(
            grading_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Overall Grade Display
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            grade_color_map = {
                'A': 'success',
                'B': 'info',
                'C': 'warning',
                'D': 'error',
                'F': 'error'
            }
            
            grade_emoji = {
                'A': '🏆',
                'B': '✅',
                'C': '⚠️',
                'D': '❌',
                'F': '⛔'
            }
            
            grade_message = {
                'A': 'Outstanding! You are beating the S&P 500 - doing better than 80%+ of professionals!',
                'B': 'Excellent! S&P 500 level performance (already beats 80% of professionals long-term).',
                'C': 'Below Market. Consider if active management is worth the effort vs. just buying SPY.',
                'D': 'Significantly Below Market. Strategy needs major improvement.',
                'F': 'Poor Performance. Switch to index funds (SPY/VOO) - simpler and better.'
            }
            
            st.markdown(f"""
                <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white;">
                    <h1 style="margin: 0; font-size: 4rem;">{grade_emoji[overall_letter]}</h1>
                    <h2 style="margin: 0.5rem 0;">Overall Grade: {overall_letter}</h2>
                    <p style="margin: 0; font-size: 1.2rem;">GPA: {gpa:.2f} / 4.0</p>
                    <p style="margin-top: 1rem; font-size: 1.1rem;">{grade_message[overall_letter]}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Grade interpretation
        st.markdown("---")
        st.markdown("#### 📖 Understanding Your Grades")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                **Grade Scale (Calibrated to S&P 500 = B):**
                - 🟢 **A (4.0):** Beating S&P 500 - You're outperforming 80%+ of professionals!
                - 🟡 **B (3.0):** S&P 500 level - Excellent (beats 80% of pros long-term)
                - 🟠 **C (2.0):** Below market - Consider switching to SPY
                - 🔴 **D (1.0):** Significantly below market - Needs major changes
                - ⛔ **F (0.0):** Poor - Just buy SPY/VOO instead
                
                **Remember:** Getting a B means you're doing as well as the best long-term 
                investment! Most active managers fail to achieve this.
            """)
        
        with col2:
            st.markdown("""
                **Overall Grade Weighting (Hedge Fund Standard):**
                - Sharpe Ratio: 25% (Risk-adjusted returns)
                - Alpha: 20% (Value added vs. market)
                - Max Drawdown: 15% (Downside protection)
                - Annual Return: 15% (Absolute performance)
                - Other metrics: 25% (Sortino, Calmar, etc.)
            """)
        
        # Action items based on grade
        st.markdown("---")
        st.markdown("#### 🎯 What Your Grade Means for Action")
        
        if overall_letter == 'A':
            st.success("""
                **Grade A - Outstanding Performance!**
                
                ✅ **What to do:**
                - Document this performance (you're beating professionals!)
                - Maintain current strategy with quarterly rebalancing
                - Consider if you can handle slight increase in risk for potentially higher returns
                - Share this report card with your financial advisor
                
                ⚠️ **Caution:**
                - Don't get overconfident - markets change
                - Ensure you can still handle the max drawdown emotionally
                - Monitor for strategy degradation (check rolling Sharpe)
            """)
        elif overall_letter == 'B':
            st.info("""
                **Grade B - Very Good Performance!**
                
                ✅ **What to do:**
                - You're beating most professionals - well done!
                - Look for specific C or D grades to improve
                - Continue current strategy with confidence
                - Monitor monthly to ensure performance persists
                
                💡 **Improvement Areas:**
                - Check which metrics are C or below
                - Consider minor optimization (Tab 7)
                - Compare to benchmarks (Tab 6) for validation
            """)
        elif overall_letter == 'C':
            st.warning("""
                **Grade C - Acceptable but Room for Improvement**
                
                ⚠️ **What to do:**
                - Review metrics graded D or F - these need attention
                - Compare to simple strategies (60/40, SPY)
                - Consider if complexity is worth the effort
                - Use Tab 7 (Optimization) to explore improvements
                
                🔍 **Key Questions:**
                - Are you beating SPY? If not, why not just buy SPY?
                - Is your Sharpe Ratio > 0.5? If not, too much risk for return
                - Can you emotionally handle the max drawdown?
            """)
        else:  # D or F
            st.error("""
                **Grade D/F - Performance Needs Major Improvement**
                
                🚨 **Immediate Actions:**
                1. **Stop and reassess** - Don't throw good money after bad
                2. **Check Tab 6** - Are you underperforming simple strategies?
                3. **Review Tab 4** - Are you in wrong regime for your strategy?
                4. **Consider alternatives:**
                   - Switch to 60/40 portfolio (simple, proven)
                   - Buy SPY index fund (beats 80% of pros long-term)
                   - Hire a professional advisor
                
                ⚠️ **Reality Check:**
                - If multiple metrics are F, strategy is fundamentally flawed
                - Don't let losses compound - cut losses and restart
                - Sometimes simplest solution (index funds) is best
            """)
        
    except Exception as e:
        st.error(f"Error calculating portfolio grades: {str(e)}")
        st.info("Ensure your portfolio has sufficient data for grading (6+ months recommended)")
    
    # Generate PyFolio Analysis
    st.markdown("---")
    st.markdown("### 📈 Your Professional Tear Sheet")
    
    try:
        # Ensure returns is a Series with datetime index
        returns_series = portfolio_returns.copy()
        if isinstance(returns_series, pd.DataFrame):
            returns_series = returns_series.iloc[:, 0]
        
        with st.spinner("Generating institutional-grade analytics..."):
            fig = pf.create_returns_tear_sheet(returns_series, return_fig=True)
            if fig is not None:
                st.pyplot(fig)
            else:
                st.warning("Could not generate returns tear sheet")
        
        st.markdown("#### 💡 How to Interpret Your Results")
        st.markdown("**Quick Assessment (30 seconds):**")
        st.markdown("""
        1. Look at Annual Returns table → Are most years positive? ✅ or ❌
        2. Check Rolling Sharpe → Is it mostly above 0.5? ✅ or ❌
        3. Review Top 5 Drawdowns → Do you recover within 12 months? ✅ or ❌
        """)
        
        st.success("**If all three are ✅:** You have an institutionally-valid strategy!")
        st.warning("**If any are ❌:** Review the specific section above to understand what needs improvement.")
        
        st.markdown("**Next Steps:**")
        st.markdown("""
        - **If metrics are strong:** Document this analysis! You now have proof 
          your strategy works at a professional level.
        - **If metrics are weak:** Use Tab 7 (Optimization) to explore improvements, 
          or consider a simpler approach (60/40 or SPY).
        - **If metrics are mixed:** Identify the specific weakness (e.g., slow recovery, 
          high volatility) and adjust your allocation accordingly.
        """)
        
        # Professional comparison
        st.markdown("---")
        st.markdown("### 🏆 How Do You Compare to Professionals?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h4>Hedge Fund Benchmark</h4>
                    <p><strong>Typical Performance:</strong></p>
                    <ul>
                        <li>Annual Return: 8-12%</li>
                        <li>Sharpe Ratio: 0.8-1.5</li>
                        <li>Max Drawdown: -15% to -25%</li>
                        <li>Win Rate: 60-70%</li>
                    </ul>
                    <p style="font-size: 0.9rem; margin-top: 1rem;">
                    <em>If you beat these, you're performing at hedge fund level!</em></p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h4>Warren Buffett Benchmark</h4>
                    <p><strong>Berkshire Hathaway:</strong></p>
                    <ul>
                        <li>Annual Return: ~20% (historical)</li>
                        <li>Sharpe Ratio: ~0.8</li>
                        <li>Max Drawdown: -50% (2008)</li>
                        <li>Win Rate: ~70%</li>
                    </ul>
                    <p style="font-size: 0.9rem; margin-top: 1rem;">
                    <em>Even Buffett has had severe drawdowns. You're in good company.</em></p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h4>S&P 500 Benchmark</h4>
                    <p><strong>Index Performance:</strong></p>
                    <ul>
                        <li>Annual Return: ~10%</li>
                        <li>Sharpe Ratio: ~0.5-0.7</li>
                        <li>Max Drawdown: -56% (2008)</li>
                        <li>Win Rate: ~55%</li>
                    </ul>
                    <p style="font-size: 0.9rem; margin-top: 1rem;">
                    <em>If you can't beat this, just buy SPY. That's okay!</em></p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="success-box">
                <h4>🎯 Reality Check</h4>
                <p><strong>Professional investors fail to beat SPY 80-90% of the time over 10+ years.</strong></p>
                <p>If your PyFolio tear sheet shows you beating SPY on a risk-adjusted basis (Sharpe ratio), 
                you're doing better than most professionals. Be proud of that!</p>
                <p><strong>Key Insight:</strong> It's not about having the highest returns. It's about having 
                good risk-adjusted returns that you can stick with through market cycles. PyFolio shows you 
                if your strategy is sustainable long-term.</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error generating PyFolio analysis: {str(e)}")
        st.info("Note: PyFolio requires sufficient historical data (typically 6+ months)")
        
        st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Troubleshooting</h4>
                <p>If PyFolio fails to generate:</p>
                <ul>
                    <li>Ensure you have at least 6 months of data</li>
                    <li>Check that your portfolio has daily returns</li>
                    <li>Verify date range includes sufficient trading days</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 4: MARKET REGIMES (NEW!)
# =============================================================================

with tab4:
    st.markdown("## 🌡️ Market Conditions & Regime Analysis")
    st.markdown("""
        <div class="info-box">
            <h4>What Are Market Regimes?</h4>
            <p>Markets behave differently in different conditions. Understanding which "regime" 
            you're in helps you know if your strategy is working as expected.</p>
            <p><strong>The 5 Regimes:</strong></p>
            <ol>
                <li><strong>🟢 Bull Market (Low Vol):</strong> Goldilocks - steady gains, low stress</li>
                <li><strong>🔵 Bull Market (High Vol):</strong> Winning but volatile - gains with anxiety</li>
                <li><strong>🟡 Sideways/Choppy:</strong> Going nowhere - range-bound, frustrating</li>
                <li><strong>🟠 Bear Market (Low Vol):</strong> Slow bleed - gradual decline</li>
                <li><strong>🔴 Bear Market (High Vol):</strong> Crisis mode - crashes and panic</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    # Detect regimes
    with st.spinner("Analyzing market regimes..."):
        regimes = detect_market_regimes(portfolio_returns, lookback=60)
        regime_stats = analyze_regime_performance(portfolio_returns, regimes)
    
    # Current Regime
    st.markdown("---")
    st.markdown("### 🎯 Current Market Regime")
    current_regime = regimes.iloc[-1]
    
    regime_colors = {
        'Bull Market (Low Vol)': '#28a745',
        'Bull Market (High Vol)': '#17a2b8',
        'Sideways/Choppy': '#ffc107',
        'Bear Market (Low Vol)': '#fd7e14',
        'Bear Market (High Vol)': '#dc3545'
    }
    
    regime_descriptions = {
        'Bull Market (Low Vol)': {
            'emoji': '🟢',
            'status': 'Excellent',
            'description': 'Best conditions for investing. Steady gains with low stress. Stay invested!',
            'action': 'Maintain current allocation. Consider adding to positions on minor dips.'
        },
        'Bull Market (High Vol)': {
            'emoji': '🔵',
            'status': 'Good but Volatile',
            'description': 'Making gains but with bumpy ride. Normal during strong growth phases.',
            'action': 'Stay the course. Volatility is creating buying opportunities. Don\'t sell on dips.'
        },
        'Sideways/Choppy': {
            'emoji': '🟡',
            'status': 'Neutral',
            'description': 'Market is range-bound. Frustrating but not dangerous.',
            'action': 'Be patient. Avoid chasing momentum. Good time for rebalancing.'
        },
        'Bear Market (Low Vol)': {
            'emoji': '🟠',
            'status': 'Caution',
            'description': 'Slow grind lower. Early warning sign of potential trouble.',
            'action': 'Review portfolio. Consider raising cash or adding defensive positions.'
        },
        'Bear Market (High Vol)': {
            'emoji': '🔴',
            'status': 'Crisis Mode',
            'description': 'High stress period with significant losses. Historically temporary.',
            'action': 'DO NOT PANIC SELL! Historically the best buying opportunity. Deep breaths.'
        }
    }
    
    regime_info = regime_descriptions[current_regime]
    
    st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {regime_colors[current_regime]};">
            <h2>{regime_info['emoji']} {current_regime}</h2>
            <h3>Status: {regime_info['status']}</h3>
            <p style="font-size: 1.1rem; margin-top: 1rem;"><strong>What This Means:</strong> 
            {regime_info['description']}</p>
            <p style="font-size: 1.1rem; margin-top: 1rem;"><strong>🎯 Action Item:</strong> 
            {regime_info['action']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Regime Timeline
    st.markdown("---")
    st.markdown("### 📊 Regime Timeline & Performance")
    fig = plot_regime_chart(regimes, portfolio_returns)
    st.pyplot(fig)
    
    # Regime timeline interpretation
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 How to Read the Regime Chart</div>
            <p><strong>Top Chart:</strong> Your portfolio value with colored backgrounds showing regimes</p>
            <p><strong>Bottom Chart:</strong> Timeline of regime changes</p>
            <p><strong>Key Insights to Look For:</strong></p>
            <ul>
                <li><strong>Big gains in green zones:</strong> Portfolio is working as designed</li>
                <li><strong>Losses in red zones:</strong> Expected, but how bad compared to benchmark?</li>
                <li><strong>Flat in yellow zones:</strong> Your capital is idle - frustrating but safe</li>
                <li><strong>Quick regime switches:</strong> Market is uncertain, be careful</li>
                <li><strong>Long red zones:</strong> True bear markets - historical best buying opportunity</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Performance by Regime
    st.markdown("---")
    st.markdown("### 📈 Performance by Regime")
    
    # Format the dataframe for display
    regime_stats_display = regime_stats.copy()
    regime_stats_display['Avg Daily Return'] = regime_stats_display['Avg Daily Return'].apply(lambda x: f"{x:.4f}")
    regime_stats_display['Volatility'] = regime_stats_display['Volatility'].apply(lambda x: f"{x:.2%}")
    regime_stats_display['Best Day'] = regime_stats_display['Best Day'].apply(lambda x: f"{x:.2%}")
    regime_stats_display['Worst Day'] = regime_stats_display['Worst Day'].apply(lambda x: f"{x:.2%}")
    regime_stats_display['Win Rate'] = regime_stats_display['Win Rate'].apply(lambda x: f"{x:.2%}")
    
    # Color-code the table
    def color_regime(val):
        color = regime_colors.get(val, '#f8f9fa')
        return f'background-color: {color}; color: white; font-weight: bold'
    
    styled_df = regime_stats_display.style.applymap(
        color_regime, subset=['Regime']
    )
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Regime performance interpretation
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 How to Use Regime Performance Data</div>
            <p><strong>What Each Column Means:</strong></p>
            <ul>
                <li><strong>Occurrences:</strong> How many days in each regime</li>
                <li><strong>Avg Daily Return:</strong> Typical daily move in that regime</li>
                <li><strong>Volatility:</strong> Annualized volatility (stress level)</li>
                <li><strong>Best/Worst Day:</strong> Extreme moves to expect</li>
                <li><strong>Win Rate:</strong> % of positive days</li>
            </ul>
            <p><strong>Key Questions to Ask:</strong></p>
            <ul>
                <li>Do you make money in bull markets? (You should!)</li>
                <li>How bad are losses in bear markets vs benchmark?</li>
                <li>Is volatility acceptable in each regime?</li>
                <li>Win rate > 50% in bull markets? Good sign.</li>
                <li>Win rate < 40% in bear markets? Portfolio may need defensive assets.</li>
            </ul>
            <p><strong>🚩 Red Flags:</strong></p>
            <ul>
                <li>Negative returns in Bull Market (Low Vol) - strategy is broken</li>
                <li>Higher losses in Bear Market (High Vol) than benchmark - insufficient protection</li>
                <li>Low win rate across all regimes - strategy is too volatile for you</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 5: FORWARD-LOOKING RISK ANALYSIS (NEW!)
# =============================================================================

with tab5:
    st.markdown("## 🔮 Forward-Looking Risk Analysis")
    st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Important Disclaimer</h4>
            <p><strong>Past performance does not guarantee future results.</strong> 
            This analysis projects future risks based on historical behavior, but markets can change.</p>
            <p>Use these projections as one tool among many for decision-making, not as a crystal ball.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate forward-looking metrics
    with st.spinner("Running forward-looking analysis..."):
        forward_metrics = calculate_forward_risk_metrics(portfolio_returns)
    
    # Expected Metrics
    st.markdown("---")
    st.markdown("### 📊 Expected Performance (Next 12 Months)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        expected_return = forward_metrics['Expected Annual Return']
        color_class = 'metric-excellent' if expected_return > 0.10 else 'metric-good' if expected_return > 0.05 else 'metric-fair'
        st.markdown(f"""
            <div class="{color_class}">
                <h4>Expected Return</h4>
                <h2>{expected_return:.2%}</h2>
                <p style="margin-top: 0.5rem;">Based on historical avg</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        expected_vol = forward_metrics['Expected Volatility']
        color_class = 'metric-excellent' if expected_vol < 0.15 else 'metric-good' if expected_vol < 0.20 else 'metric-fair'
        st.markdown(f"""
            <div class="{color_class}">
                <h4>Expected Volatility</h4>
                <h2>{expected_vol:.2%}</h2>
                <p style="margin-top: 0.5rem;">Expected fluctuation</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        prob_loss = forward_metrics['Probability of Daily Loss']
        color_class = 'metric-excellent' if prob_loss < 0.40 else 'metric-good' if prob_loss < 0.45 else 'metric-fair'
        st.markdown(f"""
            <div class="{color_class}">
                <h4>Daily Loss Probability</h4>
                <h2>{prob_loss:.1%}</h2>
                <p style="margin-top: 0.5rem;">Chance of down day</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        est_max_dd = forward_metrics['Estimated Max Drawdown']
        color_class = 'metric-excellent' if est_max_dd > -0.15 else 'metric-good' if est_max_dd > -0.25 else 'metric-poor'
        st.markdown(f"""
            <div class="{color_class}">
                <h4>Est. Max Drawdown</h4>
                <h2>{est_max_dd:.2%}</h2>
                <p style="margin-top: 0.5rem;">Worst case scenario</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Risk Metrics
    st.markdown("---")
    st.markdown("### 🎯 Value at Risk (VaR) Analysis")
    st.markdown("""
        <div class="info-box">
            <p><strong>Value at Risk (VaR)</strong> answers: "How much could I lose on a bad day?"</p>
            <p><strong>Conditional VaR (CVaR)</strong> answers: "If that bad day happens, how much worse could it get?"</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 95% Confidence Level")
        var_95 = forward_metrics['VaR (95%)']
        cvar_95 = forward_metrics['CVaR (95%)']
        
        st.markdown(f"""
            <div class="metric-card">
                <h4>VaR (95%)</h4>
                <h2>{var_95:.2%}</h2>
                <p style="margin-top: 1rem;">
                <strong>What this means:</strong> On 95% of days, your loss won't be worse than this.
                Or said differently: Only 1 in 20 days (5%) will be worse than this.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card" style="margin-top: 1rem;">
                <h4>CVaR (95%)</h4>
                <h2>{cvar_95:.2%}</h2>
                <p style="margin-top: 1rem;">
                <strong>What this means:</strong> On those 5% worst days, this is the AVERAGE loss.
                This is your "expected bad day" loss.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 99% Confidence Level")
        var_99 = forward_metrics['VaR (99%)']
        cvar_99 = forward_metrics['CVaR (99%)']
        
        st.markdown(f"""
            <div class="metric-card">
                <h4>VaR (99%)</h4>
                <h2>{var_99:.2%}</h2>
                <p style="margin-top: 1rem;">
                <strong>What this means:</strong> On 99% of days, your loss won't be worse than this.
                Only 1 in 100 days (1%) will be worse.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card" style="margin-top: 1rem;">
                <h4>CVaR (99%)</h4>
                <h2>{cvar_99:.2%}</h2>
                <p style="margin-top: 1rem;">
                <strong>What this means:</strong> On those 1% worst days, this is the AVERAGE loss.
                This is your "tail risk" exposure.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # VaR interpretation
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 How to Use VaR in Real Life</div>
            <p><strong>Example with $100,000 Portfolio:</strong></p>
            <ul>
                <li>VaR (95%) = -2.5% → On 95% of days, you'll lose less than $2,500</li>
                <li>CVaR (95%) = -3.5% → On the 5% worst days, average loss is $3,500</li>
                <li>VaR (99%) = -4.0% → Only 1% of days lose more than $4,000</li>
                <li>CVaR (99%) = -5.5% → On the very worst 1% of days, average loss is $5,500</li>
            </ul>
            <p><strong>Questions to Ask Yourself:</strong></p>
            <ul>
                <li>Can I emotionally handle the CVaR (95%) loss regularly?</li>
                <li>Can I financially survive the CVaR (99%) loss?</li>
                <li>Do I have enough liquidity to avoid selling at a loss?</li>
            </ul>
            <p><strong>🚩 Red Flags:</strong></p>
            <ul>
                <li>CVaR (95%) > -5%: You'll experience painful days frequently</li>
                <li>CVaR (99%) > -10%: Your worst days are VERY bad</li>
                <li>If these numbers scare you, your portfolio is too aggressive</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Monte Carlo Simulation
    st.markdown("---")
    st.markdown("### 🎲 Monte Carlo Simulation (1 Year Forward)")
    st.markdown("""
        <div class="info-box">
            <p><strong>What is Monte Carlo?</strong> We run 1,000+ possible future scenarios based on your 
            portfolio's historical behavior. This shows the range of possible outcomes.</p>
            <p><strong>How to read:</strong> The fan of lines shows possible paths. The colored lines show 
            key percentiles (5th to 95th). The wider the fan, the more uncertain the future.</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Running Monte Carlo simulation (this may take a moment)..."):
        simulations = monte_carlo_simulation(portfolio_returns, days_forward=252, num_simulations=1000)
    
    fig = plot_monte_carlo_simulation(simulations)
    st.pyplot(fig)
    
    # Monte Carlo interpretation
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 Understanding Monte Carlo Results</div>
            <p><strong>The Lines Explained:</strong></p>
            <ul>
                <li><strong>Green (50th %ile):</strong> Median outcome - "most likely" path</li>
                <li><strong>Dark Blue (25th & 75th %ile):</strong> "Typical" range of outcomes</li>
                <li><strong>Orange (5th %ile):</strong> Bad luck scenario - 95% chance of doing better</li>
                <li><strong>Gray (95th %ile):</strong> Good luck scenario - 95% chance of doing worse</li>
            </ul>
            <p><strong>What to Look For:</strong></p>
            <ul>
                <li><strong>Wide fan:</strong> High uncertainty, hard to predict</li>
                <li><strong>Narrow fan:</strong> More predictable outcomes</li>
                <li><strong>Most lines above 1.0:</strong> Positive expected returns</li>
                <li><strong>5th %ile below 0.85:</strong> Significant risk of 15%+ loss</li>
            </ul>
            <p><strong>Real-World Use:</strong></p>
            <ul>
                <li>Planning to retire next year? Look at 5th percentile - can you afford that outcome?</li>
                <li>Young investor? Focus on median and 75th percentile - you have time</li>
                <li>Need the money in 1 year? If 25th percentile is below 0.95, you have risk</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Scenario Analysis
    st.markdown("---")
    st.markdown("### 📊 Scenario Analysis (1 Year Forward)")
    
    final_values = simulations[-1, :]
    scenarios = {
        'Best Case (95th %ile)': np.percentile(final_values, 95),
        'Good Case (75th %ile)': np.percentile(final_values, 75),
        'Median Case (50th %ile)': np.percentile(final_values, 50),
        'Bad Case (25th %ile)': np.percentile(final_values, 25),
        'Worst Case (5th %ile)': np.percentile(final_values, 5)
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        scenario_df = pd.DataFrame({
            'Scenario': scenarios.keys(),
            'Portfolio Value': [f"${v:.2f}" for v in scenarios.values()],
            'Return': [f"{(v-1)*100:.1f}%" for v in scenarios.values()]
        })
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h4>Probability Analysis</h4>
                <p style="margin-top: 1rem;">
                    <strong>Make Money:</strong><br>
                    {:.1f}% chance<br><br>
                    <strong>Lose Money:</strong><br>
                    {:.1f}% chance<br><br>
                    <strong>Lose > 10%:</strong><br>
                    {:.1f}% chance
                </p>
            </div>
        """.format(
            (final_values > 1.0).sum() / len(final_values) * 100,
            (final_values < 1.0).sum() / len(final_values) * 100,
            (final_values < 0.9).sum() / len(final_values) * 100
        ), unsafe_allow_html=True)
    
    # Scenario interpretation
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 Using Scenarios for Decision-Making</div>
            <p><strong>Example: Planning with $100,000</strong></p>
            <ul>
                <li><strong>Best Case:</strong> Portfolio grows to $115,000 (15% gain) - Happy days!</li>
                <li><strong>Median Case:</strong> Portfolio grows to $107,000 (7% gain) - Acceptable</li>
                <li><strong>Worst Case:</strong> Portfolio drops to $92,000 (8% loss) - Ouch, but survivable?</li>
            </ul>
            <p><strong>Decision Framework:</strong></p>
            <ul>
                <li><strong>Can't afford worst case?</strong> Portfolio is too aggressive. Add bonds/cash.</li>
                <li><strong>Comfortable with worst case?</strong> You're properly positioned.</li>
                <li><strong>Disappointed by median case?</strong> Need more risk for your goals.</li>
            </ul>
            <p><strong>Important Reality Check:</strong></p>
            <ul>
                <li>These scenarios assume historical patterns continue</li>
                <li>Black swan events (2008, COVID) can exceed worst case</li>
                <li>Keep 6-12 months expenses in cash regardless of scenarios</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 6: COMPARE BENCHMARKS (ENHANCED WITH SMART SELECTION)
# =============================================================================

with tab6:
    st.markdown("## ⚖️ Compare Against Benchmarks")
    
    st.info("""
        **🎯 Smart Benchmark Selection:** Benchmarks are auto-selected based on your portfolio composition.
        This ensures you're comparing against the most relevant indices rather than generic ones.
    """)
    
    # Get smart benchmark recommendations
    smart_benchmarks = get_smart_benchmarks(list(weights.keys()), list(weights.values()))
    
    # Display recommended benchmarks
    st.markdown("### 📊 Auto-Selected Benchmarks")
    
    benchmark_info = []
    for benchmark, reason in smart_benchmarks:
        benchmark_info.append({
            'Benchmark': benchmark,
            'Reason': reason
        })
    
    if benchmark_info:
        st.dataframe(pd.DataFrame(benchmark_info), use_container_width=True, hide_index=True)
    
    # Allow manual additions
    st.markdown("#### ➕ Add Additional Benchmarks (Optional)")
    col1, col2, col3, col4 = st.columns(4)
    
    additional_benchmarks = []
    with col1:
        if st.checkbox("QQQ (Nasdaq 100)", value=False, help="Tech-heavy index"):
            additional_benchmarks.append(('QQQ', 'Nasdaq 100 comparison'))
    with col2:
        if st.checkbox("IWM (Russell 2000)", value=False, help="Small cap index"):
            additional_benchmarks.append(('IWM', 'Small cap comparison'))
    with col3:
        if st.checkbox("VT (Total World)", value=False, help="Global stocks"):
            additional_benchmarks.append(('VT', 'Global market comparison'))
    with col4:
        if st.checkbox("AGG (Total Bond)", value=False, help="Bond market"):
            additional_benchmarks.append(('AGG', 'Bond market comparison'))
    
    # Combine smart and additional benchmarks
    all_benchmarks = smart_benchmarks + additional_benchmarks
    
    # Download benchmark data
    benchmarks_data = {}
    benchmarks_metrics = {}
    
    for benchmark_symbol, reason in all_benchmarks:
        if benchmark_symbol == '60/40':
            # Create synthetic 60/40 portfolio
            spy_data = download_ticker_data(['SPY'], current['start_date'], current['end_date'])
            agg_data = download_ticker_data(['AGG'], current['start_date'], current['end_date'])
            
            if spy_data is not None and agg_data is not None:
                combined_data = pd.DataFrame({
                    'SPY': spy_data.iloc[:, 0] if isinstance(spy_data, pd.DataFrame) else spy_data,
                    'AGG': agg_data.iloc[:, 0] if isinstance(agg_data, pd.DataFrame) else agg_data
                }).dropna()
                
                portfolio_6040 = calculate_portfolio_returns(combined_data, np.array([0.6, 0.4]))
                benchmarks_data['60/40'] = portfolio_6040
                benchmarks_metrics['60/40'] = calculate_portfolio_metrics(portfolio_6040)
        else:
            # Download single benchmark
            bench_data = get_benchmark_data_openbb(benchmark_symbol, current['start_date'], current['end_date'])
            if bench_data is not None:
                bench_returns = bench_data.pct_change().dropna()
                bench_returns_series = bench_returns.iloc[:, 0] if isinstance(bench_returns, pd.DataFrame) else bench_returns
                benchmarks_data[benchmark_symbol] = bench_returns_series
                benchmarks_metrics[benchmark_symbol] = calculate_portfolio_metrics(bench_returns_series)
    
    if not benchmarks_data:
        st.warning("⚠️ Could not load benchmark data. Please check your internet connection.")
    else:
        # Enhanced Metrics Comparison Table
        st.markdown("---")
        st.markdown("### 📊 Comprehensive Metrics Comparison")
        
        comparison_rows = []
        
        # Key metrics to compare
        metric_configs = [
            ('Annual Return', 'Annual Return', 'higher_better', '%'),
            ('Sharpe Ratio', 'Sharpe Ratio', 'higher_better', 'ratio'),
            ('Sortino Ratio', 'Sortino Ratio', 'higher_better', 'ratio'),
            ('Max Drawdown', 'Max Drawdown', 'lower_better', '%'),
            ('Volatility', 'Annual Volatility', 'higher_better', '%'),
            ('Calmar Ratio', 'Calmar Ratio', 'higher_better', 'ratio'),
            ('Total Return', 'Total Return', 'higher_better', '%')
        ]
        
        for metric_display, metric_key, comparison_type, format_type in metric_configs:
            row = {'Metric': metric_display}
            
            # Add portfolio value
            port_value = metrics[metric_key]
            if format_type == '%':
                row['Your Portfolio'] = f"{port_value:.2%}"
            else:
                row['Your Portfolio'] = f"{port_value:.2f}"
            
            # Add benchmark values with comparison arrows
            for bench_name, bench_metrics in benchmarks_metrics.items():
                bench_value = bench_metrics[metric_key]
                
                # Determine if portfolio is better
                if comparison_type == 'higher_better':
                    is_better = port_value > bench_value
                    arrow = " 🟢↑" if is_better else " 🔴↓"
                else:  # lower_better
                    is_better = port_value < bench_value
                    arrow = " 🟢↑" if is_better else " 🔴↓"
                
                if format_type == '%':
                    row[bench_name] = f"{bench_value:.2%}{arrow}"
                else:
                    row[bench_name] = f"{bench_value:.2f}{arrow}"
            
            comparison_rows.append(row)
        
        comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
            <div style="text-align: center; padding: 10px; background: #f8f9fa; border-radius: 5px; margin-top: 10px;">
                <small><strong>Legend:</strong> 🟢↑ = Your portfolio better | 🔴↓ = Benchmark better</small>
            </div>
        """, unsafe_allow_html=True)
        
        # Calculate percentile ranking
        st.markdown("---")
        st.markdown("### 🏆 Percentile Ranking")
        
        # Collect all Sharpe ratios for ranking
        all_sharpes = [metrics['Sharpe Ratio']]
        for bench_metrics in benchmarks_metrics.values():
            all_sharpes.append(bench_metrics['Sharpe Ratio'])
        
        # Calculate percentile
        portfolio_sharpe = metrics['Sharpe Ratio']
        better_than_count = sum(1 for s in all_sharpes if portfolio_sharpe > s)
        percentile = (better_than_count / len(all_sharpes)) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Your Percentile",
                f"{percentile:.0f}th",
                help="Based on Sharpe Ratio vs selected benchmarks"
            )
        
        with col2:
            better_count = sum(1 for _, bench_metrics in benchmarks_metrics.items() 
                             if metrics['Sharpe Ratio'] > bench_metrics['Sharpe Ratio'])
            st.metric(
                "Benchmarks Beaten",
                f"{better_count} of {len(benchmarks_metrics)}",
                help="Number of benchmarks you outperformed on Sharpe Ratio"
            )
        
        with col3:
            rank_text = ""
            if percentile >= 80:
                rank_text = "🌟 Excellent - Top 20%"
                rank_color = "success"
            elif percentile >= 60:
                rank_text = "✅ Good - Above Average"
                rank_color = "info"
            elif percentile >= 40:
                rank_text = "⚪ Average"
                rank_color = "warning"
            else:
                rank_text = "⚠️ Below Average"
                rank_color = "error"
            
            st.metric("Rating", rank_text)
        
        # Interpretation based on ranking
        if percentile >= 70:
            st.success(f"""
                **🎉 Strong Performance!** Your portfolio is outperforming {percentile:.0f}% of selected benchmarks.
                You're delivering better risk-adjusted returns than most standard strategies.
            """)
        elif percentile >= 50:
            st.info(f"""
                **✅ Solid Performance:** Your portfolio is in the {percentile:.0f}th percentile.
                You're performing above average but there may be room for improvement.
            """)
        else:
            st.warning(f"""
                **⚠️ Performance Review Needed:** Your portfolio is in the {percentile:.0f}th percentile.
                Consider reviewing your strategy - several benchmarks are delivering better risk-adjusted returns.
            """)
        
        # Cumulative Performance Chart
        st.markdown("---")
        st.markdown("### 📈 Cumulative Performance Over Time")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot portfolio
        cum_returns_portfolio = (1 + portfolio_returns).cumprod()
        cum_returns_portfolio.plot(ax=ax, linewidth=3, label='Your Portfolio', color='#667eea')
        
        # Plot benchmarks
        colors = ['#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6f42c1', '#fd7e14']
        for i, (name, returns) in enumerate(benchmarks_data.items()):
            cum_returns_bench = (1 + returns).cumprod()
            cum_returns_bench.plot(ax=ax, linewidth=2, label=name, 
                                  color=colors[i % len(colors)], linestyle='--', alpha=0.8)
        
        ax.set_title('Performance Comparison vs Smart Benchmarks', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Smart interpretation
        st.markdown("""
            <div class="interpretation-box">
                <div class="interpretation-title">💡 How to Interpret Your Results</div>
                <p><strong>Understanding Benchmark Selection:</strong></p>
                <ul>
                    <li>Benchmarks were auto-selected based on your portfolio composition</li>
                    <li>This ensures you're comparing against relevant indices, not generic ones</li>
                    <li>A tech-heavy portfolio should compare to QQQ, not just SPY</li>
                </ul>
                <p><strong>What Good Performance Looks Like:</strong></p>
                <ul>
                    <li><strong>Above most benchmarks:</strong> Your strategy is adding value ✓</li>
                    <li><strong>Better Sharpe than SPY:</strong> You're delivering superior risk-adjusted returns ✓</li>
                    <li><strong>70th percentile or higher:</strong> You're outperforming most strategies ✓</li>
                </ul>
                <p><strong>🚩 Warning Signs:</strong></p>
                <ul>
                    <li><strong>Below 50th percentile:</strong> Majority of benchmarks are beating you</li>
                    <li><strong>Lower Sharpe than all benchmarks:</strong> Taking more risk for less return</li>
                    <li><strong>Underperforming SPY consistently:</strong> Consider switching to index fund</li>
                </ul>
                <p><strong>Decision Framework:</strong></p>
                <ul>
                    <li>If beating most benchmarks: Keep your strategy, it's working!</li>
                    <li>If average performance: Minor tweaks may help, but acceptable</li>
                    <li>If below average: Strongly consider switching to best-performing benchmark</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Rolling Sharpe Comparison
        st.markdown("---")
        st.markdown("### 📈 Rolling Sharpe Ratio (Risk-Adjusted Performance Over Time)")
        
        window = 60
        portfolio_rolling_sharpe = (portfolio_returns.rolling(window).mean() * 252) / (portfolio_returns.rolling(window).std() * np.sqrt(252))
        
        fig, ax = plt.subplots(figsize=(14, 8))
        portfolio_rolling_sharpe.plot(ax=ax, linewidth=3, label='Your Portfolio', color='#667eea')
        
        for i, (name, returns) in enumerate(benchmarks_data.items()):
            bench_rolling_sharpe = (returns.rolling(window).mean() * 252) / (returns.rolling(window).std() * np.sqrt(252))
            bench_rolling_sharpe.plot(ax=ax, linewidth=2, label=name,
                                     color=colors[i % len(colors)], linestyle='--', alpha=0.8)
        
        ax.axhline(y=1, color='#28a745', linestyle=':', linewidth=1.5, alpha=0.7, label='Good (1.0)')
        ax.axhline(y=0, color='#dc3545', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax.set_title(f'Rolling {window}-Day Sharpe Ratio Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
            <div class="interpretation-box">
                <div class="interpretation-title">💡 Rolling Sharpe Analysis</div>
                <p><strong>What This Shows:</strong> How risk-adjusted returns evolved over time</p>
                <p><strong>Key Patterns:</strong></p>
                <ul>
                    <li><strong>Consistently above benchmarks:</strong> Your strategy consistently delivers better risk-adjusted returns</li>
                    <li><strong>Converges during crises:</strong> All strategies suffer together in major crashes</li>
                    <li><strong>Diverges in recovery:</strong> Shows which strategy recovers better</li>
                    <li><strong>Recent trend matters most:</strong> Is your edge improving or deteriorating?</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)



# =============================================================================
# TAB 7: OPTIMIZATION
# =============================================================================

with tab7:
    st.markdown("## 🎯 Portfolio Optimization")
    st.markdown("""
        <div class="info-box">
            <h4>What is Portfolio Optimization?</h4>
            <p>Find the best allocation of your assets to maximize returns for a given level of risk, 
            or minimize risk for a given level of returns.</p>
            <p><strong>Maximum Sharpe Ratio:</strong> Find the allocation with the best risk-adjusted returns.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # ETF DEEP DIVE (Phase 1 OpenBB Feature)
    st.markdown("---")
    st.markdown("### 🔍 ETF Deep Dive - Know What You Own")
    
    st.info("""
        **💰 Optimize Your Costs:** Discover what's inside your ETFs and find cheaper alternatives that track the same index.
        Small differences in expense ratios compound to thousands of dollars over time!
    """)
    
    # ETF Selector
    selected_etf = st.selectbox(
        "Select an ETF to analyze:",
        list(weights.keys()),
        help="Choose an ETF from your portfolio to see detailed information"
    )
    
    if selected_etf:
        # Get expense ratio from yfinance
        try:
            etf_ticker = yf.Ticker(selected_etf)
            etf_info = etf_ticker.info
            
            # Basic Information Section
            st.markdown(f"#### 📋 {selected_etf} - Basic Information")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                expense_ratio = etf_info.get('expenseRatio', 0) if etf_info.get('expenseRatio') else 0
                st.metric(
                    "Expense Ratio",
                    f"{expense_ratio:.2%}",
                    help="Annual fee as percentage of investment"
                )
                portfolio_value = 100000  # Default
                annual_cost = portfolio_value * expense_ratio
                st.caption(f"${annual_cost:,.0f}/year on $100k")
            
            with col2:
                aum = etf_info.get('totalAssets', 0)
                if aum > 0:
                    aum_b = aum / 1e9
                    st.metric(
                        "Assets (AUM)",
                        f"${aum_b:.1f}B",
                        help="Total assets under management"
                    )
                else:
                    st.metric("Assets (AUM)", "N/A")
            
            with col3:
                div_yield = etf_info.get('yield', etf_info.get('dividendYield', 0))
                if div_yield:
                    st.metric(
                        "Dividend Yield",
                        f"{div_yield:.2%}",
                        help="Annual dividend yield"
                    )
                else:
                    st.metric("Dividend Yield", "N/A")
            
            with col4:
                category = etf_info.get('category', 'N/A')
                st.metric(
                    "Category",
                    category if category else "ETF",
                    help="Investment category"
                )
            
            # Find Cheaper Alternatives
            st.markdown("---")
            st.markdown("#### 💰 Cheaper Alternatives - Save on Fees!")
            
            alternatives = get_cheaper_etf_alternatives(selected_etf, expense_ratio)
            
            if alternatives and expense_ratio > 0:
                st.success(f"**Found {len(alternatives)} cheaper alternative(s) for {selected_etf}!**")
                
                for alt in alternatives:
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.markdown(f"**{alt['symbol']}** - {alt['name']}")
                        st.caption(f"Tracking: {alt['tracking']}")
                    
                    with col2:
                        st.metric(
                            "Expense Ratio",
                            f"{alt['expense_ratio']:.2%}"
                        )
                    
                    with col3:
                        # Calculate savings
                        user_portfolio_value = st.number_input(
                            f"Your {selected_etf} position value ($)",
                            min_value=1000,
                            max_value=10000000,
                            value=100000,
                            step=10000,
                            key=f"portfolio_value_{alt['symbol']}",
                            help="Enter your position size to calculate savings"
                        )
                        
                        savings = calculate_expense_ratio_savings(
                            expense_ratio,
                            alt['expense_ratio'],
                            user_portfolio_value
                        )
                        
                        st.metric(
                            "Annual Savings",
                            f"${savings['annual_savings']:,.0f}",
                            f"{savings['percent_cheaper']:.0f}% cheaper"
                        )
                        st.caption(f"20-year savings: ${savings['savings_20y']:,.0f}")
                
                # Summary recommendation
                best_alt = alternatives[0] if alternatives else None
                if best_alt:
                    savings = calculate_expense_ratio_savings(
                        expense_ratio,
                        best_alt['expense_ratio'],
                        user_portfolio_value
                    )
                    
                    st.markdown(f"""
                        <div class="interpretation-box">
                            <div class="interpretation-title">💡 Recommendation</div>
                            <p><strong>Switch from {selected_etf} to {best_alt['symbol']}</strong></p>
                            <ul>
                                <li>Save <strong>${savings['annual_savings']:,.0f}/year</strong> on a ${user_portfolio_value:,.0f} position</li>
                                <li>That's <strong>{savings['percent_cheaper']:.0f}% cheaper</strong> for the same exposure</li>
                                <li>Over 20 years: <strong>${savings['savings_20y']:,.0f}</strong> saved (with compound growth)</li>
                                <li>Same index, same holdings, same performance - just lower fees!</li>
                            </ul>
                            <p><strong>🎯 Action:</strong> If you're in a taxable account, check if switching triggers capital gains tax. 
                            In tax-advantaged accounts (401k, IRA), switch immediately - no tax impact!</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            elif expense_ratio > 0:
                st.info(f"**{selected_etf}** already has competitive fees. No cheaper alternatives found in our database.")
            else:
                st.warning("Could not fetch expense ratio data for this ETF.")
            
            # Holdings Information (if available from yfinance)
            st.markdown("---")
            st.markdown("#### 📊 Top Holdings")
            
            try:
                # Try to get holdings data
                # Note: yfinance may not always have this data
                st.info("**Note:** Detailed holdings data requires OpenBB. Install OpenBB for comprehensive holdings analysis.")
                
                # Placeholder for future OpenBB integration
                if OPENBB_AVAILABLE:
                    etf_data = get_etf_info_openbb(selected_etf)
                    if etf_data and not etf_data['holdings'].empty:
                        st.dataframe(etf_data['holdings'].head(10), use_container_width=True)
                    else:
                        st.caption("Holdings data not available through OpenBB for this ETF.")
                else:
                    st.caption("Install OpenBB to see top holdings, sector allocation, and more: `pip install openbb --break-system-packages`")
            except:
                st.caption("Holdings data not available.")
            
            # Performance History
            st.markdown("---")
            st.markdown("#### 📈 Performance History")
            
            # Show simple performance metrics
            etf_data_prices = download_ticker_data([selected_etf], current['start_date'], current['end_date'])
            if etf_data_prices is not None:
                etf_returns = etf_data_prices.pct_change().dropna()
                etf_metrics = calculate_portfolio_metrics(etf_returns)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Annual Return", f"{etf_metrics['Annual Return']:.2%}")
                
                with col2:
                    st.metric("Volatility", f"{etf_metrics['Annual Volatility']:.2%}")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{etf_metrics['Sharpe Ratio']:.2f}")
                
                with col4:
                    st.metric("Max Drawdown", f"{etf_metrics['Max Drawdown']:.2%}")
                
                # Simple performance chart
                cum_returns = (1 + etf_returns).cumprod()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                cum_returns.plot(ax=ax, linewidth=2, color='#667eea')
                ax.set_title(f'{selected_etf} - Cumulative Performance', fontsize=14, fontweight='bold')
                ax.set_xlabel('Date', fontsize=11)
                ax.set_ylabel('Cumulative Return', fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#f8f9fa')
                fig.patch.set_facecolor('white')
                plt.tight_layout()
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Could not fetch detailed data for {selected_etf}: {str(e)}")
            st.info("Some ETFs may have limited data available through the free tier.")
    
    # Current vs Optimal
    st.markdown("---")
    st.markdown("### 📊 Current vs Optimal Allocation")
    
    # Calculate optimal weights
    with st.spinner("Optimizing portfolio..."):
        optimal_weights = optimize_portfolio(prices, method='max_sharpe')
        optimal_returns = calculate_portfolio_returns(prices, optimal_weights)
        optimal_metrics = calculate_portfolio_metrics(optimal_returns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Current Allocation")
        current_weights_df = pd.DataFrame({
            'Ticker': list(weights.keys()),
            'Weight': [f"{w*100:.2f}%" for w in weights.values()]
        })
        st.dataframe(current_weights_df, use_container_width=True, hide_index=True)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.Set3(range(len(weights)))
        ax.pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title('Current Allocation', fontsize=14, fontweight='bold', pad=20)
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### Optimal Allocation (Max Sharpe)")
        optimal_weights_dict = {ticker: w for ticker, w in zip(prices.columns, optimal_weights)}
        optimal_weights_df = pd.DataFrame({
            'Ticker': list(optimal_weights_dict.keys()),
            'Weight': [f"{w*100:.2f}%" for w in optimal_weights_dict.values()]
        })
        st.dataframe(optimal_weights_df, use_container_width=True, hide_index=True)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(optimal_weights_dict.values(), labels=optimal_weights_dict.keys(), 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Optimal Allocation', fontsize=14, fontweight='bold', pad=20)
        st.pyplot(fig)
    
    # Metrics Comparison
    st.markdown("---")
    st.markdown("### 📈 Performance Comparison")
    
    comparison_data = {
        'Metric': ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Sortino Ratio'],
        'Current Portfolio': [
            f"{metrics['Annual Return']:.2%}",
            f"{metrics['Annual Volatility']:.2%}",
            f"{metrics['Sharpe Ratio']:.2f}",
            f"{metrics['Max Drawdown']:.2%}",
            f"{metrics['Sortino Ratio']:.2f}"
        ],
        'Optimal Portfolio': [
            f"{optimal_metrics['Annual Return']:.2%}",
            f"{optimal_metrics['Annual Volatility']:.2%}",
            f"{optimal_metrics['Sharpe Ratio']:.2f}",
            f"{optimal_metrics['Max Drawdown']:.2%}",
            f"{optimal_metrics['Sortino Ratio']:.2f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Optimization interpretation
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 Should You Switch to Optimal Allocation?</div>
            <p><strong>What Optimization Does:</strong></p>
            <ul>
                <li>Analyzes historical correlations between assets</li>
                <li>Finds allocation that maximized Sharpe ratio in the PAST</li>
                <li>Assumes future correlations will be similar to historical</li>
            </ul>
            <p><strong>When to Use Optimal Allocation:</strong></p>
            <ul>
                <li>Sharpe ratio significantly higher (0.2+ improvement)</li>
                <li>Similar or better returns with lower volatility</li>
                <li>You believe historical relationships will continue</li>
            </ul>
            <p><strong>⚠️ Important Warnings:</strong></p>
            <ul>
                <li><strong>Over-optimization risk:</strong> "Perfect" historical fit may not work going forward</li>
                <li><strong>Concentration risk:</strong> Optimal allocation often concentrates in few assets</li>
                <li><strong>Turnover costs:</strong> Switching has transaction costs and tax implications</li>
                <li><strong>Rebalancing:</strong> Optimal weights change over time - requires monitoring</li>
            </ul>
            <p><strong>Conservative Approach:</strong></p>
            <ul>
                <li>If optimal Sharpe is only slightly better (< 0.2), stick with current allocation</li>
                <li>If optimal suggests 80%+ in one asset, that's too concentrated - use judgment</li>
                <li>Consider a blend: 70% optimal + 30% equal weight</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Efficient Frontier
    st.markdown("---")
    st.markdown("### 📊 Efficient Frontier")
    
    with st.spinner("Calculating efficient frontier..."):
        results, weights_array = calculate_efficient_frontier(prices, num_portfolios=500)
        
        # Current and optimal portfolio metrics
        current_annual_return = metrics['Annual Return']
        current_annual_vol = metrics['Annual Volatility']
        
        optimal_annual_return = optimal_metrics['Annual Return']
        optimal_annual_vol = optimal_metrics['Annual Volatility']
    
    fig = plot_efficient_frontier(results, optimal_weights, optimal_annual_return, optimal_annual_vol)
    
    # Add current portfolio to plot
    ax = fig.axes[0]
    ax.scatter(current_annual_vol, current_annual_return, marker='o', color='blue',
              s=400, label='Current Portfolio', edgecolors='black', linewidths=2)
    
    # Update legend
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    
    st.pyplot(fig)
    
    # Efficient frontier interpretation
    st.markdown("""
        <div class="interpretation-box">
            <div class="interpretation-title">💡 Understanding the Efficient Frontier</div>
            <p><strong>What This Chart Shows:</strong></p>
            <ul>
                <li>Each dot = A possible portfolio allocation</li>
                <li>X-axis (Volatility) = Risk</li>
                <li>Y-axis (Return) = Expected Return</li>
                <li>Color = Sharpe Ratio (brighter yellow = better)</li>
            </ul>
            <p><strong>Key Points:</strong></p>
            <ul>
                <li><strong>Blue circle:</strong> Your current portfolio</li>
                <li><strong>Red star:</strong> Optimal portfolio (highest Sharpe)</li>
                <li><strong>Upper edge:</strong> "Efficient frontier" - best return for each risk level</li>
            </ul>
            <p><strong>How to Read Your Position:</strong></p>
            <ul>
                <li><strong>Below and left of red star:</strong> You have lower risk but also lower return</li>
                <li><strong>Above and right of red star:</strong> You have higher risk for the return</li>
                <li><strong>On the frontier:</strong> You're efficient! Can't improve without changing risk</li>
                <li><strong>Below the frontier:</strong> You're inefficient - can get better returns for same risk</li>
            </ul>
            <p><strong>Action Items:</strong></p>
            <ul>
                <li>If you're far below the frontier, consider rebalancing</li>
                <li>If you're on or near the frontier, you're doing well</li>
                <li>Remember: This is based on PAST data - future may differ!</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Action Buttons
    st.markdown("---")
    st.markdown("### 🎯 Take Action")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Apply Optimal Weights", type="primary"):
            # Update current portfolio with optimal weights
            st.session_state.portfolios[st.session_state.current_portfolio]['weights'] = optimal_weights_dict
            st.session_state.portfolios[st.session_state.current_portfolio]['returns'] = optimal_returns
            st.success("✅ Optimal weights applied! Refresh to see changes in other tabs.")
            st.balloons()
    
    with col2:
        if st.button("💾 Save as New Portfolio"):
            new_name = f"{st.session_state.current_portfolio} (Optimized)"
            st.session_state.portfolios[new_name] = {
                'tickers': tickers,
                'weights': optimal_weights_dict,
                'prices': prices,
                'returns': optimal_returns,
                'start_date': current['start_date'],
                'end_date': current['end_date']
            }
            st.success(f"✅ Saved as '{new_name}'")
    
    with col3:
        # Export optimal weights
        export_weights = pd.DataFrame({
            'Ticker': list(optimal_weights_dict.keys()),
            'Weight': list(optimal_weights_dict.values())
        })
        csv = export_weights.to_csv(index=False)
        st.download_button(
            label="📥 Export Optimal Weights",
            data=csv,
            file_name="optimal_weights.csv",
            mime="text/csv"
        )




# =============================================================================
# TAB 8: TRADING SIGNALS
# =============================================================================
with tab8:
    st.markdown("# 🚦 Trading Signals")
    st.markdown("Multi-indicator trading signals with actionable recommendations")
    st.markdown("---")
    
    if 'prices' in current:
        prices = current['prices']
        tickers = current['tickers']
        
        # Generate signals for all tickers
        signals_data = []
        for ticker in tickers:
            if ticker in prices.columns:
                signal = generate_trading_signal(prices[ticker])
                signals_data.append({
                    'Ticker': ticker,
                    'Signal': signal['signal'],
                    'Action': signal['action'],
                    'Confidence': f"{signal['confidence']:.0f}%",
                    'Score': signal['score'],
                    'RSI': f"{signal['rsi']:.1f}" if not pd.isna(signal['rsi']) else 'N/A',
                    'Key Signals': ', '.join(signal['signals'][:3])
                })
        
        # Display as table
        signals_df = pd.DataFrame(signals_data)
        
        # Style the table
        def style_signal(row):
            if 'STRONG BUY' in row['Signal'] or 'BUY' in row['Signal']:
                return ['background-color: #d4edda']*len(row)
            elif 'STRONG SELL' in row['Signal'] or 'SELL' in row['Signal']:
                return ['background-color: #f8d7da']*len(row)
            else:
                return ['background-color: #fff3cd']*len(row)
        
        styled_signals = signals_df.style.apply(style_signal, axis=1)
        st.dataframe(styled_signals, use_container_width=True, hide_index=True)
        
        # Detailed breakdown for each ticker
        st.markdown("---")
        st.markdown("## 📊 Detailed Analysis")
        
        for ticker in tickers:
            if ticker in prices.columns:
                with st.expander(f"**{ticker}** - Detailed Technical Analysis"):
                    signal = generate_trading_signal(prices[ticker])
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'BUY' in signal['signal']:
                            st.success(f"**{signal['signal']}**")
                        elif 'SELL' in signal['signal']:
                            st.error(f"**{signal['signal']}**")
                        else:
                            st.info(f"**{signal['signal']}**")
                        st.metric("Confidence", f"{signal['confidence']:.0f}%")
                    
                    with col2:
                        st.metric("Score", signal['score'], help="Range: -6 (strong sell) to +6 (strong buy)")
                        st.metric("RSI", f"{signal['rsi']:.1f}" if not pd.isna(signal['rsi']) else 'N/A')
                    
                    with col3:
                        st.metric("Action", signal['action'])
                        if signal['price_vs_sma200'] is not None:
                            st.metric("vs 200 SMA", f"{signal['price_vs_sma200']:+.2f}%")
                    
                    st.markdown("**Key Signals:**")
                    for sig in signal['signals']:
                        st.markdown(f"• {sig}")
    else:
        st.info("👆 Build a portfolio first to see trading signals")


# =============================================================================
# TAB 9: TECHNICAL CHARTS (DEEP ANALYSIS)
# =============================================================================
with tab9:
    st.markdown("# 📉 Deep Technical Analysis")
    st.markdown("Comprehensive technical analysis with support/resistance levels and key moving averages")
    st.markdown("---")
    
    if 'prices' in current:
        prices = current['prices']
        tickers = current['tickers']
        
        # Ticker selection
        selected_ticker = st.selectbox("Select ETF for Deep Analysis", tickers)
        
        if selected_ticker and selected_ticker in prices.columns:
            ticker_prices = prices[selected_ticker]
            
            # Calculate all indicators
            sma_20 = calculate_sma(ticker_prices, 20)
            sma_50 = calculate_sma(ticker_prices, 50)
            sma_200 = calculate_sma(ticker_prices, 200)
            rsi = calculate_rsi(ticker_prices)
            macd, macd_signal, macd_hist = calculate_macd(ticker_prices)
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(ticker_prices)
            
            # Calculate support and resistance
            support_resistance = calculate_support_resistance(ticker_prices)
            
            # Generate trading signal
            signal = generate_trading_signal(ticker_prices)
            
            # Display overall signal
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'BUY' in signal['signal']:
                    st.success(f"**{signal['signal']}**")
                elif 'SELL' in signal['signal']:
                    st.error(f"**{signal['signal']}**")
                else:
                    st.info(f"**{signal['signal']}**")
            
            with col2:
                st.metric("Confidence", f"{signal['confidence']:.0f}%")
            
            with col3:
                st.metric("Action", signal['action'])
            
            with col4:
                st.metric("Score", signal['score'])
            
            # Key Levels Section
            st.markdown("---")
            st.markdown("## 🎯 Key Support & Resistance Levels")
            
            current_price = ticker_prices.iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🔴 Resistance Levels**")
                st.metric("Resistance 2", f"${support_resistance['resistance_2']:.2f}", 
                         f"{((support_resistance['resistance_2']/current_price - 1)*100):+.2f}%")
                st.metric("Resistance 1", f"${support_resistance['resistance_1']:.2f}",
                         f"{((support_resistance['resistance_1']/current_price - 1)*100):+.2f}%")
                st.metric("Recent High", f"${support_resistance['recent_high']:.2f}",
                         f"{((support_resistance['recent_high']/current_price - 1)*100):+.2f}%")
            
            with col2:
                st.markdown("**📍 Current Price**")
                st.metric("", f"${current_price:.2f}", help="Current market price")
                st.metric("Pivot Point", f"${support_resistance['pivot']:.2f}",
                         f"{((support_resistance['pivot']/current_price - 1)*100):+.2f}%")
            
            with col3:
                st.markdown("**🟢 Support Levels**")
                st.metric("Support 1", f"${support_resistance['support_1']:.2f}",
                         f"{((support_resistance['support_1']/current_price - 1)*100):+.2f}%")
                st.metric("Support 2", f"${support_resistance['support_2']:.2f}",
                         f"{((support_resistance['support_2']/current_price - 1)*100):+.2f}%")
                st.metric("Recent Low", f"${support_resistance['recent_low']:.2f}",
                         f"{((support_resistance['recent_low']/current_price - 1)*100):+.2f}%")
            
            # Moving Averages Analysis
            st.markdown("---")
            st.markdown("## 📏 Moving Averages (Daily Chart)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if not pd.isna(sma_20.iloc[-1]):
                    st.metric("20-Day SMA", f"${sma_20.iloc[-1]:.2f}",
                             f"{((sma_20.iloc[-1]/current_price - 1)*100):+.2f}%")
            
            with col2:
                if not pd.isna(sma_50.iloc[-1]):
                    st.metric("50-Day SMA", f"${sma_50.iloc[-1]:.2f}",
                             f"{((sma_50.iloc[-1]/current_price - 1)*100):+.2f}%")
            
            with col3:
                if not pd.isna(sma_200.iloc[-1]):
                    st.metric("200-Day SMA", f"${sma_200.iloc[-1]:.2f}",
                             f"{((sma_200.iloc[-1]/current_price - 1)*100):+.2f}%")
            
            # Price Chart with Key Levels
            st.markdown("---")
            st.markdown("## 📊 Price Chart with Technical Indicators")
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), 
                                                gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Main price chart
            ax1.plot(ticker_prices.index, ticker_prices.values, label='Price', color='black', linewidth=2)
            
            # Plot SMAs
            if not sma_20.isna().all():
                ax1.plot(sma_20.index, sma_20.values, label='20 SMA', color='blue', alpha=0.7)
            if not sma_50.isna().all():
                ax1.plot(sma_50.index, sma_50.values, label='50 SMA', color='orange', alpha=0.7)
            if not sma_200.isna().all():
                ax1.plot(sma_200.index, sma_200.values, label='200 SMA', color='red', alpha=0.7)
            
            # Plot Bollinger Bands
            ax1.plot(bb_upper.index, bb_upper.values, 'r--', alpha=0.5, label='BB Upper')
            ax1.plot(bb_lower.index, bb_lower.values, 'g--', alpha=0.5, label='BB Lower')
            ax1.fill_between(bb_upper.index, bb_lower.values, bb_upper.values, alpha=0.1)
            
            # Plot support/resistance lines (last 100 days)
            recent_idx = ticker_prices.index[-100:] if len(ticker_prices) > 100 else ticker_prices.index
            ax1.axhline(y=support_resistance['resistance_1'], color='r', linestyle=':', alpha=0.5, label='R1')
            ax1.axhline(y=support_resistance['support_1'], color='g', linestyle=':', alpha=0.5, label='S1')
            ax1.axhline(y=current_price, color='purple', linestyle='-', linewidth=2, label='Current')
            
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.set_title(f'{selected_ticker} - Daily Chart with Key Levels', fontsize=14, fontweight='bold')
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # RSI chart
            ax2.plot(rsi.index, rsi.values, label='RSI', color='purple', linewidth=2)
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7)
            ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
            ax2.fill_between(rsi.index, 70, 100, alpha=0.1, color='red')
            ax2.fill_between(rsi.index, 0, 30, alpha=0.1, color='green')
            ax2.set_ylabel('RSI', fontsize=11)
            ax2.set_ylim(0, 100)
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # MACD chart
            ax3.plot(macd.index, macd.values, label='MACD', color='blue', linewidth=2)
            ax3.plot(macd_signal.index, macd_signal.values, label='Signal', color='red', linewidth=2)
            colors = ['green' if x > 0 else 'red' for x in macd_hist.values]
            ax3.bar(macd_hist.index, macd_hist.values, color=colors, alpha=0.3, label='Histogram')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_ylabel('MACD', fontsize=11)
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Technical Summary
            st.markdown("---")
            st.markdown("## 📋 Technical Summary")
            
            summary_text = f"""
            **Current Position Analysis:**
            - Price is {'ABOVE' if current_price > sma_200.iloc[-1] else 'BELOW'} the 200-day SMA (${sma_200.iloc[-1]:.2f})
            - Distance to Resistance 1: ${support_resistance['resistance_1'] - current_price:.2f} ({((support_resistance['resistance_1']/current_price - 1)*100):.2f}%)
            - Distance to Support 1: ${current_price - support_resistance['support_1']:.2f} ({((current_price/support_resistance['support_1'] - 1)*100):.2f}%)
            
            **Trend Analysis:**
            - Short-term (20 SMA): {'Bullish ✅' if current_price > sma_20.iloc[-1] else 'Bearish ❌'}
            - Medium-term (50 SMA): {'Bullish ✅' if current_price > sma_50.iloc[-1] else 'Bearish ❌'}
            - Long-term (200 SMA): {'Bullish ✅' if current_price > sma_200.iloc[-1] else 'Bearish ❌'}
            
            **Key Signals:**
            """
            
            for sig in signal['signals']:
                summary_text += f"\n- {sig}"
            
            st.markdown(summary_text)
            
            # Recommendation
            st.markdown("---")
            st.markdown("## 💡 Recommendation")
            
            if signal['action'] == 'Accumulate':
                st.success(f"**{signal['action'].upper()}**: Technical indicators suggest this is a good time to add to positions. Consider buying on dips toward support levels.")
            elif signal['action'] == 'Distribute':
                st.error(f"**{signal['action'].upper()}**: Technical indicators suggest reducing exposure. Consider taking profits near resistance levels.")
            else:
                st.info(f"**{signal['action'].upper()}**: Signals are mixed. Wait for clearer directional confirmation before making changes.")
        
    else:
        st.info("👆 Build a portfolio first to see technical analysis")




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