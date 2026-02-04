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
import plotly.graph_objects as go
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

# =============================================================================
# FIXED TRADING SIGNAL GENERATION - Internally Consistent
# Replace the generate_trading_signal function (around line 93-197)
# =============================================================================

# =============================================================================
# FINAL COMPLETE FIX - Trading Signals
# =============================================================================
# This file contains:
# 1. Updated function signature to accept ticker explicitly
# 2. Bond detection logic
# 3. Fixed signal display (returns list, not string)
# 4. More conservative thresholds
# =============================================================================

# =============================================================================
# FIXED SCORING LOGIC - Stays Within -6 to +6 Range
# Replace the entire generate_trading_signal function (lines 93-197)
# =============================================================================

def generate_trading_signal(prices, ticker=None):
    """
    Generate trading signal with proper scoring that stays within -6 to +6 range
    
    Scoring System (Maximum ±6):
    - Trend: ±3 points (most important)
    - Momentum: ±2 points (confirms trend)
    - Extremes: ±1 point (timing)
    Total: -6 to +6
    """
    
    # Get ticker from parameter or series name
    if ticker is None:
        ticker = prices.name if hasattr(prices, 'name') and prices.name else 'Unknown'
    
    ticker = str(ticker).upper()
    
    # =============================================================================
    # BOND ETF DETECTION
    # =============================================================================
    
    BOND_ETFS = ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'TIP', 'LQD', 'MUB', 
                 'HYG', 'JNK', 'VCIT', 'VCSH', 'BIV', 'BSV', 'VGIT', 'VGSH']
    
    if ticker in BOND_ETFS:
        return generate_bond_signal(prices, ticker)
    
    # =============================================================================
    # CALCULATE INDICATORS
    # =============================================================================
    
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
    
    # =============================================================================
    # SCORING COMPONENT 1: TREND (Maximum ±3 points)
    # =============================================================================
    
    trend_score = 0
    trend_signals = []
    trend_computation = []
    
    if not pd.isna(sma_50.iloc[-1]) and not pd.isna(sma_200.iloc[-1]):
        price_above_50 = current_price > sma_50.iloc[-1]
        price_above_200 = current_price > sma_200.iloc[-1]
        sma50_above_200 = sma_50.iloc[-1] > sma_200.iloc[-1]
        
        if price_above_50 and price_above_200 and sma50_above_200:
            trend_score = 3
            trend_signals.append("Price > 50 SMA > 200 SMA (Strong Uptrend)")
            trend_computation.append("Trend: Price > 50 SMA > 200 SMA = +3 points")
        elif price_above_200:
            trend_score = 2
            trend_signals.append("Price above 200 SMA (Uptrend)")
            trend_computation.append("Trend: Price > 200 SMA = +2 points")
        elif not price_above_50 and not price_above_200 and not sma50_above_200:
            trend_score = -3
            trend_signals.append("Price < 50 SMA < 200 SMA (Strong Downtrend)")
            trend_computation.append("Trend: Price < 50 SMA < 200 SMA = -3 points")
        elif not price_above_200:
            trend_score = -2
            trend_signals.append("Price below 200 SMA (Downtrend)")
            trend_computation.append("Trend: Price < 200 SMA = -2 points")
        else:
            trend_score = 0
            trend_signals.append("Mixed trend signals")
            trend_computation.append("Trend: Mixed signals = 0 points")
    else:
        trend_signals.append("Insufficient data for trend")
        trend_computation.append("Trend: Insufficient data = 0 points")
    
    # =============================================================================
    # SCORING COMPONENT 2: MOMENTUM (Maximum ±2 points)
    # =============================================================================
    
    momentum_score = 0
    momentum_signals = []
    momentum_computation = []
    
    macd_bullish = current_macd > current_macd_signal
    
    if current_macd > current_macd_signal and prev_macd_hist < 0 < current_macd_hist:
        momentum_score = 2
        momentum_signals.append("MACD bullish crossover")
        momentum_computation.append("Momentum: MACD bullish crossover = +2 points")
    elif current_macd < current_macd_signal and prev_macd_hist > 0 > current_macd_hist:
        momentum_score = -2
        momentum_signals.append("MACD bearish crossover")
        momentum_computation.append("Momentum: MACD bearish crossover = -2 points")
    elif macd_bullish:
        momentum_score = 1
        momentum_signals.append("MACD bullish")
        momentum_computation.append("Momentum: MACD bullish = +1 point")
    else:
        momentum_score = -1
        momentum_signals.append("MACD bearish")
        momentum_computation.append("Momentum: MACD bearish = -1 point")
    
    # =============================================================================
    # SCORING COMPONENT 3: EXTREMES (Maximum ±1 point)
    # =============================================================================
    
    extreme_score = 0
    extreme_signals = []
    extreme_computation = []
    
    # RSI component (max ±0.5)
    if current_rsi < 30:
        rsi_component = 0.5
        extreme_signals.append(f"RSI oversold ({current_rsi:.1f})")
        extreme_computation.append(f"RSI: {current_rsi:.1f} < 30 (oversold) = +0.5 points")
    elif current_rsi > 70:
        rsi_component = -0.5
        extreme_signals.append(f"RSI overbought ({current_rsi:.1f})")
        extreme_computation.append(f"RSI: {current_rsi:.1f} > 70 (overbought) = -0.5 points")
    elif current_rsi < 40:
        rsi_component = 0.25
        extreme_signals.append(f"RSI bullish lean ({current_rsi:.1f})")
        extreme_computation.append(f"RSI: {current_rsi:.1f} < 40 = +0.25 points")
    elif current_rsi > 60:
        rsi_component = -0.25
        extreme_signals.append(f"RSI bearish lean ({current_rsi:.1f})")
        extreme_computation.append(f"RSI: {current_rsi:.1f} > 60 = -0.25 points")
    else:
        rsi_component = 0
        extreme_signals.append(f"RSI neutral ({current_rsi:.1f})")
        extreme_computation.append(f"RSI: {current_rsi:.1f} (neutral) = 0 points")
    
    extreme_score += rsi_component
    
    # Bollinger Band component (max ±0.5)
    if current_price < bb_lower.iloc[-1]:
        bb_component = 0.5
        extreme_signals.append("Price below lower Bollinger Band")
        extreme_computation.append("Bollinger: Below lower band = +0.5 points")
    elif current_price > bb_upper.iloc[-1]:
        bb_component = -0.5
        extreme_signals.append("Price above upper Bollinger Band")
        extreme_computation.append("Bollinger: Above upper band = -0.5 points")
    else:
        bb_component = 0
        extreme_computation.append("Bollinger: Within bands = 0 points")
    
    extreme_score += bb_component
    
    # =============================================================================
    # TOTAL SCORE (Sum of all components: -6 to +6)
    # =============================================================================
    
    total_score = trend_score + momentum_score + extreme_score
    
    # Build computation breakdown
    score_breakdown = {
        'trend': trend_score,
        'momentum': momentum_score,
        'extremes': round(extreme_score, 2),
        'total': round(total_score, 2),
        'computation': trend_computation + momentum_computation + extreme_computation,
        'formula': f"Total = Trend({trend_score}) + Momentum({momentum_score}) + Extremes({extreme_score:.2f}) = {total_score:.2f}"
    }
    
    # =============================================================================
    # DETERMINE SIGNAL BASED ON SCORE
    # =============================================================================
    
    if total_score >= 4:
        signal = "STRONG BUY"
        action = "Accumulate"
    elif total_score >= 2:
        signal = "BUY"
        action = "Accumulate"
    elif total_score >= 0.5:
        signal = "HOLD (Bullish)"
        action = "Hold"
    elif total_score <= -4:
        signal = "STRONG SELL"
        action = "Distribute"
    elif total_score <= -2:
        signal = "SELL"
        action = "Distribute"
    elif total_score <= -0.5:
        signal = "HOLD (Bearish)"
        action = "Hold"
    else:
        signal = "HOLD"
        action = "Hold"
    
    # =============================================================================
    # CALCULATE CONFIDENCE (0-100%)
    # =============================================================================
    
    # Confidence based on:
    # 1. Score magnitude (stronger signal = higher confidence)
    # 2. Agreement between components (all bullish/bearish = higher confidence)
    
    # Base confidence from score magnitude
    base_confidence = min(abs(total_score) * 15, 100)
    
    # Agreement bonus: if all components point same direction
    components = [trend_score, momentum_score, extreme_score]
    all_positive = all(c > 0 for c in components if c != 0)
    all_negative = all(c < 0 for c in components if c != 0)
    
    if all_positive or all_negative:
        agreement_bonus = 10
    else:
        agreement_bonus = 0
    
    confidence = min(base_confidence + agreement_bonus, 100)
    
    confidence_breakdown = {
        'base': base_confidence,
        'agreement_bonus': agreement_bonus,
        'total': confidence,
        'formula': f"Confidence = min(|Score| × 15, 100) + Agreement Bonus = {base_confidence:.0f}% + {agreement_bonus}% = {confidence:.0f}%"
    }
    
    # Combine all signals
    all_signals = trend_signals + momentum_signals + extreme_signals
    
    # =============================================================================
    # RETURN RESULTS
    # =============================================================================
    
    return {
        'signal': signal,
        'action': action,
        'score': round(total_score, 1),
        'score_breakdown': score_breakdown,
        'confidence': confidence,
        'confidence_breakdown': confidence_breakdown,
        'signals': all_signals,
        'rsi': current_rsi,
        'macd': current_macd,
        'macd_signal': current_macd_signal,
        'price_vs_sma50': ((current_price / sma_50.iloc[-1]) - 1) * 100 if not pd.isna(sma_50.iloc[-1]) else None,
        'price_vs_sma200': ((current_price / sma_200.iloc[-1]) - 1) * 100 if not pd.isna(sma_200.iloc[-1]) else None
    }


# =============================================================================
# ENHANCED BOND SIGNAL LOGIC
# Replace the generate_bond_signal function with this enhanced version
# =============================================================================

def generate_bond_signal(prices, ticker):
    """
    Enhanced bond logic with proper confidence levels
    """
    
    # Calculate indicators
    sma_200 = calculate_sma(prices, 200)
    current_price = prices.iloc[-1]
    
    if len(prices) >= 60:
        recent_60d_return = (current_price / prices.iloc[-60] - 1) * 100
    else:
        recent_60d_return = 0
    
    signals_list = []
    
    # Bond type
    if ticker in ['AGG', 'BND', 'LQD']:
        bond_type = "Aggregate/Core"
    elif ticker in ['TLT', 'IEF']:
        bond_type = "Long-term Treasury"
    elif ticker in ['HYG', 'JNK']:
        bond_type = "High Yield Corporate"
    elif ticker in ['SHY', 'VCSH']:
        bond_type = "Short-term"
    elif ticker == 'TIP':
        bond_type = "Inflation-Protected"
    else:
        bond_type = "Bond"
    
    signals_list.append(f"{bond_type} - {ticker}")
    
    # Price trend
    if not pd.isna(sma_200.iloc[-1]):
        if current_price > sma_200.iloc[-1]:
            signals_list.append("Price above 200-day average")
        else:
            signals_list.append("Price below 200-day average")
    
    # Recent performance
    if abs(recent_60d_return) < 2:
        signals_list.append("Flat recent performance")
    else:
        signals_list.append(f"{'Up' if recent_60d_return > 0 else 'Down'} {abs(recent_60d_return):.1f}% over 60 days")
    
    # =============================================================================
    # BOND-SPECIFIC LOGIC WITH PROPER CONFIDENCE
    # =============================================================================
    
    # CORE BONDS (AGG, BND, LQD) - Always HOLD with HIGH confidence
    if ticker in ['AGG', 'BND', 'LQD']:
        signal = "HOLD"
        action = "Hold"
        confidence = 95  # HIGH - we're CERTAIN this should be held
        recommendation = "Hold for portfolio stability. Not a trading position - this is permanent ballast."
        
        reasoning = [
            "Core bonds are 20-40% of balanced portfolio",
            "Provides stability when stocks decline",
            "Rebalance only when allocation drifts significantly",
            "Never trade - permanent diversification holding"
        ]
    
    # TACTICAL TREASURIES (TLT, IEF)
    elif ticker in ['TLT', 'IEF']:
        trend_positive = current_price > sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) else None
        
        if trend_positive and recent_60d_return > 3:
            signal = "BUY"
            action = "Accumulate"
            confidence = 75
            recommendation = "Interest rates falling - bond prices rising. Tactical buy."
            reasoning = [
                "Rates declining benefits long-duration bonds",
                "Use 10-15% allocation as rate hedge",
                "Monitor Fed policy for reversal"
            ]
        elif trend_positive and recent_60d_return > 0:
            signal = "HOLD"
            action = "Hold"
            confidence = 65
            recommendation = "Uptrend intact. Hold current position."
            reasoning = ["Positive trend but watch Fed policy", "Good rate hedge"]
        elif not trend_positive and recent_60d_return < -3:
            signal = "SELL"
            action = "Distribute"
            confidence = 75
            recommendation = "Interest rates rising - bond prices falling. Reduce exposure."
            reasoning = [
                "Rising rates hurt long-duration bonds",
                "Consider shorter-duration alternatives"
            ]
        else:
            signal = "HOLD"
            action = "Hold"
            confidence = 50
            recommendation = "Mixed signals. Hold or wait for clarity."
            reasoning = ["Unclear rate direction"]
    
    # HIGH YIELD (HYG, JNK) - Trade like stocks
    elif ticker in ['HYG', 'JNK']:
        trend_positive = current_price > sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) else None
        
        if trend_positive and recent_60d_return > 5:
            signal = "BUY"
            action = "Accumulate"
            confidence = 70
            recommendation = "High yield strong - credit spreads tight. Risk-on."
            reasoning = [
                "Strong economy = tight credit spreads",
                "Limit to 5-10% allocation (still risky)"
            ]
        elif not trend_positive and recent_60d_return < -3:
            signal = "SELL"
            action = "Distribute"
            confidence = 80
            recommendation = "High yield weakness - recession risk. Exit."
            reasoning = [
                "Widening credit spreads signal recession",
                "High yield crashes in downturns (-20% to -30%)",
                "Switch to quality bonds (AGG/TLT)"
            ]
        else:
            signal = "HOLD"
            action = "Hold"
            confidence = 55
            recommendation = "Monitor for clear trend."
            reasoning = ["High yield is volatile - wait for clarity"]
    
    # SHORT-TERM (SHY) - Always HOLD with moderate-high confidence
    elif ticker in ['SHY', 'VCSH']:
        signal = "HOLD"
        action = "Hold"
        confidence = 85  # High - minimal uncertainty
        recommendation = "Short-term bonds as cash alternative. Minimal rate risk."
        reasoning = [
            "Short duration = minimal volatility",
            "Use for cash allocation",
            "Good when rates rising"
        ]
    
    # TIPS (Inflation-protected)
    elif ticker == 'TIP':
        if recent_60d_return > 3:
            signal = "BUY"
            action = "Accumulate"
            confidence = 65
            recommendation = "Inflation expectations rising. TIPS provide protection."
            reasoning = ["Principal adjusts with CPI", "Use 10-15% in inflationary periods"]
        else:
            signal = "HOLD"
            action = "Hold"
            confidence = 75  # High confidence for holding
            recommendation = "Hold for inflation protection."
            reasoning = ["Real yield protection", "Hedge for inflation uncertainty"]
    
    # Default
    else:
        signal = "HOLD"
        action = "Hold"
        confidence = 70
        recommendation = "Hold for bond allocation."
        reasoning = ["Bond allocation for diversification"]
    
    return {
        'signal': signal,
        'action': action,
        'score': 0,
        'score_breakdown': {
            'formula': f'{bond_type} bonds - not scored (use different logic)',
        },
        'confidence': confidence,
        'confidence_breakdown': {
            'formula': f'Confidence based on certainty of {signal} recommendation',
            'base': confidence,
            'agreement_bonus': 0,
            'total': confidence
        },
        'signals': signals_list,
        'recommendation': recommendation,
        'reasoning': reasoning,
        'rsi': None,
        'macd': None,
        'macd_signal': None,
        'price_vs_sma50': None,
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
    Plot market regime timeline with returns AND risk
    Dual-axis: Left = Cumulative Return, Right = Rolling Volatility
    """
    # Ensure returns is a Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
    
    # HIGH CONTRAST Color map for regimes
    regime_colors = {
        'Bull Market (Low Vol)': '#00C851',      # Bright green
        'Bull Market (High Vol)': '#007bff',     # Bright blue
        'Sideways/Choppy': '#ffbb33',            # Bright yellow/orange
        'Bear Market (Low Vol)': '#ff8800',      # Bright orange
        'Bear Market (High Vol)': '#ff4444'      # Bright red
    }
    
    # Calculate cumulative returns and rolling volatility
    cum_returns = (1 + returns).cumprod()
    rolling_vol = returns.rolling(60).std() * np.sqrt(252) * 100  # Annualized, as percentage
    
    # Get the full Y-axis range for returns
    y_min = cum_returns.min() * 0.95
    y_max = cum_returns.max() * 1.05
    
    # Plot regime backgrounds FIRST (behind everything) - FULL HEIGHT
    regimes_present = set()
    for regime, color in regime_colors.items():
        mask = regimes == regime
        if mask.any():
            regimes_present.add(regime)
            # Fill from bottom to top of the ENTIRE chart
            ax1.fill_between(returns.index, y_min, y_max, 
                           where=mask, alpha=0.25, color=color,
                           zorder=1)  # Behind everything
    
    # Plot cumulative returns (LEFT Y-AXIS)
    line1 = ax1.plot(cum_returns.index, cum_returns.values, linewidth=3, 
                     color='#000000', label='Portfolio Value', zorder=10)
    
    ax1.set_ylabel('Cumulative Return', fontsize=13, fontweight='bold', color='#000000')
    ax1.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax1.set_ylim(y_min, y_max)
    ax1.tick_params(axis='y', labelcolor='#000000')
    ax1.grid(True, alpha=0.3, linestyle='--', zorder=2)
    ax1.set_facecolor('#ffffff')
    
    # Create second Y-axis for VOLATILITY (RIGHT Y-AXIS)
    ax2 = ax1.twinx()
    line2 = ax2.plot(rolling_vol.index, rolling_vol.values, linewidth=2.5,
                     color='#dc3545', label='Rolling Volatility (60d)', 
                     linestyle='--', alpha=0.8, zorder=9)
    
    ax2.set_ylabel('Rolling Volatility (Annualized %)', fontsize=13, fontweight='bold', color='#dc3545')
    ax2.tick_params(axis='y', labelcolor='#dc3545')
    
    # Title
    ax1.set_title('Portfolio Performance: Return & Risk Across Market Regimes', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Create comprehensive legend
    # Portfolio and volatility lines
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    
    # Add spacer
    lines.append(plt.Line2D([0], [0], linewidth=0))
    labels.append('Market Regimes:')
    
    # Add regime colors (only those present)
    for regime, color in regime_colors.items():
        if regime in regimes_present:
            lines.append(plt.Rectangle((0,0),1,1, fc=color, alpha=0.4, ec='none'))
            labels.append(regime)
    
    # Place legend
    ax1.legend(lines, labels, loc='upper left', 
              frameon=True, shadow=True, fontsize=9, ncol=1,
              fancybox=True, framealpha=0.95)
    
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

