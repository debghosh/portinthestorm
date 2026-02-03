"""
Tab: Technical Charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from helper_functions import *


def render(tab11, portfolio_returns, prices, weights, tickers, metrics, current):
    """Render the Technical Charts tab"""
    
    with tab11:
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
                    # Defensive code: ensure signals is always a list
                    signal_list = signal.get('signals', [])
        
                    # Check if signals is a string (BUG) instead of list
                    if isinstance(signal_list, str):
                        # Wrap string in a list so it displays as one item
                        signal_list = [signal_list]
                    elif not isinstance(signal_list, list):
                        # Not a string or list - convert to empty list
                        signal_list = []
        
                    # Display signals
                    if signal_list:
                        for sig in signal_list:
                            # Each sig should be a string like "Price above 50 SMA"
                            # NOT individual characters
                            st.markdown(f"• {sig}")
                    else:
                        st.markdown("• No signals available")
                    #for sig in signal['signal']:
                    #    summary_text += f"\n- {sig}"
                    
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
                # Portfolio exists - get data and show analysis
                current = st.session_state.portfolios[st.session_state.current_portfolio]
                portfolio_returns = current['returns']
                prices = current['prices']
                weights = current['weights']
                tickers = current['tickers']
                metrics = calculate_portfolio_metrics(portfolio_returns)
        
        
        # =============================================================================
        # FOOTER
        # =============================================================================
        
