"""
Tab: Trading Signals
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from helper_functions import *


def render(tab10, portfolio_returns, prices, weights, tickers, metrics, current):
    """Render the Trading Signals tab"""
    
    def normalize_action(raw_action):
        """
        Normalize all trading signal actions to Buy/Hold/Sell
        Input: Accumulate, Hold, Distribute (from generate_trading_signal)
        Output: Buy, Hold, Sell
        """
        action_mapping = {
            'Accumulate': 'Buy',
            'Hold': 'Hold',
            'Distribute': 'Sell'
        }
        return action_mapping.get(raw_action, 'Hold')
    
    with tab10:
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
                    # Pass ticker parameter for bond detection
                    signal = generate_trading_signal(prices[ticker], ticker)
                    
                    # Normalize the action
                    normalized_action = normalize_action(signal['action'])
                    
                    # Defensive: ensure signals is a list
                    sig_list = signal.get('signals', [])
                    if isinstance(sig_list, str):
                        sig_list = [sig_list]
                    elif not isinstance(sig_list, list):
                        sig_list = []
                    
                    # Join first 3 signals
                    key_signals_text = ', '.join(sig_list[:3]) if sig_list else 'N/A'
                    
                    signals_data.append({
                        'Ticker': ticker,
                        'Signal': signal['signal'],
                        'Action': normalized_action,
                        'Confidence': f"{signal['confidence']:.0f}%",
                        'Score': signal['score'],
                        'RSI': f"{signal['rsi']:.1f}" if signal.get('rsi') and not pd.isna(signal['rsi']) else 'N/A',
                        'Key Signals': key_signals_text
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
            st.dataframe(
                styled_signals,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.NumberColumn(
                        "Score",
                        help="**Score Range: -6 to +6**\n\n"
                            "Components:\n"
                            "• Trend: ±3 (most important)\n"
                            "• Momentum: ±2 (confirms trend)\n"
                            "• Extremes: ±1 (timing)\n\n"
                            "Thresholds:\n"
                            "• ≥4: STRONG BUY\n"
                            "• ≥2: BUY\n"
                            "• -2 to +2: HOLD\n"
                            "• ≤-2: SELL\n"
                            "• ≤-4: STRONG SELL"
                    ),
                    "Confidence": st.column_config.TextColumn(
                        "Confidence",
                        help="**How Confidence is Calculated:**\n\n"
                            "Base = |Score| × 15%\n"
                            "Agreement Bonus = +10% if all indicators agree\n"
                            "Total = Base + Bonus (max 100%)\n\n"
                            "**Interpretation:**\n"
                            "• 80-100%: High conviction\n"
                            "• 60-79%: Moderate conviction\n"
                            "• 40-59%: Low conviction\n"
                            "• <40%: Very low conviction"
                    )
                }
            )
            
            # Detailed breakdown for each ticker
            st.markdown("---")
            st.markdown("## 📊 Detailed Analysis")
            
            for ticker in tickers:
                if ticker in prices.columns:
                    with st.expander(f"**{ticker}** - Detailed Technical Analysis"):
                        signal = generate_trading_signal(prices[ticker],ticker)
                        col1, col2, col3 = st.columns(3)
                        
                        # COLUMN 1: Signal, Confidence, then Key Signals below
                        with col1:
                            if 'BUY' in signal['signal']:
                                st.success(f"**{signal['signal']}**")
                            elif 'SELL' in signal['signal']:
                                st.error(f"**{signal['signal']}**")
                            else:
                                st.info(f"**{signal['signal']}**")
                            
                            conf_help = (
                                "**How Confidence is Calculated:**\n\n"
                                "Base = |Score| × 15%\n"
                                "Agreement Bonus = +10% if all indicators agree\n"
                                "Total = Base + Bonus (max 100%)\n\n"
                                "**Interpretation:**\n"
                                "• 80-100%: High conviction\n"
                                "• 60-79%: Moderate conviction\n"
                                "• 40-59%: Low conviction\n"
                                "• <40%: Very low conviction"
                            )
                            st.metric("Confidence", f"{signal['confidence']:.0f}%", help=conf_help)
                            
                            # Key Signals below Confidence
                            st.markdown("---")
                            st.markdown("**Key Signals:**")
                            signal_list = signal.get('signals', [])
                            if isinstance(signal_list, str):
                                signal_list = [signal_list]
                            elif not isinstance(signal_list, list):
                                signal_list = []
                            
                            if signal_list:
                                for sig in signal_list:
                                    st.markdown(f"• {sig}")
                            else:
                                st.markdown("• No signals available")
                        
                        # COLUMN 2: Score, RSI, then Score Breakdown below
                        with col2:
                            score_help = (
                                "**Score Range: -6 to +6**\n\n"
                                "**Components:**\n"
                                "• Trend: ±3 points (most important)\n"
                                "• Momentum: ±2 points (confirms trend)\n"
                                "• Extremes: ±1 point (timing)\n\n"
                                "**Thresholds:**\n"
                                "• ≥4: STRONG BUY\n"
                                "• ≥2: BUY\n"
                                "• -2 to +2: HOLD\n"
                                "• ≤-2: SELL\n"
                                "• ≤-4: STRONG SELL"
                            )
                            st.metric("Score", signal['score'], help=score_help)
                            
                            rsi_help = "Relative Strength Index (0-100)\n• <30: Oversold\n• >70: Overbought\n• 40-60: Neutral"
                            st.metric(
                                "RSI",
                                f"{signal['rsi']:.1f}" if signal.get('rsi') and not pd.isna(signal['rsi']) else 'N/A',
                                help=rsi_help
                            )
                            
                            # Score Breakdown below RSI
                            st.markdown("---")
                            st.markdown("**📊 Score Breakdown:**")
                            
                            if 'score_breakdown' in signal and signal.get('score', 0) != 0:
                                sb = signal['score_breakdown']
                                
                                st.markdown(f"**Trend:** {sb.get('trend', 0):+.1f} *(max ±3)*")
                                st.markdown(f"**Momentum:** {sb.get('momentum', 0):+.1f} *(max ±2)*")
                                st.markdown(f"**Extremes:** {sb.get('extremes', 0):+.2f} *(max ±1)*")
                                st.markdown(f"**Total:** {sb.get('total', 0):+.1f}")
                                
                                st.markdown("")
                                st.caption("**How Calculated:**")
                                if 'computation' in sb:
                                    for comp in sb['computation'][:3]:
                                        st.caption(f"• {comp}")
                                
                                st.caption(f"**Formula:** {sb.get('formula', 'N/A')}")
                            else:
                                st.info("N/A for bonds")
                        
                        # COLUMN 3: Action, vs 200 SMA, then Confidence Breakdown below
                        with col3:
                            st.metric("Action", signal['action'])
                            
                            if signal.get('price_vs_sma200') is not None:
                                sma_help = "Price distance from 200-day moving average\n• Positive: Above (bullish)\n• Negative: Below (bearish)"
                                st.metric("vs 200 SMA", f"{signal['price_vs_sma200']:+.2f}%", help=sma_help)
                            
                            # Confidence Breakdown below vs 200 SMA
                            st.markdown("---")
                            st.markdown("**🎯 Confidence:**")
                            
                            if 'confidence_breakdown' in signal:
                                cb = signal['confidence_breakdown']
                                
                                st.markdown(f"**Base:** {cb.get('base', 0):.0f}%")
                                st.markdown(f"**Bonus:** +{cb.get('agreement_bonus', 0)}%")
                                st.markdown(f"**Total:** {cb.get('total', 0):.0f}%")
                                
                                st.markdown("")
                                st.caption(f"**Formula:** {cb.get('formula', 'N/A')}")
                                
                                # Interpretation
                                total = cb.get('total', 0)
                                if total >= 80:
                                    st.caption("🟢 High conviction")
                                elif total >= 60:
                                    st.caption("🟡 Moderate conviction")
                                elif total >= 40:
                                    st.caption("🟠 Low conviction")
                                else:
                                    st.caption("🔴 Very low conviction")
                                
                                # Show bond reasoning if applicable
                                if 'reasoning' in signal and signal.get('score', 0) == 0:
                                    st.markdown("")
                                    st.caption("**Why:**")
                                    for reason in signal.get('reasoning', [])[:3]:
                                        st.caption(f"• {reason}")
                            else:
                                st.info("N/A")
        else:
            st.info("👆 Build a portfolio first to see trading signals")
        
        

        st.markdown("---")
        st.markdown("---")
        
        # =============================================================================
        # ETF UNIVERSE TRADING SIGNALS
        # =============================================================================
        
        st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        border-radius: 15px; color: white; margin: 2rem 0;">
                <h2 style="margin: 0; font-size: 2rem;">📡 Trading Signals for the ETF Universe</h2>
                <p style="font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.9;">
                    Real-time signals for all available ETFs
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🌐 ETF Universe Signals")
        
        try:
            # Define the ETF Universe
            etf_universe = {
                '🏢 Core Market': ['SPY', 'VOO', 'IVV', 'VTI', 'ITOT'],
                '🚀 Growth/Tech': ['QQQ', 'VUG', 'VGT', 'IWF', 'SCHG', 'MGK'],
                '💰 Dividend': ['SCHD', 'VIG', 'VYM', 'DGRO', 'NOBL', 'DVY'],
                '🛡️ Bonds': ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'TIP', 'LQD', 'MUB', 'HYG', 'JNK'],
                '🌍 International': ['VEA', 'VWO', 'VXUS', 'IEFA', 'IXUS', 'EFA'],
                '🎯 Factors': ['QUAL', 'MTUM', 'VTV', 'USMV', 'SIZE', 'VLUE']
            }
            
            # Flatten the universe
            all_etfs = []
            for category, tickers_list in etf_universe.items():
                for ticker in tickers_list:
                    all_etfs.append((category, ticker))
            
            # CRITICAL: Determine data source
            # If portfolio exists, use its date range and data for consistency
            # Otherwise, download fresh data with 180-day lookback
            
            use_portfolio_data = 'prices' in current and current['prices'] is not None
            
            if use_portfolio_data:
                # Use portfolio's data source for consistency
                portfolio_prices = current['prices']
                portfolio_tickers = set(current['tickers'])
                start_date_ref = current.get('start_date')
                end_date_ref = current.get('end_date')
                
                # Info message about data source  
                if start_date_ref and end_date_ref:
                    st.info(f"📊 Using portfolio data (through {end_date_ref.strftime('%Y-%m-%d')}) for consistent signals. "
                           f"Click '🔄 Refresh Portfolio Data' in sidebar to update to today's date.")
            else:
                # No portfolio - use fresh 180-day download
                end_date_ref = datetime.now().date()
                start_date_ref = (datetime.now() - timedelta(days=180)).date()
                portfolio_prices = None
                portfolio_tickers = set()
                st.info(f"📊 Using 180-day lookback ({start_date_ref.strftime('%Y-%m-%d')} to {end_date_ref.strftime('%Y-%m-%d')})")
            
            # Download and process
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            signals_data = []
            errors = []
            total_etfs = len(all_etfs)
            
            for idx, (category, ticker) in enumerate(all_etfs):
                progress_pct = (idx + 1) / total_etfs
                progress_bar.progress(progress_pct)
                status_text.text(f"Processing {ticker}... ({idx + 1}/{total_etfs})")
                
                try:
                    # CRITICAL: Use portfolio data if ticker is in portfolio
                    if use_portfolio_data and ticker in portfolio_tickers and ticker in portfolio_prices.columns:
                        # Use SAME data as portfolio signals for consistency
                        etf_prices = portfolio_prices[ticker].dropna()
                    else:
                        # Download fresh data (not in portfolio or no portfolio exists)
                        etf_data = download_ticker_data([ticker], start_date_ref, end_date_ref)
                        
                        if etf_data is None or etf_data.empty:
                            errors.append(f"{ticker}: No data")
                            continue
                        
                        if ticker not in etf_data.columns:
                            errors.append(f"{ticker}: Not in columns")
                            continue
                        
                        etf_prices = etf_data[ticker].dropna()
                    
                    if len(etf_prices) <= 50:
                        errors.append(f"{ticker}: Insufficient data")
                        continue
                    
                    # Generate signal using SAME function and SAME data
                    signal = generate_trading_signal(etf_prices, ticker)
                    
                    if signal is None:
                        errors.append(f"{ticker}: Signal generation failed")
                        continue
                    
                    # Normalize the action
                    normalized_action = normalize_action(signal.get('action', 'Hold'))
                    
                    score = signal.get('score', 0)
                    confidence = signal.get('confidence', 0)
                    current_price = etf_prices.iloc[-1]
                    
                    # Determine emoji and color
                    if normalized_action == 'Buy':
                        action_emoji, action_color = '🟢', '#28a745'
                    elif normalized_action == 'Sell':
                        action_emoji, action_color = '🔴', '#dc3545'
                    else:
                        action_emoji, action_color = '🟡', '#ffc107'
                    
                    signals_data.append({
                        'Category': category,
                        'Ticker': ticker,
                        'Action': normalized_action,
                        'Action_Display': f"{action_emoji} {normalized_action}",
                        'Score': score,
                        'Confidence': f"{confidence:.0f}%",
                        'Price': f"${current_price:.2f}",
                        'Action_Color': action_color
                    })
                    
                except Exception as e:
                    errors.append(f"{ticker}: {str(e)[:50]}")
                    continue
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            # Show errors if any
            if len(errors) > 0:
                with st.expander(f"⚠️ {len(errors)} ETFs had issues (click to view)"):
                    for error in errors[:20]:
                        st.text(error)
            
            # Display table if we have signals
            if signals_data:
                signals_df = pd.DataFrame(signals_data)
                
                # VERIFICATION: Compare with portfolio signals if available
                if use_portfolio_data:
                    st.markdown("---")
                    st.markdown("### ✅ Signal Verification")
                    
                    # Get portfolio signals for comparison
                    portfolio_signals = {}
                    if 'prices' in current:
                        portfolio_prices_ref = current['prices']
                        for ticker in current['tickers']:
                            if ticker in portfolio_prices_ref.columns:
                                sig = generate_trading_signal(portfolio_prices_ref[ticker], ticker)
                                portfolio_signals[ticker] = {
                                    'action': normalize_action(sig['action']),
                                    'score': sig['score']
                                }
                    
                    # Find overlapping tickers
                    etf_universe_tickers = set(signals_df['Ticker'].unique())
                    overlapping = etf_universe_tickers.intersection(portfolio_signals.keys())
                    
                    if overlapping:
                        st.info(f"📊 Comparing signals for {len(overlapping)} tickers in both Portfolio and ETF Universe")
                        
                        verification_data = []
                        mismatches = 0
                        
                        for ticker in sorted(overlapping):
                            portfolio_action = portfolio_signals[ticker]['action']
                            portfolio_score = portfolio_signals[ticker]['score']
                            
                            etf_row = signals_df[signals_df['Ticker'] == ticker].iloc[0]
                            etf_action = etf_row['Action']
                            etf_score = etf_row['Score']
                            
                            match_status = "✅ Match" if portfolio_action == etf_action else "⚠️ MISMATCH"
                            if portfolio_action != etf_action:
                                mismatches += 1
                            
                            verification_data.append({
                                'Ticker': ticker,
                                'Portfolio Signal': portfolio_action,
                                'Portfolio Score': portfolio_score,
                                'ETF Universe Signal': etf_action,
                                'ETF Universe Score': etf_score,
                                'Status': match_status
                            })
                        
                        verification_df = pd.DataFrame(verification_data)
                        st.dataframe(verification_df, use_container_width=True, hide_index=True)
                        
                        if mismatches == 0:
                            st.success(f"✅ Perfect! All {len(overlapping)} signals match between Portfolio and ETF Universe")
                        else:
                            st.error(f"⚠️ {mismatches} mismatches found! Signals should be identical for same tickers.")
                    
                    st.markdown("---")
                
                st.markdown("#### 🔍 Filter & Sort")
                col1, col2 = st.columns(2)
                
                with col1:
                    filter_action = st.multiselect(
                        "Filter by Action:",
                        ['Buy', 'Hold', 'Sell'],
                        default=['Buy', 'Hold', 'Sell'],
                        key='etf_universe_filter'
                    )
                
                with col2:
                    sort_by = st.selectbox(
                        "Sort by:",
                        ['Action', 'Score', 'Confidence', 'Ticker'],
                        key='etf_universe_sort'
                    )
                
                filtered_df = signals_df[signals_df['Action'].isin(filter_action)]
                
                if sort_by == 'Action':
                    action_order = {'Buy': 0, 'Hold': 1, 'Sell': 2}
                    filtered_df['sort_key'] = filtered_df['Action'].map(action_order)
                    filtered_df = filtered_df.sort_values('sort_key').drop('sort_key', axis=1)
                elif sort_by == 'Score':
                    filtered_df = filtered_df.sort_values('Score', ascending=False)
                elif sort_by == 'Confidence':
                    filtered_df['conf_num'] = filtered_df['Confidence'].str.rstrip('%').astype(float)
                    filtered_df = filtered_df.sort_values('conf_num', ascending=False).drop('conf_num', axis=1)
                else:
                    filtered_df = filtered_df.sort_values('Ticker')
                
                st.markdown(f"#### 📊 Showing {len(filtered_df)} ETFs")
                
                display_df = filtered_df[['Category', 'Ticker', 'Action_Display', 'Score', 'Confidence', 'Price']].copy()
                display_df.columns = ['Category', 'Ticker', 'Signal', 'Score', 'Confidence', 'Current Price']
                
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)
                
                # Summary
                st.markdown("---")
                st.markdown("### 📈 Signal Summary")
                
                col1, col2, col3 = st.columns(3)
                buy_count = len(signals_df[signals_df['Action'] == 'Buy'])
                hold_count = len(signals_df[signals_df['Action'] == 'Hold'])
                sell_count = len(signals_df[signals_df['Action'] == 'Sell'])
                
                with col1:
                    st.metric("🟢 Buy Signals", buy_count)
                with col2:
                    st.metric("🟡 Hold Signals", hold_count)
                with col3:
                    st.metric("🔴 Sell Signals", sell_count)
                
                # Top Opportunities
                st.markdown("---")
                st.markdown("### 🎯 Top Opportunities")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🟢 Top Buy Signals (Highest Score)**")
                    buy_signals = signals_df[signals_df['Action'] == 'Buy'].sort_values('Score', ascending=False).head(5)
                    if not buy_signals.empty:
                        for _, row in buy_signals.iterrows():
                            st.markdown(f"• **{row['Ticker']}** - Score: {row['Score']}, Confidence: {row['Confidence']}")
                    else:
                        st.info("No Buy signals currently")
                
                with col2:
                    st.markdown("**🔴 Top Sell Signals (Lowest Score)**")
                    sell_signals = signals_df[signals_df['Action'] == 'Sell'].sort_values('Score', ascending=True).head(5)
                    if not sell_signals.empty:
                        for _, row in sell_signals.iterrows():
                            st.markdown(f"• **{row['Ticker']}** - Score: {row['Score']}, Confidence: {row['Confidence']}")
                    else:
                        st.info("No Sell signals currently")
                
                st.markdown("---")
                st.info("💡 **Note:** These signals are based on technical analysis. Always do your own research and consider your investment goals, risk tolerance, and time horizon.")
            
            else:
                st.error("❌ Unable to generate signals for ETF universe.")
                st.markdown("**Possible causes:**")
                st.markdown("- Internet connection issues")
                st.markdown("- yfinance API rate limiting")
                if len(errors) > 0:
                    st.markdown(f"- Check the error expander above for details ({len(errors)} errors)")
        
        except Exception as e:
            st.error(f"❌ Error in ETF Universe section: {str(e)}")
            import traceback
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())

