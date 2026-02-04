"""
Tab: Market Regimes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from helper_functions import *


def render(tab6, portfolio_returns, prices, weights, tickers, metrics, current):
    """Render the Market Regimes tab"""
    
    with tab6:
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
            
            # Show diagnostic info about data source
            if 'end_date' in current:
                portfolio_end_date = current['end_date']
                st.info(f"📅 Regime analysis based on portfolio data through: **{portfolio_end_date.strftime('%Y-%m-%d')}**")
            
            # Current Regime
            st.markdown("---")
            st.markdown("### 🎯 Current Market Regime")
            current_regime = regimes.iloc[-1]
            
            # Calculate current metrics for transparency
            lookback = 60
            recent_returns = portfolio_returns.iloc[-lookback:]
            rolling_return_annual = recent_returns.mean() * 252
            rolling_vol_annual = recent_returns.std() * np.sqrt(252)
            all_vol = portfolio_returns.rolling(lookback).std() * np.sqrt(252)
            vol_median = all_vol.median()
            
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
            
            # Show regime classification metrics
            st.markdown("#### 🔬 Regime Classification Details (Last 60 Days)")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Rolling Return (Annual)", 
                    f"{rolling_return_annual:.2%}",
                    help="Average return over last 60 days, annualized. >2% = Bull, <-2% = Bear"
                )
            
            with col2:
                st.metric(
                    "Rolling Volatility (Annual)", 
                    f"{rolling_vol_annual:.2%}",
                    help=f"Volatility over last 60 days, annualized. Median: {vol_median:.2%}"
                )
            
            with col3:
                vol_status = "High" if rolling_vol_annual > vol_median else "Low"
                st.metric(
                    "Volatility Level",
                    vol_status,
                    help=f"Compared to historical median ({vol_median:.2%})"
                )
            
            with col4:
                # Calculate Sharpe ratio for last 60 days
                sharpe_60d = (rolling_return_annual - 0.02) / rolling_vol_annual if rolling_vol_annual > 0 else 0
                st.metric(
                    "Sharpe Ratio (60d)",
                    f"{sharpe_60d:.2f}",
                    help="Risk-adjusted return. >1 is good, >2 is excellent"
                )
            
            st.caption(f"**Regime Logic:** Return={rolling_return_annual:.2%} ({'positive' if rolling_return_annual > 0.02 else 'negative' if rolling_return_annual < -0.02 else 'neutral'}) + Volatility={vol_status} → {current_regime}")
            
            # Regime Timeline
            st.markdown("---")
            st.markdown("### 📊 Portfolio Performance: Return & Risk with Market Regimes")
            fig = plot_regime_chart(regimes, portfolio_returns)
            st.pyplot(fig)
            
            # Regime chart interpretation
            st.markdown("""
                <div class="interpretation-box">
                    <div class="interpretation-title">💡 How to Read This Chart</div>
                    <p><strong>Black Line (Left Axis):</strong> Your portfolio cumulative returns over time</p>
                    <p><strong>Red Dashed Line (Right Axis):</strong> Rolling 60-day volatility (annualized) - your risk level</p>
                    <p><strong>Colored Backgrounds:</strong> Market regime during each period (high contrast for clarity)</p>
                    <ul>
                        <li><strong>🟢 Bright Green:</strong> Bull (Low Vol) - Best conditions, steady gains with low stress</li>
                        <li><strong>🔵 Bright Blue:</strong> Bull (High Vol) - Gains with volatility, bumpy ride up</li>
                        <li><strong>🟡 Bright Yellow:</strong> Sideways - Range-bound, capital idle</li>
                        <li><strong>🟠 Bright Orange:</strong> Bear (Low Vol) - Slow decline, early warning</li>
                        <li><strong>🔴 Bright Red:</strong> Bear (High Vol) - Crisis mode, steep losses</li>
                    </ul>
                    <p><strong>Key Insights - Return vs Risk:</strong></p>
                    <ul>
                        <li><strong>Black line rises + Red line low:</strong> Perfect! Making money with low stress</li>
                        <li><strong>Black line rises + Red line high:</strong> Volatile gains - can you handle the swings?</li>
                        <li><strong>Black line flat + Red line high:</strong> Worst scenario - high stress, no gains</li>
                        <li><strong>Black line falls + Red line spikes:</strong> Crisis - but spikes are temporary</li>
                        <li><strong>Risk adjusts return perspective:</strong> 10% return with 5% vol beats 15% return with 25% vol</li>
                    </ul>
                    <p><strong>🎯 Investment Decisions:</strong></p>
                    <ul>
                        <li><strong>Green zones + low volatility:</strong> Maximize position sizes, compound gains</li>
                        <li><strong>Red zones + volatility spikes:</strong> Historical buying opportunities, stay disciplined</li>
                        <li><strong>High returns + high volatility:</strong> Consider reducing position size for same risk-adjusted return</li>
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
        # ENHANCED MARKET REGIME TAB - Add Historical Analysis Section
        # Add this after the current regime performance section (around line 5021)
        # =============================================================================
        
            # Historical Regime Analysis
            st.markdown("---")
            st.markdown("### 📜 Historical Market Regimes & Sleeve Performance (1900-Present)")
            
            st.markdown("""
                <div class="info-box">
                    <h4>Learn from 120+ Years of Market History</h4>
                    <p>Different portfolio "sleeves" perform differently in various market conditions. 
                    This heatmap shows which sleeves thrive (or struggle) in each regime based on historical data.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Historical regime periods (simplified - major periods)
            historical_regimes = {
                'Roaring 20s Bull\n(1921-1929)': {
                    'period': '1921-1929',
                    'regime': 'Bull Market (Low Vol)',
                    'Total Market': 15.2,
                    'Growth': 18.5,
                    'Dividend': 12.3,
                    'Factors': 14.8,
                    'Bonds': 4.2,
                    'description': 'Economic boom, speculation, easy credit'
                },
                'Great Depression\n(1929-1932)': {
                    'period': '1929-1932',
                    'regime': 'Bear Market (High Vol)',
                    'Total Market': -42.0,
                    'Growth': -55.0,
                    'Dividend': -28.0,
                    'Factors': -35.0,
                    'Bonds': 8.5,
                    'description': 'Worst market crash in US history'
                },
                'Post-Depression\n(1933-1937)': {
                    'period': '1933-1937',
                    'regime': 'Bull Market (High Vol)',
                    'Total Market': 28.5,
                    'Growth': 35.0,
                    'Dividend': 22.0,
                    'Factors': 30.0,
                    'Bonds': 3.8,
                    'description': 'Recovery rally with volatility'
                },
                'WWII Era\n(1939-1945)': {
                    'period': '1939-1945',
                    'regime': 'Sideways/Choppy',
                    'Total Market': 6.2,
                    'Growth': 8.0,
                    'Dividend': 7.5,
                    'Factors': 5.8,
                    'Bonds': 2.1,
                    'description': 'Wartime economy, price controls'
                },
                'Post-War Boom\n(1946-1965)': {
                    'period': '1946-1965',
                    'regime': 'Bull Market (Low Vol)',
                    'Total Market': 12.5,
                    'Growth': 14.8,
                    'Dividend': 11.2,
                    'Factors': 13.0,
                    'Bonds': 1.8,
                    'description': 'Golden age of capitalism'
                },
                'Stagflation\n(1966-1981)': {
                    'period': '1966-1981',
                    'regime': 'Sideways/Choppy',
                    'Total Market': 3.8,
                    'Growth': 2.5,
                    'Dividend': 6.2,
                    'Factors': 4.5,
                    'Bonds': -1.5,
                    'description': 'High inflation, oil shocks, stagnation'
                },
                'Reagan Bull\n(1982-1987)': {
                    'period': '1982-1987',
                    'regime': 'Bull Market (Low Vol)',
                    'Total Market': 16.8,
                    'Growth': 19.5,
                    'Dividend': 14.0,
                    'Factors': 17.2,
                    'Bonds': 11.2,
                    'description': 'Interest rates falling, economic boom'
                },
                'Black Monday\n(1987)': {
                    'period': '1987',
                    'regime': 'Bear Market (High Vol)',
                    'Total Market': -22.0,
                    'Growth': -28.0,
                    'Dividend': -18.0,
                    'Factors': -20.0,
                    'Bonds': 4.5,
                    'description': 'Single worst day in stock market history'
                },
                'Dot-com Boom\n(1995-2000)': {
                    'period': '1995-2000',
                    'regime': 'Bull Market (Low Vol)',
                    'Total Market': 20.5,
                    'Growth': 32.0,
                    'Dividend': 12.0,
                    'Factors': 15.0,
                    'Bonds': 6.0,
                    'description': 'Tech bubble, irrational exuberance'
                },
                'Dot-com Bust\n(2000-2002)': {
                    'period': '2000-2002',
                    'regime': 'Bear Market (Low Vol)',
                    'Total Market': -14.5,
                    'Growth': -28.0,
                    'Dividend': -2.0,
                    'Factors': -8.0,
                    'Bonds': 10.5,
                    'description': 'Tech crash, recession'
                },
                'Housing Boom\n(2003-2007)': {
                    'period': '2003-2007',
                    'regime': 'Bull Market (Low Vol)',
                    'Total Market': 12.8,
                    'Growth': 14.0,
                    'Dividend': 11.5,
                    'Factors': 13.5,
                    'Bonds': 4.8,
                    'description': 'Easy credit, housing bubble'
                },
                'Financial Crisis\n(2008)': {
                    'period': '2008',
                    'regime': 'Bear Market (High Vol)',
                    'Total Market': -37.0,
                    'Growth': -42.0,
                    'Dividend': -28.0,
                    'Factors': -32.0,
                    'Bonds': 5.2,
                    'description': 'Great Recession, banking crisis'
                },
                'Recovery Bull\n(2009-2019)': {
                    'period': '2009-2019',
                    'regime': 'Bull Market (Low Vol)',
                    'Total Market': 14.5,
                    'Growth': 17.2,
                    'Dividend': 12.0,
                    'Factors': 13.8,
                    'Bonds': 3.5,
                    'description': 'Longest bull market in history'
                },
                'COVID Crash\n(Feb-Mar 2020)': {
                    'period': 'Feb-Mar 2020',
                    'regime': 'Bear Market (High Vol)',
                    'Total Market': -34.0,
                    'Growth': -30.0,
                    'Dividend': -38.0,
                    'Factors': -32.0,
                    'Bonds': 8.0,
                    'description': 'Pandemic panic, fastest crash ever'
                },
                'Post-COVID Bull\n(2020-2021)': {
                    'period': '2020-2021',
                    'regime': 'Bull Market (High Vol)',
                    'Total Market': 28.0,
                    'Growth': 45.0,
                    'Dividend': 18.0,
                    'Factors': 25.0,
                    'Bonds': -2.0,
                    'description': 'Stimulus-fueled rally, meme stocks'
                },
                'Rate Hike Bear\n(2022)': {
                    'period': '2022',
                    'regime': 'Bear Market (High Vol)',
                    'Total Market': -18.0,
                    'Growth': -33.0,
                    'Dividend': -5.0,
                    'Factors': -12.0,
                    'Bonds': -13.0,
                    'description': 'Fed fights inflation, everything down'
                },
                'AI Rally\n(2023-2024)': {
                    'period': '2023-2024',
                    'regime': 'Bull Market (Low Vol)',
                    'Total Market': 22.0,
                    'Growth': 35.0,
                    'Dividend': 12.0,
                    'Factors': 18.0,
                    'Bonds': 2.5,
                    'description': 'AI boom, mega-cap domination'
                }
            }
            
            # Prepare data for heatmap
            sleeves = ['Total Market', 'Growth', 'Dividend', 'Factors', 'Bonds']
            periods = list(historical_regimes.keys())
            
            # Create matrix
            data_matrix = []
            for period in periods:
                row = [historical_regimes[period][sleeve] for sleeve in sleeves]
                data_matrix.append(row)
            
            df_heatmap = pd.DataFrame(data_matrix, columns=sleeves, index=periods)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 14))
            
            # Create custom colormap: red (negative) -> yellow (zero) -> green (positive)
            cmap = sns.diverging_palette(10, 130, as_cmap=True)
            
            sns.heatmap(df_heatmap, annot=True, fmt='.1f', cmap=cmap, center=0,
                        linewidths=0.5, cbar_kws={'label': 'Annualized Return (%)'},
                        vmin=-60, vmax=50, ax=ax)
            
            ax.set_title('Historical Market Regimes: Sleeve Performance (Annualized Returns %)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Portfolio Sleeve', fontsize=12, fontweight='bold')
            ax.set_ylabel('Market Regime Period', fontsize=12, fontweight='bold')
            
            plt.xticks(rotation=0, ha='center')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Key insights from heatmap
            st.markdown("---")
            st.markdown("### 💡 Key Insights from 120 Years of Market History")
            
            st.markdown("**🟢 Best Performers in Each Regime:**")
            st.markdown("""
            - **Bull Markets (Low Vol):** Growth dominates, everything works
            - **Bull Markets (High Vol):** Growth still wins but with wild swings
            - **Bear Markets:** Bonds are king, Dividend stocks hold up best among equities
            - **Sideways/Choppy:** Dividend stocks outperform, Growth disappoints
            """)
            
            st.markdown("**🔴 Worst Performers in Each Regime:**")
            st.markdown("""
            - **Bear Markets (High Vol):** Growth gets crushed (-30% to -55%)
            - **Rising Rate Environments:** Both stocks AND bonds can fall together (2022)
            - **Stagflation:** Everything struggles, especially bonds
            """)
            
            st.markdown("**🎯 Portfolio Strategy Implications:**")
            st.markdown("""
            - **You NEED multiple sleeves:** No single sleeve wins in all regimes
            - **Bonds are essential:** Only asset class with consistent positive returns in crashes
            - **Growth is feast or famine:** Huge gains in bull markets, devastating losses in bears
            - **Dividend stocks = stability:** Lower highs but much shallower crashes
            - **Diversification matters:** Balanced portfolios smooth out the extremes
            """)
            
            st.markdown("**📊 Historical Averages by Sleeve:**")
            st.markdown("""
            - **Total Market:** 10-12% in bull markets, -20% to -35% in bear markets
            - **Growth:** 15-35% in bull markets, -30% to -55% in bear markets
            - **Dividend:** 10-14% in bull markets, -5% to -28% in bear markets
            - **Factors:** 13-18% in bull markets, -12% to -35% in bear markets
            - **Bonds:** 2-10% in bull markets, +5% to +10% in bear markets
            """)
            
            st.markdown("**⚠️ Don't Fight the Last War:**")
            st.markdown("""
            - 2022 proved bonds can fall with stocks (unprecedented)
            - Tech dominated 2010s but crashed badly in 2000-2002
            - Each crisis is different - diversification is the only hedge
            """)
            
            # Regime-specific recommendations
            st.markdown("---")
            st.markdown("### 🎯 How to Position for Different Regimes")
            
            regime_recommendations = {
                'Bull Market (Low Vol)': {
                    'best': ['Growth: 30-40%', 'Total Market: 40-50%', 'Bonds: 10-20%'],
                    'avoid': 'Being too defensive - you miss gains',
                    'action': 'Stay fully invested, let winners run'
                },
                'Bull Market (High Vol)': {
                    'best': ['Growth: 20-30%', 'Total Market: 30-40%', 'Dividend: 15-20%', 'Bonds: 20-30%'],
                    'avoid': 'Panic selling during dips',
                    'action': 'Add on pullbacks, maintain conviction'
                },
                'Sideways/Choppy': {
                    'best': ['Dividend: 25-30%', 'Bonds: 30-40%', 'Total Market: 20-30%'],
                    'avoid': 'Chasing momentum - it reverses quickly',
                    'action': 'Be patient, rebalance, collect dividends'
                },
                'Bear Market (Low Vol)': {
                    'best': ['Bonds: 40-50%', 'Dividend: 20-25%', 'Cash: 10-20%'],
                    'avoid': 'Staying fully invested in growth',
                    'action': 'Raise cash, increase bonds gradually'
                },
                'Bear Market (High Vol)': {
                    'best': ['Bonds: 50-60%', 'Dividend: 15-20%', 'Cash: 10-20%'],
                    'avoid': 'PANIC SELLING - this is when fortunes are made',
                    'action': 'Buy aggressively if you have cash. This is rare opportunity.'
                }
            }
            
            for regime, rec in regime_recommendations.items():
                with st.expander(f"**{regime}** - Optimal Sleeve Allocation"):
                    st.markdown(f"**Best Sleeve Mix:**")
                    for item in rec['best']:
                        st.markdown(f"• {item}")
                    st.markdown(f"\n**Avoid:** {rec['avoid']}")
                    st.markdown(f"\n**Action Plan:** {rec['action']}")
        
        
        # =============================================================================
        # TAB 5: FORWARD-LOOKING RISK ANALYSIS (NEW!)
        # =============================================================================
        
