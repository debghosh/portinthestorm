"""
Tab: Detailed Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from helper_functions import *


def render(tab2, portfolio_returns, prices, weights, tickers, metrics, current):
    """Render the Detailed Analysis tab"""
    
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
        
        
        # =============================================================================
        # NEW TAB 3: SLEEVES ANALYSIS
        # Insert this after Detailed Analysis tab (becomes new tab3)
        # =============================================================================
        
