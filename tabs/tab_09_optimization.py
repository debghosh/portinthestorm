"""
Tab: Optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from helper_functions import *


def render(tab9, portfolio_returns, prices, weights, tickers, metrics, current):
    """Render the Optimization tab"""
    
    with tab9:
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
