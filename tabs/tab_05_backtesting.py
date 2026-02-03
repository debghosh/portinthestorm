"""
Tab: Backtesting - Portfolio vs Benchmark Analysis
Comprehensive comparison of portfolio performance against selected benchmarks
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
from helper_functions import *


def render(tab5, portfolio_returns, prices, weights, tickers, metrics, current):
    """Render the Backtesting tab with benchmark comparison"""
    
    with tab5:
        st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 15px; color: white; margin-bottom: 2rem;">
                <h1 style="margin: 0; font-size: 2.5rem;">⚔️ Portfolio vs Benchmark Battle</h1>
                <p style="font-size: 1.2rem; margin-top: 0.5rem; opacity: 0.9;">
                    How does your portfolio stack up against the market?
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # =============================================================================
        # BENCHMARK SELECTION
        # =============================================================================
        
        st.markdown("### 🎯 Select Your Benchmark")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            benchmark_choice = st.selectbox(
                "Choose comparison benchmark:",
                ["SPY (S&P 500)", "QQQ (Nasdaq-100)", "60/40 Portfolio (SPY/AGG)", "Custom Ticker"],
                key="benchmark_selector"
            )
        
        with col2:
            if benchmark_choice == "Custom Ticker":
                custom_ticker = st.text_input("Enter ticker:", value="VTI", key="custom_benchmark")
                benchmark_ticker = custom_ticker.upper()
                benchmark_name = benchmark_ticker
            elif "60/40" in benchmark_choice:
                benchmark_ticker = "6040"  # Special flag
                benchmark_name = "60/40 Portfolio"
            elif "SPY" in benchmark_choice:
                benchmark_ticker = "SPY"
                benchmark_name = "S&P 500"
            else:  # QQQ
                benchmark_ticker = "QQQ"
                benchmark_name = "Nasdaq-100"
        
        st.markdown("---")
        
        # Download benchmark data
        with st.spinner(f"Loading {benchmark_name} data..."):
            start_date = current['start_date']
            end_date = current['end_date']
            
            if benchmark_ticker == "6040":
                # Create 60/40 portfolio
                benchmark_data = download_ticker_data(['SPY', 'AGG'], start_date, end_date)
                if benchmark_data is not None and not benchmark_data.empty:
                    benchmark_weights = np.array([0.6, 0.4])
                    benchmark_returns = calculate_portfolio_returns(benchmark_data, benchmark_weights)
                else:
                    st.error("Failed to load benchmark data")
                    st.stop()
            else:
                benchmark_data = download_ticker_data([benchmark_ticker], start_date, end_date)
                if benchmark_data is not None and not benchmark_data.empty:
                    benchmark_returns = benchmark_data[benchmark_ticker].pct_change().dropna()
                else:
                    st.error("Failed to load benchmark data")
                    st.stop()
            
            # Align the returns
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_returns_aligned = portfolio_returns.loc[common_dates]
            benchmark_returns_aligned = benchmark_returns.loc[common_dates]
            
            # Calculate benchmark metrics
            benchmark_metrics = calculate_portfolio_metrics(benchmark_returns_aligned)
        
        # =============================================================================
        # HEAD-TO-HEAD COMPARISON
        # =============================================================================
        
        st.markdown("### 📊 Head-to-Head Comparison")
        
        # Calculate comparative metrics
        portfolio_total_return = (1 + portfolio_returns_aligned).prod() - 1
        benchmark_total_return = (1 + benchmark_returns_aligned).prod() - 1
        
        portfolio_cagr = metrics['Annual Return']
        benchmark_cagr = benchmark_metrics['Annual Return']
        
        portfolio_vol = metrics['Annual Volatility']
        benchmark_vol = benchmark_metrics['Annual Volatility']
        
        portfolio_sharpe = metrics['Sharpe Ratio']
        benchmark_sharpe = benchmark_metrics['Sharpe Ratio']
        
        portfolio_sortino = metrics['Sortino Ratio']
        benchmark_sortino = benchmark_metrics['Sortino Ratio']
        
        portfolio_max_dd = metrics['Max Drawdown']
        benchmark_max_dd = benchmark_metrics['Max Drawdown']
        
        portfolio_calmar = metrics['Calmar Ratio']
        benchmark_calmar = benchmark_metrics['Calmar Ratio']
        
        # Calculate Alpha (excess return adjusted for risk)
        # Alpha = Portfolio Return - (Risk-Free Rate + Beta * (Benchmark Return - Risk-Free Rate))
        risk_free_rate = 0.02  # 2% annual
        covariance = portfolio_returns_aligned.cov(benchmark_returns_aligned)
        benchmark_variance = benchmark_returns_aligned.var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
        
        alpha_annual = portfolio_cagr - (risk_free_rate + beta * (benchmark_cagr - risk_free_rate))
        
        # Correlation
        correlation = portfolio_returns_aligned.corr(benchmark_returns_aligned)
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Metric': [
                'Total Return',
                'Annual Return (CAGR)',
                'Annual Volatility',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Max Drawdown',
                'Calmar Ratio',
                'Alpha (vs Benchmark)',
                'Beta (vs Benchmark)',
                'Correlation'
            ],
            'Your Portfolio': [
                f"{portfolio_total_return*100:.2f}%",
                f"{portfolio_cagr*100:.2f}%",
                f"{portfolio_vol*100:.2f}%",
                f"{portfolio_sharpe:.2f}",
                f"{portfolio_sortino:.2f}",
                f"{portfolio_max_dd*100:.2f}%",
                f"{portfolio_calmar:.2f}",
                f"{alpha_annual*100:.2f}%",
                f"{beta:.2f}",
                f"{correlation:.2f}"
            ],
            benchmark_name: [
                f"{benchmark_total_return*100:.2f}%",
                f"{benchmark_cagr*100:.2f}%",
                f"{benchmark_vol*100:.2f}%",
                f"{benchmark_sharpe:.2f}",
                f"{benchmark_sortino:.2f}",
                f"{benchmark_max_dd*100:.2f}%",
                f"{benchmark_calmar:.2f}",
                "0.00%",
                "1.00",
                "1.00"
            ],
            'Difference': [
                f"{(portfolio_total_return - benchmark_total_return)*100:+.2f}%",
                f"{(portfolio_cagr - benchmark_cagr)*100:+.2f}%",
                f"{(portfolio_vol - benchmark_vol)*100:+.2f}%",
                f"{(portfolio_sharpe - benchmark_sharpe):+.2f}",
                f"{(portfolio_sortino - benchmark_sortino):+.2f}",
                f"{(portfolio_max_dd - benchmark_max_dd)*100:+.2f}%",
                f"{(portfolio_calmar - benchmark_calmar):+.2f}",
                f"{alpha_annual*100:.2f}%",
                f"{(beta - 1.0):+.2f}",
                f"{(correlation - 1.0):+.2f}"
            ]
        })
        
        # Style the dataframe
        def highlight_better(row):
            if row.name == 0 or row.name == 1 or row.name == 3 or row.name == 4 or row.name == 6 or row.name == 7:
                # Higher is better
                diff_val = float(row['Difference'].rstrip('%').replace('+', ''))
                if diff_val > 0:
                    return ['', 'background-color: #d4edda', '', 'background-color: #c3e6cb; font-weight: bold']
                elif diff_val < 0:
                    return ['', 'background-color: #f8d7da', '', 'background-color: #f5c6cb; font-weight: bold']
            elif row.name == 2 or row.name == 5:
                # Lower is better
                diff_val = float(row['Difference'].rstrip('%').replace('+', ''))
                if diff_val < 0:
                    return ['', 'background-color: #d4edda', '', 'background-color: #c3e6cb; font-weight: bold']
                elif diff_val > 0:
                    return ['', 'background-color: #f8d7da', '', 'background-color: #f5c6cb; font-weight: bold']
            return ['', '', '', '']
        
        styled_df = comparison_df.style.apply(highlight_better, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Interpretation
        st.markdown("---")
        st.markdown("### 🧠 What Does This Mean?")
        
        # Alpha interpretation
        if alpha_annual > 0.02:
            alpha_msg = f"🎉 **Excellent Alpha**: Your portfolio generated {alpha_annual*100:.2f}% annual alpha (risk-adjusted excess return). You're beating the benchmark even after accounting for risk!"
            alpha_color = "success"
        elif alpha_annual > 0:
            alpha_msg = f"✅ **Positive Alpha**: Your portfolio generated {alpha_annual*100:.2f}% annual alpha. You're adding value above the benchmark."
            alpha_color = "info"
        elif alpha_annual > -0.02:
            alpha_msg = f"⚠️ **Slight Underperformance**: Your portfolio has {alpha_annual*100:.2f}% annual alpha. Consider rebalancing or reviewing your strategy."
            alpha_color = "warning"
        else:
            alpha_msg = f"❌ **Negative Alpha**: Your portfolio has {alpha_annual*100:.2f}% annual alpha. You would have been better off investing in the benchmark."
            alpha_color = "error"
        
        if alpha_color == "success":
            st.success(alpha_msg)
        elif alpha_color == "info":
            st.info(alpha_msg)
        elif alpha_color == "warning":
            st.warning(alpha_msg)
        else:
            st.error(alpha_msg)
        
        # Beta interpretation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Beta Analysis:**")
            if beta > 1.1:
                st.markdown(f"• Beta = **{beta:.2f}** → High volatility relative to benchmark")
                st.markdown(f"• Your portfolio moves **{((beta-1)*100):.0f}% more** than the benchmark")
                st.markdown("• Higher risk, higher potential reward")
            elif beta > 0.9:
                st.markdown(f"• Beta = **{beta:.2f}** → Similar volatility to benchmark")
                st.markdown("• Your portfolio tracks the benchmark closely")
            else:
                st.markdown(f"• Beta = **{beta:.2f}** → Lower volatility than benchmark")
                st.markdown(f"• Your portfolio moves **{((1-beta)*100):.0f}% less** than the benchmark")
                st.markdown("• More defensive, lower risk")
        
        with col2:
            st.markdown("**🔗 Correlation Analysis:**")
            if correlation > 0.9:
                st.markdown(f"• Correlation = **{correlation:.2f}** → Very high correlation")
                st.markdown("• Portfolio moves almost identically to benchmark")
            elif correlation > 0.7:
                st.markdown(f"• Correlation = **{correlation:.2f}** → High correlation")
                st.markdown("• Portfolio generally follows benchmark trends")
            elif correlation > 0.5:
                st.markdown(f"• Correlation = **{correlation:.2f}** → Moderate correlation")
                st.markdown("• Some independence from benchmark movements")
            else:
                st.markdown(f"• Correlation = **{correlation:.2f}** → Low correlation")
                st.markdown("• Portfolio offers true diversification from benchmark")
        
        st.markdown("---")
        
        # =============================================================================
        # CUMULATIVE RETURNS CHART
        # =============================================================================
        
        st.markdown("### 📈 Cumulative Returns: Portfolio vs Benchmark")
        
        portfolio_cumulative = (1 + portfolio_returns_aligned).cumprod()
        benchmark_cumulative = (1 + benchmark_returns_aligned).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=(portfolio_cumulative - 1) * 100,
            name='Your Portfolio',
            line=dict(color='#667eea', width=3),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=(benchmark_cumulative - 1) * 100,
            name=benchmark_name,
            line=dict(color='#f5576c', width=3, dash='dash'),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Cumulative Returns Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        final_portfolio = (portfolio_cumulative.iloc[-1] - 1) * 100
        final_benchmark = (benchmark_cumulative.iloc[-1] - 1) * 100
        outperformance = final_portfolio - final_benchmark
        
        st.markdown("""
            <div class="interpretation-box">
                <div class="interpretation-title">📊 Returns Analysis</div>
        """, unsafe_allow_html=True)
        
        if outperformance > 0:
            st.markdown(f"""
                Your portfolio returned **{final_portfolio:.2f}%** vs benchmark's **{final_benchmark:.2f}%**
                
                **✅ Outperformance: +{outperformance:.2f}%**
                
                On a $100,000 investment, you made **${(outperformance/100)*100000:,.0f} more** than the benchmark!
            """)
        else:
            st.markdown(f"""
                Your portfolio returned **{final_portfolio:.2f}%** vs benchmark's **{final_benchmark:.2f}%**
                
                **❌ Underperformance: {outperformance:.2f}%**
                
                On a $100,000 investment, you made **${(outperformance/100)*100000:,.0f} less** than the benchmark.
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # =============================================================================
        # DRAWDOWN COMPARISON CHART
        # =============================================================================
        
        st.markdown("### 📉 Drawdown Comparison: Risk Analysis")
        
        # Calculate drawdowns
        portfolio_dd = (portfolio_cumulative / portfolio_cumulative.cummax() - 1) * 100
        benchmark_dd = (benchmark_cumulative / benchmark_cumulative.cummax() - 1) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_dd.index,
            y=portfolio_dd,
            name='Your Portfolio',
            fill='tozeroy',
            line=dict(color='#667eea', width=2),
            fillcolor='rgba(102, 126, 234, 0.3)',
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark_dd.index,
            y=benchmark_dd,
            name=benchmark_name,
            line=dict(color='#f5576c', width=2, dash='dash'),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Drawdown Comparison (Lower is Better Risk Management)",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown statistics
        st.markdown("#### 📊 Drawdown Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Your Max Drawdown",
                f"{portfolio_max_dd*100:.2f}%",
                delta=f"{(portfolio_max_dd - benchmark_max_dd)*100:.2f}%",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                f"{benchmark_name} Max Drawdown",
                f"{benchmark_max_dd*100:.2f}%"
            )
        
        with col3:
            avg_portfolio_dd = portfolio_dd[portfolio_dd < 0].mean()
            avg_benchmark_dd = benchmark_dd[benchmark_dd < 0].mean()
            st.metric(
                "Your Avg Drawdown",
                f"{avg_portfolio_dd:.2f}%",
                delta=f"{(avg_portfolio_dd - avg_benchmark_dd):.2f}%",
                delta_color="inverse"
            )
        
        st.markdown("""
            <div class="interpretation-box">
                <div class="interpretation-title">🛡️ Risk Management Analysis</div>
        """, unsafe_allow_html=True)
        
        if portfolio_max_dd > benchmark_max_dd:
            risk_ratio = portfolio_max_dd / benchmark_max_dd
            st.markdown(f"""
                ⚠️ Your portfolio experienced **{abs((portfolio_max_dd - benchmark_max_dd)*100):.2f}% deeper drawdowns** than the benchmark.
                
                In the worst period, your portfolio was down **{portfolio_max_dd*100:.2f}%** vs benchmark's **{benchmark_max_dd*100:.2f}%**.
                
                This suggests **higher risk** - make sure you can stomach these drops!
            """)
        else:
            st.markdown(f"""
                ✅ Your portfolio showed **better downside protection** than the benchmark!
                
                Maximum drawdown was **{portfolio_max_dd*100:.2f}%** vs benchmark's **{benchmark_max_dd*100:.2f}%**.
                
                This suggests **better risk management** - you're preserving capital in downturns.
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # =============================================================================
        # ROLLING METRICS COMPARISON
        # =============================================================================
        
        st.markdown("### 🎢 Rolling Performance Metrics")
        
        window = st.slider("Select Rolling Window (days)", 30, 365, 90, key="rolling_window_backtest")
        
        # Calculate rolling metrics
        portfolio_rolling_return = portfolio_returns_aligned.rolling(window).mean() * 252 * 100
        benchmark_rolling_return = benchmark_returns_aligned.rolling(window).mean() * 252 * 100
        
        portfolio_rolling_vol = portfolio_returns_aligned.rolling(window).std() * np.sqrt(252) * 100
        benchmark_rolling_vol = benchmark_returns_aligned.rolling(window).std() * np.sqrt(252) * 100
        
        portfolio_rolling_sharpe = (portfolio_rolling_return - 2) / portfolio_rolling_vol
        benchmark_rolling_sharpe = (benchmark_rolling_return - 2) / benchmark_rolling_vol
        
        # Plot rolling returns
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_rolling_return.index,
            y=portfolio_rolling_return,
            name='Your Portfolio',
            line=dict(color='#667eea', width=2),
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark_rolling_return.index,
            y=benchmark_rolling_return,
            name=benchmark_name,
            line=dict(color='#f5576c', width=2, dash='dash'),
        ))
        
        fig.update_layout(
            title=f"{window}-Day Rolling Annualized Return",
            xaxis_title="Date",
            yaxis_title="Annualized Return (%)",
            hovermode='x unified',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot rolling Sharpe
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_rolling_sharpe.index,
            y=portfolio_rolling_sharpe,
            name='Your Portfolio',
            line=dict(color='#667eea', width=2),
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark_rolling_sharpe.index,
            y=benchmark_rolling_sharpe,
            name=benchmark_name,
            line=dict(color='#f5576c', width=2, dash='dash'),
        ))
        
        fig.update_layout(
            title=f"{window}-Day Rolling Sharpe Ratio",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            hovermode='x unified',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # =============================================================================
        # YEAR-BY-YEAR COMPARISON
        # =============================================================================
        
        st.markdown("### 📅 Year-by-Year Performance")
        
        # Calculate annual returns
        portfolio_annual = portfolio_returns_aligned.resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
        benchmark_annual = benchmark_returns_aligned.resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
        
        if len(portfolio_annual) > 0:
            annual_df = pd.DataFrame({
                'Year': portfolio_annual.index.year,
                'Your Portfolio': portfolio_annual.values,
                benchmark_name: benchmark_annual.values,
                'Difference': (portfolio_annual - benchmark_annual).values
            })
            
            # Format the dataframe
            annual_df['Your Portfolio'] = annual_df['Your Portfolio'].apply(lambda x: f"{x:.2f}%")
            annual_df[benchmark_name] = annual_df[benchmark_name].apply(lambda x: f"{x:.2f}%")
            annual_df['Difference'] = annual_df['Difference'].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(annual_df, use_container_width=True, hide_index=True)
            
            # Calculate win rate
            wins = (portfolio_annual > benchmark_annual).sum()
            total_years = len(portfolio_annual)
            win_rate = wins / total_years * 100
            
            st.markdown(f"""
                <div class="metric-card">
                    <h4>🏆 Win Rate: {win_rate:.1f}%</h4>
                    <p>Your portfolio outperformed the benchmark in <strong>{wins} out of {total_years} years</strong></p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # =============================================================================
        # FINAL VERDICT
        # =============================================================================
        
        st.markdown("### 🎯 Final Verdict")
        
        # Score the portfolio
        score = 0
        max_score = 7
        
        # Alpha
        if alpha_annual > 0.02:
            score += 2
        elif alpha_annual > 0:
            score += 1
        
        # Sharpe Ratio
        if portfolio_sharpe > benchmark_sharpe:
            score += 1
        
        # Max Drawdown
        if portfolio_max_dd < benchmark_max_dd:
            score += 1
        
        # Total Return
        if portfolio_total_return > benchmark_total_return:
            score += 1
        
        # Volatility
        if portfolio_vol < benchmark_vol:
            score += 1
        
        # Sortino Ratio
        if portfolio_sortino > benchmark_sortino:
            score += 1
        
        score_pct = score / max_score * 100
        
        if score_pct >= 85:
            verdict_color = "#28a745"
            verdict_emoji = "🏆"
            verdict_title = "Outstanding Performance!"
            verdict_msg = "Your portfolio is crushing the benchmark across multiple metrics. Excellent work!"
        elif score_pct >= 70:
            verdict_color = "#17a2b8"
            verdict_emoji = "✅"
            verdict_title = "Strong Performance"
            verdict_msg = "Your portfolio is performing well compared to the benchmark. Keep it up!"
        elif score_pct >= 50:
            verdict_color = "#ffc107"
            verdict_emoji = "⚖️"
            verdict_title = "Mixed Results"
            verdict_msg = "Your portfolio shows both strengths and weaknesses vs the benchmark. Consider optimizing."
        else:
            verdict_color = "#dc3545"
            verdict_emoji = "⚠️"
            verdict_title = "Underperformance"
            verdict_msg = "Your portfolio is lagging the benchmark. Time to review your strategy or consider passive indexing."
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, {verdict_color}22 0%, {verdict_color}44 100%); 
                        padding: 2rem; border-radius: 15px; border-left: 5px solid {verdict_color};">
                <h2 style="margin-top: 0; color: {verdict_color};">{verdict_emoji} {verdict_title}</h2>
                <p style="font-size: 1.2rem; margin-bottom: 1rem;">{verdict_msg}</p>
                <p style="font-size: 1rem;">
                    <strong>Portfolio Score: {score}/{max_score} ({score_pct:.0f}%)</strong><br>
                    Based on: Alpha, Total Return, Risk-Adjusted Returns (Sharpe, Sortino), Volatility, and Drawdown
                </p>
            </div>
        """, unsafe_allow_html=True)
