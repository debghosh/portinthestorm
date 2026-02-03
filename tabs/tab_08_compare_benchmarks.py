"""
Tab: Compare Benchmarks
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from helper_functions import *


def render(tab8, portfolio_returns, prices, weights, tickers, metrics, current):
    """Render the Compare Benchmarks tab"""
    
    with tab8:
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
        
