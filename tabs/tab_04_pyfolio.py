"""
Tab: PyFolio Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from helper_functions import *


def render(tab4, portfolio_returns, prices, weights, tickers, metrics, current):
    """Render the PyFolio Analysis tab"""
    
    with tab4:
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
        
