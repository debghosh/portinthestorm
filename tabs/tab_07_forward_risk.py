"""
Tab: Forward Risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from helper_functions import *


def render(tab7, portfolio_returns, prices, weights, tickers, metrics, current):
    """Render the Forward Risk tab"""
    
    with tab7:
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
        
