"""
Tab: Overview
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from helper_functions import *


def render(tab1, portfolio_returns, prices, weights, tickers, metrics, current):
    """Render the Overview tab"""
    
    with tab1:
            st.markdown("""
                <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            border-radius: 15px; color: white; margin-bottom: 2rem;">
                    <h1 style="margin: 0; font-size: 2.5rem;">👨‍🍳 Your Investment Kitchen</h1>
                    <p style="font-size: 1.2rem; margin-top: 0.5rem; opacity: 0.9;">
                        Investing is like cooking - you need the right ingredients, proper proportions, and perfect timing
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # =============================================================================
            # SECTION 1: WHAT ARE YOU COOKING? (The Goal)
            # =============================================================================
            st.markdown("""
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; 
                            border-radius: 10px; border-left: 5px solid #667eea; margin-bottom: 2rem;">
                    <h2 style="margin-top: 0; color: #2c3e50;">🎯 1. What Are You Cooking?</h2>
                    <p style="font-size: 1.1rem; color: #555; margin-bottom: 0;">
                        <strong>Your Goal:</strong> Every great dish starts with knowing what you're making. 
                        What's your investment objective?
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            # Calculate values
            start_date = pd.to_datetime(current['start_date'])
            end_date = pd.to_datetime(current['end_date'])
            days_invested = (end_date - start_date).days
            years_invested = days_invested / 365.25
            total_return = metrics['Total Return']
            final_value = 100000 * (1 + total_return)
            volatility = metrics['Annual Volatility']
            
            with col1:
                st.markdown("""
                    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 100%;">
                        <h4 style="color: #667eea; margin-top: 0;">📅 Time Horizon</h4>
                """, unsafe_allow_html=True)
                st.metric("Analysis Period", f"{years_invested:.1f} years")
                st.markdown(f"*{start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')}*")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 100%;">
                        <h4 style="color: #667eea; margin-top: 0;">💰 Portfolio Value</h4>
                """, unsafe_allow_html=True)
                st.metric("$100k Invested", f"${final_value:,.0f}", f"{total_return*100:+.1f}%")
                st.markdown(f"*{metrics['Annual Return']*100:.1f}% annualized*")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 100%;">
                        <h4 style="color: #667eea; margin-top: 0;">📊 Risk Profile</h4>
                """, unsafe_allow_html=True)
                
                if volatility < 0.10:
                    risk_profile = "Conservative 🛡️"
                    risk_color = "#28a745"
                elif volatility < 0.15:
                    risk_profile = "Moderate 🎯"
                    risk_color = "#ffc107"
                elif volatility < 0.20:
                    risk_profile = "Aggressive 🚀"
                    risk_color = "#fd7e14"
                else:
                    risk_profile = "Very Aggressive ⚡"
                    risk_color = "#dc3545"
                
                st.markdown(f"<h3 style='color: {risk_color}; margin: 0.5rem 0;'>{risk_profile}</h3>", unsafe_allow_html=True)
                st.metric("Volatility", f"{volatility*100:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # =============================================================================
            # SECTION 2: YOUR INGREDIENTS (The Portfolio)
            # =============================================================================
            st.markdown("""
                <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 1.5rem; 
                            border-radius: 10px; border-left: 5px solid #28a745; margin-bottom: 2rem;">
                    <h2 style="margin-top: 0; color: #2c3e50;">🥘 2. Your Ingredients</h2>
                    <p style="font-size: 1.1rem; color: #555; margin-bottom: 0;">
                        <strong>The Portfolio:</strong> Each ETF is an ingredient. Know what each brings to the table and its quality.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Enhanced ingredient table
                ingredients_data = []
                for ticker in weights.keys():
                    weight = weights[ticker]
                    
                    if ticker in prices.columns:
                        signal_data = generate_trading_signal(prices[ticker],ticker)
                        action = signal_data['action']
                        
                        ticker_returns = prices[ticker].pct_change().dropna()
                        ticker_annual_return = (1 + ticker_returns.mean()) ** 252 - 1
                        
                        # Categorize
                        if ticker in ['SPY', 'VTI', 'QQQ', 'VOO', 'VUG']:
                            ingredient_type = "🥩 Main Course (Core Growth)"
                        elif ticker in ['AGG', 'BND', 'TLT', 'IEF', 'SHY']:
                            ingredient_type = "🥗 Stabilizer (Bonds)"
                        elif ticker in ['VEA', 'VWO', 'EFA', 'IEMG', 'VXUS']:
                            ingredient_type = "🌶️ Spice (International)"
                        elif ticker in ['GLD', 'IAU']:
                            ingredient_type = "🧂 Preservative (Gold)"
                        elif ticker in ['VYM', 'SCHD', 'DVY']:
                            ingredient_type = "💰 Dividend"
                        else:
                            ingredient_type = "🥄 Specialty"
                        
                        ingredients_data.append({
                            'Ticker': ticker,
                            'Type': ingredient_type,
                            'Portion': f"{weight*100:.1f}%",
                            'Performance': f"{ticker_annual_return*100:+.1f}%/yr",
                            'Action': action
                        })
                
                ingredients_df = pd.DataFrame(ingredients_data)
                
                def style_action(val):
                    if val == 'Accumulate':
                        return 'background-color: #d4edda; color: #155724; font-weight: bold'
                    elif val == 'Distribute':
                        return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                    elif val == 'Hold':
                        return 'background-color: #fff3cd; color: #856404; font-weight: bold'
                    return ''
                
                styled_ingredients = ingredients_df.style.applymap(style_action, subset=['Action'])
                st.dataframe(styled_ingredients, use_container_width=True, hide_index=True)
        
        
                # Ingredient Guide
                with st.expander("🧾 Ingredient Guide - What Each Type Does"):
                    st.markdown("""
                    **🥩 Main Course (Core Growth)** - Large-cap stocks (SPY, VTI, QQQ)  
                    → Provides primary growth. Like the protein in your meal.
                    
                    **🥗 Stabilizer (Bonds)** - Fixed income (AGG, BND, TLT)  
                    → Reduces volatility, provides steady income. Like vegetables that balance the meal.
                    
                    **🌶️ Spice (International)** - Foreign stocks (VEA, VWO, EFA)  
                    → Adds diversification and growth from other economies. Enhances flavor.
                    
                    **🧂 Preservative (Gold)** - Precious metals (GLD, IAU)  
                    → Inflation hedge, crisis insurance. Preserves value when market sours.
                    
                    **🥄 Specialty** - Sector-specific or thematic ETFs  
                    → Targeted exposure to specific themes. Special seasoning.
                    """)
            
            with col2:
                # Pie chart
                fig, ax = plt.subplots(figsize=(7, 7))
                colors = plt.cm.Set3(range(len(weights)))
                wedges, texts, autotexts = ax.pie(
                    weights.values(), 
                    labels=weights.keys(), 
                    autopct='%1.1f%%',
                    colors=colors, 
                    startangle=90,
                    textprops={'fontsize': 11, 'weight': 'bold'}
                )
                ax.set_title('Current Recipe', fontsize=14, fontweight='bold', pad=20)
                st.pyplot(fig)
                
                # Quality Score
                st.markdown("### ⭐ Overall Quality")
                sharpe = metrics['Sharpe Ratio']
                
                if sharpe > 1.5:
                    quality = "Excellent"
                    emoji = "🌟🌟🌟🌟🌟"
                    color = "#030804"
                elif sharpe > 1.0:
                    quality = "Very Good"
                    emoji = "🌟🌟🌟🌟"
                    color = "#06130f"
                elif sharpe > 0.5:
                    quality = "Good"
                    emoji = "🌟🌟🌟"
                    color = "#0a0802"
                elif sharpe > 0:
                    quality = "Fair"
                    emoji = "🌟🌟"
                    color = "#21140a"
                else:
                    quality = "Needs Work"
                    emoji = "🌟"
                    color = "#130607"
                
                st.markdown(f"""
                    <div style="background: {color}; color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
                        <h3 style="margin: 0; font-size: 2rem;">{emoji}</h3>
                        <h2 style="margin: 0.5rem 0;">{quality}</h2>
                        <p style="margin: 0; opacity: 0.9;">Sharpe Ratio: {sharpe:.2f}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # =============================================================================
            # SECTION 3: THE RECIPE (Allocations)
            # =============================================================================
            st.markdown("""
                <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); padding: 1.5rem; 
                            border-radius: 10px; border-left: 5px solid #ffc107; margin-bottom: 2rem;">
                    <h2 style="margin-top: 0; color: #2c3e50;">📖 3. The Recipe</h2>
                    <p style="font-size: 1.1rem; color: #555; margin-bottom: 0;">
                        <strong>Your Allocations:</strong> The proportions matter. Too much spice ruins the dish. 
                        Too little and it's bland.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🎯 Current Mix")
                
                # Categorize holdings
                growth_tickers = ['SPY', 'VTI', 'QQQ', 'VOO', 'VUG']
                bond_tickers = ['AGG', 'BND', 'TLT', 'IEF', 'SHY']
                international_tickers = ['VEA', 'VWO', 'EFA', 'IEMG', 'VXUS']
                
                growth_weight = sum([weights.get(t, 0) for t in growth_tickers])
                bond_weight = sum([weights.get(t, 0) for t in bond_tickers])
                intl_weight = sum([weights.get(t, 0) for t in international_tickers])
                other_weight = 1 - growth_weight - bond_weight - intl_weight
                
                mix_df = pd.DataFrame({
                    'Category': ['🥩 Growth', '🥗 Bonds', '🌶️ International', '🥄 Other'],
                    'Weight': [
                        f"{growth_weight*100:.1f}%",
                        f"{bond_weight*100:.1f}%",
                        f"{intl_weight*100:.1f}%",
                        f"{other_weight*100:.1f}%"
                    ]
                })
                st.dataframe(mix_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### 🎨 Style Analysis")
                
                # Determine style
                if growth_weight > 0.7:
                    style = "Aggressive Growth"
                    style_emoji = "🚀"
                    style_color = "#0f0506"
                elif growth_weight > 0.5 and bond_weight < 0.3:
                    style = "Growth"
                    style_emoji = "📈"
                    style_color = "#180f08"
                elif growth_weight > 0.4 and bond_weight > 0.3:
                    style = "Balanced"
                    style_emoji = "⚖️"
                    style_color = "#0e0b04"
                elif bond_weight > 0.5:
                    style = "Conservative"
                    style_emoji = "🛡️"
                    style_color = "#040c06"
                else:
                    style = "Custom"
                    style_emoji = "🎨"
                    style_color = "#06030b"
                
                st.markdown(f"""
                    <div style="background: {style_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                        <h2 style="margin: 0; font-size: 3rem;">{style_emoji}</h2>
                        <h4 style="margin: 0.5rem 0;">{style}</h4>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("#### 💡 Recipe Tips")
                
                # Actionable allocation guidance
                tips = []
                if growth_weight > 0.8:
                    tips.append("⚠️ Very high growth allocation - consider adding stabilizers")
                elif growth_weight < 0.3:
                    tips.append("💡 Low growth - may limit long-term returns")
                
                if bond_weight < 0.1 and growth_weight > 0.7:
                    tips.append("⚠️ High volatility risk - add some bonds for stability")
                
                if intl_weight < 0.1:
                    tips.append("🌍 Low international - missing diversification")
                elif intl_weight > 0.4:
                    tips.append("🌍 High international exposure")
                
                if not tips:
                    tips.append("✅ Allocation looks reasonable")
                
                for tip in tips:
                    st.info(tip)
            
            st.markdown("---")
            
            # =============================================================================
            # SECTION 4: WHEN TO ADD INGREDIENTS (Timing)
            # =============================================================================
            st.markdown("""
                <div style="background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); padding: 1.5rem; 
                            border-radius: 10px; border-left: 5px solid #17a2b8; margin-bottom: 2rem;">
                    <h2 style="margin-top: 0; color: #2c3e50;">⏰ 4. When to Add Ingredients</h2>
                    <p style="font-size: 1.1rem; color: #555; margin-bottom: 0;">
                        <strong>Timing Matters:</strong> Add ingredients at the right time. Don't add salt before sugar. 
                        Know when to accumulate and when to pull back.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌡️ Market Temperature")
                
                # Detect current market regime
                first_ticker = list(weights.keys())[0]
                if first_ticker in prices.columns:
                    ticker_prices = prices[first_ticker]
                    ticker_returns = ticker_prices.pct_change().dropna()
                    
                    # Calculate market metrics
                    recent_20d_return = (ticker_prices.iloc[-1] / ticker_prices.iloc[-20] - 1) * 100 if len(ticker_prices) >= 20 else 0
                    recent_60d_return = (ticker_prices.iloc[-1] / ticker_prices.iloc[-60] - 1) * 100 if len(ticker_prices) >= 60 else 0
                    vol_60d = ticker_returns.tail(60).std() * np.sqrt(252)
                    
                    # Determine temperature
                    if vol_60d > 0.35:
                        temp = "🔥 TOO HOT"
                        temp_desc = "Market Crisis / Panic"
                        temp_color = "#dc3545"
                        advice = "Step back from the stove. Don't add ingredients. Wait for market to cool."
                    elif recent_60d_return < -10 and vol_60d > 0.25:
                        temp = "❄️ COLD"
                        temp_desc = "Bear Market"
                        temp_color = "#6c757d"
                        advice = "Market cooling down. Prepare ingredients (build cash), wait for the right moment to add."
                    elif recent_60d_return > 15 and vol_60d < 0.20:
                        temp = "🌡️ WARM & STEADY"
                        temp_desc = "Bull Market"
                        temp_color = "#28a745"
                        advice = "Perfect cooking temperature! Keep adding ingredients (accumulate positions)."
                    elif recent_20d_return > 0:
                        temp = "🌤️ WARMING UP"
                        temp_desc = "Recovery"
                        temp_color = "#fd7e14"
                        advice = "Market heating up. Good time to start adding ingredients gradually."
                    else:
                        temp = "😐 LUKEWARM"
                        temp_desc = "Sideways / Choppy"
                        temp_color = "#ffc107"
                        advice = "Market temperature uncertain. Hold current recipe, wait for clearer signals."
                    
                    st.markdown(f"""
                        <div style="background: {temp_color}; color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                            <h2 style="margin: 0; font-size: 2rem;">{temp}</h2>
                            <h4 style="margin: 0.5rem 0; opacity: 0.9;">{temp_desc}</h4>
                            <p style="margin: 0.5rem 0; font-size: 0.9rem;">
                                20-day: {recent_20d_return:+.1f}% | 60-day: {recent_60d_return:+.1f}%<br>
                                Volatility: {vol_60d*100:.1f}%
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.info(f"**👨‍🍳 Chef's Advice:** {advice}")
            
            with col2:
                st.markdown("### 📋 Shopping List")
                st.markdown("*What to accumulate or reduce based on current signals*")
                
                # Create shopping list
                accumulate_list = []
                distribute_list = []
                hold_list = []
                
                for ticker in weights.keys():
                    if ticker in prices.columns:
                        signal_data = generate_trading_signal(prices[ticker],ticker)
                        if signal_data['action'] == 'Accumulate':
                            accumulate_list.append(f"**{ticker}** ({signal_data['confidence']:.0f}% confident)")
                        elif signal_data['action'] == 'Distribute':
                            distribute_list.append(f"**{ticker}** ({signal_data['confidence']:.0f}% confident)")
                        else:
                            hold_list.append(f"**{ticker}**")
                
                if accumulate_list:
                    st.markdown("**🟢 Buy More (Accumulate):**")
                    for item in accumulate_list:
                        st.markdown(f"✅ {item}")
                
                if distribute_list:
                    st.markdown("**🔴 Reduce (Distribute):**")
                    for item in distribute_list:
                        st.markdown(f"⛔ {item}")
                
                if hold_list:
                    st.markdown("**🟡 Keep Current (Hold):**")
                    for item in hold_list:
                        st.markdown(f"➖ {item}")
            
            st.markdown("---")
            
            # =============================================================================
            # SECTION 5: HOW DOES IT TASTE? (Performance - ALL 8 METRICS)
            # =============================================================================
            st.markdown("""
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1.5rem; 
                            border-radius: 10px; border-left: 5px solid #2196f3; margin-bottom: 2rem;">
                    <h2 style="margin-top: 0; color: #2c3e50;">👅 5. How Does It Taste?</h2>
                    <p style="font-size: 1.1rem; color: #555; margin-bottom: 0;">
                        <strong>The Results:</strong> Time to taste your creation. Does it meet your expectations?
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Calculate SPY
            try:
                spy_data = download_ticker_data(['SPY'], current['start_date'], current['end_date'])
                if spy_data is not None:
                    spy_returns = spy_data.pct_change().dropna()
                    spy_metrics = calculate_portfolio_metrics(spy_returns)
                else:
                    spy_metrics = None
            except:
                spy_metrics = None
            
            def get_comparison_indicator(portfolio_value, spy_value, metric_type='higher_better'):
                if spy_metrics is None:
                    return "", "white"
                if metric_type == 'higher_better':
                    return ("🟢 ↑", "#28a745") if portfolio_value > spy_value else ("🔴 ↓", "#dc3545") if portfolio_value < spy_value else ("⚪ →", "#ffc107")
                else:
                    return ("🟢 ↑", "#28a745") if portfolio_value < spy_value else ("🔴 ↓", "#dc3545") if portfolio_value > spy_value else ("⚪ →", "#ffc107")
            
            # First row of metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                metric_class = get_metric_color_class('annual_return', metrics['Annual Return'])
                arrow, color = get_comparison_indicator(metrics['Annual Return'], spy_metrics['Annual Return'] if spy_metrics else 0, 'higher_better')
                st.markdown(f"""
                    <div class="{metric_class}">
                        <h4>Annual Return {arrow}</h4>
                        <h2>{metrics['Annual Return']:.2%}</h2>
                        <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Annual Return']:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
                render_metric_explanation('annual_return')
            
            with col2:
                metric_class = get_metric_color_class('sharpe_ratio', metrics['Sharpe Ratio'])
                arrow, color = get_comparison_indicator(metrics['Sharpe Ratio'], spy_metrics['Sharpe Ratio'] if spy_metrics else 0, 'higher_better')
                st.markdown(f"""
                    <div class="{metric_class}">
                        <h4>Sharpe Ratio {arrow}</h4>
                        <h2>{metrics['Sharpe Ratio']:.2f}</h2>
                        <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Sharpe Ratio']:.2f}</p>
                    </div>
                """, unsafe_allow_html=True)
                render_metric_explanation('sharpe_ratio')
            
            with col3:
                metric_class = get_metric_color_class('max_drawdown', metrics['Max Drawdown'])
                arrow, color = get_comparison_indicator(metrics['Max Drawdown'], spy_metrics['Max Drawdown'] if spy_metrics else 0, 'lower_better')
                st.markdown(f"""
                    <div class="{metric_class}">
                        <h4>Max Drawdown {arrow}</h4>
                        <h2>{metrics['Max Drawdown']:.2%}</h2>
                        <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Max Drawdown']:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
                render_metric_explanation('max_drawdown')
            
            with col4:
                metric_class = get_metric_color_class('volatility', metrics['Annual Volatility'])
                arrow, color = get_comparison_indicator(metrics['Annual Volatility'], spy_metrics['Annual Volatility'] if spy_metrics else 0, 'lower_better')
                st.markdown(f"""
                    <div class="{metric_class}">
                        <h4>Volatility {arrow}</h4>
                        <h2>{metrics['Annual Volatility']:.2%}</h2>
                        <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Annual Volatility']:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
                render_metric_explanation('volatility')
            
            # Second row of metrics
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                metric_class = get_metric_color_class('sortino_ratio', metrics['Sortino Ratio'])
                arrow, color = get_comparison_indicator(metrics['Sortino Ratio'], spy_metrics['Sortino Ratio'] if spy_metrics else 0, 'higher_better')
                st.markdown(f"""
                    <div class="{metric_class}">
                        <h4>Sortino Ratio {arrow}</h4>
                        <h2>{metrics['Sortino Ratio']:.2f}</h2>
                        <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Sortino Ratio']:.2f}</p>
                    </div>
                """, unsafe_allow_html=True)
                render_metric_explanation('sortino_ratio')
            
            with col2:
                metric_class = get_metric_color_class('calmar_ratio', metrics['Calmar Ratio'])
                arrow, color = get_comparison_indicator(metrics['Calmar Ratio'], spy_metrics['Calmar Ratio'] if spy_metrics else 0, 'higher_better')
                st.markdown(f"""
                    <div class="{metric_class}">
                        <h4>Calmar Ratio {arrow}</h4>
                        <h2>{metrics['Calmar Ratio']:.2f}</h2>
                        <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Calmar Ratio']:.2f}</p>
                    </div>
                """, unsafe_allow_html=True)
                render_metric_explanation('calmar_ratio')
            
            with col3:
                metric_class = get_metric_color_class('win_rate', metrics['Win Rate'])
                arrow, color = get_comparison_indicator(metrics['Win Rate'], spy_metrics['Win Rate'] if spy_metrics else 0, 'higher_better')
                st.markdown(f"""
                    <div class="{metric_class}">
                        <h4>Win Rate {arrow}</h4>
                        <h2>{metrics['Win Rate']:.2%}</h2>
                        <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Win Rate']:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
                render_metric_explanation('win_rate')
            
            with col4:
                arrow, color = get_comparison_indicator(metrics['Total Return'], spy_metrics['Total Return'] if spy_metrics else 0, 'higher_better')
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>Total Return {arrow}</h4>
                        <h2>{metrics['Total Return']:.2%}</h2>
                        <p style="font-size: 0.9em; color: #888;">SPY: {spy_metrics['Total Return']:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Comparison legend
            st.markdown("""
                <div style="text-align: center; padding: 10px; margin-top: 10px; background: #f8f9fa; border-radius: 5px;">
                    <small>
                        <strong>Comparison Legend:</strong>  
                        🟢 ↑ = Better than S&P 500 | 🔴 ↓ = Worse than S&P 500 | ⚪ → = Equal
                    </small>
                </div>
            """, unsafe_allow_html=True)
            
            # Performance Chart
            st.markdown("---")
            st.markdown("### 📈 Performance Over Time")
            fig = plot_cumulative_returns(portfolio_returns, f'{st.session_state.current_portfolio} - Cumulative Returns')
            st.pyplot(fig)
            
            st.markdown("""
                <div class="interpretation-box">
                    <div class="interpretation-title">💡 What This Chart Means</div>
                    <p><strong>How to Read:</strong> Shows how $1 invested grows over time. Value of 1.5 = 50% gain.</p>
                    <p><strong>Look For:</strong> Steady upward trend = good. Sharp drops = drawdowns.</p>
                    <p><strong>Action Item:</strong> If line trends down 6+ months, consider rebalancing.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Drawdown Chart
            st.markdown("---")
            st.markdown("### 📉 Drawdown Analysis")
            fig = plot_drawdown(portfolio_returns, 'Portfolio Drawdown')
            st.pyplot(fig)
            
            st.markdown("""
                <div class="interpretation-box">
                    <div class="interpretation-title">💡 Understanding Drawdowns</div>
                    <p><strong>What This Shows:</strong> How much you're underwater from peak value.</p>
                    <p><strong>Red Flag:</strong> Drawdown exceeding -20% = bear market territory. Don't panic-sell!</p>
                    <p><strong>Psychology Check:</strong> Can you handle the deepest drawdown without selling?</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Final Verdict
            st.markdown("---")
            score = 0
            if spy_metrics:
                if metrics['Annual Return'] > spy_metrics['Annual Return']:
                    score += 1
                if metrics['Sharpe Ratio'] > spy_metrics['Sharpe Ratio']:
                    score += 1
                if abs(metrics['Max Drawdown']) < abs(spy_metrics['Max Drawdown']):
                    score += 1
                if metrics['Annual Volatility'] < spy_metrics['Annual Volatility']:
                    score += 1
            
            if score >= 3:
                verdict = "🌟 Excellent Recipe!"
                verdict_color = "#28a745"
                verdict_text = "Your portfolio is beating the market on most metrics. This is a well-balanced, high-quality recipe. Keep cooking!"
            elif score == 2:
                verdict = "👍 Good Recipe"
                verdict_color = "#20c997"
                verdict_text = "Your portfolio is competitive with the market. Some ingredients are working well. Fine-tune the recipe for even better results."
            elif score == 1:
                verdict = "🤔 Needs Adjustment"
                verdict_color = "#ffc107"
                verdict_text = "Your portfolio is underperforming on most metrics. Time to adjust the recipe - check your ingredient proportions and timing."
            else:
                verdict = "⚠️ Recipe Needs Work"
                verdict_color = "#dc3545"
                verdict_text = "Your portfolio is significantly underperforming. Consider revisiting your ingredients, proportions, and timing strategy."
            
            st.markdown(f"""
                <div style="background: {verdict_color}; color: white; padding: 2rem; border-radius: 15px; text-align: center;">
                    <h1 style="margin: 0; font-size: 3rem;">{verdict}</h1>
                    <p style="font-size: 1.2rem; margin: 1rem 0 0 0; opacity: 0.95;">
                        {verdict_text}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # =============================================================================
        # TAB 2: DETAILED ANALYSIS
        # =============================================================================
        
