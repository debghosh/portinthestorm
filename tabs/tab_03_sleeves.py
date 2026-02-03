"""
Tab: Sleeves
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from helper_functions import *


def render(tab3, portfolio_returns, prices, weights, tickers, metrics, current):
    """Render the Sleeves tab"""
    
    with tab3:
            st.markdown("## 🎯 Portfolio Sleeves Analysis")
            st.markdown("""
                <div class="info-box">
                    <h4>What Are Investment Sleeves?</h4>
                    <p>Your portfolio is like a wardrobe - different "sleeves" serve different purposes. 
                    A balanced portfolio combines multiple sleeves to achieve your goals while managing risk.</p>
                </div>
            """, unsafe_allow_html=True)
            
            if current:
                # Get current portfolio composition
                tickers = current['tickers']
                weights_dict = current['weights']
                
                # Define sleeve categories and their ETFs
                SLEEVE_DEFINITIONS = {
                    'Total Market (Core)': {
                        'etfs': ['SPY', 'VOO', 'IVV', 'VTI', 'ITOT', 'SCHB'],
                        'description': 'Broad market exposure - the foundation of your portfolio',
                        'purpose': 'Capture overall market returns with maximum diversification',
                        'typical_allocation': '30-50%',
                        'color': '#4CAF50'
                    },
                    'Growth': {
                        'etfs': ['QQQ', 'VUG', 'VGT', 'IWF', 'SCHG', 'MGK', 'VOOG', 'IVW'],
                        'description': 'High-growth companies and technology',
                        'purpose': 'Outperform during bull markets, higher risk/reward',
                        'typical_allocation': '10-30%',
                        'color': '#2196F3'
                    },
                    'Dividend & Income': {
                        'etfs': ['SCHD', 'VIG', 'VYM', 'DGRO', 'DVY', 'NOBL', 'SDY', 'HDV'],
                        'description': 'Quality dividend-paying stocks',
                        'purpose': 'Generate income, lower volatility, value tilt',
                        'typical_allocation': '10-25%',
                        'color': '#FF9800'
                    },
                    'Factors (Quality/Momentum/Value)': {
                        'etfs': ['QUAL', 'MTUM', 'VTV', 'VLUE', 'USMV', 'SIZE', 'IJR', 'VB', 'SCHA'],
                        'description': 'Factor-based strategies (quality, momentum, value, low volatility)',
                        'purpose': 'Target specific return drivers, tactical tilts',
                        'typical_allocation': '5-15%',
                        'color': '#9C27B0'
                    },
                    'Defensive Ballast (Bonds)': {
                        'etfs': ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'TIP', 'LQD', 'MUB', 'VCIT', 'BSV'],
                        'description': 'Fixed income for stability',
                        'purpose': 'Reduce volatility, preserve capital, provide dry powder',
                        'typical_allocation': '20-40%',
                        'color': '#607D8B'
                    },
                    'International': {
                        'etfs': ['VEA', 'VXUS', 'EFA', 'VWO', 'IEMG', 'IXUS'],
                        'description': 'Global diversification',
                        'purpose': 'Access non-US growth, currency diversification',
                        'typical_allocation': '10-20%',
                        'color': '#00BCD4'
                    }
                }
                
                # Categorize portfolio holdings into sleeves
                sleeve_allocation = {sleeve: 0.0 for sleeve in SLEEVE_DEFINITIONS.keys()}
                uncategorized = 0.0
                ticker_to_sleeve = {}
                
                for ticker, weight in weights_dict.items():
                    categorized = False
                    for sleeve, info in SLEEVE_DEFINITIONS.items():
                        if ticker in info['etfs']:
                            sleeve_allocation[sleeve] += weight
                            ticker_to_sleeve[ticker] = sleeve
                            categorized = True
                            break
                    if not categorized:
                        uncategorized += weight
                        ticker_to_sleeve[ticker] = 'Other'
                
                # Current Sleeve Allocation
                st.markdown("---")
                st.markdown("### 📊 Your Current Sleeve Allocation")
                
                # Create visualization
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Pie chart
                    import plotly.graph_objects as go
                    
                    labels = []
                    values = []
                    colors = []
                    
                    for sleeve, allocation in sleeve_allocation.items():
                        if allocation > 0:
                            labels.append(sleeve)
                            values.append(allocation * 100)
                            colors.append(SLEEVE_DEFINITIONS[sleeve]['color'])
                    
                    if uncategorized > 0:
                        labels.append('Other')
                        values.append(uncategorized * 100)
                        colors.append('#757575')
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        marker=dict(colors=colors),
                        hole=0.4,
                        textinfo='label+percent',
                        textposition='outside'
                    )])
                    
                    fig.update_layout(
                        title="Portfolio Sleeve Breakdown",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Current Allocation by Sleeve")
                    
                    for sleeve, allocation in sorted(sleeve_allocation.items(), key=lambda x: x[1], reverse=True):
                        if allocation > 0:
                            st.markdown(f"""
                                <div style="background-color: {SLEEVE_DEFINITIONS[sleeve]['color']}20; 
                                            padding: 10px; margin: 5px 0; border-radius: 5px;
                                            border-left: 4px solid {SLEEVE_DEFINITIONS[sleeve]['color']}">
                                    <strong>{sleeve}:</strong> {allocation*100:.1f}%<br>
                                    <small style="color: #666">{SLEEVE_DEFINITIONS[sleeve]['description']}</small>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    if uncategorized > 0:
                        st.markdown(f"""
                            <div style="background-color: #75757520; padding: 10px; margin: 5px 0; 
                                        border-radius: 5px; border-left: 4px solid #757575">
                                <strong>Other:</strong> {uncategorized*100:.1f}%<br>
                                <small style="color: #666">Uncategorized holdings</small>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Sleeve Analysis
                st.markdown("---")
                st.markdown("### 🎯 Sleeve-by-Sleeve Analysis")
                
                for sleeve, info in SLEEVE_DEFINITIONS.items():
                    allocation = sleeve_allocation[sleeve]
                    typical_range = info['typical_allocation']
                    
                    with st.expander(f"**{sleeve}** - Current: {allocation*100:.1f}% (Typical: {typical_range})"):
                        col_a, col_b = st.columns([1, 1])
                        
                        with col_a:
                            st.markdown(f"**Purpose:** {info['purpose']}")
                            st.markdown(f"**Typical Allocation:** {typical_range}")
                            
                            # Status indicator
                            typical_low = int(typical_range.split('-')[0].replace('%', ''))
                            typical_high = int(typical_range.split('-')[1].replace('%', ''))
                            current_pct = allocation * 100
                            
                            if current_pct < typical_low:
                                status = "⚠️ **Underweight** - Consider adding"
                                status_color = "#FF9800"
                            elif current_pct > typical_high:
                                status = "⚠️ **Overweight** - Consider reducing"
                                status_color = "#F44336"
                            else:
                                status = "✅ **Optimal** - Well balanced"
                                status_color = "#4CAF50"
                            
                            st.markdown(f"**Status:** <span style='color: {status_color}'>{status}</span>", 
                                        unsafe_allow_html=True)
                            
                            # Show holdings in this sleeve
                            holdings_in_sleeve = [t for t, s in ticker_to_sleeve.items() if s == sleeve]
                            if holdings_in_sleeve:
                                st.markdown(f"**Your Holdings:** {', '.join(holdings_in_sleeve)}")
                            else:
                                st.markdown("**Your Holdings:** None")
                        
                        with col_b:
                            st.markdown("**💡 Recommended ETFs for This Sleeve:**")
                            
                            # Get top 3 recommended ETFs
                            recommended = info['etfs'][:3]
                            for etf in recommended:
                                if etf in holdings_in_sleeve:
                                    st.markdown(f"• ✅ **{etf}** - *Already in portfolio*")
                                else:
                                    st.markdown(f"• {etf}")
                            
                            if len(info['etfs']) > 3:
                                with st.expander(f"See all {len(info['etfs'])} options"):
                                    st.markdown(", ".join(info['etfs']))
                
                # Recommendations
                st.markdown("---")
                st.markdown("### 💡 Portfolio Sleeve Recommendations")
                
                recommendations = []
                
                # Check each sleeve allocation
                for sleeve, info in SLEEVE_DEFINITIONS.items():
                    allocation = sleeve_allocation[sleeve]
                    typical_range = info['typical_allocation']
                    typical_low = int(typical_range.split('-')[0].replace('%', ''))
                    typical_high = int(typical_range.split('-')[1].replace('%', ''))
                    current_pct = allocation * 100
                    
                    if current_pct < typical_low:
                        gap = typical_low - current_pct
                        recommendations.append({
                            'type': 'underweight',
                            'sleeve': sleeve,
                            'current': current_pct,
                            'target': typical_low,
                            'gap': gap,
                            'action': f"Add {gap:.0f}% to {sleeve}",
                            'suggested_etfs': info['etfs'][:2]
                        })
                    elif current_pct > typical_high:
                        excess = current_pct - typical_high
                        recommendations.append({
                            'type': 'overweight',
                            'sleeve': sleeve,
                            'current': current_pct,
                            'target': typical_high,
                            'gap': excess,
                            'action': f"Reduce {sleeve} by {excess:.0f}%",
                            'suggested_etfs': []
                        })
                
                if recommendations:
                    st.markdown("**🎯 Action Items to Balance Your Sleeves:**")
                    
                    for i, rec in enumerate(recommendations, 1):
                        if rec['type'] == 'underweight':
                            st.markdown(f"""
                                <div style="background-color: #FFF3E0; padding: 15px; margin: 10px 0; 
                                            border-radius: 5px; border-left: 4px solid #FF9800">
                                    <strong>{i}. {rec['action']}</strong><br>
                                    Current: {rec['current']:.1f}% → Target: {rec['target']:.0f}%<br>
                                    <strong>Suggested ETFs:</strong> {', '.join(rec['suggested_etfs'])}
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div style="background-color: #FFEBEE; padding: 15px; margin: 10px 0; 
                                            border-radius: 5px; border-left: 4px solid #F44336">
                                    <strong>{i}. {rec['action']}</strong><br>
                                    Current: {rec['current']:.1f}% → Target: {rec['target']:.0f}%
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.success("✅ Your portfolio sleeves are well balanced!")
                
                # Sleeve Performance Comparison (if historical data available)
                st.markdown("---")
                st.markdown("### 📈 How Different Sleeves Would Have Performed")
                st.markdown("*Based on representative ETFs from each sleeve*")
                
                # This would use historical data to show how each sleeve performed
                st.info("💡 **Tip:** The Market Regime tab shows how different sleeves perform in various market conditions. "
                       "Check it out to understand when each sleeve shines!")
            
            else:
                st.info("👆 Build a portfolio first to see sleeves analysis")
        
