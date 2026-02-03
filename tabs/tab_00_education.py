"""
Tab: Portfolio Education
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from helper_functions import *


def render(tab0):
    """Render the Portfolio Education tab"""
    
    with tab0:
        st.markdown("""
            <div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 15px; color: white; margin-bottom: 2rem;">
                <h1 style="margin: 0; font-size: 3rem;">🎯 ETF Sleeve Builder</h1>
                <p style="font-size: 1.3rem; margin-top: 0.8rem; opacity: 0.95;">
                    Professional ETF analysis • Best-in-class selections • Build your optimal portfolio
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # =============================================================================
        # SLEEVE 1: CORE MARKET (Overall Market Exposure)
        # =============================================================================
        
        with st.expander("🏢 **SLEEVE 1: Core Market - Overall Market Exposure**", expanded=True):
            st.markdown("### The Foundation: Broad Market ETFs")
            st.markdown("*Your portfolio's anchor. Choose ONE as your core holding (30-60% allocation)*")
            
            core_etfs = pd.DataFrame({
                'ETF': ['SPY', 'VOO', 'IVV', 'VTI', 'ITOT'],
                'Name': ['SPDR S&P 500', 'Vanguard S&P 500', 'iShares Core S&P 500', 'Vanguard Total Market', 'iShares Core Total Market'],
                'Expense Ratio': ['0.09%', '0.03%', '0.03%', '0.03%', '0.03%'],
                'Holdings': ['503', '503', '503', '3,800+', '3,600+'],
                'Liquidity': ['Highest', 'High', 'High', 'High', 'Medium'],
                'Best For': ['Trading/Options', 'Low cost, tax efficient', 'Low cost, reliable', 'Total market exposure', 'Lower cost total market']
            })
            
            st.dataframe(core_etfs, use_container_width=True, hide_index=True)
            
            st.markdown("""
                **🎯 MY RECOMMENDATION:**
                - **Primary Choice: VOO** (Vanguard S&P 500)
                  - Lowest expense ratio among S&P 500 funds (0.03%)
                  - Excellent tax efficiency
                  - Strong tracking, massive AUM ($500B+)
                  - Best for long-term buy-and-hold investors
                
                - **Alternative: VTI** (if you want total market)
                  - Adds mid/small cap exposure (3,800+ holdings vs 503)
                  - Still only 0.03% expense ratio
                  - Slightly more volatile but better diversification
                  - Historical performance nearly identical to VOO
                
                - **For Active Traders: SPY**
                  - Most liquid ETF in the world
                  - Best for options trading
                  - Higher expense ratio (0.09%) is worth it for liquidity
                
                **⚠️ AVOID:** Redundancy - Don't own SPY + VOO + IVV (they're all the same index!)
            """)
            
            if st.button("📥 Load Core: VOO 50%", key="load_voo"):
                st.session_state.loaded_model = {
                    'name': 'Core Market - VOO',
                    'tickers': ['VOO'],
                    'weights': {'VOO': 0.50}
                }
                st.success("✅ Loaded! Add more sleeves in sidebar")
        
        # =============================================================================
        # SLEEVE 2: GROWTH
        # =============================================================================
        
        with st.expander("🚀 **SLEEVE 2: Growth - Technology & Innovation**"):
            st.markdown("### Growth & Technology ETFs")
            st.markdown("*Higher returns, higher volatility. Recommended 10-30% allocation.*")
            
            growth_etfs = pd.DataFrame({
                'ETF': ['QQQ', 'VUG', 'VGT', 'IWF', 'SCHG', 'MGK'],
                'Name': ['Invesco QQQ', 'Vanguard Growth', 'Vanguard Info Tech', 'iShares Russell 1000 Growth', 'Schwab US Large Growth', 'Vanguard Mega Cap Growth'],
                'Expense Ratio': ['0.20%', '0.04%', '0.10%', '0.19%', '0.04%', '0.07%'],
                'Top Holdings': ['AAPL, MSFT, NVDA', 'AAPL, MSFT, AMZN', '100% Tech', 'AAPL, MSFT, NVDA', 'AAPL, MSFT, AMZN', 'AAPL, MSFT, AMZN'],
                'Concentration': ['Top 10: 49%', 'Top 10: 42%', 'Top 10: 68%', 'Top 10: 45%', 'Top 10: 43%', 'Top 10: 62%'],
                '10Yr Return': ['18.9%', '15.2%', '20.1%', '15.5%', 'N/A (2009)', '16.8%']
            })
            
            st.dataframe(growth_etfs, use_container_width=True, hide_index=True)
            
            st.markdown("""
                **🎯 MY RECOMMENDATION:**
                - **Best Pure Growth: QQQ** (Nasdaq-100)
                  - Proven performer: 18.9% annualized over 10 years
                  - Concentrated tech exposure (top 10 = 49% of fund)
                  - Higher expense ratio (0.20%) but worth it for performance
                  - **Use case:** Bull markets, long time horizon (10+ years)
                  - **Risk:** Crashes hard in bear markets (-30% in 2022)
                
                - **Best Diversified Growth: VUG**
                  - Only 0.04% expense ratio (5x cheaper than QQQ)
                  - Broader growth exposure, less concentrated
                  - Better risk-adjusted returns (lower volatility than QQQ)
                  - **Use case:** Growth exposure without tech concentration risk
                
                - **Most Aggressive: VGT** (100% Technology)
                  - 20.1% annualized returns (highest)
                  - ALL technology - no other sectors
                  - Top 10 holdings = 68% of fund (extremely concentrated)
                  - **Use case:** Strong tech conviction, can stomach 40%+ drawdowns
                
                **💡 SMART COMBINATION:**
                - 60% VUG (core growth) + 40% VGT (tech tilt) = balanced growth exposure
                - Or: 100% QQQ if you want simplicity + proven track record
                
                **⚠️ WARNING:** Don't combine QQQ + VUG + VGT - massive overlap!
            """)
            
            if st.button("📥 Load Growth: QQQ 20%", key="load_qqq"):
                st.session_state.loaded_model = {
                    'name': 'Growth - QQQ',
                    'tickers': ['QQQ'],
                    'weights': {'QQQ': 0.20}
                }
                st.success("✅ Loaded!")
        
        # =============================================================================
        # SLEEVE 3: DIVIDEND & INCOME
        # =============================================================================
        
        with st.expander("💰 **SLEEVE 3: Dividend & Income - Cash Flow Generation**"):
            st.markdown("### Dividend Growth ETFs")
            st.markdown("*Income + growth. Recommended 15-25% for retirement portfolios.*")
            
            div_etfs = pd.DataFrame({
                'ETF': ['SCHD', 'VIG', 'DGRO', 'VYM', 'DVY', 'NOBL'],
                'Name': ['Schwab US Dividend Equity', 'Vanguard Dividend Appreciation', 'iShares Core Div Growth', 'Vanguard High Div Yield', 'iShares Select Dividend', 'ProShares S&P 500 Div Aristocrats'],
                'Expense Ratio': ['0.06%', '0.06%', '0.08%', '0.06%', '0.38%', '0.35%'],
                'Yield': ['3.5%', '1.8%', '2.5%', '2.9%', '3.6%', '2.1%'],
                'Div Growth': ['10yr: 12% avg', '10yr: 8% avg', '5yr: 10% avg', '10yr: 6% avg', '10yr: 5% avg', '10yr: 9% avg'],
                '10Yr Return': ['13.2%', '11.8%', 'N/A (2014)', '10.5%', '10.1%', '12.4%']
            })
            
            st.dataframe(div_etfs, use_container_width=True, hide_index=True)
            
            st.markdown("""
                **🎯 MY RECOMMENDATION:**
                - **#1 BEST: SCHD** (Schwab US Dividend Equity)
                  - **Winner**: Highest total return (13.2% annualized)
                  - 3.5% yield + 12% annual dividend growth = compounding machine
                  - Only 0.06% expense ratio
                  - Quality focused (financial health metrics)
                  - Top holdings: HD, CSCO, TXN, PEP, VZ
                  - **Perfect for:** Tax-advantaged accounts, dividend reinvestment
                
                - **Best for Qualified Dividends: VIG**
                  - Lower yield (1.8%) but higher quality companies
                  - 10+ years consecutive dividend increases required
                  - More growth-oriented than SCHD
                  - **Use case:** Taxable accounts (qualified dividends)
                
                - **Highest Yield: DVY**
                  - 3.6% yield is attractive
                  - **BUT**: High expense ratio (0.38% = 6x SCHD)
                  - Lower quality companies
                  - Underperforms SCHD on total return
                  - **Only use if:** You NEED current income NOW
                
                **💡 OPTIMAL DIVIDEND SLEEVE:**
                - **Simple:** 100% SCHD (best performance + yield balance)
                - **Advanced:** 70% SCHD + 30% VIG (yield + quality growth)
                - **Income Focus:** 80% SCHD + 20% DVY (maximize yield)
                
                **🔥 POWER MOVE:**
                - SCHD in Roth IRA: Tax-free dividends + tax-free growth = wealth building machine
                - Dividend reinvestment: 3.5% yield * 12% growth = doubles every ~5 years
                
                **⚠️ MISTAKE TO AVOID:** Don't buy SCHD in taxable account if in high tax bracket
            """)
            
            if st.button("📥 Load Dividend: SCHD 20%", key="load_schd"):
                st.session_state.loaded_model = {
                    'name': 'Dividend - SCHD',
                    'tickers': ['SCHD'],
                    'weights': {'SCHD': 0.20}
                }
                st.success("✅ Loaded!")
        
        # =============================================================================
        # SLEEVE 4: DEFENSIVE BALLAST (Bonds)
        # =============================================================================
        
        with st.expander("🛡️ **SLEEVE 4: Defensive Ballast - Bonds & Stability**"):
            st.markdown("### Bond ETFs - Your Portfolio Shock Absorber")
            st.markdown("*Reduces volatility. Recommended 20-40% for conservative portfolios.*")
            
            bond_etfs = pd.DataFrame({
                'ETF': ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'TIP', 'LQD'],
                'Name': ['iShares Core Agg Bond', 'Vanguard Total Bond', 'iShares 20+ Yr Treasury', 'iShares 7-10 Yr Treasury', 'iShares 1-3 Yr Treasury', 'iShares TIPS', 'iShares Invest Grade Corp'],
                'Expense Ratio': ['0.03%', '0.03%', '0.15%', '0.15%', '0.15%', '0.19%', '0.14%'],
                'Duration': ['6.2 years', '6.7 years', '17.5 years', '7.5 years', '1.9 years', '7.4 years', '8.6 years'],
                'Yield': ['4.5%', '4.6%', '4.7%', '4.3%', '5.1%', '5.2% (real)', '5.3%'],
                'Volatility': ['Low', 'Low', 'VERY HIGH', 'Medium', 'Very Low', 'Medium', 'Medium']
            })
            
            st.dataframe(bond_etfs, use_container_width=True, hide_index=True)
            
            st.markdown("""
                **🎯 MY RECOMMENDATION BY SCENARIO:**
                
                **Scenario 1: Normal Diversification (most people)**
                - **Best: AGG** (iShares Core Aggregate Bond)
                  - Gold standard aggregate bond fund
                  - Diversified: Treasuries + Corporates + MBS
                  - 6.2 year duration = moderate interest rate sensitivity
                  - 4.5% yield + capital appreciation potential
                  - Only 0.03% expense ratio
                  - **Use when:** You want "set and forget" bond exposure
                
                **Scenario 2: Deflation / Recession Protection**
                - **Best: TLT** (Long-term Treasuries 20+ years)
                  - **WARNING:** VERY volatile (acts like stocks sometimes)
                  - When stocks crash, TLT often surges (flight to safety)
                  - 2008: TLT +34% while SPY -37%
                  - 2022: TLT -31% (interest rates rose)
                  - **Only use if:** You're hedging a stock crash
                  - **Allocation:** 10-15% max
                
                **Scenario 3: Rising Interest Rates**
                - **Best: SHY** (Short-term 1-3 year Treasuries)
                  - Minimal interest rate risk (1.9 year duration)
                  - Currently 5.1% yield (higher than long-term!)
                  - When rates rise, SHY barely moves
                  - **Use when:** Fed is raising rates or rates volatile
                
                **Scenario 4: Inflation Protection**
                - **Best: TIP** (Treasury Inflation-Protected)
                  - Yield adjusts with inflation (5.2% real yield)
                  - Principal increases with CPI
                  - 2022: TIP -12% vs AGG -13% (better in high inflation)
                  - **Use when:** Inflation expectations rising
                
                **💡 OPTIMAL BOND SLEEVE (Current Environment - Jan 2026):**
                - **Conservative:** 60% AGG + 30% TLT + 10% TIP
                - **Moderate:** 100% AGG (simple, diversified)
                - **Defensive:** 50% AGG + 50% SHY (low volatility)
                - **Inflation Hedge:** 70% TIP + 30% SHY
                
                **🔥 CURRENT MARKET VIEW (My Opinion):**
                - Fed rate cuts likely in 2026 → TLT could benefit
                - But use SMALL allocation (10-15%) due to volatility
                - Core should be AGG for stability
                - Add TIP if inflation concerns persist
                
                **⚠️ CRITICAL WARNING:**
                - TLT is NOT a "safe" bond fund - it's highly volatile!
                - Duration = price sensitivity to rates (17.5 years = 17.5% move per 1% rate change)
                - Don't use TLT as your only bond holding!
            """)
            
            if st.button("📥 Load Bonds: AGG 25% + TLT 10%", key="load_bonds"):
                st.session_state.loaded_model = {
                    'name': 'Bonds - Balanced',
                    'tickers': ['AGG', 'TLT'],
                    'weights': {'AGG': 0.25, 'TLT': 0.10}
                }
                st.success("✅ Loaded!")
        
        # =============================================================================
        # SLEEVE 5: INTERNATIONAL
        # =============================================================================
        
        with st.expander("🌍 **SLEEVE 5: International - Global Diversification**"):
            st.markdown("### International Equity ETFs")
            st.markdown("*Reduce US dependence. Recommended 15-25% allocation.*")
            
            intl_etfs = pd.DataFrame({
                'ETF': ['VEA', 'VXUS', 'EFA', 'VWO', 'IEMG', 'IXUS'],
                'Name': ['Vanguard Developed Markets', 'Vanguard Total Intl', 'iShares MSCI EAFE', 'Vanguard Emerging Markets', 'iShares Core Emerging', 'iShares Core Total Intl'],
                'Expense Ratio': ['0.05%', '0.08%', '0.32%', '0.08%', '0.09%', '0.09%'],
                'Geography': ['Developed only', 'Dev + Emerging', 'Developed only', 'Emerging only', 'Emerging only', 'Dev + Emerging'],
                'Top Countries': ['Japan, UK, Canada', 'Japan, UK, China', 'Japan, UK, France', 'China, India, Taiwan', 'China, India, Taiwan', 'Japan, UK, China'],
                '10Yr Return': ['5.2%', '4.8%', '5.0%', '3.1%', '3.8%', '4.7%']
            })
            
            st.dataframe(intl_etfs, use_container_width=True, hide_index=True)
            
            st.markdown("""
                **🎯 MY HONEST ASSESSMENT:**
                
                **The Uncomfortable Truth:**
                - International has UNDERPERFORMED US for 15 years straight
                - VEA 10-year: 5.2% vs VOO: 13.1% (less than HALF the returns!)
                - Many investors regret international allocation
                - **BUT**: Past performance ≠ future returns
                
                **Why Still Consider International?**
                1. **Valuation:** Intl P/E ~13 vs US P/E ~21 (cheaper)
                2. **Currency hedge:** Diversifies away from USD
                3. **Different cycles:** When US slows, others may lead
                4. **Mean reversion:** Eventually valuations matter
                
                **🎯 MY RECOMMENDATION:**
                
                **For Most Investors:**
                - **Best: VEA** (Developed Markets only)
                  - 0.05% expense ratio (lowest cost)
                  - Developed markets = less risk than emerging
                  - Japan, UK, Canada = stable economies
                  - **Allocation:** 10-15% of portfolio
                  - **Why not more:** US companies already have global revenue
                
                **For Aggressive Allocation:**
                - **Best: VXUS** (Total International)
                  - Developed + Emerging in market cap weights
                  - "Set and forget" international
                  - 0.08% expense ratio
                  - **Allocation:** 20-30% if you want max diversification
                
                **For Emerging Markets Believers:**
                - **Best: VWO** (Emerging Markets)
                  - China, India, Taiwan growth story
                  - Higher risk, higher potential reward
                  - **BUT**: 10-year returns only 3.1% (weak)
                  - **Allocation:** 5-10% max, only if high conviction
                
                **💡 MY PERSONAL APPROACH:**
                - **Conservative:** 10% VEA (minimal international)
                - **Moderate:** 20% VXUS (standard diversification)
                - **Aggressive:** Skip international entirely, focus on US
                
                **🔥 CONTRARIAN VIEW:**
                - International is CHEAP relative to US
                - If US multiple compresses from 21 to 18 = -14% return
                - If Intl multiple expands from 13 to 15 = +15% return
                - Next 10 years could favor international
                - **But**: I'm not betting heavy on it (10-15% allocation only)
                
                **⚠️ WHAT I'D AVOID:**
                - EFA: 0.32% expense ratio is 6x VEA (no reason to use)
                - Heavy EM allocation: Too volatile, poor historical returns
                - 40%+ international: US companies already global
            """)
            
            if st.button("📥 Load International: VEA 15%", key="load_vea"):
                st.session_state.loaded_model = {
                    'name': 'International - VEA',
                    'tickers': ['VEA'],
                    'weights': {'VEA': 0.15}
                }
                st.success("✅ Loaded!")
        
        # =============================================================================
        # SLEEVE 6: FACTOR SLEEVES (Advanced)
        # =============================================================================
        
        with st.expander("🎯 **SLEEVE 6: Factor Sleeves - Academic Factor Premiums**"):
            st.markdown("### Factor ETFs - Beyond Market Beta")
            st.markdown("*Advanced strategy. 5-15% allocation for sophisticated investors.*")
            
            factor_etfs = pd.DataFrame({
                'ETF': ['VTV', 'MTUM', 'QUAL', 'USMV', 'SIZE', 'VLUE'],
                'Factor': ['Value', 'Momentum', 'Quality', 'Low Volatility', 'Size (Small Cap)', 'Value (Enhanced)'],
                'Name': ['Vanguard Value', 'iShares Momentum', 'iShares Quality', 'iShares Min Vol', 'iShares Size', 'iShares Enhanced Value'],
                'Expense Ratio': ['0.04%', '0.15%', '0.15%', '0.15%', '0.15%', '0.15%'],
                'Methodology': ['Low P/B, P/E', 'Price momentum', 'ROE, debt, earnings', 'Low volatility stocks', 'Small cap focus', 'Multi-factor value'],
                '10Yr Return': ['10.8%', '14.2%', '14.5%', '11.2%', '10.3%', 'N/A (2018)']
            })
            
            st.dataframe(factor_etfs, use_container_width=True, hide_index=True)
            
            st.markdown("""
                **🎯 FACTOR INVESTING - MY HONEST TAKE:**
                
                **The Theory:**
                - Academic research shows factors outperform market over long term
                - Value, Momentum, Quality, Size = documented premiums
                - **But**: Theory ≠ reality in your portfolio
                
                **The Reality:**
                - Factors go through LONG periods of underperformance
                - Value crushed 2010-2020 (growth dominated)
                - Requires discipline to hold during drawdowns
                - **Most investors quit at the worst time**
                
                **🎯 MY RECOMMENDATIONS BY FACTOR:**
                
                **1. Quality (QUAL) - BEST FOR MOST**
                - 14.5% 10-year return (outperformed market!)
                - Low debt, high ROE, stable earnings
                - Downside protection in bear markets
                - Top holdings: AAPL, MSFT, JNJ, V
                - **Use case:** Core satellite (5-10% allocation)
                - **Why:** Works in most environments
                
                **2. Momentum (MTUM) - FOR TRADERS**
                - 14.2% 10-year return (excellent)
                - Buys recent winners, sells losers
                - **BUT**: Whipsaws in volatile markets
                - Rebalances quarterly (tracking momentum shifts)
                - **Use case:** Bull market outperformance (5-10%)
                - **Warning:** Can reverse sharply
                
                **3. Value (VTV) - CONTRARIAN PLAY**
                - 10.8% 10-year return (underperformed SPY)
                - **But**: 2022 value crushed growth (+4% vs -30%)
                - Cheap stocks eventually work
                - **Use case:** IF you believe value comeback (10-15%)
                - **Problem:** Requires 10+ year horizon
                
                **4. Low Volatility (USMV) - DEFENSIVE**
                - 11.2% return with LOWER volatility
                - Best risk-adjusted returns (high Sharpe ratio)
                - Lags in bull markets, protects in bears
                - **Use case:** Risk reduction (10-15%)
                - **Best for:** Retirees, risk-averse
                
                **💡 MY PERSONAL FACTOR PORTFOLIO:**
                - **If forced to choose ONE: QUAL** (quality works everywhere)
                - **Aggressive combo:** 50% QUAL + 50% MTUM (quality + momentum)
                - **Defensive combo:** 60% QUAL + 40% USMV (quality + low vol)
                - **Contrarian:** 100% VTV (value bet)
                
                **🔥 ADVANCED STRATEGY:**
                - **Tactical rotation:** Switch factors based on market regime
                - Bull market: MTUM (momentum)
                - Bear market: USMV (low vol)
                - Recovery: VTV (value)
                - **But**: This is HARD to execute
                
                **⚠️ FACTOR INVESTING WARNINGS:**
                1. **Long dry spells:** Value dead 2010-2020
                2. **Requires conviction:** Hold through underperformance
                3. **Not "free money":** Premiums are compensation for risk/pain
                4. **Keep allocation small:** 5-15% max
                5. **Most investors fail:** Quit at bottom of cycle
                
                **MY VERDICT:**
                - Skip factors unless you're sophisticated investor
                - If you use: Start with QUAL (easiest to hold)
                - Allocation: 5-10% as satellite position
                - **Better:** Just own VOO and sleep well
            """)
            
            if st.button("📥 Load Factor: QUAL 10%", key="load_qual"):
                st.session_state.loaded_model = {
                    'name': 'Factor - Quality',
                    'tickers': ['QUAL'],
                    'weights': {'QUAL': 0.10}
                }
                st.success("✅ Loaded!")
        
        # =============================================================================
        # BUILD YOUR COMPLETE PORTFOLIO
        # =============================================================================
        
        st.markdown("---")
        st.markdown("## 🏗️ Build Your Complete Portfolio")
        st.markdown("*Combine sleeves to create your optimal allocation. Here are my suggestions based on different investor profiles:*")
        
        profile = st.selectbox("**Choose Your Investor Profile:**", [
            "Select a profile...",
            "🚀 Aggressive Growth (20s-30s, high risk tolerance)",
            "📈 Growth Focus (30s-40s, long horizon)",
            "⚖️ Balanced Growth (40s-50s, moderate risk)",
            "🛡️ Conservative Growth (50s-60s, approaching retirement)",
            "💰 Income Focus (60s+, retired)",
            "🎯 Sophisticated / Custom"
        ])
        
        if profile == "🚀 Aggressive Growth (20s-30s, high risk tolerance)":
            st.markdown("""
                ### Aggressive Growth Portfolio
                - **VOO (Core):** 40%
                - **QQQ (Growth):** 30%
                - **VEA (International):** 15%
                - **QUAL (Quality Factor):** 10%
                - **AGG (Bonds):** 5%
                
                **Total:** 100% | **Stocks/Bonds:** 95/5
                
                **Expected:** 12-15% annual return, -35% max drawdown
                
                **Why this works:**
                - Heavy growth tilt for maximum long-term returns
                - Minimal bonds (you have time to recover)
                - International for diversification
                - Quality factor for extra edge
                - Can stomach high volatility
            """)
            if st.button("📥 Load This Portfolio", key="load_aggressive"):
                st.session_state.loaded_model = {
                    'name': 'Aggressive Growth',
                    'tickers': ['VOO', 'QQQ', 'VEA', 'QUAL', 'AGG'],
                    'weights': {'VOO': 0.40, 'QQQ': 0.30, 'VEA': 0.15, 'QUAL': 0.10, 'AGG': 0.05}
                }
                st.success("✅ Loaded! Go to sidebar → Build Portfolio")
                st.balloons()
        
        elif profile == "📈 Growth Focus (30s-40s, long horizon)":
            st.markdown("""
                ### Growth Focus Portfolio
                - **VOO (Core):** 45%
                - **QQQ (Growth):** 20%
                - **SCHD (Dividend):** 15%
                - **VEA (International):** 10%
                - **AGG (Bonds):** 10%
                
                **Total:** 100% | **Stocks/Bonds:** 90/10
                
                **Expected:** 10-13% annual return, -30% max drawdown
                
                **Why this works:**
                - Strong growth core with VOO + QQQ
                - SCHD adds quality dividend growth
                - Small bond buffer for stability
                - Still aggressive but more balanced than 20s portfolio
            """)
            if st.button("📥 Load This Portfolio", key="load_growth"):
                st.session_state.loaded_model = {
                    'name': 'Growth Focus',
                    'tickers': ['VOO', 'QQQ', 'SCHD', 'VEA', 'AGG'],
                    'weights': {'VOO': 0.45, 'QQQ': 0.20, 'SCHD': 0.15, 'VEA': 0.10, 'AGG': 0.10}
                }
                st.success("✅ Loaded! Go to sidebar → Build Portfolio")
                st.balloons()
        
        elif profile == "⚖️ Balanced Growth (40s-50s, moderate risk)":
            st.markdown("""
                ### Balanced Growth Portfolio
                - **VOO (Core):** 40%
                - **SCHD (Dividend):** 20%
                - **VEA (International):** 15%
                - **AGG (Bonds):** 20%
                - **TLT (Long Bonds):** 5%
                
                **Total:** 100% | **Stocks/Bonds:** 75/25
                
                **Expected:** 8-11% annual return, -25% max drawdown
                
                **Why this works:**
                - Classic balanced approach
                - SCHD provides income + growth
                - 25% bonds for stability
                - TLT for deflation protection
                - Reasonable risk/reward
            """)
            if st.button("📥 Load This Portfolio", key="load_balanced"):
                st.session_state.loaded_model = {
                    'name': 'Balanced Growth',
                    'tickers': ['VOO', 'SCHD', 'VEA', 'AGG', 'TLT'],
                    'weights': {'VOO': 0.40, 'SCHD': 0.20, 'VEA': 0.15, 'AGG': 0.20, 'TLT': 0.05}
                }
                st.success("✅ Loaded! Go to sidebar → Build Portfolio")
                st.balloons()
        
        elif profile == "🛡️ Conservative Growth (50s-60s, approaching retirement)":
            st.markdown("""
                ### Conservative Growth Portfolio
                - **VOO (Core):** 30%
                - **SCHD (Dividend):** 25%
                - **VEA (International):** 10%
                - **AGG (Bonds):** 25%
                - **TLT (Long Bonds):** 10%
                
                **Total:** 100% | **Stocks/Bonds:** 65/35
                
                **Expected:** 7-9% annual return, -20% max drawdown
                
                **Why this works:**
                - Lower equity allocation (65%)
                - Heavy SCHD for reliable dividends
                - 35% bonds for stability
                - Preservation + moderate growth
                - Lower volatility for peace of mind
            """)
            if st.button("📥 Load This Portfolio", key="load_conservative"):
                st.session_state.loaded_model = {
                    'name': 'Conservative Growth',
                    'tickers': ['VOO', 'SCHD', 'VEA', 'AGG', 'TLT'],
                    'weights': {'VOO': 0.30, 'SCHD': 0.25, 'VEA': 0.10, 'AGG': 0.25, 'TLT': 0.10}
                }
                st.success("✅ Loaded! Go to sidebar → Build Portfolio")
                st.balloons()
        
        elif profile == "💰 Income Focus (60s+, retired)":
            st.markdown("""
                ### Income Focus Portfolio
                - **SCHD (Dividend):** 35%
                - **VOO (Core):** 20%
                - **AGG (Bonds):** 30%
                - **TLT (Long Bonds):** 10%
                - **VEA (International):** 5%
                
                **Total:** 100% | **Stocks/Bonds:** 60/40
                
                **Expected:** 6-8% annual return + 3.5% income, -18% max drawdown
                
                **Why this works:**
                - SCHD provides 3.5% yield + growth
                - 40% bonds for stability
                - Lower volatility for withdrawals
                - Classic 60/40 with income tilt
                - Sustainable for retirement spending
            """)
            if st.button("📥 Load This Portfolio", key="load_income"):
                st.session_state.loaded_model = {
                    'name': 'Income Focus',
                    'tickers': ['SCHD', 'VOO', 'AGG', 'TLT', 'VEA'],
                    'weights': {'SCHD': 0.35, 'VOO': 0.20, 'AGG': 0.30, 'TLT': 0.10, 'VEA': 0.05}
                }
                st.success("✅ Loaded! Go to sidebar → Build Portfolio")
                st.balloons()
        
        st.markdown("---")
        st.info("""
            **💡 How to Use:**
            1. Read through each sleeve above
            2. Select a pre-built profile OR build custom
            3. Click "Load This Portfolio" 
            4. Go to **sidebar** → modify if needed → "Build Portfolio"
            5. Analyze across all tabs
            
            **Remember:** These are STARTING points. Adjust based on your unique situation!
        """)

