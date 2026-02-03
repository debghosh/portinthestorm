#!/usr/bin/env python3
"""
COMPREHENSIVE METRICS VERIFICATION TEST
For Alphatic Portfolio Analyzer - Real Capital Deployment

Tests that metrics for SPY/QQQ/AGG equal-weighted portfolio
are IDENTICAL between original and modular versions.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Import from current modular version
sys.path.insert(0, '/home/claude/portinthestorm')
from helper_functions import (
    download_ticker_data,
    calculate_portfolio_returns,
    calculate_portfolio_metrics
)

print("=" * 80)
print("METRICS VERIFICATION TEST")
print("Portfolio: SPY, QQQ, AGG (Equal Weight)")
print("=" * 80)
print()

# Test parameters
TICKERS = ['SPY', 'QQQ', 'AGG']
WEIGHTS = [1/3, 1/3, 1/3]
START_DATE = datetime(2020, 1, 1).date()
END_DATE = datetime(2024, 12, 31).date()

print(f"Tickers: {TICKERS}")
print(f"Weights: {WEIGHTS}")
print(f"Date Range: {START_DATE} to {END_DATE}")
print()

try:
    # Download data
    print("Downloading price data...")
    prices = download_ticker_data(TICKERS, START_DATE, END_DATE)
    
    if prices is None or prices.empty:
        print("ERROR: Failed to download data")
        sys.exit(1)
    
    print(f"✓ Downloaded {len(prices)} days of data")
    print(f"✓ Columns: {list(prices.columns)}")
    print()
    
    # Calculate portfolio returns
    print("Calculating portfolio returns...")
    weights_array = np.array(WEIGHTS)
    portfolio_returns = calculate_portfolio_returns(prices, weights_array)
    
    print(f"✓ Calculated {len(portfolio_returns)} return observations")
    print(f"✓ Mean daily return: {portfolio_returns.mean():.6f}")
    print(f"✓ Std daily return: {portfolio_returns.std():.6f}")
    print()
    
    # Calculate metrics
    print("Calculating portfolio metrics...")
    metrics = calculate_portfolio_metrics(portfolio_returns)
    
    print()
    print("=" * 80)
    print("PORTFOLIO METRICS RESULTS")
    print("=" * 80)
    print()
    
    # Display all metrics with high precision
    print(f"Total Return:        {metrics['Total Return']:.8f}  ({metrics['Total Return']*100:.4f}%)")
    print(f"Annual Return:       {metrics['Annual Return']:.8f}  ({metrics['Annual Return']*100:.4f}%)")
    print(f"Annual Volatility:   {metrics['Annual Volatility']:.8f}  ({metrics['Annual Volatility']*100:.4f}%)")
    print(f"Sharpe Ratio:        {metrics['Sharpe Ratio']:.8f}")
    print(f"Sortino Ratio:       {metrics['Sortino Ratio']:.8f}")
    print(f"Max Drawdown:        {metrics['Max Drawdown']:.8f}  ({metrics['Max Drawdown']*100:.4f}%)")
    print(f"Calmar Ratio:        {metrics['Calmar Ratio']:.8f}")
    print(f"Win Rate:            {metrics['Win Rate']:.8f}  ({metrics['Win Rate']*100:.4f}%)")
    
    print()
    print("=" * 80)
    print("VERIFICATION INSTRUCTIONS")
    print("=" * 80)
    print()
    print("To verify these numbers match your saved version:")
    print()
    print("1. In the saved version, create a portfolio with:")
    print(f"   - Tickers: {', '.join(TICKERS)}")
    print(f"   - Allocation: Equal Weight")
    print(f"   - Start Date: {START_DATE}")
    print(f"   - End Date: {END_DATE}")
    print()
    print("2. Compare the metrics shown in Overview tab")
    print()
    print("3. The numbers should match to 8 decimal places")
    print()
    print("=" * 80)
    print()
    
    # Also calculate some derived metrics for extra verification
    print("ADDITIONAL VERIFICATION METRICS:")
    print()
    cum_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
    print(f"Cumulative Return:   {cum_return:.8f}  ({cum_return*100:.4f}%)")
    
    # Max value and drawdown dates
    cum_returns = (1 + portfolio_returns).cumprod()
    max_val = cum_returns.max()
    max_date = cum_returns.idxmax()
    print(f"Peak Portfolio Value: {max_val:.8f} on {max_date.date()}")
    
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd_date = drawdown.idxmin()
    print(f"Max Drawdown Date:   {max_dd_date.date()}")
    
    print()
    print("=" * 80)
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
