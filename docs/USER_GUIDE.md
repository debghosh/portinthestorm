# Alphatic Portfolio Analyzer - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Building Portfolios](#building-portfolios)
3. [Understanding the Analysis](#understanding-the-analysis)
4. [Comparing Performance](#comparing-performance)
5. [Portfolio Optimization](#portfolio-optimization)
6. [Tips and Best Practices](#tips-and-best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### First Launch
When you first open the application, you'll see:
- **Left Sidebar**: Portfolio builder and management tools
- **Main Area**: Tab-based interface (will show info message until first portfolio is built)
- **Navigation Tabs**: Overview, Detailed Analysis, Compare Benchmarks, Compare Portfolios, Optimization

### Your First Portfolio
1. In the left sidebar, enter a name for your portfolio (e.g., "My 60/40")
2. Enter ticker symbols in the text area (one per line or comma-separated)
3. Choose an allocation method
4. Select date range method
5. Click "🚀 Build Portfolio"

---

## Building Portfolios

### Ticker Symbols
Enter any valid stock or ETF ticker symbols. Examples:
- **Stocks**: AAPL, MSFT, GOOGL, AMZN
- **ETFs**: SPY, QQQ, IWM, AGG, GLD
- **Sector ETFs**: XLF, XLE, XLK, XLV
- **International**: EFA, EEM, VEU, VXUS

**Format Options:**
```
# One per line:
SPY
QQQ
AGG

# Or comma-separated:
SPY, QQQ, AGG
```

### Allocation Methods

#### Equal Weight
- Distributes capital equally across all assets
- Example: 3 assets → 33.33% each
- **When to use**: Simple diversification, testing ideas

#### Custom Weights
- You specify the percentage for each asset
- Must sum to 100%
- **When to use**: Specific allocation strategies, replicating existing portfolios

#### Optimize (Max Sharpe)
- Automatically finds optimal weights to maximize risk-adjusted returns
- Uses historical data and Modern Portfolio Theory
- **When to use**: Finding efficient allocations, comparing to current strategy

### Date Range Selection

#### Auto (Earliest Available)
- **Recommended for new portfolios**
- Automatically finds the earliest date where all tickers have data
- Maximizes historical analysis period
- Shows you the determined start date

#### Custom Date
- Specify your own start date
- Useful when:
  - Comparing portfolios with same time period
  - Analyzing specific market periods
  - Some tickers have limited history

**Note**: End date defaults to today but can be adjusted for historical analysis.

---

## Understanding the Analysis

### Tab 1: Overview 📈

#### Portfolio Composition
- **Weights Table**: Shows exact allocation percentages
- **Pie Chart**: Visual representation of portfolio allocation

#### Key Performance Metrics

**Returns:**
- **Annual Return**: Average yearly return (higher is better)
- **Cumulative Return**: Total return over entire period

**Risk Metrics:**
- **Annual Volatility**: Standard deviation of returns (lower is better for same return)
- **Max Drawdown**: Largest peak-to-trough decline (lower is better)

**Risk-Adjusted Returns:**
- **Sharpe Ratio**: Return per unit of risk (>1.0 is good, >2.0 is excellent)
- **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
- **Calmar Ratio**: Return divided by max drawdown
- **Omega Ratio**: Probability-weighted ratio of gains vs losses

#### Cumulative Returns Chart
- Shows portfolio value growth over time
- $1 invested at start → current value
- Useful for understanding long-term growth trajectory

#### Annual Returns Bar Chart
- Year-by-year performance
- Green bars = positive years
- Red bars = negative years
- Helps identify consistency and outlier years

#### Monthly Returns Heatmap
- Color-coded matrix: Year (rows) × Month (columns)
- Green = positive returns, Red = negative returns
- Identifies seasonal patterns and volatility clusters

---

### Tab 2: Detailed Analysis 📊

#### Rolling Metrics
- **Adjustable Window**: Use slider to change period (20-252 days)
- **Rolling Sharpe**: How risk-adjusted returns evolved over time
- **Rolling Volatility**: How portfolio risk changed over time
- **Why it matters**: Identifies unstable periods, regime changes

#### Drawdown Analysis
- **Underwater Plot**: Shows cumulative loss from peak
  - Time spent "underwater" (below previous high)
  - Depth of drawdowns
  - Recovery periods
  
- **Top 5 Drawdown Periods**: Lists worst drawdowns
  - Start and end dates
  - Duration
  - Magnitude (% loss)
  - Recovery time

#### Return Distribution
- **Histogram**: Frequency of different return levels
- **Mean Line**: Average daily return
- **Standard Deviation Lines**: ±1σ boundaries
- **Why it matters**: 
  - Assess normality of returns
  - Identify fat tails (extreme events)
  - Understand return characteristics

#### Asset Correlation Matrix
(Only shown when portfolio has 2+ assets)
- **Scale**: -1 to +1
  - +1 = Perfect positive correlation (move together)
  - 0 = No correlation (independent)
  - -1 = Perfect negative correlation (move opposite)
- **Why it matters**: Diversification effectiveness

#### Complete Metrics Table
- All calculated metrics in one place
- Can be copied or exported
- Includes additional metrics:
  - Stability of returns
  - Tail ratio
  - Daily VaR and CVaR

---

### Tab 3: Compare Benchmarks ⚖️

#### Available Benchmarks
- **SPY**: S&P 500 (large-cap US stocks)
- **QQQ**: NASDAQ-100 (tech-heavy)
- **IWM**: Russell 2000 (small-cap US)
- **AGG**: Bond Aggregate (US bonds)
- **GLD**: Gold
- **VTI**: Total US Stock Market
- **VXUS**: Total International Stocks

#### 60/40 Portfolio
- Classic balanced portfolio
- 60% stocks (SPY), 40% bonds (AGG)
- Standard benchmark for diversified investors

#### Performance Comparison Table
- Side-by-side metrics for all selections
- **Look for:**
  - Higher Sharpe = better risk-adjusted returns
  - Lower Max Drawdown = less downside risk
  - Consistency across different metrics

#### Cumulative Returns Comparison
- All portfolios/benchmarks on one chart
- **Interpret:**
  - Steeper slope = faster growth
  - Smoother line = less volatility
  - Higher ending point = better total return

#### Annual Returns Comparison
- Grouped bar chart by year
- **Identify:**
  - Years of outperformance/underperformance
  - Relative consistency
  - Bear/bull market behavior

#### Risk-Return Scatter Plot
- X-axis: Volatility (risk)
- Y-axis: Return
- **Your portfolio** = red star
- **Benchmarks** = blue dots
- **Ideal position**: Upper-left (high return, low risk)

---

### Tab 4: Compare Portfolios 🔄

#### Multi-Portfolio Analysis
- Requires 2+ saved portfolios
- Uses overlapping date range only
- Aligns all data for fair comparison

#### Performance Metrics
- Same format as benchmark comparison
- **Use to:**
  - Evaluate different strategies
  - Track portfolio evolution
  - Compare allocation methods

#### Cumulative Returns
- Multiple portfolio lines on one chart
- **Look for:**
  - Consistent outperformance
  - Lower volatility
  - Better recovery from drawdowns

#### Rolling Sharpe Comparison
- 63-day rolling window (quarterly)
- Shows relative performance stability
- **Insights:**
  - Which portfolio is more consistent?
  - Does one excel in certain conditions?
  - Are there regime changes?

#### Drawdown Comparison
- Overlaid drawdown charts
- **Key questions:**
  - Which portfolio loses less in downturns?
  - Which recovers faster?
  - Which has shallower drawdowns?

---

### Tab 5: Optimization 🎯

#### What Is Portfolio Optimization?
- Uses Modern Portfolio Theory
- Finds asset weights that maximize Sharpe ratio
- Based on historical returns and correlations
- **Goal**: Best risk-adjusted returns possible

#### Current vs Optimized
**Metrics Compared:**
- Expected return (annualized)
- Volatility (annualized)
- Sharpe ratio

**Delta Values:**
- Green = improvement
- Red = decline
- Shows what you could gain from rebalancing

#### Optimal Weights
- **Current Allocation**: Your existing weights
- **Optimal Allocation**: Mathematically derived weights
- **Compare to identify:**
  - Overweight positions
  - Underweight positions
  - Potential rebalancing opportunities

#### Efficient Frontier
- Scatter plot of thousands of random portfolios
- **Color gradient**: Sharpe ratio (darker = better)
- **Red star**: Your optimized portfolio
- **Orange diamond**: Your current portfolio
- **Interpretation:**
  - Points on the upper edge = efficient portfolios
  - Points below = suboptimal (could get better return for same risk)

#### Performance Comparison
- Historical backtest of current vs optimized
- **Remember**: Past performance ≠ future results
- **Use to:**
  - Understand potential improvement
  - See cost of current allocation
  - Inform rebalancing decisions

#### Saving Optimized Portfolio
- Creates new portfolio with optimal weights
- Saves as "{Original Name} (Optimized)"
- Can then analyze and compare like any portfolio

---

## Comparing Performance

### Choosing the Right Benchmark

**For US Large-Cap Stock Portfolios:**
- SPY (S&P 500)

**For Tech-Heavy Portfolios:**
- QQQ (NASDAQ-100)

**For Balanced Portfolios:**
- 60/40 (60% SPY, 40% AGG)

**For Global Diversified:**
- Combine SPY + VXUS in appropriate ratio

**For Income/Conservative:**
- AGG or combination with dividend ETFs

### Key Comparison Metrics

**Must Be Higher Than Benchmark:**
- Sharpe Ratio (risk-adjusted return)
- Annual Return (if taking similar risk)
- Calmar Ratio (return per unit of max drawdown)

**Should Be Similar or Lower:**
- Annual Volatility (for same return level)
- Max Drawdown (capital preservation)

**Context Matters:**
- **Higher volatility with higher return**: Acceptable if Sharpe improves
- **Lower max drawdown with similar return**: Strong capital preservation
- **Better downside capture**: Sortino ratio advantage

---

## Portfolio Optimization

### When to Optimize

**Good Candidates:**
- Portfolios with 3+ assets
- 2+ years of historical data
- Assets with varying correlations
- Rebalancing existing portfolio

**Not Recommended:**
- 2-asset portfolios (limited degrees of freedom)
- Very short history (<1 year)
- Highly correlated assets
- Already optimized portfolios

### Understanding the Results

#### Mathematical Optimization
- **Not magic**: Based on historical patterns
- **Assumptions**:
  - Historical correlations persist
  - Returns are somewhat stable
  - No transaction costs
  - No constraints on position sizes

#### Practical Considerations

**Extreme Weights:**
If optimization suggests 80% in one asset:
- Model sees it as clearly superior historically
- May not reflect future expectations
- Consider constraints or manual adjustment

**Zero Weights:**
If optimization suggests 0% in an asset:
- Historically hurt risk-adjusted returns
- Consider removing from portfolio
- Or apply minimum weight constraints

**Frequent Rebalancing:**
- Optimized weights are point-in-time
- Market conditions change
- Rebalance periodically (quarterly/annually)

### Implementation Tips

1. **Review optimal weights**: Do they make sense?
2. **Check historical performance**: Is the improvement meaningful?
3. **Consider transaction costs**: Is rebalancing worth it?
4. **Apply constraints if needed**: Min/max position sizes
5. **Monitor and reoptimize**: Quarterly or semi-annually

---

## Tips and Best Practices

### Portfolio Construction

**Diversification:**
- Use 5-15 assets for meaningful diversification
- Include different asset classes (stocks, bonds, alternatives)
- Mix geographic regions
- Include various market cap sizes

**Time Horizon:**
- Longer history (10+ years) = more reliable optimization
- Include full market cycle (bull + bear markets)
- Minimum 2-3 years for meaningful analysis

**Asset Selection:**
- Use liquid, low-cost ETFs
- Verify ticker symbols before building
- Avoid leveraged/inverse ETFs for long-term portfolios
- Consider tax efficiency

### Analysis Best Practices

**Compare Apples to Apples:**
- Use same date range for comparisons
- Account for different risk levels
- Consider transaction costs and taxes
- Normalize for market conditions

**Multiple Metrics:**
- Don't rely on single metric
- Balance return and risk metrics
- Consider worst-case scenarios (max drawdown)
- Look at rolling metrics for stability

**Context Matters:**
- Bull market: Most portfolios look good
- Bear market: Risk management shows up
- Full cycle: True quality revealed

### Common Pitfalls to Avoid

**Survivorship Bias:**
- Don't backtest only successful funds
- Include delisted tickers if replicating historical strategies

**Overfitting:**
- Don't over-optimize on historical data
- Keep strategies simple and logical
- Leave room for estimation error

**Ignoring Costs:**
- Transaction costs compound over time
- Tax considerations for taxable accounts
- Expense ratios affect long-term returns

**Recency Bias:**
- Recent performance ≠ future performance
- Don't chase last year's winners
- Consider full market cycles

---

## Troubleshooting

### Data Download Issues

**Error: "No data could be downloaded"**
- Check ticker symbols (case-insensitive)
- Verify ticker exists on Yahoo Finance
- Try different date range
- Check internet connection

**Some Tickers Failed:**
- Ticker may be delisted
- Too far back in history
- Different exchange or symbol
- Use valid alternatives

**Partial Data:**
- Some tickers have limited history
- Use custom date range
- Or remove problematic tickers

### Portfolio Analysis Issues

**Error: "No overlapping dates"**
- Portfolios have different date ranges
- Solution: Rebuild with common dates
- Or use custom date range for both

**Optimization Fails:**
- Need 2+ assets
- Requires sufficient history (252+ days recommended)
- Check for NaN values in returns
- Try removing problematic tickers

**Metrics Look Wrong:**
- Verify date range is correct
- Check for extreme outliers in data
- Ensure tickers are correct
- Compare with benchmark for sanity check

### Performance Issues

**Slow Loading:**
- Long date ranges take longer
- Many tickers increase processing time
- First download caches data
- Subsequent loads faster

**Memory Issues:**
- Close other applications
- Reduce portfolio size
- Shorter date range
- Restart application

**Visualizations Not Showing:**
- Allow time to render
- Check for errors in terminal
- Refresh browser
- Clear browser cache

### Common Questions

**Q: Why different results than other tools?**
A: Different calculation methods, data sources, or date ranges. Our calculations use industry-standard methods (PyFolio).

**Q: Can I analyze individual stocks?**
A: Yes! Works with any ticker (stocks or ETFs).

**Q: How often should I rebalance?**
A: Typically quarterly to annually. More frequent = higher costs.

**Q: Why do optimized weights seem extreme?**
A: Based on historical data patterns. May want to add constraints for practical implementation.

**Q: What's a good Sharpe ratio?**
A: >1.0 = good, >2.0 = very good, >3.0 = exceptional

**Q: Should I use auto or custom start date?**
A: Auto for maximum history, custom for specific analysis periods.

---

## Advanced Features (Coming Soon)

### Planned Enhancements
- Factor attribution (Fama-French)
- Monte Carlo forward simulations
- Regime detection and tactical allocation
- Transaction cost modeling
- Tax-aware optimization
- Custom benchmark creation
- Automated rebalancing suggestions
- PDF report generation
- Real-time data updates
- Multi-currency support

---

## Getting Help

### Resources
- **README.md**: Installation and setup
- **Sample Portfolios**: `sample_portfolios.json` for quick start
- **Test Script**: `test_installation.py` to verify setup

### Best Practices for Questions
1. Include error message (if any)
2. Describe what you were trying to do
3. Share ticker symbols and date range
4. Mention which tab/feature
5. Note your Python/OS version

---

## Glossary

**Alpha**: Excess return vs. benchmark (adjusted for beta)

**Beta**: Sensitivity to market movements (1.0 = same as market)

**Calmar Ratio**: Annual return / maximum drawdown

**Correlation**: Statistical relationship between assets (-1 to +1)

**CVaR (Conditional VaR)**: Average loss in worst 5% of cases

**Drawdown**: Decline from peak to trough

**Efficient Frontier**: Set of optimal portfolios for each risk level

**Max Drawdown**: Largest peak-to-trough decline

**Modern Portfolio Theory**: Framework for portfolio optimization (Markowitz)

**Omega Ratio**: Probability-weighted gains vs. losses

**Sharpe Ratio**: (Return - Risk-Free Rate) / Volatility

**Sortino Ratio**: Like Sharpe but only downside risk

**Stability**: Consistency of returns over time (R-squared of linear regression)

**Tail Ratio**: 95th percentile gain / 5th percentile loss

**VaR (Value at Risk)**: Loss threshold for worst 5% of days

**Volatility**: Standard deviation of returns (measure of risk)

---

*Last Updated: January 2026*
*Alphatic Portfolio Analyzer - Built for serious investors*
