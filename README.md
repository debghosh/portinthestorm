# Alphatic Portfolio Analyzer - Streamlit Web Application

A comprehensive, production-ready portfolio analysis platform built with Streamlit and PyFolio-Reloaded. Designed for sophisticated investors making real capital allocation decisions.

## 🚀 Features

### Portfolio Management
- **Build Portfolios**: Create portfolios using ticker symbols (stocks and ETFs)
- **Multiple Allocation Methods**: 
  - Equal Weight
  - Custom Weights
  - Optimized (Maximum Sharpe Ratio)
- **Automatic Start Date**: Determines earliest available data across all tickers
- **Save/Load Portfolios**: Manage multiple portfolios with persistence
- **Import/Export**: Save and restore portfolio configurations

### Comprehensive Analysis
- **Overview Tab**: Key metrics, cumulative returns, annual returns, monthly heatmap
- **Detailed Analysis Tab**: Rolling metrics, drawdown analysis, return distribution, correlation matrix
- **Benchmark Comparison**: Compare against SPY, QQQ, 60/40, and other benchmarks
- **Portfolio Comparison**: Side-by-side comparison of multiple portfolios
- **Optimization**: Modern Portfolio Theory optimization with efficient frontier visualization

### Visualizations
- Cumulative returns charts
- Rolling Sharpe ratio and volatility
- Drawdown underwater plots and periods
- Monthly returns heatmap
- Annual returns bar charts
- Return distribution histograms
- Efficient frontier scatter plots
- Correlation matrices
- Risk-return scatter plots

### Performance Metrics
- Annual Return & Volatility
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Omega Ratio
- Maximum Drawdown
- Tail Ratio
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Alpha & Beta (when compared to benchmark)
- Stability of Returns

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Running the Application

```bash
streamlit run alphatic_portfolio_app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Building Your First Portfolio

1. **Enter Portfolio Details** (Left Sidebar):
   - Portfolio Name: Give your portfolio a descriptive name
   - Tickers: Enter ticker symbols (one per line or comma-separated)
     - Example: SPY, QQQ, AGG
   - Allocation Method: Choose Equal Weight, Custom Weights, or Optimize

2. **Set Date Range**:
   - Auto (Earliest Available): Automatically finds earliest common date
   - Custom Date: Specify your own start date

3. **Build Portfolio**:
   - Click "🚀 Build Portfolio" button
   - Wait for data download and processing
   - Portfolio will appear in saved portfolios

### Analyzing Your Portfolio

**Overview Tab** 📈
- View portfolio composition (table and pie chart)
- See key performance metrics
- Examine cumulative returns
- Review annual returns and monthly heatmap

**Detailed Analysis Tab** 📊
- Analyze rolling Sharpe ratio and volatility
- Study drawdown patterns
- View return distribution
- Examine asset correlations

**Compare Benchmarks Tab** ⚖️
- Select benchmarks (SPY, QQQ, etc.)
- Include 60/40 portfolio option
- View performance comparison table
- Compare cumulative returns
- Analyze risk-return profiles

**Compare Portfolios Tab** 🔄
- Select multiple saved portfolios
- Compare performance metrics
- Overlay cumulative returns
- Compare rolling Sharpe ratios
- Analyze relative drawdowns

**Optimization Tab** 🎯
- Optimize current portfolio for max Sharpe
- View efficient frontier
- Compare current vs optimized allocation
- See weight recommendations
- Save optimized portfolio

## 📊 Example Portfolios

### Conservative 60/40
```
Tickers: SPY, AGG
Weights: 60% SPY, 40% AGG
```

### Aggressive Growth
```
Tickers: QQQ, VUG, IWM
Allocation: Equal Weight or Optimized
```

### Diversified Multi-Asset
```
Tickers: SPY, QQQ, IWM, EFA, EEM, AGG, GLD
Allocation: Optimized for Max Sharpe
```

### Factor-Based
```
Tickers: VTV (Value), VUG (Growth), MTUM (Momentum), QUAL (Quality)
Allocation: Equal Weight or Custom
```

## 🛠️ Technical Details

### Data Sources
- **Price Data**: Yahoo Finance (yfinance)
- **Historical Range**: Automatically determined or custom specified
- **Update Frequency**: Real-time on portfolio build

### Analysis Framework
- **PyFolio-Reloaded**: Core performance analytics
- **SciPy**: Portfolio optimization
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

### Optimization Algorithm
- **Method**: Sequential Least Squares Programming (SLSQP)
- **Objective**: Maximum Sharpe Ratio
- **Constraints**: Weights sum to 1, no short selling (0 ≤ w ≤ 1)
- **Risk-Free Rate**: 2% (default, can be modified in code)

## 📝 Portfolio Data Structure

Saved portfolios include:
- Ticker symbols
- Weight allocations
- Return series
- Individual asset returns
- Price data
- Date range
- Allocation method

## 🔧 Customization

### Modifying Benchmarks
Edit the `available_benchmarks` dictionary in Tab 3:
```python
available_benchmarks = {
    'SPY': 'S&P 500',
    'YOUR_TICKER': 'Your Description',
    # Add more benchmarks
}
```

### Adjusting Risk-Free Rate
Modify the `risk_free_rate` parameter in optimization functions:
```python
def optimize_portfolio(returns, risk_free_rate=0.02):  # Change 0.02 to your rate
```

### Custom Metrics
Add custom metrics in the `calculate_portfolio_metrics` function:
```python
def calculate_portfolio_metrics(returns, benchmark_returns=None):
    metrics = {}
    # Add your custom metrics here
    metrics['Your Metric'] = your_calculation(returns)
    return metrics
```

## 📈 Performance Considerations

- **Data Loading**: Initial download may take time for long historical periods
- **Portfolio Size**: Performance optimal with 3-20 assets
- **Optimization**: Efficient frontier generation uses 5000 random portfolios
- **Memory**: Large portfolios (50+ assets) may require more RAM

## ⚠️ Important Notes

### Data Accuracy
- Prices are adjusted for splits and dividends
- Missing data is handled automatically
- Failed ticker downloads are reported

### Date Alignment
- When comparing portfolios, only overlapping dates are used
- Automatic start date finds common earliest date
- Custom dates may exclude some assets if data unavailable

### Optimization Caveats
- Based on historical data (past performance ≠ future results)
- Assumes stable correlations (may change over time)
- No transaction costs or taxes included
- No constraints on sector/asset class exposure

## 🐛 Troubleshooting

### "No data could be downloaded"
- Check ticker symbols for typos
- Verify tickers exist on Yahoo Finance
- Try different date range
- Check internet connection

### "No overlapping dates"
- Selected portfolios may have different date ranges
- Use custom date range with common dates
- Rebuild portfolios with same start date

### Optimization Fails
- Need at least 2 assets in portfolio
- Ensure sufficient historical data (252+ days recommended)
- Check for invalid/missing returns

### Visualization Issues
- Close old figure windows if memory issues occur
- Reduce window size for rolling metrics
- Limit number of portfolios in comparisons

## 🔮 Future Enhancements

Planned features:
- Factor attribution analysis (Fama-French)
- Monte Carlo simulation for forward projections
- Regime detection and tactical allocation
- Transaction cost modeling
- Tax-aware optimization
- Real-time data updates
- Custom benchmarks
- PDF report generation

## 📚 References

- [PyFolio Documentation](https://github.com/quantopian/pyfolio)
- [Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
- [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp)
- [Efficient Frontier](https://www.investopedia.com/terms/e/efficientfrontier.asp)

## 📄 License

This project is for educational and personal use. Always consult with a financial advisor before making investment decisions.

## 🤝 Contributing

This is a production tool for the Alphatic platform. Feedback and suggestions are welcome!

---

**Built with** ❤️ **for serious investors**

*Comprehensive. Accurate. Production-Ready.*
