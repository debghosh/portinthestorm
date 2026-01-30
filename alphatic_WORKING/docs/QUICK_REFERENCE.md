# Alphatic Portfolio Analyzer - Quick Reference Card

## 🚀 Quick Start
```bash
streamlit run alphatic_portfolio_app.py
```

## 📊 Building Your First Portfolio

### Step 1: Enter Details (Left Sidebar)
- **Portfolio Name**: Give it a descriptive name
- **Tickers**: One per line or comma-separated
  - Example: `SPY, QQQ, AGG` or:
    ```
    SPY
    QQQ
    AGG
    ```

### Step 2: Choose Allocation
- **Equal Weight**: Simple diversification
- **Custom Weights**: Specify percentages
- **Optimize**: Maximum Sharpe ratio

### Step 3: Set Date Range
- **Auto (Recommended)**: Earliest available data
- **Custom**: Specific time period

### Step 4: Build
Click **🚀 Build Portfolio** button

---

## 🎯 Five Analysis Tabs

### 1️⃣ Overview
- Portfolio composition
- Key metrics (16+)
- Cumulative returns
- Annual returns
- Monthly heatmap

### 2️⃣ Detailed Analysis
- Rolling Sharpe & volatility
- Drawdown analysis
- Return distribution
- Correlation matrix
- Complete metrics table

### 3️⃣ Compare Benchmarks
- Select: SPY, QQQ, 60/40, etc.
- Performance table
- Charts: Cumulative, Annual, Risk-Return

### 4️⃣ Compare Portfolios
- Select 2+ saved portfolios
- Side-by-side metrics
- Overlay charts
- Rolling comparisons

### 5️⃣ Optimization
- One-click optimize
- Current vs Optimized
- Efficient frontier
- Save optimized version

---

## 📈 Key Metrics Explained

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Annual Return** | Average yearly gain | Higher better |
| **Sharpe Ratio** | Return per unit risk | >1.0 good, >2.0 excellent |
| **Max Drawdown** | Worst peak-to-trough | Lower better (closer to 0%) |
| **Volatility** | Risk/variability | Lower better for same return |
| **Sortino Ratio** | Like Sharpe, downside only | Higher better |
| **Alpha** | Excess vs benchmark | Positive better |
| **Beta** | Market sensitivity | 1.0 = market-like |

---

## 🎨 Common Portfolio Examples

### Conservative (Low Risk)
```
60/40: SPY (60%), AGG (40%)
Allocation: Custom Weights
```

### Moderate (Balanced)
```
Diversified: SPY, QQQ, IWM, AGG, GLD
Allocation: Equal Weight or Optimize
```

### Aggressive (High Risk)
```
Tech Heavy: QQQ, VGT, ARKK
Allocation: Optimize
```

### Income Focus
```
Dividend: SCHD, VYM, JEPI, AGG
Allocation: Custom or Optimize
```

---

## 🔍 Quick Analysis Workflow

### Analyzing New Portfolio
1. Build portfolio (Tab: Sidebar)
2. Check Overview (Tab 1) → Key metrics
3. Review Detailed Analysis (Tab 2) → Risk assessment
4. Compare to SPY (Tab 3) → Benchmark performance
5. Optimize if desired (Tab 5) → Improve allocation

### Comparing Strategies
1. Build 2+ portfolios with different allocations
2. Go to Tab 4 (Compare Portfolios)
3. Select all portfolios to compare
4. Review metrics table
5. Examine cumulative returns
6. Check rolling Sharpe for stability

### Optimizing Existing
1. Build portfolio with current allocation (Custom Weights)
2. Go to Tab 5 (Optimization)
3. Click "Optimize Current Portfolio"
4. Compare Current vs Optimized metrics
5. Review efficient frontier
6. Click "Save Optimized Portfolio" if better

---

## 🎯 Benchmark Selection Guide

| Your Portfolio Type | Use Benchmark |
|---------------------|---------------|
| US Large-Cap Stocks | SPY (S&P 500) |
| Tech-Heavy | QQQ (NASDAQ) |
| Small-Cap Focus | IWM (Russell 2000) |
| Balanced/Diversified | 60/40 (SPY/AGG) |
| Conservative/Bonds | AGG (Bond Aggregate) |
| Global Stocks | VTI + VXUS blend |

---

## ⚡ Pro Tips

### Best Practices
✅ Use **Auto start date** for maximum data
✅ Include **5-15 assets** for diversification
✅ Compare multiple **metrics**, not just returns
✅ Review **drawdowns** for worst-case understanding
✅ Check **correlation matrix** for true diversification

### Common Mistakes to Avoid
❌ Don't chase recent winners (recency bias)
❌ Don't over-optimize on historical data
❌ Don't ignore transaction costs in practice
❌ Don't use leveraged ETFs for long-term
❌ Don't rely on single metric (e.g., just returns)

### When to Rebalance
- After major market moves (±20%)
- Quarterly or annually (calendar-based)
- When portfolio drifts >5% from targets
- After optimization shows significant improvement

---

## 🐛 Quick Troubleshooting

### "No data could be downloaded"
→ Check ticker spelling
→ Verify internet connection
→ Try different date range

### "No overlapping dates"
→ Portfolios have different date ranges
→ Use custom date with common period
→ Or rebuild portfolios with same dates

### Optimization fails
→ Need 2+ assets
→ Need 252+ days of data
→ Remove problematic tickers

### Metrics seem wrong
→ Verify correct tickers
→ Check date range
→ Compare to known benchmark (SPY)
→ Look for outliers in data

---

## 📱 Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Rerun app | `Ctrl + R` or `⌘ + R` |
| Clear cache | `C` |
| View source | `Ctrl + Shift + I` |
| Refresh browser | `F5` |
| Stop server | `Ctrl + C` (in terminal) |

---

## 📚 Sample Portfolios to Try

Copy these into the ticker input:

**Classic 60/40**
```
SPY
AGG
```
Weights: 60% SPY, 40% AGG

**Three-Fund Portfolio**
```
VTI
VXUS
BND
```
Weights: 60% VTI, 30% VXUS, 10% BND

**All-Weather**
```
SPY
TLT
IEF
GLD
DBC
```
Let it optimize!

**Factor Tilted**
```
VTV
VUG
MTUM
QUAL
```
Equal weight or optimize

---

## 🎓 Understanding Sharpe Ratio

The most important metric:

| Sharpe | Interpretation |
|--------|----------------|
| < 0 | Losing money |
| 0 - 0.5 | Not great |
| 0.5 - 1.0 | Good |
| 1.0 - 2.0 | Very good |
| 2.0 - 3.0 | Excellent |
| > 3.0 | Exceptional (rare) |

**Formula**: (Return - Risk-Free Rate) / Volatility

**What it means**: Return you get per unit of risk taken

---

## 📊 What to Look For

### In Overview Tab
✓ Sharpe > 1.0
✓ Consistent annual returns
✓ Low max drawdown
✓ Positive Calmar ratio

### In Detailed Analysis
✓ Stable rolling Sharpe
✓ Quick recovery from drawdowns
✓ Normal return distribution
✓ Low correlations (for diversification)

### In Comparisons
✓ Outperform benchmark on Sharpe
✓ Lower volatility for same return
✓ Better worst-year performance
✓ Shallower drawdowns

### In Optimization
✓ Meaningful Sharpe improvement (>0.3)
✓ Reasonable weight distribution
✓ Historical performance gain
✓ Makes logical sense

---

## 💡 Quick Decision Framework

### Is this portfolio good?
1. Sharpe ratio > 1.0? ✓
2. Max drawdown < -30%? ✓
3. Beats benchmark? ✓
4. Consistent rolling Sharpe? ✓

**4/4 Yes → Strong portfolio**

### Should I optimize?
1. Current Sharpe < 1.5? → Yes
2. Large weight in one asset? → Yes
3. Portfolio has 3+ assets? → Yes
4. Significant historical improvement? → Yes

**3/4 Yes → Worth optimizing**

### Should I rebalance?
1. Drifted >5% from targets? → Yes
2. Optimized weights very different? → Yes
3. More than 6 months since last? → Yes
4. Major market regime change? → Yes

**2/4 Yes → Consider rebalancing**

---

## 🔗 Quick Links

**In-App Navigation:**
- Left Sidebar: Build portfolios
- Tab 1: Overview & basic metrics
- Tab 2: Deep dive analysis
- Tab 3: Benchmark comparisons
- Tab 4: Portfolio comparisons  
- Tab 5: Optimization tools

**Documentation:**
- README.md: Installation guide
- USER_GUIDE.md: Comprehensive manual
- PROJECT_SUMMARY.md: Technical details
- sample_portfolios.json: Examples to import

**Support:**
- Test installation: `python3 test_installation.py`
- Check packages: Listed in requirements.txt
- Verify data: Try downloading SPY manually

---

## 🎯 Success Metrics

Track these over time:

**Your Portfolio vs SPY:**
- [ ] Higher Sharpe ratio
- [ ] Lower max drawdown
- [ ] Better risk-adjusted returns
- [ ] More consistent performance

**Your Portfolio vs 60/40:**
- [ ] Competitive returns
- [ ] Similar or lower volatility
- [ ] Better drawdown protection
- [ ] Higher Sharpe ratio

**Portfolio Evolution:**
- [ ] Improving Sharpe over time
- [ ] Reducing drawdowns
- [ ] More stable rolling metrics
- [ ] Better diversification (lower correlations)

---

## ⏱️ Typical Session Flow

**15-minute Quick Analysis:**
1. Build portfolio (2 min)
2. Review Overview tab (3 min)
3. Check vs SPY in Tab 3 (5 min)
4. Quick optimize in Tab 5 (5 min)

**30-minute Deep Dive:**
1. Build portfolio (2 min)
2. Full Overview review (5 min)
3. Detailed Analysis tab (10 min)
4. Benchmark comparisons (8 min)
5. Optimization analysis (5 min)

**1-hour Comprehensive:**
1. Build 3 strategy variants (10 min)
2. Full analysis of each (30 min)
3. Compare all three (10 min)
4. Optimize best performer (10 min)

---

## 📞 Getting Help

### First, Try:
1. Check USER_GUIDE.md for detailed explanation
2. Run test_installation.py to verify setup
3. Review this Quick Reference card
4. Check PROJECT_SUMMARY.md for technical details

### Common Issues:
- **Import errors** → Run: `pip install -r requirements.txt`
- **Data errors** → Verify tickers on Yahoo Finance
- **Calculation errors** → Check date ranges and ticker validity
- **Performance issues** → Reduce portfolio size or date range

---

**Keep this card handy while using the application!**

*Alphatic Portfolio Analyzer - Built for serious investors*
*Quick Reference Card v1.0 - January 2026*
