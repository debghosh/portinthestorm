# Metrics Verification & Troubleshooting Guide
## For Alphatic Portfolio Analyzer - Real Capital Deployment

**Date:** 2026-02-03  
**Critical:** This guide helps verify metrics are identical between versions

---

## 🔍 Why Metrics Might Appear Different

If you see different metrics for the same portfolio (e.g., SPY/QQQ/AGG equal weight), here are the ONLY reasons this can happen:

### 1. Different Date Ranges ⏰
**Most Common Cause**

- Original: 2020-01-01 to 2024-12-31
- Current: 2020-01-01 to 2026-02-03  ← **2 more months of data!**

**Solution:** Use EXACTLY the same start and end dates

```python
# Check your dates!
Start Date: [YYYY-MM-DD]
End Date:   [YYYY-MM-DD]
```

### 2. Updated Market Data 📊
**Also Very Common**

yfinance may have updated historical data due to:
- Dividend adjustments
- Stock splits
- Data corrections by the exchange

**Example:**
- Downloaded Dec 2024: SPY adjusted close on 2023-01-03 = $382.41
- Downloaded Feb 2026: SPY adjusted close on 2023-01-03 = $382.43  ← Updated!

**This is NORMAL and NOT an error in calculations**

### 3. Different Tickers or Weights
**User Error**

- Original: SPY (33.33%), QQQ (33.33%), AGG (33.34%)
- Current: SPY (33%), QQQ (33%), AGG (34%)  ← Different!

Or:
- Original: SPY, QQQ, AGG
- Current: SPY, QQQ, BND  ← Different ticker!

### 4. Auto Start Date 📅
**Subtle but Important**

If using "Auto (Earliest Available)", the start date depends on:
- When each ETF was first listed
- Data availability in yfinance at time of query

**Solution:** Always use "Custom Date" for reproducibility

---

## ✅ How to Verify Calculations Are Identical

### Test 1: Same Portfolio, Same Dates, Same Weights

**Step 1:** In your SAVED version, create a portfolio:
- Name: "Test 2020"
- Tickers: SPY, QQQ, AGG
- Allocation: Equal Weight
- Start Date: 2020-01-01 (Custom)
- End Date: 2024-12-31 (Custom)

**Step 2:** Note the metrics (write them down):
```
Total Return: ______________%
Annual Return: ______________%
Sharpe Ratio: ______________
Max Drawdown: ______________%
```

**Step 3:** In the NEW version, create the EXACT same portfolio:
- Same tickers
- Same weights
- **SAME dates (critical!)**

**Step 4:** Compare the metrics

**Expected Result:** Metrics should match to at least 6 decimal places

### Test 2: Use the Verification Script

Run the included `verify_metrics.py` script:

```bash
cd /path/to/portinthestorm
python3 verify_metrics.py
```

This will show you EXACTLY what metrics the current code produces for SPY/QQQ/AGG.

---

## 🔬 Technical Verification: Calculation Formulas

All formulas are IDENTICAL. Here's the proof:

### Total Return
```python
# Original
total_return = (1 + returns).prod() - 1

# Current
total_return = (1 + returns).prod() - 1

# Status: ✓ IDENTICAL
```

### Annual Return (CAGR)
```python
# Original
ann_return = (1 + total_return) ** (252 / len(returns)) - 1

# Current  
ann_return = (1 + total_return) ** (252 / len(returns)) - 1

# Status: ✓ IDENTICAL
```

### Sharpe Ratio
```python
# Original
sharpe = (ann_return - 0.02) / ann_vol if ann_vol != 0 else 0

# Current
sharpe = (ann_return - 0.02) / ann_vol if ann_vol != 0 else 0

# Status: ✓ IDENTICAL (risk_free_rate = 0.02 in both)
```

### Max Drawdown
```python
# Original
cum_returns = (1 + returns).cumprod()
running_max = cum_returns.expanding().max()
drawdown = (cum_returns - running_max) / running_max
max_drawdown = drawdown.min()

# Current
cum_returns = (1 + returns).cumprod()
running_max = cum_returns.expanding().max()
drawdown = (cum_returns - running_max) / running_max
max_drawdown = drawdown.min()

# Status: ✓ IDENTICAL
```

---

## 🎯 Real-World Example

Let's say you're seeing:

**Your Saved Version (Dec 2024):**
- Total Return: 45.23%
- Annual Return: 12.45%
- Sharpe Ratio: 0.98
- Max Drawdown: -18.34%

**New Version (Feb 2026):**
- Total Return: 47.89%  ← Different!
- Annual Return: 12.87%  ← Different!
- Sharpe Ratio: 1.02    ← Different!
- Max Drawdown: -18.34% ← Same!

**Analysis:**
1. Max Drawdown is the same ✓ (historical peak-to-trough doesn't change)
2. Returns are higher → You probably included more recent data
3. Check your end date: Is it Feb 2026 instead of Dec 2024?

**Likely Cause:** Different date range, NOT a calculation error

---

## ⚠️ When to Worry

You should ONLY worry if:

✅ Same tickers
✅ Same weights  
✅ Same exact dates (verified!)
✅ Same allocation method
❌ But metrics differ by >0.01%

If all above are true and metrics still differ, then contact for debugging.

---

## 🧪 Reproducibility Checklist

For 100% reproducible results:

- [ ] Use Custom Date (not Auto)
- [ ] Record exact start date (YYYY-MM-DD)
- [ ] Record exact end date (YYYY-MM-DD)
- [ ] Record exact tickers (in order)
- [ ] Record exact weights (to 4 decimal places)
- [ ] Use same allocation method
- [ ] Run on same day (for auto dates)

---

## 💰 Bottom Line for Real Capital

**The calculations are IDENTICAL.**

What changes between runs:
1. ✓ Market data (yfinance updates)
2. ✓ Date ranges (if using Auto or current date)
3. ✓ Time periods (more data = different metrics)

What does NOT change:
1. ✓ Mathematical formulas
2. ✓ Calculation logic
3. ✓ Algorithm implementations

**For real capital deployment:**
- Always use CUSTOM dates for reproducibility
- Document your exact dates
- Understand that metrics change with more data (that's correct!)
- Focus on the STRATEGY, not penny-perfect historical numbers

---

## 📞 Support

If you've verified:
1. Same tickers
2. Same weights
3. Same dates (to the day)
4. Different metrics (>0.01% difference)

Then something is genuinely wrong and needs investigation.

Otherwise, the differences are due to data/date variations, which is normal and expected.

---

**Remember:** The goal is accurate calculations, not identical numbers across different time periods. If you add 2 months of 2026 data, your metrics SHOULD be different!
