"""
Test script to validate Alphatic Portfolio Analyzer installation
"""

import sys
import importlib

def test_imports():
    """Test that all required packages can be imported"""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    print()
    
    required_packages = {
        'streamlit': 'Streamlit',
        'yfinance': 'yfinance',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'scipy': 'SciPy',
        'pyfolio': 'PyFolio-Reloaded'
    }
    
    failed = []
    
    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {name:20} - OK")
        except ImportError as e:
            print(f"❌ {name:20} - FAILED")
            failed.append((package, name, str(e)))
    
    print()
    
    if failed:
        print("=" * 60)
        print("FAILED IMPORTS:")
        print("=" * 60)
        for package, name, error in failed:
            print(f"\n{name} ({package}):")
            print(f"  Error: {error}")
        print()
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("=" * 60)
        print("✅ All packages imported successfully!")
        print("=" * 60)
        return True


def test_versions():
    """Display versions of key packages"""
    print()
    print("=" * 60)
    print("Package Versions")
    print("=" * 60)
    print()
    
    import streamlit
    import pandas
    import numpy
    import yfinance
    import scipy
    import matplotlib
    import seaborn
    import pyfolio
    
    versions = {
        'Python': sys.version.split()[0],
        'Streamlit': streamlit.__version__,
        'Pandas': pandas.__version__,
        'NumPy': numpy.__version__,
        'yfinance': yfinance.__version__,
        'SciPy': scipy.__version__,
        'Matplotlib': matplotlib.__version__,
        'Seaborn': seaborn.__version__,
        'PyFolio': pyfolio.__version__
    }
    
    for name, version in versions.items():
        print(f"  {name:15} {version}")
    
    print()


def test_data_download():
    """Test basic data download functionality"""
    print("=" * 60)
    print("Testing Data Download")
    print("=" * 60)
    print()
    
    try:
        import yfinance as yf
        import pandas as pd
        from datetime import datetime, timedelta
        
        print("Downloading test data (SPY, last 30 days)...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = yf.download('SPY', 
                          start=start_date.strftime('%Y-%m-%d'),
                          end=end_date.strftime('%Y-%m-%d'),
                          progress=False,
                          auto_adjust=True)
        
        if data.empty:
            print("❌ No data downloaded")
            return False
        
        print(f"✅ Downloaded {len(data)} days of data")
        print(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Last close: ${data['Close'].iloc[-1]:.2f}")
        
        # Test returns calculation
        returns = data['Close'].pct_change().dropna()
        print(f"   Returns calculated: {len(returns)} values")
        print(f"   Mean daily return: {returns.mean():.4%}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Data download failed: {str(e)}")
        return False


def test_pyfolio():
    """Test PyFolio functionality"""
    print("=" * 60)
    print("Testing PyFolio Metrics")
    print("=" * 60)
    print()
    
    try:
        import pyfolio as pf
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate sample returns
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        
        print("Calculating test metrics on sample data...")
        
        metrics = {
            'Annual Return': pf.timeseries.annual_return(returns),
            'Annual Volatility': pf.timeseries.annual_volatility(returns),
            'Sharpe Ratio': pf.timeseries.sharpe_ratio(returns),
            'Max Drawdown': pf.timeseries.max_drawdown(returns),
            'Calmar Ratio': pf.timeseries.calmar_ratio(returns)
        }
        
        print("✅ PyFolio metrics calculated successfully:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"   {metric:20} {value:.4f}")
            else:
                print(f"   {metric:20} {value}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ PyFolio test failed: {str(e)}")
        return False


def test_optimization():
    """Test portfolio optimization"""
    print("=" * 60)
    print("Testing Portfolio Optimization")
    print("=" * 60)
    print()
    
    try:
        from scipy.optimize import minimize
        import numpy as np
        import pandas as pd
        
        print("Running sample optimization...")
        
        # Sample data
        np.random.seed(42)
        n_assets = 3
        n_days = 252
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_days, n_assets)),
            columns=['Asset1', 'Asset2', 'Asset3']
        )
        
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Define objective function
        def neg_sharpe(weights, mean_ret, cov_mat):
            ret = np.sum(mean_ret * weights) * 252
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat * 252, weights)))
            return -(ret - 0.02) / vol
        
        # Optimize
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial = n_assets * [1. / n_assets]
        
        result = minimize(
            neg_sharpe,
            initial,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            print("✅ Optimization successful")
            print(f"   Optimal weights: {[f'{w:.2%}' for w in result.x]}")
            print(f"   Sharpe ratio: {-result.fun:.2f}")
        else:
            print(f"❌ Optimization failed: {result.message}")
            return False
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Optimization test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print()
    print("#" * 60)
    print("# ALPHATIC PORTFOLIO ANALYZER - INSTALLATION TEST")
    print("#" * 60)
    print()
    
    # Test imports
    if not test_imports():
        print("\n❌ Installation test FAILED - Missing packages")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test versions
    test_versions()
    
    # Test data download
    if not test_data_download():
        print("\n⚠️  Data download test failed")
        print("Check your internet connection")
    
    # Test PyFolio
    if not test_pyfolio():
        print("\n⚠️  PyFolio test failed")
    
    # Test optimization
    if not test_optimization():
        print("\n⚠️  Optimization test failed")
    
    print("=" * 60)
    print("🎉 INSTALLATION TEST COMPLETE!")
    print("=" * 60)
    print()
    print("You can now run the application with:")
    print("  streamlit run alphatic_portfolio_app.py")
    print()
    print("Or use the quick start scripts:")
    print("  ./start.sh     (Linux/Mac)")
    print("  start.bat      (Windows)")
    print()


if __name__ == "__main__":
    main()
