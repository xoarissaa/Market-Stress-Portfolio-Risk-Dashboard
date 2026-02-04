"""
Test Suite for ETF Analysis Engine (Phase 2)

Tests portfolio weight normalization, regime-conditional analysis,
metrics calculations, and edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.portfolio import Portfolio
from src.metrics import PerformanceMetrics
from src.regime_context import RegimeContext


def test_weight_normalization():
    """Test that portfolio weights are automatically normalized."""
    print("\n🧪 Test 1: Weight Normalization")
    
    # Test with percentages (60, 40)
    port1 = Portfolio({'SPY': 60, 'TLT': 40})
    assert abs(sum(port1.weights) - 1.0) < 0.001, "Weights should sum to 1.0"
    assert abs(port1.weights['SPY'] - 0.6) < 0.001, "SPY should be 60%"
    print("  ✅ Percentage weights (60, 40) normalized correctly")
    
    # Test with decimals (0.6, 0.4)
    port2 = Portfolio({'SPY': 0.6, 'TLT': 0.4})
    assert abs(sum(port2.weights) - 1.0) < 0.001, "Weights should sum to 1.0"
    print("  ✅ Decimal weights (0.6, 0.4) normalized correctly")
    
    # Test with uneven weights (70, 30, 20) -> should normalize to (0.583, 0.25, 0.167)
    port3 = Portfolio({'SPY': 70, 'TLT': 30, 'GLD': 20})
    assert abs(sum(port3.weights) - 1.0) < 0.001, "Weights should sum to 1.0"
    assert abs(port3.weights['SPY'] - 70/120) < 0.001, "SPY should be 70/120"
    print("  ✅ Uneven weights (70, 30, 20) normalized correctly")
    
    print("✅ Weight normalization tests passed")


def test_portfolio_validation():
    """Test portfolio validation catches invalid inputs."""
    print("\n🧪 Test 2: Portfolio Validation")
    
    # Empty portfolio
    try:
        Portfolio({})
        assert False, "Should raise ValueError for empty portfolio"
    except ValueError as e:
        print(f"  ✅ Empty portfolio rejected: {e}")
    
    # Negative weight
    try:
        Portfolio({'SPY': 60, 'TLT': -40})
        assert False, "Should raise ValueError for negative weight"
    except ValueError as e:
        print(f"  ✅ Negative weight rejected: {e}")
    
    # Invalid ticker
    try:
        Portfolio({'': 50, 'TLT': 50})
        assert False, "Should raise ValueError for empty ticker"
    except ValueError as e:
        print(f"  ✅ Empty ticker rejected: {e}")
    
    print("✅ Portfolio validation tests passed")


def test_single_ticker_portfolio():
    """Test portfolio with single ticker."""
    print("\n🧪 Test 3: Single Ticker Portfolio")
    
    port = Portfolio({'SPY': 100})
    assert len(port.tickers) == 1, "Should have 1 ticker"
    assert port.weights['SPY'] == 1.0, "Single ticker should have 100% weight"
    print("  ✅ Single ticker portfolio created successfully")
    
    print("✅ Single ticker portfolio test passed")


def test_metrics_with_synthetic_data():
    """Test metrics calculations with known synthetic data."""
    print("\n🧪 Test 4: Metrics with Synthetic Data")
    
    # Create synthetic returns: 1% daily for 252 days
    returns = pd.Series([0.01] * 252, index=pd.date_range('2020-01-01', periods=252))
    
    # Test total return
    total_ret = PerformanceMetrics.total_return(returns)
    expected = (1.01 ** 252) - 1
    assert abs(total_ret - expected) < 0.01, f"Total return mismatch: {total_ret} vs {expected}"
    print(f"  ✅ Total return: {total_ret:.2%}")
    
    # Test CAGR (should be close to 1% * 252 compounded)
    cagr = PerformanceMetrics.cagr(returns)
    assert cagr > 0, "CAGR should be positive"
    print(f"  ✅ CAGR: {cagr:.2%}")
    
    # Test volatility
    vol = PerformanceMetrics.volatility(returns)
    assert vol >= 0, "Volatility should be non-negative"
    print(f"  ✅ Volatility: {vol:.2%}")
    
    # Test Sharpe ratio (constant returns have zero volatility, so Sharpe will be 0 or inf)
    sharpe = PerformanceMetrics.sharpe_ratio(returns)
    # For constant positive returns, Sharpe might be 0 due to zero volatility
    assert sharpe >= 0, f"Sharpe should be non-negative, got {sharpe}"
    print(f"  ✅ Sharpe Ratio: {sharpe:.2f}")
    
    # Test max drawdown (should be 0 for constant positive returns)
    mdd = PerformanceMetrics.max_drawdown(returns)
    assert mdd <= 0, "Max drawdown should be non-positive"
    print(f"  ✅ Max Drawdown: {mdd:.2%}")
    
    print("✅ Metrics calculation tests passed")


def test_downside_capture():
    """Test downside capture ratio calculation."""
    print("\n🧪 Test 5: Downside Capture Ratio")
    
    # Create synthetic data
    dates = pd.date_range('2020-01-01', periods=100)
    
    # Portfolio that falls 0.5% when benchmark falls 1%
    benchmark = pd.Series([0.01 if i % 2 == 0 else -0.01 for i in range(100)], index=dates)
    portfolio = pd.Series([0.01 if i % 2 == 0 else -0.005 for i in range(100)], index=dates)
    
    dcr = PerformanceMetrics.downside_capture_ratio(portfolio, benchmark)
    
    # Expected: -0.005 / -0.01 = 0.5
    assert abs(dcr - 0.5) < 0.01, f"DCR should be ~0.5, got {dcr}"
    print(f"  ✅ Downside Capture Ratio: {dcr:.2f} (expected ~0.5)")
    
    print("✅ Downside capture test passed")


def test_regime_conditional_metrics():
    """Test regime-conditional metrics calculation."""
    print("\n🧪 Test 6: Regime-Conditional Metrics")
    
    # Create synthetic portfolio with returns
    dates = pd.date_range('2020-01-01', periods=100)
    returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
    
    # Create synthetic regime labels (0, 1, 2)
    regime_labels = pd.Series([0] * 40 + [1] * 30 + [2] * 30, index=dates)
    
    # Create a mock portfolio
    port = Portfolio({'SPY': 100})
    port.portfolio_returns = returns
    
    # Test regime-conditional metrics for regime 0
    metrics_r0 = port.get_regime_conditional_metrics(regime_labels, 0)
    
    assert 'CAGR' in metrics_r0, "Should have CAGR metric"
    assert 'Volatility' in metrics_r0, "Should have Volatility metric"
    assert 'Days in Regime' in metrics_r0, "Should have Days in Regime"
    assert metrics_r0['Days in Regime'] == 40, f"Should have 40 days in regime 0, got {metrics_r0['Days in Regime']}"
    
    print(f"  ✅ Regime 0 metrics calculated: {metrics_r0['Days in Regime']} days")
    
    # Test regime 1
    metrics_r1 = port.get_regime_conditional_metrics(regime_labels, 1)
    assert metrics_r1['Days in Regime'] == 30, "Should have 30 days in regime 1"
    print(f"  ✅ Regime 1 metrics calculated: {metrics_r1['Days in Regime']} days")
    
    print("✅ Regime-conditional metrics test passed")


def test_regime_context_loading():
    """Test RegimeContext can load data (if available)."""
    print("\n🧪 Test 7: RegimeContext Loading")
    
    ctx = RegimeContext()
    
    try:
        ctx.load_regime_data()
        print("  ✅ Regime data loaded successfully")
        
        stats = ctx.get_regime_statistics()
        print(f"  ✅ Regime statistics: {stats['total_days']} days")
        
        labels = ctx.get_regime_labels()
        print(f"  ✅ Retrieved {len(labels)} regime labels")
        
        print("✅ RegimeContext loading test passed")
        
    except FileNotFoundError:
        print("  ⚠️  Regime data not found (run Phase 1 first)")
        print("✅ RegimeContext test skipped (data not available)")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("ETF Analysis Engine Test Suite")
    print("=" * 60)
    
    test_weight_normalization()
    test_portfolio_validation()
    test_single_ticker_portfolio()
    test_metrics_with_synthetic_data()
    test_downside_capture()
    test_regime_conditional_metrics()
    test_regime_context_loading()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
