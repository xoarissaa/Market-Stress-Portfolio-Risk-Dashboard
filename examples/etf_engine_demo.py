"""
ETF Analysis Engine Demo

Demonstrates how to use the refactored ETF Analysis Engine as a standalone module.
Shows portfolio creation, metrics calculation, and regime-conditional analysis.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.portfolio import Portfolio
from src.metrics import PerformanceMetrics
from src.regime_context import RegimeContext
from src.definitions import RegimeDefinitions


def demo_basic_portfolio():
    """Demo 1: Create a basic 60/40 portfolio and calculate metrics."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Portfolio (60/40 SPY/TLT)")
    print("="*60)
    
    # Create portfolio with percentage weights (will auto-normalize)
    portfolio = Portfolio({'SPY': 60, 'TLT': 40})
    
    print(f"\n📊 Portfolio Created:")
    print(f"  Tickers: {portfolio.tickers}")
    print(f"  Weights: {dict(portfolio.weights)}")
    
    # Fetch data
    print("\n📥 Fetching data from 2020-01-01...")
    portfolio.fetch_data(start_date="2020-01-01")
    
    # Calculate overall metrics
    print("\n📈 Overall Performance Metrics:")
    metrics = portfolio.get_performance_summary()
    
    for metric, value in metrics.items():
        if 'Ratio' in metric:
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value:.2%}")
    
    return portfolio


def demo_regime_conditional_analysis(portfolio):
    """Demo 2: Analyze portfolio performance during different market regimes."""
    print("\n" + "="*60)
    print("DEMO 2: Regime-Conditional Analysis")
    print("="*60)
    
    # Load regime context
    try:
        regime_ctx = RegimeContext()
        regime_ctx.load_regime_data()
        
        print("\n✅ Regime data loaded successfully")
        
        # Get regime statistics
        stats = regime_ctx.get_regime_statistics()
        print(f"\n📊 Regime Distribution ({stats['total_days']} total days):")
        for regime_code, pct in stats['regime_percentages'].items():
            regime_name = RegimeDefinitions.LABELS.get(regime_code, f"Regime {regime_code}")
            print(f"  {regime_name}: {pct:.1f}%")
        
        # Get regime labels aligned with portfolio
        regime_labels = regime_ctx.get_regime_labels()
        
        # Analyze performance in each regime
        print("\n📈 Performance by Market Regime:")
        print("-" * 60)
        
        for regime_code in sorted(stats['regime_counts'].keys()):
            regime_name = RegimeDefinitions.LABELS.get(regime_code, f"Regime {regime_code}")
            
            try:
                metrics = portfolio.get_regime_conditional_metrics(regime_labels, regime_code)
                
                print(f"\n🔹 {regime_name} (Regime {regime_code}):")
                print(f"  Days in regime: {metrics.get('Days in Regime', 'N/A')}")
                print(f"  CAGR: {metrics.get('CAGR', 0):.2%}")
                print(f"  Volatility: {metrics.get('Volatility', 0):.2%}")
                print(f"  Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
                print(f"  Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}")
                
            except Exception as e:
                print(f"\n🔹 {regime_name}: Error - {e}")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("Please run Phase 1 (Market Regime Detection) first:")
        print("  python src/data_loader.py")
        print("  python src/market_regime.py")


def demo_period_specific_analysis(portfolio):
    """Demo 3: Analyze portfolio during a specific time period."""
    print("\n" + "="*60)
    print("DEMO 3: Period-Specific Analysis (2022 - Bear Market)")
    print("="*60)
    
    # Analyze 2022 performance
    metrics_2022 = portfolio.get_performance_summary(
        start_date="2022-01-01",
        end_date="2022-12-31"
    )
    
    print("\n📉 2022 Performance (Bear Market Year):")
    for metric, value in metrics_2022.items():
        if 'Ratio' in metric:
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value:.2%}")


def demo_downside_capture(portfolio):
    """Demo 4: Calculate downside capture vs SPY."""
    print("\n" + "="*60)
    print("DEMO 4: Downside Capture Analysis")
    print("="*60)
    
    # Create SPY benchmark portfolio
    spy_portfolio = Portfolio({'SPY': 100})
    spy_portfolio.fetch_data(start_date="2020-01-01")
    
    # Calculate downside capture
    dcr = PerformanceMetrics.downside_capture_ratio(
        portfolio.portfolio_returns,
        spy_portfolio.portfolio_returns
    )
    
    print(f"\n🛡️ Downside Capture Ratio vs SPY: {dcr:.2f}")
    
    if dcr < 1.0:
        print(f"  ✅ Portfolio captures {dcr*100:.1f}% of SPY's downside")
        print(f"  This means when SPY falls 1%, your portfolio falls ~{dcr:.2%}")
    else:
        print(f"  ⚠️  Portfolio captures {dcr*100:.1f}% of SPY's downside")
        print(f"  This means when SPY falls 1%, your portfolio falls ~{dcr:.2%}")


def demo_alternative_portfolio():
    """Demo 5: Test with a different portfolio allocation."""
    print("\n" + "="*60)
    print("DEMO 5: Alternative Portfolio (40/30/30 SPY/TLT/GLD)")
    print("="*60)
    
    # Create a more diversified portfolio
    portfolio = Portfolio({
        'SPY': 40,
        'TLT': 30,
        'GLD': 30
    })
    
    print(f"\n📊 Portfolio Created:")
    print(f"  Tickers: {portfolio.tickers}")
    print(f"  Weights: {dict(portfolio.weights)}")
    
    # Fetch data
    print("\n📥 Fetching data...")
    portfolio.fetch_data(start_date="2020-01-01")
    
    # Calculate metrics
    print("\n📈 Overall Performance Metrics:")
    metrics = portfolio.get_performance_summary()
    
    for metric, value in metrics.items():
        if 'Ratio' in metric:
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value:.2%}")
    
    return portfolio


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("ETF ANALYSIS ENGINE DEMONSTRATION")
    print("Phase 2: Modular Portfolio Analysis with Regime Context")
    print("="*60)
    
    # Demo 1: Basic portfolio
    portfolio_60_40 = demo_basic_portfolio()
    
    # Demo 2: Regime-conditional analysis
    demo_regime_conditional_analysis(portfolio_60_40)
    
    # Demo 3: Period-specific analysis
    demo_period_specific_analysis(portfolio_60_40)
    
    # Demo 4: Downside capture
    demo_downside_capture(portfolio_60_40)
    
    # Demo 5: Alternative portfolio
    portfolio_alt = demo_alternative_portfolio()
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE")
    print("="*60)
    print("\nThe ETF Analysis Engine is ready for integration!")
    print("\nKey Features Demonstrated:")
    print("  ✅ Automatic weight normalization")
    print("  ✅ Flexible time range analysis")
    print("  ✅ Regime-conditional metrics")
    print("  ✅ Downside capture calculation")
    print("  ✅ Multiple portfolio configurations")


if __name__ == "__main__":
    main()
