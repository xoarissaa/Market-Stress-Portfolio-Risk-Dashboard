"""
Unified Comparison Engine

Compares two portfolios (or ETFs treated as 100% portfolios) with regime-conditional analysis.
Provides decision-focused summaries for practical decision-making.
"""

import pandas as pd
import numpy as np
from src.portfolio import Portfolio
from src.metrics import PerformanceMetrics
from src.definitions import RegimeDefinitions


class ComparisonEngine:
    """
    Compares two portfolios with overall and regime-conditional metrics.
    Generates decision-focused summaries.
    """
    
    def __init__(self, portfolio_a, portfolio_b, regime_labels, benchmark_returns=None):
        """
        Initialize comparison engine.
        
        :param portfolio_a: Portfolio object for option A
        :param portfolio_b: Portfolio object for option B
        :param regime_labels: pd.Series of regime labels (0, 1, 2)
        :param benchmark_returns: Optional benchmark returns (defaults to SPY if available)
        """
        self.portfolio_a = portfolio_a
        self.portfolio_b = portfolio_b
        self.regime_labels = regime_labels
        self.benchmark_returns = benchmark_returns
        
        # Results storage
        self.overall_comparison = None
        self.regime_comparisons = {}
        self.decision_summary = None
        
    def compare_overall_metrics(self):
        """
        Compare overall performance metrics between portfolios.
        
        :return: Dictionary with side-by-side metrics
        """
        metrics_a = self.portfolio_a.get_performance_summary()
        metrics_b = self.portfolio_b.get_performance_summary()
        
        # Add downside capture if benchmark available
        if self.benchmark_returns is not None:
            metrics_a['Downside Capture'] = PerformanceMetrics.downside_capture_ratio(
                self.portfolio_a.portfolio_returns,
                self.benchmark_returns
            )
            metrics_b['Downside Capture'] = PerformanceMetrics.downside_capture_ratio(
                self.portfolio_b.portfolio_returns,
                self.benchmark_returns
            )
        
        # Build comparison dict
        comparison = {
            'metrics': list(metrics_a.keys()),
            'portfolio_a': list(metrics_a.values()),
            'portfolio_b': list(metrics_b.values()),
            'winner': []
        }
        
        # Determine winners
        for metric, val_a, val_b in zip(comparison['metrics'], 
                                         comparison['portfolio_a'], 
                                         comparison['portfolio_b']):
            winner = self._determine_winner(metric, val_a, val_b)
            comparison['winner'].append(winner)
        
        self.overall_comparison = comparison
        return comparison
    
    def compare_by_regime(self, regime_code):
        """
        Compare performance during a specific market regime.
        
        :param regime_code: Regime code (0=Stable, 1=Uncertain, 2=Stress)
        :return: Dictionary with regime-specific metrics
        """
        metrics_a = self.portfolio_a.get_regime_conditional_metrics(
            self.regime_labels, regime_code
        )
        metrics_b = self.portfolio_b.get_regime_conditional_metrics(
            self.regime_labels, regime_code
        )
        
        # Build comparison dict
        comparison = {
            'regime_code': regime_code,
            'regime_name': RegimeDefinitions.LABELS.get(regime_code, f"Regime {regime_code}"),
            'metrics': list(metrics_a.keys()),
            'portfolio_a': list(metrics_a.values()),
            'portfolio_b': list(metrics_b.values()),
            'winner': []
        }
        
        # Determine winners (skip 'Days in Regime')
        for metric, val_a, val_b in zip(comparison['metrics'], 
                                         comparison['portfolio_a'], 
                                         comparison['portfolio_b']):
            if metric == 'Days in Regime':
                comparison['winner'].append(None)
            else:
                winner = self._determine_winner(metric, val_a, val_b)
                comparison['winner'].append(winner)
        
        self.regime_comparisons[regime_code] = comparison
        return comparison
    
    def compare_all_regimes(self):
        """
        Compare performance across all regimes.
        
        :return: Dictionary of regime comparisons
        """
        for regime_code in [0, 1, 2]:
            self.compare_by_regime(regime_code)
        
        return self.regime_comparisons
    
    def generate_decision_summary(self):
        """
        Generate plain-English decision-focused summary.
        
        :return: String with actionable insights
        """
        if self.overall_comparison is None:
            self.compare_overall_metrics()
        
        if not self.regime_comparisons:
            self.compare_all_regimes()
        
        insights = []
        
        # 1. Overall winner
        overall_winner = self._get_overall_winner()
        insights.append(f"**Overall Assessment:** Portfolio {overall_winner} shows stronger overall metrics.")
        
        # 2. Defensive behavior (Downside Capture + High Stress performance)
        defensive_insight = self._analyze_defensive_behavior()
        if defensive_insight:
            insights.append(defensive_insight)
        
        # 3. Volatility during stress
        stress_vol_insight = self._analyze_stress_volatility()
        if stress_vol_insight:
            insights.append(stress_vol_insight)
        
        # 4. Risk-adjusted returns
        sharpe_insight = self._analyze_risk_adjusted_returns()
        if sharpe_insight:
            insights.append(sharpe_insight)
        
        # 5. Growth potential
        growth_insight = self._analyze_growth_potential()
        if growth_insight:
            insights.append(growth_insight)
        
        self.decision_summary = "\n\n".join(insights)
        return self.decision_summary
    
    def _determine_winner(self, metric_name, value_a, value_b):
        """
        Determine which portfolio wins for a given metric.
        
        :param metric_name: Name of the metric
        :param value_a: Portfolio A value
        :param value_b: Portfolio B value
        :return: 'A', 'B', or None
        """
        # Handle None or error values
        if value_a is None or value_b is None:
            return None
        if isinstance(value_a, str) or isinstance(value_b, str):
            return None
        
        # Metrics where higher is better
        better_higher = ['Total Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
        
        # Metrics where lower is better (absolute value for Max DD)
        better_lower = ['Volatility', 'Max Drawdown', 'Downside Capture']
        
        if metric_name in better_higher:
            return 'A' if value_a > value_b else 'B'
        elif metric_name in better_lower:
            # For Max Drawdown, more negative is worse
            if metric_name == 'Max Drawdown':
                return 'A' if value_a > value_b else 'B'  # Less negative is better
            else:
                return 'A' if value_a < value_b else 'B'
        else:
            return None
    
    def _get_overall_winner(self):
        """Determine overall winner based on majority of metrics."""
        winners = [w for w in self.overall_comparison['winner'] if w is not None]
        if not winners:
            return "Neither"
        
        count_a = winners.count('A')
        count_b = winners.count('B')
        
        if count_a > count_b:
            return 'A'
        elif count_b > count_a:
            return 'B'
        else:
            return 'Neither (Tie)'
    
    def _analyze_defensive_behavior(self):
        """Analyze which portfolio is more defensive."""
        # Check downside capture
        dc_idx = None
        for i, metric in enumerate(self.overall_comparison['metrics']):
            if metric == 'Downside Capture':
                dc_idx = i
                break
        
        if dc_idx is None:
            return None
        
        dc_a = self.overall_comparison['portfolio_a'][dc_idx]
        dc_b = self.overall_comparison['portfolio_b'][dc_idx]
        
        # Check High Stress regime performance
        stress_comp = self.regime_comparisons.get(2)
        if stress_comp:
            # Find CAGR in stress regime
            cagr_idx = stress_comp['metrics'].index('CAGR') if 'CAGR' in stress_comp['metrics'] else None
            if cagr_idx:
                stress_cagr_a = stress_comp['portfolio_a'][cagr_idx]
                stress_cagr_b = stress_comp['portfolio_b'][cagr_idx]
                
                if dc_a < dc_b:
                    return f"**Defensive Behavior:** Portfolio A was more defensive, capturing {dc_a:.1%} of market downside vs {dc_b:.1%} for Portfolio B. During High Stress regimes, Portfolio A averaged {stress_cagr_a:.1%} vs {stress_cagr_b:.1%} for Portfolio B."
                else:
                    return f"**Defensive Behavior:** Portfolio B was more defensive, capturing {dc_b:.1%} of market downside vs {dc_a:.1%} for Portfolio A. During High Stress regimes, Portfolio B averaged {stress_cagr_b:.1%} vs {stress_cagr_a:.1%} for Portfolio A."
        
        return None
    
    def _analyze_stress_volatility(self):
        """Analyze volatility during stress periods."""
        stress_comp = self.regime_comparisons.get(2)
        if not stress_comp:
            return None
        
        vol_idx = stress_comp['metrics'].index('Volatility') if 'Volatility' in stress_comp['metrics'] else None
        if vol_idx is None:
            return None
        
        vol_a = stress_comp['portfolio_a'][vol_idx]
        vol_b = stress_comp['portfolio_b'][vol_idx]
        
        if vol_a < vol_b:
            diff = ((vol_b - vol_a) / vol_b) * 100
            return f"**Stress Volatility:** Portfolio A was {diff:.0f}% less volatile during High Stress periods ({vol_a:.1%} vs {vol_b:.1%})."
        else:
            diff = ((vol_a - vol_b) / vol_a) * 100
            return f"**Stress Volatility:** Portfolio B was {diff:.0f}% less volatile during High Stress periods ({vol_b:.1%} vs {vol_a:.1%})."
    
    def _analyze_risk_adjusted_returns(self):
        """Analyze Sharpe ratios."""
        sharpe_idx = None
        for i, metric in enumerate(self.overall_comparison['metrics']):
            if metric == 'Sharpe Ratio':
                sharpe_idx = i
                break
        
        if sharpe_idx is None:
            return None
        
        sharpe_a = self.overall_comparison['portfolio_a'][sharpe_idx]
        sharpe_b = self.overall_comparison['portfolio_b'][sharpe_idx]
        
        if sharpe_a > sharpe_b:
            return f"**Risk-Adjusted Returns:** Portfolio A delivered better risk-adjusted returns (Sharpe: {sharpe_a:.2f} vs {sharpe_b:.2f})."
        else:
            return f"**Risk-Adjusted Returns:** Portfolio B delivered better risk-adjusted returns (Sharpe: {sharpe_b:.2f} vs {sharpe_a:.2f})."
    
    def _analyze_growth_potential(self):
        """Analyze growth during stable periods."""
        stable_comp = self.regime_comparisons.get(0)
        if not stable_comp:
            return None
        
        cagr_idx = stable_comp['metrics'].index('CAGR') if 'CAGR' in stable_comp['metrics'] else None
        if cagr_idx is None:
            return None
        
        cagr_a = stable_comp['portfolio_a'][cagr_idx]
        cagr_b = stable_comp['portfolio_b'][cagr_idx]
        
        if cagr_a > cagr_b:
            return f"**Growth Potential:** Portfolio A showed higher returns during Stable Growth periods ({cagr_a:.1%} vs {cagr_b:.1%}), suggesting better upside capture."
        else:
            return f"**Growth Potential:** Portfolio B showed higher returns during Stable Growth periods ({cagr_b:.1%} vs {cagr_a:.1%}), suggesting better upside capture."


def format_comparison_table(comparison_dict):
    """
    Format comparison results as a pandas DataFrame for display.
    
    :param comparison_dict: Dictionary from compare_overall_metrics or compare_by_regime
    :return: pd.DataFrame
    """
    df = pd.DataFrame({
        'Metric': comparison_dict['metrics'],
        'Portfolio A': comparison_dict['portfolio_a'],
        'Portfolio B': comparison_dict['portfolio_b'],
        'Winner': comparison_dict['winner']
    })
    
    return df
