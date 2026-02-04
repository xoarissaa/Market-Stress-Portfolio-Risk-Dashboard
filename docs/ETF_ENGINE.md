# ETF Analysis Engine Documentation

## Overview

The ETF Analysis Engine is a modular Python system for analyzing portfolio performance with market regime context. It provides clean interfaces for portfolio construction, performance metrics calculation, and regime-conditional analysis.

**Phase 2 Focus:** This engine acts as "the Ship" that reacts to "the Weather" from Phase 1 (Market Stress Module).

---

## Architecture

### Core Modules

1. **`portfolio.py`** - Portfolio construction and management
2. **`metrics.py`** - Performance metrics calculations
3. **`regime_context.py`** - Interface to Phase 1 regime data
4. **`risk_engine.py`** - Advanced risk analytics (Monte Carlo, correlations)
5. **`definitions.py`** - Regime labels and descriptions (Phase 1)

### Integration Constraint

The ETF Engine **consumes** regime labels from Phase 1 but does **NOT** re-compute them. Regime data is treated as external context input.

---

## API Reference

### Portfolio Class

**Location:** `src/portfolio.py`

#### Constructor

```python
Portfolio(positions, initial_capital=10000)
```

**Parameters:**
- `positions` (dict): Ticker-weight mapping, e.g., `{'SPY': 60, 'TLT': 40}`
  - Weights can be percentages (60, 40) or decimals (0.6, 0.4)
  - Automatically normalized to sum to 1.0
- `initial_capital` (float): Starting portfolio value (default: 10,000)

**Example:**
```python
from src.portfolio import Portfolio

# Create 60/40 portfolio (auto-normalizes)
port = Portfolio({'SPY': 60, 'TLT': 40})
```

#### Methods

##### `fetch_data(start_date="2010-01-01", end_date=None)`

Fetches historical price data for portfolio assets.

**Parameters:**
- `start_date` (str): Start date in 'YYYY-MM-DD' format
- `end_date` (str, optional): End date in 'YYYY-MM-DD' format

**Returns:** DataFrame of prices

**Example:**
```python
port.fetch_data(start_date="2020-01-01")
```

##### `get_performance_summary(start_date=None, end_date=None)`

Calculate performance metrics for a specific time period.

**Parameters:**
- `start_date` (str/datetime/Timestamp, optional): Filter start date
- `end_date` (str/datetime/Timestamp, optional): Filter end date

**Returns:** Dictionary with metrics:
- `Total Return`
- `CAGR`
- `Volatility`
- `Sharpe Ratio`
- `Max Drawdown`
- `Sortino Ratio`

**Example:**
```python
# Overall metrics
metrics = port.get_performance_summary()

# 2022 only
metrics_2022 = port.get_performance_summary(
    start_date="2022-01-01",
    end_date="2022-12-31"
)
```

##### `get_regime_conditional_metrics(regime_series, regime_code)`

Calculate metrics for periods matching a specific market regime.

**Parameters:**
- `regime_series` (pd.Series): Regime labels indexed by date
- `regime_code` (int): Regime to filter (0=Stable, 1=Uncertain, 2=Stress)

**Returns:** Dictionary with metrics + `Days in Regime`

**Example:**
```python
from src.regime_context import RegimeContext

# Load regime data
ctx = RegimeContext()
regime_labels = ctx.get_regime_labels()

# Analyze performance during High Stress (regime 2)
stress_metrics = port.get_regime_conditional_metrics(regime_labels, 2)
print(f"CAGR during stress: {stress_metrics['CAGR']:.2%}")
```

---

### PerformanceMetrics Class

**Location:** `src/metrics.py`

All methods are static. Input is a pandas Series of daily returns.

#### Methods

##### `total_return(returns)`
Cumulative return over the period.

##### `cagr(returns, periods_per_year=252)`
Compound Annual Growth Rate.

##### `volatility(returns, periods_per_year=252)`
Annualized volatility (standard deviation).

##### `sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)`
Risk-adjusted return metric.

##### `sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)`
Sharpe ratio using only downside deviation.

##### `max_drawdown(returns)`
Maximum peak-to-trough decline.

##### `calmar_ratio(returns, periods_per_year=252)`
CAGR divided by absolute max drawdown.

##### `downside_capture_ratio(portfolio_returns, benchmark_returns)`
Measures portfolio's downside vs benchmark.

**Interpretation:**
- DCR < 1.0: Portfolio falls less than benchmark (good)
- DCR > 1.0: Portfolio falls more than benchmark (bad)

**Example:**
```python
from src.metrics import PerformanceMetrics

dcr = PerformanceMetrics.downside_capture_ratio(
    portfolio_returns,
    spy_returns
)
print(f"Downside Capture: {dcr:.2f}")
# Output: 0.75 means portfolio captures 75% of SPY's downside
```

---

### RegimeContext Class

**Location:** `src/regime_context.py`

Interface to Phase 1 regime data. Does NOT re-compute regimes.

#### Constructor

```python
RegimeContext(regime_data_path="data/processed_market_regimes.parquet")
```

#### Methods

##### `load_regime_data()`
Loads regime data from Phase 1 output.

**Returns:** DataFrame with regime labels and indicators

**Raises:** `FileNotFoundError` if Phase 1 hasn't been run

##### `get_regime_labels(start_date=None, end_date=None)`
Get regime labels for a date range.

**Returns:** pd.Series of regime codes (0, 1, 2) indexed by date

##### `get_regime_statistics()`
Get distribution of regimes in dataset.

**Returns:** Dictionary with:
- `total_days`
- `regime_counts`
- `regime_percentages`
- `date_range`

##### `get_regime_for_date(date)`
Get regime label for a specific date.

**Returns:** Regime code (0, 1, 2) or None

##### `get_regime_periods(regime_code)`
Get all dates for a specific regime.

**Returns:** DatetimeIndex

**Example:**
```python
from src.regime_context import RegimeContext

ctx = RegimeContext()
ctx.load_regime_data()

# Get statistics
stats = ctx.get_regime_statistics()
print(f"Total days: {stats['total_days']}")

# Get all High Stress periods
stress_dates = ctx.get_regime_periods(2)
```

---

## Usage Examples

### Example 1: Basic Portfolio Analysis

```python
from src.portfolio import Portfolio

# Create portfolio
port = Portfolio({'SPY': 60, 'TLT': 40})
port.fetch_data(start_date="2020-01-01")

# Get metrics
metrics = port.get_performance_summary()
print(f"CAGR: {metrics['CAGR']:.2%}")
print(f"Sharpe: {metrics['Sharpe Ratio']:.2f}")
```

### Example 2: Regime-Conditional Analysis

```python
from src.portfolio import Portfolio
from src.regime_context import RegimeContext
from src.definitions import RegimeDefinitions

# Setup
port = Portfolio({'SPY': 60, 'TLT': 40})
port.fetch_data(start_date="2020-01-01")

ctx = RegimeContext()
regime_labels = ctx.get_regime_labels()

# Analyze each regime
for regime_code in [0, 1, 2]:
    regime_name = RegimeDefinitions.LABELS[regime_code]
    metrics = port.get_regime_conditional_metrics(regime_labels, regime_code)
    
    print(f"\n{regime_name}:")
    print(f"  CAGR: {metrics['CAGR']:.2%}")
    print(f"  Volatility: {metrics['Volatility']:.2%}")
    print(f"  Days: {metrics['Days in Regime']}")
```

### Example 3: Period-Specific Analysis

```python
# Analyze 2022 bear market
metrics_2022 = port.get_performance_summary(
    start_date="2022-01-01",
    end_date="2022-12-31"
)

print(f"2022 Return: {metrics_2022['Total Return']:.2%}")
print(f"2022 Max DD: {metrics_2022['Max Drawdown']:.2%}")
```

### Example 4: Downside Protection Analysis

```python
from src.metrics import PerformanceMetrics

# Create benchmark
spy = Portfolio({'SPY': 100})
spy.fetch_data(start_date="2020-01-01")

# Calculate downside capture
dcr = PerformanceMetrics.downside_capture_ratio(
    port.portfolio_returns,
    spy.portfolio_returns
)

print(f"Downside Capture: {dcr:.2f}")
if dcr < 1.0:
    print(f"✅ Portfolio is more defensive than SPY")
```

---

## Metric Definitions

### CAGR (Compound Annual Growth Rate)
Annualized return assuming reinvestment.

**Formula:** `(Final Value / Initial Value)^(1/Years) - 1`

### Volatility
Annualized standard deviation of returns.

**Formula:** `σ_daily × √252`

### Sharpe Ratio
Risk-adjusted return.

**Formula:** `(Return - Risk Free Rate) / Volatility`

**Interpretation:**
- \> 1.0: Good
- \> 2.0: Very good
- \> 3.0: Excellent

### Sortino Ratio
Like Sharpe, but only penalizes downside volatility.

**Formula:** `(Return - Risk Free Rate) / Downside Deviation`

### Max Drawdown
Largest peak-to-trough decline.

**Formula:** `min((Price - Peak) / Peak)`

### Downside Capture Ratio
Portfolio's downside vs benchmark during down markets.

**Formula:** `Avg Portfolio Return (when Benchmark < 0) / Avg Benchmark Return (when Benchmark < 0)`

---

## Integration with Phase 1

Phase 1 (Market Stress Module) outputs:
- File: `data/processed_market_regimes.parquet`
- Contains: `Regime_Label` column (0, 1, 2)

Phase 2 (ETF Engine) consumes this via `RegimeContext`:

```python
# Phase 1 is locked - we only READ regime data
ctx = RegimeContext()
regime_labels = ctx.get_regime_labels()

# Use regime data for conditional analysis
port.get_regime_conditional_metrics(regime_labels, regime_code=2)
```

**Critical:** ETF Engine does NOT modify or re-compute regime labels.

---

## Testing

### Run Automated Tests

```bash
python tests/test_etf_engine.py
```

**Tests cover:**
- Weight normalization
- Portfolio validation
- Single ticker portfolios
- Metrics calculations
- Downside capture
- Regime-conditional analysis
- RegimeContext loading

### Run Demo

```bash
python examples/etf_engine_demo.py
```

**Demonstrates:**
- Basic portfolio creation
- Regime-conditional analysis
- Period-specific analysis
- Downside capture calculation
- Alternative portfolio configurations

---

## Design Principles

1. **Modularity** - Each module has a single responsibility
2. **No Re-computation** - Phase 1 regime data is consumed, not re-calculated
3. **Flexible Inputs** - Accept both percentage and decimal weights
4. **Robust Validation** - Catch errors early with clear messages
5. **Clean Interfaces** - Simple, predictable API
6. **Testability** - All components can be tested independently

---

## Future Enhancements (Out of Scope for Phase 2)

- Portfolio persistence (database)
- Investment tracking (expected vs actual)
- Gold analysis tab
- ETF universe expansion
- UI redesign

These are reserved for later phases.
