# Stage C Research Validation

Stage C adds a research/validation layer on top of the existing backtest pipeline without changing execution/lifecycle behavior.

## API

`core/research_validation.py` exposes:

- `run_backtest(config)`  
  Runs chronological `train -> validation -> test` split and returns segment results, overfitting indicators, and strategy status (`working` / `unstable` / `overfitted`).

- `run_walk_forward(config)`  
  Runs rolling windows `train -> test` and returns fold-by-fold out-of-sample metrics and stability summary.

- `run_sensitivity(config)`  
  Runs parameter grid checks (`param_grid`) on chosen window and computes robustness score.

- `analyze_portfolio(results)`  
  Produces portfolio-level analytics:
  - PnL, win rate, expectancy, profit factor
  - drawdown/equity diagnostics
  - turnover/exposure
  - contribution by strategy/instrument
  - remove-one robustness tables
  - correlation stability snapshot
  - distribution stats (tail, skew, kurtosis, losing streaks)

## Design Boundaries

- Research functions reuse `run_portfolio_backtest` and do not alter trading logic.
- No strategy optimization loop is introduced; sensitivity is diagnostic only.
- Train/test separation is chronological to avoid look-ahead leakage.
- Results stay in domain-level contracts (`Trade` rows / aggregated metrics), not simulator internals.
