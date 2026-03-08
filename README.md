This is the Technical Test for candidate Paulo Hubert.

# Repository Overview

This repository contains an end-to-end forecasting workflow for multiple time series defined by `INVENTORY_ID` and `REGION`.

Main goals:
- Explore and understand the demand data.
- Classify time series by data availability and sparsity.
- Compare forecasting models with backtesting.
- Select the best model per series using a configurable metric.
- Generate one-step and multi-step forecasts with confidence intervals.

# Project Structure

- `analytics_lib/`
  - Core reusable Python modules:
    - `classification.py`: time-series class assignment logic.
    - `forecast_models.py`: model interface and implementations (Naive, ARIMA, TSB).
    - `backtest_models.py`: backtesting utilities and error metrics.
- `notebooks/`
  - `PruebaTecnica.ipynb`: main analysis and forecasting notebook.
- `data/`
  - Input datasets used by the notebook and modules.
- `results/`
  - Tabular result artifacts (intermediate/final outputs).
- `forecasts/`
  - Forecast export files (series-level and aggregated outputs).
- `reports/`
  - Executive summary and presentation/report artifacts.

# Workflow Summary

1. Load and inspect data.
2. Classify each series (`Low data`, `Zero-inflated`, `Full series`).
3. Define allowed models per class.
4. Run class-specific backtests.
5. Build point-error and interval-coverage reports.
6. Select best model per series.
7. Fit selected models on full history and generate forecasts.

# Notes

- Confidence intervals are produced per model and used in evaluation.
- Aggregated confidence intervals are currently formed by summing lower/upper bounds across series (simple approximation).
