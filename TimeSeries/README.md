# Time Series Theory for AQI Prediction

A structured syllabus covering the time series theory. It assumes prior knowledge of classical machine learning and focuses on what is different about time series data.

## Table of Contents

1. [Foundations](#1-foundations)
2. [EDA and Plotting](#2-eda-and-plotting)
3. [Train/Test Splitting and Validation](#3-traintest-splitting-and-validation)
4. [Baseline Models](#4-baseline-models)
5. [Models](#5-models)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [MLOps for Time Series](#7-mlops-for-time-series)
8. [Out of Scope](#8-out-of-scope)

---

## 1. Foundations

- **Why time series is different:** observations are ordered and correlated, so random shuffling and i.i.d. assumptions break. Treating AQI as ordinary tabular data leaks the future into training.
- **Components:** trend, seasonality (daily, weekly, yearly), cycles, noise/residual.
- **Forecasting setups:**
  - One-step vs multi-step (forecast horizon)
  - Univariate vs multivariate
  - Exogenous variables (weather, wind, temperature, humidity)
- **AQI specifics:** the AQI index is computed from pollutant concentrations (PM2.5, PM10, NO₂, etc.). Decide whether to forecast pollutants and then convert, or forecast the AQI directly.

## 2. EDA and Plotting

- Line plot of the raw series, then zoomed-in views
- Decomposition plots (STL)
- Seasonal plots and box plots by hour, day of week and month
- Lag plots, ACF and PACF plots (and how to read them)
- Rolling mean and rolling standard deviation
- **Stationarity:**
  - What it means and why models need it
  - ADF and KPSS tests
  - Differencing, log and Box-Cox transforms
- **Data cleaning:**
  - Missing timestamps and resampling
  - Interpolation strategies
  - Sensor outliers and spikes

## 3. Train/Test Splitting and Validation

- Chronological split only, never random
- Walk-forward validation (rolling and expanding window)
- `TimeSeriesSplit` in scikit-learn
- A gap between train and test sets to avoid leakage
- **Leakage traps:**
  - Fitting scalers on the full dataset
  - Lag or rolling features computed using future rows
- Backtesting

## 4. Baseline Models

Every model must be compared against these first:

| Baseline | Idea |
|---|---|
| Naive | The last observed value is the forecast |
| Seasonal naive | The value from the same hour yesterday, or the same day last week |
| Moving average | Mean of the last *k* observations |
| Simple exponential smoothing | Exponentially weighted average of past values |

If a model cannot beat seasonal naive, it is not adding value.

## 5. Models

| Tier | Models |
|---|---|
| Statistical | ETS / Holt-Winters, ARIMA, SARIMA, SARIMAX (with exogenous variables) |
| ML with feature engineering | Lag features, rolling statistics, calendar features (hour, month, holiday, season), then XGBoost, LightGBM, Random Forest |
| Modern libraries | Prophet, Darts, sktime |
| Deep learning | LSTM / GRU, 1D CNN, Temporal Fusion Transformer, N-BEATS, PatchTST |

Suggested progression: baselines → SARIMAX → LightGBM → LSTM, compared fairly on the same validation scheme.

## 6. Evaluation Metrics

- **MAE, RMSE:** primary metrics, in the same units as AQI
- **MAPE / sMAPE:** easy to interpret, but MAPE breaks down near zero values
- **MASE:** error scaled against the naive baseline; a value below 1 means the model beats naive
- **R²:** use with caution for time series
- **Residual analysis:** residuals should resemble white noise (ACF of residuals, Ljung-Box test)
- **AQI-specific:**
  - Classification accuracy on AQI category (Good, Moderate, Poor, etc.)
  - Error during high-pollution episodes, since averages hide the spikes that matter
- **Horizon-wise error:** how error grows from 1 hour to 24 hours to 72 hours ahead

## 7. MLOps for Time Series

- **Drift:** data drift and concept drift (seasonal shifts, policy changes)
- **Retraining:** scheduled vs triggered, using a rolling window
- **Monitoring:** live error tracked against the baseline
- **Tooling:** MLflow (experiment tracking), DVC (data versioning), a prediction API and a scheduler

## 8. Out of Scope

Topics intentionally excluded to keep the focus on the project:

- State-space models and Kalman filters
- VAR / VECM
- Bayesian structural time series
- Transformer-heavy architectures beyond the models listed above
