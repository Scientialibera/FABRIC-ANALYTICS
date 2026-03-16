# Fabric Notebook
# train_exp_smoothing_module.py - Per-grain Exponential Smoothing training with MLflow

import warnings
import mlflow
import mlflow.statsmodels
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from mlflow.models.signature import infer_signature


def train_ets_single(
    series,
    trend="add",
    seasonal="add",
    seasonal_periods=12,
):
    """Fit Holt-Winters Exponential Smoothing on a single series."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
    return fit, fit.fittedvalues


def train_exp_smoothing_per_grain(
    df,
    date_column,
    grain_columns,
    target_column,
    trend="add",
    seasonal="add",
    seasonal_periods=12,
    test_ratio=0.2,
    experiment_name="demand_forecast",
    model_name="demand_model_exp_smoothing",
):
    """
    Train Exponential Smoothing per grain, log aggregate metrics to MLflow.
    Returns (all_results_df, aggregate_metrics).
    """
    from utils_module import compute_metrics, log_metrics_to_mlflow, ensure_experiment

    ensure_experiment(experiment_name)

    all_results = []
    all_y_true = []
    all_y_pred = []
    grain_count = 0

    groups = df.groupby(grain_columns)

    with mlflow.start_run(run_name="exp_smoothing_training") as run:
        mlflow.log_param("model_type", "exponential_smoothing")
        mlflow.log_param("trend", trend)
        mlflow.log_param("seasonal", seasonal)
        mlflow.log_param("seasonal_periods", seasonal_periods)
        mlflow.log_param("n_grains", len(groups))

        for grain_key, group in groups:
            group = group.sort_values(date_column).reset_index(drop=True)
            if len(group) < seasonal_periods * 2:
                continue

            series = group[target_column].values.astype(float)
            n = len(series)
            split_idx = int(n * (1 - test_ratio))
            train_series = series[:split_idx]
            test_series = series[split_idx:]

            try:
                fit, fitted = train_ets_single(train_series, trend, seasonal, seasonal_periods)
                forecast = fit.forecast(steps=len(test_series))

                all_y_true.extend(test_series.tolist())
                all_y_pred.extend(forecast.tolist())

                for i, (actual, pred) in enumerate(zip(test_series, forecast)):
                    row = {col: (grain_key[j] if isinstance(grain_key, tuple) else grain_key) for j, col in enumerate(grain_columns)}
                    row["period_offset"] = i
                    row["actual"] = float(actual)
                    row["predicted"] = float(pred)
                    row["model_type"] = "exp_smoothing"
                    all_results.append(row)

                grain_count += 1
            except Exception as e:
                grain_label = grain_key if isinstance(grain_key, str) else "|".join(str(g) for g in grain_key)
                print(f"[exp_smoothing] Skipping grain {grain_label}: {e}")

        agg_metrics = compute_metrics(all_y_true, all_y_pred) if all_y_true else {"rmse": 0, "mae": 0, "mape": 0, "r2": 0}
        log_metrics_to_mlflow(agg_metrics, prefix="test_")
        mlflow.log_metric("grains_trained", grain_count)

        mlflow.log_dict(
            {"trend": trend, "seasonal": seasonal, "seasonal_periods": seasonal_periods},
            "ets_config.json",
        )

        print(f"[exp_smoothing] Run ID: {run.info.run_id}")
        print(f"[exp_smoothing] Grains trained: {grain_count}")
        print(f"[exp_smoothing] Aggregate test metrics: {agg_metrics}")

    results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    return results_df, agg_metrics
