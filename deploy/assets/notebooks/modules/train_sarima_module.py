# Fabric Notebook
# train_sarima_module.py - Per-grain SARIMA training with MLflow

import warnings
import mlflow
import mlflow.statsmodels
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from mlflow.models.signature import infer_signature


def train_sarima_single(
    series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
):
    """Fit SARIMA on a single time series. Returns (model_fit, fitted_values)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=200)
    return fit, fit.fittedvalues


def train_sarima_per_grain(
    df,
    date_column,
    grain_columns,
    target_column,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    test_ratio=0.2,
    experiment_name="demand_forecast",
    model_name="demand_model_sarima",
):
    """
    Train SARIMA per grain combination, log aggregate metrics to MLflow.
    Returns (all_results_df, aggregate_metrics).
    """
    from utils_module import compute_metrics, log_metrics_to_mlflow, ensure_experiment

    ensure_experiment(experiment_name)

    all_results = []
    all_y_true = []
    all_y_pred = []
    grain_count = 0

    groups = df.groupby(grain_columns)

    with mlflow.start_run(run_name="sarima_training") as run:
        mlflow.log_param("model_type", "sarima")
        mlflow.log_param("order", str(order))
        mlflow.log_param("seasonal_order", str(seasonal_order))
        mlflow.log_param("n_grains", len(groups))

        for grain_key, group in groups:
            group = group.sort_values(date_column).reset_index(drop=True)
            if len(group) < max(seasonal_order[3] * 2, 10):
                continue

            series = group[target_column].values
            n = len(series)
            split_idx = int(n * (1 - test_ratio))
            train_series = series[:split_idx]
            test_series = series[split_idx:]

            try:
                fit, fitted = train_sarima_single(train_series, order, seasonal_order)
                forecast = fit.forecast(steps=len(test_series))

                all_y_true.extend(test_series.tolist())
                all_y_pred.extend(forecast.tolist())

                grain_label = grain_key if isinstance(grain_key, str) else "|".join(str(g) for g in grain_key)
                for i, (actual, pred) in enumerate(zip(test_series, forecast)):
                    row = {col: (grain_key[j] if isinstance(grain_key, tuple) else grain_key) for j, col in enumerate(grain_columns)}
                    row["period_offset"] = i
                    row["actual"] = float(actual)
                    row["predicted"] = float(pred)
                    row["model_type"] = "sarima"
                    all_results.append(row)

                grain_count += 1
            except Exception as e:
                grain_label = grain_key if isinstance(grain_key, str) else "|".join(str(g) for g in grain_key)
                print(f"[sarima] Skipping grain {grain_label}: {e}")

        agg_metrics = compute_metrics(all_y_true, all_y_pred) if all_y_true else {"rmse": 0, "mae": 0, "mape": 0, "r2": 0}
        log_metrics_to_mlflow(agg_metrics, prefix="test_")
        mlflow.log_metric("grains_trained", grain_count)

        if all_y_true:
            sample_input = pd.DataFrame({"value": all_y_true[:10]})
            sample_output = pd.Series(all_y_pred[:10], name="forecast")
            mlflow.log_dict({"order": list(order), "seasonal_order": list(seasonal_order)}, "sarima_config.json")

        print(f"[sarima] Run ID: {run.info.run_id}")
        print(f"[sarima] Grains trained: {grain_count}")
        print(f"[sarima] Aggregate test metrics: {agg_metrics}")

    results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    return results_df, agg_metrics
