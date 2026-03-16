# Fabric Notebook
# utils_module.py - Shared helpers: metrics, splits, MLflow utilities

import numpy as np
import mlflow


def compute_metrics(y_true, y_pred):
    """Compute regression/forecast metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    r2 = float(1.0 - np.sum(residuals ** 2) / max(np.sum((y_true - np.mean(y_true)) ** 2), 1e-12))

    mask = np.abs(y_true) > 1e-8
    if mask.sum() > 0:
        mape = float(np.mean(np.abs(residuals[mask] / y_true[mask])) * 100)
    else:
        mape = float("nan")

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def time_split(df, date_column, test_ratio=0.2):
    """Split a pandas DataFrame by time: the most recent `test_ratio` fraction becomes test."""
    df = df.sort_values(date_column).reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def log_metrics_to_mlflow(metrics, prefix=""):
    """Log a dict of metrics to the active MLflow run."""
    for key, value in metrics.items():
        name = f"{prefix}{key}" if prefix else key
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            mlflow.log_metric(name, value)


def ensure_experiment(experiment_name):
    """Set the active MLflow experiment, creating it if needed."""
    mlflow.set_experiment(experiment_name)
