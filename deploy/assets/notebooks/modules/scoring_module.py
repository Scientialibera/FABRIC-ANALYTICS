# Fabric Notebook
# scoring_module.py - Batch scoring using MLflow registry and Fabric PREDICT

import warnings
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime


def score_xgboost(spark, feature_df, model_name, feature_cols):
    """
    Batch-score using Fabric PREDICT / MLFlowTransformer for XGBoost.
    Falls back to direct mlflow.pyfunc.load_model if synapse.ml is unavailable.
    """
    try:
        from synapse.ml.predict import MLFlowTransformer

        latest = mlflow.MlflowClient().get_latest_versions(model_name, stages=["None", "Production"])
        if not latest:
            raise ValueError(f"No registered versions for model '{model_name}'.")
        version = latest[0].version

        spark_df = spark.createDataFrame(feature_df[feature_cols])
        transformer = MLFlowTransformer(
            inputCols=feature_cols,
            outputCol="predicted",
            modelName=model_name,
            modelVersion=int(version),
        )
        scored = transformer.transform(spark_df)
        result_pdf = scored.toPandas()
        feature_df = feature_df.copy()
        feature_df["predicted"] = result_pdf["predicted"].values
        return feature_df
    except ImportError:
        return _score_xgboost_pyfunc(feature_df, model_name, feature_cols)


def _score_xgboost_pyfunc(feature_df, model_name, feature_cols):
    """Fallback: load model via mlflow.pyfunc and predict directly."""
    latest = mlflow.MlflowClient().get_latest_versions(model_name, stages=["None", "Production"])
    if not latest:
        raise ValueError(f"No registered versions for model '{model_name}'.")
    model_uri = f"models:/{model_name}/{latest[0].version}"
    model = mlflow.pyfunc.load_model(model_uri)
    preds = model.predict(feature_df[feature_cols])
    feature_df = feature_df.copy()
    feature_df["predicted"] = preds
    return feature_df


def score_timeseries_model(
    df,
    date_column,
    grain_columns,
    target_column,
    model_type,
    forecast_horizon,
    sarima_order=None,
    sarima_seasonal_order=None,
    ets_trend="add",
    ets_seasonal="add",
    ets_seasonal_periods=12,
):
    """
    Re-fit time series model on full history per grain and forecast `horizon` steps ahead.
    Used for SARIMA and Exponential Smoothing scoring.
    """
    all_forecasts = []
    groups = df.groupby(grain_columns)

    for grain_key, group in groups:
        group = group.sort_values(date_column).reset_index(drop=True)
        series = group[target_column].values.astype(float)

        if len(series) < 10:
            continue

        try:
            if model_type == "sarima":
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = SARIMAX(
                        series,
                        order=sarima_order or (1, 1, 1),
                        seasonal_order=sarima_seasonal_order or (1, 1, 1, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=200)
                forecast = fit.forecast(steps=forecast_horizon)
            else:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = ExponentialSmoothing(
                        series,
                        trend=ets_trend,
                        seasonal=ets_seasonal,
                        seasonal_periods=ets_seasonal_periods,
                        initialization_method="estimated",
                    ).fit(optimized=True)
                forecast = fit.forecast(steps=forecast_horizon)

            last_date = pd.to_datetime(group[date_column].iloc[-1])
            for h in range(forecast_horizon):
                row = {col: (grain_key[j] if isinstance(grain_key, tuple) else grain_key) for j, col in enumerate(grain_columns)}
                row["forecast_step"] = h + 1
                row["predicted"] = float(forecast.iloc[h] if hasattr(forecast, "iloc") else forecast[h])
                row["model_type"] = model_type
                row["scored_at"] = datetime.utcnow().isoformat()
                all_forecasts.append(row)
        except Exception as e:
            grain_label = grain_key if isinstance(grain_key, str) else "|".join(str(g) for g in grain_key)
            print(f"[scoring] Skipping {model_type} grain {grain_label}: {e}")

    return pd.DataFrame(all_forecasts) if all_forecasts else pd.DataFrame()


def build_metrics_summary(predictions_df, target_column="actual", pred_column="predicted", model_type=""):
    """Build a summary metrics row from a predictions DataFrame."""
    if predictions_df.empty or target_column not in predictions_df.columns:
        return {}

    from utils_module import compute_metrics
    metrics = compute_metrics(
        predictions_df[target_column].values,
        predictions_df[pred_column].values,
    )
    metrics["model_type"] = model_type
    metrics["scored_at"] = datetime.utcnow().isoformat()
    metrics["n_predictions"] = len(predictions_df)
    return metrics
