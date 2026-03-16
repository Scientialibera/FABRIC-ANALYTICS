# Fabric Notebook
# train_xgboost_module.py - XGBoost training logic with MLflow

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from mlflow.models.signature import infer_signature


def train_xgboost(
    train_df,
    test_df,
    feature_cols,
    target_col,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    experiment_name="demand_forecast",
    model_name="demand_model_xgboost",
):
    """
    Train XGBoost regressor, log to MLflow, register model.
    Returns (model, train_preds, test_preds, metrics_dict).
    """
    from utils_module import compute_metrics, log_metrics_to_mlflow, ensure_experiment

    ensure_experiment(experiment_name)

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    with mlflow.start_run(run_name="xgboost_training") as run:
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train", len(train_df))
        mlflow.log_param("n_test", len(test_df))

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_metrics = compute_metrics(y_train, train_preds)
        test_metrics = compute_metrics(y_test, test_preds)
        log_metrics_to_mlflow(train_metrics, prefix="train_")
        log_metrics_to_mlflow(test_metrics, prefix="test_")

        importance = model.feature_importances_
        for i, col in enumerate(feature_cols):
            mlflow.log_metric(f"importance_{col}", float(importance[i]))

        signature = infer_signature(
            pd.DataFrame(X_train, columns=feature_cols),
            pd.Series(train_preds, name=target_col),
        )
        mlflow.xgboost.log_model(
            model,
            "xgboost_model",
            signature=signature,
            registered_model_name=model_name,
        )

        print(f"[xgboost] Run ID: {run.info.run_id}")
        print(f"[xgboost] Train metrics: {train_metrics}")
        print(f"[xgboost] Test metrics:  {test_metrics}")

    return model, train_preds, test_preds, test_metrics
