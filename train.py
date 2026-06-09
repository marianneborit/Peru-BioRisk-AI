"""
Peru BioRisk AI — Model Training
XGBoost/LightGBM ensemble with spatial cross-validation and MLflow tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import shap
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import BaseCrossValidator
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# ── Spatial block cross-validator ─────────────────────────────────────────────

class SpatialBlockCV(BaseCrossValidator):
    """
    Splits districts into spatial blocks of ~block_size_km and uses them
    as CV folds. Prevents geographic data leakage between train and test.
    """

    def __init__(
        self,
        n_splits: int = 5,
        block_size_km: float = 50.0,
        ubigeo_col: str = "ubigeo",
        coords_df: pd.DataFrame | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.block_size_km = block_size_km
        self.ubigeo_col = ubigeo_col
        self.coords_df = coords_df  # must have ubigeo, centroid_x_m, centroid_y_m

    def _iter_test_masks(self, X: pd.DataFrame, y: Any = None) -> Any:  # noqa: ANN401
        if self.coords_df is None:
            # Fall back to random k-fold
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_STATE)
            yield from kf._iter_test_masks(X, y)
            return

        merged = X[[self.ubigeo_col]].merge(self.coords_df, on=self.ubigeo_col, how="left")
        bx = (merged["centroid_x_m"] // (self.block_size_km * 1000)).astype(int)
        by = (merged["centroid_y_m"] // (self.block_size_km * 1000)).astype(int)
        block_id = bx.astype(str) + "_" + by.astype(str)
        unique_blocks = block_id.unique()
        np.random.seed(RANDOM_STATE)
        np.random.shuffle(unique_blocks)
        fold_assignments = np.array_split(unique_blocks, self.n_splits)

        for fold_blocks in fold_assignments:
            test_mask = block_id.isin(fold_blocks).values
            yield test_mask

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits


# ── Model config ──────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    disease: str = "dengue"
    task: str = "classification"       # classification | regression
    horizon_weeks: int = 4
    feature_cols: list[str] = field(default_factory=list)
    target_col: str = "outbreak_label"
    n_trials_optuna: int = 50
    n_cv_folds: int = 5
    block_size_km: float = 50.0
    mlflow_experiment: str = "peru-biorisk-ai"


# ── Optuna objective ──────────────────────────────────────────────────────────

def _xgb_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: BaseCrossValidator,
    task: str,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "device": "cpu",
    }

    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        if task == "classification":
            model = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))
        else:
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            scores.append(-mean_absolute_error(y_val, preds))

    return float(np.mean(scores))


# ── Ensemble builder ──────────────────────────────────────────────────────────

def train_ensemble(
    df: pd.DataFrame,
    config: ModelConfig,
    cv: BaseCrossValidator | None = None,
    coords_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Trains XGBoost + LightGBM ensemble with Optuna hyperparameter search.
    Logs all metrics, params, and SHAP values to MLflow.

    Returns a dict with trained models, feature importances, and eval metrics.
    """
    mlflow.set_experiment(config.mlflow_experiment)

    if not config.feature_cols:
        raise ValueError("feature_cols must be specified in ModelConfig")

    X = df[config.feature_cols].fillna(0).values
    y = df[config.target_col].values

    cv = cv or SpatialBlockCV(
        n_splits=config.n_cv_folds,
        block_size_km=config.block_size_km,
        coords_df=coords_df,
    )

    with mlflow.start_run(run_name=f"{config.disease}_h{config.horizon_weeks}w"):
        mlflow.log_params({
            "disease": config.disease,
            "task": config.task,
            "horizon_weeks": config.horizon_weeks,
            "n_features": len(config.feature_cols),
            "n_samples": len(df),
            "n_cv_folds": config.n_cv_folds,
        })

        # ── XGBoost tuning ────────────────────────────────────────────────
        logger.info("Tuning XGBoost (%d trials)...", config.n_trials_optuna)
        study_xgb = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study_xgb.optimize(
            lambda t: _xgb_objective(t, X, y, cv, config.task),
            n_trials=config.n_trials_optuna,
            show_progress_bar=False,
        )
        best_xgb_params = {**study_xgb.best_params, "random_state": RANDOM_STATE, "tree_method": "hist"}
        mlflow.log_params({f"xgb_{k}": v for k, v in best_xgb_params.items()})

        # ── LightGBM baseline ─────────────────────────────────────────────
        lgbm_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE,
            "verbose": -1,
        }

        # ── Final training on full data ───────────────────────────────────
        if config.task == "classification":
            xgb_model = CalibratedClassifierCV(
                XGBClassifier(**best_xgb_params, use_label_encoder=False),
                cv=3,
                method="sigmoid",
            )
            lgbm_model = CalibratedClassifierCV(
                LGBMClassifier(**lgbm_params),
                cv=3,
                method="sigmoid",
            )
        else:
            xgb_model = XGBRegressor(**best_xgb_params)
            lgbm_model = LGBMRegressor(**lgbm_params)

        xgb_model.fit(X, y)
        lgbm_model.fit(X, y)

        # ── Out-of-fold evaluation ────────────────────────────────────────
        oof_preds = np.zeros(len(y))
        for train_idx, val_idx in cv.split(X, y):
            if config.task == "classification":
                m = XGBClassifier(**best_xgb_params, use_label_encoder=False)
                m.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx] = m.predict_proba(X[val_idx])[:, 1]
            else:
                m = XGBRegressor(**best_xgb_params)
                m.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx] = m.predict(X[val_idx])

        metrics: dict[str, float] = {}
        if config.task == "classification":
            metrics["oof_auc_roc"] = float(roc_auc_score(y, oof_preds))
            metrics["oof_auc_pr"] = float(average_precision_score(y, oof_preds))
            metrics["oof_brier"] = float(brier_score_loss(y, oof_preds))
            logger.info(
                "OOF AUC-ROC=%.4f  AUC-PR=%.4f  Brier=%.4f",
                metrics["oof_auc_roc"], metrics["oof_auc_pr"], metrics["oof_brier"],
            )
        else:
            metrics["oof_mae"] = float(mean_absolute_error(y, oof_preds))
            logger.info("OOF MAE=%.4f", metrics["oof_mae"])

        mlflow.log_metrics(metrics)

        # ── SHAP values (XGBoost base estimator) ──────────────────────────
        base_xgb = xgb_model.estimator if hasattr(xgb_model, "estimator") else xgb_model
        explainer = shap.TreeExplainer(base_xgb)
        shap_values = explainer.shap_values(X[:500])  # sample for speed
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature": config.feature_cols,
            "shap_importance": mean_abs_shap,
        }).sort_values("shap_importance", ascending=False)

        mlflow.log_dict(
            feature_importance.head(30).to_dict("records"),
            "feature_importance_top30.json",
        )

        # ── Log models ────────────────────────────────────────────────────
        mlflow.sklearn.log_model(xgb_model, "xgb_model")
        mlflow.sklearn.log_model(lgbm_model, "lgbm_model")

    return {
        "xgb_model": xgb_model,
        "lgbm_model": lgbm_model,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "oof_predictions": oof_preds,
        "shap_values": shap_values,
    }


# ── Inference helper ──────────────────────────────────────────────────────────

def predict_risk(
    models: dict[str, Any],
    X: np.ndarray,
    ensemble_weights: tuple[float, float] = (0.6, 0.4),
) -> np.ndarray:
    """
    Produces ensemble predictions from XGBoost + LightGBM.
    ensemble_weights = (xgb_weight, lgbm_weight), must sum to 1.
    """
    w_xgb, w_lgbm = ensemble_weights
    xgb_pred = models["xgb_model"].predict_proba(X)[:, 1]
    lgbm_pred = models["lgbm_model"].predict_proba(X)[:, 1]
    return w_xgb * xgb_pred + w_lgbm * lgbm_pred
