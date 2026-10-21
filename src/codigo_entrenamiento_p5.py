# -*- coding: utf-8 -*-
"""Pregunta 5. Random Forest frente a XGBoost, en clasificación y en regresión.

Usa breast_cancer y auto_mpg, la semilla 42 y la misma partición de la corrida completa.
No modifica el README ni results_metrics.json. Regenera solo la figura de esta pregunta.

Desde la raíz del repositorio:

    python src/codigo_entrenamiento_p5.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_experiments as exp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from run_experiments import (
    PLOTS,
    SEED,
    attach_correlation,
    conclusion_classifier,
    conclusion_regressor,
    evaluate_classifier,
    evaluate_regressor,
    importances_from_model,
    make_preprocess,
    plot_item5,
    run_grid,
    search_info,
    to_builtin,
)


def entrenar(bc, mpg) -> dict:
    feature_names = bc["feature_names"]
    clf_records = []
    rf_base = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
    model, record, _proba = evaluate_classifier(
        "Random Forest base",
        rf_base,
        bc["X_train"],
        bc["y_train"],
        bc["X_test"],
        bc["y_test"],
        bc["cv"],
        "ensemble",
        {"n_estimators": 100, "max_features": "sqrt", "random_state": SEED},
        True,
    )
    record["importances"] = importances_from_model(model, feature_names)
    attach_correlation(record, model, bc["X_test"], "clf")
    clf_records.append(record)

    rf_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 4, 8, 12],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    rf_space = "n_estimators en {100, 200}, max_depth en {sin límite, 4, 8, 12}, min_samples_leaf en {1, 2, 4}, max_features en {sqrt, log2}."
    rf_search, rf_elapsed = run_grid(
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        rf_grid,
        bc["X_train"],
        bc["y_train"],
        bc["cv"],
        "roc_auc",
    )
    model, record, _proba = evaluate_classifier(
        "Random Forest optimizado",
        rf_search.best_estimator_,
        bc["X_train"],
        bc["y_train"],
        bc["X_test"],
        bc["y_test"],
        bc["cv"],
        "ensemble",
        to_builtin(rf_search.best_params_),
        True,
    )
    record["search"] = search_info(rf_search, rf_elapsed, "roc_auc", rf_space)
    record["importances"] = importances_from_model(model, feature_names)
    attach_correlation(record, model, bc["X_test"], "clf")
    clf_records.append(record)

    xgb_base = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,
        verbosity=0,
    )
    model, record, _proba = evaluate_classifier(
        "XGBoost base",
        xgb_base,
        bc["X_train"],
        bc["y_train"],
        bc["X_test"],
        bc["y_test"],
        bc["cv"],
        "ensemble",
        {"n_estimators": 100, "learning_rate": 0.1, "eval_metric": "logloss", "random_state": SEED},
        True,
    )
    record["importances"] = importances_from_model(model, feature_names)
    clf_records.append(record)

    xgb_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 6],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1.0, 5.0],
    }
    xgb_space = (
        "n_estimators en {100, 200}, learning_rate en {0.05, 0.1, 0.2}, max_depth en {2, 3, 4, 6}, "
        "subsample en {0.8, 1.0}, colsample_bytree en {0.8, 1.0}, reg_lambda en {1, 5}."
    )
    xgb_search, xgb_elapsed = run_grid(
        XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1, verbosity=0),
        xgb_grid,
        bc["X_train"],
        bc["y_train"],
        bc["cv"],
        "roc_auc",
    )
    model, record, _proba = evaluate_classifier(
        "XGBoost optimizado",
        xgb_search.best_estimator_,
        bc["X_train"],
        bc["y_train"],
        bc["X_test"],
        bc["y_test"],
        bc["cv"],
        "ensemble",
        to_builtin(xgb_search.best_params_),
        True,
    )
    record["search"] = search_info(xgb_search, xgb_elapsed, "roc_auc", xgb_space)
    record["importances"] = importances_from_model(model, feature_names)
    clf_records.append(record)

    reg_records = []
    rf_reg_base = Pipeline(
        [
            ("pre", make_preprocess()),
            ("model", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1)),
        ]
    )
    model, record = evaluate_regressor(
        "Random Forest base",
        rf_reg_base,
        mpg["X_train"],
        mpg["y_train"],
        mpg["X_test"],
        mpg["y_test"],
        mpg["cv"],
        "ensemble",
        {"n_estimators": 100, "max_features": 1.0, "random_state": SEED},
    )
    record["importances"] = importances_from_model(model)
    attach_correlation(record, model, mpg["X_test"], "reg")
    reg_records.append(record)

    rf_reg_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 4, 8, 12, 16],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [1.0, "sqrt", 0.5],
    }
    rf_reg_space = "n_estimators en {100, 200}, max_depth en {sin límite, 4, 8, 12, 16}, min_samples_leaf en {1, 2, 4}, max_features en {todas (1.0), sqrt, 0.5}."
    rf_reg_search, rf_reg_elapsed = run_grid(
        Pipeline(
            [
                ("pre", make_preprocess()),
                ("model", RandomForestRegressor(random_state=SEED, n_jobs=1)),
            ]
        ),
        rf_reg_grid,
        mpg["X_train"],
        mpg["y_train"],
        mpg["cv"],
        "neg_root_mean_squared_error",
    )
    model, record = evaluate_regressor(
        "Random Forest optimizado",
        rf_reg_search.best_estimator_,
        mpg["X_train"],
        mpg["y_train"],
        mpg["X_test"],
        mpg["y_test"],
        mpg["cv"],
        "ensemble",
        to_builtin(rf_reg_search.best_params_),
    )
    record["search"] = search_info(rf_reg_search, rf_reg_elapsed, "neg_root_mean_squared_error", rf_reg_space)
    record["importances"] = importances_from_model(model)
    attach_correlation(record, model, mpg["X_test"], "reg")
    reg_records.append(record)

    xgb_reg_base = Pipeline(
        [
            ("pre", make_preprocess()),
            ("model", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=SEED, n_jobs=1, verbosity=0)),
        ]
    )
    model, record = evaluate_regressor(
        "XGBoost base",
        xgb_reg_base,
        mpg["X_train"],
        mpg["y_train"],
        mpg["X_test"],
        mpg["y_test"],
        mpg["cv"],
        "ensemble",
        {"n_estimators": 100, "learning_rate": 0.1, "random_state": SEED},
    )
    record["importances"] = importances_from_model(model)
    reg_records.append(record)

    xgb_reg_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [2, 3, 4, 6],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__reg_lambda": [1.0, 5.0],
    }
    xgb_reg_search, xgb_reg_elapsed = run_grid(
        Pipeline(
            [
                ("pre", make_preprocess()),
                ("model", XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0)),
            ]
        ),
        xgb_reg_grid,
        mpg["X_train"],
        mpg["y_train"],
        mpg["cv"],
        "neg_root_mean_squared_error",
    )
    model, record = evaluate_regressor(
        "XGBoost optimizado",
        xgb_reg_search.best_estimator_,
        mpg["X_train"],
        mpg["y_train"],
        mpg["X_test"],
        mpg["y_test"],
        mpg["cv"],
        "ensemble",
        to_builtin(xgb_reg_search.best_params_),
    )
    record["search"] = search_info(xgb_reg_search, xgb_reg_elapsed, "neg_root_mean_squared_error", xgb_space)
    record["importances"] = importances_from_model(model)
    reg_records.append(record)

    plot_item5(PLOTS / "item5_rf_vs_xgboost.png", clf_records, reg_records)
    return {
        "question": "5. Entrene un modelo RandomForest y compare su eficiencia con respecto a un modelo XGBoost. ¿Cuál es mejor?",
        "classification": {
            "models": clf_records,
            "conclusion": conclusion_classifier(clf_records),
        },
        "regression": {
            "models": reg_records,
            "conclusion": conclusion_regressor(reg_records),
        },
    }

def main() -> None:
    os.chdir(exp.ROOT)
    exp.PLOTS.mkdir(parents=True, exist_ok=True)
    exp.DATA.mkdir(parents=True, exist_ok=True)
    exp.style_plots()
    cancer = exp.prepare_classification()
    mpg = exp.prepare_regression()
    resultado = entrenar(cancer, mpg)
    print(f"Pregunta 5. Mejor en clasificación: {resultado['classification']['conclusion']['winner']}")
    print(f"Pregunta 5. Mejor en regresión: {resultado['regression']['conclusion']['winner']}")


if __name__ == "__main__":
    main()
