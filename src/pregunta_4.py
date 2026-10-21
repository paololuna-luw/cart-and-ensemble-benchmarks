# -*- coding: utf-8 -*-
"""Pregunta 4. Bagging clasificador y regresor frente al árbol individual.

Usa breast_cancer y auto_mpg, la semilla 42 y la misma partición de la corrida completa.
No modifica el README ni results_metrics.json. Regenera solo la figura de esta pregunta.

Desde la raíz del repositorio:

    python src/pregunta_4.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_experiments as exp
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from run_experiments import (
    PLOTS,
    SEED,
    attach_correlation,
    evaluate_classifier,
    evaluate_regressor,
    make_mpg_tree,
    make_preprocess,
    plot_item4,
)


def entrenar(bc, mpg) -> dict:
    clf_records = []
    tree, tree_record, _proba = evaluate_classifier(
        "Árbol individual",
        DecisionTreeClassifier(random_state=SEED),
        bc["X_train"],
        bc["y_train"],
        bc["X_test"],
        bc["y_test"],
        bc["cv"],
        "individual",
        {"criterion": "gini", "random_state": SEED},
        True,
    )
    tree_record["n_estimators"] = 1
    clf_records.append(tree_record)
    for n_estimators in (10, 50, 100, 200):
        estimator = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=SEED),
            n_estimators=n_estimators,
            max_samples=1.0,
            bootstrap=True,
            oob_score=True,
            random_state=SEED,
            n_jobs=1,
        )
        model, record, _proba = evaluate_classifier(
            f"Bagging ({n_estimators})",
            estimator,
            bc["X_train"],
            bc["y_train"],
            bc["X_test"],
            bc["y_test"],
            bc["cv"],
            "ensemble",
            {
                "n_estimators": n_estimators,
                "max_samples": 1.0,
                "bootstrap": True,
                "oob_score": True,
                "base": "DecisionTreeClassifier",
                "random_state": SEED,
            },
            True,
        )
        record["n_estimators"] = n_estimators
        attach_correlation(record, model, bc["X_test"], "clf")
        clf_records.append(record)

    reg_records = []
    _model, reg_tree = evaluate_regressor(
        "Árbol individual",
        make_mpg_tree(max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="squared_error"),
        mpg["X_train"],
        mpg["y_train"],
        mpg["X_test"],
        mpg["y_test"],
        mpg["cv"],
        "individual",
        {"criterion": "squared_error", "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": SEED},
    )
    reg_tree["n_estimators"] = 1
    reg_records.append(reg_tree)
    for n_estimators in (10, 50, 100, 200):
        estimator = Pipeline(
            [
                ("pre", make_preprocess()),
                (
                    "model",
                    BaggingRegressor(
                        estimator=DecisionTreeRegressor(random_state=SEED),
                        n_estimators=n_estimators,
                        max_samples=1.0,
                        bootstrap=True,
                        oob_score=True,
                        random_state=SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        model, record = evaluate_regressor(
            f"Bagging ({n_estimators})",
            estimator,
            mpg["X_train"],
            mpg["y_train"],
            mpg["X_test"],
            mpg["y_test"],
            mpg["cv"],
            "ensemble",
            {
                "n_estimators": n_estimators,
                "max_samples": 1.0,
                "bootstrap": True,
                "oob_score": True,
                "base": "DecisionTreeRegressor",
                "random_state": SEED,
            },
        )
        record["n_estimators"] = n_estimators
        attach_correlation(record, model, mpg["X_test"], "reg")
        reg_records.append(record)
    plot_item4(PLOTS / "item4_bagging.png", clf_records, reg_records)
    bag_clf = [record for record in clf_records if record["role"] == "ensemble"]
    bag_reg = [record for record in reg_records if record["role"] == "ensemble"]
    best_clf = max(bag_clf, key=lambda record: (record["metrics"]["accuracy_test"], record["metrics"]["f1_test"], record["metrics"]["roc_auc_test"]))
    best_reg = min(bag_reg, key=lambda record: (record["metrics"]["rmse_test"], -record["metrics"]["r2_test"]))
    return {
        "question": "4. Entrene un modelo bagging regresor o clasificador y compare su eficiencia con respecto a un modelo individual. ¿Cuál es el mejor?",
        "classification": {
            "models": clf_records,
            "best_bagging": best_clf["name"],
            "individual": "Árbol individual",
        },
        "regression": {
            "models": reg_records,
            "best_bagging": best_reg["name"],
            "individual": "Árbol individual",
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
    print(f"Pregunta 4. Mejor bagging en clasificación: {resultado['classification']['best_bagging']}")
    print(f"Pregunta 4. Mejor bagging en regresión: {resultado['regression']['best_bagging']}")


if __name__ == "__main__":
    main()
