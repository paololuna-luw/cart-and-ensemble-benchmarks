# -*- coding: utf-8 -*-
"""Pregunta 3. Ensamble clasificador frente a modelos individuales.

Usa breast_cancer, la semilla 42 y la partición estratificada 80/20.
No modifica el README ni results_metrics.json. Regenera solo la figura de esta pregunta.

Desde la raíz del repositorio:

    python src/pregunta_3.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_experiments as exp
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from run_experiments import (
    PLOTS,
    SEED,
    conclusion_classifier,
    evaluate_classifier,
    individual_classifier_estimators,
    logistic_regression,
    make_cv_clf,
    make_scaled_logistic,
    plot_item3,
)


def entrenar(x_train, y_train, x_test, y_test, cv) -> dict:
    specs = [
        ("Árbol de decisión", "individual", DecisionTreeClassifier(random_state=SEED), {"criterion": "gini", "random_state": SEED}, True),
        ("SVC", "individual", individual_classifier_estimators()[1][1], {"kernel": "rbf", "probability": True, "scaler": "StandardScaler", "random_state": SEED}, True),
        ("k-NN", "individual", individual_classifier_estimators()[2][1], {"n_neighbors": 5, "scaler": "StandardScaler"}, True),
        ("Regresión logística", "individual", make_scaled_logistic(), {"penalty": "L2", "l1_ratio": 0.0, "C": 1.0, "max_iter": 10000, "scaler": "StandardScaler", "random_state": SEED}, True),
        ("Voting duro", "ensemble", VotingClassifier(estimators=individual_classifier_estimators(), voting="hard"), {"voting": "hard", "bases": "arbol, SVC, k-NN, regresion logistica"}, False),
        ("Voting suave", "ensemble", VotingClassifier(estimators=individual_classifier_estimators(), voting="soft"), {"voting": "soft", "bases": "arbol, SVC, k-NN, regresion logistica"}, True),
        (
            "Stacking",
            "ensemble",
            StackingClassifier(
                estimators=individual_classifier_estimators(),
                final_estimator=logistic_regression(),
                cv=make_cv_clf(),
                stack_method="auto",
                n_jobs=1,
                passthrough=False,
            ),
            {"final_estimator": "regresion logistica L2", "cv": "StratifiedKFold de 5 pliegues", "passthrough": False},
            True,
        ),
        (
            "AdaBoost",
            "ensemble",
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
                n_estimators=50,
                learning_rate=1.0,
                random_state=SEED,
            ),
            {"n_estimators": 50, "learning_rate": 1.0, "estimator_max_depth": 1, "random_state": SEED},
            True,
        ),
        (
            "Gradient Boosting",
            "ensemble",
            GradientBoostingClassifier(random_state=SEED),
            {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": SEED},
            True,
        ),
    ]
    records = []
    for name, role, estimator, params, needs_proba in specs:
        _model, record, _proba = evaluate_classifier(
            name, estimator, x_train, y_train, x_test, y_test, cv, role, params, needs_proba
        )
        records.append(record)
    plot_item3(PLOTS / "item3_ensemble_clf.png", records)
    individuals = [record for record in records if record["role"] == "individual"]
    ensembles = [record for record in records if record["role"] == "ensemble" and record["metrics"]["roc_auc_test"] is not None]
    return {
        "question": "3. Entrene un modelo ensemble clasificador, y compare su eficiencia con respecto a un modelo individual. ¿Cuál es mejor?",
        "models": records,
        "conclusion": conclusion_classifier(records),
        "best_individual": conclusion_classifier(individuals),
        "best_ensemble": conclusion_classifier(ensembles),
    }

def main() -> None:
    os.chdir(exp.ROOT)
    exp.PLOTS.mkdir(parents=True, exist_ok=True)
    exp.DATA.mkdir(parents=True, exist_ok=True)
    exp.style_plots()
    datos = exp.prepare_classification()
    resultado = entrenar(
        datos["X_train"], datos["y_train"], datos["X_test"], datos["y_test"], datos["cv"]
    )
    print(f"Pregunta 3. Mejor modelo: {resultado['conclusion']['winner']}")


if __name__ == "__main__":
    main()
