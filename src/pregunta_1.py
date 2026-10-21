# -*- coding: utf-8 -*-
"""Pregunta 1. Árboles con Gini y entropía frente a la regresión logística.

Usa breast_cancer, la semilla 42 y la partición estratificada 80/20.
No modifica el README ni results_metrics.json. Regenera solo la figura de esta pregunta.

Desde la raíz del repositorio:

    python src/pregunta_1.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_experiments as exp
from sklearn.tree import DecisionTreeClassifier
from run_experiments import (
    PLOTS,
    SEED,
    candidate_alphas,
    conclusion_classifier,
    evaluate_classifier,
    log,
    make_scaled_logistic,
    plot_item1,
    run_grid,
    search_info,
)


def entrenar(x_train, y_train, x_test, y_test, cv) -> dict:
    alphas = {}
    searches = {}
    for criterion in ("gini", "entropy"):
        log(f"Busqueda de ccp_alpha para {criterion}")
        alpha_grid = candidate_alphas(
            DecisionTreeClassifier(criterion=criterion, random_state=SEED),
            x_train,
            y_train,
            max_alphas=40,
        )
        search, elapsed = run_grid(
            DecisionTreeClassifier(criterion=criterion, random_state=SEED),
            {"ccp_alpha": alpha_grid},
            x_train,
            y_train,
            cv,
            "roc_auc",
        )
        alphas[criterion] = float(search.best_params_["ccp_alpha"])
        searches[criterion] = search_info(
            search,
            elapsed,
            "roc_auc",
            f"ccp_alpha sobre la ruta de poda de complejidad de costo del arbol {criterion}, con {len(alpha_grid)} valores candidatos. Se maximiza el ROC-AUC de validacion cruzada.",
        )
        searches[criterion]["best_alpha"] = alphas[criterion]
    specs = [
        ("Gini", "individual", DecisionTreeClassifier(criterion="gini", random_state=SEED), {"criterion": "gini", "random_state": SEED}, None),
        ("Gini podado", "individual", DecisionTreeClassifier(criterion="gini", random_state=SEED, ccp_alpha=alphas["gini"]), {"criterion": "gini", "ccp_alpha": alphas["gini"], "random_state": SEED}, searches["gini"]),
        ("Entropía", "individual", DecisionTreeClassifier(criterion="entropy", random_state=SEED), {"criterion": "entropy", "random_state": SEED}, None),
        ("Entropía podada", "individual", DecisionTreeClassifier(criterion="entropy", random_state=SEED, ccp_alpha=alphas["entropy"]), {"criterion": "entropy", "ccp_alpha": alphas["entropy"], "random_state": SEED}, searches["entropy"]),
        ("Regresión logística", "individual", make_scaled_logistic(), {"scaler": "StandardScaler", "penalty": "L2", "l1_ratio": 0.0, "C": 1.0, "solver": "lbfgs", "max_iter": 10000, "random_state": SEED}, None),
    ]
    fitted = []
    records = []
    for name, role, estimator, params, search in specs:
        _model, record, proba = evaluate_classifier(name, estimator, x_train, y_train, x_test, y_test, cv, role, params, True)
        if search is not None:
            record["search"] = search
        records.append(record)
        fitted.append({"record": record, "proba": proba})
    plot_item1(PLOTS / "item1_breast_cancer.png", fitted, y_test)
    return {
        "question": "1. Entrene modelos de árboles de decisión con criterios de impureza diferentes (índice gini y entropía) usando el dataset 'breast_cancer' de scikit-learn, compare su eficiencia con respecto a un modelo lineal como el Regresor Logístico. ¿Cuál es mejor?",
        "models": records,
        "conclusion": conclusion_classifier(records),
        "pruning": {"gini": searches["gini"], "entropy": searches["entropy"]},
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
    print(f"Pregunta 1. Mejor modelo: {resultado['conclusion']['winner']}")


if __name__ == "__main__":
    main()
