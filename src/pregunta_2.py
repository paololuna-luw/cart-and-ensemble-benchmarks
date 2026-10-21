# -*- coding: utf-8 -*-
"""Pregunta 2. Árboles de regresión con varias configuraciones frente a la regresión lineal.

Usa auto_mpg limpio, la semilla 42 y la partición 80/20.
No modifica el README ni results_metrics.json. Regenera solo la figura de esta pregunta.

Desde la raíz del repositorio:

    python src/pregunta_2.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_experiments as exp
from sklearn.tree import DecisionTreeRegressor
from run_experiments import (
    PLOTS,
    SEED,
    candidate_alphas,
    conclusion_regressor,
    evaluate_regressor,
    importances_from_model,
    linear_coefficient_rows,
    log,
    make_mpg_linear,
    make_mpg_tree,
    make_preprocess,
    plot_item2,
    run_grid,
    search_info,
    to_builtin,
    validation_curve_depth,
)


def entrenar(x_train, y_train, x_test, y_test, cv) -> dict:
    fixed_specs = [
        (
            "Árbol sin restricciones",
            make_mpg_tree(max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="squared_error"),
            {"criterion": "squared_error", "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": SEED},
        ),
        (
            "Árbol superficial",
            make_mpg_tree(max_depth=2, criterion="squared_error"),
            {"criterion": "squared_error", "max_depth": 2, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": SEED},
        ),
        (
            "Árbol profundidad controlada",
            make_mpg_tree(max_depth=4, min_samples_split=5, min_samples_leaf=2, criterion="squared_error"),
            {"criterion": "squared_error", "max_depth": 4, "min_samples_split": 5, "min_samples_leaf": 2, "random_state": SEED},
        ),
        (
            "Árbol error absoluto",
            make_mpg_tree(criterion="absolute_error", max_depth=5, min_samples_leaf=5),
            {"criterion": "absolute_error", "max_depth": 5, "min_samples_leaf": 5, "min_samples_split": 2, "random_state": SEED},
        ),
    ]
    records = []
    for name, estimator, params in fixed_specs:
        _model, record = evaluate_regressor(name, estimator, x_train, y_train, x_test, y_test, cv, "individual", params)
        record["importances"] = importances_from_model(_model)
        records.append(record)

    log("Busqueda de ccp_alpha para el arbol de auto_mpg")
    pre = make_preprocess()
    transformed = pre.fit_transform(x_train, y_train)
    alpha_grid = candidate_alphas(DecisionTreeRegressor(random_state=SEED), transformed, y_train, max_alphas=35)
    ccp_pipe = make_mpg_tree()
    ccp_search, ccp_elapsed = run_grid(
        ccp_pipe,
        {"model__ccp_alpha": alpha_grid},
        x_train,
        y_train,
        cv,
        "neg_root_mean_squared_error",
    )
    ccp_alpha = float(ccp_search.best_params_["model__ccp_alpha"])
    ccp_info = search_info(
        ccp_search,
        ccp_elapsed,
        "neg_root_mean_squared_error",
        f"ccp_alpha sobre la ruta de poda, con {len(alpha_grid)} valores. Se minimiza el RMSE de validacion cruzada.",
    )
    ccp_info["best_alpha"] = ccp_alpha
    ccp_model, ccp_record = evaluate_regressor(
        "Árbol poda CCP",
        make_mpg_tree(ccp_alpha=ccp_alpha),
        x_train,
        y_train,
        x_test,
        y_test,
        cv,
        "individual",
        {"criterion": "squared_error", "ccp_alpha": ccp_alpha, "random_state": SEED},
    )
    ccp_record["search"] = ccp_info
    ccp_record["importances"] = importances_from_model(ccp_model)
    records.append(ccp_record)

    log("GridSearchCV del arbol de regresion")
    grid = [
        {
            "model__criterion": ["squared_error"],
            "model__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": [1.0, "sqrt", "log2"],
        },
        {
            "model__criterion": ["absolute_error"],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__min_samples_split": [2, 10],
            "model__min_samples_leaf": [2, 5, 8],
            "model__max_features": [1.0, "sqrt"],
        },
    ]
    space = (
        "Primera malla: criterion squared_error, max_depth en {2, 3, 4, 5, 6, 8, 10, sin límite}, "
        "min_samples_split en {2, 5, 10, 20}, min_samples_leaf en {1, 2, 4, 8}, max_features en {todas (1.0), sqrt, log2}. "
        "Segunda malla: criterion absolute_error, max_depth en {3, 4, 5, 6, 8}, min_samples_split en {2, 10}, "
        "min_samples_leaf en {2, 5, 8}, max_features en {todas (1.0), sqrt}. "
        "scikit-learn 1.9 admite squared_error, absolute_error y poisson en DecisionTreeRegressor; friedman_mse ya no es válido, así que el contraste de criterion usa squared_error y absolute_error."
    )
    grid_search, grid_elapsed = run_grid(
        make_mpg_tree(),
        grid,
        x_train,
        y_train,
        cv,
        "neg_root_mean_squared_error",
    )
    grid_info = search_info(grid_search, grid_elapsed, "neg_root_mean_squared_error", space)
    best_params = to_builtin(grid_search.best_params_)
    grid_model, grid_record = evaluate_regressor(
        "Árbol GridSearchCV",
        grid_search.best_estimator_,
        x_train,
        y_train,
        x_test,
        y_test,
        cv,
        "individual",
        best_params,
    )
    grid_record["search"] = grid_info
    grid_record["importances"] = importances_from_model(grid_model)
    records.append(grid_record)

    linear_model, linear_record = evaluate_regressor(
        "Regresión lineal",
        make_mpg_linear(),
        x_train,
        y_train,
        x_test,
        y_test,
        cv,
        "individual",
        {
            "preprocesado": "one-hot de origin con referencia europe",
            "scaler": "StandardScaler",
            "estimador": "LinearRegression",
        },
    )
    linear_record["coefficients"] = linear_coefficient_rows(linear_model)
    records.append(linear_record)
    curve = validation_curve_depth(x_train, y_train, x_test, y_test)
    plot_item2(
        PLOTS / "item2_auto_mpg.png",
        records,
        curve,
        linear_record["metrics"]["r2_test"],
        linear_record["metrics"]["rmse_test"],
    )
    return {
        "question": "2. Entrene modelos de árboles de decisión con varias configuraciones de hiper parámetros usando el dataset auto_mpg, comparar su eficiencia con respecto a un modelo lineal como el Regresor Lineal. ¿Cuál es el mejor?",
        "models": records,
        "validation_curve": curve,
        "conclusion": conclusion_regressor(records),
    }

def main() -> None:
    os.chdir(exp.ROOT)
    exp.PLOTS.mkdir(parents=True, exist_ok=True)
    exp.DATA.mkdir(parents=True, exist_ok=True)
    exp.style_plots()
    datos = exp.prepare_regression()
    resultado = entrenar(
        datos["X_train"], datos["y_train"], datos["X_test"], datos["y_test"], datos["cv"]
    )
    print(f"Pregunta 2. Mejor modelo: {resultado['conclusion']['winner']}")


if __name__ == "__main__":
    main()
