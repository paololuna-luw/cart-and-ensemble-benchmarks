# -*- coding: utf-8 -*-
"""Tarea 03: CART y ensamblados.

Entrena los modelos, guarda metricas y figuras, escribe el README y ejecuta el notebook.
Todo el azar queda fijo con random_state=42.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import warnings
from importlib.metadata import version as pkg_version
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import seaborn as sns
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from scipy.stats import ttest_rel
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5
N_JOBS = -1

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
DATA = ROOT / "data"

NUMERIC_FEATURES = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
]
ORIGIN_MAP = {
    "1": "usa",
    "2": "europe",
    "3": "japan",
    1: "usa",
    2: "europe",
    3: "japan",
    "usa": "usa",
    "europe": "europe",
    "japan": "japan",
    "america": "usa",
    "american": "usa",
}

MATH_BAGGING = r"""
$$
\operatorname{Var}(\bar{X})=\rho\sigma^{2}+\frac{(1-\rho)\sigma^{2}}{B}
$$

Cuando los árboles son independientes, \(\rho = 0\) y la expresión se reduce a \(\sigma^{2}/B\).
"""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fmt(value, nd: int = 4) -> str:
    if value is None:
        return "no aplica"
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Metrica no finita al formatear: {value}")
    return f"{number:.{nd}f}"


def fmt_pm(mean, std, nd: int = 4) -> str:
    return f"{fmt(mean, nd)} ± {fmt(std, nd)}"


def fmt_alpha(value) -> str:
    number = float(value)
    if number == 0.0:
        return "0"
    if abs(number) < 1e-3:
        return f"{number:.6e}"
    return f"{number:.6g}"


def fmt_param_value(key: str, value) -> str:
    if value is None:
        if "depth" in key or "leaf_nodes" in key:
            return "sin límite"
        if "feature" in key:
            return "todas las variables"
        return "omisión"
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, float):
        if "alpha" in key or "lambda" in key or "rate" in key:
            return fmt_alpha(value)
        return f"{value:.6g}"
    return str(value)


def fmt_params(params: dict) -> str:
    if not params:
        return "configuración por omisión del estimador"
    parts = []
    for key, value in params.items():
        short = str(key).split("__")[-1]
        parts.append(f"{short}={fmt_param_value(short, value)}")
    return ", ".join(parts)


def fmt_p(p_value: float) -> str:
    if p_value < 0.0001:
        return "< 0.0001"
    return f"{p_value:.4f}"


def pp(a, b) -> float:
    return (float(a) - float(b)) * 100.0


def rel_drop(before, after) -> float:
    base = float(before)
    if base == 0.0:
        raise RuntimeError("No se puede calcular una reduccion relativa con base 0")
    return (base - float(after)) / base * 100.0


def md_table(headers: list[str], rows: list[list]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        cells = [str(cell).replace("|", "/") for cell in row]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(key): to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return to_builtin(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        if not math.isfinite(value):
            raise RuntimeError("Aparecio un float de numpy no finito")
        return value
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise RuntimeError("Aparecio un float no finito")
        return obj
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    raise TypeError(f"Objeto no serializable: {type(obj)!r}")


def assert_finite_tree(obj, path: str = "root") -> None:
    if obj is None or isinstance(obj, (str, bool, int)):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise RuntimeError(f"Valor no finito en {path}")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert_finite_tree(value, f"{path}.{key}")
        return
    if isinstance(obj, list):
        for index, value in enumerate(obj):
            assert_finite_tree(value, f"{path}[{index}]")
        return
    raise TypeError(f"Tipo inesperado en {path}: {type(obj)}")


def make_cv_clf() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def make_cv_reg() -> KFold:
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def make_preprocess() -> ColumnTransformer:
    encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore",
    )
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", encoder, ["origin"]),
        ],
        remainder="drop",
    )


def make_mpg_tree(**tree_params) -> Pipeline:
    params = {"random_state": SEED}
    params.update(tree_params)
    return Pipeline(
        [
            ("pre", make_preprocess()),
            ("model", DecisionTreeRegressor(**params)),
        ]
    )


def make_mpg_linear() -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocess()),
            ("scaler", StandardScaler()),
            ("linreg", LinearRegression()),
        ]
    )


def logistic_regression() -> LogisticRegression:
    # l1_ratio=0 es la penalizacion L2 en scikit-learn 1.9 (C=1, solver lbfgs).
    return LogisticRegression(
        max_iter=10000,
        random_state=SEED,
        solver="lbfgs",
        l1_ratio=0.0,
        C=1.0,
    )


def make_scaled_logistic() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", logistic_regression()),
        ]
    )


def clean_feature_name(name: str) -> str:
    text = str(name)
    if "__" in text:
        text = text.split("__", 1)[1]
    return text.replace("origin_", "origin=")


def timed_fit(estimator, x_train, y_train):
    model = clone(estimator)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return model, float(elapsed_ms)


def cross_validate_safe(estimator, x_train, y_train, cv, scoring):
    try:
        return cross_validate(
            estimator,
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=N_JOBS,
            error_score="raise",
        )
    except Exception as exc:
        log(f"La validacion cruzada en paralelo fallo ({type(exc).__name__}). Reintento en serie.")
        return cross_validate(
            estimator,
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )


def fold_stats(values) -> tuple[float, float, list[float]]:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise RuntimeError("La validacion cruzada devolvio un pliegue no finito")
    return float(array.mean()), float(array.std(ddof=0)), [float(v) for v in array]


def single_tree(model):
    if hasattr(model, "get_depth") and hasattr(model, "get_n_leaves"):
        return model
    if isinstance(model, Pipeline):
        for step in model.named_steps.values():
            if hasattr(step, "get_depth") and hasattr(step, "get_n_leaves"):
                return step
    return None


def positive_proba(model, x_values) -> np.ndarray:
    proba = model.predict_proba(x_values)
    classes = [int(c) for c in list(model.classes_)]
    if 1 not in classes:
        return np.zeros(len(x_values), dtype=float)
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def extract_oob(model):
    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if not hasattr(estimator, "oob_score_"):
        return None
    value = float(estimator.oob_score_)
    if not math.isfinite(value):
        return None
    return value


def candidate_alphas(estimator, x_train, y_train, max_alphas: int = 35) -> list[float]:
    path = clone(estimator).cost_complexity_pruning_path(x_train, np.asarray(y_train))
    alphas = np.asarray(path.ccp_alphas, dtype=float)
    if len(alphas) > 1:
        alphas = alphas[:-1]
    alphas = np.unique(np.round(alphas, decimals=10))
    alphas = alphas[np.isfinite(alphas) & (alphas >= 0)]
    if len(alphas) == 0:
        return [0.0]
    if len(alphas) > max_alphas:
        index = np.unique(np.linspace(0, len(alphas) - 1, max_alphas).astype(int))
        alphas = alphas[index]
    return [float(alpha) for alpha in alphas]


def run_grid(estimator, param_grid, x_train, y_train, cv, scoring: str):
    search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=N_JOBS,
        refit=True,
        error_score="raise",
    )
    started = time.perf_counter()
    try:
        search.fit(x_train, y_train)
    except Exception as exc:
        if type(exc).__name__ == "InvalidParameterError":
            raise
        log(f"GridSearchCV en paralelo fallo ({type(exc).__name__}). Reintento en serie.")
        search.set_params(n_jobs=1)
        search.fit(x_train, y_train)
    elapsed = float(time.perf_counter() - started)
    return search, elapsed


def evaluate_classifier(
    name,
    estimator,
    x_train,
    y_train,
    x_test,
    y_test,
    cv,
    role: str,
    params: dict,
    needs_proba: bool = True,
):
    log(f"Clasificador: {name}")
    model, fit_ms = timed_fit(estimator, x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    proba_test = None
    roc_test = None
    roc_train = None
    if needs_proba:
        proba_test = positive_proba(model, x_test)
        proba_train = positive_proba(model, x_train)
        roc_test = float(roc_auc_score(y_test, proba_test))
        roc_train = float(roc_auc_score(y_train, proba_train))
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
    }
    if needs_proba:
        scoring["roc_auc"] = "roc_auc"
    cv_result = cross_validate_safe(clone(estimator), x_train, y_train, cv, scoring)
    acc_mean, acc_std, acc_folds = fold_stats(cv_result["test_accuracy"])
    f1_mean, f1_std, f1_folds = fold_stats(cv_result["test_f1"])
    bal_mean, bal_std, _bal_folds = fold_stats(cv_result["test_balanced_accuracy"])
    roc_mean = roc_std = None
    roc_folds = None
    if needs_proba:
        roc_mean, roc_std, roc_folds = fold_stats(cv_result["test_roc_auc"])
    tree = single_tree(model)
    cm = confusion_matrix(y_test, pred_test, labels=[0, 1]).astype(int).tolist()
    metrics = {
        "accuracy_train": float(accuracy_score(y_train, pred_train)),
        "accuracy_test": float(accuracy_score(y_test, pred_test)),
        "precision_test": float(precision_score(y_test, pred_test, zero_division=0)),
        "recall_test": float(recall_score(y_test, pred_test, zero_division=0)),
        "f1_train": float(f1_score(y_train, pred_train, zero_division=0)),
        "f1_test": float(f1_score(y_test, pred_test, zero_division=0)),
        "balanced_accuracy_test": float(balanced_accuracy_score(y_test, pred_test)),
        "recall_malignant_test": float(recall_score(y_test, pred_test, pos_label=0, zero_division=0)),
        "roc_auc_train": roc_train,
        "roc_auc_test": roc_test,
        "cv_accuracy_mean": acc_mean,
        "cv_accuracy_std": acc_std,
        "cv_accuracy_folds": acc_folds,
        "cv_f1_mean": f1_mean,
        "cv_f1_std": f1_std,
        "cv_f1_folds": f1_folds,
        "cv_balanced_accuracy_mean": bal_mean,
        "cv_balanced_accuracy_std": bal_std,
        "cv_roc_auc_mean": roc_mean,
        "cv_roc_auc_std": roc_std,
        "cv_roc_auc_folds": roc_folds,
        "depth": None if tree is None else int(tree.get_depth()),
        "n_leaves": None if tree is None else int(tree.get_n_leaves()),
        "fit_time_ms": fit_ms,
        "confusion_matrix_test": cm,
        "oob_score": extract_oob(model),
    }
    record = {
        "name": name,
        "role": role,
        "params": params,
        "params_text": fmt_params(params),
        "metrics": metrics,
    }
    return model, record, proba_test


def evaluate_regressor(name, estimator, x_train, y_train, x_test, y_test, cv, role: str, params: dict):
    log(f"Regresor: {name}")
    model, fit_ms = timed_fit(estimator, x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }
    cv_result = cross_validate_safe(clone(estimator), x_train, y_train, cv, scoring)
    r2_mean, r2_std, r2_folds = fold_stats(cv_result["test_r2"])
    rmse_mean, rmse_std, rmse_folds = fold_stats(-np.asarray(cv_result["test_rmse"], dtype=float))
    mae_mean, mae_std, mae_folds = fold_stats(-np.asarray(cv_result["test_mae"], dtype=float))
    tree = single_tree(model)
    metrics = {
        "r2_train": float(r2_score(y_train, pred_train)),
        "r2_test": float(r2_score(y_test, pred_test)),
        "rmse_train": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "mae_train": float(mean_absolute_error(y_train, pred_train)),
        "mse_test": float(mean_squared_error(y_test, pred_test)),
        "cv_r2_mean": r2_mean,
        "cv_r2_std": r2_std,
        "cv_r2_folds": r2_folds,
        "cv_rmse_mean": rmse_mean,
        "cv_rmse_std": rmse_std,
        "cv_rmse_folds": rmse_folds,
        "cv_mae_mean": mae_mean,
        "cv_mae_std": mae_std,
        "cv_mae_folds": mae_folds,
        "depth": None if tree is None else int(tree.get_depth()),
        "n_leaves": None if tree is None else int(tree.get_n_leaves()),
        "fit_time_ms": fit_ms,
        "oob_score": extract_oob(model),
    }
    record = {
        "name": name,
        "role": role,
        "params": params,
        "params_text": fmt_params(params),
        "metrics": metrics,
    }
    return model, record


def importance_rows(names, values) -> dict:
    raw_values = np.asarray(values, dtype=float)
    total = float(raw_values.sum())
    relative = raw_values / total if total > 0 else raw_values
    raw = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(zip(names, raw_values), key=lambda item: item[1], reverse=True)
    ]
    rel = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(zip(names, relative), key=lambda item: item[1], reverse=True)
    ]
    return {"raw": raw, "relative": rel}


def importances_from_model(model, feature_names=None):
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
        if "pre" in model.named_steps:
            names = [clean_feature_name(name) for name in model.named_steps["pre"].get_feature_names_out()]
        else:
            names = feature_names
    else:
        estimator = model
        names = feature_names
    if not hasattr(estimator, "feature_importances_"):
        return None
    if names is None:
        names = [f"variable_{index}" for index in range(len(estimator.feature_importances_))]
    return importance_rows(list(names), estimator.feature_importances_)


def linear_coefficient_rows(model) -> dict:
    names = [clean_feature_name(name) for name in model.named_steps["pre"].get_feature_names_out()]
    coefficients = np.asarray(model.named_steps["linreg"].coef_, dtype=float)
    rows = [
        {"feature": name, "coefficient": float(coef)}
        for name, coef in zip(names, coefficients)
    ]
    rows.sort(key=lambda row: abs(row["coefficient"]), reverse=True)
    return {
        "intercept": float(model.named_steps["linreg"].intercept_),
        "coefficients": rows,
        "scale": "coeficientes sobre variables estandarizadas despues del one-hot",
    }


def member_prediction_matrix(model, x_values, task: str):
    if isinstance(model, Pipeline):
        last = model.steps[-1][1]
        if hasattr(last, "estimators_"):
            transformed = x_values
            for _name, step in model.steps[:-1]:
                transformed = step.transform(transformed)
            members = last.estimators_
            matrix_x = transformed
        else:
            return None
    elif hasattr(model, "estimators_"):
        members = model.estimators_
        matrix_x = x_values
    else:
        return None
    columns = []
    for member in members:
        if task == "clf":
            columns.append(positive_proba(member, matrix_x))
        else:
            columns.append(np.asarray(member.predict(matrix_x), dtype=float))
    return np.column_stack(columns)


def mean_pairwise_correlation(matrix) -> float | None:
    if matrix is None or matrix.shape[1] < 3:
        return None
    corr = np.corrcoef(matrix, rowvar=False)
    upper = corr[np.triu_indices(corr.shape[0], k=1)]
    if not np.any(np.isfinite(upper)):
        return None
    value = float(np.nanmean(upper))
    if not math.isfinite(value):
        return None
    return value


def attach_correlation(record, model, x_values, task: str) -> None:
    matrix = member_prediction_matrix(model, x_values, task)
    rho = mean_pairwise_correlation(matrix)
    if rho is None or matrix is None:
        return
    n_members = int(matrix.shape[1])
    factor = float(rho + (1.0 - rho) / n_members)
    record["ensemble_diagnostics"] = {
        "n_members": n_members,
        "mean_pairwise_prediction_correlation": rho,
        "equicorrelation_variance_factor": factor,
        "independence_variance_factor": float(1.0 / n_members),
    }


def rank_classifiers(models: list[dict]) -> list[dict]:
    ranked = [model for model in models if model["metrics"]["roc_auc_test"] is not None]
    # 10 decimales: un empate de 1e-16 no debe ganarle a un F1 mejor.
    ranked.sort(
        key=lambda model: (
            -round(float(model["metrics"]["roc_auc_test"]), 10),
            -model["metrics"]["f1_test"],
            -model["metrics"]["accuracy_test"],
            -model["metrics"]["cv_accuracy_mean"],
        )
    )
    return ranked


def rank_regressors(models: list[dict]) -> list[dict]:
    ranked = list(models)
    ranked.sort(
        key=lambda model: (
            model["metrics"]["rmse_test"],
            -model["metrics"]["r2_test"],
            -model["metrics"]["cv_r2_mean"],
        )
    )
    return ranked


def conclusion_classifier(models: list[dict], primary_metric: str = "roc_auc_test") -> dict:
    ranked = rank_classifiers(models)
    winner = ranked[0]
    runner = ranked[1]
    cv_ranked = sorted(ranked, key=lambda model: -model["metrics"]["cv_roc_auc_mean"])
    acc_pool = list(models)
    acc_pool.sort(
        key=lambda model: (
            -model["metrics"]["accuracy_test"],
            -model["metrics"]["f1_test"],
        )
    )
    return {
        "winner": winner["name"],
        "runner_up": runner["name"],
        "primary_metric": primary_metric,
        "primary_value": winner["metrics"][primary_metric],
        "runner_up_value": runner["metrics"][primary_metric],
        "cv_leader": cv_ranked[0]["name"],
        "cv_agrees": cv_ranked[0]["name"] == winner["name"],
        "accuracy_leader": acc_pool[0]["name"],
        "accuracy_leader_value": acc_pool[0]["metrics"]["accuracy_test"],
    }


def conclusion_regressor(models: list[dict]) -> dict:
    ranked = rank_regressors(models)
    winner = ranked[0]
    runner = ranked[1]
    cv_ranked = sorted(models, key=lambda model: -model["metrics"]["cv_r2_mean"])
    return {
        "winner": winner["name"],
        "runner_up": runner["name"],
        "primary_metric": "rmse_test",
        "primary_value": winner["metrics"]["rmse_test"],
        "runner_up_value": runner["metrics"]["rmse_test"],
        "cv_leader": cv_ranked[0]["name"],
        "cv_agrees": cv_ranked[0]["name"] == winner["name"],
        "winner_r2_test": winner["metrics"]["r2_test"],
        "runner_up_r2_test": runner["metrics"]["r2_test"],
    }


def t_test_sentence(model_a, model_b, fold_key: str, metric_label: str) -> str:
    folds_a = model_a["metrics"].get(fold_key)
    folds_b = model_b["metrics"].get(fold_key)
    if not folds_a or not folds_b:
        return f"No hay pliegues emparejados de {metric_label} para contrastar {model_a['name']} y {model_b['name']}."
    array_a = np.asarray(folds_a, dtype=float)
    array_b = np.asarray(folds_b, dtype=float)
    diff = array_a - array_b
    if np.allclose(diff, 0.0):
        return (
            f"En {metric_label}, los {len(array_a)} pliegues de {model_a['name']} y "
            f"{model_b['name']} coinciden."
        )
    if float(np.std(diff, ddof=0)) == 0.0:
        return (
            f"En {metric_label}, la diferencia media entre {model_a['name']} y {model_b['name']} "
            f"es {float(diff.mean()):.4f} y no cambia entre pliegues."
        )
    statistic, p_value = ttest_rel(array_a, array_b)
    if not math.isfinite(float(statistic)) or not math.isfinite(float(p_value)):
        return f"La prueba t pareada de {metric_label} no quedó definida."
    favored = model_a["name"] if float(diff.mean()) > 0 else model_b["name"]
    return (
        f"Prueba t pareada sobre los {len(array_a)} pliegues de {metric_label}: "
        f"la media de {model_a['name']} menos {model_b['name']} es {float(diff.mean()):.4f}, "
        f"t = {float(statistic):.3f}, p = {fmt_p(float(p_value))}. "
        f"El promedio de pliegues favorece a {favored}. "
        f"Con {len(array_a)} pliegues la potencia es limitada; el valor p es un apoyo descriptivo, "
        f"y el conjunto de prueba sigue siendo el criterio de decisión."
    )


def find_model(models, name: str) -> dict:
    for model in models:
        if model["name"] == name:
            return model
    raise KeyError(name)


def errors_from_cm(metrics: dict) -> tuple[int, int]:
    matrix = metrics["confusion_matrix_test"]
    errors = int(matrix[0][1] + matrix[1][0])
    total = int(sum(sum(row) for row in matrix))
    return errors, total


def style_plots() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white", pad_inches=0.35)
    plt.close(fig)


def grouped_bars(ax, labels, series, ylabel, title, ylim=None):
    palette = sns.color_palette("colorblind", max(len(series), 3))
    x_pos = np.arange(len(labels))
    width = 0.80 / max(len(series), 1)
    for index, (name, values) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        ax.bar(x_pos + offset, values, width=width * 0.92, label=name, color=palette[index])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, frameon=True)


def dynamic_ylim(values, floor=None, ceiling=None) -> tuple[float, float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    low = min(finite)
    high = max(finite)
    pad = max(0.015, (high - low) * 0.35, abs(high) * 0.05 + 0.01)
    y0 = low - pad
    y1 = high + pad
    if floor is not None:
        y0 = max(floor, y0)
    if ceiling is not None:
        y1 = min(ceiling, y1)
    if y1 <= y0:
        y1 = y0 + pad
    return y0, y1


def plot_item1(path, fitted, y_test) -> None:
    metric_specs = [
        ("accuracy_test", "Accuracy"),
        ("precision_test", "Precision"),
        ("recall_test", "Recall"),
        ("f1_test", "F1"),
        ("roc_auc_test", "ROC-AUC"),
    ]
    fig = plt.figure(figsize=(16.5, 13.5), layout="constrained")
    grid = fig.add_gridspec(3, len(fitted), height_ratios=[1.25, 1.0, 1.15])
    ax_bars = fig.add_subplot(grid[0, :])
    labels = [spec[1] for spec in metric_specs]
    series = []
    flat_values = []
    for item in fitted:
        values = [item["record"]["metrics"][key] for key, _label in metric_specs]
        series.append((item["record"]["name"], values))
        flat_values.extend(values)
    grouped_bars(
        ax_bars,
        labels,
        series,
        "Valor en prueba",
        "Metricas de prueba: arboles CART y regresion logistica",
        ylim=dynamic_ylim(flat_values, floor=0.5, ceiling=1.0),
    )
    for index, item in enumerate(fitted):
        ax_cm = fig.add_subplot(grid[1, index])
        matrix = np.asarray(item["record"]["metrics"]["confusion_matrix_test"], dtype=int)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax_cm,
            xticklabels=["Maligno", "Benigno"],
            yticklabels=["Maligno", "Benigno"] if index == 0 else False,
            annot_kws={"size": 9},
        )
        ax_cm.set_title(item["record"]["name"], fontsize=10)
        ax_cm.set_xlabel("Prediccion")
        if index == 0:
            ax_cm.set_ylabel("Clase real")
    ax_roc = fig.add_subplot(grid[2, :])
    palette = sns.color_palette("colorblind", len(fitted))
    for index, item in enumerate(fitted):
        fpr, tpr, _thresholds = roc_curve(y_test, item["proba"])
        auc = item["record"]["metrics"]["roc_auc_test"]
        ax_roc.plot(fpr, tpr, color=palette[index], lw=2, label=f"{item['record']['name']} (AUC {fmt(auc)})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="0.5", lw=1, label="Clasificador sin discriminacion")
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Tasa de verdaderos positivos")
    ax_roc.set_title("Curvas ROC en el conjunto de prueba")
    ax_roc.legend(fontsize=8, loc="lower right")
    fig.suptitle("Item 1. Cancer de mama: Gini, entropia y regresion logistica", fontsize=15)
    save_figure(fig, path)


def plot_item2(path, records, curve, linear_r2, linear_rmse) -> None:
    fig = plt.figure(figsize=(16.5, 12.5), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    names = [record["name"] for record in records]
    ax_r2 = fig.add_subplot(grid[0, 0])
    grouped_bars(
        ax_r2,
        names,
        [
            ("R2 entrenamiento", [record["metrics"]["r2_train"] for record in records]),
            ("R2 prueba", [record["metrics"]["r2_test"] for record in records]),
        ],
        "R2",
        "R2 de entrenamiento y de prueba",
        ylim=dynamic_ylim(
            [record["metrics"]["r2_train"] for record in records]
            + [record["metrics"]["r2_test"] for record in records],
            floor=0.0,
            ceiling=1.05,
        ),
    )
    ax_rmse = fig.add_subplot(grid[0, 1])
    grouped_bars(
        ax_rmse,
        names,
        [("RMSE de prueba", [record["metrics"]["rmse_test"] for record in records])],
        "RMSE (mpg)",
        "RMSE de prueba",
    )
    depths = curve["max_depth"]
    ax_curve_r2 = fig.add_subplot(grid[1, 0])
    ax_curve_r2.plot(depths, curve["r2_train"], marker="o", label="R2 entrenamiento")
    ax_curve_r2.plot(depths, curve["r2_test"], marker="o", label="R2 prueba")
    ax_curve_r2.axhline(linear_r2, color="0.3", linestyle="--", label=f"R2 prueba de la regresion lineal ({fmt(linear_r2)})")
    ax_curve_r2.axvline(curve["best_test_depth"], color="C2", linestyle=":", label=f"Mejor profundidad de prueba ({curve['best_test_depth']})")
    ax_curve_r2.set_xlabel("max_depth")
    ax_curve_r2.set_ylabel("R2")
    ax_curve_r2.set_title("Curva de validacion de max_depth (R2)")
    ax_curve_r2.legend(fontsize=8)
    ax_curve_rmse = fig.add_subplot(grid[1, 1])
    ax_curve_rmse.plot(depths, curve["rmse_train"], marker="o", label="RMSE entrenamiento")
    ax_curve_rmse.plot(depths, curve["rmse_test"], marker="o", label="RMSE prueba")
    ax_curve_rmse.axhline(linear_rmse, color="0.3", linestyle="--", label=f"RMSE prueba lineal ({fmt(linear_rmse)})")
    ax_curve_rmse.axvline(curve["best_test_depth"], color="C2", linestyle=":", label=f"Mejor profundidad de prueba ({curve['best_test_depth']})")
    ax_curve_rmse.set_xlabel("max_depth")
    ax_curve_rmse.set_ylabel("RMSE (mpg)")
    ax_curve_rmse.set_title("Curva de validacion de max_depth (RMSE)")
    ax_curve_rmse.legend(fontsize=8)
    fig.suptitle("Item 2. Auto MPG: configuraciones de arbol y regresion lineal", fontsize=15)
    save_figure(fig, path)


def plot_item3(path, records) -> None:
    fig = plt.figure(figsize=(16.5, 11.5), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    names = [record["name"] for record in records]
    ax_acc = fig.add_subplot(grid[0, 0])
    grouped_bars(
        ax_acc,
        names,
        [
            ("Accuracy entrenamiento", [record["metrics"]["accuracy_train"] for record in records]),
            ("Accuracy prueba", [record["metrics"]["accuracy_test"] for record in records]),
        ],
        "Accuracy",
        "Accuracy de entrenamiento y de prueba",
        ylim=dynamic_ylim(
            [record["metrics"]["accuracy_train"] for record in records]
            + [record["metrics"]["accuracy_test"] for record in records],
            floor=0.7,
            ceiling=1.0,
        ),
    )
    ax_f1 = fig.add_subplot(grid[0, 1])
    grouped_bars(
        ax_f1,
        names,
        [
            ("F1 de prueba", [record["metrics"]["f1_test"] for record in records]),
            ("Recall maligno", [record["metrics"]["recall_malignant_test"] for record in records]),
        ],
        "Valor",
        "F1 (clase benigna) y recall de la clase maligna",
        ylim=dynamic_ylim(
            [record["metrics"]["f1_test"] for record in records]
            + [record["metrics"]["recall_malignant_test"] for record in records],
            floor=0.7,
            ceiling=1.0,
        ),
    )
    ax_auc = fig.add_subplot(grid[1, 0])
    auc_records = [record for record in records if record["metrics"]["roc_auc_test"] is not None]
    grouped_bars(
        ax_auc,
        [record["name"] for record in auc_records],
        [("ROC-AUC de prueba", [record["metrics"]["roc_auc_test"] for record in auc_records])],
        "ROC-AUC",
        "ROC-AUC de prueba (modelos con probabilidades)",
        ylim=dynamic_ylim([record["metrics"]["roc_auc_test"] for record in auc_records], floor=0.9, ceiling=1.0),
    )
    ax_cv = fig.add_subplot(grid[1, 1])
    means = [record["metrics"]["cv_accuracy_mean"] for record in records]
    stds = [record["metrics"]["cv_accuracy_std"] for record in records]
    palette = sns.color_palette("colorblind", len(records))
    ax_cv.bar(np.arange(len(records)), means, yerr=stds, color=palette, capsize=3)
    ax_cv.set_xticks(np.arange(len(records)))
    ax_cv.set_xticklabels(names, rotation=18, ha="right")
    ax_cv.set_ylabel("Accuracy")
    ax_cv.set_title("Validacion cruzada de 5 pliegues (media y desv. estandar)")
    ax_cv.set_ylim(*dynamic_ylim(means, floor=0.85, ceiling=1.0))
    fig.suptitle("Item 3. Ensambles de clasificacion frente a modelos individuales", fontsize=15)
    save_figure(fig, path)


def plot_item4(path, clf_records, reg_records) -> None:
    fig = plt.figure(figsize=(16.5, 12.2), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    clf_names = [record["name"] for record in clf_records]
    ax_clf = fig.add_subplot(grid[0, 0])
    grouped_bars(
        ax_clf,
        clf_names,
        [
            ("Accuracy prueba", [record["metrics"]["accuracy_test"] for record in clf_records]),
            ("F1 prueba", [record["metrics"]["f1_test"] for record in clf_records]),
            ("ROC-AUC prueba", [record["metrics"]["roc_auc_test"] for record in clf_records]),
        ],
        "Valor",
        "Clasificacion: arbol individual y Bagging",
        ylim=dynamic_ylim(
            [record["metrics"]["accuracy_test"] for record in clf_records]
            + [record["metrics"]["f1_test"] for record in clf_records]
            + [record["metrics"]["roc_auc_test"] for record in clf_records],
            floor=0.85,
            ceiling=1.0,
        ),
    )
    reg_names = [record["name"] for record in reg_records]
    ax_reg = fig.add_subplot(grid[0, 1])
    grouped_bars(
        ax_reg,
        reg_names,
        [("RMSE de prueba", [record["metrics"]["rmse_test"] for record in reg_records])],
        "RMSE (mpg)",
        "Regresión: árbol individual y Bagging",
    )
    bag_clf = [record for record in clf_records if record["role"] == "ensemble"]
    individual_clf = find_model(clf_records, "Árbol individual")
    ax_clf_curve = fig.add_subplot(grid[1, 0])
    n_clf = [record["n_estimators"] for record in bag_clf]
    ax_clf_curve.plot(n_clf, [record["metrics"]["accuracy_test"] for record in bag_clf], marker="o", label="Accuracy de prueba")
    ax_clf_curve.plot(n_clf, [record["metrics"]["f1_test"] for record in bag_clf], marker="o", label="F1 de prueba")
    ax_clf_curve.plot(n_clf, [record["metrics"]["oob_score"] for record in bag_clf], marker="o", label="OOB accuracy")
    ax_clf_curve.axhline(individual_clf["metrics"]["accuracy_test"], linestyle="--", color="0.3", label="Accuracy del arbol individual")
    ax_clf_curve.set_xlabel("Numero de arboles")
    ax_clf_curve.set_ylabel("Valor")
    ax_clf_curve.set_title("Convergencia del Bagging clasificador")
    ax_clf_curve.legend(fontsize=8)
    bag_reg = [record for record in reg_records if record["role"] == "ensemble"]
    individual_reg = find_model(reg_records, "Árbol individual")
    ax_reg_curve = fig.add_subplot(grid[1, 1])
    n_reg = [record["n_estimators"] for record in bag_reg]
    ax_reg_curve.plot(n_reg, [record["metrics"]["rmse_test"] for record in bag_reg], marker="o", color="C0", label="RMSE de prueba")
    ax_reg_curve.axhline(individual_reg["metrics"]["rmse_test"], linestyle="--", color="C0", label="RMSE del arbol individual")
    ax_oob = ax_reg_curve.twinx()
    ax_oob.plot(n_reg, [record["metrics"]["oob_score"] for record in bag_reg], marker="s", color="C1", label="OOB R2")
    ax_reg_curve.set_xlabel("Numero de arboles")
    ax_reg_curve.set_ylabel("RMSE de prueba (mpg)")
    ax_oob.set_ylabel("OOB R2")
    ax_oob.set_ylim(0.0, 1.05)
    ax_reg_curve.set_title("Convergencia del Bagging regresor")
    lines_left, labels_left = ax_reg_curve.get_legend_handles_labels()
    lines_right, labels_right = ax_oob.get_legend_handles_labels()
    ax_reg_curve.legend(lines_left + lines_right, labels_left + labels_right, fontsize=8)
    fig.suptitle("Item 4. Bagging frente al arbol individual", fontsize=15)
    save_figure(fig, path)


def plot_importance_panel(ax, rows_a, rows_b, name_a, name_b, title):
    score = {}
    for row in rows_a:
        score[row["feature"]] = max(score.get(row["feature"], 0.0), row["importance"])
    for row in rows_b:
        score[row["feature"]] = max(score.get(row["feature"], 0.0), row["importance"])
    labels = [name for name, _value in sorted(score.items(), key=lambda item: item[1], reverse=True)[:8]]
    map_a = {row["feature"]: row["importance"] for row in rows_a}
    map_b = {row["feature"]: row["importance"] for row in rows_b}
    y_pos = np.arange(len(labels))
    ax.barh(y_pos + 0.16, [map_a.get(label, 0.0) for label in labels], height=0.32, label=name_a)
    ax.barh(y_pos - 0.16, [map_b.get(label, 0.0) for label in labels], height=0.32, label=name_b)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia relativa (suma 1)")
    ax.set_title(title)
    ax.legend(fontsize=8)


def plot_item5(path, clf_records, reg_records) -> None:
    fig = plt.figure(figsize=(16.5, 15.5), layout="constrained")
    grid = fig.add_gridspec(3, 2)
    clf_names = [record["name"] for record in clf_records]
    ax_clf = fig.add_subplot(grid[0, 0])
    grouped_bars(
        ax_clf,
        clf_names,
        [
            ("Accuracy", [record["metrics"]["accuracy_test"] for record in clf_records]),
            ("F1", [record["metrics"]["f1_test"] for record in clf_records]),
            ("ROC-AUC", [record["metrics"]["roc_auc_test"] for record in clf_records]),
        ],
        "Valor de prueba",
        "Clasificacion: Random Forest y XGBoost",
        ylim=dynamic_ylim(
            [record["metrics"]["accuracy_test"] for record in clf_records]
            + [record["metrics"]["f1_test"] for record in clf_records]
            + [record["metrics"]["roc_auc_test"] for record in clf_records],
            floor=0.9,
            ceiling=1.0,
        ),
    )
    reg_names = [record["name"] for record in reg_records]
    ax_reg = fig.add_subplot(grid[0, 1])
    x_pos = np.arange(len(reg_records))
    bar_width = 0.36
    rmse_bars = ax_reg.bar(
        x_pos - bar_width / 2,
        [record["metrics"]["rmse_test"] for record in reg_records],
        width=bar_width,
        color="C0",
        label="RMSE de prueba (mpg)",
    )
    ax_r2 = ax_reg.twinx()
    r2_bars = ax_r2.bar(
        x_pos + bar_width / 2,
        [record["metrics"]["r2_test"] for record in reg_records],
        width=bar_width,
        color="C1",
        label="R2 de prueba",
    )
    ax_reg.set_xticks(x_pos)
    ax_reg.set_xticklabels(reg_names, rotation=28, ha="right", fontsize=8)
    ax_reg.set_ylabel("RMSE de prueba (mpg)")
    ax_r2.set_ylabel("R2 de prueba")
    ax_r2.set_ylim(0.0, 1.05)
    ax_reg.set_title("Regresión: RMSE en el eje izquierdo y R² en el derecho")
    ax_reg.legend([rmse_bars, r2_bars], [rmse_bars.get_label(), r2_bars.get_label()], fontsize=8, loc="upper left")
    rf_clf = find_model(clf_records, "Random Forest optimizado")
    xgb_clf = find_model(clf_records, "XGBoost optimizado")
    rf_reg = find_model(reg_records, "Random Forest optimizado")
    xgb_reg = find_model(reg_records, "XGBoost optimizado")
    ax_imp_c = fig.add_subplot(grid[1, :])
    plot_importance_panel(
        ax_imp_c,
        rf_clf["importances"]["relative"],
        xgb_clf["importances"]["relative"],
        "Random Forest optimizado",
        "XGBoost optimizado",
        "Importancia relativa en clasificacion (modelos optimizados)",
    )
    ax_imp_r = fig.add_subplot(grid[2, :])
    plot_importance_panel(
        ax_imp_r,
        rf_reg["importances"]["relative"],
        xgb_reg["importances"]["relative"],
        "Random Forest optimizado",
        "XGBoost optimizado",
        "Importancia relativa en regresion (modelos optimizados)",
    )
    fig.suptitle("Item 5. Random Forest frente a XGBoost", fontsize=15)
    save_figure(fig, path)


def validation_curve_depth(x_train, y_train, x_test, y_test) -> dict:
    depths = list(range(1, 13))
    r2_train, r2_test, rmse_train, rmse_test = [], [], [], []
    for depth in depths:
        model, _ms = timed_fit(make_mpg_tree(max_depth=depth), x_train, y_train)
        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)
        r2_train.append(float(r2_score(y_train, pred_train)))
        r2_test.append(float(r2_score(y_test, pred_test)))
        rmse_train.append(float(np.sqrt(mean_squared_error(y_train, pred_train))))
        rmse_test.append(float(np.sqrt(mean_squared_error(y_test, pred_test))))
    best_index = int(np.argmin(rmse_test))
    return {
        "max_depth": depths,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "best_test_depth": int(depths[best_index]),
        "best_test_r2": r2_test[best_index],
        "best_test_rmse": rmse_test[best_index],
        "note": "Arboles con criterion squared_error, min_samples_split 2 y min_samples_leaf 1. La curva aisla el efecto de max_depth.",
    }


def search_info(search, elapsed: float, scoring: str, space_text: str) -> dict:
    n_candidates = int(len(search.cv_results_["params"]))
    return {
        "n_candidates": n_candidates,
        "n_folds": N_FOLDS,
        "n_fits": n_candidates * N_FOLDS,
        "search_time_s": elapsed,
        "scoring": scoring,
        "space_text": space_text,
        "best_params_text": fmt_params(to_builtin(search.best_params_)),
    }


def load_breast_cancer_data() -> dict:
    bunch = load_breast_cancer(as_frame=True)
    if list(bunch.target_names) != ["malignant", "benign"]:
        raise RuntimeError(f"Codificacion inesperada de breast_cancer: {list(bunch.target_names)}")
    frame = bunch.frame.copy()
    x_values = bunch.data.copy()
    y_values = bunch.target.astype(int).copy()
    y_values.name = "target"
    if x_values.shape != (569, 30):
        raise RuntimeError(f"Se esperaban 569 x 30 y se obtuvo {x_values.shape}")
    summary = []
    described = x_values.describe().T
    for feature in x_values.columns:
        summary.append(
            {
                "feature": str(feature),
                "mean": float(described.loc[feature, "mean"]),
                "std": float(described.loc[feature, "std"]),
                "min": float(described.loc[feature, "min"]),
                "max": float(described.loc[feature, "max"]),
            }
        )
    counts = y_values.value_counts().to_dict()
    return {
        "X": x_values,
        "y": y_values,
        "meta": {
            "name": "breast_cancer",
            "source": "sklearn.datasets.load_breast_cancer",
            "n_rows": int(x_values.shape[0]),
            "n_features": int(x_values.shape[1]),
            "n_missing": int(frame.isna().sum().sum()),
            "positive_class": "benigna (etiqueta 1). Precision, recall y F1 usan esta clase positiva, que es la convencion de scikit-learn.",
            "class_counts": {"malignant": int(counts.get(0, 0)), "benign": int(counts.get(1, 0))},
            "feature_summary": summary,
            "feature_names": [str(name) for name in x_values.columns],
        },
    }


def _find_column(columns, options) -> str | None:
    lowered = {str(column).lower().strip(): column for column in columns}
    for option in options:
        if option in lowered:
            return lowered[option]
    for column in columns:
        text = str(column).lower()
        for option in options:
            if option in text:
                return column
    return None


def _normalize_mpg_frame(frame: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict]:
    renamed = frame.copy()
    renamed.columns = [str(column) for column in renamed.columns]
    spec = {
        "mpg": ["mpg", "class", "target"],
        "cylinders": ["cylinders", "cylinder"],
        "displacement": ["displacement"],
        "horsepower": ["horsepower"],
        "weight": ["weight"],
        "acceleration": ["acceleration"],
        "model_year": ["model_year", "model year", "model-year", "modelyear", "model"],
        "origin": ["origin"],
        "name": ["car_name", "car name", "name"],
    }
    chosen = {}
    for key, options in spec.items():
        column = _find_column(renamed.columns, options)
        if column is not None:
            chosen[key] = column
    required = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]
    missing = [key for key in required if key not in chosen]
    if missing:
        raise ValueError(f"Faltan columnas {missing} en {source}. Columnas: {list(renamed.columns)}")
    data = pd.DataFrame()
    data["mpg"] = pd.to_numeric(renamed[chosen["mpg"]], errors="coerce")
    for key in ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]:
        data[key] = pd.to_numeric(renamed[chosen[key]], errors="coerce")
    origin_raw = renamed[chosen["origin"]]
    origin = []
    for value in origin_raw.tolist():
        if pd.isna(value):
            origin.append(np.nan)
            continue
        key = value.item() if hasattr(value, "item") and not isinstance(value, str) else value
        if key in ORIGIN_MAP:
            origin.append(ORIGIN_MAP[key])
        else:
            text = str(key).strip().lower()
            origin.append(ORIGIN_MAP.get(text, text))
    data["origin"] = origin
    n_raw = int(len(data))
    n_missing_hp = int(data["horsepower"].isna().sum())
    cleaned = data.dropna(subset=["mpg", "horsepower", "origin"]).copy()
    cleaned = cleaned[cleaned["origin"].isin(["usa", "europe", "japan"])].reset_index(drop=True)
    horsepower_mean = float(cleaned["horsepower"].mean())
    if not (40.0 <= horsepower_mean <= 200.0):
        raise ValueError(f"La media de horsepower quedo fuera del rango esperado: {horsepower_mean}")
    if not (15.0 <= float(cleaned["mpg"].mean()) <= 35.0):
        raise ValueError("La media de mpg quedo fuera del rango esperado")
    if not (350 <= len(cleaned) <= 398):
        raise ValueError(f"El numero de filas limpias es inesperado: {len(cleaned)}")
    meta = {
        "source": source,
        "n_raw": n_raw,
        "n_missing_horsepower": n_missing_hp,
        "n_clean": int(len(cleaned)),
        "dropped_rows": int(n_raw - len(cleaned)),
        "dropped_columns": ["car_name"],
        "origin_reference_for_one_hot": "europe",
    }
    return cleaned, meta


def _load_openml_mpg() -> tuple[pd.DataFrame, dict]:
    # fetch_openml llama urlopen sin timeout. Sin un limite global, una
    # conexion colgada nunca llega al respaldo de seaborn o de UCI.
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(12)
    try:
        bunch = fetch_openml(data_id=196, as_frame=True, parser="auto")
    finally:
        socket.setdefaulttimeout(previous)
    frame = bunch.data.copy()
    if _find_column(frame.columns, ["mpg", "class", "target"]) is None:
        frame["mpg"] = np.asarray(bunch.target)
    return _normalize_mpg_frame(frame, "sklearn.datasets.fetch_openml(data_id=196)")


def _load_seaborn_mpg() -> tuple[pd.DataFrame, dict]:
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)
    try:
        frame = sns.load_dataset("mpg")
    finally:
        socket.setdefaulttimeout(previous)
    return _normalize_mpg_frame(frame, "seaborn.load_dataset('mpg')")


def _load_uci_mpg() -> tuple[pd.DataFrame, dict]:
    import urllib.request

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if '"' in line:
            left, name = line.split('"', 1)
            name = name.replace('"', "").strip()
            parts = left.split()
        else:
            parts = line.split()
            name = " ".join(parts[8:])
            parts = parts[:8]
        if len(parts) < 8:
            continue
        rows.append(parts[:8] + [name])
    frame = pd.DataFrame(
        rows,
        columns=[
            "mpg",
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "model_year",
            "origin",
            "car_name",
        ],
    )
    return _normalize_mpg_frame(frame, url)


def _mpg_meta_from_clean(cleaned: pd.DataFrame, meta: dict) -> dict:
    description = cleaned[NUMERIC_FEATURES + ["mpg"]].describe().T
    summary = []
    for feature in ["mpg", *NUMERIC_FEATURES]:
        summary.append(
            {
                "feature": feature,
                "mean": float(description.loc[feature, "mean"]),
                "std": float(description.loc[feature, "std"]),
                "min": float(description.loc[feature, "min"]),
                "max": float(description.loc[feature, "max"]),
            }
        )
    correlations = []
    corr = cleaned[NUMERIC_FEATURES + ["mpg"]].corr(method="pearson")
    for feature in NUMERIC_FEATURES:
        correlations.append({"feature": feature, "correlation": float(corr.loc[feature, "mpg"])})
    correlations.sort(key=lambda row: abs(row["correlation"]), reverse=True)
    pair_corr = float(corr.loc["weight", "displacement"])
    origin_counts = {str(key): int(value) for key, value in cleaned["origin"].value_counts().to_dict().items()}
    meta = dict(meta)
    meta.update(
        {
            "target_summary": summary[0],
            "feature_summary": summary[1:],
            "correlations_with_mpg": correlations,
            "corr_weight_displacement": pair_corr,
            "origin_counts": origin_counts,
            "mpg_mean": float(cleaned["mpg"].mean()),
            "mpg_std": float(cleaned["mpg"].std(ddof=1)),
            "mpg_min": float(cleaned["mpg"].min()),
            "mpg_max": float(cleaned["mpg"].max()),
        }
    )
    return meta


def _cache_paths() -> tuple[Path, Path]:
    return DATA / "auto_mpg_clean.csv", DATA / "auto_mpg_meta.json"


def _write_mpg_cache(cleaned: pd.DataFrame, meta: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = _cache_paths()
    cleaned.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(to_builtin(meta), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_mpg_cache() -> tuple[pd.DataFrame, dict]:
    csv_path, meta_path = _cache_paths()
    cleaned = pd.read_csv(csv_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cleaned, checked = _normalize_mpg_frame(cleaned, meta.get("source", "cache local"))
    # La cache ya esta limpia: se conservan los conteos originales de la descarga.
    checked["n_raw"] = int(meta.get("n_raw", checked["n_raw"]))
    checked["n_missing_horsepower"] = int(meta.get("n_missing_horsepower", 0))
    checked["dropped_rows"] = int(meta.get("dropped_rows", 0))
    checked["source"] = str(meta.get("source", checked["source"]))
    return cleaned, _mpg_meta_from_clean(cleaned, checked)


def load_auto_mpg() -> dict:
    csv_path, meta_path = _cache_paths()
    if csv_path.exists() and meta_path.exists():
        log("auto_mpg: usando cache local data/auto_mpg_clean.csv")
        cleaned, meta = _read_mpg_cache()
    else:
        attempts = [
            ("OpenML data_id 196", _load_openml_mpg),
            ("seaborn mpg", _load_seaborn_mpg),
            ("UCI auto-mpg", _load_uci_mpg),
        ]
        errors = []
        cleaned = None
        meta = None
        for label, loader in attempts:
            try:
                log(f"auto_mpg: intentando {label}")
                cleaned, meta = loader()
                meta = _mpg_meta_from_clean(cleaned, meta)
                log(f"auto_mpg cargado desde {meta['source']} ({meta['n_clean']} filas)")
                break
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
                log(f"auto_mpg: fallo {label}")
        if cleaned is None or meta is None:
            raise RuntimeError("No se pudo cargar auto_mpg. " + " | ".join(errors))
        _write_mpg_cache(cleaned, meta)
    x_values = cleaned[NUMERIC_FEATURES + ["origin"]].copy()
    y_values = cleaned["mpg"].astype(float).copy()
    y_values.name = "mpg"
    return {"X": x_values, "y": y_values, "frame": cleaned, "meta": meta}


def split_classification(x_values, y_values):
    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_values,
    )
    return (
        x_train.reset_index(drop=True),
        x_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def split_regression(x_values, y_values):
    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    return (
        x_train.reset_index(drop=True),
        x_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def class_counts(y_values) -> dict:
    series = pd.Series(y_values)
    return {"malignant": int((series == 0).sum()), "benign": int((series == 1).sum())}


def individual_classifier_estimators():
    return [
        ("arbol", DecisionTreeClassifier(random_state=SEED)),
        (
            "svc",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="rbf", probability=True, random_state=SEED)),
                ]
            ),
        ),
        (
            "knn",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=5)),
                ]
            ),
        ),
        ("logreg", make_scaled_logistic()),
    ]


def run_item1(x_train, y_train, x_test, y_test, cv) -> dict:
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


def run_item2(x_train, y_train, x_test, y_test, cv) -> dict:
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


def run_item3(x_train, y_train, x_test, y_test, cv) -> dict:
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


def run_item4(bc, mpg) -> dict:
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


def run_item5(bc, mpg) -> dict:
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


def fmt_int_opt(value) -> str:
    if value is None:
        return "no aplica"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}"


def fmt_pp(left, right, nd: int = 2) -> str:
    text = f"{pp(left, right):.{nd}f}"
    if text.startswith("-") and float(text) == 0.0:
        return f"{0:.{nd}f}"
    return text


def clf_table(models, with_oob: bool = False) -> str:
    headers = [
        "Modelo",
        "Acc. train",
        "Acc. test",
        "Precisión",
        "Recall",
        "F1",
        "ROC-AUC",
        "Recall maligno",
        "CV accuracy (media ± desv.)",
        "Profundidad",
        "Hojas",
        "Tiempo (ms)",
    ]
    if with_oob:
        headers.append("OOB accuracy")
    rows = []
    for model in models:
        metrics = model["metrics"]
        row = [
            model["name"],
            fmt(metrics["accuracy_train"]),
            fmt(metrics["accuracy_test"]),
            fmt(metrics["precision_test"]),
            fmt(metrics["recall_test"]),
            fmt(metrics["f1_test"]),
            fmt(metrics["roc_auc_test"]),
            fmt(metrics["recall_malignant_test"]),
            fmt_pm(metrics["cv_accuracy_mean"], metrics["cv_accuracy_std"]),
            fmt_int_opt(metrics["depth"]),
            fmt_int_opt(metrics["n_leaves"]),
            fmt(metrics["fit_time_ms"], 2),
        ]
        if with_oob:
            row.append(fmt(metrics["oob_score"]))
        rows.append(row)
    return md_table(headers, rows)


def reg_table(models, with_oob: bool = False) -> str:
    headers = [
        "Modelo",
        "R² train",
        "R² test",
        "RMSE test",
        "MAE test",
        "MSE test",
        "CV R² (media ± desv.)",
        "CV RMSE (media ± desv.)",
        "Profundidad",
        "Hojas",
        "Tiempo (ms)",
    ]
    if with_oob:
        headers.append("OOB R²")
    rows = []
    for model in models:
        metrics = model["metrics"]
        row = [
            model["name"],
            fmt(metrics["r2_train"]),
            fmt(metrics["r2_test"]),
            fmt(metrics["rmse_test"]),
            fmt(metrics["mae_test"]),
            fmt(metrics["mse_test"]),
            fmt_pm(metrics["cv_r2_mean"], metrics["cv_r2_std"]),
            fmt_pm(metrics["cv_rmse_mean"], metrics["cv_rmse_std"]),
            fmt_int_opt(metrics["depth"]),
            fmt_int_opt(metrics["n_leaves"]),
            fmt(metrics["fit_time_ms"], 2),
        ]
        if with_oob:
            row.append(fmt(metrics["oob_score"]))
        rows.append(row)
    return md_table(headers, rows)


def params_table(models) -> str:
    rows = []
    for model in models:
        search = model.get("search")
        if search:
            detail = (
                f"{search['n_candidates']} combinaciones y {search['n_fits']} ajustes "
                f"en {fmt(search['search_time_s'], 2)} s. Criterio interno: {search['scoring']}. "
                f"Mejor configuración: {search['best_params_text']}."
            )
        else:
            detail = "Ajuste directo sobre el entrenamiento, sin búsqueda de hiperparámetros."
        rows.append([model["name"], model["params_text"], detail])
    return md_table(["Modelo", "Configuración", "Búsqueda"], rows)


def cm_table(models) -> str:
    rows = []
    for model in models:
        matrix = model["metrics"]["confusion_matrix_test"]
        rows.append([model["name"], matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]])
    return md_table(
        [
            "Modelo",
            "Real maligno, predicho maligno",
            "Real maligno, predicho benigno",
            "Real benigno, predicho maligno",
            "Real benigno, predicho benigno",
        ],
        rows,
    )


def summary_table(rows_in, value_headers) -> str:
    headers = ["Variable", *value_headers]
    rows = []
    for row in rows_in:
        rows.append(
            [
                row["feature"],
                fmt(row["mean"], 3),
                fmt(row["std"], 3),
                fmt(row["min"], 3),
                fmt(row["max"], 3),
            ]
        )
    return md_table(headers, rows)


def image_line(filename: str, caption: str) -> str:
    return f"![{caption}]({filename})"


def agree_sentence(conclusion: dict, cv_label: str) -> str:
    if conclusion["cv_agrees"]:
        return (
            f"La validación cruzada señala al mismo modelo, {conclusion['cv_leader']}, como líder de {cv_label}."
        )
    return (
        f"La métrica de prueba selecciona a {conclusion['winner']} y la validación cruzada selecciona a "
        f"{conclusion['cv_leader']} en {cv_label}. El ganador del ítem es el de la prueba, y esta discrepancia queda registrada."
    )


def importance_phrase(record, k: int = 3) -> str:
    payload = record.get("importances") or {}
    rows = payload.get("relative") or []
    if not rows:
        return "sin importancias disponibles"
    return ", ".join(f"{row['feature']} ({fmt(row['importance'], 3)})" for row in rows[:k])


def diagnostics_paragraph(record) -> str:
    diag = record.get("ensemble_diagnostics")
    if not diag:
        return f"No se estimó la correlación entre miembros de {record['name']}."
    return (
        f"En el conjunto de prueba, la correlación media entre las predicciones de los {diag['n_members']} "
        f"miembros de {record['name']} es {fmt(diag['mean_pairwise_prediction_correlation'], 3)}. "
        f"Bajo el modelo de equicorrelación, el factor de varianza del promedio es "
        f"{fmt(diag['equicorrelation_variance_factor'], 3)}. "
        f"Si los miembros fueran independientes, ese factor sería {fmt(diag['independence_variance_factor'], 4)}. "
        f"La correlación observada impide que la varianza caiga en la proporción ideal 1/B."
    )


def auc_margin_sentence(winner_value, runner_value) -> str:
    gap = abs(float(winner_value) - float(runner_value))
    if gap < 1e-8:
        return (
            " El ROC-AUC de prueba empata en la práctica, así que el desempate lo resuelven el F1 y la accuracy."
        )
    if gap < 0.005:
        return (
            " El margen de ROC-AUC es ajustado: la diferencia es menor que 0.005 y conviene leerla junto al F1, la accuracy y el tiempo de ajuste."
        )
    return ""


def rmse_margin_sentence(winner_rmse, runner_rmse) -> str:
    base = abs(float(runner_rmse))
    if base == 0.0:
        return ""
    relative = abs(float(winner_rmse) - float(runner_rmse)) / base
    if relative < 0.01:
        return (
            f" El margen de RMSE es ajustado: la diferencia relativa es {fmt(relative * 100, 2)}%, por debajo del 1%."
        )
    return ""


def gap_sentence(model, train_key, test_key, label) -> str:
    train = model["metrics"][train_key]
    test = model["metrics"][test_key]
    return (
        f"{model['name']} obtiene {label} de entrenamiento {fmt(train)} y de prueba {fmt(test)} "
        f"(brecha {fmt(train - test)})."
    )


def section_item1(item) -> str:
    models = item["models"]
    gini = find_model(models, "Gini")
    entropy = find_model(models, "Entropía")
    gini_pruned = find_model(models, "Gini podado")
    entropy_pruned = find_model(models, "Entropía podada")
    logistic = find_model(models, "Regresión logística")
    winner = find_model(models, item["conclusion"]["winner"])
    runner = find_model(models, item["conclusion"]["runner_up"])
    unpruned = [gini, entropy]
    better_unpruned = max(
        unpruned,
        key=lambda model: (
            model["metrics"]["roc_auc_test"],
            model["metrics"]["f1_test"],
            model["metrics"]["cv_accuracy_mean"],
            -model["metrics"]["n_leaves"],
        ),
    )
    other_unpruned = entropy if better_unpruned is gini else gini
    better_pruned = max(
        [gini_pruned, entropy_pruned],
        key=lambda model: (
            model["metrics"]["roc_auc_test"],
            model["metrics"]["f1_test"],
            -model["metrics"]["n_leaves"],
        ),
    )
    errors, total = errors_from_cm(winner["metrics"])
    logistic_errors, _total = errors_from_cm(logistic["metrics"])
    design = "\n".join(
        [
            "Se ajustan dos árboles sin poda, uno con índice de Gini y otro con entropía, y sus versiones podadas por complejidad de costo. El `ccp_alpha` de cada criterio es el que maximiza el ROC-AUC de la validación cruzada de 5 pliegues sobre el entrenamiento. La regresión logística entra en un `Pipeline` con `StandardScaler` y penalización L2 (`l1_ratio=0`, `C=1`, solver `lbfgs`, `max_iter=10000`).",
            "",
            params_table(models),
            "",
            "Precisión, recall y F1 usan la clase positiva benigna (etiqueta 1), que es la codificación de scikit-learn. El recall maligno se reporta aparte porque es la sensibilidad sobre la clase clínicamente crítica. El ROC-AUC no depende de cuál etiqueta se llame positiva.",
        ]
    )
    answer = "\n\n".join(
        [
            "El criterio de decisión es el ROC-AUC de prueba. El desempate usa F1, accuracy de prueba y la media de accuracy en la validación cruzada.",
            (
                f"Entre los árboles sin poda, {better_unpruned['name']} generaliza mejor que {other_unpruned['name']}: "
                f"ROC-AUC de prueba {fmt(better_unpruned['metrics']['roc_auc_test'])} frente a {fmt(other_unpruned['metrics']['roc_auc_test'])} "
                f"({fmt_pp(better_unpruned['metrics']['roc_auc_test'], other_unpruned['metrics']['roc_auc_test'])} puntos de ROC-AUC), "
                f"F1 {fmt(better_unpruned['metrics']['f1_test'])} frente a {fmt(other_unpruned['metrics']['f1_test'])} "
                f"y accuracy de validación cruzada {fmt_pm(better_unpruned['metrics']['cv_accuracy_mean'], better_unpruned['metrics']['cv_accuracy_std'])} "
                f"frente a {fmt_pm(other_unpruned['metrics']['cv_accuracy_mean'], other_unpruned['metrics']['cv_accuracy_std'])}. "
                f"{gini['name']} alcanza profundidad {fmt_int_opt(gini['metrics']['depth'])} y {fmt_int_opt(gini['metrics']['n_leaves'])} hojas; "
                f"{entropy['name']} alcanza profundidad {fmt_int_opt(entropy['metrics']['depth'])} y {fmt_int_opt(entropy['metrics']['n_leaves'])} hojas. "
                f"{gap_sentence(gini, 'accuracy_train', 'accuracy_test', 'accuracy')} "
                f"{gap_sentence(entropy, 'accuracy_train', 'accuracy_test', 'accuracy')}"
            ),
            (
                f"La poda reduce la complejidad. {gini_pruned['name']} queda con profundidad {fmt_int_opt(gini_pruned['metrics']['depth'])}, "
                f"{fmt_int_opt(gini_pruned['metrics']['n_leaves'])} hojas, accuracy de entrenamiento {fmt(gini_pruned['metrics']['accuracy_train'])} "
                f"y ROC-AUC de prueba {fmt(gini_pruned['metrics']['roc_auc_test'])} "
                f"(alpha {fmt_alpha(gini_pruned['params']['ccp_alpha'])}). "
                f"{entropy_pruned['name']} queda con profundidad {fmt_int_opt(entropy_pruned['metrics']['depth'])}, "
                f"{fmt_int_opt(entropy_pruned['metrics']['n_leaves'])} hojas, accuracy de entrenamiento {fmt(entropy_pruned['metrics']['accuracy_train'])} "
                f"y ROC-AUC de prueba {fmt(entropy_pruned['metrics']['roc_auc_test'])} "
                f"(alpha {fmt_alpha(entropy_pruned['params']['ccp_alpha'])}). "
                f"Entre las dos versiones podadas, el mejor ROC-AUC de prueba es el de {better_pruned['name']}."
            ),
            (
                f"**Ganador: {winner['name']}.** Su ROC-AUC de prueba es {fmt(winner['metrics']['roc_auc_test'])}, "
                f"frente a {fmt(runner['metrics']['roc_auc_test'])} de {runner['name']} "
                f"(diferencia de {fmt_pp(winner['metrics']['roc_auc_test'], runner['metrics']['roc_auc_test'])} puntos de ROC-AUC). "
                f"El F1 de prueba del ganador es {fmt(winner['metrics']['f1_test'])} y su accuracy de prueba es {fmt(winner['metrics']['accuracy_test'])}. "
                f"Comete {errors} errores en {total} casos de prueba. "
                f"La regresión logística comete {logistic_errors} errores, con recall maligno {fmt(logistic['metrics']['recall_malignant_test'])} "
                f"y F1 {fmt(logistic['metrics']['f1_test'])}. "
                f"{agree_sentence(item['conclusion'], 'ROC-AUC')} "
                f"{t_test_sentence(winner, runner, 'cv_roc_auc_folds', 'ROC-AUC')}"
            ),
            (
                (
                    "La regresión logística es el mejor modelo de este ítem. En el espacio de 30 variables citológicas, "
                    "la frontera lineal con penalización L2 aprovecha que varias mediciones (medias, errores estándar y peores valores) "
                    "se alinean con una dirección de separación estable. Cada coeficiente queda encogido por la norma L2 con C = 1, "
                    "así que el ajuste no necesita memorizar particiones. "
                    f"Los árboles sin poda alcanzan accuracy de entrenamiento {fmt(gini['metrics']['accuracy_train'])} (Gini) y "
                    f"{fmt(entropy['metrics']['accuracy_train'])} (entropía): sus cortes ortogonales llegan a aislar el entrenamiento "
                    f"y pierden parte de esa exactitud en la prueba (brechas {fmt(gini['metrics']['accuracy_train'] - gini['metrics']['accuracy_test'])} y "
                    f"{fmt(entropy['metrics']['accuracy_train'] - entropy['metrics']['accuracy_test'])}). "
                    f"La logística mantiene accuracy de entrenamiento {fmt(logistic['metrics']['accuracy_train'])} y de prueba {fmt(logistic['metrics']['accuracy_test'])}, "
                    f"con un tiempo de ajuste de {fmt(logistic['metrics']['fit_time_ms'], 2)} ms."
                )
                if winner["name"] == "Regresión logística"
                else (
                    f"El ganador es {winner['name']}, con una ventaja de "
                    f"{fmt_pp(winner['metrics']['roc_auc_test'], logistic['metrics']['roc_auc_test'])} puntos de ROC-AUC "
                    f"sobre la regresión logística ({fmt(logistic['metrics']['roc_auc_test'])}). "
                    f"En esta partición, los cortes del árbol capturan mejor la separación que el hiperplano L2. "
                    f"Aun así, los árboles sin poda muestran el sobreajuste clásico de CART: accuracy de entrenamiento "
                    f"{fmt(gini['metrics']['accuracy_train'])} (Gini) y {fmt(entropy['metrics']['accuracy_train'])} (entropía), "
                    f"con brechas de prueba de {fmt(gini['metrics']['accuracy_train'] - gini['metrics']['accuracy_test'])} y "
                    f"{fmt(entropy['metrics']['accuracy_train'] - entropy['metrics']['accuracy_test'])}. "
                    f"La logística sigue siendo el referente lineal, con F1 {fmt(logistic['metrics']['f1_test'])} "
                    f"y recall maligno {fmt(logistic['metrics']['recall_malignant_test'])}."
                )
            ),
        ]
    )
    return "\n\n".join(
        [
            "## 3. Ítem 1. Árboles de decisión: Gini, entropía y regresión logística",
            "### Enunciado oficial",
            f"> {item['question']}",
            "### Diseño del experimento y configuración de hiperparámetros",
            design,
            "### Resultados",
            clf_table(models),
            "",
            "Matrices de confusión en el conjunto de prueba, con filas de clase real y columnas de clase predicha:",
            "",
            cm_table(models),
            "",
            image_line(
                "plots/item1_breast_cancer.png",
                "Gráfico del ítem 1: métricas de prueba, matrices de confusión y curvas ROC en cáncer de mama",
            ),
            "### Respuesta Fundamentada: ¿Cuál es mejor?",
            answer,
        ]
    )


def section_item2(item, mpg_meta) -> str:
    models = item["models"]
    unrestricted = find_model(models, "Árbol sin restricciones")
    shallow = find_model(models, "Árbol superficial")
    linear = find_model(models, "Regresión lineal")
    tuned = find_model(models, "Árbol GridSearchCV")
    winner = find_model(models, item["conclusion"]["winner"])
    runner = find_model(models, item["conclusion"]["runner_up"])
    curve = item["validation_curve"]
    saturated_depth = curve["max_depth"][-1]
    saturated_r2 = curve["r2_train"][-1]
    for depth, r2_train in zip(curve["max_depth"], curve["r2_train"]):
        if r2_train >= 0.999:
            saturated_depth = depth
            saturated_r2 = r2_train
            break
    strongest = mpg_meta["correlations_with_mpg"][:3]
    corr_text = ", ".join(f"{row['feature']} (r = {fmt(row['correlation'], 3)})" for row in strongest)
    coef_rows = linear["coefficients"]["coefficients"][:5]
    coef_text = ", ".join(f"{row['feature']} ({fmt(row['coefficient'], 3)})" for row in coef_rows)
    design = "\n".join(
        [
            "Todos los modelos ven la misma matriz: seis variables numéricas y `origin` en one-hot con referencia `europe`. Los árboles reciben esa matriz en sus unidades originales. La regresión lineal la estandariza dentro del `Pipeline`, lo que no cambia las predicciones de mínimos cuadrados y sí hace comparables los coeficientes.",
            "",
            "Las seis configuraciones de árbol son: sin restricciones; profundidad 2; profundidad 4 con `min_samples_split=5` y `min_samples_leaf=2`; error absoluto con profundidad 5 y `min_samples_leaf=5`; poda por el `ccp_alpha` de menor RMSE de validación cruzada; y `GridSearchCV` sobre criterio, profundidad, tamaños mínimos de nodo y `max_features`.",
            "",
            params_table(models),
        ]
    )
    tree_beats_linear = winner["name"] != "Regresión lineal" and winner["metrics"]["rmse_test"] < linear["metrics"]["rmse_test"]
    if tree_beats_linear:
        closing = (
            f"El ganador supera a la regresión lineal porque un árbol regularizado parte el espacio de "
            f"`weight`, `horsepower`, `displacement` y `model_year` en regiones donde la pendiente de mpg cambia. "
            f"Esas interacciones no caben en un único hiperplano. Las importancias relativas del ganador se concentran en {importance_phrase(winner)}. "
            f"La regresión lineal, aun así, ya explica una parte grande de la varianza (R² de prueba {fmt(linear['metrics']['r2_test'])}) "
            f"y sus mayores coeficientes estandarizados son {coef_text}."
        )
    elif winner["name"] == "Regresión lineal":
        closing = (
            f"La regresión lineal es el mejor modelo del ítem. El RMSE de prueba del árbol más cercano, {runner['name']}, "
            f"es {fmt(runner['metrics']['rmse_test'])}, {fmt(runner['metrics']['rmse_test'] - linear['metrics']['rmse_test'])} mpg por encima. "
            f"Las correlaciones de {corr_text} muestran una señal aproximadamente lineal, y la colinealidad entre weight y displacement "
            f"(r = {fmt(mpg_meta['corr_weight_displacement'], 3)}) cabe en el ajuste de mínimos cuadrados. "
            f"Los coeficientes estandarizados de mayor magnitud son {coef_text}. "
            f"El árbol optimizado ({tuned['name']}, RMSE {fmt(tuned['metrics']['rmse_test'])}, R² {fmt(tuned['metrics']['r2_test'])}) "
            f"acerca el resultado al modelo lineal, pero sus cortes ortogonales no mejoran el hiperplano en esta muestra. "
            f"Sus importancias relativas principales son {importance_phrase(tuned)}."
        )
    else:
        closing = (
            f"El ganador es {winner['name']}, con RMSE de prueba {fmt(winner['metrics']['rmse_test'])} "
            f"y R² {fmt(winner['metrics']['r2_test'])}."
        )
    answer = "\n\n".join(
        [
            "El criterio de decisión es el menor RMSE de prueba. El desempate usa el R² de prueba y después el R² medio de validación cruzada.",
            (
                f"El árbol sin restricciones memoriza el entrenamiento: R² train {fmt(unrestricted['metrics']['r2_train'])}, "
                f"R² test {fmt(unrestricted['metrics']['r2_test'])}, RMSE test {fmt(unrestricted['metrics']['rmse_test'])} mpg, "
                f"profundidad {fmt_int_opt(unrestricted['metrics']['depth'])} y {fmt_int_opt(unrestricted['metrics']['n_leaves'])} hojas. "
                f"La regresión lineal obtiene R² train {fmt(linear['metrics']['r2_train'])}, R² test {fmt(linear['metrics']['r2_test'])} "
                f"y RMSE test {fmt(linear['metrics']['rmse_test'])}. "
                + (
                    f"El árbol sin poda queda {fmt(unrestricted['metrics']['rmse_test'] - linear['metrics']['rmse_test'])} mpg por encima de la recta en RMSE de prueba."
                    if unrestricted["metrics"]["rmse_test"] > linear["metrics"]["rmse_test"]
                    else f"En esta partición el árbol sin poda también baja el RMSE de prueba hasta {fmt(unrestricted['metrics']['rmse_test'])}, por debajo de la recta."
                )
            ),
            (
                f"El árbol superficial (max_depth = 2) muestra el otro extremo: R² test {fmt(shallow['metrics']['r2_test'])} "
                f"y RMSE test {fmt(shallow['metrics']['rmse_test'])}, con profundidad {fmt_int_opt(shallow['metrics']['depth'])} "
                f"y {fmt_int_opt(shallow['metrics']['n_leaves'])} hojas. La curva que aísla `max_depth`, con el resto de hiperparámetros en su valor base, "
                f"alcanza el menor RMSE de prueba en profundidad {curve['best_test_depth']} "
                f"(RMSE {fmt(curve['best_test_rmse'])}, R² {fmt(curve['best_test_r2'])}). "
                f"El R² de entrenamiento llega a {fmt(saturated_r2)} en profundidad {saturated_depth}. "
                f"A partir de ese punto el entrenamiento sigue cerrado y la prueba deja de mejorar."
            ),
            (
                f"**Ganador: {winner['name']}.** RMSE de prueba {fmt(winner['metrics']['rmse_test'])} mpg, "
                f"R² de prueba {fmt(winner['metrics']['r2_test'])}, MAE {fmt(winner['metrics']['mae_test'])} "
                f"y validación cruzada de R² {fmt_pm(winner['metrics']['cv_r2_mean'], winner['metrics']['cv_r2_std'])}. "
                f"El segundo modelo, {runner['name']}, tiene RMSE {fmt(runner['metrics']['rmse_test'])} "
                f"({fmt(runner['metrics']['rmse_test'] - winner['metrics']['rmse_test'])} mpg más). "
                f"{agree_sentence(item['conclusion'], 'R²')} "
                f"{t_test_sentence(winner, runner, 'cv_r2_folds', 'R²')}"
            ),
            closing,
            (
                f"La búsqueda del árbol optimizado evaluó {tuned['search']['n_candidates']} combinaciones "
                f"({tuned['search']['n_fits']} ajustes) en {fmt(tuned['search']['search_time_s'], 2)} s. "
                f"La configuración elegida es {tuned['params_text']}."
            ),
        ]
    )
    return "\n\n".join(
        [
            "## 4. Ítem 2. Árboles de regresión con varias configuraciones y regresión lineal",
            "### Enunciado oficial",
            f"> {item['question']}",
            "### Diseño del experimento y configuración de hiperparámetros",
            design,
            "### Resultados",
            reg_table(models),
            "",
            "Coeficientes de la regresión lineal sobre variables estandarizadas (intercepto "
            + fmt(linear["coefficients"]["intercept"], 4)
            + "):",
            "",
            md_table(
                ["Variable estandarizada", "Coeficiente"],
                [[row["feature"], fmt(row["coefficient"], 4)] for row in linear["coefficients"]["coefficients"]],
            ),
            "",
            image_line(
                "plots/item2_auto_mpg.png",
                "Gráfico del ítem 2: R², RMSE y curva de validación de max_depth en auto_mpg",
            ),
            "### Respuesta Fundamentada: ¿Cuál es mejor?",
            answer,
        ]
    )


def ensemble_mechanism(name: str) -> str:
    mechanisms = {
        "Voting suave": "El voting suave promedia las probabilidades de cuatro hipótesis distintas (árbol, SVM con kernel RBF, k-NN y regresión logística). Ese promedio baja la varianza de cualquier base inestable y conserva el sesgo bajo de la logística.",
        "Voting duro": "El voting duro elige la clase mayoritaria. Reduce varianza, aunque descarta la confianza de cada base y por eso no produce un ROC-AUC propio.",
        "Stacking": "El stacking entrena a las bases en pliegues internos y deja que una regresión logística meta aprenda qué modelo pesa más en cada región del espacio. Corrige errores sistemáticos que una votación con pesos fijos no corrige.",
        "AdaBoost": "AdaBoost repondera en secuencia los casos mal clasificados y suma árboles de profundidad 1. Cada paso ataca el residuo de clasificación del anterior, así que el sesgo del tocón cae sin volver al árbol profundo sin regularizar.",
        "Gradient Boosting": "El gradient boosting ajusta árboles poco profundos al gradiente de la pérdida. La corrección secuencial reduce sesgo y la poca profundidad de cada árbol limita la varianza.",
    }
    return mechanisms.get(name, f"{name} combina varios ajustes y por eso puede reducir el error de generalización de una sola hipótesis.")


def section_item3(item) -> str:
    models = item["models"]
    tree = find_model(models, "Árbol de decisión")
    logistic = find_model(models, "Regresión logística")
    hard = find_model(models, "Voting duro")
    winner = find_model(models, item["conclusion"]["winner"])
    runner = find_model(models, item["conclusion"]["runner_up"])
    best_ind = find_model(models, item["best_individual"]["winner"])
    best_ens = find_model(models, item["best_ensemble"]["winner"])
    design = "\n".join(
        [
            "Los modelos individuales son un árbol de decisión, un SVC con kernel RBF y probabilidades calibradas, un k-NN con 5 vecinos y la regresión logística L2. SVC, k-NN y logística se escalan dentro de un `Pipeline`. El árbol no se escala porque sus cortes son invariantes a una transformación monótona por variable.",
            "",
            "Los ensambles reutilizan esas cuatro bases en voting duro, voting suave y stacking con meta-aprendiz logístico y 5 pliegues internos. AdaBoost suma 50 tocones y el gradient boosting usa su configuración base (100 árboles, profundidad 3, learning rate 0.1).",
            "",
            params_table(models),
            "",
            "El voting duro no emite probabilidades, de modo que su ROC-AUC queda marcado como no aplica. El ranking principal usa el ROC-AUC de los modelos que sí producen probabilidades. El accuracy del voting duro se discute de forma explícita.",
        ]
    )
    ens_vs_tree = best_ens["metrics"]["roc_auc_test"] - tree["metrics"]["roc_auc_test"]
    ens_vs_ind = best_ens["metrics"]["roc_auc_test"] - best_ind["metrics"]["roc_auc_test"]
    if ens_vs_ind > 0:
        family = (
            f"El mejor ensamble, {best_ens['name']}, supera al mejor individual, {best_ind['name']}, "
            f"por {fmt(ens_vs_ind)} de ROC-AUC de prueba "
            f"({fmt(best_ens['metrics']['roc_auc_test'])} frente a {fmt(best_ind['metrics']['roc_auc_test'])}). "
            f"También supera al árbol individual por {fmt(ens_vs_tree)} de ROC-AUC "
            f"y por {fmt_pp(best_ens['metrics']['accuracy_test'], tree['metrics']['accuracy_test'])} puntos porcentuales de accuracy."
        )
    elif abs(ens_vs_ind) < 0.005:
        family = (
            f"El mejor ensamble ({best_ens['name']}, ROC-AUC {fmt(best_ens['metrics']['roc_auc_test'])}) y el mejor individual "
            f"({best_ind['name']}, ROC-AUC {fmt(best_ind['metrics']['roc_auc_test'])}) quedan a {fmt(abs(ens_vs_ind))} de ROC-AUC. "
            f"Frente al árbol individual, el ensamble sí abre una ventaja de {fmt(ens_vs_tree)} de ROC-AUC y de "
            f"{fmt_pp(best_ens['metrics']['accuracy_test'], tree['metrics']['accuracy_test'])} puntos porcentuales de accuracy."
        )
    else:
        family = (
            f"El mejor modelo individual, {best_ind['name']}, conserva el ROC-AUC de prueba más alto "
            f"({fmt(best_ind['metrics']['roc_auc_test'])}) y deja al mejor ensamble, {best_ens['name']}, "
            f"{fmt(abs(ens_vs_ind))} por debajo. "
            f"El ensamble sí mejora al árbol individual: ROC-AUC {fmt(best_ens['metrics']['roc_auc_test'])} frente a "
            f"{fmt(tree['metrics']['roc_auc_test'])}, y accuracy {fmt(best_ens['metrics']['accuracy_test'])} frente a "
            f"{fmt(tree['metrics']['accuracy_test'])}."
        )
    answer = "\n\n".join(
        [
            "El criterio de decisión entre modelos con probabilidades es el ROC-AUC de prueba. El desempate usa F1, accuracy de prueba y la media de accuracy en validación cruzada.",
            family,
            (
                f"**Ganador: {winner['name']}.** ROC-AUC de prueba {fmt(winner['metrics']['roc_auc_test'])}, "
                f"F1 {fmt(winner['metrics']['f1_test'])}, accuracy {fmt(winner['metrics']['accuracy_test'])} "
                f"y validación cruzada {fmt_pm(winner['metrics']['cv_accuracy_mean'], winner['metrics']['cv_accuracy_std'])}. "
                f"El segundo, {runner['name']}, queda en ROC-AUC {fmt(runner['metrics']['roc_auc_test'])} "
                f"(diferencia de {fmt_pp(winner['metrics']['roc_auc_test'], runner['metrics']['roc_auc_test'])} puntos)."
                f"{auc_margin_sentence(winner['metrics']['roc_auc_test'], runner['metrics']['roc_auc_test'])} "
                f"El líder por accuracy de prueba, contando también al voting duro, es {item['conclusion']['accuracy_leader']} "
                f"con {fmt(item['conclusion']['accuracy_leader_value'])}. "
                f"El voting duro alcanza accuracy de prueba {fmt(hard['metrics']['accuracy_test'])} y F1 {fmt(hard['metrics']['f1_test'])}. "
                f"{agree_sentence(item['conclusion'], 'ROC-AUC')} "
                f"{t_test_sentence(winner, runner, 'cv_roc_auc_folds', 'ROC-AUC')}"
            ),
            ensemble_mechanism(best_ens["name"]),
            (
                f"El árbol individual queda en accuracy de entrenamiento {fmt(tree['metrics']['accuracy_train'])} y de prueba "
                f"{fmt(tree['metrics']['accuracy_test'])}, con profundidad {fmt_int_opt(tree['metrics']['depth'])} y "
                f"{fmt_int_opt(tree['metrics']['n_leaves'])} hojas. Esa brecha es la varianza que el ensamble ataca. "
                f"La regresión logística individual sigue en ROC-AUC {fmt(logistic['metrics']['roc_auc_test'])} y F1 {fmt(logistic['metrics']['f1_test'])}: "
                f"cualquier ensamble tiene que mejorar esa base lineal, no solo al árbol."
            ),
        ]
    )
    return "\n\n".join(
        [
            "## 5. Ítem 3. Ensambles de clasificación frente a modelos individuales",
            "### Enunciado oficial",
            f"> {item['question']}",
            "### Diseño del experimento y configuración de hiperparámetros",
            design,
            "### Resultados",
            clf_table(models),
            "",
            image_line(
                "plots/item3_ensemble_clf.png",
                "Gráfico del ítem 3: modelos individuales y ensambles de clasificación",
            ),
            "### Respuesta Fundamentada: ¿Cuál es mejor?",
            answer,
        ]
    )


def section_item4(item) -> str:
    clf_models = item["classification"]["models"]
    reg_models = item["regression"]["models"]
    clf_tree = find_model(clf_models, "Árbol individual")
    reg_tree = find_model(reg_models, "Árbol individual")
    clf_ref = find_model(clf_models, "Bagging (100)")
    reg_ref = find_model(reg_models, "Bagging (100)")
    clf_best = find_model(clf_models, item["classification"]["best_bagging"])
    reg_best = find_model(reg_models, item["regression"]["best_bagging"])
    design = "\n".join(
        [
            "Escenario A, cáncer de mama: un `DecisionTreeClassifier` individual frente a `BaggingClassifier` con árboles profundos, `n_estimators` en {10, 50, 100, 200}, `max_samples=1.0`, `bootstrap=True`, `oob_score=True` y `random_state=42`.",
            "",
            "Escenario B, auto_mpg: el mismo contraste con `DecisionTreeRegressor`. El one-hot de `origin` vive en un `Pipeline` anterior al bagging, porque el ensamble convierte las filas a matriz y un transformador por nombre de columna no puede ir dentro de cada muestra bootstrap. El preprocesado se ajusta en cada pliegue de validación y en el entrenamiento completo, nunca con el conjunto de prueba.",
            "",
            "Clasificación:",
            "",
            params_table(clf_models),
            "",
            "Regresión:",
            "",
            params_table(reg_models),
        ]
    )
    clf_gain = clf_best["metrics"]["accuracy_test"] - clf_tree["metrics"]["accuracy_test"]
    reg_gain = reg_tree["metrics"]["rmse_test"] - reg_best["metrics"]["rmse_test"]
    clf_winner = clf_best["name"] if clf_gain > 0 else clf_tree["name"]
    reg_winner = reg_best["name"] if reg_gain > 0 else reg_tree["name"]
    answer = "\n\n".join(
        [
            "En clasificación gana el mayor accuracy de prueba, con desempate por F1 y ROC-AUC. En regresión gana el menor RMSE de prueba.",
            (
                f"En cáncer de mama, el árbol individual tiene accuracy de prueba {fmt(clf_tree['metrics']['accuracy_test'])}, "
                f"F1 {fmt(clf_tree['metrics']['f1_test'])} y ROC-AUC {fmt(clf_tree['metrics']['roc_auc_test'])}, "
                f"con accuracy de entrenamiento {fmt(clf_tree['metrics']['accuracy_train'])}. "
                f"El bagging de referencia, con 100 árboles, sube el accuracy de prueba a {fmt(clf_ref['metrics']['accuracy_test'])}, "
                f"el F1 a {fmt(clf_ref['metrics']['f1_test'])} y el ROC-AUC a {fmt(clf_ref['metrics']['roc_auc_test'])}. "
                f"Su OOB accuracy es {fmt(clf_ref['metrics']['oob_score'])} y su validación cruzada es "
                f"{fmt_pm(clf_ref['metrics']['cv_accuracy_mean'], clf_ref['metrics']['cv_accuracy_std'])}. "
                f"La mejor cantidad de árboles en la prueba es {clf_best['name']}, con accuracy {fmt(clf_best['metrics']['accuracy_test'])} "
                f"({fmt_pp(clf_best['metrics']['accuracy_test'], clf_tree['metrics']['accuracy_test'])} puntos porcentuales sobre el árbol) "
                f"y F1 {fmt(clf_best['metrics']['f1_test'])} "
                f"({fmt_pp(clf_best['metrics']['f1_test'], clf_tree['metrics']['f1_test'])} puntos porcentuales). "
                f"**En clasificación el mejor es {clf_winner}.** "
                f"{t_test_sentence(clf_best, clf_tree, 'cv_accuracy_folds', 'accuracy')}"
            ),
            diagnostics_paragraph(clf_ref),
            (
                f"En auto_mpg, el árbol individual pasa de R² train {fmt(reg_tree['metrics']['r2_train'])} a R² test {fmt(reg_tree['metrics']['r2_test'])}, "
                f"con RMSE de prueba {fmt(reg_tree['metrics']['rmse_test'])} y MAE {fmt(reg_tree['metrics']['mae_test'])}. "
                f"El bagging de 100 árboles obtiene R² test {fmt(reg_ref['metrics']['r2_test'])}, RMSE {fmt(reg_ref['metrics']['rmse_test'])}, "
                f"MAE {fmt(reg_ref['metrics']['mae_test'])} y OOB R² {fmt(reg_ref['metrics']['oob_score'])}. "
                f"Su validación cruzada de RMSE es {fmt_pm(reg_ref['metrics']['cv_rmse_mean'], reg_ref['metrics']['cv_rmse_std'])} "
                f"y la de R² es {fmt_pm(reg_ref['metrics']['cv_r2_mean'], reg_ref['metrics']['cv_r2_std'])}. "
                f"La mejor cantidad de árboles por RMSE de prueba es {reg_best['name']}: RMSE {fmt(reg_best['metrics']['rmse_test'])} "
                f"y R² {fmt(reg_best['metrics']['r2_test'])}. "
                f"La reducción de RMSE frente al árbol individual es {fmt(reg_tree['metrics']['rmse_test'] - reg_best['metrics']['rmse_test'])} mpg "
                f"({fmt(rel_drop(reg_tree['metrics']['rmse_test'], reg_best['metrics']['rmse_test']), 2)}% del RMSE del árbol). "
                f"**En regresión el mejor es {reg_winner}.** "
                f"{t_test_sentence(reg_best, reg_tree, 'cv_r2_folds', 'R²')}"
            ),
            diagnostics_paragraph(reg_ref),
            "\n".join(
                [
                    "El promedio de B predictores con varianza σ² y correlación ρ tiene varianza ρσ² + (1 − ρ)σ² / B. Con independencia, ρ = 0 y la varianza queda en σ² / B. El muestreo bootstrap con reemplazo genera muestras distintas; promediar los B árboles profundos deja el sesgo cerca del de un solo árbol y recorta la varianza en la medida en que esas muestras no produzcan los mismos errores. La correlación estimada arriba muestra que los árboles no son independientes, así que la ganancia empírica es real y menor que la del caso ρ = 0.",
                    "",
                    MATH_BAGGING.strip(),
                ]
            ),
        ]
    )
    return "\n\n".join(
        [
            "## 6. Ítem 4. Bagging clasificador y regresor frente al modelo individual",
            "### Enunciado oficial",
            f"> {item['question']}",
            "### Diseño del experimento y configuración de hiperparámetros",
            design,
            "### Resultados",
            "Escenario A. Clasificación en cáncer de mama:",
            "",
            clf_table(clf_models, with_oob=True),
            "",
            "Escenario B. Regresión en auto_mpg:",
            "",
            reg_table(reg_models, with_oob=True),
            "",
            image_line(
                "plots/item4_bagging.png",
                "Gráfico del ítem 4: ganancia del bagging y convergencia al aumentar el número de árboles",
            ),
            "### Respuesta Fundamentada: ¿Cuál es mejor?",
            answer,
        ]
    )


def section_item5(item) -> str:
    clf_models = item["classification"]["models"]
    reg_models = item["regression"]["models"]
    clf_winner = find_model(clf_models, item["classification"]["conclusion"]["winner"])
    clf_runner = find_model(clf_models, item["classification"]["conclusion"]["runner_up"])
    reg_winner = find_model(reg_models, item["regression"]["conclusion"]["winner"])
    reg_runner = find_model(reg_models, item["regression"]["conclusion"]["runner_up"])
    rf_clf = find_model(clf_models, "Random Forest optimizado")
    xgb_clf = find_model(clf_models, "XGBoost optimizado")
    rf_reg = find_model(reg_models, "Random Forest optimizado")
    xgb_reg = find_model(reg_models, "XGBoost optimizado")
    design = "\n".join(
        [
            "En clasificación se comparan `RandomForestClassifier(n_estimators=100, random_state=42)` y `XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='logloss', random_state=42)`, cada uno en versión base y en versión elegida por `GridSearchCV`. La búsqueda de Random Forest cubre número de árboles, profundidad, tamaño mínimo de hoja y `max_features`. La de XGBoost cubre número de árboles, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree` y `reg_lambda`. El criterio interno es el ROC-AUC.",
            "",
            "En regresión el esquema es el mismo con `RandomForestRegressor` y `XGBRegressor`, sobre la matriz ya codificada de auto_mpg. El Random Forest base de regresión usa el valor por omisión de scikit-learn, `max_features=1.0` (todas las variables). El de clasificación usa `sqrt`. La búsqueda de regresión puede elegir `sqrt` o la mitad de las variables si eso baja el RMSE. El criterio interno es el RMSE.",
            "",
            "Clasificación:",
            "",
            params_table(clf_models),
            "",
            "Regresión:",
            "",
            params_table(reg_models),
        ]
    )
    answer = "\n\n".join(
        [
            "Random Forest promedia árboles profundos entrenados en paralelo. Cada árbol ve una muestra bootstrap y, en clasificación, un subconjunto de variables de tamaño cercano a la raíz cuadrada de p. Ese submuestreo baja la correlación entre árboles y, con ella, la varianza del promedio. XGBoost suma árboles en serie. En cada paso usa una aproximación de Taylor de segundo orden de la pérdida y penaliza la norma de los pesos de las hojas con regularización L2 (`reg_lambda`) y, si se activa, L1. Por eso puede reducir sesgo y varianza a la vez, a cambio de más sensibilidad al learning rate y a la profundidad.",
            (
                f"**En clasificación gana {clf_winner['name']}.** ROC-AUC de prueba {fmt(clf_winner['metrics']['roc_auc_test'])}, "
                f"F1 {fmt(clf_winner['metrics']['f1_test'])}, accuracy {fmt(clf_winner['metrics']['accuracy_test'])} "
                f"y validación cruzada {fmt_pm(clf_winner['metrics']['cv_accuracy_mean'], clf_winner['metrics']['cv_accuracy_std'])}. "
                f"{clf_runner['name']} queda en ROC-AUC {fmt(clf_runner['metrics']['roc_auc_test'])} "
                f"(diferencia de {fmt_pp(clf_winner['metrics']['roc_auc_test'], clf_runner['metrics']['roc_auc_test'])} puntos)."
                f"{auc_margin_sentence(clf_winner['metrics']['roc_auc_test'], clf_runner['metrics']['roc_auc_test'])} "
                f"El ajuste final de {clf_winner['name']} tardó {fmt(clf_winner['metrics']['fit_time_ms'], 2)} ms y el de {clf_runner['name']} "
                f"{fmt(clf_runner['metrics']['fit_time_ms'], 2)} ms. "
                f"{agree_sentence(item['classification']['conclusion'], 'ROC-AUC')} "
                f"{t_test_sentence(clf_winner, clf_runner, 'cv_roc_auc_folds', 'ROC-AUC')}"
            ),
            (
                f"Las importancias relativas del Random Forest optimizado en clasificación se concentran en {importance_phrase(rf_clf)}. "
                f"Las de XGBoost optimizado se concentran en {importance_phrase(xgb_clf)}. "
                f"Ambas escalas están normalizadas para sumar 1; no son intercambiables en unidades brutas, y tampoco son efectos causales."
            ),
            diagnostics_paragraph(find_model(clf_models, "Random Forest base")),
            (
                f"**En regresión gana {reg_winner['name']}.** RMSE de prueba {fmt(reg_winner['metrics']['rmse_test'])} mpg, "
                f"R² de prueba {fmt(reg_winner['metrics']['r2_test'])}, MAE {fmt(reg_winner['metrics']['mae_test'])} "
                f"y validación cruzada de R² {fmt_pm(reg_winner['metrics']['cv_r2_mean'], reg_winner['metrics']['cv_r2_std'])}. "
                f"{reg_runner['name']} tiene RMSE {fmt(reg_runner['metrics']['rmse_test'])} "
                f"({fmt(reg_runner['metrics']['rmse_test'] - reg_winner['metrics']['rmse_test'])} mpg más) "
                f"y R² {fmt(reg_runner['metrics']['r2_test'])}."
                f"{rmse_margin_sentence(reg_winner['metrics']['rmse_test'], reg_runner['metrics']['rmse_test'])} "
                f"El ajuste final de {reg_winner['name']} tardó {fmt(reg_winner['metrics']['fit_time_ms'], 2)} ms y el de {reg_runner['name']} "
                f"{fmt(reg_runner['metrics']['fit_time_ms'], 2)} ms. "
                f"{agree_sentence(item['regression']['conclusion'], 'R²')} "
                f"{t_test_sentence(reg_winner, reg_runner, 'cv_r2_folds', 'R²')}"
            ),
            (
                f"En regresión, las importancias relativas del Random Forest optimizado priorizan {importance_phrase(rf_reg)}. "
                f"Las de XGBoost optimizado priorizan {importance_phrase(xgb_reg)}."
            ),
            (
                f"Conviene Random Forest cuando se quiere un ajuste paralelo, con poca sensibilidad a un learning rate y con el submuestreo de variables ya activo. En clasificación la configuración optimizada fue {rf_clf['params_text']}. "
                f"Conviene XGBoost cuando una búsqueda de learning rate, profundidad, submuestreo y `reg_lambda` mejora de forma clara la prueba. La mejor configuración de regresión que encontró esa búsqueda es {xgb_reg['params_text']}. "
                f"Cuando la diferencia de la métrica principal es pequeña, el tiempo de ajuste y la simplicidad de un bosque paralelo deciden el uso práctico. "
                f"Las búsquedas costaron {fmt(rf_clf['search']['search_time_s'], 2)} s (Random Forest, clasificación), "
                f"{fmt(xgb_clf['search']['search_time_s'], 2)} s (XGBoost, clasificación), "
                f"{fmt(rf_reg['search']['search_time_s'], 2)} s (Random Forest, regresión) y "
                f"{fmt(xgb_reg['search']['search_time_s'], 2)} s (XGBoost, regresión)."
            ),
        ]
    )
    return "\n\n".join(
        [
            "## 7. Ítem 5. Random Forest frente a XGBoost",
            "### Enunciado oficial",
            f"> {item['question']}",
            "### Diseño del experimento y configuración de hiperparámetros",
            design,
            "### Resultados",
            "Clasificación en cáncer de mama:",
            "",
            clf_table(clf_models),
            "",
            "Regresión en auto_mpg:",
            "",
            reg_table(reg_models),
            "",
            "Importancias relativas del Random Forest optimizado en clasificación:",
            "",
            md_table(
                ["Puesto", "Variable", "Importancia relativa"],
                [[str(i), row["feature"], fmt(row["importance"], 4)] for i, row in enumerate(rf_clf["importances"]["relative"][:8], start=1)],
            ),
            "",
            "Importancias relativas de XGBoost optimizado en clasificación:",
            "",
            md_table(
                ["Puesto", "Variable", "Importancia relativa"],
                [[str(i), row["feature"], fmt(row["importance"], 4)] for i, row in enumerate(xgb_clf["importances"]["relative"][:8], start=1)],
            ),
            "",
            image_line(
                "plots/item5_rf_vs_xgboost.png",
                "Gráfico del ítem 5: métricas de clasificación y regresión e importancias de Random Forest y XGBoost",
            ),
            "### Respuesta Fundamentada: ¿Cuál es mejor?",
            answer,
        ]
    )


def dataset_section(results) -> str:
    bc = results["datasets"]["breast_cancer"]
    mpg = results["datasets"]["auto_mpg"]
    n_bc = bc["n_rows"]
    mal = bc["class_counts"]["malignant"]
    ben = bc["class_counts"]["benign"]
    versions = results["meta"]["versions"]
    version_rows = [[name, versions[name]] for name in versions]
    origin_rows = [[name, str(count)] for name, count in mpg["origin_counts"].items()]
    corr_rows = [[row["feature"], fmt(row["correlation"], 4)] for row in mpg["correlations_with_mpg"]]
    return "\n\n".join(
        [
            "## 2. Metodología y entorno experimental",
            "### Arquitectura del flujo",
            "\n".join(
                [
                    "1. Cargar `breast_cancer` desde scikit-learn y `auto_mpg` desde OpenML, con respaldo en seaborn y en el archivo de UCI si la red falla. La copia limpia queda en `data/auto_mpg_clean.csv`.",
                    "2. Separar 80% de entrenamiento y 20% de prueba con `random_state=42`. En clasificación, la partición está estratificada.",
                    "3. Ajustar cada modelo solo con el entrenamiento. El escalado y el one-hot viven dentro de `Pipeline`, de modo que cada pliegue vuelve a estimarlos.",
                    "4. Elegir hiperparámetros con validación cruzada de 5 pliegues sobre el entrenamiento. El conjunto de prueba no participa en esa elección.",
                    "5. Reportar métricas de entrenamiento, de prueba y la media ± desviación estándar de los 5 pliegues (desviación con divisor N, la misma convención que `GridSearchCV`).",
                    "6. Guardar figuras, `results_metrics.json`, el informe y el notebook ejecutado.",
                ]
            ),
            "### Protocolo común",
            "\n".join(
                [
                    f"- Semilla: {results['meta']['seed']} en árboles, bosques, boosting, bagging, SVM, particiones y pliegues.",
                    "- La regresión lineal ordinaria y k-NN son deterministas y no reciben semilla.",
                    f"- Tamaño de prueba: {fmt(results['meta']['test_size'], 2)}. Pliegues: {results['meta']['n_folds']}, con `shuffle=True`.",
                    "- Clasificación: gana el mayor ROC-AUC de prueba; desempate por F1, accuracy de prueba y media de accuracy en validación cruzada.",
                    "- Regresión: gana el menor RMSE de prueba; desempate por R² de prueba y R² medio de validación cruzada.",
                    "- La validación cruzada de un modelo elegido por búsqueda está calculada en los mismos pliegues que se usaron para elegirlo, así que puede ser ligeramente optimista. La métrica de prueba permanece fuera de la selección.",
                    f"- Entorno: Python {results['meta']['python']}, {results['meta']['cpu_count']} núcleos lógicos, experimento iniciado en {results['meta']['started_at']}.",
                    f"- Duración de los experimentos, sin contar la ejecución del notebook: {fmt(results['meta']['runtime_seconds'], 2)} s.",
                ]
            ),
            "### Versiones",
            md_table(["Paquete", "Versión"], version_rows),
            "### Cáncer de mama",
            "\n".join(
                [
                    f"Fuente: `{bc['source']}`. Filas: {bc['n_rows']}. Variables numéricas: {bc['n_features']}. Valores ausentes: {bc['n_missing']}.",
                    f"Clases: maligno {mal} ({fmt(mal / n_bc, 4)}), benigno {ben} ({fmt(ben / n_bc, 4)}). La clase positiva de precisión, recall y F1 es la benigna.",
                    f"Entrenamiento: {bc['n_train']} filas (maligno {bc['train_class_counts']['malignant']}, benigno {bc['train_class_counts']['benign']}). Prueba: {bc['n_test']} filas (maligno {bc['test_class_counts']['malignant']}, benigno {bc['test_class_counts']['benign']}).",
                    "La desviación estándar descriptiva de esta tabla usa el divisor n − 1. La de la validación cruzada usa el divisor n.",
                ]
            ),
            summary_table(bc["feature_summary"], ["Media", "Desv. estándar", "Mínimo", "Máximo"]),
            "### Auto MPG",
            "\n".join(
                [
                    f"Fuente efectiva: {mpg['source']}. Filas crudas: {mpg['n_raw']}. Ausencias en `horsepower`: {mpg['n_missing_horsepower']}. Filas eliminadas: {mpg['dropped_rows']}. Filas usadas: {mpg['n_clean']}.",
                    "Se descartó el nombre del automóvil. `origin` se codificó en one-hot y la categoría de referencia del diseño es `europe`.",
                    f"MPG tiene media {fmt(mpg['mpg_mean'], 3)}, desviación estándar {fmt(mpg['mpg_std'], 3)}, mínimo {fmt(mpg['mpg_min'], 3)} y máximo {fmt(mpg['mpg_max'], 3)}.",
                    f"Entrenamiento: {mpg['n_train']} filas, media de mpg {fmt(mpg['mpg_train_mean'], 3)} (desviación {fmt(mpg['mpg_train_std'], 3)}). Prueba: {mpg['n_test']} filas, media {fmt(mpg['mpg_test_mean'], 3)} (desviación {fmt(mpg['mpg_test_std'], 3)}).",
                    f"La correlación de Pearson entre weight y displacement es {fmt(mpg['corr_weight_displacement'], 3)}.",
                ]
            ),
            "Conteos de origen después de la limpieza:",
            "",
            md_table(["Origen", "Filas"], origin_rows),
            "",
            "Correlación de Pearson con mpg:",
            "",
            md_table(["Variable", "Correlación con mpg"], corr_rows),
            "",
            summary_table(
                [mpg["target_summary"], *mpg["feature_summary"]],
                ["Media", "Desv. estándar", "Mínimo", "Máximo"],
            ),
        ]
    )


def global_section(results) -> str:
    item1 = results["item1"]["conclusion"]
    item2 = results["item2"]["conclusion"]
    item3 = results["item3"]["conclusion"]
    item4c = results["item4"]["classification"]
    item4r = results["item4"]["regression"]
    item5c = results["item5"]["classification"]["conclusion"]
    item5r = results["item5"]["regression"]["conclusion"]
    rows = [
        ["1", item1["winner"], "ROC-AUC de prueba", fmt(item1["primary_value"]), "Gini y entropía se comparan entre sí y contra la regresión logística L2."],
        ["2", item2["winner"], "RMSE de prueba (mpg)", fmt(item2["primary_value"]), "El árbol sin restricciones muestra la brecha de sobreajuste; el ganador es el menor RMSE de prueba."],
        ["3", item3["winner"], "ROC-AUC de prueba", fmt(item3["primary_value"]), "El ensamble se juzga contra el árbol y contra el mejor modelo individual."],
        ["4 clasificación", item4c["best_bagging"], "Accuracy de prueba del mejor bagging", fmt(find_model(item4c["models"], item4c["best_bagging"])["metrics"]["accuracy_test"]), "Bootstrap y promedio de árboles profundos frente a un solo árbol."],
        ["4 regresión", item4r["best_bagging"], "RMSE de prueba del mejor bagging", fmt(find_model(item4r["models"], item4r["best_bagging"])["metrics"]["rmse_test"]), "La misma comparación en millas por galón, con OOB R²."],
        ["5 clasificación", item5c["winner"], "ROC-AUC de prueba", fmt(item5c["primary_value"]), "Bosque paralelo con submuestreo de variables frente a boosting de segundo orden."],
        ["5 regresión", item5r["winner"], "RMSE de prueba (mpg)", fmt(item5r["primary_value"]), "La misma comparación sobre auto_mpg, en versión base y optimizada."],
    ]
    return "\n\n".join(
        [
            "## 8. Cuadro comparativo global y conclusiones generales",
            md_table(["Ítem", "Ganador", "Métrica principal", "Valor", "Conclusión"], rows),
            (
                "El patrón que se repite es el siguiente. Un árbol CART sin restricciones ajusta el entrenamiento hasta dejar poca o ninguna impureza y esa exactitud no se traslada completa a la prueba. "
                "La regresión regularizada, la poda, el bagging y el boosting atacan esa varianza por caminos distintos: penalización del coeficiente, eliminación de ramas, promedio de muestras bootstrap y corrección secuencial del residuo. "
                "En cáncer de mama la señal es fuerte y varios modelos superan 0.95 de ROC-AUC; la diferencia útil está en los centésimas y en el recall de la clase maligna. "
                "En auto_mpg la señal tiene una parte lineal grande (peso, potencia, cilindrada y año) y una parte de interacción que solo aparece cuando el árbol o el ensamble están regularizados. "
                f"Los ganadores declarados con la métrica de prueba son: ítem 1, {item1['winner']}; ítem 2, {item2['winner']}; ítem 3, {item3['winner']}; "
                f"ítem 4, {item4c['best_bagging']} en clasificación y {item4r['best_bagging']} en regresión; "
                f"ítem 5, {item5c['winner']} en clasificación y {item5r['winner']} en regresión."
            ),
            "### Limitaciones",
            "\n".join(
                [
                    "- Hay un solo conjunto de prueba. La validación cruzada reduce la dependencia de una partición, pero cinco pliegues dan una prueba t de baja potencia.",
                    "- La validación cruzada de los modelos sintonizados reutiliza los pliegues de la búsqueda, así que esa media puede ser optimista. La prueba no entra en la búsqueda.",
                    f"- El conjunto de prueba de cáncer de mama tiene {results['datasets']['breast_cancer']['n_test']} casos y el de auto_mpg tiene {results['datasets']['auto_mpg']['n_test']}. Un cambio de unos pocos casos mueve la tercera cifra decimal.",
                    "- Las importancias de Random Forest y de XGBoost están normalizadas para la figura. Miden contribución al criterio de cada algoritmo, no efectos causales.",
                    "- La correlación entre miembros del ensamble se calcula con sus predicciones sobre la prueba. Es una diagnóstico del desacuerdo, no la correlación teórica entre muestras de entrenamiento.",
                    "- No hay una cohorte externa ni un coste distinto para los errores malignos. El recall maligno se reporta para que esa asimetría sea visible.",
                ]
            ),
        ]
    )


def reproducibility_section() -> str:
    return "\n\n".join(
        [
            "## 9. Reproducibilidad",
            "El informe, las figuras, el JSON y el notebook salen de `src/run_experiments.py`. Volver a ejecutarlo regenera todos esos artefactos con la misma semilla.",
            "```powershell",
            "git clone https://github.com/paololuna-luw/cart-and-ensemble-benchmarks.git",
            "cd cart-and-ensemble-benchmarks",
            "python -m pip install -r requirements.txt",
            "python src/run_experiments.py",
            "```",
            "El notebook ya está ejecutado. Para regenerar sus salidas:",
            "```powershell",
            "python -m ipykernel install --user --name tarea03py --display-name \"Python 3.12 tarea03\"",
            "jupyter nbconvert --to notebook --execute Tarea_03_CART_Ensamblados.ipynb --inplace --ExecutePreprocessor.kernel_name=tarea03py",
            "```",
            "Si existe `data/auto_mpg_clean.csv`, el script lo reutiliza para que la corrida no dependa de la red. Borrar ese archivo obliga a descargar de nuevo, primero por OpenML (`data_id=196`), después por seaborn y al final por el archivo de UCI.",
            "Las tablas de este informe muestran cuatro decimales. `results_metrics.json` conserva el flotante completo de cada cálculo.",
        ]
    )


def build_readme(results) -> str:
    header = "\n\n".join(
        [
            "# Tarea de Ejecución 03: Aplicación y Evaluación Empírica de Modelos CART y Ensamblados",
            "## 1. Resumen ejecutivo",
            (
                "Este informe compara árboles CART y ensambles sobre dos tareas tabulares, con una sola semilla y un protocolo común de partición y validación cruzada. "
                f"En el ítem 1 el ganador es {results['item1']['conclusion']['winner']} "
                f"(ROC-AUC de prueba {fmt(results['item1']['conclusion']['primary_value'])}). "
                f"En el ítem 2 el ganador es {results['item2']['conclusion']['winner']} "
                f"(RMSE de prueba {fmt(results['item2']['conclusion']['primary_value'])} mpg). "
                f"En el ítem 3 el ganador es {results['item3']['conclusion']['winner']} "
                f"(ROC-AUC de prueba {fmt(results['item3']['conclusion']['primary_value'])}). "
                f"En el ítem 4 el mejor bagging de clasificación es {results['item4']['classification']['best_bagging']} "
                f"y el de regresión es {results['item4']['regression']['best_bagging']}. "
                f"En el ítem 5, la clasificación la gana {results['item5']['classification']['conclusion']['winner']} "
                f"y la regresión la gana {results['item5']['regression']['conclusion']['winner']}."
            ),
            (
                "Todos los números salen de la ejecución de `src/run_experiments.py` con `random_state=42`. "
                "No hay métricas interpoladas ni celdas pendientes. El detalle de cada contraste está en la sección del ítem, junto a la figura y a la respuesta de cuál modelo es mejor."
            ),
        ]
    )
    parts = [
        header,
        dataset_section(results),
        section_item1(results["item1"]),
        section_item2(results["item2"], results["datasets"]["auto_mpg"]),
        section_item3(results["item3"]),
        section_item4(results["item4"]),
        section_item5(results["item5"]),
        global_section(results),
        reproducibility_section(),
    ]
    return "\n\n".join(parts) + "\n"


def audit_readme(text: str) -> None:
    checks = [
        (r"\bNone\b", "None"),
        (r"\bNaN\b", "NaN"),
        (r"\bnan\b", "nan"),
    ]
    for pattern, label in checks:
        match = re.search(pattern, text)
        if match:
            start = max(0, match.start() - 120)
            end = min(len(text), match.end() + 120)
            raise RuntimeError(f"El README contiene {label}: {text[start:end]}")
    # "TODO" se busca en mayúsculas: el informe en español usa "todos" y "todo".
    for pattern, label in (
        (r"\bTODO\b", "TODO"),
        (r"\bFIXME\b", "FIXME"),
        (r"(?i)\bplaceholder\b", "placeholder"),
        (r"(?i)insertar valor", "insertar valor"),
        (r"(?i)lorem ipsum", "lorem ipsum"),
        (r"(?i)\[insert", "[insert"),
    ):
        if re.search(pattern, text):
            raise RuntimeError(f"El README contiene el marcador {label}")
    for figure in (
        "plots/item1_breast_cancer.png",
        "plots/item2_auto_mpg.png",
        "plots/item3_ensemble_clf.png",
        "plots/item4_bagging.png",
        "plots/item5_rf_vs_xgboost.png",
    ):
        if figure not in text:
            raise RuntimeError(f"El README no incluye {figure}")
    if text.count("### Respuesta Fundamentada: ¿Cuál es mejor?") != 5:
        raise RuntimeError("El README no tiene las cinco respuestas fundamentadas")


def assert_png(path: Path) -> None:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"PNG invalido: {path}")
    if len(data) < 8000:
        raise RuntimeError(f"PNG demasiado pequeno ({len(data)} bytes): {path}")


def build_and_execute_notebook(results) -> None:
    log("Preparando el notebook")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            "tarea03py",
            "--display-name",
            "Python 3.12 tarea03",
        ]
    )
    sections = [
        ("item1", "plots/item1_breast_cancer.png", "item1"),
        ("item2", "plots/item2_auto_mpg.png", "item2"),
        ("item3", "plots/item3_ensemble_clf.png", "item3"),
        ("item5clf", "plots/item5_rf_vs_xgboost.png", "item5"),
    ]
    cells = [
        new_markdown_cell(
            "# Tarea de Ejecución 03: Aplicación y Evaluación Empírica de Modelos CART y Ensamblados\n\n"
            "Notebook ejecutado. Las celdas leen `results_metrics.json`, rehacen dos ajustes de control "
            "y muestran las tablas y las figuras que respaldan el informe."
        ),
        new_code_cell(
            "\n".join(
                [
                    "import json",
                    "import math",
                    "import sys",
                    "from pathlib import Path",
                    "",
                    "import pandas as pd",
                    "from IPython.display import Image, display",
                    "",
                    "ROOT = Path.cwd().resolve()",
                    "sys.path.insert(0, str(ROOT / 'src'))",
                    "results = json.loads((ROOT / 'results_metrics.json').read_text(encoding='utf-8'))",
                    "print('Semilla:', results['meta']['seed'])",
                    "print('Python:', results['meta']['python'])",
                    "print('Inicio del experimento:', results['meta']['started_at'])",
                    "print('Ganador item 1:', results['item1']['conclusion']['winner'])",
                    "print('Ganador item 2:', results['item2']['conclusion']['winner'])",
                    "print('Ganador item 3:', results['item3']['conclusion']['winner'])",
                    "print('Mejor bagging clasificador:', results['item4']['classification']['best_bagging'])",
                    "print('Mejor bagging regresor:', results['item4']['regression']['best_bagging'])",
                    "print('Ganador item 5 clasificacion:', results['item5']['classification']['conclusion']['winner'])",
                    "print('Ganador item 5 regresion:', results['item5']['regression']['conclusion']['winner'])",
                ]
            )
        ),
        new_markdown_cell(dataset_section(results)),
        new_markdown_cell(section_item1(results["item1"])),
        new_code_cell(
            "\n".join(
                [
                    "def show_models(models):",
                    "    frame = pd.DataFrame(",
                    "        [",
                    "            {",
                    "                'modelo': model['name'],",
                    "                'accuracy_test': model['metrics'].get('accuracy_test'),",
                    "                'f1_test': model['metrics'].get('f1_test'),",
                    "                'roc_auc_test': model['metrics'].get('roc_auc_test'),",
                    "                'r2_test': model['metrics'].get('r2_test'),",
                    "                'rmse_test': model['metrics'].get('rmse_test'),",
                    "                'cv_accuracy': model['metrics'].get('cv_accuracy_mean'),",
                    "                'cv_r2': model['metrics'].get('cv_r2_mean'),",
                    "            }",
                    "            for model in models",
                    "        ]",
                    "    )",
                    "    display(frame.round(4).fillna('no aplica'))",
                    "",
                    "show_models(results['item1']['models'])",
                    "display(Image(filename='plots/item1_breast_cancer.png'))",
                ]
            )
        ),
        new_markdown_cell(section_item2(results["item2"], results["datasets"]["auto_mpg"])),
        new_code_cell(
            "\n".join(
                [
                    "show_models(results['item2']['models'])",
                    "display(Image(filename='plots/item2_auto_mpg.png'))",
                ]
            )
        ),
        new_markdown_cell(section_item3(results["item3"])),
        new_code_cell(
            "\n".join(
                [
                    "show_models(results['item3']['models'])",
                    "display(Image(filename='plots/item3_ensemble_clf.png'))",
                ]
            )
        ),
        new_markdown_cell(section_item4(results["item4"])),
        new_code_cell(
            "\n".join(
                [
                    "print('Clasificacion')",
                    "show_models(results['item4']['classification']['models'])",
                    "print('Regresion')",
                    "show_models(results['item4']['regression']['models'])",
                    "display(Image(filename='plots/item4_bagging.png'))",
                ]
            )
        ),
        new_markdown_cell(section_item5(results["item5"])),
        new_code_cell(
            "\n".join(
                [
                    "print('Clasificacion')",
                    "show_models(results['item5']['classification']['models'])",
                    "print('Regresion')",
                    "show_models(results['item5']['regression']['models'])",
                    "display(Image(filename='plots/item5_rf_vs_xgboost.png'))",
                ]
            )
        ),
        new_markdown_cell(global_section(results) + "\n\n" + reproducibility_section()),
        new_code_cell(
            "\n".join(
                [
                    "import run_experiments as exp",
                    "from sklearn.metrics import accuracy_score, r2_score",
                    "from sklearn.tree import DecisionTreeClassifier",
                    "",
                    "breast = exp.load_breast_cancer_data()",
                    "x_train, x_test, y_train, y_test = exp.split_classification(breast['X'], breast['y'])",
                    "print('breast_cancer', breast['X'].shape, 'train', len(y_train), 'test', len(y_test))",
                    "tree = DecisionTreeClassifier(criterion='gini', random_state=exp.SEED)",
                    "tree.fit(x_train, y_train)",
                    "acc = float(accuracy_score(y_test, tree.predict(x_test)))",
                    "expected_acc = next(model['metrics']['accuracy_test'] for model in results['item1']['models'] if model['name'] == 'Gini')",
                    "print('accuracy Gini', acc, expected_acc)",
                    "if abs(acc - expected_acc) > 1e-12:",
                    "    raise AssertionError('El arbol Gini no reproduce el JSON')",
                    "",
                    "mpg = exp.load_auto_mpg()",
                    "x_train, x_test, y_train, y_test = exp.split_regression(mpg['X'], mpg['y'])",
                    "print('auto_mpg', mpg['X'].shape, 'train', len(y_train), 'test', len(y_test))",
                    "shallow = exp.make_mpg_tree(max_depth=2)",
                    "shallow.fit(x_train, y_train)",
                    "score = float(r2_score(y_test, shallow.predict(x_test)))",
                    "expected_r2 = next(model['metrics']['r2_test'] for model in results['item2']['models'] if model['name'] == 'Árbol superficial')",
                    "print('R2 arbol superficial', score, expected_r2)",
                    "if abs(score - expected_r2) > 1e-10:",
                    "    raise AssertionError('El arbol superficial no reproduce el JSON')",
                    "print('Los dos ajustes de control coinciden con el JSON.')",
                ]
            )
        ),
        new_code_cell(
            "\n".join(
                [
                    "def walk(obj):",
                    "    if isinstance(obj, float):",
                    "        if not math.isfinite(obj):",
                    "            raise AssertionError('metrica no finita')",
                    "    elif isinstance(obj, dict):",
                    "        for value in obj.values():",
                    "            walk(value)",
                    "    elif isinstance(obj, list):",
                    "        for value in obj:",
                    "            walk(value)",
                    "",
                    "walk(results)",
                    "readme = Path('README.md').read_text(encoding='utf-8')",
                    "import re",
                    "if re.search(r'\\bNone\\b', readme) or re.search(r'\\bNaN\\b|\\bnan\\b', readme):",
                    "    raise AssertionError('README con marcador vacio')",
                    "for figure in ['plots/item1_breast_cancer.png', 'plots/item2_auto_mpg.png', 'plots/item3_ensemble_clf.png', 'plots/item4_bagging.png', 'plots/item5_rf_vs_xgboost.png']:",
                    "    if not Path(figure).exists():",
                    "        raise AssertionError(figure)",
                    "print('JSON finito, README limpio y cinco figuras presentes.')",
                    "print('Respuestas en el README:', readme.count('### Respuesta Fundamentada: ¿Cuál es mejor?'))",
                ]
            )
        ),
    ]
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {
            "display_name": "Python 3.12 tarea03",
            "language": "python",
            "name": "tarea03py",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    })
    path = ROOT / "Tarea_03_CART_Ensamblados.ipynb"
    nbformat.write(nb, path)
    client = NotebookClient(nb, timeout=900, kernel_name="tarea03py")
    log("Ejecutando el notebook")
    try:
        client.execute(cwd=str(ROOT))
    finally:
        nbformat.write(nb, path)
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    if len(code_cells) < 7:
        raise RuntimeError("El notebook no tiene todas las celdas de codigo")
    for index, cell in enumerate(code_cells):
        if not cell.get("outputs"):
            raise RuntimeError(f"La celda de codigo {index} no tiene salidas")
        for output in cell["outputs"]:
            if output.get("output_type") == "error":
                raise RuntimeError(f"Error en la celda {index}: {output.get('ename')}: {output.get('evalue')}")
    log(f"Notebook ejecutado: {path.name}")


def installed_versions() -> dict:
    names = [
        "scikit-learn",
        "xgboost",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "nbformat",
        "nbconvert",
        "nbclient",
        "ipykernel",
        "jupyter",
        "scipy",
    ]
    return {name: pkg_version(name) for name in names}


def main() -> None:
    os.chdir(ROOT)
    PLOTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    style_plots()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    started = time.perf_counter()
    log("Cargando breast_cancer")
    breast = load_breast_cancer_data()
    x_train, x_test, y_train, y_test = split_classification(breast["X"], breast["y"])
    breast["meta"]["n_train"] = int(len(y_train))
    breast["meta"]["n_test"] = int(len(y_test))
    breast["meta"]["train_class_counts"] = class_counts(y_train)
    breast["meta"]["test_class_counts"] = class_counts(y_test)
    bc = {
        "X_train": x_train,
        "X_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "cv": make_cv_clf(),
        "feature_names": breast["meta"]["feature_names"],
        "meta": breast["meta"],
    }
    log("Cargando auto_mpg")
    mpg_data = load_auto_mpg()
    mx_train, mx_test, my_train, my_test = split_regression(mpg_data["X"], mpg_data["y"])
    mpg_data["meta"]["n_train"] = int(len(my_train))
    mpg_data["meta"]["n_test"] = int(len(my_test))
    mpg_data["meta"]["mpg_train_mean"] = float(my_train.mean())
    mpg_data["meta"]["mpg_test_mean"] = float(my_test.mean())
    mpg_data["meta"]["mpg_train_std"] = float(my_train.std(ddof=1))
    mpg_data["meta"]["mpg_test_std"] = float(my_test.std(ddof=1))
    mpg = {
        "X_train": mx_train,
        "X_test": mx_test,
        "y_train": my_train,
        "y_test": my_test,
        "cv": make_cv_reg(),
        "meta": mpg_data["meta"],
    }
    log("Item 1")
    item1 = run_item1(bc["X_train"], bc["y_train"], bc["X_test"], bc["y_test"], bc["cv"])
    log("Item 2")
    item2 = run_item2(mpg["X_train"], mpg["y_train"], mpg["X_test"], mpg["y_test"], mpg["cv"])
    log("Item 3")
    item3 = run_item3(bc["X_train"], bc["y_train"], bc["X_test"], bc["y_test"], bc["cv"])
    log("Item 4")
    item4 = run_item4(bc, mpg)
    log("Item 5")
    item5 = run_item5(bc, mpg)
    runtime = float(time.perf_counter() - started)
    results = {
        "meta": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "n_folds": N_FOLDS,
            "python": sys.version.split()[0],
            "versions": installed_versions(),
            "cpu_count": int(os.cpu_count() or 1),
            "started_at": started_at,
            "runtime_seconds": runtime,
        },
        "datasets": {
            "breast_cancer": bc["meta"],
            "auto_mpg": mpg["meta"],
        },
        "item1": item1,
        "item2": item2,
        "item3": item3,
        "item4": item4,
        "item5": item5,
    }
    results = to_builtin(results)
    assert_finite_tree(results)
    readme = build_readme(results)
    audit_readme(readme)
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    (ROOT / "results_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    for name in (
        "item1_breast_cancer.png",
        "item2_auto_mpg.png",
        "item3_ensemble_clf.png",
        "item4_bagging.png",
        "item5_rf_vs_xgboost.png",
    ):
        assert_png(PLOTS / name)
    build_and_execute_notebook(results)
    log(f"LISTO en {runtime:.1f} s de experimentos")


if __name__ == "__main__":
    main()

