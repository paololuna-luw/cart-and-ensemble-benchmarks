# Tarea de Ejecución 03: Aplicación y Evaluación Empírica de Modelos CART y Ensamblados

## Configuración del experimento

| Campo | Contenido |
| --- | --- |
| Código por pregunta | `src/codigo_entrenamiento_p1.py` a `src/codigo_entrenamiento_p5.py` |
| Corrida completa | `src/run_experiments.py` |
| Cuaderno | `cuaderno.ipynb` |
| Métricas | `results_metrics.json` |
| Semilla | 42 |
| Partición | 80% entrenamiento / 20% prueba. En clasificación, estratificada. |
| Validación | 5 pliegues sobre el entrenamiento. Se reporta media ± desviación estándar. |
| Entorno | Python 3.12.10, scikit-learn 1.9.0, XGBoost 3.4.1 |
| Datos | breast_cancer: 569 filas y 30 variables. auto_mpg: 392 filas tras eliminar 6 ausencias de horsepower. |

### Criterio de decisión (igual en las cinco preguntas)

| Tipo de tarea | Gana | Desempate 1 | Desempate 2 | Desempate 3 |
| --- | --- | --- | --- | --- |
| Clasificación | Mayor ROC-AUC de prueba | F1 de prueba | Accuracy de prueba | Accuracy media de validación cruzada |
| Regresión | Menor RMSE de prueba | R² de prueba | R² medio de validación cruzada | — |

Precisión, recall y F1 toman como clase positiva la etiqueta benigna.

## Cuadro de dictámenes

Resumen de las cinco respuestas. El desarrollo de cada fila está en la sección enlazada.

| Pregunta | Indicación | Respuesta | Evidencia principal |
| --- | --- | --- | --- |
| [1](#pregunta-1) | Gini y entropía frente a regresión logística | Regresión logística | ROC-AUC de prueba 0.9954 |
| [2](#pregunta-2) | Árboles de regresión frente a regresión lineal | Árbol error absoluto | RMSE de prueba 2.6448 mpg |
| [3](#pregunta-3) | Ensamble clasificador frente a modelo individual | Regresión logística | ROC-AUC de prueba 0.9954 |
| [4](#pregunta-4) | Bagging frente al modelo individual | Bagging (200), en clasificación y en regresión | Accuracy 0.9474; RMSE 2.3911 mpg |
| [5](#pregunta-5) | Random Forest frente a XGBoost | Clasificación: Random Forest base. Regresión: Random Forest optimizado | ROC-AUC 0.9937; RMSE 2.3960 mpg |

<a id="pregunta-1"></a>

## Pregunta 1. Árboles de decisión: Gini, entropía y regresión logística

### Indicación

> Entrene modelos de árboles de decisión con criterios de impureza diferentes (índice gini y entropía) usando el dataset “breast_cancer” de scikit-learn, compare su eficiencia con respecto a un modelo lineal como el Regresor Logístico. ¿Cuál es mejor?

### Respuesta Fundamentada: ¿Cuál es mejor?

**Respuesta: es mejor Regresión logística.**

**Ranking por ROC-AUC de prueba**

| Puesto | Modelo | Tipo | ROC-AUC prueba | F1 prueba | Acc. train | Acc. prueba | Profundidad | Hojas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Regresión logística | Lineal | 0.9954 | 0.9861 | 0.9890 | 0.9825 | no aplica | no aplica |
| 2 | Gini podado | Árbol podado | 0.9578 | 0.9185 | 0.9473 | 0.9035 | 2 | 3 |
| 3 | Entropía podada | Árbol podado | 0.9392 | 0.9130 | 0.9604 | 0.8947 | 3 | 5 |
| 4 | Entropía | Árbol sin poda | 0.9206 | 0.9275 | 1.0000 | 0.9123 | 6 | 16 |
| 5 | Gini | Árbol sin poda | 0.9157 | 0.9286 | 1.0000 | 0.9123 | 7 | 19 |

**Datos de apoyo de la regresión logística**

| Dato | Valor |
| --- | --- |
| Validación cruzada (accuracy) | 0.9780 ± 0.0098 |
| Errores en la prueba | 2 de 114 casos |
| Recall de la clase maligna | 0.9762 |
| Ventaja de ROC-AUC sobre Gini podado (el mejor árbol) | 3.75 puntos |

**Por qué**

- Los árboles sin poda memorizan el entrenamiento (accuracy de entrenamiento 1.0000 en ambos) y pierden exactitud en la prueba.
- La poda reduce esa memorización: Gini podado baja a 3 hojas y sube su ROC-AUC de 0.9157 a 0.9578.
- Con 30 variables citológicas, la frontera lineal con penalización L2 separa las clases con menos varianza que los cortes ortogonales de un árbol CART individual.

### Métricas

| Modelo | Acc. train | Acc. test | Precisión | Recall | F1 | ROC-AUC | Recall maligno | CV accuracy (media ± desv.) | Profundidad | Hojas | Tiempo (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gini | 1.0000 | 0.9123 | 0.9559 | 0.9028 | 0.9286 | 0.9157 | 0.9286 | 0.9165 ± 0.0179 | 7 | 19 | 11.51 |
| Gini podado | 0.9473 | 0.9035 | 0.9841 | 0.8611 | 0.9185 | 0.9578 | 0.9762 | 0.9297 ± 0.0215 | 2 | 3 | 11.82 |
| Entropía | 1.0000 | 0.9123 | 0.9697 | 0.8889 | 0.9275 | 0.9206 | 0.9524 | 0.9297 ± 0.0237 | 6 | 16 | 13.83 |
| Entropía podada | 0.9604 | 0.8947 | 0.9545 | 0.8750 | 0.9130 | 0.9392 | 0.9286 | 0.9275 ± 0.0164 | 3 | 5 | 7.36 |
| Regresión logística | 0.9890 | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 | 0.9762 | 0.9780 ± 0.0098 | no aplica | no aplica | 12.91 |

### Gráfico

![Gráfico de la pregunta 1: métricas, matrices de confusión y curvas ROC](plots/item1_breast_cancer.png)

### Código correspondiente

Entrenamiento en `src/codigo_entrenamiento_p1.py`. Modelos: `DecisionTreeClassifier(criterion='gini')`, `DecisionTreeClassifier(criterion='entropy')`, sus versiones con `ccp_alpha` elegido por validación cruzada, y `Pipeline` de `StandardScaler` con `LogisticRegression`.

<a id="pregunta-2"></a>

## Pregunta 2. Árboles de regresión con varias configuraciones y regresión lineal

### Indicación

> Entrene modelos de árboles de decisión con varias configuraciones de hiper parámetros usando el dataset auto_mpg, comparar su eficiencia con respecto a un modelo lineal como el Regresor Lineal. ¿Cuál es el mejor?

### Respuesta Fundamentada: ¿Cuál es mejor?

**Respuesta: el mejor es Árbol error absoluto.**

**Ranking por RMSE de prueba (menor es mejor)**

| Puesto | Modelo | RMSE prueba (mpg) | R² prueba | MAE prueba | R² train | CV R² (media ± desv.) | Profundidad | Hojas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Árbol error absoluto | 2.6448 | 0.8629 | 2.0905 | 0.8966 | 0.8060 ± 0.0417 | 5 | 29 |
| 2 | Árbol profundidad controlada | 3.0772 | 0.8145 | 2.2782 | 0.8859 | 0.8036 ± 0.0550 | 4 | 15 |
| 3 | Árbol GridSearchCV | 3.1086 | 0.8107 | 2.0481 | 0.9254 | 0.8296 ± 0.0408 | 8 | 53 |
| 4 | Árbol poda CCP | 3.2131 | 0.7977 | 2.2556 | 0.9336 | 0.7947 ± 0.0655 | 7 | 23 |
| 5 | Regresión lineal | 3.2561 | 0.7923 | 2.4620 | 0.8287 | 0.8188 ± 0.0143 | no aplica | no aplica |
| 6 | Árbol sin restricciones | 3.3700 | 0.7775 | 2.2962 | 1.0000 | 0.7656 ± 0.0975 | 16 | 263 |
| 7 | Árbol superficial | 4.6114 | 0.5834 | 3.4993 | 0.7573 | 0.7153 ± 0.0699 | 2 | 4 |

**Hallazgos clave**

| Tema | Dato |
| --- | --- |
| Sobreajuste del árbol sin restricciones | R² de 1.0000 en entrenamiento cae a 0.7775 en prueba (profundidad 16, 263 hojas) |
| Árbol demasiado simple | Profundidad 2: RMSE 4.6114 mpg, el peor de la tabla |
| Mejor punto de la curva de `max_depth` | Profundidad 6: RMSE 3.0364 mpg, R² 0.8194 |
| Árbol error absoluto frente a regresión lineal | RMSE 0.6113 mpg menor (2.6448 frente a 3.2561) |
| Validación cruzada de R² | Favorece a Árbol GridSearchCV (0.8296 ± 0.0408). El dictamen usa el RMSE de prueba. |

**Por qué**

- El árbol regularizado parte el espacio de peso, potencia, cilindrada y año en regiones con distinta pendiente de consumo.
- Esa interacción queda fuera de un único hiperplano lineal.

### Métricas

| Modelo | R² train | R² test | RMSE test | MAE test | MSE test | CV R² (media ± desv.) | CV RMSE (media ± desv.) | Profundidad | Hojas | Tiempo (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Árbol sin restricciones | 1.0000 | 0.7775 | 3.3700 | 2.2962 | 11.3567 | 0.7656 ± 0.0975 | 3.6871 ± 0.5824 | 16 | 263 | 10.86 |
| Árbol superficial | 0.7573 | 0.5834 | 4.6114 | 3.4993 | 21.2654 | 0.7153 ± 0.0699 | 4.1464 ± 0.2577 | 2 | 4 | 9.36 |
| Árbol profundidad controlada | 0.8859 | 0.8145 | 3.0772 | 2.2782 | 9.4690 | 0.8036 ± 0.0550 | 3.4382 ± 0.2963 | 4 | 15 | 8.07 |
| Árbol error absoluto | 0.8966 | 0.8629 | 2.6448 | 2.0905 | 6.9952 | 0.8060 ± 0.0417 | 3.4442 ± 0.3331 | 5 | 29 | 13.39 |
| Árbol poda CCP | 0.9336 | 0.7977 | 3.2131 | 2.2556 | 10.3238 | 0.7947 ± 0.0655 | 3.4987 ± 0.2658 | 7 | 23 | 11.04 |
| Árbol GridSearchCV | 0.9254 | 0.8107 | 3.1086 | 2.0481 | 9.6632 | 0.8296 ± 0.0408 | 3.2216 ± 0.3100 | 8 | 53 | 9.78 |
| Regresión lineal | 0.8287 | 0.7923 | 3.2561 | 2.4620 | 10.6023 | 0.8188 ± 0.0143 | 3.3671 ± 0.4179 | no aplica | no aplica | 13.28 |

### Gráfico

![Gráfico de la pregunta 2: R², RMSE y curva de validación de max_depth](plots/item2_auto_mpg.png)

### Código correspondiente

Entrenamiento en `src/codigo_entrenamiento_p2.py`. Seis configuraciones de `DecisionTreeRegressor` (sin restricciones, profundidad 2, profundidad controlada, error absoluto, poda por complejidad de costo y `GridSearchCV`) frente a `Pipeline` de `StandardScaler` con `LinearRegression`.

<a id="pregunta-3"></a>

## Pregunta 3. Ensamble clasificador frente a modelo individual

### Indicación

> Entrene un modelo ensemble clasificador, y compare su eficiencia con respecto a un modelo individual. ¿Cuál es mejor?

### Respuesta Fundamentada: ¿Cuál es mejor?

**Respuesta: es mejor Regresión logística.**

**Ranking por ROC-AUC de prueba**

| Puesto | Modelo | Tipo | ROC-AUC prueba | F1 prueba | Acc. prueba | CV accuracy (media ± desv.) | Tiempo (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Regresión logística | Individual | 0.9954 | 0.9861 | 0.9825 | 0.9780 ± 0.0098 | 12.53 |
| 2 | SVC | Individual | 0.9950 | 0.9861 | 0.9825 | 0.9692 ± 0.0146 | 37.25 |
| 3 | Stacking | Ensamble | 0.9917 | 0.9793 | 0.9737 | 0.9780 ± 0.0098 | 507.46 |
| 4 | Voting suave | Ensamble | 0.9911 | 0.9793 | 0.9737 | 0.9758 ± 0.0128 | 74.44 |
| 5 | Gradient Boosting | Ensamble | 0.9907 | 0.9660 | 0.9561 | 0.9516 ± 0.0149 | 634.48 |
| 6 | AdaBoost | Ensamble | 0.9825 | 0.9660 | 0.9561 | 0.9670 ± 0.0197 | 253.39 |
| 7 | k-NN | Individual | 0.9788 | 0.9655 | 0.9561 | 0.9626 ± 0.0112 | 6.09 |
| 8 | Árbol de decisión | Individual | 0.9157 | 0.9286 | 0.9123 | 0.9165 ± 0.0179 | 13.11 |
| sin ROC-AUC | Voting duro | Ensamble | no aplica | 0.9861 | 0.9825 | 0.9670 ± 0.0120 | 66.50 |

**Contrastes clave**

| Contraste | Diferencia |
| --- | --- |
| Mejor ensamble (Stacking) frente al árbol individual | ROC-AUC +0.0761 y accuracy +6.14 puntos porcentuales |
| Mejor individual (Regresión logística) frente a SVC | ROC-AUC +0.03 puntos. Margen ajustado (menor que 0.005): leerlo junto al F1, la accuracy y el tiempo de ajuste. |

**Por qué**

- La votación, el apilamiento y el boosting corrigen la varianza del árbol: el árbol individual (accuracy de entrenamiento 1.0000) queda último.
- Aun así, en este conjunto la frontera lineal regularizada sigue por encima de esas combinaciones.

### Métricas

| Modelo | Acc. train | Acc. test | Precisión | Recall | F1 | ROC-AUC | Recall maligno | CV accuracy (media ± desv.) | Profundidad | Hojas | Tiempo (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Árbol de decisión | 1.0000 | 0.9123 | 0.9559 | 0.9028 | 0.9286 | 0.9157 | 0.9286 | 0.9165 ± 0.0179 | 7 | 19 | 13.11 |
| SVC | 0.9824 | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9950 | 0.9762 | 0.9692 ± 0.0146 | no aplica | no aplica | 37.25 |
| k-NN | 0.9736 | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9788 | 0.9286 | 0.9626 ± 0.0112 | no aplica | no aplica | 6.09 |
| Regresión logística | 0.9890 | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 | 0.9762 | 0.9780 ± 0.0098 | no aplica | no aplica | 12.53 |
| Voting duro | 0.9868 | 0.9825 | 0.9861 | 0.9861 | 0.9861 | no aplica | 0.9762 | 0.9670 ± 0.0120 | no aplica | no aplica | 66.50 |
| Voting suave | 0.9934 | 0.9737 | 0.9726 | 0.9861 | 0.9793 | 0.9911 | 0.9524 | 0.9758 ± 0.0128 | no aplica | no aplica | 74.44 |
| Stacking | 0.9912 | 0.9737 | 0.9726 | 0.9861 | 0.9793 | 0.9917 | 0.9524 | 0.9780 ± 0.0098 | no aplica | no aplica | 507.46 |
| AdaBoost | 1.0000 | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9825 | 0.9048 | 0.9670 ± 0.0197 | no aplica | no aplica | 253.39 |
| Gradient Boosting | 1.0000 | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9907 | 0.9048 | 0.9516 ± 0.0149 | no aplica | no aplica | 634.48 |

### Gráfico

![Gráfico de la pregunta 3: modelos individuales y ensambles de clasificación](plots/item3_ensemble_clf.png)

### Código correspondiente

Entrenamiento en `src/codigo_entrenamiento_p3.py`. Individuales: árbol, SVC con probabilidades, k-NN y regresión logística. Ensambles: `VotingClassifier` duro y suave, `StackingClassifier`, `AdaBoostClassifier` y `GradientBoostingClassifier`.

<a id="pregunta-4"></a>

## Pregunta 4. Bagging clasificador y regresor frente al modelo individual

### Indicación

> Entrene un modelo bagging regresor o clasificador  y compare su eficiencia con respecto a un modelo individual. ¿Cuál es el mejor?

### Respuesta Fundamentada: ¿Cuál es mejor?

**Respuesta: el mejor es el bagging, Bagging (200), en clasificación y en regresión.**

**Clasificación en cáncer de mama**

| Modelo | Acc. prueba | F1 prueba | ROC-AUC prueba | OOB accuracy | CV accuracy (media ± desv.) |
| --- | --- | --- | --- | --- | --- |
| Árbol individual | 0.9123 | 0.9286 | 0.9157 | no aplica | 0.9165 ± 0.0179 |
| Bagging (100) | 0.9386 | 0.9517 | 0.9927 | 0.9560 | 0.9516 ± 0.0192 |
| **Bagging (200)** | **0.9474** | **0.9583** | 0.9924 | **0.9648** | 0.9516 ± 0.0215 |

Ganancia de Bagging (200) sobre el árbol: accuracy +3.51 puntos porcentuales.

**Regresión en auto_mpg**

| Modelo | R² train | R² prueba | RMSE prueba (mpg) | MAE prueba | OOB R² |
| --- | --- | --- | --- | --- | --- |
| Árbol individual | 1.0000 | 0.7775 | 3.3700 | 2.2962 | no aplica |
| Bagging (100) | 0.9809 | 0.8871 | 2.4004 | 1.7112 | 0.8676 |
| **Bagging (200)** | 0.9821 | **0.8880** | **2.3911** | **1.6713** | **0.8692** |

Ganancia de Bagging (200) sobre el árbol: RMSE −0.9789 mpg (−29.05%).

**Por qué: varianza del promedio de B árboles**

| Caso | Varianza del promedio |
| --- | --- |
| Miembros con correlación ρ entre sí | ρσ² + (1 − ρ)σ² / B |
| Miembros independientes (ρ = 0) | σ² / B |

- El muestreo bootstrap con reemplazo y el promedio de árboles profundos reducen la varianza del predictor.
- La correlación estimada entre los miembros muestra que comparten errores, y aun así el bagging mejora la prueba en las dos tareas.

### Métricas

Clasificación en cáncer de mama:

| Modelo | Acc. train | Acc. test | Precisión | Recall | F1 | ROC-AUC | Recall maligno | CV accuracy (media ± desv.) | Profundidad | Hojas | Tiempo (ms) | OOB accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Árbol individual | 1.0000 | 0.9123 | 0.9559 | 0.9028 | 0.9286 | 0.9157 | 0.9286 | 0.9165 ± 0.0179 | 7 | 19 | 12.01 | no aplica |
| Bagging (10) | 0.9956 | 0.9298 | 0.9444 | 0.9444 | 0.9444 | 0.9899 | 0.9048 | 0.9495 ± 0.0112 | no aplica | no aplica | 90.64 | 0.9297 |
| Bagging (50) | 1.0000 | 0.9386 | 0.9452 | 0.9583 | 0.9517 | 0.9922 | 0.9048 | 0.9516 ± 0.0204 | no aplica | no aplica | 552.93 | 0.9516 |
| Bagging (100) | 1.0000 | 0.9386 | 0.9452 | 0.9583 | 0.9517 | 0.9927 | 0.9048 | 0.9516 ± 0.0192 | no aplica | no aplica | 894.52 | 0.9560 |
| Bagging (200) | 1.0000 | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9924 | 0.9286 | 0.9516 ± 0.0215 | no aplica | no aplica | 1805.80 | 0.9648 |

Regresión en auto_mpg:

| Modelo | R² train | R² test | RMSE test | MAE test | MSE test | CV R² (media ± desv.) | CV RMSE (media ± desv.) | Profundidad | Hojas | Tiempo (ms) | OOB R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Árbol individual | 1.0000 | 0.7775 | 3.3700 | 2.2962 | 11.3567 | 0.7656 ± 0.0975 | 3.6871 ± 0.5824 | 16 | 263 | 12.66 | no aplica |
| Bagging (10) | 0.9710 | 0.8534 | 2.7358 | 1.9773 | 7.4845 | 0.8592 ± 0.0309 | 2.9265 ± 0.1796 | no aplica | no aplica | 57.30 | 0.8349 |
| Bagging (50) | 0.9795 | 0.8848 | 2.4252 | 1.7292 | 5.8814 | 0.8684 ± 0.0305 | 2.8336 ± 0.2845 | no aplica | no aplica | 236.20 | 0.8615 |
| Bagging (100) | 0.9809 | 0.8871 | 2.4004 | 1.7112 | 5.7617 | 0.8653 ± 0.0326 | 2.8649 ± 0.2940 | no aplica | no aplica | 453.20 | 0.8676 |
| Bagging (200) | 0.9821 | 0.8880 | 2.3911 | 1.6713 | 5.7172 | 0.8636 ± 0.0324 | 2.8839 ± 0.2961 | no aplica | no aplica | 986.01 | 0.8692 |

### Gráfico

![Gráfico de la pregunta 4: ganancia del bagging y convergencia al aumentar el número de árboles](plots/item4_bagging.png)

### Código correspondiente

Entrenamiento en `src/codigo_entrenamiento_p4.py`. `BaggingClassifier` y `BaggingRegressor` con 10, 50, 100 y 200 árboles, `bootstrap=True` y `oob_score=True`, frente al árbol individual de cada tarea.

<a id="pregunta-5"></a>

## Pregunta 5. Random Forest frente a XGBoost

### Indicación

> Entrene un modelo RandomForest y compare su eficiencia con respecto a un modelo XGBoost. ¿Cuál es mejor?

### Respuesta Fundamentada: ¿Cuál es mejor?

**Respuesta: en clasificación es mejor Random Forest base. En regresión es mejor Random Forest optimizado.**

**Clasificación en cáncer de mama (ranking por ROC-AUC; desempate por F1 y accuracy)**

| Puesto | Modelo | ROC-AUC prueba | F1 prueba | Acc. prueba | CV accuracy (media ± desv.) |
| --- | --- | --- | --- | --- | --- |
| 1 | Random Forest base | 0.9937 | 0.9655 | 0.9561 | 0.9626 ± 0.0179 |
| 2 | XGBoost base | 0.9937 | 0.9589 | 0.9474 | 0.9648 ± 0.0162 |
| 3 | Random Forest optimizado | 0.9927 | 0.9655 | 0.9561 | 0.9560 ± 0.0120 |
| 4 | XGBoost optimizado | 0.9914 | 0.9660 | 0.9561 | 0.9736 ± 0.0226 |

- El ROC-AUC de prueba de los dos primeros empata (diferencia de 0.00 puntos), así que el desempate lo resuelven el F1 y la accuracy.
- El mejor XGBoost de la tarea es XGBoost base (ROC-AUC 0.9937, F1 0.9589).
- La validación cruzada de ROC-AUC favorece a XGBoost optimizado.

**Regresión en auto_mpg (ranking por RMSE de prueba, menor es mejor)**

| Puesto | Modelo | RMSE prueba (mpg) | R² prueba | MAE prueba | CV R² (media ± desv.) |
| --- | --- | --- | --- | --- | --- |
| 1 | Random Forest optimizado | 2.3960 | 0.8875 | 1.7061 | 0.8667 ± 0.0304 |
| 2 | Random Forest base | 2.3966 | 0.8875 | 1.7064 | 0.8667 ± 0.0305 |
| 3 | XGBoost optimizado | 2.5265 | 0.8749 | 1.8821 | 0.8671 ± 0.0373 |
| 4 | XGBoost base | 2.7145 | 0.8556 | 1.8818 | 0.8370 ± 0.0445 |

- El margen entre Random Forest optimizado y base es ajustado: diferencia relativa de 0.02%, por debajo del 1%.
- El mejor XGBoost de regresión es XGBoost optimizado (RMSE 2.5265 mpg, R² 0.8749).
- La validación cruzada de R² favorece a XGBoost optimizado.

**Por qué: cómo construye cada método sus árboles**

| Método | Forma de combinar los árboles | Efecto |
| --- | --- | --- |
| Random Forest | Promedia árboles profundos en paralelo. En clasificación cada árbol ve un subconjunto de variables de tamaño cercano a la raíz cuadrada de p. | Descorrelaciona los árboles y reduce la varianza |
| XGBoost | Suma árboles en serie con aproximación de Taylor de segundo orden. | Regularización L2 sobre las hojas |

### Métricas

Clasificación en cáncer de mama:

| Modelo | Acc. train | Acc. test | Precisión | Recall | F1 | ROC-AUC | Recall maligno | CV accuracy (media ± desv.) | Profundidad | Hojas | Tiempo (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest base | 1.0000 | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9937 | 0.9286 | 0.9626 ± 0.0179 | no aplica | no aplica | 361.36 |
| Random Forest optimizado | 0.9934 | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9927 | 0.9286 | 0.9560 ± 0.0120 | no aplica | no aplica | 866.65 |
| XGBoost base | 1.0000 | 0.9474 | 0.9459 | 0.9722 | 0.9589 | 0.9937 | 0.9048 | 0.9648 ± 0.0162 | no aplica | no aplica | 1997.92 |
| XGBoost optimizado | 1.0000 | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9914 | 0.9048 | 0.9736 ± 0.0226 | no aplica | no aplica | 176.28 |

Regresión en auto_mpg:

| Modelo | R² train | R² test | RMSE test | MAE test | MSE test | CV R² (media ± desv.) | CV RMSE (media ± desv.) | Profundidad | Hojas | Tiempo (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest base | 0.9809 | 0.8875 | 2.3966 | 1.7064 | 5.7435 | 0.8667 ± 0.0305 | 2.8535 ± 0.2911 | no aplica | no aplica | 407.54 |
| Random Forest optimizado | 0.9809 | 0.8875 | 2.3960 | 1.7061 | 5.7409 | 0.8667 ± 0.0304 | 2.8534 ± 0.2907 | no aplica | no aplica | 109.46 |
| XGBoost base | 0.9987 | 0.8556 | 2.7145 | 1.8818 | 7.3683 | 0.8370 ± 0.0445 | 3.1265 ± 0.1852 | no aplica | no aplica | 41.88 |
| XGBoost optimizado | 0.9361 | 0.8749 | 2.5265 | 1.8821 | 6.3831 | 0.8671 ± 0.0373 | 2.8296 ± 0.2216 | no aplica | no aplica | 17.55 |

### Gráfico

![Gráfico de la pregunta 5: métricas de clasificación, regresión e importancias](plots/item5_rf_vs_xgboost.png)

### Código correspondiente

Entrenamiento en `src/codigo_entrenamiento_p5.py`. `RandomForestClassifier`, `RandomForestRegressor`, `XGBClassifier` y `XGBRegressor`, cada uno en versión base y en versión elegida por `GridSearchCV`.

## Conclusiones

Las cinco indicaciones quedan respondidas con la métrica de prueba y el mismo protocolo de partición y validación cruzada.

| Pregunta | Ganador | Evidencia | Nota |
| --- | --- | --- | --- |
| 1 | Regresión logística | ROC-AUC de prueba 0.9954 | Entre los árboles sin poda, Entropía generaliza con menos hojas. Los dos llegan a accuracy de entrenamiento 1.0000. |
| 2 | Árbol error absoluto | RMSE de prueba 2.6448 mpg, R² 0.8629 | El árbol sin restricciones cae de R² 1.0000 (train) a 0.7775 (prueba); su RMSE, 3.3700 mpg, supera al de la regresión lineal, 3.2561 mpg. |
| 3 | Regresión logística | ROC-AUC 0.9954 | Stacking supera al árbol individual (0.9157) y alcanza 0.9917. |
| 4 | Bagging (200) | Clasificación: accuracy 0.9474. Regresión: RMSE 2.3911 mpg | Mejora al árbol individual en las dos tareas. |
| 5 | Clasificación: Random Forest base. Regresión: Random Forest optimizado | ROC-AUC 0.9937, F1 0.9655. RMSE 2.3960 mpg | Empate práctico en ROC-AUC; decide el F1 y la accuracy. |

**Lectura general**

- Un árbol CART sin restricciones ajusta el entrenamiento hasta dejar poca o ninguna impureza, y esa exactitud se reduce en la prueba.
- La regresión con penalización L2, la poda, el bagging y el boosting atacan esa varianza por caminos distintos.
- En cáncer de mama la señal es fuerte y varios modelos superan 0.95 de ROC-AUC.
- En auto_mpg hay una parte lineal grande y una parte de interacción que aparece cuando el árbol o el ensamble están regularizados.
- El detalle de cada contraste, con pruebas t, matrices de confusión, coeficientes e importancias, está en `cuaderno.ipynb`.

### Limitaciones

- La prueba de cáncer de mama tiene 114 casos y la de auto_mpg tiene 79. Unos pocos casos mueven la tercera cifra decimal.
- La validación cruzada de un modelo elegido por búsqueda usa los mismos pliegues de la búsqueda, así que esa media puede ser optimista. La prueba no entra en la selección.
- Con cinco pliegues, la prueba t tiene poca potencia. El dictamen usa la métrica de prueba.
- Las importancias comparan la contribución interna de cada algoritmo. No son efectos causales.

## Ubicación del material

| Requisito de la indicación | Dónde se evidencia |
| --- | --- |
| Respuesta a cada pregunta | Secciones Pregunta 1 a Pregunta 5 de este README |
| Resumen de hallazgos | Cuadro de dictámenes y sección de conclusiones |
| Métricas de entrenamiento, prueba y validación cruzada | Tabla de cada pregunta y `results_metrics.json` |
| Gráficos comparativos | Carpeta `plots/` e imágenes incrustadas bajo cada pregunta |
| Código de los modelos entrenados | `src/codigo_entrenamiento_p1.py` a `src/codigo_entrenamiento_p5.py` y `src/run_experiments.py` |
| Detalle de datos, pruebas e hiperparámetros | `cuaderno.ipynb` |

Los modelos de cada pregunta están en `src/codigo_entrenamiento_p1.py` a `src/codigo_entrenamiento_p5.py`. La corrida completa está en `src/run_experiments.py`. El cuaderno con las métricas ya calculadas es `cuaderno.ipynb`.

| Pregunta | Archivo de código | Datos | Figura |
| --- | --- | --- | --- |
| 1 | `src/codigo_entrenamiento_p1.py` | `load_breast_cancer` | `plots/item1_breast_cancer.png` |
| 2 | `src/codigo_entrenamiento_p2.py` | auto_mpg limpio, 392 filas | `plots/item2_auto_mpg.png` |
| 3 | `src/codigo_entrenamiento_p3.py` | `load_breast_cancer` | `plots/item3_ensemble_clf.png` |
| 4 | `src/codigo_entrenamiento_p4.py` | breast_cancer y auto_mpg | `plots/item4_bagging.png` |
| 5 | `src/codigo_entrenamiento_p5.py` | breast_cancer y auto_mpg | `plots/item5_rf_vs_xgboost.png` |

### Cómo ejecutarlo

Para regenerar métricas, figuras y este README:

```powershell
git clone https://github.com/paololuna-luw/cart-and-ensemble-benchmarks.git
cd cart-and-ensemble-benchmarks
python -m pip install -r requirements.txt
python src/run_experiments.py
```

Para entrenar una sola pregunta, desde la raíz del repositorio:

```powershell
python src/codigo_entrenamiento_p1.py
python src/codigo_entrenamiento_p2.py
python src/codigo_entrenamiento_p3.py
python src/codigo_entrenamiento_p4.py
python src/codigo_entrenamiento_p5.py
```

Experimento de referencia iniciado el 2026-10-20 15:59:47. Duración de los ajustes, sin el notebook: 236.86 s.
