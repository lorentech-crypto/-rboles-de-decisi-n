#!/usr/bin/env python3
"""
clasificacion_trees.py

Clasificación mediante algoritmos basados en árboles (DecisionTree, RandomForest, ExtraTrees, GradientBoosting) utilizando scikit-learn:
  - Carga el CSV (datos.csv)
  - Pre‑procesa los datos: elimina la variable más correlacionada
  - Divide en test.
  - Ajusta parámetros 
  - Evalúa cada modelo y muestra métricas, matriz de confusión y curvas ROC.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# 1. Cargar datos del archivo proporcionado por la uni

DATA_PATH = 'datos.csv'   
df = pd.read_csv(DATA_PATH)


# 2. Selección de la variable objetivo y pre‑procesamiento

TARGET_COL = 'Clase'               # Ajusta al nombre real en tu CSV
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# 2.1 Eliminar variable más correlacionada con la meta
corr_matrix = X.corrwith(y).abs().sort_values(ascending=False)
top_corr_feature = corr_matrix.index[0]
print(f'Eliminando columna más correlacionada: {top_corr_feature}')
X.drop(columns=[top_corr_feature], inplace=True)


# 3. División de datos

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)


# 4. Definición de modelos y búsqueda de parámetros

models_and_params = {
    'DecisionTree': (
        DecisionTreeClassifier(random_state=42),
        {'max_depth': [5, 10, 15], 'min_samples_split': [2, 4, 6]}
    ),
    'RandomForest': (
        RandomForestClassifier(n_estimators=200, random_state=42),
        {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    ),
    'ExtraTrees': (
        ExtraTreesClassifier(n_estimators=250, random_state=42),
        {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    ),
    'GradientBoosting': (
        GradientBoostingClassifier(random_state=42),
        {'learning_rate': [0.01, 0.1, 0.2],
         'n_estimators': [100, 150, 200],
         'max_depth': [3, 5, 7]}
    )
}

best_models = {}
for name, (estimator, param_grid) in models_and_params.items():
    print(f'\n=== {name} ===')
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f'Mejores parámetros: {grid.best_params_}')
    print(f'Mejor ROC‑AUC CV: {grid.best_score_:.4f}')


# 5. Evaluación en conjunto de validación

def evaluate(model, X, y, dataset_name='Validation'):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    print(f'\n--- {dataset_name} ---')
    print(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de confusión ({dataset_name})')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.show()

    if y_proba is not None:
        auc = roc_auc_score(y, y_proba)
        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'Curva ROC ({dataset_name})')
        plt.legend(loc='lower right')
        plt.show()
    else:
        print('El modelo no tiene predict_proba; se omite la curva ROC.')

for name, model in best_models.items():
    evaluate(model, X_val, y_val, dataset_name=f'{name} - Validation')

# 6. Guardar resultados de clasificación en CSV

results = []
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    for true_label, pred_label in zip(y_test, y_pred):
        results.append({'Model': name,
                        'True': true_label,
                        'Predicted': pred_label})

df_results = pd.DataFrame(results)
output_path = 'resultados_clasificacion.csv'
df_results.to_csv(output_path, index=False)
print(f'\nResultados guardados en {output_path}')
