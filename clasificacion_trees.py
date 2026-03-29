#!/usr/bin/env python3
"""
clasificacion_trees.py

Clasificacion mediante algoritmos basados en arboles utilizando scikit-learn:
  - DecisionTree, RandomForest, ExtraTrees, GradientBoosting
  - Carga el CSV (datos.csv) — dataset de calidad del aire por municipios
  - Crea la variable objetivo 'Clase' basada en nivel de NO2
      0 = aire bueno       (NO2 <= 40 ug/m3)
      1 = aire contaminado (NO2 >  40 ug/m3)
  - Pre-procesa: codifica categoricas, elimina variable mas correlacionada
  - Imputa valores nulos con la media (necesario para GradientBoosting)
  - Divide en entrenamiento / validacion / test
  - Ajusta parametros con GridSearchCV
  - Evalua cada modelo: metricas, matriz de confusion y curvas ROC
  - Guarda resultados en resultados_clasificacion.csv
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

###############################################################################
# 1. CARGAR DATOS
###############################################################################

DATA_PATH = 'datos.csv'
df = pd.read_csv(DATA_PATH)

print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
print("Columnas:", df.columns.tolist())

###############################################################################
# 2. CREAR VARIABLE OBJETIVO
#    Clasificacion binaria basada en nivel de NO2 (OMS: limite 40 ug/m3)
#    0 = aire bueno       (NO2 <= 40)
#    1 = aire contaminado (NO2 >  40)
###############################################################################

df = df.dropna(subset=['NO2'])
df['Clase'] = (df['NO2'] > 40).astype(int)

print(f"\nDistribucion de la variable objetivo 'Clase':")
print(df['Clase'].value_counts())
print(f"  0 (aire bueno)       : {(df['Clase']==0).sum()} registros")
print(f"  1 (aire contaminado) : {(df['Clase']==1).sum()} registros")

TARGET_COL = 'Clase'

###############################################################################
# 3. PRE-PROCESAMIENTO
###############################################################################

# Codificar variables categoricas
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Eliminar la variable mas correlacionada con la variable objetivo
# (evita data leakage y redundancia)
corr_matrix = X.corrwith(y).abs().sort_values(ascending=False)
top_corr_feature = corr_matrix.index[0]
print(f'\nEliminando columna mas correlacionada con Clase: {top_corr_feature} (r={corr_matrix.iloc[0]:.3f})')
X = X.drop(columns=[top_corr_feature])

print(f"Variables predictoras finales: {X.columns.tolist()}")

# Imputar valores nulos con la media de cada columna
# GradientBoostingClassifier no acepta NaN de forma nativa
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print(f"Valores nulos tras imputacion: {X.isnull().sum().sum()}")

###############################################################################
# 4. DIVISION DE DATOS  (70% train | 15% validacion | 15% test)
###############################################################################

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nTamano de los conjuntos:")
print(f"  Entrenamiento : {len(X_train)} registros")
print(f"  Validacion    : {len(X_val)} registros")
print(f"  Test          : {len(X_test)} registros")

###############################################################################
# 5. DEFINICION DE MODELOS Y BUSQUEDA DE PARAMETROS (GridSearchCV)
###############################################################################

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
    print(f'  Mejores parametros : {grid.best_params_}')
    print(f'  Mejor ROC-AUC CV   : {grid.best_score_:.4f}')

###############################################################################
# 6. FUNCION DE EVALUACION
###############################################################################

def evaluate(model, X, y, dataset_name='Validation'):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    print(f'\n--- {dataset_name} ---')
    print(classification_report(y, y_pred))

    # Matriz de confusion
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bueno', 'Contaminado'],
                yticklabels=['Bueno', 'Contaminado'])
    plt.title(f'Matriz de confusion — {dataset_name}')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

    # Curva ROC
    if y_proba is not None:
        auc = roc_auc_score(y, y_proba)
        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='#2E7D32', lw=2, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title(f'Curva ROC — {dataset_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
    else:
        print('El modelo no tiene predict_proba; se omite la curva ROC.')

###############################################################################
# 7. EVALUACION EN CONJUNTO DE VALIDACION
###############################################################################

for name, model in best_models.items():
    evaluate(model, X_val, y_val, dataset_name=f'{name} - Validation')

###############################################################################
# 8. EVALUACION FINAL EN CONJUNTO DE TEST
###############################################################################

print('\n' + '='*60)
print('EVALUACION FINAL EN CONJUNTO DE TEST')
print('='*60)
for name, model in best_models.items():
    evaluate(model, X_test, y_test, dataset_name=f'{name} - Test')

###############################################################################
# 9. GUARDAR RESULTADOS EN CSV
###############################################################################

results = []
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    for true_label, pred_label in zip(y_test, y_pred):
        results.append({
            'Model'    : name,
            'True'     : true_label,
            'Predicted': pred_label
        })

df_results = pd.DataFrame(results)
output_path = 'resultados_clasificacion.csv'
df_results.to_csv(output_path, index=False)
print(f'\nResultados guardados en {output_path}')
print('\n¡FIN!')
