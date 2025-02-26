#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: regression_analysis.py
But: Analyse de régression (lazyregressor et automl) sur DF2023-22-21_Concat_Finale_2.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lazypredict.Supervised import LazyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor

from tpot import TPOTRegressor

# Charger dataset
df = pd.read_csv("DF2023-22-21_Concat_Finale_2.csv")
print(df.head())
print(df.info())

# Séparer X et y
X = df.drop(['Ewltp (g/km)', 'Cn', 'year'], axis=1)
y = df['Ewltp (g/km)']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Comparaison modèles (LazyRegressor)
regressors_test = {
    LinearRegression: LinearRegression(),
    ExtraTreesRegressor: ExtraTreesRegressor(),
    RandomForestRegressor: RandomForestRegressor(),
    BaggingRegressor: BaggingRegressor(),
    KNeighborsRegressor: KNeighborsRegressor(),
}
reg = LazyRegressor(verbose=0, ignore_warnings=True, regressors=regressors_test)
models, predictions = reg.fit(X_train_scaled, X_test_scaled, y_train, y_test)
print(models)

# Optimisation pipeline (TPOT)
tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42)
tpot.fit(X_train_scaled, y_train)
print("TPOT score:", tpot.score(X_test_scaled, y_test))
tpot.export('tpot_best_pipeline_Test.py')

# Modèle final: RandomForestRegressor
model_final = RandomForestRegressor(bootstrap=False, max_features=0.75, min_samples_leaf=1,
                                    min_samples_split=9, n_estimators=100, random_state=42)
model_final.fit(X_train_scaled, y_train)
results_model_final = model_final.predict(X_test_scaled)

# Calcul RMSE
mse_mf = mean_squared_error(y_test, results_model_final)
rmse_mf = np.sqrt(mse_mf)
print("RMSE:", rmse_mf)

# Nuage de points: prédictions vs réelles
plt.figure()
plt.scatter(results_model_final, y_test, alpha=0.5)
plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color='red', lw=2)
plt.xlabel("Prédictions")
plt.ylabel("Réelles")
plt.title("Prédictions vs Réelles")
plt.show()

# Analyse erreurs
df_results_final = pd.DataFrame({'y_true': y_test, 'y_pred': results_model_final})
df_results_final['error'] = abs(df_results_final['y_true'] - df_results_final['y_pred'])
seuil = 20
outliers = df_results_final[df_results_final['error'] > seuil]
print(outliers.describe())
