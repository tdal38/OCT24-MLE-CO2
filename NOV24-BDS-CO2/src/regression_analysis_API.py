#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, BaggingRegressor
import json
import os

# Chargement du nom du fichier à partir du fichier de métadonnées :
metadata_file = os.path.join("metadata", "metadata.json")
with open(metadata_file, "r") as f:
    metadata = json.load(f)
input_filename = metadata["processed_data"]

# Chargement du dataset pour la modélisation :
df = pd.read_csv(input_filename)

# Séparation de X et y :
X = df.drop(['Ewltp (g/km)', 'Cn', 'Year'], axis=1)
y = df['Ewltp (g/km)']

# Split train/test :
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation :
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle final: RandomForestRegressor
model_final = RandomForestRegressor(bootstrap=False, max_features=0.75, min_samples_leaf=1,
                                    min_samples_split=9, n_estimators=100, random_state=42)
model_final.fit(X_train_scaled, y_train)
results_model_final = model_final.predict(X_test_scaled)

# Calcul RMSE
mse_mf = mean_squared_error(y_test, results_model_final)
rmse_mf = np.sqrt(mse_mf)
print("RMSE:", rmse_mf)

# Analyse erreurs
df_results_final = pd.DataFrame({'y_true': y_test, 'y_pred': results_model_final})
df_results_final['error'] = abs(df_results_final['y_true'] - df_results_final['y_pred'])
seuil = 20
outliers = df_results_final[df_results_final['error'] > seuil]
print(outliers.describe())
