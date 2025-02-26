#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Définir le dossier racine (parent de src)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Dossier contenant le dataset (ici "data" à la racine)
DATA_DIR = os.path.join(BASE_DIR, "data")

# Chemin complet vers le dataset
dataset_path = os.path.join(DATA_DIR, "DF2023-22-21_Concat_Finale_2.csv")

# Chargement du dataset
df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip()
print("Colonnes disponibles :", df.columns.tolist())

target = 'Ewltp (g/km)'

# 1. Modèles avec features de base (sans marques)
baseline_features = ['m (kg)', 'ec (cm3)', 'ep (KW)', 'Erwltp (g/km)', 'Fuel consumption', 'Ft_Diesel', 'Ft_Essence']
X_baseline = df[baseline_features]
y_baseline = df[target]
X_train, X_test, y_train, y_test = train_test_split(X_baseline, y_baseline, test_size=0.2, random_state=42)

# Définir le dossier de sauvegarde des modèles
MODEL_DIR = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Pipeline Random Forest
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])
pipeline_rf.fit(X_train, y_train)
rf_pred = pipeline_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, rf_pred)
r2_rf = r2_score(y_test, rf_pred)
print("Baseline - Random Forest - MSE :", mse_rf)
print("Baseline - Random Forest - R2  :", r2_rf)
joblib.dump(pipeline_rf, os.path.join(MODEL_DIR, 'pipeline_random_forest_sm.pkl'))

# Pipeline Régression Linéaire
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
pipeline_lr.fit(X_train, y_train)
lr_pred = pipeline_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, lr_pred)
r2_lr = r2_score(y_test, lr_pred)
print("\nBaseline - Régression Linéaire - MSE :", mse_lr)
print("Baseline - Régression Linéaire - R2  :", r2_lr)
joblib.dump(pipeline_lr, os.path.join(MODEL_DIR, 'pipeline_linear_regression_sm.pkl'))

# Pipeline KNN
pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor())
])
pipeline_knn.fit(X_train, y_train)
knn_pred = pipeline_knn.predict(X_test)
mse_knn = mean_squared_error(y_test, knn_pred)
r2_knn = r2_score(y_test, knn_pred)
print("\nBaseline - KNN - MSE :", mse_knn)
print("Baseline - KNN - R2  :", r2_knn)
joblib.dump(pipeline_knn, os.path.join(MODEL_DIR, 'pipeline_knn_sm.pkl'))

# Pipeline Random Forest optimisé avec TPOT (sans marques)
pipeline_rf_tpot_sm = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(
        bootstrap=False,
        max_features=0.75,
        min_samples_leaf=1,
        min_samples_split=9,
        n_estimators=100,
        random_state=42
    ))
])
pipeline_rf_tpot_sm.fit(X_train, y_train)
rf_tpot_sm_pred = pipeline_rf_tpot_sm.predict(X_test)
mse_tpot_sm = mean_squared_error(y_test, rf_tpot_sm_pred)
r2_tpot_sm = r2_score(y_test, rf_tpot_sm_pred)
print("\nBaseline - RF optimisé (TPOT) - MSE :", mse_tpot_sm)
print("Baseline - RF optimisé (TPOT) - R2  :", r2_tpot_sm)
joblib.dump(pipeline_rf_tpot_sm, os.path.join(MODEL_DIR, 'pipeline_random_forest_opt_tpot_sm.pkl'))

# 2. Modèles avec features étendues (avec marques)
extended_features = baseline_features + [
    'Mk_ALFA ROMEO', 'Mk_ALLIED VEHICLES', 'Mk_ALPINE', 'Mk_AUDI', 'Mk_BENTLEY', 'Mk_BMW',
    'Mk_CITROEN', 'Mk_CUPRA', 'Mk_DACIA', 'Mk_FIAT', 'Mk_FORD', 'Mk_HONDA', 'Mk_HYUNDAI',
    'Mk_JAGUAR', 'Mk_JEEP', 'Mk_KIA', 'Mk_LAMBORGHINI', 'Mk_LANCIA', 'Mk_LAND ROVER',
    'Mk_LEXUS', 'Mk_MASERATI', 'Mk_MAZDA', 'Mk_MERCEDES', 'Mk_MINI', 'Mk_MITSUBISHI',
    'Mk_NISSAN', 'Mk_OPEL', 'Mk_PEUGEOT', 'Mk_PORSCHE', 'Mk_RENAULT', 'Mk_SEAT',
    'Mk_SKODA', 'Mk_SUBARU', 'Mk_SUZUKI', 'Mk_TOYOTA', 'Mk_VOLKSWAGEN', 'Mk_VOLVO',
    'Mk_MAN', 'Mk_NILSSON'
]
X_extended = df[extended_features]
y_extended = df[target]
joblib.dump(extended_features, os.path.join(MODEL_DIR, 'features_rf_opt_tpot.pkl'))
print("✅ Liste des features étendues enregistrée sous 'models/features_rf_opt_tpot.pkl'")
X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X_extended, y_extended, test_size=0.2, random_state=42)

# Pipeline Random Forest pour features étendues
pipeline_rf_ext = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(
        bootstrap=False,
        max_features=0.75,
        min_samples_leaf=1,
        min_samples_split=9,
        n_estimators=100,
        random_state=42
    ))
])
pipeline_rf_ext.fit(X_train_ext, y_train_ext)
rf_ext_pred = pipeline_rf_ext.predict(X_test_ext)
mse_rf_ext = mean_squared_error(y_test_ext, rf_ext_pred)
r2_rf_ext = r2_score(y_test_ext, rf_ext_pred)
print("\nExtended - Random Forest - MSE :", mse_rf_ext)
print("Extended - Random Forest - R2  :", r2_rf_ext)
joblib.dump(pipeline_rf_ext, os.path.join(MODEL_DIR, 'pipeline_random_forest_opt_tpot.pkl'))

# Pipeline Régression Linéaire pour features étendues
pipeline_lr_ext = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
pipeline_lr_ext.fit(X_train_ext, y_train_ext)
lr_ext_pred = pipeline_lr_ext.predict(X_test_ext)
mse_lr_ext = mean_squared_error(y_test_ext, lr_ext_pred)
r2_lr_ext = r2_score(y_test_ext, lr_ext_pred)
print("\nExtended - Régression Linéaire - MSE :", mse_lr_ext)
print("Extended - Régression Linéaire - R2  :", r2_lr_ext)
joblib.dump(pipeline_lr_ext, os.path.join(MODEL_DIR, 'pipeline_linear_regression_ext.pkl'))

# Pipeline KNN pour features étendues
pipeline_knn_ext = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor())
])
pipeline_knn_ext.fit(X_train_ext, y_train_ext)
knn_ext_pred = pipeline_knn_ext.predict(X_test_ext)
mse_knn_ext = mean_squared_error(y_test_ext, knn_ext_pred)
r2_knn_ext = r2_score(y_test_ext, knn_ext_pred)
print("\nExtended - KNN - MSE :", mse_knn_ext)
print("Extended - KNN - R2  :", r2_knn_ext)
joblib.dump(pipeline_knn_ext, os.path.join(MODEL_DIR, 'pipeline_knn_ext.pkl'))
