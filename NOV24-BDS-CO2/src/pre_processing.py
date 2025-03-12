#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import des librairies générales : 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import des librairies spécifiques à l'enregistrement du fichier final : 
import os
import json
from datetime import datetime

# Chargement du dataset :
df = pd.read_csv('data.csv', low_memory=False)
pd.set_option('display.max_columns', None)
print(df.head(10))
print(df.info())
print(df.describe())

# Vérification des valeurs manquantes :
print(df.isna().mean().round(5) * 100)

# Suppression de colonnes inutiles :
cols_to_drop = ['ID', 'Country', 'VFN', 'MMS', 'Tan', 'T', 'Va', 'Ct', 'Cr', 'r', 'Mt',
                'Enedc (g/km)', 'W (mm)', 'At1 (mm)', 'At2 (mm)', 'Ernedc (g/km)', 'De',
                'Vf', 'Status', 'Date of registration', 'RLFI', 'Ve']
df = df.drop(cols_to_drop, axis=1)
df = df.drop(['Mp', 'Man', 'Mh'], axis=1)

# Suppression de "IT" et "ech" :
print(df['IT'].unique())
df = df.drop('IT', axis=1)
print(df['ech'].unique())
df = df.drop('ech', axis=1)

# Traitement de "Fm" et "Ft" :
print(df['Fm'].unique())
df = df.drop('Fm', axis=1)
print(df['Ft'].unique())
df['Ft'] = df['Ft'].str.lower()
print(df['Ft'].unique())
print(df[df['Ft'] == 'unknown'])
df = df[df['Ft'] != 'unknown']

# Regroupement des carburants :
dico_fuel = {
    'petrol': 'Essence',
    'hydrogen': 'Essence',
    'e85': 'Essence',
    'lpg': 'Essence',
    'ng': 'Essence',
    'ng-biomethane': 'Essence',
    'diesel': 'Diesel',
    'petrol/electric': 'Hybride',
    'diesel/electric': 'Hybride',
    'electric': 'Electrique'
}
df['Ft'] = df['Ft'].replace(dico_fuel)
df = df[df['Ft'] != 'Electrique']

# Gestion des valeurs manquantes :
print(df.isna().sum())
print(df.isna().mean().round(5) * 100)
print(df[df['z (Wh/km)'].isna()])
count_nan_z = df[(df['Ft'] == 'Hybride') & (df['z (Wh/km)'].isna())].shape[0]
print(count_nan_z)
df = df.drop('z (Wh/km)', axis=1)
print(df[df['Electric range (km)'].isna()])
count_nan_er = df[(df['Ft'] == 'Hybride') & (df['Electric range (km)'].isna())].shape[0]
print(count_nan_er)
df = df.drop('Electric range (km)', axis=1)
df = df.dropna(subset=['Ewltp (g/km)', 'Mk', 'Cn', 'ec (cm3)', 'ep (KW)', 'm (kg)', 'Fuel consumption ', 'Erwltp (g/km)'])

# Traitement de "Mk"

# Affichage des valeurs uniques :
print(df['Mk'].unique())

# Passage en majuscules : 
df['Mk'] = df['Mk'].str.upper()

# Liste des marques les plus répandues en Europe : 
target_brands = ['CITROEN', 'FORD', 'FIAT', 'RENAULT', 'MERCEDES', 'BMW', 'VOLKSWAGEN', 'ALPINE', 
                 'INEOS', 'LAMBORGHINI', 'TOYOTA', 'JAGUAR', 'GREAT WALL MOTOR', 'CATERHAM', 'PEUGEOT', 
                 'MAN', 'OPEL', 'ALLIED VEHICLES', 'IVECO', 'MITSUBISHI', 'DS', 'MAZDA', 'SUZUKI', 
                 'SUBARU', 'HYUNDAI', "AUDI", "NISSAN", "SKODA", "SEAT", "DACIA", "VOLVO", "KIA",
                 "LAND ROVER", "MINI", "PORSCHE", "ALFA ROMEO", "SMART", "LANCIA", "JEEP"
                 ]

# Fonction pour extraire les marques connues des chaînes de caractères : 
def extract_brand(value):
    for brand in target_brands:
        if brand in value:
            return brand
    return value
df['Mk'] = df['Mk'].apply(extract_brand)

# Correction des fautes de frappe : 
dico_marque = {
    'DS': 'CITROEN',
    'VW': 'VOLKSWAGEN',
    '?KODA': 'SKODA',
    'ŠKODA': 'SKODA',
    'PSA AUTOMOBILES SA': 'PEUGEOT',
    'FCA ITALY': 'FIAT',
    'ALFA  ROMEO': 'ALFA ROMEO',
    'LANDROVER': 'LAND ROVER'
}
df['Mk'] = df['Mk'].replace(dico_marque)

# Suppression des marques trop peu connues : 

brands_to_delete = ['TRIPOD', 'API CZ', 'MOTO STAR', 'REMOLQUES RAMIREZ', 'AIR-BRAKES', 
                    'SIN MARCA', 'WAVECAMPER', 'CASELANI', 'PANDA']
df = df[~df['Mk'].isin(brands_to_delete)]
print(df[df['Mk'].isin(brands_to_delete)])

# Suppression des occurences trop faibles : 

def filter_brands(df, col='Mk', threshold=5):
    brands = df[col].tolist()
    unique_brands = df[col].unique().tolist()
    filtered_brands = [brand for brand in unique_brands if brands.count(brand) >= threshold]
    return filtered_brands

filtered_brands = filter_brands(df, col='Mk', threshold=5)
df = df[df['Mk'].isin(filtered_brands)]


# Suppression des doublons :
nbr_doublons = df.duplicated().sum()
taille_dataset = len(df)
print(nbr_doublons, (nbr_doublons / taille_dataset) * 100)
df_clean = df.drop_duplicates()
print(df_clean.info())

# Détection et suppression d'outliers :
def detect_outliers(df, target_col, group_cols=["Cn", "Ft", "year"]):
    stats = df.groupby(group_cols).agg(**{f'{target_col}_mean': (target_col, 'mean')}).reset_index()
    df_merged = pd.merge(df, stats, on=group_cols, how="left")
    diff_col = f"diff_{target_col}"
    df_merged[diff_col] = (df_merged[target_col] - df_merged[f"{target_col}_mean"]).abs()
    q1 = df_merged[diff_col].quantile(0.25)
    q3 = df_merged[diff_col].quantile(0.75)
    iqr = q3 - q1
    seuil = (q3 + 1.5 * iqr).round(1)
    nb_outliers = len(df_merged[df_merged[diff_col] >= seuil])
    print(nb_outliers)
    df_clean_no_outliers = df_merged[df_merged[diff_col] <= seuil]
    print(len(df_clean_no_outliers))
    return df_clean_no_outliers

columns_to_filter = ['Ewltp (g/km)', 'Fuel consumption ', 'm (kg)', 'ec (cm3)', 'ep (KW)', 'Erwltp (g/km)']
df_temp = df_clean.copy()
for col in columns_to_filter:
    print(col)
    df_temp = detect_outliers(df_temp, col)
print(len(df_temp))
df_clean_no_outliers_final = df_temp
print(df_clean_no_outliers_final.head())
df_clean_no_outliers_final = df_clean_no_outliers_final[['Mk', 'Cn', 'm (kg)', 'Ewltp (g/km)', 'Ft', 'ec (cm3)', 
                                                         'ep (KW)', 'Erwltp (g/km)', 'year', 'Fuel consumption ']]

# Mise de côté des modèles hybrides trop peu représentés : 
df_clean_no_outliers_final = df_clean_no_outliers_final[df_clean_no_outliers_final['Ft'] != 'Hybride']

# Encodage des variables catégorielles :

# Encodage de "Ft" :
df_clean_no_outliers_final = pd.get_dummies(df_clean_no_outliers_final, columns=['Ft'], prefix='Ft', drop_first=False)
bool_cols = df_clean_no_outliers_final.select_dtypes(include=['bool']).columns
df_clean_no_outliers_final[bool_cols] = df_clean_no_outliers_final[bool_cols].astype(int)

# Encodage de "Mk" : 

df_clean_no_outliers_final = pd.get_dummies(df_clean_no_outliers_final, columns=['Mk'], prefix='Mk', drop_first=False)
bool_cols = df_clean_no_outliers_final.select_dtypes(include=['bool']).columns
df_clean_no_outliers_final[bool_cols] = df_clean_no_outliers_final[bool_cols].astype(int)

# Enregistrement du dataset de façon dynamique :

# Création d'un dossier "metadata":
metadata_dir = "metadata"
os.makedirs(metadata_dir, exist_ok=True) 

# Génération d'un timestamp au format YYYYMMDD_HHMMSS :
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"DF_Processed_{timestamp}.csv"

# Enregistrement du DataFrame dans le fichier avec le nom dynamique : 
df_clean_no_outliers_final.to_csv(output_filename, index=False)

# Remplissage du fichier de métadonnées : 
metadata_file = os.path.join(metadata_dir, "metadata.json")
metadata = {"processed_data": output_filename}
with open(metadata_file, "w") as f:
    json.dump(metadata, f)
