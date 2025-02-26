#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du dataset
df = pd.read_csv('data.csv', low_memory=False)
pd.set_option('display.max_columns', None)
print(df.head(10))
print(df.info())
print(df.describe())

# Vérification des valeurs manquantes
print(df.isna().mean().round(5) * 100)

# Suppression de colonnes inutiles
cols_to_drop = ['ID', 'Country', 'VFN', 'MMS', 'Tan', 'T', 'Va', 'Ct', 'Cr', 'r', 'Mt',
                'Enedc (g/km)', 'W (mm)', 'At1 (mm)', 'At2 (mm)', 'Ernedc (g/km)', 'De',
                'Vf', 'Status', 'Date of registration', 'RLFI', 'Ve']
df = df.drop(cols_to_drop, axis=1)
df = df.drop(['Mp', 'Man', 'Mh'], axis=1)

# Suppression de "IT" et "ech"
print(df['IT'].unique())
df = df.drop('IT', axis=1)
print(df['ech'].unique())
df = df.drop('ech', axis=1)

# Traitement de "Fm" et "Ft"
print(df['Fm'].unique())
df = df.drop('Fm', axis=1)
print(df['Ft'].unique())
df['Ft'] = df['Ft'].str.lower()
print(df['Ft'].unique())
print(df[df['Ft'] == 'unknown'])
df = df[df['Ft'] != 'unknown']

# Regroupement des carburants
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

# Gestion des valeurs manquantes
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
print(df['Mk'].unique())
df['Mk'] = df['Mk'].str.upper()
target_brands = ['CITROEN', 'FORD', 'FIAT', 'RENAULT', 'MERCEDES', 'BMW', 'VOLKSWAGEN', 'ALPINE', 
                 'INEOS', 'LAMBORGHINI', 'TOYOTA', 'JAGUAR', 'GREAT WALL MOTOR', 'CATERHAM', 'PEUGEOT', 
                 'MAN', 'OPEL', 'ALLIED VEHICLES', 'IVECO', 'MITSUBISHI', 'DS', 'MAZDA', 'SUZUKI', 
                 'SUBARU', 'HYUNDAI']
def extract_brand(value):
    for brand in target_brands:
        if brand in value:
            return brand
    return value
df['Mk'] = df['Mk'].apply(extract_brand)
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
print(df['Mk'].unique())
print(df['Mk'].nunique())
print(df['Mk'].value_counts())
brands_to_delete = ['TRIPOD', 'API CZ', 'MOTO STAR', 'REMOLQUES RAMIREZ', 'AIR-BRAKES', 
                    'SIN MARCA', 'WAVECAMPER', 'CASELANI', 'PANDA']
df = df[~df['Mk'].isin(brands_to_delete)]
print(df[df['Mk'].isin(brands_to_delete)])
plt.figure(figsize=(20, 6))
sns.countplot(data=df, x='Mk', order=df['Mk'].value_counts().index)
plt.xticks(rotation=45, ha='right')
plt.title('Répartition des marques')
plt.xlabel('Marques')
plt.ylabel("Occurrences")
plt.show()

# Suppression des doublons
nbr_doublons = df.duplicated().sum()
taille_dataset = len(df)
print(nbr_doublons, (nbr_doublons / taille_dataset) * 100)
df_clean = df.drop_duplicates()
print(df_clean.info())

# Boxplots des variables numériques
columns_graph = ['m (kg)', 'Ewltp (g/km)', 'ec (cm3)', 'ep (KW)', 'Erwltp (g/km)', 'Fuel consumption ']
graph_per_line = 3
n_lignes = -(-len(columns_graph) // graph_per_line)
plt.figure(figsize=(12, 4 * n_lignes))
for i, column in enumerate(columns_graph, 1):
    plt.subplot(n_lignes, graph_per_line, i)
    plt.boxplot(df_clean[column])
    plt.title(column)
    plt.ylabel('Valeurs')
plt.tight_layout()
plt.show()

# Détection et suppression d'outliers
def detect_outliers(df, target_col, group_cols=["Cn", "Ft", "year"]):
    stats = df.groupby(group_cols).agg(**{f'{target_col}_mean': (target_col, 'mean')}).reset_index()
    df_merged = pd.merge(df, stats, on=group_cols, how="left")
    diff_col = f"diff_{target_col}"
    df_merged[diff_col] = (df_merged[target_col] - df_merged[f"{target_col}_mean"]).abs()
    q1 = df_merged[diff_col].quantile(0.25)
    q3 = df_merged[diff_col].quantile(0.75)
    iqr = q3 - q1
    seuil = (q3 + 1.5 * iqr).round(1)
    plt.figure(figsize=(12, 6))
    plt.boxplot(df_merged[diff_col], vert=False, patch_artist=True,
                boxprops=dict(facecolor='skyblue', color='blue'),
                medianprops=dict(color='red'))
    plt.axvline(x=seuil, color='green', linestyle='--', linewidth=2, label=f'Seuil = {seuil}')
    plt.xlabel(diff_col)
    plt.title(f'Boxplot {diff_col}')
    plt.legend()
    plt.show()
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
plt.figure(figsize=(12, 4 * n_lignes))
for i, column in enumerate(columns_graph, 1):
    plt.subplot(n_lignes, graph_per_line, i)
    plt.boxplot(df_clean_no_outliers_final[column])
    plt.title(column)
    plt.ylabel('Valeurs')
plt.tight_layout()
plt.show()

# Distribution de "Ft"
print(df_clean_no_outliers_final['Ft'].value_counts())
plt.figure(figsize=(8, 6))
sns.countplot(x='Ft', data=df_clean_no_outliers_final)
plt.title("Distribution Ft")
plt.xlabel("Type")
plt.ylabel("Nombre")
plt.show()
df_clean_no_outliers_final = df_clean_no_outliers_final[df_clean_no_outliers_final['Ft'] != 'Hybride']
print(df_clean_no_outliers_final.info())

# Encodage des variables catégorielles
df_clean_no_outliers_final = pd.get_dummies(df_clean_no_outliers_final, columns=['Ft'], prefix='Ft', drop_first=False)
bool_cols = df_clean_no_outliers_final.select_dtypes(include=['bool']).columns
df_clean_no_outliers_final[bool_cols] = df_clean_no_outliers_final[bool_cols].astype(int)
print(df_clean_no_outliers_final.dtypes)
print(df_clean_no_outliers_final.head(10))
df_clean_no_outliers_final = pd.get_dummies(df_clean_no_outliers_final, columns=['Mk'], prefix='Mk', drop_first=False)
bool_cols = df_clean_no_outliers_final.select_dtypes(include=['bool']).columns
df_clean_no_outliers_final[bool_cols] = df_clean_no_outliers_final[bool_cols].astype(int)
print(df_clean_no_outliers_final.head())
print(df_clean_no_outliers_final.info())

# Heatmap des corrélations
columns_corr = ['m (kg)', 'Ewltp (g/km)', 'ec (cm3)', 'ep (KW)', 'Fuel consumption ', 'Erwltp (g/km)']
df_clean_no_outliers_corr = df_clean_no_outliers_final[columns_corr]
sns.heatmap(df_clean_no_outliers_corr.corr().round(2), annot=True, cmap='RdBu_r', center=0)
plt.show()

# Enregistrement du dataset
output_filename = "DF2023_Processed_2.csv"
df_clean_no_outliers_final.to_csv(output_filename, index=False)
print(output_filename)
