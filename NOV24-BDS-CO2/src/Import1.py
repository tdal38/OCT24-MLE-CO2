#!/usr/bin/env python
# coding: utf-8

# Importation des librairies pour la requête SQL et l'enregistrement du fichier final :

import requests
import urllib.parse

# Importation des librairies classiques :

import numpy as np
import pandas as pd

# Importation des librairies spécifiques à l'enregistrement du fichier final : 
import os
import json
from datetime import datetime

# Définition de la requête avec le nom du dataset (co2cars_2023Pv27):
query = """
SELECT DISTINCT [Year] AS Year, Mk, Cn, [M (kg)], [Ewltp (g/km)], Ft, [Ec (cm3)], [Ep (KW)], [Erwltp (g/km)], Fc
FROM [CO2Emission].[latest].[co2cars_2023Pv27]
WHERE Mk IS NOT NULL 
  AND Cn IS NOT NULL 
  AND [M (kg)] IS NOT NULL
  AND [Ewltp (g/km)] IS NOT NULL
  AND Ft IS NOT NULL
  AND [Ec (cm3)] IS NOT NULL
  AND [Ep (KW)] IS NOT NULL
  AND [Erwltp (g/km)] IS NOT NULL
  AND [Year] IS NOT NULL
  AND Fc IS NOT NULL
"""

# Encodage de la requête pour l'inclure dans l'URL :
encoded_query = urllib.parse.quote(query)

# Initialisation :
records = []
page = 1

# Boucle while pour parcourir toutes les pages de l'API : 
# NB : Celle-ci s'arrête quand il n'y a plus de réponse.

while True:
    url = f"https://discodata.eea.europa.eu/sql?query={encoded_query}&p={page}&nrOfHits=100000"
    response = requests.get(url)
    data = response.json()
    new_records = data.get("results", [])
    if not new_records:
        break
    
    records.extend(new_records)
    page += 1

# Transformation en DataFrame :
df = pd.DataFrame(records)

# Vérification de la colonne "Ft" : travail de catégorisation nécessaire
# Passage en minuscules des catégories en doublon

df['Ft'] = df['Ft'].str.lower()

# Suppression de ces lignes contenant un "unknown" (majoritairement composées de NaN)

df = df[df['Ft'] != 'unknown']

# Rassemblement des variables
# NB : Le dictionnaire peut être complété en cas de valeurs différentes dans le dataset utilisé

dico_fuel = {'petrol': 'Essence',
             'hydrogen' : 'Essence',
             'e85': 'Essence',
             'lpg': 'Essence',
             'ng': 'Essence',
             'ng-biomethane' : 'Essence',
             'diesel': 'Diesel',
             'petrol/electric': 'Hybride',
             'diesel/electric': 'Hybride',
             'electric' : 'Electrique'
}

df['Ft'] = df['Ft'].replace(dico_fuel)

# Mise de côté des modèles électriques (qui n'émettent pas directement de CO2)

df = df[df['Ft'] != 'Electrique']

# Traitement de la colonne "Mk" :
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

# Correction des fautes de frappe répertoriées à date : 
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

# Suppression des occurences trop faibles : 

def filter_brands(df, col='Mk', threshold=5):
    brands = df[col].tolist()
    unique_brands = df[col].unique().tolist()
    filtered_brands = [brand for brand in unique_brands if brands.count(brand) >= threshold]
    return filtered_brands

filtered_brands = filter_brands(df, col='Mk', threshold=5)
df = df[df['Mk'].isin(filtered_brands)]

# Création d'une fonction pour détecter les valeurs aberrantes dans chaque colonne :

def detect_outliers(df, target_col, group_cols=["Cn", "Ft", "Year"]):
    # Calcul de la moyenne par groupe :
    stats = (
        df.groupby(group_cols)
          .agg(**{f'{target_col}_mean': (target_col, 'mean')})
          .reset_index()
    )
    
    # Fusion du DataFrame initial avec les statistiques calculées :
    df_merged = pd.merge(df, stats, on=group_cols, how="left")
    
    # Calcul de l'écart absolu entre la valeur et la moyenne :
    diff_col = f"diff_{target_col}"
    df_merged[diff_col] = (df_merged[target_col] - df_merged[f"{target_col}_mean"]).abs()
    
    # Calcul des quartiles et de l'IQR :
    q1 = df_merged[diff_col].quantile(0.25)
    q3 = df_merged[diff_col].quantile(0.75)
    iqr = q3 - q1
    
    # Calcul du seuil (Q3 + 1.5 * IQR) :
    seuil = (q3 + 1.5 * iqr).round(1)

    # Affichage du nombre d'outliers :
    nb_outliers = len(df_merged[df_merged[diff_col] >= seuil])
    print(f'Nombre de lignes dont la valeur de "{target_col}" dépasse le seuil de {seuil}: {nb_outliers}')
    
    # Suppression des lignes présentant des outliers :
    df_clean_no_outliers = df_merged[df_merged[diff_col] <= seuil]
    print(f"Nombre de lignes après suppression des outliers : {len(df_clean_no_outliers)}")
    
    return df_clean_no_outliers

# Liste des colonnes à filtrer successivement :
columns_to_filter = ['Ewltp (g/km)', 'Fc', 'M (kg)', 'Ec (cm3)', 'Ep (KW)', 'Erwltp (g/km)']

# On part du DataFrame initial (copie pour ne pas altérer l'original) :
df_temp = df.copy()

# Boucle sur chaque colonne pour appliquer le filtrage successif des outliers :
for col in columns_to_filter:
    print(col)
    df_temp = detect_outliers(df_temp, col)

print("\nAprès filtrage successif, le nombre de lignes restantes est de :", len(df_temp))

# Suppression des valeurs aberrantes après traitement :
df_clean_no_outliers_final = df_temp

# Suppression des colonnes ajoutées pour la détection de valeurs aberrantes afin d'éviter tout risque de fuite de données :
df_clean_no_outliers_final = df_clean_no_outliers_final[['Mk', 'Cn', 'M (kg)', 'Ewltp (g/km)', 'Ft', 'Ec (cm3)', 
                                                         'Ep (KW)', 'Erwltp (g/km)', 'Year', 'Fc']]

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

# # Enregistrement du dataset de façon dynamique :
# # Création d'un dossier "metadata" :
# metadata_dir = "metadata"
# os.makedirs(metadata_dir, exist_ok=True) 

# # Génération d'un timestamp au format YYYYMMDD_HHMMSS :
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_filename = f"DF_Processed_{timestamp}.csv"

# # Enregistrement du DataFrame dans le fichier avec le nom dynamique : 
# df_clean_no_outliers_final.to_csv(output_filename, index=False)

# # Remplissage du fichier de métadonnées : 
# metadata_file = os.path.join(metadata_dir, "metadata.json")
# metadata = {"processed_data": output_filename}
# with open(metadata_file, "w") as f:
#     json.dump(metadata, f)


# Enregistrement du dataset de façon dynamique :
# Chemin vers data/raw
output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)  

# Génération d'un timestamp au format YYYYMMDD_HHMMSS :
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"{output_dir}/DF_Processed_{timestamp}.csv"

# Enregistrement du DataFrame dans le fichier avec le nom dynamique : 
df_clean_no_outliers_final.to_csv(output_filename, index=False)

# Remplissage du fichier de métadonnées dans data/raw : 
metadata_file = f"{output_dir}/metadata.json"
metadata = {"processed_data": output_filename}
with open(metadata_file, "w") as f:
    json.dump(metadata, f)