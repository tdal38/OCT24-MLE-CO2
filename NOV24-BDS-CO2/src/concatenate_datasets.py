#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concaténation des datasets pré-traités pour les années 2023, 2022 et 2021.
Le dataset final sera exporté sous le nom "DF2023-22-21_Concat_Finale_2.csv".
Assurez-vous que les fichiers "DF2023_Processed_2.csv", "DF2022_Processed_2.csv" et "DF2021_Processed_2.csv"
se trouvent dans le même dossier que ce script, ou modifiez les chemins en conséquence.
"""

import pandas as pd
import numpy as np
import os

# # Import des datasets pré-traités
# df_2023 = pd.read_csv('DF2023_Processed_2.csv')
# df_2022 = pd.read_csv('DF2022_Processed_2.csv')
# df_2021 = pd.read_csv('DF2021_Processed_2.csv')

# # Concaténation des DataFrames
# df_concat = pd.concat([df_2021, df_2022, df_2023], axis=0, ignore_index=True).fillna(0)

# print(df_concat.info())
# print(df_concat.head())

# columns_to_convert = [col for col in df_concat.columns if col.startswith('Mk')]
# df_concat[columns_to_convert] = df_concat[columns_to_convert].astype(int)

# col_exceptions = [col for col in df_concat.columns if col not in ['year', 'Cn']]
# nbr_doublons_concat = df_concat.duplicated(subset=col_exceptions).sum()
# taille_dataset_concat = len(df_concat)
# print('Nombre de doublons présents dans le dataset :', nbr_doublons_concat)
# print('Pourcentage de doublons :', (nbr_doublons_concat / taille_dataset_concat) * 100)

# df_clean_concat = df_concat.drop_duplicates(subset=col_exceptions)
# print(df_clean_concat.info())

# # Export du dataset final concaténé
# output_filename = "DF2023-22-21_Concat_Finale_2.csv"
# df_clean_concat.to_csv(output_filename, index=False)
# print("Dataset final exporté sous le nom:", output_filename)

# Chemin vers data/raw
input_dir = "data/raw"

# Import des datasets pré-traités depuis data/raw
df_2023 = pd.read_csv(f'{input_dir}/DF2023_Processed_2.csv')
df_2022 = pd.read_csv(f'{input_dir}/DF2022_Processed_2.csv')
df_2021 = pd.read_csv(f'{input_dir}/DF2021_Processed_2.csv')

# Concaténation des DataFrames
df_concat = pd.concat([df_2021, df_2022, df_2023], axis=0, ignore_index=True).fillna(0)

print(df_concat.info())
print(df_concat.head())

columns_to_convert = [col for col in df_concat.columns if col.startswith('Mk')]
df_concat[columns_to_convert] = df_concat[columns_to_convert].astype(int)

col_exceptions = [col for col in df_concat.columns if col not in ['year', 'Cn']]
nbr_doublons_concat = df_concat.duplicated(subset=col_exceptions).sum()
taille_dataset_concat = len(df_concat)
print('Nombre de doublons présents dans le dataset :', nbr_doublons_concat)
print('Pourcentage de doublons :', (nbr_doublons_concat / taille_dataset_concat) * 100)

df_clean_concat = df_concat.drop_duplicates(subset=col_exceptions)
print(df_clean_concat.info())

# Export du dataset final concaténé dans data/raw
output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)  # Crée le dossier s'il n'existe pas
output_filename = f"{output_dir}/DF2023-22-21_Concat_Finale_2.csv"
df_clean_concat.to_csv(output_filename, index=False)
print("Dataset final exporté sous le nom:", output_filename)
