#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Countplot des marques de voiture : 

plt.figure(figsize=(20, 6))
sns.countplot(data=df, x='Mk', order=df['Mk'].value_counts().index)
plt.xticks(rotation=45, ha='right')
plt.title('Répartition des marques')
plt.xlabel('Marques')
plt.ylabel("Occurrences")
plt.show()

# Boxplots des variables numériques :

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

# Outliers :

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

# Distribution de "Ft" :

plt.figure(figsize=(8, 6))
sns.countplot(x='Ft', data=df_clean_no_outliers_final)
plt.title("Distribution Ft")
plt.xlabel("Type")
plt.ylabel("Nombre")
plt.show()

# Heatmap des corrélations
columns_corr = ['m (kg)', 'Ewltp (g/km)', 'ec (cm3)', 'ep (KW)', 'Fuel consumption ', 'Erwltp (g/km)']
df_clean_no_outliers_corr = df_clean_no_outliers_final[columns_corr]
sns.heatmap(df_clean_no_outliers_corr.corr().round(2), annot=True, cmap='RdBu_r', center=0)
plt.show()