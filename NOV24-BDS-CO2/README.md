# CO2 DATASCIENTEST Project

Ce projet vise à prédire les émissions de CO₂ (WLTP) des véhicules à partir de caractéristiques techniques telles que la masse, la cylindrée, la puissance, la consommation de carburant et le type de carburant. Plusieurs modèles de machine learning sont comparés (Random Forest, Régression Linéaire, KNN), avec et sans inclusion des informations sur les marques, et une optimisation via TPOT est réalisée.

## Table des matières

- [Présentation du Projet](#présentation-du-projet)
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Modèles et Données](#modèles-et-données)
- [Téléchargement du Dataset](#téléchargement-du-dataset)
- [Pré-processing et Concaténation des Datasets](#pré-processing-et-concaténation-des-datasets)
- [Axes d'Amélioration](#axes-damélioration)
- [Licence](#licence)
- [Contributions](#contributions)

## Présentation du Projet

Face aux enjeux climatiques et aux régulations strictes sur les émissions de CO₂, il est crucial de développer des outils permettant d'estimer l'impact environnemental des véhicules. Ce projet a pour objectifs :
- D'analyser les facteurs influençant les émissions de CO₂.
- De développer et comparer plusieurs modèles de prédiction.
- D'explorer des pistes d'optimisation et d'amélioration pour des futures évolutions.

## Structure du Projet

  nov24_bds_co2
├── LICENSE
├── README.md
├── __pycache__
│   └── streamlit.cpython-313.pyc
├── app.py
├── data
│   ├── DF2023-22-21_Concat_Finale_2.csv
│   ├── features_rf_opt_tpot.pkl
│   └── shap_values_knn.pkl
├── images
│   ├── Countplot_Mk.png
│   ├── Exemple_Detect_Outliers.png
│   └── voitureco2.png
├── models
│   ├── pipeline_knn_ext.pkl
│   ├── pipeline_knn_sm.pkl
│   ├── pipeline_linear_regression_ext.pkl
│   ├── pipeline_linear_regression_sm.pkl
│   ├── pipeline_random_forest_opt_tpot.pkl
│   ├── pipeline_random_forest_opt_tpot_sm.pkl
│   └── pipeline_random_forest_sm.pkl
└── src
    ├── concatenate_datasets.py
    ├── model.py
    ├── pre_processing.py
    ├── print_tree.py
    └── regression_analysis.py

## Installation

1. Cloner le dépôt :

       git clone https://github.com/DataScientest-Studio/NOV24-BDS-CO2.git
       cd NOV24-BDS-CO2

2. Créer un environnement virtuel (optionnel, mais recommandé) :

       python -m venv venv
       source venv/bin/activate   # Sur Windows : venv\Scripts\activate

3. Installer les dépendances :

       pip install -r requirements.txt

       
## Utilisation

Pour lancer l'application Streamlit :

       streamlit run app.py

L'application s'ouvrira dans votre navigateur à l'adresse [http://localhost:8501](http://localhost:8501).

## Modèles et Données

- Les modèles entraînés (.pkl) seront enregistrés dans le dossier `models/`.
- Pour générer ces fichiers, exécutez le script `model.py` situé dans le dossier `src/`. Une fois lancé, les fichiers de modélisation seront automatiquement sauvegardés dans `models/`.
- Le prétraitement des données est réalisé via le script `pre_processing.py`.
- Les fichiers d'entrée et de sortie pour l'entraînement et l'évaluation se trouvent dans le dossier `data/`.


## Téléchargement du Dataset

Le dataset brut n'est pas inclus dans ce dépôt pour éviter de surcharger GitHub. Veuillez télécharger le dataset correspondant à l'année souhaitée et le placer dans le dossier `src_new/data` (après l'avoir renommé en `data.csv` si nécessaire).

- **Année 2023 :**  
  [Télécharger le dataset 2023](https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b?activeAccordion=1094576)

- **Année 2022 :**  
  [Télécharger le dataset 2022](https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b?activeAccordion=1094576%2C1091011)

- **Année 2021 :**  
  [Télécharger le dataset 2021](https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b?activeAccordion=1094576%2C1091011%2C1086949)

## Pré-processing et Concaténation des Datasets

Le pré-processing a été effectué séparément pour chaque dataset des années 2023, 2022 et 2021.  
Les fichiers pré-traités sont nommés :
- `DF2023_Processed_2.csv`
- `DF2022_Processed_2.csv`
- `DF2021_Processed_2.csv`

Pour obtenir le dataset final, ces fichiers ont été concaténés à l'aide du script `concatenate_datasets.py`.  
Exécutez le script en ligne de commande :

       python concatenate_datasets.py

Le fichier résultant, `DF2023-22-21_Concat_Finale_2.csv`, doit être placé dans le dossier `src_new/data`.

## Axes d'Amélioration

- Intégration de nouvelles variables explicatives.
- Optimisation des hyperparamètres avec d'autres techniques.
- Déploiement sur une plateforme cloud pour un accès public.

## Licence

Ce projet est distribué sous la licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.





MAX - MlFlow:

start a local MLflow Tracking Server :
1      mlflow server --host 127.0.0.1 --port 8080

execution du script :
1      python script/testMlFlow.py

Pour supprimer les anciennes run :
1      rm -rf mlruns/


