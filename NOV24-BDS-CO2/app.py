import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

# Chargement du dataset
data_path = "/Users/bouchaibchelaoui/Desktop/DATASCIENTEST/PROJET_CO2_DST/src_new/data"
csv_file = os.path.join(data_path, "DF2023-22-21_Concat_Finale_2.csv")
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()  # Nettoyage des noms de colonnes

target = "Ewltp (g/km)"


images_path = "/Users/bouchaibchelaoui/Desktop/DATASCIENTEST/PROJET_CO2_DST/src_new/images"


# Modèles sans marques (baseline)
path = "/Users/bouchaibchelaoui/Desktop/DATASCIENTEST/PROJET_CO2_DST/src_new/models"
# Modèles sans marques (baseline)
pipeline_rf_sm = joblib.load(os.path.join(path, "pipeline_random_forest_sm.pkl"))
pipeline_lr_sm = joblib.load(os.path.join(path, "pipeline_linear_regression_sm.pkl"))
pipeline_knn_sm = joblib.load(os.path.join(path, "pipeline_knn_sm.pkl"))
pipeline_rf_tpot_sm = joblib.load(os.path.join(path, "pipeline_random_forest_opt_tpot_sm.pkl"))

# Modèles avec marques (étendus)
pipeline_rf_ext = joblib.load(os.path.join(path, "pipeline_random_forest_opt_tpot.pkl"))
pipeline_lr_ext = joblib.load(os.path.join(path, "pipeline_linear_regression_ext.pkl"))
pipeline_knn_ext = joblib.load(os.path.join(path, "pipeline_knn_ext.pkl"))
extended_features = joblib.load(os.path.join(data_path, "features_rf_opt_tpot.pkl"))

baseline_features = ["m (kg)", "ec (cm3)", "ep (KW)", "Erwltp (g/km)", "Fuel consumption", "Ft_Diesel", "Ft_Essence"]


brand_columns = [
    'Mk_ALFA ROMEO', 'Mk_ALLIED VEHICLES', 'Mk_ALPINE', 'Mk_AUDI', 'Mk_BENTLEY', 'Mk_BMW',
    'Mk_CITROEN', 'Mk_CUPRA', 'Mk_DACIA', 'Mk_FIAT', 'Mk_FORD', 'Mk_HONDA', 'Mk_HYUNDAI',
    'Mk_JAGUAR', 'Mk_JEEP', 'Mk_KIA', 'Mk_LAMBORGHINI', 'Mk_LANCIA', 'Mk_LAND ROVER',
    'Mk_LEXUS', 'Mk_MASERATI', 'Mk_MAZDA', 'Mk_MERCEDES', 'Mk_MINI', 'Mk_MITSUBISHI',
    'Mk_NISSAN', 'Mk_OPEL', 'Mk_PEUGEOT', 'Mk_PORSCHE', 'Mk_RENAULT', 'Mk_SEAT',
    'Mk_SKODA', 'Mk_SUBARU', 'Mk_SUZUKI', 'Mk_TOYOTA', 'Mk_VOLKSWAGEN', 'Mk_VOLVO',
    'Mk_MAN', 'Mk_NILSSON'
]

extended_features = baseline_features + brand_columns


@st.cache_data(show_spinner=False)
def compute_comparison_metrics(df, baseline_features, extended_features,
                               _pipeline_rf_sm, _pipeline_lr_sm, _pipeline_knn_sm, _pipeline_rf_tpot_sm,
                               _pipeline_rf_ext, _pipeline_lr_ext, _pipeline_knn_ext):
    y_true = df["Ewltp (g/km)"]
    
    # Calcul pour les modèles sans marques (baseline)
    baseline_models = {
        "Random Forest": _pipeline_rf_sm,
        "Random Forest optimisé": _pipeline_rf_tpot_sm,
        "Régression Linéaire": _pipeline_lr_sm,
        "KNN": _pipeline_knn_sm
    }
    mse_baseline = {}
    r2_baseline = {}
    X_base = df[baseline_features]
    for name, model in baseline_models.items():
        y_pred = model.predict(X_base)
        mse_baseline[name] = mean_squared_error(y_true, y_pred)
        r2_baseline[name] = r2_score(y_true, y_pred)
    
    # Calcul pour les modèles avec marques (étendus)
    extended_models = {
        "Random Forest": _pipeline_rf_ext,
        "Random Forest optimisé": _pipeline_rf_ext,
        "Régression Linéaire": _pipeline_lr_ext,
        "KNN": _pipeline_knn_ext
    }
    mse_extended = {}
    r2_extended = {}
    X_ext = df[extended_features]
    for name, model in extended_models.items():
        y_pred = model.predict(X_ext)
        mse_extended[name] = mean_squared_error(y_true, y_pred)
        r2_extended[name] = r2_score(y_true, y_pred)
    
    data = []
    for model_name in ["Random Forest", "Random Forest optimisé", "Régression Linéaire", "KNN"]:
        data.append({
            "Modèle": model_name,
            "MSE (Sans Marques)": mse_baseline[model_name],
            "R² (Sans Marques)": r2_baseline[model_name],
            "MSE (Avec Marques)": mse_extended[model_name],
            "R² (Avec Marques)": r2_extended[model_name]
        })
    return pd.DataFrame(data)



st.markdown(
    """
    <style>
    [data-testid="stSidebar"] > div:first-child {
        background-color: #BBAE98; 
        color: white;  
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Projet CO2 DATASCIENTEST")
st.sidebar.title("Sommaire")
pages = [
    "Présentation du Projet",
    "Pre-processing",
    "DataVizualization",
    "Modélisation sans marques",
    "Modélisation avec marques",
    "Comparaison des modèles"
]

page = st.sidebar.radio("Aller vers", pages)

###############################################
# Page 1 : Présentation du Projet
###############################################

if page == pages[0]:
    st.write("# Prédiction des Émissions de CO₂")

    voitureco2_path = os.path.join(images_path, "voitureco2.png")
    image = Image.open(voitureco2_path)
    st.image(image, use_container_width=True)  
        
    st.write("## Contexte du Projet")
    st.write(
        """
        Face aux enjeux climatiques et aux régulations strictes sur les émissions de CO₂, il devient essentiel de mieux comprendre 
        et prédire l'empreinte carbone des véhicules en circulation.  
        Ce projet vise à **développer un modèle de Machine Learning permettant de prédire les émissions de CO₂ (Ewltp)**
        en fonction des caractéristiques techniques et de la consommation des véhicules.
        """
    )
    
    st.write("## Objectifs du Projet")
    st.write(
        """
        - **Analyser** les facteurs influençant les émissions de CO₂.
        - **Explorer et visualiser** les relations entre les caractéristiques des véhicules et leurs émissions.
        - **Développer un modèle de prédiction performant** basé sur l’apprentissage supervisé.
        - **Accompagner les décisions écologiques** des consommateurs et des constructeurs.
        """
    )
    
    st.write("## Données Utilisées")
    st.write(
        """
        Ce projet s’appuie sur les données officielles de l’**Agence Européenne pour l’Environnement (EEA)**, issues des bases de données **2021, 2022 et 2023**.
        L’objectif de cette fusion est d’enrichir le dataset et d’améliorer la précision du modèle en intégrant un maximum de véhicules.
        
        🔗 [Accéder au dataset de 2023](https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b)  
        🔗 [Accéder au dataset de 2022](https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b?activeAccordion=1094576%2C1091011)  
        🔗 [Accéder au dataset de 2021](https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b?activeAccordion=1094576%2C1091011%2C1086949)
        """
    )
    
    st.write("### Variables Clés du Dataset")
    st.write(
        """
        Le dataset comprend plusieurs caractéristiques essentielles des véhicules, notamment :
        - **Masse du véhicule** (`m (kg)`)
        - **Type de carburant** (`Ft`)
        - **Cylindrée du moteur** (`ec (cm3)`)
        - **Puissance du moteur** (`ep (KW)`)
        - **Consommation de carburant** (`Fuel consumption`)
        - **Réduction d’émissions WLTP** (`Erwltp (g/km)`)
        - **Émissions de CO₂ WLTP** (`Ewltp (g/km)`) – **Variable cible**
        """
    )
    
    st.write("## Qu'est-ce que la Norme WLTP ?")
    st.write(
        """
        La **norme WLTP (Worldwide Harmonized Light Vehicles Test Procedure)** est un protocole de test permettant de mesurer de manière plus précise :
        - La **consommation de carburant** des véhicules.
        - Les **émissions de CO₂** dans des conditions proches de la réalité.
        
        Avant **2018**, l’ancienne norme **NEDC** était utilisée, mais elle ne reflétait pas les conditions de conduite réelles.  
        Le **WLTP** apporte plusieurs améliorations :
        - Des **tests plus longs et réalistes**.
        - Une meilleure prise en compte de **l’impact des équipements**.
        - Une **meilleure précision des valeurs d’émissions**.
        """
    )
    
    st.write("## Méthodologie du Projet")
    st.write(
        """
        **1. Pré-traitement des Données (Pre-processing)**  
        - Nettoyage et transformation.
        - Gestion des valeurs manquantes et des outliers.
        - Normalisation des variables.
        
        **2. Analyse Exploratoire et Visualisation**  
        - Étude des corrélations.
        - Visualisation des distributions.
        
        **3. Modélisation et Comparaison**  
        - Test de plusieurs algorithmes.
        - Optimisation via GridSearchCV.
        
        **4. Déploiement avec Streamlit**  
        - Interface interactive pour la prédiction.
        """
    )
    
    st.write(
        """
        ---  
        **Explorez les étapes du projet via le menu latéral !**
        """
    )

###############################################
# Page 2 : Pré-processing des Données
###############################################


# pre processing a expliquer ici

if page == pages[1]:
    st.write("## Pré-processing des Données")
    st.write("### 1️⃣ Présentation des données")
    st.write("Initialement, le jeu de données est constitué de 3 datasets (années 2021, 2022 et 2023) pour un total d'environ 30 millions de lignes et 40 colonnes.")
    st.write("Rappel: Les datasets sont téléchargeables à cette adresse : https://www.eea.europa.eu/data-and-maps/data/co2-cars-emission-20")



    # Df pour afficher manuellement la structure des colonnes

    data_info = pd.DataFrame([
        {"Nom de la colonne": "ID", "Description": "Numéro d'identification de l'observation", "Type": "int64", "Remarque": "Pas utile pour la construction du modèle"},
        {"Nom de la colonne": "Country", "Description": "Pays de provenance du véhicule", "Type": "object", "Remarque": "Variable indicative - Pas utile pour la construction du modèle"},
        {"Nom de la colonne": "VFN", "Description": "Numéro d'identification de la famille de véhicule", "Type": "object", "Remarque": "Variable indicative - Pas utile pour la construction du modèle"},
        {"Nom de la colonne": "Mp", "Description": "Rassemblements de constructeurs", "Type": "object", "Remarque": "Variable indicative - Pas utile pour la construction du modèle"},
        {"Nom de la colonne": "Mh", "Description": "Nom du constructeur selon la dénomination standard de l'UE", "Type": "object", "Remarque": "Doublon avec 'Mp'"},
        {"Nom de la colonne": "Man", "Description": "Nom du constructeur selon la déclaration du fabricant d'origine", "Type": "object", "Remarque": "Peu d'intérêt pour la suite de l'analyse"},
        {"Nom de la colonne": "MMS", "Description": "Nom du constructeur selon la dénomination du registre des États membres", "Type": "float64", "Remarque": "A supprimer"},
        {"Nom de la colonne": "Tan", "Description": "Numéro d'homologation de type", "Type": "object", "Remarque": "Peu d'intérêt pour la suite de l'analyse"},
        {"Nom de la colonne": "T", "Description": "Type", "Type": "object", "Remarque": "Peu d'intérêt pour la suite de l'analyse"},
        {"Nom de la colonne": "Va", "Description": "Variant", "Type": "object", "Remarque": "Peu d'intérêt pour la suite de l'analyse"},
        {"Nom de la colonne": "Ve", "Description": "Version", "Type": "object", "Remarque": "Peu d'intérêt pour la suite de l'analyse"},
        {"Nom de la colonne": "Mk", "Description": "Fabriquant / Marque du véhicule", "Type": "object", "Remarque": "Intérêt pour la suite de l'analyse"},
        {"Nom de la colonne": "Cn", "Description": "Nom commercial", "Type": "object", "Remarque": "Non utilisé pour l'entraînement du modèle mais à la comparaison et l'identification des valeurs aberrantes"},
        {"Nom de la colonne": "Ct", "Description": "Catégorie du type de véhicule homologué", "Type": "object", "Remarque": "Pas de légende = Inutilisable"},
        {"Nom de la colonne": "Cr", "Description": "Catégorie du véhicule immatriculé", "Type": "object", "Remarque": "Doublon avec 'Ct'"},
        {"Nom de la colonne": "r", "Description": "Total des nouvelles immatriculations", "Type": "int64", "Remarque": "Une seule modalité : 1 sur toutes les lignes"},
        {"Nom de la colonne": "M (kg)", "Description": "Masse en ordre de marche pour un véhicule entièrement assemblé", "Type": "float64", "Remarque": "Intérêt pour la suite de l'analyse"},
        {"Nom de la colonne": "Mt", "Description": "Masse d'essai selon la procédure WLTP", "Type": "float64", "Remarque": "Doublon avec 'M (kg)'"},
        {"Nom de la colonne": "Enedc (g/km)", "Description": "Émissions spécifiques de CO2 (NEDC)", "Type": "float64", "Remarque": "Colonne uniquement composée de valeurs manquantes"},
        {"Nom de la colonne": "Ewltp (g/km)", "Description": "Émissions spécifiques de CO2 (WLTP)", "Type": "float64", "Remarque": "Variable cible"},
        {"Nom de la colonne": "W (mm)", "Description": "Empattement (distance entre les centre des roues avant et arrière d'un véhicule)", "Type": "float64", "Remarque": "Colonne uniquement composée de valeurs manquantes"},
        {"Nom de la colonne": "At1 (mm)", "Description": "Largeur de l'essieu de direction", "Type": "float64", "Remarque": "Colonne uniquement composée de valeurs manquantes"},
        {"Nom de la colonne": "At2 (mm)", "Description": "Largeur de l'essieu autre que celui de direction", "Type": "float64", "Remarque": "Colonne uniquement composée de valeurs manquantes"},
        {"Nom de la colonne": "Ft", "Description": "Type de carburant", "Type": "object", "Remarque": "Intérêt pour la suite de l'analyse"},
        {"Nom de la colonne": "Fm", "Description": "Mode de carburant", "Type": "object", "Remarque": "Doublon avec Ft"},
        {"Nom de la colonne": "Ec (cm3)", "Description": "Cylindrée du moteur", "Type": "float64", "Remarque": "Intérêt pour la suite de l'analyse"},
        {"Nom de la colonne": "Ep (KW)", "Description": "Puissance du moteur", "Type": "float64", "Remarque": "Intérêt pour la suite de l'analyse"},
        {"Nom de la colonne": "Z (Wh/km)", "Description": "Consommation d'énergie électrique", "Type": "float64", "Remarque": "Variable réservée aux véhicules électriques et hybrides rechargeables"},
        {"Nom de la colonne": "IT", "Description": "Technologie innovante ou groupe de technologies innovantes", "Type": "object", "Remarque": "Pas de légende = Inutilisable"},
        {"Nom de la colonne": "Ernedc (g/km)", "Description": "Réduction des émissions grâce à des technologies innovantes", "Type": "float64", "Remarque": "Colonne uniquement composée de valeurs manquantes"},
        {"Nom de la colonne": "Erwltp (g/km)", "Description": "Réduction des émissions grâce à des technologies innovantes (WLTP)", "Type": "float64", "Remarque": "Intérêt pour la suite de l'analyse"},
        {"Nom de la colonne": "De", "Description": "Facteur de déviation", "Type": "float64", "Remarque": "Colonne uniquement composée de valeurs manquantes"},
        {"Nom de la colonne": "Vf", "Description": "Facteur de vérification", "Type": "float64", "Remarque": "Colonne uniquement composée de valeurs manquantes"},
        {"Nom de la colonne": "Year", "Description": "Année de déclaration (et de constitution du dataset)", "Type": "int64", "Remarque": "Conserver à titre indicatif pour la fusion des différents datasets"},
        {"Nom de la colonne": "Status", "Description": "Statut des données (P = Données provisoires, F = Données finales)", "Type": "object", "Remarque": "Toutes les données sont marquées comme provisoires"},
        {"Nom de la colonne": "Date of registration", "Description": "Date d'immatriculation", "Type": "object", "Remarque": "Peu d'intérêt pour la suite de l'analyse"},
        {"Nom de la colonne": "Fuel consumption", "Description": "Consommation de carburant", "Type": "float64", "Remarque": "Intérêt pour la suite de l'analyse"},
        {"Nom de la colonne": "ech", "Description": "Normes d'émissions européennes", "Type": "object", "Remarque": "Catégories à normaliser car les données se recoupent sous différentes dénominations + Beaucoup de NaN = Finalement supprimée"},
        {"Nom de la colonne": "RLFI", "Description": "Référence d'homologation", "Type": "object", "Remarque": "Pas de légende = Inutilisable"},
        {"Nom de la colonne": "Electric range (km)", "Description": "Autonomie électrique (km)", "Type": "float64", "Remarque": "Uniquement pour les véhicules électriques et hybrides rechargeables"},
        ])


    st.write("### Structure du Dataset 📊")
    st.dataframe(data_info, hide_index=True)

    st.write("### 2️⃣ Sélection des variables pour le modèle") 
    st.write("Le dataset initial contient 40 colonnes mais une grande partie d’entre elles ont été supprimées pour des raisons de pertinence (ex : caractéristiques esthétiques du véhicule ou identifiants uniques) et de qualité des données (taux de nan trop élevé).")
    columns_to_keep = ['Mk', 'M (kg)', 'Ewltp (g/km)', 'Ft', 'Ec (cm3)', 'Ep (KW)','Z (Wh/km)', 'Erwltp (g/km)', 'Fuel consumption', 'Electric range (km)']
    st.write("Après une première sélection, voici une liste de colonnes potentiellement pertinentes pour la prédiction des émissions de CO₂ :")
    st.write(columns_to_keep)

    st.write("#### Modification de la variable 'Ft'")
    st.write("la colonne 'Ft' (type de carburant) contient une dizaine de modalité. Les véhicules ont été régroupé selon 4 classes de la manière suivante :")
    dico_ft1 = {'petrol': 'Essence',
             'hydrogen' : 'Essence',
             'e85': 'Essence',
             'lpg': 'Essence',
             'ng': 'Essence',
             'ng-biomethane' : 'Essence',
             'diesel': 'Diesel',
             'petrol/electric': 'Hybride',
             'diesel/electric': 'Hybride',
             'electric' : 'Electrique'}
    st.write(dico_ft1)

    # repartition des types de carburant
    st.write("#### Répartition des types de Carburant")
    fuel_counts = {
        "Diesel": 44,
        "Electrique": 1.5,
        "Essence": 54,
        "Hybride": 0.5
    }

    # graph des carburants
    fig, ax = plt.subplots()
    ax.pie(fuel_counts.values(), labels=fuel_counts.keys(), autopct="%1.1f%%", 
           colors=["gold", "red", "green", "lightblue"], startangle=90)
    st.pyplot(fig)



    st.write("Dans ce dataset, les véhicules électriques ne produisent pas de CO₂. Les véhicules électriques ont donc été écartés")
    st.write("Concernant les véhicules hybrides, plus de la moitié des observations non pas la colonne 'Ewltp (g/km)' (émission de CO₂) de renseignée. Avec un nombre aussi limité d’observations valides, conserver cette catégorie introduirait un biais important et ne permettrait pas une analyse mathématiquement rigoureuse. Ainsi, par souci de fiabilité et de représentativité des données, cette classe a été écartée.")
    
    st.write("#### Suppression des colonnes 'Z (Wh/km)' et 'Electric range (km)'")
    st.write("Ces deux colonnes représentent respectivement la consommation d'énergie électrique et l'autonomie électrique en km. Ces variables sont spécifiques aux véhicules électriques et hybrides. Par conséquences elles sont aussi suprimmées.")


    st.write("### 3️⃣ Traitement pour les marques des véhicules")
    st.write("Nous avons inclus la possibilité de sélectionner un certains nombres de marques pour évaluer l’effet potentiel du constructeur sur la production de CO₂")
    st.write("Cela a nécessité un travail important de standardisation, les noms des marques dans le dataset n'étaient pas uniformisés. Certaines étaient écrites en majuscules, d’autres en minuscules. Il existait plusieurs variantes pour une même marque")
    st.write("De nombreuses mentions peuvent être rassemblées sous une seule dénomination afin de réduire leur nombre. Par exemple, ‘Mercedes’, ‘Mercedes-Benz’ et ‘Mercedes Benz’ peuvent être regroupés sous ‘Mercedes’ uniquement.")
    st.write("Voici un aperçu des marques les plus représentées dans le dataset")

    countplot_path = os.path.join(images_path, "Countplot_Mk.png")
    image_marque = Image.open(countplot_path)
    st.image(image_marque, use_container_width=True)

    st.write("### 4️⃣ Suppression des doublons et valeurs manquantes")
    st.write("Bien que les datasets originaux soient extrêmement denses (près de 10 millions d’entrées chacuns), en regardant les doublons de plus près, nous pouvons constater qu’ils sont extrêmement nombreux (98% pour chaque jeu de données !).")
    st.write("En effet, ces datasets regroupent les déclarations des pays européens concernant les nouvelles immatriculations de véhicules sur leur territoire en 2021, 2022 et 2023.") 
    st.write("Les émissions de CO2 étant les mêmes pour chaque modèle de véhicule. Par exemple, 200 000 Audi A1 ont été immatriculées en 2023 sur l’ensemble de l’Union Européenne, nous avons donc 1 ligne et 199 999 doublons potentiels sur cette année spécifique. Il est donc nécessaire de supprimer les doublons, ce qui diminue grandement la taille du dataset.")
    st.write("Enfin les lignes présentant au moins une valeurs manquantes ont été supprimées.")


    st.write("### 5️⃣ Détection des Outliers")
    st.write("Pour détecter et supprimer les outliers, nous avons procédé par différentes étapes. D’abord, nous avons regroupé les véhicules présents dans le jeu de données selon leur modèle, leur type de carburant et leur année (ex. : T-ROC Essence de 2023). ")
    st.write("Ensuite, à l’aide d’une fonction d’agrégation, nous avons calculé individuellement la moyenne des colonnes numériques pour chacune de ces catégories. Nous avons ajouté cette nouvelle variable à notre jeu de données original, avant de calculer, pour chaque colonne et pour chaque ligne, la différence entre la valeur réelle et la moyenne correspondante, que nous avons stockée dans une deuxième variable.")
    st.write("Enfin, nous avons calculé l’écart interquartile de la distribution de ces différences et déterminé un seuil à ne pas dépasser. Ce seuil est ensuite utilisé pour identifier les valeurs jugées aberrantes.")
    st.write("Grâce à la définition d’une fonction et à l’utilisation d’une boucle for, nous sommes désormais capables d’implémenter cette stratégie pour toutes les colonnes numériques du jeu de données et de mettre de côté les outliers au fur et à mesure. Cela réduit la taille du jeu de données final, mais garantit des résultats fiables lors des prédictions, car le modèle n’est plus perturbé par ces valeurs aberrantes.")
    st.write("Voici un exemple graphique de la sortie de notre fonction, associée à une boucle for :")

    im_outlier = os.path.join(images_path, "Exemple_Detect_Outliers.png")
    image_marque = Image.open(im_outlier)
    st.image(image_marque, use_container_width=True)

    st.write("### 6️⃣ Concaténation des datasets")
    st.write("Afin d’enrichir un maximum le jeu de données sans altérer les résultats, le dataset initial (2023) est completé avec celui des années précedentes (2022 et 2021).")
    st.write("Après une observation approfondie des différentes colonnes et de leur remplissage, nous avons constaté que seules les deux années précédentes étaient entièrement compatibles avec notre processus de prétraitement et notre modélisation (2021 et 2022). Au-delà, nous nous exposions à des soucis de cohérence (beaucoup de valeurs manquantes dans certaines de nos colonnes et des disparités de variables avant 2016).")
    st.write("Nous avons chargé séparément les datasets et appliqué les mêmes étapes de preprocessing que pour celui de 2023. Nous avons procédé ainsi car une concaténation en tout début de traitement aurait été beaucoup trop lourde et compliquée à gérer (cela aurait créé un dataset de près de 30 millions de lignes au total).")

    st.write("### 7️⃣ Encodage et aperçu du dataset")
    st.write("Une fois le dataset final assemblé, les variables catégorielles 'Ft' et 'Mk' sont encodées (one-hot encoding).")
    st.write("Le dataset est prêt à l'emploi pour entraîner le modèle.")
    st.write(df.head())

###############################################
# Page 3 : Data Visualization
###############################################
elif page == pages[2]:
    st.write("## Data Visualization")
    st.write("### Analyse Exploratoire des Données")
    
    dico_mapping = {
        "Masse du véhicule (kg)": "m (kg)",
        "Cylindrée (cm³)": "ec (cm3)",
        "Puissance (kW)": "ep (KW)",
        "Réduction d’émissions WLTP (g/km)": "Erwltp (g/km)",
        "Consommation de carburant (L/100km)": "Fuel consumption",
        "Émissions de CO₂ (g/km)": "Ewltp (g/km)"
    }
    col_display_names = list(dico_mapping.keys())
    
    st.write("### Heatmap des Corrélations")
    true_col_names = list(dico_mapping.values())
    df_corr = df[true_col_names].corr()
    dico_inverser = {v: k for k, v in dico_mapping.items()}
    df_corr = df_corr.rename(index=dico_inverser, columns=dico_inverser)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.write("### Distribution des variables")
    col_relation_selected_display_1 = st.selectbox("Sélectionner une variable numérique :", col_display_names, key=1)
    col_relation_selected_1 = dico_mapping[col_relation_selected_display_1]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[col_relation_selected_1], bins=30, kde=True, color="blue", ax=ax)
    ax.set_xlabel(col_relation_selected_display_1)
    ax.set_ylabel("Nombre de Véhicules")
    ax.set_title(f"Distribution de {col_relation_selected_display_1}")
    st.pyplot(fig)
    
    st.write("### Détection des Outliers via Boxplots")
    cols_boxplot = ["Ewltp (g/km)", "ec (cm3)", "ep (KW)", "Fuel consumption"]
    fig, axes = plt.subplots(1, len(cols_boxplot), figsize=(18, 6))
    for i, col in enumerate(cols_boxplot):
        sns.boxplot(y=df[col], ax=axes[i], color="cyan")
        axes[i].set_title(f"Boxplot de {col}")
    st.pyplot(fig)
    
    st.write("### Relation entre une variable et les émissions de CO₂")
    col_relation_selected_display_2 = st.selectbox("Sélectionner une variable numérique :", col_display_names, key=2)
    col_relation_selected_2 = dico_mapping[col_relation_selected_display_2]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df[col_relation_selected_2], y=df["Ewltp (g/km)"], alpha=0.5, color="red", ax=ax)
    ax.set_xlabel(col_relation_selected_display_2)
    ax.set_ylabel("Émissions de CO₂ (g/km)")
    ax.set_title(f"Relation entre {col_relation_selected_display_2} et Émissions de CO₂")
    st.pyplot(fig)
    
    st.write("### Répartition des types de Carburant")
    fuel_counts = {
        "Diesel": df["Ft_Diesel"].sum(),
        "Essence": df["Ft_Essence"].sum()
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(fuel_counts.values(), labels=fuel_counts.keys(), autopct="%1.1f%%", 
           colors=["gold", "lightblue", "green"], startangle=90)
    ax.set_title("Répartition par Type de Carburant")
    st.pyplot(fig)


###############################################
# Page 4 : Modélisation sans marques (Baseline)
###############################################

elif page == "Modélisation sans marques":
    st.write("## Modélisation sans marques")
    model_choice = st.selectbox("Choisissez un modèle", 
                                ["Random Forest", "Random Forest optimisé", "Régression Linéaire", "KNN"])
    
    with st.form("prediction_form_baseline"):
        st.write("### Entrez les valeurs du véhicule pour prédire les émissions de CO₂ (sans marques)")
        col1, col2 = st.columns(2)
        with col1:
            m_kg = st.number_input("Masse du véhicule (kg)", min_value=500, max_value=3000, step=1)
            ec_cm3 = st.number_input("Cylindrée (cm³)", min_value=500, max_value=6000, step=1)
        with col2:
            ep_kw = st.number_input("Puissance (kW)", min_value=20, max_value=500, step=1)
            erwltp = st.number_input("Réduction d’émissions WLTP (g/km)", min_value=0.0, max_value=3.5, step=0.01)
        fuel_consumption = st.number_input("Consommation de carburant (L/100km)", min_value=2.0, max_value=15.0, step=0.1)
        ft = st.selectbox("Type de carburant", ["Diesel", "Essence"])
        fuel_types = {"Diesel": [1, 0], "Essence": [0, 1]}
        ft_encoded = fuel_types[ft]
        
        # Construction de l'input de base (7 valeurs)
        input_values = [m_kg, ec_cm3, ep_kw, erwltp, fuel_consumption] + ft_encoded
        input_data_df = pd.DataFrame([input_values], columns=baseline_features)
        
        submitted = st.form_submit_button("🔎 Prédire")
        if submitted:
            if model_choice == "Random Forest":
                prediction = pipeline_rf_sm.predict(input_data_df)
                
                # Extraction et affichage de l'importance des features
                rf_model = pipeline_rf_sm.named_steps["rf"]
                importances_stacking = rf_model.feature_importances_
                feature_names_stacking = list(baseline_features)
                importances_df_stacking = pd.DataFrame({
                    'Feature': feature_names_stacking,
                    'Importance': importances_stacking 
                }).sort_values(by='Importance', ascending=False)
                
                fig, ax = plt.subplots()
                ax.barh(importances_df_stacking["Feature"], importances_df_stacking["Importance"])
                ax.set_xlabel("Importance")
                ax.set_ylabel("Features")
                ax.set_title("Feature Importance - Random Forest")
                ax.invert_yaxis()  # Le plus important en haut
                st.pyplot(fig)
            
            elif model_choice == "Random Forest optimisé":
                nb_brands = len(brand_columns)
                extended_input = input_values + [0] * nb_brands
                input_data_extended = pd.DataFrame([extended_input], columns=extended_features)
                prediction = pipeline_rf_ext.predict(input_data_extended)

                rf_model_tpot = pipeline_rf_tpot_sm.named_steps["rf"]
                importances_stacking_tpot = rf_model_tpot.feature_importances_
                feature_names_stacking = list(baseline_features)
                importances_df_stacking_tpot = pd.DataFrame({
                    'Feature': feature_names_stacking,
                    'Importance': importances_stacking_tpot 
                }).sort_values(by='Importance', ascending=False)

                fig, ax = plt.subplots()
                ax.barh(importances_df_stacking_tpot["Feature"], importances_df_stacking_tpot["Importance"])
                ax.set_xlabel("Importance")
                ax.set_ylabel("Features")
                ax.set_title("Feature Importance - RandomForest optimisé (TPOT)")
                ax.invert_yaxis()
                st.pyplot(fig)

            elif model_choice == "Régression Linéaire":
                prediction = pipeline_lr_sm.predict(input_data_df)
                
                lr_model = pipeline_lr_sm.named_steps["lr"]
                coefs = lr_model.coef_
                df_coefs = pd.DataFrame({"Feature": baseline_features, "Coefficient": coefs})
                df_coefs = df_coefs.sort_values(by="Coefficient", ascending=False)
                
                fig, ax = plt.subplots()
                ax.barh(df_coefs["Feature"], df_coefs["Coefficient"])
                ax.set_xlabel("Coefficient Value")
                ax.set_ylabel("Features")
                ax.set_title("Feature Coefficients - Linear Regression")
                ax.invert_yaxis()
                st.pyplot(fig)
                
            elif model_choice == "KNN":
                prediction = pipeline_knn_sm.predict(input_data_df)
                
                # Charger les valeurs SHAP pour le modèle KNN
                data_path = "/Users/bouchaibchelaoui/Desktop/DATASCIENTEST/PROJET_CO2_DST/src_new/data"
                shap_file = os.path.join(data_path, "shap_values_knn.pkl")
                shap_values = joblib.load(shap_file)
                
                df_shap = pd.DataFrame({
                    "Feature": list(baseline_features), 
                    "Importance": np.abs(shap_values).mean(axis=0)
                }).sort_values(by="Importance", ascending=False)
                
                fig, ax = plt.subplots()
                ax.barh(df_shap["Feature"], df_shap["Importance"])
                ax.set_xlabel("Valeur SHAP")
                ax.set_ylabel("Features")
                ax.set_title("Feature Importance - KNN")
                ax.invert_yaxis()
                st.pyplot(fig)
            
            st.success(f"📊 Estimation (sans marques) : **{prediction[0]:.2f} g/km**")

###############################################
# Page 5 : Modélisation avec marques (Étendu)
###############################################
elif page == "Modélisation avec marques":
    st.write("## Modélisation avec marques")
    model_choice = st.selectbox("Choisissez un modèle", 
                                ["Random Forest", "Random Forest optimisé", "Régression Linéaire", "KNN"],
                                key="ext_model")
    
    with st.form("prediction_form_extended"):
        st.write("### Entrez les valeurs du véhicule pour prédire les émissions de CO₂ (avec marques)")
        col1, col2 = st.columns(2)
        with col1:
            m_kg = st.number_input("Masse du véhicule (kg)", min_value=500, max_value=3000, step=1, key="ext_m_kg")
            ec_cm3 = st.number_input("Cylindrée (cm³)", min_value=500, max_value=8000, step=1, key="ext_ec_cm3")
            ep_kw = st.number_input("Puissance (kW)", min_value=20, max_value=500, step=1, key="ext_ep_kw")
        with col2:
            erwltp = st.number_input("Réduction d’émissions WLTP (g/km)", min_value=0.0, max_value=5.0, step=0.01, key="ext_erwltp")
            fuel_consumption = st.number_input("Consommation de carburant (L/100km)", min_value=2.0, max_value=30.0, step=0.1, key="ext_fuel_consumption")
            ft = st.selectbox("Type de carburant", ["Diesel", "Essence"], key="ext_ft")
        # Comme les hybrides ont été supprimés, nous utilisons seulement Diesel et Essence.
        fuel_types = {"Diesel": [1, 0], "Essence": [0, 1]}
        ft_encoded = fuel_types[ft]
        
        # Sélection de la marque (bien que nous ne filtrons pas sur la marque pour la comparaison globale)
        selected_brand = st.selectbox("Sélectionnez la marque du véhicule", brand_columns, key="ext_brand")
        brand_values = [1 if col == selected_brand else 0 for col in brand_columns]
        
        # Construction de l'input complet pour le modèle étendu :
        # Baseline (7 valeurs) + fuel encoding (2 valeurs) + marque (len(brand_columns))
        extended_input = [m_kg, ec_cm3, ep_kw, erwltp, fuel_consumption] + ft_encoded + brand_values
        input_data_df = pd.DataFrame([extended_input], columns=extended_features)
        
        submitted = st.form_submit_button("🔎 Prédire")
        if submitted:
            if model_choice == "Random Forest":
                prediction = pipeline_rf_ext.predict(input_data_df)
            elif model_choice == "Random Forest optimisé":
                prediction = pipeline_rf_ext.predict(input_data_df)
            elif model_choice == "Régression Linéaire":
                prediction = pipeline_lr_ext.predict(input_data_df)
            elif model_choice == "KNN":
                prediction = pipeline_knn_ext.predict(input_data_df)
            st.success(f"📊 Estimation (avec marques) : **{prediction[0]:.2f} g/km**")

            # Liste des noms de colonnes utilisées pour l'entraînement du modèle NN
            features_for_nn = ["m (kg)", "ec (cm3)", "ep (KW)", "Fuel consumption", "Erwltp (g/km)"]

            # Créer un DataFrame pour input_vector au lieu d'un tableau NumPy
            input_vector = pd.DataFrame([[m_kg, ec_cm3, ep_kw, fuel_consumption, erwltp]], 
                                        columns=features_for_nn)

            # Utilisation de input_vector avec le modèle NN
            nn_model = NearestNeighbors(n_neighbors=1).fit(df[features_for_nn])
            distances, indices = nn_model.kneighbors(input_vector)

            # Récupération de l'indice du véhicule le plus similaire
            closest_index = indices[0][0]

            # Valeur réelle associée à ce véhicule
            point_reel = df[target].iloc[closest_index]

            # 2) Préparez la distribution globale
            actual_values = np.sort(df[target].dropna().values)

            # 3) Création du graphique
            fig, ax = plt.subplots(figsize=(8,6))

            # Scatterplot de la distribution globale (valeurs triées)
            ax.scatter(range(len(actual_values)), actual_values, color="blue", alpha=0.6, label="Valeurs réelles (triées)")

            # On identifie où se situe le point_reel dans le tableau trié
            # Méthode : on cherche l’index d’insertion de point_reel dans actual_values
            pos_point = np.searchsorted(actual_values, point_reel)

            # On affiche un marqueur vert pour la valeur réelle du véhicule similaire
            ax.scatter(pos_point, point_reel, color="green", s=100, zorder=5, label="Valeur réelle (véhicule similaire)")

            # Ligne horizontale rouge pour la valeur prédite
            ax.axhline(y=prediction[0], color="red", linestyle="--", linewidth=2, label="Valeur prédite")

            # Ajustements divers
            ax.set_xlabel("Index (valeurs triées)")
            ax.set_ylabel("Émissions de CO₂ (g/km)")
            ax.set_title("Distribution des valeurs réelles vs Valeur prédite")
            ax.legend()

            st.pyplot(fig)


###############################################
# Page 6 : Comparaison des modèles (tableau et graphique)
###############################################

elif page == "Comparaison des modèles":
    st.write("## Comparaison des performances des modèles")
    
    df_comparison = compute_comparison_metrics(
        df, baseline_features, extended_features,
        pipeline_rf_sm, pipeline_lr_sm, pipeline_knn_sm, pipeline_rf_tpot_sm,
        pipeline_rf_ext, pipeline_lr_ext, pipeline_knn_ext
    )
    
    st.write("### Résultats de l'évaluation des modèles")
    st.table(df_comparison)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(df_comparison))
    width = 0.35
    
    ax1.bar(x - width/2, df_comparison["MSE (Sans Marques)"], width, label="Sans Marques")
    ax1.bar(x + width/2, df_comparison["MSE (Avec Marques)"], width, label="Avec Marques")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_comparison["Modèle"], rotation=45)
    ax1.set_ylabel("MSE")
    ax1.set_title("Comparaison des MSE")
    ax1.legend()
    
    ax2.bar(x - width/2, df_comparison["R² (Sans Marques)"], width, label="Sans Marques")
    ax2.bar(x + width/2, df_comparison["R² (Avec Marques)"], width, label="Avec Marques")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_comparison["Modèle"], rotation=45)
    ax2.set_ylabel("R²")
    ax2.set_title("Comparaison des R²")
    ax2.legend()
    
    st.pyplot(fig)
    
    st.markdown("""
    ### **📌 Analyse des résultats**
    - **Random Forest optimisé (TPOT)** donne les meilleurs résultats avec une **faible MSE** et un **score R² proche de 1**.
    - **La Régression Linéaire** présente une **MSE élevée**, indiquant qu'elle n'est pas adaptée à ce problème.
    - **Le modèle KNN** fonctionne correctement mais est moins performant que les méthodes basées sur Random Forest.
    - **Le modèle Random Forest de base** reste intéressant, mais l'optimisation par TPOT améliore ses performances.
    
    💡 **Conclusion :** Le modèle **Random Forest optimisé avec TPOT** semble être le plus performant. 🚀
    """)

