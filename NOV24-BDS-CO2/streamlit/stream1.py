import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import mlflow
from mlflow.models import infer_signature

# Fonction pour charger et prétraiter le dataset
@st.cache_data
def load_and_preprocess_data():
    # Chargement du dataset
    df = pd.read_csv('data/DF2023-22-21_Concat_Finale_2.csv')
    
    # Suppression des espaces dans les noms de colonnes
    df.columns = df.columns.str.strip()
    
    # Sélection des features et de la target
    baseline_features = ['m (kg)', 'ec (cm3)', 'ep (KW)', 'Erwltp (g/km)', 'Fuel consumption', 'Ft_Diesel', 'Ft_Essence']
    target = 'Ewltp (g/km)'
    
    X_baseline = df[baseline_features]
    y_baseline = df[target]
    
    # Split en train/test
    X_train, X_test, y_train, y_test = train_test_split(X_baseline, y_baseline, test_size=0.2, random_state=42)
    
    return df, X_train, X_test, y_train, y_test

# Charger le dataset
df, X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Configuration de MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("MLflow Streamlit")

# Définir les onglets
tabs = ["Exploration du Dataset", "Entraînement des Modèles", "Historique des Runs", "Chargement des Modèles"]

# Création des onglets
tab1, tab2, tab3, tab4 = st.tabs(tabs)

# Onglet 1 : Exploration du dataset
with tab1:
    st.header("📊 Exploration du Dataset")

    # Affichage d'un aperçu du dataset
    st.subheader("Aperçu du Dataset 📋")
    st.dataframe(df.head(10))  # Afficher les 10 premières lignes

# Onglet 2 : Entraînement des modèles
with tab2:
    st.header("🏋️‍♂️ Entraînement des Modèles")

    # Affichage du modèle
    st.subheader("Choix du modèle 📋")

    # Sélection du modèle
    model_choice = st.selectbox(
        "Choisissez un modèle de Machine Learning",
        ["Régression Linéaire", "KNN", "Random Forest"]
    )

    # Afficher le choix sélectionné
    st.write(f"🔍 Modèle sélectionné : **{model_choice}**")

    # 📌 Entraînement et Logging MLflow
    if st.button("Entraîner le modèle"):
        
        with mlflow.start_run():
            
            # Modèle sélectionné
            if model_choice == "Régression Linéaire":
                model = LinearRegression()
            elif model_choice == "KNN":
                model = KNeighborsRegressor()
            elif model_choice == "Random Forest":
                model = RandomForestRegressor()
            
            # Entraînement
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calcul des métriques
            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred)

            # Log dans MLflow
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("RMSE", rmse)

            # Ajouter un tag d'info
            mlflow.set_tag("Training Info", f"Modèle {model_choice} entraîné sur DF2023")

            # Enregistrer le modèle MLflow
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="test_model",
                signature=signature,
                input_example=X_train,
                registered_model_name="tracking-quickstart",
            )

            # Affichage des résultats
            st.write(f"### 📊 Résultats du Modèle : {model_choice}")
            st.write(f"**R² :** {r2:.4f}") #.4f sert à avoir 4 chiffres apres la virgule
            st.write(f"**RMSE :** {rmse:.4f}") 

            # Affichage des valeurs réelles vs prédictions
            result_df = pd.DataFrame({"Valeur Réelle": y_test, "Prédiction": y_pred})
            st.dataframe(result_df.head(10))

# Onglet 3 : Historique des Runs
with tab3:
    st.header("📜 Historique des Runs MLflow")

    # Se connecter à MLflow et récupérer les runs triés par date
    mlflow.set_experiment("MLflow Streamlit")
    runs = mlflow.search_runs(order_by=["start_time DESC"])  # Tri des runs du plus récent au plus ancien

    if runs.empty:
        st.info("Aucun modèle enregistré pour l'instant. Lance un entraînement !")
    else:
        # Extraire les informations clefs des runs
        df_runs = runs[["run_id", "metrics.R2", "metrics.RMSE", "start_time", "tags.mlflow.runName"]]
        df_runs.columns = ["Run ID", "R² Score", "RMSE", "Date", "Nom du modèle"]

        # Sélectionner la dernière run par défaut
        latest_run_id = df_runs.iloc[0]["Run ID"]  # Premier élément (le plus récent)
        selected_run = st.selectbox("Affichage de la dernière Run :", list(df_runs["Nom du modèle"]), index=0)   

        # Afficher les détails de la run sélectionnée
        run_details = df_runs[df_runs["Nom du modèle"] == selected_run]
        st.write(f"📌 **Nom du modèle** : {run_details['Nom du modèle'].values[0]}")
        st.write(f"📂 **Run ID** : {run_details['Run ID'].values[0]}")
        st.write(f"📅 **Date** : {run_details['Date'].values[0]}")
        st.write(f"📊 **R² Score** : {run_details['R² Score'].values[0]:.4f}")
        st.write(f"📉 **RMSE** : {run_details['RMSE'].values[0]:.4f}")

# Onglet 4 : Chargement des modèles
with tab4:
    st.header("🚀 Charger un Modèle MLflow et Prédire")

    mlflow.set_experiment("MLflow Streamlit")
    runs = mlflow.search_runs(order_by=["start_time DESC"])  # Trier du plus récent au plus ancien

    if runs.empty:
        st.info("Aucun modèle enregistré pour l'instant. Lance un entraînement !")
    else:
        # Extraire les informations clés des modèles enregistrés
        df_models = runs[["run_id", "tags.mlflow.runName"]]
        df_models.columns = ["Run ID", "Nom du Modèle"]

        # Sélectionner un modèle via son nom
        selected_model_name = st.selectbox("🎯 Choisissez un Modèle :", df_models["Nom du Modèle"], index=0)
        selected_run_id = df_models[df_models["Nom du Modèle"] == selected_model_name]["Run ID"].values[0]  # Récupération du Run ID correspondant

        # Charger le modèle MLflow sélectionné
        model_uri = f"runs:/{selected_run_id}/test_model"
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        st.success(f"✅ Modèle '{selected_model_name}' chargé avec succès !")

        # Formulaire pour entrer les caractéristiques du véhicule
        st.write("### 🏎️ Entrez les caractéristiques du véhicule")
        with st.form("prediction_form"):
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

            # Construction de l'input (features utilisées lors de l'entraînement)
            input_values = [m_kg, ec_cm3, ep_kw, erwltp, fuel_consumption] + ft_encoded
            input_data_df = pd.DataFrame([input_values], columns=['m (kg)', 'ec (cm3)', 'ep (KW)', 'Erwltp (g/km)', 'Fuel consumption', 'Ft_Diesel', 'Ft_Essence'])
            input_data_df = input_data_df.astype({
                "m (kg)": float,
                "ec (cm3)": float,
                "ep (KW)": float,
                "Erwltp (g/km)": float,
                "Fuel consumption": float,
                "Ft_Diesel": int,
                "Ft_Essence": int
                })

            submitted = st.form_submit_button("🔎 Prédire")

        # Faire la prédiction lorsque le formulaire est soumis
        if submitted:
            prediction = loaded_model.predict(input_data_df)
            st.success(f"📊 **Prédiction des émissions de CO₂ : {prediction[0]:.2f} g/km**")

