import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error 

# Import des datasets pré-traités
df= pd.read_csv('data/DF2023-22-21_Concat_Finale_2.csv')
df.columns = df.columns.str.strip() 

# 1. Modèles avec features de base (sans marques)
baseline_features = ['m (kg)', 'ec (cm3)', 'ep (KW)', 'Erwltp (g/km)', 'Fuel consumption', 'Ft_Diesel', 'Ft_Essence']
target = 'Ewltp (g/km)'
X_baseline = df[baseline_features]
y_baseline = df[target]
X_train, X_test, y_train, y_test = train_test_split(X_baseline, y_baseline, test_size=0.2, random_state=42)

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
print(f"R2 : {r2}")
print(f"RMSE : {rmse}")

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():

    # Log the loss metric
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("rmse", rmse)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for DF2023-22-21_Concat_Finale_2")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="test_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )


# Load the model back for predictions as a generic Python Function model (optionnelle) 
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)
result = pd.DataFrame(X_test, columns=baseline_features)
result["actual_class"] = y_test
result["predicted_class"] = predictions
print(result[:4])