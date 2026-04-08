import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import socket
import subprocess
import sys
import time
import os
from mlflow import MlflowClient
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score

# Importation locale
from data_manager import load_and_preprocess_data, save_model

# --- CONFIGURATION (Notions TP) ---
HOST = "127.0.0.1"
PORT = 8080
TRACKING_URI = f"http://{HOST}:{PORT}"
MODEL_NAME = "B2M_Scoring_Champion" # Nom dans le Model Registry

def ensure_mlflow_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((HOST, PORT)) == 0: return
    
    print(f"🛰️ Lancement automatique du serveur MLflow...")
    subprocess.Popen([
        sys.executable, "-m", "mlflow", "server",
        "--host", HOST, "--port", str(PORT),
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

def run_training():
    ensure_mlflow_server()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # 1. Tags d'Expérience (Notion TP)
    exp_name = "B2M_Scoring_Loan_V2"
    exp_tags = {
        "project_name": "B2M-Banking",
        "team": "Stores-ML-B2M",
        "mlflow.note.content": "Pipeline automatisé avec Model Registry."
    }

    exp = client.get_experiment_by_name(exp_name)
    experiment_id = exp.experiment_id if exp else client.create_experiment(name=exp_name, tags=exp_tags)

    X_train, X_test, y_train, y_test, feat_names = load_and_preprocess_data()
    input_example = pd.DataFrame(X_train[:3], columns=feat_names) # Signature propre

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(n_estimators=100, scale_pos_weight=5, learning_rate=0.1)
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"Run_{name}", experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            model.fit(X_train, y_train)
            
            # Métriques
            y_pred = model.predict(X_test)
            rec = recall_score(y_test, y_pred)
            mlflow.log_metric("recall", rec)
            
            # --- AUTOMATISATION 1 : Logging avec Signature ---
            if name == "XGBoost":
                # On log le modèle et on l'enregistre dans le catalogue (Registry)
                mlflow.xgboost.log_model(
                    model, 
                    artifact_path="model", 
                    input_example=input_example,
                    registered_model_name=MODEL_NAME # <--- Création auto dans le catalogue
                )
                save_model(model, "final_model")
                
                # --- AUTOMATISATION 2 : Transition vers Production ---
                # On récupère la dernière version enregistrée
                latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version
                
                # On le marque comme "Production" (Alias moderne de MLflow 2.x)
                client.set_registered_model_alias(MODEL_NAME, "champion", latest_version)
                
                # Ou l'ancien système de "Stages" (plus compatible avec certains TPs)
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=latest_version,
                    stage="Production"
                )
                
                print(f"🏆 {name} (v{latest_version}) est maintenant le CHAMPION en Production !")

if __name__ == "__main__":
    run_training()