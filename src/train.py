import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import mlflow
import mlflow.xgboost

# Import local
from src.data_manager import load_and_clean, save_artifact

def run_training():
    # On définit le nom de l'expérience pour le prof
    mlflow.set_experiment("B2M_Scoring_Loan")
    
    print("--- Début de l'entraînement avec tracking MLflow ---")
    
    with mlflow.start_run(run_name="XGBoost_Standard_Run"):
        try:
            # 1. Chargement
            df = load_and_clean()
            X = df.drop('default', axis=1)
            y = df['default']
            
            # 2. Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            
            # 3. Paramètres du modèle (On les logge pour le prof)
            params = {
                "n_estimators": 100,
                "scale_pos_weight": 5,
                "learning_rate": 0.1,
                "max_depth": 6
            }
            mlflow.log_params(params)
            
            # 4. Entraînement
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            # 5. Métriques (On les logge aussi)
            acc = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", acc)
            
            # 6. Sauvegarde du modèle dans MLflow (Le "Lineage")
            mlflow.xgboost.log_model(model, "model_b2m")
            
            # Sauvegarde locale pour l'API
            save_artifact(model, "final_model")
            
            print(f"✅ Succès ! Précision : {acc:.4f}")
            print("📊 Allez sur http://127.0.0.1:5000 pour voir le résultat.")
            
        except Exception as e:
            print(f"❌ Erreur pendant l'entraînement : {e}")
            mlflow.log_param("error", str(e))

if __name__ == "__main__":
    run_training()
