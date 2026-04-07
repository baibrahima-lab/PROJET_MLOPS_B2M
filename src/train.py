import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score

# Import local 
from data_manager import load_and_preprocess_data, save_model

def run_training():
    mlflow.set_experiment("B2M_Scoring_Loan")
    print("🚀 Début du benchmarking MLOps (3 modèles)...")
    
    # 1. Chargement des données déjà scalées et nettoyées (outliers, NaN gérés)
    X_train, X_test, y_train, y_test, feat_names = load_and_preprocess_data()
    
    # 2. Définition du dictionnaire de modèles (Consigne : 3 algos)
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, scale_pos_weight=5, learning_rate=0.1)
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"Run_{name}"):
            try:
                print(f"--- Entraînement de {name} ---")
                
                # Entraînement
                model.fit(X_train, y_train)
                
                # Prédictions
                y_pred = model.predict(X_test)
                
                # 3. Calcul des métriques (Crucial pour le risque bancaire)
                acc = accuracy_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred) # $$Recall = \frac{TP}{TP + FN}$$
                f1 = f1_score(y_test, y_pred)
                
                # 4. Logging MLflow
                mlflow.log_param("model_type", name)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)
                
                # Sauvegarde du modèle dans MLflow (Lineage)
                if name == "XGBoost":
                    mlflow.xgboost.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")
                
                print(f"✅ {name} -> Accuracy: {acc:.4f} | Recall: {rec:.4f}")
                
                # Sauvegarde locale du "Champion" (ex: XGBoost) pour l'API
                if name == "XGBoost":
                    save_model(model, "final_model")
                    
            except Exception as e:
                print(f"❌ Erreur sur {name} : {e}")
                mlflow.log_param("error", str(e))

if __name__ == "__main__":
    run_training()