from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

MODEL_DIR = "models"
app = FastAPI()

@app.get("/")
def read_root():
    """Endpoint de vérification de santé de l'API"""
    return {"status": "online", "project": "B2M MLOps Scoring"}

# Chargement des artefacts
try:
    model = joblib.load(os.path.join(MODEL_DIR, "final_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    # feature_names contient les 7 colonnes (brutes + dti_ratio)
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
except Exception as e:
    print(f"⚠️ Erreur chargement : {e}")
    model = None

class LoanRequest(BaseModel):
    income: float
    total_debt_outstanding: float
    loan_amt_outstanding: float
    fico_score: int
    years_employed: int
    credit_lines_outstanding: int

@app.post("/predict")
def predict(request: LoanRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:
        # 1. Créer le DataFrame avec les 6 colonnes brutes
        raw_df = pd.DataFrame([request.dict()])
        
        # 2. Définir les colonnes brutes (sans le dti_ratio) pour l'imputer
        # L'imputer a été entraîné sur ces colonnes dans cet ordre
        raw_features = [
            'credit_lines_outstanding', 'loan_amt_outstanding', 
            'total_debt_outstanding', 'income', 'years_employed', 'fico_score'
        ]
        
        # 3. ÉTAPE CLÉ : Imputer UNIQUEMENT sur les colonnes brutes
        data_to_impute = raw_df[raw_features]
        imputed_values = imputer.transform(data_to_impute)
        df_clean = pd.DataFrame(imputed_values, columns=raw_features)
        
        # 4. FEATURE ENGINEERING : Ajouter le ratio après l'imputation
        df_clean['dti_ratio'] = df_clean['total_debt_outstanding'] / df_clean['income']
        
        # 5. ALIGNEMENT FINAL : Ranger les 7 colonnes dans l'ordre du modèle
        data_final = df_clean[feature_names]
        
        # 6. SCALING ET PRÉDICTION
        data_scaled = scaler.transform(data_final)
        proba = float(model.predict_proba(data_scaled)[0, 1])
        
        verdict = "REFUSÉ" if proba > 0.5 else "APPROUVÉ"
        
        return {
            "verdict": verdict,
            "probability": f"{proba:.2%}",
            "dti": round(df_clean['dti_ratio'].iloc[0], 4)
        }
    except Exception as e:
        print(f"❌ Erreur API : {e}")
        raise HTTPException(status_code=500, detail=str(e))