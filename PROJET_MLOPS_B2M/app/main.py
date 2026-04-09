import logging
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 2. Paramètres et Chargement
MODEL_PATH = "models/final_model.pkl"
SEUIL = 0.5  # Seuil de décision (50%)

app = FastAPI(title="B2M Banking API - Production Version")

# 3. Schémas de données (Pydantic)
class LoanRequest(BaseModel):
    loan_amt_outstanding: float = Field(..., gt=0)
    income: float = Field(..., gt=0)
    years_employed: int = Field(..., ge=0)
    fico_score: int = Field(..., ge=300, le=850)

class LoanResponse(BaseModel):
    prediction: int
    probability: float
    verdict: str
    timestamp: str

# 4. Gestion du modèle
def load_model():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Fichier modèle introuvable : {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# 5. Événements de cycle de vie
@app.on_event("startup")
async def startup_event():
    logger.info("�� Application B2M démarrée")
    if model:
        logger.info(f"✅ Modèle chargé avec succès depuis : {MODEL_PATH}")
    logger.info(f"⚙️ Seuil de classification : {SEUIL}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Application arrêtée")

# 6. Endpoint /predict amélioré
@app.post("/predict", response_model=LoanResponse)
def predict(request: LoanRequest):
    """Prédit le risque de défaut pour un client."""
    if model is None:
        logger.error("❌ Tentative de prédiction sans modèle chargé")
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    try:
        logger.debug(f"Requête reçue : {request}")
        
        # Préparation des données (on utilise un DataFrame pour garder les noms de colonnes)
        input_df = pd.DataFrame([request.dict()])
        
        # Calcul des probabilités
        proba = float(model.predict_proba(input_df)[0, 1])
        prediction = int(proba >= SEUIL)
        verdict = "REFUSÉ" if prediction == 1 else "APPROUVÉ"
        
        logger.info(
            f"✅ Prédiction effectuée | "
            f"FICO: {request.fico_score} | "
            f"Prob: {proba:.4f} | "
            f"Décision: {verdict}"
        )
        
        return LoanResponse(
            prediction=prediction,
            probability=round(proba, 4),
            verdict=verdict,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as exc:
        logger.error(f"❌ Erreur lors de la prédiction : {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
