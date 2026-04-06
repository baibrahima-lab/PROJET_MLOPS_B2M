import streamlit as st
import requests

# Configuration de la page
st.set_page_config(page_title="B2M Bank - Scoring", page_icon="🏦")

st.title("🏦 Système de Scoring Crédit - B2M")
st.markdown("---")

st.sidebar.header("📋 Informations du Client")

# Formulaire avec curseurs
loan = st.sidebar.number_input("Montant du prêt souhaité ($)", min_value=1000, max_value=100000, value=15000)
income = st.sidebar.number_input("Revenu Annuel ($)", min_value=10000, max_value=500000, value=45000)
years = st.sidebar.slider("Années d'ancienneté (Emploi)", 0, 40, 5)
fico = st.sidebar.slider("Score FICO (Crédit)", 300, 850, 650)

st.write("### Analyse du dossier en cours...")

# Bouton de prédiction
if st.button("Lancer l'analyse du risque"):
    # On prépare les données pour l'API
    payload = {
        "loan_amt_outstanding": float(loan),
        "income": float(income),
        "years_employed": int(years),
        "fico_score": int(fico)
    }
    
    try:
        # On appelle ton API locale (FastAPI)
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        res = response.json()
        
        # Affichage du résultat
        st.markdown("---")
        if res["verdict"] == "REFUSÉ":
            st.error(f"❌ **VERDICT : {res['verdict']}**")
            st.warning(f"Probabilité de défaut : {res['risque']}")
        else:
            st.success(f"✅ **VERDICT : {res['verdict']}**")
            st.info(f"Probabilité de défaut : {res['risque']}")
            
    except Exception as e:
        st.error("❌ Erreur : L'API n'est pas lancée. Lancez 'uvicorn app.main:app' dans un autre terminal.")

st.markdown("---")
st.caption("Projet MLOps B2M - Modèle XGBoost avec Validation Pydantic")
