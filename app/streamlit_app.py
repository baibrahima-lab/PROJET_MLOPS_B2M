import streamlit as st
import requests
import subprocess
import time
import requests

# Lancer l'API en arrière-plan si elle n'est pas détectée
try:
    requests.get("http://localhost:8000/")
except:
    subprocess.Popen(["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"])
    time.sleep(5) # On laisse 5s à l'API pour démarrer

st.set_page_config(page_title="B2M Bank - Scoring", page_icon="🏦")
st.title("🏦 Système de Scoring Crédit - B2M")

st.sidebar.header("📋 Profil Client")
income = st.sidebar.number_input("Revenu Annuel ($)", value=50000)
total_debt = st.sidebar.number_input("Dette Totale Actuelle ($)", value=15000)
loan_amt = st.sidebar.number_input("Montant du Prêt souhaité ($)", value=10000)
fico = st.sidebar.slider("Score FICO", 300, 850, 700)
years = st.sidebar.slider("Années d'ancienneté", 0, 45, 5)
credit_lines = st.sidebar.slider("Crédits en cours", 0, 20, 2)

if st.button("Lancer l'analyse"):
    payload = {
        "income": float(income),
        "total_debt_outstanding": float(total_debt),
        "loan_amt_outstanding": float(loan_amt),
        "fico_score": int(fico),
        "years_employed": int(years),
        "credit_lines_outstanding": int(credit_lines)
    }
    
    try:
        response = requests.post("http://api:8000/predict", json=payload)
        res = response.json()
        
        if response.status_code == 200:
            st.markdown("---")
            if res["verdict"] == "REFUSÉ":
                st.error(f"❌ **VERDICT : {res['verdict']}**")
            else:
                st.success(f"✅ **VERDICT : {res['verdict']}**")
            
            st.metric("Probabilité de défaut", res["probability"])
            st.info(f"Ratio DTI calculé : {res['dti']}")
        else:
            st.error(f"Erreur API : {res.get('detail', 'Inconnue')}")
    except:
        st.error("L'API n'est pas lancée (uvicorn).")


       
