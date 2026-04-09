import streamlit as st
import requests
import subprocess
import time
import requests


# On définit l'URL
API_URL = "http://127.0.0.1:8000"

def start_api():
    try:
        # On teste la connexion
        res = requests.get(API_URL, timeout=1)
        if res.status_code == 200:
            return True
    except:
        # Si ça échoue, on lance uvicorn
        subprocess.Popen(["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"])
        time.sleep(8) # On attend que ça chauffe
        return False

# APPEL DE LA FONCTION
api_is_up = start_api()

# ... (ton code de vérification globale qui affiche le bandeau vert) ...

if st.button("Lancer l'analyse"):
    # ❌ SUPPRIME l'ancien "if not api_is_up: st.error(...)" ici
    
    with st.spinner("Analyse du profil en cours..."):
        # 1. Préparer les données (tes sliders/inputs)
        data = {
            "income": income,
            "debt": debt,
            "loan_amount": loan_amount,
            "fico_score": fico_score,
            "years_employed": years_employed,
            "open_acc": open_acc
        }
        
        try:
            # 2. Envoyer la requête POST à l'API
            response = requests.post(f"{API_URL}/predict", json=data)
            
            if response.status_code == 200:
                prediction = response.json()
                # 3. Afficher le résultat
                st.metric("Probabilité de défaut", f"{prediction['probability']:.2%}")
                if prediction['prediction'] == 1:
                    st.error("⚠️ Risque élevé détecté")
                else:
                    st.success("✅ Profil validé")
            else:
                st.error(f"Erreur API : {response.status_code}")
                
        except Exception as e:
            st.error(f"Erreur de connexion : {e}")

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


       
