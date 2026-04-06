import pandas as pd
import joblib
import os

def load_and_clean():
    # Correction du nom ici : bank-risk-manage_dataset.csv
    data_path = "data/bank-risk-manage_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Nettoyage rapide
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(df.median())
    
    # On s'assure que la cible est bien nommée 'default'
    # Si ta colonne s'appelle 'loan_status' ou 'target', on la renomme
    if 'loan_status' in df.columns:
        df = df.rename(columns={'loan_status': 'default'})
    elif 'target' in df.columns:
        df = df.rename(columns={'target': 'default'})
        
    return df

def save_artifact(model, name):
    os.makedirs("models", exist_ok=True)
    path = f"models/{name}.pkl"
    joblib.dump(model, path)
    print(f"📦 Modèle sauvegardé localement dans : {path}")
