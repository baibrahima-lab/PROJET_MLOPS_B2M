import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configuration
DATA_PATH = "PROJET_MLOPS_B2M/data/bank-risk-manage_dataset.csv"
MODEL_DIR = "models"

def handle_outliers(df, columns):
    """
    Plafonne les valeurs aberrantes au 99ème percentile (Winsorization).
    Cela évite que des revenus de 2 millions d'euros ne faussent la Régression Logistique.
    """
    df_clipped = df.copy()
    for col in columns:
        if col in df_clipped.columns:
            upper_limit = df_clipped[col].quantile(0.99)
            df_clipped[col] = df_clipped[col].clip(upper=upper_limit)
    return df_clipped

def load_and_preprocess_data(save_artifacts=True):
    """
    Pipeline MLOps Complet : Nettoyage -> Imputation -> Outliers -> Engineering -> Scaling.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    # 1. Élagage initial
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    
    # 2. Gestion des données manquantes (Imputation)
    imputer = SimpleImputer(strategy='median')
    cols_to_fix = df.columns.drop('default') if 'default' in df.columns else df.columns
    df[cols_to_fix] = imputer.fit_transform(df[cols_to_fix])
    
    # 3. Gestion des valeurs aberrantes (Outliers)
    # On cible les colonnes financières identifiées dans l'EDA
    cols_outliers = ['income', 'total_debt_outstanding', 'loan_amt_outstanding']
    df = handle_outliers(df, cols_outliers)
    
    # 4. Feature Engineering (Le Ratio DTI)
    # Formule : $DTI = \frac{Total\_Debt\_Outstanding}{Income}$
    df['dti_ratio'] = df['total_debt_outstanding'] / df['income']
    
    # 5. Séparation Features/Cible
    X = df.drop(columns=['default'])
    y = df['default']
    
    # 6. Split Train/Test (Stratification pour le déséquilibre de 18.5%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 7. Scaling (Standardisation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 8. SAUVEGARDE DES OBJETS (Le "Lineage")
    if save_artifacts:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
        joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))
        joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "feature_names.pkl"))
        print(f"✅ Artefacts (Scaler, Imputer, Features) sauvegardés dans {MODEL_DIR}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

 #Test rapide pour vérifier que tout fonctionne
if __name__ == "__main__":
    try:
        print("🧪 Lancement du test de nettoyage...")
        X_train, X_test, y_train, y_test, cols = load_and_preprocess_data()
        
        print("✅ TEST RÉUSSI !")
        print(f"Structure des données : {X_train.shape[0]} lignes pour l'entraînement.")
        print(f"Colonnes générées : {list(cols)}")
        
    except Exception as e:
        print(f"❌ ÉCHEC DU TEST : {e}")

    