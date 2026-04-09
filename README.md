C'est tout à fait pertinent. Le Markdown est le standard pour la documentation technique : il permet un rendu propre, une structure hiérarchisée et une intégration directe des badges de statut.

Voici la version intégrale, **ultra-détaillée et prête à l'emploi** de ton fichier `README.md`. Elle inclut toutes les spécificités de ton projet (Python 3.12, scikit-learn 1.3.2, FastAPI, XGBoost et AWS).

---

```markdown
# 🏦 Système de Scoring de Crédit Intelligent (MLOps)

![CI Status](https://github.com/Bourko235/PROJET_MLOPS_B2M/actions/workflows/ci.yml/badge.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![AWS](https://img.shields.io/badge/Cloud-AWS_ECS-FF9900)

## 📖 Résumé du Projet
Face à l'augmentation des taux de défaut sur les prêts personnels, ce projet propose une solution industrielle de **Credit Scoring**. L'objectif est de fournir aux analystes crédit un outil capable de prédire instantanément la probabilité de défaut en s'appuyant sur un modèle de Machine Learning robuste et une architecture micro-services.

> **Impact Métier :** Réduction du risque de crédit via une évaluation automatisée et traçable de la capacité de remboursement des clients, permettant une optimisation de l'allocation du capital.

---

## 🛠️ Stack Technique
* **Data Science :** `Pandas`, `Numpy`, `Scikit-learn 1.3.2`, `XGBoost`.
* **MLOps & Tracking :** `MLflow` (Gestion des expérimentations et des artefacts).
* **Développement API :** `FastAPI` (Backend haute performance avec documentation Swagger intégrée).
* **Interface Utilisateur :** `Streamlit` (Dashboard interactif pour les prédictions en temps réel).
* **Containerisation :** `Docker` & `Docker Compose` (Isolation et reproductibilité).
* **CI/CD :** `GitHub Actions` (Validation continue des tests unitaires).
* **Cloud Infrastructure :** `Amazon ECR` (Registre d'images) & `Amazon ECS Fargate` (Déploiement Serverless).

---

## 🏗️ Architecture du Projet
Le projet respecte une séparation stricte des responsabilités pour garantir la maintenance et la scalabilité :

```text
├── .github/workflows/    # Pipelines CI (Tests) & CD (Déploiement AWS)
├── app/                  # Dossier Application
│   ├── main.py           # Backend API (FastAPI)
│   └── ui.py             # Frontend (Streamlit)
├── data/                 # Datasets de crédit (CSV)
├── models/               # Artefacts sérialisés (Scaler, Imputer, Modèle .pkl)
├── notebooks/            # Exploration de données (EDA) & Rapport d'expertise
├── src/                  # Logique Coeur
│   ├── data_manager.py   # Pipeline de preprocessing unifié
│   └── train.py          # Script d'entraînement et logs MLflow
├── tests/                # Tests unitaires (Pytest & Httpx)
├── docker-compose.yml    # Orchestration multi-conteneurs (API + UI)
├── Dockerfile            # Packaging de l'image de production
└── requirements.txt      # Dépendances (Versions figées pour la stabilité)
```

---

## 🚀 Installation et Utilisation

### 🐳 Mode Docker (Recommandé)
Pour lancer l'écosystème complet en local sans installation de dépendances :
```bash
docker-compose up --build
```
* **API Backend :** `http://localhost:8000` (Accès Swagger : `http://localhost:8000/docs`)
* **Interface UI :** `http://localhost:8501`

### 🐍 Installation Manuelle
```bash
# 1. Création et activation de l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Installation des dépendances
pip install -r requirements.txt

# 3. Lancement de l'API
uvicorn app.main:app --reload
```

---

## 🧪 Pipeline de Qualité (CI/CD)
Chaque `push` déclenche un workflow automatisé sur GitHub Actions pour garantir l'intégrité du système :
1.  **Environnement :** Setup de Python 3.12 et installation des dépendances.
2.  **Dry Run :** Vérification du fonctionnement du `data_manager.py`.
3.  **Tests Unitaires :** Validation des endpoints `/predict` et `/` (Health Check).
4.  **Contract Testing :** Vérification de la conformité des réponses JSON de l'API.

---

## 📈 Analyse et Modélisation (EDA)
L'étape d'exploration a permis de mettre en évidence les signaux faibles et forts du risque :
* **Facteur Clé :** Le **DTI Ratio (Debt-to-Income)** a été identifié comme la variable la plus discriminante.
* **Preprocessing :** Gestion des valeurs manquantes par médiane et des valeurs aberrantes par Winsorization (99e percentile).
* **Modèle :** XGBoost avec stratification pour compenser le déséquilibre des classes (18.5% de défauts).
* **Gouvernance :** Utilisation de MLflow pour le suivi des métriques (Précision, Recall, AUC).

---

## ☁️ Déploiement Cloud (AWS)
Le déploiement est orchestré de manière automatisée vers **Amazon Web Services** :
* **Storage :** Images Docker versionnées sur **Amazon ECR**.
* **Runtime :** Exécution via **Amazon ECS** sur infrastructure **Fargate** (Serverless).
* **Automatisation :** Workflow GitHub Actions dédié utilisant les accès IAM sécurisés.

---

**Auteurs :** Ibrahima Ba, Mahamat Sultan, Moustapha Mendy 

