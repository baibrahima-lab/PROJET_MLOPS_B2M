# 🏦 Système de Prédiction de Risque de Crédit (MLOps)
![CI Status](https://github.com/Bourko235/PROJET_MLOPS_B2M/actions/workflows/ci.yml/badge.svg)
![Status](https://img.shields.io/badge/Status-In--Development-yellow)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)

## 📋 Aperçu du Projet
Dans le secteur de la banque de détail, la gestion du risque de défaut sur les prêts personnels est critique. Ce projet implémente une **démarche MLOps de bout en bout (End-to-End)** pour prédire la probabilité de défaut de paiement d'un client.

**Objectif métier :** Identifier les profils à risque pour allouer le capital de manière optimale et minimiser les pertes financières de la banque.

---

## 🛠️ Stack Technique
* **Data Science :** `Pandas`, `Scikit-learn`, `XGBoost`.
* **MLOps & Tracking :** `MLflow` (Expérimentations et versioning de modèles).
* **Interface Utilisateur :** `Streamlit` (Dashboard de prédiction en temps réel).
* **DevOps :** `Docker` (Containerisation).
* **CI/CD :** `GitHub Actions` (Déploiement automatisé).
* **Cloud :** [Azure / AWS / Streamlit Cloud - À compléter].

---

## 🏗️ Architecture du Projet
```text
├── .github/workflows/   # Pipelines CI/CD
├── data/                # Datasets (Bruts et transformés)
├── src/                 # Code source
│   ├── preprocessing.py # Nettoyage et Feature Engineering
│   ├── train.py        # Entraînement et logs MLflow
│   └── app.py          # Application Streamlit
├── notebooks/           # Exploration de données (EDA)
├── requirements.txt     # Dépendances Python
└── Dockerfile           # Configuration Docker