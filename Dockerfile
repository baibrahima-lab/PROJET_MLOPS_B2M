# 1. Image de base légère avec Python 3.12
FROM python:3.12-slim

# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3. Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copier les fichiers de dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier tout le reste du projet
COPY . .

# 6. Exposer les ports (8000 pour FastAPI, 8501 pour Streamlit)
EXPOSE 8000
EXPOSE 8501

# On ne met pas de CMD ici, on va gérer ça dans le Docker Compose