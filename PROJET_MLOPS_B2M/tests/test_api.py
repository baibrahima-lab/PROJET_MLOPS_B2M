from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Serveur B2M Opérationnel", "status": "Online"}

def test_fico_validation():
    # Test un score FICO trop haut (doit être rejeté par Pydantic)
    payload = {
        "loan_amt_outstanding": 1000,
        "income": 50000,
        "years_employed": 5,
        "fico_score": 999
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # 422 = Erreur de validation
