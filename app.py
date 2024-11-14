from fastapi import FastAPI
import mlflow
import mlflow.sklearn
from pydantic import BaseModel
import numpy as np
import uvicorn
import pickle

# Charger le modèle depuis MLflow en utilisant le Run ID
model = mlflow.sklearn.load_model("runs:/978f0ccd0ba748b285d575084cdbb93c/spam_filter_model")

# Charger le tfidf_vectorizer depuis le fichier Pickle
with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf_vectorizer = pickle.load(file)

# Initialiser l'application FastAPI
app = FastAPI()

# Schéma pour la requête
class EmailRequest(BaseModel):
    text: str
        
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction de spam ! Utilisez /predict pour envoyer vos requêtes."}

# Fonction de prétraitement (assurez-vous qu'elle est importée ou définie)
def clean_text(text):
    # Nettoyage du texte, même logique que celle utilisée pendant l'entraînement
    import re
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Route pour la prédiction
@app.post("/predict")
def predict(request: EmailRequest):
    # Prétraitement du texte
    text_cleaned = clean_text(request.text)
    text_vectorized = tfidf_vectorizer.transform([text_cleaned])
    
    # Prédiction
    prediction = model.predict(text_vectorized)
    label = "spam" if prediction[0] == 1 else "ham"
    
    return {"prediction": label}

# Pour lancer le serveur, utilisez la commande suivante dans le terminal :
# uvicorn app:app --reload