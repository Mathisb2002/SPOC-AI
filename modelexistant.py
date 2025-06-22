
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Chargement du modèle pré-entraîné (léger et rapide)
MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Liste des labels du modèle (0 = ham/non-spam, 1 = spam)
LABELS = ["Email non suspect", "Attention : cet email semble suspect"]
ICONS = ["✅", "⚠️"]

# Fonction de prédiction réutilisable
def predict(text: str):


    """
    Prend un texte d'email en entrée et retourne :
    - Le label prédictif (spam ou non)
    - Le score de confiance (probabilité)
    """
    # Prétraitement et tokenization
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
    label = LABELS[pred_idx]
    icon = ICONS[pred_idx]
    return f"{icon} {label}", confidence


# Fonction pour Gradio (affiche aussi le score de confiance)
def gradio_predict(text):
    label, confidence = predict(text)
    return f"{label}\n\nScore de confiance : {confidence:.2%}"

