
# Imports nécessaires
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Chargement du modèle pré-entraîné (léger et rapide)
MODEL_NAME = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
    label = LABELS[pred_idx]
    icon = ICONS[pred_idx]
    return f"{icon} {label}", confidence

# Exemple d'email pour test rapide
EXEMPLE_EMAIL = (
    "Bonjour,\n\nVotre compte bancaire a été suspendu. Veuillez cliquer sur le lien ci-dessous pour vérifier vos informations : http://fauxsite.com\n\nMerci."
)

# Fonction pour Gradio (affiche aussi le score de confiance)
def gradio_predict(text):
    label, confidence = predict(text)
    return f"{label}\n\nScore de confiance : {confidence:.2%}"

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🕵️‍♂️ EmailSleuth
    Détecteur d'emails suspects (spam/phishing) — basé sur IA
    """)
    with gr.Row():
        email_input = gr.Textbox(label="Collez ici le texte de l'email", value=EXEMPLE_EMAIL, lines=8)
    output = gr.Textbox(label="Résultat", lines=2)
    btn = gr.Button("Analyser l'email")
    btn.click(fn=gradio_predict, inputs=email_input, outputs=output)
    gr.Markdown("""
    *Modèle utilisé : mrm8488/bert-tiny-finetuned-sms-spam-detection (Hugging Face)*
    """)

# Lancement de l'application
if __name__ == "__main__":
    demo.launch() 