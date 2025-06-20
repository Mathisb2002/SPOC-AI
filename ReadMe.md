README — EmailSleuth : Détecteur de spam/phishing pour emails
-----------------------------------------------------------
Ce script Python propose une mini-application locale pour détecter si un texte d'email est suspect (spam ou phishing) ou non, en utilisant un modèle pré-entraîné Hugging Face.

Fonctionnalités :
- Interface simple via Gradio : collez un email, obtenez une réponse et un score de confiance.
- Utilise un modèle léger (BERT-tiny fine-tuned pour le spam).
- Fonction `predict(text)` réutilisable.
- Exemple d'email inclus pour test rapide.

Dépendances :
- transformers
- torch
- gradio

Installation (si besoin) :
    pip install transformers torch gradio

Usage :
    python emailsleuth.py

Auteur : AI + GPT-4
"""
