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
Model : SVM + Dataset aveec environ 5000 mails 

source du dataset : https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification?select=email.csv
Accuracy: 0.9839
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       966
           1       0.99      0.89      0.94       149

    accuracy                           0.98      1115
   macro avg       0.99      0.94      0.96      1115
weighted avg       0.98      0.98      0.98      1115

