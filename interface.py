import gradio as gr
import csv
import random
import os

from modelexistant import gradio_predict
from newModel import predict_spam_svm


# Exemple d'email pour test rapide
EXEMPLE_EMAIL_1 = (
    "Hello,\n Your bank account has been suspended. Please click on the link below to verify your information: http://fauxsite.com \n Thank you."
)

EXEMPLE_EMAIL_2 = "Bonjour Nicolas, \n Pourrais-tu me faire un retour sur les essais du logiciel V2.0.0? \n Merci d’avance,\n Cordialement,"



with gr.Blocks() as demo:
    gr.Markdown("""
    # 🕵️‍♂️ MailShield
    Détecteur d'emails suspects (spam/phishing) — basé sur IA
    """)
    with gr.Row():
        with gr.Column():
            btn_ham = gr.Button("Exemple Ham")
            btn_spam = gr.Button("Exemple Spam")
    with gr.Row():
        with gr.Column():
            email_input = gr.Textbox(label="Collez ici le texte de l'email", value=EXEMPLE_EMAIL_1, lines=8)
    output = gr.Textbox(label="Résultat", lines=2)
    btn = gr.Button("Analyser l'email (Modèle Hugging Face)")
    btn.click(fn=gradio_predict, inputs=email_input, outputs=output)
    gr.Markdown("""
    *Modèle utilisé : AntiSpamInstitute/spam-detector-bert-MoE-v2.2(Hugging Face)*
    """)
    btn = gr.Button("Analyser l'email (Modèle entrainé pour l'occasion)")
    btn.click(fn=predict_spam_svm, inputs=email_input, outputs=output)
    gr.Markdown("""
        *Entrainement avec méthode SVM sur un dataset de 5170 mails classifés (Kaggle)*
        """)
    # Actions des boutons exemple
    btn_ham.click(fn=lambda: EXEMPLE_EMAIL_2, inputs=None, outputs=email_input)
    btn_spam.click(fn=lambda: EXEMPLE_EMAIL_1, inputs=None, outputs=email_input)

# Lancement de l'application
if __name__ == "__main__":
    demo.launch()