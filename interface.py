import gradio as gr

from modelexistant import gradio_predict
from newModel import predict_spam_svm

# Exemple d'email pour test rapide
EXEMPLE_EMAIL_1 = (
    "Hello,\n Your bank account has been suspended. Please click on the link below to verify your information: http://fauxsite.com \n Thank you."
)

EXEMPLE_EMAIL_2 = "Bonjour Nicolas, \n Pourrais-tu me faire un retour sur les essais du logiciel V2.0.0? \n Merci d’avance,\n Cordialement,"

with gr.Blocks() as demo:
    gr.Markdown("""
    # 🕵️‍♂️ EmailSleuth
    Détecteur d'emails suspects (spam/phishing) — basé sur IA
    """)
    with gr.Row():
        email_input = gr.Textbox(label="Collez ici le texte de l'email", value=EXEMPLE_EMAIL_1, lines=8)
    output = gr.Textbox(label="Résultat", lines=2)
    btn = gr.Button("Analyser l'email (Modèle Hugging Face)")
    btn.click(fn=gradio_predict, inputs=email_input, outputs=output)
    gr.Markdown("""
    *Modèle utilisé : mrm8488/bert-tiny-finetuned-sms-spam-detection (Hugging Face)*
    """)
    btn = gr.Button("Analyser l'email (Modèle entrainé pour le projet)")
    btn.click(fn=predict_spam_svm, inputs=email_input, outputs=output)
    gr.Markdown("""
        *Entrainement avec méthode SVM sur un dataset de 5170 mails classifés (Kaggle)*
        """)

# Lancement de l'application
if __name__ == "__main__":
    demo.launch()