import gradio as gr

from modelexistant import gradio_predict

# Exemple d'email pour test rapide
EXEMPLE_EMAIL = (
    "Bonjour,\n\nVotre compte bancaire a été suspendu. Veuillez cliquer sur le lien ci-dessous pour vérifier vos informations : http://fauxsite.com\n\nMerci."
)

with gr.Blocks() as demo:
    gr.Markdown("""
    # 🕵️‍♂️ EmailSleuth
    Détecteur d'emails suspects (spam/phishing) — basé sur IA
    """)
    with gr.Row():
        email_input = gr.Textbox(label="Collez ici le texte de l'email", value="", lines=8)
    output = gr.Textbox(label="Résultat", lines=2)
    btn = gr.Button("Analyser l'email")
    btn.click(fn=gradio_predict, inputs=email_input, outputs=output)
    gr.Markdown("""
    *Modèle utilisé : mrm8488/bert-tiny-finetuned-sms-spam-detection (Hugging Face)*
    """)

# Lancement de l'application
if __name__ == "__main__":
    demo.launch()