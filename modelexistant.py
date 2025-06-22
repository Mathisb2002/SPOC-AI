from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
import torch

# Load model and tokenizer
model_id = "AntiSpamInstitute/spam-detector-bert-MoE-v2.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction function
def predict_email(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    labels = ["legitimate", "phishing"]
    pred_index = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_index].item()

    return labels[pred_index], round(confidence * 100, 2)  # Return percentage

# Explanation function
def explain_email(text):
    explainer = SequenceClassificationExplainer(model=model, tokenizer=tokenizer)
    label, _ = predict_email(text)
    label_idx = "LABEL_0" if label == "legitimate" else "LABEL_1"
    word_attributions = explainer(text, class_name=label_idx)
    explainer.visualize()
    return word_attributions

# For Gradio or UI integration
def gradio_predict(text):
    label, confidence = predict_email(text)
    return f"Email is classified as '{label}' with {confidence:.2f}% confidence."

def gradio_explain(text):
    return explain_email(text)
