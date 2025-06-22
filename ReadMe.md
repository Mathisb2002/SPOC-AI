# 🕵️‍♂️ MailShield — Email Spam/Phishing Detector

MailShield is a local mini-application that detects whether an email text is suspicious (spam or phishing) or not, using either a pre-trained Hugging Face model or an SVM model trained on a public dataset.

---

## Main Features
- **Simple Gradio interface**: paste an email or use the provided examples, get an instant answer.
- **Dual AI engine**:
  - Hugging Face model (BERT-tiny fine-tuned for spam/phishing)
  - SVM model trained on a Kaggle dataset (5170 emails)
- **Example buttons**: automatically insert a real (ham or spam) email from the dataset to test the tool.
- **Confidence score** (for the Hugging Face model).
- **Clear and reusable Python code**.

---

## Installation

1. **Clone the repository (recommended)**

```bash
git clone https://github.com/Mathisb2002/SPOC-AI.git
cd SPOC-AI-main
```

2. **Or download the ZIP archive**

- Go to the [GitHub repository](https://github.com/Mathisb2002/SPOC-AI)
- Click on the green "Code" button, then "Download ZIP"
- Extract the ZIP file on your computer
- Open the project folder in your favorite code editor (VS Code, PyCharm, etc.)

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

Or, if the file does not exist:

```bash
pip install transformers torch gradio
```

---

## Launching the Application

```bash
python interface.py
```

The Gradio interface will open in your default web browser.

---

## User Guide

1. **Choose an example**:
   - Click on "Example Ham" to insert a legitimate (ham) email from the dataset.
   - Click on "Example Spam" to insert a suspicious (spam) email from the dataset.
   - You can also paste your own email text in the input box.

2. **Analyze the email**:
   - Click on "Analyze email (Hugging Face Model)" to use the BERT-tiny model.
   - Click on "Analyze email (Custom Trained Model)" to use the SVM model.

3. **Read the result**:
   - The verdict will appear in the "Result" box.
   - For the Hugging Face model, a confidence score is also displayed.

---

## Technical Details

- **Hugging Face model**: AntiSpamInstitute/spam-detector-bert-MoE-v2.2
- **SVM model**: trained on the Kaggle dataset [email.csv](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification?select=email.csv)
- **SVM Accuracy**: 0.9839

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| Ham (0)      | 0.98      | 1.00   | 0.99     | 966     |
| Spam (1)     | 0.99      | 0.89   | 0.94     | 149     |
| **Total**    | **0.98**  | **0.98**| **0.98** | 1115    |

---

## Author
AI + GPT-4

---

## Notes
- The dataset used is available on Kaggle.
- The application runs locally; no data is sent externally.
- For questions or improvements, open an issue or contact the author.

