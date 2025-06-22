import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

def prepare_data():
    df = pd.read_csv('dataset/email.csv')
    df['label'] = df['label'].map({'ham':0, 'spam':1})
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    #les mails du dataset sont en anglais, on traitera alors les mots vides de sens de la langue anglaise (stop words)
    vectorizer = CountVectorizer(stop_words='english')

    #on transforme les données en vecteurs numériques pour l'entrainement
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    #on stock notre vectorizer
    joblib.dump(vectorizer, 'models/vectorizer.joblib')

    return X_train_vec, y_train, X_test_vec, y_test

def training(X_train_vec, y_train):
    svm_model = SVC(kernel='linear', probability=True)  # kernel='linear' pour une séparation simple
    svm_model.fit(X_train_vec, y_train)
    #on stock notre modèle entrainé pour l'utiliser dans notre interface
    joblib.dump(svm_model, 'models/svm_spam_model.joblib')

def testMetrics(X_test_vec, y_test):
    #un bon moyen de tester un modèle est de calculer sa précision sur la partie du dataset dédiée aux tests

    #on importe notre modèle
    model = joblib.load('models/svm_spam_model.joblib')
    #on fait les prédictions
    y_pred = model.predict(X_test_vec)
    #sklearn fournit une fonction permettant de nous donner la précision du modèle
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(classification_report(y_test, y_pred))

def predict_spam_svm(email_text):
    model = joblib.load('models/svm_spam_model.joblib')
    vectorizer = joblib.load('models/vectorizer.joblib')

    # on prétraite notre texte (transformation en vecteur, vider les stop words ...)
    vectorized_text = vectorizer.transform([email_text]).toarray()

    # on utilise notre modèle entrainé pour prédire si un mail est spam or ham
    prediction = model.predict(vectorized_text)[0]
    proba = model.predict_proba(vectorized_text)[0][1]  # Probabilité d'être spam
    proba_confiance = proba if prediction == 1 else 1-proba
    label = 'Spam' if prediction == 1 else 'Ham'
    return f"{label}\n\nScore de confiance : {proba_confiance:.2%}"


#X_train_vec, y_train, X_test_vec, y_test = prepare_data()

#training(X_train_vec, y_train)

#testMetrics(X_test_vec, y_test)

