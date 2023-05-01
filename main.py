import pandas as pd
import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tkinter import *
from models.SVMModelSpam import SVMModelSpam, import_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from models.SVMMODELMALADIE import MaladiesCardiaques
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

svmmodelSpam = SVMModelSpam()

# Chargement des données
emails_data = import_data('datasets/labeled_emails.csv')

# Séparation des données en ensembles d'entraînement et de test
emails = emails_data['email']
labels = np.where(emails_data['label'] == 'spam', 1, 0)  # Encoder les étiquettes en 0 et 1

# Diviser les données en ensembles d'entraînement et de test
emails_train, emails_test, labels_train, labels_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# Vectoriser les courriers électroniques en utilisant la transformation TF-IDF
vectorizer = TfidfVectorizer()
features_train = vectorizer.fit_transform(emails_train)
features_test = vectorizer.transform(emails_test)

# Prédiction sur l'ensemble de test
svmmodelSpam.fit(features_train, labels_train)

mail_pred = svmmodelSpam.predict(features_test)
# Évaluer les performances du modèle
accuracy = accuracy_score(labels_test, mail_pred)
precision = precision_score(labels_test, mail_pred)
recall = recall_score(labels_test, mail_pred)
f1 = f1_score(labels_test, mail_pred)
fpr, tpr, _ = roc_curve(labels_test, mail_pred)
auc_score = auc(fpr, tpr)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("AUC score:", auc_score)




#model maladie
# Créer une instance de la classe MaladiesCardiaques
mc = MaladiesCardiaques("train.csv", "test.csv")

# Créer le modèle SVM et l'entraîner sur les données d'entraînement
model = SVC(kernel='linear', C=1, random_state=42)
model.fit(mc.X_train, mc.y_train)

# Faire des prédictions sur les données de test
y_pred = model.predict(mc.X_test)

# Évaluer la précision du modèle
accuracy = accuracy_score(mc.y_test, y_pred)
print("Accuracy:", accuracy)


# Création de l'instance de la classe MaladiesCardiaques
mc = MaladiesCardiaques("train.csv", "test.csv")
model = SVC(kernel='linear', C=1, random_state=42)
model.fit(mc.X_train, mc.y_train)

# Fonction de prédiction
def predict():
    age = int(age_entry.get())
    sex = int(sex_entry.get())
    chol = int(chol_entry.get())
    bp = int(bp_entry.get())
    data = np.array([[age, sex, chol, bp]])
    data = mc.scaler.transform(data)
    prediction = model.predict(data)[0]
    if prediction == 0:
        result_label.config(text="non malade")
    else:
        result_label.config(text="malade")

# Création de la fenêtre principale
window = tk.Tk()
window.title("Prédiction des maladies cardiaques")

# Création des champs de saisie
age_label = tk.Label(window, text="Âge:")
age_label.grid(column=0, row=0)

age_entry = tk.Entry(window)
age_entry.grid(column=1, row=0)

sex_label = tk.Label(window, text="Sexe:")
sex_label.grid(column=0, row=1)

sex_entry = tk.Entry(window)
sex_entry.grid(column=1, row=1)

chol_label = tk.Label(window, text="Taux de cholestérol:")
chol_label.grid(column=0, row=2)

chol_entry = tk.Entry(window)
chol_entry.grid(column=1, row=2)

bp_label = tk.Label(window, text="Pression artérielle:")
bp_label.grid(column=0, row=3)

bp_entry = tk.Entry(window)
bp_entry.grid(column=1, row=3)

# Bouton de prédiction
predict_button = tk.Button(window, text="Prédire", command=predict)
predict_button.grid(column=0, row=4)

# Étiquette de résultat
result_label = tk.Label(window, text="")
result_label.grid(column=1, row=4)

window.mainloop()