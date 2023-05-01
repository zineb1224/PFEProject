import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tkinter import *
from models.SVMModelSpam import SVMModelSpam, import_data
from models.SVMMODELMALADIE import SVMMODELMALADIE
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


    # Création d'une instance de la classe
    model = SVMMODELMALADIE()

    # Entraînement du modèle avec le fichier de données d'entraînement
    model.train('maladie.csv')

    # Prédiction sur les données à prédire à partir du fichier correspondant
    predictions = model.predict('maladie_to_predict.csv')

    # Affichage des prédictions
    print('Prédictions de maladies cardiaques :',predictions)





