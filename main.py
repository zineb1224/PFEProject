import pandas as pd
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tkinter import *
from models.SVMModelSpam import SVMModelSpam, import_data
from models.SVMMODELMALADIE import MaladiesCardiaques
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc ,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

print("Accuracy:", accuracy)
print("Precision:", precision)

confusion = confusion_matrix(labels_test, mail_pred)

# Afficher la matrice de confusion sous forme de heatmap

# Créer la fenêtre Tkinter
root = tk.Tk()
root.title("Matrice de confusion")

# Créer la heatmap à l'aide de seaborn et matplotlib
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion, annot=True, cmap="Blues")
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")

# Ajouter la heatmap au widget Canvas de Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# Lancer la boucle principale de Tkinter
#root.mainloop()

# Obtenez les limites du graphique
x_min, x_max = emails[:, 0].min() - 1, emails[:, 0].max() + 1
y_min, y_max = emails[:, 0].min() - 1, emails[:, 0].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Prédisez la classe pour chaque point dans le plan
Z = svmmodelSpam.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Tracez la frontière de décision SVM et les supports de vecteur
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Paired)

# Tracez les vecteurs de support
support_vectors = svmmodelSpam.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=80, facecolors='none', edgecolors='k')

# Tracez la marge
plt.plot(svmmodelSpam.coef_[0][0]*xx + svmmodelSpam.intercept_, 'k--')
plt.plot(svmmodelSpam.coef_[0][0]*xx + svmmodelSpam.intercept_+1, 'r--')
plt.plot(svmmodelSpam.coef_[0][0]*xx + svmmodelSpam.intercept_-1, 'r--')

# Étiquetez les axes
plt.xlabel('Première caractéristique')
plt.ylabel('Deuxième caractéristique')

# Afficher le graphique
plt.show()

#model maladie
# Créer une instance de la classe MaladiesCardiaques
modelmaladie = MaladiesCardiaques("train.csv", "test.csv")

# Créer le modèle SVM et l'entraîner sur les données d'entraînement
modelmaladie.fit(modelmaladie.X_train, modelmaladie.y_train)

# Faire des prédictions sur les données de test
maladie_pred = modelmaladie.predict(modelmaladie.X_test)

# Évaluer la précision du modèle
accuracy = accuracy_score(modelmaladie.y_test, maladie_pred)
print("Accuracy:", accuracy)


