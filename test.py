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