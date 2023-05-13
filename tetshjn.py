import tkinter as tk
from tkinter import ttk

# Définition de la palette de couleurs
#BG_COLOR = "#1c1c1c"
#FG_COLOR = "#d9d9d9"
#LABEL_BG_COLOR = "#3a3a3a"
#ENTRY_BG_COLOR = "#3a3a3a"
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

# Configuration du thème sombre "clam" de tkinter
#root = tk.Tk()
#root.configure(bg=BG_COLOR)
#style = ttk.Style()
#style.theme_use("clam")

# Création d'un style personnalisé pour le label
#style.configure("Dark.TLabel", background=LABEL_BG_COLOR, foreground=FG_COLOR)

# Création d'un style personnalisé pour l'entrée
#style.configure("Dark.TEntry", fieldbackground=ENTRY_BG_COLOR, foreground=FG_COLOR)

# Modification des couleurs des widgets
#ttk.Label(root, text="Username:", style="Dark.TLabel").pack(pady=20)
#entry = ttk.Entry(root, style="Dark.TEntry")
#entry.pack()

# Lancement de la boucle principale
#root.mainloop()


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# Chargement des données d'entraînement
spam_train = datasets.load_svmlight_file("chemin/vers/le/fichier/spam_train.txt")
X_train = spam_train[0]
y_train = spam_train[1]

# Entraînement du modèle SVM
model = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')
model.fit(X_train, y_train)

# Prédiction des étiquettes de classe pour les données d'entraînement
y_train_pred = model.predict(X_train)

# Tracé du graphe avec les données d'entraînement
spam_class = np.array(["non-spam", "spam"])
colors = spam_class[y_train_pred]
plt.scatter(X_train[:, 0], X_train[:, 1], c=colors)
plt.xlabel("Première caractéristique")
plt.ylabel("Deuxième caractéristique")
plt.title("Classification de spam avec SVM - données d'entraînement")
plt.show()

# Chargement des données de test
spam_test = datasets.load_svmlight_file("chemin/vers/le/fichier/spam_test.txt")
X_test = spam_test[0]
y_test = spam_test[1]

# Prédiction des étiquettes de classe pour les données de test
y_test_pred = model.predict(X_test)

# Tracé du graphe avec les données de test
colors = spam_class[y_test_pred]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colors)
plt.xlabel("Première caractéristique")
plt.ylabel("Deuxième caractéristique")
plt.title("Classification de spam avec SVM - données de test")
plt.show()

# Chargement des données d'entraînement
spam = datasets.load_svmlight_file("chemin/vers/le/fichier/spam_train.txt")
X = spam[0]
y = spam[1]

# Entraînement du modèle SVM
model = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')
model.fit(X, y)

# Prédiction des étiquettes de classe pour les données d'entraînement
y_pred = model.predict(X)

# Tracé du graphe avec les données d'entraînement
spam_class = np.array(["non-spam", "spam"])
colors = spam_class[y_pred]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.xlabel("Première caractéristique")
plt.ylabel("Deuxième caractéristique")
plt.title("Classification de spam avec SVM")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# Chargement des données de test
spam = datasets.load_svmlight_file("chemin/vers/le/fichier/spam_test.txt")
X = spam[0]
y = spam[1]

# Chargement du modèle SVM entraîné
model = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')
model.fit(X, y)

# Prédiction des étiquettes de classe pour les données de test
y_pred = model.predict(X)

# Tracé du graphe avec les données de test
spam_class = np.array(["non-spam", "spam"])
colors = spam_class[y_pred]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.xlabel("Première caractéristique")
plt.ylabel("Deuxième caractéristique")
