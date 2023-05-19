from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from models.SVMModelIris import SVMModelIris, import_dataIris
# Charger les données Iris

svmModelIris = SVMModelIris()
iris = import_dataIris()
X = iris.data[:, :2]  # Utiliser seulement les deux premières caractéristiques
y = iris.target

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle SVM
svmModelIris.fit(X_train, y_train)

# Créer une interface graphique avec Tkinter
root = tk.Tk()
root.title("Classification d'Iris avec SVM")

# Créer une zone de dessin pour afficher le graphique
figure = Figure(figsize=(6, 4), dpi=100)
subplot = figure.add_subplot(111)

# Tracer les points de données
subplot.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
subplot.set_xlabel('Longueur des sépales')
subplot.set_ylabel('Largeur des sépales')

# Tracer la ligne de séparation du SVM
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Générer une grille de points pour évaluer le modèle SVM
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.c_[XX.ravel(), YY.ravel()][:, :2]
Z = svmModelIris.predict(xy)
Z = Z.reshape(XX.shape)

# Tracer les contours de décision
subplot.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])

# Créer un canevas Tkinter pour afficher le graphique
canvas = FigureCanvasTkAgg(figure, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Exécuter l'interface graphique Tkinter
tk.mainloop()
