import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
# Chargement des données
penguins = pd.read_csv("datasets/penguins.csv")
data = penguins[["bill_length_mm",
 "bill_depth_mm",
 "flipper_length_mm",
 "body_mass_g"]].to_numpy()
labels = pd.Categorical(penguins["species"]).astype('category').codes

# Nous allons ici nous se limiter à deux espèces :
# Adelie (0) et Gentoo (2)
# et deux variables : bill_length_mm et bill_depth_mm.
y = labels
Adelie_or_Gentoo = (y == 0) | (y == 2)
X = data[:,:2][Adelie_or_Gentoo]
y = y[Adelie_or_Gentoo]
print(X.shape)
print(y.shape)
# afficher les données (pingouins Adelie et Gentoo )
plt.figure(figsize=(6, 4))
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Adelie")
plt.plot(X[:, 0][y==2], X[:, 1][y==2], "bo", label="Gentoo")
# Légende
plt.legend()
plt.xlabel("Bill length (mm)")
plt.ylabel("Bill depth (mm)")
plt.title("Données")
plt.tight_layout()
plt.show()

from sklearn.impute import SimpleImputer

# Instantiate the imputer
imputer = SimpleImputer(strategy='mean') # or 'median', 'most_frequent' etc.

# Replace missing values in X with the mean of each column
X = imputer.fit_transform(X)

# initialisation
for C1 in 100, 0.1:
 model_svc1 = svm.SVC(kernel='linear', C=C1)
 # entrainement
 model_svc1.fit(X, y)
 #Evaluation
 score1 = model_svc1.score(X, y)
 print("Score de la SVM linéaire (C=%.2f) (mean accuracy): %.2f"%( C1, score1) )
 metrics.ConfusionMatrixDisplay.from_predictions(y, model_svc1.predict(X))
 # afficher les données (pingouins Adelie et Gentoo)
 plt.figure(figsize=(6, 4))
 plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Adelie")
 plt.plot(X[:, 0][y==2], X[:, 1][y==2], "bo", label="Gentoo")
 # Limites du cadre
 ax = plt.gca()
 xlim = ax.get_xlim()
 ylim = ax.get_ylim()
 # Marquer les points vecteurs de support
 ax.scatter(model_svc1.support_vectors_[:, 0],
 model_svc1.support_vectors_[:, 1],
 linewidth=1,
 facecolors='#FFAAAA', s=180)
 # Grille de points sur lesquels appliquer le modèle
 xx = np.linspace(xlim[0], xlim[1], 30)
 yy = np.linspace(ylim[0], ylim[1], 30)
 YY, XX = np.meshgrid(yy, xx)
 xy = np.vstack([XX.ravel(), YY.ravel()]).T
 # Prédire pour les points de la grille
 Z = model_svc1.decision_function(xy).reshape(XX.shape)
 # Afficher la frontière de décision et la marge
 ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
 alpha=0.5, linestyles=['--', '-', '--'])
 # Légende
 plt.legend()
 plt.xlabel("Bill length (mm)")
 plt.ylabel("Bill depth (mm)")
 plt.title("SVM linéaire (C=%.2f)"%C1)
 plt.tight_layout()
 plt.show()