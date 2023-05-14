# Import des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Chargement des données
data = pd.read_csv("datasets/dataset_maladie.csv")
X = data[["age", "thalach"]].to_numpy()
y = data["target"].values

print(X.shape)
print(y.shape)

# Séparation des données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle SVM
model = SVC(kernel='rbf', C=1, gamma=0.005, random_state=42)
model.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred = model.predict(X_test)

# Calcul de la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Création du graphe avec la marge et les vecteurs de support
# Récupération des vecteurs de support
support_vectors = model.support_vectors_

plt.figure(figsize=(6, 6))
# afficher les données
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo")
# afficher les données
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bo")

# Limites du cadre
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# Marquer les vecteurs de support d'une croix
ax.scatter(model.support_vectors_[:, 0],
 model.support_vectors_[:, 1],
 linewidth=1,
 facecolors='#FFAAAA', s=180)
# Grille de points sur lesquels appliquer le modèle
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Prédire pour les points de la grille
Z = model.decision_function(xy).reshape(XX.shape)
# Afficher la frontière de décision et la marge
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
 alpha=0.5, linestyles=['--', '-', '--'])

# Tracé du graphe avec les vecteurs de support et les marges
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='k')
plt.xlabel('Âge')
plt.ylabel('Taux de cholestérol')
plt.show()
