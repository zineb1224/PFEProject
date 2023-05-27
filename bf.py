import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Chargement des données d'iris
iris = datasets.load_iris()
X = iris.data[:, :2]  # On utilise seulement les deux premières caractéristiques pour faciliter la visualisation
y = iris.target

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle SVM avec le noyau RBF
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Extraction des vecteurs de support
support_vectors = model.support_vectors_

# Obtention des poids et du biais
dual_coef = model.dual_coef_
intercept = model.intercept_

# Plot des données de test et des vecteurs de support
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=70, facecolors='none', edgecolors='k')

# Calcul des coordonnées de l'hyperplan
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot de l'hyperplan et des marges
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=70, facecolors='none', edgecolors='k')
plt.xlabel('Caractéristique 1')
plt.ylabel('Caractéristique 2')
plt.title('Hyperplan et marges du modèle SVM')
plt.show()
