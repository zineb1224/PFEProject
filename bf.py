import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

# Chargement du jeu de données Iris
iris = datasets.load_iris()
X = iris.data[:, :2]  # Nous prenons seulement les deux premières caractéristiques pour une visualisation en 2D
y = iris.target

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle SVM
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)


# Obtenir les coefficients du modèle SVM
w = svm_model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 10)
yy = a * xx - (svm_model.intercept_[0]) / w[1]

# Tracer l'hyperplan
plt.plot(xx, yy, 'k-')

# Tracer les marges
margin = 1 / np.sqrt(np.sum(svm_model.coef_ ** 2))
yy_down = yy - np.abs(a) * margin
yy_up = yy + np.abs(a) * margin
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')


# Tracer les points de données de test
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Set1, marker='x')

# Prédire les classes des données de test
y_pred = svm_model.predict(X_test)

# Tracer les vecteurs de support de test
plt.scatter(X_test[y_pred != y_test, 0], X_test[y_pred != y_test, 1],
            s=100, facecolors='none', edgecolors='r')

# Paramètres du graphique
plt.xlabel('Longueur des sépales')
plt.ylabel('Largeur des sépales')
plt.title('SVM avec marges et vecteurs de support')
plt.xlim(np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5)
plt.ylim(np.min(X[:, 1]) - 0.5, np.max(X[:, 1]) + 0.5)
plt.xticks(())
plt.yticks(())

# Afficher le graphique
plt.show()
