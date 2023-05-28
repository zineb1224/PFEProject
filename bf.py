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

# Extraction des indices des vecteurs de support
support_indices = model.support_
dual_coef = np.abs(model.dual_coef_)

# Limiter le nombre de vecteurs de support à afficher
num_support_vectors = 3  # Nombre souhaité de vecteurs de support à afficher par classe
selected_support_vectors = []

for class_label in np.unique(y_train):
    class_dual_coef = dual_coef[class_label - 1]
    class_support_indices = support_indices[np.where(y_train[support_indices] == class_label)]
    num_vectors = min(num_support_vectors, len(class_support_indices))
    random_indices = np.random.choice(class_support_indices, num_vectors, replace=False)
    selected_support_vectors.extend(X_train[random_indices])

selected_support_vectors = np.array(selected_support_vectors)

# Plot des données de test et des vecteurs de support sélectionnés
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
plt.scatter(selected_support_vectors[:, 0], selected_support_vectors[:, 1], s=100, facecolors='none', edgecolors='k')

# Calcul des coordonnées de l'hyperplan
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot de l'hyperplan et des marges
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
plt.scatter(selected_support_vectors[:, 0], selected_support_vectors[:, 1], s=100, facecolors='none', edgecolors='k')
plt.xlabel('Caractéristique 1')
plt.ylabel('Caractéristique 2')
plt.title('Hyperplan et marges du modèle SVM')
plt.show()
