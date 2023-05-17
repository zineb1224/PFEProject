import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from models.SVMModelMaladieCardiaque import SVMModelMaladieCardiaque , import_data
from sklearn.metrics import accuracy_score, precision_score
from sklearn import metrics


#model maladie cardiaque
svmmodelMaladieCardiaque = SVMModelMaladieCardiaque('rbf')
# Chargement des données
maladie_data = import_data("../datasets/dataset_maladie.csv")

# Séparation des données et de target
X_maladie = maladie_data[["age", "thalach"]].to_numpy()
y_maladie = maladie_data["target"].values

# Diviser les données en ensembles d'entraînement et de test
featuresMaladie_train, featuresMaladie_test, targetMaladie_train, targetMaladie_test = train_test_split(X_maladie, y_maladie, test_size = 0.2, random_state = 0)

# Prédiction sur l'ensemble de test
svmmodelMaladieCardiaque.fit(featuresMaladie_train, targetMaladie_train)

maladie_pred = svmmodelMaladieCardiaque.predict(featuresMaladie_test)
# Évaluer les performances du modèle
accuracy = accuracy_score(targetMaladie_test, maladie_pred)
precision = precision_score(targetMaladie_test, maladie_pred)
metrics.ConfusionMatrixDisplay.from_predictions(targetMaladie_test, svmmodelMaladieCardiaque.predict(featuresMaladie_test))
print("Accuracy:", accuracy)
print("Precision:", precision)

# Création du graphe avec la marge et les vecteurs de support
# Récupération des vecteurs de support
support_vectors = svmmodelMaladieCardiaque.support_vectors_

plt.figure(figsize=(6, 6))
# afficher les données
plt.plot(featuresMaladie_test[:, 0][targetMaladie_test==0], featuresMaladie_test[:, 1][targetMaladie_test==0], "yo")
# afficher les données
plt.plot(featuresMaladie_test[:, 0][targetMaladie_test==1], featuresMaladie_test[:, 1][targetMaladie_test==1], "bo")

# Limites du cadre
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
support_vectors_ = svmmodelMaladieCardiaque.support_vectors_()
# Marquer les vecteurs de support d'une croix
ax.scatter(support_vectors_[:, 0], support_vectors_[:, 1], linewidth=1, facecolors='#FFAAAA', s=180)
# Grille de points sur lesquels appliquer le modèle
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Prédire pour les points de la grille
Z = svmmodelMaladieCardiaque.decision_function(xy).reshape(XX.shape)
# Afficher la frontière de décision et la marge
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
 alpha=0.5, linestyles=['--', '-', '--'])

# Tracé du graphe avec les vecteurs de support et les marges
plt.scatter(support_vectors_[:, 0], support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.xlabel('Âge')
plt.ylabel('Taux de cholestérol')
plt.show()

