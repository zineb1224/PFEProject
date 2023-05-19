import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

from models.SVMModelDiabets import SVMModelDiabets , import_dataDiabets

# Chargement des données
svmmodelDiabets = SVMModelDiabets()
diabets = import_dataDiabets("../datasets/diabetes.csv")
data = diabets[["Glucose",
 "BloodPressure",
 "Insulin",
 "DiabetesPedigreeFunction"]].to_numpy()
labels_diabets = diabets["Outcome"]

# Nous allons ici nous se limiter à deux espèces :
# Adelie (0) et Gentoo (2)
# et deux variables : bill_length_mm et bill_depth_mm.
X = data[:, :2]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels_diabets, test_size=0.2, random_state=0)


# entrainement
svmmodelDiabets.fit(X_train, y_train)

# Evaluation
metrics.ConfusionMatrixDisplay.from_predictions(y_test, svmmodelDiabets.predict(X_test))



# afficher les données (pingouins Adelie et Gentoo)
plt.figure(figsize=(6, 4))
plt.plot(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], "yo", label="Adelie")
plt.plot(X_train[:, 0][y_train==2], X_train[:, 1][y_train==2], "bo", label="Gentoo")
# Limites du cadre
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# Récupération des vecteurs de support
support_vectors = svmmodelDiabets.support_vectors_()

# Marquer les points vecteurs de support
ax.scatter(support_vectors[:, 0],support_vectors[:, 1],linewidth=1,facecolors='#FFAAAA', s=180)
# Grille de points sur lesquels appliquer le modèle
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Prédire pour les points de la grille
Z = svmmodelDiabets.decision_function(xy).reshape(XX.shape)
# Afficher la frontière de décision et la marge
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
alpha=0.5, linestyles=['--', '-', '--'])
# Légende
plt.legend()
plt.xlabel("Bill length (mm)")
plt.ylabel("Bill depth (mm)")
plt.tight_layout()
plt.show()

# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, svmmodelDiabets.predict(X_test))
precision = precision_score(y_test, svmmodelDiabets.predict(X_test), pos_label=0)
# Calcul du score F1
f1 = f1_score(y_test, svmmodelDiabets.predict(X_test), pos_label=0)
# Affichage du score F1
print("Score F1 : ", f1)