import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from models.SVMModelSentiment import SVMModelSentiment, import_data
from sklearn.metrics import accuracy_score, precision_score
from sklearn import metrics


#model maladie cardiaque
svmmodelSentiment = SVMModelSentiment()
# Chargement des données
sentiment_data = import_data("../datasets/EcoPreprocessed.csv")

# Séparation des données et de target
X_sentiment = np.array(sentiment_data["review"])
y_sentiment = np.where(sentiment_data['division'] == 'positive', 1, 0)
# Vectorisation des textes en utilisant TF-IDF
vectorizer = TfidfVectorizer()
vecteurs_texte = vectorizer.fit_transform(X_sentiment)

# Diviser les données en ensembles d'entraînement et de test
featuresSentiment_train, featuresSentiment_test, targetSentiment_train, targetSentiment_test = train_test_split(vecteurs_texte, y_sentiment, test_size = 0.2, random_state = 0)

# Prédiction sur l'ensemble de test
svmmodelSentiment.fit(featuresSentiment_train, targetSentiment_train)

sentiment_pred = svmmodelSentiment.predict(featuresSentiment_test)
# Évaluer les performances du modèle
accuracy = accuracy_score(targetSentiment_test, sentiment_pred)
precision = precision_score(targetSentiment_test, sentiment_pred)
metrics.ConfusionMatrixDisplay.from_predictions(targetSentiment_test, svmmodelSentiment.predict(featuresSentiment_test))
print("Accuracy:", accuracy)
print("Precision:", precision)

# Création du graphe avec la marge et les vecteurs de support
# Création du graphe avec la marge et les vecteurs de support

plt.figure(figsize=(6, 6))
# afficher les données
# Plotting the data points
plt.plot(featuresSentiment_test[targetSentiment_test == 0], featuresSentiment_test[targetSentiment_test == 0], "yo")
plt.plot(featuresSentiment_test[targetSentiment_test == 1], featuresSentiment_test[targetSentiment_test == 1], "bo")

# Limites du cadre
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
support_vectors_ = svmmodelSentiment.support_vectors_()
# Marquer les vecteurs de support d'une croix
ax.scatter(support_vectors_[:, 0], support_vectors_[:, 1], linewidth=1, facecolors='#FFAAAA', s=180)
# Grille de points sur lesquels appliquer le modèle
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Prédire pour les points de la grille
Z = svmmodelSentiment.decision_function(xy).reshape(XX.shape)
# Afficher la frontière de décision et la marge
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Tracé du graphe avec les vecteurs de support et les marges
plt.scatter(support_vectors_[:, 0], support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.xlabel('Âge')
plt.ylabel('Taux de cholestérol')
plt.show()


