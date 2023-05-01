# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Charger les données
data = pd.read_csv("A:\PFEProject\datasets\maladie.csv")


# Séparer les fonctionnalités et les étiquettes
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prétraiter les données en normalisant les fonctionnalités
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer le modèle SVM et l'entraîner sur les données d'entraînement
model = SVC(kernel='linear', C=1, random_state=42)
model.fit(X_train, y_train)

# Faire des prédictions sur les données de test
y_pred = model.predict(X_test)

# Évaluer la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(data.head())
