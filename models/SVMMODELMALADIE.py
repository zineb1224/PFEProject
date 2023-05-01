# Importation des bibliothèques nécessaires
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib


class SVMMODELMALADIE:
    def __init__(self):
        self.model = svm.SVC(kernel='linear')

    def train(self, data_file_path):
        # Charger les données à partir d'un fichier CSV
        data = pd.read_csv(data_file_path)

        # Séparation des données en ensembles de formation et de test
        X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2,
                                                            random_state=42)

        # Entraînement du modèle SVM sur l'ensemble d'entraînement
        self.model.fit(X_train, y_train)

        # Prédiction sur l'ensemble de test
        y_pred = self.model.predict(X_test)

        # Calcul de la précision du modèle
        accuracy = accuracy_score(y_test, y_pred)

        # Affichage de la précision du modèle
        print('La précision du modèle SVM pour la prédiction de maladies cardiaques est de : {:.2f}%'.format(
            accuracy * 100))

        # Enregistrement du modèle entraîné
        joblib.dump(self.model, 'SVMMODELMALADIE.joblib')

    def predict(self, data_to_predict_file_path):
        # Charger le modèle entraîné
        self.model = joblib.load('SVMMODELMALADIE.joblib')

        # Charger les données à prédire à partir d'un fichier CSV
        data_to_predict = pd.read_csv(data_to_predict_file_path)

        # Prédiction des résultats sur les nouvelles données
        y_pred = self.model.predict(data_to_predict)

        # Retourner les prédictions
        return y_pred
