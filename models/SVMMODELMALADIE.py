# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MaladiesCardiaques:
    def __init__(self, train_file, test_file):
        # Charger les données d'entraînement et de test
        self.X_train, self.y_train = self.load_data(train_file)
        self.X_test, self.y_test = self.load_data(test_file)

        # Prétraiter les données en normalisant les fonctionnalités
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def load_data(self, file):
        # Charger les données à partir du fichier CSV
        data = pd.read_csv('datasets/maladie.csv')

        # Séparer les fonctionnalités et les étiquettes
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        return X, y
