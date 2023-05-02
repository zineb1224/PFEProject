# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class MaladiesCardiaques:

    def __init__(self, train_file, test_file):
        # Charger les données d'entraînement et de test
        self.model = svm.SVC()
        self.X_train, self.y_train = self.load_data(train_file)
        self.X_test, self.y_test = self.load_data(test_file)

        # Prétraiter les données en normalisant les fonctionnalités
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.train_accuracy = []
        self.test_accuracy = []

    @staticmethod
    def load_data(file):
        # Charger les données à partir du fichier CSV
        data = pd.read_csv('datasets/maladie.csv')

        # Séparer les fonctionnalités et les étiquettes
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        return x, y

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def train(self):
        for i in range(10):
            self.model.fit(self.X_train, self.y_train)
            train_acc = self.model.score(self.X_train, self.y_train)
            test_acc = self.model.score(self.X_test, self.y_test)
            self.train_accuracy.append(train_acc)
            self.test_accuracy.append(test_acc)

    def plot_accuracy(self):
        plt.plot(self.train_accuracy, label='Training accuracy')
        plt.plot(self.test_accuracy, label='Test accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()