import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from main import modelmaladie
from models.SVMMODELMALADIE import MaladiesCardiaques

class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.create_widgets()

    def create_widgets(self):
        # Création d'une figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Chargement des données et entraînement du modèle
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        X_test = np.array([[1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 9], [8, 10], [9, 11], [10, 12]])
        y_test = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

        self.ax.plot(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 'ro', label='Train: classe 0')
        self.ax.plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 'bo', label='Train: classe 1')
        self.ax.plot(X_test[y_test == 0, 0], X_test[y_test == 0, 1], 'rx', label='Test: classe 0')
        self.ax.plot(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 'bx', label='Test: classe 1')

        # Ajout des légendes
        self.ax.legend()

        # Ajout du graphique à la fenêtre
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

root = tk.Tk()
app = App(master=root)
app.mainloop()


# Train the model and get accuracy scores
accuracy_train, accuracy_test = modelmaladie.train()

# Plot the accuracy scores
plt.plot(accuracy_train, label='Training Accuracy')
plt.plot(accuracy_test, label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Scores')
plt.legend()
plt.show()

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


