import matplotlib.pyplot as plt

import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.datasets import load_iris

# Charger le dataset Iris
iris_data = load_iris()
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)

# Calculer les statistiques descriptives
stats = df.describe().loc[['min', 'max', 'mean', 'std']]

# Transposer le DataFrame pour faciliter la visualisation
stats = stats.transpose()

# Créer une fenêtre tkinter
window = tk.Tk()
window.title('Graphique')

# Créer une instance de la figure matplotlib et ajouter le graphique
figure = Figure(figsize=(10, 6))
ax = figure.add_subplot(111)
stats.plot(kind='bar', ax=ax)
ax.set_title('Statistiques descriptives du dataset Iris')
ax.set_xlabel('Variables')
ax.set_ylabel('Valeurs')
ax.legend(loc='best')

# Créer un widget de canevas tkinter pour afficher la figure
canvas = FigureCanvasTkAgg(figure, master=window)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Démarrer la boucle principale tkinter
tk.mainloop()


# Charger le dataset à partir du fichier CSV
dataset = pd.read_csv("datasets/dataset_maladie.csv")

# Calculer la moyenne pour chaque colonne
moyenne = dataset.mean()

# Calculer l'écart-type pour chaque colonne
ecart_type = dataset.std()

# Trouver la valeur minimale pour chaque colonne
minimum = dataset.min()

# Trouver la valeur maximale pour chaque colonne
maximum = dataset.max()

# Créer un DataFrame avec les statistiques
stats_df = pd.DataFrame({'Moyenne': moyenne, 'Écart-type': ecart_type, 'Minimum': minimum, 'Maximum': maximum})

# Créer une fenêtre tkinter
window = tk.Tk()
window.title('Graphique')

# Créer une instance de la figure matplotlib et ajouter le graphique
figure = Figure(figsize=(10, 6))
ax = figure.add_subplot(111)
stats_df.plot(kind='bar', ax=ax)
ax.set_title('Statistiques descriptives du dataset maladie')
ax.set_xlabel('Variables')
ax.set_ylabel('Valeurs')
ax.legend(loc='best')

# Créer un widget de canevas tkinter pour afficher la figure
canvas = FigureCanvasTkAgg(figure, master=window)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Démarrer la boucle principale tkinter
tk.mainloop()


# Charger le dataset à partir d'un fichier CSV
dataset = pd.read_csv("datasets/penguins.csv")

# Sélectionner les colonnes numériques pour le calcul des statistiques
numeric_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Calculer la moyenne pour chaque colonne
moyenne = dataset[numeric_columns].mean()

# Calculer l'écart-type pour chaque colonne
ecart_type = dataset[numeric_columns].std()

# Trouver la valeur minimale pour chaque colonne
minimum = dataset[numeric_columns].min()

# Trouver la valeur maximale pour chaque colonne
maximum = dataset[numeric_columns].max()

# Afficher les résultats
print("Moyenne :")
print(moyenne)
print("\nÉcart-type :")
print(ecart_type)
print("\nMinimum :")
print(minimum)
print("\nMaximum :")
print(maximum)

# Créer un DataFrame avec les statistiques
stats_df = pd.DataFrame({'Moyenne': moyenne, 'Écart-type': ecart_type, 'Minimum': minimum, 'Maximum': maximum})

# Afficher les statistiques dans un seul graphique
stats_df.plot(kind='bar', title='Statistiques descriptives du dataset penguins')



# Charger le dataset à partir d'un fichier CSV
dataset = pd.read_csv("datasets/diabetes.csv")

# Calculer la moyenne pour chaque colonne
moyenne = dataset.mean()

# Calculer l'écart-type pour chaque colonne
ecart_type = dataset.std()

# Trouver la valeur minimale pour chaque colonne
minimum = dataset.min()

# Trouver la valeur maximale pour chaque colonne
maximum = dataset.max()

# Afficher les résultats
print("Moyenne :")
print(moyenne)
print("\nÉcart-type :")
print(ecart_type)
print("\nMinimum :")
print(minimum)
print("\nMaximum :")
print(maximum)
# Créer un DataFrame avec les statistiques
stats_df = pd.DataFrame({'Moyenne': moyenne, 'Écart-type': ecart_type, 'Minimum': minimum, 'Maximum': maximum})

# Afficher les statistiques dans un seul graphique
stats_df.plot(kind='bar', title='Statistiques descriptives du dataset diabetes')



# Afficher le graphique
plt.show()