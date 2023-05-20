import tkinter as tk
from tkinter import ttk
from sklearn.datasets import load_iris
from random import sample
import pandas as pd


def afficher_description():
    # Obtenir l'option sélectionnée dans le combobox
    option = combobox_onglet1.get()

    # Vider le Treeview
    donnees_treeview.delete(*donnees_treeview.get_children())

    # Afficher les données correspondantes dans l'onglet 2
    if option == "Iris":
        # Obtenir 10 lignes aléatoires
        iris_data_subset = sample(iris_data, 10)
        for data in iris_data_subset:
            donnees_treeview.insert("", "end", values=data)

            # Définir le titre de l'onglet 2
        notebook.tab(onglet2, text="Iris Dataset")
        column_headings = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        description_text="Description de l'ensemble de données Iris"
    elif option == "Maladie":
        # Charger les données depuis le fichier CSV
        maladie_data = pd.read_csv('datasets/dataset_maladie.csv')

        # Obtenir 10 lignes aléatoires de l'ensemble de données Maladie
        maladie_data_subset = maladie_data.sample(10)
        for _, row in maladie_data_subset.iterrows():
            donnees_treeview.insert("", "end", values=row.tolist())

        # Définir les en-têtes des colonnes pour l'ensemble de données Maladie
        column_headings = maladie_data.columns.tolist()

        # Définir le titre de l'onglet 2 pour l'ensemble de données Maladie
        notebook.tab(onglet2, text="Maladie Dataset")
        description_text="Description de l'ensemble de données Maladie"

    elif option == "Penguins":
        # Charger les données depuis le fichier CSV
        maladie_data = pd.read_csv('datasets/penguins.csv')

        # Obtenir 10 lignes aléatoires de l'ensemble de données penguins
        maladie_data_subset = maladie_data.sample(10)
        for _, row in maladie_data_subset.iterrows():
            donnees_treeview.insert("", "end", values=row.tolist())
        # Définir les en-têtes des colonnes pour l'ensemble de données Maladie
            column_headings = maladie_data.columns.tolist()
        # Définir le titre de l'onglet 2 pour l'ensemble de données diabetes
        notebook.tab(onglet2, text="Diabetes Dataset")
        description_text="Description de l'ensemble de données Penguins"

    elif option == "Diabetes":
        # Charger les données depuis le fichier CSV
        maladie_data = pd.read_csv('datasets/diabetes.csv')

        # Obtenir 10 lignes aléatoires de l'ensemble de données diabetes
        maladie_data_subset = maladie_data.sample(10)
        for _, row in maladie_data_subset.iterrows():
            donnees_treeview.insert("", "end", values=row.tolist())
        # Définir les en-têtes des colonnes pour l'ensemble de données Maladie
            column_headings = maladie_data.columns.tolist()
        # Définir le titre de l'onglet 2 pour l'ensemble de données Diabetes
        notebook.tab(onglet2, text="Diabetes Dataset")
        description_text="Description de l'ensemble de données Diabetes"

        # Sélectionner l'onglet 2
    notebook.select(onglet2)
    # Modifier le titre de l'onglet 2 avec le texte de description correspondant
    titre_onglet2.config(text=description_text)
# Définir les en-têtes des colonnes dans le Treeview
    donnees_treeview["columns"] = column_headings
    for i, heading in enumerate(column_headings):
        donnees_treeview.heading(i, text=heading)
# Chargement du jeu de données iris
iris = load_iris()
iris_data = iris.data.tolist()  # Convertir les données en liste

# Création de la fenêtre principale
fenetre_principale = tk.Tk()
fenetre_principale.title("Application")

# Création du Notebook (les onglets)
notebook = ttk.Notebook(fenetre_principale)

# Création du premier onglet
onglet1 = ttk.Frame(notebook)
notebook.add(onglet1, text='Onglet 1')

# Création du combobox dans l'onglet 1
combobox_onglet1 = ttk.Combobox(onglet1, values=["Iris", "Maladie", "Penguins", "Diabetes"], state="readonly")
combobox_onglet1.current(0)
combobox_onglet1.pack(padx=20, pady=20)

# Création du bouton "Voir plus de description" dans l'onglet 1
bouton_onglet1 = ttk.Button(onglet1, text="Voir plus de description", command=afficher_description)
bouton_onglet1.pack(padx=20, pady=20)

# Création du deuxième onglet
onglet2 = ttk.Frame(notebook)
notebook.add(onglet2, text='Onglet 2')

# Définir le titre de l'onglet 2 avec un Label
titre_onglet2 = ttk.Label(onglet2, foreground="#FFFFFF",font=("Arial", 16, "bold"),background="#74B0FF" , padding=25)
titre_onglet2.pack(pady=20)

# Création du Treeview pour afficher les données dans l'onglet 2
donnees_treeview = ttk.Treeview(onglet2, show="headings", height=10)

# Ajouter le Treeview dans l'onglet 2
donnees_treeview.pack(padx=20, pady=20)

# L'onglet 2 est initialisé masqué
notebook.hide(onglet2)

# Affichage du Notebook
notebook.pack(expand=True, fill="both")

# Lancement de la boucle principale
fenetre_principale.mainloop()
