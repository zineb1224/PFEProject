import tkinter as tk
from tkinter import ttk
from sklearn.datasets import load_iris

def afficher_description():
    fenetre_principale.tk_setPalette(background='#333333', foreground='black')

    # Sélectionner l'onglet 2
    notebook.select(onglet2)

    # Afficher les données dans l'onglet 2
    for i in range(10):
        donnees_treeview.insert("", "end", values=iris_data[i])

# Chargement du jeu de données iris
iris = load_iris()
iris_data = iris.data[:10]  # Obtenir les 10 premières lignes des données

# Création de la fenêtre principale
fenetre_principale = tk.Tk()
fenetre_principale.title("Application")

# Création du style pour le bouton
style = ttk.Style()
style.configure("Custom.TButton", foreground="white", background="#4CAF50", padding=10)

# Création du Notebook (les onglets)
notebook = ttk.Notebook(fenetre_principale)

# Création du premier onglet
onglet1 = ttk.Frame(notebook)
notebook.add(onglet1, text='Onglet 1')

# Création du bouton dans l'onglet 1
bouton_onglet1 = ttk.Button(onglet1, text="Voir plus de description", command=afficher_description, style="Custom.TButton")
bouton_onglet1.pack(padx=20, pady=20)

# Création du deuxième onglet
onglet2 = ttk.Frame(notebook)
notebook.add(onglet2, text='Onglet 2')
# Création du cadre pour le titre
titre_cadre = ttk.Frame(onglet2, style="Title.TFrame")
titre_cadre.pack(padx=20, pady=(20, 10))

# Label pour le titre
titre_label = ttk.Label(titre_cadre, text="Titre Englobé", font=("Helvetica", 16, "bold"), foreground="#FFFFFF", background="#3F51B5")
titre_label.pack(padx=10, pady=10)

# Création de la règle de style pour le cadre du titre
fenetre_principale.style = ttk.Style()
fenetre_principale.style.configure(
    "Title.TFrame",
    background="#3F51B5",
    relief="groove",
    padding=25
)


# Partie fixe pour le texte
texte_frame = ttk.LabelFrame(onglet2)
texte_frame.pack(padx=20, pady=10)

# Label pour le texte
texte_label = ttk.Label(texte_frame, text="voila les 10 lignes datset ")
texte_label.pack(padx=10, pady=10)
# Création du Treeview pour afficher les données dans l'onglet 2
donnees_treeview = ttk.Treeview(onglet2, columns=list(range(4)), show="headings", height=10)

# Définir les en-têtes des colonnes
donnees_treeview.heading(0, text="Feature 1")
donnees_treeview.heading(1, text="Feature 2")
donnees_treeview.heading(2, text="Feature 3")
donnees_treeview.heading(3, text="Feature 4")

# Afficher les données dans l'onglet 2
for i in range(10):
    donnees_treeview.insert("", "end", values=iris_data[i])

# Ajouter le Treeview dans l'onglet 2
donnees_treeview.pack(padx=20, pady=20)

# L'onglet 2 est initialisé masqué
notebook.hide(onglet2)

# Affichage du Notebook
notebook.pack(expand=True, fill="both")

# Lancement de la boucle principale
fenetre_principale.mainloop()
