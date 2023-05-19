import tkinter as tk
from tkinter import ttk

def afficher_description():
    # Sélectionner l'onglet 2
    notebook.select(onglet2)

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

# L'onglet 2 est initialisé masqué
notebook.hide(onglet2)

# Affichage du Notebook
notebook.pack(expand=True, fill="both")

# Lancement de la boucle principale
fenetre_principale.mainloop()