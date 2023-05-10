import tkinter as tk
from tkinter import ttk

# Définition de la palette de couleurs
BG_COLOR = "#1c1c1c"
FG_COLOR = "#d9d9d9"
LABEL_BG_COLOR = "#3a3a3a"
ENTRY_BG_COLOR = "#3a3a3a"

# Configuration du thème sombre "clam" de tkinter
root = tk.Tk()
root.configure(bg=BG_COLOR)
style = ttk.Style()
style.theme_use("clam")

# Création d'un style personnalisé pour le label
style.configure("Dark.TLabel", background=LABEL_BG_COLOR, foreground=FG_COLOR)

# Création d'un style personnalisé pour l'entrée
style.configure("Dark.TEntry", fieldbackground=ENTRY_BG_COLOR, foreground=FG_COLOR)

# Modification des couleurs des widgets
ttk.Label(root, text="Username:", style="Dark.TLabel").pack(pady=20)
entry = ttk.Entry(root, style="Dark.TEntry")
entry.pack()

# Lancement de la boucle principale
root.mainloop()
