import tkinter as tk
from tkinter import ttk

root = tk.Tk()

# Création d'un style personnalisé pour Entry
style = ttk.Style()

# Configuration de la bordure et du relief pour l'Entry
style.configure("Custom.TEntry", borderwidth=0, relief="solid")

# Configuration de la bordure arrondie pour l'Entry
style.map("Custom.TEntry",
    foreground=[("focus", "black"), ("!focus", "gray")],
    background=[("focus", "white"), ("!focus", "white")],
    bordercolor=[("focus", "gray"), ("!focus", "gray")],
    borderwidth=[("focus", 2), ("!focus", 2)],
    relief=[("focus", "solid"), ("!focus", "solid")],
    focuscolor=[("focus", "white"), ("!focus", "white")],
    highlightthickness=[("focus", 2), ("!focus", 0)],
    padx=[("focus", 6), ("!focus", 6)],
    pady=[("focus", 6), ("!focus", 6)],
)

# Création d'un Entry personnalisé avec une bordure arrondie
entry = ttk.Entry(root, style="Custom.TEntry")
entry.pack(ipady=10)

root.mainloop()
