import tkinter as tk

root = tk.Tk()

# Création de la Listbox
listbox = tk.Listbox(root, width=30, height=10, font=("Arial", 12))
listbox.pack(padx=10, pady=10)

# Configuration des couleurs de fond et de texte pour la Listbox
listbox.configure(background="#F5F5F5", foreground="black")

# Configuration des couleurs de fond et de texte pour les éléments de la Listbox
listbox.configure(selectbackground="#0099CC", selectforeground="white")

# Ajout de quelques éléments à la Listbox
for i in range(10):
    listbox.insert("end", f"Item {i}")

root.mainloop()
