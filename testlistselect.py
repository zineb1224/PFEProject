import tkinter as tk

def update_label(*args):
    selected_value.set(listbox.get(tk.ACTIVE))

root = tk.Tk()

# Créer une variable de contrôle pour stocker la valeur sélectionnée
selected_value = tk.StringVar()

# Créer une liste déroulante
listbox = tk.Listbox(root)
listbox.insert(1, "Option 1")
listbox.insert(2, "Option 2")
listbox.insert(3, "Option 3")
listbox.pack()

# Lier la variable de contrôle à la liste déroulante
selected_value.trace('w', update_label)

# Créer une étiquette de texte pour afficher la valeur sélectionnée
label = tk.Label(root, textvariable=selected_value)
label.pack()

root.mainloop()
