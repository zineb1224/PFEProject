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




# Calcul des résultats
precision = 0.85
recall = 0.92
f1_score = 0.88

# Tracé du graphe
plt.figure(figsize=(8, 6))
plt.bar(['Précision', 'Rappel', 'F1-score'], [precision, recall, f1_score], color=['green', 'blue', 'orange'])
plt.ylim(0, 1)
plt.title('Résultats de la prédiction des maladies cardiaques')
plt.xlabel('Métrique')
plt.ylabel('Score')
plt.show()