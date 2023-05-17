import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Créer la fenêtre Tkinter
root = tk.Tk()
root.title("Matrice de confusion")

# Exemple de prédictions et de vraies valeurs
y_true = [0, 1, 0, 1, 1, 0, 0, 1]
y_pred = [0, 1, 1, 1, 1, 0, 1, 0]

# Créer la matrice de confusion
confusion = confusion_matrix(y_true, y_pred)

# Créer la heatmap à l'aide de seaborn et matplotlib
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion, annot=True, cmap="Blues")
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")

# Ajouter la heatmap au widget Canvas de Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# Lancer la boucle principale de Tkinter
root.mainloop()
