#import tkinter as tk
#from tkinter import scrolledtext

#class InterfaceModule(tk.Frame):
    #def __init__(self, master=None):
        #super().__init__(master)
       # self.master = master
      #  self.pack()
     #   self.create_widgets()

    #def create_widgets(self):
        # Bouton pour charger un module
        #self.load_module_button = tk.Button(self)
        #self.load_module_button["text"] = "Charger un module"
        #self.load_module_button["command"] = self.load_module
       # self.load_module_button.pack(side="top")

        # Bouton pour exécuter le module chargé
        #self.run_module_button = tk.Button(self)
        #self.run_module_button["text"] = "Exécuter le module"
       # self.run_module_button["command"] = self.run_module
      #  self.run_module_button.pack(side="top")

        # Zone de texte pour afficher le résultat
     #   self.result_text = scrolledtext.ScrolledText(self, width=40, height=10)
    #    self.result_text.pack()

   # def load_module(self):
  #      # Mettre ici le code pour charger un module
 #       self.result_text.insert(tk.END, "Module chargé\n")
#
   # def run_module(self):
  #      # Mettre ici le code pour exécuter le module chargé
 #       self.result_text.insert(tk.END, "Module exécuté\n")

#root = tk.Tk()
#app = InterfaceModule(master=root)
#app.mainloop()
from sklearn.datasets import make_classification
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd


# Création de la fenêtre principale
root = tk.Tk()
root.title("Mon application")


# Définition des fonctions

def train_model():

    # Récupération des données d'entrée
    module = module_listbox.get()
    test_size = float(test_size_entry.get())
    c_param = float(c_param_entry.get())
    h_param = float(h_param_entry.get())
    kernel_param = kernel_param_combobox.get()

    # Chargement des données
    data = pd.read_csv(module + ".csv")
    X = data.drop(columns=["target"])
    y = data["target"]

    # Séparation des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Entraînement du modèle
    clf = SVC(C=c_param, gamma=h_param, kernel=kernel_param)
    clf.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = clf.predict(X_test)

    # Entrée pour le paramètre test_size
    test_size_entry = ttk.Entry(root)
    test_size_entry.insert(tk.END, "0.2")
    test_size_entry.grid(row=2, column=1)

    # Affichage des résultats
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    train_text.insert(tk.END, f"Score sur l'ensemble d'entraînement : {train_score:.2f}\n")
    train_text.see(tk.END)
    test_text.insert(tk.END, f"Score sur l'ensemble de test : {test_score:.2f}\n")
    test_text.see(tk.END)

    # Affichage de la matrice de confusion
    disp = plot_confusion_matrix(clf, X_test, y_test, display_labels=["Classe 0", "Classe 1"], cmap=plt.cm.Blues)
    disp.ax_.set_title("Matrice de confusion")
    plt.show()


# Création des widgets

# Label pour la liste de modules
module_label = ttk.Label(root, text="Choix du module :")
module_label.grid(row=0, column=0, sticky="w")

# Liste des modules
modules = ["module1", "module2", "module3"]  # à remplacer par la liste de vos modules
module_listbox = tk.Listbox(root, height=len(modules))
for module in modules:
    module_listbox.insert(tk.END, module)
module_listbox.grid(row=1, column=0)

# Label pour la description du module
description_label = ttk.Label(root, text="Description du module :")
description_label.grid(row=0, column=1, sticky="w")

# Texte de description du module
description_text = scrolledtext.ScrolledText(root, width=30, height=5, wrap=tk.WORD)
description_text.insert(tk.END, "Description du module ici")
description_text.grid(row=1, column=1)

# Label pour le paramètre test_size
test_size_label = ttk.Label(root, text="Taille de l'ensemble de test :")
test_size_label.grid(row=2, column=0, sticky="w")

# Entrée pour le paramètre test_size


# Générer des données de classification aléatoires
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir une fonction pour entraîner le modèle SVM
def train_svm(c, h, kernel):
    # Créer une instance de modèle SVM avec les paramètres donnés
    model = SVC(C=c, kernel=kernel, gamma=h)

    # Entraîner le modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Calculer l'exactitude du modèle sur les données de test
    accuracy = model.score(X_test, y_test)

    # Retourner le modèle entraîné et son exactitude sur les données de test
    return model, accuracy

# Définir une fonction pour afficher les résultats du modèle
def show_results(model, accuracy):
    # Calculer la matrice de confusion pour les données de test
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Créer une nouvelle fenêtre pour afficher les résultats
    results_window = tk.Toplevel(root)

    # Afficher l'exactitude du modèle sur les données de test
    accuracy_label = tk.Label(results_window, text="Exactitude sur les données de test : {:.2f}%".format(accuracy * 100))
    accuracy_label.pack()

    # Afficher la matrice de confusion pour les données de test
    cm_fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title("Matrice de confusion")
    plt.colorbar()
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='white')
    cm_canvas = FigureCanvasTkAgg(cm_fig, master=results_window)
    cm_canvas.draw()
    cm_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Définir une fonction pour gérer le clic sur le bouton "Entraîner"
# Entrée pour le paramètre C
c_param_label = ttk.Label(root, text="Valeur de C :")
c_param_label.grid(row=3, column=0, sticky="w")
c_param_entry = ttk.Entry(root)
c_param_entry.insert(tk.END, "1.0")
c_param_entry.grid(row=3, column=1)

# Entrée pour le paramètre gamma
h_param_label = ttk.Label(root, text="Valeur de gamma :")
h_param_label.grid(row=4, column=0, sticky="w")
h_param_entry = ttk.Entry(root)
h_param_entry.insert(tk.END, "1.0")
h_param_entry.grid(row=4, column=1)

# Combobox pour le choix du noyau
kernel_param_label = ttk.Label(root, text="Noyau :")
kernel_param_label.grid(row=5, column=0, sticky="w")
kernel_param_combobox = ttk.Combobox(root, values=["linear", "poly", "rbf", "sigmoid"])
kernel_param_combobox.current(2)
kernel_param_combobox.grid(row=5, column=1)

def train_button_click():
    # Récupérer les valeurs des paramètres du modèle
    c_value = float(c_entry.get())
    h_value = float(h_entry.get())
    kernel_value = kernel_combobox.get()

    # Entraîner le modèle avec les paramètres donnés
    model, accuracy = train_svm(c_value, h_value, kernel_value)

    # Afficher les résultats du modèle
    show_results(model, accuracy)

# Créer la fenêtre principale de l'interface graphique
root = tk.Tk()
root.title("Interface graphique pour SVM")

# Ajouter une zone de texte pour la description du module
module_description = tk.Text(root, height=5, wrap