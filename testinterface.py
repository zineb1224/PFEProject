import tkinter as tk
from models.SVMModelSpam import SVMModelSpam, import_data
from models.SVMMODELMALADIE import MaladiesCardiaques

class MaladieApp:

    def __init__(self, master):
        self.master = master
        master.title("Prédiction de maladie cardiaque")

        # Créer les boutons pour les choix de modèle
        self.model_choice = tk.StringVar()
        self.model_choice.set("1")

        self.button_model1 = tk.Radiobutton(master, text="Module SVM Maladie 1", variable=self.model_choice, value="1")
        self.button_model1.pack()

        self.button_model2 = tk.Radiobutton(master, text="Module SVM Maladie 2", variable=self.model_choice, value="2")
        self.button_model2.pack()

        # Créer les champs pour saisir les données
        self.age_label = tk.Label(master, text="Age")
        self.age_label.pack()

        self.age_entry = tk.Entry(master)
        self.age_entry.pack()

        self.sex_label = tk.Label(master, text="Sexe (0 = Femme, 1 = Homme)")
        self.sex_label.pack()

        self.sex_entry = tk.Entry(master)
        self.sex_entry.pack()

        self.cp_label = tk.Label(master, text="Type de douleur thoracique (1-4)")
        self.cp_label.pack()

        self.cp_entry = tk.Entry(master)
        self.cp_entry.pack()

        # Créer le bouton de prédiction
        self.predict_button = tk.Button(master, text="Prédire", command=self.predict)
        self.predict_button.pack()

        # Créer la zone de texte pour afficher le résultat de la prédiction
        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

    def predict(self):
        # Obtenir les données saisies par l'utilisateur
        age = float(self.age_entry.get())
        sex = float(self.sex_entry.get())
        cp = float(self.cp_entry.get())

        # Sélectionner le modèle en fonction du choix de l'utilisateur
        if self.model_choice.get() == "1":
            model = MaladiesCardiaques()
        else:
            model = SVMModelSpam()

        # Prédire la maladie
        data = [[age, sex, cp]]
        prediction = model.predict(data)

        # Afficher le résultat de la prédiction
        if prediction[0] == 1:
            self.result_label.config(text="Malade")
        else:
            self.result_label.config(text="Non malade")

root = tk.Tk()
my_app = MaladieApp(root)
root.mainloop()
