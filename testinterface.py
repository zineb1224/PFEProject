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
import tkinter as tk
from tkinter import ttk

def getValeur():
    val = testSize.get()
    print(val)

# Création de la fenêtre principale
app = tk.Tk()
app.geometry("800x600")
app.title("Mon application")

# Création des frames
frame1 = ttk.Frame(app, padding=20)
frame2 = ttk.Frame(app, padding=20)
frame1.pack(side="left", fill="both", expand=True)
frame2.pack(side="right", fill="both", expand=True)

# Création de la liste déroulante
listbox = ttk.Combobox(frame1, values=["Dataset Spam Email", "Dataset Maladies Cardiaques"])
listbox.pack(pady=10)

# Création des champs de saisie
testSize_label = ttk.Label(frame1, text="Test size:")
testSize_label.pack(pady=10)
testSize = ttk.Entry(frame1)
testSize.pack(pady=5)

paramC_label = ttk.Label(frame1, text="Paramètre C:")
paramC_label.pack(pady=10)
paramC = ttk.Entry(frame1)
paramC.pack(pady=5)

paramKernel_label = ttk.Label(frame1, text="Paramètre Kernel:")
paramKernel_label.pack(pady=10)
paramKernel = ttk.Entry(frame1)
paramKernel.pack(pady=5)

# Bouton de validation
btn = ttk.Button(frame1, text="Valider", command=getValeur)
btn.pack(pady=10)

# Lancement de la fenêtre
app.mainloop()
