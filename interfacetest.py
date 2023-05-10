# objet : fenêtre avec deux frames
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def getValeurTestSize() :
    val = testSize.get()
    return val

def getValeurParamKernel() :
    val = paramKernel.get()
    return val
def getValeurParamC() :
    val = paramC.get()
    return val

def fitModelSpam() :
    sizetest = getValeurTestSize()
    k = getValeurParamKernel()
    C = getValeurParamC()
    print(sizetest)
    print(k)
    print(C)

# Fonction pour tracer le graphe
def tracer_graphe():
    # Créez un widget Figure de Matplotlib
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3, 4, 5], [10, 5, 20, 15, 25])
    # Créez un widget Canvas Tkinter pour afficher la figure
    canvas = FigureCanvasTkAgg(fig, master=frame_graphe)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Fonction appelée lors de la sélection d'une option dans la ComboBox
def update_label(*args):
    selected_value = combo_box.get()
    if selected_value=="Dataset Spam Email" :
        descriptiontxt.configure(text="Le fichier csv contient 5172 lignes, chaque ligne"
                                      " pour chaque e-mail. Il y a 3002 colonnes. La première colonne indique le nom de l'e-mail."
                                      " Le nom a été défini avec des chiffres et non avec le nom des destinataires pour protéger la confidentialité. "
                                      "La dernière colonne contient les libellés de prédiction : 1 pour spam, 0 pour non spam."
                                      " Les 3000 colonnes restantes sont les 3000 mots les plus courants dans tous les e-mails,"
                                      " après exclusion des caractères/mots non alphabétiques. Pour chaque ligne, "
                                      "le nombre de chaque mot (colonne) dans cet e-mail (ligne) est stocké dans les cellules respectives. "
                                       )
    elif selected_value=="Dataset Maladies Cardiaques" :
        descriptiontxt.configure(text="description de maladie cardiaque ")


#creation de splash screen
splash_root = tk.Tk()
# Centrer la fenêtre au milieu de l'écran
screen_width = splash_root.winfo_screenwidth()
screen_height = splash_root.winfo_screenheight()
x = int((screen_width / 2) - (1300 / 2))
y = int((screen_height / 2) - (750 / 2))
splash_root.geometry(f"1300x750+{x}+{y}")

# Rendre la fenêtre non-redimensionnable
splash_root.resizable(width=False, height=False)

# Créez un canevas pour ajouter une image
splash_canvas = Canvas(splash_root, width=1300, height=750)
splash_canvas.pack()

# Charger l'image et la convertir pour Tkinter
image = Image.open("imgs/AI.gif")
photo = ImageTk.PhotoImage(image)

splash_canvas.create_image(650, 375, anchor=CENTER, image=photo)

# Ajoutez un message de chargement
splash_label = Label(splash_root, text="Chargement en cours...")
splash_label.pack()

# Définissez le temps d'affichage du splash screen en millisecondes
splash_time = 3000

# Fermez la fenêtre du splash screen après le temps d'affichage
splash_root.after(splash_time, splash_root.destroy)

# Création d'un objet "fenêtre"
appSVM = tk.Tk()  # nouvelle instance de Tk
appSVM.title("Interface Home Machine")

# Centrer la fenêtre au milieu de l'écran
screen_width = appSVM.winfo_screenwidth()
screen_height = appSVM.winfo_screenheight()
x = int((screen_width / 2) - (1300 / 2))
y = int((screen_height / 2) - (750 / 2))
appSVM.geometry(f"1300x750+{x}+{y}")
appSVM.config(bg="white")

# Rendre la fenêtre non-redimensionnable
appSVM.resizable(width=False, height=False)

bg_color_frame = "#f3f3f3"

f1 = tk.LabelFrame(appSVM, bd=2, text="", bg=bg_color_frame, relief="groove")
f2 = tk.LabelFrame(appSVM, bd=2, text="", bg=bg_color_frame, relief="groove")
f1.pack(side=tk.LEFT, padx=20, pady=20)
f2.pack(side=tk.RIGHT, padx=20, pady=20)

datalabel = tk.Label(f1, text="choisir le dataset : ", bg=bg_color_frame,font=("Helvetica", 13))
datalabel.pack(padx=50, pady=10)

# Créer une liste déroulante
datasets = ["Dataset Spam Email" , "Dataset Maladies Cardiaques" ]
combo_box = ttk.Combobox(f1, values=datasets,font=("Helvetica", 12), width=35)

# Configuration des couleurs de fond et de texte pour la ComboBox
combo_box.configure(background="#F5F5F5", foreground="black")
combo_box.pack(padx=50, pady=5)


description = tk.Label(f1, text="description : ",bg=bg_color_frame, font=("Helvetica", 13))
description.pack(padx=20, pady=5)

# Création du Label pour afficher la valeur sélectionnée
descriptiontxt = tk.Label(f1, text=" ",bg=bg_color_frame , font=("Helvetica", 11) ,wraplength=360 , justify="left")
descriptiontxt.pack(padx=5, pady=5)

# Configuration de la ComboBox pour appeler la fonction update_label lors de la sélection d'une option
combo_box.bind("<<ComboboxSelected>>", update_label)

bg_color = "#fdfdfd"

# Création d'un style personnalisé pour Entry
style = ttk.Style()

# Configuration de la bordure et du relief pour l'Entry
style.configure("Custom.TEntry", borderwidth=0, relief="solid")

# Configuration de la bordure arrondie pour l'Entry
style.map("Custom.TEntry",
          foreground=[("focus", "black"), ("!focus", "gray")],
          background=[("focus", bg_color), ("!focus", bg_color)],
          bordercolor=[("focus", "gray"), ("!focus", "gray")],
          borderwidth=[("focus", 2), ("!focus", 2)],
          relief=[("focus", "solid"), ("!focus", "solid")],
          focuscolor=[("focus", "white"), ("!focus", "white")],
          highlightthickness=[("focus", 2), ("!focus", 0)],
          padx=[("focus", 6), ("!focus", 6)],
          pady=[("focus", 6), ("!focus", 6)],
          )

tstsize = tk.Label(f1, text="test size: ", bg=bg_color_frame,font=("Helvetica", 13))
tstsize.pack(padx=50, pady=10)

testSize = ttk.Entry(f1, style="Custom.TEntry" , width=40,font=("Helvetica", 11))
testSize.pack(pady=8,ipady=5)

parac = tk.Label(f1, text="parametre C: ",bg=bg_color_frame, font=("Helvetica", 13))
parac.pack(padx=50, pady=10)

paramC = ttk.Entry(f1, style="Custom.TEntry" , width=40,font=("Helvetica", 11))
paramC.pack(pady=8,ipady=5)

paramk = tk.Label(f1, text="parametre Kernel: ",bg=bg_color_frame, font=("Helvetica", 13))
paramk.pack(padx=50, pady=10)

paramKernel = ttk.Entry(f1, style="Custom.TEntry" , width=40,font=("Helvetica", 11))
paramKernel.pack(pady=8,ipady=5)

# Charger l'image et la convertir pour Tkinter
#icon_training = PhotoImage(file="imgs/training_80px.gif")

#creation de boutton pour entrainer le modele
btnTraining = tk.Button(f2 , height=4, width=26, text="Training" ,font=('Helvetica', 15), fg='#FFFFFF', bg='#9AC8EB', bd=0, command=tracer_graphe)
btnTraining.pack(padx=20,pady=5)

# Création d'un cadre dans la fenêtre Tkinter pour y afficher le graphe
frame_graphe = tk.LabelFrame(f2, bd=0, bg="#f3f3f3", relief="groove")
frame_graphe.pack()

#creation de boutton pour tester le modele
btnTesting = tk.Button(f2 , height=4, width=26, text="Testing" , font=('Helvetica', 15), fg='#FFFFFF', bg='#9AC8EB', bd=0 , command=fitModelSpam)
btnTesting.pack(padx=20,pady=5)

appSVM.mainloop()