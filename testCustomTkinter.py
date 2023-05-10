# objet : fenêtre avec deux frames
#import les biblio
import tkinter as tk
from tkinter import *
import customtkinter
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#fct pour recuperer la valeur de size du test
def getValeurTestSize() :
    val = testSize.get()
    return val
#fct pour recuperer la valeur de parametre kernel
def getValeurParamKernel() :
    val = paramKernel.get()
    return val
#fct pour recuperer la valeur de parametre C
def getValeurParamC() :
    val = paramC.get()
    return val

#fct pour entrainer le modele de svm du spam
def fitModelSpam() :
    sizetest = getValeurTestSize()
    k = getValeurParamKernel()
    C = getValeurParamC()
    print(sizetest)
    print(k)
    print(C)

#fct pour verifier que les inputs sont bien remplis et rendre le boutton de train et de test normal
def check_fields():
    if len(testSize.get()) > 0 and len(paramC.get()) > 0 and len(paramKernel.get()) > 0:
        btnTraining.config(state="normal")
        btnTesting.config(state="normal")
    else:
        btnTraining.config(state="disabled")
        btnTesting.config(state="disabled")

# Fonction pour tracer le graphe

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
appSVM = customtkinter.CTk()
appSVM.title("Interface Home Machine")
appSVM.grid_columnconfigure((0), weight=1)

# Centrer la fenêtre au milieu de l'écran
screen_width = appSVM.winfo_screenwidth()
screen_height = appSVM.winfo_screenheight()
x = int((screen_width / 2) - (1300 / 2))
y = int((screen_height / 2) - (750 / 2))
appSVM.geometry(f"1300x750+{x}+{y}")

# Rendre la fenêtre non-redimensionnable
appSVM.resizable(width=False, height=False)

customtkinter.set_default_color_theme("dark-blue")

frame1 = customtkinter.CTkFrame(master=appSVM, width=200, height=200)
frame2 = customtkinter.CTkFrame(master=appSVM, width=200, height=200)

datalabel = customtkinter.CTkLabel(appSVM, text="choisir le dataset : ",font=("Helvetica", 13))
datalabel.grid(row=0, column=0, padx=20, pady=20)

# Créer une liste déroulante
datasets = ["Dataset Spam Email" , "Dataset Maladies Cardiaques" ]
combo_box = customtkinter.CTkComboBox(appSVM, values=datasets,font=("Helvetica", 12))
combo_box.grid(row=1, column=0, padx=20, pady=20)

description = customtkinter.CTkLabel(appSVM, text="description : ", font=("Helvetica", 13))
description.grid(row=2, column=0, padx=20, pady=20)

# Création du Label pour afficher la valeur sélectionnée
descriptiontxt = customtkinter.CTkLabel(appSVM, text=" ", font=("Helvetica", 11) ,wraplength=360 , justify="left")
descriptiontxt.grid(row=3, column=0, padx=20, pady=20)

# Configuration de la ComboBox pour appeler la fonction update_label lors de la sélection d'une option
combo_box.bind("<<ComboboxSelected>>", update_label)

tstsize = customtkinter.CTkLabel(appSVM, text="test size: ",font=("Helvetica", 13))
tstsize.grid(row=4, column=0, padx=20, pady=20)

testSize = customtkinter.CTkEntry(appSVM,font=("Helvetica", 11))
testSize.grid(row=5, column=0, padx=20, pady=20)

parac = customtkinter.CTkLabel(appSVM, text="parametre C: ", font=("Helvetica", 13))
parac.grid(row=6, column=0, padx=20, pady=20)

paramC = customtkinter.CTkEntry(appSVM,font=("Helvetica", 11))
paramC.grid(row=7, column=0, padx=20, pady=20)

paramk = customtkinter.CTkLabel(appSVM, text="parametre Kernel: ", font=("Helvetica", 13))
paramk.grid(row=8, column=0, padx=20, pady=20)

paramKernel = customtkinter.CTkEntry(appSVM,font=("Helvetica", 11))
paramKernel.grid(row=9, column=0, padx=20, pady=20)
# Charger l'image et la convertir pour Tkinter
#icon_training = PhotoImage(file="imgs/training_80px.gif")

#creation de boutton pour entrainer le modele
btnTraining = customtkinter.CTkButton(appSVM , height=4, width=26, text="Training" ,font=('Helvetica', 15),state="disabled")
btnTraining.grid(row=0, column=2, padx=20, pady=20)

#creation de boutton pour tester le modele
btnTesting = customtkinter.CTkButton(appSVM , height=4, width=26, text="Testing" , font=('Helvetica', 15), command=fitModelSpam,state="disabled")
btnTesting.grid(row=1, column=2, padx=20, pady=20)

testSize.bind("<KeyRelease>", lambda event: check_fields())
paramC.bind("<KeyRelease>", lambda event: check_fields())
paramKernel.bind("<KeyRelease>", lambda event: check_fields())

appSVM.mainloop()
