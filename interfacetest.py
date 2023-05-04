# objet : fenêtre avec deux frames
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

def getValeur() :
    val = testSize.get()
    print(val)

def update_label(*args):
    selected_value.set(listbox.get(tk.ACTIVE))


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
splash_time = 5000

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

# Créer une liste déroulante
listbox = tk.Listbox(f1, width=80)
listbox.insert(1, "Dataset Spam Email")
listbox.insert(2, "Dataset Maladies Cardiaques")
listbox.pack()

# Créer une variable de contrôle pour stocker la valeur sélectionnée
selected_value = tk.StringVar()

# Lier la variable de contrôle à la liste déroulante
selected_value.trace('w', update_label)

# Créer une étiquette de texte pour afficher la valeur sélectionnée
label = tk.Label(f1, textvariable=selected_value , bg=bg_color_frame,font=("Helvetica", 14))
label.pack(padx=50, pady=10)

bg_color = "#fdfdfd"

tstsize = tk.Label(f1, text="test size: ", bg=bg_color_frame,font=("Helvetica", 14))
tstsize.pack(padx=50, pady=10)

testSize = tk.Entry(f1,width=40, bg=bg_color, fg="black" ,font=("Helvetica", 14) , bd=0, highlightthickness=1, highlightcolor="gray")
testSize.pack(padx=40, pady=2 , ipady=5)

parac = tk.Label(f1, text="parametre C: ",bg=bg_color_frame, font=("Helvetica", 14))
parac.pack(padx=50, pady=10)

paramC = tk.Entry(f1, width=40, bg=bg_color, fg="black" ,font=("Helvetica", 14) , bd=0, highlightthickness=1, highlightcolor="gray")
paramC.pack(padx=40, pady=2 , ipady=5)

paramk = tk.Label(f1, text="parametre Kernel: ",bg=bg_color_frame, font=("Helvetica", 14))
paramk.pack(padx=50, pady=10)

paramKernel = tk.Entry(f1, width=40, bg=bg_color, fg="black" ,font=("Helvetica", 14), bd=0, highlightthickness=1, highlightcolor="gray")
paramKernel.pack(padx=40, pady=2 , ipady=5)

btn = tk.Button(f1, height=1, width=10, text="Lire", command=getValeur)
btn.pack()

appSVM.mainloop()