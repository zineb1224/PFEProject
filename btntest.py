from tkinter import Tk, Button, Canvas

from PIL import Image, ImageTk


# Fonction de gestion de l'événement lors du clic sur le bouton
def clic():
    print("Clic !")

# Créer la fenêtre principale
fenetre = Tk()
# Charger l'image et la convertir pour Tkinter
icon_training = Image.open("imgs/16x16.png")
icn_training = ImageTk.PhotoImage(icon_training)
btnTraining = Button(fenetre, height=3, width=24, text="Training", font=('Helvetica', 15, "bold"), fg='#FFFFFF', bg='#76B8E0', bd=0, image=icn_training)
btnTraining.pack(padx=20, pady=5)

# Lancer la boucle principale
fenetre.mainloop()
