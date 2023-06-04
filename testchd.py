from tkinter import Tk, Button
from PIL import ImageTk, Image

# Créer une fenêtre Tkinter
fenetre = Tk()

# Charger l'image de l'icône avec Pillow
image = Image.open("imgs/16x16.png")

# Créer un widget PhotoImage à partir de l'image
icone = ImageTk.PhotoImage(image)

# Créer un bouton avec l'icône
bouton = Button(fenetre, image=icone)

# Afficher le bouton
bouton.pack()

# Lancer la boucle principale de la fenêtre
fenetre.mainloop()
