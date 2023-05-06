from tkinter import *
import tkinter as tk

appSVM = tk.Tk()  # nouvelle instance de Tk

# Charger la première image
img1 = PhotoImage(file="imgs/training_80px.gif")

# Charger la deuxième image
img2 = PhotoImage(file="imgs/training_80px.gif")

# Utiliser les images dans votre interface utilisateur tkinter
btn1 = Button(appSVM , image=img1)
btn2 = Button(appSVM , image=img2)
btn1.pack()
btn2.pack()
appSVM.mainloop()