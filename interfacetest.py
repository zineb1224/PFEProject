# objet : fenêtre avec deux frames
import tkinter as tk

def getValeur() :
    val = testSize.get()
    print(val)

# Création d'un objet "fenêtre"
appSVM = tk.Tk()  # nouvelle instance de Tk
appSVM.title("Interface Home Machine")
appSVM.geometry("1200x800")
appSVM.config(bg="black")

f1 = tk.LabelFrame(appSVM, bd=2, text="", bg="white", relief="groove")
f2 = tk.LabelFrame(appSVM, bd=2, text="", bg="white", relief="groove")
f1.pack(side=tk.LEFT, padx=20, pady=20)
f2.pack(side=tk.RIGHT, padx=20, pady=20)

# Créer une liste déroulante
listbox = tk.Listbox(f1, width=80)
listbox.insert(1, "Dataset Spam Email")
listbox.insert(2, "Dataset Maladies Cardiaques")
listbox.pack()

bg_color = "#f4f4f4"

e1 = tk.Label(f1, text="test size: ").pack(padx=50, pady=50)
testSize = tk.Entry(f1,width=40, bg=bg_color, fg="black" ,font=("Helvetica", 14) , bd=0, highlightthickness=1, highlightcolor="gray")
testSize.pack(padx=40, pady=5 , ipady=5)
e2 = tk.Label(f1, text="parametre C: ").pack(padx=50, pady=50)
paramC = tk.Entry(f1, width=40, bg=bg_color, fg="black" ,font=("Helvetica", 14) , bd=0, highlightthickness=1, highlightcolor="gray")
paramC.config(highlightbackground="white")
paramC.pack(padx=40, pady=5 , ipady=5)
e3 = tk.Label(f1, text="parametre Kernel: ").pack(padx=50, pady=50)
paramKernel = tk.Entry(f1, width=40, bg=bg_color, fg="black" ,font=("Helvetica", 14), bd=0, highlightthickness=1, highlightcolor="gray")
paramKernel.pack(padx=40, pady=5 , ipady=5)

btn = tk.Button(f1, height=1, width=10, text="Lire", command=getValeur)
btn.pack()

appSVM.mainloop()