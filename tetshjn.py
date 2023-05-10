import tkinter as tk

# Définition de la palette de couleurs
BG_COLOR = "#1c1c1c"
FG_COLOR = "#d9d9d9"
BUTTON_BG_COLOR = "#3a3a3a"
BUTTON_ACTIVE_COLOR = "#4a4a4a"

# Création de la fenêtre principale
root = tk.Tk()
root.configure(bg=BG_COLOR)

# Modification des couleurs des widgets
tk.Label(root, text="Hello World", bg=BG_COLOR, fg=FG_COLOR).pack(pady=20)
tk.Button(root, text="Click me", bg=BUTTON_BG_COLOR, fg=FG_COLOR, activebackground=BUTTON_ACTIVE_COLOR).pack()

# Lancement de la boucle principale
root.mainloop()
