import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Créez la fenêtre principale Tkinter
root = tk.Tk()
root.title("Graph Viewer")

# Créez un widget Figure de Matplotlib
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
ax.plot([1, 2, 3, 4, 5], [10, 5, 20, 15, 25])

# Créez un widget Canvas Tkinter pour afficher la figure
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# Lancez la boucle principale de la fenêtre Tkinter
tk.mainloop()