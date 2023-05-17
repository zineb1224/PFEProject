import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn import svm
from sklearn.datasets import make_blobs
import numpy as np

canvas = None
def plot_svm_graph():
    global canvas
    # Détruire le canvas s'il existe déjà
    if canvas:
        canvas.get_tk_widget().destroy()
    # Générer le jeu de données de classification binaire
    X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.6)
    # Créer le modèle SVM et l'entraîner sur les données
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)
    # Créer la grille de points pour l'affichage du graphe
    xx = np.linspace(-1, 5)
    yy = np.linspace(-1, 5)
    xx, yy = np.meshgrid(xx, yy)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    # Calculer les prédictions du modèle SVM sur la grille de points
    Z = clf.decision_function(xy).reshape(xx.shape)
    # Créer le graphe
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    # Afficher les données de classification binaire
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    # Afficher les frontières de décision du modèle SVM
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('SVM Classification')
    # Afficher le graphe dans la frame
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.get_tk_widget().pack()

def clear_graph():
    global canvas
    # Détruire le canvas s'il existe déjà
    if canvas:
        canvas.get_tk_widget().destroy()
        canvas = None

root = tk.Tk()

graph_frame = tk.Frame(root)
graph_frame.pack()

btn_frame = tk.Frame(root)
btn_frame.pack()

btn_plot = tk.Button(btn_frame, text="Plot SVM Graph", command=plot_svm_graph)
btn_plot.pack(side=tk.LEFT, padx=5)

btn_clear = tk.Button(btn_frame, text="Clear Graph", command=clear_graph)
btn_clear.pack(side=tk.LEFT)

root.mainloop()
