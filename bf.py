import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = iris.data
target = iris.target
feature_names = iris.feature_names

# Create a DataFrame with the statistics from describe
statistics = pd.DataFrame(data).describe()

# Rename the columns to match the feature names
statistics.columns = feature_names

# Create the GUI window
window = tk.Tk()
window.title("Summary Statistics of Iris Dataset")

# Create a Treeview widget for displaying the table
tree = ttk.Treeview(window)
tree["columns"] = ["Statistic"] + list(statistics.columns)
tree["show"] = "headings"

# Add column headings to the Treeview
tree.heading("Statistic", text="Statistic")
for column in statistics.columns:
    tree.heading(column, text=column)

# Add rows to the Treeview
for index, row in statistics.iterrows():
    values = [index.capitalize()] + list(row)
    tree.insert("", "end", values=values)

# Place the Treeview in the GUI window
tree.pack()

# Run the GUI event loop
window.mainloop()
