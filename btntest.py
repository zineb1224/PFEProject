import tkinter as tk
import tkinter.ttk as ttk

# Create the root window
root = tk.Tk()

# Create the notebook widget
notebook = ttk.Notebook(root)

# Create the first tab
tab1 = tk.Frame(notebook)
tab1.pack()

# Create a label in the first tab
label1 = tk.Label(tab1, text="This is the first tab.")
label1.pack()

# Create the second tab
tab2 = tk.Frame(notebook)
tab2.pack()

# Create a label in the second tab
label2 = tk.Label(tab2, text="This is the second tab.")
label2.pack()

# Add the tabs to the notebook
notebook.add(tab1, text="Tab 1")
notebook.add(tab2, text="Tab 2")

# Pack the notebook
notebook.pack()

# Start the main loop
root.mainloop()