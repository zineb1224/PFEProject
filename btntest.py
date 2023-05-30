import tkinter as tk
from ttkbootstrap import Style
from ttkbootstrap.widgets import Combobox

root = tk.Tk()

style = Style(theme='flatly')

combo_box = Combobox(root, values=['Option 1', 'Option 2', 'Option 3'], font=('Helvetica', 12), width=35)
combo_box.theme_use('custom')
combo_box.configure(style='primary', background='red')
combo_box.pack(padx=10, pady=5, ipady=2)

root.mainloop()
