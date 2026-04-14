# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:38:22 2026

@author: aneta
"""

import tkinter as tk

root = tk.Tk()
root.title("Test")
root.geometry("300x200")
label = tk.Label(root, text="Czy widzisz to okno?")
label.pack()
root.mainloop()
