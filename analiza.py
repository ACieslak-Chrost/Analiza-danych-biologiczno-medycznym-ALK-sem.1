# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 09:35:20 2026

@author: aneta
"""

import pandas as pd
import os

print("Folder:", os.getcwd())
print("Pliki:", os.listdir())

df = pd.read_csv('baza_danych_pacjentów_a.csv', sep=';')
print(df.head())