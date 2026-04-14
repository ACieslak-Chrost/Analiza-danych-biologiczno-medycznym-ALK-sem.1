# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 09:04:24 2026

@author: aneta
"""
#1. Automatyczna normalizacja płci
#Dodałam funkcję normalize_gender(), która:

#Zamienia wszystkie warianty 'k', 'K', 'kobieta', 'female' na duże 'K'

#Zamienia 'm', 'M', 'mezczyzna', 'male' na duże 'M'

#Tworzy nową kolumnę plec_normalized z ujednoliconymi wartościami

#2. Analiza według płci
#Dodałam w interfejsie:

#Opcje wyboru: "Wszyscy", "Kobiety", "Mężczyźni"

#Informację o automatycznej normalizacji (zielony tekst)

#3. Lepsze statystyki
#Przy każdej analizie pokazuje, dla której grupy płciowej są wyniki

#Wykresy mają tytuły z informacją o grupie

#Anomalie są liczone osobno dla każdej grupy

#4. Ulepszony eksport
#Przy eksporcie dodaje kolumny z informacją o anomaliach dla każdej płci

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import json
import requests
from scipy import stats
import os

class MedicalDataAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizator Danych Medycznych")
        self.root.geometry("1400x800")
        
        self.df = None
        self.measurement_columns = ['kreatynina(0,5-1,2)', 'troponina I (0-7,8))', 'WBC(4-11)', 
                                   'plt(130-450)', 'hct(38-45)', 'Na(137-145)', 
                                   'K(3,5-5,1)', 'crp(0-0,5)']
        
        self.setup_gui()
        
    def setup_gui(self):
        """Tworzy interfejs użytkownika"""
        # Główny frame podzielony na lewą i prawą część
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Lewy panel - sterowanie
        left_frame = ttk.LabelFrame(main_frame, text="Panel sterowania", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Przyciski wczytywania danych
        ttk.Button(left_frame, text="Wczytaj z pliku Excel", 
                  command=self.load_excel).grid(row=0, column=0, pady=5, sticky=tk.W)
        
        ttk.Button(left_frame, text="Wczytaj z pliku CSV", 
                  command=self.load_csv).grid(row=1, column=0, pady=5, sticky=tk.W)
        
        ttk.Button(left_frame, text="Pobierz z API", 
                  command=self.load_from_api).grid(row=2, column=0, pady=5, sticky=tk.W)
        
        ttk.Button(left_frame, text="Wczytaj z JSON", 
                  command=self.load_json).grid(row=3, column=0, pady=5, sticky=tk.W)
        
        # Separator
        ttk.Separator(left_frame, orient='horizontal').grid(row=4, column=0, pady=10, sticky=tk.EW)
        
        # Wybór parametru do analizy
        ttk.Label(left_frame, text="Wybierz parametr do analizy:").grid(row=5, column=0, pady=5, sticky=tk.W)
        self.param_var = tk.StringVar()
        self.param_combo = ttk.Combobox(left_frame, textvariable=self.param_var, 
                                        values=self.measurement_columns, state='readonly')
        self.param_combo.grid(row=6, column=0, pady=5, sticky=tk.W+tk.E)
        self.param_combo.bind('<<ComboboxSelected>>', self.update_analysis)
        
        # Zakres analizy
        ttk.Label(left_frame, text="Zakres analizy (opcjonalnie):").grid(row=7, column=0, pady=5, sticky=tk.W)
        
        range_frame = ttk.Frame(left_frame)
        range_frame.grid(row=8, column=0, pady=5)
        
        ttk.Label(range_frame, text="Od:").grid(row=0, column=0)
        self.start_range = ttk.Entry(range_frame, width=10)
        self.start_range.grid(row=0, column=1, padx=2)
        
        ttk.Label(range_frame, text="Do:").grid(row=0, column=2)
        self.end_range = ttk.Entry(range_frame, width=10)
        self.end_range.grid(row=0, column=3, padx=2)
        
        ttk.Button(range_frame, text="Zastosuj zakres", 
                  command=self.apply_range).grid(row=0, column=4, padx=5)
        
        # Przycisk eksportu
        ttk.Button(left_frame, text="Eksportuj wyniki do CSV", 
                  command=self.export_results).grid(row=9, column=0, pady=10)
        
        # Prawy panel - wyniki i wizualizacja
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame dla statystyk
        self.stats_frame = ttk.LabelFrame(right_frame, text="Wyniki statystyczne", padding="10")
        self.stats_frame.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.stats_text = tk.Text(self.stats_frame, height=10, width=60)
        self.stats_text.grid(row=0, column=0)
        
        # Frame dla wykresu
        self.plot_frame = ttk.LabelFrame(right_frame, text="Wizualizacja danych", padding="10")
        self.plot_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)
        
        # Frame dla anomalii
        self.anomaly_frame = ttk.LabelFrame(right_frame, text="Wykryte anomalie", padding="10")
        self.anomaly_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.anomaly_text = tk.Text(self.anomaly_frame, height=5, width=60)
        self.anomaly_text.grid(row=0, column=0)
        
        # Dodanie opcji analizy według płci
        self.add_gender_analysis_option(left_frame)
    
    def add_gender_analysis_option(self, parent):
        """Dodaje opcję analizy według płci"""
        ttk.Separator(parent, orient='horizontal').grid(row=10, column=0, pady=10, sticky=tk.EW)
        
        ttk.Label(parent, text="Analiza według płci:").grid(row=11, column=0, pady=5, sticky=tk.W)
        
        self.gender_var = tk.StringVar(value="wszyscy")
        gender_frame = ttk.Frame(parent)
        gender_frame.grid(row=12, column=0, pady=5)
        
        ttk.Radiobutton(gender_frame, text="Wszyscy", variable=self.gender_var, 
                       value="wszyscy", command=self.update_analysis).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(gender_frame, text="Kobiety", variable=self.gender_var, 
                       value="kobiety", command=self.update_analysis).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(gender_frame, text="Mężczyźni", variable=self.gender_var, 
                       value="mezczyzni", command=self.update_analysis).grid(row=0, column=2, padx=5)
        
        # Informacja o normalizacji płci
        ttk.Label(parent, text="✓ Automatyczna normalizacja: k/K → K, m/M → M", 
                 foreground="green").grid(row=13, column=0, pady=5)
    
    def normalize_gender(self):
        """Normalizuje wartości w kolumnie płci"""
        if self.df is not None and 'plec' in self.df.columns:
            # Konwersja na string i usunięcie białych znaków
            self.df['plec'] = self.df['plec'].astype(str).str.strip()
            
            # Mapowanie różnych wariantów na ujednolicone wartości
            gender_mapping = {
                'k': 'K', 'K': 'K', 'kobieta': 'K', 'Kobieta': 'K', 'female': 'K', 'F': 'K',
                'm': 'M', 'M': 'M', 'mezczyzna': 'M', 'Mężczyzna': 'M', 'male': 'M', 'M': 'M'
            }
            
            # Zastosowanie mapowania
            self.df['plec_normalized'] = self.df['plec'].map(gender_mapping).fillna('Inne')
            
            # Wyświetlenie informacji o normalizacji
            unique_before = self.df['plec'].unique()
            unique_after = self.df['plec_normalized'].unique()
            
            print(f"Wartości przed normalizacją: {unique_before}")
            print(f"Wartości po normalizacji: {unique_after}")
            
            # Statystyki
            gender_stats = self.df['plec_normalized'].value_counts()
            print(f"Statystyki płci po normalizacji:\n{gender_stats}")
            
            return True
        return False
    
    def load_excel(self):
        """Wczytuje dane z pliku Excel"""
        filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if filename:
            try:
                self.df = pd.read_excel(filename)
                self.normalize_gender()  # Normalizacja płci zaraz po wczytaniu
                self.process_data()
                messagebox.showinfo("Sukces", f"Wczytano {len(self.df)} rekordów z pliku Excel\n"
                                   f"Znormalizowano wartości płci")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać pliku: {str(e)}")
    
    def load_csv(self):
        """Wczytuje dane z pliku CSV"""
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            try:
                self.df = pd.read_csv(filename)
                self.normalize_gender()  # Normalizacja płci zaraz po wczytaniu
                self.process_data()
                messagebox.showinfo("Sukces", f"Wczytano {len(self.df)} rekordów z pliku CSV\n"
                                   f"Znormalizowano wartości płci")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać pliku: {str(e)}")
    
    def load_json(self):
        """Wczytuje dane z pliku JSON"""
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.df = pd.DataFrame(data)
                self.normalize_gender()  # Normalizacja płci zaraz po wczytaniu
                self.process_data()
                messagebox.showinfo("Sukces", f"Wczytano {len(self.df)} rekordów z pliku JSON\n"
                                   f"Znormalizowano wartości płci")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać pliku: {str(e)}")
    
    def load_from_api(self):
        """Pobiera dane z API"""
        # Przykładowe API - można zmienić na rzeczywiste
        api_url = tk.simpledialog.askstring("API URL", "Podaj URL API:")
        if api_url:
            try:
                response = requests.get(api_url)
                if response.status_code == 200:
                    data = response.json()
                    self.df = pd.DataFrame(data)
                    self.normalize_gender()  # Normalizacja płci zaraz po pobraniu
                    self.process_data()
                    messagebox.showinfo("Sukces", f"Pobrano {len(self.df)} rekordów z API\n"
                                       f"Znormalizowano wartości płci")
                else:
                    messagebox.showerror("Błąd", f"Błąd połączenia: {response.status_code}")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się pobrać danych: {str(e)}")
    
    def process_data(self):
        """Przetwarza wczytane dane"""
        if self.df is not None:
            # Oczyszczanie danych
            for col in self.measurement_columns:
                if col in self.df.columns:
                    # Zamiana przecinków na kropki i konwersja na liczby
                    self.df[col] = self.df[col].astype(str).str.replace(',', '.').str.replace('<', '')
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Aktualizacja combobox
            self.param_combo['values'] = [col for col in self.measurement_columns if col in self.df.columns]
    
    def apply_range(self):
        """Stosuje zakres analizy"""
        self.update_analysis()
    
    def clean_data(self, data):
        """Czyści dane z wartości odstających"""
        # Usuń wartości NaN
        data = data.dropna()
        return data
    
    def calculate_statistics(self, data):
        """Oblicza statystyki dla danych"""
        if len(data) > 0:
            stats_dict = {
                'Liczba pomiarów': len(data),
                'Średnia': np.mean(data),
                'Mediana': np.median(data),
                'Odchylenie standardowe': np.std(data),
                'Minimum': np.min(data),
                'Maksimum': np.max(data),
                'Kwartyl 25%': np.percentile(data, 25),
                'Kwartyl 75%': np.percentile(data, 75)
            }
            return stats_dict
        return {}
    
    def detect_anomalies(self, data):
        """Wykrywa anomalie w danych"""
        if len(data) > 1:
            # Metoda Z-score
            z_scores = np.abs(stats.zscore(data))
            anomalies_z = data[z_scores > 2]
            
            # Metoda IQR
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            anomalies_iqr = data[(data < lower_bound) | (data > upper_bound)]
            
            return {
                'metoda_z_score': anomalies_z.tolist() if len(anomalies_z) > 0 else [],
                'metoda_iqr': anomalies_iqr.tolist() if len(anomalies_iqr) > 0 else []
            }
        return {'metoda_z_score': [], 'metoda_iqr': []}
    
    def update_analysis(self, event=None):
        """Aktualizuje analizę dla wybranego parametru"""
        if self.df is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj dane!")
            return
        
        param = self.param_var.get()
        if not param or param not in self.df.columns:
            return
        
        # Filtruj dane według płci
        gender_filter = self.gender_var.get()
        filtered_df = self.df.copy()
        
        if gender_filter != "wszyscy" and 'plec_normalized' in filtered_df.columns:
            if gender_filter == "kobiety":
                filtered_df = filtered_df[filtered_df['plec_normalized'] == 'K']
            elif gender_filter == "mezczyzni":
                filtered_df = filtered_df[filtered_df['plec_normalized'] == 'M']
        
        # Pobierz dane
        data = filtered_df[param].copy()
        
        # Zastosuj zakres jeśli podany
        try:
            start = int(self.start_range.get()) if self.start_range.get() else 0
            end = int(self.end_range.get()) if self.end_range.get() else len(data)
            if start < end <= len(data):
                data = data.iloc[start:end]
        except ValueError:
            pass
        
        # Wyczyść dane
        data = self.clean_data(data)
        
        if len(data) == 0:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Brak danych do analizy dla wybranej płci")
            return
        
        # Oblicz statystyki
        stats = self.calculate_statistics(data)
        
        # Wyświetl statystyki
        self.stats_text.delete(1.0, tk.END)
        
        # Nagłówek z informacją o płci
        gender_info = {
            "wszyscy": "Wszyscy pacjenci",
            "kobiety": "Kobiety",
            "mezczyzni": "Mężczyźni"
        }
        
        self.stats_text.insert(tk.END, f"Analiza parametru: {param}\n")
        self.stats_text.insert(tk.END, f"Grupa: {gender_info[gender_filter]}\n")
        self.stats_text.insert(tk.END, "="*40 + "\n")
        
        for key, value in stats.items():
            self.stats_text.insert(tk.END, f"{key}: {value:.2f}\n")
        
        # Narysuj wykres
        self.ax.clear()
        self.ax.plot(data.values, marker='o', linestyle='-', linewidth=2, markersize=4)
        self.ax.set_title(f'Wykres pomiarów: {param} - {gender_info[gender_filter]}')
        self.ax.set_xlabel('Numer pomiaru')
        self.ax.set_ylabel('Wartość')
        self.ax.grid(True, alpha=0.3)
        
        # Dodanie linii normy jeśli dostępna
        norm_ranges = {
            'kreatynina(0,5-1,2)': (0.5, 1.2),
            'troponina I (0-7,8))': (0, 7.8),
            'WBC(4-11)': (4, 11),
            'plt(130-450)': (130, 450),
            'hct(38-45)': (38, 45),
            'Na(137-145)': (137, 145),
            'K(3,5-5,1)': (3.5, 5.1),
            'crp(0-0,5)': (0, 0.5)
        }
        
        if param in norm_ranges:
            lower, upper = norm_ranges[param]
            self.ax.axhline(y=lower, color='r', linestyle='--', alpha=0.5, label='Dolna granica normy')
            self.ax.axhline(y=upper, color='r', linestyle='--', alpha=0.5, label='Górna granica normy')
            self.ax.legend()
        
        self.canvas.draw()
        
        # Wykryj anomalie
        anomalies = self.detect_anomalies(data)
        
        self.anomaly_text.delete(1.0, tk.END)
        self.anomaly_text.insert(tk.END, f"Wykryte anomalie - {gender_info[gender_filter]}:\n")
        self.anomaly_text.insert(tk.END, "-"*30 + "\n")
        
        if anomalies['metoda_z_score']:
            self.anomaly_text.insert(tk.END, f"Metoda Z-score (>2σ): {len(anomalies['metoda_z_score'])} wartości\n")
            self.anomaly_text.insert(tk.END, f"Wartości: {', '.join([f'{x:.2f}' for x in anomalies['metoda_z_score'][:5]])}\n")
        else:
            self.anomaly_text.insert(tk.END, "Metoda Z-score: brak anomalii\n")
        
        if anomalies['metoda_iqr']:
            self.anomaly_text.insert(tk.END, f"Metoda IQR: {len(anomalies['metoda_iqr'])} wartości\n")
            self.anomaly_text.insert(tk.END, f"Wartości: {', '.join([f'{x:.2f}' for x in anomalies['metoda_iqr'][:5]])}\n")
        else:
            self.anomaly_text.insert(tk.END, "Metoda IQR: brak anomalii\n")
    
    def export_results(self):
        """Eksportuje wyniki analizy do pliku CSV"""
        if self.df is None:
            messagebox.showwarning("Uwaga", "Brak danych do eksportu!")
            return
        
        filename = filedialog.asksaveasfilename(defaultextension=".csv", 
                                               filetypes=[("CSV files", "*.csv")])
        if filename:
            try:
                # Przygotuj dane do eksportu
                export_df = self.df.copy()
                
                # Dodaj kolumny z analizą dla wybranego parametru
                param = self.param_var.get()
                if param and param in self.df.columns:
                    # Użyj znormalizowanej płci
                    if 'plec_normalized' in export_df.columns:
                        for gender in ['K', 'M']:
                            gender_data = self.df[self.df['plec_normalized'] == gender][param]
                            if len(gender_data) > 0:
                                anomalies = self.detect_anomalies(gender_data)
                                export_df[f'czy_anomalia_{gender}_z_score'] = False
                                export_df.loc[export_df['plec_normalized'] == gender, f'czy_anomalia_{gender}_z_score'] = \
                                    export_df[export_df['plec_normalized'] == gender][param].isin(anomalies['metoda_z_score'])
                
                export_df.to_csv(filename, index=False, encoding='utf-8')
                messagebox.showinfo("Sukces", f"Wyniki wyeksportowane do {filename}")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się eksportować: {str(e)}")

def main():
    root = tk.Tk()
    app = MedicalDataAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()