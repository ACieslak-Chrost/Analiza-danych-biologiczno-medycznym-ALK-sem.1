# -*- coding: utf-8 -*-
"""
ANALIZA DANYCH MEDYCZNYCH - HOSPITALIZOWANI vs DO DOMU
Wersja z GUI - kopiuj i uruchom w Spyder
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
import os
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GŁÓWNA KLASA APLIKACJI
# =============================================================================
class MedicalAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Analiza Danych Medycznych - Hospitalizowani vs Do Domu")
        self.root.geometry("1400x900")
        
        self.df = None
        self.df_hosp = None
        self.df_dom = None
        self.wyniki = None
        
        # Lista parametrów (dostosuj do swojego pliku!)
        self.parametry = [
            'wiek', 'RR', 'MAP', 'SpO2', 'AS', 'mleczany',
            'kreatynina(0,5-1,2)', 'troponina I (0-7,8))',
            'HGB(12,4-15,2)', 'WBC(4-11)', 'plt(130-450)',
            'hct(38-45)', 'Na(137-145)', 'K(3,5-5,1)', 'crp(0-0,5)'
        ]
        
        self.stworz_gui()
    
    # =========================================================================
    # TWORZENIE INTERFEJSU
    # =========================================================================
    def stworz_gui(self):
        """Tworzy zakładki interfejsu"""
        # Notebook (zakładki)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Zakładka 1: Wczytaj dane
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="1. WCZYTAJ DANE")
        self.tab1_wczytaj()
        
        # Zakładka 2: Analiza
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="2. ANALIZA")
        self.tab2_analiza()
        
        # Zakładka 3: Wykresy
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="3. WYKRESY")
        self.tab3_wykresy()
        
        # Zakładka 4: Eksport
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="4. EKSPORT")
        self.tab4_eksport()
    
    def tab1_wczytaj(self):
        """Zakładka wczytywania danych"""
        frame = ttk.LabelFrame(self.tab1, text="Wczytaj plik z danymi", padding=20)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Button(frame, text="Wybierz plik CSV", 
                  command=self.wczytaj_csv).pack(pady=10, ipadx=20, ipady=5)
        
        ttk.Button(frame, text="Wybierz plik Excel", 
                  command=self.wczytaj_excel).pack(pady=10, ipadx=20, ipady=5)
        
        self.info = tk.Text(frame, height=8, width=70, font=('Courier', 10))
        self.info.pack(pady=20)
        self.info.insert(tk.END, "Wybierz plik CSV lub Excel z danymi pacjentów...")
        
        ttk.Button(frame, text="Przejdź do analizy →", 
                  command=lambda: self.notebook.select(self.tab2)).pack(pady=20)
    
    def tab2_analiza(self):
        """Zakładka analizy statystycznej"""
        # Górny panel - wybór
        gorny = ttk.Frame(self.tab2)
        gorny.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(gorny, text="Wybierz parametr:", font=('Arial', 11)).pack(side='left', padx=5)
        
        self.param_var = tk.StringVar()
        self.param_combo = ttk.Combobox(gorny, textvariable=self.param_var, 
                                        values=self.parametry, width=40, state='readonly')
        self.param_combo.pack(side='left', padx=5)
        
        ttk.Button(gorny, text="ANALIZUJ WSZYSTKIE", 
                  command=self.analizuj_wszystkie).pack(side='left', padx=20, ipadx=10)
        
        # Środkowy panel - tabela wyników
        srodkowy = ttk.LabelFrame(self.tab2, text="Wyniki analizy", padding=10)
        srodkowy.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrollbar
        scroll_y = ttk.Scrollbar(srodkowy)
        scroll_y.pack(side='right', fill='y')
        
        scroll_x = ttk.Scrollbar(srodkowy, orient='horizontal')
        scroll_x.pack(side='bottom', fill='x')
        
        # Treeview (tabela)
        self.tabela = ttk.Treeview(srodkowy, 
                                   columns=('parametr', 'hosp_n', 'hosp_sr', 'hosp_std',
                                           'dom_n', 'dom_sr', 'dom_std', 'p', 'istotnosc'),
                                   show='headings',
                                   yscrollcommand=scroll_y.set,
                                   xscrollcommand=scroll_x.set,
                                   height=20)
        
        # Nagłówki
        self.tabela.heading('parametr', text='Parametr')
        self.tabela.heading('hosp_n', text='n (hosp)')
        self.tabela.heading('hosp_sr', text='Średnia hosp')
        self.tabela.heading('hosp_std', text='SD hosp')
        self.tabela.heading('dom_n', text='n (dom)')
        self.tabela.heading('dom_sr', text='Średnia dom')
        self.tabela.heading('dom_std', text='SD dom')
        self.tabela.heading('p', text='p-value')
        self.tabela.heading('istotnosc', text='Istotność')
        
        # Szerokości
        self.tabela.column('parametr', width=200)
        self.tabela.column('hosp_n', width=60)
        self.tabela.column('hosp_sr', width=90)
        self.tabela.column('hosp_std', width=90)
        self.tabela.column('dom_n', width=60)
        self.tabela.column('dom_sr', width=90)
        self.tabela.column('dom_std', width=90)
        self.tabela.column('p', width=100)
        self.tabela.column('istotnosc', width=80)
        
        self.tabela.pack(fill='both', expand=True)
        
        scroll_y.config(command=self.tabela.yview)
        scroll_x.config(command=self.tabela.xview)
        
        # Dolny panel - statystyki
        dolny = ttk.Frame(self.tab2)
        dolny.pack(fill='x', padx=10, pady=5)
        
        self.status = tk.Text(dolny, height=3, width=70)
        self.status.pack(side='left', padx=5)
        
        ttk.Button(dolny, text="→ Wykresy", 
                  command=lambda: self.notebook.select(self.tab3)).pack(side='right', padx=5)
    
    def tab3_wykresy(self):
        """Zakładka z wykresami"""
        # Wybór parametru
        gorny = ttk.Frame(self.tab3)
        gorny.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(gorny, text="Parametr do wykresu:", font=('Arial', 11)).pack(side='left', padx=5)
        
        self.wykres_param = tk.StringVar()
        self.wykres_combo = ttk.Combobox(gorny, textvariable=self.wykres_param,
                                         values=self.parametry, width=40, state='readonly')
        self.wykres_combo.pack(side='left', padx=5)
        
        ttk.Button(gorny, text="RYSUN", 
                  command=self.rysuj_wykres).pack(side='left', padx=20, ipadx=10)
        
        # Miejsce na wykres
        self.figure = plt.Figure(figsize=(12, 7), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.tab3)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
    
    def tab4_eksport(self):
        """Zakładka eksportu"""
        frame = ttk.LabelFrame(self.tab4, text="Eksportuj wyniki", padding=20)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Button(frame, text="Zapisz do CSV", 
                  command=self.export_csv).pack(pady=10, ipadx=20, ipady=5)
        
        ttk.Button(frame, text="Zapisz do SQLite", 
                  command=self.export_sql).pack(pady=10, ipadx=20, ipady=5)
        
        ttk.Button(frame, text="Zapisz wykresy", 
                  command=self.export_wykresy).pack(pady=10, ipadx=20, ipady=5)
        
        self.export_info = tk.Text(frame, height=10, width=70, font=('Courier', 10))
        self.export_info.pack(pady=20)
    
    # =========================================================================
    # FUNKCJE WCZYTYWANIA
    # =========================================================================
    def wczytaj_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if filename:
            try:
                self.df = pd.read_csv(filename, sep=';', encoding='utf-8')
                self.przetworz_dane()
                self.info.delete(1.0, tk.END)
                self.info.insert(tk.END, f"✓ Wczytano: {os.path.basename(filename)}\n")
                self.info.insert(tk.END, f"✓ Liczba wierszy: {len(self.df)}\n")
                self.info.insert(tk.END, f"✓ Kolumny: {', '.join(list(self.df.columns)[:10])}...")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie wczytano: {e}")
    
    def wczytaj_excel(self):
        filename = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])
        if filename:
            try:
                self.df = pd.read_excel(filename)
                self.przetworz_dane()
                self.info.delete(1.0, tk.END)
                self.info.insert(tk.END, f"✓ Wczytano: {os.path.basename(filename)}\n")
                self.info.insert(tk.END, f"✓ Liczba wierszy: {len(self.df)}\n")
                self.info.insert(tk.END, f"✓ Kolumny: {', '.join(list(self.df.columns)[:10])}...")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie wczytano: {e}")
    
    def przetworz_dane(self):
        """Dzieli dane na hospitalizowanych i do domu"""
        if self.df is None:
            return
        
        # Znajdź pusty wiersz (podział)
        puste = self.df[self.df.isna().all(axis=1)]
        if len(puste) > 0:
            idx = puste.index[0]
            self.df_hosp = self.df.iloc[:idx].copy().dropna(how='all')
            self.df_dom = self.df.iloc[idx+1:].copy().dropna(how='all')
            
            # Konwersja na liczby
            for col in self.parametry:
                if col in self.df_hosp.columns:
                    self.df_hosp[col] = pd.to_numeric(self.df_hosp[col], errors='coerce')
                if col in self.df_dom.columns:
                    self.df_dom[col] = pd.to_numeric(self.df_dom[col], errors='coerce')
            
            self.status.delete(1.0, tk.END)
            self.status.insert(tk.END, f"✓ Hospitalizowani: {len(self.df_hosp)} pacjentów\n")
            self.status.insert(tk.END, f"✓ Do domu: {len(self.df_dom)} pacjentów")
            
            messagebox.showinfo("OK", f"Znaleziono {len(self.df_hosp)} hospitalizowanych i {len(self.df_dom)} do domu")
        else:
            messagebox.showwarning("Uwaga", "Nie znaleziono pustego wiersza - sprawdź dane")
    
    # =========================================================================
    # ANALIZA
    # =========================================================================
    def analizuj_wszystkie(self):
        """Analizuje wszystkie parametry"""
        if self.df_hosp is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj dane!")
            return
        
        # Wyczyść tabelę
        for row in self.tabela.get_children():
            self.tabela.delete(row)
        
        wyniki = []
        
        for param in self.parametry:
            if param in self.df_hosp.columns and param in self.df_dom.columns:
                hosp = self.df_hosp[param].dropna()
                dom = self.df_dom[param].dropna()
                
                if len(hosp) > 0 and len(dom) > 0:
                    # Statystyki
                    hosp_sr = hosp.mean()
                    hosp_std = hosp.std()
                    dom_sr = dom.mean()
                    dom_std = dom.std()
                    
                    # Test Manna-Whitneya
                    stat, p = stats.mannwhitneyu(hosp, dom, alternative='two-sided')
                    
                    # Istotność
                    if p < 0.001:
                        gwiazdki = "***"
                    elif p < 0.01:
                        gwiazdki = "**"
                    elif p < 0.05:
                        gwiazdki = "*"
                    else:
                        gwiazdki = "ns"
                    
                    # Wstaw do tabeli
                    self.tabela.insert('', 'end', values=(
                        param[:20],
                        len(hosp),
                        f"{hosp_sr:.2f}",
                        f"{hosp_std:.2f}",
                        len(dom),
                        f"{dom_sr:.2f}",
                        f"{dom_std:.2f}",
                        f"{p:.4f}",
                        gwiazdki
                    ))
                    
                    wyniki.append({
                        'parametr': param,
                        'hosp_n': len(hosp),
                        'hosp_sr': hosp_sr,
                        'hosp_std': hosp_std,
                        'dom_n': len(dom),
                        'dom_sr': dom_sr,
                        'dom_std': dom_std,
                        'p_value': p,
                        'istotnosc': gwiazdki
                    })
        
        self.wyniki = pd.DataFrame(wyniki)
        messagebox.showinfo("Gotowe", f"Przeanalizowano {len(wyniki)} parametrów")
    
    # =========================================================================
    # WYKRESY
    # =========================================================================
    def rysuj_wykres(self):
        """Rysuje wykres dla wybranego parametru"""
        param = self.wykres_param.get()
        if not param or self.df_hosp is None:
            return
        
        hosp = self.df_hosp[param].dropna()
        dom = self.df_dom[param].dropna()
        
        if len(hosp) == 0 or len(dom) == 0:
            messagebox.showwarning("Uwaga", "Brak danych dla tego parametru")
            return
        
        self.ax.clear()
        
        # Wykres pudełkowy
        bp = self.ax.boxplot([hosp, dom], 
                            labels=['Hospitalizowani', 'Do domu'],
                            patch_artist=True,
                            medianprops={'color': 'red', 'linewidth': 2})
        
        bp['boxes'][0].set_facecolor('#ff9999')
        bp['boxes'][1].set_facecolor('#99ccff')
        
        # Dodaj punkty
        x_hosp = np.random.normal(1, 0.05, len(hosp))
        x_dom = np.random.normal(2, 0.05, len(dom))
        self.ax.scatter(x_hosp, hosp, alpha=0.5, color='darkred', s=40)
        self.ax.scatter(x_dom, dom, alpha=0.5, color='darkblue', s=40)
        
        # Test
        stat, p = stats.mannwhitneyu(hosp, dom, alternative='two-sided')
        
        self.ax.set_title(f'{param} - p={p:.4f}')
        self.ax.set_ylabel('Wartość')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    # =========================================================================
    # EKSPORT
    # =========================================================================
    def export_csv(self):
        if self.wyniki is None:
            messagebox.showwarning("Uwaga", "Najpierw wykonaj analizę!")
            return
        
        f = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if f:
            self.wyniki.to_csv(f, sep=';', index=False, decimal=',')
            self.export_info.insert(tk.END, f"✓ Zapisano: {f}\n")
    
    def export_sql(self):
        if self.wyniki is None:
            messagebox.showwarning("Uwaga", "Najpierw wykonaj analizę!")
            return
        
        f = filedialog.asksaveasfilename(defaultextension=".db", filetypes=[("SQLite", "*.db")])
        if f:
            conn = sqlite3.connect(f)
            self.wyniki.to_sql('wyniki', conn, if_exists='replace', index=False)
            
            meta = pd.DataFrame({
                'data': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'hosp': [len(self.df_hosp)],
                'dom': [len(self.df_dom)]
            })
            meta.to_sql('metadata', conn, if_exists='replace', index=False)
            
            conn.close()
            self.export_info.insert(tk.END, f"✓ Zapisano bazę: {f}\n")
    
    def export_wykresy(self):
        if self.df_hosp is None:
            return
        
        folder = filedialog.askdirectory()
        if folder:
            for param in self.parametry:
                if param in self.df_hosp.columns and param in self.df_dom.columns:
                    hosp = self.df_hosp[param].dropna()
                    dom = self.df_dom[param].dropna()
                    
                    if len(hosp) > 0 and len(dom) > 0:
                        plt.figure(figsize=(10, 6))
                        plt.boxplot([hosp, dom], labels=['Hosp', 'Dom'])
                        plt.title(param)
                        plt.grid(color='lightgray')
                        plt.savefig(f'{folder}/wykres_{param}.png', dpi=150)
                        plt.close()
            
            self.export_info.insert(tk.END, f"✓ Wykresy zapisane w: {folder}\n")


# =============================================================================
# URUCHOMIENIE
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalAnalyzer(root)
    root.mainloop()