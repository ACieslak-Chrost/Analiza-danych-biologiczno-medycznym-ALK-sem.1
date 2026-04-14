# -*- coding: utf-8 -*-
"""
✨ PRZEPIĘKNE GUI - ANALIZA DANYCH MEDYCZNYCH ✨
Wersja: 9.0 - Profesjonalny interfejs
Autor: Aneta
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import seaborn as sns
from scipy import stats
from math import pi
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# USTAWIENIA STYLU
# =============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

KOLORY = {
    'primary': '#2c3e50',
    'secondary': '#34495e',
    'accent1': '#e74c3c',
    'accent2': '#3498db',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'light': '#ecf0f1',
    'dark': '#2c3e50',
    'hosp': '#e74c3c',
    'dom': '#3498db',
    'bg': '#f5f5f5',
    'fg': '#2c3e50'
}

# =============================================================================
# GŁÓWNA KLASA APLIKACJI
# =============================================================================
class MedicalAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("✨ ANALIZATOR DANYCH MEDYCZNYCH ✨")
        self.root.geometry("1400x900")
        self.root.configure(bg=KOLORY['bg'])
        
        # Ikona (jeśli dostępna)
        try:
            self.root.iconbitmap(default='icon.ico')
        except:
            pass
        
        # Zmienne
        self.df = None
        self.df_hosp = None
        self.df_dom = None
        self.wyniki_df = None
        self.current_figure = None
        self.current_param = None
        
        # Listy parametrów
        self.parametry_kliniczne = [
            'wiek', 'RR', 'MAP', 'SpO2', 'AS', 'mleczany',
            'kreatynina(0,5-1,2)', 'troponina I (0-7,8))',
            'HGB(12,4-15,2)', 'WBC(4-11)', 'plt(130-450)',
            'hct(38-45)', 'Na(137-145)', 'K(3,5-5,1)', 'crp(0-0,5)'
        ]
        
        self.choroby = ['dm', 'wątroba', 'naczyniowe', 'zza', 'npl']
        
        self.setup_ui()
    
    # =========================================================================
    # INTERFEJS UŻYTKOWNIKA
    # =========================================================================
    def setup_ui(self):
        """Tworzy główny interfejs"""
        # Styl
        style = ttk.Style()
        style.theme_use('clam')
        
        # Konfiguracja kolorów
        style.configure('TNotebook', background=KOLORY['bg'], borderwidth=0)
        style.configure('TNotebook.Tab', 
                       font=('Helvetica', 12, 'bold'),
                       padding=[20, 10],
                       background=KOLORY['light'],
                       foreground=KOLORY['dark'])
        style.map('TNotebook.Tab',
                 background=[('selected', KOLORY['primary'])],
                 foreground=[('selected', 'white')])
        
        style.configure('TButton', 
                       font=('Helvetica', 11),
                       padding=10,
                       background=KOLORY['accent1'],
                       foreground='white')
        style.map('TButton',
                 background=[('active', KOLORY['accent2'])])
        
        style.configure('TLabel', 
                       font=('Helvetica', 11),
                       background=KOLORY['bg'],
                       foreground=KOLORY['dark'])
        
        style.configure('TFrame', background=KOLORY['bg'])
        style.configure('TLabelframe', 
                       background=KOLORY['bg'],
                       foreground=KOLORY['dark'],
                       font=('Helvetica', 11, 'bold'))
        style.configure('TLabelframe.Label', 
                       background=KOLORY['bg'],
                       foreground=KOLORY['dark'])
        
        # Główny kontener
        main_container = ttk.Frame(self.root, padding="15")
        main_container.pack(fill='both', expand=True)
        
        # Nagłówek
        header_frame = tk.Frame(main_container, bg=KOLORY['primary'], height=80)
        header_frame.pack(fill='x', pady=(0, 15))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, 
                              text="📊 ANALIZA PORÓWNAWCZA PACJENTÓW",
                              font=('Helvetica', 20, 'bold'),
                              bg=KOLORY['primary'],
                              fg='white')
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(header_frame,
                                 text="Przyjęci do szpitala vs Wypisani do domu",
                                 font=('Helvetica', 12),
                                 bg=KOLORY['primary'],
                                 fg='white')
        subtitle_label.pack(expand=True)
        
        # Notebook (zakładki)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True)
        
        # Tworzenie zakładek
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        self.tab5 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text="📂 WCZYTAJ DANE")
        self.notebook.add(self.tab2, text="📊 ANALIZA STATYSTYCZNA")
        self.notebook.add(self.tab3, text="📈 WYKRESY")
        self.notebook.add(self.tab4, text="📋 RAPORT")
        self.notebook.add(self.tab5, text="ℹ️ O PROGRAMIE")
        
        self.tab1_wczytaj()
        self.tab2_analiza()
        self.tab3_wykresy()
        self.tab4_raport()
        self.tab5_info()
    
    # =========================================================================
    # ZAKŁADKA 1 - WCZYTYWANIE DANYCH
    # =========================================================================
    def tab1_wczytaj(self):
        """Zakładka wczytywania danych"""
        # Główny frame
        main_frame = ttk.Frame(self.tab1, padding="30")
        main_frame.pack(fill='both', expand=True)
        
        # Ramka z przyciskami
        button_frame = tk.Frame(main_frame, bg=KOLORY['bg'])
        button_frame.pack(pady=50)
        
        # Styl przycisków
        button_style = {
            'font': ('Helvetica', 14, 'bold'),
            'bg': KOLORY['accent1'],
            'fg': 'white',
            'activebackground': KOLORY['accent2'],
            'activeforeground': 'white',
            'relief': 'flat',
            'bd': 0,
            'padx': 30,
            'pady': 15,
            'cursor': 'hand2'
        }
        
        btn_csv = tk.Button(button_frame, text="📁 WCZYTAJ PLIK CSV", 
                           command=self.wczytaj_csv, **button_style)
        btn_csv.pack(side='left', padx=20)
        
        btn_excel = tk.Button(button_frame, text="📗 WCZYTAJ PLIK EXCEL", 
                             command=self.wczytaj_excel, **button_style)
        btn_excel.pack(side='left', padx=20)
        
        # Efekt hover
        for btn in [btn_csv, btn_excel]:
            btn.bind('<Enter>', lambda e, b=btn: b.config(bg=KOLORY['accent2']))
            btn.bind('<Leave>', lambda e, b=btn: b.config(bg=KOLORY['accent1']))
        
        # Ramka informacyjna
        info_frame = tk.LabelFrame(main_frame, 
                                  text="📋 INFORMACJE O DANYCH",
                                  font=('Helvetica', 14, 'bold'),
                                  bg=KOLORY['light'],
                                  fg=KOLORY['dark'],
                                  relief='flat',
                                  bd=2,
                                  padx=20,
                                  pady=15)
        info_frame.pack(fill='both', expand=True, pady=30)
        
        # Tekst informacyjny z scrollbarem
        text_frame = tk.Frame(info_frame, bg=KOLORY['light'])
        text_frame.pack(fill='both', expand=True)
        
        self.info_text = tk.Text(text_frame,
                                 height=15,
                                 font=('Courier', 11),
                                 bg='white',
                                 fg=KOLORY['dark'],
                                 relief='flat',
                                 bd=1,
                                 padx=10,
                                 pady=10)
        self.info_text.pack(side='left', fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(text_frame, command=self.info_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.info_text.config(yscrollcommand=scrollbar.set)
        
        # Przycisk do analizy
        analyze_btn = tk.Button(main_frame,
                               text="🚀 PRZEJDŹ DO ANALIZY",
                               font=('Helvetica', 14, 'bold'),
                               bg=KOLORY['success'],
                               fg='white',
                               activebackground=KOLORY['accent2'],
                               activeforeground='white',
                               relief='flat',
                               bd=0,
                               padx=40,
                               pady=15,
                               cursor='hand2',
                               command=lambda: self.notebook.select(self.tab2))
        analyze_btn.pack(pady=20)
        analyze_btn.bind('<Enter>', lambda e: analyze_btn.config(bg=KOLORY['accent2']))
        analyze_btn.bind('<Leave>', lambda e: analyze_btn.config(bg=KOLORY['success']))
        
        # Powitalny tekst
        self.info_text.insert(1.0, """✨ Witaj w analizatorze danych medycznych!

Aby rozpocząć:
1. Kliknij przycisk "WCZYTAJ PLIK CSV" lub "WCZYTAJ PLIK EXCEL"
2. Wybierz plik z danymi pacjentów
3. Po wczytaniu przejdź do zakładki "ANALIZA STATYSTYCZNA"

Format pliku powinien zawierać:
- Pusty wiersz dzielący grupę przyjętych i wypisanych
- Kolumny z parametrami klinicznymi
- Kolumny z chorobami współistniejącymi

📊 Przyjęci do szpitala → górna część pliku
🏠 Wypisani do domu → dolna część pliku""")
    
    def wczytaj_csv(self):
        filename = filedialog.askopenfilename(
            title="Wybierz plik CSV",
            filetypes=[("Pliki CSV", "*.csv"), ("Wszystkie pliki", "*.*")]
        )
        if filename:
            try:
                self.df = pd.read_csv(filename, sep=';', encoding='utf-8')
                self.przetworz_dane()
                self.wyswietl_info(filename)
                messagebox.showinfo("✅ Sukces", 
                                  f"Plik wczytany poprawnie!\n\n"
                                  f"Liczba pacjentów: {len(self.df)}")
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Nie udało się wczytać pliku:\n{e}")
    
    def wczytaj_excel(self):
        filename = filedialog.askopenfilename(
            title="Wybierz plik Excel",
            filetypes=[("Pliki Excel", "*.xlsx *.xls"), ("Wszystkie pliki", "*.*")]
        )
        if filename:
            try:
                self.df = pd.read_excel(filename)
                self.przetworz_dane()
                self.wyswietl_info(filename)
                messagebox.showinfo("✅ Sukces", 
                                  f"Plik wczytany poprawnie!\n\n"
                                  f"Liczba pacjentów: {len(self.df)}")
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Nie udało się wczytać pliku:\n{e}")
    
    def przetworz_dane(self):
        """Przetwarza dane - dzieli na grupy"""
        if self.df is None:
            return
        
        # Znajdź pusty wiersz
        puste = self.df[self.df.isna().all(axis=1)]
        if len(puste) > 0:
            idx = puste.index[0]
            self.df_hosp = self.df.iloc[:idx].copy().dropna(how='all')
            self.df_dom = self.df.iloc[idx+1:].copy().dropna(how='all')
            
            # Konwersja na numeryczne
            for col in self.parametry_kliniczne:
                if col in self.df_hosp.columns:
                    self.df_hosp[col] = pd.to_numeric(
                        self.df_hosp[col].astype(str).str.replace(',', '.'), 
                        errors='coerce'
                    )
                if col in self.df_dom.columns:
                    self.df_dom[col] = pd.to_numeric(
                        self.df_dom[col].astype(str).str.replace(',', '.'), 
                        errors='coerce'
                    )
    
    def wyswietl_info(self, filename):
        """Wyświetla informacje o wczytanych danych"""
        self.info_text.delete(1.0, tk.END)
        
        info = f"""
╔══════════════════════════════════════════════════════════════╗
║                    INFORMACJE O DANYCH                       ║
╚══════════════════════════════════════════════════════════════╝

📁 PLIK: {os.path.basename(filename)}
📅 DATA: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 PODZIAŁ PACJENTÓW:
   • 🏥 PRZYJĘCI do szpitala: {len(self.df_hosp)} pacjentów
   • 🏠 WYPISANI do domu: {len(self.df_dom)} pacjentów
   • 👥 ŁĄCZNIE: {len(self.df)} pacjentów

📋 DOSTĘPNE PARAMETRY KLINICZNE:
"""
        for i, param in enumerate(self.parametry_kliniczne, 1):
            if param in self.df.columns:
                info += f"   {i:2d}. {param}\n"
        
        info += f"""
📊 STATYSTYKI OGÓLNE:
   • Liczba kolumn: {len(self.df.columns)}
   • Liczba wierszy: {len(self.df)}
   • Brakujące dane: {self.df.isna().sum().sum()}

✅ DANE GOTOWE DO ANALIZY!
Przejdź do zakładki "ANALIZA STATYSTYCZNA"
"""
        self.info_text.insert(1.0, info)
    
    # =========================================================================
    # ZAKŁADKA 2 - ANALIZA STATYSTYCZNA
    # =========================================================================
    def tab2_analiza(self):
        """Zakładka analizy statystycznej"""
        # Główny frame
        main_frame = ttk.Frame(self.tab2, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Panel kontrolny
        control_frame = tk.LabelFrame(main_frame,
                                     text="🎯 PARAMETRY ANALIZY",
                                     font=('Helvetica', 12, 'bold'),
                                     bg=KOLORY['light'],
                                     fg=KOLORY['dark'],
                                     relief='flat',
                                     bd=2,
                                     padx=20,
                                     pady=15)
        control_frame.pack(fill='x', pady=(0, 20))
        
        # Wybór parametru
        ttk.Label(control_frame, text="Wybierz parametr:",
                 font=('Helvetica', 11)).pack(side='left', padx=10)
        
        self.param_var = tk.StringVar()
        self.param_combo = ttk.Combobox(control_frame,
                                        textvariable=self.param_var,
                                        values=self.parametry_kliniczne,
                                        width=40,
                                        state='readonly',
                                        font=('Helvetica', 11))
        self.param_combo.pack(side='left', padx=10)
        
        # Przyciski analizy
        btn_frame = tk.Frame(control_frame, bg=KOLORY['light'])
        btn_frame.pack(side='right')
        
        analyze_one_btn = tk.Button(btn_frame,
                               text="📊 ANALIZUJ WYBRANY",
                               font=('Helvetica', 11, 'bold'),
                               bg=KOLORY['accent1'],
                               fg='white',
                               activebackground=KOLORY['accent2'],
                               activeforeground='white',
                               relief='flat',
                               bd=0,
                               padx=20,
                               pady=8,
                               cursor='hand2',
                               command=self.analizuj_pojedynczy)
        analyze_one_btn.pack(side='left', padx=5)
        analyze_one_btn.bind('<Enter>', lambda e: analyze_one_btn.config(bg=KOLORY['accent2']))
        analyze_one_btn.bind('<Leave>', lambda e: analyze_one_btn.config(bg=KOLORY['accent1']))
        
        analyze_all_btn = tk.Button(btn_frame,
                               text="📊 ANALIZUJ WSZYSTKIE",
                               font=('Helvetica', 11, 'bold'),
                               bg=KOLORY['success'],
                               fg='white',
                               activebackground=KOLORY['accent2'],
                               activeforeground='white',
                               relief='flat',
                               bd=0,
                               padx=20,
                               pady=8,
                               cursor='hand2',
                               command=self.analizuj_wszystkie)
        analyze_all_btn.pack(side='left', padx=5)
        analyze_all_btn.bind('<Enter>', lambda e: analyze_all_btn.config(bg=KOLORY['accent2']))
        analyze_all_btn.bind('<Leave>', lambda e: analyze_all_btn.config(bg=KOLORY['success']))
        
        # Tabela wyników
        table_frame = tk.LabelFrame(main_frame,
                                   text="📋 WYNIKI ANALIZY STATYSTYCZNEJ",
                                   font=('Helvetica', 12, 'bold'),
                                   bg=KOLORY['light'],
                                   fg=KOLORY['dark'],
                                   relief='flat',
                                   bd=2,
                                   padx=20,
                                   pady=15)
        table_frame.pack(fill='both', expand=True)
        
        # Treeview z scrollbarami
        tree_frame = tk.Frame(table_frame, bg=KOLORY['light'])
        tree_frame.pack(fill='both', expand=True)
        
        # Scrollbary
        vsb = tk.Scrollbar(tree_frame, orient='vertical')
        hsb = tk.Scrollbar(tree_frame, orient='horizontal')
        
        # Treeview
        columns = ('lp', 'parametr', 'hosp_n', 'hosp_sr', 'hosp_std',
                  'dom_n', 'dom_sr', 'dom_std', 'p', 'ist')
        
        self.tree = ttk.Treeview(tree_frame,
                                 columns=columns,
                                 show='headings',
                                 height=15,
                                 yscrollcommand=vsb.set,
                                 xscrollcommand=hsb.set)
        
        # Konfiguracja scrollbarów
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        
        # Nagłówki
        self.tree.heading('lp', text='LP')
        self.tree.heading('parametr', text='Parametr')
        self.tree.heading('hosp_n', text='n (hosp)')
        self.tree.heading('hosp_sr', text='Średnia hosp')
        self.tree.heading('hosp_std', text='SD hosp')
        self.tree.heading('dom_n', text='n (dom)')
        self.tree.heading('dom_sr', text='Średnia dom')
        self.tree.heading('dom_std', text='SD dom')
        self.tree.heading('p', text='p-value')
        self.tree.heading('ist', text='Ist.')
        
        # Szerokości kolumn
        self.tree.column('lp', width=50, anchor='center')
        self.tree.column('parametr', width=200)
        self.tree.column('hosp_n', width=70, anchor='center')
        self.tree.column('hosp_sr', width=100, anchor='center')
        self.tree.column('hosp_std', width=100, anchor='center')
        self.tree.column('dom_n', width=70, anchor='center')
        self.tree.column('dom_sr', width=100, anchor='center')
        self.tree.column('dom_std', width=100, anchor='center')
        self.tree.column('p', width=100, anchor='center')
        self.tree.column('ist', width=60, anchor='center')
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Kolorowanie wierszy
        self.tree.tag_configure('significant', background='#ffe6e6')
        self.tree.tag_configure('highly', background='#ffcccc')
        
        # Panel statystyk
        stats_frame = tk.Frame(main_frame, bg=KOLORY['light'], height=60)
        stats_frame.pack(fill='x', pady=(10, 0))
        
        self.stats_label = tk.Label(stats_frame,
                                   text="",
                                   font=('Helvetica', 11),
                                   bg=KOLORY['light'],
                                   fg=KOLORY['dark'])
        self.stats_label.pack(pady=10)
    
    def analizuj_pojedynczy(self):
        """Analizuje tylko wybrany parametr"""
        if self.df_hosp is None:
            messagebox.showwarning("⚠️ Uwaga", 
                                 "Najpierw wczytaj dane w zakładce 'WCZYTAJ DANE'!")
            return
        
        param = self.param_var.get()
        if not param:
            messagebox.showwarning("⚠️ Uwaga", "Wybierz parametr do analizy!")
            return
        
        # Wyczyść tabelę
        for row in self.tree.get_children():
            self.tree.delete(row)
        
        if param in self.df_hosp.columns and param in self.df_dom.columns:
            hosp = self.df_hosp[param].dropna()
            dom = self.df_dom[param].dropna()
            
            if len(hosp) > 0 and len(dom) > 0:
                hosp_sr = hosp.mean()
                hosp_std = hosp.std()
                dom_sr = dom.mean()
                dom_std = dom.std()
                
                stat, p = stats.mannwhitneyu(hosp, dom, alternative='two-sided')
                
                # Określenie istotności
                if p < 0.001:
                    stars = "***"
                    tag = 'highly'
                elif p < 0.01:
                    stars = "**"
                    tag = 'significant'
                elif p < 0.05:
                    stars = "*"
                    tag = 'significant'
                else:
                    stars = "ns"
                    tag = ''
                
                # Wstaw do tabeli
                self.tree.insert('', 'end', tags=(tag,), values=(
                    1,
                    param[:25],
                    len(hosp),
                    f"{hosp_sr:.2f}",
                    f"{hosp_std:.2f}",
                    len(dom),
                    f"{dom_sr:.2f}",
                    f"{dom_std:.2f}",
                    f"{p:.4f}",
                    stars
                ))
                
                self.stats_label.config(
                    text=f"✓ Przeanalizowano parametr: {param} • "
                         f"n(przyjęci)={len(hosp)} • n(wypisani)={len(dom)}"
                )
            else:
                messagebox.showwarning("⚠️ Uwaga", 
                                     f"Brak danych dla parametru {param}")
    
    def analizuj_wszystkie(self):
        """Przeprowadza analizę statystyczną wszystkich parametrów"""
        if self.df_hosp is None:
            messagebox.showwarning("⚠️ Uwaga", 
                                 "Najpierw wczytaj dane w zakładce 'WCZYTAJ DANE'!")
            return
        
        # Wyczyść tabelę
        for row in self.tree.get_children():
            self.tree.delete(row)
        
        wyniki = []
        
        for i, param in enumerate(self.parametry_kliniczne, 1):
            if param in self.df_hosp.columns and param in self.df_dom.columns:
                hosp = self.df_hosp[param].dropna()
                dom = self.df_dom[param].dropna()
                
                if len(hosp) > 0 and len(dom) > 0:
                    hosp_sr = hosp.mean()
                    hosp_std = hosp.std()
                    dom_sr = dom.mean()
                    dom_std = dom.std()
                    
                    stat, p = stats.mannwhitneyu(hosp, dom, alternative='two-sided')
                    
                    # Określenie istotności
                    if p < 0.001:
                        stars = "***"
                        tag = 'highly'
                    elif p < 0.01:
                        stars = "**"
                        tag = 'significant'
                    elif p < 0.05:
                        stars = "*"
                        tag = 'significant'
                    else:
                        stars = "ns"
                        tag = ''
                    
                    # Wstaw do tabeli
                    self.tree.insert('', 'end', tags=(tag,), values=(
                        i,
                        param[:25],
                        len(hosp),
                        f"{hosp_sr:.2f}",
                        f"{hosp_std:.2f}",
                        len(dom),
                        f"{dom_sr:.2f}",
                        f"{dom_std:.2f}",
                        f"{p:.4f}",
                        stars
                    ))
                    
                    wyniki.append({
                        'parametr': param,
                        'p_value': p,
                        'istotnosc': stars
                    })
        
        self.wyniki_df = pd.DataFrame(wyniki)
        istotne = sum(1 for w in wyniki if w['p_value'] < 0.05)
        wysoce = sum(1 for w in wyniki if w['p_value'] < 0.001)
        
        self.stats_label.config(
            text=f"✓ Przeanalizowano {len(wyniki)} parametrów • "
                 f"Istotne: {istotne} • Wysoce istotne: {wysoce} • "
                 f"n(przyjęci)={len(self.df_hosp)} • n(wypisani)={len(self.df_dom)}"
        )
        
        messagebox.showinfo("✅ Analiza zakończona",
                          f"Przeanalizowano {len(wyniki)} parametrów.\n"
                          f"Znaleziono {istotne} parametrów z istotnymi różnicami.")
    
    # =========================================================================
    # ZAKŁADKA 3 - WYKRESY
    # =========================================================================
    def tab3_wykresy(self):
        """Zakładka z wykresami"""
        # Główny frame
        main_frame = ttk.Frame(self.tab3, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Panel wyboru
        control_frame = tk.LabelFrame(main_frame,
                                     text="🎯 WYBIERZ PARAMETR DO WIZUALIZACJI",
                                     font=('Helvetica', 12, 'bold'),
                                     bg=KOLORY['light'],
                                     fg=KOLORY['dark'],
                                     relief='flat',
                                     bd=2,
                                     padx=20,
                                     pady=15)
        control_frame.pack(fill='x', pady=(0, 20))
        
        ttk.Label(control_frame, text="Parametr:",
                 font=('Helvetica', 11)).pack(side='left', padx=10)
        
        self.plot_param_var = tk.StringVar()
        self.plot_combo = ttk.Combobox(control_frame,
                                       textvariable=self.plot_param_var,
                                       values=self.parametry_kliniczne,
                                       width=40,
                                       state='readonly',
                                       font=('Helvetica', 11))
        self.plot_combo.pack(side='left', padx=10)
        
        # Przyciski
        btn_frame = tk.Frame(control_frame, bg=KOLORY['light'])
        btn_frame.pack(side='right')
        
        plot_btn = tk.Button(btn_frame,
                            text="📈 GENERUJ WYKRES",
                            font=('Helvetica', 11, 'bold'),
                            bg=KOLORY['accent1'],
                            fg='white',
                            activebackground=KOLORY['accent2'],
                            activeforeground='white',
                            relief='flat',
                            bd=0,
                            padx=20,
                            pady=8,
                            cursor='hand2',
                            command=self.rysuj_wykres)
        plot_btn.pack(side='left', padx=5)
        plot_btn.bind('<Enter>', lambda e: plot_btn.config(bg=KOLORY['accent2']))
        plot_btn.bind('<Leave>', lambda e: plot_btn.config(bg=KOLORY['accent1']))
        
        save_btn = tk.Button(btn_frame,
                            text="💾 ZAPISZ WYKRES",
                            font=('Helvetica', 11, 'bold'),
                            bg=KOLORY['success'],
                            fg='white',
                            activebackground=KOLORY['accent2'],
                            activeforeground='white',
                            relief='flat',
                            bd=0,
                            padx=20,
                            pady=8,
                            cursor='hand2',
                            command=self.zapisz_wykres)
        save_btn.pack(side='left', padx=5)
        save_btn.bind('<Enter>', lambda e: save_btn.config(bg=KOLORY['accent2']))
        save_btn.bind('<Leave>', lambda e: save_btn.config(bg=KOLORY['success']))
        
        # Ramka wykresu
        plot_frame = tk.LabelFrame(main_frame,
                                  text="📊 WYKRES",
                                  font=('Helvetica', 12, 'bold'),
                                  bg=KOLORY['light'],
                                  fg=KOLORY['dark'],
                                  relief='flat',
                                  bd=2,
                                  padx=20,
                                  pady=15)
        plot_frame.pack(fill='both', expand=True)
        
        # Figure i canvas
        self.figure = Figure(figsize=(12, 7), dpi=100, facecolor='white')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#f8f9fa')
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Toolbar
        toolbar_frame = tk.Frame(plot_frame, bg=KOLORY['light'])
        toolbar_frame.pack(fill='x')
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
    
    def rysuj_wykres(self):
        """Rysuje wykres dla wybranego parametru"""
        param = self.plot_param_var.get()
        if not param:
            messagebox.showwarning("⚠️ Uwaga", "Wybierz parametr do wizualizacji!")
            return
        
        if self.df_hosp is None:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj dane!")
            return
        
        hosp = self.df_hosp[param].dropna()
        dom = self.df_dom[param].dropna()
        
        if len(hosp) == 0 or len(dom) == 0:
            messagebox.showwarning("⚠️ Uwaga", f"Brak danych dla parametru {param}")
            return
        
        self.current_param = param
        
        # Wyczyść osie
        self.ax.clear()
        
        # Test statystyczny
        stat, p = stats.mannwhitneyu(hosp, dom, alternative='two-sided')
        
        # Wybór typu wykresu dla parametrów z ekstremami
        if param in ['troponina I (0-7,8))', 'crp(0-0,5)']:
            # Wykres w skali logarytmicznej
            bp = self.ax.boxplot([hosp, dom],
                                labels=['PRZYJĘCI', 'WYPISANI'],
                                patch_artist=True,
                                medianprops={'color': 'black', 'linewidth': 2})
            
            self.ax.set_yscale('log')
            self.ax.set_ylabel(f'{param} (skala log)', fontsize=11)
            
        elif param == 'kreatynina(0,5-1,2)':
            # Wykres z normami
            bp = self.ax.boxplot([hosp, dom],
                                labels=['PRZYJĘCI', 'WYPISANI'],
                                patch_artist=True,
                                medianprops={'color': 'black', 'linewidth': 2})
            
            self.ax.axhline(y=1.2, color='red', linestyle='--', alpha=0.7, label='Górna norma')
            self.ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Dolna norma')
            self.ax.legend(loc='upper right')
            self.ax.set_ylabel(param, fontsize=11)
            
        else:
            # Standardowy wykres
            bp = self.ax.boxplot([hosp, dom],
                                labels=['PRZYJĘCI', 'WYPISANI'],
                                patch_artist=True,
                                medianprops={'color': 'black', 'linewidth': 2})
            self.ax.set_ylabel(param, fontsize=11)
        
        # Kolorowanie
        bp['boxes'][0].set_facecolor(KOLORY['hosp'])
        bp['boxes'][0].set_alpha(0.8)
        bp['boxes'][1].set_facecolor(KOLORY['dom'])
        bp['boxes'][1].set_alpha(0.8)
        
        # Dodanie punktów
        x_hosp = np.random.normal(1, 0.05, len(hosp))
        x_dom = np.random.normal(2, 0.05, len(dom))
        self.ax.scatter(x_hosp, hosp, alpha=0.5, color='darkred', s=30, zorder=3)
        self.ax.scatter(x_dom, dom, alpha=0.5, color='darkblue', s=30, zorder=3)
        
        # Tytuł z p-value
        if p < 0.001:
            title = f'{param}\np < 0.001 ***'
        elif p < 0.01:
            title = f'{param}\np = {p:.4f} **'
        elif p < 0.05:
            title = f'{param}\np = {p:.4f} *'
        else:
            title = f'{param}\np = {p:.4f} (ns)'
        
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        # Statystyki opisowe
        stats_text = (f"Przyjęci:\n"
                     f"n = {len(hosp)}\n"
                     f"śr = {hosp.mean():.2f} ± {hosp.std():.2f}\n"
                     f"mediana = {np.median(hosp):.2f}\n\n"
                     f"Wypisani:\n"
                     f"n = {len(dom)}\n"
                     f"śr = {dom.mean():.2f} ± {dom.std():.2f}\n"
                     f"mediana = {np.median(dom):.2f}")
        
        self.ax.text(0.02, 0.98, stats_text,
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    fontsize=9)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def zapisz_wykres(self):
        """Zapisuje aktualny wykres do pliku"""
        if self.current_param is None:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wygeneruj wykres!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile=f'wykres_{self.current_param}.png'
        )
        
        if filename:
            try:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("✅ Sukces", f"Wykres zapisany jako:\n{filename}")
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Nie udało się zapisać pliku:\n{e}")
    
    # =========================================================================
    # ZAKŁADKA 4 - RAPORT
    # =========================================================================
    def tab4_raport(self):
        """Zakładka z raportem końcowym"""
        # Główny frame
        main_frame = ttk.Frame(self.tab4, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Panel przycisków
        btn_frame = tk.Frame(main_frame, bg=KOLORY['light'])
        btn_frame.pack(fill='x', pady=(0, 20))
        
        # Styl przycisków
        btn_style = {
            'font': ('Helvetica', 12, 'bold'),
            'fg': 'white',
            'activebackground': KOLORY['accent2'],
            'activeforeground': 'white',
            'relief': 'flat',
            'bd': 0,
            'padx': 25,
            'pady': 12,
            'cursor': 'hand2'
        }
        
        btn1 = tk.Button(btn_frame, text="📊 GENERUJ RAPORT", 
                        bg=KOLORY['accent1'], **btn_style)
        btn1.config(command=self.generuj_raport)
        btn1.pack(side='left', padx=10)
        btn1.bind('<Enter>', lambda e: btn1.config(bg=KOLORY['accent2']))
        btn1.bind('<Leave>', lambda e: btn1.config(bg=KOLORY['accent1']))
        
        btn2 = tk.Button(btn_frame, text="💾 EKSPORTUJ DO CSV", 
                        bg=KOLORY['success'], **btn_style)
        btn2.config(command=self.export_csv)
        btn2.pack(side='left', padx=10)
        btn2.bind('<Enter>', lambda e: btn2.config(bg=KOLORY['accent2']))
        btn2.bind('<Leave>', lambda e: btn2.config(bg=KOLORY['success']))
        
        btn3 = tk.Button(btn_frame, text="🔄 ODŚWIEŻ", 
                        bg=KOLORY['warning'], **btn_style)
        btn3.config(command=self.odswiez_raport)
        btn3.pack(side='left', padx=10)
        btn3.bind('<Enter>', lambda e: btn3.config(bg=KOLORY['accent2']))
        btn3.bind('<Leave>', lambda e: btn3.config(bg=KOLORY['warning']))
        
        # Ramka raportu
        report_frame = tk.LabelFrame(main_frame,
                                    text="📋 RAPORT KOŃCOWY",
                                    font=('Helvetica', 14, 'bold'),
                                    bg=KOLORY['light'],
                                    fg=KOLORY['dark'],
                                    relief='flat',
                                    bd=2,
                                    padx=20,
                                    pady=15)
        report_frame.pack(fill='both', expand=True)
        
        # Tekst raportu z scrollbarem
        text_frame = tk.Frame(report_frame, bg=KOLORY['light'])
        text_frame.pack(fill='both', expand=True)
        
        self.report_text = tk.Text(text_frame,
                                   font=('Courier', 11),
                                   bg='white',
                                   fg=KOLORY['dark'],
                                   wrap='word',
                                   padx=15,
                                   pady=15)
        self.report_text.pack(side='left', fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(text_frame, command=self.report_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.report_text.config(yscrollcommand=scrollbar.set)
    
    def generuj_raport(self):
        """Generuje raport końcowy"""
        if self.df_hosp is None:
            messagebox.showwarning("⚠️ Uwaga", "Brak danych do wygenerowania raportu!")
            return
        
        self.odswiez_raport()
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Pliki tekstowe", "*.txt"), ("PDF", "*.pdf")],
            initialfile="raport_medyczny.txt"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.report_text.get(1.0, tk.END))
                messagebox.showinfo("✅ Sukces", f"Raport zapisany jako:\n{filename}")
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Nie udało się zapisać pliku:\n{e}")
    
    def odswiez_raport(self):
        """Odświeża raport"""
        self.report_text.delete(1.0, tk.END)
        
        if self.df_hosp is None:
            self.report_text.insert(1.0, "⚡ Brak danych. Wczytaj plik w zakładce 'WCZYTAJ DANE'.")
            return
        
        # Obliczenia dla raportu
        wyniki = []
        istotne = []
        
        for param in self.parametry_kliniczne:
            if param in self.df_hosp.columns and param in self.df_dom.columns:
                hosp = self.df_hosp[param].dropna()
                dom = self.df_dom[param].dropna()
                
                if len(hosp) > 0 and len(dom) > 0:
                    stat, p = stats.mannwhitneyu(hosp, dom, alternative='two-sided')
                    roznica = hosp.mean() - dom.mean()
                    
                    wyniki.append({
                        'parametr': param,
                        'hosp_sr': hosp.mean(),
                        'dom_sr': dom.mean(),
                        'p': p,
                        'roznica': roznica
                    })
                    
                    if p < 0.05:
                        istotne.append((param, p, roznica))
        
        istotne.sort(key=lambda x: x[1])
        
        # Generuj raport
        raport = f"""
╔══════════════════════════════════════════════════════════════════╗
║              RAPORT KOŃCOWY ANALIZY MEDYCZNEJ                    ║
╚══════════════════════════════════════════════════════════════════╝

📅 Data raportu: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

📊 PODSUMOWANIE DANYCH:
────────────────────────────────────────────────────────────────────
  • 🏥 Przyjęci do szpitala: {len(self.df_hosp)} pacjentów
  • 🏠 Wypisani do domu: {len(self.df_dom)} pacjentów
  • 👥 Łącznie: {len(self.df_hosp) + len(self.df_dom)} pacjentów

📈 ISTOTNOŚĆ STATYSTYCZNA:
────────────────────────────────────────────────────────────────────
  • Parametry istotne (p < 0.05): {len(istotne)}
  • Parametry wysoce istotne (p < 0.001): {len([i for i in istotne if i[1] < 0.001])}

🔬 TOP 5 NAJBARDZIEJ ISTOTNYCH RÓŻNIC:
────────────────────────────────────────────────────────────────────
"""
        for i, (param, p, roznica) in enumerate(istotne[:5], 1):
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            else:
                stars = "*"
            
            kierunek = "⬆️ WYŻSZE" if roznica > 0 else "⬇️ NIŻSZE"
            raport += f"\n  {i}. {param:<25}\n"
            raport += f"     p = {p:.6f} {stars}\n"
            raport += f"     {kierunek} u przyjętych (różnica: {roznica:+.2f})\n"
        
        raport += f"""
{'='*70}
✅ ANALIZA ZAKOŃCZONA POMYŚLNIE
{'='*70}
"""
        self.report_text.insert(1.0, raport)
    
    def export_csv(self):
        """Eksportuje wyniki do CSV"""
        if self.wyniki_df is None:
            messagebox.showwarning("⚠️ Uwaga", 
                                 "Najpierw wykonaj analizę w zakładce 'ANALIZA STATYSTYCZNA'!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="wyniki_analizy.csv"
        )
        
        if filename:
            try:
                self.wyniki_df.to_csv(filename, sep=';', index=False, encoding='utf-8')
                messagebox.showinfo("✅ Sukces", f"Wyniki zapisane jako:\n{filename}")
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Nie udało się zapisać pliku:\n{e}")
    
    # =========================================================================
    # ZAKŁADKA 5 - O PROGRAMIE
    # =========================================================================
    def tab5_info(self):
        """Zakładka informacyjna"""
        frame = ttk.Frame(self.tab5, padding="30")
        frame.pack(fill='both', expand=True)
        
        info_text = """
╔══════════════════════════════════════════════════════════════╗
║         ANALIZATOR DANYCH MEDYCZNYCH - WERSJA 9.0           ║
╚══════════════════════════════════════════════════════════════╝

📋 OPIS PROGRAMU:
──────────────────────────────────────────────────────────────
Program służy do porównawczej analizy danych medycznych
pomiędzy pacjentami przyjętymi do szpitala a wypisanymi do domu.

📊 FUNKCJONALNOŚCI:
──────────────────────────────────────────────────────────────
• Wczytywanie danych z plików CSV i Excel
• Automatyczny podział na grupy (przyjęci/wypisani)
• Analiza statystyczna (test Manna-Whitneya)
• Wizualizacja wyników (wykresy pudełkowe)
• Generowanie raportów końcowych
• Eksport wyników do CSV

📈 PARAMETRY KLINICZNE:
──────────────────────────────────────────────────────────────
• wiek, RR, MAP, SpO2, AS, mleczany
• kreatynina(0,5-1,2), troponina I (0-7,8))
• HGB(12,4-15,2), WBC(4-11), plt(130-450)
• hct(38-45), Na(137-145), K(3,5-5,1), crp(0-0,5)

🩺 CHOROBY WSPÓŁISTNIEJĄCE:
──────────────────────────────────────────────────────────────
• dm (cukrzyca)
• wątroba (choroby wątroby)
• naczyniowe (choroby naczyniowe)
• zza (zawał)
• npl (nowotwory)

👩‍⚕️ AUTOR:
──────────────────────────────────────────────────────────────
Aneta
Wersja: 9.0 (Marzec 2026)
"""
        label = tk.Label(frame, text=info_text,
                        font=('Courier', 11),
                        bg='white',
                        fg=KOLORY['dark'],
                        justify='left',
                        padx=30,
                        pady=30)
        label.pack(fill='both', expand=True)


# =============================================================================
# URUCHOMIENIE
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalAnalyzerGUI(root)
    root.mainloop()