# -*- coding: utf-8 -*-
"""
EFEKTOWNE GUI - ANALIZA DANYCH MEDYCZNYCH
Wersja: 8.0 - czytelny interfejs
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import stats
from math import pi
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GŁÓWNA KLASA APLIKACJI
# =============================================================================
class MedicalAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 ANALIZA DANYCH MEDYCZNYCH - PRZYJĘCI vs ODESŁANI")
        self.root.geometry("1400x900")
        
        # Zmienne
        self.df = None
        self.df_hosp = None
        self.df_dom = None
        self.wyniki_df = None
        self.current_param = None
        
        # Listy parametrów
        self.parametry_kliniczne = [
            'wiek', 'RR', 'MAP', 'SpO2', 'AS', 'mleczany',
            'kreatynina(0,5-1,2)', 'troponina I (0-7,8))',
            'HGB(12,4-15,2)', 'WBC(4-11)', 'plt(130-450)',
            'hct(38-45)', 'Na(137-145)', 'K(3,5-5,1)', 'crp(0-0,5)'
        ]
        
        self.choroby = ['dm', 'wątroba', 'naczyniowe', 'zza', 'npl']
        
        # Kolory
        self.kolory = {
            'przyjeci': '#e74c3c',      # czerwony
            'odeslani': '#3498db',       # niebieski
            'tlo': '#f8f9fa',
            'tekst': '#2c3e50',
            'przycisk': '#2ecc71'
        }
        
        # Styl
        plt.style.use('ggplot')
        sns.set_palette("Set2")
        
        self.setup_gui()
    
    # =========================================================================
    # INTERFEJS UŻYTKOWNIKA
    # =========================================================================
    def setup_gui(self):
        """Tworzy interfejs z zakładkami"""
        # Styl
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook.Tab', font=('Arial', 11, 'bold'), padding=[20, 5])
        style.configure('TButton', font=('Arial', 10), padding=10)
        style.configure('TLabel', font=('Arial', 11), background=self.kolory['tlo'])
        
        # Główny kontener
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill='both', expand=True)
        
        # Nagłówek
        header = tk.Label(main_container, 
                         text="📈 ANALIZA PORÓWNAWCZA - PACJENCI PRZYJĘCI vs ODESŁANI DO DOMU",
                         font=('Arial', 16, 'bold'),
                         bg=self.kolory['tlo'],
                         fg=self.kolory['tekst'],
                         pady=15)
        header.pack(fill='x')
        
        # Notebook (zakładki)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True, pady=10)
        
        # Zakładki
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text="📂 WCZYTAJ DANE")
        self.notebook.add(self.tab2, text="📊 ANALIZA STATYSTYCZNA")
        self.notebook.add(self.tab3, text="📈 WYKRESY INDYWIDUALNE")
        self.notebook.add(self.tab4, text="📋 RAPORT KOŃCOWY")
        
        self.tab1_wczytaj()
        self.tab2_analiza()
        self.tab3_wykresy()
        self.tab4_raport()
    
    # =========================================================================
    # ZAKŁADKA 1 - WCZYTYWANIE DANYCH
    # =========================================================================
    def tab1_wczytaj(self):
        """Zakładka wczytywania danych"""
        frame = ttk.LabelFrame(self.tab1, text="📂 WCZYTAJ DANE", padding=30)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Przyciski
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=30)
        
        ttk.Button(btn_frame, text="📁 Wybierz plik CSV", 
                  command=self.wczytaj_csv,
                  width=25).pack(side='left', padx=10)
        
        ttk.Button(btn_frame, text="📗 Wybierz plik Excel", 
                  command=self.wczytaj_excel,
                  width=25).pack(side='left', padx=10)
        
        # Ramka informacyjna
        info_frame = tk.LabelFrame(frame, text="INFORMACJE O DANYCH", 
                                   font=('Arial', 12, 'bold'),
                                   bg=self.kolory['tlo'],
                                   fg=self.kolory['tekst'],
                                   padx=20, pady=15)
        info_frame.pack(fill='both', expand=True, pady=20)
        
        self.info_text = tk.Text(info_frame, height=15, width=80,
                                  font=('Courier', 11),
                                  bg='white',
                                  relief='solid',
                                  borderwidth=1)
        self.info_text.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.info_text.config(yscrollcommand=scrollbar.set)
        
        # Przycisk do analizy
        ttk.Button(frame, text="🚀 PRZEJDŹ DO ANALIZY", 
                  command=lambda: self.notebook.select(self.tab2),
                  style='Accent.TButton').pack(pady=20)
    
    def wczytaj_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if filename:
            try:
                self.df = pd.read_csv(filename, sep=';', encoding='utf-8')
                self.przetworz_dane()
                self.wyswietl_info()
                messagebox.showinfo("✅ Sukces", "Dane wczytane poprawnie!")
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Nie wczytano: {e}")
    
    def wczytaj_excel(self):
        filename = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])
        if filename:
            try:
                self.df = pd.read_excel(filename)
                self.przetworz_dane()
                self.wyswietl_info()
                messagebox.showinfo("✅ Sukces", "Dane wczytane poprawnie!")
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Nie wczytano: {e}")
    
    def przetworz_dane(self):
        """Dzieli dane na grupy"""
        if self.df is None:
            return
        
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
                        errors='coerce')
                if col in self.df_dom.columns:
                    self.df_dom[col] = pd.to_numeric(
                        self.df_dom[col].astype(str).str.replace(',', '.'), 
                        errors='coerce')
    
    def wyswietl_info(self):
        """Wyświetla informacje o danych"""
        self.info_text.delete(1.0, tk.END)
        
        info = f"""
╔══════════════════════════════════════════════════════════════╗
║                    INFORMACJE O DANYCH                       ║
╚══════════════════════════════════════════════════════════════╝

📊 PODZIAŁ PACJENTÓW:
   • Przyjęci do szpitala: {len(self.df_hosp)} pacjentów
   • Odesłani do domu: {len(self.df_dom)} pacjentów
   • Razem: {len(self.df)} pacjentów

📋 DOSTĘPNE PARAMETRY KLINICZNE:
"""
        for i, param in enumerate(self.parametry_kliniczne, 1):
            info += f"   {i:2d}. {param}\n"
        
        info += f"""
📁 NAZWA PLIKU: {os.path.basename(self.df) if hasattr(self.df, 'name') else 'wczytany z pamięci'}
📅 DATA ANALIZY: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

▶️ Przejdź do zakładki "ANALIZA STATYSTYCZNA"
"""
        self.info_text.insert(1.0, info)
    
    # =========================================================================
    # ZAKŁADKA 2 - ANALIZA STATYSTYCZNA
    # =========================================================================
    def tab2_analiza(self):
        """Zakładka analizy statystycznej"""
        # Panel kontrolny
        control_frame = ttk.LabelFrame(self.tab2, text="🎯 PARAMETRY ANALIZY", padding=15)
        control_frame.pack(fill='x', padx=15, pady=10)
        
        ttk.Label(control_frame, text="Wybierz parametr:", 
                 font=('Arial', 11, 'bold')).pack(side='left', padx=10)
        
        self.param_var = tk.StringVar()
        self.param_combo = ttk.Combobox(control_frame, 
                                        textvariable=self.param_var,
                                        values=self.parametry_kliniczne,
                                        width=40,
                                        state='readonly',
                                        font=('Arial', 11))
        self.param_combo.pack(side='left', padx=10)
        
        ttk.Button(control_frame, text="📊 ANALIZUJ WSZYSTKIE", 
                  command=self.analizuj_wszystkie,
                  style='Accent.TButton').pack(side='right', padx=10)
        
        # Tabela wyników
        table_frame = ttk.LabelFrame(self.tab2, text="📋 WYNIKI ANALIZY STATYSTYCZNEJ", padding=15)
        table_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        # Treeview
        columns = ('lp', 'parametr', 'hosp_n', 'hosp_sr', 'hosp_std', 
                  'dom_n', 'dom_sr', 'dom_std', 'p_value', 'istotnosc')
        
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Nagłówki
        self.tree.heading('lp', text='LP')
        self.tree.heading('parametr', text='Parametr')
        self.tree.heading('hosp_n', text='n (przyj)')
        self.tree.heading('hosp_sr', text='Średnia (przyj)')
        self.tree.heading('hosp_std', text='SD (przyj)')
        self.tree.heading('dom_n', text='n (odesł)')
        self.tree.heading('dom_sr', text='Średnia (odesł)')
        self.tree.heading('dom_std', text='SD (odesł)')
        self.tree.heading('p_value', text='p-value')
        self.tree.heading('istotnosc', text='Istotność')
        
        # Szerokości
        self.tree.column('lp', width=50)
        self.tree.column('parametr', width=200)
        self.tree.column('hosp_n', width=70)
        self.tree.column('hosp_sr', width=100)
        self.tree.column('hosp_std', width=100)
        self.tree.column('dom_n', width=70)
        self.tree.column('dom_sr', width=100)
        self.tree.column('dom_std', width=100)
        self.tree.column('p_value', width=100)
        self.tree.column('istotnosc', width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Statystyki podsumowujące
        summary_frame = ttk.Frame(self.tab2)
        summary_frame.pack(fill='x', padx=15, pady=5)
        
        self.summary_label = tk.Label(summary_frame, 
                                      text="", 
                                      font=('Arial', 11),
                                      bg=self.kolory['tlo'],
                                      fg=self.kolory['tekst'])
        self.summary_label.pack()
    
    def analizuj_wszystkie(self):
        """Analizuje wszystkie parametry"""
        if self.df_hosp is None:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj dane!")
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
                    
                    if p < 0.001:
                        stars = "***"
                    elif p < 0.01:
                        stars = "**"
                    elif p < 0.05:
                        stars = "*"
                    else:
                        stars = "ns"
                    
                    self.tree.insert('', 'end', values=(
                        i,
                        param[:20],
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
        
        self.summary_label.config(
            text=f"✓ Przeanalizowano {len(wyniki)} parametrów • "
                 f"Istotne statystycznie: {istotne} • "
                 f"n(przyjęci)={len(self.df_hosp)} • n(odesłani)={len(self.df_dom)}"
        )
    
    # =========================================================================
    # ZAKŁADKA 3 - WYKRESY INDYWIDUALNE
    # =========================================================================
    def tab3_wykresy(self):
        """Zakładka z wykresami"""
        # Panel wyboru
        control_frame = ttk.LabelFrame(self.tab3, text="🎯 WYBIERZ PARAMETR", padding=15)
        control_frame.pack(fill='x', padx=15, pady=10)
        
        ttk.Label(control_frame, text="Parametr do wizualizacji:", 
                 font=('Arial', 11, 'bold')).pack(side='left', padx=10)
        
        self.plot_param_var = tk.StringVar()
        self.plot_combo = ttk.Combobox(control_frame, 
                                       textvariable=self.plot_param_var,
                                       values=self.parametry_kliniczne,
                                       width=40,
                                       state='readonly',
                                       font=('Arial', 11))
        self.plot_combo.pack(side='left', padx=10)
        
        ttk.Button(control_frame, text="📈 GENERUJ WYKRES", 
                  command=self.rysuj_wykres,
                  style='Accent.TButton').pack(side='right', padx=10)
        
        # Miejsce na wykres
        plot_frame = ttk.LabelFrame(self.tab3, text="📊 WYKRES PORÓWNAWCZY", padding=15)
        plot_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        self.figure = plt.Figure(figsize=(12, 7), dpi=100, facecolor='#f8f9fa')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#ffffff')
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill='x')
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Przycisk zapisu
        ttk.Button(plot_frame, text="💾 ZAPISZ WYKRES", 
                  command=self.zapisz_wykres).pack(pady=5)
    
    def rysuj_wykres(self):
        """Rysuje wykres dla wybranego parametru"""
        param = self.plot_param_var.get()
        if not param or self.df_hosp is None:
            return
        
        self.current_param = param
        
        hosp = self.df_hosp[param].dropna()
        dom = self.df_dom[param].dropna()
        
        if len(hosp) == 0 or len(dom) == 0:
            messagebox.showwarning("⚠️ Uwaga", "Brak danych dla tego parametru")
            return
        
        self.ax.clear()
        
        # Test statystyczny
        stat, p = stats.mannwhitneyu(hosp, dom, alternative='two-sided')
        
        # Wykres pudełkowy
        bp = self.ax.boxplot([hosp, dom], 
                            labels=['PRZYJĘCI', 'ODESŁANI'],
                            patch_artist=True,
                            medianprops={'color': 'black', 'linewidth': 2},
                            boxprops={'linewidth': 2},
                            whiskerprops={'linewidth': 1.5})
        
        bp['boxes'][0].set_facecolor(self.kolory['przyjeci'])
        bp['boxes'][0].set_alpha(0.8)
        bp['boxes'][1].set_facecolor(self.kolory['odeslani'])
        bp['boxes'][1].set_alpha(0.8)
        
        # Dodanie punktów
        x_hosp = np.random.normal(1, 0.05, len(hosp))
        x_dom = np.random.normal(2, 0.05, len(dom))
        self.ax.scatter(x_hosp, hosp, alpha=0.6, color='darkred', s=50, zorder=3)
        self.ax.scatter(x_dom, dom, alpha=0.6, color='darkblue', s=50, zorder=3)
        
        # Tytuł z p-value
        if p < 0.001:
            title = f'{param}\np < 0.001 ***'
        elif p < 0.01:
            title = f'{param}\np = {p:.4f} **'
        elif p < 0.05:
            title = f'{param}\np = {p:.4f} *'
        else:
            title = f'{param}\np = {p:.4f} (ns)'
        
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        self.ax.set_ylabel('Wartość', fontsize=12)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        # Dodanie statystyk
        text = (f"Przyjęci:\nn={len(hosp)}\nśr={hosp.mean():.2f}±{hosp.std():.2f}\n\n"
                f"Odesłani:\nn={len(dom)}\nśr={dom.mean():.2f}±{dom.std():.2f}")
        
        self.ax.text(0.02, 0.98, text,
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def zapisz_wykres(self):
        """Zapisuje aktualny wykres"""
        if self.current_param:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")],
                initialfile=f'wykres_{self.current_param}.png'
            )
            if filename:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("✅ Sukces", f"Wykres zapisany jako:\n{filename}")
    
    # =========================================================================
    # ZAKŁADKA 4 - RAPORT KOŃCOWY
    # =========================================================================
    def tab4_raport(self):
        """Zakładka z raportem końcowym"""
        frame = ttk.LabelFrame(self.tab4, text="📋 PODSUMOWANIE ANALIZY", padding=20)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Przyciski akcji
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="📊 GENERUJ RAPORT PDF", 
                  command=self.generuj_raport,
                  width=20).pack(side='left', padx=10)
        
        ttk.Button(btn_frame, text="💾 EKSPORTUJ DO CSV", 
                  command=self.export_csv,
                  width=20).pack(side='left', padx=10)
        
        ttk.Button(btn_frame, text="🗑️ WYCZYŚĆ DANE", 
                  command=self.czysc_dane,
                  width=20).pack(side='left', padx=10)
        
        # Raport tekstowy
        report_frame = tk.LabelFrame(frame, text="RAPORT KOŃCOWY", 
                                     font=('Arial', 12, 'bold'),
                                     bg=self.kolory['tlo'],
                                     padx=20, pady=15)
        report_frame.pack(fill='both', expand=True, pady=20)
        
        self.report_text = tk.Text(report_frame, height=20, width=80,
                                   font=('Courier', 11),
                                   bg='white',
                                   relief='solid',
                                   borderwidth=1)
        self.report_text.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(report_frame, command=self.report_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.report_text.config(yscrollcommand=scrollbar.set)
        
        self.aktualizuj_raport()
    
    def aktualizuj_raport(self):
        """Aktualizuje raport końcowy"""
        self.report_text.delete(1.0, tk.END)
        
        if self.df_hosp is None:
            self.report_text.insert(1.0, "⚡ Brak danych. Wczytaj plik w zakładce 1.")
            return
        
        # Obliczenia dla raportu
        istotne_param = []
        for param in self.parametry_kliniczne:
            if param in self.df_hosp.columns and param in self.df_dom.columns:
                hosp = self.df_hosp[param].dropna()
                dom = self.df_dom[param].dropna()
                if len(hosp) > 0 and len(dom) > 0:
                    stat, p = stats.mannwhitneyu(hosp, dom, alternative='two-sided')
                    if p < 0.05:
                        istotne_param.append((param, p))
        
        istotne_param.sort(key=lambda x: x[1])
        
        # Generuj raport
        raport = f"""
╔══════════════════════════════════════════════════════════════════╗
║              RAPORT KOŃCOWY ANALIZY MEDYCZNEJ                    ║
╚══════════════════════════════════════════════════════════════════╝

📊 PODSUMOWANIE DANYCH:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Liczba pacjentów PRZYJĘTYCH: {len(self.df_hosp)}
  • Liczba pacjentów ODESŁANYCH: {len(self.df_dom)}
  • Łączna liczba pacjentów: {len(self.df_hosp) + len(self.df_dom)}
  • Data analizy: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

📈 ISTOTNOŚĆ STATYSTYCZNA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Parametry istotne statystycznie (p < 0.05): {len(istotne_param)}
  • Parametry wysoce istotne (p < 0.001): {sum(1 for _, p in istotne_param if p < 0.001)}

🔬 TOP 5 NAJBARDZIEJ ISTOTNYCH RÓŻNIC:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for i, (param, p) in enumerate(istotne_param[:5], 1):
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            else:
                stars = "*"
            raport += f"  {i}. {param:<30} p = {p:.6f} {stars}\n"
        
        raport += f"""
📋 PARAMETRY BEZ ISTOTNYCH RÓŻNIC:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        nieistotne = [p for p in self.parametry_kliniczne 
                     if p not in [x[0] for x in istotne_param]]
        for param in nieistotne[:8]:
            raport += f"  • {param}\n"
        
        if len(nieistotne) > 8:
            raport += f"  • ... i {len(nieistotne)-8} innych\n"
        
        raport += """
✅ ANALIZA ZAKOŃCZONA POMYŚLNIE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        self.report_text.insert(1.0, raport)
    
    def generuj_raport(self):
        """Generuje raport PDF (symulacja)"""
        if self.df_hosp is None:
            messagebox.showwarning("⚠️ Uwaga", "Brak danych do raportu!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile="raport_medyczny.txt"
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.report_text.get(1.0, tk.END))
            messagebox.showinfo("✅ Sukces", f"Raport zapisany jako:\n{filename}")
    
    def export_csv(self):
        """Eksportuje wyniki do CSV"""
        if self.wyniki_df is None:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wykonaj analizę!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="wyniki_analizy.csv"
        )
        
        if filename:
            self.wyniki_df.to_csv(filename, sep=';', index=False, encoding='utf-8')
            messagebox.showinfo("✅ Sukces", f"Wyniki zapisane jako:\n{filename}")
    
    def czysc_dane(self):
        """Czyści wszystkie dane"""
        self.df = None
        self.df_hosp = None
        self.df_dom = None
        self.wyniki_df = None
        self.current_param = None
        
        for row in self.tree.get_children():
            self.tree.delete(row)
        
        self.summary_label.config(text="")
        self.ax.clear()
        self.canvas.draw()
        self.report_text.delete(1.0, tk.END)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, "⚡ Dane wyczyszczone. Wczytaj nowy plik.")
        
        messagebox.showinfo("✅ Gotowe", "Dane zostały wyczyszczone")


# =============================================================================
# URUCHOMIENIE
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalAnalyzerGUI(root)
    root.mainloop()