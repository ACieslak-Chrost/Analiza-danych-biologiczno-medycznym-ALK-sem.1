# -*- coding: utf-8 -*-
"""
PRZEPIEKNE GUI - ANALIZA DANYCH MEDYCZNYCH
Wersja: 11.2 - Z kalkulatorem ryzyka hospitalizacji (poprawiony)
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
from scipy.stats import fisher_exact
from math import pi, exp
import os
import warnings
from datetime import datetime

# Importy dla wersji profesjonalnej
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_curve, auc, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

warnings.filterwarnings('ignore')

# =============================================================================
# USTAWIENIA STYLU
# =============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

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
    'fg': '#2c3e50',
    'risk_low': '#2ecc71',
    'risk_medium': '#f39c12',
    'risk_high': '#e74c3c'
}

# =============================================================================
# KONFIGURACJA DLA WERSJI PROFESJONALNEJ
# =============================================================================
PARAMETRY_KLINICZNE = [
    "wiek", "RR", "MAP", "SpO2", "AS", "mleczany",
    "kreatynina(0,5-1,2)", "troponina I (0-7,8))",
    "HGB(12,4-15,2)", "WBC(4-11)", "plt(130-450)",
    "hct(38-45)", "Na(137-145)", "K(3,5-5,1)", "crp(0-0,5)"
]

CHOROBY = ["dm", "wątroba", "naczyniowe", "zza", "npl"]

ZMIENNE_OBOWIAZKOWE = ["wiek"]
ZMIENNE_DODATKOWE = [
    "SpO2",
    "crp(0-0,5)",
    "kreatynina(0,5-1,2)",
    "RR_skurczowe",
    "RR_rozkurczowe",
    "troponina I (0-7,8))",
    "HGB(12,4-15,2)"
]

ZMIENNE_LOG = [
    "crp(0-0,5)",
    "troponina I (0-7,8))",
    "kreatynina(0,5-1,2)"
]

ZAKRESY_BIOLOGICZNE = {
    "wiek": (0, 120),
    "RR": (0, 300),
    "MAP": (0, 200),
    "SpO2": (0, 100),
    "AS": (0, 300),
    "mleczany": (0, 30),
    "kreatynina(0,5-1,2)": (0, 20),
    "troponina I (0-7,8))": (0, 100000),
    "HGB(12,4-15,2)": (0, 25),
    "WBC(4-11)": (0, 100),
    "plt(130-450)": (0, 2000),
    "hct(38-45)": (0, 70),
    "Na(137-145)": (100, 160),
    "K(3,5-5,1)": (2, 8),
    "crp(0-0,5)": (0, 500),
    "RR_skurczowe": (0, 300),
    "RR_rozkurczowe": (0, 200)
}

ETYKIETY = {
    "wiek": "Wiek, lata",
    "RR": "Częstość oddechów / min",
    "MAP": "Średnie ciśnienie tętnicze, mmHg",
    "SpO2": "Saturacja, %",
    "AS": "Akcja serca / min",
    "mleczany": "Mleczany, mmol/L",
    "kreatynina(0,5-1,2)": "Kreatynina, mg/dL",
    "troponina I (0-7,8))": "Troponina I",
    "HGB(12,4-15,2)": "Hemoglobina, g/dL",
    "WBC(4-11)": "Leukocyty, G/L",
    "plt(130-450)": "Płytki, G/L",
    "hct(38-45)": "Hematokryt, %",
    "Na(137-145)": "Sód, mmol/L",
    "K(3,5-5,1)": "Potas, mmol/L",
    "crp(0-0,5)": "CRP, mg/dL",
    "dm": "Cukrzyca",
    "wątroba": "Choroba wątroby",
    "naczyniowe": "Choroby naczyniowe",
    "zza": "Zespół zależności alkoholowej",
    "npl": "Nowotwór / choroba proliferacyjna",
    "log_crp(0,0-0,5)": "log(CRP)",
    "log_crp(0-0,5)": "log(CRP)",
    "log_kreatynina(0,5-1,2)": "log(kreatynina)",
    "log_troponina I (0-7,8))": "log(troponina I)",
    "RR_skurczowe": "RR skurczowe, mmHg",
    "RR_rozkurczowe": "RR rozkurczowe, mmHg"
}

# =============================================================================
# FUNKCJE POMOCNICZE DLA WERSJI PROFESJONALNEJ
# =============================================================================

def pretty_name(x: str) -> str:
    return ETYKIETY.get(x, x)

def cliff_delta(x: pd.Series, y: pd.Series) -> float:
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0
    try:
        u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        return (2 * u_stat) / (n1 * n2) - 1
    except Exception:
        return 0.0

def interpret_cliff_delta(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "mały"
    if ad < 0.33:
        return "umiarkowany"
    return "duży"

def sprawdz_epv_i_raport(df: pd.DataFrame, zmienne: list[str], outcome: str = "outcome", prog: int = 10):
    n_events = int(df[outcome].sum())
    n_vars = len(zmienne)
    epv = n_events / n_vars if n_vars > 0 else 0
    return epv >= prog, epv

def sprawdz_vif(X: pd.DataFrame):
    vif_data = pd.DataFrame()
    vif_data["zmienna"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def _wyniki_modelu_statsmodels(model, zmienne: list[str]) -> pd.DataFrame:
    rows = []
    for var in zmienne:
        ci = model.conf_int().loc[var]
        rows.append({
            "parametr": var,
            "etykieta": pretty_name(var),
            "beta": model.params[var],
            "OR": np.exp(model.params[var]),
            "ci_low": np.exp(ci[0]),
            "ci_high": np.exp(ci[1]),
            "CI_95%": f"{np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}",
            "p_value": model.pvalues[var]
        })
    return pd.DataFrame(rows)

def oblicz_MAP(skurczowe, rozkurczowe):
    """Oblicza MAP na podstawie RR skurczowego i rozkurczowego"""
    try:
        return (float(skurczowe) + 2 * float(rozkurczowe)) / 2
    except:
        return None

# =============================================================================
# GŁÓWNA KLASA APLIKACJI
# =============================================================================
class MedicalAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ANALIZATOR DANYCH MEDYCZNYCH")
        self.root.geometry("1400x900")
        self.root.configure(bg=KOLORY['bg'])
        
        # Zmienne
        self.df = None
        self.df_hosp = None
        self.df_dom = None
        self.wyniki_df = None
        self.current_figure = None
        self.current_param = None
        self.current_mode = "podstawowa"
        self.pro_btn = None
        
        # Zmienne dla kalkulatora
        self.model_predykcyjny_model = None
        self.scaler = None
        self.zmienne_modelu = None
        self.entry_vars = {}
        self.disease_vars = {}
        
        # Listy parametrów
        self.parametry_kliniczne = PARAMETRY_KLINICZNE
        self.choroby = CHOROBY
        
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
                              text="ANALIZA PORÓWNAWCZA PACJENTÓW",
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
        
        # Wybór trybu analizy
        mode_frame = tk.Frame(main_container, bg=KOLORY['light'], height=50)
        mode_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(mode_frame, text="TRYB ANALIZY:", 
                font=('Helvetica', 12, 'bold'),
                bg=KOLORY['light'],
                fg=KOLORY['dark']).pack(side='left', padx=20, pady=10)
        
        self.mode_var = tk.StringVar(value="podstawowa")
        
        mode_basic = tk.Radiobutton(mode_frame, text="PODSTAWOWA (szybka analiza)",
                                   variable=self.mode_var, value="podstawowa",
                                   font=('Helvetica', 11),
                                   bg=KOLORY['light'],
                                   fg=KOLORY['dark'],
                                   selectcolor=KOLORY['light'],
                                   command=self.zmien_tryb)
        mode_basic.pack(side='left', padx=10, pady=10)
        
        mode_pro = tk.Radiobutton(mode_frame, text="PROFESJONALNA (pełna analiza publikacyjna)",
                                 variable=self.mode_var, value="profesjonalna",
                                 font=('Helvetica', 11),
                                 bg=KOLORY['light'],
                                 fg=KOLORY['dark'],
                                 selectcolor=KOLORY['light'],
                                 command=self.zmien_tryb)
        mode_pro.pack(side='left', padx=10, pady=10)
        
        # Notebook (zakładki)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True)
        
        # Tworzenie zakładek
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        self.tab5 = ttk.Frame(self.notebook)
        self.tab6 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text=" WCZYTAJ DANE")
        self.notebook.add(self.tab2, text=" ANALIZA STATYSTYCZNA")
        self.notebook.add(self.tab3, text=" WYKRESY")
        self.notebook.add(self.tab4, text=" RAPORT")
        self.notebook.add(self.tab5, text=" O PROGRAMIE")
        self.notebook.add(self.tab6, text=" KALKULATOR RYZYKA")
        
        self.tab1_wczytaj()
        self.tab2_analiza()
        self.tab3_wykresy()
        self.tab4_raport()
        self.tab5_info()
        self.tab6_kalkulator()
        
        # Inicjalizacja - dodaj przycisk jeśli tryb profesjonalny
        if self.current_mode == "profesjonalna":
            self.root.after(100, self.dodaj_przycisk_profesjonalny)
    
    def zmien_tryb(self):
        """Zmienia tryb analizy"""
        self.current_mode = self.mode_var.get()
        if self.current_mode == "profesjonalna":
            self.dodaj_przycisk_profesjonalny()
            messagebox.showinfo("Tryb profesjonalny", 
                              "Wybrano tryb PROFESJONALNY - pełna analiza publikacyjna\n\n"
                              "RÓŻNICE W STOSUNKU DO TRYBU PODSTAWOWEGO:\n"
                              "• Tabela 1 (charakterystyka kohorty z interpretacją Cliff's delta)\n"
                              "• Analiza jednoczynnikowa z korektą FDR\n"
                              "• Trzy modele regresji logistycznej (podstawowy, rozszerzony, z redukcją)\n"
                              "• Forest plot dla OR z 95% CI\n"
                              "• Model predykcyjny z walidacją krzyżową i bootstrapem\n"
                              "• Progi kliniczne z uwzględnieniem kierunku efektu\n"
                              "• Raport z ograniczeniami (complete-case, EPV, VIF)\n"
                              "• Pliki wynikowe gotowe do publikacji")
        else:
            self.usun_przycisk_profesjonalny()
            messagebox.showinfo("Tryb podstawowy", 
                              "Wybrano tryb PODSTAWOWY - szybka analiza porównawcza\n\n"
                              "ZAKRES ANALIZY:\n"
                              "• Podstawowe statystyki opisowe\n"
                              "• Test Manna-Whitneya\n"
                              "• Wykresy pudełkowe\n"
                              "• Prosty raport końcowy")
    
    def dodaj_przycisk_profesjonalny(self):
        """Dodaje przycisk analizy profesjonalnej"""
        try:
            self.usun_przycisk_profesjonalny()
            
            for child in self.tab2.winfo_children():
                if isinstance(child, tk.LabelFrame) and child.cget('text') == "PARAMETRY ANALIZY":
                    control_frame = child
                    break
            else:
                return
            
            for child in control_frame.winfo_children():
                if isinstance(child, tk.Frame):
                    btn_frame = child
                    break
            else:
                return
            
            self.pro_btn = tk.Button(btn_frame,
                                   text="ANALIZA PROFESJONALNA",
                                   font=('Helvetica', 11, 'bold'),
                                   bg=KOLORY['warning'],
                                   fg='white',
                                   activebackground=KOLORY['accent2'],
                                   activeforeground='white',
                                   relief='flat',
                                   bd=0,
                                   padx=20,
                                   pady=8,
                                   cursor='hand2',
                                   command=self.analiza_profesjonalna)
            self.pro_btn.pack(side='left', padx=5)
            self.pro_btn.bind('<Enter>', lambda e: self.pro_btn.config(bg=KOLORY['accent2']))
            self.pro_btn.bind('<Leave>', lambda e: self.pro_btn.config(bg=KOLORY['warning']))
        except Exception as e:
            print(f"Błąd dodawania przycisku: {e}")
    
    def usun_przycisk_profesjonalny(self):
        """Usuwa przycisk analizy profesjonalnej"""
        try:
            if self.pro_btn is not None:
                self.pro_btn.destroy()
                self.pro_btn = None
        except:
            pass
    
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
        
        btn_csv = tk.Button(button_frame, text=" WCZYTAJ PLIK CSV", 
                           command=self.wczytaj_csv, **button_style)
        btn_csv.pack(side='left', padx=20)
        
        btn_excel = tk.Button(button_frame, text=" WCZYTAJ PLIK EXCEL", 
                             command=self.wczytaj_excel, **button_style)
        btn_excel.pack(side='left', padx=20)
        
        # Efekt hover
        for btn in [btn_csv, btn_excel]:
            btn.bind('<Enter>', lambda e, b=btn: b.config(bg=KOLORY['accent2']))
            btn.bind('<Leave>', lambda e, b=btn: b.config(bg=KOLORY['accent1']))
        
        # Ramka informacyjna
        info_frame = tk.LabelFrame(main_frame, 
                                  text=" INFORMACJE O DANYCH",
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
                               text=" PRZEJDŹ DO ANALIZY",
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
        self.info_text.insert(1.0, """Witaj w analizatorze danych medycznych!

Aby rozpocząć:
1. Wybierz tryb analizy (podstawowa lub profesjonalna)
2. Kliknij przycisk "WCZYTAJ PLIK CSV" lub "WCZYTAJ PLIK EXCEL"
3. Wybierz plik z danymi pacjentów (wymagana kolumna 'outcome')
4. Po wczytaniu przejdź do zakładki "ANALIZA STATYSTYCZNA"

 Tryb PODSTAWOWY - szybka analiza porównawcza
 Tryb PROFESJONALNY - pełna analiza publikacyjna""")
    
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
                messagebox.showinfo(" Sukces", 
                                  f"Plik wczytany poprawnie!\n\n"
                                  f"Liczba pacjentów: {len(self.df)}")
            except Exception as e:
                messagebox.showerror(" Błąd", f"Nie udało się wczytać pliku:\n{e}")
    
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
                messagebox.showinfo(" Sukces", 
                                  f"Plik wczytany poprawnie!\n\n"
                                  f"Liczba pacjentów: {len(self.df)}")
            except Exception as e:
                messagebox.showerror(" Błąd", f"Nie udało się wczytać pliku:\n{e}")
    
    def przetworz_dane(self):
        """Przetwarza dane - dzieli na grupy"""
        if self.df is None:
            return
        
        # Sprawdź czy jest kolumna outcome
        if 'outcome' not in self.df.columns:
            # Spróbuj znaleźć pusty wiersz (stary format)
            puste = self.df[self.df.isna().all(axis=1)]
            if len(puste) > 0:
                idx = puste.index[0]
                df_hosp = self.df.iloc[:idx].copy().dropna(how='all')
                df_dom = self.df.iloc[idx+1:].copy().dropna(how='all')
                
                # Dodaj outcome
                df_hosp['outcome'] = 1
                df_dom['outcome'] = 0
                self.df = pd.concat([df_hosp, df_dom], ignore_index=True)
        
        # Teraz już powinna być kolumna outcome
        self.df = self.df[self.df['outcome'].notna()].copy()
        self.df['outcome'] = pd.to_numeric(self.df['outcome'], errors='coerce')
        self.df = self.df[self.df['outcome'].isin([0, 1])].copy()
        
        # Jeśli są kolumny RR_skurczowe i RR_rozkurczowe, oblicz MAP
        if 'RR_skurczowe' in self.df.columns and 'RR_rozkurczowe' in self.df.columns:
            self.df['MAP'] = (self.df['RR_skurczowe'] + 2 * self.df['RR_rozkurczowe']) / 2
        
        self.df_hosp = self.df[self.df['outcome'] == 1].copy()
        self.df_dom = self.df[self.df['outcome'] == 0].copy()
        
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
 INFORMACJE O DANYCH

 PLIK: {os.path.basename(filename)}
 DATA: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
 TRYB: {"PROFESJONALNY" if self.current_mode == "profesjonalna" else "PODSTAWOWY"}

 PODZIAŁ PACJENTÓW:
   • PRZYJĘCI do szpitala: {len(self.df_hosp)} pacjentów
   • WYPISANI do domu: {len(self.df_dom)} pacjentów
   • ŁĄCZNIE: {len(self.df)} pacjentów

 DOSTĘPNE PARAMETRY KLINICZNE:
"""
        for i, param in enumerate(self.parametry_kliniczne, 1):
            if param in self.df.columns:
                info += f"   {i:2d}. {param}\n"
        
        info += f"""
 STATYSTYKI OGÓLNE:
   • Liczba kolumn: {len(self.df.columns)}
   • Liczba wierszy: {len(self.df)}

 DANE GOTOWE DO ANALIZY!
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
                                     text="PARAMETRY ANALIZY",
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
                               text=" ANALIZUJ WYBRANY",
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
                               text=" ANALIZUJ WSZYSTKIE",
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
                                   text=" WYNIKI ANALIZY STATYSTYCZNEJ",
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
            messagebox.showwarning(" Uwaga", 
                                 "Najpierw wczytaj dane w zakładce 'WCZYTAJ DANE'!")
            return
        
        param = self.param_var.get()
        if not param:
            messagebox.showwarning(" Uwaga", "Wybierz parametr do analizy!")
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
                    text=f" Przeanalizowano parametr: {param} • "
                         f"n(przyjęci)={len(hosp)} • n(wypisani)={len(dom)}"
                )
            else:
                messagebox.showwarning(" Uwaga", 
                                     f"Brak danych dla parametru {param}")
    
    def analizuj_wszystkie(self):
        """Przeprowadza analizę statystyczną wszystkich parametrów"""
        if self.df_hosp is None:
            messagebox.showwarning(" Uwaga", 
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
            text=f" Przeanalizowano {len(wyniki)} parametrów • "
                 f"Istotne: {istotne} • Wysoce istotne: {wysoce} • "
                 f"n(przyjęci)={len(self.df_hosp)} • n(wypisani)={len(self.df_dom)}"
        )
        
        messagebox.showinfo(" Analiza zakończona",
                          f"Przeanalizowano {len(wyniki)} parametrów.\n"
                          f"Znaleziono {istotne} parametrów z istotnymi różnicami.")
    
    def analiza_profesjonalna(self):
        """Pełna analiza profesjonalna (wersja publikacyjna)"""
        if self.df_hosp is None:
            messagebox.showwarning(" Uwaga", 
                                 "Najpierw wczytaj dane w zakładce 'WCZYTAJ DANE'!")
            return
        
        messagebox.showinfo(" Analiza profesjonalna", 
                          "Rozpoczynam pełną analizę publikacyjną.\n"
                          "Proszę czekać... To może potrwać kilka chwil.")
        
        try:
            # Przygotuj dane
            df_caly = pd.concat([self.df_hosp, self.df_dom])
            
            # Konwersja chorób
            mapping_tak = {"tak", "t", "yes", "y", "1", "true", "+", "tak!"}
            mapping_nie = {"nie", "n", "no", "0", "false", "-"}
            
            for col in CHOROBY:
                if col in df_caly.columns:
                    tmp = df_caly[col].astype(str).str.lower().str.strip()
                    df_caly[col] = tmp.apply(lambda x: 1 if x in mapping_tak else (0 if x in mapping_nie else np.nan))
            
            # Raport braków
            raport_brakow_df = self._raport_brakow_pro(df_caly)
            raport_brakow_df.to_csv("raport_brakow.csv", sep=";", index=False)
            
            # Walidacja zakresów
            walidacja_df = self._walidacja_zakresow(df_caly)
            walidacja_df.to_csv("walidacja_zakresow.csv", sep=";", index=False)
            
            # Tabela 1
            tabela1 = self._tabela_1_pro(df_caly)
            tabela1.to_csv("tabela_1_publikacyjna.csv", sep=";", index=False)
            
            # Analiza jednoczynnikowa
            wyniki_fdr, top5 = self._analiza_jednoczynnikowa_pro(df_caly)
            wyniki_fdr.to_csv("analiza_jednoczynnikowa_fdr.csv", sep=";", index=False)
            
            # Missingness top 5
            missing_top = self._missingness_top_pro(top5)
            missing_top.to_csv("missingness_top5.csv", sep=";", index=False)
            
            # Progi kliniczne
            progi = self._progi_kliniczne_pro(df_caly, top5)
            progi.to_csv("progi_kliniczne_eksploracyjne.csv", sep=";", index=False)
            
            # Przygotuj zmienne do modelu
            df_model, zmienne_modelu = self._przygotuj_zmienne_modelu(df_caly)
            self.zmienne_modelu = zmienne_modelu
            
            # Modele inferencyjne
            model1, wyn1, n1 = self._model_podstawowy(df_model)
            model2, wyn2, n2, epv_ok, vif2 = self._model_rozszerzony(df_model, zmienne_modelu)
            model3, wyn3, n3, _, vif3 = self._model_z_redukcja(df_model, zmienne_modelu)
            
            if wyn1 is not None:
                wyn1.to_csv("model_podstawowy.csv", sep=";", index=False)
            
            if wyn2 is not None:
                wyn2.to_csv("model_glowny_rozszerzony.csv", sep=";", index=False)
                self._forest_plot(wyn2, "forest_plot_model_glowny.png")
            
            if wyn3 is not None:
                wyn3.to_csv("model_redukowany_sensitivity.csv", sep=";", index=False)
                self._forest_plot(wyn3, "forest_plot_model_redukowany.png")
            
            if vif2 is not None:
                vif2.to_csv("vif_model_glowny.csv", sep=";", index=False)
            if vif3 is not None:
                vif3.to_csv("vif_model_redukowany.csv", sep=";", index=False)
            
            # Model predykcyjny
            pred = self._model_predykcyjny(df_model, zmienne_modelu)
            if pred is not None:
                pred['fig_roc'].savefig("krzywa_ROC.png", dpi=300, bbox_inches="tight")
                pred['fig_cal'].savefig("krzywa_kalibracji.png", dpi=300, bbox_inches="tight")
                plt.close("all")
                
                self.model_predykcyjny_model = pred['pipeline']
            
            # Raport tekstowy
            raport_txt = self._generuj_raport_tekstowy(top5, wyn2, pred, epv_ok)
            with open("raport_wyniki_i_ograniczenia.txt", "w", encoding="utf-8") as f:
                f.write(raport_txt)
            
            # Aktualizuj informację w kalkulatorze
            if hasattr(self, 'model_info_label'):
                self.model_info_label.config(text=" Model gotowy do użycia! Wprowadź dane pacjenta.")
            
            messagebox.showinfo(" Sukces", 
                              "Analiza profesjonalna zakończona!\n\n"
                              "Wygenerowane pliki:\n"
                              "• tabela_1_publikacyjna.csv\n"
                              "• analiza_jednoczynnikowa_fdr.csv\n"
                              "• modele regresji\n"
                              "• forest plot\n"
                              "• krzywa ROC\n"
                              "• raport końcowy\n\n"
                              "Model zapisany - możesz teraz korzystać z kalkulatora ryzyka!")
            
        except Exception as e:
            messagebox.showerror(" Błąd", f"Wystąpił błąd podczas analizy:\n{str(e)}")
    
    # =========================================================================
    # FUNKCJE POMOCNICZE DLA ANALIZY PROFESJONALNEJ
    # =========================================================================
    
    def _raport_brakow_pro(self, df):
        wyniki = []
        for col in df.columns:
            n_brakow = int(df[col].isna().sum())
            proc = (n_brakow / len(df)) * 100 if len(df) > 0 else 0
            wyniki.append({
                "kolumna": col,
                "braki": n_brakow,
                "procent": round(proc, 2)
            })
        return pd.DataFrame(wyniki)
    
    def _walidacja_zakresow(self, df):
        wyniki = []
        for col, (min_bio, max_bio) in ZAKRESY_BIOLOGICZNE.items():
            if col in df.columns:
                dane = df[col].dropna()
                if len(dane) > 0:
                    mask = (dane < min_bio) | (dane > max_bio)
                    n_bad = int(mask.sum())
                    wyniki.append({
                        "kolumna": col,
                        "poza_zakresem": n_bad,
                        "min_bio": min_bio,
                        "max_bio": max_bio
                    })
        return pd.DataFrame(wyniki)
    
    def _tabela_1_pro(self, df):
        wyniki = []
        
        for param in PARAMETRY_KLINICZNE:
            if param in df.columns:
                hosp = df[df['outcome'] == 1][param].dropna()
                dom = df[df['outcome'] == 0][param].dropna()
                
                if len(hosp) > 0 and len(dom) > 0:
                    p = stats.mannwhitneyu(hosp, dom).pvalue
                    d = cliff_delta(hosp, dom)
                    
                    wyniki.append({
                        "parametr": param,
                        "etykieta": pretty_name(param),
                        "hospitalizowani": f"{hosp.median():.2f} [{hosp.quantile(0.25):.2f}-{hosp.quantile(0.75):.2f}]",
                        "wypisani": f"{dom.median():.2f} [{dom.quantile(0.25):.2f}-{dom.quantile(0.75):.2f}]",
                        "p_value": p,
                        "effect_size": d,
                        "interpretacja": interpret_cliff_delta(d)
                    })
        
        for choroba in CHOROBY:
            if choroba in df.columns:
                hosp = df[df['outcome'] == 1][choroba].dropna()
                dom = df[df['outcome'] == 0][choroba].dropna()
                
                if len(hosp) > 0 and len(dom) > 0:
                    hosp_tak = int((hosp == 1).sum())
                    dom_tak = int((dom == 1).sum())
                    
                    tabela = [[hosp_tak, len(hosp) - hosp_tak], [dom_tak, len(dom) - dom_tak]]
                    _, p = fisher_exact(tabela)
                    
                    a = hosp_tak + 0.5
                    b = len(hosp) - hosp_tak + 0.5
                    c = dom_tak + 0.5
                    d = len(dom) - dom_tak + 0.5
                    or_val = (a * d) / (b * c)
                    
                    wyniki.append({
                        "parametr": choroba,
                        "etykieta": pretty_name(choroba),
                        "hospitalizowani": f"{hosp_tak}/{len(hosp)} ({100*hosp_tak/len(hosp):.1f}%)",
                        "wypisani": f"{dom_tak}/{len(dom)} ({100*dom_tak/len(dom):.1f}%)",
                        "p_value": p,
                        "effect_size": or_val,
                        "interpretacja": "OR"
                    })
        
        return pd.DataFrame(wyniki)
    
    def _analiza_jednoczynnikowa_pro(self, df):
        wyniki = []
        p_values = []
        
        for param in PARAMETRY_KLINICZNE:
            if param in df.columns:
                hosp = df[df['outcome'] == 1][param].dropna()
                dom = df[df['outcome'] == 0][param].dropna()
                
                if len(hosp) > 0 and len(dom) > 0:
                    p = stats.mannwhitneyu(hosp, dom).pvalue
                    d = cliff_delta(hosp, dom)
                    
                    wyniki.append({
                        "parametr": param,
                        "etykieta": pretty_name(param),
                        "p_raw": p,
                        "cliff_delta": d,
                        "interpretacja": interpret_cliff_delta(d),
                        "n_hosp": len(hosp),
                        "n_dom": len(dom)
                    })
                    p_values.append(p)
        
        df_wyniki = pd.DataFrame(wyniki)
        if len(df_wyniki) == 0:
            return df_wyniki, []
        
        _, p_fdr, _, _ = multipletests(p_values, method="fdr_bh")
        df_wyniki["p_fdr"] = p_fdr
        df_wyniki["istotny_fdr"] = df_wyniki["p_fdr"] < 0.05
        df_wyniki = df_wyniki.sort_values(["p_fdr", "p_raw"]).reset_index(drop=True)
        
        top5 = df_wyniki[df_wyniki["istotny_fdr"]].head(5)["parametr"].tolist()
        if len(top5) < 5:
            top5 = df_wyniki.head(5)["parametr"].tolist()
        
        return df_wyniki, top5
    
    def _missingness_top_pro(self, top_param):
        wyniki = []
        for param in top_param[:5]:
            if param in self.df_hosp.columns and param in self.df_dom.columns:
                b1 = int(self.df_hosp[param].isna().sum())
                b0 = int(self.df_dom[param].isna().sum())
                wyniki.append({
                    "parametr": param,
                    "etykieta": pretty_name(param),
                    "braki_hosp": b1,
                    "proc_hosp": round(100 * b1 / len(self.df_hosp), 2) if len(self.df_hosp) else 0,
                    "braki_dom": b0,
                    "proc_dom": round(100 * b0 / len(self.df_dom), 2) if len(self.df_dom) else 0
                })
        return pd.DataFrame(wyniki)
    
    def _progi_kliniczne_pro(self, df, top_param):
        wyniki = []
        
        for param in top_param[:5]:
            if param not in df.columns:
                continue
            
            dane = df[[param, "outcome"]].dropna()
            if len(dane) < 10:
                continue
            
            hosp_med = dane[dane["outcome"] == 1][param].median()
            dom_med = dane[dane["outcome"] == 0][param].median()
            kierunek = "wyższe" if hosp_med > dom_med else "niższe"
            
            try:
                if kierunek == "wyższe":
                    fpr, tpr, thresholds = roc_curve(dane["outcome"], dane[param])
                else:
                    fpr, tpr, thresholds = roc_curve(dane["outcome"], -dane[param])
                
                youden = tpr - fpr
                idx = int(np.argmax(youden))
                
                if kierunek == "wyższe":
                    prog = thresholds[idx]
                    y_pred = (dane[param] >= prog).astype(int)
                else:
                    prog = -thresholds[idx]
                    y_pred = (dane[param] <= prog).astype(int)
                
                tn = int(((y_pred == 0) & (dane["outcome"] == 0)).sum())
                fp = int(((y_pred == 1) & (dane["outcome"] == 0)).sum())
                fn = int(((y_pred == 0) & (dane["outcome"] == 1)).sum())
                tp = int(((y_pred == 1) & (dane["outcome"] == 1)).sum())
                
                sens = tp / (tp + fn) if (tp + fn) else 0
                spec = tn / (tn + fp) if (tn + fp) else 0
                
                wyniki.append({
                    "parametr": param,
                    "etykieta": pretty_name(param),
                    "kierunek": kierunek,
                    "prog": prog,
                    "czulosc": sens,
                    "swoistosc": spec
                })
            except:
                continue
        
        return pd.DataFrame(wyniki)
    
    def _przygotuj_zmienne_modelu(self, df):
        df_model = df.copy()
        wszystkie = ZMIENNE_OBOWIAZKOWE + ZMIENNE_DODATKOWE
        dostepne = [z for z in wszystkie if z in df_model.columns]
        
        for z in ZMIENNE_LOG:
            if z in df_model.columns:
                new_name = f"log_{z}"
                df_model[new_name] = np.log1p(df_model[z].clip(lower=0))
                if z in dostepne:
                    dostepne.remove(z)
                    dostepne.append(new_name)
        
        return df_model, dostepne
    
    def _model_podstawowy(self, df):
        if "wiek" not in df.columns:
            return None, None, None
        
        df_cc = df[["wiek", "outcome"]].dropna()
        if len(df_cc) < 10:
            return None, None, None
        
        X = sm.add_constant(df_cc["wiek"])
        y = df_cc["outcome"]
        
        try:
            model = sm.Logit(y, X).fit(disp=0)
            wyn = _wyniki_modelu_statsmodels(model, ["wiek"])
            return model, wyn, len(df_cc)
        except:
            return None, None, None
    
    def _model_rozszerzony(self, df, zmienne):
        dostepne = [z for z in zmienne if z in df.columns]
        df_cc = df[dostepne + ["outcome"]].dropna()
        
        if len(df_cc) < 10:
            return None, None, None, False, None
        
        epv_ok, _ = sprawdz_epv_i_raport(df_cc, dostepne)
        
        X = sm.add_constant(df_cc[dostepne])
        y = df_cc["outcome"]
        
        try:
            model = sm.Logit(y, X).fit(disp=0, maxiter=100)
            vif = sprawdz_vif(X)
            wyn = _wyniki_modelu_statsmodels(model, dostepne)
            return model, wyn, len(df_cc), epv_ok, vif
        except:
            return None, None, None, False, None
    
    def _model_z_redukcja(self, df, zmienne):
        dostepne = [z for z in zmienne if z in df.columns]
        df_cc = df[dostepne + ["outcome"]].dropna()
        n_events = int(df_cc["outcome"].sum())
        max_pred = int(n_events / 10)
        
        if max_pred < 1:
            return None, None, None, False, None
        
        if len(dostepne) <= max_pred:
            return self._model_rozszerzony(df, zmienne)
        
        priorytety = {
            "wiek": 10,
            "log_crp(0-0,5)": 9,
            "SpO2": 8,
            "log_kreatynina(0,5-1,2)": 7,
            "RR_skurczowe": 6,
            "RR_rozkurczowe": 5,
            "log_troponina I (0-7,8))": 4,
            "HGB(12,4-15,2)": 3
        }
        
        dostepne = sorted(dostepne, key=lambda x: priorytety.get(x, 0), reverse=True)
        wybrane = dostepne[:max_pred]
        
        return self._model_rozszerzony(df, wybrane)
    
    def _forest_plot(self, wyniki, nazwa_pliku):
        if wyniki is None or len(wyniki) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(wyniki))
        
        ax.errorbar(
            wyniki["OR"], y_pos,
            xerr=[wyniki["OR"] - wyniki["ci_low"], wyniki["ci_high"] - wyniki["OR"]],
            fmt="o", capsize=4
        )
        ax.axvline(1, linestyle="--")
        ax.set_xscale("log")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(wyniki["etykieta"])
        ax.set_xlabel("OR (95% CI)")
        ax.set_title("Niezależne czynniki związane z hospitalizacją")
        plt.tight_layout()
        plt.savefig(nazwa_pliku, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _model_predykcyjny(self, df, zmienne):
        dostepne = [z for z in zmienne if z in df.columns]
        df_cc = df[dostepne + ["outcome"]].dropna()
        
        if len(df_cc) < 20:
            return None
        
        X = df_cc[dostepne].values
        y = df_cc["outcome"].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        brier = brier_score_loss(y_test, y_prob)
        
        auc_boot = []
        for i in range(1000):
            idx = resample(range(len(y_test)), replace=True, random_state=i)
            if len(np.unique(y_test[idx])) < 2:
                continue
            auc_boot.append(roc_auc_score(y_test[idx], y_prob[idx]))
        
        auc_ci = (
            np.percentile(auc_boot, 2.5),
            np.percentile(auc_boot, 97.5)
        ) if len(auc_boot) else (roc_auc, roc_auc)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        
        fig1, ax1 = plt.subplots()
        ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax1.plot([0, 1], [0, 1], "k--")
        ax1.set_xlabel("1 - swoistość")
        ax1.set_ylabel("Czułość")
        ax1.set_title("Krzywa ROC")
        ax1.legend()
        
        fig2, ax2 = plt.subplots()
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        ax2.plot(prob_pred, prob_true, marker="o")
        ax2.plot([0, 1], [0, 1], "k--")
        ax2.set_xlabel("Prawdopodobieństwo przewidywane")
        ax2.set_ylabel("Częstość obserwowana")
        ax2.set_title(f"Kalibracja (Brier = {brier:.4f})")
        
        return {
            "auc": roc_auc,
            "auc_ci_low": auc_ci[0],
            "auc_ci_high": auc_ci[1],
            "brier": brier,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "fig_roc": fig1,
            "fig_cal": fig2,
            "pipeline": pipe
        }
    
    def _generuj_raport_tekstowy(self, top5, wyn_model_glowny, pred, epv_ok):
        lines = []
        lines.append("WYNIKI ANALIZY PROFESJONALNEJ")
        lines.append("")
        lines.append(
            f"Do analizy włączono łącznie {len(self.df_hosp) + len(self.df_dom)} pacjentów, "
            f"w tym {len(self.df_hosp)} hospitalizowanych oraz {len(self.df_dom)} wypisanych do domu."
        )
        
        lines.append("")
        lines.append("TOP 5 PARAMETRÓW (wg istotności):")
        for i, param in enumerate(top5[:5], 1):
            lines.append(f"  {i}. {pretty_name(param)}")
        
        if wyn_model_glowny is not None and len(wyn_model_glowny) > 0:
            sig = wyn_model_glowny[wyn_model_glowny["p_value"] < 0.05]
            lines.append("")
            if len(sig) > 0:
                lines.append("Niezależne czynniki ryzyka:")
                for _, row in sig.iterrows():
                    kier = "większym" if row["OR"] > 1 else "mniejszym"
                    lines.append(
                        f"  • {row['etykieta']}: OR {row['OR']:.2f} (95% CI {row['CI_95%']}; p={row['p_value']:.4f})"
                    )
        
        if pred is not None:
            lines.append("")
            lines.append(
                f"Model predykcyjny: AUC = {pred['auc']:.3f} "
                f"(95% CI {pred['auc_ci_low']:.3f}-{pred['auc_ci_high']:.3f})"
            )
        
        lines.append("")
        lines.append("OGRANICZENIA:")
        lines.append("- Analiza oparta na complete-case analysis")
        lines.append("- Progi kliniczne mają charakter eksploracyjny")
        if not epv_ok:
            lines.append("- Model główny ma ograniczone EPV - interpretować ostrożnie")
        
        return "\n".join(lines)
    
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
                                     text=" WYBIERZ PARAMETR DO WIZUALIZACJI",
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
                            text=" GENERUJ WYKRES",
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
                            text=" ZAPISZ WYKRES",
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
                                  text=" WYKRES",
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
            messagebox.showwarning(" Uwaga", "Wybierz parametr do wizualizacji!")
            return
        
        if self.df_hosp is None:
            messagebox.showwarning(" Uwaga", "Najpierw wczytaj dane!")
            return
        
        hosp = self.df_hosp[param].dropna()
        dom = self.df_dom[param].dropna()
        
        if len(hosp) == 0 or len(dom) == 0:
            messagebox.showwarning(" Uwaga", f"Brak danych dla parametru {param}")
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
            messagebox.showwarning(" Uwaga", "Najpierw wygeneruj wykres!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile=f'wykres_{self.current_param}.png'
        )
        
        if filename:
            try:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo(" Sukces", f"Wykres zapisany jako:\n{filename}")
            except Exception as e:
                messagebox.showerror(" Błąd", f"Nie udało się zapisać pliku:\n{e}")
    
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
        
        btn1 = tk.Button(btn_frame, text=" GENERUJ RAPORT", 
                        bg=KOLORY['accent1'], **btn_style)
        btn1.config(command=self.generuj_raport)
        btn1.pack(side='left', padx=10)
        btn1.bind('<Enter>', lambda e: btn1.config(bg=KOLORY['accent2']))
        btn1.bind('<Leave>', lambda e: btn1.config(bg=KOLORY['accent1']))
        
        btn2 = tk.Button(btn_frame, text=" EKSPORTUJ DO CSV", 
                        bg=KOLORY['success'], **btn_style)
        btn2.config(command=self.export_csv)
        btn2.pack(side='left', padx=10)
        btn2.bind('<Enter>', lambda e: btn2.config(bg=KOLORY['accent2']))
        btn2.bind('<Leave>', lambda e: btn2.config(bg=KOLORY['success']))
        
        btn3 = tk.Button(btn_frame, text=" ODŚWIEŻ", 
                        bg=KOLORY['warning'], **btn_style)
        btn3.config(command=self.odswiez_raport)
        btn3.pack(side='left', padx=10)
        btn3.bind('<Enter>', lambda e: btn3.config(bg=KOLORY['accent2']))
        btn3.bind('<Leave>', lambda e: btn3.config(bg=KOLORY['warning']))
        
        # Ramka raportu
        report_frame = tk.LabelFrame(main_frame,
                                    text=" RAPORT KOŃCOWY",
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
            messagebox.showwarning(" Uwaga", "Brak danych do wygenerowania raportu!")
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
                messagebox.showinfo(" Sukces", f"Raport zapisany jako:\n{filename}")
            except Exception as e:
                messagebox.showerror(" Błąd", f"Nie udało się zapisać pliku:\n{e}")
    
    def odswiez_raport(self):
        """Odświeża raport"""
        self.report_text.delete(1.0, tk.END)
        
        if self.df_hosp is None:
            self.report_text.insert(1.0, " Brak danych. Wczytaj plik w zakładce 'WCZYTAJ DANE'.")
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
 RAPORT KOŃCOWY ANALIZY MEDYCZNEJ

 Data raportu: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
 Tryb analizy: {self.current_mode.upper()}

 PODSUMOWANIE DANYCH:
   • Przyjęci do szpitala: {len(self.df_hosp)} pacjentów
   • Wypisani do domu: {len(self.df_dom)} pacjentów
   • Łącznie: {len(self.df_hosp) + len(self.df_dom)} pacjentów

 ISTOTNOŚĆ STATYSTYCZNA:
   • Parametry istotne (p < 0.05): {len(istotne)}
   • Parametry wysoce istotne (p < 0.001): {len([i for i in istotne if i[1] < 0.001])}

 TOP 5 NAJBARDZIEJ ISTOTNYCH RÓŻNIC:
"""
        for i, (param, p, roznica) in enumerate(istotne[:5], 1):
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            else:
                stars = "*"
            
            kierunek = " WYŻSZE" if roznica > 0 else " NIŻSZE"
            raport += f"\n  {i}. {param:<25}\n"
            raport += f"     p = {p:.6f} {stars}\n"
            raport += f"     {kierunek} u przyjętych (różnica: {roznica:+.2f})\n"
        
        if self.current_mode == "profesjonalna":
            raport += f"""

 Pliki analizy profesjonalnej zostały zapisane w folderze.

"""
        
        raport += f"""

 ANALIZA ZAKOŃCZONA POMYŚLNIE

"""
        self.report_text.insert(1.0, raport)
    
    def export_csv(self):
        """Eksportuje wyniki do CSV"""
        if self.wyniki_df is None:
            messagebox.showwarning(" Uwaga", 
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
                messagebox.showinfo(" Sukces", f"Wyniki zapisane jako:\n{filename}")
            except Exception as e:
                messagebox.showerror(" Błąd", f"Nie udało się zapisać pliku:\n{e}")
    
    # =========================================================================
    # ZAKŁADKA 5 - O PROGRAMIE
    # =========================================================================
    def tab5_info(self):
        """Zakładka informacyjna"""
        frame = ttk.Frame(self.tab5, padding="30")
        frame.pack(fill='both', expand=True)
        
        info_text = """
 ANALIZATOR DANYCH MEDYCZNYCH - WERSJA 11.2

 OPIS PROGRAMU:
 Program służy do porównawczej analizy danych medycznych
 pomiędzy pacjentami przyjętymi do szpitala a wypisanymi do domu.

 DWA TRYBY ANALIZY:
 PODSTAWOWY - szybka analiza:
   • Podstawowe statystyki opisowe
   • Test Manna-Whitneya
   • Wykresy pudełkowe

 PROFESJONALNY - pełna analiza publikacyjna:
   • Tabela 1 (charakterystyka kohorty z interpretacją Cliff's delta)
   • Analiza jednoczynnikowa z korektą FDR
   • Trzy modele regresji logistycznej
   • Forest plot dla OR z 95% CI
   • Model predykcyjny z walidacją krzyżową i bootstrapem
   • Progi kliniczne z uwzględnieniem kierunku efektu
   • Raport z ograniczeniami (complete-case, EPV, VIF)

 KALKULATOR RYZYKA:
   • Po analizie profesjonalnej możesz obliczyć ryzyko hospitalizacji
   • Wprowadź dane pacjenta (wiek, SpO2, CRP, RR skurczowe, RR rozkurczowe)
   • Zaznacz choroby współistniejące
   • Program obliczy prawdopodobieństwo hospitalizacji

 OBSŁUGIWANE FORMATY:
   • CSV (separator ;)
   • Excel (.xlsx, .xls)

 AUTOR:
 Aneta
 Wersja: 11.2 (Marzec 2026)
"""
        label = tk.Label(frame, text=info_text,
                        font=('Courier', 11),
                        bg='white',
                        fg=KOLORY['dark'],
                        justify='left',
                        padx=30,
                        pady=30)
        label.pack(fill='both', expand=True)
    
    # =========================================================================
    # ZAKŁADKA 6 - KALKULATOR RYZYKA HOSPITALIZACJI (POPRAWIONY)
    # =========================================================================
    def tab6_kalkulator(self):
        """Zakładka kalkulatora prawdopodobieństwa hospitalizacji"""
        
        # Główny frame
        main_frame = ttk.Frame(self.tab6, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Tytuł
        title_label = tk.Label(main_frame, 
                              text="KALKULATOR RYZYKA HOSPITALIZACJI",
                              font=('Helvetica', 16, 'bold'),
                              bg=KOLORY['light'],
                              fg=KOLORY['dark'],
                              pady=10)
        title_label.pack(fill='x', pady=(0, 20))
        
        # Ramka z informacją o modelu
        info_frame = tk.LabelFrame(main_frame,
                                  text=" MODEL PREDYKCYJNY",
                                  font=('Helvetica', 12, 'bold'),
                                  bg=KOLORY['light'],
                                  fg=KOLORY['dark'],
                                  padx=15,
                                  pady=10)
        info_frame.pack(fill='x', pady=(0, 20))
        
        self.model_info_label = tk.Label(info_frame,
                                        text="Najpierw wczytaj dane i wykonaj analizę profesjonalną",
                                        font=('Helvetica', 11),
                                        bg=KOLORY['light'],
                                        fg=KOLORY['dark'],
                                        wraplength=600)
        self.model_info_label.pack(pady=5)
        
        # Ramka z danymi pacjenta
        input_frame = tk.LabelFrame(main_frame,
                                   text=" DANE PACJENTA",
                                   font=('Helvetica', 12, 'bold'),
                                   bg=KOLORY['light'],
                                   fg=KOLORY['dark'],
                                   padx=20,
                                   pady=15)
        input_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Tworzenie pól dla parametrów - ZAMIENIAMY MAP na RR skurczowe i rozkurczowe
        parametry_do_kalkulatora = [
            ("wiek", "Wiek (lata)", 0, 120, "lat"),
            ("SpO2", "Saturacja (%)", 0, 100, "%"),
            ("crp(0-0,5)", "CRP (mg/dL)", 0, 500, "mg/dL"),
            ("RR_skurczowe", "RR skurczowe", 0, 300, "mmHg"),
            ("RR_rozkurczowe", "RR rozkurczowe", 0, 200, "mmHg")
        ]
        
        # Układ siatki
        for i, (param, label, min_val, max_val, unit) in enumerate(parametry_do_kalkulatora):
            row = i // 2
            col = i % 2
            
            # Ramka dla pojedynczego parametru
            param_frame = tk.Frame(input_frame, bg=KOLORY['light'])
            param_frame.grid(row=row, column=col, padx=20, pady=15, sticky='w')
            
            # Etykieta
            tk.Label(param_frame, 
                    text=f"{label}:",
                    font=('Helvetica', 11, 'bold'),
                    bg=KOLORY['light'],
                    fg=KOLORY['dark']).pack(anchor='w')
            
            # Pole wprowadzania
            entry_frame = tk.Frame(param_frame, bg=KOLORY['light'])
            entry_frame.pack(fill='x', pady=5)
            
            entry = tk.Entry(entry_frame,
                            font=('Helvetica', 11),
                            width=15,
                            relief='solid',
                            bd=1)
            entry.pack(side='left', padx=(0, 5))
            
            # Jednostka
            tk.Label(entry_frame,
                    text=unit,
                    font=('Helvetica', 10),
                    bg=KOLORY['light'],
                    fg=KOLORY['dark']).pack(side='left')
            
            # Zakres
            tk.Label(param_frame,
                    text=f"zakres: {min_val}-{max_val}",
                    font=('Helvetica', 9),
                    bg=KOLORY['light'],
                    fg='gray').pack(anchor='w')
            
            # Przechowaj zmienną
            self.entry_vars[param] = entry
        
        # Ramka dla chorób współistniejących
        diseases_frame = tk.LabelFrame(input_frame,
                                     text=" CHOROBY WSPÓŁISTNIEJĄCE",
                                     font=('Helvetica', 11, 'bold'),
                                     bg=KOLORY['light'],
                                     fg=KOLORY['dark'],
                                     padx=10,
                                     pady=10)
        diseases_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=20, sticky='ew')
        
        # Checkboxy dla chorób
        diseases = [
            ("dm", "Cukrzyca"),
            ("wątroba", "Choroba wątroby"),
            ("naczyniowe", "Choroby naczyniowe"),
            ("zza", "Zespół zależności alkoholowej"),
            ("npl", "Nowotwór")
        ]
        
        for i, (code, name) in enumerate(diseases):
            var = tk.IntVar()
            cb = tk.Checkbutton(diseases_frame,
                               text=name,
                               variable=var,
                               font=('Helvetica', 10),
                               bg=KOLORY['light'],
                               fg=KOLORY['dark'],
                               activebackground=KOLORY['light'])
            cb.grid(row=i//3, column=i%3, padx=20, pady=5, sticky='w')
            self.disease_vars[code] = var
        
        # Przyciski
        button_frame = tk.Frame(main_frame, bg=KOLORY['light'])
        button_frame.pack(fill='x', pady=20)
        
        calculate_btn = tk.Button(button_frame,
                                 text=" OBLICZ PRAWDOPODOBIEŃSTWO",
                                 font=('Helvetica', 14, 'bold'),
                                 bg=KOLORY['accent1'],
                                 fg='white',
                                 activebackground=KOLORY['accent2'],
                                 activeforeground='white',
                                 relief='raised',
                                 bd=3,
                                 padx=40,
                                 pady=15,
                                 cursor='hand2',
                                 command=self.oblicz_prawdopodobienstwo)
        calculate_btn.pack(side='left', padx=10, expand=True)
        
        reset_btn = tk.Button(button_frame,
                             text=" WYCZYŚĆ",
                             font=('Helvetica', 12, 'bold'),
                             bg=KOLORY['warning'],
                             fg='white',
                             activebackground=KOLORY['accent2'],
                             activeforeground='white',
                             relief='raised',
                             bd=2,
                             padx=30,
                             pady=12,
                             cursor='hand2',
                             command=self.reset_kalkulator)
        reset_btn.pack(side='left', padx=10)
        
        # Ramka wyniku
        result_frame = tk.LabelFrame(main_frame,
                                    text=" WYNIK",
                                    font=('Helvetica', 14, 'bold'),
                                    bg=KOLORY['light'],
                                    fg=KOLORY['dark'],
                                    padx=20,
                                    pady=15)
        result_frame.pack(fill='x', pady=10)
        
        # Etykieta wyniku
        self.result_label = tk.Label(result_frame,
                                    text="Wprowadź dane pacjenta i kliknij 'OBLICZ'",
                                    font=('Helvetica', 14),
                                    bg=KOLORY['light'],
                                    fg=KOLORY['dark'])
        self.result_label.pack(pady=10)
        
        # Pasek postępu dla wizualizacji ryzyka
        self.risk_canvas = tk.Canvas(result_frame,
                                    height=30,
                                    bg='white',
                                    highlightthickness=1,
                                    highlightbackground='gray')
        self.risk_canvas.pack(fill='x', pady=10)
        
        # Legenda
        legend_frame = tk.Frame(result_frame, bg=KOLORY['light'])
        legend_frame.pack(pady=5)
        
        colors = [KOLORY['risk_low'], KOLORY['risk_medium'], KOLORY['risk_high']]
        labels = ['Niskie ryzyko (<30%)', 'Średnie ryzyko (30-60%)', 'Wysokie ryzyko (>60%)']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            tk.Label(legend_frame,
                    text=label,
                    font=('Helvetica', 9),
                    bg=KOLORY['light'],
                    fg=color,
                    padx=10).pack(side='left')
    
    def oblicz_prawdopodobienstwo(self):
        """Oblicza prawdopodobieństwo hospitalizacji na podstawie modelu"""
        
        if self.model_predykcyjny_model is None:
            messagebox.showwarning(" Uwaga", 
                                 "Najpierw wykonaj analizę profesjonalną!\n\n"
                                 "Kliknij 'ANALIZA PROFESJONALNA' w zakładce 2.")
            return
        
        try:
            # Pobierz wartości z pól
            dane_pacjenta = {}
            
            # Parametry ciągłe
            for param, entry in self.entry_vars.items():
                wartosc = entry.get().strip()
                if not wartosc:
                    raise ValueError(f"Wprowadź wartość dla {param}")
                dane_pacjenta[param] = float(wartosc.replace(',', '.'))
            
            # Oblicz MAP z RR skurczowego i rozkurczowego
            if 'RR_skurczowe' in dane_pacjenta and 'RR_rozkurczowe' in dane_pacjenta:
                dane_pacjenta['MAP'] = (dane_pacjenta['RR_skurczowe'] + 2 * dane_pacjenta['RR_rozkurczowe']) / 2
            
            # Choroby współistniejące
            for disease, var in self.disease_vars.items():
                dane_pacjenta[disease] = var.get()
            
            # Przygotuj dane do predykcji (z transformacjami log)
            X_pred = pd.DataFrame([dane_pacjenta])
            
            # Dodaj transformacje log jeśli potrzebne
            for z in ZMIENNE_LOG:
                if z in X_pred.columns:
                    X_pred[f'log_{z}'] = np.log1p(X_pred[z].clip(lower=0))
            
            # Wybierz tylko te zmienne, które były w modelu
            dostepne = [z for z in self.zmienne_modelu if z in X_pred.columns]
            X_pred = X_pred[dostepne]
            
            # Predykcja
            prob = self.model_predykcyjny_model.predict_proba(X_pred)[0, 1]
            prob_percent = prob * 100
            
            # Określenie kategorii ryzyka
            if prob < 0.3:
                kategoria = "NISKIE RYZYKO"
                kolor = KOLORY['risk_low']
            elif prob < 0.6:
                kategoria = "ŚREDNIE RYZYKO"
                kolor = KOLORY['risk_medium']
            else:
                kategoria = "WYSOKIE RYZYKO"
                kolor = KOLORY['risk_high']
            
            # Wyświetl wynik
            wynik_text = f"PRAWDOPODOBIEŃSTWO HOSPITALIZACJI: {prob_percent:.1f}% - {kategoria}"
            self.result_label.config(text=wynik_text, fg=kolor)
            
            # Rysuj pasek postępu
            self.risk_canvas.delete('all')
            canvas_width = self.risk_canvas.winfo_width()
            bar_width = min(prob_percent * 3, canvas_width - 10)
            
            self.risk_canvas.create_rectangle(0, 0, bar_width, 30,
                                             fill=kolor,
                                             outline='')
            self.risk_canvas.create_text(bar_width + 10, 15,
                                        text=f"{prob_percent:.1f}%",
                                        anchor='w',
                                        font=('Helvetica', 10, 'bold'))
            
            # Dodaj linie podziału
            self.risk_canvas.create_line(90, 0, 90, 30, fill='black', dash=(2, 2))
            self.risk_canvas.create_line(180, 0, 180, 30, fill='black', dash=(2, 2))
            
        except ValueError as e:
            messagebox.showerror(" Błąd", f"Nieprawidłowa wartość: {e}")
        except Exception as e:
            messagebox.showerror(" Błąd", f"Wystąpił błąd: {e}")
    
    def reset_kalkulator(self):
        """Czyści pola w kalkulatorze"""
        for entry in self.entry_vars.values():
            entry.delete(0, tk.END)
        
        for var in self.disease_vars.values():
            var.set(0)
        
        self.result_label.config(text="Wprowadź dane pacjenta i kliknij 'OBLICZ'",
                                fg=KOLORY['dark'])
        self.risk_canvas.delete('all')


# =============================================================================
# URUCHOMIENIE
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalAnalyzerGUI(root)
    root.mainloop()