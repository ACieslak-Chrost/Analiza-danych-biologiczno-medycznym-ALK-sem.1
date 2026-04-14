
"""
✨ PRZEPIĘKNE GUI - ANALIZA DANYCH MEDYCZNYCH ✨
Wersja: 11.0 - z kalkulatorem prawdopodobieństwa hospitalizacji
Autor: Aneta
"""
# Nowy przycisk w zakładce WYKRESY

import os
import sqlite3
import warnings
from datetime import datetime
import joblib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact

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

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as pdf_canvas

warnings.filterwarnings("ignore")

# =============================================================================
# USTAWIENIA STYLU
# =============================================================================
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
np.random.seed(42)

KOLORY = {
    "primary": "#2c3e50",
    "secondary": "#34495e",
    "accent1": "#e74c3c",
    "accent2": "#3498db",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "light": "#ecf0f1",
    "dark": "#2c3e50",
    "hosp": "#e74c3c",
    "dom": "#3498db",
    "bg": "#f5f5f5",
    "fg": "#2c3e50",
}

# =============================================================================
# KONFIGURACJA
# =============================================================================
PARAMETRY_KLINICZNE = [
    "wiek", "MAP", "SpO2", "AS", "mleczany",
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
    "MAP",
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
    "log_crp(0-0,5)": "log(CRP)",
    "log_kreatynina(0,5-1,2)": "log(kreatynina)",
    "log_troponina I (0-7,8))": "log(troponina I)",
}

# =============================================================================
# WARTOŚCI KRYTYCZNE DLA SKALI RYZYKA
# =============================================================================
WARTOSCI_KRYTYCZNE = {
    "HGB(12,4-15,2)": {"low": 6.0, "opis": "Ciężka niedokrwistość"},
    "SpO2": {"low": 85.0, "opis": "Krytycznie niska saturacja"},
    "MAP": {"low": 60.0, "opis": "Krytycznie niskie ciśnienie perfuzyjne"},
    "K(3,5-5,1)": {"low": 2.8, "high": 6.0, "opis": "Krytyczne zaburzenie potasu"},
    "Na(137-145)": {"low": 120.0, "high": 155.0, "opis": "Krytyczne zaburzenie sodu"},
    "kreatynina(0,5-1,2)": {"high": 4.0, "opis": "Ciężkie upośledzenie funkcji nerek"},
    "mleczany": {"high": 4.0, "opis": "Znaczna hiperlaktatemia"},
    "crp(0-0,5)": {"high": 15.0, "opis": "Bardzo wysokie CRP"},
    "troponina I (0-7,8))": {"high": 1000.0, "opis": "Bardzo wysoka troponina"},
}

# =============================================================================
# NORMY REFERENCYJNE DO INTERPRETACJI SKALI
# Najpierw próbujemy odczytać normę bezpośrednio z nazwy kolumny, np.
# HGB(12,4-15,2), Na(137-145), crp(0-0,5).
# Punktacja nie powinna rosnąć za wyniki mieszczące się w normie.
# =============================================================================
NORMY_REFERENCYJNE_SKALI = {
    "SpO2": {"low": 95.0, "high": 100.0},
    "MAP": {"low": 70.0, "high": 105.0},
    "mleczany": {"low": 0.0, "high": 2.0},
}

# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================
def pretty_name(x):
    return ETYKIETY.get(x, x)

def odczytaj_norme_z_nazwy_kolumny(param):
    if not isinstance(param, str):
        return None
    try:
        import re
        matches = re.findall(r"\(([-0-9.,]+)-([-0-9.,]+)\)", param)
        if not matches:
            return None
        low_txt, high_txt = matches[-1]
        low = float(low_txt.replace(',', '.'))
        high = float(high_txt.replace(',', '.'))
        return {"low": low, "high": high}
    except Exception:
        return None

def cliff_delta(x, y):
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0
    try:
        u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        return (2 * u_stat) / (n1 * n2) - 1
    except Exception:
        return 0.0

def interpret_cliff_delta(d):
    ad = abs(d)
    if ad < 0.147:
        return "mały"
    if ad < 0.33:
        return "umiarkowany"
    return "duży"

def sprawdz_epv_i_raport(df, zmienne, outcome="outcome", prog=10):
    n_events = int(df[outcome].sum())
    n_vars = len(zmienne)
    epv = n_events / n_vars if n_vars > 0 else 0
    return epv >= prog, epv

def sprawdz_vif(X):
    vif_data = pd.DataFrame()
    vif_data["zmienna"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def _wyniki_modelu_statsmodels(model, zmienne):
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

def bezpieczna_liczba(x):
    if x is None or x == "":
        return np.nan
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return np.nan

def okresl_kategorie_ryzyka(p):
    if p < 0.20:
        return "NISKIE"
    if p < 0.50:
        return "UMIARKOWANE"
    if p < 0.80:
        return "WYSOKIE"
    return "BARDZO WYSOKIE"

# =============================================================================
# GŁÓWNA KLASA APLIKACJI
# =============================================================================
class MedicalAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("✨ ANALIZATOR DANYCH MEDYCZNYCH ✨")
        self.root.geometry("1500x930")
        self.root.configure(bg=KOLORY["bg"])

        # Dane
        self.df = None
        self.df_hosp = None
        self.df_dom = None
        self.wyniki_df = None
        self.current_figure = None
        self.current_param = None
        self.current_mode = "podstawowa"
        self.pro_btn = None

        # Model do kalkulatora hospitalizacji
        self.prediction_pipeline = None
        self.prediction_features = []
        self.prediction_input_vars = {}
        self.map_live_label = None
        self.prediction_model_info = ""
        self.prediction_feature_order = []
        self.prediction_model_source = "wewnętrzny"
        self.loaded_model_path = None

        # Skala ryzyka
        self.scale_current_df = None
        self.scale_frozen_df = None
        self.scale_current_meta = {}
        self.scale_frozen_meta = {}
        self.scale_input_vars = {}
        self.scale_rr_skurczowe_var = None
        self.scale_rr_rozkurczowe_var = None
        self.scale_map_live_label = None

        self.parametry_kliniczne = PARAMETRY_KLINICZNE
        self.choroby = CHOROBY

        self.setup_ui()

    # =========================================================================
    # INTERFEJS
    # =========================================================================
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TNotebook", background=KOLORY["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            font=("Helvetica", 12, "bold"),
            padding=[18, 10],
            background=KOLORY["light"],
            foreground=KOLORY["dark"]
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", KOLORY["primary"])],
            foreground=[("selected", "white")]
        )

        style.configure("TButton", font=("Helvetica", 11), padding=10)
        style.configure("TLabel", font=("Helvetica", 11), background=KOLORY["bg"], foreground=KOLORY["dark"])
        style.configure("TFrame", background=KOLORY["bg"])
        style.configure("TLabelframe", background=KOLORY["bg"], foreground=KOLORY["dark"], font=("Helvetica", 11, "bold"))
        style.configure("TLabelframe.Label", background=KOLORY["bg"], foreground=KOLORY["dark"])

        main_container = ttk.Frame(self.root, padding="15")
        main_container.pack(fill="both", expand=True)

        header_frame = tk.Frame(main_container, bg=KOLORY["primary"], height=80)
        header_frame.pack(fill="x", pady=(0, 15))
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="📊 ANALIZA PORÓWNAWCZA PACJENTÓW",
            font=("Helvetica", 20, "bold"),
            bg=KOLORY["primary"],
            fg="white"
        )
        title_label.pack(expand=True)

        subtitle_label = tk.Label(
            header_frame,
            text="Przyjęci do szpitala vs wypisani do domu",
            font=("Helvetica", 12),
            bg=KOLORY["primary"],
            fg="white"
        )
        subtitle_label.pack(expand=True)

        mode_frame = tk.Frame(main_container, bg=KOLORY["light"], height=50)
        mode_frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            mode_frame,
            text="🔧 TRYB ANALIZY:",
            font=("Helvetica", 12, "bold"),
            bg=KOLORY["light"],
            fg=KOLORY["dark"]
        ).pack(side="left", padx=20, pady=10)

        self.mode_var = tk.StringVar(value="podstawowa")

        mode_basic = tk.Radiobutton(
            mode_frame, text="📊 PODSTAWOWA",
            variable=self.mode_var, value="podstawowa",
            font=("Helvetica", 11), bg=KOLORY["light"],
            fg=KOLORY["dark"], selectcolor=KOLORY["light"],
            command=self.zmien_tryb
        )
        mode_basic.pack(side="left", padx=10, pady=10)

        mode_pro = tk.Radiobutton(
            mode_frame, text="📈 PROFESJONALNA",
            variable=self.mode_var, value="profesjonalna",
            font=("Helvetica", 11), bg=KOLORY["light"],
            fg=KOLORY["dark"], selectcolor=KOLORY["light"],
            command=self.zmien_tryb
        )
        mode_pro.pack(side="left", padx=10, pady=10)

        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill="both", expand=True)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        self.tab5 = ttk.Frame(self.notebook)
        self.tab6 = ttk.Frame(self.notebook)
        self.tab7 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text="📂 WCZYTAJ DANE")
        self.notebook.add(self.tab2, text="📊 ANALIZA STATYSTYCZNA")
        self.notebook.add(self.tab4, text="📈 WYKRESY")
        self.notebook.add(self.tab3, text="🧮 KALKULATOR HOSPITALIZACJI")
        self.notebook.add(self.tab7, text="📌 SKALA RYZYKA")
        self.notebook.add(self.tab5, text="📋 RAPORT")
        self.notebook.add(self.tab6, text="ℹ️ O PROGRAMIE")

        self.tab1_wczytaj()
        self.tab2_analiza()
        self.tab3_kalkulator()
        self.tab4_wykresy()
        self.tab5_raport()
        self.tab6_info()
        self.tab7_skala()

        if self.current_mode == "profesjonalna":
            self.root.after(100, self.dodaj_przycisk_profesjonalny)

    def zmien_tryb(self):
        self.current_mode = self.mode_var.get()

        if hasattr(self, "tree"):
            for row in self.tree.get_children():
                self.tree.delete(row)
            self._ustaw_kolumny_tabeli()

        if self.current_mode == "profesjonalna":
            self.dodaj_przycisk_profesjonalny()
            messagebox.showinfo(
                "🔬 Tryb profesjonalny",
                "Wybrano tryb PROFESJONALNY.\n\n"
                "Zakres:\n"
                "• Tabela 1\n"
                "• Analiza jednoczynnikowa z FDR\n"
                "• Modele regresji logistycznej\n"
                "• Forest plot\n"
                "• Model predykcyjny\n"
                "• Progi kliniczne\n"
                "• Raport końcowy"
            )
        else:
            self.usun_przycisk_profesjonalny()
            messagebox.showinfo(
                "📊 Tryb podstawowy",
                "Wybrano tryb PODSTAWOWY.\n\n"
                "Zakres:\n"
                "• Podstawowe statystyki\n"
                "• Test Manna-Whitneya\n"
                "• Wykresy pudełkowe"
            )

    def dodaj_przycisk_profesjonalny(self):
        try:
            self.usun_przycisk_profesjonalny()

            for child in self.tab2.winfo_children():
                if isinstance(child, tk.LabelFrame) and child.cget("text") == "🎯 PARAMETRY ANALIZY":
                    control_frame = child
                    break
            else:
                return

            btn_frame = None
            for child in control_frame.winfo_children():
                if isinstance(child, tk.Frame):
                    btn_frame = child
            if btn_frame is None:
                return

            self.pro_btn = tk.Button(
                btn_frame,
                text="🔬 ANALIZA PROFESJONALNA",
                font=("Helvetica", 11, "bold"),
                bg=KOLORY["warning"],
                fg="white",
                activebackground=KOLORY["accent2"],
                activeforeground="white",
                relief="flat",
                bd=0,
                padx=20,
                pady=8,
                cursor="hand2",
                command=self.analiza_profesjonalna
            )
            self.pro_btn.pack(side="left", padx=5)
            self.pro_btn.bind("<Enter>", lambda e: self.pro_btn.config(bg=KOLORY["accent2"]))
            self.pro_btn.bind("<Leave>", lambda e: self.pro_btn.config(bg=KOLORY["warning"]))
        except Exception as e:
            print(f"Błąd dodawania przycisku: {e}")

    def usun_przycisk_profesjonalny(self):
        try:
            if self.pro_btn is not None:
                self.pro_btn.destroy()
                self.pro_btn = None
        except Exception:
            pass

    # =========================================================================
    # ZAKŁADKA 1 - WCZYTYWANIE DANYCH
    # =========================================================================
    def tab1_wczytaj(self):
        main_frame = ttk.Frame(self.tab1, padding="30")
        main_frame.pack(fill="both", expand=True)

        button_frame = tk.Frame(main_frame, bg=KOLORY["bg"])
        button_frame.pack(pady=50)

        button_style = {
            "font": ("Helvetica", 14, "bold"),
            "bg": KOLORY["accent1"],
            "fg": "white",
            "activebackground": KOLORY["accent2"],
            "activeforeground": "white",
            "relief": "flat",
            "bd": 0,
            "padx": 30,
            "pady": 15,
            "cursor": "hand2"
        }

        btn_csv = tk.Button(button_frame, text="📁 WCZYTAJ PLIK CSV", command=self.wczytaj_csv, **button_style)
        btn_csv.pack(side="left", padx=20)

        btn_excel = tk.Button(button_frame, text="📗 WCZYTAJ PLIK EXCEL", command=self.wczytaj_excel, **button_style)
        btn_excel.pack(side="left", padx=20)

        btn_sqlite = tk.Button(button_frame, text="🗄️ WCZYTAJ BAZĘ SQLITE", command=self.wybierz_i_wczytaj_sqlite, **button_style)
        btn_sqlite.pack(side="left", padx=20)

        for btn in [btn_csv, btn_excel, btn_sqlite]:
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=KOLORY["accent2"]))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=KOLORY["accent1"]))

        info_frame = tk.LabelFrame(
            main_frame,
            text="📋 INFORMACJE O DANYCH",
            font=("Helvetica", 14, "bold"),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            relief="flat",
            bd=2,
            padx=20,
            pady=15
        )
        info_frame.pack(fill="both", expand=True, pady=30)

        text_frame = tk.Frame(info_frame, bg=KOLORY["light"])
        text_frame.pack(fill="both", expand=True)

        self.info_text = tk.Text(
            text_frame,
            height=15,
            font=("Courier", 11),
            bg="white",
            fg=KOLORY["dark"],
            relief="flat",
            bd=1,
            padx=10,
            pady=10
        )
        self.info_text.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(text_frame, command=self.info_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.info_text.config(yscrollcommand=scrollbar.set)

        analyze_btn = tk.Button(
            main_frame,
            text="🚀 PRZEJDŹ DO ANALIZY",
            font=("Helvetica", 14, "bold"),
            bg=KOLORY["success"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=40,
            pady=15,
            cursor="hand2",
            command=lambda: self.notebook.select(self.tab2)
        )
        analyze_btn.pack(pady=20)
        analyze_btn.bind("<Enter>", lambda e: analyze_btn.config(bg=KOLORY["accent2"]))
        analyze_btn.bind("<Leave>", lambda e: analyze_btn.config(bg=KOLORY["success"]))

        self.info_text.insert(
            "1.0",
            "✨ Witaj w analizatorze danych medycznych!\n\n"
            "Aby rozpocząć:\n"
            "1. Wybierz tryb analizy\n"
            "2. Wczytaj plik CSV, Excel lub bazę SQLite\n"
            "3. Program spróbuje znaleźć lub zbudować kolumnę outcome\n"
            "4. Po wczytaniu danych zostanie też zbudowany kalkulator hospitalizacji\n\n"
            "📊 Tryb PODSTAWOWY - szybka analiza\n"
            "📈 Tryb PROFESJONALNY - pełna analiza publikacyjna"
        )

    def wczytaj_csv(self):
        filename = filedialog.askopenfilename(
            title="Wybierz plik CSV",
            filetypes=[("Pliki CSV", "*.csv"), ("Wszystkie pliki", "*.*")]
        )
        if filename:
            try:
                self.df = pd.read_csv(filename, sep=";", encoding="utf-8")
                self.przetworz_dane()
                self.wyswietl_info(filename)
                self.zbuduj_model_hospitalizacji()
                messagebox.showinfo(
                    "✅ Sukces",
                    f"Plik wczytany poprawnie!\n\n"
                    f"Liczba pacjentów: {len(self.df)}\n"
                    f"Kalkulator hospitalizacji: {'gotowy' if self.prediction_pipeline is not None else 'niezbudowany'}"
                )
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
                self.zbuduj_model_hospitalizacji()
                messagebox.showinfo(
                    "✅ Sukces",
                    f"Plik wczytany poprawnie!\n\n"
                    f"Liczba pacjentów: {len(self.df)}\n"
                    f"Kalkulator hospitalizacji: {'gotowy' if self.prediction_pipeline is not None else 'niezbudowany'}"
                )
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Nie udało się wczytać pliku:\n{e}")

    def wybierz_i_wczytaj_sqlite(self):
        filename = filedialog.askopenfilename(
            title="Wybierz bazę SQLite",
            filetypes=[("Bazy SQLite", "*.sqlite *.db *.sqlite3"), ("Wszystkie pliki", "*.*")]
        )
        if not filename:
            return

        try:
            with sqlite3.connect(filename) as conn:
                tables_df = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
                    conn
                )

            tables = tables_df["name"].tolist()
            if not tables:
                messagebox.showwarning("⚠️ Uwaga", "W wybranej bazie nie znaleziono żadnych tabel.")
                return

            if len(tables) == 1:
                table_name = tables[0]
            else:
                prompt = "Dostępne tabele:\n- " + "\n- ".join(tables) + "\n\nWpisz nazwę tabeli do wczytania:"
                table_name = simpledialog.askstring(
                    "Wybór tabeli SQLite",
                    prompt,
                    initialvalue=tables[0],
                    parent=self.root
                )
                if not table_name:
                    return
                if table_name not in tables:
                    messagebox.showerror("❌ Błąd", f"Tabela '{table_name}' nie istnieje w tej bazie.")
                    return

            self.wczytaj_sqlite(filename, table_name)

        except Exception as e:
            messagebox.showerror("❌ Błąd", f"Nie udało się odczytać bazy SQLite:\n{e}")

    def wczytaj_sqlite(self, filename, table_name):
        try:
            safe_table = table_name.replace(chr(34), chr(34) * 2)
            query = f'SELECT * FROM "{safe_table}"'
            with sqlite3.connect(filename) as conn:
                self.df = pd.read_sql_query(query, conn)

            self.przetworz_dane()
            self.wyswietl_info(filename)
            if hasattr(self, "info_text") and self.info_text is not None:
                self.info_text.insert(tk.END, f"\n🗄️ TABELA SQLITE: {table_name}\n")
            self.zbuduj_model_hospitalizacji()
            messagebox.showinfo(
                "✅ Sukces",
                f"Tabela SQLite wczytana poprawnie!\n\n"
                f"Tabela: {table_name}\n"
                f"Liczba pacjentów: {len(self.df)}\n"
                f"Kalkulator hospitalizacji: {'gotowy' if self.prediction_pipeline is not None else 'niezbudowany'}"
            )
        except Exception as e:
            messagebox.showerror("❌ Błąd", f"Nie udało się wczytać tabeli SQLite:\n{e}")

    def przetworz_dane(self):
        if self.df is None:
            return

        df = self.df.copy()

        if "outcome" not in df.columns:
            puste = df[df.isna().all(axis=1)]
            if len(puste) > 0:
                idx = puste.index[0]
                df_hosp = df.iloc[:idx].copy().dropna(how="all")
                df_dom = df.iloc[idx + 1:].copy().dropna(how="all")
                df_hosp["outcome"] = 1
                df_dom["outcome"] = 0
                df = pd.concat([df_hosp, df_dom], ignore_index=True)
            else:
                raise ValueError(
                    "Brak kolumny 'outcome' i nie udało się wykryć starego formatu z pustym wierszem.\n"
                    "Dodaj kolumnę outcome (1=hospitalizacja, 0=do domu) albo użyj starego układu bazy."
                )

        df = df[df["outcome"].notna()].copy()
        df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
        df = df[df["outcome"].isin([0, 1])].copy()

        for col in self.parametry_kliniczne:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

        mapping_tak = {"tak", "t", "yes", "y", "1", "true", "+", "tak!"}
        mapping_nie = {"nie", "n", "no", "0", "false", "-"}
        for col in self.choroby:
            if col in df.columns:
                tmp = df[col].astype(str).str.lower().str.strip()
                df[col] = tmp.apply(lambda x: 1 if x in mapping_tak else (0 if x in mapping_nie else np.nan))

        self.df = df.copy()
        self.df_hosp = df[df["outcome"] == 1].copy()
        self.df_dom = df[df["outcome"] == 0].copy()

    def wyswietl_info(self, filename):
        self.info_text.delete("1.0", tk.END)

        info = f"""
╔══════════════════════════════════════════════════════════════╗
║                    INFORMACJE O DANYCH                       ║
╚══════════════════════════════════════════════════════════════╝

📁 PLIK: {os.path.basename(filename)}
📅 DATA: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
🔧 TRYB: {"PROFESJONALNY" if self.current_mode == "profesjonalna" else "PODSTAWOWY"}

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

🧮 Kalkulator hospitalizacji:
   • Budowany tylko na podstawie Twojej bazy
   • Model logistyczny uczony po wczytaniu danych

✅ DANE GOTOWE DO ANALIZY!
"""
        self.info_text.insert("1.0", info)

    # =========================================================================
    # ZAKŁADKA 2 - ANALIZA STATYSTYCZNA
    # =========================================================================
    def tab2_analiza(self):
        main_frame = ttk.Frame(self.tab2, padding="20")
        main_frame.pack(fill="both", expand=True)

        control_frame = tk.LabelFrame(
            main_frame,
            text="🎯 PARAMETRY ANALIZY",
            font=("Helvetica", 12, "bold"),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            relief="flat",
            bd=2,
            padx=20,
            pady=15
        )
        control_frame.pack(fill="x", pady=(0, 20))

        ttk.Label(control_frame, text="Wybierz parametr:", font=("Helvetica", 11)).pack(side="left", padx=10)

        self.param_var = tk.StringVar()
        self.param_combo = ttk.Combobox(
            control_frame,
            textvariable=self.param_var,
            values=self.parametry_kliniczne,
            width=40,
            state="readonly",
            font=("Helvetica", 11)
        )
        self.param_combo.pack(side="left", padx=10)

        btn_frame = tk.Frame(control_frame, bg=KOLORY["light"])
        btn_frame.pack(side="right")

        analyze_one_btn = tk.Button(
            btn_frame,
            text="📊 ANALIZUJ WYBRANY",
            font=("Helvetica", 11, "bold"),
            bg=KOLORY["accent1"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self.analizuj_pojedynczy
        )
        analyze_one_btn.pack(side="left", padx=5)
        analyze_one_btn.bind("<Enter>", lambda e: analyze_one_btn.config(bg=KOLORY["accent2"]))
        analyze_one_btn.bind("<Leave>", lambda e: analyze_one_btn.config(bg=KOLORY["accent1"]))

        analyze_all_btn = tk.Button(
            btn_frame,
            text="📊 ANALIZUJ WSZYSTKIE",
            font=("Helvetica", 11, "bold"),
            bg=KOLORY["success"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self.analizuj_wszystkie
        )
        analyze_all_btn.pack(side="left", padx=5)
        analyze_all_btn.bind("<Enter>", lambda e: analyze_all_btn.config(bg=KOLORY["accent2"]))
        analyze_all_btn.bind("<Leave>", lambda e: analyze_all_btn.config(bg=KOLORY["success"]))

        table_frame = tk.LabelFrame(
            main_frame,
            text="📋 WYNIKI ANALIZY STATYSTYCZNEJ",
            font=("Helvetica", 12, "bold"),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            relief="flat",
            bd=2,
            padx=20,
            pady=15
        )
        table_frame.pack(fill="both", expand=True)

        tree_frame = tk.Frame(table_frame, bg=KOLORY["light"])
        tree_frame.pack(fill="both", expand=True)

        vsb = tk.Scrollbar(tree_frame, orient="vertical")
        hsb = tk.Scrollbar(tree_frame, orient="horizontal")

        self.tree = ttk.Treeview(
            tree_frame,
            show="headings",
            height=15,
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set
        )

        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        self.tree.tag_configure("significant", background="#ffe6e6")
        self.tree.tag_configure("highly", background="#ffcccc")

        stats_frame = tk.Frame(main_frame, bg=KOLORY["light"], height=60)
        stats_frame.pack(fill="x", pady=(10, 0))

        self.stats_label = tk.Label(
            stats_frame,
            text="",
            font=("Helvetica", 11),
            bg=KOLORY["light"],
            fg=KOLORY["dark"]
        )
        self.stats_label.pack(pady=10)

        self._ustaw_kolumny_tabeli()

    def _ustaw_kolumny_tabeli(self):
        if self.current_mode == "podstawowa":
            columns = ("lp", "parametr", "hosp_n", "hosp_sr", "hosp_std", "dom_n", "dom_sr", "dom_std", "p", "ist")
            self.tree["columns"] = columns

            self.tree.heading("lp", text="LP")
            self.tree.heading("parametr", text="Parametr")
            self.tree.heading("hosp_n", text="n (hosp)")
            self.tree.heading("hosp_sr", text="Średnia hosp")
            self.tree.heading("hosp_std", text="SD hosp")
            self.tree.heading("dom_n", text="n (dom)")
            self.tree.heading("dom_sr", text="Średnia dom")
            self.tree.heading("dom_std", text="SD dom")
            self.tree.heading("p", text="p-value")
            self.tree.heading("ist", text="Ist.")

            self.tree.column("lp", width=50, anchor="center")
            self.tree.column("parametr", width=220)
            self.tree.column("hosp_n", width=75, anchor="center")
            self.tree.column("hosp_sr", width=100, anchor="center")
            self.tree.column("hosp_std", width=100, anchor="center")
            self.tree.column("dom_n", width=75, anchor="center")
            self.tree.column("dom_sr", width=100, anchor="center")
            self.tree.column("dom_std", width=100, anchor="center")
            self.tree.column("p", width=100, anchor="center")
            self.tree.column("ist", width=70, anchor="center")

        else:
            columns = ("lp", "parametr", "hosp_med", "dom_med", "p_raw", "p_fdr", "delta", "efekt", "ist")
            self.tree["columns"] = columns

            self.tree.heading("lp", text="LP")
            self.tree.heading("parametr", text="Parametr")
            self.tree.heading("hosp_med", text="Hosp mediana [Q1-Q3]")
            self.tree.heading("dom_med", text="Dom mediana [Q1-Q3]")
            self.tree.heading("p_raw", text="p raw")
            self.tree.heading("p_fdr", text="p FDR")
            self.tree.heading("delta", text="Cliff delta")
            self.tree.heading("efekt", text="Efekt")
            self.tree.heading("ist", text="Ist. FDR")

            self.tree.column("lp", width=50, anchor="center")
            self.tree.column("parametr", width=220)
            self.tree.column("hosp_med", width=180, anchor="center")
            self.tree.column("dom_med", width=180, anchor="center")
            self.tree.column("p_raw", width=90, anchor="center")
            self.tree.column("p_fdr", width=90, anchor="center")
            self.tree.column("delta", width=90, anchor="center")
            self.tree.column("efekt", width=100, anchor="center")
            self.tree.column("ist", width=80, anchor="center")

    def analizuj_pojedynczy(self):
        if self.df_hosp is None:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj dane!")
            return

        param = self.param_var.get()
        if not param:
            messagebox.showwarning("⚠️ Uwaga", "Wybierz parametr!")
            return

        for row in self.tree.get_children():
            self.tree.delete(row)

        if param not in self.df.columns:
            return

        hosp = self.df_hosp[param].dropna()
        dom = self.df_dom[param].dropna()

        if len(hosp) == 0 or len(dom) == 0:
            return

        if self.current_mode == "podstawowa":
            hosp_sr = hosp.mean()
            hosp_std = hosp.std()
            dom_sr = dom.mean()
            dom_std = dom.std()
            _, p = stats.mannwhitneyu(hosp, dom, alternative="two-sided")

            if p < 0.001:
                stars = "***"
                tag = "highly"
            elif p < 0.01:
                stars = "**"
                tag = "significant"
            elif p < 0.05:
                stars = "*"
                tag = "significant"
            else:
                stars = "ns"
                tag = ""

            self.tree.insert(
                "", "end", tags=(tag,),
                values=(
                    1,
                    pretty_name(param),
                    len(hosp),
                    f"{hosp_sr:.2f}",
                    f"{hosp_std:.2f}",
                    len(dom),
                    f"{dom_sr:.2f}",
                    f"{dom_std:.2f}",
                    f"{p:.4f}",
                    stars
                )
            )

            self.stats_label.config(
                text=f"✓ Przeanalizowano parametr: {pretty_name(param)} • n(przyjęci)={len(hosp)} • n(wypisani)={len(dom)}"
            )

        else:
            p_raw = stats.mannwhitneyu(hosp, dom, alternative="two-sided").pvalue
            p_fdr = p_raw
            d = cliff_delta(hosp, dom)
            efekt = interpret_cliff_delta(d)

            if p_fdr < 0.001:
                stars = "***"
                tag = "highly"
            elif p_fdr < 0.01:
                stars = "**"
                tag = "significant"
            elif p_fdr < 0.05:
                stars = "*"
                tag = "significant"
            else:
                stars = "ns"
                tag = ""

            hosp_txt = f"{hosp.median():.2f} [{hosp.quantile(0.25):.2f}-{hosp.quantile(0.75):.2f}]"
            dom_txt = f"{dom.median():.2f} [{dom.quantile(0.25):.2f}-{dom.quantile(0.75):.2f}]"

            self.tree.insert(
                "", "end", tags=(tag,),
                values=(
                    1,
                    pretty_name(param),
                    hosp_txt,
                    dom_txt,
                    f"{p_raw:.4f}",
                    f"{p_fdr:.4f}",
                    f"{d:.3f}",
                    efekt,
                    stars
                )
            )

            self.stats_label.config(
                text=f"✓ Tryb profesjonalny • {pretty_name(param)} • mediana/IQR + efekt + FDR"
            )

    def analizuj_wszystkie(self):
        if self.df_hosp is None:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj dane!")
            return

        for row in self.tree.get_children():
            self.tree.delete(row)

        if self.current_mode == "podstawowa":
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

                        _, p = stats.mannwhitneyu(hosp, dom, alternative="two-sided")

                        if p < 0.001:
                            stars = "***"
                            tag = "highly"
                        elif p < 0.01:
                            stars = "**"
                            tag = "significant"
                        elif p < 0.05:
                            stars = "*"
                            tag = "significant"
                        else:
                            stars = "ns"
                            tag = ""

                        self.tree.insert(
                            "", "end", tags=(tag,),
                            values=(
                                i,
                                pretty_name(param),
                                len(hosp),
                                f"{hosp_sr:.2f}",
                                f"{hosp_std:.2f}",
                                len(dom),
                                f"{dom_sr:.2f}",
                                f"{dom_std:.2f}",
                                f"{p:.4f}",
                                stars
                            )
                        )

                        wyniki.append({
                            "parametr": param,
                            "etykieta": pretty_name(param),
                            "p_value": p,
                            "istotnosc": stars
                        })

            self.wyniki_df = pd.DataFrame(wyniki)
            istotne = sum(1 for w in wyniki if w["p_value"] < 0.05)
            wysoce = sum(1 for w in wyniki if w["p_value"] < 0.001)

            self.stats_label.config(
                text=f"✓ Przeanalizowano {len(wyniki)} parametrów • Istotne: {istotne} • Wysoce istotne: {wysoce}"
            )

            messagebox.showinfo(
                "✅ Analiza zakończona",
                f"Przeanalizowano {len(wyniki)} parametrów.\n"
                f"Znaleziono {istotne} parametrów z istotnymi różnicami."
            )

        else:
            wyniki = []
            p_values = []

            for param in self.parametry_kliniczne:
                if param in self.df_hosp.columns and param in self.df_dom.columns:
                    hosp = self.df_hosp[param].dropna()
                    dom = self.df_dom[param].dropna()

                    if len(hosp) > 0 and len(dom) > 0:
                        p_raw = stats.mannwhitneyu(hosp, dom, alternative="two-sided").pvalue
                        d = cliff_delta(hosp, dom)

                        wyniki.append({
                            "parametr": param,
                            "etykieta": pretty_name(param),
                            "hosp_txt": f"{hosp.median():.2f} [{hosp.quantile(0.25):.2f}-{hosp.quantile(0.75):.2f}]",
                            "dom_txt": f"{dom.median():.2f} [{dom.quantile(0.25):.2f}-{dom.quantile(0.75):.2f}]",
                            "p_raw": p_raw,
                            "delta": d,
                            "efekt": interpret_cliff_delta(d),
                            "n_hosp": len(hosp),
                            "n_dom": len(dom)
                        })
                        p_values.append(p_raw)

            if len(wyniki) == 0:
                self.wyniki_df = pd.DataFrame()
                self.stats_label.config(text="Brak danych do analizy profesjonalnej.")
                return

            df_wyn = pd.DataFrame(wyniki)

            _, p_fdr, _, _ = multipletests(p_values, method="fdr_bh")
            df_wyn["p_fdr"] = p_fdr
            df_wyn = df_wyn.sort_values(["p_fdr", "p_raw"]).reset_index(drop=True)

            for i, row in df_wyn.iterrows():
                if row["p_fdr"] < 0.001:
                    stars = "***"
                    tag = "highly"
                elif row["p_fdr"] < 0.01:
                    stars = "**"
                    tag = "significant"
                elif row["p_fdr"] < 0.05:
                    stars = "*"
                    tag = "significant"
                else:
                    stars = "ns"
                    tag = ""

                self.tree.insert(
                    "", "end", tags=(tag,),
                    values=(
                        i + 1,
                        row["etykieta"],
                        row["hosp_txt"],
                        row["dom_txt"],
                        f"{row['p_raw']:.4f}",
                        f"{row['p_fdr']:.4f}",
                        f"{row['delta']:.3f}",
                        row["efekt"],
                        stars
                    )
                )

            self.wyniki_df = df_wyn.copy()

            istotne_fdr = int((df_wyn["p_fdr"] < 0.05).sum())
            wysoce_fdr = int((df_wyn["p_fdr"] < 0.001).sum())

            self.stats_label.config(
                text=f"✓ Tryb profesjonalny • {len(df_wyn)} parametrów • Istotne po FDR: {istotne_fdr} • Wysoce istotne: {wysoce_fdr}"
            )

            messagebox.showinfo(
                "✅ Analiza zakończona",
                f"Tryb profesjonalny.\nPrzeanalizowano {len(df_wyn)} parametrów.\n"
                f"Istotne po korekcji FDR: {istotne_fdr}."
            )

    # =========================================================================
    # ZAKŁADKA 3 - KALKULATOR HOSPITALIZACJI
    # =========================================================================
    def tab3_kalkulator(self):
        main_frame = ttk.Frame(self.tab3, padding="20")
        main_frame.pack(fill="both", expand=True)

        top_frame = tk.LabelFrame(
            main_frame,
            text="🧮 KALKULATOR PRAWDOPODOBIEŃSTWA HOSPITALIZACJI",
            font=("Helvetica", 12, "bold"),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            relief="flat",
            bd=2,
            padx=20,
            pady=15
        )
        top_frame.pack(fill="x", pady=(0, 15))

        self.calc_info_label = tk.Label(
            top_frame,
            text="Po wczytaniu danych program zbuduje model tylko na Twojej bazie.",
            font=("Helvetica", 11),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            justify="left"
        )
        self.calc_info_label.pack(anchor="w", pady=5)

        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill="both", expand=True)

        left_frame = tk.LabelFrame(
            middle_frame,
            text="📋 WPROWADŹ DANE PACJENTA",
            font=("Helvetica", 12, "bold"),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            relief="flat",
            bd=2,
            padx=15,
            pady=15
        )
        left_frame.pack(side="left", fill="y", expand=False, padx=(0, 10))

        entry_and_actions = tk.Frame(left_frame, bg=KOLORY["light"])
        entry_and_actions.pack(fill="both", expand=True)

        self.calc_entries_frame = tk.Frame(entry_and_actions, bg=KOLORY["light"])
        self.calc_entries_frame.pack(side="left", fill="both", expand=True)

        self._zbuduj_pola_kalkulatora()

        actions_frame = tk.Frame(entry_and_actions, bg=KOLORY["light"])
        actions_frame.pack(side="left", fill="y", padx=(12, 0))

        predict_btn = tk.Button(
            actions_frame,
            text="🔮 OBLICZ PRAWDOPODOBIEŃSTWO",
            font=("Helvetica", 11, "bold"),
            bg=KOLORY["accent1"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=14,
            pady=10,
            cursor="hand2",
            wraplength=180,
            justify="center",
            command=self.oblicz_prawdopodobienstwo_hospitalizacji
        )
        predict_btn.pack(fill="x", pady=(0, 8))
        predict_btn.bind("<Enter>", lambda e: predict_btn.config(bg=KOLORY["accent2"]))
        predict_btn.bind("<Leave>", lambda e: predict_btn.config(bg=KOLORY["accent1"]))

        clear_btn = tk.Button(
            actions_frame,
            text="🧹 WYCZYŚĆ",
            font=("Helvetica", 11, "bold"),
            bg=KOLORY["warning"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=14,
            pady=10,
            cursor="hand2",
            command=self.wyczysc_kalkulator
        )
        clear_btn.pack(fill="x", pady=(0, 12))
        clear_btn.bind("<Enter>", lambda e: clear_btn.config(bg=KOLORY["accent2"]))
        clear_btn.bind("<Leave>", lambda e: clear_btn.config(bg=KOLORY["warning"]))

        save_model_btn = tk.Button(
            actions_frame,
            text="💾 ZAPISZ MODEL",
            font=("Helvetica", 10, "bold"),
            bg=KOLORY["success"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2",
            wraplength=180,
            justify="center",
            command=self.zapisz_model_do_pliku
        )
        save_model_btn.pack(fill="x", pady=(0, 6))
        save_model_btn.bind("<Enter>", lambda e: save_model_btn.config(bg=KOLORY["accent2"]))
        save_model_btn.bind("<Leave>", lambda e: save_model_btn.config(bg=KOLORY["success"]))

        load_model_btn = tk.Button(
            actions_frame,
            text="📂 WCZYTAJ MODEL",
            font=("Helvetica", 10, "bold"),
            bg=KOLORY["primary"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2",
            wraplength=180,
            justify="center",
            command=self.wczytaj_model_z_pliku
        )
        load_model_btn.pack(fill="x", pady=(0, 6))
        load_model_btn.bind("<Enter>", lambda e: load_model_btn.config(bg=KOLORY["accent2"]))
        load_model_btn.bind("<Leave>", lambda e: load_model_btn.config(bg=KOLORY["primary"]))

        apply_model_btn = tk.Button(
            actions_frame,
            text="🧾 UŻYJ MODELU NA PLIKU",
            font=("Helvetica", 10, "bold"),
            bg=KOLORY["warning"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2",
            wraplength=180,
            justify="center",
            command=self.zastosuj_model_do_pliku
        )
        apply_model_btn.pack(fill="x")
        apply_model_btn.bind("<Enter>", lambda e: apply_model_btn.config(bg=KOLORY["accent2"]))
        apply_model_btn.bind("<Leave>", lambda e: apply_model_btn.config(bg=KOLORY["warning"]))

        right_frame = tk.LabelFrame(
            middle_frame,
            text="📊 WYNIK I INTERPRETACJA",
            font=("Helvetica", 12, "bold"),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            relief="flat",
            bd=2,
            padx=15,
            pady=15
        )
        right_frame.pack(side="left", fill="both", expand=True)

        self.result_big_label = tk.Label(
            right_frame,
            text="—",
            font=("Helvetica", 36, "bold"),
            bg="white",
            fg=KOLORY["primary"],
            width=18,
            height=2
        )
        self.result_big_label.pack(fill="x", pady=(0, 15))

        result_frame = tk.Frame(right_frame, bg=KOLORY["light"])
        result_frame.pack(fill="both", expand=True)

        result_scrollbar = tk.Scrollbar(result_frame)
        result_scrollbar.pack(side="right", fill="y")

        self.result_text = tk.Text(
            result_frame,
            font=("Courier", 11),
            bg="white",
            fg=KOLORY["dark"],
            wrap="word",
            padx=12,
            pady=12,
            yscrollcommand=result_scrollbar.set
        )
        self.result_text.pack(side="left", fill="both", expand=True)

        result_scrollbar.config(command=self.result_text.yview)

    def _zbuduj_pola_kalkulatora(self):
        for child in self.calc_entries_frame.winfo_children():
            child.destroy()

        self.prediction_input_vars = {}
        self.map_live_label = None

        pola = ZMIENNE_OBOWIAZKOWE + ZMIENNE_DODATKOWE
        unikalne_pola = []
        for p in pola:
            if p not in unikalne_pola:
                unikalne_pola.append(p)

        unikalne_pola = [p for p in unikalne_pola if p != "MAP"]

        for param in unikalne_pola:
            row = tk.Frame(self.calc_entries_frame, bg=KOLORY["light"])
            row.pack(fill="x", pady=4)

            lbl = tk.Label(
                row,
                text=pretty_name(param),
                font=("Helvetica", 10),
                bg=KOLORY["light"],
                fg=KOLORY["dark"],
                width=28,
                anchor="w"
            )
            lbl.pack(side="left", padx=(0, 8))

            var = tk.StringVar()
            ent = tk.Entry(row, textvariable=var, font=("Helvetica", 10), width=18)
            ent.pack(side="left")
            self.prediction_input_vars[param] = var

        row_rr1 = tk.Frame(self.calc_entries_frame, bg=KOLORY["light"])
        row_rr1.pack(fill="x", pady=4)

        lbl_rr1 = tk.Label(
            row_rr1,
            text="RR skurczowe, mmHg",
            font=("Helvetica", 10),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            width=28,
            anchor="w"
        )
        lbl_rr1.pack(side="left", padx=(0, 8))

        self.rr_skurczowe_var = tk.StringVar()
        ent_rr1 = tk.Entry(
            row_rr1,
            textvariable=self.rr_skurczowe_var,
            font=("Helvetica", 10),
            width=18
        )
        ent_rr1.pack(side="left")

        row_rr2 = tk.Frame(self.calc_entries_frame, bg=KOLORY["light"])
        row_rr2.pack(fill="x", pady=4)

        lbl_rr2 = tk.Label(
            row_rr2,
            text="RR rozkurczowe, mmHg",
            font=("Helvetica", 10),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            width=28,
            anchor="w"
        )
        lbl_rr2.pack(side="left", padx=(0, 8))

        self.rr_rozkurczowe_var = tk.StringVar()
        ent_rr2 = tk.Entry(
            row_rr2,
            textvariable=self.rr_rozkurczowe_var,
            font=("Helvetica", 10),
            width=18
        )
        ent_rr2.pack(side="left")

        row_map = tk.Frame(self.calc_entries_frame, bg=KOLORY["light"])
        row_map.pack(fill="x", pady=(8, 4))

        lbl_map = tk.Label(
            row_map,
            text="Wyliczone MAP, mmHg",
            font=("Helvetica", 10, "bold"),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            width=28,
            anchor="w"
        )
        lbl_map.pack(side="left", padx=(0, 8))

        self.map_live_label = tk.Label(
            row_map,
            text="—",
            font=("Helvetica", 10, "bold"),
            bg="white",
            fg=KOLORY["primary"],
            width=18,
            anchor="center",
            relief="solid",
            bd=1
        )
        self.map_live_label.pack(side="left")

        self.rr_skurczowe_var.trace_add("write", lambda *args: self.aktualizuj_map_na_zywo())
        self.rr_rozkurczowe_var.trace_add("write", lambda *args: self.aktualizuj_map_na_zywo())

    def aktualizuj_map_na_zywo(self):
        if self.map_live_label is None:
            return

        sbp = bezpieczna_liczba(self.rr_skurczowe_var.get())
        dbp = bezpieczna_liczba(self.rr_rozkurczowe_var.get())

        if pd.isna(sbp) or pd.isna(dbp):
            self.map_live_label.config(text="—")
            return

        map_val = (sbp + 2 * dbp) / 3
        self.map_live_label.config(text=f"{map_val:.1f}")

    def zbuduj_model_hospitalizacji(self):
        self.prediction_pipeline = None
        self.prediction_features = []
        self.prediction_model_info = ""
        self.prediction_feature_order = []
        self.prediction_model_source = "wewnętrzny"
        self.loaded_model_path = None
        # Skala ryzyka
        self.scale_current_df = None
        self.scale_frozen_df = None
        self.scale_current_meta = {}
        self.scale_frozen_meta = {}
        self.scale_input_vars = {}
        self.scale_rr_skurczowe_var = None
        self.scale_rr_rozkurczowe_var = None
        self.scale_map_live_label = None
        if self.df is None or "outcome" not in self.df.columns:
            self.calc_info_label.config(text="Brak danych do budowy kalkulatora.")
            return

        try:
            df_model = self.df.copy()

            wszystkie = ZMIENNE_OBOWIAZKOWE + ZMIENNE_DODATKOWE
            feature_list = []
            for col in wszystkie:
                if col in df_model.columns:
                    if col in ZMIENNE_LOG:
                        new_name = f"log_{col}"
                        df_model[new_name] = np.log1p(df_model[col].clip(lower=0))
                        feature_list.append(new_name)
                    else:
                        feature_list.append(col)

            final_features = []
            for f in feature_list:
                if f not in final_features:
                    final_features.append(f)

            if len(final_features) == 0:
                self.calc_info_label.config(
                    text="Nie udało się zbudować kalkulatora: brak odpowiednich zmiennych."
                )
                return

            df_cc = df_model[final_features + ["outcome"]].dropna().copy()

            if len(df_cc) < 20:
                self.calc_info_label.config(
                    text=f"Kalkulator niezbudowany: za mało complete-case do modelu (n={len(df_cc)})."
                )
                return

            if df_cc["outcome"].nunique() < 2:
                self.calc_info_label.config(
                    text="Kalkulator niezbudowany: outcome ma tylko jedną klasę."
                )
                return

            X = df_cc[final_features].values
            y = df_cc["outcome"].values

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=1000, random_state=42))
            ])
            pipeline.fit(X, y)

            cv = StratifiedKFold(
                n_splits=min(5, max(2, int(np.min(np.bincount(y.astype(int)))))),
                shuffle=True,
                random_state=42
            )
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
            y_prob = pipeline.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, y_prob)

            self.prediction_pipeline = pipeline
            self.prediction_features = final_features
            self.prediction_feature_order = final_features

            self.prediction_model_info = (
                f"Model gotowy.\n"
                f"• complete-case do modelu: n = {len(df_cc)}\n"
                f"• liczba cech: {len(final_features)}\n"
                f"• cechy: {', '.join([pretty_name(f) for f in final_features])}\n"
                f"• AUC na całej kohorcie treningowej: {roc_auc:.3f}\n"
                f"• CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n"
                f"• Model oparty wyłącznie na Twojej bazie"
            )
            self.calc_info_label.config(text=self.prediction_model_info)

        except Exception as e:
            self.prediction_pipeline = None
            self.calc_info_label.config(
                text=f"Nie udało się zbudować kalkulatora: {e}"
            )

    def oblicz_prawdopodobienstwo_hospitalizacji(self):
        self.result_text.delete("1.0", tk.END)

        if self.prediction_pipeline is None or len(self.prediction_features) == 0:
            messagebox.showwarning(
                "⚠️ Brak modelu",
                "Najpierw wczytaj dane. Kalkulator buduje się wyłącznie na Twojej bazie."
            )
            return

        try:
            dane_wejsciowe = {}
            for param, var in self.prediction_input_vars.items():
                dane_wejsciowe[param] = bezpieczna_liczba(var.get())

            sbp = bezpieczna_liczba(self.rr_skurczowe_var.get())
            dbp = bezpieczna_liczba(self.rr_rozkurczowe_var.get())

            if not pd.isna(sbp) and not pd.isna(dbp):
                dane_wejsciowe["MAP"] = (sbp + 2 * dbp) / 3
            else:
                dane_wejsciowe["MAP"] = np.nan

            row = {}
            for param, value in dane_wejsciowe.items():
                row[param] = value

            for col in ZMIENNE_LOG:
                if col in row:
                    row[f"log_{col}"] = np.log1p(max(row[col], 0)) if not np.isnan(row[col]) else np.nan

            missing_needed = []
            x_values = []

            for feat in self.prediction_features:
                wartosc = row.get(feat, np.nan)

                if pd.isna(wartosc):
                    if feat == "MAP":
                        missing_needed.append("RR skurczowe i RR rozkurczowe")
                    else:
                        missing_needed.append(pretty_name(feat))
                else:
                    x_values.append(wartosc)

            if missing_needed:
                missing_needed = list(dict.fromkeys(missing_needed))
                self.result_big_label.config(text="—", fg=KOLORY["primary"])
                self.result_text.insert(
                    "1.0",
                    "Brak kompletu danych do obliczenia wyniku.\n\n"
                    "Uzupełnij:\n• " + "\n• ".join(missing_needed)
                )
                return

            X_new = np.array([x_values], dtype=float)
            p_hosp = float(self.prediction_pipeline.predict_proba(X_new)[0, 1])
            kat = okresl_kategorie_ryzyka(p_hosp)

            scaler = self.prediction_pipeline.named_steps["scaler"]
            logreg = self.prediction_pipeline.named_steps["logreg"]
            z = (X_new[0] - scaler.mean_) / scaler.scale_
            contributions = z * logreg.coef_[0]

            contrib_df = pd.DataFrame({
                "feature": self.prediction_features,
                "etykieta": [pretty_name(f) for f in self.prediction_features],
                "contribution": contributions,
                "wartosc": X_new[0]
            })
            contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
            contrib_df = contrib_df.sort_values("abs_contribution", ascending=False)

            kolor = KOLORY["success"]
            if p_hosp >= 0.50:
                kolor = KOLORY["accent1"]
            elif p_hosp >= 0.20:
                kolor = KOLORY["warning"]

            self.result_big_label.config(
                text=f"{100 * p_hosp:.1f}%",
                fg=kolor
            )

            tekst = []
            tekst.append("WYNIK KALKULATORA")
            tekst.append("=" * 60)
            tekst.append(f"Prawdopodobieństwo hospitalizacji: {100 * p_hosp:.1f}%")
            tekst.append(f"Kategoria ryzyka: {kat}")
            tekst.append("")
            tekst.append("Interpretacja:")
            tekst.append("• Model wyuczony wyłącznie na Twojej bazie")
            tekst.append("• Wynik ma znaczenie wewnętrzne dla tej kohorty")
            tekst.append("• Nie jest to walidowane narzędzie zewnętrzne")
            tekst.append("")
            tekst.append("Najsilniejsze czynniki wpływające na wynik:")
            for _, rowc in contrib_df.head(5).iterrows():
                kier = "↑ zwiększa" if rowc["contribution"] > 0 else "↓ zmniejsza"
                tekst.append(
                    f"• {rowc['etykieta']}: {rowc['wartosc']:.2f}  |  {kier} ryzyko"
                )

            self.result_text.insert("1.0", "\n".join(tekst))
            self.result_text.see("1.0")

        except Exception as e:
            self.result_big_label.config(text="—", fg=KOLORY["primary"])
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(
                "1.0",
                f"Nie udało się obliczyć wyniku.\n\nSzczegóły:\n{e}"
            )
            self.result_text.see("1.0")

    def wyczysc_kalkulator(self):
        for var in self.prediction_input_vars.values():
            var.set("")

        if hasattr(self, "rr_skurczowe_var"):
            self.rr_skurczowe_var.set("")
        if hasattr(self, "rr_rozkurczowe_var"):
            self.rr_rozkurczowe_var.set("")
        if self.map_live_label is not None:
            self.map_live_label.config(text="—")

        self.result_big_label.config(text="—", fg=KOLORY["primary"])
        self.result_text.delete("1.0", tk.END)


    # =========================================================================
    # ZAKŁADKA 7 - SKALA RYZYKA
    # =========================================================================
    def tab7_skala(self):
        main_frame = ttk.Frame(self.tab7, padding="20")
        main_frame.pack(fill="both", expand=True)

        top_frame = tk.LabelFrame(
            main_frame,
            text="📌 SKALA RYZYKA HOSPITALIZACJI",
            font=("Helvetica", 12, "bold"),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            relief="flat",
            bd=2,
            padx=20,
            pady=15
        )
        top_frame.pack(fill="x", pady=(0, 15))

        self.scale_info_label = tk.Label(
            top_frame,
            text="Najpierw wczytaj dane, potem kliknij: 'Generuj skalę z aktualnej bazy'.",
            font=("Helvetica", 11),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            justify="left"
        )
        self.scale_info_label.pack(anchor="w", pady=5)

        btn_frame = tk.Frame(top_frame, bg=KOLORY["light"])
        btn_frame.pack(fill="x", pady=(10, 0))

        gen_btn = tk.Button(
            btn_frame,
            text="📌 GENERUJ SKALĘ",
            font=("Helvetica", 11, "bold"),
            bg=KOLORY["accent1"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self.generuj_skale_ryzyka
        )
        gen_btn.pack(side="left", padx=5)
        gen_btn.bind("<Enter>", lambda e: gen_btn.config(bg=KOLORY["accent2"]))
        gen_btn.bind("<Leave>", lambda e: gen_btn.config(bg=KOLORY["accent1"]))

        freeze_btn = tk.Button(
            btn_frame,
            text="❄️ ZAMROŹ SKALĘ",
            font=("Helvetica", 11, "bold"),
            bg=KOLORY["warning"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self.zamroz_skale_ryzyka
        )
        freeze_btn.pack(side="left", padx=5)
        freeze_btn.bind("<Enter>", lambda e: freeze_btn.config(bg=KOLORY["accent2"]))
        freeze_btn.bind("<Leave>", lambda e: freeze_btn.config(bg=KOLORY["warning"]))

        compare_btn = tk.Button(
            btn_frame,
            text="🆚 PORÓWNAJ SKALE",
            font=("Helvetica", 11, "bold"),
            bg=KOLORY["success"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self.porownaj_skale_ryzyka
        )
        compare_btn.pack(side="left", padx=5)
        compare_btn.bind("<Enter>", lambda e: compare_btn.config(bg=KOLORY["accent2"]))
        compare_btn.bind("<Leave>", lambda e: compare_btn.config(bg=KOLORY["success"]))

        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill="both", expand=True)

        left_frame = tk.LabelFrame(
            middle_frame,
            text="📋 AKTUALNA SKALA",
            font=("Helvetica", 12, "bold"),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            relief="flat",
            bd=2,
            padx=15,
            pady=15
        )
        left_frame.pack(side="left", fill="both", expand=False, padx=(0, 10))

        columns = ("lp", "parametr", "kierunek", "prog", "punkty", "auc")
        self.scale_tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=12)

        self.scale_tree.heading("lp", text="LP")
        self.scale_tree.heading("parametr", text="Parametr")
        self.scale_tree.heading("kierunek", text="Kierunek")
        self.scale_tree.heading("prog", text="Próg")
        self.scale_tree.heading("punkty", text="Punkty")
        self.scale_tree.heading("auc", text="AUC")

        self.scale_tree.column("lp", width=50, anchor="center")
        self.scale_tree.column("parametr", width=180)
        self.scale_tree.column("kierunek", width=90, anchor="center")
        self.scale_tree.column("prog", width=90, anchor="center")
        self.scale_tree.column("punkty", width=70, anchor="center")
        self.scale_tree.column("auc", width=70, anchor="center")

        scale_tree_scroll = tk.Scrollbar(left_frame, orient="vertical", command=self.scale_tree.yview)
        self.scale_tree.configure(yscrollcommand=scale_tree_scroll.set)

        self.scale_tree.pack(side="left", fill="both", expand=True)
        scale_tree_scroll.pack(side="right", fill="y")

        right_frame = tk.LabelFrame(
            middle_frame,
            text="🧮 KALKULATOR SKALI + PORÓWNAJ",
            font=("Helvetica", 12, "bold"),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            relief="flat",
            bd=2,
            padx=15,
            pady=15
        )
        right_frame.pack(side="left", fill="both", expand=True)

        top_calc_row = tk.Frame(right_frame, bg=KOLORY["light"])
        top_calc_row.pack(fill="x", pady=(0, 10))

        self.scale_entries_frame = tk.Frame(top_calc_row, bg=KOLORY["light"])
        self.scale_entries_frame.pack(side="left", fill="x", expand=True)

        self._zbuduj_pola_skali()

        calc_btn_col = tk.Frame(top_calc_row, bg=KOLORY["light"])
        calc_btn_col.pack(side="left", fill="y", padx=(12, 0))

        calc_btn = tk.Button(
            calc_btn_col,
            text="🧮 OBLICZ WYNIK SKALI",
            font=("Helvetica", 11, "bold"),
            bg=KOLORY["primary"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            cursor="hand2",
            wraplength=170,
            justify="center",
            command=self.oblicz_wynik_skali
        )
        calc_btn.pack(fill="x")
        calc_btn.bind("<Enter>", lambda e: calc_btn.config(bg=KOLORY["accent2"]))
        calc_btn.bind("<Leave>", lambda e: calc_btn.config(bg=KOLORY["primary"]))

        self.scale_result_big_label = tk.Label(
            right_frame,
            text="—",
            font=("Helvetica", 26, "bold"),
            bg="white",
            fg=KOLORY["primary"],
            height=1,
            anchor="center"
        )
        self.scale_result_big_label.pack(fill="x", pady=(0, 8))

        result_frame = tk.Frame(right_frame, bg=KOLORY["light"])
        result_frame.pack(fill="both", expand=True)

        scale_result_scroll = tk.Scrollbar(result_frame)
        scale_result_scroll.pack(side="right", fill="y")

        self.scale_result_text = tk.Text(
            result_frame,
            font=("Courier", 10),
            bg="white",
            fg=KOLORY["dark"],
            wrap="word",
            padx=12,
            pady=12,
            yscrollcommand=scale_result_scroll.set
        )
        self.scale_result_text.pack(side="left", fill="both", expand=True)
        scale_result_scroll.config(command=self.scale_result_text.yview)

    def _zbuduj_pola_skali(self):
        for child in self.scale_entries_frame.winfo_children():
            child.destroy()

        self.scale_input_vars = {}
        self.scale_rr_skurczowe_var = None
        self.scale_rr_rozkurczowe_var = None
        self.scale_map_live_label = None

        scale_df = self.scale_current_df if self.scale_current_df is not None else self.scale_frozen_df
        if scale_df is None or len(scale_df) == 0:
            lbl = tk.Label(
                self.scale_entries_frame,
                text="Brak aktywnej skali. Najpierw wygeneruj skalę.",
                font=("Helvetica", 10),
                bg=KOLORY["light"],
                fg=KOLORY["dark"]
            )
            lbl.pack(anchor="w")
            return

        params = scale_df["parametr"].tolist()

        for param in params:
            if param == "MAP":
                continue

            row = tk.Frame(self.scale_entries_frame, bg=KOLORY["light"])
            row.pack(fill="x", pady=3)

            lbl = tk.Label(
                row,
                text=pretty_name(param),
                font=("Helvetica", 10),
                bg=KOLORY["light"],
                fg=KOLORY["dark"],
                width=28,
                anchor="w"
            )
            lbl.pack(side="left", padx=(0, 8))

            var = tk.StringVar()
            ent = tk.Entry(row, textvariable=var, font=("Helvetica", 10), width=18)
            ent.pack(side="left")
            self.scale_input_vars[param] = var

        if "MAP" in params:
            row_rr1 = tk.Frame(self.scale_entries_frame, bg=KOLORY["light"])
            row_rr1.pack(fill="x", pady=3)

            tk.Label(
                row_rr1,
                text="RR skurczowe, mmHg",
                font=("Helvetica", 10),
                bg=KOLORY["light"],
                fg=KOLORY["dark"],
                width=28,
                anchor="w"
            ).pack(side="left", padx=(0, 8))

            self.scale_rr_skurczowe_var = tk.StringVar()
            tk.Entry(
                row_rr1,
                textvariable=self.scale_rr_skurczowe_var,
                font=("Helvetica", 10),
                width=18
            ).pack(side="left")

            row_rr2 = tk.Frame(self.scale_entries_frame, bg=KOLORY["light"])
            row_rr2.pack(fill="x", pady=3)

            tk.Label(
                row_rr2,
                text="RR rozkurczowe, mmHg",
                font=("Helvetica", 10),
                bg=KOLORY["light"],
                fg=KOLORY["dark"],
                width=28,
                anchor="w"
            ).pack(side="left", padx=(0, 8))

            self.scale_rr_rozkurczowe_var = tk.StringVar()
            tk.Entry(
                row_rr2,
                textvariable=self.scale_rr_rozkurczowe_var,
                font=("Helvetica", 10),
                width=18
            ).pack(side="left")

            row_map = tk.Frame(self.scale_entries_frame, bg=KOLORY["light"])
            row_map.pack(fill="x", pady=(6, 3))

            tk.Label(
                row_map,
                text="Wyliczone MAP, mmHg",
                font=("Helvetica", 10, "bold"),
                bg=KOLORY["light"],
                fg=KOLORY["dark"],
                width=28,
                anchor="w"
            ).pack(side="left", padx=(0, 8))

            self.scale_map_live_label = tk.Label(
                row_map,
                text="—",
                font=("Helvetica", 10, "bold"),
                bg="white",
                fg=KOLORY["primary"],
                width=18,
                anchor="center",
                relief="solid",
                bd=1
            )
            self.scale_map_live_label.pack(side="left")

            self.scale_rr_skurczowe_var.trace_add("write", lambda *args: self.aktualizuj_map_skali_na_zywo())
            self.scale_rr_rozkurczowe_var.trace_add("write", lambda *args: self.aktualizuj_map_skali_na_zywo())

    def aktualizuj_map_skali_na_zywo(self):
        if self.scale_map_live_label is None:
            return

        sbp = bezpieczna_liczba(self.scale_rr_skurczowe_var.get())
        dbp = bezpieczna_liczba(self.scale_rr_rozkurczowe_var.get())

        if pd.isna(sbp) or pd.isna(dbp):
            self.scale_map_live_label.config(text="—")
            return

        map_val = (sbp + 2 * dbp) / 3
        self.scale_map_live_label.config(text=f"{map_val:.1f}")

    def _wybierz_kandydatow_do_skali(self):
        kandydaci = [
            "crp(0-0,5)",
            "troponina I (0-7,8))",
            "SpO2",
            "HGB(12,4-15,2)",
            "hct(38-45)",
            "kreatynina(0,5-1,2)",
            "MAP",
            "wiek"
        ]
        return [k for k in kandydaci if self.df is not None and k in self.df.columns]

    def generuj_skale_ryzyka(self):
        if self.df is None or self.df_hosp is None:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj dane!")
            return

        kandydaci = self._wybierz_kandydatow_do_skali()
        wyniki = []

        for param in kandydaci:
            dane = self.df[[param, "outcome"]].dropna().copy()
            if len(dane) < 30:
                continue
            if dane["outcome"].nunique() < 2:
                continue

            hosp = dane[dane["outcome"] == 1][param]
            dom = dane[dane["outcome"] == 0][param]
            if len(hosp) < 10 or len(dom) < 10:
                continue

            p = stats.mannwhitneyu(hosp, dom, alternative="two-sided").pvalue
            kierunek = "wyższe" if hosp.median() > dom.median() else "niższe"

            try:
                if kierunek == "wyższe":
                    fpr, tpr, thresholds = roc_curve(dane["outcome"], dane[param])
                    auc_val = roc_auc_score(dane["outcome"], dane[param])
                else:
                    fpr, tpr, thresholds = roc_curve(dane["outcome"], -dane[param])
                    auc_val = roc_auc_score(dane["outcome"], -dane[param])

                youden = tpr - fpr
                idx = int(np.argmax(youden))

                if kierunek == "wyższe":
                    prog = float(thresholds[idx])
                else:
                    prog = float(-thresholds[idx])

                wyniki.append({
                    "parametr": param,
                    "etykieta": pretty_name(param),
                    "p_value": p,
                    "auc": auc_val,
                    "kierunek": kierunek,
                    "prog": prog
                })
            except Exception:
                continue

        if len(wyniki) == 0:
            messagebox.showwarning("⚠️ Uwaga", "Nie udało się wygenerować skali z aktualnej bazy.")
            return

        df_scale = pd.DataFrame(wyniki).sort_values(["p_value", "auc"], ascending=[True, False]).head(4).reset_index(drop=True)

        punkty = []
        for _, row in df_scale.iterrows():
            if row["auc"] >= 0.75:
                pkt = 2
            else:
                pkt = 1
            punkty.append(pkt)

        df_scale["punkty"] = punkty

        auc_skali = self._policz_auc_skali(df_scale, self.df)

        self.scale_current_df = df_scale.copy()
        self.scale_current_meta = {
            "n": len(self.df),
            "data": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "auc": auc_skali
        }

        self._odswiez_tabele_skali()
        self._zbuduj_pola_skali()

        self.scale_info_label.config(
            text=(
                f"Skala dynamiczna gotowa.\n"
                f"• liczba pacjentów: {len(self.df)}\n"
                f"• liczba parametrów w skali: {len(df_scale)}\n"
                f"• AUC skali: {auc_skali:.3f}\n"
                f"• wygenerowano: {self.scale_current_meta['data']}"
            )
        )

        self.scale_result_text.delete("1.0", tk.END)
        self.scale_result_text.insert(
            "1.0",
            "Wygenerowano nową skalę z aktualnej bazy.\n\n"
            "To jest wersja dynamiczna — może zmieniać się po dodaniu nowych pacjentów."
        )

    def _policz_auc_skali(self, scale_df, df):
        if scale_df is None or len(scale_df) == 0:
            return np.nan

        needed = scale_df["parametr"].tolist() + ["outcome"]
        dane = df[needed].dropna().copy()
        if len(dane) < 20 or dane["outcome"].nunique() < 2:
            return np.nan

        score_values = []
        for _, rowp in dane.iterrows():
            suma = 0
            for _, rule in scale_df.iterrows():
                val = rowp[rule["parametr"]]
                if rule["kierunek"] == "wyższe":
                    if val >= rule["prog"]:
                        suma += int(rule["punkty"])
                else:
                    if val <= rule["prog"]:
                        suma += int(rule["punkty"])
            score_values.append(suma)

        if len(set(score_values)) < 2:
            return np.nan

        return roc_auc_score(dane["outcome"], score_values)

    def _odswiez_tabele_skali(self):
        for row in self.scale_tree.get_children():
            self.scale_tree.delete(row)

        scale_df = self.scale_current_df if self.scale_current_df is not None else self.scale_frozen_df
        if scale_df is None or len(scale_df) == 0:
            return

        for i, row in scale_df.iterrows():
            self.scale_tree.insert(
                "",
                "end",
                values=(
                    i + 1,
                    row["etykieta"],
                    row["kierunek"],
                    f"{row['prog']:.2f}",
                    int(row["punkty"]),
                    f"{row['auc']:.3f}"
                )
            )

    def zamroz_skale_ryzyka(self):
        if self.scale_current_df is None or len(self.scale_current_df) == 0:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wygeneruj skalę.")
            return

        self.scale_frozen_df = self.scale_current_df.copy()
        self.scale_frozen_meta = self.scale_current_meta.copy()

        self.scale_result_text.delete("1.0", tk.END)
        self.scale_result_text.insert(
            "1.0",
            "Aktualna skala została zamrożona.\n\n"
            "To znaczy, że możesz ją traktować jako wersję finalną do porównań z przyszłymi bazami."
        )

        messagebox.showinfo("✅ Gotowe", "Skala została zamrożona.")

    def porownaj_skale_ryzyka(self):
        self.scale_result_text.delete("1.0", tk.END)

        if self.scale_frozen_df is None or len(self.scale_frozen_df) == 0:
            self.scale_result_text.insert(
                "1.0",
                "Brak skali zamrożonej.\n\nNajpierw wygeneruj skalę, a potem kliknij 'Zamroź skalę'."
            )
            return

        auc_frozen = self._policz_auc_skali(self.scale_frozen_df, self.df)
        auc_current = np.nan
        if self.scale_current_df is not None and len(self.scale_current_df) > 0:
            auc_current = self._policz_auc_skali(self.scale_current_df, self.df)

        tekst = []
        tekst.append("PORÓWNANIE SKAL")
        tekst.append("=" * 60)
        tekst.append("Skala zamrożona:")
        tekst.append(f"• pacjenci przy budowie: {self.scale_frozen_meta.get('n', '—')}")
        tekst.append(f"• data: {self.scale_frozen_meta.get('data', '—')}")
        tekst.append(f"• AUC na aktualnej bazie: {auc_frozen:.3f}" if not pd.isna(auc_frozen) else "• AUC na aktualnej bazie: brak")
        tekst.append("")

        if self.scale_current_df is not None and len(self.scale_current_df) > 0:
            tekst.append("Skala aktualna (dynamiczna):")
            tekst.append(f"• pacjenci przy budowie: {self.scale_current_meta.get('n', '—')}")
            tekst.append(f"• data: {self.scale_current_meta.get('data', '—')}")
            tekst.append(f"• AUC na aktualnej bazie: {auc_current:.3f}" if not pd.isna(auc_current) else "• AUC na aktualnej bazie: brak")
            tekst.append("")
            tekst.append("Parametry skali aktualnej:")
            for _, row in self.scale_current_df.iterrows():
                tekst.append(
                    f"• {row['etykieta']} | {row['kierunek']} | próg {row['prog']:.2f} | {int(row['punkty'])} pkt"
                )
            tekst.append("")
            tekst.append("Parametry skali zamrożonej:")
            for _, row in self.scale_frozen_df.iterrows():
                tekst.append(
                    f"• {row['etykieta']} | {row['kierunek']} | próg {row['prog']:.2f} | {int(row['punkty'])} pkt"
                )
        else:
            tekst.append("Brak aktualnej skali dynamicznej do porównania.")

        self.scale_result_text.insert("1.0", "\n".join(tekst))
        self.scale_result_text.see("1.0")

    def _sprawdz_wartosci_krytyczne_skali(self, dane):
        alerty = []
        for param, reguly in WARTOSCI_KRYTYCZNE.items():
            val = dane.get(param, np.nan)
            if pd.isna(val):
                continue

            low = reguly.get("low")
            high = reguly.get("high")
            opis = reguly.get("opis", "Wartość krytyczna")

            if low is not None and val < low:
                alerty.append(f"• {pretty_name(param)} = {val:.2f} (< {low:.2f}) — {opis}")
            elif high is not None and val > high:
                alerty.append(f"• {pretty_name(param)} = {val:.2f} (> {high:.2f}) — {opis}")

        return alerty

    def _czy_wartosc_poza_norma_dla_skali(self, param, val, kierunek):
        normy = odczytaj_norme_z_nazwy_kolumny(param)
        if normy is None:
            normy = NORMY_REFERENCYJNE_SKALI.get(param)

        if normy is None or pd.isna(val):
            return True

        low = normy.get("low")
        high = normy.get("high")

        if kierunek == "wyższe":
            return high is None or val > high
        if kierunek == "niższe":
            return low is None or val < low
        return True

    def oblicz_wynik_skali(self):
        self.scale_result_text.delete("1.0", tk.END)

        scale_df = self.scale_current_df if self.scale_current_df is not None else self.scale_frozen_df
        if scale_df is None or len(scale_df) == 0:
            self.scale_result_text.insert("1.0", "Brak aktywnej skali. Najpierw wygeneruj skalę.")
            return

        dane = {}
        for param, var in self.scale_input_vars.items():
            dane[param] = bezpieczna_liczba(var.get())

        if "MAP" in scale_df["parametr"].tolist():
            sbp = bezpieczna_liczba(self.scale_rr_skurczowe_var.get()) if self.scale_rr_skurczowe_var is not None else np.nan
            dbp = bezpieczna_liczba(self.scale_rr_rozkurczowe_var.get()) if self.scale_rr_rozkurczowe_var is not None else np.nan
            if not pd.isna(sbp) and not pd.isna(dbp):
                dane["MAP"] = (sbp + 2 * dbp) / 3
            else:
                dane["MAP"] = np.nan

        brakujace = []
        suma = 0
        trafienia = []

        for _, rule in scale_df.iterrows():
            param = rule["parametr"]
            val = dane.get(param, np.nan)

            if pd.isna(val):
                if param == "MAP":
                    brakujace.append("RR skurczowe i RR rozkurczowe")
                else:
                    brakujace.append(pretty_name(param))
                continue

            czy_spelnia = False
            if rule["kierunek"] == "wyższe" and val >= rule["prog"]:
                czy_spelnia = True
            if rule["kierunek"] == "niższe" and val <= rule["prog"]:
                czy_spelnia = True

            poza_norma = self._czy_wartosc_poza_norma_dla_skali(param, val, rule["kierunek"])

            if czy_spelnia and poza_norma:
                suma += int(rule["punkty"])
                trafienia.append(
                    f"• {rule['etykieta']}: {val:.2f} ({rule['kierunek']} niż {rule['prog']:.2f}, poza normą) → +{int(rule['punkty'])} pkt"
                )

        if brakujace:
            brakujace = list(dict.fromkeys(brakujace))

        max_pkt = int(scale_df["punkty"].sum())
        if max_pkt <= 0:
            max_pkt = 1

        if suma <= 0.25 * max_pkt:
            kat = "NISKIE"
            kolor = KOLORY["success"]
        elif suma <= 0.50 * max_pkt:
            kat = "UMIARKOWANE"
            kolor = KOLORY["warning"]
        elif suma <= 0.75 * max_pkt:
            kat = "WYSOKIE"
            kolor = KOLORY["accent1"]
        else:
            kat = "BARDZO WYSOKIE"
            kolor = KOLORY["accent1"]

        alerty_krytyczne = self._sprawdz_wartosci_krytyczne_skali(dane)
        if alerty_krytyczne:
            kat_koncowa = "BARDZO WYSOKIE"
            kolor_koncowy = KOLORY["accent1"]
        else:
            kat_koncowa = kat
            kolor_koncowy = kolor

        self.scale_result_big_label.config(text=f"{suma} pkt", fg=kolor_koncowy)

        tekst = []
        tekst.append("WYNIK SKALI RYZYKA")
        tekst.append("=" * 60)
        tekst.append(f"Suma punktów: {suma} / {max_pkt}")
        tekst.append(f"Kategoria skali: {kat}")
        if alerty_krytyczne:
            tekst.append(f"Kategoria końcowa po uwzględnieniu wartości krytycznych: {kat_koncowa}")
        else:
            tekst.append(f"Kategoria końcowa: {kat_koncowa}")
        tekst.append("")

        if trafienia:
            tekst.append("Elementy zwiększające wynik:")
            tekst.extend(trafienia)
            tekst.append("")

        if alerty_krytyczne:
            tekst.append("WARTOŚCI KRYTYCZNE:")
            tekst.extend(alerty_krytyczne)
            tekst.append("")
            tekst.append("Wniosek kliniczny:")
            tekst.append("• Niska punktacja skali nie wyklucza ciężkiego stanu")
            tekst.append("• Obecność wartości krytycznych wymaga pilnej oceny klinicznej / hospitalizacji")
            tekst.append("")

        if brakujace:
            tekst.append("Brakujące dane:")
            for b in brakujace:
                tekst.append(f"• {b}")
            tekst.append("")

        tekst.append("Uwaga:")
        tekst.append("• To uproszczona skala punktowa zbudowana na Twojej bazie")
        tekst.append("• Wartości krytyczne mają pierwszeństwo nad samą punktacją")
        tekst.append("• Skala dynamiczna może się zmieniać po dodaniu nowych pacjentów")
        tekst.append("• Skala zamrożona służy do porównań i bardziej stałego użycia")

        self.scale_result_text.insert("1.0", "\n".join(tekst))
        self.scale_result_text.see("1.0")



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
    # ZAKŁADKA 4 - WYKRESY
    # =========================================================================
    def tab4_wykresy(self):
        main_frame = ttk.Frame(self.tab4, padding="20")
        main_frame.pack(fill="both", expand=True)

        control_frame = tk.LabelFrame(
            main_frame,
            text="🎯 WYBIERZ PARAMETR DO WIZUALIZACJI",
            font=("Helvetica", 12, "bold"),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            relief="flat",
            bd=2,
            padx=20,
            pady=15
        )
        control_frame.pack(fill="x", pady=(0, 20))

        top_row = tk.Frame(control_frame, bg=KOLORY["light"])
        top_row.pack(fill="x", pady=(0, 8))

        ttk.Label(top_row, text="Parametr:", font=("Helvetica", 11)).pack(side="left", padx=10)

        self.plot_param_var = tk.StringVar()
        self.plot_combo = ttk.Combobox(
            top_row,
            textvariable=self.plot_param_var,
            values=self.parametry_kliniczne,
            width=28,
            state="readonly",
            font=("Helvetica", 11)
        )
        self.plot_combo.pack(side="left", padx=10)

        ttk.Label(top_row, text="Rodzaj wykresu:", font=("Helvetica", 11)).pack(side="left", padx=(15, 10))

        self.plot_type_var = tk.StringVar(value="Boxplot + punkty")
        self.plot_type_combo = ttk.Combobox(
            top_row,
            textvariable=self.plot_type_var,
            values=["Boxplot + punkty", "Boxplot", "Histogram", "Violinplot"],
            width=18,
            state="readonly",
            font=("Helvetica", 11)
        )
        self.plot_type_combo.pack(side="left", padx=10)

        btn_row = tk.Frame(control_frame, bg=KOLORY["light"])
        btn_row.pack(fill="x")

        plot_btn = tk.Button(
            btn_row,
            text="📈 GENERUJ WYKRES",
            font=("Helvetica", 11, "bold"),
            bg=KOLORY["accent1"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            height=2,
            cursor="hand2",
            command=self.rysuj_wykres
        )
        plot_btn.pack(side="left", padx=5)
        plot_btn.bind("<Enter>", lambda e: plot_btn.config(bg=KOLORY["accent2"]))
        plot_btn.bind("<Leave>", lambda e: plot_btn.config(bg=KOLORY["accent1"]))

        roc_multi_btn = tk.Button(
            btn_row,
            text="📊 ROC WIELU PARAMETRÓW",
            font=("Helvetica", 11, "bold"),
            bg=KOLORY["secondary"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2",
            command=self.rysuj_roc_porownawcze
        )
        roc_multi_btn.pack(side="left", padx=5)
        roc_multi_btn.bind("<Enter>", lambda e: roc_multi_btn.config(bg=KOLORY["accent2"]))
        roc_multi_btn.bind("<Leave>", lambda e: roc_multi_btn.config(bg=KOLORY["secondary"]))

        dca_btn = tk.Button(
            btn_row,
            text="📉 DCA",
            font=("Helvetica", 11, "bold"),
            bg=KOLORY["primary"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            height=2,
            cursor="hand2",
            command=self.rysuj_decision_curve
        )
        dca_btn.pack(side="left", padx=5)
        dca_btn.bind("<Enter>", lambda e: dca_btn.config(bg=KOLORY["accent2"]))
        dca_btn.bind("<Leave>", lambda e: dca_btn.config(bg=KOLORY["primary"]))

        all_plots_btn = tk.Button(
            btn_row,
            text="🖼️ GENERUJ WSZYSTKIE\nWYKRESY",
            font=("Helvetica", 11, "bold"),
            bg=KOLORY["warning"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2",
            justify="center",
            command=self.generuj_wszystkie_wykresy
        )
        all_plots_btn.pack(side="right", padx=5)
        all_plots_btn.bind("<Enter>", lambda e: all_plots_btn.config(bg=KOLORY["accent2"]))
        all_plots_btn.bind("<Leave>", lambda e: all_plots_btn.config(bg=KOLORY["warning"]))

        save_btn = tk.Button(
            btn_row,
            text="💾 ZAPISZ WYKRES",
            font=("Helvetica", 11, "bold"),
            bg=KOLORY["success"],
            fg="white",
            activebackground=KOLORY["accent2"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            height=2,
            cursor="hand2",
            command=self.zapisz_wykres
        )
        save_btn.pack(side="right", padx=5)
        save_btn.bind("<Enter>", lambda e: save_btn.config(bg=KOLORY["accent2"]))
        save_btn.bind("<Leave>", lambda e: save_btn.config(bg=KOLORY["success"]))

        plot_frame = tk.LabelFrame(
            main_frame,
            text="📊 WYKRES",
            font=("Helvetica", 12, "bold"),
            bg=KOLORY["light"],
            fg=KOLORY["dark"],
            relief="flat",
            bd=2,
            padx=20,
            pady=15
        )
        plot_frame.pack(fill="both", expand=True)

        self.figure = Figure(figsize=(12, 7), dpi=100, facecolor="white")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#f8f9fa")

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(plot_frame, bg=KOLORY["light"])
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
    
        def rysuj_wykres(self):
            param = self.plot_param_var.get()
            plot_type = self.plot_type_var.get()
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
            self.ax.clear()
    
            _, p = stats.mannwhitneyu(hosp, dom, alternative="two-sided")
    
            log_scale = param in ["troponina I (0-7,8))", "crp(0-0,5)"]
    
            if plot_type in ["Boxplot + punkty", "Boxplot"]:
                bp = self.ax.boxplot(
                    [hosp, dom],
                    labels=["PRZYJĘCI", "WYPISANI"],
                    patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2}
                )
    
                bp["boxes"][0].set_facecolor(KOLORY["hosp"])
                bp["boxes"][0].set_alpha(0.8)
                bp["boxes"][1].set_facecolor(KOLORY["dom"])
                bp["boxes"][1].set_alpha(0.8)
    
                if plot_type == "Boxplot + punkty":
                    x_hosp = np.random.normal(1, 0.05, len(hosp))
                    x_dom = np.random.normal(2, 0.05, len(dom))
                    self.ax.scatter(x_hosp, hosp, alpha=0.5, color="darkred", s=30, zorder=3)
                    self.ax.scatter(x_dom, dom, alpha=0.5, color="darkblue", s=30, zorder=3)
    
            elif plot_type == "Histogram":
                bins = 15
                self.ax.hist(hosp, bins=bins, alpha=0.6, label="PRZYJĘCI", color=KOLORY["hosp"])
                self.ax.hist(dom, bins=bins, alpha=0.6, label="WYPISANI", color=KOLORY["dom"])
                self.ax.set_xlabel(pretty_name(param), fontsize=11)
                self.ax.set_ylabel("Liczba pacjentów", fontsize=11)
                self.ax.legend()
    
            elif plot_type == "Violinplot":
                parts = self.ax.violinplot(
                    [hosp, dom],
                    positions=[1, 2],
                    showmeans=True,
                    showmedians=True
                )
    
                for i, body in enumerate(parts["bodies"]):
                    body.set_alpha(0.7)
                    body.set_facecolor(KOLORY["hosp"] if i == 0 else KOLORY["dom"])
    
                self.ax.set_xticks([1, 2])
                self.ax.set_xticklabels(["PRZYJĘCI", "WYPISANI"])
                self.ax.set_ylabel(pretty_name(param), fontsize=11)
    
            if log_scale:
                if plot_type == "Histogram":
                    self.ax.set_xscale("log")
                    self.ax.set_xlabel(f"{pretty_name(param)} (skala log)", fontsize=11)
                else:
                    self.ax.set_yscale("log")
                    self.ax.set_ylabel(f"{pretty_name(param)} (skala log)", fontsize=11)
            else:
                if plot_type != "Histogram":
                    self.ax.set_ylabel(pretty_name(param), fontsize=11)
    
            if p < 0.001:
                title = f"{pretty_name(param)}\np < 0.001 ***"
            elif p < 0.01:
                title = f"{pretty_name(param)}\np = {p:.4f} **"
            elif p < 0.05:
                title = f"{pretty_name(param)}\np = {p:.4f} *"
            else:
                title = f"{pretty_name(param)}\np = {p:.4f} (ns)"
    
            self.ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
            self.ax.grid(True, alpha=0.3, linestyle="--")
    
            stats_text = (
                f"Przyjęci:\n"
                f"n = {len(hosp)}\n"
                f"śr = {hosp.mean():.2f} ± {hosp.std():.2f}\n"
                f"mediana = {np.median(hosp):.2f}\n\n"
                f"Wypisani:\n"
                f"n = {len(dom)}\n"
                f"śr = {dom.mean():.2f} ± {dom.std():.2f}\n"
                f"mediana = {np.median(dom):.2f}"
            )
    
            self.ax.text(
                0.02, 0.98, stats_text,
                transform=self.ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                fontsize=9
            )
    
            self.figure.tight_layout()
            self.canvas.draw()
    
        def generuj_wszystkie_wykresy(self):
            if self.df_hosp is None:
                messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj dane!")
                return
    
            plot_type = self.plot_type_var.get()
            folder = filedialog.askdirectory(title="Wybierz folder do zapisu wszystkich wykresów")
    
            if not folder:
                return
    
            zapisane = 0
            pominiete = 0
    
            for param in self.parametry_kliniczne:
                if param not in self.df_hosp.columns or param not in self.df_dom.columns:
                    pominiete += 1
                    continue
    
                hosp = self.df_hosp[param].dropna()
                dom = self.df_dom[param].dropna()
    
                if len(hosp) == 0 or len(dom) == 0:
                    pominiete += 1
                    continue
    
                fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
                ax.set_facecolor("#f8f9fa")
    
                _, p = stats.mannwhitneyu(hosp, dom, alternative="two-sided")
                log_scale = param in ["troponina I (0-7,8))", "crp(0-0,5)"]
    
                if plot_type in ["Boxplot + punkty", "Boxplot"]:
                    bp = ax.boxplot(
                        [hosp, dom],
                        labels=["PRZYJĘCI", "WYPISANI"],
                        patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2}
                    )
    
                    bp["boxes"][0].set_facecolor(KOLORY["hosp"])
                    bp["boxes"][0].set_alpha(0.8)
                    bp["boxes"][1].set_facecolor(KOLORY["dom"])
                    bp["boxes"][1].set_alpha(0.8)
    
                    if plot_type == "Boxplot + punkty":
                        x_hosp = np.random.normal(1, 0.05, len(hosp))
                        x_dom = np.random.normal(2, 0.05, len(dom))
                        ax.scatter(x_hosp, hosp, alpha=0.5, color="darkred", s=30, zorder=3)
                        ax.scatter(x_dom, dom, alpha=0.5, color="darkblue", s=30, zorder=3)
    
                elif plot_type == "Histogram":
                    bins = 15
                    ax.hist(hosp, bins=bins, alpha=0.6, label="PRZYJĘCI", color=KOLORY["hosp"])
                    ax.hist(dom, bins=bins, alpha=0.6, label="WYPISANI", color=KOLORY["dom"])
                    ax.set_xlabel(pretty_name(param), fontsize=11)
                    ax.set_ylabel("Liczba pacjentów", fontsize=11)
                    ax.legend()
    
                elif plot_type == "Violinplot":
                    parts = ax.violinplot(
                        [hosp, dom],
                        positions=[1, 2],
                        showmeans=True,
                        showmedians=True
                    )
    
                    for i, body in enumerate(parts["bodies"]):
                        body.set_alpha(0.7)
                        body.set_facecolor(KOLORY["hosp"] if i == 0 else KOLORY["dom"])
    
                    ax.set_xticks([1, 2])
                    ax.set_xticklabels(["PRZYJĘCI", "WYPISANI"])
                    ax.set_ylabel(pretty_name(param), fontsize=11)
    
                if log_scale:
                    if plot_type == "Histogram":
                        ax.set_xscale("log")
                        ax.set_xlabel(f"{pretty_name(param)} (skala log)", fontsize=11)
                    else:
                        ax.set_yscale("log")
                        ax.set_ylabel(f"{pretty_name(param)} (skala log)", fontsize=11)
                else:
                    if plot_type != "Histogram":
                        ax.set_ylabel(pretty_name(param), fontsize=11)
    
                if p < 0.001:
                    title = f"{pretty_name(param)}\np < 0.001 ***"
                elif p < 0.01:
                    title = f"{pretty_name(param)}\np = {p:.4f} **"
                elif p < 0.05:
                    title = f"{pretty_name(param)}\np = {p:.4f} *"
                else:
                    title = f"{pretty_name(param)}\np = {p:.4f} (ns)"
    
                ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
                ax.grid(True, alpha=0.3, linestyle="--")
    
                stats_text = (
                    f"Przyjęci:\n"
                    f"n = {len(hosp)}\n"
                    f"śr = {hosp.mean():.2f} ± {hosp.std():.2f}\n"
                    f"mediana = {np.median(hosp):.2f}\n\n"
                    f"Wypisani:\n"
                    f"n = {len(dom)}\n"
                    f"śr = {dom.mean():.2f} ± {dom.std():.2f}\n"
                    f"mediana = {np.median(dom):.2f}"
                )
    
                ax.text(
                    0.02, 0.98, stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                    fontsize=9
                )
    
                fig.tight_layout()
    
                safe_name = (
                    param.replace("/", "_")
                         .replace("\\", "_")
                         .replace(":", "_")
                         .replace("*", "_")
                         .replace("?", "_")
                         .replace('"', "_")
                         .replace("<", "_")
                         .replace(">", "_")
                         .replace("|", "_")
                         .replace(" ", "_")
                )
    
                nazwa_pliku = f"wykres_{safe_name}_{plot_type.replace(' ', '_').replace('+', 'plus')}.png"
                sciezka = os.path.join(folder, nazwa_pliku)
    
                fig.savefig(sciezka, dpi=300, bbox_inches="tight")
                plt.close(fig)
                zapisane += 1
    
            messagebox.showinfo(
                "✅ Gotowe",
                f"Wygenerowano {zapisane} wykresów.\n"
                f"Pominięto {pominiete} parametrów bez danych.\n\n"
                f"Folder:\n{folder}"
            )
    
    def rysuj_roc_porownawcze(self):
        if self.df is None or "outcome" not in self.df.columns:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj dane!")
            return
    
        kandydaci = []
        for col in (ZMIENNE_OBOWIAZKOWE + ZMIENNE_DODATKOWE):
            if col in self.df.columns and col not in kandydaci:
                kandydaci.append(col)
    
        curves = []
        for param in kandydaci:
            dane = self.df[[param, "outcome"]].dropna().copy()
            if len(dane) < 20 or dane["outcome"].nunique() < 2:
                continue
            values = dane[param].astype(float)
            if values.nunique() < 2:
                continue
            hosp_med = dane[dane["outcome"] == 1][param].median()
            dom_med = dane[dane["outcome"] == 0][param].median()
            score = values if hosp_med >= dom_med else -values
            fpr, tpr, _ = roc_curve(dane["outcome"], score)
            auc_val = roc_auc_score(dane["outcome"], score)
            curves.append((pretty_name(param), fpr, tpr, auc_val))
    
        curves = sorted(curves, key=lambda x: x[3], reverse=True)[:6]
        if len(curves) == 0:
            messagebox.showwarning("⚠️ Uwaga", "Brak wystarczających danych do porównawczych krzywych ROC.")
            return
    
        self.ax.clear()
        for label, fpr, tpr, auc_val in curves:
            self.ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc_val:.3f})")
    
        if self.prediction_pipeline is not None and len(self.prediction_features) > 0:
            try:
                model_df, probs, valid_mask = self._oblicz_prob_z_modelu_i_df(self.df)
                y = model_df.loc[valid_mask, "outcome"]
                if len(y) >= 20 and y.nunique() == 2:
                    fpr, tpr, _ = roc_curve(y, probs.loc[valid_mask])
                    auc_val = roc_auc_score(y, probs.loc[valid_mask])
                    self.ax.plot(fpr, tpr, linewidth=3, linestyle="--", label=f"Model wieloczynnikowy (AUC={auc_val:.3f})")
            except Exception:
                pass
    
        self.ax.plot([0, 1], [0, 1], "k--", alpha=0.7)
        self.ax.set_xlabel("1 - swoistość", fontsize=11)
        self.ax.set_ylabel("Czułość", fontsize=11)
        self.ax.set_title("Porównawcze krzywe ROC kilku parametrów", fontsize=14, fontweight="bold", pad=15)
        self.ax.legend(loc="lower right", fontsize=9)
        self.ax.grid(True, alpha=0.3, linestyle="--")
        self.current_param = "ROC_porownawcze"
        self.figure.tight_layout()
        self.canvas.draw()
    
    def rysuj_decision_curve(self):
        if self.df is None or self.prediction_pipeline is None or len(self.prediction_features) == 0:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj dane i zbuduj model hospitalizacji.")
            return
    
        try:
            df_pred, probs, valid_mask = self._oblicz_prob_z_modelu_i_df(self.df)
            dane = df_pred.loc[valid_mask, ["outcome"]].copy()
            dane["prob"] = probs.loc[valid_mask].values
    
            if len(dane) < 20 or dane["outcome"].nunique() < 2:
                messagebox.showwarning("⚠️ Uwaga", "Za mało complete-case do analizy decision curve.")
                return
    
            thresholds = np.arange(0.05, 0.96, 0.01)
            n = len(dane)
            prevalence = dane["outcome"].mean()
            nb_model = []
            nb_all = []
    
            for pt in thresholds:
                pred_pos = dane["prob"] >= pt
                tp = ((pred_pos) & (dane["outcome"] == 1)).sum()
                fp = ((pred_pos) & (dane["outcome"] == 0)).sum()
                weight = pt / (1 - pt)
                nb_model.append((tp / n) - (fp / n) * weight)
                nb_all.append(prevalence - (1 - prevalence) * weight)
    
            self.ax.clear()
            self.ax.plot(thresholds, nb_model, linewidth=2.5, label="Model")
            self.ax.plot(thresholds, nb_all, linewidth=2, linestyle="--", label="Traktuj wszystkich")
            self.ax.plot(thresholds, np.zeros_like(thresholds), linewidth=2, linestyle=":", label="Traktuj nikogo")
            self.ax.set_xlabel("Próg prawdopodobieństwa", fontsize=11)
            self.ax.set_ylabel("Net benefit", fontsize=11)
            self.ax.set_title("Decision Curve Analysis", fontsize=14, fontweight="bold", pad=15)
            self.ax.legend(fontsize=9)
            self.ax.grid(True, alpha=0.3, linestyle="--")
            self.current_param = "decision_curve_analysis"
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("❌ Błąd", f"Nie udało się wygenerować DCA:\n{e}")
    
        def zapisz_wykres(self):
            if self.current_param is None:
                messagebox.showwarning("⚠️ Uwaga", "Najpierw wygeneruj wykres!")
                return
    
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
                initialfile=f"wykres_{self.current_param}.png"
            )
    
            if filename:
                try:
                    self.figure.savefig(filename, dpi=300, bbox_inches="tight")
                    messagebox.showinfo("✅ Sukces", f"Wykres zapisany jako:\n{filename}")
                except Exception as e:
                    messagebox.showerror("❌ Błąd", f"Nie udało się zapisać pliku:\n{e}")
    
        # =========================================================================
        # ZAKŁADKA 5 - RAPORT
        # =========================================================================
        def tab5_raport(self):
            main_frame = ttk.Frame(self.tab5, padding="20")
            main_frame.pack(fill="both", expand=True)
    
            btn_frame = tk.Frame(main_frame, bg=KOLORY["light"])
            btn_frame.pack(fill="x", pady=(0, 20))
    
            btn_style = {
                "font": ("Helvetica", 12, "bold"),
                "fg": "white",
                "activebackground": KOLORY["accent2"],
                "activeforeground": "white",
                "relief": "flat",
                "bd": 0,
                "padx": 25,
                "pady": 12,
                "cursor": "hand2"
            }
    
            btn1 = tk.Button(btn_frame, text="📊 GENERUJ RAPORT", bg=KOLORY["accent1"], **btn_style)
            btn1.config(command=self.generuj_raport)
            btn1.pack(side="left", padx=10)
            btn1.bind("<Enter>", lambda e: btn1.config(bg=KOLORY["accent2"]))
            btn1.bind("<Leave>", lambda e: btn1.config(bg=KOLORY["accent1"]))
    
            btn2 = tk.Button(btn_frame, text="💾 EKSPORTUJ DO CSV", bg=KOLORY["success"], **btn_style)
            btn2.config(command=self.export_csv)
            btn2.pack(side="left", padx=10)
            btn2.bind("<Enter>", lambda e: btn2.config(bg=KOLORY["accent2"]))
            btn2.bind("<Leave>", lambda e: btn2.config(bg=KOLORY["success"]))
    
            btn_pdf = tk.Button(btn_frame, text="📄 EKSPORTUJ DO PDF", bg=KOLORY["primary"], **btn_style)
            btn_pdf.config(command=self.eksportuj_pdf)
            btn_pdf.pack(side="left", padx=10)
            btn_pdf.bind("<Enter>", lambda e: btn_pdf.config(bg=KOLORY["accent2"]))
            btn_pdf.bind("<Leave>", lambda e: btn_pdf.config(bg=KOLORY["primary"]))
    
            btn3 = tk.Button(btn_frame, text="🔄 ODŚWIEŻ", bg=KOLORY["warning"], **btn_style)
            btn3.config(command=self.odswiez_raport)
            btn3.pack(side="left", padx=10)
            btn3.bind("<Enter>", lambda e: btn3.config(bg=KOLORY["accent2"]))
            btn3.bind("<Leave>", lambda e: btn3.config(bg=KOLORY["warning"]))
    
            report_frame = tk.LabelFrame(
                main_frame,
                text="📋 RAPORT KOŃCOWY",
                font=("Helvetica", 14, "bold"),
                bg=KOLORY["light"],
                fg=KOLORY["dark"],
                relief="flat",
                bd=2,
                padx=20,
                pady=15
            )
            report_frame.pack(fill="both", expand=True)
    
            text_frame = tk.Frame(report_frame, bg=KOLORY["light"])
            text_frame.pack(fill="both", expand=True)
    
            self.report_text = tk.Text(
                text_frame,
                font=("Courier", 11),
                bg="white",
                fg=KOLORY["dark"],
                wrap="word",
                padx=15,
                pady=15
            )
            self.report_text.pack(side="left", fill="both", expand=True)
    
            scrollbar = tk.Scrollbar(text_frame, command=self.report_text.yview)
            scrollbar.pack(side="right", fill="y")
            self.report_text.config(yscrollcommand=scrollbar.set)
    
        def generuj_raport(self):
            if self.df_hosp is None:
                messagebox.showwarning("⚠️ Uwaga", "Brak danych do wygenerowania raportu!")
                return
    
            self.odswiez_raport()
    
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Pliki tekstowe", "*.txt")],
                initialfile="raport_medyczny.txt"
            )
    
            if filename:
                try:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(self.report_text.get("1.0", tk.END))
                    messagebox.showinfo("✅ Sukces", f"Raport zapisany jako:\n{filename}")
                except Exception as e:
                    messagebox.showerror("❌ Błąd", f"Nie udało się zapisać pliku:\n{e}")
    
        def odswiez_raport(self):
            self.report_text.delete("1.0", tk.END)
    
            if self.df_hosp is None:
                self.report_text.insert("1.0", "Brak danych. Wczytaj plik w zakładce 'WCZYTAJ DANE'.")
                return
    
            if self.current_mode == "podstawowa":
                wyniki = []
                istotne = []
    
                for param in self.parametry_kliniczne:
                    if param in self.df_hosp.columns and param in self.df_dom.columns:
                        hosp = self.df_hosp[param].dropna()
                        dom = self.df_dom[param].dropna()
    
                        if len(hosp) > 0 and len(dom) > 0:
                            _, p = stats.mannwhitneyu(hosp, dom, alternative="two-sided")
                            roznica = hosp.mean() - dom.mean()
    
                            wyniki.append({
                                "parametr": param,
                                "hosp_sr": hosp.mean(),
                                "dom_sr": dom.mean(),
                                "p": p,
                                "roznica": roznica
                            })
    
                            if p < 0.05:
                                istotne.append((param, p, roznica))
    
                istotne.sort(key=lambda x: x[1])
    
                raport = f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              RAPORT KOŃCOWY ANALIZY MEDYCZNEJ                    ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    📅 Data raportu: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    🔧 Tryb analizy: PODSTAWOWY
    {'='*70}
    
    📊 PODSUMOWANIE DANYCH:
    ────────────────────────────────────────────────────────────────────
      • 🏥 Przyjęci do szpitala: {len(self.df_hosp)} pacjentów
      • 🏠 Wypisani do domu: {len(self.df_dom)} pacjentów
      • 👥 Łącznie: {len(self.df_hosp) + len(self.df_dom)} pacjentów
    
    📈 WYNIKI PODSTAWOWE:
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
                    raport += f"\n  {i}. {pretty_name(param):<25}\n"
                    raport += f"     p = {p:.6f} {stars}\n"
                    raport += f"     {kierunek} u przyjętych (różnica średnich: {roznica:+.2f})\n"
    
            else:
                wyniki = []
                p_values = []
    
                for param in self.parametry_kliniczne:
                    if param in self.df_hosp.columns and param in self.df_dom.columns:
                        hosp = self.df_hosp[param].dropna()
                        dom = self.df_dom[param].dropna()
    
                        if len(hosp) > 0 and len(dom) > 0:
                            p_raw = stats.mannwhitneyu(hosp, dom, alternative="two-sided").pvalue
                            d = cliff_delta(hosp, dom)
    
                            wyniki.append({
                                "parametr": param,
                                "etykieta": pretty_name(param),
                                "hosp_med": f"{hosp.median():.2f} [{hosp.quantile(0.25):.2f}-{hosp.quantile(0.75):.2f}]",
                                "dom_med": f"{dom.median():.2f} [{dom.quantile(0.25):.2f}-{dom.quantile(0.75):.2f}]",
                                "p_raw": p_raw,
                                "delta": d,
                                "efekt": interpret_cliff_delta(d)
                            })
                            p_values.append(p_raw)
    
                df_wyn = pd.DataFrame(wyniki)
    
                if len(df_wyn) > 0:
                    _, p_fdr, _, _ = multipletests(p_values, method="fdr_bh")
                    df_wyn["p_fdr"] = p_fdr
                    df_wyn = df_wyn.sort_values(["p_fdr", "p_raw"]).reset_index(drop=True)
                else:
                    df_wyn["p_fdr"] = []
    
                istotne_fdr = df_wyn[df_wyn["p_fdr"] < 0.05].copy() if len(df_wyn) > 0 else pd.DataFrame()
    
                raport = f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              RAPORT KOŃCOWY ANALIZY MEDYCZNEJ                    ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    📅 Data raportu: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    🔧 Tryb analizy: PROFESJONALNY
    {'='*70}
    
    📊 PODSUMOWANIE DANYCH:
    ────────────────────────────────────────────────────────────────────
      • 🏥 Przyjęci do szpitala: {len(self.df_hosp)} pacjentów
      • 🏠 Wypisani do domu: {len(self.df_dom)} pacjentów
      • 👥 Łącznie: {len(self.df_hosp) + len(self.df_dom)} pacjentów
    
    📈 WYNIKI PROFESJONALNE:
    ────────────────────────────────────────────────────────────────────
      • Parametry istotne po FDR (q < 0.05): {len(istotne_fdr)}
      • Parametry wysoce istotne po FDR (q < 0.001): {int((df_wyn['p_fdr'] < 0.001).sum()) if len(df_wyn) else 0}
    
    🔬 TOP 5 PARAMETRÓW:
    ────────────────────────────────────────────────────────────────────
    """
                for i, (_, row) in enumerate(df_wyn.head(5).iterrows(), 1):
                    raport += f"\n  {i}. {row['etykieta']}\n"
                    raport += f"     hosp: {row['hosp_med']}\n"
                    raport += f"     dom : {row['dom_med']}\n"
                    raport += f"     p raw = {row['p_raw']:.6f}\n"
                    raport += f"     p FDR = {row['p_fdr']:.6f}\n"
                    raport += f"     Cliff delta = {row['delta']:.3f} ({row['efekt']})\n"
    
            if self.prediction_pipeline is not None:
                raport += f"""
    {'='*70}
    🧮 KALKULATOR HOSPITALIZACJI:
    ────────────────────────────────────────────────────────────────────
    {self.prediction_model_info}
    """
    
            raport += f"""
    {'='*70}
    ✅ ANALIZA ZAKOŃCZONA POMYŚLNIE
    {'='*70}
    """
            self.report_text.insert("1.0", raport)
    
        def eksportuj_pdf(self):
            if self.df_hosp is None:
                messagebox.showwarning("⚠️ Uwaga", "Brak danych do wygenerowania PDF!")
                return
    
            self.odswiez_raport()
            filename = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF", "*.pdf")],
                initialfile="raport_medyczny.pdf"
            )
            if not filename:
                return
    
            try:
                c = pdf_canvas.Canvas(filename, pagesize=A4)
                width, height = A4
                margin = 1.5 * cm
                y = height - margin
                line_height = 14
                font_name = "Courier"
                font_size = 9
                c.setFont(font_name, font_size)
    
                text_lines = self.report_text.get("1.0", tk.END).splitlines()
                max_width = width - 2 * margin
    
                for raw_line in text_lines:
                    line = raw_line.expandtabs(4)
                    if line == "":
                        y -= line_height
                        if y < margin:
                            c.showPage()
                            c.setFont(font_name, font_size)
                            y = height - margin
                        continue
    
                    while stringWidth(line, font_name, font_size) > max_width:
                        cut = max(1, int(len(line) * max_width / max(stringWidth(line, font_name, font_size), 1)))
                        subline = line[:cut]
                        while stringWidth(subline, font_name, font_size) > max_width and len(subline) > 1:
                            subline = subline[:-1]
                        c.drawString(margin, y, subline)
                        y -= line_height
                        line = line[len(subline):]
                        if y < margin:
                            c.showPage()
                            c.setFont(font_name, font_size)
                            y = height - margin
    
                    c.drawString(margin, y, line)
                    y -= line_height
                    if y < margin:
                        c.showPage()
                        c.setFont(font_name, font_size)
                        y = height - margin
    
                c.save()
                messagebox.showinfo("✅ Sukces", f"Raport PDF zapisany jako:\n{filename}")
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Nie udało się zapisać PDF:\n{e}")
    
        def export_csv(self):
                if self.wyniki_df is None:
                    messagebox.showwarning("⚠️ Uwaga", "Najpierw wykonaj analizę!")
                    return
    
                filename = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV", "*.csv")],
                    initialfile="wyniki_analizy.csv"
                )
    
                if filename:
                    try:
                        self.wyniki_df.to_csv(filename, sep=";", index=False, encoding="utf-8")
                        messagebox.showinfo("✅ Sukces", f"Wyniki zapisane jako:\n{filename}")
                    except Exception as e:
                        messagebox.showerror("❌ Błąd", f"Nie udało się zapisać pliku:\n{e}")
    
        # =========================================================================
        # ZAKŁADKA 6 - O PROGRAMIE
        # =========================================================================
        def tab6_info(self):
            frame = ttk.Frame(self.tab6, padding="30")
            frame.pack(fill="both", expand=True)
    
            info_text = """
    ╔══════════════════════════════════════════════════════════════╗
    ║         ANALIZATOR DANYCH MEDYCZNYCH - WERSJA 11.0          ║
    ╚══════════════════════════════════════════════════════════════╝
    
    📋 OPIS PROGRAMU:
    ──────────────────────────────────────────────────────────────
    Program służy do porównawczej analizy danych medycznych
    pomiędzy pacjentami przyjętymi do szpitala a wypisanymi do domu.
    
    🧮 NOWOŚĆ:
    ──────────────────────────────────────────────────────────────
    Trzecia zakładka to kalkulator prawdopodobieństwa hospitalizacji.
    Model:
    • budowany jest dopiero po wczytaniu Twojej bazy,
    • uczy się wyłącznie na Twoich danych,
    • wykorzystuje logistyczny model predykcyjny,
    • nie korzysta z żadnych danych zewnętrznych.
    
    🔧 TRYBY ANALIZY:
    ──────────────────────────────────────────────────────────────
    📊 PODSTAWOWY:
      • Podstawowe statystyki
      • Test Manna-Whitneya
      • Wykresy pudełkowe
    
    📈 PROFESJONALNY:
      • Tabela 1
      • Analiza jednoczynnikowa z FDR
      • Regresja logistyczna
      • Forest plot
      • ROC i kalibracja
      • Raport końcowy
    
    👩‍⚕️ AUTOR:
    ──────────────────────────────────────────────────────────────
    Aneta
    Wersja 11.0
    """
            label = tk.Label(
                frame,
                text=info_text,
                font=("Courier", 11),
                bg="white",
                fg=KOLORY["dark"],
                justify="left",
                padx=30,
                pady=30
            )
            label.pack(fill="both", expand=True)
    
    
    
    # =============================================================================
    # DODANE BRAKUJĄCE METODY KLASY (patch)
    # =============================================================================
    def _patched_analiza_profesjonalna(self):
        if self.df_hosp is None:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj dane!")
            return
    
        messagebox.showinfo(
            "🔬 Analiza profesjonalna",
            "Rozpoczynam pełną analizę publikacyjną.\nTo może potrwać chwilę."
        )
    
        try:
            df_caly = self.df.copy()
    
            raport_brakow_df = self._raport_brakow_pro(df_caly)
            raport_brakow_df.to_csv("raport_brakow.csv", sep=";", index=False)
    
            walidacja_df = self._walidacja_zakresow(df_caly)
            walidacja_df.to_csv("walidacja_zakresow.csv", sep=";", index=False)
    
            tabela1 = self._tabela_1_pro(df_caly)
            tabela1.to_csv("tabela_1_publikacyjna.csv", sep=";", index=False)
    
            wyniki_fdr, top5 = self._analiza_jednoczynnikowa_pro(df_caly)
            wyniki_fdr.to_csv("analiza_jednoczynnikowa_fdr.csv", sep=";", index=False)
    
            missing_top = self._missingness_top_pro(top5)
            missing_top.to_csv("missingness_top5.csv", sep=";", index=False)
    
            progi = self._progi_kliniczne_pro(df_caly, top5)
            progi.to_csv("progi_kliniczne_eksploracyjne.csv", sep=";", index=False)
    
            df_model, zmienne_modelu = self._przygotuj_zmienne_modelu(df_caly)
    
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
    
            pred = self._model_predykcyjny(df_model, zmienne_modelu)
            if pred is not None:
                pred["fig_roc"].savefig("krzywa_ROC.png", dpi=300, bbox_inches="tight")
                pred["fig_cal"].savefig("krzywa_kalibracji.png", dpi=300, bbox_inches="tight")
                plt.close("all")
    
            raport_txt = self._generuj_raport_tekstowy(top5, wyn2, pred, epv_ok)
            with open("raport_wyniki_i_ograniczenia.txt", "w", encoding="utf-8") as f:
                f.write(raport_txt)
    
            messagebox.showinfo(
                "✅ Sukces",
                "Analiza profesjonalna zakończona!\n\nWygenerowano pliki wynikowe w katalogu roboczym."
            )
    
        except Exception as e:
            messagebox.showerror("❌ Błąd", f"Wystąpił błąd podczas analizy:\n{str(e)}")
    
    
    def _patched_zapisz_model_do_pliku(self):
        if self.prediction_pipeline is None or len(getattr(self, 'prediction_features', [])) == 0:
            messagebox.showwarning("⚠️ Uwaga", "Brak gotowego modelu do zapisania. Najpierw wczytaj dane i zbuduj model.")
            return
    
        filename = filedialog.asksaveasfilename(
            defaultextension='.joblib',
            filetypes=[('Plik modelu', '*.joblib')],
            initialfile='model_hospitalizacji.joblib'
        )
        if not filename:
            return
    
        payload = {
            'pipeline': self.prediction_pipeline,
            'features': self.prediction_features,
            'feature_order': getattr(self, 'prediction_feature_order', self.prediction_features),
            'model_info': getattr(self, 'prediction_model_info', ''),
            'saved_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        try:
            joblib.dump(payload, filename)
            messagebox.showinfo('✅ Sukces', f'Model zapisany jako\n{filename}')
        except Exception as e:
            messagebox.showerror('❌ Błąd', f'Nie udało się zapisać modelu:\n{e}')
    
    
    def _patched_wczytaj_model_z_pliku(self):
        filename = filedialog.askopenfilename(
            title='Wczytaj zapisany model',
            filetypes=[('Plik modelu', '*.joblib'), ('Wszystkie pliki', '*.*')]
        )
        if not filename:
            return
    
        try:
            payload = joblib.load(filename)
            self.prediction_pipeline = payload['pipeline']
            self.prediction_features = payload.get('features', [])
            self.prediction_feature_order = payload.get('feature_order', self.prediction_features)
            self.prediction_model_info = payload.get('model_info', 'Model wczytany z pliku.')
            if hasattr(self, 'calc_info_label'):
                self.calc_info_label.config(text=self.prediction_model_info)
            messagebox.showinfo('✅ Sukces', f'Model został wczytany:\n{filename}')
        except Exception as e:
            messagebox.showerror('❌ Błąd', f'Nie udało się wczytać modelu:\n{e}')
    
    
    def _patched_zastosuj_model_do_pliku(self):
        if self.prediction_pipeline is None or len(getattr(self, 'prediction_features', [])) == 0:
            messagebox.showwarning('⚠️ Uwaga', 'Najpierw zbuduj albo wczytaj model.')
            return
    
        filename = filedialog.askopenfilename(
            title='Wybierz plik z nowymi pacjentami',
            filetypes=[('Excel', '*.xlsx *.xls'), ('CSV', '*.csv'), ('Wszystkie pliki', '*.*')]
        )
        if not filename:
            return
    
        try:
            if filename.lower().endswith('.csv'):
                df_new = pd.read_csv(filename, sep=';', encoding='utf-8')
                if df_new.shape[1] == 1:
                    df_new = pd.read_csv(filename)
            else:
                df_new = pd.read_excel(filename)
    
            if 'MAP' in self.prediction_features and 'MAP' not in df_new.columns:
                sbp_candidates = [c for c in df_new.columns if c.lower().strip() in ['rr skurczowe', 'rr_skurczowe', 'sbp', 'rr skurczowe, mmhg']]
                dbp_candidates = [c for c in df_new.columns if c.lower().strip() in ['rr rozkurczowe', 'rr_rozkurczowe', 'dbp', 'rr rozkurczowe, mmhg']]
                if sbp_candidates and dbp_candidates:
                    sbp_col = sbp_candidates[0]
                    dbp_col = dbp_candidates[0]
                    sbp = pd.to_numeric(df_new[sbp_col].astype(str).str.replace(',', '.'), errors='coerce')
                    dbp = pd.to_numeric(df_new[dbp_col].astype(str).str.replace(',', '.'), errors='coerce')
                    df_new['MAP'] = (sbp + 2 * dbp) / 3
    
            for col in list(df_new.columns):
                if col in PARAMETRY_KLINICZNE:
                    df_new[col] = pd.to_numeric(df_new[col].astype(str).str.replace(',', '.'), errors='coerce')
    
            for col in ZMIENNE_LOG:
                if col in df_new.columns:
                    df_new[f'log_{col}'] = np.log1p(df_new[col].clip(lower=0))
    
            missing = [f for f in self.prediction_features if f not in df_new.columns]
            if missing:
                messagebox.showerror('❌ Błąd', 'Brakuje kolumn wymaganych przez model:\n• ' + '\n• '.join(missing))
                return
    
            df_pred = df_new.copy()
            X = df_pred[self.prediction_features].apply(pd.to_numeric, errors='coerce')
            valid_mask = X.notna().all(axis=1)
            probs = pd.Series(np.nan, index=df_pred.index, dtype=float)
            if valid_mask.any():
                probs.loc[valid_mask] = self.prediction_pipeline.predict_proba(X.loc[valid_mask].values)[:, 1]
    
            df_pred['p_hospitalizacji'] = probs
            df_pred['kategoria_ryzyka'] = df_pred['p_hospitalizacji'].apply(lambda p: okresl_kategorie_ryzyka(p) if pd.notna(p) else np.nan)
    
            outname = filedialog.asksaveasfilename(
                title='Zapisz wyniki predykcji',
                defaultextension='.xlsx',
                filetypes=[('Excel', '*.xlsx'), ('CSV', '*.csv')],
                initialfile='predykcja_hospitalizacji.xlsx'
            )
            if not outname:
                return
    
            if outname.lower().endswith('.csv'):
                df_pred.to_csv(outname, sep=';', index=False, encoding='utf-8')
            else:
                df_pred.to_excel(outname, index=False)
    
            messagebox.showinfo('✅ Sukces', f'Zastosowano model do nowego pliku.\nWynik zapisano jako:\n{outname}')
        except Exception as e:
            messagebox.showerror('❌ Błąd', f'Nie udało się zastosować modelu:\n{e}')
    
    
# =============================================================================
# PODPIĘCIE METOD PATCH DO KLASY
# =============================================================================
MedicalAnalyzerGUI.analiza_profesjonalna = MedicalAnalyzerGUI._patched_analiza_profesjonalna
MedicalAnalyzerGUI.zapisz_model_do_pliku = MedicalAnalyzerGUI._patched_zapisz_model_do_pliku
MedicalAnalyzerGUI.wczytaj_model_z_pliku = MedicalAnalyzerGUI._patched_wczytaj_model_z_pliku
MedicalAnalyzerGUI.zastosuj_model_do_pliku = MedicalAnalyzerGUI._patched_zastosuj_model_do_pliku

# =============================================================================
# PATCH METOD, KTÓRE WYPADŁY POZA KLASĘ PRZY SKŁADANIU PLIKU
# =============================================================================

def _patched_rysuj_wykres(self):
    param = self.plot_param_var.get()
    plot_type = self.plot_type_var.get()
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
    self.ax.clear()

    _, p = stats.mannwhitneyu(hosp, dom, alternative="two-sided")
    log_scale = param in ["troponina I (0-7,8))", "crp(0-0,5)"]

    if plot_type in ["Boxplot + punkty", "Boxplot"]:
        bp = self.ax.boxplot(
            [hosp, dom],
            labels=["PRZYJĘCI", "WYPISANI"],
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2}
        )
        bp["boxes"][0].set_facecolor(KOLORY["hosp"])
        bp["boxes"][0].set_alpha(0.8)
        bp["boxes"][1].set_facecolor(KOLORY["dom"])
        bp["boxes"][1].set_alpha(0.8)

        if plot_type == "Boxplot + punkty":
            x_hosp = np.random.normal(1, 0.05, len(hosp))
            x_dom = np.random.normal(2, 0.05, len(dom))
            self.ax.scatter(x_hosp, hosp, alpha=0.5, color="darkred", s=30, zorder=3)
            self.ax.scatter(x_dom, dom, alpha=0.5, color="darkblue", s=30, zorder=3)

    elif plot_type == "Histogram":
        bins = 15
        self.ax.hist(hosp, bins=bins, alpha=0.6, label="PRZYJĘCI", color=KOLORY["hosp"])
        self.ax.hist(dom, bins=bins, alpha=0.6, label="WYPISANI", color=KOLORY["dom"])
        self.ax.set_xlabel(pretty_name(param), fontsize=11)
        self.ax.set_ylabel("Liczba pacjentów", fontsize=11)
        self.ax.legend()

    elif plot_type == "Violinplot":
        parts = self.ax.violinplot([hosp, dom], positions=[1, 2], showmeans=True, showmedians=True)
        for i, body in enumerate(parts["bodies"]):
            body.set_alpha(0.7)
            body.set_facecolor(KOLORY["hosp"] if i == 0 else KOLORY["dom"])
        self.ax.set_xticks([1, 2])
        self.ax.set_xticklabels(["PRZYJĘCI", "WYPISANI"])
        self.ax.set_ylabel(pretty_name(param), fontsize=11)

    if log_scale:
        if plot_type == "Histogram":
            self.ax.set_xscale("log")
            self.ax.set_xlabel(f"{pretty_name(param)} (skala log)", fontsize=11)
        else:
            self.ax.set_yscale("log")
            self.ax.set_ylabel(f"{pretty_name(param)} (skala log)", fontsize=11)
    else:
        if plot_type != "Histogram":
            self.ax.set_ylabel(pretty_name(param), fontsize=11)

    if p < 0.001:
        title = f"{pretty_name(param)}\np < 0.001 ***"
    elif p < 0.01:
        title = f"{pretty_name(param)}\np = {p:.4f} **"
    elif p < 0.05:
        title = f"{pretty_name(param)}\np = {p:.4f} *"
    else:
        title = f"{pretty_name(param)}\np = {p:.4f} (ns)"

    self.ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    self.ax.grid(True, alpha=0.3, linestyle="--")

    stats_text = (
        f"Przyjęci:\n"
        f"n = {len(hosp)}\n"
        f"śr = {hosp.mean():.2f} ± {hosp.std():.2f}\n"
        f"mediana = {np.median(hosp):.2f}\n\n"
        f"Wypisani:\n"
        f"n = {len(dom)}\n"
        f"śr = {dom.mean():.2f} ± {dom.std():.2f}\n"
        f"mediana = {np.median(dom):.2f}"
    )

    self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), fontsize=9)
    self.figure.tight_layout()
    self.canvas.draw()


def _patched_generuj_wszystkie_wykresy(self):
    if self.df_hosp is None:
        messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj dane!")
        return

    plot_type = self.plot_type_var.get()
    folder = filedialog.askdirectory(title="Wybierz folder do zapisu wszystkich wykresów")
    if not folder:
        return

    zapisane = 0
    pominiete = 0
    for param in self.parametry_kliniczne:
        if param not in self.df_hosp.columns or param not in self.df_dom.columns:
            pominiete += 1
            continue
        hosp = self.df_hosp[param].dropna()
        dom = self.df_dom[param].dropna()
        if len(hosp) == 0 or len(dom) == 0:
            pominiete += 1
            continue

        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
        ax.set_facecolor("#f8f9fa")
        _, p = stats.mannwhitneyu(hosp, dom, alternative="two-sided")
        log_scale = param in ["troponina I (0-7,8))", "crp(0-0,5)"]

        if plot_type in ["Boxplot + punkty", "Boxplot"]:
            bp = ax.boxplot([hosp, dom], labels=["PRZYJĘCI", "WYPISANI"], patch_artist=True,
                            medianprops={"color": "black", "linewidth": 2})
            bp["boxes"][0].set_facecolor(KOLORY["hosp"])
            bp["boxes"][0].set_alpha(0.8)
            bp["boxes"][1].set_facecolor(KOLORY["dom"])
            bp["boxes"][1].set_alpha(0.8)
            if plot_type == "Boxplot + punkty":
                x_hosp = np.random.normal(1, 0.05, len(hosp))
                x_dom = np.random.normal(2, 0.05, len(dom))
                ax.scatter(x_hosp, hosp, alpha=0.5, color="darkred", s=30, zorder=3)
                ax.scatter(x_dom, dom, alpha=0.5, color="darkblue", s=30, zorder=3)
        elif plot_type == "Histogram":
            bins = 15
            ax.hist(hosp, bins=bins, alpha=0.6, label="PRZYJĘCI", color=KOLORY["hosp"])
            ax.hist(dom, bins=bins, alpha=0.6, label="WYPISANI", color=KOLORY["dom"])
            ax.set_xlabel(pretty_name(param), fontsize=11)
            ax.set_ylabel("Liczba pacjentów", fontsize=11)
            ax.legend()
        elif plot_type == "Violinplot":
            parts = ax.violinplot([hosp, dom], positions=[1, 2], showmeans=True, showmedians=True)
            for i, body in enumerate(parts["bodies"]):
                body.set_alpha(0.7)
                body.set_facecolor(KOLORY["hosp"] if i == 0 else KOLORY["dom"])
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["PRZYJĘCI", "WYPISANI"])
            ax.set_ylabel(pretty_name(param), fontsize=11)

        if log_scale:
            if plot_type == "Histogram":
                ax.set_xscale("log")
                ax.set_xlabel(f"{pretty_name(param)} (skala log)", fontsize=11)
            else:
                ax.set_yscale("log")
                ax.set_ylabel(f"{pretty_name(param)} (skala log)", fontsize=11)
        else:
            if plot_type != "Histogram":
                ax.set_ylabel(pretty_name(param), fontsize=11)

        if p < 0.001:
            title = f"{pretty_name(param)}\np < 0.001 ***"
        elif p < 0.01:
            title = f"{pretty_name(param)}\np = {p:.4f} **"
        elif p < 0.05:
            title = f"{pretty_name(param)}\np = {p:.4f} *"
        else:
            title = f"{pretty_name(param)}\np = {p:.4f} (ns)"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.grid(True, alpha=0.3, linestyle="--")
        stats_text = (
            f"Przyjęci:\n"
            f"n = {len(hosp)}\n"
            f"śr = {hosp.mean():.2f} ± {hosp.std():.2f}\n"
            f"mediana = {np.median(hosp):.2f}\n\n"
            f"Wypisani:\n"
            f"n = {len(dom)}\n"
            f"śr = {dom.mean():.2f} ± {dom.std():.2f}\n"
            f"mediana = {np.median(dom):.2f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), fontsize=9)
        fig.tight_layout()
        safe_name = (param.replace("/", "_").replace("\\", "_").replace(":", "_")
                     .replace("*", "_").replace("?", "_").replace('"', "_")
                     .replace("<", "_").replace(">", "_").replace("|", "_").replace(" ", "_"))
        nazwa_pliku = f"wykres_{safe_name}_{plot_type.replace(' ', '_').replace('+', 'plus')}.png"
        sciezka = os.path.join(folder, nazwa_pliku)
        fig.savefig(sciezka, dpi=300, bbox_inches="tight")
        plt.close(fig)
        zapisane += 1

    messagebox.showinfo("✅ Gotowe", f"Wygenerowano {zapisane} wykresów.\nPominięto {pominiete} parametrów bez danych.\n\nFolder:\n{folder}")


def _patched_zapisz_wykres(self):
    if self.current_param is None:
        messagebox.showwarning("⚠️ Uwaga", "Najpierw wygeneruj wykres!")
        return
    filename = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
                                            initialfile=f"wykres_{self.current_param}.png")
    if filename:
        try:
            self.figure.savefig(filename, dpi=300, bbox_inches="tight")
            messagebox.showinfo("✅ Sukces", f"Wykres zapisany jako:\n{filename}")
        except Exception as e:
            messagebox.showerror("❌ Błąd", f"Nie udało się zapisać pliku:\n{e}")


def _patched_tab5_raport(self):
    main_frame = ttk.Frame(self.tab5, padding="20")
    main_frame.pack(fill="both", expand=True)
    btn_frame = tk.Frame(main_frame, bg=KOLORY["light"])
    btn_frame.pack(fill="x", pady=(0, 20))
    btn_style = {"font": ("Helvetica", 12, "bold"), "fg": "white", "activebackground": KOLORY["accent2"],
                 "activeforeground": "white", "relief": "flat", "bd": 0, "padx": 25, "pady": 12, "cursor": "hand2"}
    btn1 = tk.Button(btn_frame, text="📊 GENERUJ RAPORT", bg=KOLORY["accent1"], command=self.generuj_raport, **btn_style)
    btn1.pack(side="left", padx=10)
    btn2 = tk.Button(btn_frame, text="💾 EKSPORTUJ DO CSV", bg=KOLORY["success"], command=self.export_csv, **btn_style)
    btn2.pack(side="left", padx=10)
    btn3 = tk.Button(btn_frame, text="🔄 ODŚWIEŻ", bg=KOLORY["warning"], command=self.odswiez_raport, **btn_style)
    btn3.pack(side="left", padx=10)
    report_frame = tk.LabelFrame(main_frame, text="📋 RAPORT KOŃCOWY", font=("Helvetica", 14, "bold"),
                                 bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=20, pady=15)
    report_frame.pack(fill="both", expand=True)
    text_frame = tk.Frame(report_frame, bg=KOLORY["light"])
    text_frame.pack(fill="both", expand=True)
    self.report_text = tk.Text(text_frame, font=("Courier", 11), bg="white", fg=KOLORY["dark"], wrap="word", padx=15, pady=15)
    self.report_text.pack(side="left", fill="both", expand=True)
    scrollbar = tk.Scrollbar(text_frame, command=self.report_text.yview)
    scrollbar.pack(side="right", fill="y")
    self.report_text.config(yscrollcommand=scrollbar.set)


def _patched_generuj_raport(self):
    if self.df_hosp is None:
        messagebox.showwarning("⚠️ Uwaga", "Brak danych do wygenerowania raportu!")
        return
    self.odswiez_raport()
    filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Pliki tekstowe", "*.txt")], initialfile="raport_medyczny.txt")
    if filename:
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.report_text.get("1.0", tk.END))
            messagebox.showinfo("✅ Sukces", f"Raport zapisany jako:\n{filename}")
        except Exception as e:
            messagebox.showerror("❌ Błąd", f"Nie udało się zapisać pliku:\n{e}")


def _patched_odswiez_raport(self):
    self.report_text.delete("1.0", tk.END)
    if self.df_hosp is None:
        self.report_text.insert("1.0", "Brak danych. Wczytaj plik w zakładce 'WCZYTAJ DANE'.")
        return
    # użyj istniejącej logiki raportu z klasy jeśli dostępna w poprzednich wersjach nie była uszkodzona
    wyniki = []
    istotne = []
    for param in self.parametry_kliniczne:
        if param in self.df_hosp.columns and param in self.df_dom.columns:
            hosp = self.df_hosp[param].dropna()
            dom = self.df_dom[param].dropna()
            if len(hosp) > 0 and len(dom) > 0:
                _, p = stats.mannwhitneyu(hosp, dom, alternative="two-sided")
                roznica = hosp.mean() - dom.mean()
                wyniki.append({"parametr": param, "hosp_sr": hosp.mean(), "dom_sr": dom.mean(), "p": p, "roznica": roznica})
                if p < 0.05:
                    istotne.append((param, p, roznica))
    istotne.sort(key=lambda x: x[1])
    raport = f"""
╔══════════════════════════════════════════════════════════════════╗
║              RAPORT KOŃCOWY ANALIZY MEDYCZNEJ                    ║
╚══════════════════════════════════════════════════════════════════╝

📅 Data raportu: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
🔧 Tryb analizy: {self.current_mode.upper()}
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
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
        kierunek = "⬆️ WYŻSZE" if roznica > 0 else "⬇️ NIŻSZE"
        raport += f"\n  {i}. {pretty_name(param):<25}\n"
        raport += f"     p = {p:.6f} {stars}\n"
        raport += f"     {kierunek} u przyjętych (różnica średnich: {roznica:+.2f})\n"
    if self.prediction_pipeline is not None:
        raport += f"""
{'='*70}
🧮 KALKULATOR HOSPITALIZACJI:
────────────────────────────────────────────────────────────────────
{self.prediction_model_info}
"""
    raport += f"""
{'='*70}
✅ ANALIZA ZAKOŃCZONA POMYŚLNIE
{'='*70}
"""
    self.report_text.insert("1.0", raport)


def _patched_export_csv(self):
    if self.wyniki_df is None:
        messagebox.showwarning("⚠️ Uwaga", "Najpierw wykonaj analizę!")
        return
    filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="wyniki_analizy.csv")
    if filename:
        try:
            self.wyniki_df.to_csv(filename, sep=";", index=False, encoding="utf-8")
            messagebox.showinfo("✅ Sukces", f"Wyniki zapisane jako:\n{filename}")
        except Exception as e:
            messagebox.showerror("❌ Błąd", f"Nie udało się zapisać pliku:\n{e}")


def _patched_tab6_info(self):
    frame = ttk.Frame(self.tab6, padding="30")
    frame.pack(fill="both", expand=True)
    info_text = """
╔══════════════════════════════════════════════════════════════╗
║         ANALIZATOR DANYCH MEDYCZNYCH - WERSJA 12.0          ║
╚══════════════════════════════════════════════════════════════╝

📋 OPIS PROGRAMU:
──────────────────────────────────────────────────────────────
Program służy do porównawczej analizy danych medycznych
pomiędzy pacjentami przyjętymi do szpitala a wypisanymi do domu.

🧮 NOWOŚĆ:
──────────────────────────────────────────────────────────────
• kalkulator prawdopodobieństwa hospitalizacji
• ROC porównawcze wielu parametrów
• decision curve analysis
• eksport raportu do PDF
• zapis i odczyt modelu

👩‍⚕️ AUTOR:
──────────────────────────────────────────────────────────────
Aneta
Wersja 12.0
"""
    label = tk.Label(frame, text=info_text, font=("Courier", 11), bg="white", fg=KOLORY["dark"], justify="left", padx=30, pady=30)
    label.pack(fill="both", expand=True)



# =============================================================================
# DODATKOWE METODY POMOCNICZE I PODPIĘCIA
# =============================================================================

def _helper_oblicz_prob_z_modelu_i_df(self, df_in):
    df_pred = df_in.copy()
    for col in ZMIENNE_LOG:
        if col in df_pred.columns:
            df_pred[f"log_{col}"] = np.log1p(df_pred[col].clip(lower=0))

    valid_mask = pd.Series(True, index=df_pred.index)
    values = []
    for feat in self.prediction_features:
        if feat not in df_pred.columns:
            valid_mask &= False
        else:
            valid_mask &= df_pred[feat].notna()
    X = df_pred.loc[valid_mask, self.prediction_features].astype(float)
    probs = pd.Series(index=df_pred.index, dtype=float)
    if len(X) > 0:
        probs.loc[valid_mask] = self.prediction_pipeline.predict_proba(X.values)[:, 1]
    return df_pred, probs, valid_mask


def _helper_raport_brakow_pro(self, df):
    wyniki = []
    for col in df.columns:
        n_brakow = int(df[col].isna().sum())
        proc = (n_brakow / len(df)) * 100 if len(df) > 0 else 0
        wyniki.append({"kolumna": col, "braki": n_brakow, "procent": round(proc, 2)})
    return pd.DataFrame(wyniki)


def _helper_walidacja_zakresow(self, df):
    wyniki = []
    for col, (min_bio, max_bio) in ZAKRESY_BIOLOGICZNE.items():
        if col in df.columns:
            dane = df[col].dropna()
            if len(dane) > 0:
                mask = (dane < min_bio) | (dane > max_bio)
                wyniki.append({"kolumna": col, "poza_zakresem": int(mask.sum()), "min_bio": min_bio, "max_bio": max_bio})
    return pd.DataFrame(wyniki)


def _helper_tabela_1_pro(self, df):
    wyniki = []
    for param in PARAMETRY_KLINICZNE:
        if param in df.columns:
            hosp = df[df["outcome"] == 1][param].dropna()
            dom = df[df["outcome"] == 0][param].dropna()
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
            hosp = df[df["outcome"] == 1][choroba].dropna()
            dom = df[df["outcome"] == 0][choroba].dropna()
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


def _helper_analiza_jednoczynnikowa_pro(self, df):
    wyniki = []
    p_values = []
    for param in PARAMETRY_KLINICZNE:
        if param in df.columns:
            hosp = df[df["outcome"] == 1][param].dropna()
            dom = df[df["outcome"] == 0][param].dropna()
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


def _helper_missingness_top_pro(self, top_param):
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


def _helper_progi_kliniczne_pro(self, df, top_param):
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
            wyniki.append({"parametr": param, "etykieta": pretty_name(param), "kierunek": kierunek, "prog": prog, "czulosc": sens, "swoistosc": spec})
        except Exception:
            continue
    return pd.DataFrame(wyniki)


def _helper_przygotuj_zmienne_modelu(self, df):
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


def _helper_model_podstawowy(self, df):
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
    except Exception:
        return None, None, None


def _helper_model_rozszerzony(self, df, zmienne):
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
    except Exception:
        return None, None, None, False, None


def _helper_model_z_redukcja(self, df, zmienne):
    dostepne = [z for z in zmienne if z in df.columns]
    df_cc = df[dostepne + ["outcome"]].dropna()
    n_events = int(df_cc["outcome"].sum())
    max_pred = int(n_events / 10)
    if max_pred < 1:
        return None, None, None, False, None
    if len(dostepne) <= max_pred:
        return self._model_rozszerzony(df, zmienne)
    priorytety = {"wiek": 10, "log_crp(0-0,5)": 9, "SpO2": 8, "log_kreatynina(0,5-1,2)": 7, "MAP": 6, "log_troponina I (0-7,8))": 5, "HGB(12,4-15,2)": 4}
    dostepne = sorted(dostepne, key=lambda x: priorytety.get(x, 0), reverse=True)
    wybrane = dostepne[:max_pred]
    return self._model_rozszerzony(df, wybrane)


def _helper_forest_plot(self, wyniki, nazwa_pliku):
    if wyniki is None or len(wyniki) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(wyniki))
    ax.errorbar(wyniki["OR"], y_pos, xerr=[wyniki["OR"] - wyniki["ci_low"], wyniki["ci_high"] - wyniki["OR"]], fmt="o", capsize=4)
    ax.axvline(1, linestyle="--")
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wyniki["etykieta"])
    ax.set_xlabel("OR (95% CI)")
    ax.set_title("Niezależne czynniki związane z hospitalizacją")
    plt.tight_layout()
    plt.savefig(nazwa_pliku, dpi=300, bbox_inches="tight")
    plt.close()


def _helper_model_predykcyjny(self, df, zmienne):
    dostepne = [z for z in zmienne if z in df.columns]
    df_cc = df[dostepne + ["outcome"]].dropna()
    if len(df_cc) < 20:
        return None
    X = df_cc[dostepne].values
    y = df_cc["outcome"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000, random_state=42))])
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
    auc_ci = (np.percentile(auc_boot, 2.5), np.percentile(auc_boot, 97.5)) if len(auc_boot) else (roc_auc, roc_auc)
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
    return {"auc": roc_auc, "auc_ci_low": auc_ci[0], "auc_ci_high": auc_ci[1], "brier": brier, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(), "fig_roc": fig1, "fig_cal": fig2}


def _patched_tab5_raport_v13(self):
    main_frame = ttk.Frame(self.tab5, padding="20")
    main_frame.pack(fill="both", expand=True)
    btn_frame = tk.Frame(main_frame, bg=KOLORY["light"])
    btn_frame.pack(fill="x", pady=(0, 20))
    btn_style = {"font": ("Helvetica", 12, "bold"), "fg": "white", "activebackground": KOLORY["accent2"], "activeforeground": "white", "relief": "flat", "bd": 0, "padx": 25, "pady": 12, "cursor": "hand2"}
    btn1 = tk.Button(btn_frame, text="📊 GENERUJ RAPORT", bg=KOLORY["accent1"], **btn_style)
    btn1.config(command=self.generuj_raport)
    btn1.pack(side="left", padx=10)
    btn2 = tk.Button(btn_frame, text="💾 EKSPORTUJ DO CSV", bg=KOLORY["success"], **btn_style)
    btn2.config(command=self.export_csv)
    btn2.pack(side="left", padx=10)
    btn3 = tk.Button(btn_frame, text="🔄 ODŚWIEŻ", bg=KOLORY["warning"], **btn_style)
    btn3.config(command=self.odswiez_raport)
    btn3.pack(side="left", padx=10)
    btn4 = tk.Button(btn_frame, text="📄 EKSPORTUJ PDF", bg=KOLORY["primary"], **btn_style)
    btn4.config(command=self.eksportuj_pdf)
    btn4.pack(side="left", padx=10)
    report_frame = tk.LabelFrame(main_frame, text="📋 RAPORT KOŃCOWY", font=("Helvetica", 14, "bold"), bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=20, pady=15)
    report_frame.pack(fill="both", expand=True)
    text_frame = tk.Frame(report_frame, bg=KOLORY["light"])
    text_frame.pack(fill="both", expand=True)
    self.report_text = tk.Text(text_frame, font=("Courier", 11), bg="white", fg=KOLORY["dark"], wrap="word", padx=15, pady=15)
    self.report_text.pack(side="left", fill="both", expand=True)
    scrollbar = tk.Scrollbar(text_frame, command=self.report_text.yview)
    scrollbar.pack(side="right", fill="y")
    self.report_text.config(yscrollcommand=scrollbar.set)


def eksportuj_pdf(self):
    if self.df_hosp is None:
        messagebox.showwarning("⚠️ Uwaga", "Brak danych do wygenerowania PDF!")
        return
    self.odswiez_raport()
    filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile="raport_medyczny.pdf")
    if not filename:
        return
    try:
        c = pdf_canvas.Canvas(filename, pagesize=A4)
        width, height = A4
        margin = 1.5 * cm
        y = height - margin
        line_height = 14
        font_name = "Courier"
        font_size = 9
        c.setFont(font_name, font_size)
        text_lines = self.report_text.get("1.0", tk.END).splitlines()
        max_width = width - 2 * margin
        for raw_line in text_lines:
            line = raw_line.expandtabs(4)
            if line == "":
                y -= line_height
                if y < margin:
                    c.showPage(); c.setFont(font_name, font_size); y = height - margin
                continue
            while stringWidth(line, font_name, font_size) > max_width:
                cut = max(1, int(len(line) * max_width / max(stringWidth(line, font_name, font_size), 1)))
                subline = line[:cut]
                while stringWidth(subline, font_name, font_size) > max_width and len(subline) > 1:
                    subline = subline[:-1]
                c.drawString(margin, y, subline)
                y -= line_height
                line = line[len(subline):]
                if y < margin:
                    c.showPage(); c.setFont(font_name, font_size); y = height - margin
            c.drawString(margin, y, line)
            y -= line_height
            if y < margin:
                c.showPage(); c.setFont(font_name, font_size); y = height - margin
        c.save()
        messagebox.showinfo("✅ Sukces", f"Raport PDF zapisany jako:\n{filename}")
    except Exception as e:
        messagebox.showerror("❌ Błąd", f"Nie udało się zapisać PDF:\n{e}")


def pokaz_waznosc_modelu(self):
    if self.prediction_pipeline is None or len(getattr(self, 'prediction_features', [])) == 0:
        messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj dane i zbuduj model.")
        return
    try:
        logreg = self.prediction_pipeline.named_steps["logreg"]
        coef = logreg.coef_[0]
        odds = np.exp(coef)
        df_imp = pd.DataFrame({
            "etykieta": [pretty_name(x) for x in self.prediction_features],
            "beta_stand": coef,
            "OR_na_1SD": odds,
            "kierunek": ["zwiększa ryzyko" if c > 0 else "zmniejsza ryzyko" for c in coef]
        }).sort_values("beta_stand", key=lambda s: s.abs(), ascending=False)
        msg = "WAŻNOŚĆ CECH MODELU\n" + "="*60 + "\n" + "\n".join(
            [f"• {r.etykieta}: beta={r.beta_stand:.3f} | OR/1SD={r.OR_na_1SD:.2f} | {r.kierunek}" for r in df_imp.itertuples()]
        )
        win = tk.Toplevel(self.root)
        win.title("🧠 Ważność cech modelu")
        win.geometry("900x700")
        txt = tk.Text(win, wrap="word", font=("Courier", 11), bg="white")
        txt.pack(fill="x")
        txt.insert("1.0", msg)
        fig = plt.Figure(figsize=(9, 4.5), dpi=100)
        ax = fig.add_subplot(111)
        show = df_imp.head(10).iloc[::-1]
        ax.barh(show["etykieta"], show["beta_stand"])
        ax.set_title("Feature importance (współczynniki standaryzowane)")
        ax.set_xlabel("Beta")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    except Exception as e:
        messagebox.showerror("❌ Błąd", f"Nie udało się pokazać ważności cech:\n{e}")

MedicalAnalyzerGUI._oblicz_prob_z_modelu_i_df = _helper_oblicz_prob_z_modelu_i_df
MedicalAnalyzerGUI._raport_brakow_pro = _helper_raport_brakow_pro
MedicalAnalyzerGUI._walidacja_zakresow = _helper_walidacja_zakresow
MedicalAnalyzerGUI._tabela_1_pro = _helper_tabela_1_pro
MedicalAnalyzerGUI._analiza_jednoczynnikowa_pro = _helper_analiza_jednoczynnikowa_pro
MedicalAnalyzerGUI._missingness_top_pro = _helper_missingness_top_pro
MedicalAnalyzerGUI._progi_kliniczne_pro = _helper_progi_kliniczne_pro
MedicalAnalyzerGUI._przygotuj_zmienne_modelu = _helper_przygotuj_zmienne_modelu
MedicalAnalyzerGUI._model_podstawowy = _helper_model_podstawowy
MedicalAnalyzerGUI._model_rozszerzony = _helper_model_rozszerzony
MedicalAnalyzerGUI._model_z_redukcja = _helper_model_z_redukcja
MedicalAnalyzerGUI._forest_plot = _helper_forest_plot
MedicalAnalyzerGUI._model_predykcyjny = _helper_model_predykcyjny
MedicalAnalyzerGUI.analiza_profesjonalna = MedicalAnalyzerGUI._patched_analiza_profesjonalna
MedicalAnalyzerGUI.zapisz_model_do_pliku = MedicalAnalyzerGUI._patched_zapisz_model_do_pliku
MedicalAnalyzerGUI.wczytaj_model_z_pliku = MedicalAnalyzerGUI._patched_wczytaj_model_z_pliku
MedicalAnalyzerGUI.zastosuj_model_do_pliku = MedicalAnalyzerGUI._patched_zastosuj_model_do_pliku
MedicalAnalyzerGUI.rysuj_wykres = _patched_rysuj_wykres
MedicalAnalyzerGUI.generuj_wszystkie_wykresy = _patched_generuj_wszystkie_wykresy
MedicalAnalyzerGUI.zapisz_wykres = _patched_zapisz_wykres
MedicalAnalyzerGUI.tab5_raport = _patched_tab5_raport_v13
MedicalAnalyzerGUI.generuj_raport = _patched_generuj_raport
MedicalAnalyzerGUI.odswiez_raport = _patched_odswiez_raport
MedicalAnalyzerGUI.export_csv = _patched_export_csv
MedicalAnalyzerGUI.tab6_info = _patched_tab6_info
MedicalAnalyzerGUI.eksportuj_pdf = eksportuj_pdf
MedicalAnalyzerGUI.pokaz_waznosc_modelu = pokaz_waznosc_modelu

# =============================================================================
# URUCHOMIENIE
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalAnalyzerGUI(root)
    root.mainloop()
