# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:13:15 2026

@author: aneta
"""

"""
✨ PRZEPIĘKNE GUI - ANALIZA DANYCH MEDYCZNYCH ✨
Wersja: 14.2 - REFAKTORYZACJA + PEŁNA FUNKCJONALNOŚĆ
Autor: Aneta
"""

import os
import sys
import sqlite3
import warnings
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any
import joblib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd

# Bezpieczne ustawienie backendu Matplotlib
def _ustaw_backend_matplotlib():
    try:
        if 'spyder' in sys.modules:
            return
        import matplotlib
        if matplotlib.get_backend() != 'TkAgg':
            matplotlib.use('TkAgg')
    except Exception:
        pass

_ustaw_backend_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

from sklearn.metrics import roc_curve, auc, brier_score_loss, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as pdf_canvas

warnings.filterwarnings("ignore")

# =============================================================================
# STAŁE GLOBALNE
# =============================================================================

# Nazwy techniczne (odporne na zmiany w plikach wejściowych)
NAZWY_TECHNICZNE = {
    "troponina I (0-7,8))": "troponina_i",
    "kreatynina(0,5-1,2)": "kreatynina",
    "crp(0-0,5)": "crp",
    "HGB(12,4-15,2)": "hgb",
    "WBC(4-11)": "wbc",
    "plt(130-450)": "plt",
    "hct(38-45)": "hct",
    "Na(137-145)": "sod",
    "K(3,5-5,1)": "potas",
}

PARAMETRY_KLINICZNE = [
    "wiek", "MAP", "SpO2", "AS", "mleczany",
    "kreatynina", "troponina_i",
    "hgb", "wbc", "plt",
    "hct", "sod", "potas", "crp"
]

CHOROBY = ["dm", "wątroba", "naczyniowe", "zza", "npl"]

ZMIENNE_OBOWIAZKOWE = ["wiek"]
ZMIENNE_DODATKOWE = [
    "SpO2", "crp", "kreatynina", "MAP", "troponina_i", "hgb"
]

ZMIENNE_LOG = ["crp", "troponina_i", "kreatynina"]

ZAKRESY_BIOLOGICZNE = {
    "wiek": (0, 120), "RR": (0, 300), "MAP": (0, 200), "SpO2": (0, 100),
    "AS": (0, 300), "mleczany": (0, 30), "kreatynina": (0, 20),
    "troponina_i": (0, 100000), "hgb": (0, 25), "wbc": (0, 100),
    "plt": (0, 2000), "hct": (0, 70), "sod": (100, 160), "potas": (2, 8), "crp": (0, 500),
}

ETYKIETY = {
    "wiek": "Wiek, lata", "MAP": "Średnie ciśnienie tętnicze, mmHg",
    "SpO2": "Saturacja, %", "AS": "Akcja serca / min", "mleczany": "Mleczany, mmol/L",
    "kreatynina": "Kreatynina, mg/dL", "troponina_i": "Troponina I",
    "hgb": "Hemoglobina, g/dL", "wbc": "Leukocyty, G/L", "plt": "Płytki, G/L",
    "hct": "Hematokryt, %", "sod": "Sód, mmol/L", "potas": "Potas, mmol/L", "crp": "CRP, mg/dL",
    "dm": "Cukrzyca", "wątroba": "Choroba wątroby", "naczyniowe": "Choroby naczyniowe",
    "zza": "Zespół zależności alkoholowej", "npl": "Nowotwór / choroba proliferacyjna",
    "log_crp": "log(CRP)", "log_kreatynina": "log(kreatynina)", "log_troponina_i": "log(troponina I)",
}

WARTOSCI_KRYTYCZNE = {
    "hgb": {"low": 6.0, "opis": "Ciężka niedokrwistość"},
    "SpO2": {"low": 85.0, "opis": "Krytycznie niska saturacja"},
    "MAP": {"low": 60.0, "opis": "Krytycznie niskie ciśnienie perfuzyjne"},
    "potas": {"low": 2.8, "high": 6.0, "opis": "Krytyczne zaburzenie potasu"},
    "sod": {"low": 120.0, "high": 155.0, "opis": "Krytyczne zaburzenie sodu"},
    "kreatynina": {"high": 4.0, "opis": "Ciężkie upośledzenie funkcji nerek"},
    "mleczany": {"high": 4.0, "opis": "Znaczna hiperlaktatemia"},
    "crp": {"high": 15.0, "opis": "Bardzo wysokie CRP"},
    "troponina_i": {"high": 1000.0, "opis": "Bardzo wysoka troponina"},
}

NORMY_REFERENCYJNE_SKALI = {
    "SpO2": {"low": 95.0, "high": 100.0},
    "MAP": {"low": 70.0, "high": 105.0},
    "mleczany": {"low": 0.0, "high": 2.0},
}

KOLORY = {
    "primary": "#2c3e50", "secondary": "#34495e", "accent1": "#e74c3c",
    "accent2": "#3498db", "success": "#2ecc71", "warning": "#f39c12",
    "light": "#ecf0f1", "dark": "#2c3e50", "hosp": "#e74c3c", "dom": "#3498db",
    "bg": "#f5f5f5", "fg": "#2c3e50",
}

BOOTSTRAP_ITERATIONS = {"szybki": 100, "dokladny": 500}
DEFAULT_BOOTSTRAP_MODE = "szybki"

# =============================================================================
# FUNKCJE POMOCNICZE (poza klasą)
# =============================================================================

def pretty_name(x):
    return ETYKIETY.get(x, x)

def nazwa_techniczna(x):
    odwrotne = {v: k for k, v in NAZWY_TECHNICZNE.items()}
    return odwrotne.get(x, x)

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

def bezpieczna_liczba(x):
    if x is None or x == "":
        return np.nan
    try:
        return float(str(x).replace(",", "."))
    except (ValueError, TypeError):
        return np.nan

def konwertuj_kolumne_na_liczby(s: pd.Series) -> pd.Series:
    """Bezpiecznie konwertuje zapisy laboratoryjne na liczby.

    Obsługuje przecinki dziesiętne oraz zapisy typu <0.01 / >100.
    Wartości nieliczbowe zamienia na NaN, żeby analiza nie przerywała pracy.
    """
    return (
        s.astype(str)
         .str.strip()
         .str.replace(",", ".", regex=False)
         .str.replace("<", "", regex=False)
         .str.replace(">", "", regex=False)
         .str.replace(" ", "", regex=False)
         .replace(["", "nan", "None", "brak", "Brak", "NULL", "NaN"], np.nan)
         .pipe(pd.to_numeric, errors="coerce")
    )

def okresl_kategorie_ryzyka(p):
    if p < 0.20:
        return "NISKIE"
    if p < 0.50:
        return "UMIARKOWANE"
    if p < 0.80:
        return "WYSOKIE"
    return "BARDZO WYSOKIE"

def dynamiczny_n_splits(y):
    y_clean = y.dropna() if hasattr(y, 'dropna') else y
    y_clean = np.array(y_clean)
    try:
        unique, counts = np.unique(y_clean, return_counts=True)
        if len(unique) < 2:
            return 2
        n_min_class = min(counts)
        return min(5, max(2, n_min_class))
    except Exception:
        return 3

def sprawdz_epv_i_raport(df, zmienne, outcome="outcome", prog=10):
    n_events = int(df[outcome].sum())
    n_vars = len(zmienne)
    epv = n_events / n_vars if n_vars > 0 else 0
    return epv >= prog, epv

def sprawdz_vif(X, include_constant=False):
    X_clean = X.copy()
    if not include_constant and X_clean.shape[1] > 0:
        first_col = X_clean.iloc[:, 0]
        if np.allclose(first_col, 1):
            X_clean = X_clean.iloc[:, 1:]
    if X_clean.shape[1] < 2:
        return pd.DataFrame({"zmienna": [], "VIF": []})
    vif_data = pd.DataFrame()
    vif_data["zmienna"] = X_clean.columns
    vif_data["VIF"] = [variance_inflation_factor(X_clean.values, i) for i in range(X_clean.shape[1])]
    return vif_data

def wyniki_modelu_statsmodels(model, zmienne):
    rows = []
    for var in zmienne:
        try:
            ci = model.conf_int().loc[var]
            rows.append({
                "parametr": var, "etykieta": pretty_name(var), "beta": model.params[var],
                "OR": np.exp(model.params[var]), "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1]),
                "CI_95%": f"{np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}", "p_value": model.pvalues[var]
            })
        except (KeyError, IndexError):
            continue
    return pd.DataFrame(rows)

def oblicz_map_z_rr(sbp, dbp):
    sbp = bezpieczna_liczba(sbp)
    dbp = bezpieczna_liczba(dbp)
    if pd.isna(sbp) or pd.isna(dbp):
        return np.nan
    return (sbp + 2 * dbp) / 3

def wykryj_srodowisko():
    if 'spyder' in sys.modules:
        return 'spyder'
    if 'ipykernel' in sys.modules:
        return 'jupyter'
    return 'terminal'


# =============================================================================
# WARSTWA 1: SILNIK STATYSTYCZNY
# =============================================================================

class StatisticsEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_hosp = df[df["outcome"] == 1].copy() if "outcome" in df.columns else None
        self.df_dom = df[df["outcome"] == 0].copy() if "outcome" in df.columns else None
        self.imputation_values = {}

    def oblicz_statystyki_parametru(self, param: str, hosp: pd.Series = None, dom: pd.Series = None) -> Optional[Dict]:
        if hosp is None or dom is None:
            if self.df_hosp is None or self.df_dom is None:
                return None
            if param not in self.df_hosp.columns or param not in self.df_dom.columns:
                return None
            hosp = self.df_hosp[param].dropna()
            dom = self.df_dom[param].dropna()
        if len(hosp) == 0 or len(dom) == 0:
            return None
        try:
            _, p_value = stats.mannwhitneyu(hosp, dom, alternative="two-sided")
        except Exception:
            p_value = 1.0
        d = cliff_delta(hosp, dom)
        if p_value < 0.001:
            stars, tag = "***", "highly"
        elif p_value < 0.01:
            stars, tag = "**", "significant"
        elif p_value < 0.05:
            stars, tag = "*", "significant"
        else:
            stars, tag = "ns", ""
        return {
            "p_value": p_value, "cliff_delta": d, "effect_size": interpret_cliff_delta(d),
            "stars": stars, "tag": tag, "hosp_n": len(hosp), "dom_n": len(dom),
            "hosp_mean": hosp.mean(), "hosp_std": hosp.std(), "dom_mean": dom.mean(), "dom_std": dom.std(),
            "hosp_median": hosp.median(), "dom_median": dom.median(),
            "hosp_q1": hosp.quantile(0.25), "hosp_q3": hosp.quantile(0.75),
            "dom_q1": dom.quantile(0.25), "dom_q3": dom.quantile(0.75),
        }

    def oblicz_tabele1(self) -> pd.DataFrame:
        wyniki = []
        for param in PARAMETRY_KLINICZNE:
            if param not in self.df.columns:
                continue
            stats_dict = self.oblicz_statystyki_parametru(param)
            if stats_dict:
                wyniki.append({
                    "parametr": param, "etykieta": pretty_name(param),
                    "hospitalizowani": f"{stats_dict['hosp_median']:.2f} [{stats_dict['hosp_q1']:.2f}-{stats_dict['hosp_q3']:.2f}]",
                    "wypisani": f"{stats_dict['dom_median']:.2f} [{stats_dict['dom_q1']:.2f}-{stats_dict['dom_q3']:.2f}]",
                    "p_value": stats_dict['p_value'], "effect_size": stats_dict['cliff_delta'],
                    "interpretacja": stats_dict['effect_size']
                })
        for choroba in CHOROBY:
            if choroba not in self.df.columns or self.df_hosp is None or self.df_dom is None:
                continue
            hosp = self.df_hosp[choroba].dropna()
            dom = self.df_dom[choroba].dropna()
            if len(hosp) > 0 and len(dom) > 0:
                hosp_tak = int((hosp == 1).sum())
                dom_tak = int((dom == 1).sum())
                tabela = [[hosp_tak, len(hosp) - hosp_tak], [dom_tak, len(dom) - dom_tak]]
                try:
                    _, p = fisher_exact(tabela)
                except Exception:
                    p = 1.0
                a, b, c, d = hosp_tak + 0.5, len(hosp) - hosp_tak + 0.5, dom_tak + 0.5, len(dom) - dom_tak + 0.5
                or_val = (a * d) / (b * c) if (b * c) > 0 else float('inf')
                wyniki.append({
                    "parametr": choroba, "etykieta": pretty_name(choroba),
                    "hospitalizowani": f"{hosp_tak}/{len(hosp)} ({100*hosp_tak/len(hosp):.1f}%)",
                    "wypisani": f"{dom_tak}/{len(dom)} ({100*dom_tak/len(dom):.1f}%)",
                    "p_value": p, "effect_size": or_val, "interpretacja": "OR"
                })
        return pd.DataFrame(wyniki)

    def analiza_jednoczynnikowa_z_fdr(self) -> Tuple[pd.DataFrame, List[str]]:
        wyniki, p_values = [], []
        for param in PARAMETRY_KLINICZNE:
            if param not in self.df.columns:
                continue
            stats_dict = self.oblicz_statystyki_parametru(param)
            if stats_dict:
                wyniki.append({
                    "parametr": param, "etykieta": pretty_name(param), "p_raw": stats_dict['p_value'],
                    "cliff_delta": stats_dict['cliff_delta'], "interpretacja": stats_dict['effect_size'],
                    "n_hosp": stats_dict['hosp_n'], "n_dom": stats_dict['dom_n'],
                    "hosp_median": stats_dict['hosp_median'], "hosp_q1": stats_dict['hosp_q1'], "hosp_q3": stats_dict['hosp_q3'],
                    "dom_median": stats_dict['dom_median'], "dom_q1": stats_dict['dom_q1'], "dom_q3": stats_dict['dom_q3']
                })
                p_values.append(stats_dict['p_value'])
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

    def przygotuj_pipeline_imputacji(self, features: List[str]) -> Pipeline:
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('logreg', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ])

    def zbuduj_model_hospitalizacji(self, progress_callback=None) -> Dict:
        wszystkie = ZMIENNE_OBOWIAZKOWE + ZMIENNE_DODATKOWE
        feature_list = []
        for col in wszystkie:
            if col in self.df.columns:
                if col in ZMIENNE_LOG:
                    new_name = f"log_{col}"
                    self.df[col] = konwertuj_kolumne_na_liczby(self.df[col])
                    self.df[new_name] = np.log1p(self.df[col].clip(lower=0))
                    feature_list.append(new_name)
                else:
                    feature_list.append(col)
        final_features = list(dict.fromkeys(feature_list))
        if len(final_features) == 0:
            return {"success": False, "error": "Brak odpowiednich zmiennych"}
        df_train, df_holdout = train_test_split(
            self.df, test_size=0.3, random_state=42,
            stratify=self.df["outcome"] if len(self.df["outcome"].unique()) > 1 else None
        )
        if len(df_train["outcome"].unique()) < 2:
            return {"success": False, "error": "Zbiór treningowy ma tylko jedną klasę"}
        X_train, y_train = df_train[final_features].copy(), df_train["outcome"].copy()
        X_holdout, y_holdout = df_holdout[final_features].copy(), df_holdout["outcome"].copy()
        pipeline = self.przygotuj_pipeline_imputacji(final_features)
        pipeline.fit(X_train, y_train)
        n_splits = dynamiczny_n_splits(y_train)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        y_prob_holdout = pipeline.predict_proba(X_holdout)[:, 1]
        auc_holdout = roc_auc_score(y_holdout, y_prob_holdout)
        ap_holdout = average_precision_score(y_holdout, y_prob_holdout)
        X_all, y_all = self.df[final_features].copy(), self.df["outcome"].copy()
        y_prob_all = pipeline.predict_proba(X_all)[:, 1]
        auc_all = roc_auc_score(y_all, y_prob_all) if len(np.unique(y_all)) > 1 else 0
        model_info = (
            f"⚠️ UWAGA: To model wewnętrzny (walidacja na holdoucie z tej samej kohorty).\n"
            f"• Zbiór treningowy: n = {len(df_train)} (hospitalizowani: {y_train.sum()})\n"
            f"• Zbiór holdout (wewnętrzna walidacja): n = {len(df_holdout)}\n"
            f"• Liczba cech w modelu: {len(final_features)}\n"
            f"• Cechy: {', '.join([pretty_name(f) for f in final_features])}\n"
            f"• AUC w CV na treningowym: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n"
            f"• AUC na holdoucie: {auc_holdout:.3f}\n"
            f"• Average Precision na holdoucie: {ap_holdout:.3f}\n"
            f"• Model NIE JEST zwalidowany zewnętrznie - wyniki mogą być optymistyczne"
        )
        return {
            "success": True, "pipeline": pipeline, "features": final_features,
            "auc_train_cv": f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
            "auc_holdout": auc_holdout, "auc_all": auc_all, "ap_holdout": ap_holdout,
            "n_train": len(df_train), "n_holdout": len(df_holdout), "n_features": len(final_features),
            "model_info": model_info
        }

    def raport_brakow(self) -> pd.DataFrame:
        wyniki = []
        for col in self.df.columns:
            n_brakow = int(self.df[col].isna().sum())
            proc = (n_brakow / len(self.df)) * 100 if len(self.df) > 0 else 0
            wyniki.append({"kolumna": col, "braki": n_brakow, "procent": round(proc, 2)})
        return pd.DataFrame(wyniki)

    def walidacja_zakresow(self) -> pd.DataFrame:
        wyniki = []
        for col, (min_bio, max_bio) in ZAKRESY_BIOLOGICZNE.items():
            if col in self.df.columns:
                dane = konwertuj_kolumne_na_liczby(self.df[col]).dropna()
                if len(dane) > 0:
                    mask = (dane < min_bio) | (dane > max_bio)
                    wyniki.append({
                        "kolumna": col, "poza_zakresem": int(mask.sum()),
                        "procent_poza": round(100 * mask.sum() / len(dane), 2),
                        "min_bio": min_bio, "max_bio": max_bio
                    })
        return pd.DataFrame(wyniki)

    def progi_kliniczne(self, top_param: List[str]) -> pd.DataFrame:
        wyniki = []
        for param in top_param[:5]:
            if param not in self.df.columns:
                continue
            dane = self.df[[param, "outcome"]].dropna()
            if len(dane) < 10 or len(dane["outcome"].unique()) < 2:
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
                if kierunek == "wyższe" and len(thresholds) > idx:
                    prog = thresholds[idx]
                elif len(thresholds) > idx:
                    prog = -thresholds[idx]
                else:
                    continue
                if pd.isna(prog):
                    continue
                if kierunek == "wyższe":
                    y_pred = (dane[param] >= prog).astype(int)
                else:
                    y_pred = (dane[param] <= prog).astype(int)
                tn = int(((y_pred == 0) & (dane["outcome"] == 0)).sum())
                fp = int(((y_pred == 1) & (dane["outcome"] == 0)).sum())
                fn = int(((y_pred == 0) & (dane["outcome"] == 1)).sum())
                tp = int(((y_pred == 1) & (dane["outcome"] == 1)).sum())
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                wyniki.append({
                    "parametr": param, "etykieta": pretty_name(param), "kierunek": kierunek,
                    "prog": round(prog, 3), "czulosc": round(sens, 3), "swoistosc": round(spec, 3),
                    "n_calosci": len(dane)
                })
            except Exception:
                continue
        return pd.DataFrame(wyniki)

    def generuj_skale_ryzyka_z_bootstrapem(self, mode: str = "szybki", progress_callback=None) -> Tuple[Optional[pd.DataFrame], Dict]:
        n_iterations = BOOTSTRAP_ITERATIONS.get(mode, 100)
        kandydaci = ["crp", "troponina_i", "SpO2", "hgb", "hct", "kreatynina", "MAP", "wiek"]
        kandydaci = [k for k in kandydaci if k in self.df.columns]
        if len(kandydaci) == 0:
            return None, {"error": "Brak kandydatów do skali"}
        wszystkie_progi = {param: [] for param in kandydaci}
        wszystkie_auc = {param: [] for param in kandydaci}
        for i in range(n_iterations):
            if progress_callback:
                progress_callback(i + 1, n_iterations)
            df_boot = self.df.sample(n=len(self.df), replace=True, random_state=i)
            for param in kandydaci:
                dane = df_boot[[param, "outcome"]].dropna()
                if len(dane) < 30 or len(dane["outcome"].unique()) < 2:
                    continue
                hosp = dane[dane["outcome"] == 1][param]
                dom = dane[dane["outcome"] == 0][param]
                if len(hosp) < 10 or len(dom) < 10:
                    continue
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
                    if kierunek == "wyższe" and len(thresholds) > idx:
                        prog = float(thresholds[idx])
                    elif len(thresholds) > idx:
                        prog = float(-thresholds[idx])
                    else:
                        continue
                    wszystkie_progi[param].append(prog)
                    wszystkie_auc[param].append(auc_val)
                except Exception:
                    continue
        wyniki = []
        for param in kandydaci:
            if len(wszystkie_progi[param]) < 10:
                continue
            progi = np.array(wszystkie_progi[param])
            auc_vals = np.array(wszystkie_auc[param])
            prog_mean, prog_std = np.mean(progi), np.std(progi)
            progi_filt = progi[np.abs(progi - prog_mean) <= 3 * prog_std] if prog_std > 0 else progi
            wyniki.append({
                "parametr": param, "etykieta": pretty_name(param), "prog_mediana": np.median(progi_filt),
                "prog_ci_low": np.percentile(progi_filt, 2.5), "prog_ci_high": np.percentile(progi_filt, 97.5),
                "auc_mediana": np.median(auc_vals), "stabilnosc": np.std(progi_filt), "n_bootstrap": len(progi_filt)
            })
        if len(wyniki) == 0:
            return None, {"error": "Nie udało się wygenerować skali"}
        df_scale = pd.DataFrame(wyniki)
        df_scale = df_scale.sort_values(["auc_mediana", "stabilnosc"], ascending=[False, True]).head(4).reset_index(drop=True)
        punkty = []
        for _, row in df_scale.iterrows():
            pkt = 2 if row["auc_mediana"] >= 0.75 else 1
            if row["stabilnosc"] < (row["prog_mediana"] * 0.1):
                pkt += 1
            punkty.append(min(pkt, 3))
        df_scale["punkty"] = punkty
        auc_skali = self._policz_auc_skali(df_scale, self.df)
        meta = {
            "n": len(self.df), "data": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "auc": auc_skali, "bootstrap_iterations": n_iterations, "mode": mode
        }
        return df_scale, meta

    def _policz_auc_skali(self, scale_df: pd.DataFrame, df: pd.DataFrame) -> float:
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
                prog = rule["prog_mediana"]
                if rule.get("kierunek", "wyższe") == "wyższe":
                    if val >= prog:
                        suma += int(rule["punkty"])
                else:
                    if val <= prog:
                        suma += int(rule["punkty"])
            score_values.append(suma)
        if len(set(score_values)) < 2:
            return np.nan
        return roc_auc_score(dane["outcome"], score_values)


# =============================================================================
# WARSTWA 2: MENEDŻER RAPORTÓW
# =============================================================================

class ReportManager:
    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.zapisane_pliki = []

    def zapisz_csv(self, df: pd.DataFrame, nazwa: str) -> str:
        sciezka = os.path.join(self.output_folder, nazwa)
        df.to_csv(sciezka, sep=";", index=False, encoding="utf-8")
        self.zapisane_pliki.append(sciezka)
        return sciezka

    def zapisz_wykres(self, fig: plt.Figure, nazwa: str) -> str:
        sciezka = os.path.join(self.output_folder, nazwa)
        fig.savefig(sciezka, dpi=300, bbox_inches="tight")
        self.zapisane_pliki.append(sciezka)
        return sciezka

    def zapisz_tekst(self, tekst: str, nazwa: str) -> str:
        sciezka = os.path.join(self.output_folder, nazwa)
        with open(sciezka, "w", encoding="utf-8") as f:
            f.write(tekst)
        self.zapisane_pliki.append(sciezka)
        return sciezka

    def utworz_forest_plot(self, wyniki_modelu: pd.DataFrame, nazwa: str, tytul: str = None) -> Optional[str]:
        if wyniki_modelu is None or len(wyniki_modelu) == 0:
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        mask = (wyniki_modelu["ci_low"] > 0) & (wyniki_modelu["ci_high"] > 0)
        plot_data = wyniki_modelu[mask].copy()
        if len(plot_data) == 0:
            plt.close(fig)
            return None
        y_pos = np.arange(len(plot_data))
        ax.errorbar(plot_data["OR"], y_pos, xerr=[plot_data["OR"] - plot_data["ci_low"], plot_data["ci_high"] - plot_data["OR"]], fmt='o', capsize=4, color=KOLORY["primary"], ecolor=KOLORY["accent2"])
        ax.axvline(1, linestyle='--', color='gray', alpha=0.7)
        ax.set_xscale('log')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_data["etykieta"])
        ax.set_xlabel("OR (95% CI)", fontsize=11)
        ax.set_title(tytul or "Niezależne czynniki związane z hospitalizacją", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.text(0.98, 0.02, "⚠️ Skala logarytmiczna", transform=ax.transAxes, fontsize=8, ha='right', va='bottom', style='italic')
        plt.tight_layout()
        return self.zapisz_wykres(fig, nazwa)

    def get_folder_path(self) -> str:
        return self.output_folder


# =============================================================================
# GŁÓWNA KLASA APLIKACJI
# =============================================================================

class MedicalAnalyzerGUI:
    PARAMETRY_KLINICZNE = PARAMETRY_KLINICZNE
    CHOROBY = CHOROBY
    KOLORY = KOLORY

    def __init__(self, root):
        self.root = root
        self.root.title("✨ ANALIZATOR DANYCH MEDYCZNYCH ✨")
        self.root.geometry("1500x930")
        self.root.configure(bg=KOLORY["bg"])
        self.df = None
        self.df_hosp = None
        self.df_dom = None
        self.wyniki_df = None
        self.current_figure = None
        self.current_param = None
        self.current_mode = "podstawowa"
        self.pro_btn = None
        self.control_frame = None
        self.btn_frame = None
        self.prediction_pipeline = None
        self.prediction_features = []
        self.prediction_input_vars = {}
        self.map_live_label = None
        self.prediction_model_info = ""
        self.prediction_feature_order = []
        self.prediction_model_source = "wewnętrzny"
        self.loaded_model_path = None
        self.model_auc_holdout = None
        self.scale_current_df = None
        self.scale_frozen_df = None
        self.scale_current_meta = {}
        self.scale_frozen_meta = {}
        self.scale_input_vars = {}
        self.scale_rr_skurczowe_var = None
        self.scale_rr_rozkurczowe_var = None
        self.scale_map_live_label = None
        self.rr_skurczowe_var = None
        self.rr_rozkurczowe_var = None
        self.otwarte_figury = []
        self.parametry_kliniczne = PARAMETRY_KLINICZNE.copy()
        self.choroby = CHOROBY.copy()
        self.zmienne_obowiazkowe = ZMIENNE_OBOWIAZKOWE.copy()
        self.zmienne_dodatkowe = ZMIENNE_DODATKOWE.copy()
        self.zmienne_log = ZMIENNE_LOG.copy()
        self.zakresy_biologiczne = ZAKRESY_BIOLOGICZNE.copy()
        self._setup_ui()

    # ===== METODY POMOCNICZE =====
    def _create_button(self, parent, text, command, color, **kwargs):
        btn = tk.Button(parent, text=text, command=command, font=("Helvetica", kwargs.get('font_size', 11), "bold"),
            bg=color, fg="white", activebackground=KOLORY["accent2"], activeforeground="white",
            relief="flat", bd=0, padx=kwargs.get('padx', 20), pady=kwargs.get('pady', 8),
            cursor="hand2", wraplength=kwargs.get('wraplength', 0), justify=kwargs.get('justify', 'center'),
            height=kwargs.get('height', 1))
        btn.bind("<Enter>", lambda e: btn.config(bg=KOLORY["accent2"]))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn

    def _sprawdz_czy_dane_wczytane(self) -> bool:
        if self.df_hosp is None:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj dane!")
            return False
        return True

    def _przygotuj_dane_dla_parametru(self, param):
        if param not in self.df.columns:
            return None, None
        hosp = self.df_hosp[param].dropna()
        dom = self.df_dom[param].dropna()
        if len(hosp) == 0 or len(dom) == 0:
            return None, None
        return hosp, dom

    def _czysc_tree(self, tree):
        for row in tree.get_children():
            tree.delete(row)

    def _okresl_istotnosc(self, p):
        if p < 0.001:
            return "***", "highly"
        elif p < 0.01:
            return "**", "significant"
        elif p < 0.05:
            return "*", "significant"
        return "ns", ""

    def _ustaw_kolumny_tabeli(self):
        if self.current_mode == "podstawowa":
            columns = ("lp", "parametr", "hosp_n", "hosp_sr", "hosp_std", "dom_n", "dom_sr", "dom_std", "p", "ist")
            self.tree["columns"] = columns
            for col in columns:
                self.tree.heading(col, text=col.upper())
            widths = [50, 220, 75, 100, 100, 75, 100, 100, 100, 70]
            for col, w in zip(columns, widths):
                self.tree.column(col, width=w, anchor="center")
        else:
            columns = ("lp", "parametr", "hosp_med", "dom_med", "p_raw", "p_fdr", "delta", "efekt", "ist")
            self.tree["columns"] = columns
            for col in columns:
                self.tree.heading(col, text=col.upper())
            widths = [50, 220, 180, 180, 90, 90, 90, 100, 80]
            for col, w in zip(columns, widths):
                self.tree.column(col, width=w, anchor="center")

    # ===== KONFIGURACJA UI =====
    def _setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=KOLORY["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", font=("Helvetica", 12, "bold"), padding=[18, 10],
                       background=KOLORY["light"], foreground=KOLORY["dark"])
        style.map("TNotebook.Tab", background=[("selected", KOLORY["primary"])], foreground=[("selected", "white")])
        style.configure("TButton", font=("Helvetica", 11), padding=10)
        style.configure("TLabel", font=("Helvetica", 11), background=KOLORY["bg"], foreground=KOLORY["dark"])
        style.configure("TFrame", background=KOLORY["bg"])
        style.configure("TLabelframe", background=KOLORY["bg"], foreground=KOLORY["dark"], font=("Helvetica", 11, "bold"))
        style.configure("TLabelframe.Label", background=KOLORY["bg"], foreground=KOLORY["dark"])
        main_container = ttk.Frame(self.root, padding="15")
        main_container.pack(fill="both", expand=True)
        self._utworz_naglowek(main_container)
        self._utworz_przelacznik_trybu(main_container)
        self._utworz_notebook(main_container)

    def _utworz_naglowek(self, parent):
        header_frame = tk.Frame(parent, bg=KOLORY["primary"], height=80)
        header_frame.pack(fill="x", pady=(0, 15))
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="📊 ANALIZA PORÓWNAWCZA PACJENTÓW", font=("Helvetica", 20, "bold"), bg=KOLORY["primary"], fg="white").pack(expand=True)
        tk.Label(header_frame, text="Przyjęci do szpitala vs wypisani do domu", font=("Helvetica", 12), bg=KOLORY["primary"], fg="white").pack(expand=True)

    def _utworz_przelacznik_trybu(self, parent):
        mode_frame = tk.Frame(parent, bg=KOLORY["light"], height=50)
        mode_frame.pack(fill="x", pady=(0, 10))
        tk.Label(mode_frame, text="🔧 TRYB ANALIZY:", font=("Helvetica", 12, "bold"), bg=KOLORY["light"], fg=KOLORY["dark"]).pack(side="left", padx=20, pady=10)
        self.mode_var = tk.StringVar(value="podstawowa")
        mode_basic = tk.Radiobutton(mode_frame, text="📊 PODSTAWOWA", variable=self.mode_var, value="podstawowa",
            font=("Helvetica", 11), bg=KOLORY["light"], fg=KOLORY["dark"], selectcolor=KOLORY["light"], command=self._zmien_tryb)
        mode_basic.pack(side="left", padx=10, pady=10)
        mode_pro = tk.Radiobutton(mode_frame, text="📈 PROFESJONALNA", variable=self.mode_var, value="profesjonalna",
            font=("Helvetica", 11), bg=KOLORY["light"], fg=KOLORY["dark"], selectcolor=KOLORY["light"], command=self._zmien_tryb)
        mode_pro.pack(side="left", padx=10, pady=10)

    def _utworz_notebook(self, parent):
        self.notebook = ttk.Notebook(parent)
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
        self._tab1_wczytaj()
        self._tab2_analiza()
        self._tab3_kalkulator()
        self._tab4_wykresy()
        self._tab5_raport()
        self._tab6_info()
        self._tab7_skala()

    # ===== ZAKŁADKA 1 - WCZYTYWANIE DANYCH =====
    def _tab1_wczytaj(self):
        main_frame = ttk.Frame(self.tab1, padding="30")
        main_frame.pack(fill="both", expand=True)
        button_frame = tk.Frame(main_frame, bg=KOLORY["bg"])
        button_frame.pack(pady=50)
        button_style = {"font": ("Helvetica", 14, "bold"), "bg": KOLORY["accent1"], "fg": "white",
            "activebackground": KOLORY["accent2"], "activeforeground": "white", "relief": "flat",
            "bd": 0, "padx": 30, "pady": 15, "cursor": "hand2"}
        btn_csv = tk.Button(button_frame, text="📁 WCZYTAJ PLIK CSV", command=self._wczytaj_csv, **button_style)
        btn_csv.pack(side="left", padx=20)
        btn_excel = tk.Button(button_frame, text="📗 WCZYTAJ PLIK EXCEL", command=self._wczytaj_excel, **button_style)
        btn_excel.pack(side="left", padx=20)
        btn_sqlite = tk.Button(button_frame, text="🗄️ WCZYTAJ BAZĘ SQLITE", command=self._wybierz_i_wczytaj_sqlite, **button_style)
        btn_sqlite.pack(side="left", padx=20)
        for btn in [btn_csv, btn_excel, btn_sqlite]:
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=KOLORY["accent2"]))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=KOLORY["accent1"]))
        info_frame = tk.LabelFrame(main_frame, text="📋 INFORMACJE O DANYCH", font=("Helvetica", 14, "bold"),
            bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=20, pady=15)
        info_frame.pack(fill="both", expand=True, pady=30)
        text_frame = tk.Frame(info_frame, bg=KOLORY["light"])
        text_frame.pack(fill="both", expand=True)
        self.info_text = tk.Text(text_frame, height=15, font=("Courier", 11), bg="white", fg=KOLORY["dark"],
            relief="flat", bd=1, padx=10, pady=10)
        self.info_text.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(text_frame, command=self.info_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.info_text.config(yscrollcommand=scrollbar.set)
        analyze_btn = self._create_button(main_frame, "🚀 PRZEJDŹ DO ANALIZY", lambda: self.notebook.select(self.tab2), KOLORY["success"], font_size=14, padx=40, pady=15)
        analyze_btn.pack(pady=20)
        self.info_text.insert("1.0", "✨ Witaj w analizatorze danych medycznych!\n\nAby rozpocząć:\n1. Wybierz tryb analizy\n2. Wczytaj plik CSV, Excel lub bazę SQLite\n3. Program spróbuje znaleźć lub zbudować kolumnę outcome\n4. Po wczytaniu danych zostanie też zbudowany kalkulator hospitalizacji\n\n📊 Tryb PODSTAWOWY - szybka analiza\n📈 Tryb PROFESJONALNY - pełna analiza publikacyjna")

    def _wczytaj_csv(self):
        filename = filedialog.askopenfilename(title="Wybierz plik CSV", filetypes=[("Pliki CSV", "*.csv"), ("Wszystkie pliki", "*.*")])
        if filename:
            try:
                encodings = ['utf-8', 'latin1', 'cp1250']
                separators = [';', ',', '\t']
                df = None
                for sep in separators:
                    for enc in encodings:
                        try:
                            df = pd.read_csv(filename, sep=sep, encoding=enc)
                            if len(df.columns) > 1:
                                break
                        except Exception:
                            continue
                    if df is not None and len(df.columns) > 1:
                        break
                if df is None:
                    df = pd.read_csv(filename)
                self.df = df
                self._normalizuj_nazwy_kolumn()
                self._przetworz_dane()
                self._wyswietl_info(filename)
                self._zbuduj_model_hospitalizacji()
                messagebox.showinfo("✅ Sukces", f"Plik wczytany poprawnie!\n\nLiczba pacjentów: {len(self.df)}\nKalkulator hospitalizacji: {'gotowy' if self.prediction_pipeline is not None else 'niezbudowany'}")
            except FileNotFoundError:
                messagebox.showerror("❌ Błąd", f"Plik nie istnieje:\n{filename}")
            except pd.errors.EmptyDataError:
                messagebox.showerror("❌ Błąd", "Plik jest pusty")
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Nie udało się wczytać pliku:\n{e}")

    def _wczytaj_excel(self):
        filename = filedialog.askopenfilename(title="Wybierz plik Excel", filetypes=[("Pliki Excel", "*.xlsx *.xls"), ("Wszystkie pliki", "*.*")])
        if filename:
            try:
                self.df = pd.read_excel(filename)
                self._normalizuj_nazwy_kolumn()
                self._przetworz_dane()
                self._wyswietl_info(filename)
                self._zbuduj_model_hospitalizacji()
                messagebox.showinfo("✅ Sukces", f"Plik wczytany poprawnie!\n\nLiczba pacjentów: {len(self.df)}\nKalkulator hospitalizacji: {'gotowy' if self.prediction_pipeline is not None else 'niezbudowany'}")
            except FileNotFoundError:
                messagebox.showerror("❌ Błąd", f"Plik nie istnieje:\n{filename}")
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Nie udało się wczytać pliku:\n{e}")

    def _wybierz_i_wczytaj_sqlite(self):
        filename = filedialog.askopenfilename(title="Wybierz bazę SQLite", filetypes=[("Bazy SQLite", "*.sqlite *.db *.sqlite3"), ("Wszystkie pliki", "*.*")])
        if not filename:
            return
        try:
            with sqlite3.connect(filename) as conn:
                tables_df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name", conn)
            tables = tables_df["name"].tolist()
            if not tables:
                messagebox.showwarning("⚠️ Uwaga", "W wybranej bazie nie znaleziono żadnych tabel.")
                return
            if len(tables) == 1:
                table_name = tables[0]
            else:
                prompt = "Dostępne tabele:\n- " + "\n- ".join(tables) + "\n\nWpisz nazwę tabeli do wczytania:"
                table_name = simpledialog.askstring("Wybór tabeli SQLite", prompt, initialvalue=tables[0], parent=self.root)
                if not table_name:
                    return
                if table_name not in tables:
                    messagebox.showerror("❌ Błąd", f"Tabela '{table_name}' nie istnieje w tej bazie.")
                    return
            self._wczytaj_sqlite(filename, table_name)
        except sqlite3.Error as e:
            messagebox.showerror("❌ Błąd", f"Błąd bazy danych SQLite:\n{e}")
        except Exception as e:
            messagebox.showerror("❌ Błąd", f"Nie udało się odczytać bazy SQLite:\n{e}")

    def _wczytaj_sqlite(self, filename, table_name):
        try:
            safe_table = table_name.replace(chr(34), chr(34) * 2)
            query = f'SELECT * FROM "{safe_table}"'
            with sqlite3.connect(filename) as conn:
                self.df = pd.read_sql_query(query, conn)
            self._normalizuj_nazwy_kolumn()
            self._przetworz_dane()
            self._wyswietl_info(filename)
            if hasattr(self, "info_text") and self.info_text is not None:
                self.info_text.insert(tk.END, f"\n🗄️ TABELA SQLITE: {table_name}\n")
            self._zbuduj_model_hospitalizacji()
            messagebox.showinfo("✅ Sukces", f"Tabela SQLite wczytana poprawnie!\n\nTabela: {table_name}\nLiczba pacjentów: {len(self.df)}\nKalkulator hospitalizacji: {'gotowy' if self.prediction_pipeline is not None else 'niezbudowany'}")
        except sqlite3.Error as e:
            messagebox.showerror("❌ Błąd", f"Błąd bazy danych SQLite:\n{e}")
        except Exception as e:
            messagebox.showerror("❌ Błąd", f"Nie udało się wczytać tabeli SQLite:\n{e}")

    def _normalizuj_nazwy_kolumn(self):
        if self.df is None:
            return
        for stara, nowa in NAZWY_TECHNICZNE.items():
            if stara in self.df.columns:
                self.df[nowa] = self.df[stara]

    def _przetworz_dane(self):
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
                raise ValueError("Brak kolumny 'outcome' i nie udało się wykryć starego formatu z pustym wierszem.\nDodaj kolumnę outcome (1=hospitalizacja, 0=do domu) albo użyj starego układu bazy.")
        df = df[df["outcome"].notna()].copy()
        df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
        df = df[df["outcome"].isin([0, 1])].copy()
        for col in self.parametry_kliniczne:
            if col in df.columns:
                df[col] = konwertuj_kolumne_na_liczby(df[col])
        mapping_tak = {"tak", "t", "yes", "y", "1", "true", "+", "tak!"}
        mapping_nie = {"nie", "n", "no", "0", "false", "-"}
        for col in self.choroby:
            if col in df.columns:
                tmp = df[col].astype(str).str.lower().str.strip()
                df[col] = tmp.apply(lambda x: 1 if x in mapping_tak else (0 if x in mapping_nie else np.nan))
        self.df = df.copy()
        self.df_hosp = df[df["outcome"] == 1].copy()
        self.df_dom = df[df["outcome"] == 0].copy()

    def _wyswietl_info(self, filename):
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
                info += f"   {i:2d}. {pretty_name(param)}\n"
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

    # ===== ZAKŁADKA 2 - ANALIZA STATYSTYCZNA =====
    def _tab2_analiza(self):
        main_frame = ttk.Frame(self.tab2, padding="20")
        main_frame.pack(fill="both", expand=True)
        self.control_frame = tk.LabelFrame(main_frame, text="🎯 PARAMETRY ANALIZY",
            font=("Helvetica", 12, "bold"), bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=20, pady=15)
        self.control_frame.pack(fill="x", pady=(0, 20))
        ttk.Label(self.control_frame, text="Wybierz parametr:", font=("Helvetica", 11)).pack(side="left", padx=10)
        self.param_var = tk.StringVar()
        self.param_combo = ttk.Combobox(self.control_frame, textvariable=self.param_var,
            values=self.parametry_kliniczne, width=40, state="readonly", font=("Helvetica", 11))
        self.param_combo.pack(side="left", padx=10)
        self.btn_frame = tk.Frame(self.control_frame, bg=KOLORY["light"])
        self.btn_frame.pack(side="right")
        analyze_one_btn = self._create_button(self.btn_frame, "📊 ANALIZUJ WYBRANY", self._analizuj_pojedynczy, KOLORY["accent1"], font_size=11, padx=20, pady=8)
        analyze_one_btn.pack(side="left", padx=5)
        analyze_all_btn = self._create_button(self.btn_frame, "📊 ANALIZUJ WSZYSTKIE", self._analizuj_wszystkie, KOLORY["success"], font_size=11, padx=20, pady=8)
        analyze_all_btn.pack(side="left", padx=5)
        if self.current_mode == "profesjonalna":
            self._dodaj_przycisk_profesjonalny_bez_szukania()
        table_frame = tk.LabelFrame(main_frame, text="📋 WYNIKI ANALIZY STATYSTYCZNEJ",
            font=("Helvetica", 12, "bold"), bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=20, pady=15)
        table_frame.pack(fill="both", expand=True)
        tree_frame = tk.Frame(table_frame, bg=KOLORY["light"])
        tree_frame.pack(fill="both", expand=True)
        vsb = tk.Scrollbar(tree_frame, orient="vertical")
        hsb = tk.Scrollbar(tree_frame, orient="horizontal")
        self.tree = ttk.Treeview(tree_frame, show="headings", height=15, yscrollcommand=vsb.set, xscrollcommand=hsb.set)
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
        self.stats_label = tk.Label(stats_frame, text="", font=("Helvetica", 11), bg=KOLORY["light"], fg=KOLORY["dark"])
        self.stats_label.pack(pady=10)
        self._ustaw_kolumny_tabeli()

    def _dodaj_przycisk_profesjonalny_bez_szukania(self):
        if hasattr(self, 'btn_frame') and self.btn_frame:
            self.pro_btn = self._create_button(self.btn_frame, "🔬 ANALIZA PROFESJONALNA", self._analiza_profesjonalna, KOLORY["warning"], font_size=11, padx=20, pady=8)
            self.pro_btn.pack(side="left", padx=5)

    def _usun_przycisk_profesjonalny(self):
        try:
            if self.pro_btn is not None:
                self.pro_btn.destroy()
                self.pro_btn = None
        except Exception:
            pass

    def _zmien_tryb(self):
        self.current_mode = self.mode_var.get()
        if hasattr(self, "tree"):
            self._czysc_tree(self.tree)
            self._ustaw_kolumny_tabeli()
        if self.current_mode == "profesjonalna":
            self._dodaj_przycisk_profesjonalny_bez_szukania()
            messagebox.showinfo("🔬 Tryb profesjonalny", "Wybrano tryb PROFESJONALNY.\n\nZakres:\n• Tabela 1\n• Analiza jednoczynnikowa z FDR\n• Modele regresji logistycznej\n• Forest plot\n• Model predykcyjny\n• Progi kliniczne\n• Raport końcowy\n\n⚠️ UWAGA: Wyniki dotyczą wyłącznie tej kohorty - brak walidacji zewnętrznej.")
        else:
            self._usun_przycisk_profesjonalny()
            messagebox.showinfo("📊 Tryb podstawowy", "Wybrano tryb PODSTAWOWY.\n\nZakres:\n• Podstawowe statystyki\n• Test Manna-Whitneya\n• Wykresy pudełkowe")

    def _analizuj_pojedynczy(self):
        if not self._sprawdz_czy_dane_wczytane():
            return
        param = self.param_var.get()
        if not param:
            messagebox.showwarning("⚠️ Uwaga", "Wybierz parametr!")
            return
        self._czysc_tree(self.tree)
        hosp, dom = self._przygotuj_dane_dla_parametru(param)
        if hosp is None or dom is None:
            messagebox.showwarning("⚠️ Uwaga", f"Brak danych dla parametru {param}")
            return
        silnik = StatisticsEngine(self.df)
        stats_dict = silnik.oblicz_statystyki_parametru(param, hosp, dom)
        if stats_dict:
            if self.current_mode == "podstawowa":
                self.tree.insert("", "end", tags=(stats_dict['tag'],), values=(1, pretty_name(param), stats_dict['hosp_n'],
                    f"{stats_dict['hosp_mean']:.2f}", f"{stats_dict['hosp_std']:.2f}", stats_dict['dom_n'],
                    f"{stats_dict['dom_mean']:.2f}", f"{stats_dict['dom_std']:.2f}", f"{stats_dict['p_value']:.4f}", stats_dict['stars']))
                self.stats_label.config(text=f"✓ Przeanalizowano parametr: {pretty_name(param)} • n(przyjęci)={stats_dict['hosp_n']} • n(wypisani)={stats_dict['dom_n']}")
            else:
                hosp_txt = f"{stats_dict['hosp_median']:.2f} [{stats_dict['hosp_q1']:.2f}-{stats_dict['hosp_q3']:.2f}]"
                dom_txt = f"{stats_dict['dom_median']:.2f} [{stats_dict['dom_q1']:.2f}-{stats_dict['dom_q3']:.2f}]"
                self.tree.insert("", "end", tags=(stats_dict['tag'],), values=(1, pretty_name(param), hosp_txt, dom_txt,
                    f"{stats_dict['p_value']:.4f}", f"{stats_dict['p_value']:.4f}", f"{stats_dict['cliff_delta']:.3f}", stats_dict['effect_size'], stats_dict['stars']))
                self.stats_label.config(text=f"✓ Tryb profesjonalny • {pretty_name(param)} • mediana/IQR + efekt + FDR")

    def _analizuj_wszystkie(self):
        if not self._sprawdz_czy_dane_wczytane():
            return
        self._czysc_tree(self.tree)
        silnik = StatisticsEngine(self.df)
        if self.current_mode == "podstawowa":
            wyniki = []
            for i, param in enumerate(self.parametry_kliniczne, 1):
                stats_dict = silnik.oblicz_statystyki_parametru(param)
                if stats_dict:
                    self.tree.insert("", "end", tags=(stats_dict['tag'],), values=(
                        i, pretty_name(param), stats_dict['hosp_n'],
                        f"{stats_dict['hosp_mean']:.2f}", f"{stats_dict['hosp_std']:.2f}",
                        stats_dict['dom_n'], f"{stats_dict['dom_mean']:.2f}", f"{stats_dict['dom_std']:.2f}",
                        f"{stats_dict['p_value']:.4f}", stats_dict['stars']
                    ))
                    wyniki.append({"parametr": param, "etykieta": pretty_name(param), 
                                   "p_value": stats_dict['p_value'], "istotnosc": stats_dict['stars']})
            self.wyniki_df = pd.DataFrame(wyniki)
            istotne = sum(1 for w in wyniki if w["p_value"] < 0.05)
            wysoce = sum(1 for w in wyniki if w["p_value"] < 0.001)
            self.stats_label.config(text=f"✓ Przeanalizowano {len(wyniki)} parametrów • Istotne: {istotne} • Wysoce istotne: {wysoce}")
            messagebox.showinfo("✅ Analiza zakończona", f"Przeanalizowano {len(wyniki)} parametrów.\nZnaleziono {istotne} parametrów z istotnymi różnicami.")
        else:
            df_wyn, _ = silnik.analiza_jednoczynnikowa_z_fdr()
            for i, row in df_wyn.iterrows():
                stars, tag = self._okresl_istotnosc(row['p_fdr'])
                hosp_txt = f"{row['hosp_median']:.2f} [{row['hosp_q1']:.2f}-{row['hosp_q3']:.2f}]"
                dom_txt = f"{row['dom_median']:.2f} [{row['dom_q1']:.2f}-{row['dom_q3']:.2f}]"
                self.tree.insert("", "end", tags=(tag,), values=(
                    i + 1, row['etykieta'], hosp_txt, dom_txt,
                    f"{row['p_raw']:.4f}", f"{row['p_fdr']:.4f}",
                    f"{row['cliff_delta']:.3f}", row['interpretacja'], stars
                ))
            self.wyniki_df = df_wyn
            istotne_fdr = int((df_wyn["p_fdr"] < 0.05).sum()) if len(df_wyn) > 0 else 0
            wysoce_fdr = int((df_wyn["p_fdr"] < 0.001).sum()) if len(df_wyn) > 0 else 0
            self.stats_label.config(text=f"✓ Tryb profesjonalny • {len(df_wyn)} parametrów • Istotne po FDR: {istotne_fdr} • Wysoce istotne: {wysoce_fdr}")
            messagebox.showinfo("✅ Analiza zakończona", f"Tryb profesjonalny.\nPrzeanalizowano {len(df_wyn)} parametrów.\nIstotne po korekcji FDR: {istotne_fdr}.")

    # ===== ZAKŁADKA 3 - KALKULATOR HOSPITALIZACJI =====
    def _tab3_kalkulator(self):
        main_frame = ttk.Frame(self.tab3, padding="20")
        main_frame.pack(fill="both", expand=True)
        top_frame = tk.LabelFrame(main_frame, text="🧮 KALKULATOR PRAWDOPODOBIEŃSTWA HOSPITALIZACJI",
            font=("Helvetica", 12, "bold"), bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=20, pady=15)
        top_frame.pack(fill="x", pady=(0, 15))
        self.calc_info_label = tk.Label(top_frame, text="Po wczytaniu danych program zbuduje model tylko na Twojej bazie.",
            font=("Helvetica", 11), bg=KOLORY["light"], fg=KOLORY["dark"], justify="left")
        self.calc_info_label.pack(anchor="w", pady=5)
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill="both", expand=True)
        left_frame = tk.LabelFrame(middle_frame, text="📋 WPROWADŹ DANE PACJENTA",
            font=("Helvetica", 12, "bold"), bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=15, pady=15)
        left_frame.pack(side="left", fill="y", expand=False, padx=(0, 10))
        entry_and_actions = tk.Frame(left_frame, bg=KOLORY["light"])
        entry_and_actions.pack(fill="both", expand=True)
        self.calc_entries_frame = tk.Frame(entry_and_actions, bg=KOLORY["light"])
        self.calc_entries_frame.pack(side="left", fill="both", expand=True)
        self._zbuduj_pola_kalkulatora()
        actions_frame = tk.Frame(entry_and_actions, bg=KOLORY["light"])
        actions_frame.pack(side="left", fill="y", padx=(12, 0))
        predict_btn = self._create_button(actions_frame, "🔮 OBLICZ PRAWDOPODOBIEŃSTWO", self._oblicz_prawdopodobienstwo, KOLORY["accent1"], font_size=11, padx=14, pady=10)
        predict_btn.pack(fill="x", pady=(0, 8))
        clear_btn = self._create_button(actions_frame, "🧹 WYCZYŚĆ", self._wyczysc_kalkulator, KOLORY["warning"], font_size=11, padx=14, pady=10)
        clear_btn.pack(fill="x", pady=(0, 12))
        save_model_btn = self._create_button(actions_frame, "💾 ZAPISZ MODEL", self._zapisz_model, KOLORY["success"], font_size=10, padx=12, pady=8)
        save_model_btn.pack(fill="x", pady=(0, 6))
        load_model_btn = self._create_button(actions_frame, "📂 WCZYTAJ MODEL", self._wczytaj_model, KOLORY["primary"], font_size=10, padx=12, pady=8)
        load_model_btn.pack(fill="x", pady=(0, 6))
        apply_model_btn = self._create_button(actions_frame, "🧾 UŻYJ MODELU NA PLIKU", self._zastosuj_model_do_pliku, KOLORY["warning"], font_size=10, padx=12, pady=8, wraplength=180, justify="center")
        apply_model_btn.pack(fill="x")
        right_frame = tk.LabelFrame(middle_frame, text="📊 WYNIK I INTERPRETACJA",
            font=("Helvetica", 12, "bold"), bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=15, pady=15)
        right_frame.pack(side="left", fill="both", expand=True)
        self.result_big_label = tk.Label(right_frame, text="—", font=("Helvetica", 36, "bold"),
            bg="white", fg=KOLORY["primary"], width=18, height=2)
        self.result_big_label.pack(fill="x", pady=(0, 15))
        result_frame = tk.Frame(right_frame, bg=KOLORY["light"])
        result_frame.pack(fill="both", expand=True)
        result_scrollbar = tk.Scrollbar(result_frame)
        result_scrollbar.pack(side="right", fill="y")
        self.result_text = tk.Text(result_frame, font=("Courier", 11), bg="white", fg=KOLORY["dark"],
            wrap="word", padx=12, pady=12, yscrollcommand=result_scrollbar.set)
        self.result_text.pack(side="left", fill="both", expand=True)
        result_scrollbar.config(command=self.result_text.yview)

    def _zbuduj_pola_kalkulatora(self):
        for child in self.calc_entries_frame.winfo_children():
            child.destroy()
        self.prediction_input_vars = {}
        self.map_live_label = None
        pola = self.zmienne_obowiazkowe + self.zmienne_dodatkowe
        unikalne_pola = []
        for p in pola:
            if p not in unikalne_pola:
                unikalne_pola.append(p)
        unikalne_pola = [p for p in unikalne_pola if p != "MAP"]
        for param in unikalne_pola:
            row = tk.Frame(self.calc_entries_frame, bg=KOLORY["light"])
            row.pack(fill="x", pady=4)
            lbl = tk.Label(row, text=pretty_name(param), font=("Helvetica", 10),
                bg=KOLORY["light"], fg=KOLORY["dark"], width=28, anchor="w")
            lbl.pack(side="left", padx=(0, 8))
            var = tk.StringVar()
            self.prediction_input_vars[param] = var
            ent = tk.Entry(row, textvariable=var, font=("Helvetica", 10), width=18)
            ent.pack(side="left")
        row_rr1 = tk.Frame(self.calc_entries_frame, bg=KOLORY["light"])
        row_rr1.pack(fill="x", pady=4)
        tk.Label(row_rr1, text="RR skurczowe, mmHg", font=("Helvetica", 10),
            bg=KOLORY["light"], fg=KOLORY["dark"], width=28, anchor="w").pack(side="left", padx=(0, 8))
        self.rr_skurczowe_var = tk.StringVar()
        tk.Entry(row_rr1, textvariable=self.rr_skurczowe_var, font=("Helvetica", 10), width=18).pack(side="left")
        row_rr2 = tk.Frame(self.calc_entries_frame, bg=KOLORY["light"])
        row_rr2.pack(fill="x", pady=4)
        tk.Label(row_rr2, text="RR rozkurczowe, mmHg", font=("Helvetica", 10),
            bg=KOLORY["light"], fg=KOLORY["dark"], width=28, anchor="w").pack(side="left", padx=(0, 8))
        self.rr_rozkurczowe_var = tk.StringVar()
        tk.Entry(row_rr2, textvariable=self.rr_rozkurczowe_var, font=("Helvetica", 10), width=18).pack(side="left")
        row_map = tk.Frame(self.calc_entries_frame, bg=KOLORY["light"])
        row_map.pack(fill="x", pady=(8, 4))
        tk.Label(row_map, text="Wyliczone MAP, mmHg", font=("Helvetica", 10, "bold"),
            bg=KOLORY["light"], fg=KOLORY["dark"], width=28, anchor="w").pack(side="left", padx=(0, 8))
        self.map_live_label = tk.Label(row_map, text="—", font=("Helvetica", 10, "bold"),
            bg="white", fg=KOLORY["primary"], width=18, anchor="center", relief="solid", bd=1)
        self.map_live_label.pack(side="left")
        self.rr_skurczowe_var.trace_add("write", lambda *args: self._aktualizuj_map_na_zywo())
        self.rr_rozkurczowe_var.trace_add("write", lambda *args: self._aktualizuj_map_na_zywo())

    def _aktualizuj_map_na_zywo(self):
        if self.map_live_label is None:
            return
        sbp = bezpieczna_liczba(self.rr_skurczowe_var.get())
        dbp = bezpieczna_liczba(self.rr_rozkurczowe_var.get())
        if pd.isna(sbp) or pd.isna(dbp):
            self.map_live_label.config(text="—")
            return
        map_val = (sbp + 2 * dbp) / 3
        self.map_live_label.config(text=f"{map_val:.1f}")

    def _zbuduj_model_hospitalizacji(self):
        self.prediction_pipeline = None
        self.prediction_features = []
        self.prediction_model_info = ""
        self.prediction_feature_order = []
        self.prediction_model_source = "wewnętrzny"
        self.loaded_model_path = None
        self.model_auc_holdout = None
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
        silnik = StatisticsEngine(self.df)
        wynik = silnik.zbuduj_model_hospitalizacji()
        if wynik["success"]:
            self.prediction_pipeline = wynik["pipeline"]
            self.prediction_features = wynik["features"]
            self.prediction_model_info = wynik["model_info"]
            self.model_auc_holdout = wynik["auc_holdout"]
            self.calc_info_label.config(text=self.prediction_model_info)
        else:
            self.calc_info_label.config(text=f"Nie udało się zbudować kalkulatora: {wynik.get('error', '')}")

    def _oblicz_prawdopodobienstwo(self):
        self.result_text.delete("1.0", tk.END)
        if self.prediction_pipeline is None or len(self.prediction_features) == 0:
            messagebox.showwarning("⚠️ Brak modelu", "Najpierw wczytaj dane. Kalkulator buduje się wyłącznie na Twojej bazie.")
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
            for col in self.zmienne_log:
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
                self.result_text.insert("1.0", "Brak kompletu danych do obliczenia wyniku.\n\nUzupełnij:\n• " + "\n• ".join(missing_needed))
                return
            X_new = np.array([x_values], dtype=float)
            p_hosp = float(self.prediction_pipeline.predict_proba(X_new)[0, 1])
            kat = okresl_kategorie_ryzyka(p_hosp)
            scaler = self.prediction_pipeline.named_steps["scaler"]
            logreg = self.prediction_pipeline.named_steps["logreg"]
            z = (X_new[0] - scaler.mean_) / scaler.scale_
            contributions = z * logreg.coef_[0]
            contrib_df = pd.DataFrame({"feature": self.prediction_features,
                "etykieta": [pretty_name(f) for f in self.prediction_features],
                "contribution": contributions, "wartosc": X_new[0]})
            contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
            contrib_df = contrib_df.sort_values("abs_contribution", ascending=False)
            kolor = KOLORY["success"]
            if p_hosp >= 0.50:
                kolor = KOLORY["accent1"]
            elif p_hosp >= 0.20:
                kolor = KOLORY["warning"]
            self.result_big_label.config(text=f"{100 * p_hosp:.1f}%", fg=kolor)
            tekst = ["WYNIK KALKULATORA", "=" * 60, f"Prawdopodobieństwo hospitalizacji: {100 * p_hosp:.1f}%",
                f"Kategoria ryzyka: {kat}", "", "Interpretacja:",
                "• Model wyuczony wyłącznie na Twojej bazie", "• Wynik ma znaczenie wewnętrzne dla tej kohorty",
                "• Nie jest to walidowane narzędzie zewnętrzne", "", "Najsilniejsze czynniki wpływające na wynik:"]
            for _, rowc in contrib_df.head(5).iterrows():
                kier = "↑ zwiększa" if rowc["contribution"] > 0 else "↓ zmniejsza"
                tekst.append(f"• {rowc['etykieta']}: {rowc['wartosc']:.2f}  |  {kier} ryzyko")
            self.result_text.insert("1.0", "\n".join(tekst))
            self.result_text.see("1.0")
        except Exception as e:
            self.result_big_label.config(text="—", fg=KOLORY["primary"])
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert("1.0", f"Nie udało się obliczyć wyniku.\n\nSzczegóły:\n{e}")
            self.result_text.see("1.0")

    def _wyczysc_kalkulator(self):
        for var in self.prediction_input_vars.values():
            var.set("")
        if hasattr(self, "rr_skurczowe_var") and self.rr_skurczowe_var is not None:
            self.rr_skurczowe_var.set("")
        if hasattr(self, "rr_rozkurczowe_var") and self.rr_rozkurczowe_var is not None:
            self.rr_rozkurczowe_var.set("")
        if self.map_live_label is not None:
            self.map_live_label.config(text="—")
        self.result_big_label.config(text="—", fg=KOLORY["primary"])
        self.result_text.delete("1.0", tk.END)

    def _zapisz_model(self):
        if self.prediction_pipeline is None or len(self.prediction_features) == 0:
            messagebox.showwarning("⚠️ Uwaga", "Brak gotowego modelu do zapisania. Najpierw wczytaj dane i zbuduj model.")
            return
        filename = filedialog.asksaveasfilename(defaultextension='.joblib',
            filetypes=[('Plik modelu', '*.joblib')], initialfile='model_hospitalizacji.joblib')
        if not filename:
            return
        payload = {'pipeline': self.prediction_pipeline, 'features': self.prediction_features,
            'feature_order': self.prediction_feature_order if hasattr(self, 'prediction_feature_order') else self.prediction_features,
            'model_info': self.prediction_model_info if hasattr(self, 'prediction_model_info') else '',
            'saved_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        try:
            joblib.dump(payload, filename)
            messagebox.showinfo('✅ Sukces', f'Model zapisany jako\n{filename}')
        except Exception as e:
            messagebox.showerror('❌ Błąd', f'Nie udało się zapisać modelu:\n{e}')

    def _wczytaj_model(self):
        filename = filedialog.askopenfilename(title='Wczytaj zapisany model',
            filetypes=[('Plik modelu', '*.joblib'), ('Wszystkie pliki', '*.*')])
        if not filename:
            return
        try:
            payload = joblib.load(filename)
            self.prediction_pipeline = payload['pipeline']
            self.prediction_features = payload.get('features', [])
            self.prediction_feature_order = payload.get('feature_order', self.prediction_features)
            self.prediction_model_info = payload.get('model_info', 'Model wczytany z pliku.')
            self.prediction_model_source = "zewnętrzny"
            self.loaded_model_path = filename
            if hasattr(self, 'calc_info_label'):
                self.calc_info_label.config(text=self.prediction_model_info)
            messagebox.showinfo('✅ Sukces', f'Model został wczytany:\n{filename}')
        except Exception as e:
            messagebox.showerror('❌ Błąd', f'Nie udało się wczytać modelu:\n{e}')

    def _zastosuj_model_do_pliku(self):
        if self.prediction_pipeline is None or len(self.prediction_features) == 0:
            messagebox.showwarning('⚠️ Uwaga', 'Najpierw zbuduj albo wczytaj model.')
            return
        filename = filedialog.askopenfilename(title='Wybierz plik z nowymi pacjentami',
            filetypes=[('Excel', '*.xlsx *.xls'), ('CSV', '*.csv'), ('Wszystkie pliki', '*.*')])
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
                if col in self.parametry_kliniczne:
                    df_new[col] = pd.to_numeric(df_new[col].astype(str).str.replace(',', '.'), errors='coerce')
            for col in self.zmienne_log:
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
            outname = filedialog.asksaveasfilename(title='Zapisz wyniki predykcji',
                defaultextension='.xlsx', filetypes=[('Excel', '*.xlsx'), ('CSV', '*.csv')],
                initialfile='predykcja_hospitalizacji.xlsx')
            if not outname:
                return
            if outname.lower().endswith('.csv'):
                df_pred.to_csv(outname, sep=';', index=False, encoding='utf-8')
            else:
                df_pred.to_excel(outname, index=False)
            messagebox.showinfo('✅ Sukces', f'Zastosowano model do nowego pliku.\nWynik zapisano jako:\n{outname}')
        except Exception as e:
            messagebox.showerror('❌ Błąd', f'Nie udało się zastosować modelu:\n{e}')

    def _pokaz_waznosc_modelu(self):
        if self.prediction_pipeline is None or len(self.prediction_features) == 0:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj dane i zbuduj model.")
            return
        try:
            logreg = self.prediction_pipeline.named_steps["logreg"]
            coef = logreg.coef_[0]
            odds = np.exp(coef)
            df_imp = pd.DataFrame({"etykieta": [pretty_name(x) for x in self.prediction_features],
                "beta_stand": coef, "OR_na_1SD": odds,
                "kierunek": ["zwiększa ryzyko" if c > 0 else "zmniejsza ryzyko" for c in coef]})
            df_imp = df_imp.sort_values("beta_stand", key=lambda s: s.abs(), ascending=False)
            msg = "WAŻNOŚĆ CECH MODELU\n" + "="*60 + "\n" + "\n".join(
                [f"• {r.etykieta}: beta={r.beta_stand:.3f} | OR/1SD={r.OR_na_1SD:.2f} | {r.kierunek}" for r in df_imp.itertuples()])
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
            self.otwarte_figury.append(fig)
        except Exception as e:
            messagebox.showerror("❌ Błąd", f"Nie udało się pokazać ważności cech:\n{e}")

    # ===== ZAKŁADKA 4 - WYKRESY =====
    def _tab4_wykresy(self):
        main_frame = ttk.Frame(self.tab4, padding="20")
        main_frame.pack(fill="both", expand=True)
        control_frame = tk.LabelFrame(main_frame, text="🎯 WYBIERZ PARAMETR DO WIZUALIZACJI",
            font=("Helvetica", 12, "bold"), bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=20, pady=15)
        control_frame.pack(fill="x", pady=(0, 20))
        top_row = tk.Frame(control_frame, bg=KOLORY["light"])
        top_row.pack(fill="x", pady=(0, 8))
        ttk.Label(top_row, text="Parametr:", font=("Helvetica", 11)).pack(side="left", padx=10)
        self.plot_param_var = tk.StringVar()
        self.plot_combo = ttk.Combobox(top_row, textvariable=self.plot_param_var,
            values=self.parametry_kliniczne, width=28, state="readonly", font=("Helvetica", 11))
        self.plot_combo.pack(side="left", padx=10)
        ttk.Label(top_row, text="Rodzaj wykresu:", font=("Helvetica", 11)).pack(side="left", padx=(15, 10))
        self.plot_type_var = tk.StringVar(value="Boxplot + punkty")
        self.plot_type_combo = ttk.Combobox(top_row, textvariable=self.plot_type_var,
            values=["Boxplot + punkty", "Boxplot", "Histogram", "Violinplot"], width=18, state="readonly", font=("Helvetica", 11))
        self.plot_type_combo.pack(side="left", padx=10)
        btn_row = tk.Frame(control_frame, bg=KOLORY["light"])
        btn_row.pack(fill="x")
        plot_btn = self._create_button(btn_row, "📈 GENERUJ WYKRES", self._rysuj_wykres, KOLORY["accent1"], font_size=11, padx=12, pady=8)
        plot_btn.pack(side="left", padx=5)
        roc_multi_btn = self._create_button(btn_row, "📊 ROC WIELU PARAMETRÓW", self._rysuj_roc_porownawcze, KOLORY["secondary"], font_size=11, padx=12, pady=8)
        roc_multi_btn.pack(side="left", padx=5)
        dca_btn = self._create_button(btn_row, "📉 DCA", self._rysuj_decision_curve, KOLORY["primary"], font_size=11, padx=12, pady=8)
        dca_btn.pack(side="left", padx=5)
        all_plots_btn = self._create_button(btn_row, "🖼️ GENERUJ WSZYSTKIE\nWYKRESY", self._generuj_wszystkie_wykresy, KOLORY["warning"], font_size=11, padx=12, pady=8, wraplength=300, justify="center")
        all_plots_btn.pack(side="right", padx=5)
        save_btn = self._create_button(btn_row, "💾 ZAPISZ WYKRES", self._zapisz_wykres, KOLORY["success"], font_size=11, padx=12, pady=8)
        save_btn.pack(side="right", padx=5)
        plot_frame = tk.LabelFrame(main_frame, text="📊 WYKRES", font=("Helvetica", 12, "bold"),
            bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=20, pady=15)
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

    def _rysuj_wykres_na_ax(self, ax, param, plot_type, hosp, dom):
        _, p = stats.mannwhitneyu(hosp, dom, alternative="two-sided")
        log_scale = param in ["troponina_i", "crp"]
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
        stats_text = (f"Przyjęci:\nn = {len(hosp)}\nśr = {hosp.mean():.2f} ± {hosp.std():.2f}\n"
            f"mediana = {np.median(hosp):.2f}\n\nWypisani:\nn = {len(dom)}\n"
            f"śr = {dom.mean():.2f} ± {dom.std():.2f}\nmediana = {np.median(dom):.2f}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), fontsize=9)
        return p

    def _rysuj_wykres(self):
        param = self.plot_param_var.get()
        plot_type = self.plot_type_var.get()
        if not param:
            messagebox.showwarning("⚠️ Uwaga", "Wybierz parametr do wizualizacji!")
            return
        if not self._sprawdz_czy_dane_wczytane():
            return
        hosp, dom = self._przygotuj_dane_dla_parametru(param)
        if hosp is None or dom is None:
            messagebox.showwarning("⚠️ Uwaga", f"Brak danych dla parametru {param}")
            return
        self.current_param = param
        self.ax.clear()
        self._rysuj_wykres_na_ax(self.ax, param, plot_type, hosp, dom)
        self.figure.tight_layout()
        self.canvas.draw()

    def _generuj_wszystkie_wykresy(self):
        if not self._sprawdz_czy_dane_wczytane():
            return
        plot_type = self.plot_type_var.get()
        folder = filedialog.askdirectory(title="Wybierz folder do zapisu wszystkich wykresów")
        if not folder:
            return
        zapisane = 0
        pominiete = 0
        for param in self.parametry_kliniczne:
            hosp, dom = self._przygotuj_dane_dla_parametru(param)
            if hosp is None or dom is None:
                pominiete += 1
                continue
            fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
            ax.set_facecolor("#f8f9fa")
            self._rysuj_wykres_na_ax(ax, param, plot_type, hosp, dom)
            fig.tight_layout()
            safe_name = (param.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
                .replace("?", "_").replace('"', "_").replace("<", "_").replace(">", "_").replace("|", "_").replace(" ", "_"))
            nazwa_pliku = f"wykres_{safe_name}_{plot_type.replace(' ', '_').replace('+', 'plus')}.png"
            sciezka = os.path.join(folder, nazwa_pliku)
            fig.savefig(sciezka, dpi=300, bbox_inches="tight")
            plt.close(fig)
            zapisane += 1
        messagebox.showinfo("✅ Gotowe", f"Wygenerowano {zapisane} wykresów.\nPominięto {pominiete} parametrów bez danych.\n\nFolder:\n{folder}")

    def _rysuj_roc_porownawcze(self):
        if not self._sprawdz_czy_dane_wczytane():
            return
        kandydaci = []
        for col in (self.zmienne_obowiazkowe + self.zmienne_dodatkowe):
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
                silnik = StatisticsEngine(self.df)
                df_pred, probs, valid_mask = self._oblicz_prob_z_modelu_i_df(self.df)
                y = df_pred.loc[valid_mask, "outcome"]
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
        self.ax.text(0.05, 0.05, "⚠️ UWAGA: Wyniki na danych treningowych\nOcena może być optymistyczna.\nBrak walidacji zewnętrznej.",
            transform=self.ax.transAxes, fontsize=8, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
        self.current_param = "ROC_porownawcze"
        self.figure.tight_layout()
        self.canvas.draw()

    def _rysuj_decision_curve(self):
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
            self.ax.text(0.05, 0.05, "⚠️ UWAGA: Wyniki na danych treningowych\nOcena może być optymistyczna.",
                transform=self.ax.transAxes, fontsize=8, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
            self.current_param = "decision_curve_analysis"
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("❌ Błąd", f"Nie udało się wygenerować DCA:\n{e}")

    def _zapisz_wykres(self):
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

    # ===== ZAKŁADKA 5 - RAPORT =====
    def _tab5_raport(self):
        main_frame = ttk.Frame(self.tab5, padding="20")
        main_frame.pack(fill="both", expand=True)
        btn_frame = tk.Frame(main_frame, bg=KOLORY["light"])
        btn_frame.pack(fill="x", pady=(0, 20))
        btn1 = self._create_button(btn_frame, "📊 GENERUJ RAPORT", self._generuj_raport, KOLORY["accent1"], font_size=12, padx=25, pady=12)
        btn1.pack(side="left", padx=10)
        btn2 = self._create_button(btn_frame, "💾 EKSPORTUJ DO CSV", self._export_csv, KOLORY["success"], font_size=12, padx=25, pady=12)
        btn2.pack(side="left", padx=10)
        btn_pdf = self._create_button(btn_frame, "📄 EKSPORTUJ DO PDF", self._eksportuj_pdf, KOLORY["primary"], font_size=12, padx=25, pady=12)
        btn_pdf.pack(side="left", padx=10)
        btn3 = self._create_button(btn_frame, "🔄 ODŚWIEŻ", self._odswiez_raport, KOLORY["warning"], font_size=12, padx=25, pady=12)
        btn3.pack(side="left", padx=10)
        report_frame = tk.LabelFrame(main_frame, text="📋 RAPORT KOŃCOWY", font=("Helvetica", 14, "bold"),
            bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=20, pady=15)
        report_frame.pack(fill="both", expand=True)
        text_frame = tk.Frame(report_frame, bg=KOLORY["light"])
        text_frame.pack(fill="both", expand=True)
        self.report_text = tk.Text(text_frame, font=("Courier", 11), bg="white", fg=KOLORY["dark"],
            wrap="word", padx=15, pady=15)
        self.report_text.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(text_frame, command=self.report_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.report_text.config(yscrollcommand=scrollbar.set)

    def _generuj_raport(self):
        if not self._sprawdz_czy_dane_wczytane():
            return
        self._odswiez_raport()
        filename = filedialog.asksaveasfilename(defaultextension=".txt",
            filetypes=[("Pliki tekstowe", "*.txt")], initialfile="raport_medyczny.txt")
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(self.report_text.get("1.0", tk.END))
                messagebox.showinfo("✅ Sukces", f"Raport zapisany jako:\n{filename}")
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Nie udało się zapisać pliku:\n{e}")

    def _odswiez_raport(self):
        self.report_text.delete("1.0", tk.END)
        if not self._sprawdz_czy_dane_wczytane():
            self.report_text.insert("1.0", "Brak danych. Wczytaj plik w zakładce 'WCZYTAJ DANE'.")
            return
        if self.current_mode == "podstawowa":
            wyniki = []
            istotne = []
            for param in self.parametry_kliniczne:
                hosp, dom = self._przygotuj_dane_dla_parametru(param)
                if hosp is None or dom is None:
                    continue
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
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                kierunek = "⬆️ WYŻSZE" if roznica > 0 else "⬇️ NIŻSZE"
                raport += f"\n  {i}. {pretty_name(param):<25}\n     p = {p:.6f} {stars}\n     {kierunek} u przyjętych (różnica średnich: {roznica:+.2f})\n"
        else:
            wyniki = []
            p_values = []
            for param in self.parametry_kliniczne:
                hosp, dom = self._przygotuj_dane_dla_parametru(param)
                if hosp is None or dom is None:
                    continue
                p_raw = stats.mannwhitneyu(hosp, dom, alternative="two-sided").pvalue
                d = cliff_delta(hosp, dom)
                wyniki.append({"parametr": param, "etykieta": pretty_name(param),
                    "hosp_med": f"{hosp.median():.2f} [{hosp.quantile(0.25):.2f}-{hosp.quantile(0.75):.2f}]",
                    "dom_med": f"{dom.median():.2f} [{dom.quantile(0.25):.2f}-{dom.quantile(0.75):.2f}]",
                    "p_raw": p_raw, "delta": d, "efekt": interpret_cliff_delta(d)})
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
                raport += f"\n  {i}. {row['etykieta']}\n     hosp: {row['hosp_med']}\n     dom : {row['dom_med']}\n     p raw = {row['p_raw']:.6f}\n     p FDR = {row['p_fdr']:.6f}\n     Cliff delta = {row['delta']:.3f} ({row['efekt']})\n"
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

    def _export_csv(self):
        if self.wyniki_df is None:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wykonaj analizę!")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV", "*.csv")], initialfile="wyniki_analizy.csv")
        if filename:
            try:
                self.wyniki_df.to_csv(filename, sep=";", index=False, encoding="utf-8")
                messagebox.showinfo("✅ Sukces", f"Wyniki zapisane jako:\n{filename}")
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Nie udało się zapisać pliku:\n{e}")

    def _eksportuj_pdf(self):
        if not self._sprawdz_czy_dane_wczytane():
            return
        self._odswiez_raport()
        filename = filedialog.asksaveasfilename(defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")], initialfile="raport_medyczny.pdf")
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

    # ===== ZAKŁADKA 6 - O PROGRAMIE =====
    def _tab6_info(self):
        frame = ttk.Frame(self.tab6, padding="30")
        frame.pack(fill="both", expand=True)
        info_text = f"""
╔══════════════════════════════════════════════════════════════╗
║         ANALIZATOR DANYCH MEDYCZNYCH - WERSJA 14.2          ║
╚══════════════════════════════════════════════════════════════╝

📋 OPIS PROGRAMU:
──────────────────────────────────────────────────────────────
Program służy do porównawczej analizy danych medycznych
pomiędzy pacjentami przyjętymi do szpitala a wypisanymi do domu.

🧮 NOWOŚĆ:
──────────────────────────────────────────────────────────────
Trzecia zakładka to kalkulator prawdopodobieństwa hospitalizacji.

🔧 TRYBY ANALIZY:
──────────────────────────────────────────────────────────────
📊 PODSTAWOWY: Podstawowe statystyki, Test Manna-Whitneya, Wykresy pudełkowe
📈 PROFESJONALNY: Tabela 1, Analiza jednoczynnikowa z FDR, Regresja logistyczna,
                  Forest plot, ROC i kalibracja, Raport końcowy

⚠️ WAŻNE OGRANICZENIA:
──────────────────────────────────────────────────────────────
• Wyniki odnoszą się WYŁĄCZNIE do wczytanej kohorty
• BRAK walidacji zewnętrznej modelu predykcyjnego
• Progi kliniczne mają charakter eksploracyjny
• Model hospitalizacji używa imputacji medianą (bez przecieku)

👩‍⚕️ AUTOR: Aneta
📅 Data kompilacji: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        label = tk.Label(frame, text=info_text, font=("Courier", 11), bg="white",
            fg=KOLORY["dark"], justify="left", padx=30, pady=30)
        label.pack(fill="both", expand=True)

    # ===== ZAKŁADKA 7 - SKALA RYZYKA =====
    def _tab7_skala(self):
        main_frame = ttk.Frame(self.tab7, padding="20")
        main_frame.pack(fill="both", expand=True)
        top_frame = tk.LabelFrame(main_frame, text="📌 SKALA RYZYKA HOSPITALIZACJI",
            font=("Helvetica", 12, "bold"), bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=20, pady=15)
        top_frame.pack(fill="x", pady=(0, 15))
        self.scale_info_label = tk.Label(top_frame,
            text="Najpierw wczytaj dane, potem kliknij: 'Generuj skalę z aktualnej bazy'.",
            font=("Helvetica", 11), bg=KOLORY["light"], fg=KOLORY["dark"], justify="left")
        self.scale_info_label.pack(anchor="w", pady=5)
        btn_frame = tk.Frame(top_frame, bg=KOLORY["light"])
        btn_frame.pack(fill="x", pady=(10, 0))
        gen_btn = self._create_button(btn_frame, "📌 GENERUJ SKALĘ", self._generuj_skale_ryzyka, KOLORY["accent1"], font_size=11, padx=20, pady=8)
        gen_btn.pack(side="left", padx=5)
        freeze_btn = self._create_button(btn_frame, "❄️ ZAMROŹ SKALĘ", self._zamroz_skale_ryzyka, KOLORY["warning"], font_size=11, padx=20, pady=8)
        freeze_btn.pack(side="left", padx=5)
        compare_btn = self._create_button(btn_frame, "🆚 PORÓWNAJ SKALE", self._porownaj_skale_ryzyka, KOLORY["success"], font_size=11, padx=20, pady=8)
        compare_btn.pack(side="left", padx=5)
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill="both", expand=True)
        left_frame = tk.LabelFrame(middle_frame, text="📋 AKTUALNA SKALA",
            font=("Helvetica", 12, "bold"), bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=15, pady=15)
        left_frame.pack(side="left", fill="both", expand=False, padx=(0, 10))
        columns = ("lp", "parametr", "kierunek", "prog", "punkty", "auc")
        self.scale_tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=12)
        self.scale_tree.heading("lp", text="LP")
        self.scale_tree.heading("parametr", text="Parametr")
        self.scale_tree.heading("kierunek", text="Kierunek")
        self.scale_tree.heading("prog", text="Próg")
        self.scale_tree.heading("punkty", text="Punkty")
        self.scale_tree.heading("auc", text="AUC")
        for col, w in [("lp", 50), ("parametr", 180), ("kierunek", 90), ("prog", 90), ("punkty", 70), ("auc", 70)]:
            self.scale_tree.column(col, width=w, anchor="center")
        scale_tree_scroll = tk.Scrollbar(left_frame, orient="vertical", command=self.scale_tree.yview)
        self.scale_tree.configure(yscrollcommand=scale_tree_scroll.set)
        self.scale_tree.pack(side="left", fill="both", expand=True)
        scale_tree_scroll.pack(side="right", fill="y")
        right_frame = tk.LabelFrame(middle_frame, text="🧮 KALKULATOR SKALI + PORÓWNAJ",
            font=("Helvetica", 12, "bold"), bg=KOLORY["light"], fg=KOLORY["dark"], relief="flat", bd=2, padx=15, pady=15)
        right_frame.pack(side="left", fill="both", expand=True)
        top_calc_row = tk.Frame(right_frame, bg=KOLORY["light"])
        top_calc_row.pack(fill="x", pady=(0, 10))
        self.scale_entries_frame = tk.Frame(top_calc_row, bg=KOLORY["light"])
        self.scale_entries_frame.pack(side="left", fill="x", expand=True)
        self._zbuduj_pola_skali()
        calc_btn_col = tk.Frame(top_calc_row, bg=KOLORY["light"])
        calc_btn_col.pack(side="left", fill="y", padx=(12, 0))
        calc_btn = self._create_button(calc_btn_col, "🧮 OBLICZ WYNIK SKALI", self._oblicz_wynik_skali, KOLORY["primary"], font_size=11, padx=18, pady=10)
        calc_btn.pack(fill="x")
        self.scale_result_big_label = tk.Label(right_frame, text="—", font=("Helvetica", 26, "bold"),
            bg="white", fg=KOLORY["primary"], height=1, anchor="center")
        self.scale_result_big_label.pack(fill="x", pady=(0, 8))
        result_frame = tk.Frame(right_frame, bg=KOLORY["light"])
        result_frame.pack(fill="both", expand=True)
        scale_result_scroll = tk.Scrollbar(result_frame)
        scale_result_scroll.pack(side="right", fill="y")
        self.scale_result_text = tk.Text(result_frame, font=("Courier", 10), bg="white", fg=KOLORY["dark"],
            wrap="word", padx=12, pady=12, yscrollcommand=scale_result_scroll.set)
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
            lbl = tk.Label(self.scale_entries_frame, text="Brak aktywnej skali. Najpierw wygeneruj skalę.",
                font=("Helvetica", 10), bg=KOLORY["light"], fg=KOLORY["dark"])
            lbl.pack(anchor="w")
            return
        params = scale_df["parametr"].tolist()
        for param in params:
            if param == "MAP":
                continue
            self.scale_input_vars[param] = tk.StringVar()
            row = tk.Frame(self.scale_entries_frame, bg=KOLORY["light"])
            row.pack(fill="x", pady=3)
            tk.Label(row, text=pretty_name(param), font=("Helvetica", 10),
                bg=KOLORY["light"], fg=KOLORY["dark"], width=28, anchor="w").pack(side="left", padx=(0, 8))
            tk.Entry(row, textvariable=self.scale_input_vars[param], font=("Helvetica", 10), width=18).pack(side="left")
        if "MAP" in params:
            self.scale_rr_skurczowe_var = tk.StringVar()
            row_rr1 = tk.Frame(self.scale_entries_frame, bg=KOLORY["light"])
            row_rr1.pack(fill="x", pady=3)
            tk.Label(row_rr1, text="RR skurczowe, mmHg", font=("Helvetica", 10),
                bg=KOLORY["light"], fg=KOLORY["dark"], width=28, anchor="w").pack(side="left", padx=(0, 8))
            tk.Entry(row_rr1, textvariable=self.scale_rr_skurczowe_var, font=("Helvetica", 10), width=18).pack(side="left")
            self.scale_rr_rozkurczowe_var = tk.StringVar()
            row_rr2 = tk.Frame(self.scale_entries_frame, bg=KOLORY["light"])
            row_rr2.pack(fill="x", pady=3)
            tk.Label(row_rr2, text="RR rozkurczowe, mmHg", font=("Helvetica", 10),
                bg=KOLORY["light"], fg=KOLORY["dark"], width=28, anchor="w").pack(side="left", padx=(0, 8))
            tk.Entry(row_rr2, textvariable=self.scale_rr_rozkurczowe_var, font=("Helvetica", 10), width=18).pack(side="left")
            row_map = tk.Frame(self.scale_entries_frame, bg=KOLORY["light"])
            row_map.pack(fill="x", pady=(6, 3))
            tk.Label(row_map, text="Wyliczone MAP, mmHg", font=("Helvetica", 10, "bold"),
                bg=KOLORY["light"], fg=KOLORY["dark"], width=28, anchor="w").pack(side="left", padx=(0, 8))
            self.scale_map_live_label = tk.Label(row_map, text="—", font=("Helvetica", 10, "bold"),
                bg="white", fg=KOLORY["primary"], width=18, anchor="center", relief="solid", bd=1)
            self.scale_map_live_label.pack(side="left")
            self.scale_rr_skurczowe_var.trace_add("write", lambda *args: self._aktualizuj_map_skali_na_zywo())
            self.scale_rr_rozkurczowe_var.trace_add("write", lambda *args: self._aktualizuj_map_skali_na_zywo())

    def _aktualizuj_map_skali_na_zywo(self):
        if self.scale_map_live_label is None:
            return
        sbp = bezpieczna_liczba(self.scale_rr_skurczowe_var.get())
        dbp = bezpieczna_liczba(self.scale_rr_rozkurczowe_var.get())
        if pd.isna(sbp) or pd.isna(dbp):
            self.scale_map_live_label.config(text="—")
            return
        map_val = (sbp + 2 * dbp) / 3
        self.scale_map_live_label.config(text=f"{map_val:.1f}")

    def _generuj_skale_ryzyka(self):
        if not self._sprawdz_czy_dane_wczytane():
            return
        mode = messagebox.askquestion("Tryb generowania skali",
            "Wybierz 'Tak' dla trybu DOKŁADNEGO (500 bootstrapów, wolniejszy)\nlub 'Nie' dla trybu SZYBKIEGO (100 bootstrapów).")
        bootstrap_mode = "dokladny" if mode == 'yes' else "szybki"
        silnik = StatisticsEngine(self.df)
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Generowanie skali")
        progress_win.geometry("400x100")
        tk.Label(progress_win, text=f"Generowanie skali (tryb {bootstrap_mode})...").pack(pady=10)
        pb = ttk.Progressbar(progress_win, length=350, mode='determinate')
        pb.pack(pady=10)
        def update_progress(current, total):
            pb['value'] = (current / total) * 100
            progress_win.update_idletasks()
        df_scale, meta = silnik.generuj_skale_ryzyka_z_bootstrapem(mode=bootstrap_mode, progress_callback=update_progress)
        progress_win.destroy()
        if df_scale is None:
            messagebox.showerror("❌ Błąd", meta.get("error", "Nie udało się wygenerować skali"))
            return
        self.scale_current_df = df_scale
        self.scale_current_meta = meta
        self._odswiez_tabele_skali()
        self._zbuduj_pola_skali()
        self.scale_info_label.config(text=f"Skala dynamiczna gotowa.\n• liczba pacjentów: {len(self.df)}\n"
            f"• liczba parametrów w skali: {len(df_scale)}\n• AUC skali: {meta.get('auc', 0):.3f}\n"
            f"• bootstrap: {meta.get('bootstrap_iterations', 0)} iteracji\n• wygenerowano: {meta.get('data', '—')}")
        self.scale_result_text.delete("1.0", tk.END)
        self.scale_result_text.insert("1.0", "Wygenerowano nową skalę z aktualnej bazy.\n\n"
            "To jest wersja dynamiczna — może zmieniać się po dodaniu nowych pacjentów.")

    def _odswiez_tabele_skali(self):
        for row in self.scale_tree.get_children():
            self.scale_tree.delete(row)
        scale_df = self.scale_current_df if self.scale_current_df is not None else self.scale_frozen_df
        if scale_df is None or len(scale_df) == 0:
            return
        for i, row in scale_df.iterrows():
            prog_text = f"{row['prog_mediana']:.2f} (95%CI: {row['prog_ci_low']:.2f}-{row['prog_ci_high']:.2f})"
            self.scale_tree.insert("", "end", values=(
                i + 1, row["etykieta"], "wyższe" if row.get("kierunek", "wyższe") == "wyższe" else "niższe",
                prog_text, int(row["punkty"]), f"{row['auc_mediana']:.3f}"
            ))

    def _zamroz_skale_ryzyka(self):
        if self.scale_current_df is None or len(self.scale_current_df) == 0:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wygeneruj skalę.")
            return
        self.scale_frozen_df = self.scale_current_df.copy()
        self.scale_frozen_meta = self.scale_current_meta.copy()
        self.scale_result_text.delete("1.0", tk.END)
        self.scale_result_text.insert("1.0", "Aktualna skala została zamrożona.\n\n"
            "To znaczy, że możesz ją traktować jako wersję finalną do porównań z przyszłymi bazami.")
        messagebox.showinfo("✅ Gotowe", "Skala została zamrożona.")

    def _porownaj_skale_ryzyka(self):
        self.scale_result_text.delete("1.0", tk.END)
        if self.scale_frozen_df is None or len(self.scale_frozen_df) == 0:
            self.scale_result_text.insert("1.0", "Brak skali zamrożonej.\n\nNajpierw wygeneruj skalę, a potem kliknij 'Zamroź skalę'.")
            return
        silnik = StatisticsEngine(self.df)
        auc_frozen = silnik._policz_auc_skali(self.scale_frozen_df, self.df)
        auc_current = np.nan
        if self.scale_current_df is not None and len(self.scale_current_df) > 0:
            auc_current = silnik._policz_auc_skali(self.scale_current_df, self.df)
        tekst = ["PORÓWNANIE SKAL", "=" * 60, "Skala zamrożona:",
            f"• pacjenci przy budowie: {self.scale_frozen_meta.get('n', '—')}",
            f"• data: {self.scale_frozen_meta.get('data', '—')}",
            f"• AUC na aktualnej bazie: {auc_frozen:.3f}" if not pd.isna(auc_frozen) else "• AUC na aktualnej bazie: brak", ""]
        if self.scale_current_df is not None and len(self.scale_current_df) > 0:
            tekst.extend(["Skala aktualna (dynamiczna):",
                f"• pacjenci przy budowie: {self.scale_current_meta.get('n', '—')}",
                f"• data: {self.scale_current_meta.get('data', '—')}",
                f"• AUC na aktualnej bazie: {auc_current:.3f}" if not pd.isna(auc_current) else "• AUC na aktualnej bazie: brak", "",
                "Parametry skali aktualnej:"])
            for _, row in self.scale_current_df.iterrows():
                tekst.append(f"• {row['etykieta']} | wyższe/niższe? | próg {row['prog_mediana']:.2f} | {int(row['punkty'])} pkt")
            tekst.extend(["", "Parametry skali zamrożonej:"])
            for _, row in self.scale_frozen_df.iterrows():
                tekst.append(f"• {row['etykieta']} | wyższe/niższe? | próg {row['prog_mediana']:.2f} | {int(row['punkty'])} pkt")
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

    def _oblicz_wynik_skali(self):
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
            if rule.get("kierunek", "wyższe") == "wyższe" and val >= rule["prog_mediana"]:
                czy_spelnia = True
            if rule.get("kierunek", "wyższe") == "niższe" and val <= rule["prog_mediana"]:
                czy_spelnia = True
            poza_norma = self._czy_wartosc_poza_norma_dla_skali(param, val, rule.get("kierunek", "wyższe"))
            if czy_spelnia and poza_norma:
                suma += int(rule["punkty"])
                trafienia.append(f"• {rule['etykieta']}: {val:.2f} (spełnia próg, poza normą) → +{int(rule['punkty'])} pkt")
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
        tekst = ["WYNIK SKALI RYZYKA", "=" * 60, f"Suma punktów: {suma} / {max_pkt}",
            f"Kategoria skali: {kat}", f"Kategoria końcowa: {kat_koncowa}", ""]
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

    # ===== METODY POMOCNICZE DLA TRYBU PROFESJONALNEGO =====
    def _oblicz_prob_z_modelu_i_df(self, df_in):
        df_pred = df_in.copy()
        for col in self.zmienne_log:
            if col in df_pred.columns:
                df_pred[col] = konwertuj_kolumne_na_liczby(df_pred[col])
                df_pred[f"log_{col}"] = np.log1p(df_pred[col].clip(lower=0))
        valid_mask = pd.Series(True, index=df_pred.index)
        for feat in self.prediction_features:
            if feat not in df_pred.columns:
                valid_mask &= False
            else:
                valid_mask &= df_pred[feat].notna()
        X = df_pred.loc[valid_mask, self.prediction_features].astype(float)
        probs = pd.Series(index=df_pred.index, dtype=float)
        if len(X) > 0 and self.prediction_pipeline is not None:
            probs.loc[valid_mask] = self.prediction_pipeline.predict_proba(X.values)[:, 1]
        return df_pred, probs, valid_mask

    def _raport_brakow_pro(self, df):
        wyniki = []
        for col in df.columns:
            n_brakow = int(df[col].isna().sum())
            proc = (n_brakow / len(df)) * 100 if len(df) > 0 else 0
            wyniki.append({"kolumna": col, "braki": n_brakow, "procent": round(proc, 2)})
        return pd.DataFrame(wyniki)

    def _walidacja_zakresow_pro(self, df):
        wyniki = []
        for col, (min_bio, max_bio) in self.zakresy_biologiczne.items():
            if col in df.columns:
                dane = konwertuj_kolumne_na_liczby(df[col]).dropna()
                if len(dane) > 0:
                    mask = (dane < min_bio) | (dane > max_bio)
                    wyniki.append({"kolumna": col, "poza_zakresem": int(mask.sum()), "min_bio": min_bio, "max_bio": max_bio})
        return pd.DataFrame(wyniki)

    def _tabela_1_pro(self, df):
        wyniki = []
        for param in self.parametry_kliniczne:
            if param in df.columns:
                hosp = df[df["outcome"] == 1][param].dropna()
                dom = df[df["outcome"] == 0][param].dropna()
                if len(hosp) > 0 and len(dom) > 0:
                    p = stats.mannwhitneyu(hosp, dom).pvalue
                    d = cliff_delta(hosp, dom)
                    wyniki.append({"parametr": param, "etykieta": pretty_name(param),
                        "hospitalizowani": f"{hosp.median():.2f} [{hosp.quantile(0.25):.2f}-{hosp.quantile(0.75):.2f}]",
                        "wypisani": f"{dom.median():.2f} [{dom.quantile(0.25):.2f}-{dom.quantile(0.75):.2f}]",
                        "p_value": p, "effect_size": d, "interpretacja": interpret_cliff_delta(d)})
        for choroba in self.choroby:
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
                    or_val = (a * d) / (b * c) if (b * c) > 0 else float('inf')
                    wyniki.append({"parametr": choroba, "etykieta": pretty_name(choroba),
                        "hospitalizowani": f"{hosp_tak}/{len(hosp)} ({100*hosp_tak/len(hosp):.1f}%)",
                        "wypisani": f"{dom_tak}/{len(dom)} ({100*dom_tak/len(dom):.1f}%)",
                        "p_value": p, "effect_size": or_val, "interpretacja": "OR"})
        return pd.DataFrame(wyniki)

    def _analiza_jednoczynnikowa_pro(self, df):
        wyniki = []
        p_values = []
        for param in self.parametry_kliniczne:
            if param in df.columns:
                hosp = df[df["outcome"] == 1][param].dropna()
                dom = df[df["outcome"] == 0][param].dropna()
                if len(hosp) > 0 and len(dom) > 0:
                    p = stats.mannwhitneyu(hosp, dom).pvalue
                    d = cliff_delta(hosp, dom)
                    wyniki.append({"parametr": param, "etykieta": pretty_name(param),
                        "p_raw": p, "cliff_delta": d, "interpretacja": interpret_cliff_delta(d),
                        "n_hosp": len(hosp), "n_dom": len(dom)})
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
                wyniki.append({"parametr": param, "etykieta": pretty_name(param),
                    "braki_hosp": b1, "proc_hosp": round(100 * b1 / len(self.df_hosp), 2) if len(self.df_hosp) else 0,
                    "braki_dom": b0, "proc_dom": round(100 * b0 / len(self.df_dom), 2) if len(self.df_dom) else 0})
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
                if kierunek == "wyższe" and len(thresholds) > idx:
                    prog = thresholds[idx]
                elif len(thresholds) > idx:
                    prog = -thresholds[idx]
                else:
                    continue
                if pd.isna(prog):
                    continue
                if kierunek == "wyższe":
                    y_pred = (dane[param] >= prog).astype(int)
                else:
                    y_pred = (dane[param] <= prog).astype(int)
                tn = int(((y_pred == 0) & (dane["outcome"] == 0)).sum())
                fp = int(((y_pred == 1) & (dane["outcome"] == 0)).sum())
                fn = int(((y_pred == 0) & (dane["outcome"] == 1)).sum())
                tp = int(((y_pred == 1) & (dane["outcome"] == 1)).sum())
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                wyniki.append({"parametr": param, "etykieta": pretty_name(param),
                    "kierunek": kierunek, "prog": prog, "czulosc": sens, "swoistosc": spec})
            except Exception:
                continue
        return pd.DataFrame(wyniki)

    def _przygotuj_zmienne_modelu(self, df):
        df_model = df.copy()

        for col in self.parametry_kliniczne:
            if col in df_model.columns:
                df_model[col] = konwertuj_kolumne_na_liczby(df_model[col])

        if "outcome" in df_model.columns:
            df_model["outcome"] = pd.to_numeric(df_model["outcome"], errors="coerce")

        wszystkie = self.zmienne_obowiazkowe + self.zmienne_dodatkowe
        dostepne = [z for z in wszystkie if z in df_model.columns]

        for z in self.zmienne_log:
            if z in df_model.columns:
                new_name = f"log_{z}"
                df_model[z] = konwertuj_kolumne_na_liczby(df_model[z])
                df_model[new_name] = np.log1p(df_model[z].clip(lower=0))
                if z in dostepne:
                    dostepne.remove(z)
                if new_name not in dostepne:
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
            wyn = wyniki_modelu_statsmodels(model, ["wiek"])
            return model, wyn, len(df_cc)
        except Exception:
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
            vif = sprawdz_vif(X, include_constant=False)
            wyn = wyniki_modelu_statsmodels(model, dostepne)
            return model, wyn, len(df_cc), epv_ok, vif
        except Exception:
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
        priorytety = {"wiek": 10, "log_crp": 9, "SpO2": 8, "log_kreatynina": 7,
            "MAP": 6, "log_troponina_i": 5, "hgb": 4}
        dostepne = sorted(dostepne, key=lambda x: priorytety.get(x, 0), reverse=True)
        wybrane = dostepne[:max_pred]
        return self._model_rozszerzony(df, wybrane)

    def _forest_plot(self, wyniki, nazwa_pliku):
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
        self.otwarte_figury.append(fig)

    def _model_predykcyjny(self, df, zmienne):
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
        for i in range(200):
            idx = resample(range(len(y_test)), replace=True, random_state=i)
            if len(np.unique(y_test[idx])) < 2:
                continue
            auc_boot.append(roc_auc_score(y_test[idx], y_prob[idx]))
        auc_ci = (np.percentile(auc_boot, 2.5), np.percentile(auc_boot, 97.5)) if len(auc_boot) else (roc_auc, roc_auc)
        n_splits = dynamiczny_n_splits(y_train)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        fig1, ax1 = plt.subplots()
        ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax1.plot([0, 1], [0, 1], "k--")
        ax1.set_xlabel("1 - swoistość")
        ax1.set_ylabel("Czułość")
        ax1.set_title("Krzywa ROC (wewnętrzna walidacja)")
        ax1.legend()
        fig2, ax2 = plt.subplots()
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        ax2.plot(prob_pred, prob_true, marker="o")
        ax2.plot([0, 1], [0, 1], "k--")
        ax2.set_xlabel("Prawdopodobieństwo przewidywane")
        ax2.set_ylabel("Częstość obserwowana")
        ax2.set_title(f"Kalibracja (Brier = {brier:.4f})")
        self.otwarte_figury.extend([fig1, fig2])
        return {"auc": roc_auc, "auc_ci_low": auc_ci[0], "auc_ci_high": auc_ci[1],
            "brier": brier, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
            "fig_roc": fig1, "fig_cal": fig2}

    def _generuj_raport_tekstowy(self, top5, wyn_model_glowny, pred, epv_ok):
        lines = ["WYNIKI ANALIZY PROFESJONALNEJ", "",
            f"Do analizy włączono łącznie {len(self.df_hosp) + len(self.df_dom)} pacjentów, "
            f"w tym {len(self.df_hosp)} hospitalizowanych oraz {len(self.df_dom)} wypisanych do domu.", "",
            "TOP 5 PARAMETRÓW (wg istotności):"]
        for i, param in enumerate(top5[:5], 1):
            lines.append(f"  {i}. {pretty_name(param)}")
        if wyn_model_glowny is not None and len(wyn_model_glowny) > 0:
            sig = wyn_model_glowny[wyn_model_glowny["p_value"] < 0.05]
            if len(sig) > 0:
                lines.extend(["", "Niezależne czynniki ryzyka:"])
                for _, row in sig.iterrows():
                    lines.append(f"  • {row['etykieta']}: OR {row['OR']:.2f} (95% CI {row['CI_95%']}; p={row['p_value']:.4f})")
        if pred is not None:
            lines.extend(["", f"Model predykcyjny: AUC = {pred['auc']:.3f} (95% CI {pred['auc_ci_low']:.3f}-{pred['auc_ci_high']:.3f})"])
        lines.extend(["", "OGRANICZENIA:", "- Analiza oparta na complete-case analysis", "- Progi kliniczne mają charakter eksploracyjny"])
        if not epv_ok:
            lines.append("- Model główny ma ograniczone EPV - interpretować ostrożnie")
        lines.extend(["", "⚠️ UWAGA: Brak walidacji zewnętrznej - wyniki dotyczą wyłącznie tej kohorty."])
        return "\n".join(lines)

    def _analiza_profesjonalna(self):
        if not self._sprawdz_czy_dane_wczytane():
            return
        
        folder = filedialog.askdirectory(
            title="Wybierz folder do zapisu wyników analizy profesjonalnej",
            initialdir=os.getcwd()
        )
        if not folder:
            messagebox.showinfo("Anulowano", "Analiza została przerwana.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(folder, f"analiza_profesjonalna_{timestamp}")
        os.makedirs(output_folder, exist_ok=True) 
        
        messagebox.showinfo("🔬 Analiza profesjonalna", 
            f"Rozpoczynam analizę.\nWyniki zostaną zapisane w:\n{output_folder}\n\nMoże to potrwać chwilę.")
        
        try:
            df_caly = self.df.copy()
            
            # Zapis plików
            self._raport_brakow_pro(df_caly).to_csv(os.path.join(output_folder, "raport_brakow.csv"), sep=";", index=False)
            self._walidacja_zakresow_pro(df_caly).to_csv(os.path.join(output_folder, "walidacja_zakresow.csv"), sep=";", index=False)
            self._tabela_1_pro(df_caly).to_csv(os.path.join(output_folder, "tabela_1_publikacyjna.csv"), sep=";", index=False)
            
            wyniki_fdr, top5 = self._analiza_jednoczynnikowa_pro(df_caly)
            wyniki_fdr.to_csv(os.path.join(output_folder, "analiza_jednoczynnikowa_fdr.csv"), sep=";", index=False)
            self._missingness_top_pro(top5).to_csv(os.path.join(output_folder, "missingness_top5.csv"), sep=";", index=False)
            self._progi_kliniczne_pro(df_caly, top5).to_csv(os.path.join(output_folder, "progi_kliniczne_eksploracyjne.csv"), sep=";", index=False)
            
            df_model, zmienne_modelu = self._przygotuj_zmienne_modelu(df_caly)
            model1, wyn1, n1 = self._model_podstawowy(df_model)
            model2, wyn2, n2, epv_ok, vif2 = self._model_rozszerzony(df_model, zmienne_modelu)
            model3, wyn3, n3, _, vif3 = self._model_z_redukcja(df_model, zmienne_modelu)
            
            if wyn1 is not None:
                wyn1.to_csv(os.path.join(output_folder, "model_podstawowy.csv"), sep=";", index=False)
            if wyn2 is not None:
                wyn2.to_csv(os.path.join(output_folder, "model_glowny_rozszerzony.csv"), sep=";", index=False)
                self._forest_plot(wyn2, os.path.join(output_folder, "forest_plot_model_glowny.png"))
            if wyn3 is not None:
                wyn3.to_csv(os.path.join(output_folder, "model_redukowany_sensitivity.csv"), sep=";", index=False)
                self._forest_plot(wyn3, os.path.join(output_folder, "forest_plot_model_redukowany.png"))
            
            if vif2 is not None:
                vif2.to_csv(os.path.join(output_folder, "vif_model_glowny.csv"), sep=";", index=False)
            if vif3 is not None:
                vif3.to_csv(os.path.join(output_folder, "vif_model_redukowany.csv"), sep=";", index=False)
            
            pred = self._model_predykcyjny(df_model, zmienne_modelu)
            if pred is not None:
                pred["fig_roc"].savefig(os.path.join(output_folder, "krzywa_ROC.png"), dpi=300, bbox_inches="tight")
                pred["fig_cal"].savefig(os.path.join(output_folder, "krzywa_kalibracji.png"), dpi=300, bbox_inches="tight")
            
            with open(os.path.join(output_folder, "raport_wyniki_i_ograniczenia.txt"), "w", encoding="utf-8") as f:
                f.write(self._generuj_raport_tekstowy(top5, wyn2, pred, epv_ok))
            
            messagebox.showinfo("✅ Sukces", f"Analiza profesjonalna zakończona!\n\nWygenerowano pliki wynikowe w:\n{output_folder}")
            
        except Exception as e:
            messagebox.showerror("❌ Błąd", f"Wystąpił błąd podczas analizy:\n{str(e)}")


# =============================================================================
# URUCHOMIENIE APLIKACJI
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalAnalyzerGUI(root)
    root.mainloop()