# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:01:10 2026

@author: aneta
"""

# -*- coding: utf-8 -*-
"""
PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 14.0 - Ostateczna korekta wszystkich błędów
Autor: Aneta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact, rankdata
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, auc, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from math import log, exp, sqrt
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# KONFIGURACJA
# =============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)
np.seterr(divide='ignore', invalid='ignore')

KOLORY = {
    'hosp': '#e74c3c',
    'dom': '#3498db',
    'istotne': '#2ecc71',
    'tlo': '#f8f9fa',
    'warning': '#f39c12'
}

# =============================================================================
# ZAKRESY BIOLOGICZNE
# =============================================================================
ZAKRESY_BIOLOGICZNE = {
    'wiek': (0, 120),
    'RR': (0, 300),
    'MAP': (0, 200),
    'SpO2': (0, 100),
    'AS': (0, 300),
    'mleczany': (0, 30),
    'kreatynina(0,5-1,2)': (0, 20),
    'troponina I (0-7,8))': (0, 100000),
    'HGB(12,4-15,2)': (0, 25),
    'WBC(4-11)': (0, 100),
    'plt(130-450)': (0, 2000),
    'hct(38-45)': (0, 70),
    'Na(137-145)': (100, 160),
    'K(3,5-5,1)': (2, 8),
    'crp(0-0,5)': (0, 500)
}

# =============================================================================
# PARAMETRY KLINICZNE - PRE-DEFINIOWANE (sens kliniczny)
# =============================================================================
PARAMETRY_KONIECZNE = ['wiek']  # wiek zawsze w modelu
PARAMETRY_OPCJONALNE = [
    'SpO2', 'MAP', 'crp(0-0,5)', 'kreatynina(0,5-1,2)',
    'troponina I (0-7,8))', 'HGB(12,4-15,2)', 'WBC(4-11)'
]

# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================

def wczytaj_dane(sciezka_pliku, separator=';'):
    """Wczytuje dane z pliku CSV"""
    try:
        df = pd.read_csv(sciezka_pliku, sep=separator, encoding='utf-8')
        print(f"✓ Wczytano plik: {os.path.basename(sciezka_pliku)}")
        print(f"  Liczba wierszy: {len(df)}")
        print(f"  Liczba kolumn: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"✗ Błąd wczytywania: {e}")
        return None


def przygotuj_dane_z_outcome(df, id_pacjenta=None):
    """
    Przygotowuje dane - wymaga kolumny 'outcome'
    """
    df_copy = df.copy()
    
    if 'outcome' not in df_copy.columns:
        print("\n" + "="*70)
        print("❌ BRAK KOLUMNY 'outcome' W PLIKU!")
        print("="*70)
        print("\nDodaj kolumnę 'outcome' z wartościami:")
        print("  1 = hospitalizowani")
        print("  0 = do domu")
        return None, None, None
    
    # Sprawdź poprawność wartości
    df_copy = df_copy[df_copy['outcome'].notna()]
    df_copy['outcome'] = pd.to_numeric(df_copy['outcome'], errors='coerce')
    df_copy = df_copy[df_copy['outcome'].isin([0, 1])]
    
    if len(df_copy) == 0:
        print("✗ Brak poprawnych wartości w kolumnie 'outcome'")
        return None, None, None
    
    # Sprawdź duplikaty po ID jeśli dostępne
    if id_pacjenta and id_pacjenta in df_copy.columns:
        duplikaty_id = df_copy[id_pacjenta].duplicated().sum()
        if duplikaty_id > 0:
            print(f"\n⚠️ Znaleziono {duplikaty_id} duplikatów ID pacjenta")
            df_copy = df_copy.drop_duplicates(subset=[id_pacjenta], keep='first')
            print(f"  Po usunięciu: {len(df_copy)} unikalnych pacjentów")
    
    # Podział na grupy
    df_hosp = df_copy[df_copy['outcome'] == 1].copy()
    df_dom = df_copy[df_copy['outcome'] == 0].copy()
    
    print(f"\n✓ Podział danych (na podstawie kolumny 'outcome'):")
    print(f"  • Hospitalizowani (outcome=1): {len(df_hosp)} ({len(df_hosp)/len(df_copy)*100:.1f}%)")
    print(f"  • Do domu (outcome=0): {len(df_dom)} ({len(df_dom)/len(df_copy)*100:.1f}%)")
    print(f"  • Razem: {len(df_copy)}")
    
    return df_hosp, df_dom, df_copy


def konwertuj_na_numeryczne(df, kolumny):
    """Konwertuje kolumny na typ numeryczny"""
    df_copy = df.copy()
    
    for col in kolumny:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(
                df_copy[col].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
    
    return df_copy


def konwertuj_choroby(df, kolumny):
    """Konwertuje kolumny z chorobami na wartości binarne"""
    df_copy = df.copy()
    
    mapping_tak = ['tak', 't', 'yes', 'y', '1', 'true', '+', 'tak!', 'TAK', 'T']
    mapping_nie = ['nie', 'n', 'no', '0', 'false', '-', 'NIE', 'N']
    
    for col in kolumny:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.lower().str.strip()
            df_copy[col] = df_copy[col].apply(
                lambda x: 1 if x in mapping_tak else (0 if x in mapping_nie else np.nan)
            )
    
    return df_copy


def raport_brakow(df):
    """Generuje raport braków danych"""
    print("\n" + "="*70)
    print("RAPORT BRAKÓW DANYCH")
    print("="*70)
    
    for col in df.columns:
        n_brakow = df[col].isna().sum()
        proc_brakow = (n_brakow / len(df)) * 100
        if proc_brakow > 0:
            print(f"  {col:<30} braki: {n_brakow:3d} ({proc_brakow:5.1f}%)")


def walidacja_zakresow_biologicznych(df, zakresy):
    """Walidacja zakresów biologicznie możliwych"""
    print("\n" + "="*70)
    print("WALIDACJA ZAKRESÓW BIOLOGICZNYCH")
    print("="*70)
    
    znaleziono = False
    for col, (min_bio, max_bio) in zakresy.items():
        if col in df.columns:
            dane = df[col].dropna()
            if len(dane) > 0:
                poza = ((dane < min_bio) | (dane > max_bio)).sum()
                if poza > 0:
                    znaleziono = True
                    print(f"\n  ⚠️ {col}: {poza} wartości poza zakresem biologicznym")
                    print(f"     Zakres: {min_bio}-{max_bio}")
                    print(f"     Problemowe: {dane[((dane < min_bio) | (dane > max_bio))].tolist()}")
    
    if not znaleziono:
        print("  ✓ Wszystkie wartości w zakresach biologicznych")


def sprawdz_epv(df, zmienne, outcome='outcome', prog=10):
    """Sprawdza Events Per Variable dla modelu"""
    n_events = df[outcome].sum()
    n_vars = len(zmienne)
    epv = n_events / n_vars if n_vars > 0 else 0
    
    print(f"\n📊 Events Per Variable (EPV): {epv:.1f}")
    print(f"  • Liczba zdarzeń (hospitalizacji): {n_events}")
    print(f"  • Liczba predyktorów: {n_vars}")
    
    if epv < prog:
        print(f"  ⚠️ EPV < {prog} - model może być niestabilny!")
        print(f"  Zalecane: maksymalnie {int(n_events/prog)} predyktorów")
    else:
        print(f"  ✓ EPV OK (≥{prog})")
    
    return epv


# =============================================================================
# POPRAWIONY CLIFF'S DELTA
# =============================================================================

def cliff_delta_poprawiony(x, y):
    """
    Poprawna implementacja Cliff's delta z U-statystyki
    
    Returns:
        delta: wartość między -1 a 1
        interpretation: 
            |delta| < 0.147 - mały efekt
            |delta| < 0.33 - średni efekt
            |delta| >= 0.33 - duży efekt
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0
    
    # U-statystyka z Manna-Whitneya
    U, _ = stats.mannwhitneyu(x, y, alternative='two-sided', method='exact')
    
    # Cliff's delta = (2U)/(n1*n2) - 1
    delta = (2 * U) / (n1 * n2) - 1
    
    return delta


# =============================================================================
# ANALIZA OPISOWA
# =============================================================================

def tabela_opisowa_kompletna(df_hosp, df_dom, parametry_ciagle, choroby):
    """
    Kompletna Tabela 1: ciągłe + kategorialne
    """
    print("\n" + "="*80)
    print("TABELA 1: CHARAKTERYSTYKA KOHORTY")
    print("="*80)
    
    wyniki = []
    
    # Zmienne ciągłe - mediana [IQR]
    print("\n--- ZMIENNE CIĄGŁE ---")
    print("{:<25} {:>8} {:>25} {:>8} {:>25} {:>12} {:>10}".format(
        "Parametr", "n_hosp", "Hosp (mediana [IQR])", "n_dom", "Dom (mediana [IQR])", "p-value", "Cliff's d"
    ))
    print("-"*120)
    
    for param in parametry_ciagle:
        if param in df_hosp.columns:
            hosp = df_hosp[param].dropna()
            dom = df_dom[param].dropna()
            
            if len(hosp) > 0 and len(dom) > 0:
                hosp_med = hosp.median()
                hosp_q1 = hosp.quantile(0.25)
                hosp_q3 = hosp.quantile(0.75)
                dom_med = dom.median()
                dom_q1 = dom.quantile(0.25)
                dom_q3 = dom.quantile(0.75)
                
                hosp_stat = f"{hosp_med:.2f} [{hosp_q1:.2f}-{hosp_q3:.2f}]"
                dom_stat = f"{dom_med:.2f} [{dom_q1:.2f}-{dom_q3:.2f}]"
                
                # Test Manna-Whitneya
                stat, p = stats.mannwhitneyu(hosp, dom)
                
                # Cliff's delta
                d = cliff_delta_poprawiony(hosp, dom)
                
                wyniki.append({
                    'typ': 'ciągły',
                    'parametr': param,
                    'hosp_n': len(hosp),
                    'hosp_stat': hosp_stat,
                    'dom_n': len(dom),
                    'dom_stat': dom_stat,
                    'p_value': p,
                    'effect_size': d
                })
                
                print("{:<25} {:>3}   {:25} {:>3}   {:25}   {:<8.4f}   {:>6.2f}".format(
                    param[:24],
                    len(hosp),
                    hosp_stat,
                    len(dom),
                    dom_stat,
                    p,
                    d
                ))
    
    # Zmienne kategorialne - n (%)
    print("\n--- ZMIENNE KATEGORIALNE ---")
    print("{:<25} {:>20} {:>20} {:>12} {:>10}".format(
        "Choroba", "Hospitalizowani", "Do domu", "p-value", "OR"
    ))
    print("-"*90)
    
    for choroba in choroby:
        if choroba in df_hosp.columns:
            hosp = df_hosp[choroba].dropna()
            dom = df_dom[choroba].dropna()
            
            if len(hosp) > 0 and len(dom) > 0:
                hosp_tak = (hosp == 1).sum()
                hosp_n = len(hosp)
                dom_tak = (dom == 1).sum()
                dom_n = len(dom)
                
                hosp_proc = (hosp_tak / hosp_n) * 100
                dom_proc = (dom_tak / dom_n) * 100
                
                # Poprawka 0.5 dla OR
                a = hosp_tak + 0.5
                b = hosp_n - hosp_tak + 0.5
                c = dom_tak + 0.5
                d = dom_n - dom_tak + 0.5
                
                oddsratio = (a * d) / (b * c)
                
                # Test Fishera
                tabela = [[hosp_tak, hosp_n - hosp_tak], [dom_tak, dom_n - dom_tak]]
                _, p = fisher_exact(tabela)
                
                hosp_stat = f"{hosp_tak}/{hosp_n} ({hosp_proc:.1f}%)"
                dom_stat = f"{dom_tak}/{dom_n} ({dom_proc:.1f}%)"
                
                wyniki.append({
                    'typ': 'kategorialny',
                    'parametr': choroba,
                    'hosp_n': hosp_n,
                    'hosp_stat': hosp_stat,
                    'dom_n': dom_n,
                    'dom_stat': dom_stat,
                    'p_value': p,
                    'effect_size': oddsratio
                })
                
                print("{:<25} {:>20} {:>20}   {:<8.4f}   {:>6.2f}".format(
                    choroba,
                    hosp_stat,
                    dom_stat,
                    p,
                    oddsratio
                ))
    
    return pd.DataFrame(wyniki)


# =============================================================================
# POPRAWIONY BOOTSTRAP
# =============================================================================

def bootstrap_prog(df_caly, param, n_bootstrap=1000, kierunek='wyższe'):
    """
    Bootstrap dla progów klinicznych z prawidłowym random_state
    """
    dane = df_caly[[param, 'outcome']].dropna()
    
    if len(dane) < 10:
        return None
    
    progi_boot = []
    
    for i in range(n_bootstrap):
        # KAŻDA iteracja ma inny random_state!
        boot_sample = resample(dane, replace=True, random_state=i)
        
        try:
            fpr, tpr, progi = roc_curve(boot_sample['outcome'], boot_sample[param])
            
            # Jeśli kierunek jest odwrotny (niższe = gorsze), odwróć predyktor
            if kierunek == 'niższe':
                fpr, tpr, progi = roc_curve(boot_sample['outcome'], -boot_sample[param])
            
            youden = tpr - fpr
            if len(youden) > 0:
                opt_idx = np.argmax(youden)
                if kierunek == 'niższe':
                    progi_boot.append(-progi[opt_idx])
                else:
                    progi_boot.append(progi[opt_idx])
        except:
            continue
    
    if len(progi_boot) < n_bootstrap * 0.5:
        return None
    
    return {
        'mediana': np.median(progi_boot),
        'ci_2.5': np.percentile(progi_boot, 2.5),
        'ci_97.5': np.percentile(progi_boot, 97.5),
        'wszystkie': progi_boot
    }


def progi_kliniczne_z_bootstrapem_top5(df_caly, top_param, n_bootstrap=1000):
    """
    Progi dla top 5 parametrów z bootstrapem
    """
    print("\n" + "="*80)
    print("PROGI KLINICZNE Z BOOTSTRAPEM")
    print("="*80)
    
    wyniki = []
    
    for param in top_param:
        if param not in df_caly.columns:
            continue
        
        # Określ kierunek na podstawie median w grupach
        hosp_med = df_caly[df_caly['outcome'] == 1][param].median()
        dom_med = df_caly[df_caly['outcome'] == 0][param].median()
        
        kierunek = 'wyższe' if hosp_med > dom_med else 'niższe'
        
        print(f"\n{param} (kierunek: {kierunek})")
        
        # Bootstrap
        prog = bootstrap_prog(df_caly, param, n_bootstrap, kierunek)
        
        if prog is None:
            print("  ⚠️ Nie udało się oszacować progu")
            continue
        
        # Ocena na oryginalnych danych
        dane = df_caly[[param, 'outcome']].dropna()
        
        if kierunek == 'wyższe':
            y_pred = (dane[param] >= prog['mediana']).astype(int)
        else:
            y_pred = (dane[param] <= prog['mediana']).astype(int)
        
        y_true = dane['outcome']
        
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        
        czulosc = tp / (tp + fn) if (tp + fn) > 0 else 0
        swoistosc = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"  Próg: {prog['mediana']:.2f} (95% CI: {prog['ci_2.5']:.2f}-{prog['ci_97.5']:.2f})")
        print(f"  Czułość: {czulosc:.3f}")
        print(f"  Swoistość: {swoistosc:.3f}")
        
        wyniki.append({
            'parametr': param,
            'kierunek': kierunek,
            'prog': prog['mediana'],
            'prog_ci_2.5': prog['ci_2.5'],
            'prog_ci_97.5': prog['ci_97.5'],
            'czulosc': czulosc,
            'swoistosc': swoistosc
        })
    
    return pd.DataFrame(wyniki)


# =============================================================================
# ANALIZA WIELOCZYNNIKOWA - POPRAWIONA
# =============================================================================

def analiza_wieloczynnikowa_kliniczna(df_caly, parametry_kliniczne, wiek_zawsze=True):
    """
    Regresja logistyczna z pre-definiowanymi zmiennymi klinicznymi
    """
    print("\n" + "="*80)
    print("ANALIZA WIELOCZYNNIKOWA - REGRESJA LOGISTYCZNA")
    print("="*80)
    
    # Wybór zmiennych: sens kliniczny + EPV
    print("\n1. WYBÓR ZMIENNYCH (sens kliniczny)")
    
    # Lista zmiennych do rozważenia
    potencjalne = ['wiek', 'SpO2', 'crp(0-0,5)', 'kreatynina(0,5-1,2)']
    
    # Dodaj jeśli dostępne
    dostepne = [p for p in potencjalne if p in df_caly.columns]
    
    # Sprawdź EPV
    n_events = df_caly['outcome'].sum()
    max_pred = int(n_events / 10)
    
    if max_pred < len(dostepne):
        print(f"  ⚠️ EPV ogranicza liczbę predyktorów")
        print(f"  Wybieram {max_pred} z {len(dostepne)}")
        dostepne = dostepne[:max_pred]
    
    print(f"  Zmienne w modelu: {', '.join(dostepne)}")
    
    if len(dostepne) == 0:
        print("  Brak predyktorów!")
        return None, None, None
    
    # Przygotowanie danych
    df_model = df_caly[dostepne + ['outcome']].dropna()
    print(f"\n2. DANE DO MODELU: {len(df_model)} obserwacji")
    
    # VIF
    print("\n3. OCENA WSPÓŁLINIOWOŚCI (VIF)")
    X_vif = df_model[dostepne]
    X_vif = sm.add_constant(X_vif)
    
    vif_data = pd.DataFrame()
    vif_data['zmienna'] = X_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    
    for _, row in vif_data.iterrows():
        if row['zmienna'] != 'const':
            if row['VIF'] > 10:
                print(f"  ⚠️ {row['zmienna']}: VIF={row['VIF']:.2f} (wysoka współliniowość)")
            elif row['VIF'] > 5:
                print(f"  • {row['zmienna']}: VIF={row['VIF']:.2f} (umiarkowana współliniowość)")
            else:
                print(f"  ✓ {row['zmienna']}: VIF={row['VIF']:.2f}")
    
    # Regresja logistyczna z obsługą błędów
    print("\n4. MODEL REGRESJI LOGISTYCZNEJ")
    
    X = df_model[dostepne]
    X = sm.add_constant(X)
    y = df_model['outcome']
    
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100, method='bfgs')
        print(model.summary().tables[1])
    except Exception as e:
        print(f"  ⚠️ Błąd estymacji: {e}")
        print("  Próbuję z prostszym modelem...")
        
        # Próba z mniejszą liczbą zmiennych
        dostepne = dostepne[:min(2, len(dostepne))]
        X = df_model[dostepne]
        X = sm.add_constant(X)
        
        try:
            model = sm.Logit(y, X).fit(disp=0, maxiter=100)
            print(model.summary().tables[1])
        except:
            print("  ✗ Nie udało się oszacować modelu")
            return None, None, None
    
    # Wyniki
    wyniki = []
    for i, param in enumerate(['const'] + dostepne):
        if param in model.params.index:
            or_val = np.exp(model.params[param])
            ci_low, ci_high = np.exp(model.conf_int().loc[param])
            p_val = model.pvalues[param]
            
            wyniki.append({
                'parametr': param,
                'OR': or_val,
                'CI_95%': f"{ci_low:.2f}-{ci_high:.2f}",
                'p_value': p_val
            })
            
            if param != 'const':
                gwiazdki = ""
                if p_val < 0.001:
                    gwiazdki = "***"
                elif p_val < 0.01:
                    gwiazdki = "**"
                elif p_val < 0.05:
                    gwiazdki = "*"
                
                print(f"  {param}: OR={or_val:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f}), p={p_val:.4f} {gwiazdki}")
    
    return model, pd.DataFrame(wyniki), df_model


# =============================================================================
# OCENA PREDYKCYJNA Z BŁĘDAMI
# =============================================================================

def ocena_predykcyjna_z_ci(df_model, dostepne, n_bootstrap=1000):
    """
    Ocena modelu z bootstrapem dla AUC
    """
    print("\n" + "="*80)
    print("OCENA PREDYKCYJNA Z BOOTSTRAPEM")
    print("="*80)
    
    X = df_model[dostepne]
    y = df_model['outcome']
    
    # Podział na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n1. PODZIAŁ DANYCH:")
    print(f"  • Treningowy: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  • Testowy: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Trenuj model
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    
    # Predykcje na zbiorze testowym
    y_pred_prob = logreg.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Bootstrap dla AUC
    auc_boot = []
    for i in range(n_bootstrap):
        boot_idx = resample(range(len(y_test)), replace=True, random_state=i)
        if len(np.unique(y_test.iloc[boot_idx])) < 2:
            continue
        auc_boot.append(roc_auc_score(y_test.iloc[boot_idx], y_pred_prob[boot_idx]))
    
    auc_ci_low = np.percentile(auc_boot, 2.5)
    auc_ci_high = np.percentile(auc_boot, 97.5)
    
    print(f"\n2. METRYKI NA ZBIORZE TESTOWYM:")
    print(f"  • AUC: {roc_auc:.3f} (95% CI: {auc_ci_low:.3f}-{auc_ci_high:.3f})")
    
    # Brier score
    brier = brier_score_loss(y_test, y_pred_prob)
    print(f"  • Brier score: {brier:.4f} (0 = idealny)")
    
    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    czulosc = tp / (tp + fn) if (tp + fn) > 0 else 0
    swoistosc = tn / (tn + fp) if (tn + fp) > 0 else 0
    dokladnosc = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"  • Czułość: {czulosc:.3f}")
    print(f"  • Swoistość: {swoistosc:.3f}")
    print(f"  • Dokładność: {dokladnosc:.3f}")
    print(f"  • PPV: {ppv:.3f}")
    print(f"  • NPV: {npv:.3f}")
    
    # Walidacja krzyżowa
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(logreg, X_train, y_train, cv=cv, scoring='roc_auc')
    
    print(f"\n3. WALIDACJA KRZYŻOWA (5-fold CV):")
    print(f"  • AUC (CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Wykresy
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC (AUC = {roc_auc:.2f}, 95% CI: {auc_ci_low:.2f}-{auc_ci_high:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Krzywa ROC z bootstrapem')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
    ax2.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Idealna kalibracja')
    ax2.set_xlabel('Średnie prawdopodobieństwo przewidywane')
    ax2.set_ylabel('Zaobserwowana częstość')
    ax2.set_title(f'Krzywa kalibracji (Brier = {brier:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    return {
        'auc': roc_auc,
        'auc_ci_low': auc_ci_low,
        'auc_ci_high': auc_ci_high,
        'brier': brier,
        'czulosc': czulosc,
        'swoistosc': swoistosc,
        'dokladnosc': dokladnosc,
        'ppv': ppv,
        'npv': npv,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'fig_roc': fig1,
        'fig_cal': fig2
    }


# =============================================================================
# WYKRESY
# =============================================================================

def wykres_pudelkowy_z_kierunkiem(df_hosp, df_dom, param, nazwa_pliku):
    """
    Wykres pudełkowy z zaznaczeniem kierunku
    """
    hosp = df_hosp[param].dropna()
    dom = df_dom[param].dropna()
    
    if len(hosp) == 0 or len(dom) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Test statystyczny
    stat, p = stats.mannwhitneyu(hosp, dom)
    d = cliff_delta_poprawiony(hosp, dom)
    
    # Kierunek
    hosp_med = hosp.median()
    dom_med = dom.median()
    kierunek = "↑ wyższe u hosp" if hosp_med > dom_med else "↓ niższe u hosp"
    
    # Wykres 1
    bp1 = ax1.boxplot([hosp, dom],
                     labels=['PRZYJĘCI', 'WYPISANI'],
                     patch_artist=True,
                     medianprops={'color': 'black', 'linewidth': 2})
    
    bp1['boxes'][0].set_facecolor(KOLORY['hosp'])
    bp1['boxes'][0].set_alpha(0.8)
    bp1['boxes'][1].set_facecolor(KOLORY['dom'])
    bp1['boxes'][1].set_alpha(0.8)
    
    np.random.seed(42)
    x_hosp = np.random.normal(1, 0.05, len(hosp))
    x_dom = np.random.normal(2, 0.05, len(dom))
    ax1.scatter(x_hosp, hosp, alpha=0.3, color='darkred', s=20)
    ax1.scatter(x_dom, dom, alpha=0.3, color='darkblue', s=20)
    
    ax1.set_title(f'{param}')
    ax1.set_ylabel(param)
    ax1.grid(True, alpha=0.3)
    
    # Wykres 2 - statystyki
    ax2.axis('off')
    text = f"""
    {param}
    
    Hospitalizowani:
    n = {len(hosp)}
    mediana = {hosp_med:.2f}
    IQR = {hosp.quantile(0.25):.2f}-{hosp.quantile(0.75):.2f}
    
    Wypisani:
    n = {len(dom)}
    mediana = {dom_med:.2f}
    IQR = {dom.quantile(0.25):.2f}-{dom.quantile(0.75):.2f}
    
    p = {p:.4f}
    Cliff's d = {d:.2f}
    {kierunek}
    """
    ax2.text(0.1, 0.5, text, fontsize=12, transform=ax2.transAxes,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(nazwa_pliku, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Wykres: {nazwa_pliku}")


# =============================================================================
# FOREST PLOT
# =============================================================================

def forest_plot_ostateczny(wyniki_wielo, nazwa_pliku='forest_plot.png'):
    """
    Forest plot z OR i CI
    """
    if wyniki_wielo is None or len(wyniki_wielo) == 0:
        return
    
    df_plot = wyniki_wielo[wyniki_wielo['parametr'] != 'const'].copy()
    
    if len(df_plot) == 0:
        return
    
    # Parsuj CI
    df_plot[['ci_low', 'ci_high']] = df_plot['CI_95%'].str.split('-', expand=True).astype(float)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(df_plot))
    
    ax.errorbar(df_plot['OR'], y_pos, 
                xerr=[df_plot['OR'] - df_plot['ci_low'], df_plot['ci_high'] - df_plot['OR']],
                fmt='o', color='darkblue', ecolor='gray', capsize=5, markersize=8)
    
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='OR = 1')
    
    ax.set_xscale('log')
    ax.set_xlabel('OR (95% CI) - skala logarytmiczna')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['parametr'])
    ax.set_title('Niezależne czynniki ryzyka hospitalizacji')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Dodaj wartości OR
    for i, (_, row) in enumerate(df_plot.iterrows()):
        ax.text(row['OR'] * 1.1, i, f"{row['OR']:.2f}", 
                verticalalignment='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(nazwa_pliku, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Forest plot: {nazwa_pliku}")


# =============================================================================
# GŁÓWNA FUNKCJA
# =============================================================================

def main(sciezka_pliku, id_pacjenta=None):
    """
    Główna funkcja analizy
    """
    print("\n" + "="*80)
    print("PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # =========================================================================
    # ETAP 1: WCZYTYWANIE
    # =========================================================================
    
    df = wczytaj_dane(sciezka_pliku)
    if df is None:
        return
    
    df_hosp, df_dom, df_caly = przygotuj_dane_z_outcome(df, id_pacjenta)
    if df_hosp is None:
        return
    
    # Listy parametrów
    parametry_kliniczne = [
        'wiek', 'RR', 'MAP', 'SpO2', 'AS', 'mleczany',
        'kreatynina(0,5-1,2)', 'troponina I (0-7,8))',
        'HGB(12,4-15,2)', 'WBC(4-11)', 'plt(130-450)',
        'hct(38-45)', 'Na(137-145)', 'K(3,5-5,1)', 'crp(0-0,5)'
    ]
    
    choroby = ['dm', 'wątroba', 'naczyniowe', 'zza', 'npl']
    
    # Konwersja
    print("\n" + "="*70)
    print("KONWERSJA DANYCH")
    print("="*70)
    
    df_hosp = konwertuj_na_numeryczne(df_hosp, parametry_kliniczne)
    df_dom = konwertuj_na_numeryczne(df_dom, parametry_kliniczne)
    df_caly = konwertuj_na_numeryczne(df_caly, parametry_kliniczne)
    
    df_hosp = konwertuj_choroby(df_hosp, choroby)
    df_dom = konwertuj_choroby(df_dom, choroby)
    df_caly = konwertuj_choroby(df_caly, choroby)
    
    # =========================================================================
    # ETAP 2: KONTROLA JAKOŚCI
    # =========================================================================
    
    raport_brakow(df_caly)
    walidacja_zakresow_biologicznych(df_caly, ZAKRESY_BIOLOGICZNE)
    
    # =========================================================================
    # ETAP 3: TABELA 1
    # =========================================================================
    
    tabela = tabela_opisowa_kompletna(df_hosp, df_dom, parametry_kliniczne, choroby)
    tabela.to_csv('tabela_1_opisowa.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 4: ANALIZA JEDNOCZYNNIKOWA
    # =========================================================================
    
    print("\n" + "="*80)
    print("ANALIZA JEDNOCZYNNIKOWA - TOP 5")
    print("="*80)
    
    # Oblicz p-value dla wszystkich
    wyniki_jedno = []
    for param in parametry_kliniczne:
        if param in df_caly.columns:
            hosp = df_caly[df_caly['outcome'] == 1][param].dropna()
            dom = df_caly[df_caly['outcome'] == 0][param].dropna()
            if len(hosp) > 0 and len(dom) > 0:
                _, p = stats.mannwhitneyu(hosp, dom)
                wyniki_jedno.append((param, p))
    
    # Sortuj i weź top 5
    wyniki_jedno.sort(key=lambda x: x[1])
    top5_param = [x[0] for x in wyniki_jedno[:5]]
    
    print(f"Top 5 parametrów: {', '.join(top5_param)}")
    
    # =========================================================================
    # ETAP 5: PROGI KLINICZNE
    # =========================================================================
    
    progi = progi_kliniczne_z_bootstrapem_top5(df_caly, top5_param)
    if progi is not None and len(progi) > 0:
        progi.to_csv('progi_kliniczne.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 6: ANALIZA WIELOCZYNNIKOWA
    # =========================================================================
    
    model, wyniki_wielo, df_model = analiza_wieloczynnikowa_kliniczna(
        df_caly, parametry_kliniczne, wiek_zawsze=True
    )
    
    if model is not None:
        wyniki_wielo.to_csv('analiza_wieloczynnikowa.csv', sep=';', index=False)
        forest_plot_ostateczny(wyniki_wielo)
        
        # =====================================================================
        # ETAP 7: OCENA PREDYKCYJNA
        # =====================================================================
        
        dostepne = wyniki_wielo[wyniki_wielo['parametr'] != 'const']['parametr'].tolist()
        metryki = ocena_predykcyjna_z_ci(df_model, dostepne)
        
        metryki['fig_roc'].savefig('krzywa_ROC.png', dpi=300, bbox_inches='tight')
        metryki['fig_cal'].savefig('krzywa_kalibracji.png', dpi=300, bbox_inches='tight')
        plt.close('all')
    
    # =========================================================================
    # ETAP 8: WYKRESY
    # =========================================================================
    
    print("\n" + "="*80)
    print("GENEROWANIE WYKRESÓW")
    print("="*80)
    
    for i, param in enumerate(top5_param, 1):
        nazwa = f'wykres_{i}_{param}.png'.replace('(', '').replace(')', '').replace(' ', '_')
        wykres_pudelkowy_z_kierunkiem(df_hosp, df_dom, param, nazwa)
    
    # =========================================================================
    # PODSUMOWANIE
    # =========================================================================
    
    print("\n" + "="*80)
    print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE")
    print("="*80)
    
    print("\nWygenerowane pliki:")
    print("  • tabela_1_opisowa.csv")
    if progi is not None:
        print("  • progi_kliniczne.csv")
    if model is not None:
        print("  • analiza_wieloczynnikowa.csv")
        print("  • forest_plot.png")
        print("  • krzywa_ROC.png")
        print("  • krzywa_kalibracji.png")
    for i in range(1, 6):
        print(f"  • wykres_{i}_*.png")
    
    print("\n" + "="*80)


# =============================================================================
# URUCHOMIENIE
# =============================================================================
if __name__ == "__main__":
    # Podaj ścieżkę do pliku z kolumną 'outcome'
    # Opcjonalnie: podaj nazwę kolumny z ID pacjenta
    sciezka = 'BAZA_DANYCH_PACJENTOW_B.csv'
    main(sciezka, id_pacjenta=None)  # id_pacjenta='patient_id' jeśli masz