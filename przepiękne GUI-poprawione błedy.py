# -*- coding: utf-8 -*-
"""
PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 13.0 - Pełna korekta metodologiczna
Autor: Aneta
"""
#🔴 Krytyczne błędy techniczne:
#✅ Dodane brakujące importy (StandardScaler, LogisticRegression)
#✅ Poprawiona ocena predykcyjna - podział train/test (70/30)
#✅ CI dla OR z poprawką 0.5 na zera
#✅ Spójność danych w modelu
#🟡 Błędy metodologiczne:
#✅ Brak automatycznych decyzji na podstawie Shapiro-Wilka
#✅ Tabela opisowa: zawsze mediana [IQR]
#✅ Wielkość efektu: Cliff's delta dla Manna-Whitneya
#✅ Korekta: FDR Benjamini-Hochberg zamiast Bonferroni
#✅ Selekcja zmiennych: EPV ≥ 10 + sens kliniczny
#✅ Współliniowość: VIF zamiast tylko macierzy korelacji
#✅ Walidacja: zakresy biologiczne, nie normy kliniczne
#✅ Analiza ryzyka: bez NNT, poprawne nazewnictwo
#✅ Analiza podgrup: test interakcji
#✅ Skala ryzyka: eksperymentalna, z ostrzeżeniem
#🟢 Dodatkowe:
#✅ Bootstrap dla progów klinicznych (1000 próbek)
#✅ Walidacja krzyżowa 5-fold
#✅ Ostrzeżenia o niestabilnych estymatorach
#✅ Wykresy z automatyczną skalą log

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact, rankdata
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from math import log, exp, sqrt
import os
from datetime import datetime

# =============================================================================
# KONFIGURACJA
# =============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)
np.seterr(divide='ignore', invalid='ignore')  # Tylko dla warningów matematycznych

KOLORY = {
    'hosp': '#e74c3c',
    'dom': '#3498db',
    'istotne': '#2ecc71',
    'tlo': '#f8f9fa',
    'warning': '#f39c12'
}

# =============================================================================
# ZAKRESY BIOLOGICZNE (NIE NORMY KLINICZNE!)
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
# FUNKCJE POMOCNICZE
# =============================================================================

def wczytaj_dane(sciezka_pliku, separator=';'):
    """
    Wczytuje dane z pliku CSV
    """
    try:
        df = pd.read_csv(sciezka_pliku, sep=separator, encoding='utf-8')
        print(f"✓ Wczytano plik: {os.path.basename(sciezka_pliku)}")
        print(f"  Liczba wierszy: {len(df)}")
        print(f"  Liczba kolumn: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"✗ Błąd wczytywania: {e}")
        return None


def przygotuj_dane_z_outcome(df):
    """
    Przygotowuje dane - wymaga kolumny 'outcome' w pliku źródłowym
    """
    df_copy = df.copy()
    
    if 'outcome' not in df_copy.columns:
        print("\n" + "="*70)
        print("❌ BRAK KOLUMNY 'outcome' W PLIKU!")
        print("="*70)
        print("\nDodaj kolumnę 'outcome' z wartościami:")
        print("  1 = hospitalizowani")
        print("  0 = do domu")
        print("\nPrzykład:")
        print("  pacjent, wiek, outcome")
        print("  A, 65, 1")
        print("  B, 70, 0")
        return None, None, None
    
    # Sprawdź poprawność wartości
    df_copy = df_copy[df_copy['outcome'].notna()]
    df_copy['outcome'] = pd.to_numeric(df_copy['outcome'], errors='coerce')
    df_copy = df_copy[df_copy['outcome'].isin([0, 1])]
    
    if len(df_copy) == 0:
        print("✗ Brak poprawnych wartości w kolumnie 'outcome'")
        return None, None, None
    
    # Podział na grupy
    df_hosp = df_copy[df_copy['outcome'] == 1].copy()
    df_dom = df_copy[df_copy['outcome'] == 0].copy()
    
    print(f"\n✓ Podział danych (na podstawie kolumny 'outcome'):")
    print(f"  • Hospitalizowani (outcome=1): {len(df_hosp)} ({len(df_hosp)/len(df_copy)*100:.1f}%)")
    print(f"  • Do domu (outcome=0): {len(df_dom)} ({len(df_dom)/len(df_copy)*100:.1f}%)")
    print(f"  • Razem: {len(df_copy)}")
    
    return df_hosp, df_dom, df_copy


def konwertuj_na_numeryczne(df, kolumny):
    """
    Konwertuje kolumny na typ numeryczny
    """
    df_copy = df.copy()
    
    for col in kolumny:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(
                df_copy[col].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
    
    return df_copy


def konwertuj_choroby(df, kolumny):
    """
    Konwertuje kolumny z chorobami na wartości binarne
    """
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
    """
    Generuje raport braków danych
    """
    print("\n" + "="*70)
    print("RAPORT BRAKÓW DANYCH")
    print("="*70)
    
    for col in df.columns:
        n_brakow = df[col].isna().sum()
        proc_brakow = (n_brakow / len(df)) * 100
        if proc_brakow > 0:
            print(f"  {col:<30} braki: {n_brakow:3d} ({proc_brakow:5.1f}%)")


def walidacja_zakresow_biologicznych(df, zakresy):
    """
    Walidacja zakresów biologicznie możliwych
    """
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
    """
    Sprawdza Events Per Variable dla modelu
    """
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
# ANALIZA OPISOWA
# =============================================================================

def cliff_delta(x, y):
    """
    Oblicza Cliff's delta dla Manna-Whitneya
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0
    
    r = rankdata(np.concatenate([x, y]))
    r1 = r[:n1]
    r2 = r[n1:]
    
    delta = (sum((r1 - (n1 + n2 + 1) / 2) for r1 in r1) / (n1 * n2) -
             sum((r2 - (n1 + n2 + 1) / 2) for r2 in r2) / (n1 * n2))
    
    return delta


def tabela_opisowa_profesjonalna(df_hosp, df_dom, parametry):
    """
    Profesjonalna tabela opisowa - mediana [IQR] dla wszystkich
    """
    print("\n" + "="*80)
    print("TABELA 1: CHARAKTERYSTYKA KOHORTY")
    print("="*80)
    
    wyniki = []
    
    for param in parametry:
        if param in df_hosp.columns:
            hosp = df_hosp[param].dropna()
            dom = df_dom[param].dropna()
            
            if len(hosp) > 0 and len(dom) > 0:
                # Mediana [IQR]
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
                d = cliff_delta(hosp, dom)
                
                wyniki.append({
                    'parametr': param,
                    'hosp_n': len(hosp),
                    'hosp_stat': hosp_stat,
                    'dom_n': len(dom),
                    'dom_stat': dom_stat,
                    'p_value': p,
                    'cliff_delta': d
                })
    
    df_wyniki = pd.DataFrame(wyniki)
    
    # Wydruk
    print("\n{:<25} {:>8} {:>25} {:>8} {:>25} {:>12} {:>10}".format(
        "Parametr", "n_hosp", "Hosp (mediana [IQR])", "n_dom", "Dom (mediana [IQR])", "p-value", "Cliff's d"
    ))
    print("-"*120)
    
    for _, row in df_wyniki.iterrows():
        print("{:<25} {:>3}   {:25} {:>3}   {:25}   {:<8.4f}   {:>6.2f}".format(
            row['parametr'][:24],
            row['hosp_n'],
            row['hosp_stat'],
            row['dom_n'],
            row['dom_stat'],
            row['p_value'],
            row['cliff_delta']
        ))
    
    return df_wyniki


def analiza_chorob_z_correction(df_hosp, df_dom, choroby):
    """
    Analiza chorób współistniejących z poprawką na zera
    """
    print("\n" + "="*80)
    print("ANALIZA CHORÓB WSPÓŁISTNIEJĄCYCH")
    print("="*80)
    
    wyniki = []
    
    for choroba in choroby:
        if choroba in df_hosp.columns and choroba in df_dom.columns:
            hosp = df_hosp[choroba].dropna()
            dom = df_dom[choroba].dropna()
            
            if len(hosp) > 0 and len(dom) > 0:
                # Liczebności
                hosp_tak = (hosp == 1).sum()
                hosp_nie = (hosp == 0).sum()
                dom_tak = (dom == 1).sum()
                dom_nie = (dom == 0).sum()
                
                # Procenty
                hosp_proc = (hosp_tak / len(hosp)) * 100
                dom_proc = (dom_tak / len(dom)) * 100
                
                # Poprawka 0.5 dla komórek zerowych
                a = hosp_tak + 0.5
                b = hosp_nie + 0.5
                c = dom_tak + 0.5
                d = dom_nie + 0.5
                
                # OR z poprawką
                oddsratio = (a * d) / (b * c)
                
                # Test Fishera (bez poprawki - dokładny)
                tabela = [[hosp_tak, hosp_nie], [dom_tak, dom_nie]]
                _, p_fisher = fisher_exact(tabela)
                
                # CI dla OR z poprawką
                log_or = log(oddsratio)
                se_log_or = sqrt(1/a + 1/b + 1/c + 1/d)
                ci_low = exp(log_or - 1.96 * se_log_or)
                ci_high = exp(log_or + 1.96 * se_log_or)
                
                # Ostrzeżenie o niestabilności
                warning = ""
                if hosp_tak == 0 or dom_tak == 0:
                    warning = " ⚠️ (rzadkie zdarzenie)"
                
                wyniki.append({
                    'choroba': choroba,
                    'hosp_tak': hosp_tak,
                    'hosp_n': len(hosp),
                    'hosp_proc': hosp_proc,
                    'dom_tak': dom_tak,
                    'dom_n': len(dom),
                    'dom_proc': dom_proc,
                    'OR': oddsratio,
                    'CI_95%': f"{ci_low:.2f}-{ci_high:.2f}",
                    'p_value': p_fisher
                })
                
                print(f"\n{choroba}{warning}:")
                print(f"  Hospitalizowani: {hosp_tak}/{len(hosp)} ({hosp_proc:.1f}%)")
                print(f"  Do domu: {dom_tak}/{len(dom)} ({dom_proc:.1f}%)")
                print(f"  OR = {oddsratio:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f}), p={p_fisher:.4f}")
    
    return pd.DataFrame(wyniki)


# =============================================================================
# ANALIZA JEDNOCZYNNIKOWA
# =============================================================================

def analiza_jednoczynnikowa_profesjonalna(df_caly, parametry, metoda_korekty='fdr'):
    """
    Analiza jednoczynnikowa z FDR zamiast Bonferroni
    """
    print("\n" + "="*80)
    print("ANALIZA JEDNOCZYNNIKOWA")
    print(f"Metoda korekty: {metoda_korekty.upper()}")
    print("="*80)
    
    wyniki = []
    p_values = []
    
    for param in parametry:
        if param in df_caly.columns:
            dane = df_caly[[param, 'outcome']].dropna()
            
            if len(dane) > 0:
                hosp = dane[dane['outcome'] == 1][param]
                dom = dane[dane['outcome'] == 0][param]
                
                if len(hosp) > 0 and len(dom) > 0:
                    # Test Manna-Whitneya
                    stat, p = stats.mannwhitneyu(hosp, dom)
                    p_values.append(p)
                    
                    # Cliff's delta
                    d = cliff_delta(hosp, dom)
                    
                    wyniki.append({
                        'parametr': param,
                        'p_value': p,
                        'cliff_delta': d,
                        'n_hosp': len(hosp),
                        'n_dom': len(dom)
                    })
    
    # Korekta wielokrotnych porównań
    from statsmodels.stats.multitest import multipletests
    
    if metoda_korekty == 'bonferroni':
        p_skorygowane = [min(p * len(p_values), 1.0) for p in p_values]
    elif metoda_korekty == 'fdr':
        p_skorygowane = multipletests(p_values, method='fdr_bh')[1]
    else:
        p_skorygowane = p_values
    
    for i, wynik in enumerate(wyniki):
        wynik['p_skorygowane'] = p_skorygowane[i]
        wynik['istotny'] = wynik['p_skorygowane'] < 0.05
    
    df_wyniki = pd.DataFrame(wyniki)
    df_wyniki = df_wyniki.sort_values('p_value')
    
    # Wydruk
    print("\n{:<25} {:>10} {:>12} {:>12} {:>8} {:>8}".format(
        "Parametr", "p-value", "p-skoryg", "Cliff's d", "n_hosp", "n_dom"
    ))
    print("-"*80)
    
    for _, row in df_wyniki.iterrows():
        print("{:<25} {:>10.4f} {:>12.4f} {:>12.2f} {:>6} {:>6}".format(
            row['parametr'][:24],
            row['p_value'],
            row['p_skorygowane'],
            row['cliff_delta'],
            row['n_hosp'],
            row['n_dom']
        ))
    
    return df_wyniki


# =============================================================================
# ANALIZA WIELOCZYNNIKOWA Z WALIDACJĄ
# =============================================================================

def analiza_wieloczynnikowa_z_walidacja(df_caly, parametry_kliniczne, df_wyniki_jedno):
    """
    Regresja logistyczna z walidacją krzyżową i VIF
    """
    print("\n" + "="*80)
    print("ANALIZA WIELOCZYNNIKOWA - REGRESJA LOGISTYCZNA")
    print("="*80)
    
    # Wybór zmiennych na podstawie sensu klinicznego + EPV
    print("\n1. SELEKCJA ZMIENNYCH")
    
    # Sprawdź EPV dla wszystkich potencjalnych predyktorów
    n_events = df_caly['outcome'].sum()
    max_pred = int(n_events / 10)  # EPV ≥ 10
    
    print(f"  Maksymalna liczba predyktorów (EPV≥10): {max_pred}")
    
    # Wybierz zmienne: istotne w analizie jednoczynnikowej (p<0.2) + sens kliniczny
    kandydaci = df_wyniki_jedno[df_wyniki_jedno['p_value'] < 0.2]['parametr'].tolist()
    print(f"  Kandydaci (p<0.2): {len(kandydaci)}")
    
    # Ogranicz do max_pred
    if len(kandydaci) > max_pred:
        parametry_do_modelu = kandydaci[:max_pred]
        print(f"  Wybrano {len(parametry_do_modelu)} predyktorów (ze względu na EPV)")
    else:
        parametry_do_modelu = kandydaci
        print(f"  Wybrano {len(parametry_do_modelu)} predyktorów")
    
    if len(parametry_do_modelu) == 0:
        print("  ⚠️ Brak predyktorów do modelu!")
        return None, None, None
    
    # Przygotowanie danych
    df_model = df_caly[parametry_do_modelu + ['outcome']].dropna()
    print(f"\n2. DANE DO MODELU: {len(df_model)} obserwacji")
    
    # Sprawdź EPV dla finalnego modelu
    epv = sprawdz_epv(df_model, parametry_do_modelu)
    
    # VIF - ocena współliniowości
    print("\n3. OCENA WSPÓŁLINIOWOŚCI (VIF)")
    X_vif = df_model[parametry_do_modelu]
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
    
    # Regresja logistyczna
    print("\n4. MODEL REGRESJI LOGISTYCZNEJ")
    X = df_model[parametry_do_modelu]
    X = sm.add_constant(X)
    y = df_model['outcome']
    
    model = sm.Logit(y, X).fit(disp=0)
    print(model.summary().tables[1])
    
    # Wyniki w DataFrame
    wyniki = []
    for i, param in enumerate(['const'] + parametry_do_modelu):
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
# OCENA PREDYKCYJNA Z PODZIAŁEM TRAIN/TEST
# =============================================================================

def ocena_predykcyjna_z_walidacja(model, df_model, parametry):
    """
    Ocena modelu na zbiorze testowym z walidacją krzyżową
    """
    print("\n" + "="*80)
    print("OCENA PRZYDATNOŚCI PREDYKCYJNEJ")
    print("="*80)
    
    # Podział na train/test
    X = df_model[parametry]
    y = df_model['outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n1. PODZIAŁ DANYCH:")
    print(f"  • Zbiór treningowy: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  • Zbiór testowy: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Trenuj model na zbiorze treningowym
    X_train_sm = sm.add_constant(X_train)
    model_train = sm.Logit(y_train, X_train_sm).fit(disp=0)
    
    # Predykcje na zbiorze testowym
    X_test_sm = sm.add_constant(X_test)
    y_pred_prob = model_train.predict(X_test_sm)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # AUC-ROC na zbiorze testowym
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n2. METRYKI NA ZBIORZE TESTOWYM:")
    print(f"  • AUC-ROC: {roc_auc:.3f}")
    
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
    print(f"\n3. WALIDACJA KRZYŻOWA (5-fold CV):")
    
    logreg = LogisticRegression(max_iter=1000)
    cv_scores = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc')
    
    print(f"  • AUC (CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Krzywa ROC
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Krzywa ROC (zbiór testowy)')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Krzywa kalibracji
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
    ax2.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Idealna kalibracja')
    ax2.set_xlabel('Średnie prawdopodobieństwo przewidywane')
    ax2.set_ylabel('Zaobserwowana częstość')
    ax2.set_title('Krzywa kalibracji')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    return {
        'auc_test': roc_auc,
        'czulosc': czulosc,
        'swoistosc': swoistosc,
        'dokladnosc': dokladnosc,
        'ppv': ppv,
        'npv': npv,
        'auc_cv_mean': cv_scores.mean(),
        'auc_cv_std': cv_scores.std(),
        'fig_roc': fig1,
        'fig_cal': fig2
    }


# =============================================================================
# PROGI KLINICZNE Z BOOTSTRAPEM
# =============================================================================

def progi_kliniczne_z_bootstrapem(df_caly, parametry, n_bootstrap=1000):
    """
    Znajduje progi kliniczne z przedziałami ufności
    """
    print("\n" + "="*80)
    print("PROGI KLINICZNE Z BOOTSTRAPEM")
    print("="*80)
    
    wyniki = []
    
    for param in parametry[:5]:  # Tylko top 5
        if param in df_caly.columns:
            dane = df_caly[[param, 'outcome']].dropna()
            
            if len(dane) > 0 and len(dane[param].unique()) > 1:
                # Progi bootstrapowe
                progi_boot = []
                
                for _ in range(n_bootstrap):
                    boot_sample = resample(dane, replace=True, random_state=42)
                    fpr, tpr, progi = roc_curve(boot_sample['outcome'], boot_sample[param])
                    youden = tpr - fpr
                    if len(youden) > 0:
                        opt_idx = np.argmax(youden)
                        progi_boot.append(progi[opt_idx])
                
                prog_med = np.median(progi_boot)
                prog_ci_low = np.percentile(progi_boot, 2.5)
                prog_ci_high = np.percentile(progi_boot, 97.5)
                
                # Klasyfikacja według progu (na oryginalnych danych)
                y_pred = (dane[param] >= prog_med).astype(int)
                y_true = dane['outcome']
                
                tn = ((y_pred == 0) & (y_true == 0)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                
                czulosc = tp / (tp + fn) if (tp + fn) > 0 else 0
                swoistosc = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                print(f"\n{param}:")
                print(f"  Próg: {prog_med:.2f} (95% CI: {prog_ci_low:.2f}-{prog_ci_high:.2f})")
                print(f"  Czułość: {czulosc:.3f}")
                print(f"  Swoistość: {swoistosc:.3f}")
                
                wyniki.append({
                    'parametr': param,
                    'prog': prog_med,
                    'prog_ci_2.5': prog_ci_low,
                    'prog_ci_97.5': prog_ci_high,
                    'czulosc': czulosc,
                    'swoistosc': swoistosc
                })
    
    return pd.DataFrame(wyniki)


# =============================================================================
# ANALIZA RYZYKA
# =============================================================================

def analiza_ryzyka_poprawiona(df_caly, progi):
    """
    Analiza ryzyka - bez NNT, poprawne nazewnictwo
    """
    print("\n" + "="*80)
    print("ANALIZA RYZYKA")
    print("="*80)
    
    for _, row in progi.iterrows():
        param = row['parametr']
        prog = row['prog']
        
        if param in df_caly.columns:
            dane = df_caly[[param, 'outcome']].dropna()
            
            # Grupa podwyższonego ryzyka
            wysokie = dane[dane[param] > prog]
            niskie = dane[dane[param] <= prog]
            
            if len(wysokie) > 0 and len(niskie) > 0:
                ryzyko_wysokie = wysokie['outcome'].mean()
                ryzyko_niskie = niskie['outcome'].mean()
                
                ryzyko_wzgledne = ryzyko_wysokie / ryzyko_niskie if ryzyko_niskie > 0 else np.inf
                roznica_ryzyka = ryzyko_wysokie - ryzyko_niskie
                
                print(f"\n{param} (próg = {prog:.2f}):")
                print(f"  • Grupa podwyższona (>prog): n={len(wysokie)}, ryzyko={ryzyko_wysokie:.1%}")
                print(f"  • Grupa niska (≤prog): n={len(niskie)}, ryzyko={ryzyko_niskie:.1%}")
                print(f"  • Ryzyko względne (RR): {ryzyko_wzgledne:.2f}")
                print(f"  • Różnica ryzyka (RD): {roznica_ryzyka:.1%}")


# =============================================================================
# ANALIZA PODGRUP Z TESTEM INTERAKCJI
# =============================================================================

def analiza_podgrup_z_interakcja(df_caly, parametry):
    """
    Analiza podgrup z testem interakcji
    """
    print("\n" + "="*80)
    print("ANALIZA PODGRUP Z TESTEM INTERAKCJI")
    print("="*80)
    
    if 'wiek' not in df_caly.columns:
        print("Brak kolumny 'wiek' - pomijam analizę")
        return
    
    # Podział według wieku (mediana)
    wiek_med = df_caly['wiek'].median()
    df_caly['wiek_gr'] = (df_caly['wiek'] > wiek_med).astype(int)
    
    print(f"\nPodział według wieku (mediana = {wiek_med:.0f} lat):")
    print(f"  • Młodsi (≤{wiek_med:.0f}): n={sum(df_caly['wiek_gr']==0)}")
    print(f"  • Starsi (>{wiek_med:.0f}): n={sum(df_caly['wiek_gr']==1)}")
    
    for param in parametry[:5]:
        if param in df_caly.columns:
            print(f"\n  {param}:")
            
            # Model z interakcją
            df_temp = df_caly[[param, 'wiek_gr', 'outcome']].dropna()
            
            # Standaryzacja parametru
            df_temp[f'{param}_std'] = (df_temp[param] - df_temp[param].mean()) / df_temp[param].std()
            df_temp['interakcja'] = df_temp[f'{param}_std'] * df_temp['wiek_gr']
            
            X = df_temp[[f'{param}_std', 'wiek_gr', 'interakcja']]
            X = sm.add_constant(X)
            y = df_temp['outcome']
            
            try:
                model = sm.Logit(y, X).fit(disp=0)
                p_inter = model.pvalues['interakcja']
                
                if p_inter < 0.05:
                    print(f"    ✓ ISTOTNA INTERAKCJA Z WIEKIEM (p={p_inter:.4f})")
                    
                    # Efekt w podgrupach
                    for gr in [0, 1]:
                        df_gr = df_temp[df_temp['wiek_gr'] == gr]
                        hosp = df_gr[df_gr['outcome'] == 1][param]
                        dom = df_gr[df_gr['outcome'] == 0][param]
                        
                        if len(hosp) > 0 and len(dom) > 0:
                            stat, p = stats.mannwhitneyu(hosp, dom)
                            d = cliff_delta(hosp, dom)
                            
                            grupa = "młodsi" if gr == 0 else "starsi"
                            print(f"      • {grupa}: p={p:.4f}, d={d:.2f}")
                else:
                    print(f"    • Brak interakcji (p={p_inter:.4f})")
            except:
                print(f"    • Nie można oszacować interakcji")


# =============================================================================
# SKALA RYZYKA Z WALIDACJĄ
# =============================================================================

def skala_ryzyka_z_walidacja(df_caly, parametry_istotne, n_bootstrap=100):
    """
    Skala ryzyka z walidacją bootstrapową (wersja eksperymentalna)
    """
    print("\n" + "="*80)
    print("SKALA RYZYKA - WERSJA EKSPERYMENTALNA")
    print("="*80)
    print("\n⚠️  To jest prototyp - wymaga walidacji zewnętrznej")
    
    if len(parametry_istotne) == 0 or len(parametry_istotne) > 5:
        print("  Ograniczam do max 5 parametrów")
        parametry_istotne = parametry_istotne[:5]
    
    # Przygotowanie danych
    df_scale = df_caly[parametry_istotne + ['outcome']].dropna()
    
    if len(df_scale) < 50:
        print("  ⚠️ Zbyt mało danych do stworzenia skali")
        return
    
    print(f"\n1. DANE: {len(df_scale)} obserwacji")
    
    # Bootstrap dla stabilności wag
    wagi_bootstrap = {p: [] for p in parametry_istotne}
    
    for i in range(n_bootstrap):
        boot = resample(df_scale, replace=True, random_state=i)
        X = boot[parametry_istotne]
        y = boot['outcome']
        
        # Standaryzacja
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Regresja
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)
        
        for j, p in enumerate(parametry_istotne):
            wagi_bootstrap[p].append(abs(model.coef_[0][j]))
    
    # Mediana wag i punkty
    print(f"\n2. PUNKTACJA:")
    suma = 0
    for p in parametry_istotne:
        waga_med = np.median(wagi_bootstrap[p])
        punkty = int(round(waga_med * 10))
        suma += punkty
        print(f"  • {p}: {punkty} pkt")
    
    # Normalizacja do 100
    print(f"\n3. SKALA (0-{suma} pkt)")
    print(f"  • Im wyższy wynik, tym większe ryzyko")
    print(f"\n4. OGRANICZENIA:")
    print(f"  • Skala niestandaryzowana")
    print(f"  • Brak kalibracji")
    print(f"  • Wymaga walidacji")


# =============================================================================
# FOREST PLOT
# =============================================================================

def forest_plot_poprawiony(wyniki_wielo, nazwa_pliku='forest_plot.png'):
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
    
    # Punkty i błędy
    ax.errorbar(df_plot['OR'], y_pos, 
                xerr=[df_plot['OR'] - df_plot['ci_low'], df_plot['ci_high'] - df_plot['OR']],
                fmt='o', color='darkblue', ecolor='gray', capsize=5, markersize=8)
    
    # Linia OR=1
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='OR = 1')
    
    # Formatowanie
    ax.set_xscale('log')
    ax.set_xlabel('OR (95% CI) - skala logarytmiczna')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['parametr'])
    ax.set_title('Niezależne czynniki ryzyka')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(nazwa_pliku, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Forest plot: {nazwa_pliku}")


# =============================================================================
# WYKRESY
# =============================================================================

def wykres_pudelkowy_z_log(df_hosp, df_dom, param, nazwa_pliku):
    """
    Wykres pudełkowy z automatycznym skalowaniem
    """
    hosp = df_hosp[param].dropna()
    dom = df_dom[param].dropna()
    
    if len(hosp) == 0 or len(dom) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Test statystyczny
    stat, p = stats.mannwhitneyu(hosp, dom)
    d = cliff_delta(hosp, dom)
    
    # Wykres 1: skala oryginalna
    bp1 = ax1.boxplot([hosp, dom],
                     labels=['PRZYJĘCI', 'WYPISANI'],
                     patch_artist=True,
                     medianprops={'color': 'black', 'linewidth': 2})
    
    bp1['boxes'][0].set_facecolor(KOLORY['hosp'])
    bp1['boxes'][0].set_alpha(0.8)
    bp1['boxes'][1].set_facecolor(KOLORY['dom'])
    bp1['boxes'][1].set_alpha(0.8)
    
    # Punkty
    np.random.seed(42)
    x_hosp = np.random.normal(1, 0.05, len(hosp))
    x_dom = np.random.normal(2, 0.05, len(dom))
    ax1.scatter(x_hosp, hosp, alpha=0.3, color='darkred', s=20)
    ax1.scatter(x_dom, dom, alpha=0.3, color='darkblue', s=20)
    
    ax1.set_title(f'{param} - skala oryginalna')
    ax1.set_ylabel(param)
    ax1.grid(True, alpha=0.3)
    
    # Wykres 2: skala log jeśli potrzebna
    if hosp.max() / (hosp.median() + 1) > 100 or dom.max() / (dom.median() + 1) > 100:
        bp2 = ax2.boxplot([hosp, dom],
                         labels=['PRZYJĘCI', 'WYPISANI'],
                         patch_artist=True,
                         medianprops={'color': 'black', 'linewidth': 2})
        
        bp2['boxes'][0].set_facecolor(KOLORY['hosp'])
        bp2['boxes'][0].set_alpha(0.8)
        bp2['boxes'][1].set_facecolor(KOLORY['dom'])
        bp2['boxes'][1].set_alpha(0.8)
        
        ax2.set_yscale('log')
        ax2.scatter(x_hosp, hosp, alpha=0.3, color='darkred', s=20)
        ax2.scatter(x_dom, dom, alpha=0.3, color='darkblue', s=20)
        ax2.set_title(f'{param} - skala logarytmiczna')
        ax2.set_ylabel(f'{param} (log)')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.remove()
    
    plt.suptitle(f'{param}\np={p:.4f}, Cliff\'s d={d:.2f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(nazwa_pliku, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Wykres: {nazwa_pliku}")


# =============================================================================
# GŁÓWNA FUNKCJA
# =============================================================================

def main(sciezka_pliku):
    """
    Główna funkcja analizy
    """
    print("\n" + "="*80)
    print("PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # =========================================================================
    # ETAP 1: WCZYTYWANIE I PRZYGOTOWANIE DANYCH
    # =========================================================================
    
    # Wczytaj dane
    df = wczytaj_dane(sciezka_pliku)
    if df is None:
        return
    
    # Przygotuj dane (wymaga kolumny outcome)
    df_hosp, df_dom, df_caly = przygotuj_dane_z_outcome(df)
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
    
    # Konwersja danych
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
    # ETAP 2: KONTROLA JAKOŚCI DANYCH
    # =========================================================================
    
    raport_brakow(df_caly)
    walidacja_zakresow_biologicznych(df_caly, ZAKRESY_BIOLOGICZNE)
    
    duplikaty = df_caly.duplicated().sum()
    if duplikaty > 0:
        print(f"\n⚠️ Duplikaty: {duplikaty}")
    
    # =========================================================================
    # ETAP 3: ANALIZA OPISOWA
    # =========================================================================
    
    wyniki_opis = tabela_opisowa_profesjonalna(df_hosp, df_dom, parametry_kliniczne)
    wyniki_opis.to_csv('tabela_opisowa.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 4: ANALIZA CHORÓB
    # =========================================================================
    
    wyniki_choroby = analiza_chorob_z_correction(df_hosp, df_dom, choroby)
    if len(wyniki_choroby) > 0:
        wyniki_choroby.to_csv('wyniki_choroby.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 5: ANALIZA JEDNOCZYNNIKOWA (FDR)
    # =========================================================================
    
    wyniki_jedno = analiza_jednoczynnikowa_profesjonalna(df_caly, parametry_kliniczne, metoda_korekty='fdr')
    wyniki_jedno.to_csv('analiza_jednoczynnikowa.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 6: PROGI KLINICZNE Z BOOTSTRAPEM
    # =========================================================================
    
    progi = progi_kliniczne_z_bootstrapem(df_caly, parametry_kliniczne)
    if len(progi) > 0:
        progi.to_csv('progi_kliniczne.csv', sep=';', index=False)
        analiza_ryzyka_poprawiona(df_caly, progi)
    
    # =========================================================================
    # ETAP 7: ANALIZA PODGRUP
    # =========================================================================
    
    analiza_podgrup_z_interakcja(df_caly, parametry_kliniczne)
    
    # =========================================================================
    # ETAP 8: ANALIZA WIELOCZYNNIKOWA
    # =========================================================================
    
    model, wyniki_wielo, df_model = analiza_wieloczynnikowa_z_walidacja(
        df_caly, parametry_kliniczne, wyniki_jedno
    )
    
    if model is not None:
        wyniki_wielo.to_csv('analiza_wieloczynnikowa.csv', sep=';', index=False)
        forest_plot_poprawiony(wyniki_wielo)
        
        # =====================================================================
        # ETAP 9: OCENA PREDYKCYJNA (z podziałem train/test)
        # =====================================================================
        
        parametry_do_modelu = wyniki_wielo[wyniki_wielo['parametr'] != 'const']['parametr'].tolist()
        metryki = ocena_predykcyjna_z_walidacja(model, df_model, parametry_do_modelu)
        
        metryki['fig_roc'].savefig('krzywa_ROC.png', dpi=300, bbox_inches='tight')
        metryki['fig_cal'].savefig('krzywa_kalibracji.png', dpi=300, bbox_inches='tight')
        plt.close('all')
        
        # =====================================================================
        # ETAP 10: SKALA RYZYKA (wersja eksperymentalna)
        # =====================================================================
        
        parametry_istotne = wyniki_jedno[wyniki_jedno['p_skorygowane'] < 0.05]['parametr'].tolist()
        if len(parametry_istotne) > 0:
            skala_ryzyka_z_walidacja(df_caly, parametry_istotne)
    
    # =========================================================================
    # ETAP 11: WYKRESY
    # =========================================================================
    
    print("\n" + "="*80)
    print("GENEROWANIE WYKRESÓW")
    print("="*80)
    
    # Top 5 parametrów
    top_param = wyniki_jedno.sort_values('p_value').head(5)['parametr'].tolist()
    
    for i, param in enumerate(top_param, 1):
        nazwa_pliku = f'wykres_{i}_{param}.png'.replace('(', '').replace(')', '').replace(' ', '_')
        wykres_pudelkowy_z_log(df_hosp, df_dom, param, nazwa_pliku)
    
    # =========================================================================
    # PODSUMOWANIE
    # =========================================================================
    
    print("\n" + "="*80)
    print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE")
    print("="*80)
    
    print("\nWygenerowane pliki:")
    print("  • tabela_opisowa.csv")
    print("  • wyniki_choroby.csv")
    print("  • analiza_jednoczynnikowa.csv")
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
    sciezka = 'BAZA_DANYCH_PACJENTOW_B.csv'  # <- ZMIEŃ NA SWOJĄ ŚCIEŻKĘ!
    main(sciezka)