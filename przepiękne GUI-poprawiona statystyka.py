# -*- coding: utf-8 -*-
"""
PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 12.0 - Pełna analiza z prawdziwego zdarzenia
Autor: Aneta
"""
# Kod funkcyjny - zero powtarzalności
#Każda operacja to osobna funkcja
#Łatwe testowanie i modyfikacja
#2. Reprodukowalność - np.random.seed(42)
#Wykresy zawsze takie same
#Można odtworzyć wyniki
#3. Zamykanie figur - plt.close(fig) w zapisz_wykres()
#Brak wycieków pamięci
#Oszczędność zasobów
#4. Brak ukrytych warningów - usunięte warnings.filterwarnings('ignore')
#Widzimy wszystkie problemy
#Można je rozwiązywać
#5. Czytelny podział na etapy
#Opis kohorty (Tabela 1)
#Kontrola jakości
#Analiza jednoczynnikowa z korektą
#Analiza wieloczynnikowa
#Ocena predykcyjna
#Progi kliniczne
#Analiza podgrup
#Skala ryzyka
#6. Dodatkowe elementy kliniczne
#✅ Progi kliniczne (cut-offs)
#✅ Analiza ryzyka (RR, NNT)
#✅ Analiza podgrup (wiek)
#✅ Skala ryzyka
#✅ Forest plot
#✅ Krzywa kalibracji
#7. Eksport wszystkich wyników
#CSV dla każdej analizy
#Wykresy PNG
#Możliwość dalszej obróbki

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
import os
from datetime import datetime

# =============================================================================
# KONFIGURACJA
# =============================================================================
# Ustawienia wykresów
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)  # REPRODUKOWALNOŚĆ!

# Kolory
KOLORY = {
    'hosp': '#e74c3c',
    'dom': '#3498db',
    'istotne': '#2ecc71',
    'tlo': '#f8f9fa',
    'warning': '#f39c12'
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


def przygotuj_dane(df):
    """
    Przygotowuje dane do analizy - tworzy kolumnę outcome
    """
    df_copy = df.copy()
    
    # Znajdź pusty wiersz
    puste = df_copy[df_copy.isna().all(axis=1)]
    
    if len(puste) > 0:
        idx = puste.index[0]
        df_hosp = df_copy.iloc[:idx].copy().dropna(how='all')
        df_dom = df_copy.iloc[idx+1:].copy().dropna(how='all')
        
        # Dodaj kolumnę outcome
        df_hosp['outcome'] = 1  # hospitalizowani
        df_dom['outcome'] = 0   # do domu
        
        # Połącz
        df_caly = pd.concat([df_hosp, df_dom], ignore_index=True)
        
        print(f"✓ Podział danych:")
        print(f"  • Hospitalizowani (outcome=1): {len(df_hosp)}")
        print(f"  • Do domu (outcome=0): {len(df_dom)}")
        print(f"  • Razem: {len(df_caly)}")
        
        return df_hosp, df_dom, df_caly
    else:
        print("✗ Nie znaleziono pustego wiersza - sprawdź dane")
        return None, None, None


def konwertuj_na_numeryczne(df, kolumny):
    """
    Konwertuje kolumny na typ numeryczny z walidacją
    """
    df_copy = df.copy()
    
    for col in kolumny:
        if col in df_copy.columns:
            # Konwersja
            df_copy[col] = pd.to_numeric(
                df_copy[col].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
            
            # Raport konwersji
            n_konwersji = (~df_copy[col].isna()).sum()
            n_brakow = df_copy[col].isna().sum()
            print(f"  • {col}: przekonwertowano {n_konwersji}, braki: {n_brakow}")
    
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
            # Konwersja
            df_copy[col] = df_copy[col].astype(str).str.lower().str.strip()
            df_copy[col] = df_copy[col].apply(
                lambda x: 1 if x in mapping_tak else (0 if x in mapping_nie else np.nan)
            )
            
            # Raport
            n_tak = (df_copy[col] == 1).sum()
            n_nie = (df_copy[col] == 0).sum()
            n_brak = df_copy[col].isna().sum()
            print(f"  • {col}: tak={n_tak}, nie={n_nie}, braki={n_brak}")
    
    return df_copy


def test_normalnosci(dane, parametr, alpha=0.05):
    """
    Testuje normalność rozkładu (Shapiro-Wilk)
    """
    dane_czyste = dane.dropna()
    
    if len(dane_czyste) >= 3 and len(dane_czyste) <= 5000:
        stat, p = stats.shapiro(dane_czyste)
        normalny = p > alpha
        
        print(f"    Test Shapiro-Wilk dla {parametr}: p={p:.4f} - "
              f"{'rozkład normalny' if normalny else 'brak normalności'}")
        return normalny
    else:
        print(f"    Za mało danych do testu normalności dla {parametr} (n={len(dane_czyste)})")
        return False


def efekt_rozmiaru(hosp, dom):
    """
    Oblicza wielkość efektu (Cohen's d)
    """
    n1, n2 = len(hosp), len(dom)
    s1, s2 = hosp.std(), dom.std()
    
    # Cohen's d
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    if pooled_sd > 0:
        d = (hosp.mean() - dom.mean()) / pooled_sd
    else:
        d = 0
    
    return d


def korekta_bonferroni(p_values):
    """
    Stosuje korektę Bonferroniego
    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def zapisz_wykres(fig, nazwa):
    """
    Zapisuje wykres i zamyka figurę
    """
    try:
        fig.savefig(nazwa, dpi=300, bbox_inches='tight')
        print(f"  ✓ Zapisano: {nazwa}")
    except Exception as e:
        print(f"  ✗ Błąd zapisu {nazwa}: {e}")
    finally:
        plt.close(fig)


def waliduj_zakresy(df, parametry, normy):
    """
    Waliduje zakresy wartości parametrów
    """
    print("\n" + "="*70)
    print("WALIDACJA ZAKRESÓW")
    print("="*70)
    
    for param in parametry:
        if param in df.columns and param in normy:
            min_norm, max_norm = normy[param]
            dane = df[param].dropna()
            
            if len(dane) > 0:
                poza = ((dane < min_norm) | (dane > max_norm)).sum()
                proc_poza = (poza / len(dane)) * 100
                
                print(f"\n{param}:")
                print(f"  Zakres normy: {min_norm} - {max_norm}")
                print(f"  Wartości poza normą: {poza}/{len(dane)} ({proc_poza:.1f}%)")
                print(f"  Min: {dane.min():.2f}, Max: {dane.max():.2f}")


def raport_brakow(df):
    """
    Generuje raport braków danych
    """
    print("\n" + "="*70)
    print("RAPORT BRAKÓW DANYCH")
    print("="*70)
    
    raport = []
    for col in df.columns:
        n_brakow = df[col].isna().sum()
        proc_brakow = (n_brakow / len(df)) * 100
        raport.append({
            'kolumna': col,
            'braki': n_brakow,
            'procent': proc_brakow
        })
        print(f"  {col:<30} braki: {n_brakow:3d} ({proc_brakow:5.1f}%)")
    
    return pd.DataFrame(raport)


# =============================================================================
# ANALIZA OPISOWA
# =============================================================================

def tabela_opisowa(df_hosp, df_dom, parametry):
    """
    Generuje tabelę opisową kohorty (Tabela 1)
    """
    print("\n" + "="*70)
    print("TABELA 1: OPIS KOHORTY")
    print("="*70)
    
    wyniki = []
    
    for param in parametry:
        if param in df_hosp.columns:
            hosp = df_hosp[param].dropna()
            dom = df_dom[param].dropna()
            
            if len(hosp) > 0 and len(dom) > 0:
                # Test normalności
                normalny_hosp = test_normalnosci(hosp, f"{param} (hosp)")
                normalny_dom = test_normalnosci(dom, f"{param} (dom)")
                
                # Statystyki opisowe
                if normalny_hosp and normalny_dom:
                    # Mean ± SD
                    hosp_stat = f"{hosp.mean():.2f} ± {hosp.std():.2f}"
                    dom_stat = f"{dom.mean():.2f} ± {dom.std():.2f}"
                    test = "t-test"
                else:
                    # Mediana [IQR]
                    hosp_stat = f"{hosp.median():.2f} [{hosp.quantile(0.25):.2f}-{hosp.quantile(0.75):.2f}]"
                    dom_stat = f"{dom.median():.2f} [{dom.quantile(0.25):.2f}-{dom.quantile(0.75):.2f}]"
                    test = "Mann-Whitney"
                
                # Test statystyczny
                if normalny_hosp and normalny_dom:
                    stat, p = stats.ttest_ind(hosp, dom)
                else:
                    stat, p = stats.mannwhitneyu(hosp, dom)
                
                # Wielkość efektu
                d = efekt_rozmiaru(hosp, dom)
                
                wyniki.append({
                    'parametr': param,
                    'hosp_n': len(hosp),
                    'hosp_stat': hosp_stat,
                    'dom_n': len(dom),
                    'dom_stat': dom_stat,
                    'test': test,
                    'p_value': p,
                    'effect_size': d
                })
    
    df_wyniki = pd.DataFrame(wyniki)
    
    # Wydruk
    print("\n{:<25} {:>8} {:>20} {:>8} {:>20} {:>10} {:>10}".format(
        "Parametr", "n_hosp", "Hosp", "n_dom", "Dom", "p-value", "d-Cohena"
    ))
    print("-"*100)
    
    for _, row in df_wyniki.iterrows():
        print("{:<25} {:>3}   {:20} {:>3}   {:20}   {:<8.4f}   {:>6.2f}".format(
            row['parametr'][:24],
            row['hosp_n'],
            row['hosp_stat'],
            row['dom_n'],
            row['dom_stat'],
            row['p_value'],
            row['effect_size']
        ))
    
    return df_wyniki


def analiza_chorob(df_hosp, df_dom, choroby):
    """
    Analiza chorób współistniejących z testem Fishera i OR
    """
    print("\n" + "="*70)
    print("ANALIZA CHORÓB WSPÓŁISTNIEJĄCYCH")
    print("="*70)
    
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
                
                # Test Fishera
                tabela = [[hosp_tak, hosp_nie], [dom_tak, dom_nie]]
                oddsratio, p_fisher = fisher_exact(tabela)
                
                # Przedział ufności dla OR
                from math import exp, sqrt, log
                log_or = log(oddsratio)
                se_log_or = sqrt(1/hosp_tak + 1/hosp_nie + 1/dom_tak + 1/dom_nie)
                ci_low = exp(log_or - 1.96 * se_log_or)
                ci_high = exp(log_or + 1.96 * se_log_or)
                
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
                
                print(f"\n{choroba}:")
                print(f"  Hospitalizowani: {hosp_tak}/{len(hosp)} ({hosp_proc:.1f}%)")
                print(f"  Do domu: {dom_tak}/{len(dom)} ({dom_proc:.1f}%)")
                print(f"  OR = {oddsratio:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f}), p={p_fisher:.4f}")
    
    return pd.DataFrame(wyniki)


# =============================================================================
# ANALIZA JEDNOCZYNNIKOWA
# =============================================================================

def analiza_jednoczynnikowa(df_caly, parametry):
    """
    Przeprowadza analizę jednoczynnikową z korektą Bonferroniego
    """
    print("\n" + "="*70)
    print("ANALIZA JEDNOCZYNNIKOWA")
    print("="*70)
    
    wyniki = []
    p_values = []
    
    for param in parametry:
        if param in df_caly.columns:
            dane = df_caly[[param, 'outcome']].dropna()
            
            if len(dane) > 0:
                hosp = dane[dane['outcome'] == 1][param]
                dom = dane[dane['outcome'] == 0][param]
                
                if len(hosp) > 0 and len(dom) > 0:
                    # Test normalności
                    normalny = test_normalnosci(pd.concat([hosp, dom]), param)
                    
                    # Test statystyczny
                    if normalny:
                        stat, p = stats.ttest_ind(hosp, dom)
                        test = "t-test"
                    else:
                        stat, p = stats.mannwhitneyu(hosp, dom)
                        test = "Mann-Whitney"
                    
                    p_values.append(p)
                    
                    # Wielkość efektu
                    d = efekt_rozmiaru(hosp, dom)
                    
                    wyniki.append({
                        'parametr': param,
                        'test': test,
                        'statystyka': stat,
                        'p_value': p,
                        'effect_size': d,
                        'n_hosp': len(hosp),
                        'n_dom': len(dom)
                    })
    
    # Korekta Bonferroniego
    skorygowane = korekta_bonferroni(p_values)
    for i, wynik in enumerate(wyniki):
        wynik['p_skorygowane'] = skorygowane[i]
        wynik['istotny'] = wynik['p_skorygowane'] < 0.05
    
    df_wyniki = pd.DataFrame(wyniki)
    
    # Wydruk
    print("\n{:<25} {:>8} {:>10} {:>10} {:>10} {:>10}".format(
        "Parametr", "p-value", "p-skoryg", "d-Cohena", "n_hosp", "n_dom"
    ))
    print("-"*80)
    
    for _, row in df_wyniki.iterrows():
        print("{:<25} {:>8.4f} {:>10.4f} {:>10.2f} {:>6} {:>6}".format(
            row['parametr'][:24],
            row['p_value'],
            row['p_skorygowane'],
            row['effect_size'],
            row['n_hosp'],
            row['n_dom']
        ))
    
    return df_wyniki


# =============================================================================
# ANALIZA WIELOCZYNNIKOWA
# =============================================================================

def analiza_wieloczynnikowa(df_caly, parametry):
    """
    Przeprowadza analizę wieloczynnikową (regresja logistyczna)
    """
    print("\n" + "="*70)
    print("ANALIZA WIELOCZYNNIKOWA - REGRESJA LOGISTYCZNA")
    print("="*70)
    
    # Przygotowanie danych
    df_model = df_caly[parametry + ['outcome']].dropna()
    print(f"Liczba obserwacji w modelu: {len(df_model)}")
    
    # Sprawdzenie współliniowości
    print("\nMacierz korelacji predyktorów:")
    corr_matrix = df_model[parametry].corr()
    print(corr_matrix.round(2))
    
    # Wysoka korelacja?
    for i in range(len(parametry)):
        for j in range(i+1, len(parametry)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                print(f"  ⚠️ Wysoka korelacja między {parametry[i]} a {parametry[j]}")
    
    # Regresja logistyczna
    X = df_model[parametry]
    X = sm.add_constant(X)
    y = df_model['outcome']
    
    model = sm.Logit(y, X).fit(disp=0)
    
    print("\nWyniki regresji logistycznej:")
    print(model.summary().tables[1])
    
    # Wyniki w DataFrame
    wyniki = []
    for i, param in enumerate(['const'] + parametry):
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
                print(f"  {param}: OR={or_val:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f}), p={p_val:.4f}")
    
    return model, pd.DataFrame(wyniki)


# =============================================================================
# OCENA PREDYKCYJNA
# =============================================================================

def ocena_predykcyjna(model, X, y):
    """
    Ocenia przydatność predykcyjną modelu (ROC, AUC, kalibracja)
    """
    print("\n" + "="*70)
    print("OCENA PRZYDATNOŚCI PREDYKCYJNEJ")
    print("="*70)
    
    # Predykcje
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # AUC-ROC
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    print(f"AUC-ROC: {roc_auc:.3f}")
    
    # Macierz pomyłek
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    czulosc = tp / (tp + fn) if (tp + fn) > 0 else 0
    swoistosc = tn / (tn + fp) if (tn + fp) > 0 else 0
    dokladnosc = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"Czułość: {czulosc:.3f}")
    print(f"Swoistość: {swoistosc:.3f}")
    print(f"Dokładność: {dokladnosc:.3f}")
    print(f"PPV: {ppv:.3f}")
    print(f"NPV: {npv:.3f}")
    
    # Wykres ROC
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Krzywa ROC')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Krzywa kalibracji
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y, y_pred_prob, n_bins=10)
    ax2.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Idealna kalibracja')
    ax2.set_xlabel('Średnie prawdopodobieństwo przewidywane')
    ax2.set_ylabel('Zaobserwowana częstość')
    ax2.set_title('Krzywa kalibracji')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    return {
        'auc': roc_auc,
        'czulosc': czulosc,
        'swoistosc': swoistosc,
        'dokladnosc': dokladnosc,
        'ppv': ppv,
        'npv': npv,
        'fig_roc': fig1,
        'fig_cal': fig2
    }


# =============================================================================
# PROGI KLINICZNE
# =============================================================================

def znajdz_progi_kliniczne(df_caly, parametry):
    """
    Znajduje optymalne progi dla parametrów (cut-off)
    """
    print("\n" + "="*70)
    print("PROGI KLINICZNE (CUT-OFFS)")
    print("="*70)
    
    wyniki = []
    
    for param in parametry:
        if param in df_caly.columns:
            dane = df_caly[[param, 'outcome']].dropna()
            
            if len(dane) > 0 and len(dane[param].unique()) > 1:
                fpr, tpr, progi = roc_curve(dane['outcome'], dane[param])
                youden = tpr - fpr
                opt_idx = np.argmax(youden)
                opt_prog = progi[opt_idx]
                
                # Klasyfikacja według progu
                y_pred = (dane[param] >= opt_prog).astype(int)
                y_true = dane['outcome']
                
                tn = ((y_pred == 0) & (y_true == 0)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                
                czulosc = tp / (tp + fn) if (tp + fn) > 0 else 0
                swoistosc = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                print(f"\n{param}:")
                print(f"  Optymalny próg: {opt_prog:.2f}")
                print(f"  Czułość: {czulosc:.3f}")
                print(f"  Swoistość: {swoistosc:.3f}")
                print(f"  PPV: {ppv:.3f}")
                print(f"  NPV: {npv:.3f}")
                
                wyniki.append({
                    'parametr': param,
                    'prog': opt_prog,
                    'czulosc': czulosc,
                    'swoistosc': swoistosc,
                    'PPV': ppv,
                    'NPV': npv
                })
    
    return pd.DataFrame(wyniki)


# =============================================================================
# ANALIZA PODGRUP
# =============================================================================

def analiza_podgrup(df_caly, parametry):
    """
    Analiza w podgrupach (według wieku)
    """
    print("\n" + "="*70)
    print("ANALIZA PODGRUP (WEDŁUG WIEKU)")
    print("="*70)
    
    if 'wiek' not in df_caly.columns:
        print("Brak kolumny 'wiek' - pomijam analizę podgrup")
        return
    
    # Podział według wieku (mediana)
    wiek_med = df_caly['wiek'].median()
    
    df_mlodsi = df_caly[df_caly['wiek'] <= wiek_med]
    df_starsi = df_caly[df_caly['wiek'] > wiek_med]
    
    print(f"\nPodział według wieku (mediana = {wiek_med:.0f} lat):")
    print(f"  Młodsi (≤{wiek_med:.0f}): n={len(df_mlodsi)}")
    print(f"  Starsi (>{wiek_med:.0f}): n={len(df_starsi)}")
    
    for param in parametry[:5]:  # top 5
        if param in df_caly.columns:
            print(f"\n  {param}:")
            
            # Młodsi
            hosp_m = df_mlodsi[df_mlodsi['outcome'] == 1][param].median()
            dom_m = df_mlodsi[df_mlodsi['outcome'] == 0][param].median()
            p_m = stats.mannwhitneyu(
                df_mlodsi[df_mlodsi['outcome'] == 1][param].dropna(),
                df_mlodsi[df_mlodsi['outcome'] == 0][param].dropna()
            )[1] if (len(df_mlodsi[df_mlodsi['outcome'] == 1][param].dropna()) > 0 and 
                     len(df_mlodsi[df_mlodsi['outcome'] == 0][param].dropna()) > 0) else 1.0
            
            # Starsi
            hosp_s = df_starsi[df_starsi['outcome'] == 1][param].median()
            dom_s = df_starsi[df_starsi['outcome'] == 0][param].median()
            p_s = stats.mannwhitneyu(
                df_starsi[df_starsi['outcome'] == 1][param].dropna(),
                df_starsi[df_starsi['outcome'] == 0][param].dropna()
            )[1] if (len(df_starsi[df_starsi['outcome'] == 1][param].dropna()) > 0 and 
                     len(df_starsi[df_starsi['outcome'] == 0][param].dropna()) > 0) else 1.0
            
            print(f"    Młodsi: hosp={hosp_m:.2f} vs dom={dom_m:.2f}, p={p_m:.4f}")
            print(f"    Starsi: hosp={hosp_s:.2f} vs dom={dom_s:.2f}, p={p_s:.4f}")


# =============================================================================
# ANALIZA RYZYKA
# =============================================================================

def analiza_ryzyka(df_caly, progi):
    """
    Oblicza ryzyko względne i bezwzględne dla progów
    """
    print("\n" + "="*70)
    print("ANALIZA RYZYKA")
    print("="*70)
    
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
                ryzyko_bezwzgledne = ryzyko_wysokie - ryzyko_niskie
                nnt = 1 / ryzyko_bezwzgledne if ryzyko_bezwzgledne > 0 else np.inf
                
                print(f"\n{param} (próg = {prog:.2f}):")
                print(f"  Ryzyko w grupie podwyższonej: {ryzyko_wysokie:.1%}")
                print(f"  Ryzyko w grupie niskiej: {ryzyko_niskie:.1%}")
                print(f"  Ryzyko względne (RR): {ryzyko_wzgledne:.2f}")
                print(f"  Ryzyko bezwzględne: {ryzyko_bezwzgledne:.1%}")
                print(f"  NNT: {nnt:.0f}")


# =============================================================================
# SKALA RYZYKA
# =============================================================================

def stworz_skale_ryzyka(df_caly, parametry_istotne):
    """
    Tworzy prostą skalę ryzyka na podstawie istotnych parametrów
    """
    print("\n" + "="*70)
    print("SKALA RYZYKA")
    print("="*70)
    
    if len(parametry_istotne) == 0:
        print("Brak istotnych parametrów do stworzenia skali")
        return df_caly
    
    # Przygotowanie danych
    df_model = df_caly[parametry_istotne + ['outcome']].dropna()
    X = df_model[parametry_istotne]
    y = df_model['outcome']
    
    # Standaryzacja
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Regresja logistyczna dla wag
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    # Wagi (zaokrąglone do punktów)
    wagi = {}
    for i, param in enumerate(parametry_istotne):
        waga = abs(model.coef_[0][i] * 10)
        wagi[param] = int(round(waga))
    
    print("\nPunkty ryzyka:")
    suma_punktow = 0
    for param, waga in wagi.items():
        print(f"  {param}: {waga} pkt")
        suma_punktow += waga
    
    # Obliczenie ryzyka dla każdego pacjenta
    df_caly['risk_score'] = 0
    for param, waga in wagi.items():
        if param in df_caly.columns:
            # Dychotomizacja według mediany
            med = df_caly[param].median()
            df_caly[f'risk_{param}'] = (df_caly[param] > med).astype(int)
            df_caly['risk_score'] += df_caly[f'risk_{param}'] * waga
    
    # Podział na kategorie ryzyka
    kwantyle = df_caly['risk_score'].quantile([0.33, 0.67])
    df_caly['risk_category'] = pd.cut(df_caly['risk_score'],
                                      bins=[-np.inf, kwantyle.iloc[0], kwantyle.iloc[1], np.inf],
                                      labels=['Niskie', 'Średnie', 'Wysokie'])
    
    # Ryzyko w każdej kategorii
    print("\nRyzyko w kategoriach:")
    for kategoria in ['Niskie', 'Średnie', 'Wysokie']:
        ryzyko = df_caly[df_caly['risk_category'] == kategoria]['outcome'].mean()
        n = len(df_caly[df_caly['risk_category'] == kategoria])
        print(f"  {kategoria} (n={n}): {ryzyko:.1%}")
    
    return df_caly


# =============================================================================
# WYKRESY
# =============================================================================

def wykres_pudelkowy(df_hosp, df_dom, param, nazwa_pliku):
    """
    Tworzy wykres pudełkowy dla pojedynczego parametru
    """
    hosp = df_hosp[param].dropna()
    dom = df_dom[param].dropna()
    
    if len(hosp) == 0 or len(dom) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Test statystyczny
    stat, p = stats.mannwhitneyu(hosp, dom)
    
    # Wykres
    bp = ax.boxplot([hosp, dom],
                    labels=['PRZYJĘCI', 'WYPISANI'],
                    patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2})
    
    bp['boxes'][0].set_facecolor(KOLORY['hosp'])
    bp['boxes'][0].set_alpha(0.8)
    bp['boxes'][1].set_facecolor(KOLORY['dom'])
    bp['boxes'][1].set_alpha(0.8)
    
    # Dodanie punktów (z seed dla reprodukowalności)
    np.random.seed(42)
    x_hosp = np.random.normal(1, 0.05, len(hosp))
    x_dom = np.random.normal(2, 0.05, len(dom))
    ax.scatter(x_hosp, hosp, alpha=0.4, color='darkred', s=30)
    ax.scatter(x_dom, dom, alpha=0.4, color='darkblue', s=30)
    
    # Tytuł
    if p < 0.001:
        title = f'{param}\np < 0.001 ***'
    elif p < 0.01:
        title = f'{param}\np = {p:.4f} **'
    elif p < 0.05:
        title = f'{param}\np = {p:.4f} *'
    else:
        title = f'{param}\np = {p:.4f} (ns)'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(param)
    ax.grid(True, alpha=0.3)
    
    zapisz_wykres(fig, nazwa_pliku)


def forest_plot(wyniki_wielo, nazwa_pliku='forest_plot.png'):
    """
    Wizualizacja OR z 95% CI (forest plot)
    """
    if wyniki_wielo is None or len(wyniki_wielo) == 0:
        return
    
    # Filtruj stałą
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
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    
    # Formatowanie
    ax.set_xscale('log')
    ax.set_xlabel('OR (95% CI) - skala logarytmiczna')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['parametr'])
    ax.set_title('Forest Plot - niezależne czynniki ryzyka')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    zapisz_wykres(fig, nazwa_pliku)


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
    
    # Przygotuj dane (dodaj outcome)
    df_hosp, df_dom, df_caly = przygotuj_dane(df)
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
    
    # Normy do walidacji
    normy = {
        'kreatynina(0,5-1,2)': (0.5, 1.2),
        'troponina I (0-7,8))': (0, 7.8),
        'WBC(4-11)': (4, 11),
        'plt(130-450)': (130, 450),
        'hct(38-45)': (38, 45),
        'Na(137-145)': (137, 145),
        'K(3,5-5,1)': (3.5, 5.1),
        'crp(0-0,5)': (0, 0.5)
    }
    
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
    
    # Raport braków
    raport_brakow(df_caly)
    
    # Walidacja zakresów
    waliduj_zakresy(df_caly, parametry_kliniczne, normy)
    
    # Duplikaty
    duplikaty = df_caly.duplicated().sum()
    print(f"\nDuplikaty: {duplikaty}")
    
    # =========================================================================
    # ETAP 3: ANALIZA OPISOWA (Tabela 1)
    # =========================================================================
    
    wyniki_opis = tabela_opisowa(df_hosp, df_dom, parametry_kliniczne)
    wyniki_opis.to_csv('tabela_opisowa.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 4: ANALIZA CHORÓB
    # =========================================================================
    
    wyniki_choroby = analiza_chorob(df_hosp, df_dom, choroby)
    if len(wyniki_choroby) > 0:
        wyniki_choroby.to_csv('wyniki_choroby.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 5: ANALIZA JEDNOCZYNNIKOWA
    # =========================================================================
    
    wyniki_jedno = analiza_jednoczynnikowa(df_caly, parametry_kliniczne)
    wyniki_jedno.to_csv('analiza_jednoczynnikowa.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 6: PROGI KLINICZNE
    # =========================================================================
    
    progi = znajdz_progi_kliniczne(df_caly, parametry_kliniczne)
    if len(progi) > 0:
        progi.to_csv('progi_kliniczne.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 7: ANALIZA RYZYKA
    # =========================================================================
    
    if len(progi) > 0:
        analiza_ryzyka(df_caly, progi)
    
    # =========================================================================
    # ETAP 8: ANALIZA PODGRUP
    # =========================================================================
    
    analiza_podgrup(df_caly, parametry_kliniczne)
    
    # =========================================================================
    # ETAP 9: ANALIZA WIELOCZYNNIKOWA
    # =========================================================================
    
    # Wybierz parametry do modelu (p < 0.1 w analizie jednoczynnikowej)
    parametry_do_modelu = wyniki_jedno[wyniki_jedno['p_value'] < 0.1]['parametr'].tolist()
    
    if len(parametry_do_modelu) > 0:
        model, wyniki_wielo = analiza_wieloczynnikowa(df_caly, parametry_do_modelu)
        wyniki_wielo.to_csv('analiza_wieloczynnikowa.csv', sep=';', index=False)
        
        # Forest plot
        forest_plot(wyniki_wielo)
        
        # =====================================================================
        # ETAP 10: OCENA PREDYKCYJNA
        # =====================================================================
        
        X = df_caly[parametry_do_modelu]
        X = sm.add_constant(X)
        y = df_caly['outcome']
        
        metryki = ocena_predykcyjna(model, X, y)
        zapisz_wykres(metryki['fig_roc'], 'krzywa_ROC.png')
        zapisz_wykres(metryki['fig_cal'], 'krzywa_kalibracji.png')
        
        # =====================================================================
        # ETAP 11: SKALA RYZYKA
        # =====================================================================
        
        parametry_istotne = wyniki_jedno[wyniki_jedno['istotny']]['parametr'].tolist()
        if len(parametry_istotne) > 0:
            df_caly = stworz_skale_ryzyka(df_caly, parametry_istotne[:5])  # max 5 parametrów
            df_caly[['risk_score', 'risk_category']].to_csv('skala_ryzyka.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 12: WYKRESY DLA NAJWAŻNIEJSZYCH PARAMETRÓW
    # =========================================================================
    
    print("\n" + "="*70)
    print("GENEROWANIE WYKRESÓW")
    print("="*70)
    
    # Top 5 istotnych parametrów
    top_param = wyniki_jedno.sort_values('p_value').head(5)['parametr'].tolist()
    
    for i, param in enumerate(top_param, 1):
        nazwa_pliku = f'wykres_{i}_{param}.png'.replace('(', '').replace(')', '').replace(' ', '_')
        wykres_pudelkowy(df_hosp, df_dom, param, nazwa_pliku)
    
    # Specjalne wykresy dla parametrów z ekstremami
    if 'troponina I (0-7,8))' in df_caly.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        hosp = df_hosp['troponina I (0-7,8))'].dropna()
        dom = df_dom['troponina I (0-7,8))'].dropna()
        
        ax.boxplot([hosp, dom], labels=['PRZYJĘCI', 'WYPISANI'])
        ax.set_yscale('log')
        ax.set_title('Troponina I - skala logarytmiczna')
        ax.set_ylabel('Troponina (log)')
        ax.grid(True, alpha=0.3)
        
        zapisz_wykres(fig, 'wykres_troponina_log.png')
    
    if 'crp(0-0,5)' in df_caly.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        hosp = df_hosp['crp(0-0,5)'].dropna()
        dom = df_dom['crp(0-0,5)'].dropna()
        
        ax.boxplot([hosp, dom], labels=['PRZYJĘCI', 'WYPISANI'])
        ax.set_yscale('log')
        ax.set_title('CRP - skala logarytmiczna')
        ax.set_ylabel('CRP (log)')
        ax.grid(True, alpha=0.3)
        
        zapisz_wykres(fig, 'wykres_crp_log.png')
    
    # =========================================================================
    # PODSUMOWANIE
    # =========================================================================
    
    print("\n" + "="*80)
    print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE")
    print("="*80)
    
    # Podsumowanie plików
    print(f"\nWygenerowane pliki:")
    print(f"  • tabela_opisowa.csv")
    print(f"  • wyniki_choroby.csv")
    print(f"  • analiza_jednoczynnikowa.csv")
    if len(progi) > 0:
        print(f"  • progi_kliniczne.csv")
    if len(parametry_do_modelu) > 0:
        print(f"  • analiza_wieloczynnikowa.csv")
        print(f"  • forest_plot.png")
        print(f"  • krzywa_ROC.png")
        print(f"  • krzywa_kalibracji.png")
    if 'risk_score' in df_caly.columns:
        print(f"  • skala_ryzyka.csv")
    print(f"  • wykres_1_... do wykres_5_... .png")
    print(f"  • wykres_troponina_log.png")
    print(f"  • wykres_crp_log.png")
    
    print("\n" + "="*80)


# =============================================================================
# URUCHOMIENIE
# =============================================================================
if __name__ == "__main__":
    # Podaj ścieżkę do swojego pliku
    sciezka = 'BAZA_DANYCH_PACJENTOW_B.csv'  # <- ZMIEŃ NA SWOJĄ ŚCIEŻKĘ!
    main(sciezka)