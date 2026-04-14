# -*- coding: utf-8 -*-
"""
PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 10.0 - Pełna analiza z prawdziwego zdarzenia
Autor: Aneta
"""
#ETAP 1: Opis kohorty
#Tabela z medianą/IQR lub średnią/SD (zależnie od rozkładu)
#Test normalności Shapiro-Wilka#Wielkość efektu (Cohen's d)
#✅ ETAP 2: Kontrola jakości
#Raport braków danych
#Walidacja zakresów wartości
#Wykrywanie duplikatów
#Bez ignorowania warningów!
#✅ ETAP 3: Analiza jednoczynnikowa
#Odpowiedni test (t-test lub Mann-Whitney)
#Korekta Bonferroniego
#Effect size
#✅ ETAP 4: Analiza wieloczynnikowa
#Regresja logistyczna
#OR z 95% CI
#Ocena współliniowości
#✅ ETAP 5: Ocena predykcyjna
#Krzywa ROC
#AUC
#Czułość/swoistość
#✅ ETAP 6: Interpretacja kliniczna
#Które parametry są niezależne
#Które mają znaczenie praktyczne
#Które są wtórne
#✅ POPRAWKI TECHNICZNE:
#✅ Funkcje pomocnicze - zero powtarzalnego kodu
#✅ Seed losowy - reprodukowalność wykresów
#✅ plt.close() - zamykanie figur
#✅ Bez warnings.filterwarnings('ignore') - widzimy problemy#✅ Czytelny podział na funkcje

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact, chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import logit
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import warnings

# =============================================================================
# KONFIGURACJA
# =============================================================================
# UWAGA: Nie ignorujemy warningów - chcemy je widzieć i analizować!
# warnings.filterwarnings('ignore') - WYKREŚLONE!

# Ustawienia wykresów
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)  # REPRODUKOWALNOŚĆ!

# Kolory
KOLORY = {
    'hosp': '#e74c3c',
    'dom': '#3498db',
    'istotne': '#2ecc71',
    'tlo': '#f8f9fa'
}

# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================

def wczytaj_dane(sciezka_pliku, separator=';'):
    """
    Wczytuje dane z pliku CSV
    
    Parameters:
    -----------
    sciezka_pliku : str
        Ścieżka do pliku CSV
    separator : str
        Separator w pliku (domyślnie ';')
    
    Returns:
    --------
    pd.DataFrame
        Wczytane dane
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


def przygotuj_dane(df, kolumna_wyniku='hosp'):
    """
    Przygotowuje dane do analizy - tworzy kolumnę outcome
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame z danymi
    kolumna_wyniku : str
        Nazwa kolumny z wynikiem (domyślnie 'hosp')
    
    Returns:
    --------
    tuple
        (df_hosp, df_dom, df_caly)
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
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame do konwersji
    kolumny : list
        Lista kolumn do konwersji
    
    Returns:
    --------
    pd.DataFrame
        DataFrame z przekonwertowanymi kolumnami
    """
    df_copy = df.copy()
    
    for col in kolumny:
        if col in df_copy.columns:
            # Zapisz oryginalne wartości
            original = df_copy[col].copy()
            
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
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame do konwersji
    kolumny : list
        Lista kolumn z chorobami
    
    Returns:
    --------
    pd.DataFrame
        DataFrame z przekonwertowanymi kolumnami
    """
    df_copy = df.copy()
    
    mapping_tak = ['tak', 't', 'yes', 'y', '1', 'true', '+', 'tak!', 'TAK', 'T']
    mapping_nie = ['nie', 'n', 'no', '0', 'false', '-', 'NIE', 'N']
    
    for col in kolumny:
        if col in df_copy.columns:
            original = df_copy[col].copy()
            
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
    
    Parameters:
    -----------
    dane : pd.Series
        Dane do testu
    parametr : str
        Nazwa parametru
    alpha : float
        Poziom istotności
    
    Returns:
    --------
    bool
        True jeśli rozkład normalny
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
    Oblicza wielkość efektu (Cohen's d lub r)
    
    Parameters:
    -----------
    hosp : pd.Series
        Dane grupy hospitalizowanych
    dom : pd.Series
        Dane grupy do domu
    
    Returns:
    --------
    float
        Wielkość efektu
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


def korekta_bonferroni(p_values, alpha=0.05):
    """
    Stosuje korektę Bonferroniego
    
    Parameters:
    -----------
    p_values : list
        Lista wartości p
    alpha : float
        Poziom istotności
    
    Returns:
    --------
    list
        Skorygowane wartości p
    """
    n = len(p_values)
    skorygowane = [min(p * n, 1.0) for p in values]
    return skorygowane


def zapisz_wykres(fig, nazwa, zamykaj=True):
    """
    Zapisuje wykres i zamyka figurę
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figura do zapisania
    nazwa : str
        Nazwa pliku
    zamykaj : bool
        Czy zamknąć figurę po zapisie
    """
    try:
        fig.savefig(nazwa, dpi=300, bbox_inches='tight')
        print(f"  ✓ Zapisano: {nazwa}")
    except Exception as e:
        print(f"  ✗ Błąd zapisu {nazwa}: {e}")
    finally:
        if zamykaj:
            plt.close(fig)


def waliduj_zakresy(df, parametry, normy):
    """
    Waliduje zakresy wartości parametrów
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dane do walidacji
    parametry : list
        Lista parametrów
    normy : dict
        Słownik z normami {parametr: (min, max)}
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
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dane do analizy
    
    Returns:
    --------
    pd.DataFrame
        Raport braków
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
    Generuje tabelę opisową kohorty
    
    Parameters:
    -----------
    df_hosp : pd.DataFrame
        Dane hospitalizowanych
    df_dom : pd.DataFrame
        Dane do domu
    parametry : list
        Lista parametrów do analizy
    
    Returns:
    --------
    pd.DataFrame
        Tabela opisowa
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
        "Parametr", "n_hosp", "Hosp", "n_dom", "Dom", "p-value", "Cohen's d"
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
    Analiza chorób współistniejących
    
    Parameters:
    -----------
    df_hosp : pd.DataFrame
        Dane hospitalizowanych
    df_dom : pd.DataFrame
        Dane do domu
    choroby : list
        Lista chorób
    
    Returns:
    --------
    pd.DataFrame
        Wyniki analizy
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

def analiza_jednoczynnikowa(df_caly, parametry, kolumna_wyniku='outcome'):
    """
    Przeprowadza analizę jednoczynnikową
    
    Parameters:
    -----------
    df_caly : pd.DataFrame
        Pełne dane z kolumną outcome
    parametry : list
        Lista parametrów
    kolumna_wyniku : str
        Nazwa kolumny z wynikiem
    
    Returns:
    --------
    pd.DataFrame
        Wyniki analizy
    """
    print("\n" + "="*70)
    print("ANALIZA JEDNOCZYNNIKOWA")
    print("="*70)
    
    wyniki = []
    p_values = []
    
    for param in parametry:
        if param in df_caly.columns:
            dane = df_caly[[param, kolumna_wyniku]].dropna()
            
            if len(dane) > 0:
                hosp = dane[dane[kolumna_wyniku] == 1][param]
                dom = dane[dane[kolumna_wyniku] == 0][param]
                
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
    print("\n{:<25} {:>8} {:>10} {:>10} {:>10}".format(
        "Parametr", "p-value", "p-skoryg", "d-Cohena", "Istotny"
    ))
    print("-"*70)
    
    for _, row in df_wyniki.iterrows():
        print("{:<25} {:>8.4f} {:>10.4f} {:>10.2f} {:>10}".format(
            row['parametr'][:24],
            row['p_value'],
            row['p_skorygowane'],
            row['effect_size'],
            "✓" if row['istotny'] else "✗"
        ))
    
    return df_wyniki


# =============================================================================
# ANALIZA WIELOCZYNNIKOWA
# =============================================================================

def analiza_wieloczynnikowa(df_caly, parametry, kolumna_wyniku='outcome'):
    """
    Przeprowadza analizę wieloczynnikową (regresja logistyczna)
    
    Parameters:
    -----------
    df_caly : pd.DataFrame
        Pełne dane z kolumną outcome
    parametry : list
        Lista parametrów do modelu
    kolumna_wyniku : str
        Nazwa kolumny z wynikiem
    
    Returns:
    --------
    tuple
        (model, wyniki_df)
    """
    print("\n" + "="*70)
    print("ANALIZA WIELOCZYNNIKOWA - REGRESJA LOGISTYCZNA")
    print("="*70)
    
    # Przygotowanie danych
    df_model = df_caly[parametry + [kolumna_wyniku]].dropna()
    print(f"Liczba obserwacji w modelu: {len(df_model)}")
    
    # Sprawdzenie współliniowości
    print("\nMacierz korelacji predyktorów:")
    corr_matrix = df_model[parametry].corr()
    print(corr_matrix.round(2))
    
    # Wysoka korelacja?
    for i in range(len(parametry)):
        for j in range(i+1, len(parametry)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                print(f"  UWAGA: Wysoka korelacja między {parametry[i]} a {parametry[j]}")
    
    # Regresja logistyczna
    X = df_model[parametry]
    X = sm.add_constant(X)
    y = df_model[kolumna_wyniku]
    
    model = sm.Logit(y, X).fit(disp=0)
    
    print("\nWyniki regresji logistycznej:")
    print(model.summary().tables[1])
    
    # Wyniki w DataFrame
    wyniki = []
    for i, param in enumerate(['const'] + parametry):
        if param in model.params.index:
            or_val = np.exp(model.params[param])
            ci_low = np.exp(model.conf_int().loc[param][0])
            ci_high = np.exp(model.conf_int().loc[param][1])
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
    Ocenia przydatność predykcyjną modelu
    
    Parameters:
    -----------
    model : statsmodels.discrete.discrete_model.BinaryResultsWrapper
        Wytrenowany model
    X : pd.DataFrame
        Predyktory
    y : pd.Series
        Wartości rzeczywiste
    
    Returns:
    --------
    dict
        Metryki oceny
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
    
    print(f"Czułość: {czulosc:.3f}")
    print(f"Swoistość: {swoistosc:.3f}")
    print(f"Dokładność: {dokladnosc:.3f}")
    
    # Wykres ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Krzywa ROC')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return {
        'auc': roc_auc,
        'czulosc': czulosc,
        'swoistosc': swoistosc,
        'dokladnosc': dokladnosc,
        'figura': fig
    }


# =============================================================================
# WYKRESY
# =============================================================================

def wykres_pudelkowy(df_hosp, df_dom, param, nazwa_pliku):
    """
    Tworzy wykres pudełkowy dla pojedynczego parametru
    
    Parameters:
    -----------
    df_hosp : pd.DataFrame
        Dane hospitalizowanych
    df_dom : pd.DataFrame
        Dane do domu
    param : str
        Nazwa parametru
    nazwa_pliku : str
        Nazwa pliku do zapisu
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
    
    # Dodanie punktów (z seed dla reprodukowalności!)
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


# =============================================================================
# INTERPRETACJA KLINICZNA
# =============================================================================

def interpretacja_kliniczna(wyniki_jedno, wyniki_wielo, wyniki_choroby):
    """
    Generuje interpretację kliniczną wyników
    
    Parameters:
    -----------
    wyniki_jedno : pd.DataFrame
        Wyniki analizy jednoczynnikowej
    wyniki_wielo : pd.DataFrame
        Wyniki analizy wieloczynnikowej
    wyniki_choroby : pd.DataFrame
        Wyniki analizy chorób
    """
    print("\n" + "="*70)
    print("INTERPRETACJA KLINICZNA")
    print("="*70)
    
    # 1. Istotne statystycznie (po korekcie)
    istotne = wyniki_jedno[wyniki_jedno['istotny'] == True]
    
    print("\n1. PARAMETRY ISTOTNE STATYSTYCZNIE (po korekcie Bonferroniego):")
    for _, row in istotne.iterrows():
        print(f"  • {row['parametr']}: p_skoryg={row['p_skorygowane']:.4f}, d={row['effect_size']:.2f}")
    
    # 2. Wielkość efektu
    print("\n2. WIELKOŚĆ EFEKTU (Cohen's d):")
    for _, row in wyniki_jedno.iterrows():
        if abs(row['effect_size']) >= 0.8:
            print(f"  • {row['parametr']}: d={row['effect_size']:.2f} - duży efekt")
        elif abs(row['effect_size']) >= 0.5:
            print(f"  • {row['parametr']}: d={row['effect_size']:.2f} - średni efekt")
        elif abs(row['effect_size']) >= 0.2:
            print(f"  • {row['parametr']}: d={row['effect_size']:.2f} - mały efekt")
    
    # 3. Niezależne czynniki (regresja)
    print("\n3. NIEZALEŻNE CZYNNIKI RYZYKA (regresja logistyczna):")
    for _, row in wyniki_wielo.iterrows():
        if row['parametr'] != 'const' and row['p_value'] < 0.05:
            print(f"  • {row['parametr']}: OR={row['OR']:.2f} (95% CI: {row['CI_95%']})")
    
    # 4. Choroby współistniejące
    if wyniki_choroby is not None and len(wyniki_choroby) > 0:
        print("\n4. CHOROBY WSPÓŁISTNIEJĄCE:")
        for _, row in wyniki_choroby.iterrows():
            if row['p_value'] < 0.05:
                print(f"  • {row['choroba']}: OR={row['OR']:.2f} (95% CI: {row['CI_95%']})")
    
    # 5. Znaczenie praktyczne
    print("\n5. ZNACZENIE PRAKTYCZNE:")
    print("  Parametry, które są zarówno:")
    print("  • istotne statystycznie (p_skoryg < 0.05)")
    print("  • mają co najmniej średni efekt (|d| ≥ 0.5)")
    print("  • pozostają istotne w modelu wieloczynnikowym")
    
    for _, row in istotne.iterrows():
        if abs(row['effect_size']) >= 0.5:
            czy_w_modelu = False
            if wyniki_wielo is not None:
                czy_w_modelu = any(
                    (w['parametr'] == row['parametr'] and w['p_value'] < 0.05)
                    for _, w in wyniki_wielo.iterrows()
                )
            if czy_w_modelu:
                print(f"  ✓ {row['parametr']} - silny, niezależny czynnik")
            else:
                print(f"  • {row['parametr']} - istotny, ale zależny od innych czynników")


# =============================================================================
# GŁÓWNA FUNKCJA
# =============================================================================

def main(sciezka_pliku):
    """
    Główna funkcja analizy
    
    Parameters:
    -----------
    sciezka_pliku : str
        Ścieżka do pliku z danymi
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
    
    # Normy
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
    # ETAP 3: ANALIZA OPISOWA
    # =========================================================================
    
    wyniki_opis = tabela_opisowa(df_hosp, df_dom, parametry_kliniczne)
    wyniki_opis.to_csv('tabela_opisowa.csv', sep=';', index=False)
    
    wyniki_choroby = analiza_chorob(df_hosp, df_dom, choroby)
    if len(wyniki_choroby) > 0:
        wyniki_choroby.to_csv('wyniki_choroby.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 4: ANALIZA JEDNOCZYNNIKOWA
    # =========================================================================
    
    wyniki_jedno = analiza_jednoczynnikowa(df_caly, parametry_kliniczne)
    wyniki_jedno.to_csv('analiza_jednoczynnikowa.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 5: ANALIZA WIELOCZYNNIKOWA
    # =========================================================================
    
    # Wybierz parametry do modelu (istotne w analizie jednoczynnikowej)
    parametry_do_modelu = wyniki_jedno[wyniki_jedno['p_value'] < 0.1]['parametr'].tolist()
    
    if len(parametry_do_modelu) > 0:
        model, wyniki_wielo = analiza_wieloczynnikowa(df_caly, parametry_do_modelu)
        wyniki_wielo.to_csv('analiza_wieloczynnikowa.csv', sep=';', index=False)
        
        # =====================================================================
        # ETAP 6: OCENA PREDYKCYJNA
        # =====================================================================
        
        X = df_caly[parametry_do_modelu]
        X = sm.add_constant(X)
        y = df_caly['outcome']
        
        metryki = ocena_predykcyjna(model, X, y)
        zapisz_wykres(metryki['figura'], 'krzywa_ROC.png')
    else:
        print("\nBrak parametrów do modelu wieloczynnikowego (p<0.1)")
        model, wyniki_wielo = None, None
    
    # =========================================================================
    # ETAP 7: WYKRESY DLA NAJWAŻNIEJSZYCH PARAMETRÓW
    # =========================================================================
    
    print("\n" + "="*70)
    print("GENEROWANIE WYKRESÓW")
    print("="*70)
    
    # Top 5 istotnych parametrów
    top_param = wyniki_jedno.sort_values('p_value').head(5)['parametr'].tolist()
    
    for i, param in enumerate(top_param, 1):
        nazwa_pliku = f'wykres_{i}_{param}.png'.replace('(', '').replace(')', '').replace(' ', '_')
        wykres_pudelkowy(df_hosp, df_dom, param, nazwa_pliku)
    
    # Specjalne wykresy dla troponiny i CRP
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
    # ETAP 8: INTERPRETACJA KLINICZNA
    # =========================================================================
    
    interpretacja_kliniczna(wyniki_jedno, wyniki_wielo, wyniki_choroby)
    
    # =========================================================================
    # PODSUMOWANIE
    # =========================================================================
    
    print("\n" + "="*80)
    print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE")
    print("="*80)
    print(f"\nWygenerowane pliki:")
    print(f"  • tabela_opisowa.csv")
    print(f"  • wyniki_choroby.csv")
    print(f"  • analiza_jednoczynnikowa.csv")
    if wyniki_wielo is not None:
        print(f"  • analiza_wieloczynnikowa.csv")
    print(f"  • krzywa_ROC.png")
    print(f"  • wykres_1_... do wykres_5_... .png")
    print(f"  • wykres_troponina_log.png")
    print(f"  • wykres_crp_log.png")
    print("="*80)


# =============================================================================
# URUCHOMIENIE
# =============================================================================
if __name__ == "__main__":
    # main('BAZA_DANYCH_PACJENTOW_B.csv')  # Odkomentuj i podaj ścieżkę
    print("Aby uruchomić, odkomentuj linię z main() i podaj ścieżkę do pliku")