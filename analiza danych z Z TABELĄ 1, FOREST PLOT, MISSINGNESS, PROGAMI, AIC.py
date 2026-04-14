# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:10:13 2026

@author: aneta
"""
# -*- coding: utf-8 -*-
"""
PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 20.0 - FINAŁ Z KOMPLETNYM RAPORTEM
Autor: Aneta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import (roc_curve, auc, confusion_matrix, brier_score_loss,
                            roc_auc_score)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
# PRE-DEFINIOWANY PLAN ANALIZY
# =============================================================================
ZMIENNE_OBOWIAZKOWE = ['wiek']
ZMIENNE_DODATKOWE = [
    'SpO2',
    'crp(0-0,5)',
    'kreatynina(0,5-1,2)',
    'MAP',
    'troponina I (0-7,8))',
    'HGB(12,4-15,2)'
]

ZMIENNE_LOG = [
    'crp(0-0,5)',
    'troponina I (0-7,8))',
    'kreatynina(0,5-1,2)'
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
    """Przygotowuje dane z kolumną outcome"""
    df_copy = df.copy()
    
    if 'outcome' not in df_copy.columns:
        print("\n" + "="*70)
        print("❌ BRAK KOLUMNY 'outcome' W PLIKU!")
        return None, None, None
    
    df_copy = df_copy[df_copy['outcome'].notna()]
    df_copy['outcome'] = pd.to_numeric(df_copy['outcome'], errors='coerce')
    df_copy = df_copy[df_copy['outcome'].isin([0, 1])]
    
    if len(df_copy) == 0:
        print("✗ Brak poprawnych wartości w kolumnie 'outcome'")
        return None, None, None
    
    if id_pacjenta and id_pacjenta in df_copy.columns:
        duplikaty_id = df_copy[id_pacjenta].duplicated().sum()
        if duplikaty_id > 0:
            print(f"\n⚠️ Znaleziono {duplikaty_id} duplikatów ID pacjenta")
            df_copy = df_copy.drop_duplicates(subset=[id_pacjenta], keep='first')
    
    df_hosp = df_copy[df_copy['outcome'] == 1].copy()
    df_dom = df_copy[df_copy['outcome'] == 0].copy()
    
    print(f"\n✓ Podział danych:")
    print(f"  • Hospitalizowani (outcome=1): {len(df_hosp)} ({len(df_hosp)/len(df_copy)*100:.1f}%)")
    print(f"  • Do domu (outcome=0): {len(df_dom)} ({len(df_dom)/len(df_copy)*100:.1f}%)")
    print(f"  • Razem: {len(df_copy)}")
    
    return df_hosp, df_dom, df_caly


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


def cliff_delta(x, y):
    """Cliff's delta"""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0
    try:
        U, _ = stats.mannwhitneyu(x, y, alternative='two-sided', method='asymptotic')
        return (2 * U) / (n1 * n2) - 1
    except:
        return 0


def sprawdz_epv_i_raport(df, zmienne, outcome='outcome', prog=10):
    """Sprawdza EPV i zwraca status"""
    n_events = df[outcome].sum()
    n_vars = len(zmienne)
    epv = n_events / n_vars if n_vars > 0 else 0
    
    print(f"\n📊 EPV: {epv:.1f} (zdarzeń={n_events}, predyktorów={n_vars})")
    
    if epv < prog:
        print(f"  ⚠️ EPV < {prog} - model niestabilny!")
        return False, epv
    else:
        print(f"  ✓ EPV OK")
        return True, epv


def sprawdz_vif(X):
    """Sprawdza VIF i zwraca ostrzeżenia"""
    vif_data = pd.DataFrame()
    vif_data['zmienna'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    ostrzezenia = []
    for _, row in vif_data.iterrows():
        if row['zmienna'] != 'const':
            if row['VIF'] > 10:
                ostrzezenia.append(f"  ⚠️ {row['zmienna']}: VIF={row['VIF']:.2f} (wysoka)")
            elif row['VIF'] > 5:
                ostrzezenia.append(f"  • {row['zmienna']}: VIF={row['VIF']:.2f} (umiarkowana)")
    
    return ostrzezenia, vif_data


# =============================================================================
# 1️⃣ TABELA 1 - KOMPLETNA
# =============================================================================

def tabela_1_kompletna(df_hosp, df_dom, parametry_ciagle, choroby):
    """
    Kompletna Tabela 1 z:
    - zmiennymi ciągłymi (mediana [IQR], p-value, Cliff's delta)
    - zmiennymi kategorialnymi (n (%), p-value, OR)
    """
    print("\n" + "="*80)
    print("📊 TABELA 1: CHARAKTERYSTYKA KOHORTY")
    print("="*80)
    print(f"  Hospitalizowani: n = {len(df_hosp)}")
    print(f"  Do domu: n = {len(df_dom)}")
    print(f"  RAZEM: n = {len(df_hosp) + len(df_dom)}")
    
    wyniki = []
    
    # ========== ZMIENNE CIĄGŁE ==========
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
                
                _, p = stats.mannwhitneyu(hosp, dom)
                d = cliff_delta(hosp, dom)
                
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
                    param[:24], len(hosp), hosp_stat, len(dom), dom_stat, p, d
                ))
    
    # ========== ZMIENNE KATEGORIALNE ==========
    print("\n--- ZMIENNE KATEGORIALNE ---")
    print("{:<25} {:>25} {:>25} {:>12} {:>10}".format(
        "Choroba", "Hospitalizowani", "Do domu", "p-value", "OR"
    ))
    print("-"*100)
    
    for choroba in choroby:
        if choroba in df_hosp.columns:
            hosp = df_hosp[choroba].dropna()
            dom = df_dom[choroba].dropna()
            
            if len(hosp) > 0 and len(dom) > 0:
                hosp_tak = (hosp == 1).sum()
                dom_tak = (dom == 1).sum()
                
                hosp_stat = f"{hosp_tak}/{len(hosp)} ({hosp_tak/len(hosp)*100:.1f}%)"
                dom_stat = f"{dom_tak}/{len(dom)} ({dom_tak/len(dom)*100:.1f}%)"
                
                # Poprawka 0.5 dla OR
                a = hosp_tak + 0.5
                b = len(hosp) - hosp_tak + 0.5
                c = dom_tak + 0.5
                d = len(dom) - dom_tak + 0.5
                oddsratio = (a * d) / (b * c)
                
                tabela = [[hosp_tak, len(hosp)-hosp_tak], [dom_tak, len(dom)-dom_tak]]
                _, p = fisher_exact(tabela)
                
                wyniki.append({
                    'typ': 'kategorialny',
                    'parametr': choroba,
                    'hosp_stat': hosp_stat,
                    'dom_stat': dom_stat,
                    'p_value': p,
                    'effect_size': oddsratio
                })
                
                print("{:<25} {:>25} {:>25}   {:<8.4f}   {:>6.2f}".format(
                    choroba, hosp_stat, dom_stat, p, oddsratio
                ))
    
    return pd.DataFrame(wyniki)


# =============================================================================
# 3️⃣ RAPORT MISSINGNESS DLA TOP PARAMETRÓW
# =============================================================================

def raport_missingness_top(df_hosp, df_dom, top_param):
    """
    Raport braków danych tylko dla top parametrów
    """
    print("\n" + "="*80)
    print("📋 RAPORT BRAKÓW - TOP 5 PARAMETRÓW")
    print("="*80)
    
    for param in top_param[:5]:
        if param in df_hosp.columns and param in df_dom.columns:
            brak_hosp = df_hosp[param].isna().sum()
            brak_dom = df_dom[param].isna().sum()
            proc_hosp = (brak_hosp / len(df_hosp)) * 100
            proc_dom = (brak_dom / len(df_dom)) * 100
            
            print(f"\n{param}:")
            print(f"  Hospitalizowani: {brak_hosp}/{len(df_hosp)} ({proc_hosp:.1f}%)")
            print(f"  Do domu: {brak_dom}/{len(df_dom)} ({proc_dom:.1f}%)")


# =============================================================================
# 4️⃣ PROGI KLINICZNE - POPRAWIONE Z KIERUNKIEM
# =============================================================================

def progi_kliniczne_poprawione(df, top_param):
    """
    Progi kliniczne z uwzględnieniem kierunku efektu
    """
    print("\n" + "="*80)
    print("🎯 PROGI KLINICZNE (z kierunkiem efektu)")
    print("="*80)
    
    wyniki = []
    
    for param in top_param[:5]:
        if param not in df.columns:
            continue
        
        dane = df[[param, 'outcome']].dropna()
        if len(dane) < 10:
            continue
        
        # Określenie kierunku na podstawie median
        hosp_med = dane[dane['outcome'] == 1][param].median()
        dom_med = dane[dane['outcome'] == 0][param].median()
        kierunek = 'wyższe' if hosp_med > dom_med else 'niższe'
        
        # ROC z odpowiednim kierunkiem
        if kierunek == 'wyższe':
            fpr, tpr, progi = roc_curve(dane['outcome'], dane[param])
        else:
            fpr, tpr, progi = roc_curve(dane['outcome'], -dane[param])
        
        youden = tpr - fpr
        opt_idx = np.argmax(youden)
        
        if kierunek == 'wyższe':
            prog_opt = progi[opt_idx]
            y_pred = (dane[param] >= prog_opt).astype(int)
        else:
            prog_opt = -progi[opt_idx]
            y_pred = (dane[param] <= prog_opt).astype(int)
        
        # Metryki
        tn = ((y_pred == 0) & (dane['outcome'] == 0)).sum()
        fp = ((y_pred == 1) & (dane['outcome'] == 0)).sum()
        fn = ((y_pred == 0) & (dane['outcome'] == 1)).sum()
        tp = ((y_pred == 1) & (dane['outcome'] == 1)).sum()
        
        czulosc = tp / (tp + fn) if (tp + fn) > 0 else 0
        swoistosc = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\n{param} (kierunek: {kierunek} = większe ryzyko):")
        print(f"  Próg: {prog_opt:.2f}")
        print(f"  Czułość: {czulosc:.3f}")
        print(f"  Swoistość: {swoistosc:.3f}")
        
        wyniki.append({
            'parametr': param,
            'kierunek': kierunek,
            'prog': prog_opt,
            'czulosc': czulosc,
            'swoistosc': swoistosc
        })
    
    return pd.DataFrame(wyniki)


# =============================================================================
# TRANSFORMACJE
# =============================================================================

def przygotuj_zmienne_do_modelu(df, zmienne_obow, zmienne_dod, zmienne_log):
    """
    Przygotowuje listę zmiennych z transformacjami log
    """
    df_model = df.copy()
    wszystkie = zmienne_obow + zmienne_dod
    dostepne = [z for z in wszystkie if z in df_model.columns]
    
    for z in zmienne_log:
        if z in df_model.columns:
            df_model[f'log_{z}'] = np.log1p(df_model[z].clip(lower=0))
            if z in dostepne:
                dostepne.remove(z)
                dostepne.append(f'log_{z}')
    
    return df_model, dostepne


# =============================================================================
# ANALIZA JEDNOCZYNNIKOWA Z FDR
# =============================================================================

def analiza_jednoczynnikowa(df_caly, parametry):
    """
    Analiza jednoczynnikowa z FDR
    """
    print("\n" + "="*80)
    print("📈 ANALIZA JEDNOCZYNNIKOWA Z FDR")
    print("="*80)
    
    wyniki = []
    p_values_raw = []
    
    for param in parametry:
        if param in df_caly.columns:
            hosp = df_caly[df_caly['outcome'] == 1][param].dropna()
            dom = df_caly[df_caly['outcome'] == 0][param].dropna()
            
            if len(hosp) > 0 and len(dom) > 0:
                _, p = stats.mannwhitneyu(hosp, dom)
                d = cliff_delta(hosp, dom)
                
                wyniki.append({
                    'parametr': param,
                    'p_raw': p,
                    'cliff_delta': d,
                    'n_hosp': len(hosp),
                    'n_dom': len(dom)
                })
                p_values_raw.append(p)
    
    if len(p_values_raw) > 0:
        _, p_adjusted, _, _ = multipletests(p_values_raw, method='fdr_bh')
        for i, w in enumerate(wyniki):
            w['p_fdr'] = p_adjusted[i]
            w['istotny'] = w['p_fdr'] < 0.05
    
    df_wyniki = pd.DataFrame(wyniki).sort_values('p_raw')
    
    print("\n{:<25} {:>10} {:>12} {:>10} {:>8} {:>8}".format(
        "Parametr", "p_raw", "p_FDR", "Cliff's d", "n_hosp", "n_dom"
    ))
    print("-"*75)
    
    for _, row in df_wyniki.iterrows():
        print("{:<25} {:>10.4f} {:>12.4f} {:>10.2f} {:>6} {:>6}".format(
            row['parametr'][:24], row['p_raw'], row['p_fdr'],
            row['cliff_delta'], row['n_hosp'], row['n_dom']
        ))
    
    top5 = df_wyniki.head(5)['parametr'].tolist()
    return df_wyniki, top5


# =============================================================================
# MODELE INFERENCYJNE
# =============================================================================

def model_podstawowy(df, outcome='outcome'):
    """Model podstawowy (tylko wiek)"""
    print("\n" + "="*80)
    print("📊 MODEL 1: PODSTAWOWY (TYLKO WIEK)")
    print("="*80)
    
    if 'wiek' not in df.columns:
        print("  ✗ Brak wieku")
        return None, None, None
    
    df_cc = df[['wiek', outcome]].dropna()
    print(f"  Complete-case: n={len(df_cc)}")
    
    if len(df_cc) < 10:
        return None, None, None
    
    X = sm.add_constant(df_cc['wiek'])
    y = df_cc[outcome]
    
    try:
        model = sm.Logit(y, X).fit(disp=0)
        print(model.summary().tables[1])
        
        wyniki = pd.DataFrame([{
            'parametr': 'wiek',
            'OR': np.exp(model.params['wiek']),
            'CI_95%': f"{np.exp(model.conf_int().loc['wiek'])[0]:.2f}-{np.exp(model.conf_int().loc['wiek'])[1]:.2f}",
            'p_value': model.pvalues['wiek']
        }])
        
        return model, wyniki, len(df_cc)
    except:
        return None, None, None


def model_rozszerzony(df, zmienne, outcome='outcome'):
    """Model rozszerzony (wiek + biomarkery) - GŁÓWNY"""
    print("\n" + "="*80)
    print("📊 MODEL 2: ROZSZERZONY (GŁÓWNY)")
    print("="*80)
    
    dostepne = [z for z in zmienne if z in df.columns]
    print(f"  Zmienne: {', '.join(dostepne)}")
    
    df_cc = df[dostepne + [outcome]].dropna()
    print(f"  Complete-case: n={len(df_cc)}")
    
    if len(df_cc) < 10:
        return None, None, None, False
    
    epv_ok, epv = sprawdz_epv_i_raport(df_cc, dostepne)
    
    X = sm.add_constant(df_cc[dostepne])
    y = df_cc[outcome]
    
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        print(model.summary().tables[1])
        
        ostrzezenia, _ = sprawdz_vif(X)
        for o in ostrzezenia:
            print(o)
        
        # 5️⃣ PSEUDO R² I AIC
        print(f"\n📈 Pseudo R² (McFadden): {model.prsquared:.4f}")
        print(f"📈 AIC: {model.aic:.2f}")
        
        wyniki = []
        for param in dostepne:
            wyniki.append({
                'parametr': param,
                'OR': np.exp(model.params[param]),
                'CI_95%': f"{np.exp(model.conf_int().loc[param])[0]:.2f}-{np.exp(model.conf_int().loc[param])[1]:.2f}",
                'p_value': model.pvalues[param]
            })
        
        return model, pd.DataFrame(wyniki), len(df_cc), epv_ok
    except Exception as e:
        print(f"  ⚠️ Błąd: {e}")
        return None, None, None, False


def model_z_redukcja(df, zmienne, outcome='outcome'):
    """Model z redukcją - SENSITIVITY ANALYSIS"""
    print("\n" + "="*80)
    print("📊 MODEL 3: Z REDUKCJĄ (SENSITIVITY ANALYSIS)")
    print("="*80)
    
    dostepne = [z for z in zmienne if z in df.columns]
    df_cc = df[dostepne + [outcome]].dropna()
    
    n_events = df_cc[outcome].sum()
    max_pred = int(n_events / 10)
    
    print(f"  Maksymalna liczba predyktorów (EPV≥10): {max_pred}")
    
    if len(dostepne) <= max_pred:
        print("  ✓ EPV OK - używam modelu rozszerzonego")
        return model_rozszerzony(df, zmienne, outcome)
    
    # Redukcja według priorytetów klinicznych
    priorytety = {
        'wiek': 10,
        'log_crp(0-0,5)': 9,
        'SpO2': 8,
        'log_kreatynina(0,5-1,2)': 7,
        'MAP': 6
    }
    
    dostepne.sort(key=lambda x: priorytety.get(x, 0), reverse=True)
    wybrane = dostepne[:max_pred]
    
    print(f"  Wybrane: {', '.join(wybrane)}")
    
    return model_rozszerzony(df, wybrane, outcome)


# =============================================================================
# 2️⃣ FOREST PLOT
# =============================================================================

def forest_plot(wyniki, nazwa_pliku='forest_plot.png'):
    """
    Forest plot z OR i 95% CI
    """
    if wyniki is None or len(wyniki) == 0:
        return
    
    df_plot = wyniki.copy()
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
    ax.set_title('Czynniki ryzyka hospitalizacji')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Dodanie wartości OR
    for i, (_, row) in enumerate(df_plot.iterrows()):
        ax.text(row['OR'] * 1.1, i, f"{row['OR']:.2f}", 
                verticalalignment='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(nazwa_pliku, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Forest plot: {nazwa_pliku}")


# =============================================================================
# MODEL PREDYKCYJNY
# =============================================================================

def model_predykcyjny(df, zmienne, outcome='outcome'):
    """Model predykcyjny z walidacją"""
    print("\n" + "="*80)
    print("🤖 MODEL PREDYKCYJNY (sklearn)")
    print("="*80)
    
    dostepne = [z for z in zmienne if z in df.columns]
    df_cc = df[dostepne + [outcome]].dropna()
    
    if len(df_cc) < 20:
        print("  ✗ Zbyt mało danych")
        return None
    
    X = df_cc[dostepne].values
    y = df_cc[outcome].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    auc_boot = []
    for i in range(1000):
        boot_idx = resample(range(len(y_test)), replace=True, random_state=i)
        if len(np.unique(y_test[boot_idx])) < 2:
            continue
        try:
            auc_boot.append(roc_auc_score(y_test[boot_idx], y_pred_prob[boot_idx]))
        except:
            continue
    
    auc_ci = (np.percentile(auc_boot, 2.5), np.percentile(auc_boot, 97.5)) if auc_boot else (roc_auc, roc_auc)
    
    brier = brier_score_loss(y_test, y_pred_prob)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    
    print(f"\n  AUC: {roc_auc:.3f} (95% CI: {auc_ci[0]:.3f}-{auc_ci[1]:.3f})")
    print(f"  Brier: {brier:.4f}")
    print(f"  CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Krzywa ROC')
    ax1.legend()
    
    fig2, ax2 = plt.subplots()
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
    ax2.plot(prob_pred, prob_true, marker='o')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Observed')
    ax2.set_title(f'Kalibracja (Brier={brier:.4f})')
    
    return {
        'auc': roc_auc,
        'auc_ci': auc_ci,
        'brier': brier,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'fig_roc': fig1,
        'fig_cal': fig2
    }


# =============================================================================
# WYKRESY
# =============================================================================

def wykres_pudelkowy(df_hosp, df_dom, param, nazwa):
    """Wykres pudełkowy"""
    hosp = df_hosp[param].dropna()
    dom = df_dom[param].dropna()
    
    if len(hosp) == 0 or len(dom) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    _, p = stats.mannwhitneyu(hosp, dom)
    d = cliff_delta(hosp, dom)
    
    ax1.boxplot([hosp, dom], labels=['HOSP', 'DOM'], patch_artist=True,
                boxprops=dict(facecolor=KOLORY['hosp']),
                medianprops=dict(color='black'))
    ax1.set_title(param)
    ax1.grid(True, alpha=0.3)
    
    ax2.axis('off')
    text = f"""
    {param}
    HOSP: n={len(hosp)}, mediana={hosp.median():.2f}
    DOM: n={len(dom)}, mediana={dom.median():.2f}
    p = {p:.4f}
    Cliff's d = {d:.2f}
    """
    ax2.text(0.1, 0.5, text, fontsize=12, transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(nazwa, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {nazwa}")


# =============================================================================
# PORÓWNANIE MODELI Z AIC
# =============================================================================

def porownaj_modele_z_aic(model1, model2, model3, n1, n2, n3, epv_ok):
    """
    Porównanie modeli z uwzględnieniem AIC i pseudo R²
    """
    print("\n" + "="*80)
    print("📊 PORÓWNANIE MODELI INFERENCYJNYCH")
    print("="*80)
    
    print(f"\n{'Model':<20} {'n':<8} {'Zmienne':<15} {'Pseudo R²':<12} {'AIC':<10} {'Status':<12}")
    print("-"*80)
    
    if model1 is not None:
        print(f"{'Podstawowy':<20} {n1:<8} {'wiek':<15} {model1.prsquared:.4f}     {model1.aic:.2f}   {'informacyjny':<12}")
    
    if model2 is not None:
        status = "GŁÓWNY" if epv_ok else "niestabilny"
        print(f"{'Rozszerzony':<20} {n2:<8} {len(model2.params)-1:<15} {model2.prsquared:.4f}     {model2.aic:.2f}   {status:<12}")
    
    if model3 is not None and model3 is not model2:
        print(f"{'Z redukcją':<20} {n3:<8} {len(model3.params)-1:<15} {model3.prsquared:.4f}     {model3.aic:.2f}   {'sensitivity':<12}")
    
    print("\n🎯 WNIOSKI KLINICZNE:")
    
    if model2 is not None:
        print("\n  Model główny (rozszerzony) - czynniki niezależne:")
        for param in model2.params.index:
            if param != 'const' and model2.pvalues[param] < 0.05:
                or_val = np.exp(model2.params[param])
                ci = np.exp(model2.conf_int().loc[param])
                print(f"    • {param}: OR={or_val:.2f} (95% CI: {ci[0]:.2f}-{ci[1]:.2f})")


# =============================================================================
# GŁÓWNA FUNKCJA
# =============================================================================

def main(sciezka_pliku):
    """Główna funkcja"""
    
    print("\n" + "="*80)
    print("🏥 PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH - WERSJA 20.0")
    print("="*80)
    
    # =========================================================================
    # 1. WCZYTYWANIE
    # =========================================================================
    df = wczytaj_dane(sciezka_pliku)
    if df is None:
        return
    
    df_hosp, df_dom, df_caly = przygotuj_dane_z_outcome(df)
    if df_hosp is None:
        return
    
    # =========================================================================
    # 2. KONWERSJA
    # =========================================================================
    parametry = [
        'wiek', 'RR', 'MAP', 'SpO2', 'AS', 'mleczany',
        'kreatynina(0,5-1,2)', 'troponina I (0-7,8))',
        'HGB(12,4-15,2)', 'WBC(4-11)', 'plt(130-450)',
        'hct(38-45)', 'Na(137-145)', 'K(3,5-5,1)', 'crp(0-0,5)'
    ]
    choroby = ['dm', 'wątroba', 'naczyniowe', 'zza', 'npl']
    
    df_hosp = konwertuj_na_numeryczne(df_hosp, parametry)
    df_dom = konwertuj_na_numeryczne(df_dom, parametry)
    df_caly = konwertuj_na_numeryczne(df_caly, parametry)
    
    df_hosp = konwertuj_choroby(df_hosp, choroby)
    df_dom = konwertuj_choroby(df_dom, choroby)
    df_caly = konwertuj_choroby(df_caly, choroby)
    
    # =========================================================================
    # 3. TABELA 1
    # =========================================================================
    tabela = tabela_1_kompletna(df_hosp, df_dom, parametry, choroby)
    tabela.to_csv('tabela_1.csv', sep=';', index=False)
    
    # =========================================================================
    # 4. ANALIZA JEDNOCZYNNIKOWA
    # =========================================================================
    wyniki_fdr, top5 = analiza_jednoczynnikowa(df_caly, parametry)
    wyniki_fdr.to_csv('analiza_jednoczynnikowa.csv', sep=';', index=False)
    
    # =========================================================================
    # 5. RAPORT MISSINGNESS DLA TOP 5
    # =========================================================================
    raport_missingness_top(df_hosp, df_dom, top5)
    
    # =========================================================================
    # 6. PROGI KLINICZNE
    # =========================================================================
    progi = progi_kliniczne_poprawione(df_caly, top5)
    if progi is not None:
        progi.to_csv('progi_kliniczne.csv', sep=';', index=False)
    
    # =========================================================================
    # 7. TRANSFORMACJE
    # =========================================================================
    df_model, zmienne = przygotuj_zmienne_do_modelu(
        df_caly, ZMIENNE_OBOWIAZKOWE, ZMIENNE_DODATKOWE, ZMIENNE_LOG
    )
    
    # =========================================================================
    # 8. MODELE INFERENCYJNE
    # =========================================================================
    model1, wyn1, n1 = model_podstawowy(df_model)
    model2, wyn2, n2, epv_ok = model_rozszerzony(df_model, zmienne)
    model3, wyn3, n3, _ = model_z_redukcja(df_model, zmienne)
    
    if wyn1 is not None:
        wyn1.to_csv('model_podstawowy.csv', sep=';', index=False)
    if wyn2 is not None:
        wyn2.to_csv('model_rozszerzony.csv', sep=';', index=False)
        forest_plot(wyn2, 'forest_plot_rozszerzony.png')
    if wyn3 is not None and wyn3 is not wyn2:
        wyn3.to_csv('model_z_redukcja.csv', sep=';', index=False)
        forest_plot(wyn3, 'forest_plot_redukcja.png')
    
    # =========================================================================
    # 9. PORÓWNANIE MODELI Z AIC
    # =========================================================================
    porownaj_modele_z_aic(model1, model2, model3, n1, n2, n3, epv_ok)
    
    # =========================================================================
    # 10. MODEL PREDYKCYJNY
    # =========================================================================
    pred = model_predykcyjny(df_model, zmienne)
    if pred:
        pred['fig_roc'].savefig('krzywa_ROC.png', dpi=300)
        pred['fig_cal'].savefig('krzywa_kalibracji.png', dpi=300)
        plt.close('all')
    
    # =========================================================================
    # 11. WYKRESY DLA TOP 5
    # =========================================================================
    print("\n" + "="*80)
    print("📈 GENEROWANIE WYKRESÓW")
    print("="*80)
    
    for i, param in enumerate(top5[:5], 1):
        nazwa = f'wykres_{i}_{param}.png'.replace('(', '').replace(')', '').replace(' ', '_')
        wykres_pudelkowy(df_hosp, df_dom, param, nazwa)
    
    # =========================================================================
    # 12. PODSUMOWANIE
    # =========================================================================
    print("\n" + "="*80)
    print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE")
    print("="*80)
    
    print("\n📁 WYGENEROWANE PLIKI:")
    print("  • tabela_1.csv")
    print("  • analiza_jednoczynnikowa.csv")
    print("  • progi_kliniczne.csv")
    if wyn1 is not None:
        print("  • model_podstawowy.csv")
    if wyn2 is not None:
        print("  • model_rozszerzony.csv")
        print("  • forest_plot_rozszerzony.png")
    if wyn3 is not None and wyn3 is not wyn2:
        print("  • model_z_redukcja.csv")
        print("  • forest_plot_redukcja.png")
    if pred is not None:
        print("  • krzywa_ROC.png")
        print("  • krzywa_kalibracji.png")
    for i in range(1, 6):
        print(f"  • wykres_{i}_*.png")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main('BAZA_DANYCH_PACJENTOW_B.csv')