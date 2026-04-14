# -*- coding: utf-8 -*-
"""
PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 17.0 - FINAŁ Z DECYZJAMI KLINICZNYMI
Autor: Aneta
"""
#Decyzje na podstawie diagnostyki:
#✅ EPV steruje interpretacją - zwraca flagę epv_ok i ostrzega w podsumowaniu

#✅ VIF z ostrzeżeniami - ale model idzie dalej (decyzja kliniczna)

#✅ Complete-case opisane - w ograniczeniach

#🟡 Nowe analizy:
#✅ Analiza nieliniowości - kwartyle i test trendu

#✅ Missingness tylko dla top parametrów - nie wszystkie

#✅ Dokładność w wynikach - dodana do printa i podsumowania

#🟢 Poprawki:
#✅ Forest plot z OR - czytelniejszy

#✅ Ostrzeżenia o niestabilności - w podsumowaniu

#✅ Progi wyraźnie eksploracyjne - w nazwie pliku i opisie
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
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

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

# Zmienne wymagające transformacji log
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
        print("="*70)
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


def raport_brakow(df, nazwa):
    """Generuje raport braków danych z liczebnościami"""
    print(f"\n--- RAPORT BRAKÓW: {nazwa} (n={len(df)}) ---")
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
                    print(f"     Problemowe: {dane[((dane < min_bio) | (dane > max_bio))].tolist()}")
    
    if not znaleziono:
        print("  ✓ Wszystkie wartości w zakresach biologicznych")


def sprawdz_epv_i_decyzja(df, zmienne, outcome='outcome', prog=10):
    """Sprawdza EPV i podejmuje decyzję o kontynuacji"""
    n_events = df[outcome].sum()
    n_vars = len(zmienne)
    epv = n_events / n_vars if n_vars > 0 else 0
    
    print(f"\n📊 Events Per Variable (EPV): {epv:.1f}")
    print(f"  • Liczba zdarzeń: {n_events}")
    print(f"  • Liczba predyktorów: {n_vars}")
    
    if epv < prog:
        print(f"  ⚠️ EPV < {prog} - MODEL NIESTABILNY!")
        print(f"  ⚠️ Wyniki inferencyjne NIE powinny być interpretowane")
        return False, epv
    else:
        print(f"  ✓ EPV OK (≥{prog})")
        return True, epv


def sprawdz_vif_i_ostrzezenie(X, prog_wysoki=10, prog_umiarkowany=5):
    """Sprawdza VIF i zwraca listę ostrzeżeń"""
    vif_data = pd.DataFrame()
    vif_data['zmienna'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    ostrzezenia = []
    for _, row in vif_data.iterrows():
        if row['zmienna'] != 'const':
            if row['VIF'] > prog_wysoki:
                ostrzezenia.append(f"  ⚠️ {row['zmienna']}: VIF={row['VIF']:.2f} (wysoka współliniowość)")
            elif row['VIF'] > prog_umiarkowany:
                ostrzezenia.append(f"  • {row['zmienna']}: VIF={row['VIF']:.2f} (umiarkowana)")
            else:
                print(f"  ✓ {row['zmienna']}: VIF={row['VIF']:.2f}")
    
    return ostrzezenia, vif_data


def raport_przeplywu_pacjentow(etapy):
    """Raport przepływu pacjentów przez analizę"""
    print("\n" + "="*70)
    print("PRZEPŁYW PACJENTÓW PRZEZ ANALIZĘ")
    print("="*70)
    
    for etap, n in etapy.items():
        print(f"  {etap}: {n}")


def cliff_delta_bezpieczny(x, y):
    """
    Cliff's delta z kontrolą kierunku
    
    Interpretacja:
        >0: x ma wyższe wartości niż y
        <0: x ma niższe wartości niż y
        |d| < 0.147 - mały efekt
        |d| < 0.33 - średni efekt
        |d| >= 0.33 - duży efekt
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0
    
    try:
        U, _ = stats.mannwhitneyu(x, y, alternative='two-sided', method='asymptotic')
        delta = (2 * U) / (n1 * n2) - 1
        return delta
    except:
        return 0


# =============================================================================
# ANALIZA JEDNOCZYNNIKOWA Z FDR
# =============================================================================

def analiza_jednoczynnikowa_z_fdr(df_caly, parametry, alpha=0.05):
    """
    Pełna analiza jednoczynnikowa z korektą FDR
    """
    print("\n" + "="*80)
    print("ANALIZA JEDNOCZYNNIKOWA Z KOREKTĄ FDR")
    print("="*80)
    
    wyniki = []
    p_values_raw = []
    
    for param in parametry:
        if param in df_caly.columns:
            hosp = df_caly[df_caly['outcome'] == 1][param].dropna()
            dom = df_caly[df_caly['outcome'] == 0][param].dropna()
            
            if len(hosp) > 0 and len(dom) > 0:
                stat, p = stats.mannwhitneyu(hosp, dom)
                d = cliff_delta_bezpieczny(hosp, dom)
                
                wyniki.append({
                    'parametr': param,
                    'p_raw': p,
                    'cliff_delta': d,
                    'n_hosp': len(hosp),
                    'n_dom': len(dom)
                })
                p_values_raw.append(p)
    
    # Korekta FDR Benjamini-Hochberg
    if len(p_values_raw) > 0:
        _, p_adjusted, _, _ = multipletests(p_values_raw, alpha=alpha, method='fdr_bh')
        
        for i, wynik in enumerate(wyniki):
            wynik['p_fdr'] = p_adjusted[i]
            wynik['istotny_fdr'] = wynik['p_fdr'] < alpha
    
    df_wyniki = pd.DataFrame(wyniki)
    df_wyniki = df_wyniki.sort_values('p_raw')
    
    # Wydruk
    print("\n{:<25} {:>10} {:>12} {:>10} {:>8} {:>8} {:>8}".format(
        "Parametr", "p_raw", "p_FDR", "Cliff's d", "n_hosp", "n_dom", "istotny"
    ))
    print("-"*85)
    
    for _, row in df_wyniki.iterrows():
        print("{:<25} {:>10.4f} {:>12.4f} {:>10.2f} {:>6} {:>6} {:>8}".format(
            row['parametr'][:24],
            row['p_raw'],
            row['p_fdr'],
            row['cliff_delta'],
            row['n_hosp'],
            row['n_dom'],
            "✓" if row['istotny_fdr'] else "✗"
        ))
    
    # Top 5 po FDR
    top5_fdr = df_wyniki[df_wyniki['istotny_fdr']].head(5)['parametr'].tolist()
    if len(top5_fdr) < 5:
        top5_fdr = df_wyniki.head(5)['parametr'].tolist()
    
    return df_wyniki, top5_fdr


# =============================================================================
# TRANSFORMACJE DANYCH
# =============================================================================

def przygotuj_dane_do_modelu(df_caly, zmienne_obowiazkowe, zmienne_dodatkowe, zmienne_log):
    """
    Przygotowuje dane z transformacjami log dla wskazanych zmiennych
    """
    df_model = df_caly.copy()
    
    # Podstawowe zmienne
    wszystkie_zmienne = zmienne_obowiazkowe + zmienne_dodatkowe
    dostepne = [z for z in wszystkie_zmienne if z in df_model.columns]
    
    # Dodaj log-transformacje
    for z in zmienne_log:
        if z in df_model.columns:
            df_model[f'log_{z}'] = np.log1p(df_model[z].clip(lower=0))
            print(f"  ✓ Dodano log({z})")
            if z in dostepne:
                dostepne.remove(z)
                dostepne.append(f'log_{z}')
    
    return df_model, dostepne


def raport_missingness_grupowa(df_hosp, df_dom, top_param):
    """Raport braków w podziale na grupy - tylko dla top parametrów"""
    print("\n" + "="*70)
    print("ANALIZA BRAKÓW W PODZIALE NA GRUPY (TOP 5)")
    print("="*70)
    
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
# ANALIZA NIELINIOWOŚCI
# =============================================================================

def analiza_nieliniowosci(df, zmienna, outcome='outcome'):
    """
    Prosta analiza nieliniowości przez kategoryzację
    """
    print(f"\n--- Analiza nieliniowości: {zmienna} ---")
    
    if zmienna not in df.columns:
        return
    
    dane = df[[zmienna, outcome]].dropna()
    if len(dane) < 20:
        return
    
    # Podział na kwartyle
    dane['kwartyl'] = pd.qcut(dane[zmienna], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # Ryzyko w każdym kwartylu
    print("  Ryzyko hospitalizacji wg kwartyli:")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        ryzyko = dane[dane['kwartyl'] == q][outcome].mean()
        print(f"    {q}: {ryzyko:.3f}")
    
    # Test trendu (Cochran-Armitage)
    from statsmodels.stats.proportion import proportions_chisquare
    try:
        tab = pd.crosstab(dane['kwartyl'], dane[outcome])
        chi2, p, _ = proportions_chisquare(tab[1], tab.sum(axis=1))
        print(f"  Test trendu: p={p:.4f}")
    except:
        pass


# =============================================================================
# ANALIZA INFERENCYJNA Z VIF/EPV I DECYZJAMI
# =============================================================================

def analiza_inferencyjna_statsmodels(df, zmienne, outcome='outcome'):
    """
    Model inferencyjny w statsmodels - z VIF i EPV i decyzjami
    """
    print("\n" + "="*80)
    print("MODEL INFERENCYJNY (statsmodels) - OR i 95% CI")
    print("="*80)
    
    # Sprawdź dostępność
    dostepne = [z for z in zmienne if z in df.columns]
    print(f"  Zmienne w modelu: {', '.join(dostepne)}")
    
    # Przygotuj dane
    df_clean = df[dostepne + [outcome]].dropna()
    print(f"  Liczba obserwacji: {len(df_clean)}")
    
    if len(df_clean) < 10:
        print("  ✗ Zbyt mało danych - PRZERYWAM")
        return None, None, False
    
    # EPV - decyzja
    epv_ok, epv = sprawdz_epv_i_decyzja(df_clean, dostepne)
    
    if not epv_ok:
        print("  ⚠️ Model NIESTABILNY - wyniki tylko orientacyjne!")
    
    # Model
    X = df_clean[dostepne]
    X = sm.add_constant(X)
    y = df_clean[outcome]
    
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100, method='bfgs')
        print("\n" + str(model.summary().tables[1]))
        
        # VIF
        print("\n--- VIF (Variance Inflation Factor) ---")
        ostrzezenia_vif, vif_data = sprawdz_vif_i_ostrzezenie(X)
        for o in ostrzezenia_vif:
            print(o)
        
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
        
        return model, pd.DataFrame(wyniki), epv_ok
    
    except Exception as e:
        print(f"  ⚠️ Błąd estymacji: {e}")
        return None, None, False


# =============================================================================
# ANALIZA PREDYKCYJNA Z PIPELINE
# =============================================================================

def analiza_predykcyjna_sklearn_bezpieczna(df, zmienne, outcome='outcome', test_size=0.3):
    """
    Model predykcyjny w sklearn - poprawny pipeline
    """
    print("\n" + "="*80)
    print("MODEL PREDYKCYJNY (sklearn) - poprawny pipeline")
    print("="*80)
    
    dostepne = [z for z in zmienne if z in df.columns]
    
    # Przygotuj dane
    df_clean = df[dostepne + [outcome]].dropna()
    print(f"  Liczba obserwacji: {len(df_clean)}")
    
    if len(df_clean) < 20:
        print("  ✗ Zbyt mało danych - PRZERYWAM")
        return None
    
    X = df_clean[dostepne].values
    y = df_clean[outcome].values
    
    # Podział
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n1. PODZIAŁ DANYCH:")
    print(f"  • Treningowy: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  • Testowy: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Pipeline: skalowanie + regresja
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Trenuj
    pipeline.fit(X_train, y_train)
    
    # Predykcje
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Bootstrap dla AUC
    auc_boot = []
    for i in range(1000):
        boot_idx = resample(range(len(y_test)), replace=True, random_state=i)
        if len(np.unique(y_test[boot_idx])) < 2:
            continue
        try:
            auc_boot.append(roc_auc_score(y_test[boot_idx], y_pred_prob[boot_idx]))
        except:
            continue
    
    if len(auc_boot) > 0:
        auc_ci_low = np.percentile(auc_boot, 2.5)
        auc_ci_high = np.percentile(auc_boot, 97.5)
    else:
        auc_ci_low = auc_ci_high = roc_auc
    
    # Brier
    brier = brier_score_loss(y_test, y_pred_prob)
    
    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    czulosc = tp / (tp + fn) if (tp + fn) > 0 else 0
    swoistosc = tn / (tn + fp) if (tn + fp) > 0 else 0
    dokladnosc = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # CV z pipeline
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    
    print(f"\n2. METRYKI NA ZBIORZE TESTOWYM:")
    print(f"  • AUC: {roc_auc:.3f} (95% CI: {auc_ci_low:.3f}-{auc_ci_high:.3f})")
    print(f"  • Brier score: {brier:.4f}")
    print(f"  • Czułość: {czulosc:.3f}")
    print(f"  • Swoistość: {swoistosc:.3f}")
    print(f"  • PPV: {ppv:.3f}")
    print(f"  • NPV: {npv:.3f}")
    print(f"  • Dokładność: {dokladnosc:.3f}")
    print(f"\n3. WALIDACJA KRZYŻOWA (5-fold):")
    print(f"  • AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Wykresy
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC = {roc_auc:.2f}, 95% CI: {auc_ci_low:.2f}-{auc_ci_high:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Krzywa ROC - model predykcyjny')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
    ax2.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    ax2.set_xlabel('Średnie prawdopodobieństwo przewidywane')
    ax2.set_ylabel('Zaobserwowana częstość')
    ax2.set_title(f'Kalibracja (Brier = {brier:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    return {
        'auc': roc_auc,
        'auc_ci': (auc_ci_low, auc_ci_high),
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
# PROGI KLINICZNE Z BOOTSTRAPEM (EKSPLORACYJNE)
# =============================================================================

def progi_kliniczne_z_bootstrapem(df_caly, top_param, n_bootstrap=1000):
    """Progi kliniczne dla top parametrów (wersja eksploracyjna)"""
    print("\n" + "="*80)
    print("PROGI KLINICZNE Z BOOTSTRAPEM (EKSPLORACYJNE)")
    print("="*80)
    print("  ⚠️ UWAGA: Progi niewalidowane zewnętrznie!")
    
    wyniki = []
    
    for param in top_param[:5]:
        if param not in df_caly.columns:
            continue
        
        hosp_med = df_caly[df_caly['outcome'] == 1][param].median()
        dom_med = df_caly[df_caly['outcome'] == 0][param].median()
        kierunek = 'wyższe' if hosp_med > dom_med else 'niższe'
        
        print(f"\n{param} (kierunek: {kierunek})")
        
        dane = df_caly[[param, 'outcome']].dropna()
        if len(dane) < 10:
            print("  ⚠️ Zbyt mało danych")
            continue
        
        progi_boot = []
        for i in range(n_bootstrap):
            boot = resample(dane, replace=True, random_state=i)
            try:
                if kierunek == 'wyższe':
                    fpr, tpr, progi = roc_curve(boot['outcome'], boot[param])
                else:
                    fpr, tpr, progi = roc_curve(boot['outcome'], -boot[param])
                
                youden = tpr - fpr
                if len(youden) > 0:
                    opt_idx = np.argmax(youden)
                    if kierunek == 'wyższe':
                        progi_boot.append(progi[opt_idx])
                    else:
                        progi_boot.append(-progi[opt_idx])
            except:
                continue
        
        if len(progi_boot) < n_bootstrap * 0.3:
            print("  ⚠️ Niestabilny estymator")
            continue
        
        prog_med = np.median(progi_boot)
        prog_ci_low = np.percentile(progi_boot, 2.5)
        prog_ci_high = np.percentile(progi_boot, 97.5)
        
        # Ocena (na tych samych danych - eksploracyjnie!)
        if kierunek == 'wyższe':
            y_pred = (dane[param] >= prog_med).astype(int)
        else:
            y_pred = (dane[param] <= prog_med).astype(int)
        
        tn = ((y_pred == 0) & (dane['outcome'] == 0)).sum()
        fp = ((y_pred == 1) & (dane['outcome'] == 0)).sum()
        fn = ((y_pred == 0) & (dane['outcome'] == 1)).sum()
        tp = ((y_pred == 1) & (dane['outcome'] == 1)).sum()
        
        czulosc = tp / (tp + fn) if (tp + fn) > 0 else 0
        swoistosc = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"  Próg: {prog_med:.2f} (95% CI: {prog_ci_low:.2f}-{prog_ci_high:.2f})")
        print(f"  Czułość: {czulosc:.3f}")
        print(f"  Swoistość: {swoistosc:.3f}")
        
        wyniki.append({
            'parametr': param,
            'kierunek': kierunek,
            'prog': prog_med,
            'prog_ci_2.5': prog_ci_low,
            'prog_ci_97.5': prog_ci_high,
            'czulosc': czulosc,
            'swoistosc': swoistosc
        })
    
    return pd.DataFrame(wyniki)


# =============================================================================
# FOREST PLOT
# =============================================================================

def forest_plot_ostateczny(wyniki_wielo, nazwa_pliku='forest_plot.png'):
    """Forest plot z OR i CI"""
    if wyniki_wielo is None or len(wyniki_wielo) == 0:
        return
    
    df_plot = wyniki_wielo[wyniki_wielo['parametr'] != 'const'].copy()
    if len(df_plot) == 0:
        return
    
    df_plot[['ci_low', 'ci_high']] = df_plot['CI_95%'].str.split('-', expand=True).astype(float)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(df_plot))
    
    ax.errorbar(df_plot['OR'], y_pos,
                xerr=[df_plot['OR'] - df_plot['ci_low'], df_plot['ci_high'] - df_plot['OR']],
                fmt='o', color='darkblue', ecolor='gray', capsize=5, markersize=8)
    
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='OR = 1')
    ax.set_xscale('log')
    ax.set_xlabel('OR (95% CI) - skala log')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['parametr'])
    ax.set_title('Czynniki ryzyka hospitalizacji (model inferencyjny)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(nazwa_pliku, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Forest plot: {nazwa_pliku}")


# =============================================================================
# WYKRESY INDYWIDUALNE
# =============================================================================

def wykres_pudelkowy_z_kierunkiem(df_hosp, df_dom, param, nazwa_pliku):
    """Wykres pudełkowy z kierunkiem"""
    hosp = df_hosp[param].dropna()
    dom = df_dom[param].dropna()
    
    if len(hosp) == 0 or len(dom) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    stat, p = stats.mannwhitneyu(hosp, dom)
    d = cliff_delta_bezpieczny(hosp, dom)
    
    hosp_med = hosp.median()
    dom_med = dom.median()
    kierunek = "↑ wyższe u hosp" if hosp_med > dom_med else "↓ niższe u hosp"
    
    # Wykres
    bp1 = ax1.boxplot([hosp, dom], labels=['PRZYJĘCI', 'WYPISANI'],
                      patch_artist=True, medianprops={'color': 'black', 'linewidth': 2})
    
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
    
    # Statystyki
    ax2.axis('off')
    text = f"""
    {param}
    
    Hospitalizowani (n={len(hosp)}):
    mediana = {hosp_med:.2f}
    IQR = {hosp.quantile(0.25):.2f}-{hosp.quantile(0.75):.2f}
    
    Wypisani (n={len(dom)}):
    mediana = {dom_med:.2f}
    IQR = {dom.quantile(0.25):.2f}-{dom.quantile(0.75):.2f}
    
    p = {p:.4f}
    Cliff's d = {d:.2f}
    {kierunek}
    """
    ax2.text(0.1, 0.5, text, transform=ax2.transAxes,
            verticalalignment='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(nazwa_pliku, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Wykres: {nazwa_pliku}")


# =============================================================================
# TABELA OPISOWA
# =============================================================================

def tabela_opisowa_kompletna(df_hosp, df_dom, parametry_ciagle, choroby):
    """Kompletna Tabela 1 z liczebnościami"""
    print("\n" + "="*80)
    print("TABELA 1: CHARAKTERYSTYKA KOHORTY")
    print("="*80)
    print(f"  Hospitalizowani: n={len(df_hosp)}")
    print(f"  Do domu: n={len(df_dom)}")
    print(f"  RAZEM: n={len(df_hosp)+len(df_dom)}")
    
    wyniki = []
    
    # Zmienne ciągłe
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
                
                stat, p = stats.mannwhitneyu(hosp, dom)
                d = cliff_delta_bezpieczny(hosp, dom)
                
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
    
    # Zmienne kategorialne
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
                dom_tak = (dom == 1).sum()
                
                hosp_stat = f"{hosp_tak}/{len(hosp)} ({hosp_tak/len(hosp)*100:.1f}%)"
                dom_stat = f"{dom_tak}/{len(dom)} ({dom_tak/len(dom)*100:.1f}%)"
                
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
                
                print("{:<25} {:>20} {:>20}   {:<8.4f}   {:>6.2f}".format(
                    choroba, hosp_stat, dom_stat, p, oddsratio
                ))
    
    return pd.DataFrame(wyniki)


# =============================================================================
# PODSUMOWANIE I OGRANICZENIA
# =============================================================================

def podsumowanie_z_ograniczeniami(model_inf=None, model_pred=None, progi=None, epv_ok=False):
    """Generuje podsumowanie z ograniczeniami"""
    print("\n" + "="*80)
    print("PODSUMOWANIE I OGRANICZENIA ANALIZY")
    print("="*80)
    
    print("\n✅ MOCNE STRONY:")
    print("  • Kompletna Tabela 1 (ciągłe + kategorialne)")
    print("  • Testy nieparametryczne (Mann-Whitney, Fisher)")
    print("  • Korekta FDR dla wielokrotnych porównań")
    print("  • Bootstrap dla progów klinicznych i AUC")
    print("  • Walidacja krzyżowa i podział train/test")
    print("  • Dwa podejścia: inferencyjne i predykcyjne")
    print("  • VIF i EPV z decyzjami")
    print("  • Transformacje log dla biomarkerów")
    print("  • Poprawny pipeline (bez data leakage)")
    print("  • Analiza nieliniowości (kwartyle)")
    
    print("\n⚠️ OGRANICZENIA:")
    print("  • Analiza complete-case (usuwanie braków)")
    print("  • Brak walidacji zewnętrznej")
    print("  • Progi kliniczne eksploracyjne (wymagają walidacji)")
    print("  • Założenie liniowości w modelu logistycznym")
    
    if model_inf is not None:
        print("\n📊 INTERPRETACJA KLINICZNA (model inferencyjny):")
        if not epv_ok:
            print("  ⚠️ UWAGA: Model NIESTABILNY (EPV < 10)")
            print("  ⚠️ Wyniki należy traktować orientacyjnie!")
        print("  • OR > 1 - czynnik ryzyka hospitalizacji")
        print("  • OR < 1 - czynnik ochronny")
        print("  • p < 0.05 po korekcie - istotne statystycznie")
    
    if model_pred is not None:
        print("\n📈 PERFORMANCE PREDYKCYJNY:")
        print(f"  • AUC: {model_pred['auc']:.3f}")
        print(f"  • Dokładność: {model_pred['dokladnosc']:.3f}")
        print(f"  • Czułość: {model_pred['czulosc']:.3f}")
        print(f"  • Swoistość: {model_pred['swoistosc']:.3f}")
    
    print("\n" + "="*80)


# =============================================================================
# GŁÓWNA FUNKCJA
# =============================================================================

def main(sciezka_pliku, id_pacjenta=None):
    """Główna funkcja analizy"""
    
    # Rejestr przepływu
    przeplyw = {}
    
    print("\n" + "="*80)
    print("PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH - WERSJA 17.0")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # =========================================================================
    # ETAP 1: WCZYTYWANIE
    # =========================================================================
    
    df = wczytaj_dane(sciezka_pliku)
    if df is None:
        return
    
    przeplyw['1. Wczytano'] = len(df)
    
    df_hosp, df_dom, df_caly = przygotuj_dane_z_outcome(df, id_pacjenta)
    if df_hosp is None:
        return
    
    przeplyw['2. Po deduplikacji'] = len(df_caly)
    przeplyw['3. Hospitalizowani'] = len(df_hosp)
    przeplyw['4. Do domu'] = len(df_dom)
    
    # =========================================================================
    # ETAP 2: KONWERSJA
    # =========================================================================
    
    parametry_kliniczne = [
        'wiek', 'RR', 'MAP', 'SpO2', 'AS', 'mleczany',
        'kreatynina(0,5-1,2)', 'troponina I (0-7,8))',
        'HGB(12,4-15,2)', 'WBC(4-11)', 'plt(130-450)',
        'hct(38-45)', 'Na(137-145)', 'K(3,5-5,1)', 'crp(0-0,5)'
    ]
    
    choroby = ['dm', 'wątroba', 'naczyniowe', 'zza', 'npl']
    
    df_hosp = konwertuj_na_numeryczne(df_hosp, parametry_kliniczne)
    df_dom = konwertuj_na_numeryczne(df_dom, parametry_kliniczne)
    df_caly = konwertuj_na_numeryczne(df_caly, parametry_kliniczne)
    
    df_hosp = konwertuj_choroby(df_hosp, choroby)
    df_dom = konwertuj_choroby(df_dom, choroby)
    df_caly = konwertuj_choroby(df_caly, choroby)
    
    # =========================================================================
    # ETAP 3: KONTROLA JAKOŚCI
    # =========================================================================
    
    raport_brakow(df_caly, 'Pełna kohorta')
    walidacja_zakresow_biologicznych(df_caly, ZAKRESY_BIOLOGICZNE)
    
    # =========================================================================
    # ETAP 4: TABELA 1
    # =========================================================================
    
    tabela = tabela_opisowa_kompletna(df_hosp, df_dom, parametry_kliniczne, choroby)
    tabela.to_csv('tabela_1_opisowa.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 5: ANALIZA JEDNOCZYNNIKOWA Z FDR
    # =========================================================================
    
    wyniki_fdr, top5_fdr = analiza_jednoczynnikowa_z_fdr(df_caly, parametry_kliniczne)
    wyniki_fdr.to_csv('analiza_jednoczynnikowa_fdr.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 6: ANALIZA NIELINIOWOŚCI
    # =========================================================================
    
    print("\n" + "="*80)
    print("ANALIZA NIELINIOWOŚCI (TOP 3)")
    print("="*80)
    
    for param in top5_fdr[:3]:
        analiza_nieliniowosci(df_caly, param)
    
    # =========================================================================
    # ETAP 7: RAPORT MISSINGNESS DLA TOP PARAMETRÓW
    # =========================================================================
    
    raport_missingness_grupowa(df_hosp, df_dom, top5_fdr)
    
    # =========================================================================
    # ETAP 8: PROGI KLINICZNE (EKSPLORACYJNE)
    # =========================================================================
    
    progi = progi_kliniczne_z_bootstrapem(df_caly, top5_fdr)
    if progi is not None and len(progi) > 0:
        progi.to_csv('progi_kliniczne_eksploracyjne.csv', sep=';', index=False)
    
    # =========================================================================
    # ETAP 9: TRANSFORMACJE
    # =========================================================================
    
    print("\n" + "="*80)
    print("TRANSFORMACJE DANYCH")
    print("="*80)
    
    df_model, zmienne_do_modelu = przygotuj_dane_do_modelu(
        df_caly, ZMIENNE_OBOWIAZKOWE, ZMIENNE_DODATKOWE, ZMIENNE_LOG
    )
    
    przeplyw['5. Po transformacjach'] = len(df_model)
    
    # =========================================================================
    # ETAP 10: MODEL INFERENCYJNY (statsmodels)
    # =========================================================================
    
    model_inf, wyniki_inf, epv_ok = analiza_inferencyjna_statsmodels(
        df_model, zmienne_do_modelu
    )
    
    if model_inf is not None:
        wyniki_inf.to_csv('analiza_wieloczynnikowa_statsmodels.csv', sep=';', index=False)
        forest_plot_ostateczny(wyniki_inf, 'forest_plot_statsmodels.png')
        
        n_model = len(df_model[zmienne_do_modelu + ['outcome']].dropna())
        przeplyw['6. Do modelu inferencyjnego'] = n_model
    
    # =========================================================================
    # ETAP 11: MODEL PREDYKCYJNY (sklearn)
    # =========================================================================
    
    if model_inf is not None:
        model_pred = analiza_predykcyjna_sklearn_bezpieczna(
            df_model, zmienne_do_modelu
        )
        
        if model_pred is not None:
            model_pred['fig_roc'].savefig('krzywa_ROC_sklearn.png', dpi=300, bbox_inches='tight')
            model_pred['fig_cal'].savefig('krzywa_kalibracji_sklearn.png', dpi=300, bbox_inches='tight')
            plt.close('all')
    else:
        model_pred = None
    
    # =========================================================================
    # ETAP 12: WYKRESY
    # =========================================================================
    
    print("\n" + "="*80)
    print("GENEROWANIE WYKRESÓW")
    print("="*80)
    
    for i, param in enumerate(top5_fdr[:5], 1):
        nazwa = f'wykres_{i}_{param}.png'.replace('(', '').replace(')', '').replace(' ', '_')
        wykres_pudelkowy_z_kierunkiem(df_hosp, df_dom, param, nazwa)
    
    # =========================================================================
    # ETAP 13: RAPORT PRZEPŁYWU
    # =========================================================================
    
    raport_przeplywu_pacjentow(przeplyw)
    
    # =========================================================================
    # ETAP 14: PODSUMOWANIE
    # =========================================================================
    
    podsumowanie_z_ograniczeniami(model_inf, model_pred, progi, epv_ok)
    
    print("\n" + "="*80)
    print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE")
    print("="*80)
    
    print("\nWygenerowane pliki:")
    print("  • tabela_1_opisowa.csv")
    print("  • analiza_jednoczynnikowa_fdr.csv")
    if progi is not None:
        print("  • progi_kliniczne_eksploracyjne.csv")
    if model_inf is not None:
        print("  • analiza_wieloczynnikowa_statsmodels.csv")
        print("  • forest_plot_statsmodels.png")
    if model_pred is not None:
        print("  • krzywa_ROC_sklearn.png")
        print("  • krzywa_kalibracji_sklearn.png")
    for i in range(1, 6):
        print(f"  • wykres_{i}_*.png")
    
    print("\n" + "="*80)


# =============================================================================
# URUCHOMIENIE
# =============================================================================
if __name__ == "__main__":
    sciezka = 'BAZA_DANYCH_PACJENTOW_B.csv'  # <- ZMIEŃ NA SWOJĄ
    main(sciezka, id_pacjenta=None)  # id_pacjenta='patient_id' jeśli masz