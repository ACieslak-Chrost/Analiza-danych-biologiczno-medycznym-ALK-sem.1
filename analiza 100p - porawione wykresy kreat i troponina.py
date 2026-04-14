
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 18:46:12 2026

@author: aneta
"""

# -*- coding: utf-8 -*-
"""
PEŁNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 7.0 - wszystkie wykresy z poprzednich wersji
Autor: Aneta
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Wyłącza okna, tylko zapis
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import pi
import sqlite3
import os
import warnings
warnings.filterwarnings('ignore')

# Ustawienia wykresów
plt.style.use('ggplot')
sns.set_palette("Set2")
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

print("="*80)
print("PEŁNA ANALIZA DANYCH PACJENTÓW - WSZYSTKIE WYKRESY")
print("Wersja 7.0 - przywrócone wszystkie wykresy")
print("="*80)

# =============================================================================
# 1. WCZYTYWANIE I PRZYGOTOWANIE DANYCH
# =============================================================================

print("\n1. WCZYTYWANIE DANYCH...")

# Wczytaj dane
df = pd.read_csv('BAZA_DANYCH_PACJENTOW_B.csv', sep=';', encoding='utf-8')

# Znajdź podział
puste_wiersze = df[df.isna().all(axis=1)]
if len(puste_wiersze) > 0:
    indeks_podzialu = puste_wiersze.index[0]
    df_hosp = df.iloc[:indeks_podzialu].copy().dropna(how='all')
    df_dom = df.iloc[indeks_podzialu+1:].copy().dropna(how='all')
    print(f"✓ Hospitalizowani: {len(df_hosp)} pacjentów")
    print(f"✓ Do domu: {len(df_dom)} pacjentów")
else:
    print("✗ Nie znaleziono podziału")
    exit()

# =============================================================================
# 2. FUNKCJE POMOCNICZE
# =============================================================================

def convert_to_numeric(df, columns):
    """Konwertuje kolumny na typ numeryczny"""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    return df

def convert_choroba(wartosc):
    """Konwertuje różne formy zapisu chorób"""
    if pd.isna(wartosc):
        return np.nan
    if isinstance(wartosc, str):
        wartosc = wartosc.lower().strip()
        if wartosc in ['tak', 't', 'yes', 'y', '1', 'true', '+', 'tak!']:
            return 1
        elif wartosc in ['nie', 'n', 'no', '0', 'false', '-']:
            return 0
    return np.nan

# Listy parametrów
parametry_kliniczne = [
    'wiek', 'RR', 'MAP', 'SpO2', 'AS', 'mleczany',
    'kreatynina(0,5-1,2)', 'troponina I (0-7,8))',
    'HGB(12,4-15,2)', 'WBC(4-11)', 'plt(130-450)',
    'hct(38-45)', 'Na(137-145)', 'K(3,5-5,1)', 'crp(0-0,5)'
]

choroby = ['dm', 'wątroba', 'naczyniowe', 'zza', 'npl']

# Konwersja danych
df_hosp = convert_to_numeric(df_hosp, parametry_kliniczne)
df_dom = convert_to_numeric(df_dom, parametry_kliniczne)

# =============================================================================
# 3. ANALIZA STATYSTYCZNA - TABELA
# =============================================================================

print("\n" + "="*80)
print("2. ANALIZA STATYSTYCZNA - TABELA WYNIKÓW")
print("="*80)

wyniki = []
print("\n{:<25} {:>12} {:>12} {:>12} {:>10}".format(
    "Parametr", "Hosp (n)", "Hosp (śr±SD)", "Dom (śr±SD)", "p-value"
))
print("-"*75)

for param in parametry_kliniczne:
    if param in df_hosp.columns:
        hosp = df_hosp[param].dropna()
        dom = df_dom[param].dropna()
        
        if len(hosp) > 0 and len(dom) > 0:
            hosp_sr = hosp.mean()
            hosp_std = hosp.std()
            dom_sr = dom.mean()
            dom_std = dom.std()
            
            # Test Manna-Whitneya
            stat, p_value = stats.mannwhitneyu(hosp, dom, alternative='two-sided')
            
            # Określenie istotności
            if p_value < 0.001:
                gwiazdki = "***"
            elif p_value < 0.01:
                gwiazdki = "**"
            elif p_value < 0.05:
                gwiazdki = "*"
            else:
                gwiazdki = "ns"
            
            print("{:<25} {:>5}   {:>6.2f}±{:<5.2f} {:>6.2f}±{:<5.2f}  p={:<.4f} {}".format(
                param[:24], len(hosp), hosp_sr, hosp_std, dom_sr, dom_std, p_value, gwiazdki
            ))
            
            wyniki.append({
                'parametr': param, 
                'hosp_n': len(hosp), 
                'hosp_sr': hosp_sr, 
                'hosp_std': hosp_std,
                'dom_n': len(dom), 
                'dom_sr': dom_sr, 
                'dom_std': dom_std,
                'roznica': hosp_sr - dom_sr,
                'p_value': p_value, 
                'istotnosc': gwiazdki
            })

df_wyniki = pd.DataFrame(wyniki)
istotne = df_wyniki[df_wyniki['p_value'] < 0.05].sort_values('p_value')
parametry_istotne = istotne['parametr'].tolist()

print(f"\n✓ Znaleziono {len(parametry_istotne)} parametrów z istotnymi różnicami")

# =============================================================================
# 3. GENEROWANIE WYKRESÓW...
# =============================================================================

print("\n3. GENEROWANIE WYKRESÓW...")

# =============================================================================
# WYKRES DLA TROPONINY (z wartościami ekstremalnymi)
# =============================================================================

if 'troponina I (0-7,8))' in df_hosp.columns:
    print("\n✓ Wykres: Troponina I (skala logarytmiczna)")
    
    hosp_trop = df_hosp['troponina I (0-7,8))'].dropna()
    dom_trop = df_dom['troponina I (0-7,8))'].dropna()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Wykres 1: Skala logarytmiczna
    axes[0].boxplot([hosp_trop, dom_trop], labels=['Przyjęci', 'Wypisani'])
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Troponina (skala log)')
    axes[0].set_title('A) Troponina - skala logarytmiczna')
    axes[0].grid(True, alpha=0.3)
    
    # Wykres 2: Bez wartości odstających (tylko do 100)
    hosp_trop_bez = hosp_trop[hosp_trop < 100]
    dom_trop_bez = dom_trop[dom_trop < 100]
    
    axes[1].boxplot([hosp_trop_bez, dom_trop_bez], labels=['Przyjęci', 'Wypisani'])
    axes[1].set_ylabel('Troponina (wartości <100)')
    axes[1].set_title('B) Troponina - bez wartości ekstremalnych')
    axes[1].grid(True, alpha=0.3)
    
    # Wykres 3: Statystyki
    stat, p = stats.mannwhitneyu(hosp_trop, dom_trop, alternative='two-sided')
    
    tekst = f"p-value: {p:.6f}\n"
    tekst += f"Przyjęci: n={len(hosp_trop)}, mediana={np.median(hosp_trop):.2f}\n"
    tekst += f"Wypisani: n={len(dom_trop)}, mediana={np.median(dom_trop):.2f}\n"
    tekst += f"Max przyjęci: {hosp_trop.max():.2f}\n"
    tekst += f"Max wypisani: {dom_trop.max():.2f}"
    
    axes[2].text(0.1, 0.5, tekst, fontsize=12, transform=axes[2].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[2].axis('off')
    axes[2].set_title('C) Podsumowanie')
    
    plt.suptitle('TROPONINA I - ANALIZA SZCZEGÓŁOWA', fontweight='bold')
    plt.tight_layout()
    plt.savefig('wykres_troponina_szczegolowa.png', dpi=150, bbox_inches='tight')
    # USUNIĘTO plt.show()

# =============================================================================
# WYKRES DLA KREATYNINY
# =============================================================================

if 'kreatynina(0,5-1,2)' in df_hosp.columns:
    print("\n✓ Wykres: Kreatynina")
    
    hosp_kreat = df_hosp['kreatynina(0,5-1,2)'].dropna()
    dom_kreat = df_dom['kreatynina(0,5-1,2)'].dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Wykres pudełkowy
    bp = axes[0].boxplot([hosp_kreat, dom_kreat], 
                        labels=['Przyjęci', 'Wypisani'],
                        patch_artist=True)
    
    bp['boxes'][0].set_facecolor('#ff9999')
    bp['boxes'][1].set_facecolor('#99ccff')
    
    # Dodanie punktów
    x_hosp = np.random.normal(1, 0.05, len(hosp_kreat))
    x_dom = np.random.normal(2, 0.05, len(dom_kreat))
    axes[0].scatter(x_hosp, hosp_kreat, alpha=0.6, color='darkred', s=40)
    axes[0].scatter(x_dom, dom_kreat, alpha=0.6, color='darkblue', s=40)
    
    axes[0].axhline(y=1.2, color='red', linestyle='--', alpha=0.5, label='Górna norma')
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Dolna norma')
    axes[0].set_ylabel('Kreatynina (mg/dL)')
    axes[0].set_title('A) Rozkład kreatyniny')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(hosp_kreat, bins=15, alpha=0.7, color='red', label='Przyjęci', density=True)
    axes[1].hist(dom_kreat, bins=15, alpha=0.7, color='blue', label='Wypisani', density=True)
    axes[1].axvline(x=1.2, color='red', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Kreatynina (mg/dL)')
    axes[1].set_ylabel('Gęstość')
    axes[1].set_title('B) Rozkład - porównanie')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Test statystyczny
    stat, p = stats.mannwhitneyu(hosp_kreat, dom_kreat, alternative='two-sided')
    plt.suptitle(f'KREATYNINA - porównanie grup (p={p:.4f})', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('wykres_kreatynina.png', dpi=150, bbox_inches='tight')
    # USUNIĘTO plt.show()

# =============================================================================
# 4. WYKRES 1: PARAMETRY ISTOTNE - WYKRESY PUDEŁKOWE
# =============================================================================

print("\n✓ Wykres 1: Parametry z istotnymi różnicami")

if len(parametry_istotne) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(parametry_istotne[:6]):
        if i < 6 and param in df_hosp.columns:
            dane_hosp = df_hosp[param].dropna()
            dane_dom = df_dom[param].dropna()
            
            dane = pd.DataFrame({
                'wartosc': pd.concat([dane_hosp, dane_dom]),
                'grupa': ['Hospitalizowani']*len(dane_hosp) + ['Do domu']*len(dane_dom)
            })
            
            p_val = istotne[istotne['parametr'] == param]['p_value'].values[0]
            
            sns.boxplot(data=dane, x='grupa', y='wartosc', ax=axes[i], 
                       palette=['#ff6b6b', '#4ecdc4'])
            axes[i].set_title(f'{param}\np={p_val:.4f}', fontweight='bold')
            axes[i].set_xlabel('')
            sns.stripplot(data=dane, x='grupa', y='wartosc', ax=axes[i], 
                         color='black', alpha=0.5, size=3)
    
    for j in range(i+1, 6):
        axes[j].set_visible(False)
    
    plt.suptitle('RYSUNEK 1: PARAMETRY Z ISTOTNYMI RÓŻNICAMI', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('wykres1_parametry_istotne.png', dpi=150, bbox_inches='tight')
    # USUNIĘTO plt.show()

# =============================================================================
# 5. WYKRES 2: MAP - SZCZEGÓŁOWA ANALIZA
# =============================================================================

print("\n✓ Wykres 2: Szczegółowa analiza MAP")

if 'MAP' in df_hosp.columns:
    hosp_map = df_hosp['MAP'].dropna()
    dom_map = df_dom['MAP'].dropna()
    
    stat, p_value_map = stats.mannwhitneyu(hosp_map, dom_map, alternative='two-sided')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Wykres pudełkowy
    dane_map = pd.DataFrame({
        'MAP': pd.concat([hosp_map, dom_map]),
        'Grupa': ['Hospitalizowani']*len(hosp_map) + ['Do domu']*len(dom_map)
    })
    
    sns.boxplot(data=dane_map, x='Grupa', y='MAP', ax=axes[0], 
                palette=['#ff6b6b', '#4ecdc4'])
    axes[0].set_title(f'MAP - porównanie grup\np={p_value_map:.4f}', fontweight='bold')
    axes[0].set_ylabel('MAP (mmHg)')
    sns.stripplot(data=dane_map, x='Grupa', y='MAP', ax=axes[0], 
                  color='black', alpha=0.5, size=4)
    
    # Histogram
    axes[1].hist(hosp_map, bins=15, alpha=0.7, color='red', label='Hospitalizowani', density=True)
    axes[1].hist(dom_map, bins=15, alpha=0.7, color='blue', label='Do domu', density=True)
    axes[1].set_xlabel('MAP (mmHg)')
    axes[1].set_ylabel('Gęstość')
    axes[1].set_title('Rozkład MAP')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # MAP vs Wiek
    hosp_idx = hosp_map.dropna().index
    dom_idx = dom_map.dropna().index
    
    hosp_wiek = df_hosp.loc[hosp_idx, 'wiek'].dropna()
    hosp_map_clean = hosp_map.loc[hosp_wiek.index]
    
    dom_wiek = df_dom.loc[dom_idx, 'wiek'].dropna()
    dom_map_clean = dom_map.loc[dom_wiek.index]
    
    axes[2].scatter(hosp_wiek, hosp_map_clean, color='red', alpha=0.6, s=60, label='Hospitalizowani')
    axes[2].scatter(dom_wiek, dom_map_clean, color='blue', alpha=0.6, s=60, label='Do domu')
    axes[2].set_xlabel('Wiek (lata)')
    axes[2].set_ylabel('MAP (mmHg)')
    axes[2].set_title('MAP vs Wiek')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('RYSUNEK 2: SZCZEGÓŁOWA ANALIZA MAP', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('wykres2_MAP_analiza.png', dpi=150, bbox_inches='tight')
    # USUNIĘTO plt.show()

# =============================================================================
# 6. WYKRES 3: MACIERZ KORELACJI (pierwsza wersja)
# =============================================================================

print("\n✓ Wykres 3: Macierz korelacji (podstawowa)")

parametry_korelacji1 = ['wiek', 'SpO2', 'HGB(12,4-15,2)', 'crp(0-0,5)', 
                       'kreatynina(0,5-1,2)', 'WBC(4-11)']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Korelacja - hospitalizowani
df_hosp_corr = df_hosp[parametry_korelacji1].dropna()
if len(df_hosp_corr) > 0:
    corr_hosp = df_hosp_corr.corr()
    sns.heatmap(corr_hosp, annot=True, cmap='RdBu_r', center=0, 
                square=True, linewidths=1, ax=axes[0], cbar=False,
                vmin=-1, vmax=1, fmt='.2f')
    axes[0].set_title('A) HOSPITALIZOWANI', fontweight='bold')

# Korelacja - do domu
df_dom_corr = df_dom[parametry_korelacji1].dropna()
if len(df_dom_corr) > 0:
    corr_dom = df_dom_corr.corr()
    sns.heatmap(corr_dom, annot=True, cmap='RdBu_r', center=0, 
                square=True, linewidths=1, ax=axes[1], cbar=True,
                vmin=-1, vmax=1, fmt='.2f')
    axes[1].set_title('B) DO DOMU', fontweight='bold')

plt.suptitle('RYSUNEK 3: MACIERZ KORELACJI - PODSTAWOWA', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres3_macierz_korelacji_podstawowa.png', dpi=150, bbox_inches='tight')
# USUNIĘTO plt.show()

# =============================================================================
# 7. WYKRES 4: WYKRESY ROZRZTU (pierwsza wersja)
# =============================================================================

print("\n✓ Wykres 4: Wykresy rozrzutu (podstawowe)")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Wykres rozrzutu - wiek vs SpO2
axes[0].scatter(df_hosp['wiek'], df_hosp['SpO2'], c='red', alpha=0.6, s=80, label='Hospitalizowani')
axes[0].scatter(df_dom['wiek'], df_dom['SpO2'], c='blue', alpha=0.6, s=80, label='Do domu')
axes[0].set_xlabel('Wiek (lata)')
axes[0].set_ylabel('SpO2 (%)')
axes[0].set_title('A) Wiek vs Saturacja')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Wykres rozrzutu - HGB vs CRP
axes[1].scatter(df_hosp['HGB(12,4-15,2)'], df_hosp['crp(0-0,5)'], c='red', alpha=0.6, s=80, label='Hospitalizowani')
axes[1].scatter(df_dom['HGB(12,4-15,2)'], df_dom['crp(0-0,5)'], c='blue', alpha=0.6, s=80, label='Do domu')
axes[1].set_xlabel('Hemoglobina (HGB)')
axes[1].set_ylabel('CRP (stan zapalny)')
axes[1].set_title('B) Hemoglobina vs CRP')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')  # Skala logarytmiczna dla CRP

plt.suptitle('RYSUNEK 4: WYKRESY ROZRZTU - PODSTAWOWE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres4_wykresy_rozrzutu_podstawowe.png', dpi=150, bbox_inches='tight')
# USUNIĘTO plt.show()

# =============================================================================
# 8. WYKRES 5: ROZKŁADY PARAMETRÓW (pierwsza wersja)
# =============================================================================

print("\n✓ Wykres 5: Rozkłady parametrów (podstawowe)")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

parametry_rozklady1 = ['SpO2', 'HGB(12,4-15,2)', 'crp(0-0,5)', 'wiek', 'kreatynina(0,5-1,2)', 'WBC(4-11)']

for i, param in enumerate(parametry_rozklady1):
    if i < 6 and param in df_hosp.columns:
        # Histogram i KDE
        axes[i].hist(df_hosp[param].dropna(), bins=15, alpha=0.5, color='red', density=True, label='Hosp')
        axes[i].hist(df_dom[param].dropna(), bins=15, alpha=0.5, color='blue', density=True, label='Dom')
        
        sns.kdeplot(data=df_hosp[param].dropna(), ax=axes[i], color='darkred', linewidth=2)
        sns.kdeplot(data=df_dom[param].dropna(), ax=axes[i], color='darkblue', linewidth=2)
        
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Gęstość')
        axes[i].set_title(f'Rozkład: {param}')
        axes[i].legend()

plt.suptitle('RYSUNEK 5: ROZKŁADY PARAMETRÓW - PODSTAWOWE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres5_rozklady_podstawowe.png', dpi=150, bbox_inches='tight')
# USUNIĘTO plt.show()

# =============================================================================
# 9. WYKRES 6: MACIERZ KORELACJI (z MAP)
# =============================================================================

print("\n✓ Wykres 6: Macierz korelacji (z MAP)")

parametry_korelacji2 = ['wiek', 'MAP', 'SpO2', 'HGB(12,4-15,2)', 'crp(0-0,5)', 'kreatynina(0,5-1,2)']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Korelacja - hospitalizowani
df_hosp_corr2 = df_hosp[parametry_korelacji2].dropna()
if len(df_hosp_corr2) > 0:
    corr_hosp = df_hosp_corr2.corr()
    sns.heatmap(corr_hosp, annot=True, cmap='RdBu_r', center=0, 
                square=True, linewidths=1, ax=axes[0], cbar=False,
                vmin=-1, vmax=1, fmt='.2f')
    axes[0].set_title('A) HOSPITALIZOWANI', fontweight='bold')

# Korelacja - do domu
df_dom_corr2 = df_dom[parametry_korelacji2].dropna()
if len(df_dom_corr2) > 0:
    corr_dom = df_dom_corr2.corr()
    sns.heatmap(corr_dom, annot=True, cmap='RdBu_r', center=0, 
                square=True, linewidths=1, ax=axes[1], cbar=True,
                vmin=-1, vmax=1, fmt='.2f')
    axes[1].set_title('B) DO DOMU', fontweight='bold')

plt.suptitle('RYSUNEK 6: MACIERZ KORELACJI (z MAP)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres6_macierz_korelacji_z_MAP.png', dpi=150, bbox_inches='tight')
# USUNIĘTO plt.show()

# =============================================================================
# 10. WYKRES 7: WYKRESY ROZRZTU (z MAP)
# =============================================================================

print("\n✓ Wykres 7: Wykresy rozrzutu (z MAP)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Wykres 1: Wiek vs SpO2
axes[0].scatter(df_hosp['wiek'], df_hosp['SpO2'], c='red', alpha=0.6, s=60, label='Hosp')
axes[0].scatter(df_dom['wiek'], df_dom['SpO2'], c='blue', alpha=0.6, s=60, label='Dom')
axes[0].set_xlabel('Wiek (lata)')
axes[0].set_ylabel('SpO2 (%)')
axes[0].set_title('A) Wiek vs Saturacja')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Wykres 2: HGB vs CRP
axes[1].scatter(df_hosp['HGB(12,4-15,2)'], df_hosp['crp(0-0,5)'], c='red', alpha=0.6, s=60, label='Hosp')
axes[1].scatter(df_dom['HGB(12,4-15,2)'], df_dom['crp(0-0,5)'], c='blue', alpha=0.6, s=60, label='Dom')
axes[1].set_xlabel('Hemoglobina (HGB)')
axes[1].set_ylabel('CRP')
axes[1].set_title('B) HGB vs CRP')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Wykres 3: MAP vs Wiek
axes[2].scatter(df_hosp['wiek'], df_hosp['MAP'], c='red', alpha=0.6, s=60, label='Hosp')
axes[2].scatter(df_dom['wiek'], df_dom['MAP'], c='blue', alpha=0.6, s=60, label='Dom')
axes[2].set_xlabel('Wiek (lata)')
axes[2].set_ylabel('MAP (mmHg)')
axes[2].set_title('C) MAP vs Wiek')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('RYSUNEK 7: WYKRESY ROZRZTU (z MAP)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres7_wykresy_rozrzutu_z_MAP.png', dpi=150, bbox_inches='tight')
# USUNIĘTO plt.show()

# =============================================================================
# 11. WYKRES 8: ROZKŁADY PARAMETRÓW (z MAP)
# =============================================================================

print("\n✓ Wykres 8: Rozkłady parametrów (z MAP)")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

parametry_rozklady2 = ['MAP', 'SpO2', 'HGB(12,4-15,2)', 'crp(0-0,5)', 'wiek', 'kreatynina(0,5-1,2)']

for i, param in enumerate(parametry_rozklady2):
    if i < 6 and param in df_hosp.columns:
        axes[i].hist(df_hosp[param].dropna(), bins=15, alpha=0.5, color='red', density=True, label='Hosp')
        axes[i].hist(df_dom[param].dropna(), bins=15, alpha=0.5, color='blue', density=True, label='Dom')
        
        sns.kdeplot(data=df_hosp[param].dropna(), ax=axes[i], color='darkred', linewidth=2)
        sns.kdeplot(data=df_dom[param].dropna(), ax=axes[i], color='darkblue', linewidth=2)
        
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Gęstość')
        axes[i].set_title(f'Rozkład: {param}')
        axes[i].legend()

plt.suptitle('RYSUNEK 8: ROZKŁADY PARAMETRÓW (z MAP)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres8_rozklady_z_MAP.png', dpi=150, bbox_inches='tight')
# USUNIĘTO plt.show()

# =============================================================================
# 12. WYKRES 9: CHOROBY WSPÓŁISTNIEJĄCE
# =============================================================================

print("\n✓ Wykres 9: Choroby współistniejące")

wyniki_choroby = []
for choroba in choroby:
    if choroba in df_hosp.columns:
        hosp_val = df_hosp[choroba].apply(convert_choroba)
        dom_val = df_dom[choroba].apply(convert_choroba)
        
        hosp_proc = hosp_val.mean() * 100 if hosp_val.count() > 0 else 0
        dom_proc = dom_val.mean() * 100 if dom_val.count() > 0 else 0
        
        wyniki_choroby.append({
            'choroba': choroba,
            'hosp': hosp_proc,
            'dom': dom_proc,
            'roznica': hosp_proc - dom_proc
        })

df_chor = pd.DataFrame(wyniki_choroby)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Wykres słupkowy
x = range(len(df_chor))
width = 0.35

axes[0].bar([i - width/2 for i in x], df_chor['hosp'], width, label='Hospitalizowani', 
           color='#ff6b6b', edgecolor='black', linewidth=1)
axes[0].bar([i + width/2 for i in x], df_chor['dom'], width, label='Do domu', 
           color='#4ecdc4', edgecolor='black', linewidth=1)

axes[0].set_xlabel('Choroba')
axes[0].set_ylabel('Procent pacjentów (%)')
axes[0].set_title('A) Częstość występowania')
axes[0].set_xticks(x)
axes[0].set_xticklabels(df_chor['choroba'])
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Dodaj wartości
for i, row in df_chor.iterrows():
    axes[0].text(i - width/2, row['hosp'] + 1, f"{row['hosp']:.0f}%", 
                ha='center', va='bottom', fontsize=9)
    axes[0].text(i + width/2, row['dom'] + 1, f"{row['dom']:.0f}%", 
                ha='center', va='bottom', fontsize=9)

# Wykres różnic
kolory = ['green' if x > 0 else 'red' for x in df_chor['roznica']]
axes[1].bar(df_chor['choroba'], df_chor['roznica'], color=kolory, edgecolor='black', linewidth=1)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_xlabel('Choroba')
axes[1].set_ylabel('Różnica procentowa (hosp - dom)')