# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 10:21:22 2026

@author: aneta
"""

# -*- coding: utf-8 -*-
"""
PEŁNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 2.0 - wszystkie wykresy i analizy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
print("PEŁNA ANALIZA DANYCH PACJENTÓW - HOSPITALIZOWANI vs DO DOMU")
print("="*80)

# =============================================================================
# 1. WCZYTYWANIE I PRZYGOTOWANIE DANYCH
# =============================================================================

print("\n1. WCZYTYWANIE DANYCH...")

# Wczytaj dane
df = pd.read_csv('baza_danych_pacjentów_a.csv', sep=';', encoding='utf-8')

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
                'parametr': param, 'hosp_n': len(hosp), 'hosp_sr': hosp_sr, 'hosp_std': hosp_std,
                'dom_n': len(dom), 'dom_sr': dom_sr, 'dom_std': dom_std,
                'p_value': p_value, 'istotnosc': gwiazdki
            })

# =============================================================================
# 4. WYKRES 1: PORÓWNANIE PARAMETRÓW - WYKRESY PUDEŁKOWE
# =============================================================================

print("\n" + "="*80)
print("3. GENEROWANIE WYKRESÓW...")
print("="*80)

# Wybierz parametry z istotnymi różnicami
df_wyniki = pd.DataFrame(wyniki)
istotne = df_wyniki[df_wyniki['p_value'] < 0.05].sort_values('p_value')
parametry_istotne = istotne['parametr'].tolist()

print(f"\n✓ Znaleziono {len(parametry_istotne)} parametrów z istotnymi różnicami")

if len(parametry_istotne) > 0:
    # Wykres 1.1: Wykresy pudełkowe dla istotnych parametrów
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(parametry_istotne[:6]):  # Maksymalnie 6 wykresów
        if i < 6 and param in df_hosp.columns:
            dane_hosp = df_hosp[param].dropna()
            dane_dom = df_dom[param].dropna()
            
            dane = pd.DataFrame({
                'wartosc': pd.concat([dane_hosp, dane_dom]),
                'grupa': ['Hospitalizowani']*len(dane_hosp) + ['Do domu']*len(dane_dom)
            })
            
            p_val = istotne[istotne['parametr'] == param]['p_value'].values[0]
            gwiazdki = istotne[istotne['parametr'] == param]['istotnosc'].values[0]
            
            sns.boxplot(data=dane, x='grupa', y='wartosc', ax=axes[i], 
                       palette=['#ff6b6b', '#4ecdc4'])
            axes[i].set_title(f'{param}\np={p_val:.4f} {gwiazdki}', fontweight='bold')
            axes[i].set_xlabel('')
            sns.stripplot(data=dane, x='grupa', y='wartosc', ax=axes[i], 
                         color='black', alpha=0.5, size=3)
    
    # Ukryj puste wykresy
    for j in range(i+1, 6):
        axes[j].set_visible(False)
    
    plt.suptitle('RYSUNEK 1: PARAMETRY Z ISTOTNYMI RÓŻNICAMI', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('wykres1_parametry_istotne.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Zapisano: wykres1_parametry_istotne.png")

# =============================================================================
# 5. WYKRES 2: MACIERZ KORELACJI
# =============================================================================

print("\nGenerowanie macierzy korelacji...")

# Wybierz parametry do korelacji
parametry_korelacji = ['wiek', 'SpO2', 'HGB(12,4-15,2)', 'crp(0-0,5)', 
                       'kreatynina(0,5-1,2)', 'WBC(4-11)']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Korelacja - hospitalizowani
df_hosp_corr = df_hosp[parametry_korelacji].dropna()
if len(df_hosp_corr) > 0:
    corr_hosp = df_hosp_corr.corr()
    sns.heatmap(corr_hosp, annot=True, cmap='RdBu_r', center=0, 
                square=True, linewidths=1, ax=axes[0], cbar=False,
                vmin=-1, vmax=1, fmt='.2f')
    axes[0].set_title('A) HOSPITALIZOWANI', fontweight='bold')

# Korelacja - do domu
df_dom_corr = df_dom[parametry_korelacji].dropna()
if len(df_dom_corr) > 0:
    corr_dom = df_dom_corr.corr()
    sns.heatmap(corr_dom, annot=True, cmap='RdBu_r', center=0, 
                square=True, linewidths=1, ax=axes[1], cbar=True,
                vmin=-1, vmax=1, fmt='.2f')
    axes[1].set_title('B) DO DOMU', fontweight='bold')

plt.suptitle('RYSUNEK 2: MACIERZ KORELACJI - PORÓWNANIE GRUP', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres2_macierz_korelacji.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Zapisano: wykres2_macierz_korelacji.png")

# =============================================================================
# 6. WYKRES 3: WIEK vs SpO2 (WYKRES ROZRZTU)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Wykres rozrzutu - wiek vs SpO2
axes[0].scatter(df_hosp['wiek'], df_hosp['SpO2'], c='red', alpha=0.6, s=80, label='Hospitalizowani')
axes[0].scatter(df_dom['wiek'], df_dom['SpO2'], c='blue', alpha=0.6, s=80, label='Do domu')
axes[0].set_xlabel('Wiek (lata)')
axes[0].set_ylabel('SpO2 (%)')
axes[0].set_title('A) Wiek vs Saturacja')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Linia trendu dla hospitalizowanych
if len(df_hosp[['wiek', 'SpO2']].dropna()) > 1:
    z_hosp = np.polyfit(df_hosp['wiek'].dropna(), df_hosp['SpO2'].dropna(), 1)
    p_hosp = np.poly1d(z_hosp)
    x_trend = np.linspace(df_hosp['wiek'].min(), df_hosp['wiek'].max(), 50)
    axes[0].plot(x_trend, p_hosp(x_trend), 'r--', alpha=0.5, linewidth=2)

# Wykres rozrzutu - HGB vs CRP
axes[1].scatter(df_hosp['HGB(12,4-15,2)'], df_hosp['crp(0-0,5)'], c='red', alpha=0.6, s=80, label='Hospitalizowani')
axes[1].scatter(df_dom['HGB(12,4-15,2)'], df_dom['crp(0-0,5)'], c='blue', alpha=0.6, s=80, label='Do domu')
axes[1].set_xlabel('Hemoglobina (HGB)')
axes[1].set_ylabel('CRP (stan zapalny)')
axes[1].set_title('B) Hemoglobina vs CRP')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')  # Skala logarytmiczna dla CRP

plt.suptitle('RYSUNEK 3: WYKRESY ROZRZTU - ZALEŻNOŚCI MIĘDZY PARAMETRAMI', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres3_wykresy_rozrzutu.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Zapisano: wykres3_wykresy_rozrzutu.png")

# =============================================================================
# 7. WYKRES 4: ROZKŁADY PARAMETRÓW (HISTOGRAMY + KDE)
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

parametry_rozklady = ['SpO2', 'HGB(12,4-15,2)', 'crp(0-0,5)', 'wiek', 'kreatynina(0,5-1,2)', 'WBC(4-11)']

for i, param in enumerate(parametry_rozklady):
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

plt.suptitle('RYSUNEK 4: ROZKŁADY PARAMETRÓW - PORÓWNANIE GRUP', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres4_rozklady.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Zapisano: wykres4_rozklady.png")

# =============================================================================
# 8. WYKRES 5: CHOROBY WSPÓŁISTNIEJĄCE
# =============================================================================

print("\nAnaliza chorób współistniejących...")

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
            'dom': dom_proc
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
axes[1].bar(df_chor['choroba'], df_chor['hosp'] - df_chor['dom'], 
           color=['green' if x > 0 else 'red' for x in df_chor['hosp'] - df_chor['dom']],
           edgecolor='black', linewidth=1)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_xlabel('Choroba')
axes[1].set_ylabel('Różnica procentowa (hosp - dom)')
axes[1].set_title('B) Różnica w częstości')
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('RYSUNEK 5: CHOROBY WSPÓŁISTNIEJĄCE - PORÓWNANIE GRUP', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres5_choroby.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Zapisano: wykres5_choroby.png")

# =============================================================================
# 9. WYKRES 6: PROFIL PACJENTA - RADAR CHART
# =============================================================================

from math import pi

# Wybierz parametry do profilu
parametry_profil = ['SpO2', 'HGB(12,4-15,2)', 'crp(0-0,5)', 'wiek']
df_profil = pd.DataFrame()

for param in parametry_profil:
    if param in df_hosp.columns:
        # Normalizacja do zakresu 0-100
        wszystkie = pd.concat([df_hosp[param], df_dom[param]]).dropna()
        min_val = wszystkie.min()
        max_val = wszystkie.max()
        range_val = max_val - min_val
        
        if range_val > 0:
            hosp_norm = ((df_hosp[param].mean() - min_val) / range_val) * 100
            dom_norm = ((df_dom[param].mean() - min_val) / range_val) * 100
        else:
            hosp_norm = dom_norm = 50
        
        df_profil[param] = [hosp_norm, dom_norm]

if len(df_profil) > 0:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    categories = list(df_profil.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Hospitalizowani
    values = df_profil.iloc[0].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, color='red', label='Hospitalizowani')
    ax.fill(angles, values, alpha=0.25, color='red')
    
    # Do domu
    values = df_profil.iloc[1].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, color='blue', label='Do domu')
    ax.fill(angles, values, alpha=0.25, color='blue')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_title('RYSUNEK 6: PROFIL PACJENTA (wartości znormalizowane)', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('wykres6_profil.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Zapisano: wykres6_profil.png")

# =============================================================================
# 10. PODSUMOWANIE I ZAPIS WYNIKÓW
# =============================================================================

print("\n" + "="*80)
print("4. PODSUMOWANIE - CZYNNIKI RYZYKA HOSPITALIZACJI")
print("="*80)

print("\n🔴 CZYNNIKI ZWIĘKSZAJĄCE RYZYKO HOSPITALIZACJI:\n")

for _, row in istotne.iterrows():
    kierunek = "⬆️ WYŻSZE" if row['hosp_sr'] > row['dom_sr'] else "⬇️ NIŻSZE"
    print(f"  • {row['parametr']}: {kierunek} u hospitalizowanych")
    print(f"    (hosp: {row['hosp_sr']:.2f} vs dom: {row['dom_sr']:.2f}, p={row['p_value']:.4f} {row['istotnosc']})")

print("\n" + "="*80)
print("✓ ZAPISANO PLIKI:")
print("="*80)

# Zapisz wyniki do CSV
df_wyniki.to_csv('wyniki_parametry.csv', sep=';', index=False, decimal=',')
df_chor.to_csv('wyniki_choroby.csv', sep=';', index=False, decimal=',')

print("\nPliki danych:")
print("  • wyniki_parametry.csv - statystyki parametrów")
print("  • wyniki_choroby.csv - analiza chorób")

print("\nWykresy (6 plików PNG):")
print("  • wykres1_parametry_istotne.png - parametry z istotnymi różnicami")
print("  • wykres2_macierz_korelacji.png - korelacje między parametrami")
print("  • wykres3_wykresy_rozrzutu.png - zależności (wiek/SpO2, HGB/CRP)")
print("  • wykres4_rozklady.png - rozkłady parametrów")
print("  • wykres5_choroby.png - choroby współistniejące")
print("  • wykres6_profil.png - profil pacjenta (radar chart)")

print("\n" + "="*80)
print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE!")
print("="*80)