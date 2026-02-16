# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 10:40:31 2026

@author: aneta
"""

# -*- coding: utf-8 -*-
"""
PEŁNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 3.0 - wszystkie parametry + MAP
Autor: Analiza danych pacjentów
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import pi
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
print("Wersja z MAP (Średnie Ciśnienie Tętnicze)")
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
# 3. ANALIZA STATYSTYCZNA - TABELA WSZYSTKICH PARAMETRÓW
# =============================================================================

print("\n" + "="*80)
print("2. ANALIZA STATYSTYCZNA - WSZYSTKIE PARAMETRY")
print("="*80)

wyniki = []
print("\n{:<25} {:>6} {:>12} {:>12} {:>12}".format(
    "Parametr", "n_hosp", "Hosp (śr±SD)", "Dom (śr±SD)", "p-value"
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
            
            print("{:<25} {:>3}   {:>6.2f}±{:<5.2f} {:>6.2f}±{:<5.2f}  p={:<.4f} {}".format(
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

# =============================================================================
# 4. RANKING PARAMETRÓW WG ISTOTNOŚCI
# =============================================================================

print("\n" + "="*80)
print("3. RANKING PARAMETRÓW WG ISTOTNOŚCI")
print("="*80)

df_wyniki = pd.DataFrame(wyniki)
df_wyniki = df_wyniki.sort_values('p_value')

print("\n{:<4} {:<25} {:>12} {:>6} {:>12}".format(
    "Rank", "Parametr", "p-value", "Istot.", "Różnica"
))
print("-"*60)

for i, row in df_wyniki.iterrows():
    rank = i + 1
    if row['p_value'] < 0.001:
        poziom = "***"
    elif row['p_value'] < 0.01:
        poziom = "**"
    elif row['p_value'] < 0.05:
        poziom = "*"
    else:
        poziom = "ns"
    
    kierunek = "+" if row['roznica'] > 0 else "-"
    print("{:<4} {:<25} {:>8.4f} {:>6} {:>+8.2f}".format(
        rank, row['parametr'][:24], row['p_value'], poziom, row['roznica']
    ))

# =============================================================================
# 5. WYKRES 1: PARAMETRY ISTOTNE - WYKRESY PUDEŁKOWE
# =============================================================================

print("\n4. GENEROWANIE WYKRESÓW...")

istotne = df_wyniki[df_wyniki['p_value'] < 0.05]
parametry_istotne = istotne['parametr'].tolist()

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
    plt.show()
    print("✓ Zapisano: wykres1_parametry_istotne.png")

# =============================================================================
# 6. WYKRES 2: MAP - SZCZEGÓŁOWA ANALIZA
# =============================================================================

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
    axes[2].scatter(df_hosp['wiek'], hosp_map, color='red', alpha=0.6, s=60, label='Hospitalizowani')
    axes[2].scatter(df_dom['wiek'], dom_map, color='blue', alpha=0.6, s=60, label='Do domu')
    axes[2].set_xlabel('Wiek (lata)')
    axes[2].set_ylabel('MAP (mmHg)')
    axes[2].set_title('MAP vs Wiek')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('RYSUNEK 2: SZCZEGÓŁOWA ANALIZA MAP', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('wykres2_MAP_analiza.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Zapisano: wykres2_MAP_analiza.png")

# =============================================================================
# 7. WYKRES 3: MACIERZ KORELACJI
# =============================================================================

parametry_korelacji = ['wiek', 'MAP', 'SpO2', 'HGB(12,4-15,2)', 'crp(0-0,5)', 'kreatynina(0,5-1,2)']

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

plt.suptitle('RYSUNEK 3: MACIERZ KORELACJI (z MAP)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres3_macierz_korelacji.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Zapisano: wykres3_macierz_korelacji.png")

# =============================================================================
# 8. WYKRES 4: WYKRESY ROZRZTU
# =============================================================================

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

plt.suptitle('RYSUNEK 4: WYKRESY ROZRZTU - ZALEŻNOŚCI MIĘDZY PARAMETRAMI', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres4_wykresy_rozrzutu.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Zapisano: wykres4_wykresy_rozrzutu.png")

# =============================================================================
# 9. WYKRES 5: ROZKŁADY PARAMETRÓW
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

parametry_rozklady = ['MAP', 'SpO2', 'HGB(12,4-15,2)', 'crp(0-0,5)', 'wiek', 'kreatynina(0,5-1,2)']

for i, param in enumerate(parametry_rozklady):
    if i < 6 and param in df_hosp.columns:
        axes[i].hist(df_hosp[param].dropna(), bins=15, alpha=0.5, color='red', density=True, label='Hosp')
        axes[i].hist(df_dom[param].dropna(), bins=15, alpha=0.5, color='blue', density=True, label='Dom')
        
        sns.kdeplot(data=df_hosp[param].dropna(), ax=axes[i], color='darkred', linewidth=2)
        sns.kdeplot(data=df_dom[param].dropna(), ax=axes[i], color='darkblue', linewidth=2)
        
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Gęstość')
        axes[i].set_title(f'Rozkład: {param}')
        axes[i].legend()

plt.suptitle('RYSUNEK 5: ROZKŁADY PARAMETRÓW', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres5_rozklady.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Zapisano: wykres5_rozklady.png")

# =============================================================================
# 10. WYKRES 6: CHOROBY WSPÓŁISTNIEJĄCE
# =============================================================================

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
axes[1].set_title('B) Różnica w częstości')
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('RYSUNEK 6: CHOROBY WSPÓŁISTNIEJĄCE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres6_choroby.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Zapisano: wykres6_choroby.png")

# =============================================================================
# 11. WYKRES 7: PROFIL PACJENTA (RADAR CHART)
# =============================================================================

parametry_profil = ['MAP', 'SpO2', 'HGB(12,4-15,2)', 'crp(0-0,5)', 'wiek']
df_profil = pd.DataFrame()

for param in parametry_profil:
    if param in df_hosp.columns:
        wszystkie = pd.concat([df_hosp[param], df_dom[param]]).dropna()
        min_val = wszystkie.min()
        max_val = wszystkie.max()
        range_val = max_val - min_val
        
        if range_val > 0:
            if param == 'crp(0-0,5)':  # Dla CRP - wyższe = gorzej
                hosp_norm = 100 - ((df_hosp[param].mean() - min_val) / range_val) * 100
                dom_norm = 100 - ((df_dom[param].mean() - min_val) / range_val) * 100
            else:  # Dla pozostałych - wyższe = lepiej
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
    ax.set_title('RYSUNEK 7: PROFIL PACJENTA (100 = najlepiej)', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('wykres7_profil.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Zapisano: wykres7_profil.png")

# =============================================================================
# 12. PODSUMOWANIE I WNIOSKI
# =============================================================================

print("\n" + "="*80)
print("5. PODSUMOWANIE - CZYNNIKI RYZYKA HOSPITALIZACJI")
print("="*80)

print("\n🔴 CZYNNIKI ZWIĘKSZAJĄCE RYZYKO HOSPITALIZACJI:\n")

istotne = df_wyniki[df_wyniki['p_value'] < 0.05].sort_values('p_value')
for _, row in istotne.iterrows():
    kierunek = "⬆️ WYŻSZE" if row['roznica'] > 0 else "⬇️ NIŻSZE"
    print(f"  • {row['parametr']}: {kierunek} u hospitalizowanych")
    print(f"    (hosp: {row['hosp_sr']:.2f} vs dom: {row['dom_sr']:.2f}, p={row['p_value']:.4f})")

# Statystyki MAP
if 'MAP' in df_hosp.columns:
    print(f"\n  • MAP: p={p_value_map:.4f}")
    if p_value_map < 0.05:
        if hosp_map.mean() > dom_map.mean():
            print(f"    (hosp: {hosp_map.mean():.1f} vs dom: {dom_map.mean():.1f} - WYŻSZE u hosp)")
        else:
            print(f"    (hosp: {hosp_map.mean():.1f} vs dom: {dom_map.mean():.1f} - NIŻSZE u hosp)")
    else:
        print(f"    (hosp: {hosp_map.mean():.1f} vs dom: {dom_map.mean():.1f} - różnica NIEISTOTNA)")

print("\n" + "="*80)
print("✓ ZAPISANO PLIKI:")
print("="*80)

# Zapisz wyniki do CSV
df_wyniki.to_csv('wyniki_parametry.csv', sep=';', index=False, decimal=',')
df_chor.to_csv('wyniki_choroby.csv', sep=';', index=False, decimal=',')

print("\nPliki danych:")
print("  • wyniki_parametry.csv - statystyki wszystkich parametrów (z MAP)")
print("  • wyniki_choroby.csv - analiza chorób")

print("\nWykresy (7 plików PNG):")
print("  • wykres1_parametry_istotne.png - parametry z istotnymi różnicami")
print("  • wykres2_MAP_analiza.png - szczegółowa analiza MAP")
print("  • wykres3_macierz_korelacji.png - korelacje z MAP")
print("  • wykres4_wykresy_rozrzutu.png - zależności między parametrami")
print("  • wykres5_rozklady.png - rozkłady parametrów (z MAP)")
print("  • wykres6_choroby.png - choroby współistniejące")
print("  • wykres7_profil.png - profil pacjenta (z MAP)")

print("\n" + "="*80)
print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE!")
print("="*80)