# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 09:54:34 2026

@author: aneta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Ustawienie stylu wykresów
plt.style.use('ggplot')
sns.set_palette("Set2")

print("=== ANALIZA PORÓWNAWCZA: HOSPITALIZOWANI vs WYPISANI DO DOMU ===\n")

# 1. WCZYTAJ DANE I PODZIEL
df = pd.read_csv('baza_danych_pacjentów_a.csv', sep=';')

# Znajdź pusty wiersz (separator między grupami)
puste_wiersze = df[df.isna().all(axis=1)]
if len(puste_wiersze) > 0:
    indeks_podzialu = puste_wiersze.index[0]
    
    # Podziel na grupy
    df_hosp = df.iloc[:indeks_podzialu].copy()      # hospitalizowani (wiersze 2-30)
    df_dom = df.iloc[indeks_podzialu+1:].copy()     # do domu (wiersze 32-52)
    
    # Oczyść z wierszy z samymi separatorami
    df_hosp = df_hosp.dropna(how='all')
    df_dom = df_dom.dropna(how='all')
    
    print(f"Grupa HOSPITALIZOWANI: {len(df_hosp)} pacjentów")
    print(f"Grupa DO DOMU: {len(df_dom)} pacjentów")
    print("-" * 60)

# 2. PARAMETRY DO ANALIZY (wybierz te, które mają znaczenie kliniczne)
parametry_kliniczne = [
    'wiek',
    'RR',           # ciśnienie
    'SpO2',         # saturacja
    'mleczany',     # kwas mlekowy
    'kreatynina(0,5-1,2)',
    'troponina I (0-7,8))',
    'HGB(12,4-15,2)',
    'WBC(4-11)',
    'plt(130-450)',
    'hct(38-45)',
    'Na(137-145)',
    'K(3,5-5,1)',
    'crp(0-0,5)'
]

# Konwersja na typ liczbowy (zamiana przecinków na kropki)
for df in [df_hosp, df_dom]:
    for col in parametry_kliniczne:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

print("\n=== 1. STATYSTYKI OPISOWE ===\n")

wyniki = []
for param in parametry_kliniczne:
    if param in df_hosp.columns:
        hosp = df_hosp[param].dropna()
        dom = df_dom[param].dropna()
        
        if len(hosp) > 0 and len(dom) > 0:
            print(f"\n{param}:")
            print(f"  Hospitalizowani: n={len(hosp):2d}, śr={hosp.mean():7.2f} ± {hosp.std():.2f}")
            print(f"  Do domu:         n={len(dom):2d}, śr={dom.mean():7.2f} ± {dom.std():.2f}")
            
            # Test statystyczny (t-test lub Mann-Whitney)
            if len(hosp) > 5 and len(dom) > 5:  # minimalna liczba do testu
                stat, p_value = stats.mannwhitneyu(hosp, dom, alternative='two-sided')
                gwiazdki = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"  Test U Manna-Whitneya: p={p_value:.4f} {gwiazdki}")
                
                wyniki.append({
                    'parametr': param,
                    'hosp_sr': hosp.mean(),
                    'hosp_std': hosp.std(),
                    'dom_sr': dom.mean(),
                    'dom_std': dom.std(),
                    'roznica': hosp.mean() - dom.mean(),
                    'p_value': p_value,
                    'istotnosc': gwiazdki
                })

print("\n" + "="*60)
print("\n=== 2. WIZUALIZACJA - WYKRESY PUDEŁKOWE ===")

# Wybierz parametry z istotnymi różnicami
df_wyniki = pd.DataFrame(wyniki)
istotne = df_wyniki[df_wyniki['p_value'] < 0.05].sort_values('p_value')

print(f"\nZnaleziono {len(istotne)} parametrów z istotnymi różnicami (p<0.05):")
for _, row in istotne.iterrows():
    print(f"  {row['parametr']}: p={row['p_value']:.4f} {row['istotnosc']}")

# Wykresy dla najważniejszych parametrów (max 6)
najwazniejsze = istotne.head(6)['parametr'].tolist()

if len(najwazniejsze) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(najwazniejsze):
        if i < 6 and param in df_hosp.columns:
            dane = pd.DataFrame({
                'wartosc': pd.concat([df_hosp[param], df_dom[param]]),
                'grupa': ['Hospitalizowani']*len(df_hosp) + ['Do domu']*len(df_dom)
            })
            dane = dane.dropna()
            
            sns.boxplot(data=dane, x='grupa', y='wartosc', ax=axes[i])
            axes[i].set_title(f'{param}\np={istotne.iloc[i]["p_value"]:.4f}')
            axes[i].set_xlabel('')
    
    plt.tight_layout()
    plt.show()

print("\n" + "="*60)
print("\n=== 3. PORÓWNANIE CHORÓB WSPÓŁISTNIEJĄCYCH ===")

choroby = ['dm', 'wątroba', 'naczyniowe', 'zza', 'npl']
for choroba in choroby:
    if choroba in df_hosp.columns:
        hosp_proc = (df_hosp[choroba].astype(str).str.lower() == 'tak').mean() * 100
        dom_proc = (df_dom[choroba].astype(str).str.lower() == 'tak').mean() * 100
        
        print(f"\n{choroba}:")
        print(f"  Hospitalizowani: {hosp_proc:.1f}%")
        print(f"  Do domu:         {dom_proc:.1f}%")

print("\n" + "="*60)
print("\n=== 4. PODSUMOWANIE - CZYNNIKI RYZYKA HOSPITALIZACJI ===")

print("\nCzynniki zwiększające ryzyko hospitalizacji:")
df_wyniki = df_wyniki.sort_values('p_value')
for _, row in df_wyniki.iterrows():
    if row['p_value'] < 0.05:
        kierunek = "WYŻSZE" if row['hosp_sr'] > row['dom_sr'] else "NIŻSZE"
        print(f"  • {row['parametr']}: {kierunek} u hospitalizowanych (p={row['p_value']:.4f})")

print("\n" + "="*60)
print("\n✅ Analiza zakończona!")

# Zapisz wyniki do pliku
df_wyniki.to_csv('wyniki_analizy.csv', sep=';', index=False)
print("\nSzczegółowe wyniki zapisano w pliku: 'wyniki_analizy.csv'")