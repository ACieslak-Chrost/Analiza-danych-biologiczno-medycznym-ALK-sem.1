# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 10:03:53 2026

@author: aneta
"""

# -*- coding: utf-8 -*-
"""
KOMPLETNA ANALIZA DANYCH PACJENTÓW
- Porównanie hospitalizowani vs do domu
- Wszystkie parametry kliniczne
- Choroby współistniejące z poprawną konwersją
- Testy statystyczne i wizualizacje
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Ustawienie stylu wykresów
plt.style.use('ggplot')
sns.set_palette("Set2")
sns.set_style("whitegrid")

print("="*80)
print("KOMPLETNA ANALIZA PORÓWNAWCZA: HOSPITALIZOWANI vs WYPISANI DO DOMU")
print("="*80)

# =============================================================================
# 1. WCZYTAJ DANE I PODZIEL NA GRUPY
# =============================================================================

# Wczytaj wszystkie dane
df = pd.read_csv('baza_danych_pacjentów_a.csv', sep=';')

# Znajdź pusty wiersz (separator między grupami)
puste_wiersze = df[df.isna().all(axis=1)]
if len(puste_wiersze) > 0:
    indeks_podzialu = puste_wiersze.index[0]
    
    # Podziel na grupy
    df_hosp = df.iloc[:indeks_podzialu].copy()      # hospitalizowani
    df_dom = df.iloc[indeks_podzialu+1:].copy()     # do domu
    
    # Oczyść z wierszy z samymi separatorami
    df_hosp = df_hosp.dropna(how='all')
    df_dom = df_dom.dropna(how='all')
    
    print(f"\n✓ Grupa HOSPITALIZOWANI: {len(df_hosp)} pacjentów")
    print(f"✓ Grupa DO DOMU: {len(df_dom)} pacjentów")
else:
    print("✗ Nie znaleziono pustego wiersza - sprawdź strukturę pliku")
    exit()

# =============================================================================
# 2. FUNKCJA DO KONWERSJI CHORÓB NA WARTOŚCI LOGICZNE
# =============================================================================

def convert_choroba(wartosc):
    """
    Konwertuje różne formy zapisu na wartości logiczne
    """
    if pd.isna(wartosc):
        return np.nan
    if isinstance(wartosc, str):
        wartosc = wartosc.lower().strip()
        # Wszystkie formy "tak"
        if wartosc in ['tak', 't', 'yes', 'y', '1', 'true', '+', 'tak!', 'TAK']:
            return True
        # Wszystkie formy "nie"
        elif wartosc in ['nie', 'n', 'no', '0', 'false', '-', 'NIE']:
            return False
        # Brak danych
        elif wartosc in ['bd', 'brak', 'nan', 'null', 'none', '']:
            return np.nan
    return np.nan

# =============================================================================
# 3. PRZYGOTOWANIE DANYCH - KONWERSJA NA TYPY LICZBOWE
# =============================================================================

# Lista wszystkich parametrów do analizy
parametry_kliniczne = [
    'wiek',
    'RR',           
    'MAP',
    'SpO2',         
    'AS',
    'mleczany',     
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
for df_grupa in [df_hosp, df_dom]:
    for col in parametry_kliniczne:
        if col in df_grupa.columns:
            # Zamiana przecinków na kropki i konwersja na float
            df_grupa[col] = pd.to_numeric(
                df_grupa[col].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )

print("\n" + "="*80)
print("CZĘŚĆ 1: PARAMETRY KLINICZNE")
print("="*80)

# =============================================================================
# 4. ANALIZA STATYSTYCZNA PARAMETRÓW KLINICZNYCH
# =============================================================================

wyniki_parametry = []

print("\n{:<25} {:>15} {:>15} {:>15} {:>15}".format(
    "Parametr", "Hosp (śr±SD)", "Dom (śr±SD)", "Różnica", "p-value"
))
print("-"*85)

for param in parametry_kliniczne:
    if param in df_hosp.columns:
        # Pobierz dane bez NaN
        hosp = df_hosp[param].dropna()
        dom = df_dom[param].dropna()
        
        if len(hosp) > 0 and len(dom) > 0:
            # Podstawowe statystyki
            hosp_sr = hosp.mean()
            hosp_std = hosp.std()
            dom_sr = dom.mean()
            dom_std = dom.std()
            roznica = hosp_sr - dom_sr
            
            # Test statystyczny (Mann-Whitney U - nie wymaga normalności rozkładu)
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
            
            # Wyświetl wyniki
            print("{:<25} {:>8.2f}±{:<5.2f} {:>8.2f}±{:<5.2f} {:>+8.2f}    p={:<.4f} {}".format(
                param[:24], hosp_sr, hosp_std, dom_sr, dom_std, roznica, p_value, gwiazdki
            ))
            
            # Zapisz do DataFrame
            wyniki_parametry.append({
                'parametr': param,
                'hosp_n': len(hosp),
                'hosp_sr': hosp_sr,
                'hosp_std': hosp_std,
                'dom_n': len(dom),
                'dom_sr': dom_sr,
                'dom_std': dom_std,
                'roznica': roznica,
                'p_value': p_value,
                'istotnosc': gwiazdki
            })

# =============================================================================
# 5. WIZUALIZACJA - WYKRESY PUDEŁKOWE DLA ISTOTNYCH PARAMETRÓW
# =============================================================================

df_wyniki = pd.DataFrame(wyniki_parametry)
istotne = df_wyniki[df_wyniki['p_value'] < 0.05].sort_values('p_value')

print("\n" + "="*80)
print(f"✓ Znaleziono {len(istotne)} parametrów z istotnymi różnicami (p<0.05)")

if len(istotne) > 0:
    print("\nNajistotniejsze parametry:")
    for _, row in istotne.head().iterrows():
        kierunek = "↑ WYŻSZE" if row['roznica'] > 0 else "↓ NIŻSZE"
        print(f"  • {row['parametr']}: {kierunek} u hospitalizowanych (p={row['p_value']:.4f} {row['istotnosc']})")
    
    # Wykresy pudełkowe dla najważniejszych parametrów
    n_wykresow = min(len(istotne), 6)
    if n_wykresow > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (_, row) in enumerate(istotne.head(n_wykresow).iterrows()):
            param = row['parametr']
            
            # Przygotuj dane
            dane_hosp = df_hosp[param].dropna()
            dane_dom = df_dom[param].dropna()
            
            # Stwórz DataFrame dla seaborn
            dane_wykres = pd.DataFrame({
                'wartosc': pd.concat([dane_hosp, dane_dom]),
                'grupa': ['Hospitalizowani']*len(dane_hosp) + ['Do domu']*len(dane_dom)
            })
            
            # Rysuj wykres pudełkowy
            sns.boxplot(data=dane_wykres, x='grupa', y='wartosc', ax=axes[i], palette=['#ff9999', '#66b3ff'])
            axes[i].set_title(f'{param}\np={row["p_value"]:.4f} {row["istotnosc"]}', fontweight='bold')
            axes[i].set_xlabel('')
            
            # Dodaj punkty danych
            sns.stripplot(data=dane_wykres, x='grupa', y='wartosc', ax=axes[i], 
                         color='black', alpha=0.5, size=3)
        
        # Ukryj puste wykresy
        for j in range(i+1, 6):
            axes[j].set_visible(False)
        
        plt.suptitle('Parametry z istotnymi różnicami między grupami', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

print("\n" + "="*80)
print("CZĘŚĆ 2: CHOROBY WSPÓŁISTNIEJĄCE")
print("="*80)

# =============================================================================
# 6. ANALIZA CHORÓB WSPÓŁISTNIEJĄCYCH
# =============================================================================

choroby = ['dm', 'wątroba', 'naczyniowe', 'zza', 'npl']
wyniki_choroby = []

print("\n{:<15} {:>20} {:>20} {:>15}".format(
    "Choroba", "Hospitalizowani", "Do domu", "p-value"
))
print("-"*75)

for choroba in choroby:
    if choroba in df_hosp.columns:
        # Konwertuj wartości
        hosp_values = df_hosp[choroba].apply(convert_choroba)
        dom_values = df_dom[choroba].apply(convert_choroba)
        
        # Policz procenty (bez braków danych)
        hosp_tak = hosp_values.sum() / hosp_values.count() * 100 if hosp_values.count() > 0 else 0
        dom_tak = dom_values.sum() / dom_values.count() * 100 if dom_values.count() > 0 else 0
        
        # Liczba pacjentów z danymi
        hosp_n = hosp_values.count()
        dom_n = dom_values.count()
        
        # Test chi-kwadrat
        if hosp_n > 0 and dom_n > 0:
            tabela = [[hosp_values.sum(), hosp_n - hosp_values.sum()],
                     [dom_values.sum(), dom_n - dom_values.sum()]]
            
            # Sprawdź czy można wykonać test chi-kwadrat
            if min(tabela[0][0], tabela[0][1], tabela[1][0], tabela[1][1]) > 0:
                chi2, p_value, dof, expected = stats.chi2_contingency(tabela)
                
                # Określenie istotności
                if p_value < 0.001:
                    gwiazdki = "***"
                elif p_value < 0.01:
                    gwiazdki = "**"
                elif p_value < 0.05:
                    gwiazdki = "*"
                else:
                    gwiazdki = "ns"
                
                # Wyświetl wyniki
                print("{:<15} {:>6.1f}% ({:2d}/{:<2d}) {:>6.1f}% ({:2d}/{:<2d})   p={:<.4f} {}".format(
                    choroba, 
                    hosp_tak, int(hosp_values.sum()), hosp_n,
                    dom_tak, int(dom_values.sum()), dom_n,
                    p_value, gwiazdki
                ))
                
                wyniki_choroby.append({
                    'choroba': choroba,
                    'hosp_proc': hosp_tak,
                    'hosp_tak': int(hosp_values.sum()),
                    'hosp_n': hosp_n,
                    'dom_proc': dom_tak,
                    'dom_tak': int(dom_values.sum()),
                    'dom_n': dom_n,
                    'roznica': hosp_tak - dom_tak,
                    'p_value': p_value,
                    'istotnosc': gwiazdki
                })

# =============================================================================
# 7. WIZUALIZACJA CHORÓB WSPÓŁISTNIEJĄCYCH
# =============================================================================

if len(wyniki_choroby) > 0:
    df_choroby = pd.DataFrame(wyniki_choroby)
    
    # Wykres słupkowy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Wykres 1: Porównanie procentów
    x = range(len(df_choroby))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], df_choroby['hosp_proc'], width, 
            label='Hospitalizowani', color='#ff9999', alpha=0.8)
    ax1.bar([i + width/2 for i in x], df_choroby['dom_proc'], width, 
            label='Do domu', color='#66b3ff', alpha=0.8)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_choroby['choroba'])
    ax1.set_ylabel('Procent pacjentów (%)')
    ax1.set_title('Choroby współistniejące - porównanie grup')
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Dodaj wartości na słupkach
    for i, row in df_choroby.iterrows():
        ax1.text(i - width/2, row['hosp_proc'] + 2, f"{row['hosp_proc']:.0f}%", 
                ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, row['dom_proc'] + 2, f"{row['dom_proc']:.0f}%", 
                ha='center', va='bottom', fontsize=9)
    
    # Dodaj gwiazdki istotności
    for i, row in df_choroby.iterrows():
        if row['istotnosc'] != 'ns':
            max_val = max(row['hosp_proc'], row['dom_proc'])
            ax1.text(i, max_val + 5, row['istotnosc'], 
                    ha='center', fontsize=14, fontweight='bold')
    
    # Wykres 2: Różnica procentowa
    kolory = ['green' if x > 0 else 'red' for x in df_choroby['roznica']]
    ax2.bar(df_choroby['choroba'], df_choroby['roznica'], color=kolory, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Różnica procentowa (hosp - dom)')
    ax2.set_title('Różnica w częstości występowania')
    
    # Dodaj wartości na słupkach
    for i, row in df_choroby.iterrows():
        ax2.text(i, row['roznica'] + (2 if row['roznica'] > 0 else -4), 
                f"{row['roznica']:+.1f}%", ha='center', fontsize=10,
                color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

print("\n" + "="*80)
print("CZĘŚĆ 3: PODSUMOWANIE I WNIOSKI")
print("="*80)

# =============================================================================
# 8. PODSUMOWANIE - CZYNNIKI RYZYKA HOSPITALIZACJI
# =============================================================================

print("\n🔴 CZYNNIKI ZWIĘKSZAJĄCE RYZYKO HOSPITALIZACJI:")
print("-" * 50)

# Czynniki z parametrów klinicznych
istotne_param = df_wyniki[df_wyniki['p_value'] < 0.05].sort_values('p_value')
if len(istotne_param) > 0:
    print("\n📊 PARAMETRY KLINICZNE:")
    for _, row in istotne_param.iterrows():
        kierunek = "⬆️ WYŻSZE" if row['roznica'] > 0 else "⬇️ NIŻSZE"
        print(f"  • {row['parametr']}: {kierunek} u hospitalizowanych")
        print(f"    (hosp: {row['hosp_sr']:.2f} vs dom: {row['dom_sr']:.2f}, p={row['p_value']:.4f})")

# Czynniki z chorób współistniejących
if len(wyniki_choroby) > 0:
    df_choroby_istotne = df_choroby[df_choroby['p_value'] < 0.05].sort_values('p_value')
    if len(df_choroby_istotne) > 0:
        print("\n🏥 CHOROBY WSPÓŁISTNIEJĄCE:")
        for _, row in df_choroby_istotne.iterrows():
            print(f"  • {row['choroba']}: częstsza u hospitalizowanych o {row['roznica']:+.1f}%")
            print(f"    (hosp: {row['hosp_proc']:.1f}% vs dom: {row['dom_proc']:.1f}%, p={row['p_value']:.4f})")

print("\n" + "="*80)
print("✓ Analiza zakończona pomyślnie!")
print("="*80)

# =============================================================================
# 9. ZAPIS WYNIKÓW DO PLIKÓW CSV
# =============================================================================

# Zapisz szczegółowe wyniki
df_wyniki.to_csv('wyniki_parametry_kliniczne.csv', sep=';', index=False, decimal=',')
print(f"\n✓ Zapisano wyniki parametrów: 'wyniki_parametry_kliniczne.csv'")

if len(wyniki_choroby) > 0:
    df_choroby.to_csv('wyniki_choroby_wspolistniejace.csv', sep=';', index=False, decimal=',')
    print(f"✓ Zapisano wyniki chorób: 'wyniki_choroby_wspolistniejace.csv'")

# Stwórz raport podsumowujący
with open('raport_końcowy.txt', 'w', encoding='utf-8') as f:
    f.write("RAPORT Z ANALIZY DANYCH PACJENTÓW\n")
    f.write("="*50 + "\n\n")
    f.write(f"Liczba hospitalizowanych: {len(df_hosp)}\n")
    f.write(f"Liczba wypisanych do domu: {len(df_dom)}\n\n")
    
    f.write("ISTOTNE PARAMETRY KLINICZNE:\n")
    for _, row in istotne_param.iterrows():
        f.write(f"- {row['parametr']}: p={row['p_value']:.4f}\n")
    
    if len(wyniki_choroby) > 0 and len(df_choroby_istotne) > 0:
        f.write("\nISTOTNE CHOROBY WSPÓŁISTNIEJĄCE:\n")
        for _, row in df_choroby_istotne.iterrows():
            f.write(f"- {row['choroba']}: p={row['p_value']:.4f}\n")
    
    f.write("\n" + "="*50 + "\n")
    f.write("Koniec raportu\n")

print(f"✓ Zapisano raport: 'raport_końcowy.txt'")
print("\n" + "="*80)