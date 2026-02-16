# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 11:13:24 2026

@author: aneta
"""

# -*- coding: utf-8 -*-
"""
PEŁNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 4.1 - z bazą danych SQL (poprawiony zapis)
Autor: Analiza danych pacjentów
"""

import pandas as pd
import numpy as np
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
print("PEŁNA ANALIZA DANYCH PACJENTÓW - HOSPITALIZOWANI vs DO DOMU")
print("Wersja 4.1 - z bazą danych SQL")
print("="*80)

# =============================================================================
# 1. TWORZENIE BAZY DANYCH SQL Z PLIKU CSV
# =============================================================================

print("\n1. TWORZENIE BAZY DANYCH SQL...")

def create_database_from_csv(csv_file, db_file='pacjenci.db'):
    """Tworzy bazę SQLite z pliku CSV"""
    try:
        # Wczytaj CSV
        df = pd.read_csv(csv_file, sep=';', encoding='utf-8')
        
        # Oczyszczenie nazw kolumn (SQL nie lubi nawiasów i myślników)
        df.columns = [col.replace('(', '_').replace(')', '_').replace('-', '_').replace(' ', '_') for col in df.columns]
        
        # Połącz z bazą SQLite
        conn = sqlite3.connect(db_file)
        
        # Zapisz DataFrame do bazy
        df.to_sql('pacjenci', conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"✓ Utworzono bazę danych: {db_file}")
        return True
    except Exception as e:
        print(f"✗ Błąd tworzenia bazy: {e}")
        return False

# Tworzymy bazę
create_database_from_csv('baza_danych_pacjentów_a.csv')

# =============================================================================
# 2. POBIERANIE DANYCH Z BAZY SQL
# =============================================================================

print("\n2. POBIERANIE DANYCH Z BAZY SQL...")

conn = sqlite3.connect('pacjenci.db')
df = pd.read_sql_query("SELECT * FROM pacjenci", conn)
conn.close()

print(f"✓ Pobrano {len(df)} wierszy z bazy SQL")

# =============================================================================
# 3. PRZYKŁADOWE ZAPYTANIA SQL
# =============================================================================

print("\n3. PRZYKŁADOWE ZAPYTANIA SQL:")

conn = sqlite3.connect('pacjenci.db')

# Zapytanie 1: Liczba pacjentów według płci
query1 = """
SELECT plec, COUNT(*) as liczba 
FROM pacjenci 
WHERE plec IS NOT NULL AND plec != ''
GROUP BY plec
"""
df_plec = pd.read_sql_query(query1, conn)
print("\n✓ Liczba pacjentów według płci:")
print(df_plec.to_string(index=False))

# Zapytanie 2: Średni wiek
query2 = """
SELECT 
    AVG(CAST(wiek AS FLOAT)) as sredni_wiek,
    MIN(CAST(wiek AS FLOAT)) as min_wiek,
    MAX(CAST(wiek AS FLOAT)) as max_wiek
FROM pacjenci 
WHERE wiek IS NOT NULL AND wiek != ''
"""
df_wiek = pd.read_sql_query(query2, conn)
print("\n✓ Statystyki wieku:")
print(df_wiek.to_string(index=False))

conn.close()

# =============================================================================
# 4. PRZYGOTOWANIE DANYCH DO ANALIZY (podział na grupy)
# =============================================================================

print("\n4. PRZYGOTOWANIE DANYCH DO ANALIZY...")

# Znajdź podział (pusty wiersz)
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
# 5. FUNKCJE POMOCNICZE
# =============================================================================

def convert_to_numeric(df, columns):
    """Konwertuje kolumny na typ numeryczny"""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
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
    'kreatynina_0,5_1,2_', 'troponina_I__0_7,8__',
    'HGB_12,4_15,2_', 'WBC_4_11_', 'plt_130_450_',
    'hct_38_45_', 'Na_137_145_', 'K_3,5_5,1_', 'crp_0_0,5_'
]

choroby = ['dm', 'wątroba', 'naczyniowe', 'zza', 'npl']

# Konwersja danych
df_hosp = convert_to_numeric(df_hosp, parametry_kliniczne)
df_dom = convert_to_numeric(df_dom, parametry_kliniczne)

# =============================================================================
# 6. ANALIZA STATYSTYCZNA - TABELA WSZYSTKICH PARAMETRÓW
# =============================================================================

print("\n" + "="*80)
print("6. ANALIZA STATYSTYCZNA - WSZYSTKIE PARAMETRY")
print("="*80)

wyniki = []
print("\n{:<30} {:>6} {:>12} {:>12} {:>12}".format(
    "Parametr", "n_hosp", "Hosp (śr±SD)", "Dom (śr±SD)", "p-value"
))
print("-"*85)

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
            
            print("{:<30} {:>3}   {:>6.2f}±{:<5.2f} {:>6.2f}±{:<5.2f}  p={:<.4f} {}".format(
                param[:29], len(hosp), hosp_sr, hosp_std, dom_sr, dom_std, p_value, gwiazdki
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
# 7. RANKING PARAMETRÓW WG ISTOTNOŚCI
# =============================================================================

print("\n" + "="*80)
print("7. RANKING PARAMETRÓW WG ISTOTNOŚCI")
print("="*80)

df_wyniki = pd.DataFrame(wyniki)
df_wyniki = df_wyniki.sort_values('p_value')

print("\n{:<4} {:<30} {:>12} {:>6} {:>12}".format(
    "Rank", "Parametr", "p-value", "Istot.", "Różnica"
))
print("-"*70)

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
    print("{:<4} {:<30} {:>8.4f} {:>6} {:>+8.2f}".format(
        rank, row['parametr'][:29], row['p_value'], poziom, row['roznica']
    ))

# =============================================================================
# 8. ANALIZA CHORÓB WSPÓŁISTNIEJĄCYCH
# =============================================================================

print("\n" + "="*80)
print("8. ANALIZA CHORÓB WSPÓŁISTNIEJĄCYCH")
print("="*80)

wyniki_choroby = []
for choroba in choroby:
    if choroba in df_hosp.columns:
        hosp_val = df_hosp[choroba].apply(convert_choroba)
        dom_val = df_dom[choroba].apply(convert_choroba)
        
        hosp_proc = hosp_val.mean() * 100 if hosp_val.count() > 0 else 0
        dom_proc = dom_val.mean() * 100 if dom_val.count() > 0 else 0
        
        wyniki_choroby.append({
            'choroba': choroba,
            'hosp_proc': hosp_proc,
            'hosp_n': hosp_val.count(),
            'dom_proc': dom_proc,
            'dom_n': dom_val.count(),
            'roznica': hosp_proc - dom_proc
        })
        
        print(f"\n{choroba}:")
        print(f"  Hospitalizowani: {hosp_proc:.1f}% ({int(hosp_val.sum())}/{hosp_val.count()})")
        print(f"  Do domu: {dom_proc:.1f}% ({int(dom_val.sum())}/{dom_val.count()})")

df_chor = pd.DataFrame(wyniki_choroby)

# =============================================================================
# 9. ZAPIS WYNIKÓW DO BAZY SQL
# =============================================================================

print("\n9. ZAPISYWANIE WYNIKÓW DO BAZY SQL...")

def save_results_to_sql(df_wyniki, df_chor, db_file='pacjenci.db'):
    """Zapisuje wyniki analizy do bazy SQL"""
    try:
        conn = sqlite3.connect(db_file)
        
        # Funkcja pomocnicza: oczyszcza nazwy kolumn dla SQL
        def clean_col_names(df):
            df_clean = df.copy()
            new_cols = []
            for col in df_clean.columns:
                col_clean = str(col)
                col_clean = col_clean.replace('(', '_')
                col_clean = col_clean.replace(')', '_')
                col_clean = col_clean.replace('-', '_')
                col_clean = col_clean.replace(' ', '_')
                col_clean = col_clean.replace(',', '_')
                col_clean = col_clean.replace('.', '_')
                new_cols.append(col_clean)
            df_clean.columns = new_cols
            return df_clean
        
        # Oczyść nazwy kolumn
        df_wyniki_clean = clean_col_names(df_wyniki)
        df_wyniki_clean.to_sql('wyniki_parametry', conn, if_exists='replace', index=False)
        print("  ✓ Zapisano tabelę: wyniki_parametry")
        
        if not df_chor.empty:
            df_chor_clean = clean_col_names(df_chor)
            df_chor_clean.to_sql('wyniki_choroby', conn, if_exists='replace', index=False)
            print("  ✓ Zapisano tabelę: wyniki_choroby")
        
        from datetime import datetime
        data_analizy = pd.DataFrame({
            'data_analizy': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'liczba_hosp': [len(df_hosp)],
            'liczba_dom': [len(df_dom)]
        })
        data_analizy.to_sql('metadata', conn, if_exists='replace', index=False)
        print("  ✓ Zapisano tabelę: metadata")
        
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"\n  ✓ Tabele w bazie: {', '.join([t[0] for t in tables])}")
        
        conn.close()
        print("\n✓ WSZYSTKIE WYNIKI ZAPISANE DO BAZY SQL")
        return True
    except Exception as e:
        print(f"✗ Błąd zapisu do bazy: {e}")
        return False

# =============================================================================
# 10. WYKRES DLA MAP
# =============================================================================

print("\n10. GENEROWANIE WYKRESÓW...")

if 'MAP' in df_hosp.columns:
    plt.figure(figsize=(10, 6))
    
    hosp_map = df_hosp['MAP'].dropna()
    dom_map = df_dom['MAP'].dropna()
    
    stat_map, p_value_map = stats.mannwhitneyu(hosp_map, dom_map, alternative='two-sided')
    
    bp = plt.boxplot([hosp_map, dom_map], labels=['Hospitalizowani', 'Do domu'], 
                patch_artist=True, 
                medianprops=dict(color='red', linewidth=2))
    
    bp['boxes'][0].set_facecolor('#ff9999')
    bp['boxes'][1].set_facecolor('#99ccff')
    
    plt.ylabel('MAP (mmHg)')
    plt.title(f'Porównanie MAP (p={p_value_map:.4f})')
    plt.grid(True, alpha=0.3)
    
    x_hosp = np.random.normal(1, 0.05, size=len(hosp_map))
    x_dom = np.random.normal(2, 0.05, size=len(dom_map))
    plt.scatter(x_hosp, hosp_map, alpha=0.6, color='darkred', s=50)
    plt.scatter(x_dom, dom_map, alpha=0.6, color='darkblue', s=50)
    
    plt.tight_layout()
    plt.savefig('wykres_MAP_SQL.png', dpi=150)
    plt.show()
    print("✓ Zapisano: wykres_MAP_SQL.png")

# =============================================================================
# 11. PODSUMOWANIE
# =============================================================================

print("\n" + "="*80)
print("11. PODSUMOWANIE - CZYNNIKI RYZYKA HOSPITALIZACJI")
print("="*80)

print("\n🔴 CZYNNIKI Z ISTOTNYMI RÓŻNICAMI (p < 0.05):\n")

istotne = df_wyniki[df_wyniki['p_value'] < 0.05].sort_values('p_value')
for _, row in istotne.iterrows():
    kierunek = "⬆️ WYŻSZE" if row['roznica'] > 0 else "⬇️ NIŻSZE"
    print(f"  • {row['parametr']}: {kierunek} u hospitalizowanych")
    print(f"    (hosp: {row['hosp_sr']:.2f} vs dom: {row['dom_sr']:.2f}, p={row['p_value']:.4f})")

# =============================================================================
# 12. ZAPIS WYNIKÓW DO CSV
# =============================================================================

print("\n" + "="*80)
print("12. ZAPIS WYNIKÓW")
print("="*80)

df_wyniki.to_csv('wyniki_parametry.csv', sep=';', index=False, decimal=',')
print("✓ Zapisano: wyniki_parametry.csv")

if not df_chor.empty:
    df_chor.to_csv('wyniki_choroby.csv', sep=';', index=False, decimal=',')
    print("✓ Zapisano: wyniki_choroby.csv")

# Zapisz do SQL
save_results_to_sql(df_wyniki, df_chor)

print("\n" + "="*80)
print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE!")
print("="*80)
print("\nPODSUMOWANIE:")
print(f"  • Dane: {len(df_hosp)} hosp + {len(df_dom)} dom = {len(df)} pacjentów")
print(f"  • Istotne parametry: {len(istotne)}")
print(f"  • Baza SQL: pacjenci.db")
print(f"  • Wyniki CSV: 2 pliki")
print(f"  • Wykres: wykres_MAP_SQL.png")
print("="*80)