

# -*- coding: utf-8 -*-
"""
PEŁNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 4.0 - z obsługą bazy SQL
Autor: Aneta
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
print("Wersja 4.0 - z bazą danych SQL")
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

# Tworzymy bazę (jeśli nie istnieje)
if not os.path.exists('pacjenci.db'):
    create_database_from_csv('baza_danych_pacjentów_a.csv')
else:
    print("✓ Baza danych już istnieje: pacjenci.db")

# =============================================================================
# 2. POBIERANIE DANYCH Z BAZY SQL
# =============================================================================

print("\n2. POBIERANIE DANYCH Z BAZY SQL...")

def load_from_sql(db_file='pacjenci.db', table='pacjenci'):
    """Pobiera dane z bazy SQLite"""
    try:
        conn = sqlite3.connect(db_file)
        
        # Zapytanie SQL - pobierz wszystko
        query = f"SELECT * FROM {table}"
        df = pd.read_sql_query(query, conn)
        
        conn.close()
        print(f"✓ Pobrano {len(df)} wierszy z bazy SQL")
        return df
    except Exception as e:
        print(f"✗ Błąd pobierania z bazy: {e}")
        return None

# Pobieramy dane
df = load_from_sql()

if df is None:
    print("✗ Nie udało się pobrać danych. Sprawdzam plik CSV...")
    df = pd.read_csv('baza_danych_pacjentów_a.csv', sep=';', encoding='utf-8')
    print("✓ Wczytano z pliku CSV")

# =============================================================================
# 3. PRZYKŁADOWE ZAPYTANIA SQL (opcjonalnie)
# =============================================================================

print("\n3. PRZYKŁADOWE ZAPYTANIA SQL:")

conn = sqlite3.connect('pacjenci.db')

# Zapytanie 1: Liczba pacjentów według płci
query1 = """
SELECT "pleć", COUNT(*) as liczba 
FROM pacjenci 
WHERE "pleć" IS NOT NULL 
GROUP BY "pleć"
"""
df_plec = pd.read_sql_query(query1, conn)
print("\n✓ Liczba pacjentów według płci:")
print(df_plec.to_string(index=False))

# Zapytanie 2: Średni wiek dla hospitalizowanych i do domu
# Najpierw musimy znaleźć podział (to skomplikowane w SQL, więc pokazuję jako przykład)
print("\n✓ Przykład zapytania - statystyki wieku:")
query2 = """
SELECT 
    AVG(CAST(REPLACE(wiek, ',', '.') AS FLOAT)) as sredni_wiek,
    MIN(CAST(REPLACE(wiek, ',', '.') AS FLOAT)) as min_wiek,
    MAX(CAST(REPLACE(wiek, ',', '.') AS FLOAT)) as max_wiek
FROM pacjenci 
WHERE wiek IS NOT NULL AND wiek != ''
"""
df_wiek = pd.read_sql_query(query2, conn)
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
# 6. ANALIZA STATYSTYCZNA - TABELA WSZYSTKICH PARAMETRÓW
# =============================================================================

print("\n" + "="*80)
print("6. ANALIZA STATYSTYCZNA - WSZYSTKIE PARAMETRY")
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
# 7. RANKING PARAMETRÓW WG ISTOTNOŚCI
# =============================================================================

print("\n" + "="*80)
print("7. RANKING PARAMETRÓW WG ISTOTNOŚCI")
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
# 8. ZAPIS WYNIKÓW DO BAZY SQL
# =============================================================================

print("\n8. ZAPISYWANIE WYNIKÓW DO BAZY SQL...")

def save_results_to_sql(df_wyniki, df_chor, db_file='pacjenci.db'):
    """Zapisuje wyniki analizy do bazy SQL"""
    try:
        conn = sqlite3.connect(db_file)
        
        # Zapisz wyniki parametrów
        df_wyniki.to_sql('wyniki_parametry', conn, if_exists='replace', index=False)
        
        # Zapisz wyniki chorób
        df_chor.to_sql('wyniki_choroby', conn, if_exists='replace', index=False)
        
        # Dodatkowo zapisz datę analizy
        from datetime import datetime
        data_analizy = pd.DataFrame({
            'data': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'liczba_hosp': [len(df_hosp)],
            'liczba_dom': [len(df_dom)]
        })
        data_analizy.to_sql('metadata', conn, if_exists='replace', index=False)
        
        conn.close()
        print("✓ Wyniki zapisane do bazy SQL")
        return True
    except Exception as e:
        print(f"✗ Błąd zapisu do bazy: {e}")
        return False

# =============================================================================
# 9. WYKRESY (skrócona wersja, ale można rozwinąć jak wcześniej)
# =============================================================================

print("\n9. GENEROWANIE WYKRESÓW...")

# Prosty wykres dla MAP jeśli istnieje
if 'MAP' in df_hosp.columns:
    plt.figure(figsize=(10, 6))
    
    hosp_map = df_hosp['MAP'].dropna()
    dom_map = df_dom['MAP'].dropna()
    
    plt.boxplot([hosp_map, dom_map], labels=['Hospitalizowani', 'Do domu'], 
                patch_artist=True, 
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
    
    plt.ylabel('MAP (mmHg)')
    plt.title('Porównanie MAP - dane z bazy SQL')
    plt.grid(True, alpha=0.3)
    
    # Dodaj punkty
    x_hosp = np.random.normal(1, 0.05, size=len(hosp_map))
    x_dom = np.random.normal(2, 0.05, size=len(dom_map))
    plt.scatter(x_hosp, hosp_map, alpha=0.6, color='blue', s=50)
    plt.scatter(x_dom, dom_map, alpha=0.6, color='green', s=50)
    
    plt.tight_layout()
    plt.savefig('wykres_MAP_SQL.png', dpi=150)
    plt.show()
    print("✓ Zapisano: wykres_MAP_SQL.png")

# =============================================================================
# 10. PODSUMOWANIE
# =============================================================================

print("\n" + "="*80)
print("10. PODSUMOWANIE - CZYNNIKI RYZYKA HOSPITALIZACJI")
print("="*80)

print("\n🔴 CZYNNIKI ZWIĘKSZAJĄCE RYZYKO HOSPITALIZACJI:\n")

istotne = df_wyniki[df_wyniki['p_value'] < 0.05].sort_values('p_value')
for _, row in istotne.iterrows():
    kierunek = "⬆️ WYŻSZE" if row['roznica'] > 0 else "⬇️ NIŻSZE"
    print(f"  • {row['parametr']}: {kierunek} u hospitalizowanych")
    print(f"    (hosp: {row['hosp_sr']:.2f} vs dom: {row['dom_sr']:.2f}, p={row['p_value']:.4f})")

# =============================================================================
# 11. ZAPIS WSZYSTKICH WYNIKÓW
# =============================================================================

print("\n" + "="*80)
print("11. ZAPIS WYNIKÓW")
print("="*80)

# Zapisz do CSV
df_wyniki.to_csv('wyniki_parametry.csv', sep=';', index=False, decimal=',')
print("✓ Zapisano: wyniki_parametry.csv")

# Zapisz do SQL
save_results_to_sql(df_wyniki, pd.DataFrame())  # df_chor do uzupełnienia

print("\n" + "="*80)
print("✅ ANALIZA Z SQL ZAKOŃCZONA POMYŚLNIE!")
print("="*80)
print("\nCo zostało zrobione:")
print("  • Utworzono bazę danych SQLite z pliku CSV")
print("  • Pobrano dane z bazy SQL")
print("  • Wykonano przykładowe zapytania SQL")
print("  • Przeprowadzono analizę statystyczną")
print("  • Zapisano wyniki z powrotem do bazy SQL")
print("  • Wygenerowano wykresy")