# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 17:19:25 2026

@author: aneta
"""

# -*- coding: utf-8 -*-
"""
PEŁNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 6.0 - optymalne połączenie z bazą SQL
Autor: Aneta
"""

import os
import sqlite3
import warnings
from datetime import datetime
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# Ustawienia wykresów
plt.style.use('ggplot')
sns.set_palette('Set2')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

print('=' * 80)
print('PEŁNA ANALIZA DANYCH PACJENTÓW - HOSPITALIZOWANI vs DO DOMU')
print('Wersja 6.0 - optymalne połączenie z bazą SQL')
print('=' * 80)


def convert_to_numeric(data, columns):
    """Konwertuje kolumny na typ numeryczny"""
    for col in columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    return data


def convert_choroba(value):
    """Konwertuje różne formy zapisu chorób"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ['tak', 't', 'yes', 'y', '1', 'true', '+', 'tak!']:
            return 1
        if value_lower in ['nie', 'n', 'no', '0', 'false', '-']:
            return 0
    return np.nan


def save_results_to_sql(df_params, df_diseases, hosp_count, dom_count,
                        db_file='pacjenci.db'):
    """Zapisuje wyniki analizy do bazy SQL"""
    try:
        conn = sqlite3.connect(db_file)

        def clean_col_names(data):
            """Oczyszcza nazwy kolumn dla SQL"""
            clean = data.copy()
            new_cols = []
            for col in clean.columns:
                col_clean = str(col)
                for char in ['(', ')', '-', ' ', ',', '.']:
                    col_clean = col_clean.replace(char, '_')
                new_cols.append(col_clean)
            clean.columns = new_cols
            return clean

        clean_params = clean_col_names(df_params)
        clean_params.to_sql('wyniki_parametry', conn,
                            if_exists='replace', index=False)
        print('  ✓ Zapisano tabelę: wyniki_parametry')

        if not df_diseases.empty:
            clean_diseases = clean_col_names(df_diseases)
            clean_diseases.to_sql('wyniki_choroby', conn,
                                  if_exists='replace', index=False)
            print('  ✓ Zapisano tabelę: wyniki_choroby')

        metadata = pd.DataFrame({
            'data_analizy': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'liczba_hosp': [hosp_count],
            'liczba_dom': [dom_count]
        })
        metadata.to_sql('metadata', conn, if_exists='replace', index=False)
        print('  ✓ Zapisano tabelę: metadata')

        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table_names = [t[0] for t in tables]
        print(f'\n  ✓ Tabele w bazie: {", ".join(table_names)}')
        conn.close()
        print('\n✓ WSZYSTKIE WYNIKI ZAPISANE DO BAZY SQL')
        return True
    except Exception as e:
        print(f'✗ Błąd zapisu do bazy: {e}')
        return False


# =============================================================================
# GŁÓWNA CZĘŚĆ PROGRAMU - JEDNO POŁĄCZENIE DO BAZY!
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # 1. TWORZENIE BAZY DANYCH SQL Z PLIKU CSV
    # -------------------------------------------------------------------------
    print("\n1. TWORZENIE BAZY DANYCH SQL...")

    # Wczytaj CSV
    df_csv = pd.read_csv('baza_danych_pacjentów_a.csv', sep=';', encoding='utf-8')

    # Oczyszczenie nazw kolumn (SQL nie lubi nawiasów i myślników)
    df_csv.columns = [
        col.replace('(', '_').replace(')', '_')
        .replace('-', '_').replace(' ', '_')
        for col in df_csv.columns
    ]

    # JEDNO połączenie do WSZYSTKIEGO
    conn = sqlite3.connect('pacjenci.db')

    # Zapisz dane do bazy
    df_csv.to_sql('pacjenci', conn, if_exists='replace', index=False)
    print("✓ Utworzono bazę danych: pacjenci.db")

    # -------------------------------------------------------------------------
    # 2. PRZYKŁADOWE ZAPYTANIA SQL (na tym samym połączeniu!)
    # -------------------------------------------------------------------------
    print("\n2. PRZYKŁADOWE ZAPYTANIA SQL:")

    # Zapytanie 1: Liczba pacjentów według płci
    QUERY1 = """
    SELECT plec, COUNT(*) as liczba
    FROM pacjenci
    WHERE plec IS NOT NULL AND plec != ''
    GROUP BY plec
    """
    df_plec = pd.read_sql_query(QUERY1, conn)
    print("\n✓ Liczba pacjentów według płci:")
    print(df_plec.to_string(index=False))

    # Zapytanie 2: Statystyki wieku
    QUERY2 = """
    SELECT
        AVG(CAST(wiek AS FLOAT)) as sredni_wiek,
        MIN(CAST(wiek AS FLOAT)) as min_wiek,
        MAX(CAST(wiek AS FLOAT)) as max_wiek
    FROM pacjenci
    WHERE wiek IS NOT NULL AND wiek != ''
    """
    df_wiek = pd.read_sql_query(QUERY2, conn)
    print("\n✓ Statystyki wieku:")
    print(df_wiek.to_string(index=False))

    # -------------------------------------------------------------------------
    # 3. POBIERANIE DANYCH DO ANALIZY (ciągle to samo połączenie!)
    # -------------------------------------------------------------------------
    print("\n3. POBIERANIE DANYCH DO ANALIZY...")

    df = pd.read_sql_query("SELECT * FROM pacjenci", conn)

    # -------------------------------------------------------------------------
    # 4. PRZYGOTOWANIE DANYCH (podział na grupy)
    # -------------------------------------------------------------------------
    print("\n4. PRZYGOTOWANIE DANYCH DO ANALIZY...")

    # Znajdź podział (pusty wiersz)
    empty_rows = df[df.isna().all(axis=1)]
    if len(empty_rows) > 0:
        split_idx = empty_rows.index[0]
        df_hosp = df.iloc[:split_idx].copy().dropna(how='all')
        df_dom = df.iloc[split_idx + 1:].copy().dropna(how='all')
        print(f"✓ Hospitalizowani: {len(df_hosp)} pacjentów")
        print(f"✓ Do domu: {len(df_dom)} pacjentów")
    else:
        print("✗ Nie znaleziono podziału")
        conn.close()
        exit()

    # -------------------------------------------------------------------------
    # 5. KONIEC – zamykamy połączenie z bazą (wykorzystane do wszystkiego)
    # -------------------------------------------------------------------------
    conn.close()
    print("✓ Zamknięto połączenie z bazą danych")

    # Listy parametrów
    CLINICAL_PARAMS = [
        'wiek', 'RR', 'MAP', 'SpO2', 'AS', 'mleczany',
        'kreatynina_0,5_1,2_', 'troponina_I__0_7,8__',
        'HGB_12,4_15,2_', 'WBC_4_11_', 'plt_130_450_',
        'hct_38_45_', 'Na_137_145_', 'K_3,5_5,1_', 'crp_0_0,5_'
    ]

    DISEASES = ['dm', 'wątroba', 'naczyniowe', 'zza', 'npl']

    # Konwersja danych
    df_hosp = convert_to_numeric(df_hosp, CLINICAL_PARAMS)
    df_dom = convert_to_numeric(df_dom, CLINICAL_PARAMS)

    # =========================================================================
    # 6. ANALIZA STATYSTYCZNA - TABELA WSZYSTKICH PARAMETRÓW
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. ANALIZA STATYSTYCZNA - WSZYSTKIE PARAMETRY")
    print("=" * 80)

    results = []
    print("\n{:<30} {:>6} {:>12} {:>12} {:>12}".format(
        "Parametr", "n_hosp", "Hosp (śr±SD)", "Dom (śr±SD)", "p-value"
    ))
    print("-" * 85)

    for param in CLINICAL_PARAMS:
        if param in df_hosp.columns:
            hosp = df_hosp[param].dropna()
            dom = df_dom[param].dropna()

            if len(hosp) > 0 and len(dom) > 0:
                hosp_mean = hosp.mean()
                hosp_std = hosp.std()
                dom_mean = dom.mean()
                dom_std = dom.std()
                diff = hosp_mean - dom_mean

                stat, p_val = stats.mannwhitneyu(hosp, dom,
                                                 alternative='two-sided')

                if p_val < 0.001:
                    stars = "***"
                elif p_val < 0.01:
                    stars = "**"
                elif p_val < 0.05:
                    stars = "*"
                else:
                    stars = "ns"

                print(
                    f"{param[:29]:<30} {len(hosp):>3}   "
                    f"{hosp_mean:>6.2f}±{hosp_std:<5.2f} "
                    f"{dom_mean:>6.2f}±{dom_std:<5.2f}  "
                    f"p={p_val:<.4f} {stars}"
                )

                results.append({
                    'parametr': param,
                    'hosp_n': len(hosp),
                    'hosp_sr': hosp_mean,
                    'hosp_std': hosp_std,
                    'dom_n': len(dom),
                    'dom_sr': dom_mean,
                    'dom_std': dom_std,
                    'roznica': diff,
                    'p_value': p_val,
                    'istotnosc': stars
                })

    # =========================================================================
    # 7. RANKING PARAMETRÓW WG ISTOTNOŚCI
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. RANKING PARAMETRÓW WG ISTOTNOŚCI")
    print("=" * 80)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('p_value')

    print("\n{:<4} {:<30} {:>12} {:>6} {:>12}".format(
        "Rank", "Parametr", "p-value", "Istot.", "Różnica"
    ))
    print("-" * 70)

    for idx, row in df_results.iterrows():
        rank = idx + 1
        if row['p_value'] < 0.001:
            level = "***"
        elif row['p_value'] < 0.01:
            level = "**"
        elif row['p_value'] < 0.05:
            level = "*"
        else:
            level = "ns"

        sign = "+" if row['roznica'] > 0 else "-"
        print(
            f"{rank:<4} {row['parametr'][:29]:<30} "
            f"{row['p_value']:>8.4f} {level:>6} "
            f"{sign}{abs(row['roznica']):>+7.2f}"
        )

    # =========================================================================
    # 8. ANALIZA CHORÓB WSPÓŁISTNIEJĄCYCH
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. ANALIZA CHORÓB WSPÓŁISTNIEJĄCYCH")
    print("=" * 80)

    disease_results = []
    for disease in DISEASES:
        if disease in df_hosp.columns:
            hosp_vals = df_hosp[disease].apply(convert_choroba)
            dom_vals = df_dom[disease].apply(convert_choroba)

            hosp_pct = (hosp_vals.mean() * 100
                        if hosp_vals.count() > 0 else 0)
            dom_pct = (dom_vals.mean() * 100
                       if dom_vals.count() > 0 else 0)

            disease_results.append({
                'choroba': disease,
                'hosp_proc': hosp_pct,
                'hosp_n': hosp_vals.count(),
                'dom_proc': dom_pct,
                'dom_n': dom_vals.count(),
                'roznica': hosp_pct - dom_pct
            })

            print(f"\n{disease}:")
            hosp_sum = int(hosp_vals.sum()) if hosp_vals.count() > 0 else 0
            dom_sum = int(dom_vals.sum()) if dom_vals.count() > 0 else 0
            print(
                f"  Hospitalizowani: {hosp_pct:.1f}% "
                f"({hosp_sum}/{hosp_vals.count()})"
            )
            print(
                f"  Do domu: {dom_pct:.1f}% "
                f"({dom_sum}/{dom_vals.count()})"
            )

    df_disease = pd.DataFrame(disease_results)

    # =========================================================================
    # 9. WYKRES DLA MAP
    # =========================================================================
    print("\n10. GENEROWANIE WYKRESÓW...")

    if 'MAP' in df_hosp.columns:
        plt.figure(figsize=(10, 6))

        hosp_map = df_hosp['MAP'].dropna()
        dom_map = df_dom['MAP'].dropna()

        stat_map, p_val_map = stats.mannwhitneyu(
            hosp_map, dom_map, alternative='two-sided'
        )

        bp = plt.boxplot(
            [hosp_map, dom_map],
            labels=['Hospitalizowani', 'Do domu'],
            patch_artist=True,
            medianprops={'color': 'red', 'linewidth': 2}
        )

        bp['boxes'][0].set_facecolor('#ff9999')
        bp['boxes'][1].set_facecolor('#99ccff')

        plt.ylabel('MAP (mmHg)')
        plt.title(f'Porównanie MAP (p={p_val_map:.4f})')
        plt.grid(True, alpha=0.3)

        x_hosp = np.random.normal(1, 0.05, size=len(hosp_map))
        x_dom = np.random.normal(2, 0.05, size=len(dom_map))
        plt.scatter(x_hosp, hosp_map, alpha=0.6, color='darkred', s=50)
        plt.scatter(x_dom, dom_map, alpha=0.6, color='darkblue', s=50)

        plt.tight_layout()
        plt.savefig('wykres_MAP_SQL.png', dpi=150)
        plt.show()
        print("✓ Zapisano: wykres_MAP_SQL.png")

    # =========================================================================
    # 10. PODSUMOWANIE - CZYNNIKI RYZYKA HOSPITALIZACJI
    # =========================================================================
    print("\n" + "=" * 80)
    print("11. PODSUMOWANIE - CZYNNIKI RYZYKA HOSPITALIZACJI")
    print("=" * 80)

    print("\n🔴 CZYNNIKI Z ISTOTNYMI RÓŻNICAMI (p < 0.05):\n")

    significant = df_results[df_results['p_value'] < 0.05].sort_values('p_value')
    for _, row in significant.iterrows():
        direction = "⬆️ WYŻSZE" if row['roznica'] > 0 else "⬇️ NIŻSZE"
        print(
            f"  • {row['parametr']}: {direction} u hospitalizowanych\n"
            f"    (hosp: {row['hosp_sr']:.2f} vs dom: {row['dom_sr']:.2f}, "
            f"p={row['p_value']:.4f})"
        )

    # =========================================================================
    # 11. ZAPIS WYNIKÓW
    # =========================================================================
    print("\n" + "=" * 80)
    print("12. ZAPIS WYNIKÓW")
    print("=" * 80)

    df_results.to_csv('wyniki_parametry.csv', sep=';',
                      index=False, decimal=',')
    print("✓ Zapisano: wyniki_parametry.csv")

    if not df_disease.empty:
        df_disease.to_csv('wyniki_choroby.csv', sep=';',
                          index=False, decimal=',')
        print("✓ Zapisano: wyniki_choroby.csv")

    # Zapis do SQL (tu tworzymy NOWE połączenie, bo to osobna operacja)
    save_results_to_sql(df_results, df_disease,
                        len(df_hosp), len(df_dom))

    print("\n" + "=" * 80)
    print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE!")
    print("=" * 80)
    print("\nPODSUMOWANIE:")
    print(f"  • Dane: {len(df_hosp)} hosp + {len(df_dom)} dom = "
          f"{len(df)} pacjentów")
    print(f"  • Istotne parametry: {len(significant)}")
    print(f"  • Baza SQL: pacjenci.db")
    print(f"  • Wyniki CSV: 2 pliki")
    print(f"  • Wykres: wykres_MAP_SQL.png")
    print("=" * 80)