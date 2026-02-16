# Analiza Danych Medycznych Pacjentów

## 📋 Opis projektu
Projekt analizuje dane medyczne pacjentów, porównując grupę hospitalizowanych 
z grupą wypisanych do domu. Celem jest identyfikacja czynników ryzyka hospitalizacji.

## 📊 Dane
- 51 pacjentów (29 hospitalizowanych, 21 do domu)
- Parametry: wiek, MAP, SpO2, HGB, CRP, choroby współistniejące
- Źródło: plik CSV, baza SQLite

## 🔬 Wyniki
**4 istotne czynniki ryzyka hospitalizacji (p < 0.05):**

| Parametr | Hospitalizowani | Do domu | p-value | Wniosek |
|----------|-----------------|---------|---------|---------|
| HGB (hemoglobina) | 11.73 | 13.97 | 0.0059 | ↓ NIŻSZA u hosp. |
| SpO2 (saturacja) | 93.7% | 96.3% | 0.0066 | ↓ NIŻSZA u hosp. |
| HCT (hematokryt) | 35.8 | 41.7 | 0.0100 | ↓ NIŻSZY u hosp. |
| CRP (stan zapalny) | 7.82 | 3.02 | 0.0453 | ↑ WYŻSZY u hosp. |

## 🛠️ Technologie
- Python 3.13
- Pandas, NumPy
- Matplotlib, Seaborn
- SciPy
- SQLite3

## 📁 Historia wersji

| Plik | Opis |
|------|------|
| `analiza.py` | Pierwsze wczytanie danych |
| `analiza 2.py` | Podstawowa analiza statystyczna |
| `analiza 3.py` | Dodanie testów statystycznych |
| `analiza4.py` | Kompletna analiza z chorobami |
| `analiza 5.py` | Pełne wykresy (6 wykresów) |
| `analiza 6.py` | Dodanie MAP |
| `analiza7.py` | Wersja z SQL |
| `analiza 8.py` | Poprawiony zapis do SQL |
| `Analiza 9.py` | Ostateczna wersja |

## 🚀 Uruchomienie
```bash
pip install pandas numpy matplotlib seaborn scipy
python "Analiza 9.py"