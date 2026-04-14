
# -*- coding: utf-8 -*-
"""
PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 21.0 - kompletna, spójna i klinicznie użyteczna
Autor: Aneta
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import fisher_exact

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

from sklearn.metrics import roc_curve, auc, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


# =============================================================================
# KONFIGURACJA
# =============================================================================

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
np.random.seed(42)

KOLORY = {
    "hosp": "#e74c3c",
    "dom": "#3498db",
    "warning": "#f39c12",
}

ZAKRESY_BIOLOGICZNE = {
    "wiek": (0, 120),
    "RR": (0, 300),
    "MAP": (0, 200),
    "SpO2": (0, 100),
    "AS": (0, 300),
    "mleczany": (0, 30),
    "kreatynina(0,5-1,2)": (0, 20),
    "troponina I (0-7,8))": (0, 100000),
    "HGB(12,4-15,2)": (0, 25),
    "WBC(4-11)": (0, 100),
    "plt(130-450)": (0, 2000),
    "hct(38-45)": (0, 70),
    "Na(137-145)": (100, 160),
    "K(3,5-5,1)": (2, 8),
    "crp(0-0,5)": (0, 500),
}

PARAMETRY_KLINICZNE = [
    "wiek", "RR", "MAP", "SpO2", "AS", "mleczany",
    "kreatynina(0,5-1,2)", "troponina I (0-7,8))",
    "HGB(12,4-15,2)", "WBC(4-11)", "plt(130-450)",
    "hct(38-45)", "Na(137-145)", "K(3,5-5,1)", "crp(0-0,5)"
]

CHOROBY = ["dm", "wątroba", "naczyniowe", "zza", "npl"]

ZMIENNE_OBOWIAZKOWE = ["wiek"]
ZMIENNE_DODATKOWE = [
    "SpO2",
    "crp(0-0,5)",
    "kreatynina(0,5-1,2)",
    "MAP",
    "troponina I (0-7,8))",
    "HGB(12,4-15,2)",
]

ZMIENNE_LOG = [
    "crp(0-0,5)",
    "troponina I (0-7,8))",
    "kreatynina(0,5-1,2)",
]


# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================

def wczytaj_dane(sciezka_pliku: str, separator: str = ";") -> pd.DataFrame | None:
    """Wczytuje dane z pliku CSV."""
    try:
        df = pd.read_csv(sciezka_pliku, sep=separator, encoding="utf-8")
        print(f"✓ Wczytano plik: {os.path.basename(sciezka_pliku)}")
        print(f"  Liczba wierszy: {len(df)}")
        print(f"  Liczba kolumn: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"✗ Błąd wczytywania: {e}")
        return None


def przygotuj_dane_z_outcome(df: pd.DataFrame, id_pacjenta: str | None = None):
    """
    Przygotowuje dane z kolumną outcome.
    outcome: 1 = hospitalizowani, 0 = do domu
    """
    df_copy = df.copy()

    if "outcome" not in df_copy.columns:
        print("\n" + "=" * 70)
        print("❌ BRAK KOLUMNY 'outcome' W PLIKU!")
        print("=" * 70)
        return None, None, None

    df_copy = df_copy[df_copy["outcome"].notna()].copy()
    df_copy["outcome"] = pd.to_numeric(df_copy["outcome"], errors="coerce")
    df_copy = df_copy[df_copy["outcome"].isin([0, 1])].copy()

    if len(df_copy) == 0:
        print("✗ Brak poprawnych wartości w kolumnie 'outcome'")
        return None, None, None

    if id_pacjenta and id_pacjenta in df_copy.columns:
        duplikaty_id = df_copy[id_pacjenta].duplicated().sum()
        if duplikaty_id > 0:
            print(f"\n⚠️ Znaleziono {duplikaty_id} duplikatów ID pacjenta")
            df_copy = df_copy.drop_duplicates(subset=[id_pacjenta], keep="first").copy()

    df_hosp = df_copy[df_copy["outcome"] == 1].copy()
    df_dom = df_copy[df_copy["outcome"] == 0].copy()
    df_caly = df_copy.copy()

    print(f"\n✓ Podział danych:")
    print(f"  • Hospitalizowani (outcome=1): {len(df_hosp)} ({len(df_hosp)/len(df_caly)*100:.1f}%)")
    print(f"  • Do domu (outcome=0): {len(df_dom)} ({len(df_dom)/len(df_caly)*100:.1f}%)")
    print(f"  • Razem (df_caly): {len(df_caly)}")

    return df_hosp, df_dom, df_caly


def konwertuj_na_numeryczne(df: pd.DataFrame, kolumny: list[str]) -> pd.DataFrame:
    """Konwertuje wskazane kolumny na typ numeryczny."""
    df_copy = df.copy()
    for col in kolumny:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(
                df_copy[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )
    return df_copy


def konwertuj_choroby(df: pd.DataFrame, kolumny: list[str]) -> pd.DataFrame:
    """Konwertuje kolumny z chorobami na wartości 0/1."""
    df_copy = df.copy()

    mapping_tak = {"tak", "t", "yes", "y", "1", "true", "+", "tak!"}
    mapping_nie = {"nie", "n", "no", "0", "false", "-"}

    for col in kolumny:
        if col in df_copy.columns:
            tmp = df_copy[col].astype(str).str.lower().str.strip()
            df_copy[col] = tmp.apply(lambda x: 1 if x in mapping_tak else (0 if x in mapping_nie else np.nan))

    return df_copy


def raport_brakow(df: pd.DataFrame, nazwa: str) -> pd.DataFrame:
    """Raport braków danych."""
    print(f"\n--- RAPORT BRAKÓW: {nazwa} (n={len(df)}) ---")
    raport = []

    for col in df.columns:
        n_brakow = int(df[col].isna().sum())
        proc_brakow = (n_brakow / len(df)) * 100 if len(df) > 0 else 0
        raport.append({
            "kolumna": col,
            "braki": n_brakow,
            "procent_brakow": round(proc_brakow, 2)
        })
        if proc_brakow > 0:
            print(f"  {col:<30} braki: {n_brakow:3d} ({proc_brakow:5.1f}%)")

    return pd.DataFrame(raport)


def walidacja_zakresow_biologicznych(df: pd.DataFrame, zakresy: dict) -> pd.DataFrame:
    """Walidacja biologicznie możliwych zakresów wartości."""
    print("\n" + "=" * 70)
    print("WALIDACJA ZAKRESÓW BIOLOGICZNYCH")
    print("=" * 70)

    wyniki = []
    znaleziono = False

    for col, (min_bio, max_bio) in zakresy.items():
        if col in df.columns:
            dane = df[col].dropna()
            if len(dane) > 0:
                mask = (dane < min_bio) | (dane > max_bio)
                poza = int(mask.sum())
                wyniki.append({
                    "kolumna": col,
                    "n_poza_zakresem": poza,
                    "min_bio": min_bio,
                    "max_bio": max_bio
                })
                if poza > 0:
                    znaleziono = True
                    print(f"\n  ⚠️ {col}: {poza} wartości poza zakresem biologicznym")
                    print(f"     Problemowe: {dane[mask].tolist()}")

    if not znaleziono:
        print("  ✓ Wszystkie wartości w zakresach biologicznych")

    return pd.DataFrame(wyniki)


def cliff_delta(x: pd.Series, y: pd.Series) -> float:
    """Cliff's delta."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0
    try:
        u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        return (2 * u_stat) / (n1 * n2) - 1
    except Exception as e:
        print(f"⚠️ Błąd Cliff's delta: {e}")
        return 0.0


def sprawdz_epv_i_raport(df: pd.DataFrame, zmienne: list[str], outcome: str = "outcome", prog: int = 10):
    """Sprawdza EPV."""
    n_events = int(df[outcome].sum())
    n_vars = len(zmienne)
    epv = n_events / n_vars if n_vars > 0 else 0

    print(f"\n📊 EPV: {epv:.1f} (zdarzeń={n_events}, predyktorów={n_vars})")

    if epv < prog:
        print(f"  ⚠️ EPV < {prog} - model może być niestabilny")
        return False, epv
    print("  ✓ EPV OK")
    return True, epv


def sprawdz_vif(X: pd.DataFrame):
    """Sprawdza VIF."""
    vif_data = pd.DataFrame()
    vif_data["zmienna"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    ostrzezenia = []
    for _, row in vif_data.iterrows():
        if row["zmienna"] != "const":
            if row["VIF"] > 10:
                ostrzezenia.append(f"  ⚠️ {row['zmienna']}: VIF={row['VIF']:.2f} (wysoka)")
            elif row["VIF"] > 5:
                ostrzezenia.append(f"  • {row['zmienna']}: VIF={row['VIF']:.2f} (umiarkowana)")

    return ostrzezenia, vif_data


def raport_missingness_top(df_hosp: pd.DataFrame, df_dom: pd.DataFrame, top_param: list[str]) -> pd.DataFrame:
    """Raport braków dla top parametrów."""
    print("\n" + "=" * 80)
    print("📋 RAPORT BRAKÓW - TOP 5 PARAMETRÓW")
    print("=" * 80)

    wyniki = []
    for param in top_param[:5]:
        if param in df_hosp.columns and param in df_dom.columns:
            brak_hosp = int(df_hosp[param].isna().sum())
            brak_dom = int(df_dom[param].isna().sum())
            proc_hosp = (brak_hosp / len(df_hosp)) * 100 if len(df_hosp) > 0 else 0
            proc_dom = (brak_dom / len(df_dom)) * 100 if len(df_dom) > 0 else 0

            print(f"\n{param}:")
            print(f"  Hospitalizowani: {brak_hosp}/{len(df_hosp)} ({proc_hosp:.1f}%)")
            print(f"  Do domu: {brak_dom}/{len(df_dom)} ({proc_dom:.1f}%)")

            wyniki.append({
                "parametr": param,
                "braki_hosp": brak_hosp,
                "proc_braki_hosp": round(proc_hosp, 2),
                "braki_dom": brak_dom,
                "proc_braki_dom": round(proc_dom, 2),
            })

    return pd.DataFrame(wyniki)


# =============================================================================
# TABELA 1
# =============================================================================

def tabela_1_kompletna(df_hosp: pd.DataFrame, df_dom: pd.DataFrame,
                       parametry_ciagle: list[str], choroby: list[str]) -> pd.DataFrame:
    """Kompletna Tabela 1."""
    print("\n" + "=" * 80)
    print("📊 TABELA 1: CHARAKTERYSTYKA KOHORTY")
    print("=" * 80)
    print(f"  Hospitalizowani: n = {len(df_hosp)}")
    print(f"  Do domu: n = {len(df_dom)}")
    print(f"  RAZEM: n = {len(df_hosp) + len(df_dom)}")

    wyniki = []

    print("\n--- ZMIENNE CIĄGŁE ---")
    print("{:<25} {:>8} {:>25} {:>8} {:>25} {:>12} {:>10}".format(
        "Parametr", "n_hosp", "Hosp (mediana [IQR])", "n_dom", "Dom (mediana [IQR])", "p-value", "Cliff's d"
    ))
    print("-" * 120)

    for param in parametry_ciagle:
        if param in df_hosp.columns and param in df_dom.columns:
            hosp = df_hosp[param].dropna()
            dom = df_dom[param].dropna()

            if len(hosp) > 0 and len(dom) > 0:
                hosp_stat = f"{hosp.median():.2f} [{hosp.quantile(0.25):.2f}-{hosp.quantile(0.75):.2f}]"
                dom_stat = f"{dom.median():.2f} [{dom.quantile(0.25):.2f}-{dom.quantile(0.75):.2f}]"

                _, p = stats.mannwhitneyu(hosp, dom)
                d = cliff_delta(hosp, dom)

                wyniki.append({
                    "typ": "ciągły",
                    "parametr": param,
                    "hosp_n": len(hosp),
                    "hosp_stat": hosp_stat,
                    "dom_n": len(dom),
                    "dom_stat": dom_stat,
                    "p_value": p,
                    "effect_size": d
                })

                print("{:<25} {:>3}   {:25} {:>3}   {:25}   {:<8.4f}   {:>6.2f}".format(
                    param[:24], len(hosp), hosp_stat, len(dom), dom_stat, p, d
                ))

    print("\n--- ZMIENNE KATEGORIALNE ---")
    print("{:<25} {:>25} {:>25} {:>12} {:>10}".format(
        "Choroba", "Hospitalizowani", "Do domu", "p-value", "OR"
    ))
    print("-" * 100)

    for choroba in choroby:
        if choroba in df_hosp.columns and choroba in df_dom.columns:
            hosp = df_hosp[choroba].dropna()
            dom = df_dom[choroba].dropna()

            if len(hosp) > 0 and len(dom) > 0:
                hosp_tak = int((hosp == 1).sum())
                dom_tak = int((dom == 1).sum())

                hosp_stat = f"{hosp_tak}/{len(hosp)} ({hosp_tak/len(hosp)*100:.1f}%)"
                dom_stat = f"{dom_tak}/{len(dom)} ({dom_tak/len(dom)*100:.1f}%)"

                a = hosp_tak + 0.5
                b = len(hosp) - hosp_tak + 0.5
                c = dom_tak + 0.5
                d = len(dom) - dom_tak + 0.5
                oddsratio = (a * d) / (b * c)

                tabela = [[hosp_tak, len(hosp) - hosp_tak], [dom_tak, len(dom) - dom_tak]]
                _, p = fisher_exact(tabela)

                wyniki.append({
                    "typ": "kategorialny",
                    "parametr": choroba,
                    "hosp_stat": hosp_stat,
                    "dom_stat": dom_stat,
                    "p_value": p,
                    "effect_size": oddsratio
                })

                print("{:<25} {:>25} {:>25}   {:<8.4f}   {:>6.2f}".format(
                    choroba, hosp_stat, dom_stat, p, oddsratio
                ))

    return pd.DataFrame(wyniki)


# =============================================================================
# ANALIZA JEDNOCZYNNIKOWA Z FDR
# =============================================================================

def analiza_jednoczynnikowa(df_caly: pd.DataFrame, parametry: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Analiza jednoczynnikowa z FDR."""
    print("\n" + "=" * 80)
    print("📈 ANALIZA JEDNOCZYNNIKOWA Z FDR")
    print("=" * 80)

    wyniki = []
    p_values_raw = []

    for param in parametry:
        if param in df_caly.columns:
            hosp = df_caly[df_caly["outcome"] == 1][param].dropna()
            dom = df_caly[df_caly["outcome"] == 0][param].dropna()

            if len(hosp) > 0 and len(dom) > 0:
                _, p = stats.mannwhitneyu(hosp, dom)
                d = cliff_delta(hosp, dom)

                wyniki.append({
                    "parametr": param,
                    "p_raw": p,
                    "cliff_delta": d,
                    "n_hosp": len(hosp),
                    "n_dom": len(dom)
                })
                p_values_raw.append(p)

    df_wyniki = pd.DataFrame(wyniki)
    if len(df_wyniki) == 0:
        return df_wyniki, []

    _, p_adjusted, _, _ = multipletests(p_values_raw, method="fdr_bh")
    df_wyniki["p_fdr"] = p_adjusted
    df_wyniki["istotny"] = df_wyniki["p_fdr"] < 0.05
    df_wyniki = df_wyniki.sort_values(["p_fdr", "p_raw"]).reset_index(drop=True)

    print("\n{:<25} {:>10} {:>12} {:>10} {:>8} {:>8}".format(
        "Parametr", "p_raw", "p_FDR", "Cliff's d", "n_hosp", "n_dom"
    ))
    print("-" * 75)

    for _, row in df_wyniki.iterrows():
        print("{:<25} {:>10.4f} {:>12.4f} {:>10.2f} {:>6} {:>6}".format(
            row["parametr"][:24], row["p_raw"], row["p_fdr"],
            row["cliff_delta"], row["n_hosp"], row["n_dom"]
        ))

    istotne = df_wyniki[df_wyniki["istotny"]].sort_values(["p_fdr", "p_raw"])
    if len(istotne) >= 5:
        top5 = istotne.head(5)["parametr"].tolist()
    else:
        top5 = df_wyniki.head(5)["parametr"].tolist()

    return df_wyniki, top5


# =============================================================================
# PROGI KLINICZNE
# =============================================================================

def progi_kliniczne_poprawione(df: pd.DataFrame, top_param: list[str]) -> pd.DataFrame:
    """
    Progi kliniczne eksploracyjne z uwzględnieniem kierunku efektu.
    """
    print("\n" + "=" * 80)
    print("🎯 PROGI KLINICZNE (eksploracyjne, z kierunkiem efektu)")
    print("=" * 80)

    wyniki = []

    for param in top_param[:5]:
        if param not in df.columns:
            continue

        dane = df[[param, "outcome"]].dropna()
        if len(dane) < 10:
            continue

        hosp_med = dane[dane["outcome"] == 1][param].median()
        dom_med = dane[dane["outcome"] == 0][param].median()
        kierunek = "wyższe" if hosp_med > dom_med else "niższe"

        try:
            if kierunek == "wyższe":
                fpr, tpr, progi = roc_curve(dane["outcome"], dane[param])
            else:
                fpr, tpr, progi = roc_curve(dane["outcome"], -dane[param])

            youden = tpr - fpr
            opt_idx = np.argmax(youden)

            if kierunek == "wyższe":
                prog_opt = progi[opt_idx]
                y_pred = (dane[param] >= prog_opt).astype(int)
            else:
                prog_opt = -progi[opt_idx]
                y_pred = (dane[param] <= prog_opt).astype(int)

            tn = int(((y_pred == 0) & (dane["outcome"] == 0)).sum())
            fp = int(((y_pred == 1) & (dane["outcome"] == 0)).sum())
            fn = int(((y_pred == 0) & (dane["outcome"] == 1)).sum())
            tp = int(((y_pred == 1) & (dane["outcome"] == 1)).sum())

            czulosc = tp / (tp + fn) if (tp + fn) > 0 else 0
            swoistosc = tn / (tn + fp) if (tn + fp) > 0 else 0

            print(f"\n{param} (kierunek: {kierunek} = większe ryzyko):")
            print(f"  Próg: {prog_opt:.2f}")
            print(f"  Czułość: {czulosc:.3f}")
            print(f"  Swoistość: {swoistosc:.3f}")

            wyniki.append({
                "parametr": param,
                "kierunek": kierunek,
                "prog": prog_opt,
                "czulosc": czulosc,
                "swoistosc": swoistosc
            })

        except Exception as e:
            print(f"⚠️ Błąd wyznaczania progu dla {param}: {e}")

    return pd.DataFrame(wyniki)


# =============================================================================
# TRANSFORMACJE
# =============================================================================

def przygotuj_zmienne_do_modelu(df: pd.DataFrame, zmienne_obow: list[str],
                                zmienne_dod: list[str], zmienne_log: list[str]):
    """Przygotowuje zmienne do modeli."""
    df_model = df.copy()
    wszystkie = zmienne_obow + zmienne_dod
    dostepne = [z for z in wszystkie if z in df_model.columns]

    for z in zmienne_log:
        if z in df_model.columns:
            nowa = f"log_{z}"
            df_model[nowa] = np.log1p(df_model[z].clip(lower=0))
            if z in dostepne:
                dostepne.remove(z)
                dostepne.append(nowa)

    return df_model, dostepne


# =============================================================================
# MODELE INFERENCYJNE
# =============================================================================

def model_podstawowy(df: pd.DataFrame, outcome: str = "outcome"):
    """Model podstawowy: tylko wiek."""
    print("\n" + "=" * 80)
    print("📊 MODEL 1: PODSTAWOWY (TYLKO WIEK)")
    print("=" * 80)

    if "wiek" not in df.columns:
        print("  ✗ Brak wieku")
        return None, None, None

    df_cc = df[["wiek", outcome]].dropna()
    print(f"  Complete-case: n={len(df_cc)}")

    if len(df_cc) < 10:
        return None, None, None

    X = sm.add_constant(df_cc["wiek"])
    y = df_cc[outcome]

    try:
        model = sm.Logit(y, X).fit(disp=0)

        print(model.summary().tables[1])

        ci = model.conf_int().loc["wiek"]
        wyniki = pd.DataFrame([{
            "parametr": "wiek",
            "OR": np.exp(model.params["wiek"]),
            "ci_low": np.exp(ci[0]),
            "ci_high": np.exp(ci[1]),
            "CI_95%": f"{np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}",
            "p_value": model.pvalues["wiek"]
        }])

        return model, wyniki, len(df_cc)

    except Exception as e:
        print(f"⚠️ Błąd modelu podstawowego: {e}")
        return None, None, None


def model_rozszerzony(df: pd.DataFrame, zmienne: list[str], outcome: str = "outcome"):
    """Model rozszerzony: główny model inferencyjny."""
    print("\n" + "=" * 80)
    print("📊 MODEL 2: ROZSZERZONY (GŁÓWNY)")
    print("=" * 80)

    dostepne = [z for z in zmienne if z in df.columns]
    print(f"  Zmienne: {', '.join(dostepne)}")

    df_cc = df[dostepne + [outcome]].dropna()
    print(f"  Complete-case: n={len(df_cc)}")

    if len(df_cc) < 10:
        return None, None, None, False

    epv_ok, _ = sprawdz_epv_i_raport(df_cc, dostepne)

    X = sm.add_constant(df_cc[dostepne])
    y = df_cc[outcome]

    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        print(model.summary().tables[1])

        ostrzezenia, _ = sprawdz_vif(X)
        for o in ostrzezenia:
            print(o)

        print(f"\n📈 Pseudo R² (McFadden): {model.prsquared:.4f}")
        print(f"📈 AIC: {model.aic:.2f}")

        wyniki = []
        for param in dostepne:
            ci = model.conf_int().loc[param]
            wyniki.append({
                "parametr": param,
                "OR": np.exp(model.params[param]),
                "ci_low": np.exp(ci[0]),
                "ci_high": np.exp(ci[1]),
                "CI_95%": f"{np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}",
                "p_value": model.pvalues[param]
            })

        return model, pd.DataFrame(wyniki), len(df_cc), epv_ok

    except Exception as e:
        print(f"⚠️ Błąd modelu rozszerzonego: {e}")
        return None, None, None, False


def model_z_redukcja(df: pd.DataFrame, zmienne: list[str], outcome: str = "outcome"):
    """
    Model redukowany jako sensitivity analysis przy niskim EPV.
    """
    print("\n" + "=" * 80)
    print("📊 MODEL 3: Z REDUKCJĄ (SENSITIVITY ANALYSIS)")
    print("=" * 80)

    dostepne = [z for z in zmienne if z in df.columns]
    df_cc = df[dostepne + [outcome]].dropna()

    n_events = int(df_cc[outcome].sum())
    max_pred = int(n_events / 10)

    if max_pred < 1:
        print("  ✗ Za mało zdarzeń do modelu redukowanego")
        return None, None, None, False

    print(f"  Maksymalna liczba predyktorów (EPV≥10): {max_pred}")

    if len(dostepne) <= max_pred:
        print("  ✓ EPV OK - używam modelu rozszerzonego")
        return model_rozszerzony(df, zmienne, outcome)

    priorytety = {
        "wiek": 10,
        "log_crp(0-0,5)": 9,
        "SpO2": 8,
        "log_kreatynina(0,5-1,2)": 7,
        "MAP": 6,
        "log_troponina I (0-7,8))": 5,
        "HGB(12,4-15,2)": 4
    }

    dostepne.sort(key=lambda x: priorytety.get(x, 0), reverse=True)
    wybrane = dostepne[:max_pred]

    print(f"  Wybrane: {', '.join(wybrane)}")

    return model_rozszerzony(df, wybrane, outcome)


# =============================================================================
# PORÓWNANIE MODELI
# =============================================================================

def porownaj_modele_z_aic(model1, model2, model3, n1, n2, n3, epv_ok):
    """Porównanie modeli inferencyjnych."""
    print("\n" + "=" * 80)
    print("📊 PORÓWNANIE MODELI INFERENCYJNYCH")
    print("=" * 80)

    print(f"\n{'Model':<20} {'n':<8} {'Zmienne':<15} {'Pseudo R²':<12} {'AIC':<10} {'Status':<12}")
    print("-" * 80)

    if model1 is not None:
        print(f"{'Podstawowy':<20} {n1:<8} {'wiek':<15} {model1.prsquared:.4f}     {model1.aic:.2f}   {'informacyjny':<12}")

    if model2 is not None:
        status = "GŁÓWNY" if epv_ok else "niestabilny"
        print(f"{'Rozszerzony':<20} {n2:<8} {len(model2.params)-1:<15} {model2.prsquared:.4f}     {model2.aic:.2f}   {status:<12}")

    if model3 is not None and model3 is not model2:
        print(f"{'Z redukcją':<20} {n3:<8} {len(model3.params)-1:<15} {model3.prsquared:.4f}     {model3.aic:.2f}   {'sensitivity':<12}")

    if model2 is not None:
        print("\n🎯 WNIOSKI KLINICZNE:")
        print("\n  Model główny (rozszerzony) - czynniki niezależne:")
        for param in model2.params.index:
            if param != "const" and model2.pvalues[param] < 0.05:
                or_val = np.exp(model2.params[param])
                ci = np.exp(model2.conf_int().loc[param])
                print(f"    • {param}: OR={or_val:.2f} (95% CI: {ci[0]:.2f}-{ci[1]:.2f})")


# =============================================================================
# FOREST PLOT
# =============================================================================

def forest_plot(wyniki: pd.DataFrame, nazwa_pliku: str = "forest_plot.png"):
    """Forest plot z OR i 95% CI."""
    if wyniki is None or len(wyniki) == 0:
        return

    if not {"ci_low", "ci_high", "OR"}.issubset(wyniki.columns):
        print("⚠️ Brak kolumn liczbowych CI do forest plot")
        return

    df_plot = wyniki.copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(df_plot))

    ax.errorbar(
        df_plot["OR"], y_pos,
        xerr=[df_plot["OR"] - df_plot["ci_low"], df_plot["ci_high"] - df_plot["OR"]],
        fmt="o", color="darkblue", ecolor="gray", capsize=5, markersize=8
    )

    ax.axvline(x=1, color="red", linestyle="--", alpha=0.7, label="OR = 1")
    ax.set_xscale("log")
    ax.set_xlabel("OR (95% CI) - skala logarytmiczna")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["parametr"])
    ax.set_title("Czynniki ryzyka hospitalizacji")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    for i, (_, row) in enumerate(df_plot.iterrows()):
        ax.text(row["OR"] * 1.1, i, f"{row['OR']:.2f}", verticalalignment="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(nazwa_pliku, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Forest plot: {nazwa_pliku}")


# =============================================================================
# MODEL PREDYKCYJNY
# =============================================================================

def model_predykcyjny(df: pd.DataFrame, zmienne: list[str], outcome: str = "outcome"):
    """Model predykcyjny z poprawnym pipeline."""
    print("\n" + "=" * 80)
    print("🤖 MODEL PREDYKCYJNY (sklearn)")
    print("=" * 80)

    dostepne = [z for z in zmienne if z in df.columns]
    df_cc = df[dostepne + [outcome]].dropna()

    if len(df_cc) < 20:
        print("  ✗ Zbyt mało danych")
        return None

    X = df_cc[dostepne].values
    y = df_cc[outcome].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    auc_boot = []
    for i in range(1000):
        boot_idx = resample(range(len(y_test)), replace=True, random_state=i)
        if len(np.unique(y_test[boot_idx])) < 2:
            continue
        try:
            auc_boot.append(roc_auc_score(y_test[boot_idx], y_pred_prob[boot_idx]))
        except Exception as e:
            if i == 0:
                print(f"⚠️ Bootstrap AUC: {e}")

    auc_ci = (
        np.percentile(auc_boot, 2.5),
        np.percentile(auc_boot, 97.5)
    ) if len(auc_boot) > 0 else (roc_auc, roc_auc)

    brier = brier_score_loss(y_test, y_pred_prob)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

    print(f"\n  AUC: {roc_auc:.3f} (95% CI: {auc_ci[0]:.3f}-{auc_ci[1]:.3f})")
    print(f"  Brier: {brier:.4f}")
    print(f"  CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Krzywa ROC")
    ax1.legend()

    fig2, ax2 = plt.subplots()
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
    ax2.plot(prob_pred, prob_true, marker="o")
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Observed")
    ax2.set_title(f"Kalibracja (Brier={brier:.4f})")

    return {
        "auc": roc_auc,
        "auc_ci": auc_ci,
        "brier": brier,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "fig_roc": fig1,
        "fig_cal": fig2
    }


# =============================================================================
# WYKRESY
# =============================================================================

def wykres_pudelkowy(df_hosp: pd.DataFrame, df_dom: pd.DataFrame, param: str, nazwa: str):
    """Wykres pudełkowy dla pojedynczego parametru."""
    hosp = df_hosp[param].dropna()
    dom = df_dom[param].dropna()

    if len(hosp) == 0 or len(dom) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    _, p = stats.mannwhitneyu(hosp, dom)
    d = cliff_delta(hosp, dom)

    bp = ax1.boxplot([hosp, dom], labels=["HOSP", "DOM"], patch_artist=True, medianprops=dict(color="black"))
    bp["boxes"][0].set_facecolor(KOLORY["hosp"])
    bp["boxes"][1].set_facecolor(KOLORY["dom"])

    ax1.set_title(param)
    ax1.grid(True, alpha=0.3)

    ax2.axis("off")
    text = (
        f"{param}\n"
        f"HOSP: n={len(hosp)}, mediana={hosp.median():.2f}\n"
        f"DOM: n={len(dom)}, mediana={dom.median():.2f}\n"
        f"p = {p:.4f}\n"
        f"Cliff's d = {d:.2f}"
    )
    ax2.text(0.1, 0.5, text, fontsize=12, transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(nazwa, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {nazwa}")


# =============================================================================
# PODSUMOWANIE KOŃCOWE
# =============================================================================

def podsumowanie_koncowe(model1, model2, model3, pred, epv_ok):
    """Końcowe ograniczenia i interpretacja."""
    print("\n" + "=" * 80)
    print("PODSUMOWANIE I OGRANICZENIA ANALIZY")
    print("=" * 80)

    print("\n✅ MOCNE STRONY:")
    print("  • Kompletna Tabela 1 dla zmiennych ciągłych i kategorialnych")
    print("  • Testy nieparametryczne (Mann-Whitney, Fisher)")
    print("  • Korekta FDR dla wielokrotnych porównań")
    print("  • Raport missingness dla kluczowych parametrów")
    print("  • Model podstawowy, rozszerzony i redukowany")
    print("  • Diagnostyka EPV, VIF, pseudo R² i AIC")
    print("  • Transformacje log dla biomarkerów silnie skośnych")
    print("  • Model predykcyjny z poprawnym pipeline bez leakage")
    print("  • ROC, kalibracja i bootstrap CI dla AUC")

    print("\n⚠️ OGRANICZENIA:")
    print("  • Analiza modeli inferencyjnych oparta o complete-case")
    print("  • Brak imputacji braków danych")
    print("  • Progi kliniczne mają charakter eksploracyjny")
    print("  • Brak walidacji zewnętrznej")
    print("  • Regresja logistyczna zakłada liniowość efektu na skali logit")
    print("  • Model predykcyjny nie służy do wnioskowania przyczynowego")

    if model2 is not None and not epv_ok:
        print("\n  ⚠️ Model rozszerzony ma niskie EPV - interpretować ostrożnie.")

    if pred is not None:
        print(f"\n📈 Model predykcyjny: AUC={pred['auc']:.3f}, Brier={pred['brier']:.4f}")

    print("\n" + "=" * 80)


# =============================================================================
# GŁÓWNA FUNKCJA
# =============================================================================

def main(sciezka_pliku: str, id_pacjenta: str | None = None):
    print("\n" + "=" * 80)
    print("🏥 PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH - WERSJA 21.0")
    print(datetime.now().strftime("Data uruchomienia: %Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    # 1. Wczytywanie
    df = wczytaj_dane(sciezka_pliku)
    if df is None:
        return

    df_hosp, df_dom, df_caly = przygotuj_dane_z_outcome(df, id_pacjenta=id_pacjenta)
    if df_hosp is None:
        return

    # 2. Konwersja
    df_hosp = konwertuj_na_numeryczne(df_hosp, PARAMETRY_KLINICZNE)
    df_dom = konwertuj_na_numeryczne(df_dom, PARAMETRY_KLINICZNE)
    df_caly = konwertuj_na_numeryczne(df_caly, PARAMETRY_KLINICZNE)

    df_hosp = konwertuj_choroby(df_hosp, CHOROBY)
    df_dom = konwertuj_choroby(df_dom, CHOROBY)
    df_caly = konwertuj_choroby(df_caly, CHOROBY)

    # 3. Kontrola jakości
    raport_brakow(df_caly, "Pełna kohorta").to_csv("raport_brakow.csv", sep=";", index=False)
    walidacja_zakresow_biologicznych(df_caly, ZAKRESY_BIOLOGICZNE).to_csv(
        "walidacja_zakresow.csv", sep=";", index=False
    )

    # 4. Tabela 1
    tabela = tabela_1_kompletna(df_hosp, df_dom, PARAMETRY_KLINICZNE, CHOROBY)
    tabela.to_csv("tabela_1.csv", sep=";", index=False)

    # 5. Analiza jednoczynnikowa
    wyniki_fdr, top5 = analiza_jednoczynnikowa(df_caly, PARAMETRY_KLINICZNE)
    wyniki_fdr.to_csv("analiza_jednoczynnikowa.csv", sep=";", index=False)

    # 6. Missingness top 5
    raport_missingness_top(df_hosp, df_dom, top5).to_csv("missingness_top5.csv", sep=";", index=False)

    # 7. Progi kliniczne
    progi = progi_kliniczne_poprawione(df_caly, top5)
    if len(progi) > 0:
        progi.to_csv("progi_kliniczne.csv", sep=";", index=False)

    # 8. Transformacje
    df_model, zmienne = przygotuj_zmienne_do_modelu(
        df_caly, ZMIENNE_OBOWIAZKOWE, ZMIENNE_DODATKOWE, ZMIENNE_LOG
    )

    # 9. Modele inferencyjne
    model1, wyn1, n1 = model_podstawowy(df_model)
    model2, wyn2, n2, epv_ok = model_rozszerzony(df_model, zmienne)
    model3, wyn3, n3, _ = model_z_redukcja(df_model, zmienne)

    if wyn1 is not None:
        wyn1.to_csv("model_podstawowy.csv", sep=";", index=False)

    if wyn2 is not None:
        wyn2.to_csv("model_rozszerzony.csv", sep=";", index=False)
        forest_plot(wyn2, "forest_plot_rozszerzony.png")

    if wyn3 is not None and not (wyn2 is not None and wyn3.equals(wyn2)):
        wyn3.to_csv("model_z_redukcja.csv", sep=";", index=False)
        forest_plot(wyn3, "forest_plot_redukcja.png")

    # 10. Porównanie modeli
    porownaj_modele_z_aic(model1, model2, model3, n1, n2, n3, epv_ok)

    # 11. Model predykcyjny
    pred = model_predykcyjny(df_model, zmienne)
    if pred is not None:
        pred["fig_roc"].savefig("krzywa_ROC.png", dpi=300, bbox_inches="tight")
        pred["fig_cal"].savefig("krzywa_kalibracji.png", dpi=300, bbox_inches="tight")
        plt.close("all")

    # 12. Wykresy dla top 5
    print("\n" + "=" * 80)
    print("📈 GENEROWANIE WYKRESÓW")
    print("=" * 80)

    for i, param in enumerate(top5[:5], 1):
        nazwa = f"wykres_{i}_{param}.png".replace("(", "").replace(")", "").replace(" ", "_")
        wykres_pudelkowy(df_hosp, df_dom, param, nazwa)

    # 13. Podsumowanie
    podsumowanie_koncowe(model1, model2, model3, pred, epv_ok)

    print("\n" + "=" * 80)
    print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE")
    print("=" * 80)

    print("\n📁 WYGENEROWANE PLIKI:")
    print("  • raport_brakow.csv")
    print("  • walidacja_zakresow.csv")
    print("  • tabela_1.csv")
    print("  • analiza_jednoczynnikowa.csv")
    print("  • missingness_top5.csv")
    if len(progi) > 0:
        print("  • progi_kliniczne.csv")
    if wyn1 is not None:
        print("  • model_podstawowy.csv")
    if wyn2 is not None:
        print("  • model_rozszerzony.csv")
        print("  • forest_plot_rozszerzony.png")
    if wyn3 is not None and not (wyn2 is not None and wyn3.equals(wyn2)):
        print("  • model_z_redukcja.csv")
        print("  • forest_plot_redukcja.png")
    if pred is not None:
        print("  • krzywa_ROC.png")
        print("  • krzywa_kalibracji.png")
    for i in range(1, min(5, len(top5)) + 1):
        print(f"  • wykres_{i}_*.png")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main("BAZA_DANYCH_PACJENTOW_B.csv", id_pacjenta=None) 