# -*- coding: utf-8 -*-

"""
PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH
Wersja: 22.0 - wersja publikacyjna
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
    "HGB(12,4-15,2)"
]

ZMIENNE_LOG = [
    "crp(0-0,5)",
    "troponina I (0-7,8))",
    "kreatynina(0,5-1,2)"
]

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

ETYKIETY = {
    "wiek": "Wiek, lata",
    "RR": "Częstość oddechów / min",
    "MAP": "Średnie ciśnienie tętnicze, mmHg",
    "SpO2": "Saturacja, %",
    "AS": "Akcja serca / min",
    "mleczany": "Mleczany, mmol/L",
    "kreatynina(0,5-1,2)": "Kreatynina, mg/dL",
    "troponina I (0-7,8))": "Troponina I",
    "HGB(12,4-15,2)": "Hemoglobina, g/dL",
    "WBC(4-11)": "Leukocyty, G/L",
    "plt(130-450)": "Płytki, G/L",
    "hct(38-45)": "Hematokryt, %",
    "Na(137-145)": "Sód, mmol/L",
    "K(3,5-5,1)": "Potas, mmol/L",
    "crp(0-0,5)": "CRP, mg/dL",
    "dm": "Cukrzyca",
    "wątroba": "Choroba wątroby",
    "naczyniowe": "Choroby naczyniowe",
    "zza": "Zespół zakrzepowo-zatorowy / wywiad",
    "npl": "Nowotwór / choroba proliferacyjna",
    "log_crp(0,0-0,5)": "log(CRP)",
    "log_crp(0-0,5)": "log(CRP)",
    "log_kreatynina(0,5-1,2)": "log(kreatynina)",
    "log_troponina I (0-7,8))": "log(troponina I)",
}


# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================

def pretty_name(x: str) -> str:
    return ETYKIETY.get(x, x)


def wczytaj_dane(sciezka_pliku: str, separator: str = ";") -> pd.DataFrame | None:
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
    df_copy = df.copy()

    if "outcome" not in df_copy.columns:
        print("\n❌ BRAK KOLUMNY 'outcome' W PLIKU!")
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
            print(f"⚠️ Usunięto {duplikaty_id} duplikatów ID pacjenta")
            df_copy = df_copy.drop_duplicates(subset=[id_pacjenta], keep="first").copy()

    df_hosp = df_copy[df_copy["outcome"] == 1].copy()
    df_dom = df_copy[df_copy["outcome"] == 0].copy()
    df_caly = df_copy.copy()

    print(f"\n✓ Hospitalizowani: {len(df_hosp)}")
    print(f"✓ Do domu: {len(df_dom)}")
    print(f"✓ Razem: {len(df_caly)}")

    return df_hosp, df_dom, df_caly


def konwertuj_na_numeryczne(df: pd.DataFrame, kolumny: list[str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col in kolumny:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(
                df_copy[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )
    return df_copy


def konwertuj_choroby(df: pd.DataFrame, kolumny: list[str]) -> pd.DataFrame:
    df_copy = df.copy()
    mapping_tak = {"tak", "t", "yes", "y", "1", "true", "+", "tak!"}
    mapping_nie = {"nie", "n", "no", "0", "false", "-"}

    for col in kolumny:
        if col in df_copy.columns:
            tmp = df_copy[col].astype(str).str.lower().str.strip()
            df_copy[col] = tmp.apply(lambda x: 1 if x in mapping_tak else (0 if x in mapping_nie else np.nan))

    return df_copy


def raport_brakow(df: pd.DataFrame, nazwa: str) -> pd.DataFrame:
    print(f"\n--- RAPORT BRAKÓW: {nazwa} (n={len(df)}) ---")
    wyniki = []
    for col in df.columns:
        n_brakow = int(df[col].isna().sum())
        proc = (n_brakow / len(df)) * 100 if len(df) > 0 else 0
        wyniki.append({
            "kolumna": col,
            "braki": n_brakow,
            "procent": round(proc, 2)
        })
        if proc > 0:
            print(f"{col:<30} {n_brakow:>4} ({proc:5.1f}%)")
    return pd.DataFrame(wyniki)


def walidacja_zakresow_biologicznych(df: pd.DataFrame, zakresy: dict) -> pd.DataFrame:
    print("\n--- WALIDACJA ZAKRESÓW BIOLOGICZNYCH ---")
    wyniki = []

    for col, (min_bio, max_bio) in zakresy.items():
        if col in df.columns:
            dane = df[col].dropna()
            if len(dane) > 0:
                mask = (dane < min_bio) | (dane > max_bio)
                n_bad = int(mask.sum())
                wyniki.append({
                    "kolumna": col,
                    "poza_zakresem": n_bad,
                    "min_bio": min_bio,
                    "max_bio": max_bio
                })
                if n_bad > 0:
                    print(f"⚠️ {col}: {n_bad} poza zakresem biologicznym")

    if len(wyniki) == 0:
        print("Brak danych do walidacji.")

    return pd.DataFrame(wyniki)


def cliff_delta(x: pd.Series, y: pd.Series) -> float:
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0
    try:
        u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        return (2 * u_stat) / (n1 * n2) - 1
    except Exception:
        return 0.0


def interpret_cliff_delta(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "mały"
    if ad < 0.33:
        return "umiarkowany"
    return "duży"


def sprawdz_epv_i_raport(df: pd.DataFrame, zmienne: list[str], outcome: str = "outcome", prog: int = 10):
    n_events = int(df[outcome].sum())
    n_vars = len(zmienne)
    epv = n_events / n_vars if n_vars > 0 else 0
    print(f"\nEPV = {epv:.1f} (zdarzeń={n_events}, predyktorów={n_vars})")
    return epv >= prog, epv


def sprawdz_vif(X: pd.DataFrame):
    vif_data = pd.DataFrame()
    vif_data["zmienna"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


# =============================================================================
# TABELA 1
# =============================================================================

def tabela_1_kompletna(df_hosp: pd.DataFrame, df_dom: pd.DataFrame,
                       parametry_ciagle: list[str], choroby: list[str]) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("TABELA 1: CHARAKTERYSTYKA KOHORTY")
    print("=" * 80)

    wyniki = []

    for param in parametry_ciagle:
        if param in df_hosp.columns and param in df_dom.columns:
            hosp = df_hosp[param].dropna()
            dom = df_dom[param].dropna()

            if len(hosp) > 0 and len(dom) > 0:
                p = stats.mannwhitneyu(hosp, dom).pvalue
                d = cliff_delta(hosp, dom)

                wyniki.append({
                    "typ": "ciągła",
                    "parametr": param,
                    "etykieta": pretty_name(param),
                    "hospitalizowani": f"{hosp.median():.2f} [{hosp.quantile(0.25):.2f}-{hosp.quantile(0.75):.2f}]",
                    "wypisani": f"{dom.median():.2f} [{dom.quantile(0.25):.2f}-{dom.quantile(0.75):.2f}]",
                    "n_hosp": len(hosp),
                    "n_dom": len(dom),
                    "p_value": p,
                    "effect_size": d,
                    "interpretacja_efektu": interpret_cliff_delta(d)
                })

    for choroba in choroby:
        if choroba in df_hosp.columns and choroba in df_dom.columns:
            hosp = df_hosp[choroba].dropna()
            dom = df_dom[choroba].dropna()

            if len(hosp) > 0 and len(dom) > 0:
                hosp_tak = int((hosp == 1).sum())
                dom_tak = int((dom == 1).sum())

                tabela = [[hosp_tak, len(hosp) - hosp_tak], [dom_tak, len(dom) - dom_tak]]
                _, p = fisher_exact(tabela)

                a = hosp_tak + 0.5
                b = len(hosp) - hosp_tak + 0.5
                c = dom_tak + 0.5
                d = len(dom) - dom_tak + 0.5
                or_val = (a * d) / (b * c)

                wyniki.append({
                    "typ": "kategorialna",
                    "parametr": choroba,
                    "etykieta": pretty_name(choroba),
                    "hospitalizowani": f"{hosp_tak}/{len(hosp)} ({100*hosp_tak/len(hosp):.1f}%)",
                    "wypisani": f"{dom_tak}/{len(dom)} ({100*dom_tak/len(dom):.1f}%)",
                    "n_hosp": len(hosp),
                    "n_dom": len(dom),
                    "p_value": p,
                    "effect_size": or_val,
                    "interpretacja_efektu": "OR"
                })

    tabela = pd.DataFrame(wyniki)
    print(tabela[["etykieta", "hospitalizowani", "wypisani", "p_value"]].to_string(index=False))
    return tabela


# =============================================================================
# ANALIZA JEDNOCZYNNIKOWA
# =============================================================================

def analiza_jednoczynnikowa(df_caly: pd.DataFrame, parametry: list[str]) -> tuple[pd.DataFrame, list[str]]:
    print("\n" + "=" * 80)
    print("ANALIZA JEDNOCZYNNIKOWA Z FDR")
    print("=" * 80)

    wyniki = []
    p_values = []

    for param in parametry:
        if param in df_caly.columns:
            hosp = df_caly[df_caly["outcome"] == 1][param].dropna()
            dom = df_caly[df_caly["outcome"] == 0][param].dropna()

            if len(hosp) > 0 and len(dom) > 0:
                p = stats.mannwhitneyu(hosp, dom).pvalue
                d = cliff_delta(hosp, dom)

                wyniki.append({
                    "parametr": param,
                    "etykieta": pretty_name(param),
                    "p_raw": p,
                    "cliff_delta": d,
                    "interpretacja_efektu": interpret_cliff_delta(d),
                    "n_hosp": len(hosp),
                    "n_dom": len(dom)
                })
                p_values.append(p)

    df_wyniki = pd.DataFrame(wyniki)
    if len(df_wyniki) == 0:
        return df_wyniki, []

    _, p_fdr, _, _ = multipletests(p_values, method="fdr_bh")
    df_wyniki["p_fdr"] = p_fdr
    df_wyniki["istotny_fdr"] = df_wyniki["p_fdr"] < 0.05
    df_wyniki = df_wyniki.sort_values(["p_fdr", "p_raw"]).reset_index(drop=True)

    top5 = df_wyniki[df_wyniki["istotny_fdr"]].head(5)["parametr"].tolist()
    if len(top5) < 5:
        top5 = df_wyniki.head(5)["parametr"].tolist()

    print(df_wyniki[["etykieta", "p_raw", "p_fdr", "cliff_delta"]].to_string(index=False))
    return df_wyniki, top5


# =============================================================================
# RAPORT BRAKÓW DLA TOP 5
# =============================================================================

def raport_missingness_top(df_hosp: pd.DataFrame, df_dom: pd.DataFrame, top_param: list[str]) -> pd.DataFrame:
    wyniki = []
    for param in top_param[:5]:
        if param in df_hosp.columns and param in df_dom.columns:
            b1 = int(df_hosp[param].isna().sum())
            b0 = int(df_dom[param].isna().sum())
            wyniki.append({
                "parametr": param,
                "etykieta": pretty_name(param),
                "braki_hosp": b1,
                "proc_hosp": round(100 * b1 / len(df_hosp), 2) if len(df_hosp) else 0,
                "braki_dom": b0,
                "proc_dom": round(100 * b0 / len(df_dom), 2) if len(df_dom) else 0
            })
    return pd.DataFrame(wyniki)


# =============================================================================
# PROGI KLINICZNE
# =============================================================================

def progi_kliniczne_poprawione(df: pd.DataFrame, top_param: list[str]) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("PROGI KLINICZNE - ANALIZA EKSPLORACYJNA")
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
                fpr, tpr, thresholds = roc_curve(dane["outcome"], dane[param])
            else:
                fpr, tpr, thresholds = roc_curve(dane["outcome"], -dane[param])

            youden = tpr - fpr
            idx = int(np.argmax(youden))

            if kierunek == "wyższe":
                prog = thresholds[idx]
                y_pred = (dane[param] >= prog).astype(int)
            else:
                prog = -thresholds[idx]
                y_pred = (dane[param] <= prog).astype(int)

            tn = int(((y_pred == 0) & (dane["outcome"] == 0)).sum())
            fp = int(((y_pred == 1) & (dane["outcome"] == 0)).sum())
            fn = int(((y_pred == 0) & (dane["outcome"] == 1)).sum())
            tp = int(((y_pred == 1) & (dane["outcome"] == 1)).sum())

            sens = tp / (tp + fn) if (tp + fn) else 0
            spec = tn / (tn + fp) if (tn + fp) else 0

            wyniki.append({
                "parametr": param,
                "etykieta": pretty_name(param),
                "kierunek": kierunek,
                "prog": prog,
                "czulosc": sens,
                "swoistosc": spec
            })

        except Exception as e:
            print(f"⚠️ Błąd progu dla {param}: {e}")

    return pd.DataFrame(wyniki)


# =============================================================================
# TRANSFORMACJE
# =============================================================================

def przygotuj_zmienne_do_modelu(df: pd.DataFrame, zmienne_obow: list[str],
                                zmienne_dod: list[str], zmienne_log: list[str]):
    df_model = df.copy()
    wszystkie = zmienne_obow + zmienne_dod
    dostepne = [z for z in wszystkie if z in df_model.columns]

    for z in zmienne_log:
        if z in df_model.columns:
            new_name = f"log_{z}"
            df_model[new_name] = np.log1p(df_model[z].clip(lower=0))
            if z in dostepne:
                dostepne.remove(z)
                dostepne.append(new_name)

    return df_model, dostepne


# =============================================================================
# MODELE INFERENCYJNE
# =============================================================================

def _wyniki_modelu_statsmodels(model, zmienne: list[str]) -> pd.DataFrame:
    rows = []
    for var in zmienne:
        ci = model.conf_int().loc[var]
        rows.append({
            "parametr": var,
            "etykieta": pretty_name(var),
            "beta": model.params[var],
            "OR": np.exp(model.params[var]),
            "ci_low": np.exp(ci[0]),
            "ci_high": np.exp(ci[1]),
            "CI_95%": f"{np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}",
            "p_value": model.pvalues[var]
        })
    return pd.DataFrame(rows)


def model_podstawowy(df: pd.DataFrame, outcome: str = "outcome"):
    print("\n" + "=" * 80)
    print("MODEL 1: PODSTAWOWY (TYLKO WIEK)")
    print("=" * 80)

    if "wiek" not in df.columns:
        return None, None, None

    df_cc = df[["wiek", outcome]].dropna()
    if len(df_cc) < 10:
        return None, None, None

    X = sm.add_constant(df_cc["wiek"])
    y = df_cc[outcome]

    try:
        model = sm.Logit(y, X).fit(disp=0)
        wyn = _wyniki_modelu_statsmodels(model, ["wiek"])
        print(model.summary().tables[1])
        return model, wyn, len(df_cc)
    except Exception as e:
        print(f"⚠️ Model podstawowy: {e}")
        return None, None, None


def model_rozszerzony(df: pd.DataFrame, zmienne: list[str], outcome: str = "outcome"):
    print("\n" + "=" * 80)
    print("MODEL 2: ROZSZERZONY (GŁÓWNY)")
    print("=" * 80)

    dostepne = [z for z in zmienne if z in df.columns]
    df_cc = df[dostepne + [outcome]].dropna()

    if len(df_cc) < 10:
        return None, None, None, False, None

    epv_ok, epv = sprawdz_epv_i_raport(df_cc, dostepne)

    X = sm.add_constant(df_cc[dostepne])
    y = df_cc[outcome]

    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        vif = sprawdz_vif(X)
        print(model.summary().tables[1])
        print(f"Pseudo R² McFadden: {model.prsquared:.4f}")
        print(f"AIC: {model.aic:.2f}")
        return model, _wyniki_modelu_statsmodels(model, dostepne), len(df_cc), epv_ok, vif
    except Exception as e:
        print(f"⚠️ Model rozszerzony: {e}")
        return None, None, None, False, None


def model_z_redukcja(df: pd.DataFrame, zmienne: list[str], outcome: str = "outcome"):
    print("\n" + "=" * 80)
    print("MODEL 3: Z REDUKCJĄ (SENSITIVITY ANALYSIS)")
    print("=" * 80)

    dostepne = [z for z in zmienne if z in df.columns]
    df_cc = df[dostepne + [outcome]].dropna()
    n_events = int(df_cc[outcome].sum())
    max_pred = int(n_events / 10)

    if max_pred < 1:
        return None, None, None, False, None

    if len(dostepne) <= max_pred:
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

    dostepne = sorted(dostepne, key=lambda x: priorytety.get(x, 0), reverse=True)
    wybrane = dostepne[:max_pred]

    print("Wybrane zmienne:", ", ".join(map(pretty_name, wybrane)))
    return model_rozszerzony(df, wybrane, outcome)


# =============================================================================
# FOREST PLOT
# =============================================================================

def forest_plot(wyniki: pd.DataFrame, nazwa_pliku: str):
    if wyniki is None or len(wyniki) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(wyniki))

    ax.errorbar(
        wyniki["OR"], y_pos,
        xerr=[wyniki["OR"] - wyniki["ci_low"], wyniki["ci_high"] - wyniki["OR"]],
        fmt="o", capsize=4
    )
    ax.axvline(1, linestyle="--")
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wyniki["etykieta"])
    ax.set_xlabel("OR (95% CI)")
    ax.set_title("Niezależne czynniki związane z hospitalizacją")
    plt.tight_layout()
    plt.savefig(nazwa_pliku, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# MODEL PREDYKCYJNY
# =============================================================================

def model_predykcyjny(df: pd.DataFrame, zmienne: list[str], outcome: str = "outcome"):
    print("\n" + "=" * 80)
    print("MODEL PREDYKCYJNY")
    print("=" * 80)

    dostepne = [z for z in zmienne if z in df.columns]
    df_cc = df[dostepne + [outcome]].dropna()

    if len(df_cc) < 20:
        return None

    X = df_cc[dostepne].values
    y = df_cc[outcome].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42))
    ])

    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    brier = brier_score_loss(y_test, y_prob)

    auc_boot = []
    for i in range(1000):
        idx = resample(range(len(y_test)), replace=True, random_state=i)
        if len(np.unique(y_test[idx])) < 2:
            continue
        auc_boot.append(roc_auc_score(y_test[idx], y_prob[idx]))

    auc_ci = (
        np.percentile(auc_boot, 2.5),
        np.percentile(auc_boot, 97.5)
    ) if len(auc_boot) else (roc_auc, roc_auc)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    print(f"AUC: {roc_auc:.3f} (95% CI {auc_ci[0]:.3f}-{auc_ci[1]:.3f})")
    print(f"Brier: {brier:.4f}")
    print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_xlabel("1 - swoistość")
    ax1.set_ylabel("Czułość")
    ax1.set_title("Krzywa ROC")
    ax1.legend()

    fig2, ax2 = plt.subplots()
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ax2.plot(prob_pred, prob_true, marker="o")
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_xlabel("Prawdopodobieństwo przewidywane")
    ax2.set_ylabel("Częstość obserwowana")
    ax2.set_title(f"Kalibracja (Brier = {brier:.4f})")

    return {
        "auc": roc_auc,
        "auc_ci_low": auc_ci[0],
        "auc_ci_high": auc_ci[1],
        "brier": brier,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "fig_roc": fig1,
        "fig_cal": fig2
    }


# =============================================================================
# WYKRESY TOP 5
# =============================================================================

def wykres_pudelkowy(df_hosp: pd.DataFrame, df_dom: pd.DataFrame, param: str, nazwa: str):
    hosp = df_hosp[param].dropna()
    dom = df_dom[param].dropna()

    if len(hosp) == 0 or len(dom) == 0:
        return

    p = stats.mannwhitneyu(hosp, dom).pvalue
    d = cliff_delta(hosp, dom)

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([hosp, dom], labels=["Hospitalizowani", "Wypisani"], patch_artist=True)
    bp["boxes"][0].set_facecolor(KOLORY["hosp"])
    bp["boxes"][1].set_facecolor(KOLORY["dom"])
    ax.set_title(f"{pretty_name(param)}\np={p:.4f}, Cliff's d={d:.2f}")
    ax.set_ylabel(pretty_name(param))
    plt.tight_layout()
    plt.savefig(nazwa, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# RAPORT TEKSTOWY
# =============================================================================

def generuj_raport_tekstowy(df_hosp, df_dom, wyn_jedno, wyn_model_glowny, pred, epv_ok) -> str:
    lines = []
    lines.append("WYNIKI")
    lines.append("")
    lines.append(
        f"Do analizy włączono łącznie {len(df_hosp) + len(df_dom)} pacjentów, "
        f"w tym {len(df_hosp)} hospitalizowanych oraz {len(df_dom)} wypisanych do domu."
    )

    if wyn_jedno is not None and len(wyn_jedno) > 0:
        ist = wyn_jedno[wyn_jedno["istotny_fdr"]]
        if len(ist) > 0:
            lines.append("")
            lines.append("W analizie jednoczynnikowej po korekcji FDR istotne różnice obserwowano dla:")
            for _, row in ist.head(5).iterrows():
                lines.append(
                    f"- {row['etykieta']} (p_FDR={row['p_fdr']:.4f}, "
                    f"Cliff's d={row['cliff_delta']:.2f})."
                )

    if wyn_model_glowny is not None and len(wyn_model_glowny) > 0:
        sig = wyn_model_glowny[wyn_model_glowny["p_value"] < 0.05]
        lines.append("")
        if len(sig) > 0:
            lines.append("W modelu wieloczynnikowym niezależnie z hospitalizacją związane były:")
            for _, row in sig.iterrows():
                kier = "większym" if row["OR"] > 1 else "mniejszym"
                lines.append(
                    f"- {row['etykieta']} (OR {row['OR']:.2f}; 95% CI {row['CI_95%']}; p={row['p_value']:.4f}), "
                    f"co wiązało się z {kier} prawdopodobieństwem hospitalizacji."
                )
        else:
            lines.append("W modelu wieloczynnikowym nie wykazano niezależnych czynników przy p<0,05.")

    if pred is not None:
        lines.append("")
        lines.append(
            f"Model predykcyjny osiągnął AUC {pred['auc']:.3f} "
            f"(95% CI {pred['auc_ci_low']:.3f}-{pred['auc_ci_high']:.3f}) "
            f"oraz Brier score {pred['brier']:.4f}."
        )

    lines.append("")
    lines.append("OGRANICZENIA")
    lines.append("")
    lines.append("- Analiza modeli wieloczynnikowych była oparta na complete-case analysis.")
    lines.append("- Progi kliniczne mają charakter eksploracyjny i wymagają walidacji zewnętrznej.")
    lines.append("- Regresja logistyczna zakłada liniowość efektów na skali logit.")
    if not epv_ok:
        lines.append("- Model główny miał ograniczone EPV, dlatego wyniki należy interpretować ostrożnie.")

    return "\n".join(lines)


# =============================================================================
# GŁÓWNA FUNKCJA
# =============================================================================

def main(sciezka_pliku: str, id_pacjenta: str | None = None):
    print("\n" + "=" * 80)
    print("PROFESJONALNA ANALIZA DANYCH MEDYCZNYCH - WERSJA 22.0")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    df = wczytaj_dane(sciezka_pliku)
    if df is None:
        return

    df_hosp, df_dom, df_caly = przygotuj_dane_z_outcome(df, id_pacjenta=id_pacjenta)
    if df_hosp is None:
        return

    df_hosp = konwertuj_na_numeryczne(df_hosp, PARAMETRY_KLINICZNE)
    df_dom = konwertuj_na_numeryczne(df_dom, PARAMETRY_KLINICZNE)
    df_caly = konwertuj_na_numeryczne(df_caly, PARAMETRY_KLINICZNE)

    df_hosp = konwertuj_choroby(df_hosp, CHOROBY)
    df_dom = konwertuj_choroby(df_dom, CHOROBY)
    df_caly = konwertuj_choroby(df_caly, CHOROBY)

    raport_brakow(df_caly, "Pełna kohorta").to_csv("raport_brakow.csv", sep=";", index=False)
    walidacja_zakresow_biologicznych(df_caly, ZAKRESY_BIOLOGICZNE).to_csv(
        "walidacja_zakresow.csv", sep=";", index=False
    )

    tabela1 = tabela_1_kompletna(df_hosp, df_dom, PARAMETRY_KLINICZNE, CHOROBY)
    tabela1.to_csv("tabela_1_publikacyjna.csv", sep=";", index=False)

    wyn_jedno, top5 = analiza_jednoczynnikowa(df_caly, PARAMETRY_KLINICZNE)
    wyn_jedno.to_csv("analiza_jednoczynnikowa_fdr.csv", sep=";", index=False)

    missing_top = raport_missingness_top(df_hosp, df_dom, top5)
    missing_top.to_csv("missingness_top5.csv", sep=";", index=False)

    progi = progi_kliniczne_poprawione(df_caly, top5)
    progi.to_csv("progi_kliniczne_eksploracyjne.csv", sep=";", index=False)

    df_model, zmienne_modelu = przygotuj_zmienne_do_modelu(
        df_caly, ZMIENNE_OBOWIAZKOWE, ZMIENNE_DODATKOWE, ZMIENNE_LOG
    )

    model1, wyn1, n1 = model_podstawowy(df_model)
    model2, wyn2, n2, epv_ok, vif2 = model_rozszerzony(df_model, zmienne_modelu)
    model3, wyn3, n3, _, vif3 = model_z_redukcja(df_model, zmienne_modelu)

    if wyn1 is not None:
        wyn1.to_csv("model_podstawowy.csv", sep=";", index=False)

    if wyn2 is not None:
        wyn2.to_csv("model_glowny_rozszerzony.csv", sep=";", index=False)
        forest_plot(wyn2, "forest_plot_model_glowny.png")

    if wyn3 is not None:
        wyn3.to_csv("model_redukowany_sensitivity.csv", sep=";", index=False)
        forest_plot(wyn3, "forest_plot_model_redukowany.png")

    if vif2 is not None:
        vif2.to_csv("vif_model_glowny.csv", sep=";", index=False)
    if vif3 is not None:
        vif3.to_csv("vif_model_redukowany.csv", sep=";", index=False)

    pred = model_predykcyjny(df_model, zmienne_modelu)
    if pred is not None:
        pred["fig_roc"].savefig("krzywa_ROC.png", dpi=300, bbox_inches="tight")
        pred["fig_cal"].savefig("krzywa_kalibracji.png", dpi=300, bbox_inches="tight")
        plt.close("all")

    for i, param in enumerate(top5[:5], 1):
        nazwa = f"wykres_{i}_{param}.png".replace("(", "").replace(")", "").replace(" ", "_")
        wykres_pudelkowy(df_hosp, df_dom, param, nazwa)

    raport_txt = generuj_raport_tekstowy(df_hosp, df_dom, wyn_jedno, wyn2, pred, epv_ok)
    with open("raport_wyniki_i_ograniczenia.txt", "w", encoding="utf-8") as f:
        f.write(raport_txt)

    print("\n" + "=" * 80)
    print("ANALIZA ZAKOŃCZONA POMYŚLNIE")
    print("=" * 80)
    print("\nWygenerowane pliki:")
    print("• raport_brakow.csv")
    print("• walidacja_zakresow.csv")
    print("• tabela_1_publikacyjna.csv")
    print("• analiza_jednoczynnikowa_fdr.csv")
    print("• missingness_top5.csv")
    print("• progi_kliniczne_eksploracyjne.csv")
    if wyn1 is not None:
        print("• model_podstawowy.csv")
    if wyn2 is not None:
        print("• model_glowny_rozszerzony.csv")
        print("• forest_plot_model_glowny.png")
        print("• vif_model_glowny.csv")
    if wyn3 is not None:
        print("• model_redukowany_sensitivity.csv")
        print("• forest_plot_model_redukowany.png")
        print("• vif_model_redukowany.csv")
    if pred is not None:
        print("• krzywa_ROC.png")
        print("• krzywa_kalibracji.png")
    print("• raport_wyniki_i_ograniczenia.txt")


if __name__ == "__main__":
    main("BAZA_DANYCH_PACJENTOW_B.csv", id_pacjenta=None)