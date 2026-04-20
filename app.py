from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR      = Path(__file__).parent
ARTEFACTS_DIR = BASE_DIR / "model_artefacts"
DATA_PATH     = BASE_DIR / "data" / "raw" / "otodom_all.csv"
SCRAPED_DATE  = "April 2026"

LICZBA_PIETER_DEFAULT = 5  # median floors in building, used when not provided by user

st.set_page_config(
    page_title="Polish Apartment Price Estimator",
    page_icon=None,
    layout="wide",
)

C_INK    = "#0f1729"
C_SLATE  = "#3a4560"
C_STEEL  = "#6b7a99"
C_SILVER = "#9ba8c2"
C_MIST   = "#e4e9f2"
C_SNOW   = "#f4f6fb"
C_WHITE  = "#ffffff"

C_BLUE   = "#1a56db"
C_BLUE_D = "#1045b5"
C_BLUE_L = "#e8eefb"
C_TEAL   = "#0ea271"
C_TEAL_BG = "rgba(14,162,113,0.08)"

FIG_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "figure.dpi": 140,
    "figure.facecolor": C_WHITE,
    "axes.facecolor": C_WHITE,
    "text.color": C_INK,
    "axes.labelcolor": C_STEEL,
    "xtick.color": C_STEEL,
    "ytick.color": C_SLATE,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8.5,
    "font.size": 9,
    "axes.grid": False,
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=Plus+Jakarta+Sans:wght@700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

.stApp {{ background: {C_SNOW} !important; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{
    display: none !important;
    visibility: hidden !important;
}}
.main .block-container {{
    max-width: 1100px !important;
    padding: 2rem 2rem 4rem 2rem !important;
    margin: 0 auto !important;
}}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 0 !important;
    background: transparent !important;
    border-bottom: 2px solid {C_MIST} !important;
    padding-bottom: 0 !important;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    color: {C_STEEL} !important;
    padding: 0.7rem 1.5rem !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 0 !important;
}}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {{ color: {C_SLATE} !important; }}
[data-testid="stTabs"] [aria-selected="true"][data-baseweb="tab"] {{
    color: {C_BLUE} !important;
    font-weight: 700 !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    background-color: {C_BLUE} !important;
    height: 2.5px !important;
    border-radius: 2px 2px 0 0 !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {{ padding-top: 1.75rem !important; }}

label[data-testid="stWidgetLabel"] p {{
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: {C_SLATE} !important;
    margin-bottom: 0.15rem !important;
}}

[data-baseweb="select"],
[data-baseweb="select"] > div,
[data-baseweb="select"] > div > div {{
    background: {C_WHITE} !important;
    background-color: {C_WHITE} !important;
    color: {C_INK} !important;
}}
[data-baseweb="select"] > div {{
    border: 1.5px solid {C_MIST} !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    box-shadow: 0 1px 3px rgba(15,23,41,0.04) !important;
}}
[data-baseweb="select"] > div:focus-within {{
    border-color: {C_BLUE} !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,0.10) !important;
}}
[data-baseweb="popover"],
[data-baseweb="popover"] ul,
[data-baseweb="popover"] [role="listbox"] {{
    background: {C_WHITE} !important;
    background-color: {C_WHITE} !important;
}}
[data-baseweb="popover"] [role="option"] {{
    color: {C_INK} !important;
    background: {C_WHITE} !important;
}}
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] [role="option"]:hover {{
    background: {C_BLUE_L} !important;
    background-color: {C_BLUE_L} !important;
}}

[data-testid="stNumberInput"],
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] > div > div {{ background: transparent !important; }}
[data-testid="stNumberInput"] input,
[data-baseweb="input"],
[data-baseweb="input"] input {{
    border: 1.5px solid {C_MIST} !important;
    border-radius: 10px !important;
    background: {C_WHITE} !important;
    background-color: {C_WHITE} !important;
    color: {C_INK} !important;
    font-size: 13px !important;
    box-shadow: 0 1px 3px rgba(15,23,41,0.04) !important;
}}
[data-baseweb="input"]:focus-within,
[data-testid="stNumberInput"] input:focus {{
    border-color: {C_BLUE} !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,0.10) !important;
    outline: none !important;
}}
[data-testid="stNumberInput"] button {{
    background: {C_WHITE} !important;
    background-color: {C_WHITE} !important;
    border: 1.5px solid {C_MIST} !important;
    color: {C_SLATE} !important;
    border-radius: 8px !important;
}}
[data-testid="stNumberInput"] button:hover {{
    background: {C_SNOW} !important;
    border-color: {C_SILVER} !important;
}}

.stButton {{ padding-top: 1.4rem !important; }}
.stButton > button[kind="primary"] {{
    background: {C_BLUE} !important;
    background-color: {C_BLUE} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 2px 8px rgba(26,86,219,0.25) !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    min-height: 42px !important;
    transition: background-color 0.2s, box-shadow 0.2s, transform 0.1s !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: {C_BLUE_D} !important;
    background-color: {C_BLUE_D} !important;
    box-shadow: 0 4px 14px rgba(26,86,219,0.30) !important;
    transform: translateY(-1px) !important;
}}
.stButton > button[kind="primary"]:active {{ transform: translateY(0) !important; }}

[data-testid="metric-container"] {{
    background: {C_WHITE} !important;
    background-color: {C_WHITE} !important;
    border-radius: 12px !important;
    padding: 1.25rem 1.5rem !important;
    box-shadow: 0 1px 4px rgba(15,23,41,0.06), 0 1px 2px rgba(15,23,41,0.03) !important;
    border: 1px solid {C_MIST} !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.65rem !important;
    color: {C_INK} !important;
    letter-spacing: -0.01em !important;
}}
[data-testid="stMetricLabel"] p {{
    font-size: 10.5px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: {C_STEEL} !important;
}}
[data-testid="stMetricDelta"] {{ font-weight: 600 !important; font-size: 12px !important; }}

.data-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 13px;
}}
.data-table th {{
    background: {C_SNOW};
    padding: 0.75rem 1rem;
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {C_STEEL};
    text-align: left;
    border-bottom: 1.5px solid {C_MIST};
}}
.data-table td {{
    padding: 0.75rem 1rem;
    color: {C_INK};
    border-bottom: 1px solid rgba(228,233,242,0.6);
}}
.data-table tbody tr:last-child td {{ border-bottom: none; }}
.data-table tbody tr:hover td {{ background: {C_BLUE_L}; }}
.table-wrap {{
    background: {C_WHITE};
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(15,23,41,0.06), 0 1px 2px rgba(15,23,41,0.03);
    border: 1px solid {C_MIST};
}}

.chart-label {{
    font-size: 13px;
    font-weight: 700;
    color: {C_INK};
    margin-bottom: 0.6rem;
    padding-left: 0.1rem;
    line-height: 1.4;
}}
.chart-label span {{
    font-weight: 500;
    color: {C_STEEL};
    font-size: 12px;
}}

.section-divider {{
    border: none;
    border-top: 1.5px solid {C_MIST};
    margin: 1.25rem 0 1.75rem 0;
}}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {C_SILVER}; border-radius: 3px; }}

[data-testid="stAlert"] {{
    border-radius: 10px !important;
    border: 1px solid {C_MIST} !important;
    font-size: 13px !important;
}}

.stats-panel {{
    display: flex;
    align-items: center;
    background: {C_WHITE};
    border: 1px solid {C_MIST};
    border-left: 4px solid {C_TEAL};
    border-radius: 12px;
    padding: 1.1rem 2rem;
    margin: 0 auto 0.5rem auto;
    max-width: 600px;
    box-shadow: 0 1px 4px rgba(15,23,41,0.05);
    gap: 0;
}}
.stats-panel .stat {{ flex: 1; text-align: center; padding: 0 0.75rem; }}
.stats-panel .stat + .stat {{ border-left: 1px solid {C_MIST}; }}
.stats-panel .stat-value {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 1.35rem;
    color: {C_BLUE};
    line-height: 1.2;
}}
.stats-panel .stat-label {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {C_STEEL};
    margin-top: 0.15rem;
}}

.about-section {{
    background: {C_WHITE};
    border: 1px solid {C_MIST};
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 4px rgba(15,23,41,0.05);
    font-size: 14px;
    line-height: 1.7;
    color: {C_INK};
}}

@media (max-width: 768px) {{
    [data-testid="column"] {{
        width: 100% !important;
        flex: none !important;
        min-width: 100% !important;
    }}
    .stats-panel {{
        flex-direction: column;
        gap: 0.75rem;
        padding: 1rem 1.25rem;
    }}
    .stats-panel .stat + .stat {{
        border-left: none;
        border-top: 1px solid {C_MIST};
        padding-top: 0.75rem;
    }}
}}
.about-section h3 {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 1.1rem;
    color: {C_INK};
    margin: 0 0 0.75rem 0;
    letter-spacing: -0.01em;
}}
.about-section ul {{ margin: 0.5rem 0; padding-left: 1.25rem; }}
.about-section li {{ margin-bottom: 0.35rem; }}
.about-section code {{
    background: {C_SNOW};
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    font-size: 12.5px;
    color: {C_BLUE};
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load artefacts + data
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artefacts() -> tuple:
    model = joblib.load(ARTEFACTS_DIR / "xgb_model.joblib")
    with open(ARTEFACTS_DIR / "target_enc_city.json", encoding="utf-8") as f:
        enc_city: dict[str, float] = json.load(f)
    with open(ARTEFACTS_DIR / "target_enc_neighborhood.json", encoding="utf-8") as f:
        enc_neighborhood: dict[str, float] = json.load(f)
    with open(ARTEFACTS_DIR / "city_neighborhoods.json", encoding="utf-8") as f:
        city_neighborhoods: dict[str, list[str]] = json.load(f)
    with open(ARTEFACTS_DIR / "meta.json") as f:
        meta: dict = json.load(f)
    return model, enc_city, enc_neighborhood, city_neighborhoods, meta


def _load_model_stats() -> dict:
    for name in ("production_meta.json", "candidate_meta.json"):
        path = ARTEFACTS_DIR / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {"mae": 97_270, "mape": 7.9, "r2": 0.903}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["price", "area_m2"])
    df = df[(df["price"] >= 50_000) & (df["price"] <= 5_000_000)]
    df = df[(df["area_m2"] >= 15) & (df["area_m2"] <= 250)]
    df["price_per_m2"] = df["price"] / df["area_m2"]
    df = df[df["price_per_m2"] <= 40_000]
    df["neighborhood"] = df["neighborhood"].fillna(df["city"])
    return df.reset_index(drop=True)


model, enc_city, enc_neighborhood, city_neighborhoods, meta = load_artefacts()
model_stats  = _load_model_stats()
df_ref       = load_data()
CITIES_SORTED = sorted(city_neighborhoods.keys())
city_median   = df_ref.groupby("city")["price_per_m2"].median().to_dict()

ROK_BUDOWY_RANGE = meta.get("rok_budowy_range", [1950, 2025])
STAN_RAW_OPTIONS = meta.get("ohe_categories", {}).get("stan_wykonczenia", [])

_STAN_LABELS = {
    "ready_to_use":   "Move-in ready",
    "to_completion":  "Needs finishing",
    "shell":          "Shell / raw",
    "high_standard":  "High standard",
    "for_renovation": "For renovation",
}


def _stan_label(code: str) -> str:
    return _STAN_LABELS.get(code, code.replace("_", " ").title())


STAN_OPTIONS = ["— not specified —"] + STAN_RAW_OPTIONS


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _build_input_vector(
    city: str,
    neighborhood: str,
    area: float,
    rooms: int,
    floor: int,
    rok_budowy: int,
    stan_wykonczenia: str,
    winda: bool,
) -> pd.DataFrame:
    city_val = enc_city.get(city, enc_city["__unknown__"])
    nb_val   = enc_neighborhood.get(neighborhood, enc_neighborhood["__unknown__"])

    features = meta.get("features")

    if not features:
        # old meta.json without feature list — fall back to 6-feature baseline
        return pd.DataFrame([{
            "area_m2": area, "rooms": rooms, "floor": floor,
            "city_enc": city_val, "neighborhood_enc": nb_val, "is_private_owner": 0,
        }])

    feat: dict[str, float] = {col: 0 for col in features}
    feat.update({
        "area_m2":          area,
        "rooms":            rooms,
        "floor":            floor,
        "city_enc":         city_val,
        "neighborhood_enc": nb_val,
        "is_private_owner": 0,
        "rok_budowy":       rok_budowy,
        "liczba_pieter":    LICZBA_PIETER_DEFAULT,
        "winda":            int(winda),
    })
    for cat in STAN_RAW_OPTIONS:
        col = f"stan_wykonczenia_{cat}"
        if col in feat:
            feat[col] = 1 if (stan_wykonczenia and cat == stan_wykonczenia) else 0

    return pd.DataFrame([feat])[features]


def predict_price(
    city: str,
    neighborhood: str,
    area: float,
    rooms: int,
    floor: int,
    rok_budowy: int,
    stan_wykonczenia: str,
    winda: bool,
) -> tuple[float, float]:
    X = _build_input_vector(city, neighborhood, area, rooms, floor, rok_budowy, stan_wykonczenia, winda)
    price = float(np.expm1(model.predict(X)[0]))
    return price, price / area


def predict_all_cities(
    area: float, rooms: int, floor: int,
    rok_budowy: int, stan_wykonczenia: str, winda: bool,
) -> pd.DataFrame:
    rows = []
    for c in CITIES_SORTED:
        nb = city_neighborhoods[c][0]
        try:
            price, ppm2 = predict_price(c, nb, area, rooms, floor, rok_budowy, stan_wykonczenia, winda)
            rows.append({"city": c, "price": price, "ppm2": ppm2})
        except Exception:
            pass
    return pd.DataFrame(rows).sort_values("ppm2", ascending=True)


def predict_all_neighborhoods(
    city: str, area: float, rooms: int, floor: int,
    rok_budowy: int, stan_wykonczenia: str, winda: bool,
) -> pd.DataFrame:
    rows = []
    for nb in city_neighborhoods.get(city, [city]):
        try:
            price, ppm2 = predict_price(city, nb, area, rooms, floor, rok_budowy, stan_wykonczenia, winda)
            rows.append({"neighborhood": nb, "price": price, "ppm2": ppm2})
        except Exception:
            pass
    return pd.DataFrame(rows).sort_values("ppm2", ascending=True)


def similar_listings(city, neighborhood, area, rooms, price, n=6):
    pool = df_ref[df_ref["city"] == city].copy()
    if pool.empty:
        return pd.DataFrame()
    pool["score"] = (
        ((pool["area_m2"] - area) / area).abs() * 0.5
        + ((pool["rooms"] - rooms) / rooms).abs() * 0.3
        + ((pool["price"] - price) / price).abs() * 0.2
    )
    top = (
        pool.sort_values("score").head(n)
        [["neighborhood", "area_m2", "rooms", "floor", "price", "price_per_m2", "url"]]
        .reset_index(drop=True)
    )
    top["floor"] = top["floor"].fillna(0).astype(int)
    top["rooms"] = top["rooms"].fillna(0).astype(int)
    top["price"] = top["price"].astype(int)
    top["price_per_m2"] = top["price_per_m2"].round(0).astype(int)
    return top


def closest_listing_url(
    city: str, area: float, budget: int, rooms_min: int,
    neighborhood: str | None = None,
) -> str | None:
    base = (
        (df_ref["city"] == city)
        & (df_ref["price"] <= budget)
        & (df_ref["rooms"] >= rooms_min)
        & df_ref["url"].notna()
    )
    pool = df_ref[base & (df_ref["neighborhood"] == neighborhood)].copy() if neighborhood else pd.DataFrame()
    if pool.empty:
        pool = df_ref[base].copy()
    if pool.empty:
        return None
    pool["_dist"] = (pool["area_m2"] - area).abs()
    return pool.sort_values("_dist").iloc[0]["url"]


def _feat_dict(
    city: str, nb: str, area: float,
    rooms: int, floor: int,
    rok_budowy: int, stan_wykonczenia: str, winda: bool,
) -> dict:
    """Build a feature dict for one row (used for batch assembly)."""
    city_val = enc_city.get(city, enc_city["__unknown__"])
    nb_val   = enc_neighborhood.get(nb, enc_neighborhood["__unknown__"])
    features = meta.get("features") or [
        "area_m2", "rooms", "floor", "city_enc", "neighborhood_enc", "is_private_owner",
    ]
    feat: dict = {col: 0 for col in features}
    feat.update({
        "area_m2": area, "rooms": rooms, "floor": floor,
        "city_enc": city_val, "neighborhood_enc": nb_val,
        "is_private_owner": 0,
        "rok_budowy": rok_budowy, "liczba_pieter": LICZBA_PIETER_DEFAULT,
        "winda": int(winda),
    })
    for cat in STAN_RAW_OPTIONS:
        col = f"stan_wykonczenia_{cat}"
        if col in feat:
            feat[col] = 1 if (stan_wykonczenia and cat == stan_wykonczenia) else 0
    return feat


def _batch_predict_areas(
    pairs: list[tuple[str, str]],
    areas: np.ndarray,
    rooms: int,
    rok_budowy: int,
    stan_wykonczenia: str,
    winda: bool,
) -> np.ndarray:
    features = meta.get("features") or [
        "area_m2", "rooms", "floor", "city_enc", "neighborhood_enc", "is_private_owner",
    ]
    rows = [
        _feat_dict(c, nb, float(areas[i]), rooms, 2, rok_budowy, stan_wykonczenia, winda)
        for i, (c, nb) in enumerate(pairs)
    ]
    X = pd.DataFrame(rows)[features]
    return np.expm1(model.predict(X))


def reverse_lookup(
    selected_city: str,
    selected_nb: str,
    budget: int,
    rooms: int,
    rok_budowy: int,
    stan_wykonczenia: str,
    winda: bool,
    n_results: int = 10,
) -> pd.DataFrame:
    # Build the list of city-neighborhood pairs to search
    if selected_city == "All cities":
        pairs = [(c, city_neighborhoods[c][0]) for c in CITIES_SORTED]
    elif selected_nb == "All neighborhoods":
        pairs = [
            (selected_city, nb)
            for nb in city_neighborhoods.get(selected_city, [selected_city])
        ]
    else:
        pairs = [(selected_city, selected_nb)]

    n = len(pairs)
    lo = np.full(n, 15.0)
    hi = np.full(n, 200.0)

    for _ in range(14):
        mid = (lo + hi) / 2
        prices = _batch_predict_areas(pairs, mid, rooms, rok_budowy, stan_wykonczenia, winda)
        lo = np.where(prices <= budget, mid, lo)
        hi = np.where(prices > budget, mid, hi)

    final_areas  = np.round(lo, 1)
    final_prices = _batch_predict_areas(pairs, final_areas, rooms, rok_budowy, stan_wykonczenia, winda)

    rows = []
    for i, (city, nb) in enumerate(pairs):
        area  = final_areas[i]
        price = final_prices[i]
        if area >= 18:
            rows.append({
                "city": city, "neighborhood": nb, "area_m2": area,
                "est_price": int(price), "ppm2": int(price / area),
                "budget_used_pct": round(price / budget * 100, 1),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("area_m2", ascending=False).reset_index(drop=True)

    # for all-cities mode keep best result per city
    if selected_city == "All cities":
        df = df.drop_duplicates(subset=["city"]).head(n_results)

    return df


def sensitivity_chart(
    city, neighborhood, area, rooms, floor,
    rok_budowy, stan_wykonczenia, winda, base_price,
):
    plt.rcParams.update(FIG_STYLE)

    rok_lo = max(ROK_BUDOWY_RANGE[0], rok_budowy - 10)
    rok_hi = min(ROK_BUDOWY_RANGE[1], rok_budowy + 10)

    # each entry: (label, price_low, price_high)
    impacts = []

    candidates = [
        ("area_m2", f"Area ±20%  ({int(area*0.8)}–{int(area*1.2)} m²)",
         lambda: (
             predict_price(city, neighborhood, area * 0.8, rooms, floor, rok_budowy, stan_wykonczenia, winda)[0],
             predict_price(city, neighborhood, area * 1.2, rooms, floor, rok_budowy, stan_wykonczenia, winda)[0],
         )),
        ("rooms", f"Rooms ±1  ({max(1, rooms-1)}–{min(5, rooms+1)})",
         lambda: (
             predict_price(city, neighborhood, area, max(1, rooms - 1), floor, rok_budowy, stan_wykonczenia, winda)[0],
             predict_price(city, neighborhood, area, min(5, rooms + 1), floor, rok_budowy, stan_wykonczenia, winda)[0],
         )),
        ("floor", f"Floor ±1  ({max(0, floor-1)}–{min(10, floor+1)})",
         lambda: (
             predict_price(city, neighborhood, area, rooms, max(0, floor - 1), rok_budowy, stan_wykonczenia, winda)[0],
             predict_price(city, neighborhood, area, rooms, min(10, floor + 1), rok_budowy, stan_wykonczenia, winda)[0],
         )),
        ("rok_budowy", f"Year built ±10  ({rok_lo}–{rok_hi})",
         lambda: (
             predict_price(city, neighborhood, area, rooms, floor, rok_lo, stan_wykonczenia, winda)[0],
             predict_price(city, neighborhood, area, rooms, floor, rok_hi, stan_wykonczenia, winda)[0],
         )),
    ]

    candidates.append((
        "winda",
        "Elevator  (No → Yes)",
        lambda: (
            predict_price(city, neighborhood, area, rooms, floor, rok_budowy, stan_wykonczenia, False)[0],
            predict_price(city, neighborhood, area, rooms, floor, rok_budowy, stan_wykonczenia, True)[0],
        ),
    ))

    # stan_wykonczenia comparison: worst vs best available category
    if len(STAN_RAW_OPTIONS) >= 2:
        candidates.append((
            "stan_wykonczenia",
            f"Condition  ({_stan_label(STAN_RAW_OPTIONS[0])} → {_stan_label(STAN_RAW_OPTIONS[-1])})",
            lambda: (
                predict_price(city, neighborhood, area, rooms, floor, rok_budowy, STAN_RAW_OPTIONS[0], winda)[0],
                predict_price(city, neighborhood, area, rooms, floor, rok_budowy, STAN_RAW_OPTIONS[-1], winda)[0],
            ),
        ))

    for _key, label, fn in candidates:
        try:
            p_lo, p_hi = fn()
            impacts.append({
                "label": label,
                "low":  (p_lo - base_price) / 1_000,
                "high": (p_hi - base_price) / 1_000,
            })
        except Exception:
            pass

    if not impacts:
        return plt.figure()

    df_imp = pd.DataFrame(impacts)
    df_imp["abs_max"] = df_imp[["low", "high"]].abs().max(axis=1)
    df_imp = df_imp.sort_values("abs_max", ascending=True)

    fig, ax = plt.subplots(figsize=(6, max(2.6, len(df_imp) * 0.65)))
    for _, row in df_imp.iterrows():
        ax.barh(row["label"], row["low"],  color=C_BLUE, alpha=0.65, edgecolor="none", height=0.45)
        ax.barh(row["label"], row["high"], color=C_TEAL, alpha=0.85, edgecolor="none", height=0.45)
    ax.axvline(0, color=C_STEEL, linewidth=0.7, alpha=0.35)
    ax.set_xlabel("Price change (PLN thousands)", fontsize=8, color=C_STEEL)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_price(val) -> str:
    return f"{int(val):,}".replace(",", "\u202f")


def fmt_ppm2(val) -> str:
    return f"{int(val):,}".replace(",", "\u202f")


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

_r2   = f"{model_stats.get('r2', 0.903):.3f}"
_mae  = f"{int(model_stats.get('mae', 97_270)):,}"
_mape = f"{model_stats.get('mape', 7.9):.1f}%"
_n    = str(len(CITIES_SORTED))

st.markdown(f"""
<div style="text-align:center; padding:1.25rem 0 0.75rem 0;">
    <div style="
        font-family:'Plus Jakarta Sans',sans-serif;
        font-size:1.85rem; font-weight:800;
        color:{C_INK}; line-height:1.15;
        margin-bottom:0.4rem; letter-spacing:-0.02em;
    ">Polish Apartment<br>Price Estimator</div>
    <div style="
        font-size:13px; color:{C_STEEL};
        max-width:520px; margin:0 auto 1rem auto; line-height:1.6;
    ">
        Select city, neighborhood, and property parameters, then click the
        button to generate a price estimate powered by XGBoost trained on
        Otodom.pl listings and their detail pages.
    </div>
</div>

<div class="stats-panel">
    <div class="stat">
        <div class="stat-value">{_r2}</div>
        <div class="stat-label">R²</div>
    </div>
    <div class="stat">
        <div class="stat-value">{_mae}</div>
        <div class="stat-label">MAE (PLN)</div>
    </div>
    <div class="stat">
        <div class="stat-value">{_mape}</div>
        <div class="stat-label">Median MAPE</div>
    </div>
    <div class="stat">
        <div class="stat-value">{_n}</div>
        <div class="stat-label">Cities</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_est, tab_rev, tab_data = st.tabs(["Estimate Price", "Reverse Lookup", "About the Project"])

# ===========================================================================
# TAB 1 — ESTIMATE PRICE
# ===========================================================================

with tab_est:
    c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 1.4, 1, 1, 1.4])

    city: str        = c1.selectbox("City", CITIES_SORTED, key="est_city")
    neighborhood: str = c2.selectbox("Neighborhood", city_neighborhoods.get(city, [city]), key="est_nb")
    area: float      = c3.number_input("Area (m²)", min_value=15, max_value=200, value=55, step=1, key="est_area")
    rooms: int       = c4.selectbox("Rooms", [1, 2, 3, 4, 5], index=1, key="est_rooms")
    floor: int       = c5.number_input("Floor", min_value=0, max_value=10, value=2, step=1, key="est_floor")
    run_est          = c6.button("Estimate Price", type="primary", width="stretch", key="btn_est")

    d1, d2, d3 = st.columns([1, 2, 0.8])
    rok_budowy: int = d1.number_input(
        "Year built",
        min_value=ROK_BUDOWY_RANGE[0], max_value=ROK_BUDOWY_RANGE[1],
        value=2000, step=1, key="est_rok",
    )
    stan_raw: str = d2.selectbox(
        "Condition",
        STAN_OPTIONS,
        format_func=lambda x: _stan_label(x) if x != "— not specified —" else x,
        key="est_stan",
    )
    stan_wykonczenia: str = stan_raw if stan_raw != "— not specified —" else ""
    winda: bool = d3.selectbox("Elevator", ["No", "Yes"], key="est_winda") == "Yes"

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if not run_est:
        st.markdown(f"""
        <div style="text-align:center;padding:3.5rem 0;color:{C_STEEL};font-size:14px;line-height:1.6;">
            Fill in the parameters above and click
            <strong style="color:{C_BLUE};">Estimate Price</strong>.
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Calculating estimate…"):
            price, ppm2 = predict_price(city, neighborhood, area, rooms, floor, rok_budowy, stan_wykonczenia, winda)
            median_city = city_median.get(city, ppm2)
            delta_pct   = (ppm2 - median_city) / median_city * 100

        if price < 300_000:
            st.warning(
                "Estimate below 300,000 PLN — the model has limited training data in this "
                "price range. Treat as indicative only."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Estimated Price",  f"{fmt_price(price)} PLN")
        m2.metric("Price per m²",     f"{fmt_ppm2(ppm2)} PLN/m²")
        m3.metric("City Median",      f"{fmt_ppm2(median_city)} PLN/m²")
        m4.metric("vs City Median",   f"{delta_pct:+.1f} %",
                  delta=f"{delta_pct:+.1f} %", delta_color="inverse")

        st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown(
                f'<div class="chart-label">Price distribution '
                f'<span>— {neighborhood} vs {city}</span></div>',
                unsafe_allow_html=True,
            )
            nb_data   = df_ref[(df_ref["city"] == city) & (df_ref["neighborhood"] == neighborhood)]["price_per_m2"]
            city_data = df_ref[df_ref["city"] == city]["price_per_m2"]
            has_nb    = len(nb_data) >= 10

            plt.rcParams.update(FIG_STYLE)
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.hist(city_data, bins=35, color=C_BLUE, alpha=0.12, edgecolor="none", label=f"{city} (all)")
            if has_nb:
                ax.hist(nb_data, bins=20, color=C_BLUE, alpha=0.55, edgecolor="none", label=neighborhood)
            ax.axvline(ppm2, color=C_TEAL, linewidth=2.5, label=f"Estimate  {fmt_ppm2(ppm2)}")
            ax.axvline(city_data.median(), color=C_STEEL, linewidth=1.2, linestyle="--",
                       label=f"Median  {fmt_ppm2(city_data.median())}")
            ax.set_xlabel("PLN / m²", fontsize=8)
            ax.set_ylabel("Listings", fontsize=8)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))
            ax.legend(frameon=False, fontsize=7.5, loc="upper right")
            plt.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close()

        with col_r:
            st.markdown(
                '<div class="chart-label">Cross-city benchmarking '
                '<span>— same spec, PLN/m²</span></div>',
                unsafe_allow_html=True,
            )
            cp_df  = predict_all_cities(area, rooms, floor, rok_budowy, stan_wykonczenia, winda)
            colors = [C_TEAL if c == city else C_BLUE for c in cp_df["city"]]
            alphas = [1.0  if c == city else 0.50   for c in cp_df["city"]]

            plt.rcParams.update(FIG_STYLE)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            bars = ax2.barh(cp_df["city"], cp_df["ppm2"], edgecolor="none", height=0.55)
            for bar, col, alpha in zip(bars, colors, alphas):
                bar.set_facecolor(col)
                bar.set_alpha(alpha)
            for i, (_, row) in enumerate(cp_df.iterrows()):
                ax2.text(row["ppm2"] + 60, i, fmt_ppm2(row["ppm2"]), va="center", fontsize=7, color=C_SLATE)
            ax2.set_xlabel("Predicted PLN / m²", fontsize=8)
            ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))
            ax2.set_xlim(0, cp_df["ppm2"].max() * 1.22)
            plt.tight_layout()
            st.pyplot(fig2, width="stretch")
            plt.close()

        st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)

        st.markdown(
            f'<div class="chart-label">Comparable listings '
            f'<span>— real Otodom.pl data ({SCRAPED_DATE}) · links may expire after sale</span></div>',
            unsafe_allow_html=True,
        )

        sim = similar_listings(city, neighborhood, area, rooms, price)
        if not sim.empty:
            rows_html = ""
            for _, row in sim.iterrows():
                link = (
                    f'<a href="{row["url"]}" target="_blank" '
                    f'style="color:{C_BLUE};text-decoration:none;font-weight:600;">↗&nbsp;View</a>'
                    if pd.notna(row["url"]) and row["url"]
                    else f'<span style="color:{C_SILVER};">—</span>'
                )
                rows_html += f"""
                <tr>
                    <td style="font-weight:600;">{row['neighborhood']}</td>
                    <td>{row['area_m2']:.0f} m²</td>
                    <td>{int(row['rooms'])}</td>
                    <td>{int(row['floor'])}</td>
                    <td style="font-weight:700;">{fmt_price(row['price'])}</td>
                    <td style="color:{C_BLUE};font-weight:700;">{fmt_ppm2(row['price_per_m2'])}</td>
                    <td>{link}</td>
                </tr>"""
            st.markdown(f"""
            <div class="table-wrap">
              <table class="data-table">
                <thead><tr>
                  <th>Neighborhood</th><th>Area</th><th>Rooms</th>
                  <th>Floor</th><th>Price (PLN)</th><th>PLN/m²</th><th>Source</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("No comparable listings found for this city.")

        st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)

        col_l2, col_r2 = st.columns(2)

        with col_l2:
            st.markdown(
                f'<div class="chart-label">Neighborhood ranking '
                f'<span>— {city}, PLN/m²</span></div>',
                unsafe_allow_html=True,
            )
            nb_df     = predict_all_neighborhoods(city, area, rooms, floor, rok_budowy, stan_wykonczenia, winda)
            colors_nb = [C_TEAL if n == neighborhood else C_BLUE for n in nb_df["neighborhood"]]
            alphas_nb = [1.0  if n == neighborhood else 0.50   for n in nb_df["neighborhood"]]

            plt.rcParams.update(FIG_STYLE)
            fig3, ax3 = plt.subplots(figsize=(6, max(3.5, len(nb_df) * 0.38)))
            bars3 = ax3.barh(nb_df["neighborhood"], nb_df["ppm2"], edgecolor="none", height=0.55)
            for bar, col, alpha in zip(bars3, colors_nb, alphas_nb):
                bar.set_facecolor(col)
                bar.set_alpha(alpha)
            ax3.set_xlabel("Predicted PLN / m²", fontsize=8)
            ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))
            ax3.set_xlim(0, nb_df["ppm2"].max() * 1.18)
            plt.tight_layout()
            st.pyplot(fig3, width="stretch")
            plt.close()

        with col_r2:
            st.markdown(
                '<div class="chart-label">Price sensitivity '
                '<span>— impact of changing each parameter</span></div>',
                unsafe_allow_html=True,
            )
            fig4 = sensitivity_chart(
                city, neighborhood, area, rooms, floor,
                rok_budowy, stan_wykonczenia, winda, price,
            )
            st.pyplot(fig4, width="stretch")
            plt.close()


# ===========================================================================
# TAB 2 — REVERSE LOOKUP
# ===========================================================================

with tab_rev:
    r1, r2, r3, r4 = st.columns([2, 2, 1, 1.4])

    city_options_rl = ["All cities"] + CITIES_SORTED
    selected_city_rl: str = r1.selectbox("City", city_options_rl, key="rl_city")

    nb_options_rl = (
        ["All neighborhoods"] + city_neighborhoods.get(selected_city_rl, [])
        if selected_city_rl != "All cities" else ["All neighborhoods"]
    )
    selected_nb_rl: str = r2.selectbox(
        "Neighborhood",
        nb_options_rl,
        disabled=(selected_city_rl == "All cities"),
        key="rl_nb",
    )

    budget: int   = r3.number_input(
        "Budget (PLN)", min_value=100_000, max_value=3_000_000,
        value=600_000, step=10_000, key="rl_budget",
    )
    rooms_rl: int = r4.selectbox("Rooms (min)", [1, 2, 3, 4, 5], index=1, key="rl_rooms")

    d1, d2, d3, d4 = st.columns([1, 2, 0.8, 1.5])
    rok_budowy_rl: int = d1.number_input(
        "Year built",
        min_value=ROK_BUDOWY_RANGE[0], max_value=ROK_BUDOWY_RANGE[1],
        value=2000, step=1, key="rl_rok",
    )
    stan_raw_rl: str = d2.selectbox(
        "Condition",
        STAN_OPTIONS,
        format_func=lambda x: _stan_label(x) if x != "— not specified —" else x,
        key="rl_stan",
    )
    stan_wykonczenia_rl: str = stan_raw_rl if stan_raw_rl != "— not specified —" else ""
    winda_rl: bool = d3.selectbox("Elevator", ["No", "Yes"], key="rl_winda") == "Yes"
    run_rev        = d4.button("Find Apartments", type="primary", width="stretch", key="btn_rev")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if not run_rev:
        st.markdown(f"""
        <div style="text-align:center;padding:3.5rem 0;color:{C_STEEL};font-size:14px;line-height:1.6;">
            Set your budget and click <strong style="color:{C_BLUE};">Find Apartments</strong>.
        </div>
        """, unsafe_allow_html=True)

    else:
        spinner_msg = (
            "Searching across all cities…"
            if selected_city_rl == "All cities"
            else f"Searching neighborhoods in {selected_city_rl}…"
            if selected_nb_rl == "All neighborhoods"
            else "Calculating…"
        )
        try:
            with st.spinner(spinner_msg):
                results = reverse_lookup(
                    selected_city_rl, selected_nb_rl,
                    budget, rooms_rl, rok_budowy_rl, stan_wykonczenia_rl, winda_rl,
                )
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            results = pd.DataFrame()

        if results.empty:
            st.warning("No results found. Try increasing the budget or reducing room count.")
        else:
            best = results.iloc[0]

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {C_BLUE} 0%, {C_BLUE_D} 100%);
                border-radius: 14px; padding: 1.75rem 2rem;
                margin-bottom: 1.75rem; position: relative; overflow: hidden;
                box-shadow: 0 4px 20px rgba(26,86,219,0.25);
            ">
                <div style="position:absolute;right:-2rem;top:-2rem;width:8rem;height:8rem;
                            background:rgba(255,255,255,0.06);border-radius:50%;"></div>
                <div style="position:relative;z-index:1;">
                    <div style="font-size:10.5px;font-weight:700;text-transform:uppercase;
                                letter-spacing:0.14em;color:rgba(255,255,255,0.55);margin-bottom:0.5rem;">
                        Best purchasing power
                    </div>
                    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.8rem;
                                font-weight:800;color:#fff;margin-bottom:0.35rem;letter-spacing:-0.01em;">
                        {best['area_m2']} m² in {best['city']}
                    </div>
                    <div style="font-size:13px;color:rgba(255,255,255,0.6);">
                        {best['neighborhood']}&nbsp;&nbsp;·&nbsp;&nbsp;{fmt_price(best['est_price'])} PLN
                        &nbsp;&nbsp;·&nbsp;&nbsp;{fmt_ppm2(best['ppm2'])} PLN/m²
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # table first
            st.markdown(
                f'<div class="chart-label">Results '
                f'<span>— {len(results)} results sorted by maximum area</span></div>',
                unsafe_allow_html=True,
            )

            rows_html = ""
            for _, row in results.iterrows():
                pct = row["budget_used_pct"]
                url = closest_listing_url(row["city"], row["area_m2"], budget, rooms_rl, row["neighborhood"])
                link_cell = (
                    f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
                    f'style="color:{C_BLUE};font-weight:600;text-decoration:none;font-size:12px;">View →</a>'
                    if url else
                    f'<span style="color:{C_SILVER};font-size:11px;">—</span>'
                )
                rows_html += f"""
                <tr>
                    <td style="font-weight:700;">{row['city']}</td>
                    <td>{row['neighborhood']}</td>
                    <td style="font-weight:600;">{row['area_m2']} m²</td>
                    <td style="font-weight:700;">{fmt_price(row['est_price'])}</td>
                    <td style="color:{C_STEEL};">{fmt_ppm2(row['ppm2'])}</td>
                    <td>
                        <span style="display:inline-block;padding:0.2rem 0.65rem;
                                     background:{C_TEAL_BG};color:{C_TEAL};
                                     border-radius:6px;font-size:11px;font-weight:700;">
                            {pct} %
                        </span>
                    </td>
                    <td>{link_cell}</td>
                </tr>"""

            st.markdown(f"""
            <div class="table-wrap">
              <table class="data-table">
                <thead><tr>
                  <th>City</th><th>Neighborhood</th><th>Area</th>
                  <th>Est. Price (PLN)</th><th>PLN/m²</th><th>% of Budget</th>
                  <th>Listing</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)

            # chart below
            label_y = "neighborhood" if selected_city_rl != "All cities" else "city"
            chart_subtitle = (
                "all cities · one representative neighborhood each"
                if selected_city_rl == "All cities"
                else f"{selected_city_rl} · all neighborhoods"
                if selected_nb_rl == "All neighborhoods"
                else f"{selected_city_rl} · {selected_nb_rl}"
            )
            st.markdown(
                f'<div class="chart-label">Purchasing power '
                f'<span>— max area (m²) for {fmt_price(budget)} PLN · {chart_subtitle}</span></div>',
                unsafe_allow_html=True,
            )

            plt.rcParams.update(FIG_STYLE)
            fig, ax = plt.subplots(figsize=(9, max(3.5, len(results) * 0.42)))
            y_vals  = results[label_y]
            colors  = [C_TEAL if i == 0 else C_BLUE for i in range(len(results))]
            alphas  = [1.0    if i == 0 else 0.55   for i in range(len(results))]
            bars    = ax.barh(y_vals, results["area_m2"], edgecolor="none", height=0.55)
            for bar, col, alpha in zip(bars, colors, alphas):
                bar.set_facecolor(col)
                bar.set_alpha(alpha)
            for i, (_, row) in enumerate(results.iterrows()):
                label_txt = (
                    f"{row['area_m2']} m²  ·  {row['neighborhood']}  ·  {fmt_price(row['est_price'])} PLN"
                    if selected_city_rl == "All cities"
                    else f"{row['area_m2']} m²  ·  {fmt_price(row['est_price'])} PLN"
                )
                ax.text(row["area_m2"] + 0.4, i, label_txt, va="center", fontsize=7.5, color=C_SLATE)
            ax.set_xlabel("Max area you can buy (m²)", fontsize=8)
            ax.set_xlim(0, results["area_m2"].max() * 1.65)
            plt.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close()


# ===========================================================================
# TAB 3 — ABOUT THE PROJECT
# ===========================================================================

with tab_data:
    n_listings     = len(df_ref)
    n_cities       = df_ref["city"].nunique()
    n_neighborhoods = df_ref["neighborhood"].nunique()
    median_ppm2    = int(df_ref["price_per_m2"].median())

    st.markdown(f"""
    <div class="about-section">
        <h3>About This Project</h3>
        <p>
            End-to-end ML pipeline for predicting apartment prices in Poland.
            The dataset was scraped from <strong>Otodom.pl</strong>, Poland's largest real
            estate portal, in <strong>{SCRAPED_DATE}</strong> using <code>requests</code>
            + <code>BeautifulSoup</code>. Each listing page embeds a
            <code>__NEXT_DATA__</code> JSON block (Next.js SSR) containing the full
            structured property data — no headless browser required.
        </p>
        <p>
            Neighborhood data comes from <code>location.reverseGeocoding.locations</code>
            filtered by <code>locationLevel == "district"</code>, giving proper district
            names (Mokotów, Wola, Żoliborz) rather than the city-level fallback that
            the simpler address field returns for ~95% of listings.
        </p>
        <p>
            After collecting list-view data, a second scraping pass fetched individual
            listing pages for all ~36k listings to extract detail features: year built,
            building type, heating type, finish condition, and amenity flags (elevator,
            balcony, garage, etc.).
        </p>
    </div>
    """, unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Listings",       f"{n_listings:,}")
    d2.metric("Cities",         str(n_cities))
    d3.metric("Neighborhoods",  str(n_neighborhoods))
    d4.metric("Median PLN/m²",  f"{median_ppm2:,}")

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="about-section">
        <h3>Model</h3>
        <p>
            <strong>XGBoost regressor</strong> trained on log-transformed prices
            (<code>log1p</code> → <code>expm1</code> at inference). 80/20 train-test split.
            City and neighborhood are encoded using <strong>target encoding</strong>
            (mean log-price per location, fit on the training set only to prevent leakage).
        </p>
        <ul>
            <li><code>area_m2</code>, <code>rooms</code>, <code>floor</code> — core property dimensions</li>
            <li><code>city_enc</code>, <code>neighborhood_enc</code> — target-encoded location</li>
            <li><code>rok_budowy</code> — year built (#3 in SHAP importance)</li>
            <li><code>stan_wykonczenia</code> — finish condition, one-hot encoded</li>
            <li><code>winda</code>, <code>balkon</code>, <code>garaz</code> and 10 other boolean amenity flags</li>
            <li><code>rodzaj_zabudowy</code>, <code>ogrzewanie</code>, <code>rynek</code>
                and 3 other categoricals — one-hot encoded</li>
        </ul>
        <p>
            Test set performance: <strong>R² = {_r2}</strong> ·
            <strong>MAE = {_mae} PLN</strong> · <strong>Median MAPE = {_mape}</strong>
        </p>
        <h3>Limitations</h3>
        <p>
            Prices are <em>asking prices</em> from Otodom.pl listings, not closed
            transaction prices. The model performs best in the 300,000–800,000 PLN
            range (MAPE ~9–10%) and is less reliable below 300,000 PLN due to limited
            training examples in that segment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    chart_l, chart_r = st.columns(2)

    with chart_l:
        st.markdown('<div class="chart-label">Listings per city</div>', unsafe_allow_html=True)
        city_counts = df_ref["city"].value_counts().sort_values(ascending=True)
        plt.rcParams.update(FIG_STYLE)
        fig_dc, ax_dc = plt.subplots(figsize=(6, 4))
        ax_dc.barh(city_counts.index, city_counts.values, edgecolor="none", height=0.55, color=C_BLUE, alpha=0.55)
        for i, (cname, cnt) in enumerate(city_counts.items()):
            ax_dc.text(cnt + 8, i, str(cnt), va="center", fontsize=8, color=C_SLATE)
        ax_dc.set_xlabel("Number of listings", fontsize=8)
        ax_dc.set_xlim(0, city_counts.max() * 1.15)
        plt.tight_layout()
        st.pyplot(fig_dc, width="stretch")
        plt.close()

    with chart_r:
        st.markdown('<div class="chart-label">Median price per m² by city</div>', unsafe_allow_html=True)
        med_series = df_ref.groupby("city")["price_per_m2"].median().sort_values(ascending=True)
        plt.rcParams.update(FIG_STYLE)
        fig_mp, ax_mp = plt.subplots(figsize=(6, 4))
        ax_mp.barh(med_series.index, med_series.values, edgecolor="none", height=0.55, color=C_TEAL, alpha=0.70)
        for i, (cname, val) in enumerate(med_series.items()):
            ax_mp.text(val + 60, i, fmt_ppm2(val), va="center", fontsize=8, color=C_SLATE)
        ax_mp.set_xlabel("Median PLN / m²", fontsize=8)
        ax_mp.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))
        ax_mp.set_xlim(0, med_series.max() * 1.18)
        plt.tight_layout()
        st.pyplot(fig_mp, width="stretch")
        plt.close()

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    chart_l2, chart_r2 = st.columns(2)

    with chart_l2:
        st.markdown('<div class="chart-label">Price per m² distribution — all cities</div>', unsafe_allow_html=True)
        plt.rcParams.update(FIG_STYLE)
        fig_pd, ax_pd = plt.subplots(figsize=(6, 3.2))
        ax_pd.hist(df_ref["price_per_m2"], bins=50, color=C_BLUE, alpha=0.55, edgecolor="none")
        ax_pd.axvline(df_ref["price_per_m2"].median(), color=C_TEAL, linewidth=2,
                      label=f"Median  {fmt_ppm2(df_ref['price_per_m2'].median())}")
        ax_pd.set_xlabel("PLN / m²", fontsize=8)
        ax_pd.set_ylabel("Listings", fontsize=8)
        ax_pd.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))
        ax_pd.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_pd, width="stretch")
        plt.close()

    with chart_r2:
        st.markdown('<div class="chart-label">Room count distribution</div>', unsafe_allow_html=True)
        rooms_counts = df_ref["rooms"].dropna().astype(int).value_counts().sort_index()
        plt.rcParams.update(FIG_STYLE)
        fig_rc, ax_rc = plt.subplots(figsize=(6, 3.2))
        bars_rc = ax_rc.bar(rooms_counts.index.astype(str), rooms_counts.values,
                            color=C_BLUE, alpha=0.55, edgecolor="none", width=0.6)
        for bar, val in zip(bars_rc, rooms_counts.values):
            ax_rc.text(bar.get_x() + bar.get_width() / 2, val + 30, str(val),
                       ha="center", fontsize=8.5, color=C_SLATE)
        ax_rc.set_xlabel("Rooms", fontsize=8)
        ax_rc.set_ylabel("Listings", fontsize=8)
        ax_rc.set_ylim(0, rooms_counts.max() * 1.15)
        plt.tight_layout()
        st.pyplot(fig_rc, width="stretch")
        plt.close()

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="chart-label">Summary statistics by city</div>', unsafe_allow_html=True)

    city_stats = (
        df_ref.groupby("city")
        .agg(
            listings=("price", "count"),
            median_price=("price", "median"),
            median_ppm2=("price_per_m2", "median"),
            median_area=("area_m2", "median"),
        )
        .sort_values("median_ppm2", ascending=False)
        .reset_index()
    )

    rows_stat = ""
    for _, row in city_stats.iterrows():
        rows_stat += f"""
        <tr>
            <td style="font-weight:700;">{row['city']}</td>
            <td>{int(row['listings'])}</td>
            <td style="font-weight:600;">{fmt_price(row['median_price'])}</td>
            <td style="color:{C_BLUE};font-weight:700;">{fmt_ppm2(row['median_ppm2'])}</td>
            <td>{row['median_area']:.0f} m²</td>
        </tr>"""

    st.markdown(f"""
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr>
          <th>City</th><th>Listings</th><th>Median Price (PLN)</th>
          <th>Median PLN/m²</th><th>Median Area</th>
        </tr></thead>
        <tbody>{rows_stat}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("<div style='height:3rem;'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="border-top:1.5px solid {C_MIST};padding:1.25rem 0;
            display:flex;justify-content:space-between;flex-wrap:wrap;
            gap:0.5rem;align-items:center;">
    <div style="font-size:11.5px;color:{C_SILVER};">
        Data: Otodom.pl ({SCRAPED_DATE}) · XGBoost · {_n} cities ·
        R²={_r2} · MAE={_mae} PLN · Median MAPE={_mape}
    </div>
    <div style="font-size:11.5px;color:{C_SILVER};">Built with Streamlit</div>
</div>
""", unsafe_allow_html=True)
