from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAY_AMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
MONTH_LABELS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


@dataclass
class StoryPack:
    metrics: dict[str, str]
    variable_dictionary: list[dict[str, str]]
    charts: dict[str, str]


VARIABLE_DICTIONARY = [
    {"name": "LIMIT_BAL", "group": "Profil", "description": "Montant du crédit accordé (NTD)."},
    {"name": "SEX", "group": "Profil", "description": "Genre recodé (male/female)."},
    {"name": "EDUCATION", "group": "Profil", "description": "Niveau d'éducation recodé (graduate/university/high_school/others/unknown)."},
    {"name": "MARRIAGE", "group": "Profil", "description": "Statut marital recodé (married/single/other)."},
    {"name": "AGE", "group": "Profil", "description": "Âge du client."},
    {"name": "PAY_0 ... PAY_6", "group": "Paiement", "description": "Historique des retards de paiement."},
    {"name": "BILL_AMT1 ... BILL_AMT6", "group": "Facturation", "description": "Montants facturés d'avril à septembre 2005."},
    {"name": "PAY_AMT1 ... PAY_AMT6", "group": "Remboursement", "description": "Montants réellement payés sur la même période."},
    {"name": "DEFAULT", "group": "Cible", "description": "1 = défaut le mois suivant, 0 = pas de défaut."},
    {"name": "n_months_late", "group": "Indicateur dérivé", "description": "Nombre de mois avec retard > 0."},
    {"name": "credit_utilization", "group": "Indicateur dérivé", "description": "Facturation moyenne / limite de crédit."},
    {"name": "repayment_ratio", "group": "Indicateur dérivé", "description": "Remboursement moyen / facturation moyenne."},
]


def _to_plot(fig: Any) -> str:
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        height=420,
        autosize=True,
        hovermode="closest",
    )
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={
            "displayModeBar": True,
            "responsive": True,
            "scrollZoom": True,
            "doubleClick": "reset",
        },
    )


def _recode_idempotent(series: pd.Series, mapping: dict[int, str]) -> pd.Series:
    known_labels = set(mapping.values())
    numeric_series = pd.to_numeric(series, errors="coerce")
    mapped = numeric_series.map(mapping)
    return series.where(series.isin(known_labels), mapped)


@lru_cache(maxsize=1)
def load_clean_data(dataset_path: str) -> pd.DataFrame:
    df = pd.read_excel(Path(dataset_path), header=1)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    if "DEFAULT" not in df.columns:
        target_col = [c for c in df.columns if "default" in c.lower() and "payment" in c.lower()][0]
        df = df.rename(columns={target_col: "DEFAULT"})

    df["EDUCATION"] = _recode_idempotent(df["EDUCATION"], {
        1: "graduate", 2: "university", 3: "high_school", 4: "others", 0: "unknown", 5: "unknown", 6: "unknown",
    })
    df["MARRIAGE"] = _recode_idempotent(df["MARRIAGE"], {1: "married", 2: "single", 0: "other", 3: "other"})
    df["SEX"] = _recode_idempotent(df["SEX"], {1: "male", 2: "female"})

    df["DEFAULT"] = pd.to_numeric(df["DEFAULT"], errors="coerce").fillna(0).astype(int)

    for col in PAY_COLS + BILL_COLS + PAY_AMT_COLS + ["LIMIT_BAL", "AGE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    df["avg_pay_delay"] = df[PAY_COLS].mean(axis=1)
    df["max_pay_delay"] = df[PAY_COLS].max(axis=1)
    df["n_months_late"] = (df[PAY_COLS] > 0).sum(axis=1)
    df["avg_bill_amt"] = df[BILL_COLS].mean(axis=1)
    df["avg_pay_amt"] = df[PAY_AMT_COLS].mean(axis=1)

    df["credit_utilization"] = (df["avg_bill_amt"] / df["LIMIT_BAL"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["repayment_ratio"] = np.where(df["avg_bill_amt"] > 0, df["avg_pay_amt"] / df["avg_bill_amt"], 0)
    df["credit_utilization"] = df["credit_utilization"].clip(0, 5)
    df["repayment_ratio"] = pd.Series(df["repayment_ratio"], index=df.index).fillna(0).clip(0, 10)

    return df


def _kpi_metrics(df: pd.DataFrame) -> dict[str, str]:
    return {
        "clients": f"{len(df):,}".replace(",", " "),
        "default_rate": f"{(df['DEFAULT'].mean() * 100):.2f}%",
        "n_defaults": f"{int(df['DEFAULT'].sum()):,}".replace(",", " "),
        "credit_mean": f"{df['LIMIT_BAL'].mean():,.0f} NTD".replace(",", " "),
        "credit_median": f"{df['LIMIT_BAL'].median():,.0f} NTD".replace(",", " "),
        "age_mean": f"{df['AGE'].mean():.1f} ans",
        "age_median": f"{df['AGE'].median():.0f} ans",
    }


def _chart_repartition_cible(df: pd.DataFrame) -> str:
    c = df["DEFAULT"].value_counts().sort_index()
    fig = go.Figure()
    fig.add_bar(
        x=["No Default (0)", "Default (1)"],
        y=[c.get(0, 0), c.get(1, 0)],
        marker_color=["#2ecc71", "#e74c3c"],
    )
    fig.update_layout(title="1. Répartition de la Cible<br><sup>(Le défaut est-il fréquent ?)</sup>", yaxis_title="Nombre de clients")
    return _to_plot(fig)


def _chart_distribution_credit(df: pd.DataFrame) -> str:
    fig = px.histogram(df, x="LIMIT_BAL", nbins=50, title="2. Distribution du Crédit Accordé<br><sup>(Les faibles crédits défautent-ils plus ?)</sup>")
    fig.add_vline(x=df["LIMIT_BAL"].mean(), line_dash="dash", line_color="red")
    fig.add_vline(x=df["LIMIT_BAL"].median(), line_dash="dash", line_color="orange")
    fig.update_xaxes(title="LIMIT_BAL")
    return _to_plot(fig)


def _chart_distribution_age(df: pd.DataFrame) -> str:
    fig = px.histogram(df, x="AGE", nbins=30, title="3. Distribution d'Âge<br><sup>(L'âge aide-t-il à segmenter ?)</sup>")
    fig.add_vline(x=df["AGE"].mean(), line_dash="dash", line_color="red")
    fig.add_vline(x=df["AGE"].median(), line_dash="dash", line_color="orange")
    return _to_plot(fig)


def _rate_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tmp = df.groupby(col)["DEFAULT"].agg(["sum", "count"]).reset_index()
    tmp["rate"] = np.where(tmp["count"] > 0, tmp["sum"] / tmp["count"] * 100, 0)
    return tmp.sort_values("rate", ascending=False)


def _chart_defaut_genre(df: pd.DataFrame) -> str:
    s = _rate_by_category(df, "SEX")
    fig = px.bar(s, x="SEX", y="rate", title="4. Taux de Défaut par Genre<br><sup>(Profil socio-démo)</sup>", color="SEX")
    fig.update_yaxes(title="Taux de défaut (%)")
    return _to_plot(fig)


def _chart_defaut_education(df: pd.DataFrame) -> str:
    s = _rate_by_category(df, "EDUCATION")
    fig = px.bar(s, x="EDUCATION", y="rate", title="5. Taux de Défaut par Éducation<br><sup>(Variables secondaires)</sup>", color_discrete_sequence=["#9b59b6"])
    fig.update_yaxes(title="Taux de défaut (%)")
    return _to_plot(fig)


def _chart_defaut_marriage(df: pd.DataFrame) -> str:
    s = _rate_by_category(df, "MARRIAGE")
    fig = px.bar(s, x="MARRIAGE", y="rate", title="6. Taux de Défaut par État Civil<br><sup>(Profil socio-démo)</sup>", color_discrete_sequence=["#f39c12"])
    fig.update_yaxes(title="Taux de défaut (%)")
    return _to_plot(fig)


def _chart_pay0(df: pd.DataFrame) -> str:
    pay0 = df.groupby("PAY_0")["DEFAULT"].agg(["sum", "count"]).reset_index().sort_values("PAY_0")
    pay0 = pay0[pay0["count"] > 10].copy()
    pay0["rate"] = np.where(pay0["count"] > 0, pay0["sum"] / pay0["count"] * 100, 0)
    fig = px.bar(pay0, x="PAY_0", y="rate", title="⭐ 1. Taux de Défaut par PAY_0<br><sup>(Le dernier statut de paiement est-il critique ?)</sup>", color_discrete_sequence=["#e74c3c"])
    fig.update_yaxes(title="Taux de défaut (%)", range=[0, 100])
    return _to_plot(fig)


def _chart_heatmap_retard(df: pd.DataFrame) -> str:
    rows = []
    for default_val in [0, 1]:
        subset = df[df["DEFAULT"] == default_val]
        rows.append([subset[col].mean() for col in PAY_COLS])
    heatmap_data = np.array(rows)

    fig = px.imshow(
        heatmap_data,
        x=PAY_COLS,
        y=["No Default", "Default"],
        text_auto=".2f",
        color_continuous_scale="RdYlGn_r",
        title="2. Heatmap : Retard Moyen (PAY_0..PAY_6) par Classe<br><sup>(Retard ponctuel ou persistant ?)</sup>",
    )
    fig.update_coloraxes(colorbar_title="Retard moyen (mois)")
    return _to_plot(fig)


def _chart_evolution_retard(df: pd.DataFrame) -> str:
    nondef = [df[df["DEFAULT"] == 0][c].mean() for c in PAY_COLS]
    defe = [df[df["DEFAULT"] == 1][c].mean() for c in PAY_COLS]

    fig = go.Figure()
    fig.add_bar(name="No Default", x=PAY_COLS, y=nondef, marker_color="#2ecc71")
    fig.add_bar(name="Default", x=PAY_COLS, y=defe, marker_color="#e74c3c")
    fig.update_layout(
        barmode="group",
        title="3. Évolution du Retard Moyen par Période<br><sup>(Temporalité)</sup>",
        yaxis_title="Retard moyen (mois)",
    )
    return _to_plot(fig)


def _chart_corr_pay(df: pd.DataFrame) -> str:
    corr = df[PAY_COLS].corr()
    fig = px.imshow(
        corr,
        x=PAY_COLS,
        y=PAY_COLS,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="4. Corrélation entre PAY_0..PAY_6<br><sup>(Retards persistants ?)</sup>",
    )
    fig.update_coloraxes(colorbar_title="Corrélation")
    return _to_plot(fig)


def _long_for_box(df: pd.DataFrame, cols: list[str], value_name: str) -> pd.DataFrame:
    tmp = df[["DEFAULT"] + cols].copy()
    tmp["classe"] = np.where(tmp["DEFAULT"] == 1, "Default (1)", "No Default (0)")
    long = tmp.melt(id_vars=["classe"], value_vars=cols, var_name="variable", value_name=value_name)
    return long


def _chart_box_bill(df: pd.DataFrame) -> str:
    long = _long_for_box(df, BILL_COLS, "Montant")
    fig = px.box(
        long,
        x="variable",
        y="Montant",
        color="classe",
        title="1. Distribution des Factures (BILL_AMT1..6)<br>par Classe",
        points=False,
    )
    fig.update_yaxes(type="log")
    return _to_plot(fig)


def _chart_box_payamt(df: pd.DataFrame) -> str:
    long = _long_for_box(df, PAY_AMT_COLS, "Montant")
    fig = px.box(
        long,
        x="variable",
        y="Montant",
        color="classe",
        title="2. Distribution des Remboursements (PAY_AMT1..6)<br>par Classe",
        points=False,
    )
    fig.update_yaxes(type="log")
    return _to_plot(fig)


def _chart_scatter_credit_bill(df: pd.DataFrame) -> str:
    sample = df.sample(n=min(3500, len(df)), random_state=42)
    sample["classe"] = np.where(sample["DEFAULT"] == 1, "Default", "No Default")
    fig = px.scatter(
        sample,
        x="LIMIT_BAL",
        y="avg_bill_amt",
        color="classe",
        opacity=0.45,
        title="3. Relation Crédit vs Facturation Moyenne<br><sup>(Couple crédit/comportement)</sup>",
    )
    return _to_plot(fig)


def _chart_repayment_quantile(df: pd.DataFrame) -> str:
    work = df.copy()
    work["repayment_quantile"] = pd.qcut(
        work["repayment_ratio"],
        q=5,
        labels=["Q1 (worst)", "Q2", "Q3", "Q4", "Q5 (best)"],
        duplicates="drop",
    )
    rep = work.groupby("repayment_quantile", observed=True)["DEFAULT"].agg(["sum", "count"]).reset_index()
    rep["rate"] = np.where(rep["count"] > 0, rep["sum"] / rep["count"] * 100, 0)
    fig = px.bar(
        rep,
        x="repayment_quantile",
        y="rate",
        title="⭐ 4. Taux de Défaut par Ratio de Remboursement<br><sup>(Capacité vs Facturation)</sup>",
        color="repayment_quantile",
        color_discrete_sequence=["#e74c3c", "#f39c12", "#f1c40f", "#3498db", "#2ecc71"],
    )
    fig.update_yaxes(title="Taux de défaut (%)")
    return _to_plot(fig)


def build_storypack(dataset_path: str) -> StoryPack:
    df = load_clean_data(dataset_path)
    charts = {
        # Axe 1
        "axis1_target": _chart_repartition_cible(df),
        "axis1_limit": _chart_distribution_credit(df),
        "axis1_age": _chart_distribution_age(df),
        "axis1_sex": _chart_defaut_genre(df),
        "axis1_edu": _chart_defaut_education(df),
        "axis1_marriage": _chart_defaut_marriage(df),
        # Axe 2
        "axis2_pay0": _chart_pay0(df),
        "axis2_heatmap": _chart_heatmap_retard(df),
        "axis2_evolution": _chart_evolution_retard(df),
        "axis2_corr": _chart_corr_pay(df),
        # Axe 3
        "axis3_bill": _chart_box_bill(df),
        "axis3_payamt": _chart_box_payamt(df),
        "axis3_scatter": _chart_scatter_credit_bill(df),
        "axis3_repay": _chart_repayment_quantile(df),
    }

    return StoryPack(metrics=_kpi_metrics(df), variable_dictionary=VARIABLE_DICTIONARY, charts=charts)
