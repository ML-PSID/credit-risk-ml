from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from analysis_web import load_clean_data

CAT_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC_FEATURES = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "avg_pay_delay", "max_pay_delay", "n_months_late",
    "avg_bill_amt", "avg_pay_amt", "credit_utilization", "repayment_ratio",
]

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=60, b=20),
    autosize=True,
    hovermode="closest",
    font=dict(family="Syne, sans-serif", color="#efece5"),
)


@dataclass
class MLPack:
    metrics: dict[str, str]
    charts: dict[str, str]


def _to_plot(fig: Any, height: int = 420) -> str:
    fig.update_layout(height=height, **DARK_LAYOUT)
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


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_encoded = enc.fit_transform(df[CAT_FEATURES])
    cat_df = pd.DataFrame(cat_encoded, columns=CAT_FEATURES, index=df.index)

    X = pd.concat([cat_df, df[NUMERIC_FEATURES]], axis=1).values
    y = df["DEFAULT"].values
    feature_names = CAT_FEATURES + NUMERIC_FEATURES
    return X, y, feature_names


def _chart_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> str:
    cm = confusion_matrix(y_test, y_pred)
    labels = ["No Default (0)", "Default (1)"]
    annotations = [
        [f"VN<br>{cm[0,0]:,}", f"FP<br>{cm[0,1]:,}"],
        [f"FN<br>{cm[1,0]:,}", f"VP<br>{cm[1,1]:,}"],
    ]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=annotations,
        texttemplate="%{text}",
        colorscale=[[0, "#163044"], [0.5, "rgba(255,106,61,0.53)"], [1, "#ff6a3d"]],
        showscale=False,
        hovertemplate="Réel: %{y}<br>Prédit: %{x}<br>N = %{z}<extra></extra>",
    ))
    fig.update_layout(title="Matrice de Confusion", xaxis_title="Prédit", yaxis_title="Réel")
    return _to_plot(fig, height=400)


def _chart_feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> str:
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    colors = [
        "#ff6a3d" if v > np.percentile(vals, 75)
        else "#00b894" if v > np.percentile(vals, 50)
        else "#c9c4b5"
        for v in vals
    ]

    fig = go.Figure(go.Bar(
        x=vals[::-1],
        y=names[::-1],
        orientation="h",
        marker_color=colors[::-1],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(title="Top 20 — Importance des Variables (Gini)", xaxis_title="Importance moyenne (Gini)")
    return _to_plot(fig, height=520)


def _chart_roc(y_test: np.ndarray, y_proba: np.ndarray, auc: float) -> str:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#ff6a3d", width=2.5), name=f"RF (AUC = {auc:.3f})")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#c9c4b5", dash="dash", width=1), name="Aléatoire")
    fig.update_layout(
        title="Courbe ROC",
        xaxis_title="Taux de Faux Positifs",
        yaxis_title="Taux de Vrais Positifs",
        legend=dict(x=0.55, y=0.1),
    )
    return _to_plot(fig, height=420)


def _chart_precision_recall(y_test: np.ndarray, y_proba: np.ndarray) -> str:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    baseline = y_test.mean()

    fig = go.Figure()
    fig.add_scatter(x=recall, y=precision, mode="lines", line=dict(color="#00b894", width=2.5), name=f"RF (AP = {ap:.3f})")
    fig.add_hline(y=baseline, line_dash="dash", line_color="#c9c4b5", annotation_text=f"Baseline ({baseline:.2f})")
    fig.update_layout(
        title="Courbe Précision-Rappel",
        xaxis_title="Rappel",
        yaxis_title="Précision",
        legend=dict(x=0.55, y=0.9),
    )
    return _to_plot(fig, height=420)


def _chart_scores_bar(acc: float, prec: float, rec: float, f1: float, auc: float) -> str:
    metrics_names = ["Accuracy", "Précision", "Rappel", "F1-Score", "AUC-ROC"]
    values = [acc, prec, rec, f1, auc]
    colors = ["#ff6a3d", "#ffd166", "#00b894", "#0984e3", "#6c5ce7"]

    fig = go.Figure(go.Bar(
        x=metrics_names,
        y=values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(title="Synthèse des Métriques", yaxis=dict(range=[0, 1.1], title="Score"))
    return _to_plot(fig, height=400)


def _chart_proba_dist(y_test: np.ndarray, y_proba: np.ndarray) -> str:
    df_plot = pd.DataFrame({"proba": y_proba, "classe": np.where(y_test == 1, "Default (1)", "No Default (0)")})
    fig = px.histogram(
        df_plot,
        x="proba",
        color="classe",
        nbins=60,
        barmode="overlay",
        opacity=0.72,
        color_discrete_map={"Default (1)": "#ff6a3d", "No Default (0)": "#00b894"},
        title="Distribution des Probabilités Prédites",
        labels={"proba": "P(défaut)", "classe": "Classe réelle"},
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#ffd166", annotation_text="Seuil 0.5")
    fig.update_layout(xaxis_title="Probabilité de défaut prédite", yaxis_title="Nombre d'observations")
    return _to_plot(fig, height=400)


def _chart_learning_curve(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
) -> str:
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    train_scores, test_scores = [], []

    for frac in fractions:
        n = max(int(len(X_train) * frac), 50)
        idx = np.random.default_rng(42).choice(len(X_train), size=n, replace=False)
        Xtr, ytr = X_train[idx], y_train[idx]

        clf = RandomForestClassifier(
            n_estimators=50, max_depth=12, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf.fit(Xtr, ytr)
        train_scores.append(roc_auc_score(ytr, clf.predict_proba(Xtr)[:, 1]))
        test_scores.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    sizes = [int(len(X_train) * f) for f in fractions]

    fig = go.Figure()
    fig.add_scatter(x=sizes, y=train_scores, mode="lines+markers", name="Train AUC", line=dict(color="#ffd166", width=2), marker=dict(size=7))
    fig.add_scatter(x=sizes, y=test_scores, mode="lines+markers", name="Test AUC", line=dict(color="#00b894", width=2), marker=dict(size=7))
    fig.update_layout(
        title="Courbe d'Apprentissage (AUC-ROC)",
        xaxis_title="Taille de l'ensemble d'entraînement",
        yaxis_title="AUC-ROC",
        legend=dict(x=0.65, y=0.1),
    )
    return _to_plot(fig, height=420)


@lru_cache(maxsize=1)
def build_ml_pack(dataset_path: str) -> MLPack:
    df = load_clean_data(dataset_path)
    X, y, feature_names = _prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "accuracy": f"{acc:.3f}",
        "precision": f"{prec:.3f}",
        "recall": f"{rec:.3f}",
        "f1": f"{f1:.3f}",
        "auc_roc": f"{auc:.3f}",
        "n_train": f"{len(X_train):,}".replace(",", " "),
        "n_test": f"{len(X_test):,}".replace(",", " "),
    }

    charts = {
        "confusion": _chart_confusion_matrix(y_test, y_pred),
        "importance": _chart_feature_importance(model, feature_names),
        "roc": _chart_roc(y_test, y_proba, auc),
        "pr": _chart_precision_recall(y_test, y_proba),
        "scores": _chart_scores_bar(acc, prec, rec, f1, auc),
        "proba_dist": _chart_proba_dist(y_test, y_proba),
        "learning": _chart_learning_curve(X_train, X_test, y_train, y_test),
    }

    return MLPack(metrics=metrics, charts=charts)
