from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "data" / "notebook" / "default of credit card clients.xls"
CAT_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC_FEATURES = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "avg_pay_delay", "max_pay_delay", "n_months_late",
    "avg_bill_amt", "avg_pay_amt", "credit_utilization", "repayment_ratio",
]
EXTRA_FEATURES = ["trend_pay_delay", "n_zero_payment", "last_pay_ratio", "bill_acceleration"]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAY_AMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
RL_PARAM_DIST = {"logisticregression__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]}


def _recode_idempotent(series, mapping):
    known_labels = set(mapping.values())
    numeric_series = pd.to_numeric(series, errors="coerce")
    mapped = numeric_series.map(mapping)
    return series.where(series.isin(known_labels), mapped)


def _load_clean_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_excel(DATASET, header=1)
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
    df["credit_utilization"] = (df["avg_bill_amt"] / df["LIMIT_BAL"]).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 5)
    df["repayment_ratio"] = np.where(df["avg_bill_amt"] > 0, df["avg_pay_amt"] / df["avg_bill_amt"], 0)
    df["repayment_ratio"] = pd.Series(df["repayment_ratio"], index=df.index).fillna(0).clip(0, 10)

    df["trend_pay_delay"] = df["PAY_0"] - df["PAY_6"]
    df["n_zero_payment"] = (df[PAY_AMT_COLS] == 0).sum(axis=1)
    df["last_pay_ratio"] = df["PAY_AMT1"] / (df["BILL_AMT1"].abs() + 1)
    df["bill_acceleration"] = df["BILL_AMT1"] - df["BILL_AMT6"]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_encoded = pd.DataFrame(enc.fit_transform(df[CAT_FEATURES]), columns=CAT_FEATURES, index=df.index)
    all_numeric = NUMERIC_FEATURES + EXTRA_FEATURES
    X = pd.concat([cat_encoded, df[all_numeric]], axis=1).values
    y = df["DEFAULT"].values
    feature_names = cat_encoded.columns.tolist() + all_numeric
    return X, y, feature_names


def _find_optimal_threshold(y_test, y_proba, min_precision=0.42):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    mask = precisions[:-1] >= min_precision
    if mask.any():
        best_idx = int(recalls[:-1][mask].argmax())
        return float(thresholds[mask][best_idx])
    return 0.35


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#0f1a22",
        "axes.facecolor": "#163044",
        "axes.edgecolor": "#c9c4b5",
        "axes.labelcolor": "#efece5",
        "xtick.color": "#efece5",
        "ytick.color": "#efece5",
        "text.color": "#efece5",
        "font.size": 11,
        "axes.titleweight": "bold",
    })


def _save(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(BASE_DIR / filename, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _fit_rl():
    X, y, feature_names = _load_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    search = RandomizedSearchCV(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        ),
        param_distributions=RL_PARAM_DIST,
        n_iter=len(RL_PARAM_DIST["logisticregression__C"]),
        scoring="recall",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        random_state=42,
        refit=True,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = _find_optimal_threshold(y_test, y_proba, min_precision=0.42)
    y_pred = (y_proba >= threshold).astype(int)
    scores = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Précision": precision_score(y_test, y_pred, zero_division=0),
        "Rappel": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
    }
    return X, y, X_train, X_test, y_train, y_test, model, y_proba, y_pred, threshold, scores, feature_names


def _chart_resultats(y_test, y_pred, scores, threshold) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Régression Logistique — Résultats au seuil optimisé", fontsize=15)

    ax = axes[0]
    im = ax.imshow(cm, cmap="Greens")
    ax.set_title(f"Matrice de confusion (seuil = {threshold:.2f})")
    ax.set_xticks([0, 1], ["No Default", "Default"])
    ax.set_yticks([0, 1], ["No Default", "Default"])
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="#0f1a22", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    names = list(scores.keys())
    vals = list(scores.values())
    colors = ["#00b894", "#ffd166", "#ff6a3d", "#0984e3", "#6c5ce7"]
    ax.bar(names, vals, color=colors)
    ax.set_title("Synthèse des métriques")
    ax.set_ylim(0, 1.05)
    for idx, val in enumerate(vals):
        ax.text(idx, val + 0.02, f"{val:.3f}", ha="center")

    _save(fig, "charts_rl_resultats.png")


def _chart_roc_pr(y_test, y_proba, threshold) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    idx = int(np.argmin(np.abs(thresholds - threshold))) if len(thresholds) else 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Régression Logistique — Courbes ROC et Précision-Rappel", fontsize=15)

    axes[0].plot(fpr, tpr, color="#00b894", linewidth=2.5, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#c9c4b5")
    axes[0].set_title("Courbe ROC")
    axes[0].set_xlabel("Taux de faux positifs")
    axes[0].set_ylabel("Taux de vrais positifs")
    axes[0].legend()

    axes[1].plot(recall, precision, color="#00b894", linewidth=2.5, label=f"AP = {ap:.3f}")
    axes[1].scatter([recall[idx]], [precision[idx]], color="#ffd166", s=100, marker="*", label=f"Seuil {threshold:.2f}")
    axes[1].axhline(y_test.mean(), linestyle="--", color="#c9c4b5", label=f"Baseline {y_test.mean():.2f}")
    axes[1].set_title("Courbe Précision-Rappel")
    axes[1].set_xlabel("Rappel")
    axes[1].set_ylabel("Précision")
    axes[1].legend()

    _save(fig, "charts_rl_roc_pr.png")


def _chart_coefficients(model, feature_names) -> None:
    coefs = model.named_steps["logisticregression"].coef_[0]
    idx = np.argsort(np.abs(coefs))[::-1][:20]
    names = [feature_names[i] for i in idx][::-1]
    vals = coefs[idx][::-1]
    colors = ["#00b894" if v > 0 else "#ff6a3d" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#c9c4b5", linewidth=1)
    ax.set_title("Régression Logistique — Top 20 coefficients standardisés")
    ax.set_xlabel("Coefficient")
    _save(fig, "charts_rl_coefficients.png")


def _chart_cv(X, y, model) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        clone(model),
        X,
        y,
        cv=cv,
        scoring={"recall": "recall", "precision": "precision", "f1": "f1", "auc_roc": "roc_auc"},
        n_jobs=1,
    )
    folds = np.arange(1, 6)
    metrics = {
        "Rappel": scores["test_recall"],
        "Précision": scores["test_precision"],
        "F1": scores["test_f1"],
        "AUC-ROC": scores["test_auc_roc"],
    }
    colors = ["#ff6a3d", "#ffd166", "#00b894", "#6c5ce7"]

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.18
    for idx, (name, vals) in enumerate(metrics.items()):
        offset = (idx - 1.5) * width
        ax.bar(folds + offset, vals, width=width, label=f"{name} moy. {vals.mean():.3f}", color=colors[idx])
    ax.set_title("Régression Logistique — Validation croisée stratifiée 5 folds")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(folds)
    ax.legend()
    _save(fig, "charts_rl_cv.png")


def _chart_learning(X_train, X_test, y_train, y_test, best_c) -> None:
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    train_scores, test_scores = [], []
    rng = np.random.default_rng(42)

    for frac in fractions:
        n = max(int(len(X_train) * frac), 50)
        idx = rng.choice(len(X_train), size=n, replace=False)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000, random_state=42),
        )
        clf.fit(X_train[idx], y_train[idx])
        train_scores.append(roc_auc_score(y_train[idx], clf.predict_proba(X_train[idx])[:, 1]))
        test_scores.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    sizes = [int(len(X_train) * f) for f in fractions]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, train_scores, marker="o", color="#ffd166", label="Train AUC")
    ax.plot(sizes, test_scores, marker="o", color="#00b894", label="Test AUC")
    ax.set_title("Régression Logistique — Courbe d'apprentissage")
    ax.set_xlabel("Taille de l'ensemble d'entraînement")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    _save(fig, "charts_rl_learning.png")


def main() -> None:
    _style()
    X, y, X_train, X_test, y_train, y_test, model, y_proba, y_pred, threshold, scores, feature_names = _fit_rl()
    _chart_resultats(y_test, y_pred, scores, threshold)
    _chart_roc_pr(y_test, y_proba, threshold)
    _chart_coefficients(model, feature_names)
    _chart_cv(X, y, model)
    _chart_learning(X_train, X_test, y_train, y_test, model.named_steps["logisticregression"].C)
    print("Images RL générées : charts_rl_resultats.png, charts_rl_roc_pr.png, charts_rl_coefficients.png, charts_rl_cv.png, charts_rl_learning.png")


if __name__ == "__main__":
    main()
