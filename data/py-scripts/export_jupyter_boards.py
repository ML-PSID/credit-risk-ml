from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / 'data' / 'notebook' / 'default of credit card clients.xls'
DEFAULT_OUTPUT_DIR = ROOT / 'data' / 'exports' / 'boards'

COLORS = {
    'default': '#e65d1b',
    'non_default': '#2fb8a9',
    'accent': '#f4b36a',
    'ink': '#f6efe3',
    'grid': 'rgba(255,255,255,0.1)',
}


def recode_idempotent(series: pd.Series, mapping: dict[int, str]) -> pd.Series:
    known_labels = set(mapping.values())
    numeric_series = pd.to_numeric(series, errors='coerce')
    mapped = numeric_series.map(mapping)
    return series.where(series.isin(known_labels), mapped)


def prepare_dataset() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE, sheet_name=0, header=1).copy()
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    if 'EDUCATION' in df.columns:
        df['EDUCATION'] = recode_idempotent(
            df['EDUCATION'],
            {1: 'graduate', 2: 'university', 3: 'high_school', 4: 'others', 0: 'unknown', 5: 'unknown', 6: 'unknown'},
        )
    if 'MARRIAGE' in df.columns:
        df['MARRIAGE'] = recode_idempotent(df['MARRIAGE'], {1: 'married', 2: 'single', 0: 'other', 3: 'other'})
    if 'SEX' in df.columns:
        df['SEX'] = recode_idempotent(df['SEX'], {1: 'male', 2: 'female'})

    if 'DEFAULT' not in df.columns:
        candidates = [c for c in df.columns if 'default' in c.lower() and 'payment' in c.lower()]
        if not candidates:
            raise ValueError('Target column not found in dataset.')
        df = df.rename(columns={candidates[0]: 'DEFAULT'})

    df['DEFAULT'] = pd.to_numeric(df['DEFAULT'], errors='coerce').fillna(0).astype(int)

    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    df['avg_pay_delay'] = df[pay_cols].mean(axis=1)
    df['n_months_late'] = (df[pay_cols] > 0).sum(axis=1)
    df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
    df['avg_pay_amt'] = df[pay_amt_cols].mean(axis=1)
    df['repayment_ratio'] = np.where(df['avg_bill_amt'] > 0, df['avg_pay_amt'] / df['avg_bill_amt'], 0)
    df['repayment_ratio'] = pd.Series(df['repayment_ratio']).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 10)

    return df


def base_layout(title: str, height: int = 360) -> dict:
    return {
        'title': {'text': title, 'font': {'color': COLORS['ink'], 'size': 16}},
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': COLORS['ink']},
        'height': height,
        'margin': {'l': 50, 'r': 20, 't': 55, 'b': 50},
        'xaxis': {'gridcolor': COLORS['grid'], 'zerolinecolor': COLORS['grid']},
        'yaxis': {'gridcolor': COLORS['grid'], 'zerolinecolor': COLORS['grid']},
        'legend': {'orientation': 'h', 'y': -0.2},
    }


def kpis(df: pd.DataFrame) -> dict[str, str]:
    return {
        'Nombre de clients': f"{len(df):,}".replace(',', ' '),
        'Taux de défaut': f"{df['DEFAULT'].mean():.2%}",
        'Nombre de défauts': f"{int(df['DEFAULT'].sum()):,}".replace(',', ' '),
        'Crédit moyen': f"{df['LIMIT_BAL'].mean():,.0f} NT$".replace(',', ' '),
        'Crédit médian': f"{df['LIMIT_BAL'].median():,.0f} NT$".replace(',', ' '),
        'Âge moyen': f"{df['AGE'].mean():.1f} ans",
    }


def chart_default_distribution(df: pd.DataFrame) -> dict:
    counts = df['DEFAULT'].value_counts().sort_index()
    data = [{
        'type': 'bar',
        'x': ['Hors défaut', 'Défaut'],
        'y': [int(counts.get(0, 0)), int(counts.get(1, 0))],
        'marker': {'color': [COLORS['non_default'], COLORS['default']]},
    }]
    layout = base_layout('Répartition défaut / hors défaut')
    return {'data': data, 'layout': layout}


def chart_histograms(df: pd.DataFrame) -> dict:
    def_rows = df[df['DEFAULT'] == 1]['AGE']
    non_rows = df[df['DEFAULT'] == 0]['AGE']
    bins = np.arange(20, 81, 4)
    def_counts, edges = np.histogram(def_rows, bins=bins)
    non_counts, _ = np.histogram(non_rows, bins=bins)
    labels = [f'{int(edges[i])}-{int(edges[i + 1]) - 1}' for i in range(len(edges) - 1)]
    data = [
        {'type': 'bar', 'name': 'Défaut', 'x': labels, 'y': def_counts.tolist(), 'marker': {'color': COLORS['default']}, 'opacity': 0.82},
        {'type': 'bar', 'name': 'Hors défaut', 'x': labels, 'y': non_counts.tolist(), 'marker': {'color': COLORS['non_default']}, 'opacity': 0.7},
    ]
    layout = base_layout('Histogramme AGE (bins)')
    layout['barmode'] = 'group'
    return {'data': data, 'layout': layout}


def chart_limit_hist(df: pd.DataFrame) -> dict:
    def_rows = df[df['DEFAULT'] == 1]['LIMIT_BAL']
    non_rows = df[df['DEFAULT'] == 0]['LIMIT_BAL']
    bins = np.arange(0, 650000, 50000)
    def_counts, edges = np.histogram(def_rows, bins=bins)
    non_counts, _ = np.histogram(non_rows, bins=bins)
    labels = [f'{int(edges[i] / 1000)}k-{int(edges[i + 1] / 1000)}k' for i in range(len(edges) - 1)]
    data = [
        {'type': 'bar', 'name': 'Défaut', 'x': labels, 'y': def_counts.tolist(), 'marker': {'color': COLORS['default']}, 'opacity': 0.82},
        {'type': 'bar', 'name': 'Hors défaut', 'x': labels, 'y': non_counts.tolist(), 'marker': {'color': COLORS['non_default']}, 'opacity': 0.7},
    ]
    layout = base_layout('Histogramme LIMIT_BAL (bins)')
    layout['barmode'] = 'group'
    return {'data': data, 'layout': layout}


def chart_rates_by_categorical(df: pd.DataFrame, col: str, title: str) -> dict:
    grp = (df.groupby(col)['DEFAULT'].mean() * 100).sort_values(ascending=False)
    data = [{'type': 'bar', 'x': grp.index.astype(str).tolist(), 'y': grp.values.round(2).tolist(), 'marker': {'color': COLORS['accent']}}]
    layout = base_layout(title)
    layout['yaxis']['title'] = 'Taux de défaut (%)'
    return {'data': data, 'layout': layout}


def chart_pay0_default(df: pd.DataFrame) -> dict:
    grp = df.groupby('PAY_0')['DEFAULT'].agg(['mean', 'count'])
    grp = grp[grp['count'] > 10]
    data = [{'type': 'bar', 'x': grp.index.astype(str).tolist(), 'y': (grp['mean'] * 100).round(2).tolist(), 'marker': {'color': COLORS['default']}}]
    layout = base_layout('Taux de défaut par PAY_0')
    layout['yaxis']['title'] = 'Taux de défaut (%)'
    return {'data': data, 'layout': layout}


def chart_pay_heatmap(df: pd.DataFrame) -> dict:
    cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    z = [
        [float(df[df['DEFAULT'] == 0][c].mean()) for c in cols],
        [float(df[df['DEFAULT'] == 1][c].mean()) for c in cols],
    ]
    data = [{'type': 'heatmap', 'z': z, 'x': cols, 'y': ['Hors défaut', 'Défaut'], 'colorscale': 'RdYlGn_r'}]
    layout = base_layout('Heatmap retards PAY_* par classe')
    return {'data': data, 'layout': layout}


def chart_monthly_finance(df: pd.DataFrame) -> dict:
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    months = ['Sep', 'Aoû', 'Juil', 'Juin', 'Mai', 'Avr']

    d = df[df['DEFAULT'] == 1]
    n = df[df['DEFAULT'] == 0]
    data = [
        {'type': 'scatter', 'mode': 'lines+markers', 'name': 'Facturé défaut', 'x': months, 'y': d[bill_cols].mean().round(0).tolist(), 'line': {'color': COLORS['default'], 'width': 3}},
        {'type': 'scatter', 'mode': 'lines+markers', 'name': 'Payé défaut', 'x': months, 'y': d[pay_cols].mean().round(0).tolist(), 'line': {'color': COLORS['accent'], 'width': 3}},
        {'type': 'scatter', 'mode': 'lines+markers', 'name': 'Facturé hors défaut', 'x': months, 'y': n[bill_cols].mean().round(0).tolist(), 'line': {'color': '#bfe8dc', 'width': 2}},
        {'type': 'scatter', 'mode': 'lines+markers', 'name': 'Payé hors défaut', 'x': months, 'y': n[pay_cols].mean().round(0).tolist(), 'line': {'color': '#d9f2ec', 'width': 2}},
    ]
    layout = base_layout('Facturé vs payé (moyennes mensuelles)')
    return {'data': data, 'layout': layout}


def chart_scatter(df: pd.DataFrame) -> dict:
    sample = df.sample(min(2500, len(df)), random_state=42)
    d = sample[sample['DEFAULT'] == 1]
    n = sample[sample['DEFAULT'] == 0]
    data = [
        {'type': 'scattergl', 'mode': 'markers', 'name': 'Défaut', 'x': d['BILL_AMT1'].tolist(), 'y': d['PAY_AMT1'].tolist(), 'marker': {'color': COLORS['default'], 'size': 6, 'opacity': 0.5}},
        {'type': 'scattergl', 'mode': 'markers', 'name': 'Hors défaut', 'x': n['BILL_AMT1'].tolist(), 'y': n['PAY_AMT1'].tolist(), 'marker': {'color': COLORS['non_default'], 'size': 6, 'opacity': 0.45}},
    ]
    layout = base_layout('Nuage BILL_AMT1 vs PAY_AMT1')
    layout['xaxis']['title'] = 'BILL_AMT1'
    layout['yaxis']['title'] = 'PAY_AMT1'
    return {'data': data, 'layout': layout}


def chart_repayment_quantile(df: pd.DataFrame) -> dict:
    temp = df.copy()
    temp['repayment_quantile'] = pd.qcut(temp['repayment_ratio'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
    grp = (temp.groupby('repayment_quantile', observed=True)['DEFAULT'].mean() * 100).round(2)
    data = [{'type': 'bar', 'x': grp.index.astype(str).tolist(), 'y': grp.values.tolist(), 'marker': {'color': ['#e74c3c', '#f39c12', '#f1c40f', '#3498db', '#2ecc71'][: len(grp)]}}]
    layout = base_layout('Taux de défaut par quantile de repayment_ratio')
    return {'data': data, 'layout': layout}


def chart_feature_importance(df: pd.DataFrame) -> dict:
    features = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'avg_pay_delay', 'n_months_late', 'avg_bill_amt', 'avg_pay_amt', 'repayment_ratio', 'LIMIT_BAL', 'AGE']
    corr = df[features + ['DEFAULT']].corr()['DEFAULT'].drop('DEFAULT').abs().sort_values(ascending=True)
    top = corr.tail(12)
    data = [{'type': 'bar', 'orientation': 'h', 'x': top.values.round(3).tolist(), 'y': top.index.tolist(), 'marker': {'color': '#e67e22'}}]
    layout = base_layout('Top variables (|corrélation| avec DEFAULT)', height=420)
    return {'data': data, 'layout': layout}


def export(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_dataset()

    manifest = {
        'title': 'Dashboard Analytique - UCI Credit Card (Interactif)',
        'kpis': kpis(df),
        'source': str(DATA_FILE),
        'generated_at': pd.Timestamp.utcnow().isoformat(),
        'sections': [
            {
                'id': 'axe-1',
                'title': 'Axe 1 - Profil du portefeuille',
                'question': 'Comment se répartit le risque dans la population ?',
                'insight': 'Le défaut reste minoritaire mais structurel, avec des différences visibles selon les segments.',
                'charts': [
                    chart_default_distribution(df),
                    chart_histograms(df),
                    chart_limit_hist(df),
                    chart_rates_by_categorical(df, 'SEX', 'Taux de défaut par sexe'),
                    chart_rates_by_categorical(df, 'EDUCATION', 'Taux de défaut par éducation'),
                    chart_rates_by_categorical(df, 'MARRIAGE', 'Taux de défaut par état civil'),
                ],
            },
            {
                'id': 'axe-2',
                'title': 'Axe 2 - Comportement de paiement',
                'question': 'Les retards PAY_* constituent-ils des signaux précoces ?',
                'insight': 'PAY_0 et la persistance des retards sont des indicateurs majeurs du risque.',
                'charts': [
                    chart_pay0_default(df),
                    chart_pay_heatmap(df),
                ],
            },
            {
                'id': 'axe-3',
                'title': 'Axe 3 - Exposition financière',
                'question': 'Le risque vient-il du volume ou du déséquilibre facturé/payé ?',
                'insight': 'Le déséquilibre remboursement/facturation discrimine mieux que le montant brut isolé.',
                'charts': [
                    chart_monthly_finance(df),
                    chart_scatter(df),
                    chart_repayment_quantile(df),
                ],
            },
            {
                'id': 'axe-4',
                'title': 'Axe 4 - Variables pertinentes',
                'question': 'Quelles variables prioriser pour la modélisation ?',
                'insight': 'Les variables comportementales dominent les descriptives dans l’explication du défaut.',
                'charts': [
                    chart_feature_importance(df),
                ],
            },
        ],
    }

    with open(output_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False)

    print(f'Export done: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export interactive dashboard data for frontend.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    export(args.output_dir)
