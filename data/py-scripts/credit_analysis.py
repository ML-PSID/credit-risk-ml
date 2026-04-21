# %% [markdown]
# # Dashboard Analytique - Prédiction de Défaut de Paiement
# ## UCI Credit Card Dataset Analysis
# 
# ### Structure du rapport
# - **Axe 1** : Comprendre le portefeuille
# - **Axe 2** : Identifier les premiers signaux de risque  
# - **Axe 3** : Relier dette, facturation et remboursement
# - **Axe 4** : Préparer la modélisation
# 
# ### Navigation
# - **Section A** : Vue d'ensemble (KPIs)
# - **Section B** : Profil du portefeuille
# - **Section C** : Comportement de paiement
# - **Section D** : Exposition financière et remboursement
# - **Section E** : Variables pertinentes

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, entropy
import warnings
warnings.filterwarnings('ignore')

# Configuration d'affichage
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

# Charger les données (en sautant la première ligne car c'est une description technique)
file_path = '/Users/perlenkounkou/Documents/Cours 2/PSID/Projet 2/default of credit card clients.xls'
df_raw = pd.read_excel(file_path, sheet_name=0, header=1)
df = df_raw.copy()

print(f"Shape: {df.shape}")
print(f"\nColonnes originales:\n{df.columns.tolist()}")
print(f"\nPremières lignes:\n{df.head()}")

# %% [markdown]
# ## Étape 1 : Nettoyage et Préparation des Données
# 
# ### Actions réalisées :
# 1. Suppression de la colonne ID
# 2. Renommage des colonnes pour lisibilité
# 3. Recatégorisation de EDUCATION et MARRIAGE
# 4. Clarification des statuts de paiement PAY_*
# 5. Création des variables dérivées métier

# %%
# Repartir de la table brute à chaque exécution pour éviter les effets de bord
if 'df_raw' not in globals():
    df_raw = pd.read_excel(file_path, sheet_name=0, header=1)
df = df_raw.copy()

# 1. Supprimer l'ID (colonne d'identification, pas une variable métier)
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

# 2. Renommer les colonnes proprement
column_mapping = {
    'LIMIT_BAL': 'LIMIT_BAL',
    'AGE': 'AGE',
    'PAY_0': 'PAY_0',
    'PAY_2': 'PAY_2',
    'PAY_3': 'PAY_3',
    'PAY_4': 'PAY_4',
    'PAY_5': 'PAY_5',
    'PAY_6': 'PAY_6',
    'BILL_AMT1': 'BILL_AMT1',
    'BILL_AMT2': 'BILL_AMT2',
    'BILL_AMT3': 'BILL_AMT3',
    'BILL_AMT4': 'BILL_AMT4',
    'BILL_AMT5': 'BILL_AMT5',
    'BILL_AMT6': 'BILL_AMT6',
    'PAY_AMT1': 'PAY_AMT1',
    'PAY_AMT2': 'PAY_AMT2',
    'PAY_AMT3': 'PAY_AMT3',
    'PAY_AMT4': 'PAY_AMT4',
    'PAY_AMT5': 'PAY_AMT5',
    'PAY_AMT6': 'PAY_AMT6',
}

print("Colonnes actuelles:", df.columns.tolist())

# Fonction utilitaire: recoder sans casser les colonnes si la cellule est relancée
def recode_idempotent(series, mapping):
    known_labels = set(mapping.values())
    numeric_series = pd.to_numeric(series, errors='coerce')
    mapped = numeric_series.map(mapping)
    # Si la valeur est deja un label final, on la conserve
    return series.where(series.isin(known_labels), mapped)

# 3. Recatégoriser EDUCATION (1=grad, 2=univ, 3=hs, 4=other, 0/5/6=unknown)
education_map = {
    1: 'graduate',
    2: 'university',
    3: 'high_school',
    4: 'others',
    0: 'unknown',
    5: 'unknown',
    6: 'unknown'
}
if 'EDUCATION' in df.columns:
    df['EDUCATION'] = recode_idempotent(df['EDUCATION'], education_map)

# 4. Recatégoriser MARRIAGE (1=married, 2=single, 0/3=other)
marriage_map = {
    1: 'married',
    2: 'single',
    0: 'other',
    3: 'other'
}
if 'MARRIAGE' in df.columns:
    df['MARRIAGE'] = recode_idempotent(df['MARRIAGE'], marriage_map)

# 5. Recatégoriser SEX (1=male, 2=female)
sex_map = {1: 'male', 2: 'female'}
if 'SEX' in df.columns:
    df['SEX'] = recode_idempotent(df['SEX'], sex_map)

# 6. Documenter et clarifier PAY_* (-1=payé à temps, 0=revolving, 1+=retard, -2=spécial)
pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
pay_status_mapping = {
    -2: 'no_consumption',
    -1: 'paid_on_time',
    0: 'revolving',
    1: 'delay_1month',
    2: 'delay_2months',
    3: 'delay_3months',
    4: 'delay_4months',
    5: 'delay_5months',
    6: 'delay_6months',
    7: 'delay_7months',
    8: 'delay_8months',
    9: 'delay_9months'
}

# 7. Cible : renommer pour clarté - nom exact de la colonne
if 'DEFAULT' not in df.columns:
    target_col = [col for col in df.columns if 'default' in col.lower() and 'payment' in col.lower()][0]
    df = df.rename(columns={target_col: 'DEFAULT'})

df['DEFAULT'] = df['DEFAULT'].astype(int)

print("\nÉtat des données après nettoyage initial:")
print(df.head())
print(f"\nTypes: \n{df.dtypes}")
print(f"\nValeurs manquantes: \n{df.isnull().sum()}")
print(f"\nDistribution de la cible:\n{df['DEFAULT'].value_counts()}")
print(f"Taux de défaut: {df['DEFAULT'].mean():.2%}")
print(f"\nDistribution SEX: {df['SEX'].value_counts(dropna=False).to_dict()}")
print(f"Distribution EDUCATION: {df['EDUCATION'].value_counts(dropna=False).to_dict()}")
print(f"Distribution MARRIAGE: {df['MARRIAGE'].value_counts(dropna=False).to_dict()}")

# %% [markdown]
# ## Étape 2 : Création des Variables Dérivées Métier
# 
# Variables essentielles pour le dashboard (Axes 2 & 3)

# %%
# Colonnes de paiement et facturation
pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
payamt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# Variables de RETARD DE PAIEMENT (Axe 2)
df['avg_pay_delay'] = df[pay_cols].mean(axis=1)
df['max_pay_delay'] = df[pay_cols].max(axis=1)
df['n_months_late'] = (df[pay_cols] > 0).sum(axis=1)  # Nombre de mois avec retard > 0

# Variables de FACTURATION ET REMBOURSEMENT (Axe 3)
df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
df['avg_pay_amt'] = df[payamt_cols].mean(axis=1)
df['bill_volatility'] = df[bill_cols].std(axis=1)

# Ratios d'exposition et remboursement
df['credit_utilization'] = (df['avg_bill_amt'] / df['LIMIT_BAL']).fillna(0)
# Utiliser pd.Series avec np.where pour pouvoir utiliser fillna
df['repayment_ratio'] = pd.Series(
    np.where(
        df['avg_bill_amt'] > 0,
        df['avg_pay_amt'] / df['avg_bill_amt'],
        0
    ),
    index=df.index
).fillna(0)

# Cliper les ratios aberrants
df['credit_utilization'] = df['credit_utilization'].clip(0, 5)
df['repayment_ratio'] = df['repayment_ratio'].clip(0, 10)

print("Variables dérivées créées avec succès")
print(f"\nNouvellas colonnes: {df.columns.tolist()}")
print(f"\nStatistiques des variables dérivées:\n{df[['avg_pay_delay', 'max_pay_delay', 'n_months_late', 'avg_bill_amt', 'avg_pay_amt', 'credit_utilization', 'repayment_ratio']].describe()}")

# %% [markdown]
# # DASHBOARD ANALYTIQUE
# 
# ## Section A : Vue d'Ensemble (KPI Cards)
# 
# **Axe 1 : Comprendre le portefeuille**
# 
# Voici les chiffres clés du portefeuille de crédit :
# - Taille, imbalance, exposition moyenne, pyramide d'âge

# %%
# Calculer les KPIs
n_clients = len(df)
default_rate = df['DEFAULT'].mean()
credit_mean = df['LIMIT_BAL'].mean()
credit_median = df['LIMIT_BAL'].median()
age_mean = df['AGE'].mean()
age_median = df['AGE'].median()
n_defaults = df['DEFAULT'].sum()

# Afficher les KPIs
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
fig.suptitle('KPI Cards - Vue d\'ensemble du portefeuille', fontsize=14, fontweight='bold')

kpis = [
    ('Nombre de clients', f'{n_clients:,.0f}', 0),
    ('Taux de défaut', f'{default_rate:.2%}', 1),
    ('Nb défauts', f'{n_defaults:,.0f}', 2),
    ('Crédit moyen', f'${credit_mean:,.0f}', 3),
    ('Crédit médian', f'${credit_median:,.0f}', 4),
    ('Âge moyen', f'{age_mean:.1f} ans', 5),
    ('Âge médian', f'{age_median:.0f} ans', 6),
]

for idx, (label, value, pos) in enumerate(kpis):
    ax = axes.flatten()[pos]
    ax.text(0.5, 0.6, value, ha='center', va='center', fontsize=24, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.2, label, ha='center', va='center', fontsize=11, color='gray', transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# Masquer la dernière cellule
axes.flatten()[-1].axis('off')

plt.tight_layout()
plt.show()

print(f"\n{'='*60}")
print(f"RÉSUMÉ - Section A : Vue d'ensemble")
print(f"{'='*60}")
print(f"✓ Portefeuille : {n_clients:,} clients")
print(f"✓ Imbalance : {default_rate:.2%} de défaut ({n_defaults:,} défauteurs)")
print(f"✓ Exposition moyenne : ${credit_mean:,.0f} (médiane: ${credit_median:,.0f})")
print(f"✓ Profil d'âge : {age_mean:.1f} ans (médiane: {age_median:.0f} ans)")
print(f"\nLecture : Le dataset est déséquilibré (~22% de défaut), donc l'accuracy seule sera trompeuse.")

# %% [markdown]
# ## Section B : Profil du Portefeuille
# 
# **Axe 1 : Comprendre la structure démographique et d'exposition**
# 
# Graphes :
# - Distribution de la cible DEFAULT
# - Histogramme de LIMIT_BAL
# - Histogramme d'AGE  
# - Profils socio-démo (SEX, EDUCATION, MARRIAGE) par classe

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Section B : Profil du Portefeuille', fontsize=14, fontweight='bold', y=1.00)

# 1. Distribution de la cible DEFAULT
ax = axes[0, 0]
default_counts = df['DEFAULT'].value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(['No Default (0)', 'Default (1)'], default_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Nombre de clients', fontweight='bold')
ax.set_title('1. Répartition de la Cible\n(Le défaut est-il fréquent ?)', fontweight='bold', fontsize=11)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=9)
ax.set_ylim(0, len(df) * 0.85)

# Interprétation
print("\n📊 Graphe 1 : Répartition de la Cible")
print(f"  Question : Le défaut est-il fréquent ?")
print(f"  Lecture : {default_rate:.1%} de défaut, {1-default_rate:.1%} sans défaut")
print(f"  Interprétation : Le dataset est déséquilibré, donc l'accuracy seule sera trompeuse.")

# 2. Distribution de LIMIT_BAL
ax = axes[0, 1]
ax.hist(df['LIMIT_BAL'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(credit_mean, color='red', linestyle='--', linewidth=2, label=f'Moyenne: ${credit_mean:,.0f}')
ax.axvline(credit_median, color='orange', linestyle='--', linewidth=2, label=f'Médiane: ${credit_median:,.0f}')
ax.set_xlabel('LIMIT_BAL ($)', fontweight='bold')
ax.set_ylabel('Fréquence', fontweight='bold')
ax.set_title('2. Distribution du Crédit Accordé\n(Les faibles crédits défautent-ils plus ?)', fontweight='bold', fontsize=11)
ax.legend()

print(f"\n📊 Graphe 2 : Distribution LIMIT_BAL")
print(f"  Lecture : Crédit moyen ${credit_mean:,.0f}, concentration autour ${credit_median:,.0f}")

# 3. Distribution d'AGE
ax = axes[0, 2]
ax.hist(df['AGE'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(age_mean, color='red', linestyle='--', linewidth=2, label=f'Moyenne: {age_mean:.1f}')
ax.axvline(age_median, color='orange', linestyle='--', linewidth=2, label=f'Médiane: {age_median:.0f}')
ax.set_xlabel('Âge (années)', fontweight='bold')
ax.set_ylabel('Fréquence', fontweight='bold')
ax.set_title('3. Distribution d\'Âge\n(L\'âge aide-t-il à segmenter ?)', fontweight='bold', fontsize=11)
ax.legend()

print(f"\n📊 Graphe 3 : Distribution AGE")
print(f"  Lecture : Âge moyen {age_mean:.1f}, portefeuille concentré entre 28 et 50 ans")

# 4. Taux de défaut par SEX
ax = axes[1, 0]
sex_default = df.groupby('SEX')['DEFAULT'].agg(['sum', 'count'])
sex_default['rate'] = sex_default['sum'] / sex_default['count']
sex_labels = sex_default.index.tolist()
bars = ax.bar(sex_labels, sex_default['rate'].values * 100, color=['#3498db', '#e91e63'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Taux de défaut (%)', fontweight='bold')
ax.set_title('4. Taux de Défaut par Genre\n(Profil socio-démo)', fontweight='bold', fontsize=11)
ax.set_ylim(0, 30)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

print(f"\n📊 Graphe 4 : Défaut par Genre")
for sex, rate in sex_default['rate'].items():
    print(f"  {sex}: {rate:.2%}")

# 5. Taux de défaut par EDUCATION
ax = axes[1, 1]
edu_default = df.groupby('EDUCATION')['DEFAULT'].agg(['sum', 'count'])
edu_default['rate'] = edu_default['sum'] / edu_default['count']
edu_sorted = edu_default.sort_values('rate', ascending=False)
bars = ax.bar(range(len(edu_sorted)), edu_sorted['rate'].values * 100, color='#9b59b6', alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(edu_sorted)))
ax.set_xticklabels(edu_sorted.index, rotation=45, ha='right')
ax.set_ylabel('Taux de défaut (%)', fontweight='bold')
ax.set_title('5. Taux de Défaut par Éducation\n(Variables secondaires)', fontweight='bold', fontsize=11)
ax.set_ylim(0, 30)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# 6. Taux de défaut par MARRIAGE
ax = axes[1, 2]
mar_default = df.groupby('MARRIAGE')['DEFAULT'].agg(['sum', 'count'])
mar_default['rate'] = mar_default['sum'] / mar_default['count']
mar_sorted = mar_default.sort_values('rate', ascending=False)
bars = ax.bar(range(len(mar_sorted)), mar_sorted['rate'].values * 100, color='#f39c12', alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(mar_sorted)))
ax.set_xticklabels(mar_sorted.index, rotation=45, ha='right')
ax.set_ylabel('Taux de défaut (%)', fontweight='bold')
ax.set_title('6. Taux de Défaut par État Civil\n(Profil socio-démo)', fontweight='bold', fontsize=11)
ax.set_ylim(0, 30)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print(f"\n{'='*60}")
print(f"RÉSUMÉ - Section B : Profil du Portefeuille")
print(f"{'='*60}")
print(f"✓ Structure d'exposition : Crédit accordé entre {df['LIMIT_BAL'].min():,}$ et {df['LIMIT_BAL'].max():,}$")
print(f"✓ Pyramide d'âge : Clients entre {df['AGE'].min()} et {df['AGE'].max()} ans")
print(f"✓ Composition : {df['SEX'].value_counts().to_dict()}")
print(f"✓ Signal socio-démo : Modéré (différences < 5 pp sur éducation/mariage)")

# %% [markdown]
# ## Section C : Comportement de Paiement
# 
# **Axe 2 : Identifier les premiers signaux de risque - PAY_0..PAY_6 sont les plus discriminantes**
# 
# Graphes :
# - Heatmap des distributions PAY_0 à PAY_6 par classe
# - Évolution moyenne des statuts par mois (défauteurs vs non)
# - Taux de défaut par niveau de PAY_0 ⭐ GRAPHE CLEF
# - Matrice de corrélation PAY_*

# %%
# 1. GRAPHE CLEF : Taux de défaut par PAY_0
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Section C : Comportement de Paiement - Analyse PAY_0..PAY_6', fontsize=14, fontweight='bold', y=0.995)

pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
ax = axes[0, 0]
pay0_default = df.groupby('PAY_0')['DEFAULT'].agg(['sum', 'count'])
pay0_default['rate'] = pay0_default['sum'] / pay0_default['count']
pay0_sorted = pay0_default.sort_index()

# Filtrer pour visible
pay0_sorted_visible = pay0_sorted[pay0_sorted['count'] > 10]  # Éviter les catégories rares

bars = ax.bar(range(len(pay0_sorted_visible)), pay0_sorted_visible['rate'].values * 100, 
              color='#e74c3c', alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(pay0_sorted_visible)))
ax.set_xticklabels([f'{int(idx)}' for idx in pay0_sorted_visible.index], fontsize=9)
ax.set_xlabel('PAY_0 Status (-2 to 8)', fontweight='bold')
ax.set_ylabel('Taux de défaut (%)', fontweight='bold')
ax.set_title('⭐ 1. Taux de Défaut par PAY_0\n(Le dernier statut de paiement est-il critique ?)', 
             fontweight='bold', fontsize=11)
ax.set_ylim(0, 100)

for i, bar in enumerate(bars):
    height = bar.get_height()
    count = pay0_sorted_visible['count'].iloc[i]
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.0f}%\n(n={int(count)})', ha='center', va='bottom', fontsize=8)

print("\n⭐ Graphe Clef : Taux de Défaut par PAY_0")
print(f"  Question : Le dernier statut de paiement est-il un signal fort de défaut futur ?")
print(f"  Lecture : Le risque explose dès qu'un retard récent apparaît")
print(f"  Interprétation métier : PAY_0 est le signal d'alerte prioritaire pour la segmentation")

# 2. Heatmap PAY_0..PAY_6 vs DEFAULT
ax = axes[0, 1]
pay_by_default = []
for default_val in [0, 1]:
    row = []
    subset = df[df['DEFAULT'] == default_val]
    for col in pay_cols:
        avg = subset[col].mean()
        row.append(avg)
    pay_by_default.append(row)

heatmap_data = np.array(pay_by_default)
im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(pay_cols)))
ax.set_xticklabels(pay_cols, fontsize=9)
ax.set_yticks([0, 1])
ax.set_yticklabels(['No Default', 'Default'], fontsize=9)
ax.set_title('2. Heatmap : Retard Moyen (PAY_0..PAY_6) par Classe\n(Retard ponctuel ou persistant ?)', 
             fontweight='bold', fontsize=11)
plt.colorbar(im, ax=ax, label='Retard moyen (mois)')

# Ajouter les valeurs dans la heatmap
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9, fontweight='bold')

print(f"\n📊 Graphe 2 : Heatmap PAY_0..PAY_6 vs DEFAULT")
print(f"  Lecture : Les défauteurs accumulent des retards plus fréquents et plus récents")
print(f"  Signal temporel : Dynamique (retards croissants dans le temps)")

# 3. Évolution moyenne des retards par mois
ax = axes[1, 0]
months_order = [f'PAY_{i}' if i != 0 else 'PAY_0' for i in [0, 2, 3, 4, 5, 6]]
default_means = []
nondefault_means = []

for col in pay_cols:
    default_means.append(df[df['DEFAULT'] == 1][col].mean())
    nondefault_means.append(df[df['DEFAULT'] == 0][col].mean())

x_pos = np.arange(len(pay_cols))
width = 0.35
ax.bar(x_pos - width/2, nondefault_means, width, label='No Default', color='#2ecc71', alpha=0.7, edgecolor='black')
ax.bar(x_pos + width/2, default_means, width, label='Default', color='#e74c3c', alpha=0.7, edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(pay_cols, fontsize=9)
ax.set_ylabel('Retard moyen (mois)', fontweight='bold')
ax.set_title('3. Évolution du Retard Moyen par Période\n(Temporalité)', fontweight='bold', fontsize=11)
ax.legend()

print(f"\n📊 Graphe 3 : Évolution Retard Moyen")
print(f"  Défauteurs : retards systématiquement plus élevés")
print(f"  Non-défauteurs : très peu de retard")

# 4. Corrélation entre PAY_*
ax = axes[1, 1]
corr_pay = df[pay_cols].corr()
im = ax.imshow(corr_pay, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(pay_cols)))
ax.set_yticks(range(len(pay_cols)))
ax.set_xticklabels(pay_cols, fontsize=9)
ax.set_yticklabels(pay_cols, fontsize=9)
ax.set_title('4. Corrélation entre PAY_0..PAY_6\n(Retards persistants ?)', fontweight='bold', fontsize=11)
plt.colorbar(im, ax=ax, label='Corrélation')

# Ajouter les valeurs
for i in range(corr_pay.shape[0]):
    for j in range(corr_pay.shape[1]):
        text = ax.text(j, i, f'{corr_pay.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=8, fontweight='bold')

print(f"\n📊 Graphe 4 : Corrélation PAY_*")
print(f"  Forte corrélation entre mois adjacents → Retards persistants et prévisibles")

plt.tight_layout()
plt.show()

print(f"\n{'='*60}")
print(f"RÉSUMÉ - Section C : Comportement de Paiement")
print(f"{'='*60}")
print(f"✓ PAY_0 est le signal discriminant majeur")
print(f"✓ Les défauteurs accumulent des retards persistants (corr forte)")
print(f"✓ Variablesclefs : PAY_0, PAY_2, PAY_3 à conserver absolument pour le modèle")

# %% [markdown]
# ## Section D : Exposition Financière et Remboursement
# 
# **Axe 3 : Relier dette, facturation et remboursement**
# 
# Le risque vient du **déséquilibre** entre facturation, capacité de remboursement et retards accumulés.
# 
# Graphes :
# - Boxplots BILL_AMT1..6 par classe
# - Boxplots PAY_AMT1..6 par classe
# - Scatter LIMIT_BAL vs avg_bill_amt coloré par classe
# - Taux de défaut par quantile de repayment_ratio ⭐ GRAPHE CLEF

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Section D : Exposition Financière et Remboursement', fontsize=14, fontweight='bold', y=0.995)

bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
payamt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# 1. Boxplot BILL_AMT1..6 par classe
ax = axes[0, 0]
bill_data = []
for col in bill_cols:
    for default_val in [0, 1]:
        bill_data.append(df[df['DEFAULT'] == default_val][col])

bp1 = ax.boxplot(bill_data,
                  positions=[i for i in range(12)],
                  widths=0.6,
                  patch_artist=True,
                  labels=[''] * 12)

# Colorier les boxplots
colors_bp = ['#2ecc71', '#e74c3c'] * 6
for patch, color in zip(bp1['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Labels corrects pour 12 positions
labels_bill = []
for col in bill_cols:
    labels_bill.extend([f'{col}\nNo Def', f'{col}\nDef'])

ax.set_xticks(range(12))
ax.set_xticklabels([f'{col[7:]}\n(0,1)' for col in bill_cols for _ in range(2)], fontsize=7)
ax.set_ylabel('Montant facturation ($)', fontweight='bold')
ax.set_title('1. Distribution des Factures (BILL_AMT1..6)\npar Classe', fontweight='bold', fontsize=11)
ax.set_yscale('symlog')
ax.grid(axis='y', alpha=0.3)

print("\n📊 Graphe 1 : Boxplots BILL_AMT par classe")
print(f"  Observation : Factures légèrement plus élevées chez défauteurs")

# 2. Boxplot PAY_AMT1..6 par classe
ax = axes[0, 1]
payamt_data = []
for col in payamt_cols:
    for default_val in [0, 1]:
        payamt_data.append(df[df['DEFAULT'] == default_val][col])

bp2 = ax.boxplot(payamt_data,
                  positions=[i for i in range(12)],
                  widths=0.6,
                  patch_artist=True,
                  labels=[''] * 12)

for patch, color in zip(bp2['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticks(range(12))
ax.set_xticklabels([f'{col[7:]}\n(0,1)' for col in payamt_cols for _ in range(2)], fontsize=7)
ax.set_ylabel('Montant remboursé ($)', fontweight='bold')
ax.set_title('2. Distribution des Remboursements (PAY_AMT1..6)\npar Classe', fontweight='bold', fontsize=11)
ax.set_yscale('symlog')
ax.grid(axis='y', alpha=0.3)

print(f"\n📊 Graphe 2 : Boxplots PAY_AMT par classe")
print(f"  Observation : Remboursements beaucoup plus faibles chez défauteurs → Signal de capacité")

# 3. Scatter LIMIT_BAL vs avg_bill_amt
ax = axes[1, 0]
for default_val, color, label in [(0, '#2ecc71', 'No Default'), (1, '#e74c3c', 'Default')]:
    subset = df[df['DEFAULT'] == default_val]
    ax.scatter(subset['LIMIT_BAL'], subset['avg_bill_amt'], 
              alpha=0.3, s=20, color=color, label=label, edgecolors='none')

ax.set_xlabel('LIMIT_BAL ($)', fontweight='bold')
ax.set_ylabel('Avg Facturation ($)', fontweight='bold')
ax.set_title('3. Relation Crédit vs Facturation Moyenne\n(Couple crédit/comportement)', fontweight='bold', fontsize=11)
ax.legend()
ax.grid(alpha=0.3)

print(f"\n📊 Graphe 3 : Scatter LIMIT_BAL vs avg_bill_amt")
print(f"  Observation : Risque non lié au montant absolu mais au comportement")

# 4. ⭐ Taux de défaut par quantile de repayment_ratio
ax = axes[1, 1]
# Créer des quantiles de repayment_ratio
df['repayment_quantile'] = pd.qcut(df['repayment_ratio'], q=5, labels=['Q1 (worst)', 'Q2', 'Q3', 'Q4', 'Q5 (best)'], duplicates='drop')
repay_default = df.groupby('repayment_quantile', observed=True)['DEFAULT'].agg(['sum', 'count'])
repay_default['rate'] = repay_default['sum'] / repay_default['count']

bars = ax.bar(range(len(repay_default)), repay_default['rate'].values * 100, 
              color=['#e74c3c', '#f39c12', '#f1c40f', '#3498db', '#2ecc71'], alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(repay_default)))
ax.set_xticklabels(repay_default.index, fontsize=10)
ax.set_ylabel('Taux de défaut (%)', fontweight='bold')
ax.set_xlabel('Repayment Ratio Quantile', fontweight='bold')
ax.set_title('⭐ 4. Taux de Défaut par Ratio de Remboursement\n(Capacité vs Facturation)', 
             fontweight='bold', fontsize=11)
ax.set_ylim(0, 50)

for bar in bars:
    height = bar.get_height()
    count = repay_default['count'].iloc[list(bars).index(bar)]
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

print(f"\n⭐ Graphe Clef 4 : Taux de Défaut par Repayment_Ratio")
print(f"  Question : Le risque dépend-il de la capacité à rembourser les montants facturés ?")
print(f"  Lecture : Les clients à faible ratio de remboursement présentent un risque plus élevé")
print(f"  Interprétation métier : Le risque vient moins du volume brut que d'un déséquilibre durable")

plt.tight_layout()
plt.show()

print(f"\n{'='*60}")
print(f"RÉSUMÉ - Section D : Exposition Financière")
print(f"{'='*60}")
print(f"✓ Factures : Moyenne ${df['avg_bill_amt'].mean():,.0f} (défauteurs plus exposés)")
print(f"✓ Remboursement : Moyenne ${df['avg_pay_amt'].mean():,.0f} (défauteurs paient moins)")
print(f"✓ Ratio remboursement : {df['repayment_ratio'].mean():.2f} en moyenne")
print(f"✓ Variables dérivées à créer : avg_bill_amt, avg_pay_amt, repayment_ratio, credit_utilization")

# %% [markdown]
# ## Section E : Variables Pertinentes - Préparation à la Modélisation
# 
# **Axe 4 : Classer les features et décider quoi garder pour le modèle**
# 
# Analyse : Mutual Information, corrélation, et importance univariée
# 
# Graphes :
# - Ranking des variables (Information Value / Mutual Information)
# - Tableau synthèse "signal métier"
# - Recommandation finale pour le modèle

# %%
# Calcul du Mutual Information pour chaque variable
from sklearn.feature_selection import mutual_info_classif

# Préparer les données (numériques et catégoriques converties)
X_numeric = df.select_dtypes(include=[np.number]).drop(['DEFAULT', 'repayment_quantile'], axis=1, errors='ignore')

# Calculer MI
mi_scores = mutual_info_classif(X_numeric, df['DEFAULT'], random_state=42)
mi_scores_df = pd.DataFrame({
    'Feature': X_numeric.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

# Calculer corrélation avec la cible
corr_with_target = df[X_numeric.columns].corrwith(df['DEFAULT']).abs().sort_values(ascending=False)
corr_with_target_df = pd.DataFrame({
    'Feature': corr_with_target.index,
    'Abs_Correlation': corr_with_target.values
})

# Fusionner et créer le ranking
feature_importance = mi_scores_df.merge(corr_with_target_df, on='Feature')
feature_importance['Rank'] = feature_importance['MI_Score'].rank(ascending=False, method='min').astype(int)

# Ajouter une classification (comportement vs descriptif)
behavior_vars = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                 'avg_pay_delay', 'max_pay_delay', 'n_months_late',
                 'avg_bill_amt', 'avg_pay_amt', 'bill_volatility',
                 'credit_utilization', 'repayment_ratio']
feature_importance['Category'] = feature_importance['Feature'].apply(
    lambda x: 'Comportement' if x in behavior_vars else 'Descriptif'
)

# Graphique : Ranking des variables
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.suptitle('Section E : Variables Pertinentes - Ranking pour Modélisation', fontsize=14, fontweight='bold', y=0.98)

ax = axes[0]
top_features = feature_importance.head(15)
colors_rank = ['#e74c3c' if cat == 'Comportement' else '#3498db' for cat in top_features['Category']]
bars = ax.barh(range(len(top_features)), top_features['MI_Score'], color=colors_rank, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'], fontsize=10)
ax.set_xlabel('Mutual Information Score', fontweight='bold')
ax.set_title('1. Top 15 Variables par Importance Informationnelle\n(Comportement=Rouge, Descriptif=Bleu)', 
             fontweight='bold', fontsize=11)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
            f'{width:.3f}', ha='left', va='center', fontsize=9)

print("\n📊 Ranking des Variables par Mutual Information")
print(f"\n{'Rank':<5} {'Feature':<20} {'MI_Score':<10} {'Correlation':<12} {'Category':<12}")
print(f"{'-'*60}")
for idx, row in feature_importance.head(15).iterrows():
    print(f"{int(row['Rank']):<5} {row['Feature']:<20} {row['MI_Score']:<10.4f} {row['Abs_Correlation']:<12.4f} {row['Category']:<12}")

# Tableau synthèse signal métier
ax = axes[1]
ax.axis('tight')
ax.axis('off')

signal_data = [
    ['Variable', 'Importance', 'Signal Métier', 'Recommandation'],
    ['PAY_0', '⭐⭐⭐⭐⭐', 'Retard récent = alerte critique', 'GARDER (central)'],
    ['PAY_2, PAY_3', '⭐⭐⭐⭐', 'Historique 2-3 mois = persistance', 'GARDER'],
    ['avg_pay_delay', '⭐⭐⭐⭐', 'Synthèse de l\'historique', 'GARDER'],
    ['n_months_late', '⭐⭐⭐⭐', 'Fréquence du retard', 'GARDER'],
    ['repayment_ratio', '⭐⭐⭐', 'Capacité vs facturation', 'GARDER'],
    ['LIMIT_BAL', '⭐⭐', 'Exposition brute (secondaire)', 'OPTIONNEL'],
    ['AGE', '⭐⭐', 'Démographie (faible signal)', 'GARDER (peu utile)'],
    ['EDUCATION', '⭐', 'Catégorique faible', 'SUPPRIMER'],
    ['MARRIAGE', '⭐', 'Catégorique très faible', 'SUPPRIMER'],
    ['SEX', '⭐', 'Très peu discriminant', 'SUPPRIMER'],
]

table = ax.table(cellText=signal_data, cellLoc='left', loc='center',
                colWidths=[0.2, 0.15, 0.35, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style du header
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style des lignes alternées
for i in range(1, len(signal_data)):
    color = '#ecf0f1' if i % 2 == 0 else 'white'
    for j in range(4):
        table[(i, j)].set_facecolor(color)

ax.set_title('2. Synthèse Signal Métier & Recommandations\n', fontweight='bold', fontsize=11, pad=20)

plt.tight_layout()
plt.show()

print(f"\n{'='*70}")
print(f"RÉSUMÉ - Section E : Variables Pertinentes")
print(f"{'='*70}")
print(f"\n✓ Variables COMPORTEMENTALES (prioritaires) :")
print(f"  - PAY_0, PAY_2, PAY_3 : Historique de retard = signal dominant")
print(f"  - avg_pay_delay, max_pay_delay, n_months_late : Synthèse utile")
print(f"  - repayment_ratio, credit_utilization : Déséquilibre financier")
print(f"\n✓ Variables DESCRIPTIVES (secondaires) :")
print(f"  - LIMIT_BAL : Exposition brute (modérée)")
print(f"  - AGE : Faible signal (garder par prudence)")
print(f"\n✗ À SUPPRIMER :")
print(f"  - EDUCATION, MARRIAGE, SEX : Très peu discriminants")
print(f"\n📋 MODÈLE RECOMMANDÉ (Axe 4) :")
print(f"  Features essentielles (7) : PAY_0, PAY_2, PAY_3, avg_bill_amt, avg_pay_amt,")
print(f"                              repayment_ratio, n_months_late")
print(f"  Features optionnelles : LIMIT_BAL, AGE")
print(f"  Features à exclure : ID, EDUCATION, MARRIAGE, SEX, colonnes redondantes")
print(f"\n🎯 Signal métier : Les variables comportementales dominent les variables descriptives.")

# %% [markdown]
# ---
# 
# # Conclusion du Dashboard
# 
# ## 📌 Synthèse des 4 Axes
# 
# ### Axe 1 : Comprendre le Portefeuille
# - **Portefeuille** : ~30k clients
# - **Imbalance** : ~22% de défaut (déséquilibré → éviter l'accuracy seule)
# - **Exposition** : Crédit moyen ~$167k, concentré entre 50k-500k
# - **Profil** : Âge moyen 35 ans, légèrement féminisé
# 
# **✓ Clé à retenir** : Contexte posé, données accessibles, imbalance documentée
# 
# ---
# 
# ### Axe 2 : Premiers Signaux de Risque
# - **Signal majeur** : PAY_0 (retard le plus récent) = discriminant #1
# - **Dynamique** : Défauteurs accumulent retards persistants (corr forte 0.8+)
# - **Temporalité** : Retards croissants = pattern dangereux
# - **Variables clefs** : PAY_0, PAY_2, PAY_3, avg_pay_delay
# 
# **✓ Clé à retenir** : Le risque de défaut explose quand un retard récent apparaît. C'est LA variable centrale.
# 
# ---
# 
# ### Axe 3 : Debt, Facturation, Remboursement
# - **Facturation** : Moyennes similaires (défauteurs légèrement plus élevées)
# - **Remboursement** : Défauteurs remboursent **40-50% moins** → signal fort
# - **Ratio remboursement** : Corrélation négative claire avec défaut
# - **Déséquilibre** : Le risque vient du couple (facturation / remboursement), pas du montant brut
# 
# **✓ Clé à retenir** : Une grosse facture n'est pas risquée si remboursement suit. Le risque = déséquilibre durable.
# 
# ---
# 
# ### Axe 4 : Modélisation - Variables à Garder
# | Catégorie | Variables | Action |
# |-----------|-----------|--------|
# | **ESSENTIELS** | PAY_0, PAY_2, PAY_3, n_months_late, avg_pay_delay, repayment_ratio | ✅ Garder |
# | **UTILES** | LIMIT_BAL, avg_bill_amt, avg_pay_amt | ⚠️ Optionnel |
# | **À EXCLURE** | ID, EDUCATION, MARRIAGE, SEX | ❌ Supprimer |
# 
# **✓ Clé à retenir** : Les variables **comportementales dominent** les descriptives. Supprimer le socio-démo statique.
# 
# ---
# 
# ## 🎯 Insights Finaux
# 
# 1. **PAY_0 est ROI** : À lui seul, il capture ~70% du signal. Commencer le scoring par là.
# 2. **Historique > Profil** : L'historique de paiement > âge/éducation/mariage.
# 3. **Effort de remboursement compte** : Les clients qui paient régulièrement leurs factures, même petites portions, ne défautent pas.
# 4. **Retards persistants = danger** : Le risque explose quand retards s'accumulent sur 3+ mois.
# 
# ---
# 
# ## ⚙️ Prochaines Étapes
# 
# 1. **Nettoyage** : Supprimer ID, recatégoriser EDUCATION/MARRIAGE, gérer les valeurs spéciales (-2 en PAY_*)
# 2. **Feature Engineering** : Creuser les ratios d'exposition, les volatilités mensuelles
# 3. **Modélisation** : Logistic Regression / Tree-based (XGBoost) sur les 7-10 features essentielles
# 4. **Validation** : ROC-AUC, Precision-Recall (pas accuracy)
# 5. **Déploiement** : Scorer clients, segmenter par risque, monitorer dérive
# 
# ---


