"""
MINI-PROJET : PRÉVISION DE FAILLITE D'ENTREPRISE (VERSION SIMPLIFIÉE)
======================================================================
Dataset simplifié avec seulement 5 features essentielles pour KNN et SVM
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🏢 MINI-PROJET : PRÉVISION DE FAILLITE (VERSION SIMPLIFIÉE)")
print("=" * 80)

# ============================================================
# 1. CHARGEMENT
# ============================================================
print("\n📂 Chargement des datasets...")

df_adherents = pd.read_csv('prepareAdherents/dataset_encoded_adherents.csv')
df_salaries = pd.read_csv('dataset_salaries_enriched.csv')

print(f"✓ Adherents : {df_adherents.shape[0]:,} entreprises")
print(f"✓ Salariés  : {df_salaries.shape[0]:,} lignes")

# Agréger par entreprise
agg = df_salaries.groupby('affiliateNumber').agg({
    'nb_salaries_declares': 'first',
    'ratio_salaries_immatriculations': 'first',
    'entreprise_peu_salaries': 'first',
    'entreprise_ancienne': 'first',
    'compte_risque': 'first',
    'bank_accountDefaultState': 'first'
}).reset_index()

# Merge
df = df_adherents.merge(agg, on='affiliateNumber', how='inner', suffixes=('_adh', '_sal'))
print(f"✓ Merge : {df.shape[0]:,} entreprises")

# Utiliser la target du dataset salaires (plus fiable)
if 'bank_accountDefaultState_sal' in df.columns:
    df['bank_accountDefaultState'] = df['bank_accountDefaultState_sal']
elif 'bank_accountDefaultState' not in df.columns:
    df['bank_accountDefaultState'] = df['bank_accountDefaultState_adh']

# ============================================================
# 2. TARGET
# ============================================================
print("\n🎯 Variable cible : bank_accountDefaultState")
target_dist = df['bank_accountDefaultState'].value_counts()
for val, count in target_dist.items():
    label = "SAINE" if val == 0 else "FAILLITE"
    pct = count / len(df) * 100
    print(f"  {label:10s} : {count:8,} ({pct:5.2f}%)")

# ============================================================
# 3. FEATURES SIMPLIFIÉES (5 features essentielles)
# ============================================================
print("\n" + "=" * 80)
print("📊 CRÉATION DE 5 FEATURES ESSENTIELLES")
print("=" * 80)

# 1. FIRM AGE (Ancienneté)
print("\n1. Firm Age (années)")
df['firm_age_years'] = df['anciennete_affiliation_months'] / 12
print(f"   ✓ Moyenne : {df['firm_age_years'].mean():.2f} ans")

# 2. FIRM SIZE (Taille - log pour normalisation)
print("\n2. Firm Size (log employés)")
df['firm_size_log'] = np.log1p(df['nb_salaries_declares'])
print(f"   ✓ Moyenne : {df['firm_size_log'].mean():.2f}")

# 3. DEBT RISK (Risque financier composite)
print("\n3. Debt Risk Score")
# Gérer les suffixes potentiels
bank_state_col = 'bank_accountState_adh' if 'bank_accountState_adh' in df.columns else 'bank_accountState'
compte_risque_col = 'compte_risque' if 'compte_risque' in df.columns else 'compte_risque_sal'

df['debt_risk_score'] = (
    df[bank_state_col] * 0.4 +       # État compte bancaire
    df[compte_risque_col] * 0.6       # Compte à risque
)
print(f"   ✓ Moyenne : {df['debt_risk_score'].mean():.4f}")

# 4. STABILITY INDEX (Stabilité entreprise)
print("\n4. Stability Index")
df['stability_index'] = (
    (df['anciennete_affiliation_months'] / 200) * 0.5 +  # Ancienneté
    df['ratio_salaries_immatriculations'] * 0.3 +         # Stabilité RH
    (1 - df['entreprise_peu_salaries']) * 0.2             # Taille suffisante
)
np.random.seed(42)
noise = np.random.normal(0, 0.05, len(df))
df['stability_index'] = np.clip(df['stability_index'] + noise, 0, 1)
print(f"   ✓ Moyenne : {df['stability_index'].mean():.4f}")

# 5. REGIONAL RISK (Risque régional)
print("\n5. Regional Risk")
region_default = df.groupby('directionRegionale')['bank_accountDefaultState'].mean()
df['regional_risk'] = df['directionRegionale'].map(region_default)
print(f"   ✓ Moyenne : {df['regional_risk'].mean():.4f}")

# ============================================================
# 4. SÉLECTION FINALE (7 colonnes : 5 features + ID + target)
# ============================================================
print("\n" + "=" * 80)
print("🎯 DATASET FINAL SIMPLIFIÉ")
print("=" * 80)

features_final = [
    'affiliateNumber',        # ID
    'firm_age_years',         # 1. Âge entreprise
    'firm_size_log',          # 2. Taille (log)
    'debt_risk_score',        # 3. Risque dette
    'stability_index',        # 4. Stabilité
    'regional_risk',          # 5. Risque régional
    'bank_accountDefaultState'  # TARGET
]

df_final = df[features_final].copy()

print(f"\n✓ Dataset : {df_final.shape[0]:,} entreprises x {df_final.shape[1]} colonnes")
print(f"\nFeatures sélectionnées (5) :")
for i, col in enumerate(features_final[1:-1], 1):
    print(f"  {i}. {col}")

# ============================================================
# 5. NETTOYAGE
# ============================================================
print("\n🧹 Nettoyage...")

# Gérer NaN
if df_final.isnull().sum().sum() > 0:
    for col in df_final.columns:
        if df_final[col].isnull().sum() > 0:
            df_final[col].fillna(df_final[col].median(), inplace=True)
    print(f"✓ NaN remplis")

# Supprimer doublons
before = len(df_final)
df_final = df_final.drop_duplicates()
print(f"✓ {before - len(df_final):,} doublons supprimés")

print(f"\n📊 Dataset final : {len(df_final):,} lignes")

# ============================================================
# 6. STATISTIQUES PAR TARGET
# ============================================================
print("\n" + "=" * 80)
print("📊 ANALYSE PAR STATUT (SAINE vs FAILLITE)")
print("=" * 80)

stats_cols = ['firm_age_years', 'firm_size_log', 'debt_risk_score', 
              'stability_index', 'regional_risk']

print("\nMoyennes par statut :")
comparison = df_final.groupby('bank_accountDefaultState')[stats_cols].mean()
comparison.index = ['SAINE (0)', 'FAILLITE (1)']
print(comparison.to_string())

# Différences
print("\n📈 Différences SAINE vs FAILLITE :")
saine = comparison.loc['SAINE (0)']
faillite = comparison.loc['FAILLITE (1)']
for col in stats_cols:
    diff_pct = ((faillite[col] - saine[col]) / saine[col] * 100) if saine[col] != 0 else 0
    symbol = "⬆️" if diff_pct > 0 else "⬇️"
    print(f"  {col:25s} : {symbol} {diff_pct:+6.1f}%")

# ============================================================
# 7. SAUVEGARDE
# ============================================================
print("\n" + "=" * 80)
print("💾 SAUVEGARDE")
print("=" * 80)

output_file = 'dataset_bankruptcy_prediction.csv'
df_final.to_csv(output_file, index=False)
print(f"\n✓ {output_file}")
print(f"  Lignes  : {len(df_final):,}")
print(f"  Colonnes: {len(features_final)}")

# Liste features
features_info = pd.DataFrame({
    'Feature': features_final,
    'Description': [
        'Identifiant entreprise',
        'Âge entreprise (années)',
        'Taille entreprise (log employés)',
        'Score risque dette/compte',
        'Indice stabilité entreprise',
        'Risque régional (taux défaut)',
        'Target: 0=Saine, 1=Faillite'
    ],
    'Type': [
        'ID',
        'Numérique',
        'Numérique',
        'Numérique',
        'Numérique',
        'Numérique',
        'Binaire'
    ]
})

features_info.to_csv('features_list_bankruptcy.csv', index=False)
print(f"✓ features_list_bankruptcy.csv")

# ============================================================
# 8. RÉSUMÉ
# ============================================================
print("\n" + "=" * 80)
print("✅ MINI-PROJET TERMINÉ")
print("=" * 80)

print(f"\n📊 Dataset simplifié :")
print(f"  • Entreprises : {len(df_final):,}")
print(f"  • Features    : 5 (simples et efficaces)")
print(f"  • Target      : bank_accountDefaultState")

print(f"\n🎯 Distribution :")
for val, count in df_final['bank_accountDefaultState'].value_counts().sort_index().items():
    label = "SAINE" if val == 0 else "FAILLITE"
    pct = count / len(df_final) * 100
    print(f"  {label:10s} : {count:8,} ({pct:5.2f}%)")

print(f"\n💡 Features adaptées pour KNN & SVM :")
print(f"  1. firm_age_years      - Ancienneté entreprise")
print(f"  2. firm_size_log       - Taille (normalisée)")
print(f"  3. debt_risk_score     - Risque financier")
print(f"  4. stability_index     - Stabilité globale")
print(f"  5. regional_risk       - Contexte régional")

print(f"\n🚀 Prochaine étape : Créer modèles KNN & SVM from scratch")

print("\n" + "=" * 80)
