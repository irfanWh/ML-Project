"""
ENRICHISSEMENT DATASET SALARIES avec AGREGATIONS ADHERENTS
============================================================
Ce script merge les datasets salariés et adhérents pour créer un dataset enrichi
avec des features agrégées par entreprise (affiliateNumber)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔄 ENRICHISSEMENT DATASET SALARIES AVEC INFORMATIONS ADHERENTS")
print("=" * 80)

# ============================================================
# 1. CHARGEMENT DES DATASETS
# ============================================================
print("\n📂 Chargement des datasets...")

# Dataset Salariés (principal)
df_salaries = pd.read_csv('prepareSalaires/dataset_encoded_salaries.csv')
print(f"✓ Dataset Salariés chargé : {df_salaries.shape[0]:,} lignes x {df_salaries.shape[1]} colonnes")
print(f"  Colonnes : {list(df_salaries.columns)}")

# Dataset Adhérents (entreprises)
df_adherents = pd.read_csv('prepareAdherents/dataset_encoded_adherents.csv')
print(f"✓ Dataset Adhérents chargé : {df_adherents.shape[0]:,} lignes x {df_adherents.shape[1]} colonnes")
print(f"  Colonnes : {list(df_adherents.columns)}")

# ============================================================
# 2. AGRÉGATIONS PAR ENTREPRISE (affiliateNumber)
# ============================================================
print("\n📊 Calcul des agrégations par entreprise...")

# Compter le nombre de salariés par entreprise
agg_salaries_count = df_salaries.groupby('affiliateNumber').size().reset_index(name='nb_salaries_declares')
print(f"✓ Nombre de salariés par entreprise calculé")

# Calcul du nombre d'immatriculations uniques par entreprise
agg_immatriculations = df_salaries.groupby('affiliateNumber')['immatriculationNumber'].nunique().reset_index(name='nb_immatriculations_uniques')
print(f"✓ Nombre d'immatriculations uniques calculé")

# Mode de demande dominant par entreprise
agg_demandMode = df_salaries.groupby('affiliateNumber')['demandMode'].agg(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
).reset_index(name='demandMode_dominant')
print(f"✓ Mode de demande dominant calculé")

# Agrégation de l'ancienneté moyenne des salariés par entreprise
agg_anciennete = df_salaries.groupby('affiliateNumber')['anciennete_adhesion_months'].agg([
    ('anciennete_adhesion_mean', 'mean'),
    ('anciennete_adhesion_min', 'min'),
    ('anciennete_adhesion_max', 'max'),
    ('anciennete_adhesion_std', 'std')
]).reset_index()
print(f"✓ Statistiques d'ancienneté des salariés calculées")

# ============================================================
# 3. MERGE DES AGRÉGATIONS
# ============================================================
print("\n🔗 Fusion des agrégations...")

# Fusionner toutes les agrégations
df_agg = agg_salaries_count.copy()
df_agg = df_agg.merge(agg_immatriculations, on='affiliateNumber', how='left')
df_agg = df_agg.merge(agg_demandMode, on='affiliateNumber', how='left')
df_agg = df_agg.merge(agg_anciennete, on='affiliateNumber', how='left')

print(f"✓ Agrégations fusionnées : {df_agg.shape}")

# ============================================================
# 4. MERGE AVEC ADHERENTS
# ============================================================
print("\n🔗 Merge avec le dataset Adhérents...")

# Fusionner les agrégations avec les adhérents
df_adherents_enriched = df_adherents.merge(df_agg, on='affiliateNumber', how='left')

print(f"✓ Dataset Adhérents enrichi : {df_adherents_enriched.shape}")
print(f"  Nouvelles colonnes : {[c for c in df_adherents_enriched.columns if c not in df_adherents.columns]}")

# Remplir les NaN pour les entreprises sans salariés déclarés
df_adherents_enriched['nb_salaries_declares'] = df_adherents_enriched['nb_salaries_declares'].fillna(0).astype(int)
df_adherents_enriched['nb_immatriculations_uniques'] = df_adherents_enriched['nb_immatriculations_uniques'].fillna(0).astype(int)
df_adherents_enriched['demandMode_dominant'] = df_adherents_enriched['demandMode_dominant'].fillna(-1).astype(int)

# ============================================================
# 5. ENRICHISSEMENT DES SALARIÉS
# ============================================================
print("\n🔗 Enrichissement du dataset Salariés...")

# Merge salariés avec adhérents enrichis
df_salaries_enriched = df_salaries.merge(
    df_adherents_enriched, 
    on='affiliateNumber', 
    how='left',
    suffixes=('_salarie', '_entreprise')
)

print(f"✓ Dataset Salariés enrichi : {df_salaries_enriched.shape}")
print(f"  Total colonnes : {len(df_salaries_enriched.columns)}")

# ============================================================
# 6. CRÉATION DE FEATURES SUPPLÉMENTAIRES
# ============================================================
print("\n🎯 Création de features supplémentaires...")

# Ratio salariés/immatriculations
df_salaries_enriched['ratio_salaries_immatriculations'] = (
    df_salaries_enriched['nb_salaries_declares'] / 
    (df_salaries_enriched['nb_immatriculations_uniques'] + 1)  # +1 pour éviter division par 0
)

# Écart d'ancienneté (salarie vs entreprise)
if 'anciennete_adhesion_months_salarie' in df_salaries_enriched.columns:
    df_salaries_enriched['ecart_anciennete'] = (
        df_salaries_enriched['anciennete_adhesion_months_entreprise'] - 
        df_salaries_enriched['anciennete_adhesion_months_salarie']
    )

# Indicateur entreprise avec peu de salariés déclarés
df_salaries_enriched['entreprise_peu_salaries'] = (df_salaries_enriched['nb_salaries_declares'] < 5).astype(int)

# Indicateur entreprise nouvelle (< 12 mois)
df_salaries_enriched['entreprise_nouvelle'] = (df_salaries_enriched['anciennete_affiliation_months'] < 12).astype(int)

# Indicateur entreprise ancienne (> 10 ans = 120 mois)
df_salaries_enriched['entreprise_ancienne'] = (df_salaries_enriched['anciennete_affiliation_months'] > 120).astype(int)

# Indicateur compte bancaire à risque
df_salaries_enriched['compte_risque'] = (
    (df_salaries_enriched['bank_accountDefaultState'] == 1) | 
    (df_salaries_enriched['bank_accountState'] == 0)
).astype(int)

print(f"✓ Features créées : ratio_salaries_immatriculations, ecart_anciennete, entreprise_peu_salaries, etc.")

# ============================================================
# 7. STATISTIQUES FINALES
# ============================================================
print("\n" + "=" * 80)
print("📈 STATISTIQUES DU DATASET ENRICHI")
print("=" * 80)

print(f"\n📊 Dataset Salariés Enrichi :")
print(f"  • Total lignes : {len(df_salaries_enriched):,}")
print(f"  • Total colonnes : {len(df_salaries_enriched.columns)}")
print(f"  • Entreprises uniques : {df_salaries_enriched['affiliateNumber'].nunique():,}")

print(f"\n📊 Distribution nb_salaries_declares par entreprise :")
print(df_salaries_enriched['nb_salaries_declares'].describe())

print(f"\n📊 Distribution typeAdherent :")
print(df_salaries_enriched['typeAdherent'].value_counts())

print(f"\n📊 Distribution modaliteTelepaiement :")
print(df_salaries_enriched['modaliteTelepaiement'].value_counts())

print(f"\n📊 Entreprises avec compte à risque :")
print(f"  • Compte risque = 1 : {df_salaries_enriched['compte_risque'].sum():,} salariés")
print(f"  • Compte risque = 0 : {(df_salaries_enriched['compte_risque'] == 0).sum():,} salariés")

print(f"\n📊 Distribution taille entreprise :")
print(f"  • Peu de salariés (< 5) : {df_salaries_enriched['entreprise_peu_salaries'].sum():,} salariés")
print(f"  • Entreprises nouvelles (< 12 mois) : {df_salaries_enriched['entreprise_nouvelle'].sum():,} salariés")
print(f"  • Entreprises anciennes (> 10 ans) : {df_salaries_enriched['entreprise_ancienne'].sum():,} salariés")

# ============================================================
# 8. SAUVEGARDE
# ============================================================
print("\n💾 Sauvegarde des datasets enrichis...")

# Sauvegarder le dataset salariés enrichi
output_salaries = 'dataset_salaries_enriched.csv'
df_salaries_enriched.to_csv(output_salaries, index=False)
print(f"✓ {output_salaries} sauvegardé ({len(df_salaries_enriched):,} lignes)")

# Sauvegarder le dataset adhérents enrichi
output_adherents = 'dataset_adherents_enriched.csv'
df_adherents_enriched.to_csv(output_adherents, index=False)
print(f"✓ {output_adherents} sauvegardé ({len(df_adherents_enriched):,} lignes)")

# Sauvegarder les mappings de colonnes
print("\n📋 Liste des colonnes du dataset enrichi :")
for i, col in enumerate(df_salaries_enriched.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "=" * 80)
print("✅ ENRICHISSEMENT TERMINÉ AVEC SUCCÈS !")
print("=" * 80)
print(f"\n📁 Fichiers créés :")
print(f"  • {output_salaries}")
print(f"  • {output_adherents}")
print(f"\n🎯 Prêt pour les projets ML avec KNN/SVM !")
