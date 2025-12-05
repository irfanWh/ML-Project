import pandas as pd
import numpy as np

# ==============================
# CHARGEMENT DES DONNÉES
# ==============================

print("=== CHARGEMENT DES DATASETS ===\n")

# Charger les adhérents (entreprises)
adherents = pd.read_csv('prepareAdherents/dataset_encoded_adherents.csv')
print(f"✓ Adhérents (entreprises) : {len(adherents)} lignes, {len(adherents.columns)} colonnes")

# Charger les salariés
salaries = pd.read_csv('prepareSalaires/salaries_combined_data.csv')
print(f"✓ Salariés : {len(salaries)} lignes, {len(salaries.columns)} colonnes")

print(f"\nColonnes adhérents : {list(adherents.columns)}")
print(f"Colonnes salariés : {list(salaries.columns)}")


# ==============================
# AGRÉGATION DES SALARIÉS PAR ENTREPRISE
# ==============================

print("\n\n=== AGRÉGATION DES SALARIÉS PAR ENTREPRISE ===\n")

# Agréger les métriques des salariés par affiliateNumber (entreprise)
salaries_agregats = salaries.groupby('affiliateNumber').agg({
    'nombre_jours': ['sum', 'mean', 'max', 'min'],  # Jours travaillés
    'salaire': ['sum', 'mean', 'median', 'max', 'min', 'std'],  # Salaires
    'demandMode': 'mean',  # Proportion de demandMode = 1
    'anciennete_adhesion_months': 'mean'  # Ancienneté moyenne
}).reset_index()

# Renommer les colonnes agrégées
salaries_agregats.columns = [
    'affiliateNumber',
    'total_jours_travailles',
    'moyenne_jours_travailles',
    'max_jours_travailles',
    'min_jours_travailles',
    'masse_salariale_totale',
    'salaire_moyen',
    'salaire_median',
    'salaire_max',
    'salaire_min',
    'salaire_std',
    'taux_demandMode',
    'anciennete_moyenne_salaries'
]

# Nombre de salariés par entreprise
nb_salaries_par_entreprise = salaries.groupby('affiliateNumber').size().reset_index(name='nb_salaries')

# Fusionner les agrégats
salaries_agregats = salaries_agregats.merge(nb_salaries_par_entreprise, on='affiliateNumber', how='left')

print(f"✓ Agrégation effectuée : {len(salaries_agregats)} entreprises analysées")
print(f"\nAperçu des agrégats :")
print(salaries_agregats.head(10))


# ==============================
# ENRICHISSEMENT AVEC FEATURES CALCULÉES
# ==============================

print("\n\n=== ENRICHISSEMENT DES AGRÉGATS ===\n")

# 1. Salaire moyen par salarié effectif (qui travaille)
salaries_actifs = salaries[salaries['nombre_jours'] > 0]
salaires_actifs_agregats = salaries_actifs.groupby('affiliateNumber').agg({
    'salaire': 'mean',
    'nombre_jours': 'count'
}).reset_index()
salaires_actifs_agregats.columns = ['affiliateNumber', 'salaire_moyen_actifs', 'nb_salaries_actifs']

salaries_agregats = salaries_agregats.merge(salaires_actifs_agregats, on='affiliateNumber', how='left')

# 2. Taux de salariés actifs
salaries_agregats['taux_salaries_actifs'] = (salaries_agregats['nb_salaries_actifs'] / salaries_agregats['nb_salaries'] * 100).fillna(0)

# 3. Indicateur de grande entreprise
salaries_agregats['est_grande_entreprise'] = (salaries_agregats['nb_salaries'] >= 50).astype(int)

# 4. Catégorisation de la masse salariale
salaries_agregats['categorie_masse_salariale'] = pd.cut(
    salaries_agregats['masse_salariale_totale'],
    bins=[0, 50000, 200000, 500000, 1000000, float('inf')],
    labels=['Très faible', 'Faible', 'Moyenne', 'Élevée', 'Très élevée']
)

# 5. Coefficient de variation salariale (dispersion)
salaries_agregats['coef_variation_salaire'] = (salaries_agregats['salaire_std'] / salaries_agregats['salaire_moyen']).fillna(0)

print("✓ Features calculées ajoutées")
print(f"\nNouvelles colonnes : {list(salaries_agregats.columns)}")


# ==============================
# CROISEMENT AVEC LES CARACTÉRISTIQUES DES ENTREPRISES
# ==============================

print("\n\n=== CROISEMENT ENTREPRISES ↔ SALARIÉS ===\n")

# Fusionner les agrégats des salariés avec les caractéristiques des entreprises
dataset_complet = adherents.merge(
    salaries_agregats,
    on='affiliateNumber',
    how='left'  # Left join pour garder toutes les entreprises
)

# Remplir les NaN (entreprises sans salariés dans le dataset)
dataset_complet['nb_salaries'] = dataset_complet['nb_salaries'].fillna(0)
dataset_complet['masse_salariale_totale'] = dataset_complet['masse_salariale_totale'].fillna(0)

print(f"✓ Dataset complet créé : {len(dataset_complet)} lignes, {len(dataset_complet.columns)} colonnes")
print(f"\nEntreprises avec salariés : {dataset_complet['nb_salaries'].gt(0).sum()}")
print(f"Entreprises sans salariés : {dataset_complet['nb_salaries'].eq(0).sum()}")


# ==============================
# ANALYSES STATISTIQUES
# ==============================

print("\n\n=== ANALYSES STATISTIQUES ===\n")

# 1. Distribution du nombre de salariés
print("Distribution du nombre de salariés par entreprise :")
print(dataset_complet['nb_salaries'].describe())

# 2. Statistiques par type d'adhérent
print("\n\nStatistiques par type d'adhérent :")
stats_par_type = dataset_complet.groupby('typeAdherent').agg({
    'nb_salaries': ['mean', 'median', 'sum'],
    'masse_salariale_totale': ['mean', 'sum'],
    'salaire_moyen': 'mean'
})
print(stats_par_type)

# 3. Statistiques par région
print("\n\nTop 5 régions par nombre de salariés :")
stats_par_region = dataset_complet.groupby('directionRegionale').agg({
    'nb_salaries': 'sum',
    'masse_salariale_totale': 'sum',
    'affiliateNumber': 'count'
}).sort_values('nb_salaries', ascending=False).head(5)
stats_par_region.columns = ['Total salariés', 'Masse salariale totale', 'Nombre entreprises']
print(stats_par_region)

# 4. Corrélation entre ancienneté et nombre de salariés
correlation_anciennete = dataset_complet[['anciennete_adhesion_months', 'nb_salaries', 'masse_salariale_totale']].corr()
print("\n\nCorrélations :")
print(correlation_anciennete)


# ==============================
# ENRICHISSEMENT POUR PROJET ML (DÉTECTION ANOMALIES)
# ==============================

print("\n\n=== ENRICHISSEMENT POUR DÉTECTION D'ANOMALIES ===\n")

# Filtrer seulement les entreprises avec des salariés
dataset_ml = dataset_complet[dataset_complet['nb_salaries'] > 0].copy()

print(f"Entreprises avec salariés : {len(dataset_ml)}")

# ========== FEATURES ESSENTIELLES SEULEMENT ==========

# 1. Salaire par jour (normalisation importante)
dataset_ml['salaire_par_jour'] = (dataset_ml['masse_salariale_totale'] / dataset_ml['total_jours_travailles'].replace(0, 1)).fillna(0)

# 2. Jours travaillés par salarié (productivité)
dataset_ml['jours_par_salarie'] = (dataset_ml['total_jours_travailles'] / dataset_ml['nb_salaries']).fillna(0)

# 3. Écart au salaire moyen régional (contexte géographique)
salaire_moyen_region = dataset_ml.groupby('directionRegionale')['salaire_moyen'].transform('mean')
dataset_ml['ecart_salaire_region'] = dataset_ml['salaire_moyen'] - salaire_moyen_region

# 4. Indicateur de compte à risque
dataset_ml['compte_a_risque'] = (
    (dataset_ml['bank_accountDefaultState'] == 1) | 
    (dataset_ml['bank_accountState'] == 0)
).astype(int)

print(f"✓ 4 features essentielles créées")

nouvelles_features = ['salaire_par_jour', 'jours_par_salarie', 'ecart_salaire_region', 'compte_a_risque']
print(f"\nFeatures ajoutées : {nouvelles_features}")

# ========== NETTOYAGE FINAL ==========

# Remplacer les valeurs infinies par NaN
dataset_ml = dataset_ml.replace([np.inf, -np.inf], np.nan)

# Supprimer les lignes avec trop de NaN (optionnel)
dataset_ml = dataset_ml.dropna(subset=['nb_salaries', 'salaire_moyen', 'masse_salariale_totale'])

# Remplir les NaN restants par 0 pour les features calculées
features_numeriques = dataset_ml.select_dtypes(include=[np.number]).columns
dataset_ml[features_numeriques] = dataset_ml[features_numeriques].fillna(0)

print(f"\n✓ Nettoyage effectué")
print(f"Dataset final pour ML : {len(dataset_ml)} lignes")


# ==============================
# SAUVEGARDE DES RÉSULTATS
# ==============================

print("\n\n=== SAUVEGARDE DES RÉSULTATS ===\n")

# SÉLECTION DES COLONNES ESSENTIELLES POUR LE ML
colonnes_ml = [
    # ID de l'entreprise (IMPORTANT : ne pas utiliser pour training, uniquement pour identifier)
    'affiliateNumber',
    
    # Informations de base
    'directionRegionale',
    'anciennete_adhesion_months',
    
    # Indicateurs salariés (agrégats)
    'nb_salaries',
    'masse_salariale_totale',
    'salaire_moyen',
    'salaire_median',
    'salaire_max',
    'salaire_min',
    'salaire_std',
    
    # Activité
    'total_jours_travailles',
    'taux_salaries_actifs',
    
    # Features engineered (les 4 essentielles)
    'salaire_par_jour',
    'jours_par_salarie',
    'ecart_salaire_region',
    'compte_a_risque'
]

# 1. Dataset ML SIMPLIFIÉ (seulement colonnes importantes)
dataset_ml_simple = dataset_ml[colonnes_ml].copy()
dataset_ml_simple.to_csv('dataset_ml_anomalies.csv', index=False)
print(f"✓ Dataset ML simplifié sauvegardé : dataset_ml_anomalies.csv ({len(dataset_ml_simple)} lignes, {len(colonnes_ml)} colonnes)")
print(f"  ⚠️  affiliateNumber inclus pour identification (ne pas utiliser pour training)")

# 2. Dataset complet original
dataset_complet.to_csv('dataset_entreprises_salaries_complet.csv', index=False)
print(f"✓ Dataset complet sauvegardé : dataset_entreprises_salaries_complet.csv")

# 3. Agrégats salariés uniquement
salaries_agregats.to_csv('agregats_salaries_par_entreprise.csv', index=False)
print(f"✓ Agrégats salariés sauvegardés : agregats_salaries_par_entreprise.csv")

# 4. Statistiques récapitulatives
stats_recap = pd.DataFrame({
    'Métrique': [
        'Total entreprises',
        'Entreprises avec salariés',
        'Total salariés',
        'Masse salariale totale',
        'Salaire moyen global',
        'Entreprises à risque'
    ],
    'Valeur': [
        len(dataset_complet),
        len(dataset_ml),
        dataset_ml['nb_salaries'].sum(),
        f"{dataset_ml['masse_salariale_totale'].sum():,.2f}",
        f"{dataset_ml['salaire_moyen'].mean():,.2f}",
        dataset_ml['compte_a_risque'].sum()
    ]
})
stats_recap.to_csv('statistiques_recap.csv', index=False)
print(f"✓ Statistiques sauvegardées : statistiques_recap.csv")

print("\n\n=== TERMINÉ ===")
print("\nFichiers générés :")
print("  1. dataset_ml_anomalies.csv ⭐ ({} COLONNES SEULEMENT)".format(len(colonnes_ml)))
print("  2. dataset_entreprises_salaries_complet.csv")
print("  3. agregats_salaries_par_entreprise.csv")
print("  4. statistiques_recap.csv")
print(f"\n✓ Prêt pour le Projet 5 : Détection d'anomalies salariales !")
