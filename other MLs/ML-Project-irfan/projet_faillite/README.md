# 🏢 PROJET : PRÉVISION DE FAILLITE D'ENTREPRISE

## 📋 Description du Projet

Mini-projet de **Machine Learning** pour prédire la **faillite d'entreprises** en utilisant des algorithmes **KNN** et **SVM** implémentés **from scratch**.

---

## 🎯 Objectif

Prédire si une entreprise présente un **risque de faillite** (défaut bancaire) en analysant 5 features simples et efficaces.

---

## 📊 Dataset

- **Source** : `dataset_bankruptcy_prediction.csv`
- **Entreprises** : 161,254
- **Features** : 5 (numériques)
- **Target** : `bank_accountDefaultState` (0=Saine, 1=Faillite)
- **Distribution** :
  - ✅ Saines : 149,466 (92.69%)
  - ⚠️ Faillites : 11,788 (7.31%)

---

## 📁 Structure du Projet

```
projet_faillite/
│
├── create_features_bankruptcy_prediction.py  # Script création dataset
├── dataset_bankruptcy_prediction.csv         # Dataset ML (161,254 × 7)
├── features_list_bankruptcy.csv              # Documentation features
│
├── model_bankruptcy_prediction.ipynb         # Notebook training KNN & SVM
│
├── interface_bankruptcy_prediction.py        # Interface Streamlit (5 pages)
│
├── knn_scratch_bankruptcy.pkl                # Modèle KNN entraîné
├── svm_scratch_bankruptcy.pkl                # Modèle SVM entraîné
├── bankruptcy_features.pkl                   # Liste des features
├── bankruptcy_train_mean.pkl                 # Moyennes normalisation
├── bankruptcy_train_std.pkl                  # Écarts-types normalisation
├── best_k_bankruptcy.pkl                     # Meilleur k pour KNN
│
└── README.md                                 # Ce fichier
```

---

## 🔧 Les 5 Features

| # | Feature | Description | Rôle | Impact |
|---|---------|-------------|------|--------|
| 1 | **firm_age_years** | Âge entreprise (années) | Maturité vs obsolescence | ⭐⭐⭐ |
| 2 | **firm_size_log** | Taille (log employés) | Capacité vs complexité | ⭐⭐ |
| 3 | **debt_risk_score** | Score risque dette (0-1) | Santé financière directe | ⭐⭐⭐⭐⭐ |
| 4 | **stability_index** | Indice stabilité (0-1) | Santé organisationnelle | ⭐⭐⭐⭐ |
| 5 | **regional_risk** | Risque régional (%) | Contexte économique | ⭐⭐ |

### 📚 Explication Détaillée des Features

#### 1️⃣ **firm_age_years** - Âge de l'Entreprise
- **Définition** : Ancienneté depuis la création (en années)
- **Calcul** : `anciennete_affiliation_months / 12`
- **Insight** : Entreprises en faillite sont +32% plus anciennes (paradoxe du déclin)

#### 2️⃣ **firm_size_log** - Taille (Logarithmique)
- **Définition** : Taille normalisée par logarithme
- **Calcul** : `log(nombre_employés + 1)`
- **Pourquoi log ?** : Évite que grandes entreprises écrasent petites dans le modèle
- **Insight** : Grandes entreprises +29% plus à risque (bureaucratie)

#### 3️⃣ **debt_risk_score** - Risque d'Endettement
- **Définition** : Score composite risque financier (0 à 1)
- **Calcul** : `bank_accountState*0.4 + compte_risque*0.6`
- **Insight** : **Feature la plus importante** ! Entreprises faillite ont +43% de score
- **Interprétation** :
  - 0.0-0.3 → ✅ Faible risque
  - 0.3-0.6 → ⚠️ Risque modéré
  - 0.6-1.0 → 🚨 Risque élevé

#### 4️⃣ **stability_index** - Indice de Stabilité
- **Définition** : Score composite stabilité organisationnelle (0 à 1)
- **Calcul** : Composite de ancienneté/200*0.5 + ratio_salaries*0.3 + taille*0.2 + noise
- **Insight** : Faillites ont +23% stabilité (paradoxe : stabilité = stagnation)
- **Interprétation** :
  - 0.0-0.3 → 🚨 Instable
  - 0.3-0.7 → ⚠️ Moyenne
  - 0.7-1.0 → ✅ Stable (attention si combiné avec autres signaux négatifs)

#### 5️⃣ **regional_risk** - Risque Régional
- **Définition** : Taux de défaut moyen dans la région
- **Calcul** : `nb_faillites_region / nb_total_region`
- **Insight** : Contexte économique local influence la survie
- **Interprétation** :
  - 0.00-0.05 → ✅ Région prospère
  - 0.05-0.10 → ⚠️ Moyenne
  - 0.10-0.20 → 🚨 Région en difficulté

---

## 🤖 Algorithmes Implémentés FROM SCRATCH

### 🔵 KNN (K-Nearest Neighbors)
- **Méthode** : Distance euclidienne + vote majoritaire
- **K optimal** : Déterminé par cross-validation
- **Avantages** : Simple, interprétable, rapide
- **Dataset entraînement** : 20,000 échantillons

### 🟢 SVM (Support Vector Machine)
- **Méthode** : Gradient descent (1000 itérations)
- **Kernel** : Linéaire
- **Avantages** : Hyperplan optimal, robuste
- **Dataset entraînement** : 10,000 échantillons

---

## 🚀 Utilisation

### 1️⃣ Créer le Dataset
```bash
python create_features_bankruptcy_prediction.py
```
**Output** :
- `dataset_bankruptcy_prediction.csv` (161,254 entreprises × 7 colonnes)
- `features_list_bankruptcy.csv` (documentation)

### 2️⃣ Entraîner les Modèles
Ouvrir et exécuter le notebook :
```bash
jupyter notebook model_bankruptcy_prediction.ipynb
```
Ou dans VS Code : Run All Cells

**Output** (6 fichiers .pkl) :
- `knn_scratch_bankruptcy.pkl`
- `svm_scratch_bankruptcy.pkl`
- `bankruptcy_features.pkl`
- `bankruptcy_train_mean.pkl`
- `bankruptcy_train_std.pkl`
- `best_k_bankruptcy.pkl`

### 3️⃣ Lancer l'Interface
```bash
streamlit run interface_bankruptcy_prediction.py
```

---

## 💻 Interface Streamlit (5 Pages)

### 🏠 Page 1 : Accueil
- Vue d'ensemble du projet
- Métriques clés (2 algorithmes, 5 features)
- Description des features
- Statistiques dataset

### 📊 Page 2 : Exploration Données
- Aperçu du dataset (premières lignes)
- Statistiques descriptives
- Distribution de la target (graphiques)
- Comparaison SAINE vs FAILLITE

### 🔵 Page 3 : Prédiction KNN
- Formulaire de saisie (5 features)
- Bouton prédiction avec KNN
- Résultat visuel (✅ Saine / ⚠️ Faillite)
- Affichage des features utilisées

### 🟢 Page 4 : Prédiction SVM
- Formulaire de saisie (5 features)
- Bouton prédiction avec SVM
- Résultat visuel (✅ Saine / ⚠️ Faillite)
- Affichage des features utilisées

### 📈 Page 5 : Comparaison Modèles
- Tableau comparatif KNN vs SVM
- Points forts du mini-projet
- Importance des features
- Recommandations

---

## 📈 Résultats Attendus

### Exemple Entreprise SAINE ✅
```
firm_age_years = 8 ans         → Mature
firm_size_log = 3.5            → ~30 employés (PME)
debt_risk_score = 0.2          → Bon payeur
stability_index = 0.7          → Stable
regional_risk = 0.04           → Région prospère
→ PRÉDICTION : SAINE
```

### Exemple Entreprise FAILLITE 🚨
```
firm_age_years = 22 ans        → Très ancienne (déclin)
firm_size_log = 5.2            → ~180 employés (grande)
debt_risk_score = 0.85         → Défauts fréquents
stability_index = 0.9          → Trop stable (stagnation)
regional_risk = 0.15           → Région en crise
→ PRÉDICTION : FAILLITE
```

---

## 🔍 Insights Clés du Dataset

| Insight | Valeur | Interprétation |
|---------|--------|----------------|
| Grandes entreprises = Plus de risque | +29% taille | Bureaucratie, moins d'agilité |
| Entreprises anciennes = Plus de risque | +32% âge | Déclin après maturité |
| Stabilité élevée ≠ Bon signe | +23% stabilité | Stagnation vs croissance |
| Risque dette = Indicateur #1 | +43% score | Prédicteur le plus fort |

---

## ⚙️ Configuration Requise

### Packages Python
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit pickle-mixin
```

### Environnement
- Python 3.12+
- Jupyter Notebook
- Streamlit 1.30+

---

## 🎓 Points Pédagogiques

### Apprentissage
- ✅ Implémentation FROM SCRATCH de KNN et SVM
- ✅ Feature engineering simple et efficace
- ✅ Normalisation manuelle (StandardScaler)
- ✅ Cross-validation pour optimisation hyperparamètres
- ✅ Métriques de classification (Accuracy, Precision, Recall, F1)

### Mini-Projet Adapté
- **Simplicité** : 5 features claires vs 19 complexes initialement
- **Rapidité** : Sampling (20k/10k/5k) vs dataset complet (161k)
- **Interprétabilité** : Features métier compréhensibles
- **Pédagogie** : From scratch pour comprendre les algorithmes

---

## 📝 Notes Importantes

⚠️ **Avant de lancer l'interface** :
1. Exécuter `create_features_bankruptcy_prediction.py` pour générer le dataset
2. Exécuter le notebook `model_bankruptcy_prediction.ipynb` pour générer les .pkl
3. Vérifier que les 11 fichiers sont présents dans le dossier

💡 **Class Imbalance** : Dataset déséquilibré (92.69% vs 7.31%)
- Approche actuelle : Stratified split
- Amélioration possible : SMOTE, class weights

---

## 🏆 Auteur

**Projet Mini-ML** - Prévision Faillite d'Entreprise  
KNN & SVM FROM SCRATCH | 5 Features | Classification Binaire

---

## 📅 Dernière Mise à Jour

3 Décembre 2025
