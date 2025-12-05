# 🎤 PRÉSENTATION PROJET FAILLITE - 2 MINUTES

## ⏱️ TIMING : 2 minutes (120 secondes)

---

## 📌 SLIDE 1 : INTRODUCTION (20 secondes)

**Bonjour,**

Je vais vous présenter mon mini-projet de **Machine Learning** : 
**"Prévision de Faillite d'Entreprise"**

### Objectif
Prédire si une entreprise risque la **faillite** en analysant **5 indicateurs simples**.

### Approche
- **2 algorithmes** implémentés **from scratch** : KNN et SVM
- **161,254 entreprises** analysées
- **Interface interactive** avec Streamlit

---

## 📊 SLIDE 2 : LE DATASET (25 secondes)

### Source des Données
- Dataset d'entreprises marocaines affiliées
- **161,254 entreprises** avec leurs données financières
- **Distribution** :
  - ✅ 92.69% entreprises saines (149,466)
  - ⚠️ 7.31% en faillite (11,788)

### Les 5 Features Clés
1. **Âge entreprise** - Maturité (années)
2. **Taille** - Nombre employés (log)
3. **Risque dette** ⭐⭐⭐⭐⭐ - Score financier (0-1)
4. **Stabilité** - Indice organisation (0-1)
5. **Risque régional** - Contexte économique (%)

**Insight clé** : Le risque de dette est le prédicteur #1 (+43% pour faillites)

---

## 🤖 SLIDE 3 : LES ALGORITHMES (20 secondes)

### KNN (K-Nearest Neighbors)
- Distance euclidienne + vote majoritaire
- K optimal trouvé par cross-validation
- ✅ Simple et interprétable

### SVM (Support Vector Machine)
- Gradient descent (1000 itérations)
- Hyperplan optimal pour séparation
- ✅ Robuste pour classification binaire

**Point fort** : Implémentés **FROM SCRATCH** sans bibliothèques ML

---

## 💻 DÉMO 1 : NOTEBOOK (25 secondes)

### Structure du Notebook (12 sections)

**[OUVRIR : `model_bankruptcy_prediction.ipynb`]**

1. **Chargement** - 161,254 entreprises
2. **Split stratifié** - 80/20 pour équilibrer les classes
3. **Normalisation** - StandardScaler from scratch
4. **Optimisation K** - Cross-validation sur 6 valeurs
5. **KNN Training** - 20,000 échantillons (rapide)
6. **SVM Training** - 10,000 échantillons avec gradient descent
7. **Évaluations** - Métriques + matrices de confusion
8. **Comparaison** - KNN vs SVM (table + graphique)
9. **Sauvegarde** - 6 fichiers .pkl générés

**Résultat** : Modèles prêts pour prédiction en production !

---

## 🖥️ DÉMO 2 : INTERFACE STREAMLIT (25 secondes)

### 5 Pages Interactives

**[LANCER : `streamlit run interface_bankruptcy_prediction.py`]**

#### Page 1 : Accueil 🏠
- Vue d'ensemble : 2 algos, 5 features, 161K entreprises

#### Page 2 : Exploration 📊
- Statistiques descriptives
- Graphiques distribution (saines vs faillites)
- Comparaison moyennes par feature

#### Page 3 : Prédiction KNN 🔵
**[DÉMO INTERACTIVE]**
```
Exemple 1 - Entreprise SAINE :
- Âge : 8 ans
- Employés : 30
- Risque dette : 0.2 (faible)
- Stabilité : 0.7
- Région : 4%
→ RÉSULTAT : ✅ ENTREPRISE SAINE
```

#### Page 4 : Prédiction SVM 🟢
**[DÉMO INTERACTIVE]**
```
Exemple 2 - Entreprise FAILLITE :
- Âge : 22 ans (très ancienne)
- Employés : 180
- Risque dette : 0.85 (élevé !)
- Stabilité : 0.9 (stagnation)
- Région : 15% (crise)
→ RÉSULTAT : ⚠️ RISQUE DE FAILLITE
```

#### Page 5 : Comparaison 📈
- Tableau KNN vs SVM
- Importance des features

---

## 🎯 SLIDE 4 : RÉSULTATS & INSIGHTS (15 secondes)

### Découvertes Surprenantes

| Insight | Impact |
|---------|--------|
| 🏢 Grandes entreprises = **+29% risque** | Bureaucratie |
| 📅 Entreprises anciennes = **+32% risque** | Déclin après maturité |
| 📊 Stabilité élevée = **+23% risque** | Stagnation vs croissance |
| 💳 Risque dette = **+43%** | **Prédicteur #1** |

### Performance
- ✅ Modèles légers et rapides
- ✅ 5 features interprétables
- ✅ Interface intuitive

---

## 🏆 SLIDE 5 : CONCLUSION (10 secondes)

### Ce Projet Démontre

✅ **Feature Engineering** - Simplification 19 → 5 features  
✅ **Implémentation from scratch** - Compréhension profonde KNN/SVM  
✅ **Visualisation** - Interface complète et professionnelle  
✅ **Insights métier** - Découvertes contre-intuitives  

### Applications
- Banques : Évaluation crédit entreprises
- Assureurs : Tarification risque
- Investisseurs : Due diligence

---

## 🎬 SCRIPT PRÉSENTATION ORALE

### **[0:00 - 0:20] INTRODUCTION**
*"Bonjour, je vais vous présenter mon projet de prévision de faillite d'entreprise. L'objectif est de prédire si une entreprise risque la faillite en analysant 5 indicateurs simples. J'ai implémenté 2 algorithmes from scratch - KNN et SVM - sur un dataset de 161,254 entreprises marocaines, avec une interface Streamlit interactive."*

### **[0:20 - 0:45] DATASET**
*"Le dataset contient 161 mille entreprises avec une distribution déséquilibrée : 93% saines et 7% en faillite. J'ai sélectionné 5 features clés : l'âge de l'entreprise, sa taille en logarithme, un score de risque dette, un indice de stabilité, et le risque régional. Le point intéressant : le risque de dette est le prédicteur numéro 1, avec un écart de 43% entre entreprises saines et en faillite."*

### **[0:45 - 1:05] ALGORITHMES**
*"Pour les algorithmes, j'ai codé KNN avec distance euclidienne et vote majoritaire, et SVM avec gradient descent sur 1000 itérations. Les deux sont implémentés from scratch sans utiliser scikit-learn pour les modèles. Le notebook contient 12 sections : du chargement jusqu'à la sauvegarde de 6 fichiers pickle."*

### **[1:05 - 1:30] DÉMO NOTEBOOK**
*[Montrer notebook ouvert]  
"Dans le notebook, vous voyez le pipeline complet : normalisation manuelle, optimisation du K par cross-validation, training sur des échantillons de 20K et 10K pour rapidité, évaluation avec métriques et matrices de confusion, et comparaison visuelle KNN vs SVM."*

### **[1:30 - 1:55] DÉMO INTERFACE**
*[Montrer Streamlit]  
"L'interface a 5 pages. Dans la page prédiction, je saisis les données : par exemple une entreprise de 8 ans, 30 employés, risque dette faible à 0.2... Le modèle prédit : Entreprise SAINE. Maintenant un cas à risque : 22 ans, 180 employés, mais risque dette à 0.85 et région en crise à 15%... Résultat : RISQUE DE FAILLITE. L'interface affiche aussi les graphiques d'exploration et la comparaison des modèles."*

### **[1:55 - 2:10] INSIGHTS & CONCLUSION**
*"Les insights sont surprenants : les grandes entreprises ont 29% plus de risque, les anciennes aussi à 32%, et une stabilité trop élevée indique souvent une stagnation. En conclusion, ce projet démontre le feature engineering, l'implémentation from scratch, et produit des insights métier exploitables pour les banques ou investisseurs. Merci !"*

---

## 📋 CHECKLIST AVANT PRÉSENTATION

### Préparation Technique
- [ ] Notebook exécuté (tous les .pkl générés)
- [ ] Interface Streamlit testée (lance correctement)
- [ ] Données chargées (dataset_bankruptcy_prediction.csv présent)
- [ ] Graphiques s'affichent correctement
- [ ] Les 2 exemples de prédiction testés

### Fichiers à Avoir Ouverts
1. `model_bankruptcy_prediction.ipynb` (dans VS Code ou Jupyter)
2. Terminal prêt : `streamlit run interface_bankruptcy_prediction.py`
3. Ce fichier de présentation (pour référence)

### Exemples Prédiction à Préparer

**Exemple 1 - SAINE** ✅
```
Âge : 8 ans
Employés : 30
Risque dette : 0.2
Stabilité : 0.7
Région : 4%
```

**Exemple 2 - FAILLITE** ⚠️
```
Âge : 22 ans
Employés : 180
Risque dette : 0.85
Stabilité : 0.9
Région : 15%
```

---

## 🎯 POINTS CLÉS À RETENIR

1. **Mini-projet pédagogique** - Simple mais complet
2. **From scratch** - Compréhension profonde des algos
3. **5 features** - Simplification efficace (19→5)
4. **Insights métier** - Découvertes contre-intuitives
5. **Interface pro** - Streamlit avec 5 pages interactives
6. **Dataset réel** - 161K entreprises marocaines

---

## ⏱️ TIMING DÉTAILLÉ

| Section | Durée | Cumul |
|---------|-------|-------|
| Introduction | 20s | 0:20 |
| Dataset | 25s | 0:45 |
| Algorithmes | 20s | 1:05 |
| Démo Notebook | 25s | 1:30 |
| Démo Interface | 25s | 1:55 |
| Insights + Conclusion | 15s | 2:10 |
| **TOTAL** | **130s** | **2:10** |

*Marge : -10s (ajuster en parlant légèrement plus vite)*

---

## 💡 CONSEILS PRÉSENTATION

### À Faire ✅
- Parler clairement et avec assurance
- Montrer les résultats visuels (graphiques)
- Faire la démo interactive en direct
- Expliquer les insights surprenants
- Être enthousiaste sur les découvertes

### À Éviter ❌
- Trop de détails techniques
- Lire les slides mot à mot
- Rester sur le code trop longtemps
- Oublier de conclure sur l'utilité métier
- Dépasser 2min 15s

---

## 🎬 BONNE PRÉSENTATION ! 🚀
