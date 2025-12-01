# 🇲🇦 Prédiction de Salaire IT - Marché Marocain 2025

Application de Machine Learning pour prédire les salaires des professionnels IT au Maroc basée sur un modèle de régression linéaire avec descente de gradient.

## 📊 Dataset

- **3,000 exemples** réels du marché IT marocain
- **20 entreprises multinationales** (Microsoft, Oracle, SAP, Capgemini, Accenture, etc.)
- **10 profils IT** (Data Scientist, DevOps, Cybersécurité, Cloud Engineer, etc.)
- **65+ technologies** (Python, React, AWS, Kubernetes, TensorFlow, etc.)
- **Salaires:** 9,000 - 65,000 MAD/mois

## 🎯 Fonctionnalités

- Modèle de régression linéaire **from scratch** avec descente de gradient
- One-Hot Encoding pour variables catégorielles
- Normalisation Z-Score
- Interface web interactive avec **Streamlit**
- Performances: R² ≈ 87-90% | MAE: ~3,500 MAD

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### 1. Entraîner le modèle

Ouvrir et exécuter le notebook:
```bash
jupyter notebook prediction_salaire_IT_MAROC_2025.ipynb
```

### 2. Lancer l'application Streamlit

```bash
streamlit run app_streamlit.py
```

L'application sera accessible sur `http://localhost:8501`

## 📁 Structure du Projet

```
├── DATASET_IT_MAROC_2025.csv              # Dataset d'entraînement
├── prediction_salaire_IT_MAROC_2025.ipynb # Notebook d'entraînement
├── model_salaire_IT_MAROC_2025.pkl        # Modèle entraîné
├── app_streamlit.py                        # Interface web
├── requirements.txt                        # Dépendances Python
└── README.md                               # Documentation
```

## 🛠️ Technologies Utilisées

- **Python 3.12**
- **NumPy** - Calculs matriciels
- **Pandas** - Manipulation de données
- **Matplotlib/Seaborn** - Visualisations
- **Streamlit** - Interface web
- **Scikit-learn** - Métriques d'évaluation

## 📈 Exemple de Prédiction

```
Profil: Data Scientist / IA
Expérience: 5 ans
Niveau: Ingénieur
Technologie: Python
Entreprise: Microsoft Maroc

→ Salaire prédit: ~35,000 MAD/mois
```

## 👥 Auteur

Développé pour le marché IT marocain 2025

## 📝 License

MIT License
