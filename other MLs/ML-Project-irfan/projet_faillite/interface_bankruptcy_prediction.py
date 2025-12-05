import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# MODÈLES FROM SCRATCH (nécessaires pour unpickle)
# ============================================================

class KNNFromScratch:
    """KNN implémenté from scratch"""
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def _predict_single(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        from collections import Counter
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])


class SVMFromScratch:
    """SVM implémenté from scratch"""
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0
        
        for iteration in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
        return self
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.where(linear_output >= 0, 1, 0)

# ============================================================
# CONFIGURATION PAGE
# ============================================================

st.set_page_config(
    page_title="Prévision Faillite Entreprise",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e8f4f8 0%, #ffffff 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

@st.cache_resource
def load_models():
    """Charger les modèles et paramètres"""
    try:
        with open('projet_faillite/knn_scratch_bankruptcy.pkl', 'rb') as f:
            knn_model = pickle.load(f)
        with open('projet_faillite/svm_scratch_bankruptcy.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        with open('projet_faillite/bankruptcy_features.pkl', 'rb') as f:
            features = pickle.load(f)
        with open('projet_faillite/bankruptcy_train_mean.pkl', 'rb') as f:
            train_mean = pickle.load(f)
        with open('projet_faillite/bankruptcy_train_std.pkl', 'rb') as f:
            train_std = pickle.load(f)
        with open('projet_faillite/best_k_bankruptcy.pkl', 'rb') as f:
            best_k = pickle.load(f)
        
        return knn_model, svm_model, features, train_mean, train_std, best_k
    except FileNotFoundError:
        st.error("⚠️ Modèles non trouvés. Veuillez d'abord exécuter le notebook `model_bankruptcy_prediction.ipynb`")
        return None, None, None, None, None, None

@st.cache_data
def load_dataset():
    """Charger le dataset"""
    try:
        df = pd.read_csv('projet_faillite/dataset_bankruptcy_prediction.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset non trouvé : `dataset_bankruptcy_prediction.csv`")
        return None

def predict_bankruptcy(features_values, model, train_mean, train_std):
    """Faire une prédiction"""
    # Créer DataFrame
    input_df = pd.DataFrame([features_values])
    
    # Normaliser
    input_scaled = (input_df - train_mean) / train_std
    
    # Prédire
    prediction = model.predict(input_scaled.values)
    
    return prediction[0]

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🏢 Navigation")
page = st.sidebar.radio(
    "Choisir une page :",
    ["🏠 Accueil", "📊 Exploration Données", "🔵 Prédiction KNN", "🟢 Prédiction SVM", "📈 Comparaison Modèles"]
)

# ============================================================
# PAGE 1 : ACCUEIL
# ============================================================

if page == "🏠 Accueil":
    st.markdown('<div class="main-header">🏢 PRÉVISION DE FAILLITE D\'ENTREPRISE</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Objectif du Projet
    
    Ce mini-projet utilise des algorithmes **KNN** et **SVM** implémentés **from scratch** pour prédire 
    si une entreprise est à risque de **faillite** (défaut bancaire).
    
    ---
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("🤖 Algorithmes", "2", "KNN & SVM")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("📊 Features", "5", "Simples")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("🎯 Target", "Binaire", "0/1")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Description des features
    st.subheader("📋 Les 5 Features")
    
    features_desc = pd.DataFrame({
        'Feature': [
            '1. firm_age_years',
            '2. firm_size_log',
            '3. debt_risk_score',
            '4. stability_index',
            '5. regional_risk'
        ],
        'Description': [
            'Âge de l\'entreprise (années)',
            'Taille entreprise (log employés)',
            'Score risque financier (0-1)',
            'Indice stabilité globale (0-1)',
            'Risque régional - taux défaut (%)'
        ]
    })
    
    st.table(features_desc)
    
    st.markdown("---")
    
    # Dataset info
    df = load_dataset()
    if df is not None:
        st.subheader("📊 Informations Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entreprises", f"{len(df):,}")
        with col2:
            st.metric("Entreprises Saines", f"{(df['bank_accountDefaultState']==0).sum():,}")
        with col3:
            st.metric("Entreprises Faillite", f"{(df['bank_accountDefaultState']==1).sum():,}")
        with col4:
            st.metric("Features", "5")
    
    st.markdown("---")
    
    st.info("👈 Utilisez le menu latéral pour naviguer entre les différentes pages")

# ============================================================
# PAGE 2 : EXPLORATION DONNÉES
# ============================================================

elif page == "📊 Exploration Données":
    st.markdown('<div class="main-header">📊 EXPLORATION DES DONNÉES</div>', unsafe_allow_html=True)
    
    df = load_dataset()
    
    if df is not None:
        # Aperçu données
        st.subheader("🔍 Aperçu du Dataset")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Statistiques
        st.subheader("📈 Statistiques Descriptives")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Distribution target
        st.subheader("🎯 Distribution de la Target")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            target_counts = df['bank_accountDefaultState'].value_counts()
            colors = ['#2ecc71', '#e74c3c']
            ax.bar(['SAINE (0)', 'FAILLITE (1)'], target_counts.values, color=colors)
            ax.set_ylabel('Nombre d\'entreprises', fontsize=12)
            ax.set_title('Distribution des Classes', fontsize=14, fontweight='bold')
            for i, v in enumerate(target_counts.values):
                ax.text(i, v, f'{v:,}', ha='center', va='bottom')
            st.pyplot(fig)
        
        with col2:
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(target_counts.values, labels=['SAINE (0)', 'FAILLITE (1)'], 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Répartition Target', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Analyse par statut
        st.subheader("📊 Comparaison SAINE vs FAILLITE")
        
        features_cols = ['firm_age_years', 'firm_size_log', 'debt_risk_score', 
                        'stability_index', 'regional_risk']
        
        comparison = df.groupby('bank_accountDefaultState')[features_cols].mean()
        comparison.index = ['SAINE (0)', 'FAILLITE (1)']
        
        st.dataframe(comparison.T, use_container_width=True)
        
        # Graphique comparaison
        fig, ax = plt.subplots(figsize=(12, 6))
        comparison.T.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
        ax.set_title('Moyennes des Features par Statut', fontsize=14, fontweight='bold')
        ax.set_ylabel('Valeur Moyenne')
        ax.set_xlabel('Features')
        ax.legend(['SAINE', 'FAILLITE'])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================
# PAGE 3 : PRÉDICTION KNN
# ============================================================

elif page == "🔵 Prédiction KNN":
    st.markdown('<div class="main-header">🔵 PRÉDICTION KNN FROM SCRATCH</div>', unsafe_allow_html=True)
    
    # Charger modèles
    knn_model, _, features, train_mean, train_std, best_k = load_models()
    
    if knn_model is not None:
        st.markdown(f"""
        <div class="info-box">
            <strong>ℹ️ Modèle KNN</strong><br>
            K = {best_k} voisins | Distance euclidienne | Vote majoritaire
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📝 Saisir les Informations de l'Entreprise")
        
        col1, col2 = st.columns(2)
        
        with col1:
            firm_age = st.number_input(
                "🏢 Âge de l'entreprise (années)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                help="Ancienneté de l'entreprise en années"
            )
            
            firm_size = st.number_input(
                "👥 Nombre d'employés",
                min_value=1,
                max_value=10000,
                value=10,
                step=1,
                help="Nombre total d'employés"
            )
            firm_size_log = np.log1p(firm_size)
            
            debt_risk = st.slider(
                "💳 Score Risque Dette",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Score composite du risque financier (0=faible, 1=élevé)"
            )
        
        with col2:
            stability = st.slider(
                "📊 Indice de Stabilité",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Indice de stabilité de l'entreprise (0=instable, 1=stable)"
            )
            
            regional_risk = st.slider(
                "🗺️ Risque Régional (%)",
                min_value=0.0,
                max_value=20.0,
                value=7.0,
                step=0.5,
                help="Taux de défaut dans la région"
            ) / 100
        
        st.markdown("---")
        
        # Bouton prédiction
        if st.button("🔮 Prédire avec KNN", type="primary", use_container_width=True):
            # Préparer features
            features_values = {
                'firm_age_years': firm_age,
                'firm_size_log': firm_size_log,
                'debt_risk_score': debt_risk,
                'stability_index': stability,
                'regional_risk': regional_risk
            }
            
            # Prédire
            prediction = predict_bankruptcy(features_values, knn_model, train_mean, train_std)
            
            # Afficher résultat
            st.markdown("### 🎯 Résultat de la Prédiction")
            
            if prediction == 0:
                st.markdown("""
                <div class="success-box">
                    <h2 style="color: #28a745; margin: 0;">✅ ENTREPRISE SAINE</h2>
                    <p style="margin: 0.5rem 0 0 0;">
                        Le modèle KNN prédit que cette entreprise est <strong>financièrement stable</strong> 
                        et présente un <strong>faible risque de faillite</strong>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="danger-box">
                    <h2 style="color: #dc3545; margin: 0;">⚠️ RISQUE DE FAILLITE</h2>
                    <p style="margin: 0.5rem 0 0 0;">
                        Le modèle KNN prédit que cette entreprise présente un <strong>risque élevé de faillite</strong>. 
                        Une attention particulière est recommandée.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Afficher features
            st.markdown("#### 📊 Features Utilisées")
            
            features_df = pd.DataFrame({
                'Feature': [
                    'Âge entreprise',
                    'Taille (log)',
                    'Risque dette',
                    'Stabilité',
                    'Risque régional'
                ],
                'Valeur': [
                    f"{firm_age:.1f} ans",
                    f"{firm_size_log:.3f}",
                    f"{debt_risk:.2f}",
                    f"{stability:.2f}",
                    f"{regional_risk:.4f}"
                ]
            })
            
            st.table(features_df)

# ============================================================
# PAGE 4 : PRÉDICTION SVM
# ============================================================

elif page == "🟢 Prédiction SVM":
    st.markdown('<div class="main-header">🟢 PRÉDICTION SVM FROM SCRATCH</div>', unsafe_allow_html=True)
    
    # Charger modèles
    _, svm_model, features, train_mean, train_std, _ = load_models()
    
    if svm_model is not None:
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ Modèle SVM</strong><br>
            Gradient Descent | 1000 itérations | Hyperplan optimal
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📝 Saisir les Informations de l'Entreprise")
        
        col1, col2 = st.columns(2)
        
        with col1:
            firm_age_svm = st.number_input(
                "🏢 Âge de l'entreprise (années)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                key="svm_age"
            )
            
            firm_size_svm = st.number_input(
                "👥 Nombre d'employés",
                min_value=1,
                max_value=10000,
                value=10,
                step=1,
                key="svm_size"
            )
            firm_size_log_svm = np.log1p(firm_size_svm)
            
            debt_risk_svm = st.slider(
                "💳 Score Risque Dette",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.1,
                key="svm_debt"
            )
        
        with col2:
            stability_svm = st.slider(
                "📊 Indice de Stabilité",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="svm_stability"
            )
            
            regional_risk_svm = st.slider(
                "🗺️ Risque Régional (%)",
                min_value=0.0,
                max_value=20.0,
                value=7.0,
                step=0.5,
                key="svm_regional"
            ) / 100
        
        st.markdown("---")
        
        # Bouton prédiction
        if st.button("🔮 Prédire avec SVM", type="primary", use_container_width=True):
            # Préparer features
            features_values_svm = {
                'firm_age_years': firm_age_svm,
                'firm_size_log': firm_size_log_svm,
                'debt_risk_score': debt_risk_svm,
                'stability_index': stability_svm,
                'regional_risk': regional_risk_svm
            }
            
            # Prédire
            prediction_svm = predict_bankruptcy(features_values_svm, svm_model, train_mean, train_std)
            
            # Afficher résultat
            st.markdown("### 🎯 Résultat de la Prédiction")
            
            if prediction_svm == 0:
                st.markdown("""
                <div class="success-box">
                    <h2 style="color: #28a745; margin: 0;">✅ ENTREPRISE SAINE</h2>
                    <p style="margin: 0.5rem 0 0 0;">
                        Le modèle SVM prédit que cette entreprise est <strong>financièrement stable</strong> 
                        et présente un <strong>faible risque de faillite</strong>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="danger-box">
                    <h2 style="color: #dc3545; margin: 0;">⚠️ RISQUE DE FAILLITE</h2>
                    <p style="margin: 0.5rem 0 0 0;">
                        Le modèle SVM prédit que cette entreprise présente un <strong>risque élevé de faillite</strong>. 
                        Une attention particulière est recommandée.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Afficher features
            st.markdown("#### 📊 Features Utilisées")
            
            features_df_svm = pd.DataFrame({
                'Feature': [
                    'Âge entreprise',
                    'Taille (log)',
                    'Risque dette',
                    'Stabilité',
                    'Risque régional'
                ],
                'Valeur': [
                    f"{firm_age_svm:.1f} ans",
                    f"{firm_size_log_svm:.3f}",
                    f"{debt_risk_svm:.2f}",
                    f"{stability_svm:.2f}",
                    f"{regional_risk_svm:.4f}"
                ]
            })
            
            st.table(features_df_svm)

# ============================================================
# PAGE 5 : COMPARAISON MODÈLES
# ============================================================

elif page == "📈 Comparaison Modèles":
    st.markdown('<div class="main-header">📈 COMPARAISON KNN vs SVM</div>', unsafe_allow_html=True)
    
    st.subheader("🎯 Avantages du Mini-Projet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Points Forts
        
        - **Simplicité** : 5 features faciles à comprendre
        - **Rapidité** : Modèles légers et rapides
        - **Interprétabilité** : Features métier claires
        - **From Scratch** : Compréhension complète des algos
        - **Équilibré** : Complexité adaptée pour apprentissage
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Modèles Implémentés
        
        - **KNN** : K plus proches voisins
        - **SVM** : Support Vector Machine
        - **Normalisation** : StandardScaler from scratch
        - **Métriques** : Accuracy, Precision, Recall, F1
        - **Visualisations** : Matrices de confusion
        """)
    
    st.markdown("---")
    
    # Tableau comparatif
    st.subheader("📊 Comparaison Technique")
    
    comparison_data = {
        'Aspect': [
            'Algorithme',
            'Implémentation',
            'Complexité',
            'Distance/Méthode',
            'Temps Entraînement',
            'Interprétabilité',
            'Adapté pour'
        ],
        'KNN': [
            'K-Nearest Neighbors',
            'From Scratch',
            'Simple',
            'Distance Euclidienne',
            'Instantané (lazy)',
            '⭐⭐⭐',
            'Petits datasets'
        ],
        'SVM': [
            'Support Vector Machine',
            'From Scratch',
            'Moyenne',
            'Gradient Descent',
            'Itératif (1000 iter)',
            '⭐⭐',
            'Classification binaire'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
    
    st.markdown("---")
    
    st.subheader("💡 Features Utilisées (5)")
    
    features_impact = pd.DataFrame({
        'Rang': [1, 2, 3, 4, 5],
        'Feature': [
            'debt_risk_score',
            'stability_index',
            'firm_age_years',
            'regional_risk',
            'firm_size_log'
        ],
        'Impact Prévu': [
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐⭐',
            '⭐⭐⭐',
            '⭐⭐',
            '⭐⭐'
        ]
    })
    
    st.table(features_impact)
    
    st.markdown("---")
    
    st.info("💡 **Note** : Pour voir les performances réelles, exécutez d'abord le notebook `model_bankruptcy_prediction.ipynb`")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏢 <strong>Mini-Projet : Prévision de Faillite d'Entreprise</strong></p>
    <p>KNN & SVM FROM SCRATCH | 5 Features | Classification Binaire</p>
</div>
""", unsafe_allow_html=True)
