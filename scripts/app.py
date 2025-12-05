import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# Page Config
st.set_page_config(page_title="ML Project Portfolio", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "Company Classification", "Salary Prediction (Marouane)", "Salary Prediction (Asaad)", "Bankruptcy Prediction (Irfan)"])

# ==========================================
# 0. Home / Dashboard
# ==========================================
def show_home():
    st.title("🤖 Machine Learning Project Portfolio")
    
    st.markdown("""
    ### Welcome!
    This portfolio showcases three distinct Machine Learning projects integrated into a single application.
    
    #### 👥 Project Team
    - **Abdelali Saadali**
    - **Asaad FETHALLAH**
    - **Marouane Mounir**
    - **Muhammad Irfan Wahyudi**
    
    ---
    
    ### 🚀 Projects Overview
    
    #### 1. 🏢 Company Classification AI
    *By Abdelali Saadali*
    - **Goal**: Classify companies based on worker data patterns.
    - **Algorithm**: K-Means Clustering (implemented from scratch).
    - **Key Features**: Visualizes stability, seasonality, and potential anomalies in company workforce data.
    
    #### 2. 🇲🇦 Salary Prediction (Marouane)
    *By Marouane Mounir*
    - **Goal**: Predict IT salaries in the Moroccan market for 2025.
    - **Algorithm**: Linear Regression (Gradient Descent from scratch).
    - **Key Features**: Takes into account profile, experience, education, technology, and target company.
    
    #### 3. 💵 Salary Predictor (Asaad)
    *By Asaad FETHALLAH*
    - **Goal**: Predict employee salaries based on historical data.
    - **Algorithm**: Linear Regression.
    - **Key Features**: Predicts salary for specific employees based on month and year.

    #### 4. 🏢 Bankruptcy Prediction (Irfan)
    *By Muhammad Irfan Wahyudi*
    - **Goal**: Predict company bankruptcy risk.
    - **Algorithm**: KNN & SVM (implemented from scratch).
    - **Key Features**: Analyzes financial stability, debt risk, and regional factors.
    
    ---
    
    ### 🛠️ Technologies Used
    - **Python**: Core programming language.
    - **Streamlit**: For building this interactive web application.
    - **Scikit-learn / NumPy / Pandas**: For data processing and machine learning.
    - **Matplotlib / Seaborn**: For data visualization.
    
    👈 **Use the sidebar to navigate between the projects.**
    """)

# ==========================================
# 1. Company Classification App
# ==========================================
def show_company_classification():
    st.title("🏢 Company Classification AI")
    st.markdown("""
    This application uses Machine Learning (K-Means Clustering) to classify companies into 4 categories based on their monthly worker data:
    - **Entreprises stables**: Consistent work days, high full-time ratio.
    - **Entreprises saisonnières**: Fewer work days, lower pay (likely part-time/seasonal).
    - **Entreprises irrégulières**: High variance in work days, mixed workforce.
    - **Entreprises potentiellement frauduleuses**: Anomalous data (e.g., extremely high salaries).
    """)

    # Load Data
    @st.cache_data
    def load_data():
        if os.path.exists('processed_features.csv'):
            df = pd.read_csv('processed_features.csv')
            return df
        return None

    df = load_data()

    if df is None:
        st.error("Data not found. Please run the analysis notebook/script first.")
        return

    # Load Adherents for Names
    try:
        adherents_path = '../Data CNSS/ADHERENTS.csv'
        if os.path.exists(adherents_path):
            adherents = pd.read_csv(adherents_path)
            adherents = adherents.rename(columns={'bank_adherent_adherentMandataire': 'ID_adherent'})
            adherents = adherents[['ID_adherent', 'companyName']]
            adherents = adherents.drop_duplicates(subset=['ID_adherent'])
            df = df.merge(adherents, on='ID_adherent', how='left')
            df['companyName'] = df['companyName'].fillna("Unknown")
        else:
            df['companyName'] = "Unknown"
            
    except Exception as e:
        st.warning(f"Could not load company names: {e}")
        df['companyName'] = "Unknown"

    # Sort by ID_adherent
    df = df.sort_values('ID_adherent')

    # Cluster Mapping
    CLUSTER_MAP = {
        0: "Entreprises saisonnières",
        1: "Entreprises irrégulières",
        2: "Entreprises potentiellement frauduleuses",
        3: "Entreprises stables"
    }

    COLOR_MAP = {
        "Entreprises stables": "green",
        "Entreprises saisonnières": "orange",
        "Entreprises irrégulières": "yellow",
        "Entreprises potentiellement frauduleuses": "red"
    }

    if 'cluster' in df.columns:
        df['Category'] = df['cluster'].map(CLUSTER_MAP)
    else:
        st.error("Cluster column not found in data.")
        return

    # Sidebar - Company Selection
    st.sidebar.header("Select Company")
    df['label'] = df['ID_adherent'].astype(str) + " - " + df['companyName']
    selected_label = st.sidebar.selectbox("Choose a Company", df['label'].unique())
    selected_id = int(selected_label.split(" - ")[0])

    # Main Content
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Company Details")
        company_data = df[df['ID_adherent'] == selected_id].iloc[0]
        
        category = company_data['Category']
        color = COLOR_MAP.get(category, "grey")
        
        st.markdown(f"### Classification: :{color}[{category}]")
        
        st.metric("Number of Workers", int(company_data['num_workers']))
        st.metric("Average Days Worked", f"{company_data['avg_days']:.2f}")
        st.metric("Average Salary", f"{company_data['avg_salary']:.2f} MAD")
        st.metric("Full Time Ratio", f"{company_data['full_time_ratio']*100:.1f}%")
        
        st.markdown("---")
        st.write("**Raw Stats:**")
        st.write(company_data[['std_days', 'std_salary', 'total_salary']])

    with col2:
        st.subheader("Cluster Visualization")
        
        # Scatter Plot 1
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='avg_days', y='std_days', hue='Category', palette=COLOR_MAP, alpha=0.6, ax=ax)
        sns.scatterplot(x=[company_data['avg_days']], y=[company_data['std_days']], color='black', s=200, marker='X', label=f"Selected ({selected_id})", ax=ax)
        plt.title("Work Stability: Average Days vs Standard Deviation")
        st.pyplot(fig)
        
        # Scatter Plot 2
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='avg_days', y='avg_salary', hue='Category', palette=COLOR_MAP, alpha=0.6, ax=ax2)
        sns.scatterplot(x=[company_data['avg_days']], y=[company_data['avg_salary']], color='black', s=200, marker='X', label=f"Selected ({selected_id})", ax=ax2)
        plt.yscale('log')
        plt.title("Salary vs Work Days (Log Scale Salary)")
        st.pyplot(fig2)

    st.subheader("All Companies Data")
    st.dataframe(df)

# ==========================================
# 2. Salary Prediction (Marouane)
# ==========================================
def show_salary_prediction_marouane():
    st.title("🇲🇦 Prédiction de Salaire IT - Marché Marocain 2025 (Marouane)")
    
    model_path = '../other MLs/ML-Project-marouane/model_salaire_IT_MAROC_2025.pkl'
    
    @st.cache_resource
    def load_marouane_model():
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file not found at {model_path}")
            return None

    model_params = load_marouane_model()
    
    if model_params:
        # Prediction Logic
        def predict_salary(profil, experience, niveau_etude, technologie, entreprise, model_params):
            theta = model_params['theta']
            mean = model_params['mean']
            std = model_params['std']
            feature_columns = model_params['feature_columns']
            
            input_data = pd.DataFrame({
                'Entreprise': [entreprise],
                'Profil': [profil],
                'Experience': [experience],
                'Niveau_Etude': [niveau_etude],
                'Technologie': [technologie],
                'Salaire': [0]
            })
            
            input_encoded = pd.get_dummies(input_data, columns=['Entreprise', 'Profil', 'Niveau_Etude', 'Technologie'], drop_first=False)
            input_encoded = input_encoded.drop('Salaire', axis=1)
            
            for col in feature_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            input_encoded = input_encoded[feature_columns]
            X_input = input_encoded.values
            X_input_norm = (X_input - mean) / std
            X_input_final = np.hstack((X_input_norm, np.ones((1, 1))))
            
            return X_input_final.dot(theta)[0][0]

        # UI
        col1, col2 = st.columns(2)
        with col1:
            profil = st.selectbox("Profil IT", sorted(model_params['profils_list']))
            experience = st.slider("Années d'expérience", 0, 15, 5)
            niveau_etude = st.selectbox("Niveau d'étude", sorted(model_params['niveaux_list']))
        
        with col2:
            technologie = st.selectbox("Technologie", sorted(model_params['technologies_list']))
            entreprise = st.selectbox("Entreprise", sorted(model_params['entreprises_list']))
            
        if st.button("Prédire le Salaire (Marouane)"):
            salaire = predict_salary(profil, experience, niveau_etude, technologie, entreprise, model_params)
            st.success(f"💰 Salaire Estimé: {salaire:,.2f} MAD")

# ==========================================
# 3. Salary Prediction (Asaad)
# ==========================================
def show_salary_prediction_asaad():
    st.title("Salary Predictor (Asaad)")
    
    model_path = '../other MLs/ML-Project-Asaad/AsaadModel/Asaad_Salaries_Models.pkl'
    
    @st.cache_resource
    def load_asaad_model():
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file not found at {model_path}")
            return None

    data = load_asaad_model()
    
    if data:
        models = data["models"]
        month_map = data["month_map"]
        
        def predict(full_name, month, year):
            if full_name not in models:
                return None
            m = models[full_name]
            x = np.array([month_map[month], year])
            x = (x - m["X_mean"]) / m["X_std"]
            x = np.hstack([1, x]).reshape(1,-1)
            y_norm = x @ m["w"]
            y = float(y_norm * m["y_std"] + m["y_mean"])
            return y

        full_name = st.text_input("Enter Employee's full name :")
        month = st.text_input("Enter a valid month :")
        year = st.number_input("Enter a valid year :", min_value=2000, max_value=2100, value=2025)
        
        if st.button("Predict Salary (Asaad)"):
            if not full_name or not month:
                st.warning("Fill all form spots.")
            else:
                month = month.lower()
                if month not in month_map:
                    st.error("Invalid Month !")
                else:
                    salaire = predict(full_name, month, year)
                    if salaire is None:
                        st.error("Unknown Employee.")
                    else:
                        st.success(f"💵 Predicted Salary: {salaire:.2f} MAD")

# ==========================================
# 4. Bankruptcy Prediction (Irfan)
# ==========================================

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

def show_bankruptcy_prediction_irfan():
    # CSS personnalisé pour cette section
    st.markdown("""
    <style>
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

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Bankruptcy Navigation :",
        ["🏠 Accueil", "📊 Exploration Données", "🔵 Prédiction KNN", "🟢 Prédiction SVM", "📈 Comparaison Modèles"]
    )

    # Base path for Irfan's project files
    BASE_PATH = '../other MLs/ML-Project-irfan/projet_faillite/'

    @st.cache_resource
    def load_irfan_models():
        """Charger les modèles et paramètres"""
        try:
            with open(BASE_PATH + 'knn_scratch_bankruptcy.pkl', 'rb') as f:
                knn_model = pickle.load(f)
            with open(BASE_PATH + 'svm_scratch_bankruptcy.pkl', 'rb') as f:
                svm_model = pickle.load(f)
            with open(BASE_PATH + 'bankruptcy_features.pkl', 'rb') as f:
                features = pickle.load(f)
            with open(BASE_PATH + 'bankruptcy_train_mean.pkl', 'rb') as f:
                train_mean = pickle.load(f)
            with open(BASE_PATH + 'bankruptcy_train_std.pkl', 'rb') as f:
                train_std = pickle.load(f)
            with open(BASE_PATH + 'best_k_bankruptcy.pkl', 'rb') as f:
                best_k = pickle.load(f)
            
            return knn_model, svm_model, features, train_mean, train_std, best_k
        except FileNotFoundError:
            st.error(f"⚠️ Modèles non trouvés dans {BASE_PATH}. Vérifiez le chemin.")
            return None, None, None, None, None, None

    @st.cache_data
    def load_irfan_dataset():
        """Charger le dataset"""
        try:
            if os.path.exists(BASE_PATH + 'dataset_bankruptcy_prediction.csv'):
                df = pd.read_csv(BASE_PATH + 'dataset_bankruptcy_prediction.csv')
                return df
            else:
                return None
        except FileNotFoundError:
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
    # PAGE 1 : ACCUEIL
    # ============================================================
    if page == "🏠 Accueil":
        st.title("🏢 Prévision de Faillite d'Entreprise (Irfan)")
        
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
        df = load_irfan_dataset()
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
        else:
            st.info("ℹ️ Le dataset original n'est pas disponible pour l'affichage des statistiques, mais les modèles sont chargés et prêts pour la prédiction.")

    # ============================================================
    # PAGE 2 : EXPLORATION DONNÉES
    # ============================================================
    elif page == "📊 Exploration Données":
        st.title("📊 Exploration des Données")
        
        df = load_irfan_dataset()
        
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
        else:
            st.warning("⚠️ Le fichier de données `dataset_bankruptcy_prediction.csv` est introuvable.")
            st.info("Vous pouvez toujours utiliser les pages de **Prédiction** (KNN & SVM) car les modèles sont pré-entraînés.")

    # ============================================================
    # PAGE 3 : PRÉDICTION KNN
    # ============================================================
    elif page == "🔵 Prédiction KNN":
        st.title("🔵 Prédiction KNN From Scratch")
        
        # Charger modèles
        knn_model, _, features, train_mean, train_std, best_k = load_irfan_models()
        
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
        st.title("🟢 Prédiction SVM From Scratch")
        
        # Charger modèles
        _, svm_model, features, train_mean, train_std, _ = load_irfan_models()
        
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
        st.title("📈 Comparaison KNN vs SVM")
        
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


# ==========================================
# Main Routing
# ==========================================
if app_mode == "Home":
    show_home()
elif app_mode == "Company Classification":
    show_company_classification()
elif app_mode == "Salary Prediction (Marouane)":
    show_salary_prediction_marouane()
elif app_mode == "Salary Prediction (Asaad)":
    show_salary_prediction_asaad()
elif app_mode == "Bankruptcy Prediction (Irfan)":
    show_bankruptcy_prediction_irfan()
