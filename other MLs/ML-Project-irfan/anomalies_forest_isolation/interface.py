import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ========================================
# CONFIGURATION DE LA PAGE
# ========================================
st.set_page_config(
    page_title="Détection d'Anomalies Salariales",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CHARGEMENT DES MODÈLES ET MAPPINGS
# ========================================
@st.cache_resource
def load_models():
    """Charge tous les modèles et objets sauvegardés"""
    try:
        model = pickle.load(open('isolation_forest_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        features_list = pickle.load(open('features_list.pkl', 'rb'))
        pca = pickle.load(open('pca_model.pkl', 'rb'))
        return model, scaler, features_list, pca
    except Exception as e:
        st.error(f"Erreur de chargement des modèles : {e}")
        return None, None, None, None

@st.cache_data
def load_encoding_mappings():
    """Charge les mappings de décodage des colonnes catégorielles"""
    try:
        mappings_df = pd.read_csv('prepareAdherents/encodage_mappings_adherents.csv')
        
        # Créer des dictionnaires de décodage par colonne
        decode_maps = {}
        for col in mappings_df['column'].unique():
            col_data = mappings_df[mappings_df['column'] == col]
            decode_maps[col] = dict(zip(col_data['code'], col_data['original_value']))
        
        return decode_maps
    except Exception as e:
        st.warning(f"Mappings non chargés : {e}")
        return {}

def decode_dataframe(df, decode_maps):
    """Décode les colonnes catégorielles d'un dataframe"""
    df_decoded = df.copy()
    
    for col, mapping in decode_maps.items():
        if col in df_decoded.columns:
            df_decoded[col] = df_decoded[col].map(mapping).fillna(df_decoded[col])
    
    return df_decoded

model, scaler, features_list, pca = load_models()
decode_maps = load_encoding_mappings()

# ========================================
# SIDEBAR - NAVIGATION
# ========================================
st.sidebar.title("🔍 Détection d'Anomalies")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📊 Analyser Dataset", "🔎 Prédire Anomalie", "📈 Visualisations", "ℹ️ À Propos"]
)

st.sidebar.markdown("---")
st.sidebar.info("**Modèle** : Isolation Forest\n\n**Features** : 13\n\n**Contamination** : 5%")

# ========================================
# PAGE 1 : ACCUEIL
# ========================================
if page == "🏠 Accueil":
    st.title("🔍 Système de Détection d'Anomalies Salariales")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🤖 Modèle", "Isolation Forest", "100 arbres")
    with col2:
        st.metric("📊 Features", "13", "Sélectionnées")
    with col3:
        st.metric("🎯 Précision", "95%", "Contamination 5%")
    
    st.markdown("---")
    
    st.header("📌 Fonctionnalités")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Analyser Dataset")
        st.write("""
        - Charger un fichier CSV d'entreprises
        - Détecter automatiquement les anomalies
        - Télécharger les résultats
        - Statistiques détaillées
        """)
        
        st.subheader("🔎 Prédire Anomalie")
        st.write("""
        - Saisir manuellement les données d'une entreprise
        - Obtenir une prédiction en temps réel
        - Score d'anomalie détaillé
        """)
    
    with col2:
        st.subheader("📈 Visualisations")
        st.write("""
        - Distribution des scores d'anomalie
        - Analyse PCA en 2D
        - Comparaison par région
        - Graphiques interactifs
        """)
        
        st.subheader("ℹ️ À Propos")
        st.write("""
        - Informations sur le modèle
        - Description des features
        - Documentation technique
        """)
    
    st.markdown("---")
    st.success("✅ Modèles chargés avec succès ! Utilisez le menu de navigation pour commencer.")

# ========================================
# PAGE 2 : ANALYSER DATASET
# ========================================
elif page == "📊 Analyser Dataset":
    st.title("📊 Analyse de Dataset Complet")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📁 Charger un fichier CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Fichier chargé : {len(df)} entreprises")
        
        # Vérifier les colonnes
        missing_features = [f for f in features_list if f not in df.columns]
        
        if missing_features:
            st.error(f"❌ Colonnes manquantes : {missing_features}")
        else:
            # Préparer les données
            X = df[features_list].copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # Normaliser
            X_scaled = scaler.transform(X)
            
            # Prédire
            predictions = model.predict(X_scaled)
            scores = model.score_samples(X_scaled)
            
            # Ajouter au dataframe
            df['prediction'] = predictions
            df['score_anomalie'] = scores
            df['est_anomalie'] = (predictions == -1).astype(int)
            
            # Statistiques
            nb_anomalies = (df['est_anomalie'] == 1).sum()
            nb_normaux = (df['est_anomalie'] == 0).sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🚨 Anomalies", nb_anomalies, f"{nb_anomalies/len(df)*100:.1f}%")
            with col2:
                st.metric("✅ Normaux", nb_normaux, f"{nb_normaux/len(df)*100:.1f}%")
            with col3:
                st.metric("📊 Total", len(df))
            
            st.markdown("---")
            
            # Tableau des anomalies
            st.subheader("🚨 Top 20 Entreprises Anormales")
            anomalies = df[df['est_anomalie'] == 1].nsmallest(20, 'score_anomalie')
            
            if len(anomalies) > 0:
                # Décoder les colonnes catégorielles
                anomalies_display = decode_dataframe(anomalies, decode_maps)
                
                cols_display = ['affiliateNumber', 'directionRegionale', 'nb_salaries', 'salaire_moyen', 
                               'masse_salariale_totale', 'taux_salaries_actifs', 'score_anomalie']
                cols_available = [c for c in cols_display if c in anomalies_display.columns]
                st.dataframe(anomalies_display[cols_available], use_container_width=True)
            else:
                st.info("Aucune anomalie détectée.")
            
            st.markdown("---")
            
            # Télécharger les résultats
            col1, col2 = st.columns(2)
            
            with col1:
                # Décoder avant téléchargement
                df_decoded = decode_dataframe(df, decode_maps)
                csv_all = df_decoded.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger Résultats Complets",
                    data=csv_all,
                    file_name="resultats_anomalies.csv",
                    mime="text/csv"
                )
            
            with col2:
                anomalies_decoded = decode_dataframe(df[df['est_anomalie'] == 1], decode_maps)
                csv_anomalies = anomalies_decoded.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger Anomalies Seulement",
                    data=csv_anomalies,
                    file_name="entreprises_anormales.csv",
                    mime="text/csv"
                )

# ========================================
# PAGE 3 : PRÉDIRE ANOMALIE
# ========================================
elif page == "🔎 Prédire Anomalie":
    st.title("🔎 Prédire une Anomalie")
    st.markdown("---")
    
    st.info("💡 Saisissez les informations d'une entreprise pour obtenir une prédiction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Informations Générales")
        
        # Dropdown pour Direction Régionale avec noms décodés
        if 'directionRegionale' in decode_maps:
            region_options = decode_maps['directionRegionale']
            region_names = list(region_options.values())
            region_selected = st.selectbox("Direction Régionale", region_names, index=6 if len(region_names) > 6 else 0)
            # Trouver le code correspondant
            direction_regionale = [k for k, v in region_options.items() if v == region_selected][0]
        else:
            direction_regionale = st.number_input("Direction Régionale", min_value=0, max_value=20, value=6)
        
        nb_salaries = st.number_input("Nombre de Salariés", min_value=1, max_value=5000, value=50)
        masse_salariale = st.number_input("Masse Salariale Totale (€)", min_value=0.0, value=100000.0, step=1000.0)
        salaire_moyen = st.number_input("Salaire Moyen (€)", min_value=0.0, value=2000.0, step=100.0)
        salaire_median = st.number_input("Salaire Médian (€)", min_value=0.0, value=1800.0, step=100.0)
        salaire_max = st.number_input("Salaire Maximum (€)", min_value=0.0, value=3500.0, step=100.0)
        salaire_min = st.number_input("Salaire Minimum (€)", min_value=0.0, value=1500.0, step=100.0)
    
    with col2:
        st.subheader("📊 Statistiques")
        salaire_std = st.number_input("Écart-type Salaire (€)", min_value=0.0, value=500.0, step=50.0)
        total_jours = st.number_input("Total Jours Travaillés", min_value=1, max_value=50000, value=1200)
        taux_actifs = st.slider("Taux Salariés Actifs (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
        salaire_par_jour = st.number_input("Salaire par Jour (€)", min_value=0.0, value=83.33, step=1.0)
        jours_par_salarie = st.number_input("Jours par Salarié", min_value=1, max_value=365, value=24)
        ecart_region = st.number_input("Écart Salaire Région (€)", min_value=-10000.0, max_value=10000.0, value=0.0, step=100.0)
        compte_risque = st.selectbox("Compte à Risque", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    
    st.markdown("---")
    
    if st.button("🔍 ANALYSER", type="primary", use_container_width=True):
        # Créer le vecteur de features
        input_data = pd.DataFrame({
            'nb_salaries': [nb_salaries],
            'masse_salariale_totale': [masse_salariale],
            'salaire_moyen': [salaire_moyen],
            'salaire_median': [salaire_median],
            'salaire_max': [salaire_max],
            'salaire_min': [salaire_min],
            'salaire_std': [salaire_std],
            'total_jours_travailles': [total_jours],
            'taux_salaries_actifs': [taux_actifs],
            'salaire_par_jour': [salaire_par_jour],
            'jours_par_salarie': [jours_par_salarie],
            'ecart_salaire_region': [ecart_region],
            'compte_a_risque': [compte_risque]
        })
        
        # Normaliser
        input_scaled = scaler.transform(input_data)
        
        # Prédire
        prediction = model.predict(input_scaled)[0]
        score = model.score_samples(input_scaled)[0]
        
        st.markdown("---")
        
        # Résultat
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == -1:
                st.error("🚨 ANOMALIE DÉTECTÉE")
            else:
                st.success("✅ ENTREPRISE NORMALE")
        
        with col2:
            st.metric("Score d'Anomalie", f"{score:.4f}")
        
        with col3:
            risk_level = "ÉLEVÉ" if score < -0.3 else "MOYEN" if score < -0.1 else "FAIBLE"
            color = "🔴" if score < -0.3 else "🟠" if score < -0.1 else "🟢"
            st.metric("Niveau de Risque", f"{color} {risk_level}")
        
        st.markdown("---")
        
        # Interprétation
        st.subheader("📋 Interprétation")
        
        # Afficher la région décodée
        if 'directionRegionale' in decode_maps:
            region_name = decode_maps['directionRegionale'].get(direction_regionale, f"Région {direction_regionale}")
            st.write(f"**Région** : {region_name}")
        
        if prediction == -1:
            st.warning("""
            ⚠️ **Cette entreprise présente des caractéristiques anormales.**
            
            Recommandations :
            - Vérifier la cohérence des données salariales
            - Analyser les écarts par rapport à la région
            - Investiguer si compte à risque activé
            - Comparer avec des entreprises similaires
            """)
        else:
            st.info("""
            ✅ **Cette entreprise présente un profil normal.**
            
            Les caractéristiques salariales sont cohérentes avec les entreprises similaires.
            """)

# ========================================
# PAGE 4 : VISUALISATIONS
# ========================================
elif page == "📈 Visualisations":
    st.title("📈 Visualisations et Analyses")
    st.markdown("---")
    
    # Charger le dataset complet
    try:
        df_viz = pd.read_csv('resultats_anomalies_detection.csv')
        
        tab1, tab2, tab3 = st.tabs(["📊 Distribution Scores", "🗺️ PCA 2D", "📍 Analyse Régionale"])
        
        # TAB 1: Distribution
        with tab1:
            st.subheader("Distribution des Scores d'Anomalie")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogramme
            ax1.hist(df_viz['score_anomalie'], bins=50, color='skyblue', edgecolor='black')
            ax1.axvline(df_viz['score_anomalie'].mean(), color='red', linestyle='--', 
                       label=f"Moyenne: {df_viz['score_anomalie'].mean():.3f}")
            ax1.set_title('Distribution des Scores', fontweight='bold')
            ax1.set_xlabel('Score d\'anomalie')
            ax1.set_ylabel('Fréquence')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Boxplot
            normal = df_viz[df_viz['est_anomalie'] == 0]['score_anomalie']
            anomalie = df_viz[df_viz['est_anomalie'] == 1]['score_anomalie']
            ax2.boxplot([normal, anomalie], labels=['Normal', 'Anomalie'])
            ax2.set_title('Comparaison des Scores', fontweight='bold')
            ax2.set_ylabel('Score d\'anomalie')
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        # TAB 2: PCA
        with tab2:
            st.subheader("Visualisation PCA 2D")
            
            X_viz = df_viz[features_list].copy()
            X_viz = X_viz.replace([np.inf, -np.inf], np.nan).fillna(X_viz.median())
            X_scaled_viz = scaler.transform(X_viz)
            X_pca_viz = pca.transform(X_scaled_viz)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = ['blue' if x == 0 else 'red' for x in df_viz['est_anomalie']]
            ax.scatter(X_pca_viz[:, 0], X_pca_viz[:, 1], c=colors, alpha=0.5, s=30)
            
            ax.set_title('Détection d\'Anomalies - PCA 2D', fontsize=16, fontweight='bold')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
            ax.legend(['Normal', 'Anomalie'])
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        # TAB 3: Régions
        with tab3:
            st.subheader("Analyse par Région")
            
            if 'directionRegionale' in df_viz.columns:
                # Décoder les régions
                df_viz_decoded = decode_dataframe(df_viz, decode_maps)
                
                region_stats = df_viz_decoded.groupby('directionRegionale').agg({
                    'est_anomalie': ['sum', 'mean', 'count']
                }).round(3)
                
                region_stats.columns = ['Nb_Anomalies', 'Taux_Anomalie', 'Total']
                region_stats = region_stats.sort_values('Nb_Anomalies', ascending=False)
                
                st.dataframe(region_stats, use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                region_stats['Nb_Anomalies'].plot(kind='bar', color='coral', edgecolor='black', ax=ax)
                ax.set_title('Nombre d\'Anomalies par Région', fontsize=14, fontweight='bold')
                ax.set_xlabel('Direction Régionale')
                ax.set_ylabel('Nombre d\'Anomalies')
                ax.grid(True, alpha=0.3, axis='y')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Colonne 'directionRegionale' non trouvée dans le dataset.")
    
    except FileNotFoundError:
        st.error("❌ Fichier 'resultats_anomalies_detection.csv' introuvable. Veuillez d'abord analyser un dataset.")

# ========================================
# PAGE 5 : À PROPOS
# ========================================
elif page == "ℹ️ À Propos":
    st.title("ℹ️ À Propos du Système")
    st.markdown("---")
    
    st.header("🤖 Modèle : Isolation Forest")
    st.write("""
    L'**Isolation Forest** est un algorithme de détection d'anomalies non supervisé qui :
    - Isole les observations en sélectionnant aléatoirement une feature
    - Sépare les valeurs entre le min et max de cette feature
    - Les anomalies nécessitent moins de partitions (sont isolées plus rapidement)
    - Score négatif = plus l'entreprise est anormale
    """)
    
    st.markdown("---")
    
    st.header("📊 Features Utilisées (13)")
    
    features_info = {
        "nb_salaries": "Nombre de salariés dans l'entreprise",
        "masse_salariale_totale": "Somme totale des salaires (€)",
        "salaire_moyen": "Salaire moyen des employés (€)",
        "salaire_median": "Salaire médian (€)",
        "salaire_max": "Salaire maximum (€)",
        "salaire_min": "Salaire minimum (€)",
        "salaire_std": "Écart-type des salaires (€)",
        "total_jours_travailles": "Total de jours travaillés",
        "taux_salaries_actifs": "Pourcentage de salariés actifs (%)",
        "salaire_par_jour": "Coût salarial quotidien (€/jour)",
        "jours_par_salarie": "Moyenne de jours par salarié",
        "ecart_salaire_region": "Différence par rapport à la moyenne régionale (€)",
        "compte_a_risque": "Indicateur de compte bancaire à risque (0/1)"
    }
    
    for feature, description in features_info.items():
        st.write(f"**{feature}** : {description}")
    
    st.markdown("---")
    
    st.header("⚙️ Paramètres du Modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**n_estimators** : 100 arbres")
        st.write("**contamination** : 0.05 (5% anomalies)")
        st.write("**random_state** : 42")
    
    with col2:
        st.write("**Normalisation** : StandardScaler")
        st.write("**PCA** : 2 composantes principales")
        st.write("**Score seuil** : -0.1 (personnalisable)")
    
    st.markdown("---")
    
    st.success("✅ Système développé pour détecter les anomalies dans les données salariales des entreprises.")
