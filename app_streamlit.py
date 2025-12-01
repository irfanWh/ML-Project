import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Salaire IT",
    page_icon="💼",
    layout="wide"
)

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    """Charge les paramètres du modèle depuis le fichier pickle"""
    try:
        with open('model_salaire_IT_MAROC_2025.pkl', 'rb') as f:
            model_params = pickle.load(f)
        return model_params
    except FileNotFoundError:
        st.error("❌ Fichier du modèle introuvable! Veuillez d'abord entraîner le modèle en exécutant le notebook 'prediction_salaire_IT_MAROC_2025.ipynb'")
        return None

# Fonction de prédiction
def predict_salary(profil, experience, niveau_etude, technologie, entreprise, model_params):
    """
    Prédit le salaire d'un professionnel IT au Maroc
    
    Args:
    - profil: profil IT (string) - ex: Data Scientist / IA, DevOps Engineer, etc.
    - experience: années d'expérience (int)
    - niveau_etude: niveau d'étude (string) - Bac+2, Bac+3, Master, Ingénieur, Doctorat
    - technologie: technologie/compétence (string)
    - entreprise: nom de l'entreprise (string)
    - model_params: dictionnaire contenant les paramètres du modèle
    
    Returns:
    - salaire_predit: salaire prédit en MAD/mois ou message d'erreur
    """
    # Récupération des paramètres
    theta = model_params['theta']
    mean = model_params['mean']
    std = model_params['std']
    feature_columns = model_params['feature_columns']
    
    # Vérifier que les valeurs existent
    if entreprise not in model_params['entreprises_list']:
        return None, f"Erreur: Entreprise '{entreprise}' non reconnue"
    if profil not in model_params['profils_list']:
        return None, f"Erreur: Profil '{profil}' non reconnu"
    if niveau_etude not in model_params['niveaux_list']:
        return None, f"Erreur: Niveau d'étude '{niveau_etude}' non reconnu"
    if technologie not in model_params['technologies_list']:
        return None, f"Erreur: Technologie '{technologie}' non reconnue"
    
    # Créer un DataFrame avec les mêmes colonnes que le dataset MAROC 2025
    input_data = pd.DataFrame({
        'Entreprise': [entreprise],
        'Profil': [profil],
        'Experience': [experience],
        'Niveau_Etude': [niveau_etude],
        'Technologie': [technologie],
        'Salaire': [0]
    })
    
    # One-Hot Encoding
    input_encoded = pd.get_dummies(input_data, 
                                    columns=['Entreprise', 'Profil', 'Niveau_Etude', 'Technologie'],
                                    drop_first=False)
    
    # Supprimer la colonne Salaire
    input_encoded = input_encoded.drop('Salaire', axis=1)
    
    # Ajouter les colonnes manquantes
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Réorganiser les colonnes
    input_encoded = input_encoded[feature_columns]
    
    # Convertir en numpy array
    X_input = input_encoded.values
    
    # Normalisation
    X_input_norm = (X_input - mean) / std
    
    # Ajouter la colonne de biais
    X_input_final = np.hstack((X_input_norm, np.ones((1, 1))))
    
    # Prédiction
    salaire_predit = X_input_final.dot(theta)[0][0]
    
    return salaire_predit, None

# Titre principal
st.title("🇲🇦 Prédiction de Salaire IT - Marché Marocain 2025")
st.markdown("---")

# Chargement du modèle
model_params = load_model()

if model_params is not None:
    # Affichage des informations sur le modèle
    with st.expander("ℹ️ Informations sur le modèle", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score (Train)", f"{model_params['r2_train']:.4f}")
        with col2:
            st.metric("R² Score (Test)", f"{model_params['r2_test']:.4f}")
        with col3:
            st.metric("Performance", f"{model_params['r2_test']*100:.2f}%")
        
        st.info(f"""
        **Modèle:** Régression Linéaire Multiple (from scratch avec Descente de Gradient)
        
        **Dataset:** DATASET_IT_MAROC_2025.csv (3000 exemples réels)
        
        **Marché:** 20 entreprises multinationales au Maroc
        
        **Features utilisées:** 
        - Profil IT (10 profils: IA, Cyber, Cloud, DevOps, etc.) - One-Hot Encoded
        - Années d'expérience (0-15 ans)
        - Niveau d'étude (Bac+2 → Doctorat) - One-Hot Encoded
        - Technologie (65+ technologies) - One-Hot Encoded
        - Entreprise (20 multinationales) - One-Hot Encoded
        
        **Performances:** R² ≈ {model_params['r2_test']*100:.1f}% | MAE: {model_params.get('mae_test', 0):,.0f} MAD
        """)
    
    st.markdown("---")
    
    # Interface de saisie
    st.header("🔍 Saisir les informations du professionnel IT")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Récupération des listes déroulantes
        profils_disponibles = sorted(model_params['profils_list'])
        niveaux_etude_disponibles = sorted(model_params['niveaux_list'])
        
        profil = st.selectbox(
            "👨‍💻 Profil IT",
            options=profils_disponibles,
            help="Sélectionnez le profil professionnel (ex: Data Scientist, DevOps, etc.)"
        )
        
        experience = st.slider(
            "⏱️ Années d'expérience",
            min_value=0,
            max_value=15,
            value=5,
            step=1,
            help="Nombre d'années d'expérience professionnelle dans le domaine IT"
        )
        
        niveau_etude = st.selectbox(
            "🎓 Niveau d'étude",
            options=niveaux_etude_disponibles,
            help="Niveau d'études le plus élevé (Bac+2, Bac+3, Master, Ingénieur, Doctorat)"
        )
    
    with col2:
        technologies_disponibles = sorted(model_params['technologies_list'])
        entreprises_disponibles = sorted(model_params['entreprises_list'])
        
        technologie = st.selectbox(
            "💻 Technologie/Compétence",
            options=technologies_disponibles,
            help="Technologie ou compétence principale utilisée"
        )
        
        entreprise = st.selectbox(
            "🏢 Entreprise visée",
            options=entreprises_disponibles,
            help="Entreprise multinationale au Maroc"
        )
    
    st.markdown("---")
    
    # Bouton de prédiction
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🎯 Prédire le Salaire", use_container_width=True, type="primary")
    
    # Prédiction
    if predict_button:
        with st.spinner("Calcul en cours..."):
            salaire, erreur = predict_salary(
                profil=profil,
                experience=experience,
                niveau_etude=niveau_etude,
                technologie=technologie,
                entreprise=entreprise,
                model_params=model_params
            )
        
        if erreur:
            st.error(erreur)
        else:
            st.markdown("---")
            st.success("✅ Prédiction effectuée avec succès!")
            
            # Affichage du résultat
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 30px;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                ">
                    <h2 style="color: white; margin: 0; font-size: 1.5em;">💰 Salaire Estimé</h2>
                    <h1 style="color: white; margin: 10px 0; font-size: 3em; font-weight: bold;">
                        {salaire:,.2f} MAD
                    </h1>
                    <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 1.1em;">par mois</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Récapitulatif
            st.markdown("---")
            st.subheader("📋 Récapitulatif des informations")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                - **Profil:** {profil}
                - **Expérience:** {experience} ans
                - **Niveau d'étude:** {niveau_etude}
                """)
            with col2:
                st.markdown(f"""
                - **Technologie:** {technologie}
                - **Entreprise:** {entreprise}
                """)
            
            # Note sur la prédiction
            st.info(f"""
            ℹ️ **Note:** Cette prédiction est basée sur un modèle entraîné avec 3,000 exemples réels du marché IT marocain 2025.
            Le salaire réel peut varier selon la négociation, les avantages, la localisation (Casablanca, Rabat, etc.), 
            le contexte économique et les compétences spécifiques. Salaires typiques au Maroc: 9,000 → 65,000 MAD/mois.
            """)
    
    # Statistiques supplémentaires
    st.markdown("---")
    st.subheader("📊 Statistiques du modèle - Marché IT Marocain")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Profils IT", len(model_params['profils_list']))
    with col2:
        st.metric("Technologies", len(model_params['technologies_list']))
    with col3:
        st.metric("Entreprises", len(model_params['entreprises_list']))
    with col4:
        st.metric("Niveaux d'étude", len(model_params['niveaux_list']))

else:
    st.warning("""
    ⚠️ **Le modèle n'est pas encore disponible!**
    
    Pour utiliser cette application, vous devez d'abord:
    1. Ouvrir et exécuter le notebook `prediction_salaire_IT_MAROC_2025.ipynb`
    2. Entraîner le modèle en exécutant toutes les cellules (environ 10-15 minutes)
    3. Le fichier `model_salaire_IT_MAROC_2025.pkl` sera généré automatiquement
    4. Relancer cette application Streamlit avec: `streamlit run app_streamlit.py`
    
    📊 **Dataset utilisé:** DATASET_IT_MAROC_2025.csv (3000 exemples, 20 entreprises multinationales)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <p>🇲🇦 Développé pour le marché IT marocain avec ❤️</p>
    <p>Utilise: Streamlit, NumPy, Pandas | Modèle: Régression Linéaire (From Scratch avec Descente de Gradient)</p>
    <p><small>Dataset: 3000 exemples | 20 entreprises multinationales | 10 profils IT | 65+ technologies</small></p>
</div>
""", unsafe_allow_html=True)
