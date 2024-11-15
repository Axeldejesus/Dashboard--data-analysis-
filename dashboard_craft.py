import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from sklearn.cluster import KMeans

# Configuration de la page
st.set_page_config(page_title="Analyse et Prédictions des Crafts", layout="wide")

# Fonction pour charger et prétraiter les données
@st.cache_data
def load_data():
  df = pd.read_csv("craft.csv")
  df = df.replace('nan', np.nan)
  df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
  return df

# Fonction pour encoder les caractéristiques catégorielles
def encode_features(df):
  le = LabelEncoder()
  df_encoded = df.copy()
  categorical_columns = ['baseItem', 'mat1', 'mat2', 'mat3']
  
  for col in categorical_columns:
      if col in df.columns:
          df_encoded[col + '_encoded'] = le.fit_transform(df[col].fillna('missing'))
  
  return df_encoded

# Fonction pour entraîner le modèle prédictif
@st.cache_resource
def train_cost_predictor(df):
  df_encoded = encode_features(df)
  features = ['category', 'tier', 'nMats', 'baseItem_encoded', 'mat1_encoded']
  
  X = df_encoded[features].fillna(-1)
  y = df_encoded['cost'].fillna(df_encoded['cost'].mean())
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  model = RandomForestRegressor(n_estimators=100, random_state=42)  
  model.fit(X_train, y_train)
  
  return model, features

# Chargement des données
try:
  df = load_data()
  
  # Interface utilisateur
  st.title("🔮 Analyse et Prédictions des Crafts")
  
  # Onglets pour différentes sections
  tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Exploration des données", "🤖 Prédictions", "🔍 Analyse avancée", "📈 Tendances", "📚 Guide d'utilisation"])
  
  with tab1:
      st.header("Exploration des données")
      st.write("Visualisez et filtrez les données de crafting.")
      
      # Champ de recherche par nom
      search_term = st.text_input("Rechercher par nom d'item")
      
      # Filtres pour l'exploration des données
      category_filter = st.multiselect("Sélectionnez une catégorie", options=df['category'].unique())
      tier_filter = st.multiselect("Sélectionnez un tier", options=df['tier'].unique())
      
      # Application des filtres
      filtered_df = df.copy()
      
      if search_term:
          filtered_df = filtered_df[filtered_df['baseItem'].str.contains(search_term, case=False, na=False)]
      
      if category_filter:
          filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
      
      if tier_filter:
          filtered_df = filtered_df[filtered_df['tier'].isin(tier_filter)]
      
      # Affichage du tableau de données filtré
      st.dataframe(filtered_df)

  with tab2:
      st.header("Prédictions de coût")
      st.write("Prédisez le coût d'un craft en fonction de ses caractéristiques.")
      
      # Entraînement du modèle
      model, features = train_cost_predictor(df)
      
      # Interface de prédiction
      pred_category = st.selectbox("Catégorie", sorted(df['category'].unique()))
      pred_tier = st.selectbox("Tier", sorted(df['tier'].unique()))
      pred_nmats = st.number_input("Nombre de matériaux", min_value=1, max_value=10)
      
      # Faire une prédiction
      if st.button("Prédire le coût"):
          df_encoded = encode_features(df)
          sample = pd.DataFrame({
              'category': [pred_category],
              'tier': [pred_tier],
              'nMats': [pred_nmats],
              'baseItem_encoded': [0],
              'mat1_encoded': [0]
          })
          
          prediction = model.predict(sample)[0]
          st.success(f"Coût prédit: {prediction:.2f}")
          
          # Importance des caractéristiques
          feature_importance = pd.DataFrame({
              'Feature': features,
              'Importance': model.feature_importances_
          }).sort_values('Importance', ascending=False)
          
          st.write("Importance des caractéristiques dans la prédiction:")
          fig_importance = px.bar(feature_importance, x='Feature', y='Importance')
          st.plotly_chart(fig_importance)

  with tab3:
      st.header("Analyse avancée")
      st.write("Analysez les données de manière plus approfondie.")
      
      # Clustering des items
      X_cluster = df[['cost', 'nMats']].fillna(df[['cost', 'nMats']].mean())
      kmeans = KMeans(n_clusters=3, random_state=42)
      df['cluster'] = kmeans.fit_predict(X_cluster)
      
      # Visualisation des clusters avec explication détaillée
      st.subheader("Clustering des items par coût et matériaux")
      st.markdown("""
      **Comment lire ce graphique:**
      - Chaque point représente un item différent
      - Axe horizontal (x) : Coût de l'item
      - Axe vertical (y) : Nombre de matériaux nécessaires
      - Les couleurs regroupent les items ayant des caractéristiques similaires
      
      **Ce que cela signifie:**
      - Les points proches = items similaires en termes de coût et matériaux
      - Groupes distincts = différentes catégories de complexité/coût
      - Utile pour identifier les items exceptionnels ou standards
      """)
      fig_cluster = px.scatter(df, x='cost', y='nMats', color='cluster', 
                             title="Clustering des items par coût et matériaux")
      st.plotly_chart(fig_cluster)
      
      # Analyse des coûts avec explication détaillée
      st.subheader("Analyse des coûts par nombre de matériaux")
      st.markdown("""
      **Comment lire ce graphique:**
      - Axe horizontal (x) : Nombre de matériaux requis
      - Axe vertical (y) : Coût moyen en pièces d'or
      - Chaque barre représente un niveau de complexité différent
      
      **Ce que cela signifie:**
      - Plus la barre est haute, plus les items sont coûteux
      - Permet de voir si plus de matériaux = coût plus élevé
      - Aide à estimer le coût selon la complexité de fabrication
      """)
      cost_by_nmats = df.groupby('nMats')['cost'].mean().reset_index()
      fig_cost_by_nmats = px.bar(cost_by_nmats, x='nMats', y='cost', 
                                title="Coût moyen par nombre de matériaux")
      st.plotly_chart(fig_cost_by_nmats)

  with tab4:
      st.header("Analyse des tendances")
      st.write("Découvrez les tendances dans les données de crafting.")
      
      # Évolution du coût par tier avec explication détaillée
      st.subheader("Évolution du coût moyen par tier")
      st.markdown("""
      **Comment lire ce graphique:**
      - Axe horizontal (x) : Niveaux de tier (qualité)
      - Axe vertical (y) : Coût moyen en pièces d'or
      - La ligne montre l'évolution des coûts entre les tiers
      
      **Ce que cela signifie:**
      - Pente montante = les tiers supérieurs coûtent plus cher
      - Plateaux = zones où les coûts sont stables
      - Aide à prévoir les coûts selon le tier visé
      """)
      cost_trend = df.groupby('tier')['cost'].mean().reset_index()
      fig_trend = px.line(cost_trend, x='tier', y='cost', 
                         title="Évolution du coût moyen par tier")
      st.plotly_chart(fig_trend)
      
      # Distribution des coûts avec explication détaillée
      st.subheader("Distribution des coûts")
      st.markdown("""
      **Comment lire ce graphique:**
      - Axe horizontal (x) : Plages de coût
      - Axe vertical (y) : Nombre d'items dans chaque plage
      - Les pics montrent les coûts les plus fréquents
      
      **Ce que cela signifie:**
      - Pics élevés = coûts très communs
      - Distribution étalée = grande variété de coûts
      - Aide à identifier les gammes de prix normales
      """)
      fig_cost_distribution = px.histogram(df, x='cost', nbins=50, 
                                         title="Distribution des coûts")
      st.plotly_chart(fig_cost_distribution)
      
      # Complexité par catégorie avec explication détaillée
      st.subheader("Complexité par catégorie")
      st.markdown("""
      **Comment lire ce graphique:**
      - Axe horizontal (x) : Catégories d'items
      - Axe vertical (y) : Nombre moyen de matériaux
      - Chaque barre représente une catégorie différente
      
      **Ce que cela signifie:**
      - Barres hautes = catégories complexes à fabriquer
      - Barres basses = catégories simples
      - Aide à choisir des projets selon leur complexité
      """)
      complexity_analysis = df.groupby('category')['nMats'].mean().reset_index()
      fig_complexity = px.bar(complexity_analysis, x='category', y='nMats', 
                            title="Complexité moyenne par catégorie")
      st.plotly_chart(fig_complexity)

  with tab5:
      st.title("📚 Guide d'utilisation du Dashboard")
      
      st.markdown("""
      ### 🎯 Objectif du Dashboard
      Ce dashboard vous permet d'analyser en détail les items de crafting. Il est conçu pour être accessible à tous, que vous soyez débutant ou expert.
      
      ### 📊 Exploration des données
      - **Fonctionnalité** : Permet de filtrer et de visualiser les données de crafting.
      - **Utilisation** : Recherchez par nom d'item et appliquez des filtres par catégorie et tier pour explorer les données spécifiques.
      
      ### 🔮 Prédictions
      - **Fonctionnalité** : Utilise un modèle de régression Random Forest pour prédire le coût d'un item.
      - **Utilisation** : Sélectionnez la catégorie, le tier, et le nombre de matériaux pour obtenir une estimation du coût.
      - **Graphique d'importance** : Montre quels facteurs influencent le plus le coût.
      
      ### 🔍 Analyse Avancée
      - **Fonctionnalité** : Comprend des analyses comme le clustering des items par coût et nombre de matériaux.
      - **Utilisation** : Identifiez des groupes d'items similaires pour comprendre les patterns dans les données.
      
      ### 📈 Analyse des Tendances
      - **Fonctionnalité** : Visualise les tendances des coûts moyens par tier et la distribution des coûts.
      - **Utilisation** : Comprenez comment les coûts évoluent avec le tier et explorez la répartition des coûts.
      
      ### 💡 Conseils d'utilisation
      - Commencez par l'Exploration des données pour avoir une idée générale.
      - Utilisez les Prédictions pour estimer le coût de nouvelles combinaisons.
      - Explorez l'Analyse Avancée pour découvrir des patterns intéressants.
      - Utilisez les filtres dans l'Exploration des données pour des recherches spécifiques.
      """)
      
      # Section FAQ
      st.subheader("❓ Questions Fréquentes")
      
      with st.expander("Comment interpréter le graphique de clustering ?"):
          st.write("Le clustering regroupe les items similaires. Chaque couleur représente un groupe différent d'items ayant des caractéristiques similaires en termes de coût et de matériaux.")
      
      with st.expander("Comment sont calculées les prédictions ?"):
          st.write("Les prédictions utilisent un modèle de Random Forest qui apprend à partir des données existantes. Il prend en compte la catégorie, le tier, et le nombre de matériaux pour estimer le coût probable d'un item.")

except Exception as e:
  st.error(f"Erreur lors de l'analyse: {str(e)}")
  st.info("Vérifiez que le fichier 'craft.csv' est présent et correctement formaté.")