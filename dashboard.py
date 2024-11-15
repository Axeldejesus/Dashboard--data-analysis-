import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

# Configuration de la page 
st.set_page_config(layout="wide", page_title="Mysree Forest Analytics Pro")

# Titre principal 
st.markdown("""
  <h1 style='text-align: center; color: #4CAF50; animation: fadeIn 2s;'>
      🌳 Mysree Forest Advanced Analytics
  </h1>
  """, unsafe_allow_html=True)

# Chargement et préparation des données
@st.cache_data
def load_data():
  df = pd.read_csv('mysree_forest_outdoors_complete.csv')
  # Expression régulière
  df['Rarity_Value'] = df['Item_Rarity'].str.extract(r'(\d+)').astype(int)
  return df

df = load_data()

# Création des onglets
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Vue d'ensemble", "🔮 Prédictions", "🎯 Analyse Avancée", "📝 Données", "❓ Explications"])

with tab1:
  # Métriques améliorées
  col1, col2, col3, col4 = st.columns(4)
  with col1:
      st.metric("Total Items", len(df), 
               delta=f"{len(df[df['Rarity_Value']>5])} items rares")
  with col2:
      st.metric("Rareté Moyenne", 
               round(df['Rarity_Value'].mean(), 2),
               delta=f"{round(df['Rarity_Value'].std(), 2)} σ")
  with col3:
      st.metric("Item le Plus Rare", 
               df.loc[df['Rarity_Value'].idxmax(), 'Name'],
               delta=f"IR{df['Rarity_Value'].max()}")
  with col4:
      st.metric("% Drops Monstres", 
               f"{round(len(df[df['Drop_Type']=='Monster Drop'])/len(df)*100, 1)}%")

  # Visualisations améliorées
  col1, col2 = st.columns(2)
  
  with col1:
      st.subheader("📊 Distribution de la Rareté")
      fig_rarity = px.histogram(df, 
                              x='Rarity_Value',
                              title='Distribution des Valeurs de Rareté',
                              color_discrete_sequence=['#2ecc71'])
      fig_rarity.add_vline(x=df['Rarity_Value'].mean(), 
                         line_dash="dash", 
                         annotation_text="Moyenne")
      fig_rarity.update_layout(template="plotly_dark")
      st.plotly_chart(fig_rarity, use_container_width=True)

  with col2:
      st.subheader("🔄 Hiérarchie des Items")
      fig_sunburst = px.sunburst(df, 
                               path=['Drop_Type', 'Source', 'Item_Rarity'],
                               values='Rarity_Value',
                               title='Hiérarchie des Items')
      fig_sunburst.update_layout(template="plotly_dark")
      st.plotly_chart(fig_sunburst, use_container_width=True)

with tab2:
  st.subheader("🔮 Prédiction de Rareté")
  
  # Préparation du modèle
  X = pd.get_dummies(df[['Drop_Type', 'Source']])
  y = df['Rarity_Value']
  
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X, y)
  
  # Interface de prédiction
  col1, col2 = st.columns(2)
  with col1:
      pred_drop_type = st.selectbox("Type de Drop", df['Drop_Type'].unique())
      pred_source = st.selectbox("Source", df['Source'].unique())
  
  pred_data = pd.DataFrame([[pred_drop_type, pred_source]], 
                          columns=['Drop_Type', 'Source'])
  pred_data = pd.get_dummies(pred_data)
  pred_data = pred_data.reindex(columns=X.columns, fill_value=0)
  
  prediction = model.predict(pred_data)[0]
  
  with col2:
      st.metric("Rareté Prédite", f"IR{round(prediction)}")
      
  # Importance des caractéristiques
  importance_df = pd.DataFrame({
      'Feature': X.columns,
      'Importance': model.feature_importances_
  }).sort_values('Importance', ascending=False).head(10)
  
  st.plotly_chart(px.bar(importance_df, 
                        x='Importance', 
                        y='Feature',
                        title='Importance des Caractéristiques'),
                  use_container_width=True)

with tab3:
  st.subheader("🎯 Analyse Avancée")
  
  # Clustering
  X_cluster = StandardScaler().fit_transform(df[['Rarity_Value']])
  kmeans = KMeans(n_clusters=3, random_state=42)
  df['Cluster'] = kmeans.fit_predict(X_cluster)
  
  col1, col2 = st.columns(2)
  
  with col1:
      fig_cluster = px.scatter(df, 
                             x='Rarity_Value',
                             y='Level',
                             color='Cluster',
                             hover_data=['Name'],
                             title='Clustering des Items')
      st.plotly_chart(fig_cluster, use_container_width=True)
  
  with col2:
      # Top items par cluster
      st.subheader("Items Représentatifs par Cluster")
      for cluster in range(3):
          st.write(f"Cluster {cluster}:")
          st.write(df[df['Cluster'] == cluster].nlargest(3, 'Rarity_Value')[['Name', 'Rarity_Value']])

with tab4:
  # Filtres améliorés
  col1, col2, col3 = st.columns(3)
  with col1:
      selected_rarity = st.multiselect("Rareté", 
                                     sorted(df['Item_Rarity'].unique()))
  with col2:
      selected_drop_type = st.multiselect("Type de Drop", 
                                        df['Drop_Type'].unique())
  with col3:
      selected_source = st.multiselect("Source", 
                                     df['Source'].unique())
  
  # Filtrage
  filtered_df = df.copy()
  if selected_rarity:
      filtered_df = filtered_df[filtered_df['Item_Rarity'].isin(selected_rarity)]
  if selected_drop_type:
      filtered_df = filtered_df[filtered_df['Drop_Type'].isin(selected_drop_type)]
  if selected_source:
      filtered_df = filtered_df[filtered_df['Source'].isin(selected_source)]
  
  # Affichage avec style
  st.dataframe(filtered_df.style.background_gradient(subset=['Rarity_Value']))

# Sidebar avec statistiques supplémentaires
st.sidebar.title("📊 Statistiques Globales")
st.sidebar.metric("Items les plus communs", 
               df['Name'].value_counts().index[0])
st.sidebar.metric("Source la plus fréquente", 
               df['Source'].value_counts().index[0])
st.sidebar.metric("Rareté Médiane", 
               f"IR{df['Rarity_Value'].median()}")

with tab5:
  st.title("📚 Guide d'utilisation du Dashboard")
  
  st.markdown("""
  ### 🎯 Objectif du Dashboard
  Ce dashboard vous permet d'analyser en détail les items de Mysree Forest. Il est conçu pour être accessible à tous, que vous soyez débutant ou expert.
  
  ### 📊 Vue d'ensemble
  Dans cet onglet, vous trouverez :
  - **Total Items** : Nombre total d'items dans la base de données, avec le nombre d'items rares (rareté > 5)
  - **Rareté Moyenne** : La valeur moyenne de rareté des items, avec l'écart-type (σ) qui indique la dispersion des valeurs
  - **Item le Plus Rare** : L'item ayant la plus haute valeur de rareté
  - **% Drops Monstres** : Pourcentage d'items provenant des monstres
  
  Les graphiques montrent :
  - La **Distribution de Rareté** : Comment les valeurs de rareté sont réparties (plus la barre est haute, plus il y a d'items avec cette rareté)
  - La **Hiérarchie des Items** : Un graphique interactif montrant les relations entre Type de Drop, Source et Rareté
  
  ### 🔮 Prédictions
  Cette section utilise l'intelligence artificielle pour :
  - Prédire la rareté probable d'un item basé sur son type de drop et sa source
  - Montrer quels facteurs influencent le plus la rareté (graphique d'importance)
  
  Comment l'utiliser :
  1. Sélectionnez un type de drop
  2. Choisissez une source
  3. Le système prédit automatiquement la rareté probable
  
  ### 🎯 Analyse Avancée
  Cette section utilise le clustering (regroupement automatique) pour :
  - Identifier des groupes d'items similaires
  - Les clusters sont des groupes d'items ayant des caractéristiques similaires :
      - **Cluster 0** : Items communs
      - **Cluster 1** : Items de rareté moyenne
      - **Cluster 2** : Items rares
  
  Le graphique montre :
  - Chaque point représente un item
  - Les couleurs indiquent les différents clusters
  - Survolez les points pour voir les détails
  
  ### 📝 Données
  Cette section permet d'explorer les données brutes :
  - Utilisez les filtres pour affiner votre recherche
  - Les colonnes peuvent être triées
  - La coloration indique la rareté (plus c'est foncé, plus c'est rare)
  
  ### 📊 Statistiques Globales (Barre latérale)
  La barre latérale montre :
  - L'item le plus commun dans la base de données
  - La source la plus fréquente d'items
  - La valeur médiane de rareté (IR = Indice de Rareté)
  
  ### 💡 Conseils d'utilisation
  - Commencez par la Vue d'ensemble pour avoir une idée générale
  - Utilisez les Prédictions pour estimer la rareté de nouvelles combinaisons
  - Explorez l'Analyse Avancée pour découvrir des patterns intéressants
  - Utilisez les filtres dans Données pour des recherches spécifiques
  """)
  
  # Section FAQ
  st.subheader("❓ Questions Fréquentes")
  
  with st.expander("Que signifie σ (sigma) dans les métriques ?"):
      st.write("σ est l'écart-type, qui mesure la dispersion des valeurs autour de la moyenne. Plus il est grand, plus les valeurs sont dispersées.")
  
  with st.expander("Comment interpréter le graphique de clustering ?"):
      st.write("Le clustering regroupe les items similaires. Chaque couleur représente un groupe différent d'items ayant des caractéristiques similaires en termes de rareté et de niveau.")
  
  with st.expander("Comment sont calculées les prédictions ?"):
      st.write("Les prédictions utilisent un modèle de Random Forest qui apprend à partir des données existantes. Il prend en compte le type de drop et la source pour estimer la rareté probable d'un item.")
