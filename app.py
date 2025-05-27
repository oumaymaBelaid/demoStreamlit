import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

#  Config de la page
st.set_page_config(page_title="Segmentation Clients", layout="wide")
st.title(" Segmentation des clients du centre commercial")

#  Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

#  Aperçu rapide
st.subheader("Aperçu des données")
st.write(df.head())

#  Choix utilisateur
st.sidebar.title("Paramètres")
n_clusters = st.sidebar.slider("Nombre de clusters", 2, 8, 3)
selected_features = st.sidebar.multiselect(
    "Colonnes à utiliser", 
    ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], 
    default=['Age', 'Annual Income (k$)']
)

#  Préparation des données
df_copy = df.copy()
df_copy['Gender'] = df_copy['Gender'].map({'Male': 0, 'Female': 1})

if selected_features:
    scaler = StandardScaler()
    X = scaler.fit_transform(df_copy[selected_features])

    #  K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    df_copy['Cluster'] = kmeans.fit_predict(X)

    #  Affichage des clusters
    st.subheader("Visualisation des clusters")
    if len(selected_features) >= 2:
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=selected_features[0], 
            y=selected_features[1], 
            hue='Cluster', 
            data=df_copy, 
            palette='Set2'
        )
        st.pyplot(fig)
    else:
        st.write("Veuillez sélectionner au moins deux colonnes.")

    #  Moyennes par cluster
    st.subheader("Moyennes par cluster")
    st.write(df_copy.groupby('Cluster')[selected_features].mean())

   #  Stratégie marketing détaillée selon les clusters
st.subheader("Stratégie marketing par cluster")

# Description manuelle des clusters 
cluster_strategies = {
    0: {
        "label": "Riches économes",
        "description": "Revenu élevé, dépenses faibles",
        "marketing": [
            "💼 Promouvoir des placements ou services haut de gamme mais utiles",
            "📈 Mettre en avant la qualité/prix plutôt que le luxe",
        ]
    },
    1: {
        "label": "Dépensiers moyens",
        "description": "Âge moyen, dépenses élevées",
        "marketing": [
            "🎯 Proposer des offres personnalisées pour les fidéliser",
            "🎁 Cartes de fidélité, offres groupées et services réguliers",
        ]
    },
    2: {
        "label": "Jeunes dépensiers",
        "description": "Jeunes, dépenses élevées",
        "marketing": [
            "📱 Campagnes sur TikTok, Instagram, etc.",
            "🔥 Offres flash, expériences fun et tendance",
        ]
    },
    3: {
        "label": "Faible pouvoir d'achat",
        "description": "Revenu faible, dépenses faibles",
        "marketing": [
            "💸 Promos et bons de réduction",
            "🛍️ Offres économiques et packs d'entrée de gamme",
        ]
    },
    4: {
        "label": "Cibles premium",
        "description": "Revenu élevé, dépenses élevées",
        "marketing": [
            "🌟 Produits de luxe, expériences VIP",
            "🔒 Services personnalisés et exclusifs",
        ]
    },
}

# Affichage
for cluster_id in sorted(df_copy['Cluster'].unique()):
    data = cluster_strategies.get(cluster_id, {})
    st.markdown(f"###  Cluster {cluster_id} : {data.get('label', 'Inconnu')}")
    st.write(f"**Profil :** {data.get('description', 'Non défini')}")
    st.write("**Stratégies proposées :**")
    for strategy in data.get("marketing", []):
        st.write("-", strategy)
