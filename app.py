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

    #  Stratégie marketing simple
    st.subheader("Stratégie marketing proposée")

    for cluster in range(n_clusters):
        st.markdown(f"###  Cluster {cluster}")
        cluster_data = df_copy[df_copy['Cluster'] == cluster][selected_features].mean()

        strategies = []

        # Exemple simple de règles
        if 'Annual Income (k$)' in selected_features:
            if cluster_data['Annual Income (k$)'] > 70:
                strategies.append("🛍️ Offrir des produits de luxe")
            elif cluster_data['Annual Income (k$)'] < 40:
                strategies.append("💸 Proposer des réductions")

        if 'Age' in selected_features:
            if cluster_data['Age'] < 25:
                strategies.append("📱 Cibler via les réseaux sociaux")
            elif cluster_data['Age'] > 50:
                strategies.append("📧 Utiliser l'emailing ou les appels")

        if 'Spending Score (1-100)' in selected_features:
            if cluster_data['Spending Score (1-100)'] > 70:
                strategies.append("🎁 Fidéliser avec des avantages exclusifs")
            elif cluster_data['Spending Score (1-100)'] < 40:
                strategies.append("📣 Relancer avec des offres personnalisées")

        for s in strategies:
            st.write("-", s)

        if not strategies:
            st.write("Aucune stratégie spécifique")
