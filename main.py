# Importations

# Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
# Fonctions implementées
from utils import *


# Indicateurs

if "selected_features" not in st.session_state:
            st.session_state.selected_features = [
                'Delai_Prescription_Facturation',
                'Beneficiaires_Mensuels',
                'Depenses_Mensuelles',
                'Quantite_Acte',
                'Montant de la dépense - Prestation seule',
                'Proportion_Jeunes',
                'Age_Superieur_18',
                'Moyenne_Age_Etablissement',
                'Distance_Beneficiaire_Etablissement_km',
                'Nombre_Prescripteurs',
                'Pourcentage_ORL',
                'Pourcentage_Autres_Specialites',
                'Frequence_Distance_Suspecte',
                'Distance_Prescripteur_Etablissement_km'
            ]

# Interface principale
def main():

    # Titres
    st.set_page_config(page_title="Application de Détection Avancée", layout="wide",initial_sidebar_state="expanded")
    st.header("🔍 Analyse des Anomalies dans les Données d'Audioprothèses")
    st.markdown("Veuillez téléverser deux fichiers CSV : votre jeu de données principal et le fichier de correspondance des régions.")

    st.subheader("Téléversement du Jeu de Données Principal")

    #Depot de données
    file1 = st.file_uploader("Choisir le premier fichier CSV", type="csv", key="file1")

    #Données de mapping (regions)
    file2 = r"points-extremes-des-departements-metropolitains-de-france.csv"

    if file1 is not None and file2 is not None:
        df1 = pd.read_csv(file1)
        print(df1.columns)
        df2 = pd.read_csv(file2)

        df1 = process_fraud_data(df1)
        df1 = process_location_data(df1, df2)

        df1.rename(columns={
            'Délai prescription-facturation': 'Delai_Prescription_Facturation',
            'Bénéficiaires par mois': 'Beneficiaires_Mensuels',
            'Dépenses par mois': 'Depenses_Mensuelles',
            'Quantité d\'acte - Prestation seule (pas presta. de réf.)': 'Quantite_Acte' , 
            'Proportion jeunes': 'Proportion_Jeunes',
            'Age supérieur à 18': 'Age_Superieur_18',
            'Moyenne âge par établissement': 'Moyenne_Age_Etablissement',
            'Distance benef ex (km)': 'Distance_Beneficiaire_Etablissement_km',
            'Nombre de prescripteurs': 'Nombre_Prescripteurs',
            'Pourcentage ORL': 'Pourcentage_ORL',
            'Pourcentage autres': 'Pourcentage_Autres_Specialites',
            'Frequence_dist_suspecte': 'Frequence_Distance_Suspecte',
            'Distance pr etab (km)': 'Distance_Prescripteur_Etablissement_km'
        }, inplace=True)


        tab1, tab2, tab3 = st.tabs(["Aperçu des Données", "Analyse Avancée", "Visualisations"])

        with tab1:
            st.subheader("📌 Aperçu du Jeu de Données")
            st.write(df1.head())
            
            st.subheader("📊 Statistiques Descriptives")
            st.write(df1.describe())

        with tab2:
            st.header("🚨 Tableau de Bord d'Analyse Avancée")
            st.markdown("---")

            st.subheader("🎛️ Sélection des Variables")

            selected_features = st.multiselect(
                "🔧 Sélectionnez les variables à analyser :",
                options=st.session_state.selected_features,
                default=st.session_state.selected_features,
                key="feature_selector"
            )

            # Mettre à jour `selected_features` dans `st.session_state` sans recharger tout
            if selected_features != st.session_state.selected_features:
                st.session_state.selected_features = selected_features

            st.success(f"✅ Variables sélectionnées avec succés")

            if st.button("🚀 Lancer l'Analyse"):
                with st.spinner("🔎 Analyse en cours..."):
                    st.markdown("---")
                    st.header("📈 Résultats de l'Analyse Avancée")

                    df1[st.session_state.selected_features] = df1[st.session_state.selected_features].fillna(df1[st.session_state.selected_features].mean())

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df1[st.session_state.selected_features])

                    model = IsolationForest(contamination=0.01, random_state=42)
                    df1['Anomalie'] = model.fit_predict(X_scaled)
                    df1['Score_Anomalie'] = model.decision_function(X_scaled)

                    df1['Anomalie'] = df1['Anomalie'].map({1: 0, -1: 1})

                    st.session_state.anomalies = df1[df1['Anomalie'] == 1]
                    st.session_state.normales = df1[df1['Anomalie'] == 0]

                    st.subheader("📊 Résumé de l'Analyse")
                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric("💼 Total des Transactions", len(df1))
                    col2.metric("⚠️ Anomalies Potentielles", df1['Anomalie'].sum())

                    pourcentage_anomalies = int((df1['Anomalie'].sum() / len(df1)) * 100) + 1
                    col3.metric("📉 % Anomalies", f"{pourcentage_anomalies} %")

                    perte_potentielle = df1[df1['Anomalie'] == 1]['Montant de la dépense - Prestation seule'].sum()
                    col4.metric("💰 Estimation des Pertes Potentielles", f"€ {perte_potentielle:,.2f}")


                    st.subheader("🔍 Caractéristiques des Anomalies")
                    anomalies = df1[df1['Anomalie'] == 1]
                    normales = df1[df1['Anomalie'] == 0]

                    anomalies = anomalies.drop_duplicates()
                    normales = normales.drop_duplicates()

                    st.markdown("**📊 Comparaison des Moyennes**")
                    compare_df = pd.DataFrame({
                        'Normal': normales[st.session_state.selected_features].mean(),
                        'Anomalie': anomalies[st.session_state.selected_features].mean()
                    }).style.format("{:.2f}")
                    st.dataframe(compare_df)

                    st.subheader("🔥 Anomalies les Plus Significatives")
                    anomalies_sorted = anomalies.sort_values(by='Score_Anomalie', ascending=True)
                    st.dataframe(anomalies_sorted.head(10)[['N° PS exécutant Statistique', 'Score_Anomalie'] + st.session_state.selected_features])


        with tab3:
            if 'anomalies' in st.session_state and 'normales' in st.session_state:
                anomalies = st.session_state.anomalies
                normales = st.session_state.normales

                # Création des sous-tabs
                sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Indicateurs et comparaisons", "Analyse de régions", "Analyse d'établissements"])

                with sub_tab1:  # Indicateurs et comparaisons
                    st.write("Comparaison du Délai prescription-facturation")
                    mean_anomalies = anomalies['Delai_Prescription_Facturation'].mean()
                    mean_normales = normales['Delai_Prescription_Facturation'].mean()
                    categories = ['Anomalies', 'Normales']
                    means = [mean_anomalies, mean_normales]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#1f77b4', '#ff7f0e']
                    bars = ax.barh(categories, means, color=colors, edgecolor='black', linewidth=1.2)

                    for bar in bars:
                        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                                f'{bar.get_width():.2f}', va='center', ha='left', fontsize=12, color='black')

                    ax.set_xlabel("Moyenne du Délai prescription-facturation", fontsize=14)
                    ax.set_ylabel("Catégories", fontsize=14)
                    ax.set_title("Comparaison des Moyennes du Délai prescription-facturation", fontsize=16, fontweight='bold')
                    ax.set_facecolor('whitesmoke')
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.5)
                        spine.set_color('gray')

                    st.pyplot(fig)
                    
                    
                    st.write("Distribution des Distances Beneficiare - etablisssement dans les Anomalies")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(anomalies['Distance_Prescripteur_Etablissement_km'], bins=30, kde=True, color='orange', ax=ax)
                    ax.set_title("Distribution des Distances (Anomalies)", fontsize=16)
                    ax.set_xlabel('Distance_Prescripteur_Etablissement_km', fontsize=12)
                    ax.set_ylabel("Fréquence", fontsize=12)

                    st.pyplot(fig)

                                        
                    st.write(anomalies.columns)   

                    st.write("Distribution des Distances Prescripteur - etablissement dans les Anomalies")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(anomalies['Distance_Beneficiaire_Etablissement_km'], bins=30, kde=True, color='orange', ax=ax)
                    ax.set_title("Distribution des Distances (Anomalies)", fontsize=16)
                    ax.set_xlabel("Distance benef ex (km)", fontsize=12)
                    ax.set_ylabel("Fréquence", fontsize=12)

                    st.pyplot(fig)


                    st.write("Top 3 Établissements par Fréquence de Distances Suspectes")

                    # Top 3 dans les anomalies
                    top3_anomalies = (
                        anomalies.groupby('N° PS exécutant Statistique')['Frequence_Distance_Suspecte']
                        .mean()
                        .nlargest(3)
                        .reset_index()
                    )
                    top3_anomalies['Catégorie'] = 'Anomalies'

                    # Top 3 dans les normales (en s'assurant qu'ils sont différents de ceux dans anomalies)
                    top3_normales = (
                        normales[~normales['N° PS exécutant Statistique'].isin(top3_anomalies['N° PS exécutant Statistique'])]
                        .groupby('N° PS exécutant Statistique')['Frequence_Distance_Suspecte']
                        .mean()
                        .nlargest(3)
                        .reset_index()
                    )
                    top3_normales['Catégorie'] = 'Normales'

                    # Combiner les deux DataFrames
                    top3 = pd.concat([top3_anomalies, top3_normales])

                    # Visualisation
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.barplot(
                        data=top3,
                        y='N° PS exécutant Statistique',
                        x='Frequence_Distance_Suspecte',
                        hue='Catégorie',
                        palette={'Anomalies': '#ff7f0e', 'Normales': '#1f77b4'},
                        edgecolor='black'
                    )
                    
                    ax.set_xlabel("Fréquence de Distances Suspectes", fontsize=14)
                    ax.set_ylabel("Établissements", fontsize=14)
                    ax.set_title("Top 3 Établissements par Fréquence de Distances Suspectes", fontsize=16, fontweight='bold')
                    ax.set_facecolor('whitesmoke')
                    ax.legend(title='Catégorie')

                    for spine in ax.spines.values():
                        spine.set_linewidth(1.5)
                        spine.set_color('gray')

                    st.pyplot(fig)


                with sub_tab2:  # Analyse de régions
                    st.write("Répartition des Départements dans les Anomalies")
                    top_departments = anomalies['Département d\'exercice du PS exécutant'].value_counts().head(5)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.pie(top_departments, labels=top_departments.index, autopct='%1.1f%%', startangle=90, wedgeprops={'width': 0.3})
                    centre_circle = plt.Circle((0, 0), 0.50, color='white', fc='white', lw=0)
                    ax.add_artist(centre_circle)
                    ax.set_title("Répartition des Départements (Anomalies)", fontsize=14)

                    st.pyplot(fig)

                    # Mapping des départements vers les régions
                    departement_to_region = {
                        '01': 'Auvergne-Rhône-Alpes', '02': 'Hauts-de-France', '03': 'Auvergne-Rhône-Alpes', '04': 'Provence-Alpes-Côte d\'Azur',
                        '05': 'Provence-Alpes-Côte d\'Azur', '06': 'Provence-Alpes-Côte d\'Azur', '07': 'Auvergne-Rhône-Alpes', '08': 'Grand Est',
                        '09': 'Occitanie', '10': 'Grand Est', '11': 'Occitanie', '12': 'Occitanie', '13': 'Provence-Alpes-Côte d\'Azur', '14': 'Normandie',
                        '15': 'Auvergne-Rhône-Alpes', '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine', '18': 'Centre-Val de Loire', '19': 'Nouvelle-Aquitaine',
                        '20': 'Corse', '21': 'Bourgogne-Franche-Comté', '22': 'Bretagne', '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine', '25': 'Bourgogne-Franche-Comté',
                        '26': 'Auvergne-Rhône-Alpes', '27': 'Normandie', '28': 'Centre-Val de Loire', '29': 'Bretagne', '30': 'Occitanie', '31': 'Occitanie',
                        '32': 'Occitanie', '33': 'Nouvelle-Aquitaine', '34': 'Occitanie', '35': 'Bretagne', '36': 'Centre-Val de Loire', '37': 'Centre-Val de Loire',
                        '38': 'Auvergne-Rhône-Alpes', '39': 'Bourgogne-Franche-Comté', '40': 'Nouvelle-Aquitaine', '41': 'Centre-Val de Loire', '42': 'Auvergne-Rhône-Alpes',
                        '43': 'Auvergne-Rhône-Alpes', '44': 'Pays de la Loire', '45': 'Centre-Val de Loire', '46': 'Occitanie', '47': 'Nouvelle-Aquitaine',
                        '48': 'Occitanie', '49': 'Pays de la Loire', '50': 'Normandie', '51': 'Grand Est', '52': 'Grand Est', '53': 'Pays de la Loire', '54': 'Grand Est',
                        '55': 'Grand Est', '56': 'Bretagne', '57': 'Grand Est', '58': 'Bourgogne-Franche-Comté', '59': 'Hauts-de-France', '60': 'Hauts-de-France',
                        '61': 'Normandie', '62': 'Hauts-de-France', '63': 'Auvergne-Rhône-Alpes', '64': 'Nouvelle-Aquitaine', '65': 'Occitanie', '66': 'Occitanie',
                        '67': 'Grand Est', '68': 'Grand Est', '69': 'Auvergne-Rhône-Alpes', '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté', '72': 'Pays de la Loire',
                        '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes', '75': 'Île-de-France', '76': 'Normandie', '77': 'Île-de-France', '78': 'Île-de-France',
                        '79': 'Nouvelle-Aquitaine', '80': 'Hauts-de-France', '81': 'Occitanie', '82': 'Occitanie', '83': 'Provence-Alpes-Côte d\'Azur', '84': 'Provence-Alpes-Côte d\'Azur',
                        '85': 'Pays de la Loire', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine', '88': 'Grand Est', '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',
                        '91': 'Île-de-France', '92': 'Île-de-France', '93': 'Île-de-France', '94': 'Île-de-France', '95': 'Île-de-France', '96': 'Outre-mer', '97': 'Outre-mer',
                        '98': 'Outre-mer', '99': 'Outre-mer'
                    }

                    # Comptage des départements les plus fréquents dans anomalies
                    top_departments = anomalies['Département d\'exercice du PS exécutant'].value_counts().head(10)

                    # Mappage des départements vers les régions
                    top_departments_regions = top_departments.index.astype(str).map(departement_to_region)

                    # Comptage des régions les plus fréquentes dans anomalies (après mappage)
                    region_counts = top_departments_regions.value_counts()

                    # Mappage des départements aux régions
                    department_to_region = top_departments.index.to_series().map(departement_to_region)

                    # Calcul de la somme des pourcentages des départements dans chaque région
                    region_percentage = top_departments.groupby(top_departments_regions).sum() / top_departments.sum() * 100

                    # barh chart pour les régions
                    plt.figure(figsize=(10, 6))

                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                    # graphique en barh
                    bars = region_percentage.plot(kind='barh', color=colors, edgecolor='black', linewidth=1.2)

                    # labels sur les barres
                    for index, value in enumerate(region_percentage):
                        # Ajouter l'étiquette dans la barre, avec la somme des pourcentages des départements
                        plt.text(value + 0.1, index, f'{value:.1f}%', va='center', ha='left', fontsize=12, color='black')

                    # titres et des labels
                    plt.xlabel("Somme des pourcentages des départements", fontsize=14)
                    plt.ylabel("Régions", fontsize=14)
                    plt.title("Répartition des régions les plus fréquentes dans les anomalies", fontsize=16, fontweight='bold')
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)

                    plt.gca().set_facecolor('whitesmoke')
                    for spine in plt.gca().spines.values():
                        spine.set_linewidth(1.5)
                        spine.set_color('gray')

                    st.pyplot()

                
                with sub_tab3:
                    etablissement_anomalie_id = anomalies['N° PS exécutant Statistique'].value_counts().idxmax()
                    etablissement_normal_id = normales['N° PS exécutant Statistique'].value_counts().idxmax()

                    # Filtrer les données pour ces deux établissements
                    anomalies_etablissement = anomalies[anomalies['N° PS exécutant Statistique'] == etablissement_anomalie_id]
                    normales_etablissement = normales[normales['N° PS exécutant Statistique'] == etablissement_normal_id]

                    # Calculer les remboursements totaux par mois pour anomalies et normales
                    anomalies_mois = anomalies_etablissement.groupby(['Année de remboursement', 'Mois de remboursement'])['Depenses_Mensuelles'].sum().reset_index()
                    normales_mois = normales_etablissement.groupby(['Année de remboursement', 'Mois de remboursement'])['Depenses_Mensuelles'].sum().reset_index()

                    # Tracer le graphique
                    fig, ax = plt.subplots(figsize=(12, 6))  # Créer un objet figure et axe
                    sns.lineplot(data=anomalies_mois, x='Mois de remboursement', y='Depenses_Mensuelles', label=f'Anomalies ({etablissement_anomalie_id})', color='red', marker='o', ax=ax)
                    sns.lineplot(data=normales_mois, x='Mois de remboursement', y='Depenses_Mensuelles', label=f'Normales ({etablissement_normal_id})', color='blue', marker='o', ax=ax)

                    # Ajouter des détails au graphique
                    ax.set_title(f'Depenses_Mensuelles : {etablissement_anomalie_id} (Anomalies) vs {etablissement_normal_id} (Normales)')
                    ax.set_xlabel('Mois')
                    ax.set_ylabel('Dépenses')
                    ax.legend(title='Catégorie', loc='upper left')
                    ax.set_xticklabels(ax.get_xticks(), rotation=45)
                    ax.grid(True)

                    st.pyplot(fig)

                    # Affichage des établissements sélectionnés pour confirmation
                    st.write(f"Établissement le plus fréquent dans les anomalies : {etablissement_anomalie_id}")
                    st.write(f"Établissement le plus fréquent dans les normales : {etablissement_normal_id}")

if __name__ == "__main__":
    main()
