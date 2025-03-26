# Importations

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import shap

def process_fraud_data(df):
    df = df[(df['Quantité d\'acte - Prestation seule (pas presta. de réf.)'] > 0) &
            (df['Montant de la dépense - Prestation seule'] > 0)]
    
    df['Délai prescription-facturation'] = (
        (df['Année de remboursement'] - df['Année de prescription']) * 12 +
        (df['Mois de remboursement'] - df['Mois de prescription'])
    )
    
    remboursements_par_mois = df.groupby(
        ['N° PS exécutant Statistique', 'Année de remboursement', 'Mois de remboursement']
    )['Nombre de bénéficiaires'].sum().reset_index(name='Bénéficiaires par mois')
    df = df.merge(remboursements_par_mois, on=['N° PS exécutant Statistique', 'Année de remboursement', 'Mois de remboursement'])
    
    depenses_par_mois = df.groupby(
        ['N° PS exécutant Statistique', 'Année de remboursement', 'Mois de remboursement']
    )['Montant de la dépense - Prestation seule'].sum().reset_index(name='Dépenses par mois')
    df = df.merge(depenses_par_mois, on=['N° PS exécutant Statistique', 'Année de remboursement', 'Mois de remboursement'])
    
    proportion_jeunes = df.groupby('N° PS exécutant Statistique')['Age du bénéficiaire'].apply(lambda x: (x < 18).mean())
    df = df.merge(proportion_jeunes.rename('Proportion jeunes'), on='N° PS exécutant Statistique')
    
    df['Age supérieur à 18'] = (df['Age du bénéficiaire'] > 18).astype(int)
    df = df[df['Délai prescription-facturation'] <= 8]
    
    prescripteurs_par_etablissement = df.groupby('N° PS exécutant Statistique')['N° PS prescripteur Statistique'].nunique().reset_index()
    prescripteurs_par_etablissement.columns = ['N° PS exécutant Statistique', 'Nombre de prescripteurs']
    df = df.merge(prescripteurs_par_etablissement, on='N° PS exécutant Statistique', how='left')
    
    prescripteurs_orl = df[df['Libellé spécialité/nat. activité du PS prescripteur'] == 'OTO RHINO-LARYNGOLOGIE'] \
        .groupby('N° PS exécutant Statistique')['N° PS prescripteur Statistique'].nunique().reset_index()
    prescripteurs_orl.columns = ['N° PS exécutant Statistique', 'Nombre de prescripteurs ORL']
    df = df.merge(prescripteurs_orl, on='N° PS exécutant Statistique', how='left')
    
    df['Pourcentage ORL'] = (df['Nombre de prescripteurs ORL'] / df['Nombre de prescripteurs']) * 100
    df['Pourcentage autres'] = 100 - df['Pourcentage ORL']
    
    df = df.drop_duplicates()
    df['Moyenne âge par établissement'] = df.groupby('N° PS exécutant Statistique')['Age du bénéficiaire'].transform('mean')
    
    return df

def process_location_data(df1, df2):

    df2['Latitude moyenne'] = (
    (df2['Latitude la plus au nord'] + df2['Latitude la plus au sud']) / 2
        )
    df2['Longitude moyenne'] = (
    (df2['Longitude la plus à l’est'] + df2['Longitude la plus à l’ouest']) / 2
        )
    
    departments_benef = df2.rename(columns={
        'Departement': 'Département du bénéficiaire',
        'Latitude moyenne': 'Latitude bénéficiaire',
        'Longitude moyenne': 'Longitude bénéficiaire'
    })
    departments_execut = df2.rename(columns={
        'Departement': "Département d'exercice du PS exécutant",
        'Latitude moyenne': 'Latitude exécutant',
        'Longitude moyenne': 'Longitude exécutant'
    })
    
    print(df2.columns)
    departments_prescripteur = df2.rename(columns={
        'Departement': 'Département du cabinet principal du PS Prescripteur',
        'Latitude moyenne': 'Latitude prescripteur',
        'Longitude moyenne': 'Longitude prescripteur'
    })

    
    # Correction des codes départementaux pour la Corse
    departments_benef.loc[departments_benef['Département du bénéficiaire'] == '2A', 'Département du bénéficiaire'] = "200"
    departments_benef.loc[departments_benef['Département du bénéficiaire'] == '2B', 'Département du bénéficiaire'] = "201"
    
    departments_execut.loc[departments_execut["Département d'exercice du PS exécutant"] == '2A', "Département d'exercice du PS exécutant"] = "200"
    departments_execut.loc[departments_execut["Département d'exercice du PS exécutant"] == '2B', "Département d'exercice du PS exécutant"] = "201"
    
    departments_prescripteur.loc[departments_prescripteur['Département du cabinet principal du PS Prescripteur'] == '2A', 'Département du cabinet principal du PS Prescripteur'] = "200"
    departments_prescripteur.loc[departments_prescripteur['Département du cabinet principal du PS Prescripteur'] == '2B', 'Département du cabinet principal du PS Prescripteur'] = "201"
    
    # entier
    departments_benef['Département du bénéficiaire'] = departments_benef['Département du bénéficiaire'].astype(int)
    departments_execut["Département d'exercice du PS exécutant"] = departments_execut["Département d'exercice du PS exécutant"].astype(int)
    departments_prescripteur['Département du cabinet principal du PS Prescripteur'] = departments_prescripteur['Département du cabinet principal du PS Prescripteur'].astype(int)

    # Fusion des données de localisation
    df1 = df1.merge(departments_benef, on='Département du bénéficiaire', how='left')
    df1 = df1.merge(departments_execut, on="Département d'exercice du PS exécutant", how='left')
    df1 = df1.merge(departments_prescripteur, on='Département du cabinet principal du PS Prescripteur', how='left')
    
    # Calcul de la distance bénéficiaire - établissement
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Rayon de la Terre en km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    df1['Distance benef ex (km)'] = haversine(
        df1['Latitude bénéficiaire'],
        df1['Longitude bénéficiaire'],
        df1['Latitude exécutant'],
        df1['Longitude exécutant']
    )

    df1['Distance pr etab (km)'] = haversine(
        df1['Latitude prescripteur'],
        df1['Longitude prescripteur'],
        df1['Latitude exécutant'],
        df1['Longitude exécutant']
    )
    
    # Filtrage sur les distances
    df1['Distance benef ex (km)'] = df1['Distance benef ex (km)'].apply(
        lambda x: 0 if x < 50 else x
    )

    # Filtrage sur les distances
    df1['Distance pr etab (km)'] = df1['Distance pr etab (km)'].apply(
        lambda x: 0 if x < 50 else x
    )

    # Calcul de la fréquence des grandes distances pour chaque établissement
    df1['Grande distance'] = (df1['Distance benef ex (km)'] > 50).astype(int)
    frequence_dist = df1.groupby('N° PS exécutant Statistique')['Grande distance'].mean().reset_index()
    frequence_dist.columns = ['N° PS exécutant Statistique', 'Frequence_dist_suspecte']

    # Normalisation entre 0 et 1
    min_val = frequence_dist['Frequence_dist_suspecte'].min()
    max_val = frequence_dist['Frequence_dist_suspecte'].max()
    frequence_dist['Frequence_dist_suspecte'] = (frequence_dist['Frequence_dist_suspecte'] - min_val) / (max_val - min_val)
    
    df1 = df1.merge(frequence_dist, on='N° PS exécutant Statistique', how='left')
    
    return df1

def calculer_importance_permutation(model, X_scaled, feature_names, sample_size=1000, random_state=None):
    """
    Calcule l'importance des features avec la permutation des scores d'anomalie sur un échantillon.
    
    :param model: Modèle IsolationForest entraîné
    :param X_scaled: Données normalisées utilisées pour l'entraînement du modèle
    :param feature_names: Liste des noms des features utilisées
    :param sample_size: Taille de l'échantillon à utiliser (par défaut 1000)
    :param random_state: Graine aléatoire pour la reproductibilité
    :return: DataFrame contenant l'impact moyen de chaque variable
    """
    np.random.seed(random_state)

    # Prendre un échantillon aléatoire des données
    indices_sample = np.random.choice(X_scaled.shape[0], min(sample_size, X_scaled.shape[0]), replace=False)
    X_sample = X_scaled[indices_sample]

    scores_original = model.decision_function(X_sample)  # Scores d'anomalie de base
    importances = []

    for i in range(X_sample.shape[1]):
        X_permuted = X_sample.copy()
        np.random.shuffle(X_permuted[:, i])  # Permuter la colonne i
        scores_permuted = model.decision_function(X_permuted)
        impact = np.abs(scores_original - scores_permuted).mean()  # Différence moyenne
        importances.append(impact)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": importances
    })

    print(importance_df.sort_values(by="Impact", ascending=False))

    return importance_df.sort_values(by="Impact", ascending=False)


