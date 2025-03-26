# Analyse des Anomalies dans les Données d'Audioprothèses avec Streamlit

## Description
Cette application Streamlit permet d'analyser les anomalies dans un ensemble de données d'audioprothèses en utilisant l'algorithme Isolation Forest. Elle permet d'explorer les données, de détecter les valeurs aberrantes et de visualiser les résultats de l'analyse.

## Fonctionnalités
- **Chargement des données** : Importation des données au format CSV.
- **Exploration des données** : Affichage des statistiques descriptives et des distributions des variables.
- **Détection des anomalies** : Utilisation d'Isolation Forest pour identifier les valeurs aberrantes.
- **Visualisation** : Graphiques interactifs pour observer les anomalies.

## Installation
Assurez-vous d'avoir Python installé sur votre machine. Clonez ce dépôt et installez les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

## Utilisation
Lancez l'application avec la commande suivante :

```bash
streamlit run app.py
```

## Structure du projet
```
/
|-- app.py                # Code principal de l'application
|-- requirements.txt      # Liste des dépendances
|-- data/                 # Dossier contenant les fichiers de données
|-- README.md             # Documentation du projet
```

## Configuration
Si nécessaire, modifiez les hyperparamètres d'Isolation Forest directement dans `app.py`.

## Auteur
Ali Anass

## Licence
MIT License

