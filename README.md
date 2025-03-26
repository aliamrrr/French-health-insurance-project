🕵️♂️ Détection d'Anomalies dans les Remboursements d'Audioprothèses

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)

Application d'analyse avancée pour détecter les comportements suspects dans les remboursements d'appareils auditifs grâce au machine learning.

## 📋 Table des Matières
- [Prérequis](#-prérequis)
- [🚀 Installation](#-installation)
- [📁 Structure des Fichiers](#-structure-des-fichiers)
- [🖥️ Utilisation](#%EF%B8%8F-utilisation)
- [✨ Fonctionnalités](#-fonctionnalités)
- [📚 Dépendances](#-dépendances)


## 🔧 Prérequis
- Ordinateur Windows/MacOS/Linux
- 500 Mo d'espace disque libre
- Connexion internet stable

## 🚀 Installation
### Étape 1 : Installer Python
1. Téléchargez Python 3.9+ sur [python.org](https://www.python.org/downloads/)
2. **Windows** : Cochez "Add Python to PATH" pendant l'installation  
   **MacOS** : Utilisez le package .pkg  
   **Linux** : `sudo apt-get install python3`

### Étape 2 : Télécharger les Fichiers
1. [Téléchargez le dossier du projet (.ZIP)](https://github.com/votrecompte/audioprotheses/archive/main.zip)
2. Décompressez le dossier où vous voulez

### Étape 3 : Installer les Dépendances
Ouvrez un terminal dans le dossier du projet et exécutez :
```bash
pip install -r requirements.txt
```

## 📁 Structure des Fichiers
```
audioprotheses/
├── main.py              - Code principal de l'application
├── requirements.txt     - Liste des dépendances
├── regions.csv          - Fichier de mapping des régions (à télécharger)
└── donnees_audioprotheses.csv  - Jeu de données principal (à fournir par l'utilisateur)
```

## 🖥️ Utilisation
1. Placez votre fichier CSV de données dans le dossier principal
2. Soyez sur d'avoir le fichier "points-extremes-des-departements-metropolitains-de-france.csv" dans votre dossier de travail
3. Lancez l'application avec :
```bash
streamlit run main.py
```
4. Suivez les instructions dans le navigateur qui s'ouvre automatiquement

## ✨ Fonctionnalités

### 🕵️ Analyse Intelligente
- Détection automatique des anomalies avec Isolation Forest
- Score de risque personnalisé pour chaque établissement
- Estimation des pertes financières potentielles

### 📊 Tableaux de Bord Interactifs
```python
Tabs:
1️⃣ Aperçu des Données - Exploration rapide du dataset
2️⃣ Analyse Avancée    - Détection d'anomalies en temps réel
3️⃣ Visualisations     - Graphiques dynamiques et exportables
```


## 📚 Dépendances
| Librairie      | Version | Usage                 |
|----------------|---------|-----------------------|
| Streamlit      | 1.22.0  | Interface utilisateur |
| scikit-learn   | 1.2.2   | Machine Learning      |
| Pandas         | 1.5.3   | Analyse des données   |
| Matplotlib     | 3.7.1   | Visualisations        |
