

Un tableau de bord professionnel pour visualiser et analyser les données de rendez-vous à partir d'une base de données MySQL.

## Description

Cette application Streamlit offre une interface utilisateur moderne et interactive pour explorer et visualiser les données de rendez-vous stockées dans une base de données MySQL. Elle permet de filtrer les données par région, date, plage horaire, jour de la semaine, mois et semaine, tout en affichant des visualisations complexes comme des graphiques, des tableaux croisés et des cartes thermiques.

## Fonctionnalités Principales

- **Analyse temporelle** : Visualisation des tendances par jour de la semaine, plage horaire, mois
- **Analyse géographique** : Répartition des rendez-vous par région
- **Tableau de bord interactif** : Filtres dynamiques et mise à jour en temps réel
- **Visualisations avancées** :
  - Graphiques à barres empilées
  - Cartes thermiques (heatmaps)
  - Tableaux croisés dynamiques
  - Indicateurs KPI
- **Export des données** : Possibilité d'exporter les données filtrées

## Architecture Technique

### Structure du Projet

```bash
rndv_ghandi/
├── app/                      # Package principal de l'application
│   ├── components/           # Composants réutilisables de l'UI
│   │   ├── __init__.py
│   │   ├── data_display.py   # Affichage des tableaux et données brutes
│   │   ├── metrics.py        # Calcul et affichage des indicateurs KPI
│   │   ├── sidebar.py        # Configuration des filtres utilisateur
│   │   ├── visualizations.py # Graphiques de base (barres, lignes)
│   │   └── advanced_visualizations.py # Visualisations complexes (heatmaps)
│   ├── pages/                # Pages de l'application
│   │   ├── __init__.py
│   │   ├── dashboard.py      # Page principale avec vue d'ensemble
│   │   └── advanced_dashboard.py # Page avec analyses avancées
│   └── __init__.py
├── config/                   # Configuration
│   ├── __init__.py
│   ├── config.py             # Paramètres globaux de l'application
│   └── .env                  # Variables sensibles (non versionnées)
├── database/                 # Couche d'accès aux données
│   ├── __init__.py
│   └── connection.py         # Gestion des connexions MySQL
├── services/                 # Services métier
│   ├── __init__.py
│   └── data_service.py       # Logique de récupération/transformation des données
├── utils/                    # Utilitaires
│   ├── __init__.py
│   ├── logger.py             # Configuration centralisée du logging
│   └── translations.py       # Gestion des traductions (fr/en)
├── tests/                    # Tests unitaires et d'intégration
├── app.py                    # Point d'entrée Streamlit
├── requirements.txt          # Dépendances Python
└── README.md                 # Documentation
```

### Flux de Données

1. **Initialisation** :
   - Chargement de la configuration depuis `config/`
   - Établissement de la connexion MySQL via `database/connection.py`
   - Configuration des logs via `utils/logger.py`

2. **Exécution** :
   - Streamlit lance `app.py` qui charge les pages depuis `app/pages/`
   - Chaque page utilise des composants depuis `app/components/`
   - Les composants appellent `services/data_service.py` pour récupérer les données
   - Le service de données interroge la base MySQL via la connexion établie

3. **Affichage** :
   - Les données transformées sont affichées via les composants d'interface
   - Les interactions utilisateur déclenchent des rechargements des données

## Installation

### Prérequis

- Python 3.7+
- MySQL Server 8.0+
- Base de données 'ghandi' avec table 'vicidial_rdv'
- Compte MySQL avec droits de lecture sur la table

### Configuration

1. Installer les dépendances :

```bash
pip install -r requirements.txt
```

3. (Optionnel) Personnaliser l'interface dans `config/config.py` :

```python
APP_CONFIG = {
    "page_title": "Dashboard Rendez-vous",
}
```

## Utilisation

Lancer l'application :

```bash
streamlit run app.py
```

L'application sera accessible à : [http://localhost:8501](http://localhost:8501)

### Fonctionnalités Avancées

1. **Filtres** :
   - Sélection multiple de régions
   - Plage de dates avec calendrier interactif
   - Filtres par jour de semaine/heure

2. **Visualisations** :
   - Heatmap des rendez-vous par heure/jour
   - Graphiques empilés par type de rendez-vous
   - Carte géographique des régions (si données GPS disponibles)

3. **Performance** :
   - Cache des requêtes SQL
   - Chargement asynchrone des données
   - Optimisation des requêtes

## Licence

Ce projet est sous licence propriétaire. Tous droits réservés.
