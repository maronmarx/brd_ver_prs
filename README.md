# Dashboard de Rendez-vous Ghandi

Un tableau de bord professionnel pour visualiser et analyser les données de rendez-vous à partir d'une base de données MySQL.

## Description

Cette application Streamlit offre une interface utilisateur moderne et interactive pour explorer et visualiser les données de rendez-vous stockées dans une base de données MySQL. Elle permet de filtrer les données par région, date, plage horaire, jour de la semaine, mois et semaine, tout en affichant des visualisations complexes comme des graphiques, des tableaux croisés et des cartes thermiques.

## Structure du Projet

```
rndv_ghandi/
├── app/                      # Package principal de l'application
│   ├── components/           # Composants réutilisables de l'UI
│   │   ├── __init__.py
│   │   ├── data_display.py   # Affichage des données et tableaux
│   │   ├── metrics.py        # Indicateurs de performance
│   │   ├── sidebar.py        # Filtres de la barre latérale
│   │   └── visualizations.py # Graphiques et visualisations
│   ├── pages/                # Pages de l'application
│   │   ├── __init__.py
│   │   └── dashboard.py      # Page principale du tableau de bord
│   └── __init__.py
├── config/                   # Configuration
│   ├── __init__.py
│   ├── config.py             # Paramètres de configuration
│   └── .env                  # Variables d'environnement
├── database/                 # Couche d'accès aux données
│   ├── __init__.py
│   └── connection.py         # Gestionnaire de connexion à la BD
├── services/                 # Services métier
│   ├── __init__.py
│   └── data_service.py       # Service d'accès aux données
├── utils/                    # Utilitaires
│   ├── __init__.py
│   ├── logger.py             # Configuration du journal
│   └── translations.py       # Traductions fr/en
├── tests/                    # Tests unitaires et d'intégration
├── app.py                    # Point d'entrée de l'application
└── README.md                 # Documentation
```

## Installation

### Prérequis

- Python 3.7+
- MySQL Server
- Base de données 'ghandi' avec table 'vicidial_rdv'

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Configuration

1. Modifier le fichier `config/.env` pour configurer les paramètres de connexion à la base de données:

```
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=root
DB_NAME=ghandi
TABLE_NAME=vicidial_rdv
```

## Utilisation

Lancer l'application:

```bash
streamlit run app.py
```

L'application sera accessible à l'adresse: http://localhost:8501

## Fonctionnalités

- **Filtrage des données**: par région, date, plage horaire, jour, mois et semaine
- **Visualisations**: graphiques à barres, cartes thermiques, tableaux croisés
- **Métriques**: indicateurs clés de performance avec mise à jour en temps réel
- **Analyse par région**: visualisation des rendez-vous par région géographique
- **Analyse temporelle**: tendances par jour de la semaine, plage horaire, etc.
- **Actualisation automatique**: mise à jour périodique des données

## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `config/config.py`, notamment:

- `APP_CONFIG`: paramètres de l'interface utilisateur
- `CACHE_CONFIG`: configuration du cache
- `LOGGING_CONFIG`: configuration de la journalisation

## Structure de la base de données

La table `vicidial_rdv` doit inclure au minimum les colonnes suivantes:
- `client_postal_code`: code postal du client (utilisé pour la région)
- `last_local_call_time`: horodatage du rendez-vous

## Licence

Ce projet est sous licence privée.
