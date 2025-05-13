import streamlit as st
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import time
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import numpy as np
import warnings

# Import app modules
from app.pages.dashboard import render_dashboard
from app.pages.advanced_dashboard import render_advanced_dashboard

# Ignorer les avertissements de pandas
warnings.filterwarnings('ignore')

# Configuration de la connexion MySQL
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "123"
DB_NAME = "ghandi"
TABLE_NAME = "vicidial_rdv"

# Style CSS pour un dashboard professionnel
dashboard_style = """
<style>
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 95%;
}
.stApp {
    background-color: #f0f2f6;
}
h1, h2, h3 {
    color: #1E3A8A;
    font-weight: 600;
}
h1 {
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
}
h2 {
    font-size: 1.8rem;
    margin-bottom: 0.4rem;
}
h3 {
    font-size: 1.4rem;
    margin-bottom: 0.3rem;
}
/* Metric styling */
div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px 20px;
    border-radius: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    margin-bottom: 10px !important;
    transition: transform 0.3s, box-shadow 0.3s;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
div[data-testid="stMetric"] > div {
    justify-content: center;
}
div[data-testid="stMetric"] label {
    color: #555;
    font-weight: 500;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #1E3A8A;
    font-weight: 600;
}
/* Dataframe styling */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
div[data-testid="stDataFrame"] td {
    font-size: 0.9rem;
}
div[data-testid="stDataFrame"] th {
    background-color: #f0f2f6;
    color: #1E3A8A;
    font-weight: 600;
}
/* Chart containers */
div[data-testid="stPlotlyChart"] {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: transform 0.3s, box-shadow 0.3s;
}
div[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.dashboard-container {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 25px;
    border: 1px solid rgba(0,0,0,0.03);
}
.filters-active {
    background-color: #f0f8ff;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 20px;
    border-left: 4px solid #0079FF;
}
.info-card {
    background-color: rgba(0, 121, 255, 0.1);
    border-left: 5px solid #0079FF;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    transition: transform 0.3s, box-shadow 0.3s;
}
.info-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background-color: white;
    border-radius: 4px 4px 0px 0px;
    border: none;
    padding: 8px 16px;
    color: #666;
    font-weight: 400;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background-color: #1E3A8A !important;
    color: white !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-list"] button:hover {
    background-color: rgba(30, 58, 138, 0.1);
    color: #1E3A8A;
}
.stTabs [data-baseweb="tab-border"] {
    background-color: #f0f2f6;
}
/* Sidebar */
section[data-testid="stSidebar"] > div {
    background-color: white;
    padding: 1rem;
    border-radius: 0 15px 15px 0;
    box-shadow: 2px 0 10px rgba(0,0,0,0.05);
}
section[data-testid="stSidebar"] hr {
    margin: 10px 0;
}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #1E3A8A;
}
</style>
"""

# Dictionnaires de traduction
JOURS_SEMAINE = {
    "Monday": "Lundi",
    "Tuesday": "Mardi",
    "Wednesday": "Mercredi",
    "Thursday": "Jeudi",
    "Friday": "Vendredi",
    "Saturday": "Samedi",
    "Sunday": "Dimanche"
}

MOIS_ANNEE = {
    "January": "Janvier",
    "February": "Février",
    "March": "Mars",
    "April": "Avril",
    "May": "Mai",
    "June": "Juin",
    "July": "Juillet",
    "August": "Août",
    "September": "Septembre",
    "October": "Octobre",
    "November": "Novembre",
    "December": "Décembre"
}

# Dictionnaires inversés pour la conversion
JOURS_SEMAINE_INVERSE = {v: k for k, v in JOURS_SEMAINE.items()}
MOIS_ANNEE_INVERSE = {v: k for k, v in MOIS_ANNEE.items()}

# Ordre des jours pour le tri
JOURS_ORDRE = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


# Fonction de gestion des erreurs
def safe_execute(func, fallback_value=None, *args, **kwargs):
    """Exécute une fonction avec gestion d'erreur et valeur de repli"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.warning(f"Une erreur est survenue mais a été gérée: {str(e)}")
        return fallback_value


def fetch_data(region=None, date=None, time_filter_type=None, time_filter_value=None,
               day_filter=None, month_filter=None, week_filter=None, year_filter=None):
    try:
        engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

        if time_filter_type and time_filter_value:
            if time_filter_type == "minute":
                query = f'''
                    SELECT 
                        LEFT(LPAD(client_postal_code, 5, '0'), 2) AS region,
                        CONCAT(
                            LPAD(HOUR(last_local_call_time), 2, '0'), ':', 
                            LPAD(FLOOR(MINUTE(last_local_call_time) / {time_filter_value}) * {time_filter_value}, 2, '0'), 
                            ' - ', 
                            CASE 
                                WHEN FLOOR(MINUTE(last_local_call_time) / {time_filter_value}) * {time_filter_value} + {time_filter_value} >= 60 THEN 
                                    CONCAT(LPAD(HOUR(last_local_call_time) + 1, 2, '0'), ':', 
                                    LPAD(FLOOR(MINUTE(last_local_call_time) / {time_filter_value}) * {time_filter_value} + {time_filter_value} - 60, 2, '0'))
                                ELSE 
                                    CONCAT(LPAD(HOUR(last_local_call_time), 2, '0'), ':', 
                                    LPAD(FLOOR(MINUTE(last_local_call_time) / {time_filter_value}) * {time_filter_value} + {time_filter_value}, 2, '0'))
                            END
                        ) AS plage_horaire,
                        COUNT(*) AS nombre_rendez_vous
                    FROM {TABLE_NAME}
                    WHERE 1=1
                '''
            else:
                query = f'''
                    SELECT 
                        LEFT(LPAD(client_postal_code, 5, '0'), 2) AS region,
                        CONCAT(
                            LPAD(FLOOR(HOUR(last_local_call_time) / {time_filter_value}) * {time_filter_value}, 2, '0'), ':00 - ', 
                            CASE 
                                WHEN FLOOR(HOUR(last_local_call_time) / {time_filter_value}) * {time_filter_value} + {time_filter_value} >= 24 THEN 
                                    CONCAT(LPAD(FLOOR(HOUR(last_local_call_time) / {time_filter_value}) * {time_filter_value} + {time_filter_value} - 24, 2, '0'), ':00')
                                ELSE 
                                    CONCAT(LPAD(FLOOR(HOUR(last_local_call_time) / {time_filter_value}) * {time_filter_value} + {time_filter_value}, 2, '0'), ':00')
                            END
                        ) AS plage_horaire,
                        COUNT(*) AS nombre_rendez_vous
                    FROM {TABLE_NAME}
                    WHERE 1=1
                '''
        elif day_filter is not None:  # Nouvelle logique pour l'analyse par jour de la semaine
            # Requête modifiée pour obtenir des informations sur tous les jours de la semaine par région
            query = f'''
                SELECT 
                    LEFT(LPAD(client_postal_code, 5, '0'), 2) AS region,
                    CASE 
                        WHEN DAYNAME(last_local_call_time) = 'Monday' THEN 'Lundi'
                        WHEN DAYNAME(last_local_call_time) = 'Tuesday' THEN 'Mardi'
                        WHEN DAYNAME(last_local_call_time) = 'Wednesday' THEN 'Mercredi'
                        WHEN DAYNAME(last_local_call_time) = 'Thursday' THEN 'Jeudi'
                        WHEN DAYNAME(last_local_call_time) = 'Friday' THEN 'Vendredi'
                        WHEN DAYNAME(last_local_call_time) = 'Saturday' THEN 'Samedi'
                        WHEN DAYNAME(last_local_call_time) = 'Sunday' THEN 'Dimanche'
                    END AS jour_semaine,
                    COUNT(*) AS nombre_rendez_vous
                FROM {TABLE_NAME}
                WHERE 1=1
            '''
        elif month_filter:
            query = f'''
                SELECT 
                    LEFT(LPAD(client_postal_code, 5, '0'), 2) AS region,
                    YEAR(last_local_call_time) AS annee,
                    '{MOIS_ANNEE[month_filter]}' AS mois,
                    COUNT(*) AS nombre_rendez_vous
                FROM {TABLE_NAME}
                WHERE MONTHNAME(last_local_call_time) = '{month_filter}'
            '''
        elif week_filter is not None and year_filter is not None:
            first_day = datetime.strptime(f'{year_filter}-W{week_filter}-1', "%Y-W%W-%w").date()
            last_day = first_day + timedelta(days=6)
            query = f'''
                SELECT 
                    LEFT(LPAD(client_postal_code, 5, '0'), 2) AS region,
                    CASE 
                        WHEN DAYNAME(last_local_call_time) = 'Monday' THEN 'Lundi'
                        WHEN DAYNAME(last_local_call_time) = 'Tuesday' THEN 'Mardi'
                        WHEN DAYNAME(last_local_call_time) = 'Wednesday' THEN 'Mercredi'
                        WHEN DAYNAME(last_local_call_time) = 'Thursday' THEN 'Jeudi'
                        WHEN DAYNAME(last_local_call_time) = 'Friday' THEN 'Vendredi'
                        WHEN DAYNAME(last_local_call_time) = 'Saturday' THEN 'Samedi'
                        WHEN DAYNAME(last_local_call_time) = 'Sunday' THEN 'Dimanche'
                    END AS jour,
                    COUNT(*) AS nombre_rendez_vous
                FROM {TABLE_NAME}
                WHERE YEARWEEK(last_local_call_time, 1) = YEARWEEK('{first_day}', 1)
            '''
        else:
            query = f'''
                SELECT 
                    LEFT(LPAD(client_postal_code, 5, '0'), 2) AS region,
                    COUNT(*) AS nombre_rendez_vous
                FROM {TABLE_NAME}
                WHERE 1=1
            '''

        # Appliquer le filtre de région à toutes les requêtes
        if region:
            # Support for multiple regions separated by a hyphen
            if '-' in region:
                region1, region2 = region.split('-')
                # Sélectionner uniquement les deux régions spécifiques, pas la plage entière
                query += f" AND LEFT(LPAD(client_postal_code, 5, '0'), 2) IN ('{region1.strip()}', '{region2.strip()}')"
            else:
                query += f" AND LEFT(LPAD(client_postal_code, 5, '0'), 2) = '{region.strip()}'"

        if date:
            query += f" AND DATE(last_local_call_time) = '{date}'"

        # Ajout des clauses GROUP BY et ORDER BY en fonction du type de filtre
        if time_filter_type and time_filter_value:
            if time_filter_type == "minute":
                query += f" GROUP BY region, HOUR(last_local_call_time), FLOOR(MINUTE(last_local_call_time) / {time_filter_value}), plage_horaire"
                query += f" ORDER BY region, HOUR(last_local_call_time), FLOOR(MINUTE(last_local_call_time) / {time_filter_value})"
            else:
                query += f" GROUP BY region, FLOOR(HOUR(last_local_call_time) / {time_filter_value}), plage_horaire"
                query += f" ORDER BY region, FLOOR(HOUR(last_local_call_time) / {time_filter_value})"
        elif day_filter is not None:
            # Pour l'analyse par jour, grouper par région et jour de la semaine
            query += " GROUP BY region, jour_semaine ORDER BY region, FIELD(jour_semaine, 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche')"
        elif month_filter:
            # GROUP BY pour le filtre par mois
            query += " GROUP BY region, annee, mois ORDER BY region, annee"
        elif week_filter is not None and year_filter is not None:
            # GROUP BY pour le filtre par semaine
            query += " GROUP BY region, jour"
            query += " ORDER BY region, FIELD(jour, 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche')"
        else:
            # GROUP BY par défaut
            query += " GROUP BY region ORDER BY region"

        return pd.read_sql(query, con=engine)
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {str(e)}")
        # Retourner un DataFrame vide en cas d'erreur
        return pd.DataFrame()


def analyse_jours_par_region(df):
    """
    Analyse les données par jour de la semaine pour chaque région
    et identifie la région avec le plus de rendez-vous pour chaque jour
    """
    if df.empty or 'jour_semaine' not in df.columns:
        return pd.DataFrame(), {}

    try:
        # Création d'un pivot pour avoir les jours en colonnes
        pivot_df = df.pivot_table(
            index='region',
            columns='jour_semaine',
            values='nombre_rendez_vous',
            aggfunc='sum',
            fill_value=0
        ).reset_index()

        # Réordonner les colonnes selon l'ordre des jours
        jour_cols = [j for j in JOURS_ORDRE if j in pivot_df.columns]
        pivot_df = pivot_df[['region'] + jour_cols]

        # Ajouter une colonne de total par région
        pivot_df['Total'] = pivot_df[jour_cols].sum(axis=1)

        # Trier par total décroissant
        pivot_df = pivot_df.sort_values('Total', ascending=False)

        # Trouver la région avec le plus de RDV pour chaque jour
        top_regions = {}
        for jour in jour_cols:
            if df[df['jour_semaine'] == jour].empty:
                top_regions[jour] = {'region': 'N/A', 'nombre': 0}
                continue
                
            max_region = df[df['jour_semaine'] == jour].loc[df[df['jour_semaine'] == jour]['nombre_rendez_vous'].idxmax()]
            top_regions[jour] = {
                'region': max_region['region'],
                'nombre': int(max_region['nombre_rendez_vous'])
            }

        return pivot_df, top_regions
    except Exception as e:
        st.error(f"Erreur lors de l'analyse par jour: {str(e)}")
        return pd.DataFrame(), {}


def generate_color_palette(n_colors):
    """Génère une palette de couleurs pour les graphiques"""
    import plotly.colors as pc
    
    if n_colors <= 10:
        return pc.qualitative.D3[:n_colors]
    else:
        return pc.sample_colorscale('Viridis', n_colors)


def display_donut_chart(data, title="Distribution des régions"):
    """Affiche un graphique en anneau (donut chart)"""
    if data.empty:
        st.warning("Aucune donnée disponible pour le graphique en anneau.")
        return
        
    try:
        fig = go.Figure(data=[go.Pie(
            labels=data['region'],
            values=data['nombre_rendez_vous'],
            hole=0.5,
            marker=dict(colors=generate_color_palette(len(data)))
        )])
        
        fig.update_layout(
            title=title,
            font=dict(size=12),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de l'affichage du graphique en anneau: {str(e)}")


def display_active_filters(filters):
    """Affiche les filtres actifs"""
    active_filters = []
    if filters.get('region'):
        active_filters.append(f"🔍 Région: {filters['region']}")
    if filters.get('date'):
        active_filters.append(f"📅 Date: {filters['date']}")
    if filters.get('time_filter_type') and filters.get('time_filter_value'):
        unit = "minutes" if filters['time_filter_type'] == "minute" else "heures"
        active_filters.append(f"⏱️ {filters['time_filter_value']} {unit}")
    if filters.get('day_filter'):
        active_filters.append(f"📅 Analyse par jour de la semaine")
    if filters.get('month_filter'):
        active_filters.append(f"📅 Mois: {MOIS_ANNEE.get(filters['month_filter'], filters['month_filter'])}")
    if filters.get('use_week_filter'):
        active_filters.append(f"🗓️ Semaine {filters['week_filter']} de {filters['year_filter']}")

    if active_filters:
        st.markdown(f"""
        <div class="filters-active">
            <p style="margin:0; font-weight: 500;">Filtres actifs: {' | '.join(active_filters)}</p>
        </div>
        """, unsafe_allow_html=True)


def display_heatmap(data, x_field, y_field, value_field, title=None):
    """Affiche une heatmap avec les données fournies"""
    if data.empty:
        st.warning("Aucune donnée disponible pour la heatmap.")
        return
        
    try:
        heat_fig = px.density_heatmap(
            data,
            x=x_field,
            y=y_field,
            z=value_field,
            category_orders={"jour": JOURS_ORDRE} if x_field == "jour" or x_field == "jour_semaine" else None,
            color_continuous_scale='Viridis',
            labels={x_field: x_field.capitalize(), y_field: y_field.capitalize(), value_field: 'Nombre de RDV'}
        )

        heat_fig.update_layout(
            title=title or f"Heatmap de {value_field} par {x_field} et {y_field}",
            xaxis_title=x_field.capitalize(),
            yaxis_title=y_field.capitalize(),
            font=dict(size=12)
        )

        st.plotly_chart(heat_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de l'affichage de la heatmap: {str(e)}")


def display_radar_chart(df, regions, value_col, days_col='jour_semaine', title="Performance par jour"):
    """Affiche un graphique radar pour comparer les performances des régions par jour"""
    if df.empty:
        st.warning("Aucune donnée disponible pour le graphique radar.")
        return
        
    try:
        # Si plus de 5 régions, prendre seulement les 5 plus importantes
        if len(regions) > 5:
            top_regions = df.groupby('region')[value_col].sum().nlargest(5).index.tolist()
        else:
            top_regions = regions.tolist()
            
        # Créer figure
        fig = go.Figure()
        
        colors = generate_color_palette(len(top_regions))
        
        for i, region in enumerate(top_regions):
            region_data = df[df['region'] == region]
            
            if not region_data.empty:
                # S'assurer que toutes les journées sont présentes
                all_days = {}
                for day in JOURS_ORDRE:
                    day_value = region_data[region_data[days_col] == day][value_col].sum()
                    all_days[day] = day_value if not pd.isna(day_value) else 0
                
                fig.add_trace(go.Scatterpolar(
                    r=[all_days.get(day, 0) for day in JOURS_ORDRE],
                    theta=JOURS_ORDRE,
                    fill='toself',
                    name=f'Région {region}',
                    line_color=colors[i]
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, df[value_col].max() * 1.1]
                )
            ),
            title=title,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de l'affichage du graphique radar: {str(e)}")


def display_distribution_chart(df, col='nombre_rendez_vous', title="Distribution des rendez-vous"):
    """Affiche un histogramme avec courbe de densité"""
    if df.empty:
        st.warning("Aucune donnée disponible pour l'histogramme.")
        return
        
    try:
        # Filtrer les valeurs pour éviter les erreurs
        filtered_values = df[col].dropna()
        
        if len(filtered_values) < 2:
            st.warning(f"Pas assez de données pour afficher la distribution de {col}.")
            return
            
        # Créer l'histogramme avec courbe de densité
        fig = ff.create_distplot(
            [filtered_values], 
            group_labels=[col],
            bin_size=max(1, (filtered_values.max() - filtered_values.min()) / 20),
            curve_type='normal',
            colors=['#1E3A8A']
        )
        
        fig.update_layout(
            title=title,
            xaxis_title=col,
            yaxis_title="Fréquence",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de l'affichage de la distribution: {str(e)}")
        # Fallback: afficher un histogramme simple
        try:
            st.write("Affichage d'un histogramme simplifié:")
            alt_fig = px.histogram(df, x=col, title=title)
            st.plotly_chart(alt_fig, use_container_width=True)
        except:
            st.error("Impossible d'afficher l'histogramme alternatif.")


def display_combined_chart(df, title="Performance par région"):
    """Affiche un graphique combiné barres + ligne"""
    if df.empty:
        st.warning("Aucune donnée disponible pour le graphique combiné.")
        return
        
    try:
        # Top 10 régions par nombre de RDV
        top_df = df.nlargest(10, 'nombre_rendez_vous')
        
        # Créer un graphique avec deux axes Y
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Ajouter les barres
        fig.add_trace(
            go.Bar(
                x=top_df['region'],
                y=top_df['nombre_rendez_vous'],
                name="Nombre de RDV",
                marker_color='#1E3A8A'
            ),
            secondary_y=False
        )
        
        # Calculer le pourcentage cumulé
        total = top_df['nombre_rendez_vous'].sum()
        top_df = top_df.copy()
        top_df['pct_cumul'] = top_df['nombre_rendez_vous'].cumsum() / total * 100
        
        # Ajouter la ligne de pourcentage cumulé
        fig.add_trace(
            go.Scatter(
                x=top_df['region'],
                y=top_df['pct_cumul'],
                name="% cumulé",
                line=dict(color='#FF5733', width=3)
            ),
            secondary_y=True
        )
        
        # Mise à jour des layouts
        fig.update_layout(
            title=title,
            xaxis_title="Région",
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Mettre à jour les titres des axes y
        fig.update_yaxes(title_text="Nombre de RDV", secondary_y=False)
        fig.update_yaxes(title_text="Pourcentage cumulé", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de l'affichage du graphique combiné: {str(e)}")


def validate_region(input_str):
    """Valide le format de la région (désactivation de la validation comme demandé)"""
    return input_str.strip() if input_str else ''


# Initialisation des variables de session
if "last_data" not in st.session_state:
    st.session_state.last_data = pd.DataFrame()
if "history" not in st.session_state:
    st.session_state.history = []
if "last_update" not in st.session_state:
    st.session_state.last_update = None
if "dashboard_mode" not in st.session_state:
    st.session_state.dashboard_mode = "standard"  # Default to standard dashboard

# Configuration de la page
st.set_page_config(page_title="📊 Dashboard Temps Réel - Rendez-vous", layout="wide")

# Appliquer le style CSS
st.markdown(dashboard_style, unsafe_allow_html=True)

# Dashboard header with toggle
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊 Tableau de bord des Rendez-vous")
    st.caption(f"Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}")
with col2:
    # Add toggle for switching between standard and advanced dashboard
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    dashboard_toggle = st.selectbox(
        "Version du tableau de bord:",
        ["Standard", "Avancé 2025"],
        index=0 if st.session_state.dashboard_mode == "standard" else 1,
        key="dashboard_selector"
    )
    st.session_state.dashboard_mode = "standard" if dashboard_toggle == "Standard" else "advanced"

# Widgets de filtrage dans la barre latérale
st.sidebar.header("🔍 Filtres")

# Entrée de région (sans validation comme demandé)
region_input = st.sidebar.text_input(
    "Entrer la région (2 chiffres ou 00-99)",
    "",
    help="Entrez un code de région à 2 chiffres (00-99) ou deux codes séparés par un tiret (ex: 00-99)"
)
region = region_input.strip() if region_input else None

date = st.sidebar.date_input("Sélectionner la date", None)

# Filtres temporels
st.sidebar.header("📅 Filtres Temporels")
time_filter_type = st.sidebar.radio("Type de filtre horaire", ["Aucun", "Minute", "Heure"], index=0)

time_filter_value = None
if time_filter_type == "Minute":
    time_filter_value = st.sidebar.selectbox("Interval en minutes", [5, 10, 15, 30, 60], index=2)
elif time_filter_type == "Heure":
    time_filter_value = st.sidebar.selectbox("Interval en heures", [1, 2, 3, 4, 6, 12], index=0)

# Nouvelle logique pour le jour de la semaine - choix simple sans filtre
jour_semaine_choice = st.sidebar.checkbox("Analyser par jour de la semaine", value=False)
day_filter = True if jour_semaine_choice else None

# Filtre par mois
month_filter = st.sidebar.selectbox("Filtrer par mois",
                                  ["Tous"] + list(MOIS_ANNEE.values()),
                                  index=0)
month_filter = None if month_filter == "Tous" else MOIS_ANNEE_INVERSE[month_filter]

# Filtre par semaine
st.sidebar.header("🗓️ Filtre par Semaine")
year_filter = st.sidebar.number_input("Année", min_value=2000, max_value=2100, value=datetime.now().year)
week_filter = st.sidebar.number_input("Numéro de semaine", min_value=1, max_value=53,
                                    value=datetime.now().isocalendar()[1])
use_week_filter = st.sidebar.checkbox("Utiliser le filtre par semaine")

# Création des conteneurs pour l'interface
data_placeholder = st.empty()
chart_placeholder = st.empty()
details_placeholder = st.empty()  # Pour les détails supplémentaires

# Collect filter settings
active_filters = {
    'region': region,
    'date': date.strftime('%Y-%m-%d') if date else None,
    'time_filter_type': time_filter_type.lower() if time_filter_type != "Aucun" else None,
    'time_filter_value': time_filter_value,
    'day_filter': day_filter,
    'month_filter': month_filter,
    'use_week_filter': use_week_filter,
    'week_filter': week_filter if use_week_filter else None,
    'year_filter': year_filter if use_week_filter else None
}

# Render the appropriate dashboard based on the mode
if st.session_state.dashboard_mode == "standard":
    # Add refresh button
    st.sidebar.button("🔄 Rafraîchir les données", key="refresh_standard")
    
    # Get data for standard dashboard
    with st.spinner("Chargement des données..."):
        # Fetch data or use cached data from session state
        data = fetch_data(
            region,
            date.strftime('%Y-%m-%d') if date else None,
            time_filter_type.lower() if time_filter_type != "Aucun" else None,
            time_filter_value,
            day_filter,
            month_filter,
            week_filter if use_week_filter else None,
            year_filter if use_week_filter else None
        )
    
    # Update session state for standard dashboard
    st.session_state.last_data = data
    
    # Display the standard dashboard
    display_active_filters(active_filters)
    
    # Use dashboard module to render
    data_placeholder = st.empty()
    chart_placeholder = st.empty()
    details_placeholder = st.empty()
    
    with data_placeholder:
        st.subheader("📌 Données mises à jour")
    
    # Render dashboard content
    if data is not None and not data.empty:
        render_dashboard(data, active_filters)
    else:
        st.warning("Aucune donnée disponible pour les filtres sélectionnés.")
        
else:  # Advanced dashboard
    # Add refresh button with unique key
    if st.sidebar.button("🔄 Rafraîchir les données", key="refresh_advanced"):
        # Force refresh in session state
        st.session_state.force_refresh = True
    
    # Add description about advanced features
    st.sidebar.markdown("""
    <div style="background-color: rgba(0, 121, 255, 0.1); 
                padding: 10px; 
                border-radius: 10px; 
                margin-top: 20px;
                border-left: 3px solid #0079FF;">
        <p style="margin:0; font-size: 0.9em;">
            <strong>Tableau de bord avancé 2025</strong><br>
            Visualisations interactives avec analyses prédictives et intelligence artificielle.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render the advanced dashboard
    render_advanced_dashboard(active_filters)

# Add information about auto-refresh
st.sidebar.markdown("""
<div style="margin-top: 30px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; font-size: 0.8em;">
    <p style="color: #666;">Les données sont automatiquement mises à jour. Vous pouvez également cliquer sur 'Rafraîchir les données' pour une mise à jour immédiate.</p>
</div>
""", unsafe_allow_html=True)
