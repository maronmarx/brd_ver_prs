"""
Dashboard page module.
Provides the main dashboard rendering functionality.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from typing import Dict, Any
import io

from utils.translations import MOIS_ANNEE, JOURS_ORDRE

# Get logger
logger = logging.getLogger('rndv_ghandi.dashboard')


def fetch_and_process_data(filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Fetch data from the service based on filters.
    This function is a wrapper for backward compatibility.
    
    Args:
        filters (Dict[str, Any]): Dictionary of filter values
        
    Returns:
        pd.DataFrame: Processed data
    """
    try:
        # In the modular version, we receive data directly
        # This function is kept for backward compatibility
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def display_data_section(data: pd.DataFrame, filters: Dict[str, Any]):
    """
    Display the main data section of the dashboard.
    
    Args:
        data (pd.DataFrame): Data to display
        filters (Dict[str, Any]): Dictionary of filter values
    """
    st.markdown('<h3 style="text-align: center; margin-bottom: 20px;">Données des Rendez-vous</h3>', unsafe_allow_html=True)
    
    # Calculate main metrics
    total_rdv = data['nombre_rendez_vous'].sum() if not data.empty else 0
    
    if 'plage_horaire' in data.columns:
        cols = st.columns(3)
        cols[0].metric("📍 Total RDV", f"{total_rdv:,}")
        cols[1].metric("📊 Régions", f"{data['region'].nunique()}")
        cols[2].metric("⏱️ Plages", f"{data['plage_horaire'].nunique()}")

        # Add download button based on the data columns present
    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True) # Add spacing after button

    # Display data table (keep existing logic for displaying the correct table)
    if not data.empty and 'jour_semaine' not in data.columns:
        # Use display_dataframe function from data_display module to ensure correct sorting
        from app.components.data_display import display_dataframe
        display_dataframe(data)

        # Add download button for the main data table when day filter is not active
        csv_buffer = io.StringIO()
        # Sort data for download to match display order
        sorted_data = data.sort_values(by='nombre_rendez_vous', ascending=False)
        sorted_data.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Télécharger en CSV",
            data=csv_buffer.getvalue(),
            file_name='donnees_rendez_vous.csv',
            mime='text/csv',
            key='download_main_data'
        )

    elif 'jour_semaine' in data.columns:
         # For day filter, display pivot table
        if not data.empty: # Redundant check, but keeping for safety
            # Analyse des données par jour de la semaine
            pivot_data, top_regions = analyse_jours_par_region(data) # Regenerate for display

            # Add download button for the pivot table
            if not pivot_data.empty:
                csv_buffer = io.StringIO()
                
                # Create a copy for CSV export
                csv_pivot_data = pivot_data.copy()
                
                # Format percentage columns for CSV
                jour_cols = [j for j in JOURS_ORDRE if j in pivot_data.columns]
                percentage_cols = [f'{jour} (%)' for jour in jour_cols] + ['Total (%)']
                
                for col in percentage_cols:
                    if col in csv_pivot_data.columns:
                        csv_pivot_data[col] = csv_pivot_data[col].apply(lambda x: f'{x:.2f}%' if pd.notna(x) else '')
                
                csv_pivot_data.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Télécharger l'analyse par jour en CSV", 
                    data=csv_buffer.getvalue(),
                    file_name='analyse_jours_par_region.csv',
                    mime='text/csv',
                    key='download_day_analysis'
                )

            if not pivot_data.empty:
                st.write("### 📊 Répartition des rendez-vous par jour de la semaine et par région")

                # Formater le dataframe pivot pour l'affichage
                formatted_pivot = pivot_data.copy()
                jour_cols = [j for j in JOURS_ORDRE if j in formatted_pivot.columns]

                # Define formatting dictionary
                format_dict = {col: '{:,.0f}' for col in jour_cols + ['Total']}
                for jour in jour_cols:
                    format_dict[f'{jour} (%)'] = '{:.1f}%'
                format_dict['Total (%)'] = '{:.1f}%'

                st.dataframe(
                    formatted_pivot
                    .style
                    .background_gradient(subset=jour_cols + ['Total'], cmap='viridis')
                    .format(format_dict)
                    .set_properties(**{'text-align': 'center', 'border': '1px solid grey'}),
                    use_container_width=True
                )
    # No need for explicit display logic for other elifs as they don't have separate tables


def display_chart_section(data: pd.DataFrame, filters: Dict[str, Any]):
    """
    Display the chart section of the dashboard.
    
    Args:
        data (pd.DataFrame): Data to display
        filters (Dict[str, Any]): Dictionary of filter values
    """
    st.markdown('<h3 style="text-align: center; margin-bottom: 20px;">Visualisations</h3>', unsafe_allow_html=True)
    
    # Show charts based on available data dimensions
    if 'plage_horaire' in data.columns:
        # Time slots analysis
        fig = px.bar(
            data, 
            x='plage_horaire', 
            y='nombre_rendez_vous', 
            color='region',
            title=f"Rendez-vous par plage horaire",
            labels={'plage_horaire': 'Plage horaire', 'nombre_rendez_vous': 'Nombre de RDV', 'region': 'Région'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True, key="plage_horaire_chart")
    elif 'jour_semaine' in data.columns:
        # Day of week analysis - call the details section to display region cards and visualizations
        display_details_section(data, filters, prefix="main_")
    elif 'annee' in data.columns and 'mois' in data.columns:
        # Monthly analysis
        fig = px.bar(
            data, 
            x='region', 
            y='nombre_rendez_vous', 
            color='annee',
            title=f"Rendez-vous par région et par année",
            labels={'region': 'Région', 'nombre_rendez_vous': 'Nombre de RDV', 'annee': 'Année'}
        )
        st.plotly_chart(fig, use_container_width=True, key="monthly_analysis_chart")
    elif 'jour' in data.columns:
        # Weekly analysis
        fig = px.bar(
            data, 
            x='region', 
            y='nombre_rendez_vous', 
            color='jour',
            title=f"Rendez-vous par région et par jour",
            labels={'region': 'Région', 'nombre_rendez_vous': 'Nombre de RDV', 'jour': 'Jour'},
            category_orders={"jour": JOURS_ORDRE}
        )
        st.plotly_chart(fig, use_container_width=True, key="weekly_analysis_chart")
    else:
        # General region analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            bar_fig = px.bar(
                data, 
                x='region', 
                y='nombre_rendez_vous',
                title="Rendez-vous par région", 
                color='nombre_rendez_vous',
                color_continuous_scale='Viridis',
                labels={'region': 'Région', 'nombre_rendez_vous': 'Nombre de RDV'}
            )
            st.plotly_chart(bar_fig, use_container_width=True, key="region_bar_chart")
        
        with col2:
            # Donut chart
            from app.components.visualizations import display_donut_chart
            display_donut_chart(data, value_col='nombre_rendez_vous', name_col='region', title="Distribution des rendez-vous par région")

def display_details_section(data: pd.DataFrame, filters: Dict[str, Any], prefix: str = ""):
    """
    Display detailed analysis section.
    
    Args:
        data (pd.DataFrame): Data to display
        filters (Dict[str, Any]): Dictionary of filter values
    """
    if 'jour_semaine' not in data.columns:
        st.info("L'analyse par jour de la semaine n'est pas disponible avec les filtres actuels.")
        return
    
    # Perform detailed analysis by day of week
    pivot_data, top_regions = analyse_jours_par_region(data)
    
    if not pivot_data.empty:
        # Display region cards
        st.markdown("<h4 style='text-align: center;'>Régions avec le plus de rendez-vous par jour</h4>", unsafe_allow_html=True)
        
        cards_cols = st.columns(len(top_regions))
        
        for i, (jour, data) in enumerate(top_regions.items()):
            with cards_cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        background-color: rgba(0, 121, 255, 0.1);
                        border-left: 5px solid #0079FF;
                        padding: 10px;
                        border-radius: 5px;
                        margin-bottom: 10px;
                    ">
                        <h4 style="color: #0079FF; margin: 0;">{jour}</h4>
                        <p style="margin: 5px 0 0 0;">
                            <span style="font-weight: bold;">Région:</span> {data['region']}
                        </p>
                        <p style="margin: 5px 0 0 0;">
                            <span style="font-weight: bold;">Rendez-vous:</span> {data['nombre']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Create heatmap for days by region
        st.markdown("<h4 style='text-align: center;'>Carte thermique des rendez-vous</h4>", unsafe_allow_html=True)
        
        # Prepare data for heatmap
        jour_cols = [j for j in JOURS_ORDRE if j in pivot_data.columns]
        heatmap_data = []
        
        for _, row in pivot_data.iterrows():
            region = row['region']
            for jour in jour_cols:
                heatmap_data.append({
                    'region': region,
                    'jour': jour,
                    'nombre': row[jour]
                })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Display heatmap
        heat_fig = px.density_heatmap(
            heatmap_df,
            x='jour',
            y='region',
            z='nombre',
            category_orders={"jour": JOURS_ORDRE},
            color_continuous_scale='Viridis',
            labels={'jour': 'Jour de la semaine', 'region': 'Région', 'nombre': 'Nombre de RDV'}
        )
        
        heat_fig.update_layout(
            title="Intensité des rendez-vous par jour et par région",
            font=dict(size=12),
            yaxis_title="Région",
            xaxis_title="Jour de la semaine"
        )
        
        st.plotly_chart(heat_fig, use_container_width=True, key=f"{prefix}heatmap_details")
        
        # Create bar chart for daily totals
        daily_totals = pivot_data[jour_cols].sum().reset_index()
        daily_totals.columns = ['jour', 'total']
        daily_totals = daily_totals.sort_values('jour', key=lambda x: [JOURS_ORDRE.index(i) for i in x])
        
        bar_fig = px.bar(
            daily_totals,
            x='jour',
            y='total',
            color='total',
            text='total',
            title="Nombre total de rendez-vous par jour de la semaine",
            labels={'jour': 'Jour de la semaine', 'total': 'Nombre total de RDV'},
            color_continuous_scale='Viridis'
        )
        
        bar_fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        bar_fig.update_layout(
            xaxis=dict(categoryorder='array', categoryarray=JOURS_ORDRE),
            yaxis_title="Nombre de rendez-vous",
            showlegend=False
        )
        
        st.plotly_chart(bar_fig, use_container_width=True, key=f"{prefix}daily_totals_chart")

def analyse_jours_par_region(df: pd.DataFrame):
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

        # Ajouter une colonne de total par région et trier par ordre descendant
        pivot_df['Total'] = pivot_df[jour_cols].sum(axis=1)
        pivot_df = pivot_df.sort_values('Total', ascending=False)

        # Calculer les pourcentages par jour (par rapport au total de la colonne du jour)
        for jour in jour_cols:
            total_jour = pivot_df[jour].sum()
            if total_jour > 0:
                pivot_df[f'{jour} (%)'] = (pivot_df[jour] / total_jour) * 100
            else:
                pivot_df[f'{jour} (%)'] = 0

        # Calculer le pourcentage total par région (par rapport au grand total)
        grand_total = pivot_df['Total'].sum()
        if grand_total > 0:
            pivot_df['Total (%)'] = (pivot_df['Total'] / grand_total) * 100
        else:
            pivot_df['Total (%)'] = 0

        # Réordonner les colonnes pour inclure les pourcentages à côté des valeurs
        ordered_cols = ['region']
        for jour in jour_cols:
            ordered_cols.extend([jour, f'{jour} (%)'])
        ordered_cols.extend(['Total', 'Total (%)'])
        pivot_df = pivot_df[ordered_cols]

        # Trouver la région avec le plus de RDV pour chaque jour
        top_regions = {}
        for jour in jour_cols:
            if df[df['jour_semaine'] == jour].empty:
                top_regions[jour] = {'region': 'N/A', 'nombre': 0}
                continue

            # Find the region with the max number of appointments for the current day
            max_region_row = df[df['jour_semaine'] == jour].loc[df[df['jour_semaine'] == jour]['nombre_rendez_vous'].idxmax()]

            top_regions[jour] = {
                'region': max_region_row['region'],
                'nombre': int(max_region_row['nombre_rendez_vous'])
            }

        return pivot_df, top_regions
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse par jour: {str(e)}")
        return pd.DataFrame(), {}


def render_dashboard(data: pd.DataFrame, filters: Dict[str, Any]):
    """
    Render the main dashboard based on filters.
    
    Args:
        filters (Dict[str, Any]): Dictionary of filter values
    """
    # Set page configuration for a ultra-modern 2025 professional look
    st.markdown("""
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f2f6;
        padding: 0px 20px;
        border-radius: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 10px 20px;
        margin: 5px 5px;
        transition: all 0.3s ease;
        font-weight: 500;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0079FF, #00DFA2) !important;
        color: white !important;
        box-shadow: 0 4px 10px rgba(0, 121, 255, 0.3) !important;
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .dashboard-container {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        border: 1px solid rgba(0,0,0,0.03);
        transition: all 0.3s ease;
    }
    .dashboard-container:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    .css-1aumxhk {
        background-color: #f8f9fa;
        background-image: linear-gradient(to bottom, #f0f2f6, #ffffff);
    }
    /* Metric styling */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 10px !important;
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
    }
    .chart-title {
        color: #1E3A8A;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
        text-align: center;
    }
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 10px 0;
        background: linear-gradient(45deg, #0079FF, #00DFA2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        color: #555;
        font-size: 1rem;
    }
    .trend-indicator-up {
        color: #00DFA2;
        font-weight: 600;
    }
    .trend-indicator-down {
        color: #FF0060;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display dashboard header with current time
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h1 style="margin: 0;">📊 Tableau de bord des Rendez-vous</h1>
        <p style="color: gray; margin: 0;">Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different dashboard views
    tabs = st.tabs(["📊 Vue principale", "🔍 Analyse détaillée", "📈 Tendances", "🗺️ Carte des régions"])
    
    # Check if we need to refresh data (either first load or sync button clicked)
    if 'last_data' not in st.session_state or st.session_state.get('force_refresh', False):
        # Fetch data based on filter settings
        with st.spinner("Chargement des données..."):
            data = fetch_and_process_data(filters)
        
        # Update session state
        if data is not None:
            st.session_state.last_data = data
            st.session_state.last_update = datetime.now()
            st.session_state.force_refresh = False
            
            # Update history (keep last 5 updates)
            st.session_state.history.append({
                'timestamp': datetime.now(),
                'data': data.copy() if not data.empty else pd.DataFrame()
            })
            if len(st.session_state.history) > 5:
                st.session_state.history.pop(0)
        else:
            # No data available
            st.warning("Aucune donnée disponible pour les filtres sélectionnés.")
            return
    
    # Use data from session state
    data = st.session_state.last_data
    
    # Tab 1: Main Dashboard View
    with tabs[0]:
        if not data.empty:
            # Display modern KPI cards
            from app.components.visualizations import display_kpi_cards
            display_kpi_cards(data)
            
            # Display main data section
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            display_data_section(data, filters)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Afficher les filtres actifs
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
                <div style="background-color: #f0f8ff; padding: 10px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #0079FF;">
                    <p style="margin:0; font-weight: 500;">Filtres actifs: {' | '.join(active_filters)}</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Display main chart section
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            display_chart_section(data, filters)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Aucune donnée disponible pour les filtres sélectionnés.")
    
    # Tab 2: Detailed Analysis
    with tabs[1]:
        if not data.empty:
            # Affichage d'un en-tête moderne pour l'analyse détaillée
            st.markdown("""
            <div style="text-align: center; margin-bottom: 20px; padding: 20px; background: linear-gradient(to right, rgba(0, 121, 255, 0.1), rgba(0, 223, 162, 0.1)); border-radius: 15px;">
                <h2 style="margin:0; color: #1E3A8A; font-weight: 700;">🔍 Analyse Détaillée des Rendez-vous</h2>
                <p style="color: #666; margin-top: 5px;">Découvrez des insights avancés avec nos visualisations spécialisées</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Section 1: Métriques avancées
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            st.markdown('<h3 style="text-align: center; margin-bottom: 20px;">📊 Métriques Avancées</h3>', unsafe_allow_html=True)
            
            # Statistiques avancées - calculer quelques métriques intéressantes
            col1, col2, col3, col4 = st.columns(4)
            
            # Stats par région
            regions_stats = data.groupby('region')['nombre_rendez_vous'].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
            regions_stats.columns = ['region', 'total', 'moyenne', 'ecart_type', 'minimum', 'maximum']
            
            # Région avec le plus de RDV
            top_region = regions_stats.loc[regions_stats['total'].idxmax()]
            # Région avec le moins de RDV
            bottom_region = regions_stats.loc[regions_stats['total'].idxmin()]
            # Région avec la plus grande variabilité
            if 'ecart_type' in regions_stats.columns and not regions_stats['ecart_type'].isna().all():
                # Filtrer les NaN avant de chercher le maximum
                filtered_stats = regions_stats.dropna(subset=['ecart_type'])
                if not filtered_stats.empty:
                    variable_region = filtered_stats.loc[filtered_stats['ecart_type'].idxmax()]
                else:
                    variable_region = pd.Series({'region': 'N/A', 'ecart_type': 0})
            else:
                variable_region = pd.Series({'region': 'N/A', 'ecart_type': 0})
            
            # Statistiques globales
            total_rdv = data['nombre_rendez_vous'].sum()
            avg_rdv = total_rdv / data['region'].nunique() if data['region'].nunique() > 0 else 0
            
            # Afficher les métriques dans des cartes modernes
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <span style="font-size: 1.2rem; color: #0079FF;">🔝 Top Région</span>
                    <div class="stat-value">{top_region['region']}</div>
                    <div class="stat-label">{int(top_region['total'])} rendez-vous</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <span style="font-size: 1.2rem; color: #9376E0;">📉 Région Minimum</span>
                    <div class="stat-value">{bottom_region['region']}</div>
                    <div class="stat-label">{int(bottom_region['total'])} rendez-vous</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <span style="font-size: 1.2rem; color: #00DFA2;">📊 Moyenne par Région</span>
                    <div class="stat-value">{avg_rdv:.1f}</div>
                    <div class="stat-label">rendez-vous / région</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                st.markdown(f"""
                <div class="stat-card">
                    <span style="font-size: 1.2rem; color: #FF7676;">📈 Plus Grande Variabilité</span>
                    <div class="stat-value">{variable_region['region']}</div>
                    <div class="stat-label">σ = {variable_region['ecart_type']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Section 2: Distribution des rendez-vous
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            
            # Créer des histogrammes de distribution
            import plotly.figure_factory as ff
            
            st.markdown('<h3 style="text-align: center; margin-bottom: 20px;">📊 Distribution des Rendez-vous</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Préparer les données pour l'histogramme
                region_counts = data.groupby('region')['nombre_rendez_vous'].sum().reset_index()
                region_counts = region_counts.sort_values('nombre_rendez_vous', ascending=False)
                
                try:
                    # Essayer d'utiliser create_distplot (nécessite scipy)
                    import scipy
                    region_values = region_counts['nombre_rendez_vous'].values
                    
                    fig = ff.create_distplot(
                        [region_values], 
                        ['Distribution des RDV par région'], 
                        bin_size=max(int(region_values.max() / 10), 1),
                        curve_type='normal'
                    )
                    
                    fig.update_layout(
                        title="Distribution Statistique des Rendez-vous",
                        xaxis_title="Nombre de Rendez-vous",
                        yaxis_title="Densité",
                        font=dict(size=12),
                        margin=dict(t=60, b=50, l=50, r=50)
                    )
                    
                except (ImportError, ModuleNotFoundError):
                    # Alternative si scipy n'est pas disponible: utiliser un histogramme simple
                    import plotly.express as px
                    
                    fig = px.histogram(
                        region_counts,
                        x='nombre_rendez_vous',
                        nbins=20,
                        marginal='box',
                        title="Distribution Statistique des Rendez-vous",
                        labels={'nombre_rendez_vous': 'Nombre de Rendez-vous'},
                        color_discrete_sequence=['#0079FF']
                    )
                    
                    fig.update_layout(
                        xaxis_title="Nombre de Rendez-vous",
                        yaxis_title="Nombre de Régions",
                        font=dict(size=12),
                        margin=dict(t=60, b=50, l=50, r=50)
                    )
                    
                except Exception as e:
                    # En cas d'erreur inattendue, afficher un message et utiliser un graphique de base
                    st.warning(f"Impossible de générer la distribution avancée: {str(e)}")
                    
                    # Utiliser un graphique à barres basique comme alternative de secours
                    fig = px.bar(
                        region_counts.head(10),
                        x='region',
                        y='nombre_rendez_vous', 
                        title="Top 10 Régions par Nombre de Rendez-vous",
                        color='nombre_rendez_vous',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        xaxis_title="Région",
                        yaxis_title="Nombre de Rendez-vous", 
                        font=dict(size=12)
                    )
                
                st.plotly_chart(fig, use_container_width=True, key="distribution_chart")
                
            with col2:
                from app.components.visualizations import display_donut_chart
                # Utiliser un graphique en anneau pour les proportions
                region_data = data.groupby('region')['nombre_rendez_vous'].sum().reset_index()
                # Limiter à top 5 pour la lisibilité
                top_regions = region_data.sort_values('nombre_rendez_vous', ascending=False).head(5)
                # Ajouter une catégorie 'Autres' pour le reste
                other_sum = region_data[~region_data['region'].isin(top_regions['region'])]['nombre_rendez_vous'].sum()
                if other_sum > 0:
                    other_row = pd.DataFrame({'region': ['Autres'], 'nombre_rendez_vous': [other_sum]})
                    top_regions = pd.concat([top_regions, other_row])
                
                display_donut_chart(top_regions, title="Répartition des Top 5 Régions")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Section 3: Visualisations temporelles avancées
            if 'jour_semaine' in data.columns:
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                st.markdown('<h3 style="text-align: center; margin-bottom: 20px;">📅 Analyse Temporelle Avancée</h3>', unsafe_allow_html=True)
                
                # Afficher les détails de la semaine
                display_details_section(data, filters, prefix="detailed_")
                
                # Afficher le graphique radar
                from app.components.visualizations import display_radar_chart
                display_radar_chart(data, title="Profil des régions par jour de la semaine")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Section 4: Analyse comparative des régions
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            st.markdown('<h3 style="text-align: center; margin-bottom: 20px;">🔄 Analyse Comparative des Régions</h3>', unsafe_allow_html=True)
            
            # Afficher les filtres actifs
            active_filters = []
            if filters.get('region'):
                active_filters.append(f"🔍 Région: {filters['region']}")
            if filters.get('date'):
                active_filters.append(f"📅 Date: {filters['date']}")
            if filters.get('time_filter_type') and filters.get('time_filter_value'):
                unit = "minutes" if filters['time_filter_type'] == "minute" else "heures"
                active_filters.append(f"⏱️ {filters['time_filter_value']} {unit}")
            
            if active_filters:
                st.markdown(f"""
                <div style="background-color: #f0f8ff; padding: 10px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #0079FF;">
                    <p style="margin:0; font-weight: 500;">Filtres actifs: {' | '.join(active_filters)}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Créer un tableau de comparaison des régions
            import plotly.graph_objects as go
            from app.components.visualizations import generate_color_palette
            
            # Préparer les données (en utilisant les mêmes données filtrées)
            region_comparison = data.groupby('region')['nombre_rendez_vous'].agg(['sum', 'count', 'mean']).reset_index()
            region_comparison.columns = ['region', 'total', 'count', 'average']
            region_comparison['percentage'] = region_comparison['total'] / region_comparison['total'].sum() * 100
            region_comparison = region_comparison.sort_values('total', ascending=False)
            
            # Créer le graphique en barres
            colors = generate_color_palette(len(region_comparison))
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=region_comparison['region'],
                y=region_comparison['total'],
                name='Total RDV',
                marker_color=colors,
                text=region_comparison['total'].apply(lambda x: f"{x:,.0f}"),
                textposition='auto'
            ))
            
            # Ajouter une ligne pour le pourcentage
            fig.add_trace(go.Scatter(
                x=region_comparison['region'],
                y=region_comparison['percentage'],
                mode='lines+markers+text',
                name='Pourcentage',
                yaxis='y2',
                line=dict(color='#FF0060', width=2),
                marker=dict(size=8, color='#FF0060'),
                text=region_comparison['percentage'].apply(lambda x: f"{x:.1f}%"),
                textposition='top center'
            ))
            
            # Configurer la mise en page
            fig.update_layout(
                title="Comparaison des Régions par Nombre de Rendez-vous",
                xaxis=dict(title="Région"),
                yaxis=dict(title="Nombre de RDV", side='left'),
                yaxis2=dict(title="Pourcentage (%)", side='right', overlaying='y', showgrid=False),
                legend=dict(x=0.01, y=0.99, orientation='h'),
                font=dict(size=12),
                margin=dict(t=60, b=50, l=50, r=50),
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True, key="region_comparison_chart")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.warning("Aucune donnée disponible pour l'analyse détaillée.")
    
    # Tab 3: Trends and Insights
    with tabs[2]:
        if not data.empty:
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            st.subheader("📈 Analyse des tendances")
            
            # Afficher les filtres actifs
            active_filters = []
            if filters.get('region'):
                active_filters.append(f"🔍 Région: {filters['region']}")
            if filters.get('date'):
                active_filters.append(f"📅 Date: {filters['date']}")
            if filters.get('time_filter_type') and filters.get('time_filter_value'):
                unit = "minutes" if filters['time_filter_type'] == "minute" else "heures"
                active_filters.append(f"⏱️ {filters['time_filter_value']} {unit}")
            if filters.get('month_filter'):
                active_filters.append(f"📅 Mois: {MOIS_ANNEE.get(filters['month_filter'], filters['month_filter'])}")
            
            if active_filters:
                st.markdown(f"""
                <div style="background-color: #f0f8ff; padding: 10px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #0079FF;">
                    <p style="margin:0; font-weight: 500;">Filtres actifs: {' | '.join(active_filters)}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add donut chart for regions distribution
            from app.components.visualizations import display_donut_chart, display_combined_chart
            
            col1, col2 = st.columns(2)
            with col1:
                # Group by region to avoid duplicates
                region_data = data.groupby('region')['nombre_rendez_vous'].sum().reset_index()
                # Get top 10 regions for better visualization
                top_regions = region_data.sort_values('nombre_rendez_vous', ascending=False).head(10)
                display_donut_chart(top_regions, value_col='nombre_rendez_vous', name_col='region', title="Top 10 régions par nombre de rendez-vous")
            
            with col2:
                # Display combined chart with Pareto principle
                display_combined_chart(data, title="Analyse Pareto des rendez-vous par région")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Aucune donnée disponible pour l'analyse des tendances.")
    
    # Tab 4: Region Map
    with tabs[3]:
        if not data.empty:
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            
            # Afficher les filtres actifs
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
                <div style="background-color: #f0f8ff; padding: 10px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #0079FF;">
                    <p style="margin:0; font-weight: 500;">Filtres actifs: {' | '.join(active_filters)}</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.subheader("🗺️ Carte des Rendez-vous par Région")
            from app.components.visualizations import display_region_map
            
            # Group by region to get aggregated data
            region_data = data.groupby('region')['nombre_rendez_vous'].sum().reset_index()
            display_region_map(region_data, title="Carte des rendez-vous par région")
            
            st.markdown('</div>', unsafe_allow_html=True)
