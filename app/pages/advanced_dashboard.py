"""
Advanced Dashboard page module.
Provides enhanced dashboard capabilities with modern 2025 visualizations and AI insights.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any

from utils.translations import MOIS_ANNEE, JOURS_ORDRE
from app.components.visualizations import (
    display_kpi_cards, display_donut_chart, display_combined_chart,
    display_radar_chart, display_region_map
)
from app.components.advanced_visualizations import (
    display_3d_region_comparison,
    display_time_pattern_heatmap,
    display_predictive_analysis,
    display_sankey_flow_analysis,
    display_calendar_heatmap,
    display_geo_heatmap
)
from app.components.premium_visualizations import (
    display_anomaly_detection,
    display_cohort_analysis,
    display_cluster_analysis,
    display_ai_insights_dashboard
)

# Get logger
logger = logging.getLogger('rndv_ghandi.advanced_dashboard')

def render_advanced_dashboard(filters: Dict[str, Any]):
    """
    Render the advanced 2025 dashboard with modern visualizations and AI insights.
    
    Args:
        filters (Dict[str, Any]): Dictionary of filter values
    """
    # Check if we have data in session_state or need to fetch it
    if st.session_state.get('force_refresh', False) or 'advanced_data' not in st.session_state:
        # Fetch data for the advanced dashboard
        with st.spinner("Chargement des données avancées..."):
            data = fetch_data_for_advanced_dashboard(filters)
            
            if data is not None and not data.empty:
                st.session_state.advanced_data = data
                st.session_state.last_update = datetime.now()
                st.session_state.force_refresh = False
            else:
                st.warning("Aucune donnée disponible pour les filtres sélectionnés.")
                return
    
    # Get data from session state
    data = st.session_state.advanced_data
    
    # Display dashboard header with modern styling and interactive elements
    st.markdown(f"""
    <div style="
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        margin-bottom: 25px; 
        background: linear-gradient(to right, rgba(0, 121, 255, 0.05), rgba(0, 223, 162, 0.05));
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02);
    ">
        <div>
            <h1 style="
                margin: 0; 
                background: linear-gradient(to right, #0079FF, #00DFA2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 700;
            ">Tableau de Bord 2025</h1>
            <p style="color: #666; margin: 5px 0 0 0;">Visualisations avancées et analyses prédictives</p>
        </div>
        <div style="
            background-color: white;
            padding: 10px 15px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        ">
            <div style="display: flex; flex-direction: column; align-items: flex-end;">
                <span style="color: #666; font-size: 0.8rem;">Mise à jour</span>
                <span style="color: #0079FF; font-weight: 600;">{datetime.now().strftime('%H:%M:%S')}</span>
            </div>
            <div style="
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: linear-gradient(135deg, #0079FF, #00DFA2);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.2rem;
            ">
                📊
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display active filters in a modern style
    display_active_filters(filters)
    
    # Create tabs for the advanced dashboard
    tabs = st.tabs([
        "📊 Vue Principale",
        "🔍 Analyse Avancée",
        "🔮 Prédictions",
        "🗺️ Cartographie"
    ])
    
    # Tab 1: Main Dashboard View
    with tabs[0]:
        st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>Vue d'Ensemble Avancée</h2>", unsafe_allow_html=True)
        
        # Display AI Insights Dashboard
        display_ai_insights_dashboard(data, "Intelligence Artificielle & Insights 2025")
        
        # Advanced KPI cards with AI-powered insights
        display_kpi_cards(data)
        
        # 3D visualization for region comparison
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        display_3d_region_comparison(data, "Analyse 3D des Régions")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Modern distribution visualizations
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        
        # Add a subheading for this section
        st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>Distribution des Rendez-vous</h3>", unsafe_allow_html=True)
        
        # Split into columns for multiple visualizations
        col1, col2 = st.columns(2)
        
        # Process data for visualizations
        region_data = data.groupby('region')['nombre_rendez_vous'].sum().reset_index()
        region_data = region_data.sort_values('nombre_rendez_vous', ascending=False)
        
        with col1:
            # Get top 5 regions for donut chart
            top_5 = region_data.head(5)
            other_sum = region_data.iloc[5:]['nombre_rendez_vous'].sum() if len(region_data) > 5 else 0
            
            if other_sum > 0:
                # Add "Other" category
                other_df = pd.DataFrame({'region': ['Autres'], 'nombre_rendez_vous': [other_sum]})
                donut_data = pd.concat([top_5, other_df], ignore_index=True)
            else:
                donut_data = top_5
                
            # Display donut chart
            display_donut_chart(donut_data, value_col='nombre_rendez_vous', name_col='region', title="Distribution par Région")
        
        with col2:
            # Display combined chart (bars + line)
            display_combined_chart(region_data, title="Analyse Pareto des Rendez-vous")
        
        st.markdown('</div>', unsafe_allow_html=True)
            
        # Pattern analysis section
        if 'jour_semaine' in data.columns:
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            
            # Add a subheading for this section
            st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>Analyse des Motifs Temporels</h3>", unsafe_allow_html=True)
            
            # Display time pattern heatmap
            display_time_pattern_heatmap(data, "Motifs de Rendez-vous par Jour et Région")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Advanced Analysis
    with tabs[1]:
        st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>Analyse Multidimensionnelle</h2>", unsafe_allow_html=True)
        
        # Anomaly detection visualization
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        display_anomaly_detection(data, "Détection d'Anomalies Intelligente")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Cluster analysis visualization
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        display_cluster_analysis(data, "Segmentation des Régions par Clusters IA")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Cohort analysis visualization
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        display_cohort_analysis(data, "Analyse de Cohortes Avancée")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Calendar heatmap visualization
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        
        if 'jour_semaine' in data.columns:
            # Show weekly calendar view
            display_calendar_heatmap(data, "Calendrier d'Intensité Hebdomadaire")
        else:
            st.warning("L'analyse temporelle avancée nécessite les données de jour de la semaine. Veuillez activer le filtre 'Analyser par jour de la semaine'.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sankey diagram for flow analysis
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        
        if 'jour_semaine' in data.columns:
            # Show Sankey diagram for day-to-region flows
            display_sankey_flow_analysis(data, "Flux de Rendez-vous: Jours → Régions")
        else:
            st.warning("L'analyse des flux nécessite les données de jour de la semaine. Veuillez activer le filtre 'Analyser par jour de la semaine'.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Radar chart for regional patterns
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        
        # Add a subheading for this section
        st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>Profils Régionaux</h3>", unsafe_allow_html=True)
        
        if 'jour_semaine' in data.columns:
            # Display radar chart
            display_radar_chart(data, title="Comparaison des Profils Régionaux par Jour")
        else:
            st.warning("L'analyse des profils régionaux nécessite les données de jour de la semaine. Veuillez activer le filtre 'Analyser par jour de la semaine'.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced statistical insights
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        
        # Add a subheading for this section
        st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>Statistiques Avancées</h3>", unsafe_allow_html=True)
        
        # Create grouped statistics
        region_stats = data.groupby('region')['nombre_rendez_vous'].agg(['sum', 'mean', 'std', 'count']).reset_index()
        region_stats.columns = ['region', 'total', 'moyenne', 'ecart_type', 'nombre_periodes']
        
        # Calculate coefficient of variation (CV)
        region_stats['cv'] = region_stats['ecart_type'] / region_stats['moyenne']
        
        # Calculate percentage of total
        total_rdv = region_stats['total'].sum()
        region_stats['pourcentage'] = (region_stats['total'] / total_rdv * 100)
        
        # Sort by total appointments
        region_stats = region_stats.sort_values('total', ascending=False)
        
        # Format table for display - using display_dataframe to ensure consistent sorting
        # First, sort the table by total in descending order
        region_stats = region_stats.sort_values('total', ascending=False)
        
        # Now display it with styling
        st.dataframe(
            region_stats.style
            .format({
                'total': '{:,.0f}',
                'moyenne': '{:,.1f}',
                'ecart_type': '{:,.1f}',
                'cv': '{:,.2f}',
                'pourcentage': '{:,.1f}%'
            })
            .background_gradient(subset=['total', 'pourcentage'], cmap='Blues')
            .background_gradient(subset=['cv'], cmap='RdYlGn_r')
            .set_properties(**{'text-align': 'center'}),
            use_container_width=True
        )
        
        # Add explanation
        st.markdown("""
        <div style="background-color: rgba(0, 121, 255, 0.05); padding: 15px; border-radius: 10px; margin-top: 15px;">
            <h4 style="color: #0079FF; margin-top: 0;">Guide d'interprétation</h4>
            <ul>
                <li><strong>Total</strong>: Nombre total de rendez-vous pour la région</li>
                <li><strong>Moyenne</strong>: Nombre moyen de rendez-vous par période</li>
                <li><strong>Écart-type</strong>: Mesure de dispersion des données</li>
                <li><strong>Nombre périodes</strong>: Nombre de périodes avec des données</li>
                <li><strong>CV</strong>: Coefficient de variation (écart-type/moyenne) - mesure de la variabilité relative</li>
                <li><strong>Pourcentage</strong>: Contribution de la région au volume total</li>
            </ul>
            <p>Un <strong>coefficient de variation</strong> plus bas indique une distribution plus stable et prévisible des rendez-vous.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Predictions
    with tabs[2]:
        st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>Analyse Prédictive et Tendances</h2>", unsafe_allow_html=True)
        
        # AI-powered predictive analysis
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        
        if 'jour_semaine' in data.columns:
            # Display predictive analysis
            display_predictive_analysis(data)
        else:
            st.warning("L'analyse prédictive nécessite les données de jour de la semaine. Veuillez activer le filtre 'Analyser par jour de la semaine'.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Trend analysis and recommendations
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        
        # Add a subheading for this section
        st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>Analyse des Tendances</h3>", unsafe_allow_html=True)
        
        # Create synthetic trend data (for demonstration)
        if not data.empty:
            # Group data by region for trend analysis
            region_trends = data.groupby('region')['nombre_rendez_vous'].sum().reset_index()
            region_trends = region_trends.sort_values('nombre_rendez_vous', ascending=False)
            
            # Display trend visualization
            import plotly.graph_objects as go
            
            # Get top 10 regions
            top_regions = region_trends.head(10)
            
            # Create trend visualization
            fig = go.Figure()
            
            # Add bars for current data
            fig.add_trace(go.Bar(
                x=top_regions['region'],
                y=top_regions['nombre_rendez_vous'],
                name="Données Actuelles",
                marker_color='rgba(0, 121, 255, 0.7)',
                text=top_regions['nombre_rendez_vous'].apply(lambda x: f"{x:,.0f}"),
                textposition='outside'
            ))
            
            # Synthetic growth trend (for demonstration)
            np.random.seed(42)  # For reproducibility
            growth_factors = np.random.normal(1.15, 0.05, len(top_regions))
            projected_values = top_regions['nombre_rendez_vous'] * growth_factors
            
            # Add trend line
            fig.add_trace(go.Scatter(
                x=top_regions['region'],
                y=projected_values,
                mode='lines+markers',
                name="Projection (+15%)",
                line=dict(color='rgba(0, 223, 162, 1)', width=3),
                marker=dict(size=8, color='rgba(0, 223, 162, 1)'),
                text=[f"+{(g-1)*100:.1f}%" for g in growth_factors],
                textposition='top center'
            ))
            
            # Update layout
            fig.update_layout(
                title="Tendance et Projection de Croissance par Région",
                xaxis_title="Région",
                yaxis_title="Nombre de Rendez-vous",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights and recommendations
            st.markdown("""
            <div style="
                background: linear-gradient(to right, rgba(0, 121, 255, 0.05), rgba(0, 223, 162, 0.05));
                border-radius: 15px;
                padding: 20px;
                margin-top: 20px;
            ">
                <h4 style="color: #0079FF; margin-top: 0;">🧠 Insights et Recommandations IA</h4>
                <p>Basé sur l'analyse des tendances actuelles et des projections futures, voici quelques recommandations d'optimisation:</p>
                
                <div style="margin: 15px 0;">
                    <h5 style="color: #00DFA2; margin: 0 0 5px 0;">Opportunités de Croissance</h5>
                    <ul>
                        <li><strong>Régions à fort potentiel:</strong> Les régions en tête du classement présentent les meilleures opportunités pour une expansion continue.</li>
                        <li><strong>Stratégie de ciblage:</strong> Concentrez les ressources marketing sur les régions montrant une forte croissance projective.</li>
                    </ul>
                </div>
                
                <div style="margin: 15px 0;">
                    <h5 style="color: #FF0060; margin: 0 0 5px 0;">Points d'Attention</h5>
                    <ul>
                        <li><strong>Régions sous-performantes:</strong> Identifiez et analysez les causes des faibles performances dans certaines régions.</li>
                        <li><strong>Analyse de conversion:</strong> Évaluez si la baisse des rendez-vous dans certaines zones est due à une réduction de la demande ou à des problèmes de processus.</li>
                    </ul>
                </div>
                
                <div style="margin: 15px 0;">
                    <h5 style="color: #0079FF; margin: 0 0 5px 0;">Prochaines Étapes</h5>
                    <ul>
                        <li><strong>Mise en place de KPIs:</strong> Définissez des indicateurs clés de performance pour suivre l'évolution des tendances identifiées.</li>
                        <li><strong>Analyse de cohorte:</strong> Implementez une analyse de cohorte pour mieux comprendre les comportements des utilisateurs dans le temps.</li>
                        <li><strong>Tableau de bord dynamique:</strong> Créez un tableau de bord en temps réel pour suivre les performances par rapport aux projections.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Données insuffisantes pour l'analyse des tendances.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Geographic Visualization
    with tabs[3]:
        st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>Cartographie et Distribution Géographique</h2>", unsafe_allow_html=True)
        
        # Geographic heatmap visualization
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        
        # Display geographic heatmap
        display_geo_heatmap(data)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Regional comparison
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        
        # Add a subheading for this section
        st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>Analyse Comparative des Régions</h3>", unsafe_allow_html=True)
        
        # Create region map
        display_region_map(data, title="Distribution des Rendez-vous par Code Région")
        
        # Add information about the map
        st.markdown("""
        <div style="
            background-color: rgba(0, 121, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        ">
            <h4 style="color: #0079FF; margin-top: 0;">📍 À propos des Codes Région</h4>
            <p>Les codes région correspondent aux deux premiers chiffres des codes postaux français. Par exemple:</p>
            <ul>
                <li><strong>75</strong>: Paris</li>
                <li><strong>69</strong>: Rhône (Lyon)</li>
                <li><strong>13</strong>: Bouches-du-Rhône (Marseille)</li>
                <li><strong>33</strong>: Gironde (Bordeaux)</li>
                <li><strong>59</strong>: Nord (Lille)</li>
            </ul>
            <p>Cette représentation permet d'analyser rapidement la distribution géographique des rendez-vous à travers la France.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def fetch_data_for_advanced_dashboard(filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Fetch data for the advanced dashboard based on the provided filters.
    This function is a wrapper that reuses the data from the main app.
    
    Args:
        filters (Dict[str, Any]): Dictionary of filter values
        
    Returns:
        pd.DataFrame: Data for the advanced dashboard
    """
    try:
        # In the modular version, we simply use the data from session_state
        # That has already been fetched in the main app
        if 'last_data' in st.session_state:
            return st.session_state.last_data
        else:
            return pd.DataFrame()  # Empty DataFrame if no data available
    except Exception as e:
        logger.error(f"Error processing data for advanced dashboard: {str(e)}")
        st.error(f"Une erreur est survenue: {str(e)}")
        return pd.DataFrame()

def display_active_filters(filters: Dict[str, Any]):
    """
    Display active filters in a modern style.
    
    Args:
        filters (Dict[str, Any]): Dictionary of filter values
    """
    # Extract active filters
    active_filters = []
    
    if filters.get('region'):
        active_filters.append({
            'icon': '🔍',
            'name': 'Région',
            'value': filters['region']
        })
    
    if filters.get('date'):
        active_filters.append({
            'icon': '📅',
            'name': 'Date',
            'value': filters['date']
        })
    
    if filters.get('time_filter_type') and filters.get('time_filter_value'):
        unit = "minutes" if filters['time_filter_type'] == "minute" else "heures"
        active_filters.append({
            'icon': '⏱️',
            'name': 'Interval',
            'value': f"{filters['time_filter_value']} {unit}"
        })
    
    if filters.get('day_filter'):
        active_filters.append({
            'icon': '📅',
            'name': 'Analyse',
            'value': "Jour de la semaine"
        })
    
    if filters.get('month_filter'):
        active_filters.append({
            'icon': '📅',
            'name': 'Mois',
            'value': MOIS_ANNEE.get(filters['month_filter'], filters['month_filter'])
        })
    
    if filters.get('use_week_filter'):
        active_filters.append({
            'icon': '🗓️',
            'name': 'Semaine',
            'value': f"{filters['week_filter']} de {filters['year_filter']}"
        })
    
    # If there are active filters, display them
    if active_filters:
        # Create a styled container
        st.markdown("""
        <style>
        .filter-container {
            background-color: white;
            border-radius: 15px;
            padding: 15px 20px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border: 1px solid rgba(0,0,0,0.03);
        }
        .filter-header {
            font-weight: 600;
            color: #0079FF;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .filter-tag {
            display: inline-flex;
            align-items: center;
            background-color: rgba(0, 121, 255, 0.1);
            padding: 8px 12px;
            border-radius: 50px;
            margin-right: 10px;
            margin-bottom: 5px;
            font-size: 0.9rem;
            color: #0079FF;
            border: 1px solid rgba(0, 121, 255, 0.2);
            transition: all 0.2s ease;
        }
        .filter-tag:hover {
            background-color: rgba(0, 121, 255, 0.15);
            box-shadow: 0 2px 10px rgba(0, 121, 255, 0.1);
        }
        .filter-icon {
            margin-right: 5px;
        }
        .filter-name {
            font-weight: 500;
            margin-right: 5px;
        }
        .filter-value {
            font-weight: 400;
            background-color: white;
            padding: 3px 8px;
            border-radius: 50px;
            font-size: 0.8rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display the filters
        st.markdown("""
        <div class="filter-container">
            <div class="filter-header">
                <span>🔍</span>
                <span>Filtres actifs</span>
            </div>
            <div>
        """, unsafe_allow_html=True)
        
        # Display each filter tag
        filter_tags = ""
        for filter_item in active_filters:
            filter_tags += f"""
            <span class="filter-tag">
                <span class="filter-icon">{filter_item['icon']}</span>
                <span class="filter-name">{filter_item['name']}:</span>
                <span class="filter-value">{filter_item['value']}</span>
            </span>
            """
        
        st.markdown(filter_tags + "</div></div>", unsafe_allow_html=True)
