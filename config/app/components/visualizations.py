"""
Visualizations component module.
Provides functions to create and display modern visualizations in the dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple, Any
import colorsys

from utils.translations import JOURS_ORDRE

# Color palette for consistent styling
COLORS = {
    'primary': '#0079FF',
    'secondary': '#00DFA2',
    'accent': '#F6FA70',
    'warning': '#FF0060',
    'background': '#f0f2f6',
    'text': '#212529',
    'green': '#00DFA2',
    'red': '#FF0060',
    'blue': '#0079FF',
    'yellow': '#F6FA70',
    'purple': '#9376E0',
    'orange': '#FF7676',
}

# Generate a color palette for charts
def generate_color_palette(n_colors):
    """
    Generate a color palette with n colors.
    
    Args:
        n_colors (int): Number of colors to generate
        
    Returns:
        List[str]: List of hex color codes
    """
    # Use predefined color palette for small numbers of colors
    if n_colors <= 10:
        return pc.qualitative.D3[:n_colors]
    
    # For larger palettes, generate custom colors
    palette = []
    base_colors = [COLORS['blue'], COLORS['green'], COLORS['purple'], 
                   COLORS['orange'], COLORS['yellow']]
    
    # Add base colors
    palette.extend(base_colors[:min(len(base_colors), n_colors)])
    
    # Generate additional colors if needed
    if n_colors > len(base_colors):
        for i in range(n_colors - len(base_colors)):
            hue = i / (n_colors - len(base_colors))
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            palette.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
    
    return palette

def display_kpi_cards(data: pd.DataFrame):
    """
    Display modern KPI cards for a dashboard
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
    """
    if data.empty:
        st.warning("Aucune donnée disponible pour les KPIs.")
        return
    
    total_rdv = data['nombre_rendez_vous'].sum()
    regions_count = data['region'].nunique()
    
    # Calculate additional metrics depending on data structure
    if 'jour_semaine' in data.columns:
        # Analyze by day of week
        days_count = data['jour_semaine'].nunique()
        
        # Find top day
        day_counts = data.groupby('jour_semaine')['nombre_rendez_vous'].sum()
        top_day = day_counts.idxmax() if not day_counts.empty else 'N/A'
        top_day_value = day_counts.max() if not day_counts.empty else 0
        
        # Find top region
        region_counts = data.groupby('region')['nombre_rendez_vous'].sum()
        top_region = region_counts.idxmax() if not region_counts.empty else 'N/A'
        top_region_value = region_counts.max() if not region_counts.empty else 0
        
        # Create a 4-column layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 10px; padding: 20px; text-align: center; 
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); height: 100%;">
                <div style="font-size: 0.9rem; color: #777; margin-bottom: 5px;">Total Rendez-vous</div>
                <div style="font-size: 2rem; font-weight: 700; color: #0079FF; margin-bottom: 5px;">{total_rdv:,}</div>
                <div style="font-size: 0.8rem; color: #555; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #0079FF; margin-right: 5px;">📊</span> Tous départements confondus
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 10px; padding: 20px; text-align: center; 
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); height: 100%;">
                <div style="font-size: 0.9rem; color: #777; margin-bottom: 5px;">Régions Actives</div>
                <div style="font-size: 2rem; font-weight: 700; color: #00DFA2; margin-bottom: 5px;">{regions_count}</div>
                <div style="font-size: 0.8rem; color: #555; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #00DFA2; margin-right: 5px;">🌍</span> Départements avec rendez-vous
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 10px; padding: 20px; text-align: center; 
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); height: 100%;">
                <div style="font-size: 0.9rem; color: #777; margin-bottom: 5px;">Jour le Plus Actif</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #FF7676; margin-bottom: 5px;">{top_day}</div>
                <div style="font-size: 0.8rem; color: #555; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #FF7676; margin-right: 5px;">📅</span> {top_day_value:,} rendez-vous
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 10px; padding: 20px; text-align: center; 
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); height: 100%;">
                <div style="font-size: 0.9rem; color: #777; margin-bottom: 5px;">Région Top Performer</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #9376E0; margin-bottom: 5px;">{top_region}</div>
                <div style="font-size: 0.8rem; color: #555; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #9376E0; margin-right: 5px;">🏆</span> {top_region_value:,} rendez-vous
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Simplified version for data without day analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 10px; padding: 20px; text-align: center; 
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); height: 100%;">
                <div style="font-size: 0.9rem; color: #777; margin-bottom: 5px;">Total Rendez-vous</div>
                <div style="font-size: 2rem; font-weight: 700; color: #0079FF; margin-bottom: 5px;">{total_rdv:,}</div>
                <div style="font-size: 0.8rem; color: #555; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #0079FF; margin-right: 5px;">📊</span> Tous départements confondus
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 10px; padding: 20px; text-align: center; 
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); height: 100%;">
                <div style="font-size: 0.9rem; color: #777; margin-bottom: 5px;">Régions Actives</div>
                <div style="font-size: 2rem; font-weight: 700; color: #00DFA2; margin-bottom: 5px;">{regions_count}</div>
                <div style="font-size: 0.8rem; color: #555; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #00DFA2; margin-right: 5px;">🌍</span> Départements avec rendez-vous
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_region_bar_chart(data: pd.DataFrame, title: str = "Rendez-vous par région"):
    """
    Display a bar chart for appointments by region.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        title (str, optional): Chart title. Defaults to "Rendez-vous par région".
    """
    if data.empty or 'region' not in data.columns:
        st.warning("Aucune donnée disponible pour la visualisation par région.")
        return
    
    # Group by region if needed
    if 'region' in data.columns and 'nombre_rendez_vous' in data.columns:
        plot_data = data.groupby('region')['nombre_rendez_vous'].sum().reset_index()
    else:
        plot_data = data
        
    # Create the bar chart
    fig = px.bar(
        plot_data, 
        x='region', 
        y='nombre_rendez_vous',
        color='nombre_rendez_vous',
        title=title,
        color_continuous_scale='Viridis',
        labels={'region': 'Région', 'nombre_rendez_vous': 'Nombre de RDV'}
    )
    
    fig.update_layout(
        xaxis_title="Région",
        yaxis_title="Nombre de rendez-vous",
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_month_bar_chart(data: pd.DataFrame, title: str = "Rendez-vous par mois"):
    """
    Display a bar chart for appointments by month.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        title (str, optional): Chart title. Defaults to "Rendez-vous par mois".
    """
    if data.empty or ('mois' not in data.columns and 'annee' not in data.columns):
        st.warning("Aucune donnée disponible pour la visualisation par mois.")
        return
    
    # Create the visualization
    fig = px.bar(
        data, 
        x='region', 
        y='nombre_rendez_vous', 
        color='annee' if 'annee' in data.columns else None,
        title=title,
        labels={
            'region': 'Région', 
            'nombre_rendez_vous': 'Nombre de RDV', 
            'annee': 'Année'
        }
    )
    
    fig.update_layout(
        xaxis_title="Région",
        yaxis_title="Nombre de rendez-vous",
        legend_title="Année" if 'annee' in data.columns else None,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_week_bar_chart(data: pd.DataFrame, title: str = "Rendez-vous par semaine"):
    """
    Display a bar chart for appointments by week.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        title (str, optional): Chart title. Defaults to "Rendez-vous par semaine".
    """
    if data.empty or 'jour' not in data.columns:
        st.warning("Aucune donnée disponible pour la visualisation par semaine.")
        return
    
    # Create the visualization
    fig = px.bar(
        data, 
        x='region', 
        y='nombre_rendez_vous', 
        color='jour',
        title=title,
        category_orders={"jour": JOURS_ORDRE},
        labels={
            'region': 'Région', 
            'nombre_rendez_vous': 'Nombre de RDV', 
            'jour': 'Jour'
        }
    )
    
    fig.update_layout(
        xaxis_title="Région",
        yaxis_title="Nombre de rendez-vous",
        legend_title="Jour",
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_timeslot_bar_chart(data: pd.DataFrame):
    """
    Display a bar chart for timeslot analysis.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
    """
    if data.empty or 'plage_horaire' not in data.columns:
        st.warning("Aucune donnée disponible pour l'analyse par plage horaire.")
        return
    
    # Créer le graphique à barres pour les plages horaires
    fig = px.bar(
        data, 
        x='plage_horaire', 
        y='nombre_rendez_vous', 
        color='region',
        title=f"Rendez-vous par plage horaire",
        labels={'plage_horaire': 'Plage horaire', 'nombre_rendez_vous': 'Nombre de RDV', 'region': 'Région'}
    )
    
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def display_week_bar_chart(data: pd.DataFrame, title: str = "RDV par semaine"):
    """
    Display a bar chart for appointments by week.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        title (str, optional): Chart title. Defaults to "RDV par semaine".
    """
    if data.empty or 'jour' not in data.columns:
        st.warning("Aucune donnée disponible pour la visualisation par semaine.")
        return

    fig = px.bar(
        data, 
        x='region', 
        y='nombre_rendez_vous', 
        color='jour',
        title=title,
        category_orders={"jour": JOURS_ORDRE},
        labels={
            'region': 'Région', 
            'nombre_rendez_vous': 'Nombre de RDV', 
            'jour': 'Jour'
        }
    )
    
    fig.update_layout(
        xaxis_title="Région",
        yaxis_title="Nombre de rendez-vous",
        legend_title="Jour",
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_day_total_bar_chart(daily_totals: pd.DataFrame, title: str = "Nombre total de rendez-vous par jour de la semaine"):
    """
    Display a bar chart for total appointments by day of the week.
    
    Args:
        daily_totals (pd.DataFrame): DataFrame containing daily totals
        title (str, optional): Chart title. Defaults to "Nombre total de rendez-vous par jour de la semaine".
    """
    if daily_totals.empty or 'jour' not in daily_totals.columns:
        st.warning("Aucune donnée disponible pour la visualisation par jour de la semaine.")
        return

    fig = px.bar(
        daily_totals,
        x='jour',
        y='total',
        color='total',
        text='total',
        title=title,
        labels={
            'jour': 'Jour de la semaine', 
            'total': 'Nombre total de RDV'
        },
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=JOURS_ORDRE),
        yaxis_title="Nombre de rendez-vous",
        showlegend=False,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_heatmap(heatmap_data: pd.DataFrame, title: str = "Intensité des rendez-vous par jour de la semaine et par région"):
    """
    Display a heatmap for appointments by day and region.
    
    Args:
        heatmap_data (pd.DataFrame): DataFrame containing heatmap data
        title (str, optional): Chart title. Defaults to "Intensité des rendez-vous par jour de la semaine et par région".
    """
    if heatmap_data.empty or 'jour' not in heatmap_data.columns:
        st.warning("Aucune donnée disponible pour la carte thermique.")
        return

    heat_fig = px.density_heatmap(
        heatmap_data,
        x='jour',
        y='region',
        z='nombre',
        category_orders={"jour": JOURS_ORDRE},
        color_continuous_scale='Viridis',
        labels={
            'jour': 'Jour de la semaine', 
            'region': 'Région', 
            'nombre': 'Nombre de RDV'
        }
    )

    heat_fig.update_layout(
        title=title,
        xaxis_title="Jour de la semaine",
        yaxis_title="Région",
        font=dict(size=12)
    )

    st.plotly_chart(heat_fig, use_container_width=True)


def display_top_regions_cards(top_regions: Dict[str, Dict[str, Any]]):
    """
    Display cards showing top regions for each day of the week.
    
    Args:
        top_regions (Dict[str, Dict[str, Any]]): Dictionary with top region data by day
    """
    if not top_regions:
        st.warning("Aucune donnée disponible pour les meilleures régions.")
        return
        
    st.write("### 🔍 Régions avec le plus de rendez-vous par jour de la semaine")

    # Create cards to display the top regions
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
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
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


def display_donut_chart(data: pd.DataFrame, value_col: str = 'nombre_rendez_vous', name_col: str = 'region', 
                        title: str = "Répartition des rendez-vous par région"):
    """
    Display a donut chart for appointment distribution.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        value_col (str): Column name for values
        name_col (str): Column name for categories
        title (str): Chart title
    """
    if data.empty or value_col not in data.columns or name_col not in data.columns:
        st.warning(f"Aucune donnée disponible pour le graphique en anneau.")
        return
    
    # Generate a color palette for the chart
    colors = generate_color_palette(len(data))
    
    # Create a donut chart
    fig = go.Figure(data=[go.Pie(
        labels=data[name_col],
        values=data[value_col],
        hole=0.5,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title=title,
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=60, b=60, l=60, r=60)
    )
    
    # Add a center text showing total
    total = data[value_col].sum()
    fig.add_annotation(
        text=f"<b>Total<br>{total}</b>",
        font=dict(size=16),
        showarrow=False,
        x=0.5,
        y=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_region_map(data: pd.DataFrame, value_col: str = 'nombre_rendez_vous', 
                       title: str = "Carte des rendez-vous par région"):
    """
    Display a simplified region map for France using a scatter plot.
    This is a simplified version as a full choropleth map would require GeoJSON data.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        value_col (str): Column name for values
        title (str): Chart title
    """
    if data.empty or 'region' not in data.columns:
        st.warning(f"Aucune donnée disponible pour la carte des régions.")
        return
    
    # Create a stylish container for the visualization
    st.markdown(f"""
    <div style="
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    ">
        <h3 style="text-align: center; color: {COLORS['text']};">{title}</h3>
        <p style="text-align: center; color: gray; margin-bottom: 20px;">
            Visualisation de la distribution des rendez-vous par région
        </p>
    """, unsafe_allow_html=True)
    
    # Display a placeholder message
    st.info("🗺️ Une carte choroplèthe détaillée des régions pourrait être intégrée ici en utilisant les codes géographiques des régions françaises et une bibliothèque comme folium ou en utilisant GeoJSON avec Plotly.")
    
    # Display a bar chart as an alternative
    fig = px.bar(
        data.sort_values(value_col, ascending=False), 
        x='region', 
        y=value_col,
        color=value_col,
        color_continuous_scale='YlOrRd',
        title="Distribution des rendez-vous par code région",
        labels={
            'region': 'Code Région',
            value_col: 'Nombre de rendez-vous'
        },
        text=value_col
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        xaxis_title="Code Région",
        yaxis_title="Nombre de rendez-vous",
        coloraxis_showscale=True,
        font=dict(size=12),
        margin=dict(t=80, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def display_radar_chart(data: pd.DataFrame, title: str = "Comparaison des rendez-vous par jour"):
    """
    Display a radar chart for comparing appointments by day of the week across regions.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        title (str): Chart title
    """
    if data.empty or 'jour_semaine' not in data.columns:
        st.warning(f"Aucune donnée disponible pour le graphique radar.")
        return
    
    # Prepare data for radar chart - get top 5 regions by total appointments
    pivot_data = data.pivot_table(
        index='region',
        columns='jour_semaine',
        values='nombre_rendez_vous',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Add total column
    jour_cols = [j for j in JOURS_ORDRE if j in pivot_data.columns]
    pivot_data['Total'] = pivot_data[jour_cols].sum(axis=1)
    
    # Get top 5 regions
    top_regions = pivot_data.sort_values('Total', ascending=False).head(5)['region'].tolist()
    filtered_data = pivot_data[pivot_data['region'].isin(top_regions)]
    
    # Create radar chart
    fig = go.Figure()
    
    colors = generate_color_palette(len(filtered_data))
    
    for i, (_, row) in enumerate(filtered_data.iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=[row[day] for day in jour_cols],
            theta=jour_cols,
            fill='toself',
            name=f"Région {row['region']}",
            line_color=colors[i],
            fillcolor='rgba(0, 121, 255, 0.3)'  # Using rgba for proper transparency
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=True
            )
        ),
        showlegend=True,
        title=title,
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=80, b=80, l=60, r=60)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_kpi_cards(data: pd.DataFrame):
    """
    Display modern KPI cards with key metrics.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
    """
    if data.empty:
        st.warning("Aucune donnée disponible pour les indicateurs KPI.")
        return
    
    # Calculate KPIs
    total_rdv = data['nombre_rendez_vous'].sum()
    total_regions = data['region'].nunique()
    
    # Find top region
    top_region_data = data.groupby('region')['nombre_rendez_vous'].sum().reset_index()
    top_region = top_region_data.loc[top_region_data['nombre_rendez_vous'].idxmax()]
    
    # Find average per region
    avg_per_region = total_rdv / total_regions if total_regions > 0 else 0
    
    # Create modern KPI cards
    st.markdown("""
    <style>
    .kpi-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        text-align: center;
        height: 100%;
    }
    .kpi-icon {
        font-size: 24px;
        margin-bottom: 10px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .kpi-label {
        font-size: 14px;
        color: gray;
    }
    .kpi-blue { color: #0079FF; }
    .kpi-green { color: #00DFA2; }
    .kpi-purple { color: #9376E0; }
    .kpi-orange { color: #FF7676; }
    </style>
    """, unsafe_allow_html=True)
    
    # Display cards in a row
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon kpi-blue">📊</div>
            <div class="kpi-value">{total_rdv:,}</div>
            <div class="kpi-label">Total des rendez-vous</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon kpi-green">🏙️</div>
            <div class="kpi-value">{total_regions}</div>
            <div class="kpi-label">Régions actives</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon kpi-purple">🏆</div>
            <div class="kpi-value">{top_region['region']}</div>
            <div class="kpi-label">Région la plus active<br>({top_region['nombre_rendez_vous']:,} RDV)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon kpi-orange">📈</div>
            <div class="kpi-value">{avg_per_region:.1f}</div>
            <div class="kpi-label">Moyenne de RDV par région</div>
        </div>
        """, unsafe_allow_html=True)


def display_combined_chart(data: pd.DataFrame, title: str = "Analyse combinée des rendez-vous"):
    """
    Display a combined chart with bars and line for comprehensive analysis.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        title (str): Chart title
    """
    if data.empty or 'region' not in data.columns:
        st.warning(f"Aucune donnée disponible pour le graphique combiné.")
        return
    
    # Prepare data
    region_counts = data.groupby('region')['nombre_rendez_vous'].sum().reset_index()
    region_counts = region_counts.sort_values('nombre_rendez_vous', ascending=False)
    
    # Calculate cumulative percentage
    total = region_counts['nombre_rendez_vous'].sum()
    region_counts['cumulative'] = region_counts['nombre_rendez_vous'].cumsum()
    region_counts['cumulative_percent'] = region_counts['cumulative'] / total * 100
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bars
    fig.add_trace(
        go.Bar(
            x=region_counts['region'],
            y=region_counts['nombre_rendez_vous'],
            name="Nombre de RDV",
            marker_color=COLORS['blue'],
            opacity=0.8
        ),
        secondary_y=False,
    )
    
    # Add line
    fig.add_trace(
        go.Scatter(
            x=region_counts['region'],
            y=region_counts['cumulative_percent'],
            name="% Cumulé",
            marker_color=COLORS['green'],
            line=dict(width=3)
        ),
        secondary_y=True,
    )
    
    # Configure layout
    fig.update_layout(
        title=title,
        xaxis_title="Région",
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=80, b=50, l=50, r=50),
        barmode='group'
    )
    
    # Add axis titles
    fig.update_yaxes(
        title_text="Nombre de rendez-vous", 
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Pourcentage cumulé", 
        secondary_y=True
    )
    
    # Add a horizontal line at 80% for Pareto principle
    fig.add_shape(
        type="line",
        x0=region_counts['region'].iloc[0],
        y0=80,
        x1=region_counts['region'].iloc[-1],
        y1=80,
        line=dict(
            color="red",
            width=2,
            dash="dash",
        ),
        secondary_y=True
    )
    
    # Add annotation for 80% line
    fig.add_annotation(
        x=region_counts['region'].iloc[len(region_counts)//2],
        y=83,
        text="Seuil 80%",
        showarrow=False,
        font=dict(
            size=12,
            color="red"
        ),
        secondary_y=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
