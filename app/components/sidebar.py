"""
Sidebar component module.
Provides functions to create and manage the sidebar filters.
"""
import streamlit as st
import re
from datetime import datetime
from typing import Dict, Any, Tuple

from utils.translations import MOIS_ANNEE
from services.data_service import appointment_service


def create_sidebar() -> Dict[str, Any]:
    """
    Create the sidebar with all filter options.
    
    Returns:
        Dict[str, Any]: Dictionary containing all selected filter values
    """
    st.sidebar.header("🔍 Filtres")
    
    # Function to validate region - validation supprimée selon la demande
    def validate_region(input_str):
        """Valide le format de la région (validation désactivée)"""
        return input_str.strip() if input_str else ''
    
    # Initialiser les variables de session state pour détecter les changements
    for key in ['previous_region', 'previous_date', 'previous_time_type', 
                'previous_time_value', 'previous_day_filter', 'previous_month_filter',
                'previous_year', 'previous_week', 'previous_use_week']:
        if key not in st.session_state:
            st.session_state[key] = None
    
    # Region filter avec détection de changement
    region_input = st.sidebar.text_input(
        "Entrer la région (2 chiffres ou 00-99)",
        "",
        key="region_input",
        help="Entrez un code de région à 2 chiffres (00-99) ou deux codes séparés par un tiret (ex: 00-99)",
        on_change=lambda: setattr(st.session_state, 'force_refresh', True)
    )
    region = validate_region(region_input)
    
    # Mettre à jour la session state et rafraîchir si nécessaire
    if region != st.session_state.previous_region:
        st.session_state.previous_region = region
        st.session_state.force_refresh = True
    
    # Date filter avec détection de changement
    date = st.sidebar.date_input(
        "Sélectionner la date", 
        None,
        key="date_input",
        on_change=lambda: setattr(st.session_state, 'force_refresh', True)
    )
    date_str = date.strftime('%Y-%m-%d') if date else None
    
    # Create temporal filters section
    st.sidebar.header("📅 Filtres Temporels")
    
    # Time filter options avec détection de changement
    time_filter_type = st.sidebar.radio(
        "Type de filtre horaire", 
        ["Aucun", "Minute", "Heure"], 
        index=0,
        key="time_filter_type",
        on_change=lambda: setattr(st.session_state, 'force_refresh', True)
    )
    
    time_filter_value = None
    if time_filter_type == "Minute":
        time_filter_value = st.sidebar.selectbox(
            "Interval en minutes", 
            [5, 10, 15, 30, 60], 
            index=2,
            key="minute_value",
            on_change=lambda: setattr(st.session_state, 'force_refresh', True)
        )
    elif time_filter_type == "Heure":
        time_filter_value = st.sidebar.selectbox(
            "Interval en heures", 
            [1, 2, 3, 4, 6, 12], 
            index=0,
            key="hour_value",
            on_change=lambda: setattr(st.session_state, 'force_refresh', True)
        )
    
    # Day of week analysis option avec détection de changement
    jour_semaine_choice = st.sidebar.checkbox(
        "Analyser par jour de la semaine", 
        value=False,
        key="day_filter",
        on_change=lambda: setattr(st.session_state, 'force_refresh', True)
    )
    day_filter = True if jour_semaine_choice else None
    
    # Month filter option avec détection de changement
    month_filter = st.sidebar.selectbox(
        "Filtrer par mois",
        ["Tous"] + list(MOIS_ANNEE.values()),
        index=0,
        key="month_filter",
        on_change=lambda: setattr(st.session_state, 'force_refresh', True)
    )
    month_filter_english = None
    if month_filter != "Tous":
        for english, french in MOIS_ANNEE.items():
            if french == month_filter:
                month_filter_english = english
                break
    
    # Week filter section
    st.sidebar.header("🗓️ Filtre par Semaine")
    current_year = datetime.now().year
    current_week = datetime.now().isocalendar()[1]
    
    year_filter = st.sidebar.number_input(
        "Année", 
        min_value=2000, 
        max_value=2100, 
        value=current_year,
        key="year_filter",
        on_change=lambda: setattr(st.session_state, 'force_refresh', True)
    )
    
    week_filter = st.sidebar.number_input(
        "Numéro de semaine", 
        min_value=1, 
        max_value=53, 
        value=current_week,
        key="week_filter",
        on_change=lambda: setattr(st.session_state, 'force_refresh', True)
    )
    
    use_week_filter = st.sidebar.checkbox(
        "Utiliser le filtre par semaine",
        key="use_week_filter",
        on_change=lambda: setattr(st.session_state, 'force_refresh', True)
    )
    
    # Create the filter dictionary
    filters = {
        'region': region if region else None,
        'date': date_str,
        'time_filter_type': time_filter_type.lower() if time_filter_type != "Aucun" else None,
        'time_filter_value': time_filter_value,
        'day_filter': day_filter,
        'month_filter': month_filter_english,
        'week_filter': week_filter if use_week_filter else None,
        'year_filter': year_filter if use_week_filter else None,
        'use_week_filter': use_week_filter
    }
    
    # Add an apply filters button to force refresh
    st.sidebar.markdown("---")
    if st.sidebar.button("📊 Appliquer les filtres", help="Appliquer les filtres et actualiser les données"):
        st.session_state.force_refresh = True
    
    return filters


def get_filter_summary(filters: Dict[str, Any]) -> str:
    """
    Generate a summary of the active filters for display.
    
    Args:
        filters (Dict[str, Any]): Dictionary of filter values
        
    Returns:
        str: Summary of active filters
    """
    summary_parts = []
    
    if filters['region']:
        summary_parts.append(f"Région: {filters['region']}")
    
    if filters['date']:
        summary_parts.append(f"Date: {filters['date']}")
    
    if filters['time_filter_type']:
        unit = 'minutes' if filters['time_filter_type'] == 'minute' else 'heures'
        summary_parts.append(f"Filtre horaire: {filters['time_filter_value']} {unit}")
    
    if filters['day_filter']:
        summary_parts.append("Analyse par jour de la semaine")
    
    if filters['month_filter']:
        for english, french in MOIS_ANNEE.items():
            if english == filters['month_filter']:
                summary_parts.append(f"Mois: {french}")
                break
    
    if filters['use_week_filter']:
        summary_parts.append(f"Semaine {filters['week_filter']} de {filters['year_filter']}")
    
    if not summary_parts:
        return "Aucun filtre actif"
    
    return " | ".join(summary_parts)
