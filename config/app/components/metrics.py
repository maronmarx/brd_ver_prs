"""
Metrics component module.
Provides functions to display metrics in the dashboard.
"""
import streamlit as st
from typing import List, Optional, Dict, Any


def display_metrics(metrics: List[Dict[str, Any]], num_columns: int = 3):
    """
    Display metrics in a row with a specified number of columns.
    
    Args:
        metrics (List[Dict[str, Any]]): List of metric dictionaries with keys 'label', 'value', and optionally 'icon'
        num_columns (int, optional): Number of columns to display. Defaults to 3.
    """
    # Create columns
    cols = st.columns(num_columns)
    
    # Display each metric in a column
    for i, metric in enumerate(metrics):
        col_index = i % num_columns
        with cols[col_index]:
            label = f"{metric.get('icon', '')} {metric['label']}"
            st.metric(label=label, value=metric['value'], delta=metric.get('delta'))


def display_basic_metrics(total_rdv: int, regions_count: int, additional_metrics: Optional[List[Dict[str, Any]]] = None):
    """
    Display basic metrics for the dashboard.
    
    Args:
        total_rdv (int): Total number of appointments
        regions_count (int): Number of regions
        additional_metrics (List[Dict[str, Any]], optional): Additional metrics to display
    """
    metrics = [
        {'icon': '📍', 'label': 'Total RDV', 'value': f"{total_rdv:,}"},
        {'icon': '📊', 'label': 'Régions', 'value': f"{regions_count}"}
    ]
    
    if additional_metrics:
        metrics.extend(additional_metrics)
    
    # Determine the number of columns based on the metrics count
    num_columns = min(len(metrics), 4)  # Max 4 columns to prevent tiny metrics
    
    display_metrics(metrics, num_columns)


def display_time_filter_metrics(total_rdv: int, regions_count: int, time_ranges_count: int):
    """
    Display metrics for time filter view.
    
    Args:
        total_rdv (int): Total number of appointments
        regions_count (int): Number of regions
        time_ranges_count (int): Number of time ranges
    """
    metrics = [
        {'icon': '📍', 'label': 'Total RDV', 'value': f"{total_rdv:,}"},
        {'icon': '📊', 'label': 'Régions', 'value': f"{regions_count}"},
        {'icon': '⏱️', 'label': 'Plages', 'value': f"{time_ranges_count}"}
    ]
    
    display_metrics(metrics)


def display_day_filter_metrics(total_rdv: int, regions_count: int, days_count: int):
    """
    Display metrics for day filter view.
    
    Args:
        total_rdv (int): Total number of appointments
        regions_count (int): Number of regions
        days_count (int): Number of days
    """
    metrics = [
        {'icon': '📍', 'label': 'Total RDV', 'value': f"{total_rdv:,}"},
        {'icon': '📊', 'label': 'Régions', 'value': f"{regions_count}"},
        {'icon': '📅', 'label': 'Jours', 'value': f"{days_count}"}
    ]
    
    display_metrics(metrics)


def display_month_filter_metrics(total_rdv: int, regions_count: int, month: str, years_count: int):
    """
    Display metrics for month filter view.
    
    Args:
        total_rdv (int): Total number of appointments
        regions_count (int): Number of regions
        month (str): Month name
        years_count (int): Number of years
    """
    metrics = [
        {'icon': '📍', 'label': 'Total RDV', 'value': f"{total_rdv:,}"},
        {'icon': '📅', 'label': 'Mois', 'value': month},
        {'icon': '📆', 'label': 'Années', 'value': f"{years_count}"},
        {'icon': '📊', 'label': 'Régions', 'value': f"{regions_count}"}
    ]
    
    display_metrics(metrics, 4)


def display_week_filter_metrics(total_rdv: int, regions_count: int, week_number: int, period: str):
    """
    Display metrics for week filter view.
    
    Args:
        total_rdv (int): Total number of appointments
        regions_count (int): Number of regions
        week_number (int): Week number
        period (str): Date period
    """
    metrics = [
        {'icon': '📍', 'label': 'Total RDV', 'value': f"{total_rdv:,}"},
        {'icon': '📅', 'label': 'Semaine', 'value': f"Semaine {week_number}"},
        {'icon': '📆', 'label': 'Période', 'value': period},
        {'icon': '📊', 'label': 'Régions', 'value': f"{regions_count}"}
    ]
    
    display_metrics(metrics, 4)
