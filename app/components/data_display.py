"""
Data display component module.
Provides functions to format and display data tables in the dashboard.
"""
import streamlit as st
import pandas as pd
from typing import List, Optional

from utils.translations import JOURS_ORDRE


def display_dataframe(df: pd.DataFrame, use_container_width: bool = True):
    """
    Display a formatted DataFrame with styling.
    
    Args:
        df (pd.DataFrame): The DataFrame to display
        use_container_width (bool, optional): Whether to use full container width. Defaults to True.
    """
    if df.empty:
        st.warning("Aucune donnée disponible.")
        return
    
    # Apply styling to the DataFrame
    styled_df = df.style.background_gradient(subset='nombre_rendez_vous', cmap='viridis')
    
    # Format numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    format_dict = {col: '{:,.0f}' for col in numeric_cols}
    styled_df = styled_df.format(format_dict)
    
    # Set cell properties for better visualization
    styled_df = styled_df.set_properties(**{'text-align': 'center', 'border': '1px solid grey'})
    
    # Display the styled DataFrame
    st.dataframe(styled_df, use_container_width=use_container_width)


def display_pivot_table(pivot_df: pd.DataFrame, use_container_width: bool = True):
    """
    Display a formatted pivot table with styling.
    
    Args:
        pivot_df (pd.DataFrame): The pivot DataFrame to display
        use_container_width (bool, optional): Whether to use full container width. Defaults to True.
    """
    if pivot_df.empty:
        st.warning("Aucune donnée disponible pour le tableau pivot.")
        return
    
    # Identify day columns that exist in the pivot table
    jour_cols = [j for j in JOURS_ORDRE if j in pivot_df.columns]
    
    if 'Total' in pivot_df.columns:
        cols_to_style = jour_cols + ['Total']
    else:
        cols_to_style = jour_cols
    
    # Apply styling to the pivot table
    styled_pivot = pivot_df.style.background_gradient(subset=cols_to_style, cmap='viridis')
    
    # Format numeric columns
    format_dict = {col: '{:,.0f}' for col in cols_to_style}
    styled_pivot = styled_pivot.format(format_dict)
    
    # Set cell properties for better visualization
    styled_pivot = styled_pivot.set_properties(**{'text-align': 'center', 'border': '1px solid grey'})
    
    # Display the styled pivot table
    st.dataframe(styled_pivot, use_container_width=use_container_width)


def prepare_heatmap_data(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for the heatmap visualization.
    
    Args:
        pivot_df (pd.DataFrame): Pivot table with days as columns
        
    Returns:
        pd.DataFrame: Reshaped DataFrame suitable for heatmap visualization
    """
    if pivot_df.empty:
        return pd.DataFrame()
    
    # Identify day columns that exist in the pivot table
    jour_cols = [j for j in JOURS_ORDRE if j in pivot_df.columns]
    
    # Prepare data for the heatmap
    heatmap_data = []
    for _, row in pivot_df.iterrows():
        region = row['region']
        for jour in jour_cols:
            heatmap_data.append({
                'region': region,
                'jour': jour,
                'nombre': row[jour]
            })
    
    return pd.DataFrame(heatmap_data)


def prepare_daily_totals(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare daily totals data for bar chart visualization.
    
    Args:
        pivot_df (pd.DataFrame): Pivot table with days as columns
        
    Returns:
        pd.DataFrame: DataFrame with daily totals
    """
    if pivot_df.empty:
        return pd.DataFrame()
    
    # Identify day columns that exist in the pivot table
    jour_cols = [j for j in JOURS_ORDRE if j in pivot_df.columns]
    
    if not jour_cols:
        return pd.DataFrame()
    
    # Calculate total for each day
    daily_totals = pivot_df[jour_cols].sum().reset_index()
    daily_totals.columns = ['jour', 'total']
    
    # Order days according to JOURS_ORDRE
    daily_totals = daily_totals.sort_values('jour', key=lambda x: [JOURS_ORDRE.index(i) for i in x])
    
    return daily_totals
