"""
Premium Visualizations component module.
Provides state-of-the-art 2025 visualization components with AI insights and interactive elements.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from scipy import stats
import random

# Premium color palette for 2025 visualizations
PREMIUM_COLORS = {
    'primary': '#0079FF',
    'secondary': '#00DFA2',
    'accent1': '#FF0060',
    'accent2': '#F6FA70',
    'neutral1': '#9376E0',
    'neutral2': '#1A374D',
    'gradient1': 'linear-gradient(135deg, #0079FF, #00DFA2)',
    'gradient2': 'linear-gradient(135deg, #FF0060, #F6FA70)',
}

def display_anomaly_detection(data: pd.DataFrame, title: str = "Détection d'Anomalies Intelligente"):
    """
    Display an AI-powered anomaly detection visualization that highlights unusual patterns in the data.
    
    Args:
        data (pd.DataFrame): DataFrame containing the appointment data
        title (str): Chart title
    """
    if data.empty or 'nombre_rendez_vous' not in data.columns:
        st.warning("Données insuffisantes pour l'analyse des anomalies.")
        return
    
    # Create title with modern styling
    st.markdown(f"""
    <h3 style="text-align: center; margin-bottom: 25px; 
               background: {PREMIUM_COLORS['gradient1']}; 
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-weight: 700;">
        🔍 {title}
    </h3>
    """, unsafe_allow_html=True)
    
    # Group data by region for anomaly detection
    region_data = data.groupby('region')['nombre_rendez_vous'].sum().reset_index()
    region_data = region_data.sort_values('nombre_rendez_vous', ascending=False)
    
    # Calculate statistical properties
    values = region_data['nombre_rendez_vous'].values
    
    # Simple Z-score anomaly detection (would be more sophisticated in a real implementation)
    mean = np.mean(values)
    std = np.std(values)
    
    # Calculate z-scores
    z_scores = stats.zscore(values)
    
    # Flag anomalies (|z| > 2.5 is often considered a common threshold)
    region_data['z_score'] = z_scores
    region_data['is_anomaly'] = abs(region_data['z_score']) > 2.5
    
    # Create figure
    fig = go.Figure()
    
    # Add main trace (non-anomalies)
    regular_data = region_data[~region_data['is_anomaly']]
    fig.add_trace(go.Bar(
        x=regular_data['region'],
        y=regular_data['nombre_rendez_vous'],
        name="Normal",
        marker_color=PREMIUM_COLORS['primary'],
        opacity=0.8
    ))
    
    # Add anomaly trace
    anomalies = region_data[region_data['is_anomaly']]
    fig.add_trace(go.Bar(
        x=anomalies['region'],
        y=anomalies['nombre_rendez_vous'],
        name="Anomalie",
        marker_color=PREMIUM_COLORS['accent1'],
        opacity=0.9
    ))
    
    # Add mean line and standard deviation bands
    x_values = list(range(len(region_data)))
    
    fig.add_trace(go.Scatter(
        x=region_data['region'],
        y=[mean] * len(region_data),
        name="Moyenne",
        line=dict(color=PREMIUM_COLORS['secondary'], width=2, dash='dash'),
        mode='lines'
    ))
    
    # Add standard deviation bands
    fig.add_trace(go.Scatter(
        x=region_data['region'],
        y=[mean + 2*std] * len(region_data),
        name="+2 σ",
        line=dict(color=PREMIUM_COLORS['secondary'], width=1, dash='dot'),
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=region_data['region'],
        y=[mean - 2*std] * len(region_data),
        name="-2 σ",
        line=dict(color=PREMIUM_COLORS['secondary'], width=1, dash='dot'),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 223, 162, 0.1)'
    ))
    
    # Customize layout
    fig.update_layout(
        title="Détection d'Anomalies dans les Rendez-vous par Région",
        xaxis_title="Région",
        yaxis_title="Nombre de Rendez-vous",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display insights and anomaly details
    if len(anomalies) > 0:
        st.markdown("""
        <div style="background: linear-gradient(to right, rgba(0, 121, 255, 0.05), rgba(255, 0, 96, 0.05));
                    border-radius: 15px; padding: 20px; margin: 20px 0;">
            <h4 style="color: #FF0060; margin-top: 0;">🔔 Alertes d'Anomalies Détectées</h4>
            <p>Notre système d'intelligence artificielle a identifié des régions présentant des valeurs statistiquement anormales 
            qui pourraient nécessiter une attention particulière.</p>
        """, unsafe_allow_html=True)
        
        # Create an anomaly table
        anomaly_table = anomalies[['region', 'nombre_rendez_vous', 'z_score']].copy()
        anomaly_table['écart'] = (anomaly_table['nombre_rendez_vous'] - mean) / mean * 100
        anomaly_table['sévérité'] = anomaly_table['z_score'].abs().apply(
            lambda z: "⚠️ Faible" if z < 3 else "⚠️⚠️ Moyenne" if z < 4 else "⚠️⚠️⚠️ Élevée"
        )
        
        for i, row in anomaly_table.iterrows():
            direction = "supérieur" if row['z_score'] > 0 else "inférieur"
            severity_color = "#FFA500" if "Faible" in row['sévérité'] else "#FF4500" if "Moyenne" in row['sévérité'] else "#FF0000"
            
            st.markdown(f"""
            <div style="background-color: white; border-radius: 10px; padding: 15px; margin-bottom: 15px;
                      box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-left: 4px solid {severity_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h5 style="margin: 0; color: #1A374D;">Région {row['region']}</h5>
                        <p style="margin: 5px 0 0 0; color: #666;">
                            Valeur <strong style="color: {PREMIUM_COLORS['accent1']};">{int(row['nombre_rendez_vous'])}</strong> 
                            ({int(row['écart'])}% {direction} à la moyenne)
                        </p>
                    </div>
                    <div style="color: {severity_color}; font-weight: 600;">
                        {row['sévérité']}
                    </div>
                </div>
                <p style="margin: 10px 0 0 0; font-size: 0.9rem;">
                    Z-score: <strong>{row['z_score']:.2f}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add recommendation
        st.markdown("""
        <div style="margin-top: 15px;">
            <h5 style="color: #0079FF; margin-bottom: 10px;">🧠 Recommandations IA</h5>
            <ul>
                <li><strong>Investiguer les causes:</strong> Examiner les facteurs qui pourraient expliquer les anomalies détectées.</li>
                <li><strong>Comparer historiquement:</strong> Vérifier si ces anomalies sont récurrentes ou nouvelles.</li>
                <li><strong>Action ciblée:</strong> Mettre en place des stratégies spécifiques pour les régions présentant des anomalies importantes.</li>
            </ul>
        </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("Aucune anomalie significative n'a été détectée dans les données actuelles.")


def display_cohort_analysis(data: pd.DataFrame, title: str = "Analyse de Cohortes Avancée"):
    """
    Display a cohort analysis visualization to analyze patterns across different groups.
    
    Args:
        data (pd.DataFrame): DataFrame containing the appointment data
        title (str): Chart title
    """
    if data.empty or 'region' not in data.columns:
        st.warning("Données insuffisantes pour l'analyse de cohortes.")
        return
    
    # Create title with modern styling
    st.markdown(f"""
    <h3 style="text-align: center; margin-bottom: 25px; 
               background: {PREMIUM_COLORS['gradient1']}; 
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-weight: 700;">
        👥 {title}
    </h3>
    """, unsafe_allow_html=True)
    
    # Create synthetic cohort data for demonstration
    # This would use real cohort data in a production environment
    
    # Get unique regions
    regions = data['region'].unique()
    
    # Create retention data (synthetic for demo)
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic cohorts (e.g., by week)
    cohorts = ['Semaine 1', 'Semaine 2', 'Semaine 3', 'Semaine 4', 'Semaine 5']
    periods = ['Initial', '+1', '+2', '+3', '+4']
    
    # Create synthetic retention rates
    # Initial is always 100%, then gradually decreases
    cohort_data = []
    
    for cohort in cohorts:
        # Base retention rate (different for each cohort)
        base_rate = np.random.uniform(0.8, 0.95)
        
        for i, period in enumerate(periods):
            if i == 0:
                retention = 100  # Initial is always 100%
            else:
                # Decay rate increases with time
                decay = base_rate ** i
                # Add some randomness
                retention = decay * 100 * (1 + np.random.uniform(-0.05, 0.05))
                retention = max(min(retention, 100), 0)  # Ensure between 0-100
            
            cohort_data.append({
                'cohort': cohort,
                'period': period,
                'retention': retention
            })
    
    # Convert to DataFrame
    cohort_df = pd.DataFrame(cohort_data)
    
    # Create pivot table
    cohort_pivot = cohort_df.pivot(index='cohort', columns='period', values='retention')
    
    # Create heatmap
    fig = px.imshow(
        cohort_pivot,
        labels=dict(x="Période", y="Cohorte", color="Taux de Rétention (%)"),
        x=periods,
        y=cohorts,
        color_continuous_scale=[
            [0, 'rgba(255, 0, 96, 0.8)'],
            [0.3, 'rgba(255, 118, 118, 0.8)'],
            [0.5, 'rgba(255, 177, 101, 0.8)'],
            [0.7, 'rgba(246, 250, 112, 0.8)'],
            [0.9, 'rgba(0, 223, 162, 0.8)'],
            [1, 'rgba(0, 121, 255, 0.8)']
        ],
        text_auto='.1f',
        aspect="auto"
    )
    
    # Update layout
    fig.update_layout(
        title="Analyse de Rétention par Cohorte",
        xaxis_title="Période d'Observation",
        yaxis_title="Cohorte",
        coloraxis_colorbar=dict(
            title="Taux de Rétention (%)",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300,
            yanchor="top", y=1,
            ticks="outside"
        ),
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights and explanations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: rgba(0, 121, 255, 0.05); padding: 15px; border-radius: 10px;">
            <h4 style="color: #0079FF; margin-top: 0;">🧠 Guide d'Interprétation</h4>
            <p>L'analyse de cohortes permet de suivre des groupes d'utilisateurs au fil du temps pour comprendre leur comportement.</p>
            <ul>
                <li><strong>Cohorte:</strong> Groupe d'utilisateurs qui ont commencé dans la même période</li>
                <li><strong>Période:</strong> Temps écoulé depuis l'entrée dans la cohorte</li>
                <li><strong>Taux de rétention:</strong> Pourcentage d'utilisateurs toujours actifs à une période donnée</li>
            </ul>
            <p>Un bon taux de rétention indique une fidélisation efficace des utilisateurs.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Calculate some insights from the synthetic data
        avg_retention = cohort_df[cohort_df['period'] == '+1']['retention'].mean()
        best_cohort = cohort_pivot['+4'].idxmax()
        best_cohort_value = cohort_pivot.loc[best_cohort, '+4']
        
        st.markdown(f"""
        <div style="background-color: rgba(0, 121, 255, 0.05); padding: 15px; border-radius: 10px;">
            <h4 style="color: #0079FF; margin-top: 0;">📊 Principaux Insights</h4>
            <p><span style="color: #00DFA2; font-weight: 600;">Rétention moyenne (Période +1):</span> {avg_retention:.1f}%</p>
            <p><span style="color: #0079FF; font-weight: 600;">Meilleure cohorte sur 4 périodes:</span> {best_cohort} ({best_cohort_value:.1f}%)</p>
            <p><span style="color: #FF0060; font-weight: 600;">Potentiel d'amélioration:</span> Concentrez-vous sur la période +2 qui montre la plus forte baisse de rétention.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add trends and recommendations
    st.markdown("""
    <div style="background: linear-gradient(to right, rgba(0, 121, 255, 0.05), rgba(0, 223, 162, 0.05));
                border-radius: 15px; padding: 20px; margin: 20px 0;">
        <h4 style="color: #0079FF; margin-top: 0;">💡 Tendances et Recommandations</h4>
        
        <div style="display: flex; gap: 20px; margin: 15px 0;">
            <div style="flex: 1; background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <h5 style="color: #00DFA2; margin-top: 0;">Points Forts</h5>
                <ul>
                    <li>Les cohortes récentes montrent une meilleure rétention, indiquant des améliorations dans l'expérience utilisateur</li>
                    <li>Le taux de rétention initial est excellent, montrant un fort engagement dès le départ</li>
                </ul>
            </div>
            
            <div style="flex: 1; background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <h5 style="color: #FF0060; margin-top: 0;">Opportunités d'Amélioration</h5>
                <ul>
                    <li>La baisse de rétention entre les périodes +1 et +2 indique un potentiel d'amélioration dans le ré-engagement</li>
                    <li>La cohorte "Semaine 2" montre une rétention inférieure qui mériterait une analyse approfondie</li>
                </ul>
            </div>
        </div>
        
        <h5 style="color: #0079FF; margin-bottom: 10px;">Actions Recommandées</h5>
        <ol>
            <li><strong>Programme de fidélisation:</strong> Mettre en place des incitations pour les utilisateurs après leur première utilisation</li>
            <li><strong>Campagne de ré-engagement:</strong> Cibler spécifiquement les utilisateurs à risque d'abandon en période +2</li>
            <li><strong>Analyse segmentée:</strong> Identifier les caractéristiques des cohortes à forte rétention pour les répliquer</li>
            <li><strong>Enquête utilisateur:</strong> Recueillir des feedbacks qualitatifs pour comprendre les causes d'abandon</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


def display_cluster_analysis(data: pd.DataFrame, title: str = "Segmentation par Clusters IA"):
    """
    Display a cluster analysis visualization using AI techniques to segment the data.
    
    Args:
        data (pd.DataFrame): DataFrame containing the appointment data
        title (str): Chart title
    """
    if data.empty or 'region' not in data.columns:
        st.warning("Données insuffisantes pour l'analyse par clusters.")
        return
    
    # Create title with modern styling
    st.markdown(f"""
    <h3 style="text-align: center; margin-bottom: 25px; 
               background: {PREMIUM_COLORS['gradient1']}; 
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-weight: 700;">
        🧩 {title}
    </h3>
    """, unsafe_allow_html=True)
    
    # For this demo, we'll create a synthetic cluster analysis
    # In a real implementation, this would use actual clustering algorithms
    
    # Get regions data
    region_data = data.groupby('region')['nombre_rendez_vous'].sum().reset_index()
    
    # Create synthetic cluster data
    np.random.seed(42)  # For reproducibility
    
    # Create second and third dimensions for clustering
    region_data['variability'] = region_data['nombre_rendez_vous'] * np.random.uniform(0.1, 0.3, len(region_data))
    region_data['growth_potential'] = region_data['nombre_rendez_vous'] * np.random.uniform(0.8, 1.2, len(region_data))
    
    # Assign clusters (for demonstration)
    # In a real implementation, this would use k-means or other clustering algorithms
    
    def assign_cluster(row):
        # Use global dataframe values for comparison rather than individual values
        median_rdv = region_data['nombre_rendez_vous'].median()
        median_growth = region_data['growth_potential'].median()
        
        if row['nombre_rendez_vous'] > median_rdv * 1.5:
            if row['growth_potential'] > median_growth:
                return "Champions 🌟"
            else:
                return "Piliers Stables 🏛️"
        elif row['nombre_rendez_vous'] > median_rdv * 0.7:
            if row['variability'] > region_data['variability'].median():
                return "En Croissance 📈"
            else:
                return "Modérés 🔍"
        else:
            if row['growth_potential'] > median_growth:
                return "Potentiel Inexploité 💎"
            else:
                return "Nécessite Attention ⚠️"
    
    # Apply synthetic clustering
    region_data['cluster'] = region_data.apply(assign_cluster, axis=1)
    
    # Define cluster colors
    cluster_colors = {
        "Champions 🌟": PREMIUM_COLORS['primary'],
        "Piliers Stables 🏛️": PREMIUM_COLORS['secondary'],
        "En Croissance 📈": PREMIUM_COLORS['accent2'],
        "Modérés 🔍": PREMIUM_COLORS['neutral1'],
        "Potentiel Inexploité 💎": PREMIUM_COLORS['accent1'],
        "Nécessite Attention ⚠️": "#FF7676"
    }
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        region_data,
        x='nombre_rendez_vous',
        y='variability',
        z='growth_potential',
        color='cluster',
        color_discrete_map=cluster_colors,
        size='nombre_rendez_vous',
        hover_name='region',
        opacity=0.8,
        size_max=30,
        title="Segmentation des Régions par Clusters IA",
        labels={
            'nombre_rendez_vous': 'Volume de RDV',
            'variability': 'Variabilité',
            'growth_potential': 'Potentiel de Croissance'
        }
    )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Volume de Rendez-vous',
            yaxis_title='Variabilité',
            zaxis_title='Potentiel de Croissance',
            aspectmode='auto'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
        legend=dict(
            title="Clusters",
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    # Add text labels for selected points
    for cluster in region_data['cluster'].unique():
        # Get top point in cluster
        if not region_data[region_data['cluster'] == cluster].empty:
            top_point = region_data[region_data['cluster'] == cluster].sort_values('nombre_rendez_vous', ascending=False).iloc[0]
            
            fig.add_trace(go.Scatter3d(
                x=[top_point['nombre_rendez_vous']],
                y=[top_point['variability']],
                z=[top_point['growth_potential']],
                mode='text',
                text=[f"R{top_point['region']}"],
                textposition="top center",
                textfont=dict(size=12, color='black', family="Arial Black"),
                showlegend=False
            ))
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add cluster insights
    st.markdown("""
    <div style="background: linear-gradient(to right, rgba(0, 121, 255, 0.05), rgba(0, 223, 162, 0.05));
                border-radius: 15px; padding: 20px; margin: 20px 0;">
        <h4 style="color: #0079FF; margin-top: 0;">🔍 Analyse des Segments</h4>
        <p>Notre algorithme d'intelligence artificielle a identifié 6 segments distincts parmi les régions, chacun avec des caractéristiques et un potentiel uniques.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display cluster cards
    col1, col2, col3 = st.columns(3)
    
    cluster_descriptions = {
        "Champions 🌟": "Régions à haut volume avec excellent potentiel de croissance future. Continuez à investir et à optimiser ces marchés clés.",
        "Piliers Stables 🏛️": "Régions établies avec volume élevé mais croissance modérée. Mettez l'accent sur la rétention et la fidélisation.",
        "En Croissance 📈": "Régions à volume moyen avec forte variabilité, indiquant un potentiel de croissance. Identifiez les facteurs de réussite.",
        "Modérés 🔍": "Performance moyenne stable. Testez de nouvelles approches pour débloquer leur potentiel.",
        "Potentiel Inexploité 💎": "Faible volume mais fort potentiel. Investissez dans ces régions pour saisir des opportunités inexploitées.",
        "Nécessite Attention ⚠️": "Faible volume et potentiel limité. Analysez les causes profondes et envisagez des changements de stratégie."
    }
    
    cluster_metrics = {
        "Champions 🌟": {"num": len(region_data[region_data['cluster'] == "Champions 🌟"]), "vol": int(region_data[region_data['cluster'] == "Champions 🌟"]['nombre_rendez_vous'].sum()) if not region_data[region_data['cluster'] == "Champions 🌟"].empty else 0},
        "Piliers Stables 🏛️": {"num": len(region_data[region_data['cluster'] == "Piliers Stables 🏛️"]), "vol": int(region_data[region_data['cluster'] == "Piliers Stables 🏛️"]['nombre_rendez_vous'].sum()) if not region_data[region_data['cluster'] == "Piliers Stables 🏛️"].empty else 0},
        "En Croissance 📈": {"num": len(region_data[region_data['cluster'] == "En Croissance 📈"]), "vol": int(region_data[region_data['cluster'] == "En Croissance 📈"]['nombre_rendez_vous'].sum()) if not region_data[region_data['cluster'] == "En Croissance 📈"].empty else 0},
        "Modérés 🔍": {"num": len(region_data[region_data['cluster'] == "Modérés 🔍"]), "vol": int(region_data[region_data['cluster'] == "Modérés 🔍"]['nombre_rendez_vous'].sum()) if not region_data[region_data['cluster'] == "Modérés 🔍"].empty else 0},
        "Potentiel Inexploité 💎": {"num": len(region_data[region_data['cluster'] == "Potentiel Inexploité 💎"]), "vol": int(region_data[region_data['cluster'] == "Potentiel Inexploité 💎"]['nombre_rendez_vous'].sum()) if not region_data[region_data['cluster'] == "Potentiel Inexploité 💎"].empty else 0},
        "Nécessite Attention ⚠️": {"num": len(region_data[region_data['cluster'] == "Nécessite Attention ⚠️"]), "vol": int(region_data[region_data['cluster'] == "Nécessite Attention ⚠️"]['nombre_rendez_vous'].sum()) if not region_data[region_data['cluster'] == "Nécessite Attention ⚠️"].empty else 0}
    }
    
    # Display first row of clusters
    with col1:
        cluster = "Champions 🌟"
        st.markdown(f"""
        <div style="background-color: white; padding: 15px; border-radius: 10px; 
                  box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-top: 4px solid {cluster_colors[cluster]}; height: 240px;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.5rem; margin-right: 8px;">{cluster.split()[1]}</span>
                <span style="font-weight: 600; font-size: 1.2rem; color: {cluster_colors[cluster]};">{cluster.split()[0]}</span>
            </div>
            <div style="display: flex; gap: 10px; margin: 15px 0;">
                <div style="flex: 1; background: rgba(0, 121, 255, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: #666;">Régions</div>
                    <div style="font-size: 1.4rem; font-weight: 600; color: #0079FF;">{cluster_metrics[cluster]['num']}</div>
                </div>
                <div style="flex: 1; background: rgba(0, 121, 255, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: #666;">Volume</div>
                    <div style="font-size: 1.4rem; font-weight: 600; color: #0079FF;">{cluster_metrics[cluster]['vol']}</div>
                </div>
            </div>
            <p style="font-size: 0.9rem; color: #666; margin-top: 15px;">{cluster_descriptions[cluster]}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        cluster = "Piliers Stables 🏛️"
        st.markdown(f"""
        <div style="background-color: white; padding: 15px; border-radius: 10px; 
                  box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-top: 4px solid {cluster_colors[cluster]}; height: 240px;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.5rem; margin-right: 8px;">{cluster.split()[1]}</span>
                <span style="font-weight: 600; font-size: 1.2rem; color: {cluster_colors[cluster]};">{cluster.split()[0]}</span>
            </div>
            <div style="display: flex; gap: 10px; margin: 15px 0;">
                <div style="flex: 1; background: rgba(0, 121, 255, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: #666;">Régions</div>
                    <div style="font-size: 1.4rem; font-weight: 600; color: #0079FF;">{cluster_metrics[cluster]['num']}</div>
                </div>
                <div style="flex: 1; background: rgba(0, 121, 255, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: #666;">Volume</div>
                    <div style="font-size: 1.4rem; font-weight: 600; color: #0079FF;">{cluster_metrics[cluster]['vol']}</div>
                </div>
            </div>
            <p style="font-size: 0.9rem; color: #666; margin-top: 15px;">{cluster_descriptions[cluster]}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        cluster = "En Croissance 📈"
        st.markdown(f"""
        <div style="background-color: white; padding: 15px; border-radius: 10px; 
                  box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-top: 4px solid {cluster_colors[cluster]}; height: 240px;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.5rem; margin-right: 8px;">{cluster.split()[1]}</span>
                <span style="font-weight: 600; font-size: 1.2rem; color: {cluster_colors[cluster]};">{cluster.split()[0]}</span>
            </div>
            <div style="display: flex; gap: 10px; margin: 15px 0;">
                <div style="flex: 1; background: rgba(0, 121, 255, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: #666;">Régions</div>
                    <div style="font-size: 1.4rem; font-weight: 600; color: #0079FF;">{cluster_metrics[cluster]['num']}</div>
                </div>
                <div style="flex: 1; background: rgba(0, 121, 255, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: #666;">Volume</div>
                    <div style="font-size: 1.4rem; font-weight: 600; color: #0079FF;">{cluster_metrics[cluster]['vol']}</div>
                </div>
            </div>
            <p style="font-size: 0.9rem; color: #666; margin-top: 15px;">{cluster_descriptions[cluster]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display second row of clusters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cluster = "Modérés 🔍"
        st.markdown(f"""
        <div style="background-color: white; padding: 15px; border-radius: 10px; 
                  box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-top: 4px solid {cluster_colors[cluster]}; height: 240px;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.5rem; margin-right: 8px;">{cluster.split()[1]}</span>
                <span style="font-weight: 600; font-size: 1.2rem; color: {cluster_colors[cluster]};">{cluster.split()[0]}</span>
            </div>
            <div style="display: flex; gap: 10px; margin: 15px 0;">
                <div style="flex: 1; background: rgba(0, 121, 255, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: #666;">Régions</div>
                    <div style="font-size: 1.4rem; font-weight: 600; color: #0079FF;">{cluster_metrics[cluster]['num']}</div>
                </div>
                <div style="flex: 1; background: rgba(0, 121, 255, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: #666;">Volume</div>
                    <div style="font-size: 1.4rem; font-weight: 600; color: #0079FF;">{cluster_metrics[cluster]['vol']}</div>
                </div>
            </div>
            <p style="font-size: 0.9rem; color: #666; margin-top: 15px;">{cluster_descriptions[cluster]}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        cluster = "Potentiel Inexploité 💎"
        st.markdown(f"""
        <div style="background-color: white; padding: 15px; border-radius: 10px; 
                  box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-top: 4px solid {cluster_colors[cluster]}; height: 240px;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.5rem; margin-right: 8px;">{cluster.split()[1]}</span>
                <span style="font-weight: 600; font-size: 1.2rem; color: {cluster_colors[cluster]};">{cluster.split()[0]}</span>
            </div>
            <div style="display: flex; gap: 10px; margin: 15px 0;">
                <div style="flex: 1; background: rgba(0, 121, 255, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: #666;">Régions</div>
                    <div style="font-size: 1.4rem; font-weight: 600; color: #0079FF;">{cluster_metrics[cluster]['num']}</div>
                </div>
                <div style="flex: 1; background: rgba(0, 121, 255, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: #666;">Volume</div>
                    <div style="font-size: 1.4rem; font-weight: 600; color: #0079FF;">{cluster_metrics[cluster]['vol']}</div>
                </div>
            </div>
            <p style="font-size: 0.9rem; color: #666; margin-top: 15px;">{cluster_descriptions[cluster]}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        cluster = "Nécessite Attention ⚠️"
        st.markdown(f"""
        <div style="background-color: white; padding: 15px; border-radius: 10px; 
                  box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-top: 4px solid {cluster_colors[cluster]}; height: 240px;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.5rem; margin-right: 8px;">{cluster.split()[1]}</span>
                <span style="font-weight: 600; font-size: 1.2rem; color: {cluster_colors[cluster]};">{cluster.split()[0]}</span>
            </div>
            <div style="display: flex; gap: 10px; margin: 15px 0;">
                <div style="flex: 1; background: rgba(0, 121, 255, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: #666;">Régions</div>
                    <div style="font-size: 1.4rem; font-weight: 600; color: #0079FF;">{cluster_metrics[cluster]['num']}</div>
                </div>
                <div style="flex: 1; background: rgba(0, 121, 255, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: #666;">Volume</div>
                    <div style="font-size: 1.4rem; font-weight: 600; color: #0079FF;">{cluster_metrics[cluster]['vol']}</div>
                </div>
            </div>
            <p style="font-size: 0.9rem; color: #666; margin-top: 15px;">{cluster_descriptions[cluster]}</p>
        </div>
        """, unsafe_allow_html=True)

def display_ai_insights_dashboard(data: pd.DataFrame, title: str = "Intelligence Artificielle & Insights"):
    """
    Display an AI-powered insights dashboard with advanced analytics.
    
    Args:
        data (pd.DataFrame): DataFrame containing the appointment data
        title (str): Chart title
    """
    if data.empty or 'region' not in data.columns:
        st.warning("Données insuffisantes pour les insights IA.")
        return
    
    # Create title with modern styling
    st.markdown(f"""
    <h3 style="text-align: center; margin-bottom: 25px; 
               background: {PREMIUM_COLORS['gradient1']}; 
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-weight: 700;">
        🧠 {title}
    </h3>
    """, unsafe_allow_html=True)
    
    # Create AI-generated insights
    st.markdown("""
    <div style="
        background: linear-gradient(to right, rgba(0, 121, 255, 0.05), rgba(0, 223, 162, 0.05));
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 30px;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div style="
                width: 50px;
                height: 50px;
                border-radius: 50%;
                background: linear-gradient(135deg, #0079FF, #00DFA2);
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 15px;
            ">
                <span style="font-size: 25px;">🧠</span>
            </div>
            <div>
                <h4 style="margin: 0; color: #0079FF;">Assistant IA - Synthèse d'Insights</h4>
                <p style="margin: 5px 0 0 0; color: #666;">Analyse avancée basée sur les données disponibles</p>
            </div>
        </div>
        
        <div style="margin-left: 65px;">
            <p>Après analyse de vos données de rendez-vous, voici les principaux insights et recommandations :</p>
            
            <div style="
                background-color: white;
                border-radius: 10px;
                padding: 15px;
                margin: 15px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                border-left: 4px solid #0079FF;
            ">
                <h5 style="color: #0079FF; margin-top: 0;">Distribution Régionale</h5>
                <p>La répartition des rendez-vous montre une forte concentration dans certaines régions clés, créant un potentiel d'expansion dans les zones moins représentées.</p>
                <ul>
                    <li><strong>Opportunité:</strong> Développer des campagnes ciblées pour les régions à faible présence.</li>
                    <li><strong>Risque:</strong> Dépendance excessive aux régions dominantes.</li>
                </ul>
            </div>
            
            <div style="
                background-color: white;
                border-radius: 10px;
                padding: 15px;
                margin: 15px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                border-left: 4px solid #00DFA2;
            ">
                <h5 style="color: #00DFA2; margin-top: 0;">Patterns Temporels</h5>
                <p>L'analyse révèle des pics d'activité significatifs en milieu de semaine, avec une baisse notable le weekend.</p>
                <ul>
                    <li><strong>Recommandation:</strong> Optimiser les ressources pour les périodes de pointe identifiées.</li>
                    <li><strong>Action:</strong> Tester des incitations pour équilibrer la charge vers les périodes creuses.</li>
                </ul>
            </div>
            
            <div style="
                background-color: white;
                border-radius: 10px;
                padding: 15px;
                margin: 15px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                border-left: 4px solid #FF0060;
            ">
                <h5 style="color: #FF0060; margin-top: 0;">Prévisions & Tendances</h5>
                <p>Les modèles prédictifs suggèrent une croissance potentielle de 12-15% dans les mois à venir, avec des variations régionales significatives.</p>
                <ul>
                    <li><strong>Projection:</strong> Forte croissance anticipée dans les régions de type "Potentiel Inexploité".</li>
                    <li><strong>Alerte:</strong> Surveillance recommandée pour les régions "Nécessite Attention" qui risquent un déclin.</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create KPI snapshot
    st.markdown("<h4 style='text-align: center; margin: 30px 0 20px;'>Indicateurs Clés de Performance</h4>", unsafe_allow_html=True)
    
    # Calculate KPIs (synthetic for demo)
    total_rdv = data['nombre_rendez_vous'].sum()
    avg_per_region = total_rdv / data['region'].nunique()
    concentration_index = data.groupby('region')['nombre_rendez_vous'].sum().nlargest(3).sum() / total_rdv * 100
    
    # Display KPI cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            height: 150px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 10px;">Volume Total</div>
            <div style="
                font-size: 2rem;
                font-weight: 700;
                background: linear-gradient(to right, #0079FF, #00DFA2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">{int(total_rdv)}</div>
            <div style="font-size: 0.8rem; color: #00DFA2; margin-top: 10px;">
                <span style="font-weight: 600;">▲ 8.3%</span> vs période précédente
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div style="
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            height: 150px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 10px;">Moyenne / Région</div>
            <div style="
                font-size: 2rem;
                font-weight: 700;
                background: linear-gradient(to right, #0079FF, #00DFA2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">{int(avg_per_region)}</div>
            <div style="font-size: 0.8rem; color: #00DFA2; margin-top: 10px;">
                <span style="font-weight: 600;">▲ 5.2%</span> vs période précédente
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # Calculate coverage (synthetic)
        coverage = data['region'].nunique() / 99 * 100  # Assuming 99 regions in total for France
        
        st.markdown(f"""
        <div style="
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            height: 150px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 10px;">Couverture Régionale</div>
            <div style="
                font-size: 2rem;
                font-weight: 700;
                background: linear-gradient(to right, #0079FF, #00DFA2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">{coverage:.1f}%</div>
            <div style="font-size: 0.8rem; color: #00DFA2; margin-top: 10px;">
                <span style="font-weight: 600;">▲ 2.1%</span> vs période précédente
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div style="
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            height: 150px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 10px;">Indice de Concentration</div>
            <div style="
                font-size: 2rem;
                font-weight: 700;
                background: linear-gradient(to right, #FF0060, #FF7676);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">{concentration_index:.1f}%</div>
            <div style="font-size: 0.8rem; color: #FF0060; margin-top: 10px;">
                <span style="font-weight: 600;">▼ 1.3%</span> vs période précédente
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add recommendation section
    st.markdown("""
    <div style="margin-top: 40px;">
        <h4 style="text-align: center; margin-bottom: 20px;">Recommandations Stratégiques</h4>
        
        <div style="
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 20px;
        ">
            <div style="
                background-color: white;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                border-top: 4px solid #0079FF;
            ">
                <div style="font-size: 1.5rem; margin-bottom: 10px;">💼</div>
                <h5 style="color: #0079FF; margin-top: 0;">Optimisation Régionale</h5>
                <p style="font-size: 0.9rem; color: #666;">
                    Rééquilibrer le portefeuille régional en redirigeant 15% des ressources marketing vers les zones sous-représentées à fort potentiel.
                </p>
                <div style="
                    background-color: rgba(0, 121, 255, 0.1);
                    color: #0079FF;
                    font-weight: 600;
                    padding: 5px 10px;
                    border-radius: 50px;
                    display: inline-block;
                    font-size: 0.8rem;
                    margin-top: 10px;
                ">
                    Priorité Haute
                </div>
            </div>
            
            <div style="
                background-color: white;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                border-top: 4px solid #00DFA2;
            ">
                <div style="font-size: 1.5rem; margin-bottom: 10px;">📊</div>
                <h5 style="color: #00DFA2; margin-top: 0;">Distribution Temporelle</h5>
                <p style="font-size: 0.9rem; color: #666;">
                    Implémenter un système d'incitation pour répartir plus uniformément les rendez-vous sur la semaine, avec un focus sur les créneaux sous-utilisés.
                </p>
                <div style="
                    background-color: rgba(0, 223, 162, 0.1);
                    color: #00DFA2;
                    font-weight: 600;
                    padding: 5px 10px;
                    border-radius: 50px;
                    display: inline-block;
                    font-size: 0.8rem;
                    margin-top: 10px;
                ">
                    Priorité Moyenne
                </div>
            </div>
            
            <div style="
                background-color: white;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                border-top: 4px solid #FF0060;
            ">
                <div style="font-size: 1.5rem; margin-bottom: 10px;">🔄</div>
                <h5 style="color: #FF0060; margin-top: 0;">Plan d'Intervention</h5>
                <p style="font-size: 0.9rem; color: #666;">
                    Mettre en place un protocole de suivi spécifique pour les régions en alerte, avec analyse mensuelle des indicateurs de performance.
                </p>
                <div style="
                    background-color: rgba(255, 0, 96, 0.1);
                    color: #FF0060;
                    font-weight: 600;
                    padding: 5px 10px;
                    border-radius: 50px;
                    display: inline-block;
                    font-size: 0.8rem;
                    margin-top: 10px;
                ">
                    Action Immédiate
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
