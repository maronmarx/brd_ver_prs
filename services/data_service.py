"""
Data service module.
Provides services for retrieving and processing appointment data.
"""
import pandas as pd
from datetime import datetime, timedelta
import re
import logging
from typing import Dict, Any, Tuple, List, Optional

from sqlalchemy import create_engine
from utils.translations import JOURS_SEMAINE, MOIS_ANNEE, JOURS_SEMAINE_INVERSE, MOIS_ANNEE_INVERSE, JOURS_ORDRE
from config.config import DB_CONFIG

# Get logger
logger = logging.getLogger('rndv_ghandi.data_service')

class AppointmentDataService:
    """
    Service for retrieving and processing appointment data.
    """
    def __init__(self):
        """
        Initialize the appointment data service.
        """
        self.db_host = DB_CONFIG['host']
        self.db_user = DB_CONFIG['user']
        self.db_password = DB_CONFIG['password']
        self.db_name = DB_CONFIG['database']
        self.table_name = DB_CONFIG['table']
        
    def validate_region(self, input_str: str) -> str:
        """
        Validate the format of the region (2 digits or 2x2 digits separated by a hyphen).
        
        Args:
            input_str (str): Input string to validate
            
        Returns:
            str: Validated region string or empty string if invalid
        """
        if not input_str:
            return ''
            
        input_str = input_str.strip()
        if re.match(r'^(\d{2}(-\d{2})?)$', input_str):
            parts = input_str.split('-')
            if len(parts) == 1:
                # Single region case
                if 0 <= int(parts[0]) <= 99:
                    return input_str
            elif len(parts) == 2:
                # Region range case
                if 0 <= int(parts[0]) <= 99 and 0 <= int(parts[1]) <= 99 and int(parts[0]) <= int(parts[1]):
                    return input_str
        return ''

    def fetch_data(self, 
                  region: Optional[str] = None, 
                  date: Optional[str] = None,
                  time_filter_type: Optional[str] = None, 
                  time_filter_value: Optional[int] = None,
                  day_filter: Optional[bool] = None, 
                  month_filter: Optional[str] = None,
                  week_filter: Optional[int] = None, 
                  year_filter: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch appointment data from the database based on various filters.
        Using the original query building approach with direct string formatting.
        
        Args:
            region (str, optional): Region code filter
            date (str, optional): Date filter in YYYY-MM-DD format
            time_filter_type (str, optional): Type of time filter ('minute' or 'hour')
            time_filter_value (int, optional): Value for time filtering
            day_filter (bool, optional): Flag to enable day of week analysis
            month_filter (str, optional): Month name filter in English
            week_filter (int, optional): Week number filter
            year_filter (int, optional): Year filter
            
        Returns:
            pd.DataFrame: Filtered appointment data
        """
        try:
            # Create engine for database connection
            engine = create_engine(f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}")
            
            # Build the appropriate query based on filters (using original approach)
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
                        FROM {self.table_name}
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
                        FROM {self.table_name}
                        WHERE 1=1
                    '''
            elif day_filter is not None:  # Analysis by day of week
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
                    FROM {self.table_name}
                    WHERE 1=1
                '''
            elif month_filter:
                query = f'''
                    SELECT 
                        LEFT(LPAD(client_postal_code, 5, '0'), 2) AS region,
                        YEAR(last_local_call_time) AS annee,
                        '{MOIS_ANNEE[month_filter]}' AS mois,
                        COUNT(*) AS nombre_rendez_vous
                    FROM {self.table_name}
                    WHERE MONTHNAME(last_local_call_time) = '{month_filter}'
                '''
            elif week_filter is not None and year_filter is not None:
                first_day = datetime.strptime(f'{year_filter}-W{week_filter}-1', "%Y-W%W-%w").date()
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
                    FROM {self.table_name}
                    WHERE YEARWEEK(last_local_call_time, 1) = YEARWEEK('{first_day}', 1)
                '''
            else:
                query = f'''
                    SELECT 
                        LEFT(LPAD(client_postal_code, 5, '0'), 2) AS region,
                        COUNT(*) AS nombre_rendez_vous
                    FROM {self.table_name}
                    WHERE 1=1
                '''

            # Apply region filter to all queries
            if region:
                # Support for multiple regions separated by a hyphen
                if '-' in region:
                    region1, region2 = region.split('-')
                    # Select only the two specific regions, not the entire range
                    query += f" AND LEFT(LPAD(client_postal_code, 5, '0'), 2) IN ('{region1.strip()}', '{region2.strip()}')"
                else:
                    query += f" AND LEFT(LPAD(client_postal_code, 5, '0'), 2) = '{region.strip()}'"

            # Apply date filter if specified
            if date:
                query += f" AND DATE(last_local_call_time) = '{date}'"

            # Add GROUP BY and ORDER BY clauses based on filter type
            if time_filter_type and time_filter_value:
                if time_filter_type == "minute":
                    query += f" GROUP BY region, HOUR(last_local_call_time), FLOOR(MINUTE(last_local_call_time) / {time_filter_value}), plage_horaire"
                    query += f" ORDER BY region, HOUR(last_local_call_time), FLOOR(MINUTE(last_local_call_time) / {time_filter_value})"
                else:
                    query += f" GROUP BY region, FLOOR(HOUR(last_local_call_time) / {time_filter_value}), plage_horaire"
                    query += f" ORDER BY region, FLOOR(HOUR(last_local_call_time) / {time_filter_value})"
            elif day_filter is not None:
                # For day analysis, group by region and day of week
                query += " GROUP BY region, jour_semaine ORDER BY region, FIELD(jour_semaine, 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche')"
            elif month_filter:
                # GROUP BY for month filter
                query += " GROUP BY region, annee, mois ORDER BY region, annee"
            elif week_filter is not None and year_filter is not None:
                # GROUP BY for week filter
                query += " GROUP BY region, jour"
                query += " ORDER BY region, FIELD(jour, 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche')"
            else:
                # Default GROUP BY
                query += " GROUP BY region ORDER BY region"

            # Log the query for debugging
            logger.debug(f"Executing query: {query}")
            
            # Execute the query and return results
            return pd.read_sql(query, con=engine)
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def analyse_jours_par_region(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze data by day of the week for each region and identify
        the region with the most appointments for each day.
        
        Args:
            df (pd.DataFrame): DataFrame with appointment data
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Pivot table and dictionary of top regions per day
        """
        if df.empty or 'jour_semaine' not in df.columns:
            return pd.DataFrame(), {}

        # Create a pivot table with days as columns
        pivot_df = df.pivot_table(
            index='region',
            columns='jour_semaine',
            values='nombre_rendez_vous',
            aggfunc='sum',
            fill_value=0
        ).reset_index()

        # Reorder columns according to day order
        jour_cols = [j for j in JOURS_ORDRE if j in pivot_df.columns]
        pivot_df = pivot_df[['region'] + jour_cols]

        # Add total column per region
        pivot_df['Total'] = pivot_df[jour_cols].sum(axis=1)

        # Sort by total in descending order
        pivot_df = pivot_df.sort_values('Total', ascending=False)

        # Find the region with the most appointments for each day
        top_regions = {}
        for jour in jour_cols:
            day_data = df[df['jour_semaine'] == jour]
            if not day_data.empty:
                max_region = day_data.loc[day_data['nombre_rendez_vous'].idxmax()]
                top_regions[jour] = {
                    'region': max_region['region'],
                    'nombre': int(max_region['nombre_rendez_vous'])
                }

        return pivot_df, top_regions

# Create a singleton instance
appointment_service = AppointmentDataService()
