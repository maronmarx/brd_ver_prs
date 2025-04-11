"""
Database connection module.
Provides a connection to the MySQL database for data operations.
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
import mysql.connector
from mysql.connector import Error as MySQLError

from config.config import DB_CONFIG

# Set up logger
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager class to handle connections and queries.
    """
    def __init__(self, config=None):
        """
        Initialize the database manager with configuration.
        
        Args:
            config (dict, optional): Database configuration. Defaults to DB_CONFIG.
        """
        self.config = config or DB_CONFIG
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """
        Initialize the SQLAlchemy engine with the current configuration.
        """
        try:
            connection_string = (
                f"mysql+mysqlconnector://{self.config['user']}:{self.config['password']}"
                f"@{self.config['host']}/{self.config['database']}"
            )
            self.engine = create_engine(connection_string)
            logger.info("SQLAlchemy engine initialized successfully.")
        except SQLAlchemyError as e:
            logger.error("Failed to initialize SQLAlchemy engine: %s", str(e))
            raise
    
    @contextmanager
    def get_connection(self):
        """
        Provide a context manager for database connections.
        
        Yields:
            Connection: A SQLAlchemy database connection.
        """
        if not self.engine:
            self._initialize_engine()
        
        conn = None
        try:
            conn = self.engine.connect()
            logger.debug("Database connection established.")
            yield conn
        except SQLAlchemyError as e:
            logger.error("Database connection error: %s", str(e))
            raise
        finally:
            if conn:
                conn.close()
                logger.debug("Database connection closed.")

    def execute_query(self, query, params=None):
        """
        Execute a query and return results as a pandas DataFrame.
        
        Args:
            query (str): SQL query to execute.
            params (dict or list, optional): Parameters for the query. Defaults to None.
            
        Returns:
            pandas.DataFrame: Results of the query.
        """
        import pandas as pd
        
        try:
            with self.get_connection() as conn:
                logger.debug("Executing query: %s", query)
                if params:
                    if isinstance(params, dict):
                        # For named parameters
                        return pd.read_sql_query(query, conn, params=params)
                    else:
                        # For positional parameters
                        return pd.read_sql_query(query, conn, params=tuple(params))
                else:
                    return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error("Error executing query: %s", str(e))
            logger.error("Query: %s", query)
            if params:
                logger.error("Parameters: %s", str(params))
            return pd.DataFrame()  # Return empty DataFrame on error

    def execute_raw_query(self, query, params=None):
        """
        Execute a raw SQL query directly with mysql.connector.
        Useful for operations that don't return data (INSERT, UPDATE, DELETE).
        
        Args:
            query (str): SQL query to execute.
            params (tuple, optional): Parameters for the query. Defaults to None.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        conn = None
        cursor = None
        try:
            conn = mysql.connector.connect(
                host=self.config['host'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database']
            )
            cursor = conn.cursor()
            logger.debug("Executing raw query: %s", query)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            logger.debug("Query executed successfully.")
            return True
        except MySQLError as e:
            logger.error("Error executing raw query: %s", str(e))
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

# Singleton instance of DatabaseManager
db_manager = DatabaseManager()
